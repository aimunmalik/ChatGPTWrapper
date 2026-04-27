from unittest.mock import MagicMock

from anna_chat.bedrock_client import BedrockResponse
from anna_chat.translate import (
    SYSTEM_PROMPT_TEMPLATE,
    TranslationResult,
    _dedupe_overlap,
    translate_text,
)


def _stub_bedrock(responses: list[BedrockResponse]) -> MagicMock:
    """Build a fake `BedrockClient` whose `invoke` returns each response in turn."""
    client = MagicMock()
    client.invoke.side_effect = responses
    return client


def test_translate_text_empty_input_returns_empty_result_no_invokes():
    client = _stub_bedrock([])
    result = translate_text("   ", "Spanish", bedrock=client)
    assert result == TranslationResult(text="", input_tokens=0, output_tokens=0)
    assert client.invoke.call_count == 0


def test_translate_text_single_chunk_invokes_once_with_system_prompt():
    client = _stub_bedrock(
        [
            BedrockResponse(
                text="Hola mundo.",
                input_tokens=10,
                output_tokens=4,
                stop_reason="end_turn",
            )
        ]
    )
    result = translate_text("Hello world.", "Spanish", bedrock=client)
    assert result.text == "Hola mundo."
    assert result.input_tokens == 10
    assert result.output_tokens == 4
    assert client.invoke.call_count == 1
    call_kwargs = client.invoke.call_args.kwargs
    assert call_kwargs["system"] == SYSTEM_PROMPT_TEMPLATE.format(
        target_language_label="Spanish"
    )
    # The user message MUST be the chunk text verbatim per the contract.
    assert call_kwargs["messages"][0]["content"] == "Hello world."
    # max_tokens at the contract's per-call limit.
    assert call_kwargs["max_tokens"] == 4096


def test_translate_text_multi_chunk_concatenates_and_sums_tokens():
    # Build a body that the chunker is guaranteed to split — two paragraphs
    # each well over the 3000-token target. Use a callable side_effect so we
    # don't have to predict the exact chunk count.
    long_para = ("word " * 4000).strip()
    multi_para = long_para + "\n\n" + long_para
    counter = {"i": 0}

    def _fake_invoke(**kwargs):
        counter["i"] += 1
        # Make each call's output marker distinct so concatenation is
        # observable. Prepend a unique tag, then a Spanish-ish payload.
        return BedrockResponse(
            text=f"chunk-{counter['i']}\n\npalabra palabra",
            input_tokens=3000,
            output_tokens=2500,
            stop_reason="end_turn",
        )

    client = MagicMock()
    client.invoke.side_effect = _fake_invoke
    result = translate_text(multi_para, "Spanish", bedrock=client)
    # The chunker must have produced multiple windows; otherwise the test
    # would silently pass with one call.
    assert client.invoke.call_count >= 2
    assert result.input_tokens == 3000 * client.invoke.call_count
    assert result.output_tokens == 2500 * client.invoke.call_count
    # Each chunk's marker survives concatenation.
    assert "chunk-1" in result.text
    assert f"chunk-{client.invoke.call_count}" in result.text
    # Concatenation uses double-newline boundaries between chunks.
    assert "\n\n" in result.text


def test_dedupe_overlap_drops_leading_line_when_duplicated():
    prev_tail = "Final paragraph.\nDuplicated header line."
    next_text = "Duplicated header line.\nNew content here."
    cleaned = _dedupe_overlap(prev_tail, next_text)
    assert cleaned == "New content here."


def test_dedupe_overlap_passthrough_when_no_match():
    prev_tail = "Something completely different."
    next_text = "Brand new opening line.\nMore content."
    cleaned = _dedupe_overlap(prev_tail, next_text)
    assert cleaned == next_text


def test_dedupe_overlap_handles_empty_inputs():
    assert _dedupe_overlap("", "abc") == "abc"
    assert _dedupe_overlap("abc", "") == ""


def test_translate_text_does_not_log_chunk_text(caplog):
    """Sanity check: even at debug, no source/translation appears in logs."""
    import logging

    secret = "patient-name-acme-corp-12345"
    client = _stub_bedrock(
        [
            BedrockResponse(
                text="translated-output-marker-67890",
                input_tokens=5,
                output_tokens=5,
                stop_reason="end_turn",
            )
        ]
    )
    with caplog.at_level(logging.DEBUG, logger="anna_chat"):
        translate_text(secret, "Spanish", bedrock=client)
    rendered = "\n".join(rec.getMessage() for rec in caplog.records)
    assert secret not in rendered
    assert "translated-output-marker" not in rendered
