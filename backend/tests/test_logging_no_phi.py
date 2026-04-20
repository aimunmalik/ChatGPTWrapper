import io
import json
import logging

from anna_chat.logging_config import JsonFormatter, configure_logging


def test_json_formatter_drops_phi_fields():
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="x.py",
        lineno=1,
        msg="chat_turn_complete",
        args=(),
        exc_info=None,
    )
    record.userId = "u_123"
    record.conversationId = "c_abc"
    record.latencyMs = 42
    record.content = "patient: John Doe, DOB 01/01/2000, diagnosis..."
    record.message = "same PHI-bearing text that must not be logged"
    record.messages = [{"role": "user", "content": "PHI"}]
    record.body = "PHI"
    record.prompt = "PHI"
    record.completion = "PHI"
    record.text = "PHI"
    record.assistantMessage = "PHI"
    record.userMessage = "PHI"

    formatted = JsonFormatter().format(record)
    payload = json.loads(formatted)

    assert payload["userId"] == "u_123"
    assert payload["conversationId"] == "c_abc"
    assert payload["latencyMs"] == 42
    assert payload["level"] == "INFO"
    assert payload["message"] == "chat_turn_complete"

    for phi_field in [
        "content",
        "messages",
        "body",
        "prompt",
        "completion",
        "text",
        "assistantMessage",
        "userMessage",
    ]:
        assert phi_field not in payload, (
            f"PHI field '{phi_field}' leaked into log output: {payload}"
        )


def test_configure_logging_uses_json_handler():
    buf = io.StringIO()
    configure_logging()
    logger = logging.getLogger("anna_chat_test")
    for handler in logging.getLogger().handlers:
        handler.stream = buf

    logger.info("event_with_metadata", extra={"userId": "u_x", "latencyMs": 10})

    line = buf.getvalue().strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["userId"] == "u_x"
    assert payload["latencyMs"] == 10
    assert payload["level"] == "INFO"


def test_content_field_on_record_never_appears():
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="x.py",
        lineno=1,
        msg="x",
        args=(),
        exc_info=None,
    )
    record.content = "patient name: Jane Roe"
    formatted = JsonFormatter().format(record)
    assert "Jane Roe" not in formatted
    assert "patient" not in formatted
