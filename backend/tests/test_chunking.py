from anna_chat.chunking import (
    TOKEN_PER_WORD,
    ChunkWindow,
    approx_token_count,
    chunk_text,
)


def test_empty_input_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  \n ") == []


def test_single_short_paragraph_returns_one_chunk():
    text = "This is a short paragraph about ABA transitions."
    chunks = chunk_text(text, target_tokens=800, overlap_tokens=100)
    assert len(chunks) == 1
    assert isinstance(chunks[0], ChunkWindow)
    assert chunks[0].text == text
    # Approx tokens = word count × 1.3
    expected_tokens = int(len(text.split()) * TOKEN_PER_WORD)
    assert chunks[0].approx_tokens == expected_tokens


def test_approx_token_count_matches_formula():
    assert approx_token_count("") == 0
    assert approx_token_count("one two three four five") == int(5 * TOKEN_PER_WORD)


def test_long_text_splits_into_multiple_chunks_with_overlap():
    # Build a body of paragraphs, each ~50 words. With target_tokens small we
    # force multiple windows.
    paragraphs = [
        " ".join([f"word{i}_{j}" for j in range(50)]) for i in range(6)
    ]
    text = "\n\n".join(paragraphs)
    chunks = chunk_text(text, target_tokens=80, overlap_tokens=20)
    # 6 paragraphs × ~65 tokens each, target 80 → roughly one paragraph per
    # window, several windows total.
    assert len(chunks) >= 3
    # Each chunk should respect the target (within one paragraph's worth of
    # slack because we only split on paragraph boundaries).
    for c in chunks:
        assert c.approx_tokens <= 80 * 2  # 2x slack — we never split paragraphs


def test_overlap_is_carried_between_windows():
    # Two paragraphs that fit individually but not together.
    p1 = " ".join([f"first{i}" for i in range(60)])
    p2 = " ".join([f"second{i}" for i in range(60)])
    text = f"{p1}\n\n{p2}"
    chunks = chunk_text(text, target_tokens=90, overlap_tokens=30)
    assert len(chunks) >= 2
    # The second window should include tail words from the first paragraph
    # (overlap carry-over). Check that at least one `first*` word appears at
    # the start of window 2.
    window2 = chunks[1].text
    overlap_tokens = window2.split()[:30]
    assert any(w.startswith("first") for w in overlap_tokens)


def test_paragraph_preservation_means_blank_line_separator():
    # Two short paragraphs fit together in one window but the separator is
    # preserved as a blank line.
    text = "Para one with some words.\n\nPara two with more words here."
    chunks = chunk_text(text, target_tokens=800, overlap_tokens=50)
    assert len(chunks) == 1
    assert "\n\n" in chunks[0].text


def test_oversized_paragraph_is_force_split():
    # Single paragraph far bigger than target forces a word-level split.
    huge = " ".join([f"tok{i}" for i in range(500)])
    chunks = chunk_text(huge, target_tokens=100, overlap_tokens=20)
    assert len(chunks) >= 3
    # Windows after the first should start with tail words of the previous
    # (overlap). The first window has no carry (new text).
    for c in chunks[1:]:
        assert c.text.startswith("tok")
    # Collectively windows should preserve all original tokens (ignoring
    # overlap duplication).
    all_words = set()
    for c in chunks:
        all_words.update(c.text.split())
    assert "tok0" in all_words
    assert "tok499" in all_words


def test_normalizes_mixed_line_endings():
    text = "Para one.\r\n\r\nPara two.\r\n\r\nPara three."
    chunks = chunk_text(text, target_tokens=800, overlap_tokens=0)
    assert len(chunks) == 1
    assert "Para one." in chunks[0].text
    assert "Para three." in chunks[0].text


def test_chunk_window_default_metadata_is_none():
    # The basic chunker leaves page_number / section_title unset so the
    # caller (ingest handler) can fill them in.
    text = "hello world"
    chunks = chunk_text(text)
    assert chunks[0].page_number is None
    assert chunks[0].section_title is None
