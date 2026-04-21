from decimal import Decimal

from anna_chat.ddb import Conversation, Message, Repository, _floats_to_decimal


def test_sort_key_is_lexicographically_ordered():
    k1 = Repository._sort_key(1_000_000_000_000, "m_a")
    k2 = Repository._sort_key(1_000_000_000_001, "m_b")
    assert k1 < k2


def test_sort_key_has_fixed_width_timestamp():
    k = Repository._sort_key(1_700_000_000_000, "m_xyz")
    ts_part, id_part = k.split("#")
    assert len(ts_part) == 13
    assert id_part == "m_xyz"


def test_conversation_is_serializable():
    c = Conversation(
        userId="u1",
        conversationId="c1",
        title="hello",
        createdAt=1,
        updatedAt=2,
        model="m",
    )
    from dataclasses import asdict

    d = asdict(c)
    assert d["userId"] == "u1"
    assert d["title"] == "hello"


def test_floats_to_decimal_handles_nested_sources():
    """Guard against the 'DDB rejects floats' class of bug that kept blowing
    up the chat handler: retrieval sources have float scores, embeddings
    have float vectors, and anywhere one escapes into put_item we 500."""
    item = {
        "content": "hello",
        "inputTokens": 10,  # ints stay ints
        "sources": [
            {
                "index": 1,
                "docTitle": "Paper",
                "score": 0.842,
                "pageNumber": None,
            },
            {"index": 2, "docTitle": "Other", "score": 0.673, "pageNumber": 4},
        ],
        "embedding": [0.1, 0.2, 0.3],
    }
    out = _floats_to_decimal(item)
    # All floats -> Decimal
    assert isinstance(out["sources"][0]["score"], Decimal)
    assert out["sources"][0]["score"] == Decimal("0.842")
    assert isinstance(out["sources"][1]["score"], Decimal)
    assert all(isinstance(x, Decimal) for x in out["embedding"])
    # Non-floats preserved
    assert out["content"] == "hello"
    assert out["inputTokens"] == 10 and isinstance(out["inputTokens"], int)
    assert out["sources"][0]["pageNumber"] is None
    assert out["sources"][1]["pageNumber"] == 4


def test_message_defaults():
    m = Message(
        conversationId="c1",
        sortKey="0000000000000#m_x",
        userId="u1",
        role="user",
        content="hi",
        messageId="m_x",
    )
    assert m.inputTokens == 0
    assert m.outputTokens == 0
    assert m.ttl == 0
    assert m.model == ""
