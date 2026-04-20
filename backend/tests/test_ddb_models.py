from anna_chat.ddb import Conversation, Message, Repository


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
