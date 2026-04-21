import io
import json
import logging
import math
from typing import Any

from anna_chat import kb_retrieve
from anna_chat.kb_repo import Chunk
from anna_chat.logging_config import configure_logging


class _FakeEmbeddings:
    """Deterministic embedder for retrieval tests.

    `embeddings_by_text` maps exact input strings to the vector they should
    return, so tests can assert the scoring outcome without a real Bedrock
    call.
    """

    def __init__(self, embeddings_by_text: dict[str, list[float]]):
        self._map = embeddings_by_text

    def embed(self, text: str) -> list[float]:
        if text in self._map:
            return self._map[text]
        # Fallback: zero vector — will score 0 against everything.
        dim = len(next(iter(self._map.values()))) if self._map else 4
        return [0.0] * dim


class _FakeRepo:
    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks

    def scan_all_chunks(self) -> list[Chunk]:
        return list(self._chunks)


def _chunk(
    idx: int,
    embedding: list[float],
    *,
    text: str | None = None,
    doc_title: str = "Doc",
    source_type: str = "training",
    page: int | None = None,
) -> Chunk:
    return Chunk(
        kbDocId=f"kd_{idx}",
        chunkIdx=idx,
        chunkText=text or f"text-{idx}",
        chunkTokens=10,
        embedding=embedding,
        docTitle=doc_title,
        sourceType=source_type,
        pageNumber=page,
        sectionTitle=None,
        createdAt=0,
    )


def _unit(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec))
    return [x / n for x in vec] if n else vec


def _setup(monkeypatch: Any) -> None:
    # Clear cache between tests so one test doesn't pollute the next.
    kb_retrieve.invalidate_cache()
    monkeypatch.setattr(kb_retrieve.time, "monotonic", lambda: 0.0)


def test_empty_kb_returns_empty_list(monkeypatch):
    _setup(monkeypatch)
    embeddings = _FakeEmbeddings({"hello": [1.0, 0.0, 0.0, 0.0]})
    repo = _FakeRepo([])
    out = kb_retrieve.retrieve(
        "hello", embeddings=embeddings, repo=repo, top_k=5, min_score=0.0
    )
    assert out == []


def test_cosine_similarity_ranks_top_matches(monkeypatch):
    _setup(monkeypatch)
    # Three chunks with embeddings pointing in different directions; query
    # aligns exactly with the second chunk.
    query_vec = _unit([0.0, 1.0, 0.0, 0.0])
    chunks = [
        _chunk(0, _unit([1.0, 0.0, 0.0, 0.0]), text="orthogonal"),
        _chunk(1, _unit([0.0, 1.0, 0.0, 0.0]), text="perfect match"),
        _chunk(2, _unit([0.0, 0.9, 0.1, 0.0]), text="close match"),
    ]
    embeddings = _FakeEmbeddings({"q": query_vec})
    repo = _FakeRepo(chunks)
    out = kb_retrieve.retrieve(
        "q", embeddings=embeddings, repo=repo, top_k=5, min_score=0.0
    )
    assert len(out) == 3
    # Ranked by score desc.
    assert out[0].chunk_idx == 1
    assert out[1].chunk_idx == 2
    assert out[2].chunk_idx == 0
    # Perfect alignment → score ≈ 1.0
    assert abs(out[0].score - 1.0) < 1e-6
    # Orthogonal → score ≈ 0
    assert abs(out[2].score) < 1e-6


def test_threshold_filter_drops_low_scores(monkeypatch):
    _setup(monkeypatch)
    query_vec = _unit([1.0, 0.0, 0.0])
    chunks = [
        _chunk(0, _unit([1.0, 0.0, 0.0])),      # score 1.0
        _chunk(1, _unit([0.9, 0.1, 0.0])),      # ~0.994
        _chunk(2, _unit([0.0, 1.0, 0.0])),      # 0
    ]
    embeddings = _FakeEmbeddings({"q": query_vec})
    repo = _FakeRepo(chunks)
    out = kb_retrieve.retrieve(
        "q", embeddings=embeddings, repo=repo, top_k=5, min_score=0.35
    )
    # Only the two above 0.35 should survive.
    assert len(out) == 2
    assert {c.chunk_idx for c in out} == {0, 1}


def test_top_k_limits_results(monkeypatch):
    _setup(monkeypatch)
    query_vec = _unit([1.0, 0.0])
    chunks = [_chunk(i, _unit([1.0 - i * 0.01, 0.0 + i * 0.01])) for i in range(10)]
    embeddings = _FakeEmbeddings({"q": query_vec})
    repo = _FakeRepo(chunks)
    out = kb_retrieve.retrieve(
        "q", embeddings=embeddings, repo=repo, top_k=3, min_score=0.0
    )
    assert len(out) == 3
    # First three should be the closest-to-unit-x vectors.
    assert [c.chunk_idx for c in out] == [0, 1, 2]


def test_returned_chunk_has_embedding_stripped(monkeypatch):
    _setup(monkeypatch)
    query_vec = _unit([1.0, 0.0])
    chunks = [_chunk(0, _unit([1.0, 0.0]))]
    embeddings = _FakeEmbeddings({"q": query_vec})
    repo = _FakeRepo(chunks)
    out = kb_retrieve.retrieve(
        "q", embeddings=embeddings, repo=repo, top_k=5, min_score=0.0
    )
    # RetrievedChunk does not expose embedding.
    assert not hasattr(out[0], "embedding")
    assert out[0].chunk_text == "text-0"
    assert out[0].doc_title == "Doc"


def test_query_and_chunk_text_never_logged(monkeypatch):
    _setup(monkeypatch)
    buf = io.StringIO()
    configure_logging()
    for handler in logging.getLogger().handlers:
        handler.stream = buf

    secret_query = "patient John Doe has had meltdowns during transitions"
    secret_chunk_text = "ANNA protocol says patient-specific plan X for Jane Roe"
    query_vec = _unit([1.0, 0.0])
    chunks = [_chunk(0, _unit([1.0, 0.0]), text=secret_chunk_text)]
    embeddings = _FakeEmbeddings({secret_query: query_vec})
    repo = _FakeRepo(chunks)

    kb_retrieve.retrieve(
        secret_query, embeddings=embeddings, repo=repo, top_k=5, min_score=0.0
    )

    output = buf.getvalue()
    assert "John Doe" not in output
    assert "Jane Roe" not in output
    assert "patient" not in output
    assert "meltdowns" not in output
    assert "transitions" not in output
    # Metadata should still be present.
    # At least one JSON line should mention the log event.
    assert "kb_retrieve_complete" in output
    # Validate it's the structured JSON our formatter emits.
    last_line = [line for line in output.strip().splitlines() if line][-1]
    payload = json.loads(last_line)
    assert payload.get("chunksReturned") == 1
    assert payload.get("chunksScanned") == 1


def test_empty_query_returns_empty(monkeypatch):
    _setup(monkeypatch)
    embeddings = _FakeEmbeddings({})
    repo = _FakeRepo([_chunk(0, [1.0, 0.0])])
    assert kb_retrieve.retrieve(
        "", embeddings=embeddings, repo=repo, top_k=5, min_score=0.0
    ) == []
    assert kb_retrieve.retrieve(
        "   ", embeddings=embeddings, repo=repo, top_k=5, min_score=0.0
    ) == []


def test_cache_refreshed_after_ttl(monkeypatch):
    _setup(monkeypatch)
    # Fake monotonic clock we can advance.
    clock = {"t": 100.0}
    monkeypatch.setattr(kb_retrieve.time, "monotonic", lambda: clock["t"])

    query_vec = _unit([1.0, 0.0])
    embeddings = _FakeEmbeddings({"q": query_vec})

    scan_calls = {"n": 0}
    initial_chunks = [_chunk(0, _unit([1.0, 0.0]), text="first")]

    class CountingRepo:
        def scan_all_chunks(self):
            scan_calls["n"] += 1
            return initial_chunks

    repo = CountingRepo()
    kb_retrieve.retrieve(
        "q", embeddings=embeddings, repo=repo, top_k=5, min_score=0.0
    )
    # Within the TTL window → no re-scan.
    clock["t"] = 200.0  # +100s, under 5-min TTL
    kb_retrieve.retrieve(
        "q", embeddings=embeddings, repo=repo, top_k=5, min_score=0.0
    )
    assert scan_calls["n"] == 1

    # Beyond TTL → re-scan.
    clock["t"] = 100.0 + kb_retrieve.CACHE_TTL_SECONDS + 1
    kb_retrieve.retrieve(
        "q", embeddings=embeddings, repo=repo, top_k=5, min_score=0.0
    )
    assert scan_calls["n"] == 2


def test_empty_cache_uses_short_ttl(monkeypatch):
    """An empty scan result should NOT be cached for the full 5 minutes —
    otherwise a warm Lambda that scanned before the first doc was ingested
    would keep serving 'no KB material' well past the admin upload."""
    _setup(monkeypatch)
    clock = {"t": 100.0}
    monkeypatch.setattr(kb_retrieve.time, "monotonic", lambda: clock["t"])

    query_vec = _unit([1.0, 0.0])
    embeddings = _FakeEmbeddings({"q": query_vec})

    scan_calls = {"n": 0}
    returned: list = []  # empty first, populated later

    class CountingRepo:
        def scan_all_chunks(self):
            scan_calls["n"] += 1
            return list(returned)

    repo = CountingRepo()

    # First call: KB empty → cached as [].
    kb_retrieve.retrieve(
        "q", embeddings=embeddings, repo=repo, top_k=5, min_score=0.0
    )
    assert scan_calls["n"] == 1

    # 5 seconds later: still within EMPTY_CACHE_TTL_SECONDS → no rescan.
    clock["t"] = 105.0
    kb_retrieve.retrieve(
        "q", embeddings=embeddings, repo=repo, top_k=5, min_score=0.0
    )
    assert scan_calls["n"] == 1

    # Past the short TTL (not past the 5-min full TTL) → should rescan.
    clock["t"] = 100.0 + kb_retrieve.EMPTY_CACHE_TTL_SECONDS + 1
    # Simulate admin upload finishing — DDB now has a chunk.
    returned.append(_chunk(0, _unit([1.0, 0.0]), text="new upload"))
    results = kb_retrieve.retrieve(
        "q", embeddings=embeddings, repo=repo, top_k=5, min_score=0.0
    )
    assert scan_calls["n"] == 2
    assert len(results) == 1
    assert results[0].chunk_text == "new upload"
