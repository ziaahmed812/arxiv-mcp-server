"""Tests for semantic search and reindex tools."""

import json
from pathlib import Path

import pytest

from arxiv_mcp_server.tools import semantic_search as semantic_module

np = pytest.importorskip("numpy")


class DummyModel:
    """Deterministic embedding model for tests."""

    def encode(self, text, convert_to_numpy=True, normalize_embeddings=True):
        vector = np.array(
            [
                float("transformer" in text.lower()),
                float("vision" in text.lower()),
                float("graph" in text.lower()),
            ],
            dtype=np.float32,
        )
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector


class DummySentenceTransformer(DummyModel):
    """Minimal SentenceTransformer stand-in for tests."""

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs


@pytest.fixture
def semantic_test_env(monkeypatch, temp_storage_args):
    """Configure semantic search module to use a temporary index and dummy model."""
    monkeypatch.setattr(
        semantic_module, "SentenceTransformer", DummySentenceTransformer
    )
    semantic_module._model = None
    return temp_storage_args


@pytest.mark.asyncio
async def test_semantic_search_free_text(semantic_test_env):
    """Semantic text query should rank closest abstract first."""
    semantic_module._upsert_index_record(
        paper_id="2401.00001",
        title="Vision Transformers",
        abstract="transformer model for vision",
        authors=["Author 1"],
        categories=["cs.CV"],
    )
    semantic_module._upsert_index_record(
        paper_id="2401.00002",
        title="Graph Methods",
        abstract="graph neural network approach",
        authors=["Author 2"],
        categories=["cs.LG"],
    )

    response = await semantic_module.handle_semantic_search(
        {"query": "vision transformer", "max_results": 2}
    )

    payload = json.loads(response[0].text)
    assert payload["total_results"] == 2
    assert payload["papers"][0]["id"] == "2401.00001"


@pytest.mark.asyncio
async def test_semantic_search_by_paper_id(semantic_test_env):
    """similar-to-paper mode excludes the source paper from results."""
    semantic_module._upsert_index_record(
        paper_id="2402.00001",
        title="Transformer Baselines",
        abstract="transformer pretraining method",
        authors=["Author 1"],
        categories=["cs.LG"],
    )
    semantic_module._upsert_index_record(
        paper_id="2402.00002",
        title="Vision Transformer Variant",
        abstract="vision transformer architecture",
        authors=["Author 2"],
        categories=["cs.CV"],
    )

    response = await semantic_module.handle_semantic_search(
        {"paper_id": "2402.00001", "max_results": 3}
    )

    payload = json.loads(response[0].text)
    assert payload["mode"] == "similar_to_paper"
    assert all(p["id"] != "2402.00001" for p in payload["papers"])


@pytest.mark.asyncio
async def test_reindex_uses_local_markdown_ids(monkeypatch, semantic_test_env):
    """Reindex should walk bundled markdown files and attempt indexing each ID."""
    bundle_one = semantic_test_env / "2301.00001v1"
    bundle_two = semantic_test_env / "2301.00002v2"
    bundle_one.mkdir(parents=True, exist_ok=True)
    bundle_two.mkdir(parents=True, exist_ok=True)
    Path(bundle_one, "paper.md").write_text("paper", encoding="utf-8")
    Path(bundle_two, "paper.md").write_text("paper", encoding="utf-8")

    indexed_ids = []

    def _mock_index(paper_id):
        indexed_ids.append(paper_id)
        return True

    monkeypatch.setattr(semantic_module, "index_paper_by_id", _mock_index)

    response = await semantic_module.handle_reindex({"clear_existing": True})

    payload = json.loads(response[0].text)
    assert payload["status"] == "success"
    assert set(indexed_ids) == {"2301.00001v1", "2301.00002v2"}


def test_get_model_uses_current_sentence_transformers_signature(
    monkeypatch, semantic_test_env
):
    """Model loading should avoid passing removed constructor kwargs."""
    captured = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs):
            captured["model_name"] = model_name
            captured["kwargs"] = kwargs

    monkeypatch.setattr(semantic_module, "SentenceTransformer", FakeSentenceTransformer)
    semantic_module._model = None

    model = semantic_module._get_model()

    assert isinstance(model, FakeSentenceTransformer)
    assert captured["model_name"] == semantic_module.EMBEDDING_MODEL_NAME
    assert captured["kwargs"] == {}
