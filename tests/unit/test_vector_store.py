"""Unit tests for Qdrant VectorStore (mocked client, no live Docker required)."""

from __future__ import annotations

from dataclasses import fields
from unittest.mock import MagicMock

import pytest

from src.vector_store.qdrant_client_wrapper import VectorStore
from src.vector_store.types import (
    CollectionNotFound,
    InvalidEmbeddingDim,
    PointRecord,
    RetrievalSpec,
    SearchResult,
)


def _mock_scored_point(score: float, payload: dict) -> MagicMock:
    point = MagicMock()
    point.score = score
    point.payload = payload
    return point


def _mock_query_response(points: list) -> MagicMock:
    response = MagicMock()
    response.points = points
    return response


@pytest.fixture
def store() -> VectorStore:
    """VectorStore with mocked Qdrant client and embedding models."""
    vs = VectorStore(host="localhost", port=6333, embedding_dim=384)
    vs._client = MagicMock()
    vs._client.collection_exists.return_value = True
    vs._dense_model = MagicMock()
    vs._sparse_model = MagicMock()

    def _dense_embed(_texts):
        dense_vec = MagicMock()
        dense_vec.astype.return_value = [0.1] * 384
        return iter([dense_vec])

    def _sparse_embed(_texts):
        sparse_raw = MagicMock()
        sparse_raw.indices.tolist.return_value = [1, 2]
        sparse_raw.values.tolist.return_value = [0.5, 0.3]
        return [sparse_raw]

    vs._dense_model.embed.side_effect = _dense_embed
    vs._sparse_model.embed.side_effect = _sparse_embed

    return vs


def test_search_hybrid_maps_fusion_score(store: VectorStore) -> None:
    payload = {
        "text": "Fama-French three-factor model",
        "doc_id": "ff1993",
        "chunk_id": "ff1993::0",
        "source": "arxiv",
        "paper_id": "ff1993",
        "title": "Common risk factors",
        "section": "Introduction",
        "year": 1993,
        "authors": ["Fama", "French"],
        "url": "https://example.com",
    }
    store._client.query_points.return_value = _mock_query_response(
        [_mock_scored_point(0.87, payload)]
    )

    results = store.search("Fama-French factor model", "papers")
    assert len(results) == 1
    assert results[0].fusion_score == 0.87
    assert results[0].text == payload["text"]
    assert results[0].paper_id == "ff1993"
    assert "FusionQuery" in str(store._client.query_points.call_args)


def test_search_empty_returns_list(store: VectorStore) -> None:
    store._client.query_points.return_value = _mock_query_response([])
    results = store.search("nonexistent query xyz", "papers")
    assert results == []


def test_search_dense_mode(store: VectorStore) -> None:
    store._client.query_points.return_value = _mock_query_response([])
    store.search("test", "papers", RetrievalSpec(mode="dense", top_k=3))
    call_kwargs = store._client.query_points.call_args.kwargs
    assert call_kwargs["using"] == "dense"
    assert call_kwargs["limit"] == 3


def test_upsert_auto_embed(store: VectorStore) -> None:
    store._client.collection_exists.return_value = False
    item = PointRecord(
        id="test-uuid",
        text="Momentum strategy buys winners",
        payload={"doc_id": "p1", "chunk_id": "p1::0", "source": "arxiv"},
    )
    result = store.upsert("papers", [item])
    assert result.upserted_count == 1
    assert result.point_ids == ["test-uuid"]
    store._client.upsert.assert_called_once()
    store._dense_model.embed.assert_called()
    store._sparse_model.embed.assert_called()


def test_upsert_invalid_dim_raises(store: VectorStore) -> None:
    item = PointRecord(
        id="bad-dim",
        text="text",
        dense_vector=[0.1] * 128,
        payload={},
    )
    with pytest.raises(InvalidEmbeddingDim):
        store.upsert("papers", [item])


def test_collection_not_found(store: VectorStore) -> None:
    store._client.collection_exists.return_value = False
    with pytest.raises(CollectionNotFound):
        store.search("query", "papers")


def test_search_result_field_order() -> None:
    """SearchResult field order matches interface contract §C1."""
    names = [f.name for f in fields(SearchResult)]
    required = [
        "text",
        "fusion_score",
        "doc_id",
        "chunk_id",
        "source",
        "paper_id",
        "title",
        "section",
        "year",
    ]
    defaults = ["authors", "url", "metadata"]
    assert names[: len(required)] == required
    assert names[len(required) :] == defaults


def test_search_batch(store: VectorStore) -> None:
    store._client.query_points.return_value = _mock_query_response([])
    batch = store.search_batch(["q1", "q2"], "papers")
    assert len(batch) == 2
    assert store._client.query_points.call_count == 2


def test_delete(store: VectorStore) -> None:
    store.delete("papers", ["id-1", "id-2"])
    store._client.delete.assert_called_once()


def test_delete_empty_ids_noop(store: VectorStore) -> None:
    store.delete("papers", [])
    store._client.delete.assert_not_called()
