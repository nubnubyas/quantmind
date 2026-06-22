"""§C1 VectorStore shared types (interface contract v1.0)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol


Collection = Literal["papers", "docs", "concepts", "code_snippets"]


@dataclass
class SearchResult:
    # Field order: required fields before fields with defaults (dataclass rule).
    text: str
    fusion_score: float
    doc_id: str
    chunk_id: str
    source: str
    paper_id: str | None
    title: str | None
    section: str | None
    year: int | None
    authors: list[str] = field(default_factory=list)
    url: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalSpec:
    mode: Literal["hybrid", "dense", "sparse"] = "hybrid"
    top_k: int = 5
    prefetch_k: int = 20
    score_threshold: float | None = None
    filters: dict | None = None
    rerank: bool = False


@dataclass
class PointRecord:
    id: str
    text: str
    dense_vector: list[float] | None = None
    sparse_vector: dict | None = None
    payload: dict = field(default_factory=dict)


@dataclass
class UpsertResult:
    upserted_count: int
    point_ids: list[str]
    errors: list[str]


class VectorStoreUnavailable(Exception):
    """Retrieval service is unavailable."""


class CollectionNotFound(Exception):
    """Collection does not exist."""


class InvalidEmbeddingDim(Exception):
    """Vector dimension mismatch."""


class BackendTimeout(Exception):
    """Qdrant timeout."""


class VectorStore(Protocol):
    def search(
        self,
        query: str,
        collection: Collection,
        spec: RetrievalSpec | None = None,
    ) -> list[SearchResult]: ...

    def search_batch(
        self,
        queries: list[str],
        collection: Collection,
        spec: RetrievalSpec | None = None,
    ) -> list[list[SearchResult]]: ...

    def upsert(self, collection: Collection, items: list[PointRecord]) -> UpsertResult: ...

    def delete(self, collection: Collection, ids: list[str]) -> None: ...
