"""Qdrant-backed VectorStore implementation (interface contract §C1)."""

from __future__ import annotations

import os
import uuid
from typing import Any

from dotenv import load_dotenv
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    PointStruct,
    Prefetch,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from src.vector_store.types import (
    BackendTimeout,
    Collection,
    CollectionNotFound,
    InvalidEmbeddingDim,
    PointRecord,
    RetrievalSpec,
    SearchResult,
    UpsertResult,
    VectorStoreUnavailable,
)

COLLECTIONS: tuple[Collection, ...] = ("papers", "docs", "concepts", "code_snippets")
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


def make_point_id(paper_id: str, chunk_index: int) -> str:
    """Deterministic UUID5 for idempotent upsert by paper_id + chunk_index."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{paper_id}::{chunk_index}"))


class VectorStore:
    """Hybrid dense+sparse retrieval over Qdrant (RRF fusion)."""

    def __init__(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        embedding_model: str | None = None,
        embedding_dim: int | None = None,
        timeout: float = 30.0,
    ) -> None:
        load_dotenv()
        self._host = host or os.getenv("QDRANT_HOST", "localhost")
        self._port = port or int(os.getenv("QDRANT_PORT", "6333"))
        self._embedding_model = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
        )
        self._embedding_dim = embedding_dim or int(os.getenv("EMBEDDING_DIM", "384"))
        self._timeout = timeout
        self._client: QdrantClient | None = None
        self._dense_model: TextEmbedding | None = None
        self._sparse_model: SparseTextEmbedding | None = None
        self._collections_ready: set[str] = set()

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            try:
                self._client = QdrantClient(
                    host=self._host,
                    port=self._port,
                    timeout=self._timeout,
                )
            except Exception as exc:  # noqa: BLE001
                raise VectorStoreUnavailable(
                    f"Cannot connect to Qdrant at {self._host}:{self._port}"
                ) from exc
        return self._client

    def _get_dense_model(self) -> TextEmbedding:
        if self._dense_model is None:
            self._dense_model = TextEmbedding(self._embedding_model)
        return self._dense_model

    def _get_sparse_model(self) -> SparseTextEmbedding:
        if self._sparse_model is None:
            self._sparse_model = SparseTextEmbedding("Qdrant/bm25")
        return self._sparse_model

    def _embed_dense(self, text: str) -> list[float]:
        vec = list(next(self._get_dense_model().embed([text])).astype(float))
        if len(vec) != self._embedding_dim:
            raise InvalidEmbeddingDim(
                f"Expected dense dim {self._embedding_dim}, got {len(vec)}"
            )
        return vec

    def _embed_sparse(self, text: str) -> SparseVector:
        raw = list(self._get_sparse_model().embed([text]))[0]
        return SparseVector(
            indices=raw.indices.tolist(),
            values=raw.values.tolist(),
        )

    def _sparse_from_dict(self, data: dict) -> SparseVector:
        return SparseVector(indices=data["indices"], values=data["values"])

    def _ensure_collection(self, collection: Collection) -> None:
        if collection in self._collections_ready:
            return
        client = self._get_client()
        try:
            if not client.collection_exists(collection):
                client.create_collection(
                    collection_name=collection,
                    vectors_config={
                        DENSE_VECTOR_NAME: VectorParams(
                            size=self._embedding_dim,
                            distance=Distance.COSINE,
                        ),
                    },
                    sparse_vectors_config={
                        SPARSE_VECTOR_NAME: SparseVectorParams(
                            index=SparseIndexParams(on_disk=False),
                        ),
                    },
                )
            self._collections_ready.add(collection)
        except Exception as exc:  # noqa: BLE001
            if self._is_timeout(exc):
                raise BackendTimeout(f"Timeout ensuring collection {collection}") from exc
            raise VectorStoreUnavailable(str(exc)) from exc

    def _assert_collection_exists(self, collection: Collection) -> None:
        try:
            if not self._get_client().collection_exists(collection):
                raise CollectionNotFound(f"Collection '{collection}' does not exist")
        except CollectionNotFound:
            raise
        except Exception as exc:  # noqa: BLE001
            if self._is_timeout(exc):
                raise BackendTimeout(f"Timeout checking collection {collection}") from exc
            raise VectorStoreUnavailable(str(exc)) from exc

    @staticmethod
    def _is_timeout(exc: Exception) -> bool:
        name = type(exc).__name__
        msg = str(exc).lower()
        return "timeout" in name.lower() or "deadline" in msg or "timed out" in msg

    def _build_filter(self, filters: dict | None) -> Filter | None:
        if not filters:
            return None
        conditions = [
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in filters.items()
        ]
        return Filter(must=conditions) if conditions else None

    def _point_to_search_result(self, point: Any) -> SearchResult:
        payload = point.payload or {}
        authors = payload.get("authors") or []
        if not isinstance(authors, list):
            authors = [authors]
        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        return SearchResult(
            text=payload.get("text", ""),
            fusion_score=float(point.score or 0.0),
            doc_id=payload.get("doc_id", ""),
            chunk_id=payload.get("chunk_id", ""),
            source=payload.get("source", ""),
            paper_id=payload.get("paper_id"),
            title=payload.get("title"),
            section=payload.get("section"),
            year=payload.get("year"),
            authors=authors,
            url=payload.get("url"),
            metadata=metadata,
        )

    def search(
        self,
        query: str,
        collection: Collection,
        spec: RetrievalSpec | None = None,
    ) -> list[SearchResult]:
        """Hybrid, dense, or sparse search over a collection."""
        spec = spec or RetrievalSpec()
        self._assert_collection_exists(collection)
        q_filter = self._build_filter(spec.filters)

        try:
            dense_vec = self._embed_dense(query)
            sparse_vec = self._embed_sparse(query)
            client = self._get_client()

            if spec.mode == "dense":
                response = client.query_points(
                    collection_name=collection,
                    query=dense_vec,
                    using=DENSE_VECTOR_NAME,
                    limit=spec.top_k,
                    query_filter=q_filter,
                    with_payload=True,
                    score_threshold=spec.score_threshold,
                )
            elif spec.mode == "sparse":
                response = client.query_points(
                    collection_name=collection,
                    query=sparse_vec,
                    using=SPARSE_VECTOR_NAME,
                    limit=spec.top_k,
                    query_filter=q_filter,
                    with_payload=True,
                    score_threshold=spec.score_threshold,
                )
            else:
                response = client.query_points(
                    collection_name=collection,
                    prefetch=[
                        Prefetch(
                            query=dense_vec,
                            using=DENSE_VECTOR_NAME,
                            limit=spec.prefetch_k,
                            filter=q_filter,
                        ),
                        Prefetch(
                            query=sparse_vec,
                            using=SPARSE_VECTOR_NAME,
                            limit=spec.prefetch_k,
                            filter=q_filter,
                        ),
                    ],
                    query=FusionQuery(fusion=Fusion.RRF),
                    limit=spec.top_k,
                    query_filter=q_filter,
                    with_payload=True,
                    score_threshold=spec.score_threshold,
                )
            return [self._point_to_search_result(p) for p in response.points]
        except (CollectionNotFound, InvalidEmbeddingDim, BackendTimeout):
            raise
        except Exception as exc:  # noqa: BLE001
            if self._is_timeout(exc):
                raise BackendTimeout(f"Search timed out on {collection}") from exc
            raise VectorStoreUnavailable(str(exc)) from exc

    def search_batch(
        self,
        queries: list[str],
        collection: Collection,
        spec: RetrievalSpec | None = None,
    ) -> list[list[SearchResult]]:
        """Run search for each query independently."""
        return [self.search(q, collection, spec) for q in queries]

    def upsert(self, collection: Collection, items: list[PointRecord]) -> UpsertResult:
        """Upsert points; auto-embed when vectors are omitted."""
        if not items:
            return UpsertResult(upserted_count=0, point_ids=[], errors=[])

        self._ensure_collection(collection)
        points: list[PointStruct] = []
        errors: list[str] = []
        point_ids: list[str] = []

        for item in items:
            try:
                dense = item.dense_vector
                if dense is None:
                    dense = self._embed_dense(item.text)
                elif len(dense) != self._embedding_dim:
                    raise InvalidEmbeddingDim(
                        f"Point {item.id}: expected dim {self._embedding_dim}, "
                        f"got {len(dense)}"
                    )

                if item.sparse_vector is None:
                    sparse = self._embed_sparse(item.text)
                else:
                    sparse = self._sparse_from_dict(item.sparse_vector)

                payload = {"text": item.text, **item.payload}
                points.append(
                    PointStruct(
                        id=item.id,
                        vector={DENSE_VECTOR_NAME: dense, SPARSE_VECTOR_NAME: sparse},
                        payload=payload,
                    )
                )
                point_ids.append(item.id)
            except InvalidEmbeddingDim:
                raise
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{item.id}: {exc}")

        if not points:
            return UpsertResult(upserted_count=0, point_ids=[], errors=errors)

        try:
            self._get_client().upsert(collection_name=collection, points=points)
        except Exception as exc:  # noqa: BLE001
            if self._is_timeout(exc):
                raise BackendTimeout(f"Upsert timed out on {collection}") from exc
            raise VectorStoreUnavailable(str(exc)) from exc

        return UpsertResult(
            upserted_count=len(points),
            point_ids=point_ids,
            errors=errors,
        )

    def delete(self, collection: Collection, ids: list[str]) -> None:
        """Delete points by id from a collection."""
        if not ids:
            return
        self._assert_collection_exists(collection)
        try:
            self._get_client().delete(
                collection_name=collection,
                points_selector=qmodels.PointIdsList(points=ids),
            )
        except CollectionNotFound:
            raise
        except Exception as exc:  # noqa: BLE001
            if self._is_timeout(exc):
                raise BackendTimeout(f"Delete timed out on {collection}") from exc
            raise VectorStoreUnavailable(str(exc)) from exc
