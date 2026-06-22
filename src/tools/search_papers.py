"""T1: Hybrid search over indexed papers and documents."""

from __future__ import annotations

from src.tools._helpers import fail, map_vector_store_error, ok, search_result_to_dict
from src.tools.types import ToolResult
from src.vector_store import VectorStore as QdrantVectorStore
from src.vector_store.types import Collection, RetrievalSpec, VectorStore

TOOL_NAME = "search_papers"


def search_papers(
    query: str,
    collection: Collection = "papers",
    spec: RetrievalSpec | None = None,
    *,
    vector_store: VectorStore | None = None,
) -> ToolResult:
    store = vector_store or QdrantVectorStore()
    try:
        results = store.search(query, collection, spec)
    except Exception as exc:  # noqa: BLE001
        return map_vector_store_error(TOOL_NAME, exc)

    return ok(
        TOOL_NAME,
        "search_results",
        results=[search_result_to_dict(r) for r in results],
        count=len(results),
    )
