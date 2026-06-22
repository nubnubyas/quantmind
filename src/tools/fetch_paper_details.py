"""T2: Fetch paper metadata and abstract from Qdrant payload."""

from __future__ import annotations

from src.tools._helpers import fail, map_vector_store_error, ok
from src.tools.types import ToolResult
from src.vector_store import VectorStore as QdrantVectorStore
from src.vector_store.types import RetrievalSpec, SearchResult, VectorStore

TOOL_NAME = "fetch_paper_details"


def _merge_chunks(results: list[SearchResult]) -> dict:
    title = None
    authors: list[str] = []
    year = None
    url = None
    source = None
    texts: list[str] = []

    for r in results:
        texts.append(r.text)
        if title is None and r.title:
            title = r.title
        if not authors and r.authors:
            authors = r.authors
        if year is None and r.year is not None:
            year = r.year
        if url is None and r.url:
            url = r.url
        if source is None and r.source:
            source = r.source

    return {
        "title": title,
        "authors": authors,
        "year": year,
        "url": url,
        "source": source,
        "abstract": "\n\n".join(texts),
        "chunk_count": len(results),
    }


def fetch_paper_details(
    paper_id: str,
    *,
    vector_store: VectorStore | None = None,
) -> ToolResult:
    store = vector_store or QdrantVectorStore()
    spec = RetrievalSpec(top_k=20, filters={"paper_id": paper_id})

    try:
        results = store.search(paper_id, "papers", spec)
    except Exception as exc:  # noqa: BLE001
        return map_vector_store_error(TOOL_NAME, exc)

    if not results:
        return fail(TOOL_NAME, f"Paper '{paper_id}' not found", "NOT_FOUND")

    merged = _merge_chunks(results)
    return ok(
        TOOL_NAME,
        "paper_details",
        paper_id=paper_id,
        title=merged["title"],
        authors=merged["authors"],
        year=merged["year"],
        url=merged["url"],
        source=merged["source"],
        abstract=merged["abstract"],
        chunk_count=merged["chunk_count"],
    )
