"""End-to-end ingestion: fetch → parse → chunk → optional Qdrant upsert."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from src.ingestion.arxiv_fetcher import fetch_paper
from src.ingestion.chunker import chunk_blocks
from src.ingestion.pdf_parser import parse_pdf
from src.vector_store.qdrant_client_wrapper import VectorStore, make_point_id
from src.vector_store.types import PointRecord, UpsertResult

logger = logging.getLogger(__name__)

# Five stable q-fin papers for local ingestion acceptance.
SEED_PAPER_IDS: list[str] = [
    "1208.2775",   # Price momentum strategy (Barroso & Santa-Clara)
    "2002.01800",  # Sharpe Ratio Analysis in High Dimensions
    "2006.02467",  # Fama-French three-factor regression model
    "0801.4047",   # No arbitrage conditions for trading strategies
    "1511.07101",  # Risk-return empirical study (statistical methods)
]

DEFAULT_CACHE_DIR = Path("data/sample_papers")


@dataclass
class ChunkRecord:
    """One chunk ready for vector indexing."""

    id: str
    paper_id: str
    chunk_index: int
    text: str
    section: str | None
    title: str
    authors: list[str]
    year: int | None
    url: str
    source: str = "arxiv"
    doc_id: str = ""
    chunk_id: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.doc_id:
            self.doc_id = self.paper_id
        if not self.chunk_id:
            self.chunk_id = f"{self.paper_id}::{self.chunk_index}"


def _metadata_to_chunk(
    meta,
    chunk,
) -> ChunkRecord:
    point_id = make_point_id(meta.paper_id, chunk.chunk_index)
    return ChunkRecord(
        id=point_id,
        paper_id=meta.paper_id,
        chunk_index=chunk.chunk_index,
        text=chunk.text,
        section=chunk.section,
        title=meta.title,
        authors=meta.authors,
        year=meta.year,
        url=meta.url,
        doc_id=meta.paper_id,
        chunk_id=f"{meta.paper_id}::{chunk.chunk_index}",
    )


def pipeline(
    paper_ids: list[str],
    *,
    cache_dir: Path | None = None,
) -> list[ChunkRecord]:
    """Fetch, parse, and chunk papers; idempotent by paper_id + chunk_index."""
    cache = cache_dir or DEFAULT_CACHE_DIR
    all_chunks: list[ChunkRecord] = []

    for paper_id in paper_ids:
        try:
            meta = fetch_paper(paper_id, cache)
            blocks = parse_pdf(meta.pdf_path)
            text_chunks = chunk_blocks(blocks)
            if not text_chunks and meta.abstract:
                from src.ingestion.pdf_parser import TextBlock

                text_chunks = chunk_blocks(
                    [TextBlock(text=meta.abstract, section="Abstract", page_num=0)]
                )
            for tc in text_chunks:
                if tc.text.strip():
                    all_chunks.append(_metadata_to_chunk(meta, tc))
            logger.info(
                "Ingested %s: %d chunks", meta.paper_id, len(text_chunks)
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to ingest %s: %s", paper_id, exc)

    return all_chunks


def upsert_chunks(
    vector_store: VectorStore,
    chunks: list[ChunkRecord],
) -> UpsertResult:
    """Upsert chunks into the papers collection (idempotent point ids)."""
    items = [
        PointRecord(
            id=chunk.id,
            text=chunk.text,
            payload={
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "paper_id": chunk.paper_id,
                "title": chunk.title,
                "section": chunk.section,
                "year": chunk.year,
                "authors": chunk.authors,
                "url": chunk.url,
                "metadata": chunk.metadata,
            },
        )
        for chunk in chunks
    ]
    return vector_store.upsert("papers", items)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    store = VectorStore()
    records = pipeline(SEED_PAPER_IDS)
    print(f"Pipeline produced {len(records)} chunks from {len(SEED_PAPER_IDS)} papers")
    if not records:
        raise SystemExit("No chunks produced; check arXiv fetch errors above.")
    result = upsert_chunks(store, records)
    print(f"Upserted {result.upserted_count} points, errors: {result.errors}")
    hits = store.search("Fama-French factor model", "papers")
    if hits:
        print(f"Top hit fusion_score={hits[0].fusion_score:.4f}: {hits[0].text[:80]}...")
    else:
        print("No search hits (unexpected after upsert).")
