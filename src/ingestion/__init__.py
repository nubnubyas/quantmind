"""Data ingestion pipeline for arXiv papers."""

from src.ingestion.pipeline import ChunkRecord, SEED_PAPER_IDS, pipeline, upsert_chunks

__all__ = ["ChunkRecord", "SEED_PAPER_IDS", "pipeline", "upsert_chunks"]
