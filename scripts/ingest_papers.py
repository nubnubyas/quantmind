#!/usr/bin/env python3
"""Ingest papers into Qdrant and verify collection health."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.ingestion.pipeline import SEED_PAPER_IDS, pipeline, upsert_chunks
from src.vector_store.qdrant_client_wrapper import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VERIFICATION_QUERIES = [
    ("动量因子在A股市场", "momentum / A-share"),
    ("Sharpe Ratio calculation", "Sharpe ratio"),
    ("Fama-French five factor model", "Fama-French"),
    ("low volatility anomaly", "low vol"),
    ("backtrader dual moving average", "backtest"),
]


def main() -> None:
    store = VectorStore()

    # 1. Verify Qdrant reachable
    hits = store.search("test", "papers")
    print(f"Qdrant papers collection: reachable ({len(hits)} hits for probe query)")

    # 2. Ingest
    print(f"Ingesting {len(SEED_PAPER_IDS)} papers...")
    records = pipeline(SEED_PAPER_IDS)
    unique_papers = {r.paper_id for r in records}
    print(f"Pipeline produced {len(records)} chunks from {len(unique_papers)} papers")

    if not records:
        print("ERROR: No chunks produced!")
        sys.exit(1)

    # 3. Upsert
    result = upsert_chunks(store, records)
    print(f"Upserted {result.upserted_count} points, errors: {result.errors}")

    if result.errors:
        print("WARNING: Some upsert errors occurred; check logs above.")

    # 4. Verify with diverse queries
    print("\nVerification queries:")
    for query, topic in VERIFICATION_QUERIES:
        hits = store.search(query, "papers")
        if hits:
            top_titles = [h.title[:60] for h in hits[:3] if h.title]
            print(f"  [{topic}] -> {top_titles}")
        else:
            print(f"  [{topic}] -> NO RESULTS (data gap!)")

    print(f"\nUnique papers ingested this run: {len(unique_papers)}")
    if len(unique_papers) < 20:
        print(f"WARNING: Expected >= 20 papers, got {len(unique_papers)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
