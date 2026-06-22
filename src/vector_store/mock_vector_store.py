"""Mock VectorStore for B1 development (interface contract §C1)."""

from __future__ import annotations

from copy import deepcopy

from src.vector_store.types import (
    Collection,
    PointRecord,
    RetrievalSpec,
    SearchResult,
    UpsertResult,
)


def _base_results() -> list[SearchResult]:
    return [
        SearchResult(
            text=(
                "The Fama-French three-factor model explains stock returns using "
                "market excess return (Rm-Rf), size (SMB), and value (HML) factors. "
                "It extends CAPM by accounting for the size and book-to-market premiums."
            ),
            fusion_score=0.92,
            doc_id="ff1993",
            chunk_id="ff1993-c0",
            source="arxiv",
            paper_id="ff1993",
            title="Fama-French Three-Factor Model",
            section="Introduction",
            year=1993,
            authors=["Eugene Fama", "Kenneth French"],
            url="https://example.com/ff1993",
        ),
        SearchResult(
            text=(
                "The Sharpe ratio measures risk-adjusted return as excess return per unit "
                "of volatility. Higher Sharpe indicates better return per unit of risk."
            ),
            fusion_score=0.78,
            doc_id="sharpe1966",
            chunk_id="sharpe1966-c0",
            source="manual",
            paper_id="sharpe1966",
            title="Sharpe Ratio and Risk-Adjusted Returns",
            section="Definition",
            year=1966,
            authors=["William Sharpe"],
        ),
        SearchResult(
            text=(
                "Momentum investing exploits the tendency of assets with strong recent "
                "performance to continue outperforming in the short to medium term."
            ),
            fusion_score=0.71,
            doc_id="mom1997",
            chunk_id="mom1997-c0",
            source="arxiv",
            paper_id="mom1997",
            title="Momentum Factor Anomaly",
            section="Empirical Evidence",
            year=1997,
            authors=["Narasimhan Jegadeesh", "Sheridan Titman"],
        ),
    ]


class MockVectorStore:
    """Hardcoded SearchResults for Research Subgraph development."""

    def __init__(self, *, low_confidence: bool = False) -> None:
        self._low_confidence = low_confidence
        self._depth_fetched = False

    def mark_depth_fetched(self) -> None:
        self._depth_fetched = True

    def _build_results(self, spec: RetrievalSpec | None) -> list[SearchResult]:
        results = deepcopy(_base_results())
        if self._low_confidence:
            results[0].fusion_score = 0.1
        elif self._depth_fetched:
            results[0].fusion_score = min(0.95, results[0].fusion_score + 0.05)
            extra = deepcopy(results[0])
            extra.chunk_id = "ff1993-c1"
            extra.section = "Factor Definitions"
            extra.fusion_score = 0.88
            results.append(extra)

        top_k = (spec.top_k if spec else 5) or 5
        return results[:top_k]

    def search(
        self,
        query: str,
        collection: Collection,
        spec: RetrievalSpec | None = None,
    ) -> list[SearchResult]:
        del query, collection
        return self._build_results(spec)

    def search_batch(
        self,
        queries: list[str],
        collection: Collection,
        spec: RetrievalSpec | None = None,
    ) -> list[list[SearchResult]]:
        return [self.search(q, collection, spec) for q in queries]

    def upsert(self, collection: Collection, items: list[PointRecord]) -> UpsertResult:
        del collection
        return UpsertResult(upserted_count=0, point_ids=[], errors=[])

    def delete(self, collection: Collection, ids: list[str]) -> None:
        del collection, ids
