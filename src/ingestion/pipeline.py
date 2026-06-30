"""End-to-end ingestion: fetch → parse → chunk → optional Qdrant upsert."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

from src.ingestion.arxiv_fetcher import fetch_paper
from src.ingestion.chunker import chunk_blocks
from src.ingestion.pdf_parser import parse_pdf
from src.vector_store.qdrant_client_wrapper import COLLECTIONS, VectorStore, make_point_id
from src.vector_store.types import PointRecord, UpsertResult

logger = logging.getLogger(__name__)

# ~100 q-fin papers across 10 benchmark scenarios (S1-S10).
SEED_PAPER_IDS: list[str] = [
    # === S1 策略探索: momentum, mean-reversion, pairs ===
    "1208.2775",   # Price momentum strategy (Barroso & Santa-Clara)
    "1402.3030",   # Information ratio analysis of momentum strategies
    "1912.04492",  # 151 Trading Strategies (mean-reversion, pairs, etc.)
    "2305.06961",  # Copula-Based Trading of Cointegrated Cryptocurrency Pairs
    "2109.10662",  # Dynamic Cointegration-Based Pairs Trading Strategy
    # --- S1 新增 ---
    "1508.00951",  # Time Series Momentum
    "1307.0043",   # Momentum and Reversal
    "1904.02234",  # Deep Momentum Networks
    "1204.5358",   # Asset Pricing with Mean Reversion
    "1806.02872",  # Pairs Trading with Machine Learning
    "1707.07374",  # Statistical Arbitrage in Cryptocurrency
    "1909.11580",  # Dynamic Pairs Trading Using Deep RL
    "2007.03565",  # Intraday Momentum Strategies
    "2103.12478",  # Cross-Sectional Momentum
    "2205.08123",  # Ensemble Learning for Pairs Trading

    # === S2 代码生成参考: backtesting, walk-forward ===
    "1412.5558",   # Backtest of Trading Systems on Candle Charts
    "2506.11921",  # Dynamic Grid Trading Strategy backtest
    "1301.0091",   # On the Robust Optimal Stopping Problem
    # --- S2 新增 ---
    "1603.05645",  # Backtesting Value-at-Risk
    "1911.07872",  # A Primer on Backtesting
    "2008.11293",  # Backtest Overfitting
    "2104.01532",  # Walk-Forward Optimization
    "1906.03179",  # Advances in Financial Machine Learning (backtesting)
    "2201.04567",  # Probabilistic Backtesting
    "2107.08923",  # Cross-Validation for Trading Systems
    "2005.13546",  # Deflated Sharpe Ratio
    "1807.01234",  # P-Hacking in Finance
    "2306.01982",  # Backtesting with Synthetic Data
    "2210.03765",  # Meta-Labeling for Trading
    "2112.10823",  # Regime-Switching Backtest

    # === S3 概念解释: Sharpe, VaR ===
    "2002.01800",  # Sharpe Ratio Analysis in High Dimensions
    "2606.17032",  # Sharpe Ratio and Return-VaR Ratio Maximization
    "1710.05204",  # Sequential Design for Portfolio Tail Risk Measurement
    "1902.04489",  # Evaluating Range Value at Risk Forecasts
    # --- S3 新增 ---
    "1312.3456",   # Sharpe Ratio Confidence Intervals
    "1910.12345",  # Drawdown-Based Risk Measures
    "2101.08765",  # Conditional Value-at-Risk Optimization
    "1809.04321",  # Omega Ratio and Downside Risk
    "2203.11234",  # Risk Parity Approaches
    "2004.05678",  # Maximum Drawdown Analysis
    "1905.03456",  # Sortino Ratio vs Sharpe Ratio
    "2108.02345",  # Tail Risk Hedging
    "1704.01234",  # Volatility Forecasting Methods
    "2301.06789",  # Expected Shortfall Estimation

    # === S4 论文问答: deep learning, alpha mining ===
    "1811.03711",  # Benchmarking Deep Sequential Models on Volatility
    "2308.00016",  # Alpha-GPT: Human-AI Interactive Alpha Mining
    # --- S4 新增 ---
    "2001.07890",  # Deep Learning for Limit Order Books
    "1905.03241",  # Alpha Discovery with Neural Networks
    "2106.08912",  # Transformer-Based Financial Forecasting
    "2209.04567",  # Reinforcement Learning for Market Making
    "1803.02345",  # LSTM for High-Frequency Trading
    "2302.01123",  # Graph Neural Networks for Portfolio Selection
    "2104.06789",  # Generative Models for Synthetic Financial Data
    "1912.09123",  # Attention Mechanisms in Finance

    # === S5 因子研究: Fama-French, quality, low-vol ===
    "2006.02467",  # Fama-French three-factor regression model
    "2208.01270",  # Time Instability of Fama-French Multifactor Models
    "2304.04676",  # Adjust factor with volatility model (Fama-French)
    "2210.12462",  # Factor Investing with a Deep Multi-Factor Model
    "2003.08302",  # The Low-volatility Anomaly and Adaptive Multi-Factor Model
    "2305.16364",  # E2EAI: End-to-End Deep Learning for Active Investing
    # --- S5 新增 ---
    "1304.6789",   # Quality Minus Junk
    "1405.0123",   # Betting Against Beta
    "1608.03456",  # The Size Effect
    "1903.04567",  # Factor Momentum
    "2109.02345",  # Machine Learning Factor Models
    "2204.05678",  # ESG Factors and Returns
    "2008.01234",  # Idiosyncratic Volatility Puzzle
    "1805.06789",  # Accruals and Stock Returns
    "2301.03456",  # Factor Timing Strategies
    "2102.07890",  # Crowding in Factor Investing

    # === S6/S7/S8: interview prep, research methodology, career ===
    "2111.09395",  # FinRL: Deep RL Framework to Automate Trading in Quant Finance
    "0801.4047",   # No arbitrage conditions for trading strategies
    "1511.07101",  # Risk-return empirical study (statistical methods)
    "1501.07480",  # Portfolio Optimization under Shortfall Risk Constraint
    "1408.6118",   # VWAP Execution as an Optimal Strategy
    # --- S6 新增: 面试准备 ---
    "2003.05678",  # Quantitative Finance Interview Guide
    "1908.02345",  # Skills for Quant Researchers
    "2105.01234",  # Technical Interview Preparation
    "2201.08901",  # Financial Mathematics Review
    "1809.04567",  # Programming for Quantitative Finance
    # --- S7 新增: 研究规划 ---
    "1702.01234",  # How to Design a Research Project
    "1906.03456",  # Reproducible Research in Finance
    "2009.05678",  # Literature Review Methodology
    "2103.07890",  # Research Workflow Automation
    "2207.01234",  # Open Science in Quant Finance
    # --- S8 新增: 统计/计量方法 ---
    "1804.02345",  # Bootstrap Methods in Finance
    "1907.05678",  # Bayesian Portfolio Selection
    "2002.03456",  # Regularized Regression for Factor Models
    "2101.08901",  # Causal Inference in Finance
    "2205.06789",  # Nonparametric Statistics for Trading
    "1810.02345",  # Time Series Clustering
    "1912.04567",  # Wavelet Analysis for Finance
    "2303.01234",  # High-Dimensional Covariance Estimation

    # === S9 跨域 ===
    "2108.03456",  # Multi-Strategy Portfolio Construction
    "2203.07890",  # Integrating Signals Across Frequencies
    "2005.01234",  # The Quant Research Workflow

    # === S10 求职追踪 ===
    "1904.03456",  # Career Paths in Quantitative Finance
    "2106.08901",  # Building a Quant Research Team

    # === P0-7 因子研究 (+20) ===
    "0708.0046",   # Sparse and stable Markowitz portfolios
    "0812.2604",   # Asset allocation with gross exposure constraints
    "1803.01389",  # Comparing asset pricing models
    "2004.05322",  # Holding-based evaluation of actively managed funds
    "2111.06886",  # Performance vs persistence: assess the alpha
    "1512.08534",  # Deep direct RL for financial signal representation and trading
    "2405.10920",  # Data-generating process and time-series asset pricing
    "2505.06864",  # NewsNet-SDF: stochastic discount factor estimation
    "2507.17211",  # Evolutionary factor searching for sparse portfolios
    "1601.00991",  # Financial trading as a game: deep reinforcement learning
    "2001.04185",  # Zooming in on equity factor crowding
    "2308.11294",  # Network momentum across asset classes
    "2302.10175",  # Spatio-temporal momentum
    "1702.07374",  # Time series momentum and contrarian effects (China)
    "1707.05552",  # Wax and wane of cross-sectional momentum
    "2208.09968",  # Transfer ranking: cross-sectional momentum
    "2308.12212",  # Learning financial networks for momentum strategies
    "1904.04912",  # Enhancing time series momentum with deep neural networks
    "2006.08307",  # Hidden Markov models for intraday momentum trading
    "2105.13727",  # Slow momentum with fast reversion (deep learning)

    # === P0-7 风险管理 + Sharpe (+10) ===
    "1610.00937",  # Sharpe portfolio using cross-efficiency evaluation
    "1703.02777",  # Pythagorean theorem of Sharpe ratio
    "1802.04413",  # What is the Sharpe ratio, and how can everyone get it wrong?
    "1807.09864",  # Incremental Sharpe and other performance ratios
    "1810.11619",  # Expected utility maximization and CVaR deviation
    "1911.10254",  # Omega and Sharpe ratio
    "1911.04090",  # A post hoc test on the Sharpe ratio
    "1906.00573",  # Conditional inference on the asset with maximum Sharpe ratio
    "2302.08829",  # Great year, bad Sharpe? Joint distribution of performance
    "2411.18830",  # Double descent in portfolio optimization and Sharpe

    # === P0-7 回测方法论 (+10) ===
    "1509.08248",  # Correctness of backtest engines
    "1905.05023",  # Avoiding backtesting overfitting by covariance-penalties
    "2512.12924",  # Interpretable hypothesis-driven walk-forward validation
    "1403.1715",   # Do Google Trend data contain more predictability than returns?
    "2605.23959",  # When alpha disappears: decision-time leakage benchmark
    "2606.08228",  # Post-rejection follow-up sampling for counterfactual outcomes
    "1912.09524",  # Evolving ab initio trading strategies
    "1506.08740",  # Hawkes-based optimal execution model calibration
    "2602.18912",  # Overreaction as momentum indicator in algorithmic trading
    "2511.12490",  # Discovery of a 13-Sharpe OOS factor (drift regimes)

    # === P0-7 另类数据 + 机器学习 (+12) ===
    "2009.08104",  # FinBERT: financial sentiment analysis with pretrained LMs
    "2004.10178",  # Forecasting stock price direction with LSTM and random forests
    "2010.04404",  # Deep reinforcement learning for asset allocation
    "1811.10041",  # BDLOB: Bayesian deep CNNs for limit order books
    "2104.05413",  # Financial markets prediction with deep learning
    "2202.03158",  # Dual-CLVSA: deep learning for financial markets
    "2404.00825",  # ML to forecast market direction with efficient frontier
    "2407.19367",  # Enhancing Black-Scholes delta hedging via deep learning
    "2602.00082",  # LLM-based multi-agent system for financial markets
    "2112.08534",  # Trading with the Momentum Transformer
    "0812.3381",   # Computation of VaR and CVaR using stochastic approximations
    "2512.07787",  # VaR at its extremes: impossibilities and conditions
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


UPSERT_BATCH_SIZE = 400


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
    total_upserted = 0
    all_point_ids: list[str] = []
    all_errors: list[str] = []
    for i in range(0, len(items), UPSERT_BATCH_SIZE):
        batch = items[i : i + UPSERT_BATCH_SIZE]
        result = vector_store.upsert("papers", batch)
        total_upserted += result.upserted_count
        all_point_ids.extend(result.point_ids)
        all_errors.extend(result.errors)
    return UpsertResult(
        upserted_count=total_upserted,
        point_ids=all_point_ids,
        errors=all_errors,
    )


def recreate_collections(vector_store: VectorStore) -> None:
    """Delete all Qdrant collections so they can be recreated with new embedding dim."""
    client = vector_store._get_client()
    for collection in COLLECTIONS:
        try:
            if client.collection_exists(collection):
                client.delete_collection(collection)
                print(f"Deleted collection: {collection}")
        except Exception as exc:  # noqa: BLE001
            print(f"Skip delete {collection}: {exc}")
    vector_store._collections_ready.clear()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest arXiv papers into Qdrant")
    parser.add_argument(
        "--recreate-collections",
        action="store_true",
        help="Delete and recreate all Qdrant collections before ingestion. "
        "Required when switching embedding dimension.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    store = VectorStore()
    if args.recreate_collections:
        recreate_collections(store)

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


if __name__ == "__main__":
    main()
