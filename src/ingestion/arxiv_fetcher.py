"""arXiv paper search and PDF download for the ingestion pipeline."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import arxiv
import requests

_CLIENT = arxiv.Client(delay_seconds=3.0, num_retries=3)
_USER_AGENT = "QuantMind/1.0 (arxiv ingestion)"


@dataclass
class PaperMetadata:
    """Metadata and local path for a fetched arXiv paper."""

    paper_id: str
    title: str
    authors: list[str]
    year: int | None
    url: str
    pdf_path: Path
    abstract: str


def normalize_paper_id(paper_id: str) -> str:
    """Strip arxiv: prefix and version suffix (e.g. 2310.12345v2 -> 2310.12345)."""
    pid = paper_id.strip()
    if pid.lower().startswith("arxiv:"):
        pid = pid[6:]
    pid = re.sub(r"v\d+$", "", pid)
    return pid


def _download_pdf(url: str, dest: Path) -> None:
    """Download PDF bytes from arXiv."""
    resp = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


def fetch_paper(paper_id: str, cache_dir: Path) -> PaperMetadata:
    """Download a paper PDF (cached) and return metadata."""
    pid = normalize_paper_id(paper_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = cache_dir / f"{pid}.pdf"

    search = arxiv.Search(id_list=[pid])
    results = list(_CLIENT.results(search))
    if not results:
        raise ValueError(f"Paper not found on arXiv: {pid}")

    paper = results[0]
    if not pdf_path.exists():
        if not paper.pdf_url:
            raise ValueError(f"No PDF URL for paper: {pid}")
        _download_pdf(paper.pdf_url, pdf_path)
        time.sleep(3)  # respect arXiv rate limit after PDF fetch

    year = paper.published.year if paper.published else None
    return PaperMetadata(
        paper_id=pid,
        title=paper.title,
        authors=[a.name for a in paper.authors],
        year=year,
        url=paper.entry_id,
        pdf_path=pdf_path,
        abstract=paper.summary,
    )


def search_quant_papers(query: str, max_results: int = 10) -> list[PaperMetadata]:
    """Search q-fin category papers; metadata only (no PDF download)."""
    search = arxiv.Search(
        query=f"cat:q-fin.* AND ({query})",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    out: list[PaperMetadata] = []
    for paper in _CLIENT.results(search):
        pid = normalize_paper_id(paper.get_short_id())
        year = paper.published.year if paper.published else None
        out.append(
            PaperMetadata(
                paper_id=pid,
                title=paper.title,
                authors=[a.name for a in paper.authors],
                year=year,
                url=paper.entry_id,
                pdf_path=Path(""),
                abstract=paper.summary,
            )
        )
    return out
