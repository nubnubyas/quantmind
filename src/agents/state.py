"""§C0 LangGraph State Schema (interface contract v1.0)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages

from src.sandbox import SandboxResult
from src.vector_store.types import SearchResult

Mode = Literal["research", "codegen", "planning", "interview"]


class RouteDecision(TypedDict):
    primary_mode: Mode
    fanout_modes: list[Mode]
    reason: str
    confidence: float


class QueryIntent(TypedDict):
    keywords: list[str]
    domain: str
    intent: str
    doc_type: str | None
    time_range: str | None


@dataclass
class VerificationResult:
    answers_question: bool
    claims_grounded: bool
    no_hallucination: bool
    uncertainty_stated: bool
    score: float
    threshold_used: float
    failure_reasons: list[str]
    allow_output: bool


class Citation(TypedDict):
    paper_id: str
    title: str
    section: str | None
    relevance_score: float


class SubgraphOutput(TypedDict):
    mode: Mode
    result: str | None
    citations: list[Citation]
    error: str | None


def _merge_named_outputs(
    a: dict[str, SubgraphOutput],
    b: dict[str, SubgraphOutput],
) -> dict[str, SubgraphOutput]:
    """Each subgraph writes under its mode key; different keys avoid silent overwrite."""
    return {**a, **b}


def _merge_retry_counts(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    merged = dict(a)
    for k, v in b.items():
        merged[k] = max(merged.get(k, 0), v)
    return merged


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    route: Optional[RouteDecision]

    query_intent: Optional[QueryIntent]
    retrieved: Optional[list[SearchResult]]
    confidence: Optional[float]
    draft_answer: Optional[str]
    verification: Optional[VerificationResult]

    strategy_spec: Optional[dict]
    generated_code: Optional[str]
    sandbox_result: Optional[SandboxResult]

    final_response: Optional[str]
    citations: Optional[list[Citation]]

    retry_counts: Annotated[dict[str, int], _merge_retry_counts]
    subgraph_outputs: Annotated[dict[str, SubgraphOutput], _merge_named_outputs]


__all__ = [
    "AgentState",
    "Citation",
    "Mode",
    "QueryIntent",
    "RouteDecision",
    "SubgraphOutput",
    "VerificationResult",
    "_merge_named_outputs",
    "_merge_retry_counts",
]
