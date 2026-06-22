"""Research Subgraph — Plan-then-Execute-then-Verify (technical plan §2.3)."""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from src.agents.state import (
    AgentState,
    Citation,
    QueryIntent,
    SubgraphOutput,
    VerificationResult,
)
from src.config.llm_client import LLMClient
from src.vector_store.types import RetrievalSpec, SearchResult, VectorStore

CONFIDENCE_THRESHOLD = 0.5
VERIFY_THRESHOLD = 0.75
DEPTH_SCORE_THRESHOLD = 0.85


class QueryIntentSchema(BaseModel):
    keywords: list[str] = Field(description="Key terms from the user query")
    domain: str = Field(description="Domain e.g. factor investing, risk, execution")
    intent: str = Field(description="What the user wants e.g. explain, compare, apply")
    doc_type: str | None = Field(default=None, description="paper/concept/code or null")
    time_range: str | None = Field(default=None, description="Time range if mentioned")


class VerificationJudgeSchema(BaseModel):
    answers_question: bool = Field(description="Does the answer address the specific question?")
    claims_grounded: bool = Field(description="Are key claims supported by retrieved passages?")
    no_hallucination: bool = Field(description="No unsupported new claims introduced?")
    uncertainty_stated: bool = Field(description="Uncertainty stated when evidence is thin?")
    score: float = Field(ge=0.0, le=1.0, description="Overall verification score 0-1")
    failure_reasons: list[str] = Field(default_factory=list)


def _last_user_text(state: AgentState) -> str:
    messages = state.get("messages") or []
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content", ""))
        role = getattr(msg, "type", None) or getattr(msg, "role", None)
        if role in ("human", "user"):
            content = getattr(msg, "content", "")
            return content if isinstance(content, str) else str(content)
    return ""


def _query_text(state: AgentState) -> str:
    intent = state.get("query_intent")
    if intent and intent.get("keywords"):
        return " ".join(intent["keywords"])
    return _last_user_text(state)


def _results_to_citations(results: list[SearchResult]) -> list[Citation]:
    citations: list[Citation] = []
    for r in results:
        citations.append(
            {
                "paper_id": r.paper_id or r.doc_id,
                "title": r.title or r.doc_id,
                "section": r.section,
                "relevance_score": r.fusion_score,
            }
        )
    return citations


def _format_sources(results: list[SearchResult]) -> str:
    blocks: list[str] = []
    for i, r in enumerate(results, 1):
        blocks.append(
            f"[{i}] title={r.title!r} paper_id={r.paper_id!r} section={r.section!r}\n{r.text}"
        )
    return "\n\n".join(blocks)


def compile_research_subgraph(
    vector_store: VectorStore,
    llm_client: LLMClient,
    *,
    checkpointer=None,
) -> CompiledStateGraph:
    """Build and compile the Research subgraph on AgentState."""
    depth_expanded = False

    def parse_query(state: AgentState) -> dict:
        user_text = _last_user_text(state)
        raw = llm_client.chat_structured(
            [
                {
                    "role": "system",
                    "content": (
                        "Extract structured query intent for a quant finance research assistant. "
                        "Return keywords, domain, intent, optional doc_type and time_range."
                    ),
                },
                {"role": "user", "content": user_text},
            ],
            QueryIntentSchema,
        )
        query_intent: QueryIntent = {
            "keywords": raw["keywords"],
            "domain": raw["domain"],
            "intent": raw["intent"],
            "doc_type": raw.get("doc_type"),
            "time_range": raw.get("time_range"),
        }
        return {"query_intent": query_intent}

    def hybrid_search(state: AgentState) -> dict:
        nonlocal depth_expanded
        query = _query_text(state)
        top_k = 10 if depth_expanded else 5
        prefetch_k = 40 if depth_expanded else 20
        spec = RetrievalSpec(mode="hybrid", top_k=top_k, prefetch_k=prefetch_k)
        retrieved = vector_store.search(query, "papers", spec)
        return {"retrieved": retrieved}

    def check_confidence(state: AgentState) -> dict:
        retrieved = state.get("retrieved") or []
        confidence = retrieved[0].fusion_score if retrieved else 0.0
        return {"confidence": confidence}

    def route_after_confidence(state: AgentState) -> Literal["graceful_decline", "decide_depth"]:
        confidence = state.get("confidence") or 0.0
        if confidence < CONFIDENCE_THRESHOLD:
            return "graceful_decline"
        return "decide_depth"

    def decide_depth(state: AgentState) -> dict:
        return {}

    def route_after_depth(
        state: AgentState,
    ) -> Literal["fetch_more", "synthesize_answer"]:
        nonlocal depth_expanded
        retrieved = state.get("retrieved") or []
        if not retrieved:
            return "fetch_more"
        if depth_expanded:
            return "synthesize_answer"
        top_score = retrieved[0].fusion_score
        if len(retrieved) < 3 or top_score < DEPTH_SCORE_THRESHOLD:
            return "fetch_more"
        return "synthesize_answer"

    def fetch_more(state: AgentState) -> dict:
        nonlocal depth_expanded
        del state
        depth_expanded = True
        return {}

    def synthesize_answer(state: AgentState) -> dict:
        user_text = _last_user_text(state)
        retrieved = state.get("retrieved") or []
        sources = _format_sources(retrieved)
        result = llm_client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a quant research assistant. Answer using ONLY the provided sources. "
                        "Cite sources inline as [1], [2], etc. State uncertainty when evidence is weak."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {user_text}\n\nSources:\n{sources}",
                },
            ]
        )
        return {
            "draft_answer": result.text,
            "citations": _results_to_citations(retrieved),
        }

    def verify_answer(state: AgentState) -> dict:
        user_text = _last_user_text(state)
        draft = state.get("draft_answer") or ""
        retrieved = state.get("retrieved") or []
        sources = _format_sources(retrieved)
        raw = llm_client.chat_structured(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a strict verifier. Score the draft against four criteria:\n"
                        "1) answers_question 2) claims_grounded 3) no_hallucination "
                        "4) uncertainty_stated.\n"
                        "Return booleans, an overall score 0-1, and failure_reasons."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {user_text}\n\nDraft answer:\n{draft}\n\nSources:\n{sources}"
                    ),
                },
            ],
            VerificationJudgeSchema,
        )
        threshold = VERIFY_THRESHOLD
        score = float(raw["score"])
        allow_output = score >= threshold
        verification = VerificationResult(
            answers_question=bool(raw["answers_question"]),
            claims_grounded=bool(raw["claims_grounded"]),
            no_hallucination=bool(raw["no_hallucination"]),
            uncertainty_stated=bool(raw["uncertainty_stated"]),
            score=score,
            threshold_used=threshold,
            failure_reasons=list(raw.get("failure_reasons") or []),
            allow_output=allow_output,
        )
        updates: dict = {"verification": verification}
        if not allow_output:
            current = state.get("retry_counts", {}).get("research", 0)
            updates["retry_counts"] = {"research": current + 1}
        return updates

    def route_after_verify(
        state: AgentState,
    ) -> Literal["format_response", "hybrid_search", "format_doubtful"]:
        verification = state.get("verification")
        if verification and verification.allow_output:
            return "format_response"
        retries = state.get("retry_counts", {}).get("research", 0)
        if retries <= 1:
            return "hybrid_search"
        return "format_doubtful"

    def format_response(state: AgentState) -> dict:
        draft = state.get("draft_answer") or ""
        citations = state.get("citations") or []
        cite_lines = "\n".join(
            f"- [{c['title']}] (relevance={c['relevance_score']:.2f})" for c in citations
        )
        final = f"{draft}\n\nReferences:\n{cite_lines}" if cite_lines else draft
        subgraph_output: SubgraphOutput = {
            "mode": "research",
            "result": final,
            "citations": citations,
            "error": None,
        }
        return {
            "final_response": final,
            "subgraph_outputs": {"research": subgraph_output},
        }

    def format_doubtful(state: AgentState) -> dict:
        draft = state.get("draft_answer") or ""
        verification = state.get("verification")
        reasons = ", ".join(verification.failure_reasons) if verification else "low confidence"
        header = (
            "⚠️ 以下部分未经充分验证，请谨慎参考。\n"
            f"验证说明: {reasons}\n\n"
        )
        final = header + draft
        citations = state.get("citations") or []
        subgraph_output: SubgraphOutput = {
            "mode": "research",
            "result": final,
            "citations": citations,
            "error": "verification_failed_after_retry",
        }
        return {
            "final_response": final,
            "subgraph_outputs": {"research": subgraph_output},
        }

    def graceful_decline(state: AgentState) -> dict:
        user_text = _last_user_text(state)
        intent = state.get("query_intent")
        result = llm_client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Retrieval confidence was too low. Respond in Chinese. "
                        "Say information is insufficient and give exactly 3 suggested "
                        "research directions the user could try."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Query: {user_text}\nIntent: {intent}",
                },
            ]
        )
        subgraph_output: SubgraphOutput = {
            "mode": "research",
            "result": result.text,
            "citations": [],
            "error": "low_confidence",
        }
        return {
            "final_response": result.text,
            "subgraph_outputs": {"research": subgraph_output},
        }

    builder = StateGraph(AgentState)
    builder.add_node("parse_query", parse_query)
    builder.add_node("hybrid_search", hybrid_search)
    builder.add_node("check_confidence", check_confidence)
    builder.add_node("decide_depth", decide_depth)
    builder.add_node("fetch_more", fetch_more)
    builder.add_node("synthesize_answer", synthesize_answer)
    builder.add_node("verify_answer", verify_answer)
    builder.add_node("format_response", format_response)
    builder.add_node("format_doubtful", format_doubtful)
    builder.add_node("graceful_decline", graceful_decline)

    builder.add_edge(START, "parse_query")
    builder.add_edge("parse_query", "hybrid_search")
    builder.add_edge("hybrid_search", "check_confidence")
    builder.add_conditional_edges("check_confidence", route_after_confidence)
    builder.add_conditional_edges("decide_depth", route_after_depth)
    builder.add_edge("fetch_more", "hybrid_search")
    builder.add_edge("synthesize_answer", "verify_answer")
    builder.add_conditional_edges("verify_answer", route_after_verify)
    builder.add_edge("format_response", END)
    builder.add_edge("format_doubtful", END)
    builder.add_edge("graceful_decline", END)

    return builder.compile(checkpointer=checkpointer)
