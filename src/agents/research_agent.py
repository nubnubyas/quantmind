"""Research Subgraph — Plan-then-Execute-then-Verify (technical plan §2.3)."""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from src.agents._utils import last_user_text
from src.agents.state import (
    AgentState,
    Citation,
    QueryIntent,
    SubgraphOutput,
    VerificationResult,
)
from src.config.llm_client import LLMClient
from src.tools.search_web import search_web
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


def _query_text(state: AgentState) -> str:
    intent = state.get("query_intent")
    if intent and intent.get("keywords"):
        return " ".join(intent["keywords"])
    return last_user_text(state)


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
        source_type = f"source={r.source}" if r.source == "web" else f"paper_id={r.paper_id!r}"
        blocks.append(
            f"[{i}] title={r.title!r} {source_type} section={r.section!r}\n{r.text}"
        )
    return "\n\n".join(blocks)


def _web_to_search_results(web_data: list[dict]) -> list[SearchResult]:
    """Convert DuckDuckGo web search results to the SearchResult format expected by
    synthesize_answer and verify_answer nodes."""
    return [
        SearchResult(
            text=item.get("snippet", ""),
            fusion_score=0.5,
            doc_id=f"web-{i}",
            chunk_id=f"web-{i}",
            source="web",
            paper_id=None,
            title=item.get("title"),
            section=None,
            year=None,
            url=item.get("url"),
        )
        for i, item in enumerate(web_data)
    ]


def compile_research_subgraph(
    vector_store: VectorStore,
    llm_client: LLMClient,
    *,
    checkpointer=None,
) -> CompiledStateGraph:
    """Build and compile the Research subgraph on AgentState."""
    depth_expanded = False

    def parse_query(state: AgentState) -> dict:
        user_text = last_user_text(state)
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
        qdrant_results = vector_store.search(query, "papers", spec)

        web_result = search_web(query, max_results=3)
        web_results: list[SearchResult] = []
        if web_result.ok:
            web_data = web_result.data.get("results") or []
            if web_data:
                web_results = [
                    SearchResult(
                        text=item.get("snippet", ""),
                        fusion_score=0.45,
                        doc_id=f"web-{i}",
                        chunk_id=f"web-{i}",
                        source="web",
                        paper_id=None,
                        title=item.get("title"),
                        section=None,
                        year=None,
                        url=item.get("url"),
                    )
                    for i, item in enumerate(web_data)
                ]

        merged: list[SearchResult] = []
        web_idx = 0
        for i, qr in enumerate(qdrant_results):
            merged.append(qr)
            if (i + 1) % 2 == 0 and web_idx < len(web_results):
                merged.append(web_results[web_idx])
                web_idx += 1
        merged.extend(web_results[web_idx:])

        # 降权 Alpha-GPT 综述 chunk（当查询不涉及 alpha mining 时）
        query_lower = query.lower()
        is_alpha_query = any(
            kw in query_lower for kw in ["alpha", "gpt", "mining", "signal generation"]
        )
        if not is_alpha_query:
            alpha_chunks = []
            other_chunks = []
            for r in merged:
                title_lower = (r.title or "").lower()
                text_lower = (r.text or "").lower()
                if "alpha-gpt" in title_lower or "alpha-gpt" in text_lower[:300]:
                    alpha_chunks.append(r)
                else:
                    other_chunks.append(r)
            merged = other_chunks + alpha_chunks

        return {"retrieved": merged}

    def check_confidence(state: AgentState) -> dict:
        retrieved = state.get("retrieved") or []
        confidence = retrieved[0].fusion_score if retrieved else 0.0
        return {"confidence": confidence}

    def route_after_confidence(state: AgentState) -> Literal["web_search", "decide_depth"]:
        confidence = state.get("confidence") or 0.0
        if confidence < CONFIDENCE_THRESHOLD:
            return "web_search"
        return "decide_depth"

    def web_search(state: AgentState) -> dict:
        user_text = last_user_text(state)
        result = search_web(user_text, max_results=5)
        if result.ok:
            web_data = result.data.get("results") or []
            if web_data:
                return {"retrieved": _web_to_search_results(web_data)}
        return {"retrieved": []}

    def check_web_results(state: AgentState) -> dict:
        retrieved = state.get("retrieved") or []
        if retrieved:
            return {"confidence": 0.5}
        return {}

    def route_after_web(
        state: AgentState,
    ) -> Literal["synthesize_answer", "graceful_decline"]:
        retrieved = state.get("retrieved") or []
        if retrieved:
            return "synthesize_answer"
        return "graceful_decline"

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
        user_text = last_user_text(state)
        retrieved = state.get("retrieved") or []
        sources = _format_sources(retrieved)
        result = llm_client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a quant research assistant. Answer the user's question "
                        "using the provided sources.\n\n"
                        "CRITICAL RULES:\n"
                        "1. Use at least 2 DIFFERENT sources in your answer. "
                        "Do not rely on a single source even if it looks comprehensive.\n"
                        "2. Sources marked with 'source=web' are web search results — "
                        "treat them as equally valid as paper sources.\n"
                        "3. If sources disagree or cover different aspects, "
                        "synthesize a balanced view.\n"
                        "4. Cite sources inline as [1], [2], etc. "
                        "State uncertainty when evidence is weak or thin.\n"
                        "5. Answer the SPECIFIC question asked — do not summarize a paper "
                        "unless it directly answers the question."
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
        user_text = last_user_text(state)
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
        user_text = last_user_text(state)
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
    builder.add_node("web_search", web_search)
    builder.add_node("check_web_results", check_web_results)
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
    builder.add_edge("web_search", "check_web_results")
    builder.add_conditional_edges("check_web_results", route_after_web)
    builder.add_conditional_edges("decide_depth", route_after_depth)
    builder.add_edge("fetch_more", "hybrid_search")
    builder.add_edge("synthesize_answer", "verify_answer")
    builder.add_conditional_edges("verify_answer", route_after_verify)
    builder.add_edge("format_response", END)
    builder.add_edge("format_doubtful", END)
    builder.add_edge("graceful_decline", END)

    return builder.compile(checkpointer=checkpointer)
