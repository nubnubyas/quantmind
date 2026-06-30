"""Parent Graph — Intent Router + Supervisor fan-out over four subgraphs."""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from src.agents._utils import last_user_text
from src.agents.state import AgentState, Mode, RouteDecision
from src.config.llm_client import LLMClient

ROUTER_SYSTEM_PROMPT = """\
You are an intent router for QuantMind, a quant finance AI assistant.
Classify the user query into one primary mode and optional fan-out modes.

Modes:
- research: concept explanations, paper retrieval, factor research, market analysis
- codegen: strategy backtesting, code generation, implementation tasks
- planning: research planning, learning roadmaps, study paths
- interview: interview questions, interview preparation, mock interviews

Rules:
- Strategy backtest / code generation → primary_mode=codegen
- Research planning / learning roadmap → primary_mode=planning
- Interview questions / interview prep → primary_mode=interview
- Concept explanation / paper search / factor research → primary_mode=research
- Cross-domain queries (e.g. "how to present my momentum factor research in a Citadel interview")
  → primary_mode=research, fanout_modes=["interview"]

Set fanout_modes to [] for single-domain queries.
Provide reason and confidence (0.0-1.0).\
"""


class RouterSchema(BaseModel):
    primary_mode: Literal["research", "codegen", "planning", "interview"] = Field(
        description="Primary domain of the query"
    )
    fanout_modes: list[Literal["research", "codegen", "planning", "interview"]] = Field(
        default_factory=list,
        description="Additional domains for cross-domain queries",
    )
    reason: str = Field(description="Classification rationale")
    confidence: float = Field(ge=0.0, le=1.0, description="Classification confidence")


def _make_classify_intent(llm_client: LLMClient):
    def classify_intent(state: AgentState) -> dict:
        user_text = last_user_text(state)
        raw = llm_client.chat_structured(
            [
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            RouterSchema,
        )
        route: RouteDecision = {
            "primary_mode": raw["primary_mode"],
            "fanout_modes": raw.get("fanout_modes") or [],
            "reason": raw["reason"],
            "confidence": raw["confidence"],
        }
        if route["confidence"] < 0.5:
            original_reason = route["reason"]
            route = {
                "primary_mode": "research",
                "fanout_modes": [],
                "reason": (
                    f"Low confidence ({route['confidence']:.2f}), fallback to research. "
                    f"Original: {original_reason}"
                ),
                "confidence": route["confidence"],
            }
        return {"route": route}

    return classify_intent


def route_to_subgraphs(state: AgentState) -> list[Send]:
    route = state["route"]
    modes: list[Mode] = [route["primary_mode"]] + route.get("fanout_modes", [])
    unique_modes = list(dict.fromkeys(modes))
    # route is already on state from router; omit from Send arg to avoid concurrent writes
    return [Send(mode, {}) for mode in unique_modes]


def merge_results(state: AgentState) -> dict:
    outputs = state.get("subgraph_outputs") or {}
    if len(outputs) <= 1:
        return {}

    parts: list[str] = []
    for mode, out in outputs.items():
        if out.get("error") is None and out.get("result"):
            parts.append(f"## {mode.upper()}\n\n{out['result']}")

    combined = "\n\n---\n\n".join(parts)
    draft = f"以下是对您问题的综合分析：\n\n{combined}"
    return {"draft_answer": draft}


def format_final(state: AgentState) -> dict:
    outputs = state.get("subgraph_outputs") or {}
    route = state.get("route")
    primary = route.get("primary_mode", "research") if route else "research"

    if state.get("draft_answer"):
        final = state["draft_answer"]
    elif primary in outputs and outputs[primary].get("result"):
        final = outputs[primary]["result"]
    else:
        errors: list[str] = []
        for mode, out in outputs.items():
            if out.get("error"):
                errors.append(f"{mode}: {out['error']}")
        if errors:
            final = "抱歉，无法处理您的请求。\n错误信息：\n" + "\n".join(errors)
        else:
            final = "抱歉，无法处理您的请求。"

    return {"final_response": final}


def compile_parent_graph(
    research_subgraph: CompiledStateGraph,
    codegen_subgraph: CompiledStateGraph,
    planning_subgraph: CompiledStateGraph,
    interview_subgraph: CompiledStateGraph,
    llm_client: LLMClient,
    *,
    checkpointer=None,
) -> CompiledStateGraph:
    """Build and compile the parent graph routing to four injected subgraphs."""
    builder = StateGraph(AgentState)
    builder.add_node("router", _make_classify_intent(llm_client))
    builder.add_node("research", research_subgraph)
    builder.add_node("codegen", codegen_subgraph)
    builder.add_node("planning", planning_subgraph)
    builder.add_node("interview", interview_subgraph)
    builder.add_node("merge_results", merge_results)
    builder.add_node("format_final", format_final)

    builder.add_edge(START, "router")
    builder.add_conditional_edges("router", route_to_subgraphs)
    for mode in ("research", "codegen", "planning", "interview"):
        builder.add_edge(mode, "merge_results")
    builder.add_edge("merge_results", "format_final")
    builder.add_edge("format_final", END)

    return builder.compile(checkpointer=checkpointer)
