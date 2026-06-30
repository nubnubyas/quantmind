"""Unit tests for Router + Supervisor Parent Graph (task pack B4)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from src.agents.state import AgentState, SubgraphOutput
from src.agents.supervisor import (
    _make_classify_intent,
    compile_parent_graph,
    format_final,
    merge_results,
    route_to_subgraphs,
)


def _initial_state(message: str, user_id: str = "test_user") -> dict:
    return {
        "messages": [HumanMessage(content=message)],
        "user_id": user_id,
        "route": None,
        "query_intent": None,
        "retrieved": None,
        "confidence": None,
        "draft_answer": None,
        "verification": None,
        "strategy_spec": None,
        "generated_code": None,
        "sandbox_result": None,
        "final_response": None,
        "citations": None,
        "retry_counts": {},
        "subgraph_outputs": {},
    }


def _subgraph_output(
    mode: str,
    result: str = "mock result",
    error: str | None = None,
) -> SubgraphOutput:
    return {
        "mode": mode,  # type: ignore[typeddict-item]
        "result": result,
        "citations": [],
        "error": error,
    }


def _mock_subgraph(
    mode: str,
    result: str = "mock result",
    error: str | None = None,
):
    def pass_node(state: AgentState) -> dict:
        return {
            "subgraph_outputs": {
                mode: _subgraph_output(mode, result=result, error=error),
            },
        }

    builder = StateGraph(AgentState)
    builder.add_node("pass", pass_node)
    builder.add_edge(START, "pass")
    builder.add_edge("pass", END)
    return builder.compile()


@pytest.fixture
def mock_llm():
    return MagicMock()


def _run_router(mock_llm: MagicMock, message: str) -> dict:
    classify = _make_classify_intent(mock_llm)
    return classify(_initial_state(message))


def test_route_research(mock_llm):
    mock_llm.chat_structured.return_value = {
        "primary_mode": "research",
        "fanout_modes": [],
        "reason": "Concept explanation about momentum factor",
        "confidence": 0.92,
    }
    result = _run_router(mock_llm, "解释动量因子")
    route = result["route"]
    assert route["primary_mode"] == "research"
    assert route["fanout_modes"] == []


def test_route_codegen(mock_llm):
    mock_llm.chat_structured.return_value = {
        "primary_mode": "codegen",
        "fanout_modes": [],
        "reason": "Strategy backtest code request",
        "confidence": 0.95,
    }
    result = _run_router(mock_llm, "写一个 RSI 策略回测代码")
    route = result["route"]
    assert route["primary_mode"] == "codegen"
    assert route["fanout_modes"] == []


def test_route_planning(mock_llm):
    mock_llm.chat_structured.return_value = {
        "primary_mode": "planning",
        "fanout_modes": [],
        "reason": "Learning roadmap request",
        "confidence": 0.88,
    }
    result = _run_router(mock_llm, "帮我规划学习期权定价")
    route = result["route"]
    assert route["primary_mode"] == "planning"
    assert route["fanout_modes"] == []


def test_route_interview(mock_llm):
    mock_llm.chat_structured.return_value = {
        "primary_mode": "interview",
        "fanout_modes": [],
        "reason": "Interview question generation",
        "confidence": 0.91,
    }
    result = _run_router(mock_llm, "生成 Citadel 量化研究员的面试题")
    route = result["route"]
    assert route["primary_mode"] == "interview"
    assert route["fanout_modes"] == []


def test_route_cross_domain(mock_llm):
    mock_llm.chat_structured.return_value = {
        "primary_mode": "research",
        "fanout_modes": ["interview"],
        "reason": "Cross-domain: factor research presentation for interview",
        "confidence": 0.85,
    }
    result = _run_router(mock_llm, "如何在 Citadel 面试中介绍我的动量因子研究？")
    route = result["route"]
    assert route["primary_mode"] == "research"
    assert route["fanout_modes"] == ["interview"]


def test_route_low_confidence_fallback(mock_llm):
    mock_llm.chat_structured.return_value = {
        "primary_mode": "codegen",
        "fanout_modes": [],
        "reason": "Uncertain classification",
        "confidence": 0.3,
    }
    result = _run_router(mock_llm, "something ambiguous")
    route = result["route"]
    assert route["primary_mode"] == "research"
    assert route["fanout_modes"] == []
    assert "Low confidence" in route["reason"]


def test_route_to_subgraphs_deduplicates():
    state = {
        **_initial_state("test"),
        "route": {
            "primary_mode": "research",
            "fanout_modes": ["research", "interview"],
            "reason": "cross",
            "confidence": 0.9,
        },
    }
    sends = route_to_subgraphs(state)  # type: ignore[arg-type]
    node_names = [s.node for s in sends]
    assert node_names == ["research", "interview"]


def test_merge_cross_domain():
    state = {
        **_initial_state("test"),
        "subgraph_outputs": {
            "research": _subgraph_output("research", result="动量因子研究内容"),
            "interview": _subgraph_output("interview", result="面试准备建议"),
        },
    }
    result = merge_results(state)  # type: ignore[arg-type]
    assert "draft_answer" in result
    draft = result["draft_answer"]
    assert "RESEARCH" in draft
    assert "INTERVIEW" in draft
    assert "动量因子研究内容" in draft
    assert "面试准备建议" in draft


def test_merge_single_domain_passthrough():
    state = {
        **_initial_state("test"),
        "subgraph_outputs": {
            "research": _subgraph_output("research", result="only one"),
        },
    }
    result = merge_results(state)  # type: ignore[arg-type]
    assert result == {}


def test_format_final_single_domain():
    state = {
        **_initial_state("test"),
        "route": {
            "primary_mode": "research",
            "fanout_modes": [],
            "reason": "test",
            "confidence": 0.9,
        },
        "subgraph_outputs": {
            "research": _subgraph_output("research", result="Research answer text"),
        },
    }
    result = format_final(state)  # type: ignore[arg-type]
    assert result["final_response"] == "Research answer text"


def test_format_final_cross_domain_uses_draft():
    state = {
        **_initial_state("test"),
        "route": {
            "primary_mode": "research",
            "fanout_modes": ["interview"],
            "reason": "cross",
            "confidence": 0.9,
        },
        "subgraph_outputs": {
            "research": _subgraph_output("research", result="Research only"),
            "interview": _subgraph_output("interview", result="Interview only"),
        },
        "draft_answer": "以下是对您问题的综合分析：\n\n## RESEARCH\n\nResearch only\n\n---\n\n## INTERVIEW\n\nInterview only",
    }
    result = format_final(state)  # type: ignore[arg-type]
    assert "RESEARCH" in result["final_response"]
    assert "INTERVIEW" in result["final_response"]


def test_format_final_all_errors():
    state = {
        **_initial_state("test"),
        "route": {
            "primary_mode": "research",
            "fanout_modes": [],
            "reason": "test",
            "confidence": 0.9,
        },
        "subgraph_outputs": {
            "research": _subgraph_output("research", result=None, error="search failed"),
            "codegen": _subgraph_output("codegen", result=None, error="code failed"),
        },
    }
    result = format_final(state)  # type: ignore[arg-type]
    assert "抱歉" in result["final_response"]
    assert "research: search failed" in result["final_response"]
    assert "codegen: code failed" in result["final_response"]


def test_compile_parent_graph(mock_llm):
    research = _mock_subgraph("research", result="Research mock output")
    codegen = _mock_subgraph("codegen", result="Codegen mock output")
    planning = _mock_subgraph("planning", result="Planning mock output")
    interview = _mock_subgraph("interview", result="Interview mock output")

    mock_llm.chat_structured.return_value = {
        "primary_mode": "research",
        "fanout_modes": [],
        "reason": "research query",
        "confidence": 0.95,
    }

    graph = compile_parent_graph(
        research,
        codegen,
        planning,
        interview,
        mock_llm,
    )
    result = graph.invoke(_initial_state("解释动量因子"))

    assert result.get("final_response") == "Research mock output"
    assert "research" in (result.get("subgraph_outputs") or {})


def test_compile_parent_graph_cross_domain_fanout(mock_llm):
    research = _mock_subgraph("research", result="Research side")
    codegen = _mock_subgraph("codegen", result="Codegen side")
    planning = _mock_subgraph("planning", result="Planning side")
    interview = _mock_subgraph("interview", result="Interview side")

    mock_llm.chat_structured.return_value = {
        "primary_mode": "research",
        "fanout_modes": ["interview"],
        "reason": "cross domain",
        "confidence": 0.9,
    }

    graph = compile_parent_graph(
        research,
        codegen,
        planning,
        interview,
        mock_llm,
    )
    result = graph.invoke(_initial_state("如何在 Citadel 面试中介绍我的动量因子研究？"))

    outputs = result.get("subgraph_outputs") or {}
    assert "research" in outputs
    assert "interview" in outputs
    assert "codegen" not in outputs
    assert "planning" not in outputs
    final = result.get("final_response") or ""
    assert "RESEARCH" in final
    assert "INTERVIEW" in final
