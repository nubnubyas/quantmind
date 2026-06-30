"""Unit tests for Planning Subgraph (task pack B3)."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage
from langgraph.store.memory import InMemoryStore

from src.agents.planning_agent import compile_planning_subgraph
from src.memory import UserMemory
from src.tools.types import ToolResult


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


def _plan_steps() -> list[dict]:
    return [
        {
            "step_id": "s1",
            "title": "Foundations",
            "description": "Learn volatility basics",
            "status": "pending",
        },
        {
            "step_id": "s2",
            "title": "Arbitrage strategies",
            "description": "Study vol arb mechanics",
            "status": "pending",
        },
        {
            "step_id": "s3",
            "title": "Implementation",
            "description": "Build a simple vol arb model",
            "status": "pending",
        },
    ]


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.chat_structured.return_value = {
        "goal_text": "系统学习波动率套利",
        "current_level": "beginner",
        "num_steps": 5,
    }
    return llm


@pytest.fixture
def memory_store():
    return InMemoryStore()


@pytest.fixture
def memory(memory_store):
    return UserMemory(store=memory_store)


def test_compile_signature():
    sig = inspect.signature(compile_planning_subgraph)
    params = list(sig.parameters.keys())
    assert params == ["llm_client", "memory", "checkpointer"]
    assert sig.parameters["memory"].default is None
    assert sig.parameters["checkpointer"].kind == inspect.Parameter.KEYWORD_ONLY


def test_happy_path(mock_llm, memory):
    plan_data = {
        "result_type": "research_plan",
        "plan_id": "plan-1",
        "goal": "系统学习波动率套利",
        "steps": _plan_steps(),
        "existing": False,
    }
    with patch(
        "src.agents.planning_agent.create_research_plan",
        return_value=ToolResult(ok=True, tool_name="create_research_plan", data=plan_data),
    ):
        graph = compile_planning_subgraph(mock_llm, memory)
        result = graph.invoke(_initial_state("我想系统学习波动率套利"))

    assert result.get("final_response")
    subgraph = (result.get("subgraph_outputs") or {}).get("planning", {})
    assert subgraph.get("mode") == "planning"
    assert subgraph.get("error") is None
    assert subgraph.get("result")
    assert "s1" in (subgraph.get("result") or "")
    assert "Foundations" in (subgraph.get("result") or "")


def test_idempotent_same_goal(mock_llm, memory):
    tool_llm = MagicMock()
    tool_llm.chat_structured.return_value = {
        "steps": [
            {
                "step_id": "s1",
                "title": "Read foundational paper",
                "description": "Study vol surface",
                "status": "pending",
            }
        ]
    }

    graph = compile_planning_subgraph(mock_llm, memory)
    message = "我想系统学习波动率套利"
    state = _initial_state(message, user_id="user_plan_1")

    with patch("src.agents.planning_agent.create_research_plan") as mock_t6:
        from src.tools.create_research_plan import create_research_plan as real_t6

        def _delegate(**kwargs):
            kwargs["memory"] = memory
            kwargs["llm"] = tool_llm
            return real_t6(**kwargs)

        mock_t6.side_effect = _delegate

        first = graph.invoke(state)
        second = graph.invoke(state)

    assert first.get("final_response")
    assert second.get("final_response")
    assert "已有计划" in (second.get("final_response") or "")
    assert tool_llm.chat_structured.call_count == 1


def test_llm_error_path(mock_llm, memory):
    mock_llm.chat_structured.side_effect = RuntimeError("LLM unavailable")
    graph = compile_planning_subgraph(mock_llm, memory)
    result = graph.invoke(_initial_state("我想系统学习波动率套利"))

    subgraph = (result.get("subgraph_outputs") or {}).get("planning", {})
    assert subgraph.get("error")
    assert "LLM unavailable" in (subgraph.get("error") or "")
    assert "LLM unavailable" in (result.get("final_response") or "")


def test_format_plan_output(mock_llm, memory):
    plan_data = {
        "result_type": "research_plan",
        "plan_id": "plan-fmt",
        "steps": _plan_steps(),
        "existing": False,
    }
    with patch(
        "src.agents.planning_agent.create_research_plan",
        return_value=ToolResult(ok=True, tool_name="create_research_plan", data=plan_data),
    ):
        graph = compile_planning_subgraph(mock_llm, memory)
        result = graph.invoke(_initial_state("plan format test"))

    final = result.get("final_response") or ""
    for step in _plan_steps():
        assert step["step_id"] in final
        assert step["title"] in final
        assert step["description"] in final
        assert step["status"] in final


def test_tool_failure_path(mock_llm, memory):
    with patch(
        "src.agents.planning_agent.create_research_plan",
        return_value=ToolResult(
            ok=False,
            tool_name="create_research_plan",
            data={"result_type": "error"},
            error="LLM_ERROR: timeout",
            error_code="LLM_ERROR",
            retryable=True,
        ),
    ):
        graph = compile_planning_subgraph(mock_llm, memory)
        result = graph.invoke(_initial_state("我想系统学习波动率套利"))

    subgraph = (result.get("subgraph_outputs") or {}).get("planning", {})
    assert subgraph.get("error")
    assert "timeout" in (result.get("final_response") or "")


def test_missing_user_id_uses_fallback(mock_llm, memory):
    """When user_id is absent from state, planning subgraph uses fallback instead of crashing."""
    state = _initial_state("我想学习波动率套利")
    del state["user_id"]

    plan_data = {
        "result_type": "research_plan",
        "plan_id": "plan-fb",
        "steps": [_plan_steps()[0]],
        "existing": False,
    }
    with patch(
        "src.agents.planning_agent.create_research_plan",
        return_value=ToolResult(ok=True, tool_name="create_research_plan", data=plan_data),
    ):
        graph = compile_planning_subgraph(mock_llm, memory)
        result = graph.invoke(state)

    assert result.get("final_response")
