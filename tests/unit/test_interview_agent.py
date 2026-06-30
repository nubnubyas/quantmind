"""Unit tests for Interview Subgraph (task pack B3)."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage
from langgraph.store.memory import InMemoryStore

from src.agents.interview_agent import compile_interview_subgraph
from src.memory import UserMemory
from src.tools.types import ToolResult

LONG_JD = (
    "Quantitative Researcher at Citadel. Requirements: strong Python, statistics, "
    "machine learning, and experience with factor models and portfolio optimization."
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


def _questions_fixture() -> list[dict]:
    return [
        {
            "question": "Explain momentum factor construction.",
            "category": "quant",
            "difficulty": "medium",
        },
        {
            "question": "How would you debug a backtest pipeline?",
            "category": "engineering",
            "difficulty": "hard",
        },
        {
            "question": "What is the Sharpe ratio?",
            "category": "quant",
            "difficulty": "easy",
        },
    ]


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.chat_structured.return_value = {
        "jd_text": LONG_JD,
        "company": "Citadel",
        "focus_areas": ["factor models"],
    }
    return llm


@pytest.fixture
def memory_store():
    return InMemoryStore()


@pytest.fixture
def memory(memory_store):
    return UserMemory(store=memory_store)


def test_compile_signature():
    sig = inspect.signature(compile_interview_subgraph)
    params = list(sig.parameters.keys())
    assert params == ["llm_client", "memory", "checkpointer"]
    assert sig.parameters["memory"].default is None
    assert sig.parameters["checkpointer"].kind == inspect.Parameter.KEYWORD_ONLY


def test_happy_path(mock_llm, memory):
    questions = _questions_fixture()
    tool_data = {
        "result_type": "interview_questions",
        "questions": questions,
        "count": len(questions),
    }
    with patch(
        "src.agents.interview_agent.generate_interview_questions",
        return_value=ToolResult(ok=True, tool_name="generate_interview_questions", data=tool_data),
    ):
        graph = compile_interview_subgraph(mock_llm, memory)
        result = graph.invoke(_initial_state(f"Generate questions for:\n{LONG_JD}"))

    assert result.get("final_response")
    subgraph = (result.get("subgraph_outputs") or {}).get("interview", {})
    assert subgraph.get("mode") == "interview"
    assert subgraph.get("error") is None
    assert subgraph.get("result")
    assert "共 3 题" in (subgraph.get("result") or "")


def test_no_profile_new_user(mock_llm, memory):
    tool_data = {
        "result_type": "interview_questions",
        "questions": _questions_fixture()[:1],
        "count": 1,
    }
    with patch(
        "src.agents.interview_agent.generate_interview_questions",
        return_value=ToolResult(ok=True, tool_name="generate_interview_questions", data=tool_data),
    ) as mock_t5:
        graph = compile_interview_subgraph(mock_llm, memory)
        result = graph.invoke(_initial_state(LONG_JD, user_id="brand_new_user"))

    assert result.get("final_response")
    subgraph = (result.get("subgraph_outputs") or {}).get("interview", {})
    assert subgraph.get("error") is None
    mock_t5.assert_called_once()


def test_jd_parse_failure_short_text(mock_llm, memory):
    mock_llm.chat_structured.return_value = {
        "jd_text": "hi",
        "company": None,
        "focus_areas": None,
    }
    graph = compile_interview_subgraph(mock_llm, memory)
    result = graph.invoke(_initial_state("hi"))

    subgraph = (result.get("subgraph_outputs") or {}).get("interview", {})
    assert subgraph.get("error")
    assert "无法生成面试题" in (result.get("final_response") or "")


def test_jd_parse_llm_failure(mock_llm, memory):
    mock_llm.chat_structured.side_effect = RuntimeError("parse failed")
    graph = compile_interview_subgraph(mock_llm, memory)
    result = graph.invoke(_initial_state(LONG_JD))

    subgraph = (result.get("subgraph_outputs") or {}).get("interview", {})
    assert subgraph.get("error")
    assert "parse failed" in (result.get("final_response") or "")


def test_profile_personalization(mock_llm, memory):
    memory.update_profile(
        "user_profile_1",
        {
            "research_interests": ["momentum"],
            "target_roles": ["quant researcher"],
        },
    )
    tool_data = {
        "result_type": "interview_questions",
        "questions": _questions_fixture()[:1],
        "count": 1,
    }
    with patch(
        "src.agents.interview_agent.generate_interview_questions",
        return_value=ToolResult(ok=True, tool_name="generate_interview_questions", data=tool_data),
    ) as mock_t5:
        graph = compile_interview_subgraph(mock_llm, memory)
        graph.invoke(_initial_state(LONG_JD, user_id="user_profile_1"))

    focus_areas = mock_t5.call_args.kwargs.get("focus_areas") or []
    assert "momentum" in focus_areas
    assert "quant researcher" in focus_areas


def test_format_questions_output(mock_llm, memory):
    questions = _questions_fixture()
    tool_data = {
        "result_type": "interview_questions",
        "questions": questions,
        "count": len(questions),
    }
    with patch(
        "src.agents.interview_agent.generate_interview_questions",
        return_value=ToolResult(ok=True, tool_name="generate_interview_questions", data=tool_data),
    ):
        graph = compile_interview_subgraph(mock_llm, memory)
        result = graph.invoke(_initial_state(LONG_JD))

    final = result.get("final_response") or ""
    assert "## quant" in final
    assert "## engineering" in final
    assert "[easy]" in final
    assert "[medium]" in final
    assert "[hard]" in final
