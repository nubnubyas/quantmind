"""Unit tests for BenchmarkRunner (E2 Phase 4)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langgraph.types import Command

from src.eval.evaluators import CitationEvaluator, CodeEvaluator, JudgeEvaluator, ItemResult
from src.eval.runner import BenchmarkRunner


def _bench_item(item_id: str = "bench_001", **overrides) -> dict:
    base = {
        "id": item_id,
        "scenario": "S1",
        "query": "Explain momentum factor",
        "expected_behavior": "Define momentum and cite evidence",
        "difficulty": "easy",
        "eval_criteria": {
            "factual_grounding": True,
            "cites_sources": True,
            "uncertainty_stated": False,
            "requires_code": False,
        },
        "tags": [],
    }
    base.update(overrides)
    return base


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.chat_structured.return_value = {
        "answers_question": 0.9,
        "matches_expected": 0.9,
        "overall_score": 0.9,
        "reason": "Good answer",
    }
    return llm


@pytest.fixture
def evaluators(mock_llm):
    return [
        JudgeEvaluator(mock_llm),
        CodeEvaluator(),
        CitationEvaluator(),
    ]


def test_run_single_item(evaluators):
    graph = MagicMock()
    graph.invoke.return_value = {
        "final_response": "Momentum is past return ranking [1].",
        "citations": [{"paper_id": "p1", "title": "Momentum", "section": None, "relevance_score": 0.9}],
        "sandbox_result": None,
        "route": {"primary_mode": "research", "fanout_modes": [], "reason": "r", "confidence": 0.9},
    }

    runner = BenchmarkRunner(graph, evaluators)
    item = _bench_item()
    result = runner.run_item(item)

    assert isinstance(result, ItemResult)
    assert result.bench_id == "bench_001"
    assert result.response == "Momentum is past return ranking [1]."
    assert result.error is None
    assert len(result.evals) == 2
    assert {ev.evaluator for ev in result.evals} == {"judge", "citation"}
    graph.invoke.assert_called_once()


def test_run_batch(evaluators):
    graph = MagicMock()
    graph.invoke.return_value = {
        "final_response": "Answer without citations",
        "citations": [],
        "sandbox_result": None,
        "route": None,
    }

    runner = BenchmarkRunner(graph, evaluators)
    items = [_bench_item("bench_001"), _bench_item("bench_002"), _bench_item("bench_003")]
    results = runner.run_all(items)

    assert len(results) == 3
    assert graph.invoke.call_count == 3


def test_run_with_interrupt(evaluators):
    graph = MagicMock()
    interrupt = SimpleNamespace(id="intr-1", value={"name": "RSI Strategy"})
    graph.invoke.side_effect = [
        {"__interrupt__": [interrupt]},
        {
            "final_response": "```python\nimport backtrader as bt\n```",
            "citations": [],
            "sandbox_result": None,
            "route": None,
        },
    ]

    runner = BenchmarkRunner(graph, evaluators)
    item = _bench_item(
        eval_criteria={
            "factual_grounding": False,
            "cites_sources": False,
            "uncertainty_stated": False,
            "requires_code": True,
        }
    )
    result = runner.run_item(item)

    assert "backtrader" in (result.response or "")
    assert graph.invoke.call_count == 2
    resume_call = graph.invoke.call_args_list[1]
    assert isinstance(resume_call.args[0], Command)
    assert resume_call.args[0].resume == {"intr-1": interrupt.value}
    assert any(ev.evaluator == "code" for ev in result.evals)


def test_run_error_handling(evaluators):
    graph = MagicMock()
    graph.invoke.side_effect = RuntimeError("graph failed")

    runner = BenchmarkRunner(graph, evaluators)
    result = runner.run_item(_bench_item())

    assert result.error == "graph failed"
    assert result.response is None
    assert result.evals == []
