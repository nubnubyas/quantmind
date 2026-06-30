"""Unit tests for benchmark evaluators (E2 Phase 4)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.eval.evaluators import CitationEvaluator, CodeEvaluator, JudgeEvaluator
from src.sandbox import SandboxResult


@pytest.fixture
def mock_llm():
    return MagicMock()


def test_judge_high_score(mock_llm):
    mock_llm.chat_structured.return_value = {
        "answers_question": 0.9,
        "matches_expected": 0.85,
        "overall_score": 0.85,
        "reason": "Covers key momentum factor points.",
    }
    evaluator = JudgeEvaluator(mock_llm)
    result = evaluator.evaluate(
        query="Explain momentum factor",
        expected_behavior="Define momentum and cite A-share evidence",
        response="Momentum is past 12-1 month return ranking [1].",
        criteria={},
    )
    assert result.passed is True
    assert result.score >= 0.7
    assert result.evaluator == "judge"


def test_judge_low_score(mock_llm):
    mock_llm.chat_structured.return_value = {
        "answers_question": 0.1,
        "matches_expected": 0.05,
        "overall_score": 0.1,
        "reason": "Off-topic response.",
    }
    evaluator = JudgeEvaluator(mock_llm)
    result = evaluator.evaluate(
        query="Explain momentum factor",
        expected_behavior="Define momentum and cite A-share evidence",
        response="The weather is nice today.",
        criteria={},
    )
    assert result.passed is False
    assert result.score < 0.3


def test_code_syntax_valid():
    evaluator = CodeEvaluator()
    response = """Here is the strategy:

```python
import backtrader as bt

class MyStrategy(bt.Strategy):
  pass
```
"""
    result = evaluator.evaluate(response, None)
    assert result.passed is True
    assert result.score == 0.5


def test_code_syntax_error():
    evaluator = CodeEvaluator()
    response = """```python
def broken(
    pass
```"""
    result = evaluator.evaluate(response, None)
    assert result.passed is False
    assert result.score == 0.0


def test_citation_present_in_response():
    evaluator = CitationEvaluator()
    citations = [{"paper_id": "p1", "title": "Momentum", "section": None, "relevance_score": 0.9}]
    response = "Momentum works in A-shares [1]."
    result = evaluator.evaluate(citations, {"cites_sources": True}, response)
    assert result.passed is True
    assert result.score == 1.0


def test_citation_missing():
    evaluator = CitationEvaluator()
    result = evaluator.evaluate([], {"cites_sources": True}, "No inline markers here.")
    assert result.passed is False
    assert result.score == 0.0


def test_code_sandbox_success_without_syntax():
    evaluator = CodeEvaluator()
    sandbox = SandboxResult(
        success=True,
        phase="output",
        stdout="ok",
        stderr="",
        exit_code=0,
        timed_out=False,
        error=None,
        error_code=None,
    )
    result = evaluator.evaluate("no fenced code", sandbox)
    assert result.passed is True
    assert result.score == 0.8
