"""Unit tests for CodeGen Subgraph (task pack B2)."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.agents.codegen_agent import SAMPLE_DATA_PATH, compile_codegen_subgraph
from src.config.llm_client import LLMResult
from src.sandbox import SandboxResult, SandboxRunner


def _strategy_spec_raw(**overrides: object) -> dict:
    base = {
        "name": "RSI Mean Reversion",
        "description": "RSI 均值回归策略",
        "framework": "backtrader",
        "parameters": {"rsi_period": 14, "buy_threshold": 30},
        "signal_logic": "Buy when RSI < 30, sell when RSI > 70",
        "asset_class": "equity",
        "timeframe": "daily",
    }
    base.update(overrides)
    return base


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


def _sandbox_success(stdout: str = "Final Portfolio Value: 100000.00") -> SandboxResult:
    return SandboxResult(
        success=True,
        phase="output",
        stdout=stdout,
        stderr="",
        exit_code=0,
        timed_out=False,
        error=None,
        error_code=None,
    )


def _sandbox_failure(error: str = "SyntaxError: invalid syntax") -> SandboxResult:
    return SandboxResult(
        success=False,
        phase="syntax",
        stdout="",
        stderr="",
        exit_code=None,
        timed_out=False,
        error=error,
        error_code="SYNTAX_ERROR",
    )


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.chat_structured.return_value = _strategy_spec_raw()
    llm.chat.return_value = LLMResult(
        text="import backtrader as bt\n\nclass MyStrategy(bt.Strategy):\n    pass\nprint('ok')",
        model="deepseek-chat",
        usage={"input_tokens": 10, "output_tokens": 50, "total_tokens": 60},
        latency_ms=500,
    )
    return llm


@pytest.fixture
def mock_sandbox():
    runner = MagicMock(spec=SandboxRunner)
    runner.run.return_value = _sandbox_success()
    return runner


@pytest.fixture
def checkpointer():
    return InMemorySaver()


def test_compile_signature():
    sig = inspect.signature(compile_codegen_subgraph)
    params = list(sig.parameters.keys())
    assert params == ["llm_client", "sandbox_runner", "checkpointer"]
    assert sig.parameters["sandbox_runner"].default is None
    assert sig.parameters["checkpointer"].kind == inspect.Parameter.KEYWORD_ONLY


def test_happy_path_with_interrupt(mock_llm, mock_sandbox, checkpointer):
    graph = compile_codegen_subgraph(mock_llm, mock_sandbox, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "codegen-happy-1"}}

    r1 = graph.invoke(_initial_state("RSI 均值回归策略"), config)
    interrupts = r1.get("__interrupt__") or []
    assert interrupts, "expected interrupt at confirm_with_user"
    assert interrupts[0].value["name"] == "RSI Mean Reversion"

    edited = {**interrupts[0].value, "parameters": {"rsi_period": 21}}
    r2 = graph.invoke(Command(resume={interrupts[0].id: edited}), config)

    assert r2.get("strategy_spec") == edited
    assert r2.get("generated_code")
    assert r2.get("sandbox_result") and r2["sandbox_result"].success
    assert r2.get("final_response")

    subgraph = (r2.get("subgraph_outputs") or {}).get("codegen", {})
    assert subgraph.get("mode") == "codegen"
    assert subgraph.get("error") is None
    assert "Sandbox output" in (subgraph.get("result") or "")

    mock_sandbox.run.assert_called_once()
    call_kwargs = mock_sandbox.run.call_args.kwargs
    assert call_kwargs["sample_data_path"] == SAMPLE_DATA_PATH
    assert call_kwargs["timeout_s"] == 30


def test_generate_code_uses_confirmed_spec(mock_llm, mock_sandbox, checkpointer):
    graph = compile_codegen_subgraph(mock_llm, mock_sandbox, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "codegen-spec-1"}}

    r1 = graph.invoke(_initial_state("RSI strategy"), config)
    intr = r1["__interrupt__"][0]
    edited = {
        **intr.value,
        "signal_logic": "Buy when RSI < 25, sell when RSI > 75",
    }
    graph.invoke(Command(resume={intr.id: edited}), config)

    user_msg = mock_llm.chat.call_args[0][0][1]["content"]
    assert "Buy when RSI < 25" in user_msg


def test_sandbox_failure_retry_exhausted(mock_llm, mock_sandbox, checkpointer):
    mock_sandbox.run.return_value = _sandbox_failure("RuntimeError: boom")
    graph = compile_codegen_subgraph(mock_llm, mock_sandbox, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "codegen-fail-1"}}

    r1 = graph.invoke(_initial_state("Broken strategy"), config)
    intr = r1["__interrupt__"][0]
    r2 = graph.invoke(Command(resume={intr.id: intr.value}), config)

    assert r2.get("retry_counts", {}).get("codegen") == 2
    assert mock_sandbox.run.call_count == 2

    subgraph = (r2.get("subgraph_outputs") or {}).get("codegen", {})
    assert subgraph.get("error") == "sandbox_failed_after_retry"
    assert "RuntimeError: boom" in (r2.get("final_response") or "")


def test_sandbox_failure_retry_success(mock_llm, mock_sandbox, checkpointer):
    mock_sandbox.run.side_effect = [
        _sandbox_failure("first attempt failed"),
        _sandbox_success("Final Portfolio Value: 105000.00"),
    ]
    graph = compile_codegen_subgraph(mock_llm, mock_sandbox, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "codegen-retry-ok-1"}}

    r1 = graph.invoke(_initial_state("Retry strategy"), config)
    intr = r1["__interrupt__"][0]
    r2 = graph.invoke(Command(resume={intr.id: intr.value}), config)

    assert r2.get("retry_counts", {}).get("codegen") == 1
    assert mock_sandbox.run.call_count == 2

    subgraph = (r2.get("subgraph_outputs") or {}).get("codegen", {})
    assert subgraph.get("error") is None
    assert "105000.00" in (r2.get("final_response") or "")


def test_confirm_node_no_side_effects_before_interrupt(mock_llm, mock_sandbox, checkpointer):
    side_effect_calls: list[str] = []

    def fake_side_effect(*_args, **_kwargs):
        side_effect_calls.append("sandbox_run")
        return _sandbox_success()

    mock_sandbox.run.side_effect = fake_side_effect

    with patch("src.agents.codegen_agent.generate_backtest_code") as mock_t3:
        mock_t3.side_effect = lambda **_kw: side_effect_calls.append("generate") or MagicMock(
            ok=True, data={"code": "print('x')"}
        )

        graph = compile_codegen_subgraph(mock_llm, mock_sandbox, checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "codegen-side-effect-1"}}
        r1 = graph.invoke(_initial_state("Side effect check"), config)

        assert r1.get("__interrupt__"), "should pause before generate/sandbox"
        assert side_effect_calls == [], "no T3 or sandbox calls before interrupt resume"

        intr = r1["__interrupt__"][0]
        graph.invoke(Command(resume={intr.id: intr.value}), config)
        assert "generate" in side_effect_calls
        assert "sandbox_run" in side_effect_calls
