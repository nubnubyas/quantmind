"""Unit tests for Streamlit UI API client and response handling."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.api.models import ChatResponse, InterruptPayload
from src.ui.streamlit_app import (
    api_chat,
    api_resume,
    apply_chat_response,
    build_edited_spec,
    process_chat_response,
)


def _mock_response(payload: dict, status_code: int = 200) -> MagicMock:
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = payload
    mock.raise_for_status = MagicMock()
    return mock


def test_api_chat_ok():
    payload = {
        "thread_id": "thread-1",
        "status": "ok",
        "response": "Momentum factors explain cross-sectional returns.",
        "citations": [
            {
                "paper_id": "2301.12345",
                "title": "Momentum Strategies",
                "section": "Abstract",
                "relevance_score": 0.92,
            }
        ],
        "interrupt": None,
    }
    with patch("src.ui.streamlit_app.requests.post", return_value=_mock_response(payload)) as post:
        result = api_chat("user-1", "解释动量因子", "thread-1")

    assert result is not None
    assert result.status == "ok"
    assert result.response == "Momentum factors explain cross-sectional returns."
    assert len(result.citations) == 1
    assert result.citations[0]["title"] == "Momentum Strategies"
    assert result.interrupt is None

    post.assert_called_once()
    call_kwargs = post.call_args.kwargs
    assert call_kwargs["json"] == {
        "user_id": "user-1",
        "thread_id": "thread-1",
        "message": "解释动量因子",
    }
    assert call_kwargs["timeout"] == 120


def test_interrupt_trigger_and_resume():
    interrupt_payload = InterruptPayload(
        interrupt_id="interrupt-abc",
        type="confirm_strategy",
        strategy_spec={
            "name": "RSI Mean Reversion",
            "framework": "backtrader",
            "parameters": {"rsi_period": 14},
            "signal_logic": "RSI<30 buy, RSI>70 sell",
        },
    )
    chat_response = ChatResponse(
        thread_id="thread-2",
        status="interrupt",
        interrupt=interrupt_payload,
    )

    message = process_chat_response(chat_response)
    assert message["role"] == "assistant"
    assert message["interrupt"] is not None
    assert message["interrupt"]["interrupt_id"] == "interrupt-abc"
    assert message["citations"] is None

    new_messages, pending = apply_chat_response(chat_response, None)
    assert len(new_messages) == 1
    assert pending is not None
    assert pending.interrupt_id == "interrupt-abc"

    edited_spec, error = build_edited_spec(
        interrupt_payload.strategy_spec,
        name="RSI Updated",
        framework="vectorbt",
        parameters_str='{"rsi_period": 10}',
        signal_logic="RSI<25 buy",
    )
    assert error is None
    assert edited_spec["name"] == "RSI Updated"
    assert edited_spec["framework"] == "vectorbt"
    assert edited_spec["parameters"] == {"rsi_period": 10}

    resume_payload = {
        "thread_id": "thread-2",
        "status": "ok",
        "response": "class RSIStrategy(bt.Strategy):\n    ...\n\nSandbox execution passed.",
        "citations": [],
        "interrupt": None,
    }
    with patch(
        "src.ui.streamlit_app.requests.post",
        return_value=_mock_response(resume_payload),
    ) as post:
        result = api_resume("thread-2", interrupt_payload, edited_spec)

    assert result is not None
    assert result.status == "ok"
    assert "RSIStrategy" in (result.response or "")

    post.assert_called_once()
    assert post.call_args.kwargs["json"] == {
        "thread_id": "thread-2",
        "interrupt_id": "interrupt-abc",
        "edited_spec": edited_spec,
    }

    final_messages, final_pending = apply_chat_response(result, pending)
    assert final_pending is None
    assert final_messages[0]["content"].startswith("class RSIStrategy")


@patch("src.ui.streamlit_app.st.error")
def test_api_error_handling(mock_error):
    with patch(
        "src.ui.streamlit_app.requests.post",
        side_effect=requests.exceptions.ConnectionError("Connection refused"),
    ):
        result = api_chat("user-1", "hello", "thread-1")

    assert result is None
    mock_error.assert_called_once()
    assert "API 请求失败" in mock_error.call_args.args[0]

    interrupt = InterruptPayload(
        interrupt_id="i1",
        type="confirm_strategy",
        strategy_spec={"name": "test"},
    )
    with patch(
        "src.ui.streamlit_app.requests.post",
        side_effect=requests.exceptions.ConnectionError("Connection refused"),
    ):
        resume_result = api_resume("thread-1", interrupt, {"name": "test"})

    assert resume_result is None
    assert mock_error.call_count == 2
    assert "Resume 请求失败" in mock_error.call_args.args[0]


def test_build_edited_spec_invalid_json():
    spec, error = build_edited_spec(
        {"name": "x"},
        name="x",
        framework="backtrader",
        parameters_str="not-json",
        signal_logic="logic",
    )
    assert spec is None
    assert error is not None
    assert "JSON" in error
