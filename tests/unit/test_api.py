"""Unit tests for FastAPI integration layer (interface contract §C5)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from src.api.main import create_app
from src.api.models import ChatRequest, ChatResponse, InterruptPayload, ResumeRequest


@pytest.fixture
def mock_graph():
    return MagicMock()


@asynccontextmanager
async def api_client(mock_graph: MagicMock) -> AsyncIterator[tuple[AsyncClient, MagicMock]]:
    test_app = create_app(graph=mock_graph)
    async with test_app.router.lifespan_context(test_app):
        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac, mock_graph


@pytest.mark.anyio
async def test_chat_ok(mock_graph):
    mock_graph.invoke.return_value = {
        "final_response": "Momentum factors explain cross-sectional returns.",
        "citations": [
            {
                "paper_id": "2301.12345",
                "title": "Momentum Strategies",
                "section": "Abstract",
                "relevance_score": 0.92,
            }
        ],
    }

    async with api_client(mock_graph) as (ac, graph):
        response = await ac.post(
            "/chat",
            json={
                "user_id": "user-1",
                "thread_id": "thread-1",
                "message": "Explain momentum factor",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["thread_id"] == "thread-1"
    assert body["status"] == "ok"
    assert body["response"] == "Momentum factors explain cross-sectional returns."
    assert len(body["citations"]) == 1
    assert body["citations"][0]["paper_id"] == "2301.12345"
    assert body["interrupt"] is None

    graph.invoke.assert_called_once()
    invoke_state = graph.invoke.call_args.args[0]
    invoke_config = graph.invoke.call_args.kwargs["config"]
    assert invoke_state["user_id"] == "user-1"
    assert len(invoke_state["messages"]) == 1
    assert isinstance(invoke_state["messages"][0], HumanMessage)
    assert invoke_state["messages"][0].content == "Explain momentum factor"
    assert invoke_config == {"configurable": {"thread_id": "thread-1"}}


@pytest.mark.anyio
async def test_chat_interrupt(mock_graph):
    mock_graph.invoke.return_value = {
        "__interrupt__": [
            SimpleNamespace(
                id="interrupt-abc",
                value={
                    "strategy_spec": {
                        "signal": "MA crossover",
                        "period": "20/60",
                    }
                },
            )
        ],
    }

    async with api_client(mock_graph) as (ac, _graph):
        response = await ac.post(
            "/chat",
            json={
                "user_id": "user-1",
                "thread_id": "thread-2",
                "message": "Build a momentum strategy",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["thread_id"] == "thread-2"
    assert body["status"] == "interrupt"
    assert body["response"] is None
    assert body["citations"] == []
    assert body["interrupt"] == {
        "interrupt_id": "interrupt-abc",
        "type": "confirm_strategy",
        "strategy_spec": {"signal": "MA crossover", "period": "20/60"},
    }


@pytest.mark.anyio
async def test_resume(mock_graph):
    mock_graph.invoke.return_value = {
        "final_response": "Strategy confirmed and code generated.",
        "citations": [],
    }

    async with api_client(mock_graph) as (ac, graph):
        response = await ac.post(
            "/resume",
            json={
                "thread_id": "thread-2",
                "interrupt_id": "interrupt-abc",
                "edited_spec": {"signal": "MA crossover", "period": "10/30", "approved": True},
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["thread_id"] == "thread-2"
    assert body["status"] == "ok"
    assert body["response"] == "Strategy confirmed and code generated."
    assert body["interrupt"] is None

    graph.invoke.assert_called_once()
    invoke_arg = graph.invoke.call_args.args[0]
    invoke_config = graph.invoke.call_args.kwargs["config"]
    assert isinstance(invoke_arg, Command)
    assert invoke_arg.resume == {
        "interrupt-abc": {"signal": "MA crossover", "period": "10/30", "approved": True},
    }
    assert invoke_config == {"configurable": {"thread_id": "thread-2"}}


def test_models_match_contract():
    assert set(ChatRequest.model_fields) == {"user_id", "thread_id", "message"}
    assert set(ResumeRequest.model_fields) == {"thread_id", "interrupt_id", "edited_spec"}
    assert set(InterruptPayload.model_fields) == {"interrupt_id", "type", "strategy_spec"}
    assert set(ChatResponse.model_fields) == {
        "thread_id",
        "status",
        "response",
        "citations",
        "interrupt",
    }

    ok = ChatResponse(
        thread_id="t1",
        status="ok",
        response="hello",
        citations=[
            {
                "paper_id": "p1",
                "title": "Title",
                "section": None,
                "relevance_score": 0.8,
            }
        ],
    )
    assert ok.status == "ok"
    assert ok.interrupt is None

    interrupted = ChatResponse(
        thread_id="t1",
        status="interrupt",
        interrupt=InterruptPayload(
            interrupt_id="i1",
            type="confirm_strategy",
            strategy_spec={"signal": "momentum"},
        ),
    )
    assert interrupted.status == "interrupt"
    assert interrupted.response is None
