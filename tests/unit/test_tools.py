"""Unit tests for T1–T8 tools (interface contract §C2)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langgraph.store.memory import InMemoryStore
from sqlalchemy import text

from src.config.llm_client import LLMResult
from src.db import ApplicationRepo, create_all, get_session
from src.memory import UserMemory
from src.tools import (
    ToolResult,
    create_research_plan,
    explain_concept,
    fetch_paper_details,
    generate_backtest_code,
    generate_interview_questions,
    search_papers,
    track_application,
    update_user_memory,
)
from src.vector_store.mock_vector_store import MockVectorStore
from src.vector_store.types import RetrievalSpec, SearchResult


@pytest.fixture(scope="module")
def setup_db(requires_db):
    """Create all tables once per module. Skipped when PostgreSQL unreachable."""
    create_all()
    yield


@pytest.fixture
def db_session(setup_db):
    with get_session() as session:
        yield session
        session.execute(text("DELETE FROM research_notes"))
        session.execute(text("DELETE FROM applications"))
        session.execute(text("DELETE FROM ingestion_log"))


@pytest.fixture
def memory_store():
    return InMemoryStore()


@pytest.fixture
def memory(memory_store):
    return UserMemory(store=memory_store)


def test_search_papers_returns_results():
    result = search_papers("momentum factor", vector_store=MockVectorStore())

    assert isinstance(result, ToolResult)
    assert result.ok is True
    assert result.tool_name == "search_papers"
    assert result.data["result_type"] == "search_results"
    assert len(result.data["results"]) > 0
    assert result.data["count"] == len(result.data["results"])


def test_fetch_paper_details_from_payload():
    mock_store = MagicMock()
    mock_store.search.return_value = [
        SearchResult(
            text="Abstract paragraph one.",
            fusion_score=0.9,
            doc_id="doc1",
            chunk_id="c0",
            source="arxiv",
            paper_id="2301.12345",
            title="Momentum Strategies",
            section="Abstract",
            year=2023,
            authors=["Author A"],
            url="https://arxiv.org/abs/2301.12345",
        ),
        SearchResult(
            text="Abstract paragraph two.",
            fusion_score=0.85,
            doc_id="doc1",
            chunk_id="c1",
            source="arxiv",
            paper_id="2301.12345",
            title="Momentum Strategies",
            section="Abstract",
            year=2023,
            authors=["Author A"],
        ),
    ]

    result = fetch_paper_details("2301.12345", vector_store=mock_store)

    assert result.ok is True
    assert result.data["result_type"] == "paper_details"
    assert result.data["paper_id"] == "2301.12345"
    assert result.data["title"] == "Momentum Strategies"
    assert result.data["authors"] == ["Author A"]
    assert result.data["chunk_count"] == 2
    assert "Abstract paragraph one" in result.data["abstract"]


def test_fetch_paper_details_not_found():
    mock_store = MagicMock()
    mock_store.search.return_value = []

    result = fetch_paper_details("missing-id", vector_store=mock_store)

    assert result.ok is False
    assert result.error_code == "NOT_FOUND"


def test_generate_backtest_code():
    mock_llm = MagicMock()
    mock_llm.chat.return_value = LLMResult(
        text="import backtrader as bt\n\nclass MyStrategy(bt.Strategy):\n    pass",
        model="deepseek-v4-flash",
        usage={"input_tokens": 10, "output_tokens": 50, "total_tokens": 60},
        latency_ms=1200,
    )

    result = generate_backtest_code(
        "Buy when RSI < 30, sell when RSI > 70",
        framework="backtrader",
        llm=mock_llm,
    )

    assert result.ok is True
    assert result.data["result_type"] == "backtest_code"
    assert "backtrader" in result.data["code"]
    assert result.data["framework"] == "backtrader"


def test_explain_concept():
    mock_store = MagicMock()
    mock_store.search.return_value = [
        SearchResult(
            text="Sharpe ratio measures excess return per unit of volatility.",
            fusion_score=0.8,
            doc_id="sharpe",
            chunk_id="c0",
            source="manual",
            paper_id=None,
            title="Sharpe Ratio",
            section=None,
            year=None,
        )
    ]
    mock_llm = MagicMock()
    mock_llm.chat.return_value = LLMResult(
        text="The Sharpe ratio is a risk-adjusted performance metric.",
        model="deepseek-v4-flash",
        usage={},
        latency_ms=500,
    )

    result = explain_concept(
        "Sharpe ratio",
        depth="technical",
        vector_store=mock_store,
        llm=mock_llm,
    )

    assert result.ok is True
    assert result.data["result_type"] == "concept_explanation"
    assert result.data["concept"] == "Sharpe ratio"
    assert "Sharpe" in result.data["explanation"]
    assert len(result.data["sources"]) == 1


def test_generate_interview_questions():
    mock_llm = MagicMock()
    mock_llm.chat_structured.return_value = {
        "questions": [
            {
                "question": "Explain momentum factor construction.",
                "category": "quant",
                "difficulty": "medium",
            }
        ]
    }

    result = generate_interview_questions(
        "Quant researcher role requiring Python and statistics.",
        company="Two Sigma",
        num_questions=1,
        llm=mock_llm,
    )

    assert result.ok is True
    assert result.data["result_type"] == "interview_questions"
    assert result.data["count"] == 1
    assert result.data["questions"][0]["question"]


def test_create_research_plan_idempotent(memory):
    mock_llm = MagicMock()
    mock_llm.chat_structured.return_value = {
        "steps": [
            {
                "step_id": "s1",
                "title": "Read foundational paper",
                "description": "Study Fama-French model",
                "status": "pending",
            }
        ]
    }

    first = create_research_plan(
        "user1",
        "Learn factor investing",
        current_level="beginner",
        memory=memory,
        llm=mock_llm,
    )
    second = create_research_plan(
        "user1",
        "Learn factor investing",
        current_level="beginner",
        memory=memory,
        llm=mock_llm,
    )

    assert first.ok is True
    assert second.ok is True
    assert first.data["plan_id"] == second.data["plan_id"]
    assert second.data["existing"] is True
    assert mock_llm.chat_structured.call_count == 1


def test_update_user_memory_profile(memory):
    result = update_user_memory(
        "user_alice",
        "profile",
        {"research_interests": ["momentum", "stat_arb"], "skill_level": "intermediate"},
        memory=memory,
    )

    assert result.ok is True
    assert result.data["result_type"] == "memory_update"
    assert result.data["memory_type"] == "profile"
    assert result.data["stored"]["research_interests"] == ["momentum", "stat_arb"]

    profile = memory.get_profile("user_alice")
    assert profile is not None
    assert profile.skill_level == "intermediate"


def test_update_user_memory_bookmark(memory):
    result = update_user_memory(
        "user1",
        "bookmark",
        {"paper_id": "paper-123", "note": "Key reference"},
        memory=memory,
    )

    assert result.ok is True
    bookmarks = memory.list_bookmarks("user1")
    assert len(bookmarks) == 1
    assert bookmarks[0]["paper_id"] == "paper-123"


def test_track_application_idempotent_create(db_session):
    repo = ApplicationRepo(db_session)

    first = track_application(
        "user1",
        "Citadel",
        "Quant Researcher",
        "create",
        repo=repo,
        session=db_session,
    )
    second = track_application(
        "user1",
        "Citadel",
        "Quant Researcher",
        "create",
        repo=repo,
        session=db_session,
    )

    assert first.ok is True
    assert second.ok is True
    assert first.data["app_id"] == second.data["app_id"]
    assert second.data["existing"] is True

    apps = repo.list_by_user("user1")
    assert len(apps) == 1


def test_track_application_update_status(db_session):
    repo = ApplicationRepo(db_session)

    created = track_application(
        "user1",
        "Jane Street",
        "Trader",
        "create",
        repo=repo,
        session=db_session,
    )
    updated = track_application(
        "user1",
        "Jane Street",
        "Trader",
        "update_status",
        status="applied",
        repo=repo,
        session=db_session,
    )

    assert updated.ok is True
    assert updated.data["status"] == "applied"
    assert updated.data["app_id"] == created.data["app_id"]
