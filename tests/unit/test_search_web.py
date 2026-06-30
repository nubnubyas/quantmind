"""Unit tests for T9 search_web tool."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.tools.search_web import search_web
from src.tools.types import ToolResult


@patch("duckduckgo_search.DDGS")
def test_search_web_happy_path(mock_ddgs_cls):
    mock_ddgs = MagicMock()
    mock_ddgs.text.return_value = [
        {"title": "Sharpe Ratio", "href": "https://example.com/sharpe", "body": "Definition..."},
        {"title": "Risk Metrics", "href": "https://example.com/risk", "body": "Overview..."},
        {"title": "Portfolio Theory", "href": "https://example.com/portfolio", "body": "Basics..."},
    ]
    mock_ddgs_cls.return_value.__enter__.return_value = mock_ddgs
    mock_ddgs_cls.return_value.__exit__.return_value = None

    result = search_web("Sharpe Ratio calculation", max_results=3)

    assert isinstance(result, ToolResult)
    assert result.ok is True
    assert result.tool_name == "search_web"
    assert result.data["result_type"] == "web_search_results"
    assert result.data["count"] == 3
    assert len(result.data["results"]) == 3
    first = result.data["results"][0]
    assert first["title"] == "Sharpe Ratio"
    assert first["url"] == "https://example.com/sharpe"
    assert first["snippet"] == "Definition..."


@patch("duckduckgo_search.DDGS")
def test_search_web_empty(mock_ddgs_cls):
    mock_ddgs = MagicMock()
    mock_ddgs.text.return_value = []
    mock_ddgs_cls.return_value.__enter__.return_value = mock_ddgs
    mock_ddgs_cls.return_value.__exit__.return_value = None

    result = search_web("nonexistent query xyz")

    assert result.ok is True
    assert result.data["result_type"] == "web_search_results"
    assert result.data["results"] == []
    assert result.data["count"] == 0


def test_search_web_validation_error():
    result = search_web("")
    assert result.ok is False
    assert result.error_code == "VALIDATION_ERROR"

    result = search_web("   ")
    assert result.ok is False
    assert result.error_code == "VALIDATION_ERROR"


@patch("duckduckgo_search.DDGS")
def test_search_web_error(mock_ddgs_cls):
    mock_ddgs = MagicMock()
    mock_ddgs.text.side_effect = RuntimeError("connection failed")
    mock_ddgs_cls.return_value.__enter__.return_value = mock_ddgs
    mock_ddgs_cls.return_value.__exit__.return_value = None

    result = search_web("Sharpe Ratio")

    assert result.ok is False
    assert result.error_code == "WEB_SEARCH_ERROR"
    assert result.retryable is False
    assert "connection failed" in (result.error or "")


@patch("duckduckgo_search.DDGS")
def test_search_web_rate_limit(mock_ddgs_cls):
    mock_ddgs = MagicMock()
    mock_ddgs.text.side_effect = RuntimeError("RatelimitException: HTTP 202")
    mock_ddgs_cls.return_value.__enter__.return_value = mock_ddgs
    mock_ddgs_cls.return_value.__exit__.return_value = None

    result = search_web("Sharpe Ratio")

    assert result.ok is False
    assert result.error_code == "WEB_SEARCH_ERROR"
    assert result.retryable is True
