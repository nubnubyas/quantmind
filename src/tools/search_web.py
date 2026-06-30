"""T9: Web search via DuckDuckGo (fallback when Qdrant retrieval is insufficient)."""

from __future__ import annotations

from src.tools._helpers import fail, ok
from src.tools.types import ToolResult

TOOL_NAME = "search_web"


def search_web(
    query: str,
    max_results: int = 5,
) -> ToolResult:
    """Search the web using DuckDuckGo and return structured results.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return (default 5, max 10).

    Returns:
        ToolResult with data={"result_type": "web_search_results", "results": [...]}
    """
    if not query or not query.strip():
        return fail(TOOL_NAME, "Empty search query", "VALIDATION_ERROR")

    max_results = min(max(max_results, 1), 10)

    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            raw = list(ddgs.text(query.strip(), max_results=max_results))

        if not raw:
            return ok(TOOL_NAME, "web_search_results", results=[], count=0)

        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in raw
        ]
        return ok(TOOL_NAME, "web_search_results", results=results, count=len(results))

    except ImportError:
        return fail(
            TOOL_NAME,
            "duckduckgo_search package not installed. Run: pip install duckduckgo_search",
            "DEPENDENCY_MISSING",
        )
    except Exception as exc:
        msg = str(exc)
        retryable = "rate" in msg.lower() or "timeout" in msg.lower() or "429" in msg
        return fail(TOOL_NAME, msg, "WEB_SEARCH_ERROR", retryable=retryable)
