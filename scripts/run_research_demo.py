#!/usr/bin/env python3
"""Run Research Subgraph demo and acceptance checks for task pack B1."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv(ROOT / ".env")

# LangSmith APAC tracing
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_PROJECT", "quantmind")
os.environ.setdefault(
    "LANGSMITH_ENDPOINT", "https://apac.api.smith.langchain.com"
)

from src.agents.research_agent import compile_research_subgraph
from src.agents.state import _merge_retry_counts
from src.config.llm_client import create_llm_client
from src.tools.types import ToolResult
from src.vector_store.mock_vector_store import MockVectorStore
from src.vector_store.qdrant_client_wrapper import VectorStore

MOCK_PAPER_IDS = frozenset({"ff1993", "sharpe1966", "mom1997"})

DIVIDER = "\n" + "=" * 60 + "\n"


def create_vector_store() -> VectorStore:
    return VectorStore(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
    )


def _initial_state(user_id: str, message: str) -> dict:
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


def test_reducer() -> None:
    merged = _merge_retry_counts({"research": 1}, {"research": 2})
    assert merged["research"] == 2, merged
    print("✅ reducer _merge_retry_counts OK")


def test_llm_ping() -> None:
    client = create_llm_client()
    result = client.chat([{"role": "user", "content": "Reply with exactly: pong"}])
    assert result.text, "empty LLM response"
    assert result.usage["total_tokens"] >= 0
    print(f"✅ LLMClient.chat OK model={result.model} latency_ms={result.latency_ms}")
    print(f"   usage={result.usage}")


def test_happy_path() -> None:
    print(DIVIDER + "Happy path (real VectorStore): 解释动量因子")
    graph = compile_research_subgraph(create_vector_store(), create_llm_client())
    result = graph.invoke(_initial_state("demo_user", "解释动量因子"))
    assert result.get("final_response"), "missing final_response"
    print(result["final_response"][:500], "..." if len(result["final_response"]) > 500 else "")
    citations = result.get("citations") or []
    assert citations, "expected citations from real retrieval"
    paper_ids = [c["paper_id"] for c in citations]
    assert not MOCK_PAPER_IDS.intersection(paper_ids), (
        f"got mock paper_ids: {paper_ids}"
    )
    print(f"✅ citations count={len(citations)} paper_ids={paper_ids}")
    verification = result.get("verification")
    if verification:
        print(
            f"   verification score={verification.score:.2f} "
            f"allow_output={verification.allow_output}"
        )
    confidence = result.get("confidence")
    if confidence is not None:
        print(f"   confidence={confidence:.2f}")


def test_graceful_decline() -> None:
    print(DIVIDER + "Graceful decline: low confidence mock")
    empty_web = ToolResult(
        ok=True,
        tool_name="search_web",
        data={"result_type": "web_search_results", "results": [], "count": 0},
    )
    with patch("src.agents.research_agent.search_web", return_value=empty_web):
        graph = compile_research_subgraph(
            MockVectorStore(low_confidence=True), create_llm_client()
        )
        result = graph.invoke(_initial_state("demo_user", "解释 Fama-French 三因子模型"))
    final = result.get("final_response") or ""
    subgraph = (result.get("subgraph_outputs") or {}).get("research", {})
    assert subgraph.get("error") == "low_confidence", subgraph
    assert final, "graceful_decline should still produce final_response"
    print(final[:400], "...")
    print("✅ graceful_decline triggered")


def test_verify_retry() -> None:
    print(DIVIDER + "Verify retry: simulated first failure")
    graph = compile_research_subgraph(
        MockVectorStore(), create_llm_client(verify_fail_once=True)
    )
    result = graph.invoke(_initial_state("demo_user", "解释 Fama-French 三因子模型"))
    retries = (result.get("retry_counts") or {}).get("research", 0)
    assert retries >= 1, f"expected retry_counts['research'] >= 1, got {retries}"
    assert result.get("final_response"), "missing final_response after retry"
    print(f"✅ verify retry OK retry_counts['research']={retries}")
    verification = result.get("verification")
    if verification:
        print(f"   final verification score={verification.score:.2f}")


def print_langsmith_trace() -> None:
    try:
        from langsmith import Client

        client = Client()
        project = os.getenv("LANGSMITH_PROJECT", "quantmind")
        runs = list(client.list_runs(project_name=project, limit=1))
        if runs:
            url = client.get_run_url(run=runs[0])
            print(f"LangSmith latest trace: {url}")
        else:
            print("LangSmith: no runs found yet in project 'quantmind'")
    except Exception as exc:  # noqa: BLE001
        print(f"LangSmith: could not fetch trace URL ({exc})")


def main() -> None:
    print("QuantMind B1 Research Subgraph Demo")
    test_reducer()
    test_llm_ping()
    test_happy_path()
    test_graceful_decline()
    test_verify_retry()
    print(DIVIDER + "All acceptance checks passed.")
    print_langsmith_trace()


if __name__ == "__main__":
    main()
