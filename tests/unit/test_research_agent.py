"""Unit tests for Research Subgraph web search fallback (T9)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from src.agents.research_agent import _web_to_search_results, compile_research_subgraph
from src.config.llm_client import LLMResult
from src.tools.types import ToolResult
from src.vector_store.mock_vector_store import MockVectorStore
from src.vector_store.types import SearchResult


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


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.chat_structured.side_effect = [
        {
            "keywords": ["Sharpe", "Ratio"],
            "domain": "risk",
            "intent": "explain",
            "doc_type": None,
            "time_range": None,
        },
        {
            "answers_question": True,
            "claims_grounded": True,
            "no_hallucination": True,
            "uncertainty_stated": True,
            "score": 0.9,
            "failure_reasons": [],
        },
    ]
    llm.chat.return_value = LLMResult(
        text="Sharpe ratio measures risk-adjusted return.",
        model="mock",
        usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        latency_ms=100,
    )
    return llm


def _web_search_results(count: int = 2) -> ToolResult:
    results = [
        {
            "title": f"Web Result {i}",
            "url": f"https://example.com/{i}",
            "snippet": f"Snippet about Sharpe ratio {i}.",
        }
        for i in range(count)
    ]
    return ToolResult(
        ok=True,
        tool_name="search_web",
        data={"result_type": "web_search_results", "results": results, "count": count},
    )


def test_web_search_always_called_in_hybrid(mock_llm):
    """Verify web search is always called as supplemental source, even with high Qdrant confidence."""
    mock_llm.chat_structured.side_effect = [
        {
            "keywords": ["test"],
            "domain": "test",
            "intent": "test",
            "doc_type": None,
            "time_range": None,
        },
        {
            "answers_question": True,
            "claims_grounded": True,
            "no_hallucination": True,
            "uncertainty_stated": True,
            "score": 0.9,
            "failure_reasons": [],
        },
    ]
    mock_llm.chat.return_value = LLMResult(
        text="Test response.",
        model="mock",
        usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        latency_ms=100,
    )

    with patch("src.agents.research_agent.search_web") as mock_web:
        mock_web.return_value = ToolResult(
            ok=True,
            tool_name="search_web",
            data={"result_type": "web_search_results", "results": [], "count": 0},
        )
        graph = compile_research_subgraph(MockVectorStore(), mock_llm)
        graph.invoke(_initial_state("explain momentum factor"))

    mock_web.assert_called_once()


def test_web_search_fallback(mock_llm):
    with patch(
        "src.agents.research_agent.search_web",
        return_value=_web_search_results(2),
    ) as mock_search:
        graph = compile_research_subgraph(MockVectorStore(low_confidence=True), mock_llm)
        result = graph.invoke(_initial_state("什么是 Sharpe Ratio？"))

    assert mock_search.call_count >= 1
    subgraph = (result.get("subgraph_outputs") or {}).get("research", {})
    assert subgraph.get("error") != "low_confidence"
    assert result.get("final_response")


def test_web_search_fallback_then_decline(mock_llm):
    mock_llm.chat_structured.side_effect = [
        {
            "keywords": ["Fama", "French"],
            "domain": "factor",
            "intent": "explain",
            "doc_type": None,
            "time_range": None,
        },
    ]
    mock_llm.chat.return_value = LLMResult(
        text="信息不足，建议从以下方向继续研究...",
        model="mock",
        usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        latency_ms=100,
    )

    empty_web = ToolResult(
        ok=True,
        tool_name="search_web",
        data={"result_type": "web_search_results", "results": [], "count": 0},
    )
    with patch("src.agents.research_agent.search_web", return_value=empty_web):
        graph = compile_research_subgraph(MockVectorStore(low_confidence=True), mock_llm)
        result = graph.invoke(_initial_state("解释 Fama-French 三因子模型"))

    subgraph = (result.get("subgraph_outputs") or {}).get("research", {})
    assert subgraph.get("error") == "low_confidence"
    assert result.get("final_response")


def test_web_to_search_results_mapping():
    """Verify field mapping and SearchResult dataclass field order."""
    web_data = [
        {
            "title": "Sharpe Ratio Explained",
            "url": "https://example.com/sharpe",
            "snippet": "The Sharpe ratio measures risk-adjusted return.",
        },
        {
            "title": "Risk Metrics Overview",
            "url": "https://example.com/risk",
            "snippet": "Various risk metrics including Sharpe, Sortino, etc.",
        },
    ]
    results = _web_to_search_results(web_data)

    assert len(results) == 2
    # First result
    r0 = results[0]
    assert r0.text == "The Sharpe ratio measures risk-adjusted return."
    assert r0.fusion_score == 0.5
    assert r0.doc_id == "web-0"
    assert r0.chunk_id == "web-0"
    assert r0.source == "web"
    assert r0.paper_id is None
    assert r0.title == "Sharpe Ratio Explained"
    assert r0.section is None
    assert r0.year is None
    assert r0.url == "https://example.com/sharpe"
    # Second result
    r1 = results[1]
    assert r1.doc_id == "web-1"
    assert r1.title == "Risk Metrics Overview"
    # Empty input
    assert _web_to_search_results([]) == []


def test_web_search_retry_loop(mock_llm):
    """When verify fails on web-sourced content, retry loop calls web_search
    again (bounded by retry limit, max 2 calls to web_search total)."""
    call_count = [0]

    def _counted_web_search(*args, **kwargs):
        call_count[0] += 1
        return _web_search_results(2)

    mock_llm.chat_structured.side_effect = [
        # parse_query
        {
            "keywords": ["test"],
            "domain": "test",
            "intent": "test",
            "doc_type": None,
            "time_range": None,
        },
        # 1st verify_answer: fail to trigger retry
        {
            "answers_question": False,
            "claims_grounded": False,
            "no_hallucination": False,
            "uncertainty_stated": False,
            "score": 0.3,
            "failure_reasons": ["retry test"],
        },
        # 2nd verify_answer: fail again to trigger format_doubtful
        {
            "answers_question": False,
            "claims_grounded": False,
            "no_hallucination": False,
            "uncertainty_stated": False,
            "score": 0.3,
            "failure_reasons": ["still failing"],
        },
    ]
    mock_llm.chat.return_value = LLMResult(
        text="Mocked draft for retry test.",
        model="mock",
        usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        latency_ms=100,
    )

    with patch(
        "src.agents.research_agent.search_web",
        side_effect=_counted_web_search,
    ):
        graph = compile_research_subgraph(MockVectorStore(low_confidence=True), mock_llm)
        result = graph.invoke(_initial_state("test query"))

    # hybrid_search always calls web; low_confidence may also route to web_search fallback; retries add more
    assert 2 <= call_count[0] <= 4, f"Expected 2-4 web_search calls, got {call_count[0]}"
    # Should end in format_doubtful (verify failed twice → retries exhausted)
    subgraph = (result.get("subgraph_outputs") or {}).get("research", {})
    assert subgraph.get("error") == "verification_failed_after_retry"
    assert result.get("final_response")


def test_hybrid_search_interleaves_web_results():
    """Web search results should be interleaved, not appended at end."""
    mock_vs = MagicMock()
    mock_vs.search.return_value = [
        SearchResult(
            text="qdrant1",
            fusion_score=0.7,
            source="arxiv",
            doc_id="p1",
            chunk_id="p1-0",
            paper_id="p1",
            title="Paper 1",
            section=None,
            year=None,
        ),
        SearchResult(
            text="qdrant2",
            fusion_score=0.6,
            source="arxiv",
            doc_id="p2",
            chunk_id="p2-0",
            paper_id="p2",
            title="Paper 2",
            section=None,
            year=None,
        ),
        SearchResult(
            text="qdrant3",
            fusion_score=0.5,
            source="arxiv",
            doc_id="p3",
            chunk_id="p3-0",
            paper_id="p3",
            title="Paper 3",
            section=None,
            year=None,
        ),
    ]
    mock_llm = MagicMock()
    mock_llm.chat_structured.side_effect = [
        {
            "keywords": ["test"],
            "domain": "test",
            "intent": "test",
            "doc_type": None,
            "time_range": None,
        },
        {
            "answers_question": True,
            "claims_grounded": True,
            "no_hallucination": True,
            "uncertainty_stated": True,
            "score": 0.9,
            "failure_reasons": [],
        },
    ]
    mock_llm.chat.return_value = LLMResult(
        text="Test response.",
        model="mock",
        usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        latency_ms=100,
    )

    web_data = [
        {"snippet": "web snippet 1", "title": "Web 1", "url": "https://a.com"},
        {"snippet": "web snippet 2", "title": "Web 2", "url": "https://b.com"},
    ]
    mock_web_result = ToolResult(
        ok=True,
        tool_name="search_web",
        data={"result_type": "web_search_results", "results": web_data, "count": 2},
    )

    with patch("src.agents.research_agent.search_web", return_value=mock_web_result):
        graph = compile_research_subgraph(mock_vs, mock_llm)
        result = graph.invoke(_initial_state("test query"))

    retrieved = result.get("retrieved") or []
    sources = [r.source for r in retrieved]
    assert "web" in sources, "Web results should be present"
    web_indices = [i for i, s in enumerate(sources) if s == "web"]
    last_qdrant_idx = max(i for i, s in enumerate(sources) if s != "web")
    assert any(i < last_qdrant_idx for i in web_indices), (
        f"Web results should be interleaved, not all at end. Sources: {sources}"
    )


def test_synthesize_prompt_requires_multiple_sources():
    """synthesize_answer prompt must instruct LLM to use multiple sources."""
    mock_vs = MagicMock()
    mock_vs.search.return_value = [
        SearchResult(
            text="source A content about Sharpe Ratio",
            fusion_score=0.7,
            source="arxiv",
            doc_id="p1",
            chunk_id="p1-0",
            paper_id="p1",
            title="Sharpe Ratio Analysis",
            section="Introduction",
            year=None,
        ),
        SearchResult(
            text="web: Sharpe = (Rp - Rf) / σp",
            fusion_score=0.45,
            source="web",
            doc_id="web-0",
            chunk_id="web-0",
            paper_id=None,
            title="Sharpe Ratio Definition",
            section=None,
            year=None,
        ),
    ]
    mock_llm = MagicMock()
    mock_llm.chat_structured.side_effect = [
        {
            "keywords": ["Sharpe", "Ratio"],
            "domain": "risk",
            "intent": "explain",
            "doc_type": None,
            "time_range": None,
        },
        {
            "answers_question": True,
            "claims_grounded": True,
            "no_hallucination": True,
            "uncertainty_stated": True,
            "score": 0.9,
            "failure_reasons": [],
        },
    ]
    mock_llm.chat.return_value = LLMResult(
        text="The Sharpe Ratio = (Rp - Rf) / σp [2]. It measures risk-adjusted return [1].",
        model="deepseek-v4-flash",
        usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        latency_ms=500,
    )

    web_result = ToolResult(
        ok=False,
        tool_name="search_web",
        data={},
        error="no results",
    )

    with patch("src.agents.research_agent.search_web", return_value=web_result):
        graph = compile_research_subgraph(mock_vs, mock_llm)
        graph.invoke(_initial_state("What is Sharpe Ratio?"))

    call_args = mock_llm.chat.call_args
    system_prompt = call_args[0][0][0]["content"]
    assert "at least 2 DIFFERENT sources" in system_prompt.lower() or (
        "different sources" in system_prompt.lower()
    ), f"Prompt should require multiple sources. Got: {system_prompt[:200]}"


def test_hybrid_search_demotes_alpha_gpt_chunks():
    """Alpha-GPT survey chunks should be moved to end for non-alpha queries."""
    alpha_chunk = SearchResult(
        text="Alpha-GPT's workflow consists of three distinct stages for human-AI collaboration.",
        fusion_score=0.95,
        source="arxiv",
        doc_id="alpha-gpt",
        chunk_id="alpha-gpt-0",
        paper_id="alpha-gpt",
        title="Alpha-GPT: Human-AI Interactive Alpha Mining",
        section="Introduction",
        year=2024,
    )
    sharpe_chunk = SearchResult(
        text="The Sharpe ratio measures risk-adjusted return as excess return per unit of volatility.",
        fusion_score=0.78,
        source="arxiv",
        doc_id="sharpe1966",
        chunk_id="sharpe1966-0",
        paper_id="sharpe1966",
        title="Sharpe Ratio and Risk-Adjusted Returns",
        section="Definition",
        year=1966,
    )
    mock_vs = MagicMock()
    mock_vs.search.return_value = [alpha_chunk, sharpe_chunk]
    mock_llm = MagicMock()
    mock_llm.chat_structured.side_effect = [
        {
            "keywords": ["Sharpe", "Ratio"],
            "domain": "risk",
            "intent": "explain",
            "doc_type": None,
            "time_range": None,
        },
        {
            "answers_question": True,
            "claims_grounded": True,
            "no_hallucination": True,
            "uncertainty_stated": True,
            "score": 0.9,
            "failure_reasons": [],
        },
    ]
    mock_llm.chat.return_value = LLMResult(
        text="Sharpe ratio measures risk-adjusted return.",
        model="mock",
        usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        latency_ms=100,
    )

    empty_web = ToolResult(
        ok=True,
        tool_name="search_web",
        data={"result_type": "web_search_results", "results": [], "count": 0},
    )

    with patch("src.agents.research_agent.search_web", return_value=empty_web):
        graph = compile_research_subgraph(mock_vs, mock_llm)
        result = graph.invoke(_initial_state("What is Sharpe Ratio?"))

    retrieved = result.get("retrieved") or []
    assert len(retrieved) >= 2
    assert retrieved[0].paper_id == "sharpe1966"
    assert retrieved[-1].paper_id == "alpha-gpt"


def test_hybrid_search_keeps_alpha_gpt_for_alpha_queries():
    """Alpha-GPT chunks should stay in place when query is about alpha mining."""
    alpha_chunk = SearchResult(
        text="Alpha-GPT's workflow consists of three distinct stages for human-AI collaboration.",
        fusion_score=0.95,
        source="arxiv",
        doc_id="alpha-gpt",
        chunk_id="alpha-gpt-0",
        paper_id="alpha-gpt",
        title="Alpha-GPT: Human-AI Interactive Alpha Mining",
        section="Introduction",
        year=2024,
    )
    other_chunk = SearchResult(
        text="Some other quant content.",
        fusion_score=0.5,
        source="arxiv",
        doc_id="other",
        chunk_id="other-0",
        paper_id="other",
        title="Other Paper",
        section=None,
        year=None,
    )
    mock_vs = MagicMock()
    mock_vs.search.return_value = [alpha_chunk, other_chunk]
    mock_llm = MagicMock()
    mock_llm.chat_structured.side_effect = [
        {
            "keywords": ["Alpha-GPT", "mining"],
            "domain": "alpha",
            "intent": "explain",
            "doc_type": None,
            "time_range": None,
        },
        {
            "answers_question": True,
            "claims_grounded": True,
            "no_hallucination": True,
            "uncertainty_stated": True,
            "score": 0.9,
            "failure_reasons": [],
        },
    ]
    mock_llm.chat.return_value = LLMResult(
        text="Alpha-GPT uses human-AI collaboration for alpha mining.",
        model="mock",
        usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        latency_ms=100,
    )

    empty_web = ToolResult(
        ok=True,
        tool_name="search_web",
        data={"result_type": "web_search_results", "results": [], "count": 0},
    )

    with patch("src.agents.research_agent.search_web", return_value=empty_web):
        graph = compile_research_subgraph(mock_vs, mock_llm)
        result = graph.invoke(_initial_state("Explain Alpha-GPT alpha mining workflow"))

    retrieved = result.get("retrieved") or []
    assert retrieved[0].paper_id == "alpha-gpt"
