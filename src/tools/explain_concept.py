"""T4: Explain a quantitative finance concept with RAG + LLM."""

from __future__ import annotations

from src.config.llm_client import LLMClient, create_llm_client
from src.tools._helpers import fail, map_vector_store_error, ok, search_result_to_dict
from src.tools.types import ToolResult
from src.vector_store.qdrant_client_wrapper import VectorStore
from src.vector_store.types import RetrievalSpec

TOOL_NAME = "explain_concept"

_DEPTH_INSTRUCTIONS = {
    "intuition": "Explain with intuitive analogies, minimal math.",
    "technical": "Explain with technical precision, include key formulas where relevant.",
    "mathematical": "Provide a rigorous mathematical treatment with definitions and notation.",
}


def explain_concept(
    concept: str,
    depth: str = "technical",
    *,
    vector_store: VectorStore | None = None,
    llm: LLMClient | None = None,
) -> ToolResult:
    store = vector_store or VectorStore()
    client = llm or create_llm_client()

    try:
        results = store.search(concept, "concepts", RetrievalSpec(top_k=3))
    except Exception as exc:  # noqa: BLE001
        return map_vector_store_error(TOOL_NAME, exc)

    context = "\n\n".join(r.text for r in results) if results else "No retrieved context."
    depth_instruction = _DEPTH_INSTRUCTIONS.get(depth, _DEPTH_INSTRUCTIONS["technical"])

    try:
        llm_result = client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a quantitative finance educator. "
                        f"{depth_instruction} "
                        "Ground your explanation in the provided context when available."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Concept: {concept}\n\nContext:\n{context}",
                },
            ]
        )
    except Exception as exc:  # noqa: BLE001
        return fail(TOOL_NAME, str(exc), "LLM_ERROR", retryable=True)

    return ok(
        TOOL_NAME,
        "concept_explanation",
        concept=concept,
        depth=depth,
        explanation=llm_result.text,
        sources=[search_result_to_dict(r) for r in results],
        model=llm_result.model,
        latency_ms=llm_result.latency_ms,
    )
