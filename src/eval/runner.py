"""Benchmark runner: invoke agent per item and run evaluators."""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from src.eval.evaluators import (
    CitationEvaluator,
    CodeEvaluator,
    EvalResult,
    ItemResult,
    JudgeEvaluator,
)


class BenchmarkRunner:
    def __init__(
        self,
        graph: CompiledStateGraph,
        evaluators: list[Any],
        *,
        checkpointer=None,
    ) -> None:
        self.graph = graph
        self.evaluators = evaluators
        self.checkpointer = checkpointer

    def run_item(self, item: dict) -> ItemResult:
        start = time.perf_counter()
        thread_id = f"eval-{item['id']}"
        criteria = item.get("eval_criteria", {})

        try:
            initial_state = {
                "messages": [HumanMessage(content=item["query"])],
                "user_id": "eval_user",
            }
            config = {"configurable": {"thread_id": thread_id}}
            result = self.graph.invoke(initial_state, config=config)

            if result.get("__interrupt__"):
                interrupt_value = result["__interrupt__"][0]
                result = self.graph.invoke(
                    Command(resume={interrupt_value.id: interrupt_value.value}),
                    config=config,
                )

            latency_ms = int((time.perf_counter() - start) * 1000)
            response = result.get("final_response") or ""
            citations = result.get("citations") or []
            sandbox_result = result.get("sandbox_result")
            route = result.get("route")

            error = None
            if not response.strip():
                response = None
                error = "No response from agent"

            evals = self._run_evaluators(
                item=item,
                criteria=criteria,
                response=response or "",
                citations=citations,
                sandbox_result=sandbox_result,
            )

            return ItemResult(
                bench_id=item["id"],
                scenario=item["scenario"],
                difficulty=item["difficulty"],
                query=item["query"],
                response=response,
                latency_ms=latency_ms,
                route=dict(route) if route else None,
                evals=evals,
                error=error,
            )
        except Exception as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            return ItemResult(
                bench_id=item["id"],
                scenario=item["scenario"],
                difficulty=item["difficulty"],
                query=item["query"],
                response=None,
                latency_ms=latency_ms,
                route=None,
                evals=[],
                error=str(exc),
            )

    def _run_evaluators(
        self,
        *,
        item: dict,
        criteria: dict,
        response: str,
        citations: list,
        sandbox_result,
    ) -> list[EvalResult]:
        evals: list[EvalResult] = []
        for evaluator in self.evaluators:
            try:
                if isinstance(evaluator, JudgeEvaluator):
                    evals.append(
                        evaluator.evaluate(
                            item["query"],
                            item["expected_behavior"],
                            response,
                            criteria,
                        )
                    )
                elif isinstance(evaluator, CodeEvaluator):
                    if criteria.get("requires_code"):
                        evals.append(evaluator.evaluate(response, sandbox_result))
                elif isinstance(evaluator, CitationEvaluator):
                    if criteria.get("cites_sources"):
                        evals.append(
                            evaluator.evaluate(citations, criteria, response)
                        )
                else:
                    evals.append(evaluator.evaluate(item, {"final_response": response}))
            except Exception as exc:
                name = getattr(evaluator, "__class__", type(evaluator)).__name__.lower()
                if "judge" in name:
                    evaluator_name = "judge"
                elif "code" in name:
                    evaluator_name = "code"
                elif "citation" in name:
                    evaluator_name = "citation"
                else:
                    evaluator_name = name
                evals.append(
                    EvalResult(
                        evaluator=evaluator_name,
                        passed=False,
                        score=0.0,
                        reason=str(exc),
                    )
                )
        return evals

    def run_all(self, items: list[dict], *, limit: int | None = None) -> list[ItemResult]:
        subset = items[:limit] if limit else items
        return [self.run_item(item) for item in subset]
