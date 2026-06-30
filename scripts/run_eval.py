#!/usr/bin/env python3
"""Run QuantMind benchmark evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_PROJECT", "quantmind")
os.environ.setdefault("LANGSMITH_ENDPOINT", "https://apac.api.smith.langchain.com")

from langgraph.checkpoint.sqlite import SqliteSaver

from src.agents.codegen_agent import compile_codegen_subgraph
from src.agents.interview_agent import compile_interview_subgraph
from src.agents.planning_agent import compile_planning_subgraph
from src.agents.research_agent import compile_research_subgraph
from src.agents.supervisor import compile_parent_graph
from src.config.llm_client import LLMClient
from src.eval.evaluators import CitationEvaluator, CodeEvaluator, JudgeEvaluator
from src.eval.report import MetricsReport
from src.eval.runner import BenchmarkRunner
from src.memory import UserMemory
from src.sandbox import SandboxRunner
from src.vector_store.qdrant_client_wrapper import VectorStore


def load_benchmark(path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _item_to_dict(item) -> dict:
    data = asdict(item)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="QuantMind Benchmark Runner")
    parser.add_argument("--limit", type=int, default=None, help="Limit items to run")
    parser.add_argument("--scenario", type=str, default=None, help="Filter by scenario (S1-S10)")
    parser.add_argument("--output", type=str, default="eval_report.md", help="Output report path")
    parser.add_argument(
        "--json-output",
        type=str,
        default="eval_results.json",
        help="Raw results JSON",
    )
    args = parser.parse_args()

    bench_path = ROOT / "data" / "benchmark" / "benchmark_v1.jsonl"
    items = load_benchmark(str(bench_path))
    if args.scenario:
        items = [i for i in items if i["scenario"] == args.scenario]
    if args.limit:
        items = items[: args.limit]

    print(f"Running evaluation on {len(items)} items...")

    with SqliteSaver.from_conn_string("eval_quantmind.sqlite") as checkpointer:
        vector_store = VectorStore()
        llm_client = LLMClient()
        sandbox_runner = SandboxRunner()
        memory = UserMemory()

        research = compile_research_subgraph(vector_store, llm_client, checkpointer=checkpointer)
        codegen = compile_codegen_subgraph(llm_client, sandbox_runner, checkpointer=checkpointer)
        planning = compile_planning_subgraph(llm_client, memory, checkpointer=checkpointer)
        interview = compile_interview_subgraph(llm_client, memory, checkpointer=checkpointer)

        graph = compile_parent_graph(
            research,
            codegen,
            planning,
            interview,
            llm_client,
            checkpointer=checkpointer,
        )

        evaluators = [
            JudgeEvaluator(llm_client),
            CodeEvaluator(sandbox_runner),
            CitationEvaluator(),
        ]

        runner = BenchmarkRunner(graph, evaluators, checkpointer=checkpointer)
        results = runner.run_all(items)

        report = MetricsReport.from_results(results)
        markdown = report.to_markdown()
        print(markdown)

        Path(args.output).write_text(markdown, encoding="utf-8")
        Path(args.json_output).write_text(
            json.dumps([_item_to_dict(r) for r in results], ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\nReport saved to {args.output}")
        print(f"Raw results saved to {args.json_output}")


if __name__ == "__main__":
    main()
