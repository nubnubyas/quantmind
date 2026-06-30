"""Aggregate benchmark item results into a metrics report."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import mean

from src.eval.evaluators import ItemResult


def _item_passed(item: ItemResult) -> bool:
    return bool(item.evals) and all(e.passed for e in item.evals) and not item.error


def _group_rate(items: list[ItemResult], key_fn) -> dict[str, dict]:
    groups: dict[str, list[ItemResult]] = {}
    for item in items:
        key = key_fn(item)
        groups.setdefault(key, []).append(item)

    out: dict[str, dict] = {}
    for key, group in sorted(groups.items()):
        passed = sum(1 for item in group if _item_passed(item))
        total = len(group)
        out[key] = {
            "total": total,
            "passed": passed,
            "rate": passed / total if total else 0.0,
        }
    return out


def _evaluator_stats(results: list[ItemResult]) -> dict[str, dict]:
    stats: dict[str, dict[str, int]] = {}
    for item in results:
        for ev in item.evals:
            bucket = stats.setdefault(ev.evaluator, {"total": 0, "passed": 0})
            bucket["total"] += 1
            if ev.passed:
                bucket["passed"] += 1

    return {
        name: {
            "total": data["total"],
            "passed": data["passed"],
            "rate": data["passed"] / data["total"] if data["total"] else 0.0,
        }
        for name, data in sorted(stats.items())
    }


def _percentile(values: list[int], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return float(ordered[idx])


@dataclass
class MetricsReport:
    total: int
    passed: int
    overall_pass_rate: float
    by_scenario: dict[str, dict] = field(default_factory=dict)
    by_difficulty: dict[str, dict] = field(default_factory=dict)
    by_evaluator: dict[str, dict] = field(default_factory=dict)
    code_exec_rate: float = 0.0
    citation_rate: float = 0.0
    cross_domain_route_accuracy: float = 0.0
    graceful_decline_rate: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0

    @classmethod
    def from_results(cls, results: list[ItemResult]) -> MetricsReport:
        total = len(results)
        passed = sum(1 for item in results if _item_passed(item))

        s2_code = [
            ev
            for item in results
            if item.scenario == "S2"
            for ev in item.evals
            if ev.evaluator == "code"
        ]
        code_exec_rate = (
            sum(1 for ev in s2_code if ev.passed) / len(s2_code) if s2_code else 0.0
        )

        citation_evals = [ev for item in results for ev in item.evals if ev.evaluator == "citation"]
        citation_rate = (
            sum(1 for ev in citation_evals if ev.passed) / len(citation_evals)
            if citation_evals
            else 0.0
        )

        s9_items = [item for item in results if item.scenario == "S9"]
        if s9_items:
            routed = sum(
                1
                for item in s9_items
                if item.route and item.route.get("fanout_modes")
            )
            cross_domain_route_accuracy = routed / len(s9_items)
        else:
            cross_domain_route_accuracy = 0.0

        declined = sum(1 for item in results if item.error or item.response is None)
        graceful_decline_rate = declined / total if total else 0.0

        latencies = [item.latency_ms for item in results]
        avg_latency_ms = mean(latencies) if latencies else 0.0
        p95_latency_ms = _percentile(latencies, 95.0)

        return cls(
            total=total,
            passed=passed,
            overall_pass_rate=passed / total if total else 0.0,
            by_scenario=_group_rate(results, lambda item: item.scenario),
            by_difficulty=_group_rate(results, lambda item: item.difficulty),
            by_evaluator=_evaluator_stats(results),
            code_exec_rate=code_exec_rate,
            citation_rate=citation_rate,
            cross_domain_route_accuracy=cross_domain_route_accuracy,
            graceful_decline_rate=graceful_decline_rate,
            avg_latency_ms=avg_latency_ms,
            p95_latency_ms=p95_latency_ms,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            "# QuantMind Benchmark Evaluation Report",
            "",
            "## Summary",
            f"- Total items: {self.total}",
            f"- Pass rate: {self.overall_pass_rate:.2f}",
            f"- Avg latency: {self.avg_latency_ms:.0f}ms / P95: {self.p95_latency_ms:.0f}ms",
            "",
            "## By Scenario",
            "| Scenario | Total | Passed | Rate |",
            "|----------|-------|--------|------|",
        ]
        for scenario, data in self.by_scenario.items():
            lines.append(
                f"| {scenario} | {data['total']} | {data['passed']} | {data['rate']:.2f} |"
            )

        lines.extend(
            [
                "",
                "## By Difficulty",
                "| Difficulty | Total | Passed | Rate |",
                "|------------|-------|--------|------|",
            ]
        )
        for difficulty, data in self.by_difficulty.items():
            lines.append(
                f"| {difficulty} | {data['total']} | {data['passed']} | {data['rate']:.2f} |"
            )

        lines.extend(
            [
                "",
                "## By Evaluator",
                "| Evaluator | Total | Passed | Rate |",
                "|-----------|-------|--------|------|",
            ]
        )
        for evaluator, data in self.by_evaluator.items():
            lines.append(
                f"| {evaluator} | {data['total']} | {data['passed']} | {data['rate']:.2f} |"
            )

        lines.extend(
            [
                "",
                "## Special Metrics",
                f"- Code execution rate (S2): {self.code_exec_rate:.2f}",
                f"- Citation presence rate: {self.citation_rate:.2f}",
                f"- Cross-domain route accuracy (S9): {self.cross_domain_route_accuracy:.2f}",
                f"- Graceful decline rate: {self.graceful_decline_rate:.2f}",
                f"- Avg latency: {self.avg_latency_ms:.0f}ms / P95: {self.p95_latency_ms:.0f}ms",
            ]
        )
        return "\n".join(lines) + "\n"
