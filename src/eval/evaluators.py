"""E2 Phase 4: benchmark evaluators (judge, code, citation)."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from src.config.llm_client import LLMClient
from src.sandbox import SandboxResult, SandboxRunner

_JUDGE_SYSTEM_PROMPT = """\
You are an evaluator for a quant finance AI assistant. Score the agent's response against the expected behavior description.

Scoring rubric:
- answers_question (0.0-1.0): Does the response address the user's query?
- matches_expected (0.0-1.0): Does the response cover the key points in expected_behavior?
- overall_score (0.0-1.0): Holistic quality score.

Expected behavior is a description of what a good answer should contain, NOT a reference answer.\
"""

_CODE_BLOCK_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
_CITATION_MARKER_RE = re.compile(r"\[\d+\]")
_PASS_THRESHOLD = 0.7


class JudgeSchema(BaseModel):
    answers_question: float = Field(ge=0.0, le=1.0)
    matches_expected: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(description="Brief explanation of the score")


@dataclass
class EvalResult:
    """Single evaluator's verdict for one benchmark item."""

    evaluator: str
    passed: bool
    score: float
    reason: str
    details: dict = field(default_factory=dict)


@dataclass
class ItemResult:
    """All evaluator results for one benchmark item."""

    bench_id: str
    scenario: str
    difficulty: str
    query: str
    response: str | None
    latency_ms: int
    route: dict | None
    evals: list[EvalResult]
    error: str | None = None


def _extract_python_blocks(response: str) -> list[str]:
    return _CODE_BLOCK_RE.findall(response or "")


class JudgeEvaluator:
    """LLM-as-judge: compare response against expected_behavior."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    def evaluate(
        self,
        query: str,
        expected_behavior: str,
        response: str,
        criteria: dict,
    ) -> EvalResult:
        del criteria  # reserved for future criteria-aware judging
        if not response or not response.strip():
            return EvalResult(
                evaluator="judge",
                passed=False,
                score=0.0,
                reason="Empty response from agent",
            )

        try:
            raw = self._llm.chat_structured(
                [
                    {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Query:\n{query}\n\n"
                            f"Expected behavior:\n{expected_behavior}\n\n"
                            f"Agent response:\n{response}"
                        ),
                    },
                ],
                JudgeSchema,
            )
            score = float(raw.get("overall_score", 0.0))
            passed = score >= _PASS_THRESHOLD
            return EvalResult(
                evaluator="judge",
                passed=passed,
                score=score,
                reason=str(raw.get("reason", "")),
                details={
                    "answers_question": raw.get("answers_question"),
                    "matches_expected": raw.get("matches_expected"),
                    "overall_score": score,
                },
            )
        except Exception as exc:
            return EvalResult(
                evaluator="judge",
                passed=False,
                score=0.0,
                reason=str(exc),
            )


class CodeEvaluator:
    """Validate generated code via ast.parse() and optional sandbox execution."""

    def __init__(self, sandbox_runner: SandboxRunner | None = None) -> None:
        self._sandbox_runner = sandbox_runner

    def evaluate(
        self,
        response: str,
        sandbox_result: SandboxResult | None,
    ) -> EvalResult:
        blocks = _extract_python_blocks(response or "")
        if not blocks and not sandbox_result:
            return EvalResult(
                evaluator="code",
                passed=False,
                score=0.0,
                reason="No python code block found in response",
            )

        syntax_ok = False
        if blocks:
            syntax_ok = all(self._syntax_ok(block) for block in blocks)
        elif sandbox_result and sandbox_result.success:
            syntax_ok = False

        if sandbox_result and sandbox_result.success:
            score = 1.0 if syntax_ok else 0.8
            return EvalResult(
                evaluator="code",
                passed=True,
                score=score,
                reason="Sandbox execution succeeded",
                details={"phase": sandbox_result.phase, "syntax_ok": syntax_ok},
            )

        if syntax_ok:
            return EvalResult(
                evaluator="code",
                passed=True,
                score=0.5,
                reason="Syntax check passed (no sandbox result)",
                details={"phase": "syntax"},
            )

        return EvalResult(
            evaluator="code",
            passed=False,
            score=0.0,
            reason="Syntax error in extracted python code",
            details={"blocks_found": len(blocks)},
        )

    @staticmethod
    def _syntax_ok(code: str) -> bool:
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False


class CitationEvaluator:
    """Check whether citations are present and properly structured when required."""

    def evaluate(
        self,
        citations: list[dict] | None,
        criteria: dict,
        response: str,
    ) -> EvalResult:
        del criteria
        has_citations = bool(citations)
        has_markers = bool(_CITATION_MARKER_RE.search(response or ""))
        passed = has_citations and has_markers
        score = 1.0 if passed else 0.0
        if passed:
            reason = "Citations list and inline markers present"
        elif not has_citations and not has_markers:
            reason = "Missing citations list and inline citation markers"
        elif not has_citations:
            reason = "Citations list is empty"
        else:
            reason = "Response missing inline citation markers like [1]"

        return EvalResult(
            evaluator="citation",
            passed=passed,
            score=score,
            reason=reason,
            details={
                "citations_count": len(citations or []),
                "has_markers": has_markers,
            },
        )
