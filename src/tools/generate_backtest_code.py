"""T3: Generate backtesting code via LLM."""

from __future__ import annotations

from src.config.llm_client import LLMClient, create_llm_client
from src.tools._helpers import fail, ok
from src.tools.types import ToolResult

TOOL_NAME = "generate_backtest_code"

_DEPTH_PROMPTS = {
    "backtrader": (
        "Generate complete, runnable Python backtesting code using the backtrader library. "
        "Include imports, a Strategy class, data loading, and cerebro setup. "
        "Output only Python code, no markdown fences."
    ),
    "vectorbt": (
        "Generate complete, runnable Python backtesting code using the vectorbt library. "
        "Include imports, signal logic, and portfolio simulation. "
        "Output only Python code, no markdown fences."
    ),
}


def generate_backtest_code(
    strategy_description: str,
    framework: str = "backtrader",
    asset_class: str | None = None,
    timeframe: str | None = None,
    *,
    llm: LLMClient | None = None,
) -> ToolResult:
    client = llm or create_llm_client()
    system = _DEPTH_PROMPTS.get(framework, _DEPTH_PROMPTS["backtrader"])

    user_parts = [f"Strategy: {strategy_description}", f"Framework: {framework}"]
    if asset_class:
        user_parts.append(f"Asset class: {asset_class}")
    if timeframe:
        user_parts.append(f"Timeframe: {timeframe}")

    try:
        result = client.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": "\n".join(user_parts)},
            ]
        )
    except Exception as exc:  # noqa: BLE001
        return fail(TOOL_NAME, str(exc), "LLM_ERROR", retryable=True)

    code = result.text.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    return ok(
        TOOL_NAME,
        "backtest_code",
        code=code,
        framework=framework,
        model=result.model,
        latency_ms=result.latency_ms,
    )
