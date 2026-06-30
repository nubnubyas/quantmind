"""T3: Generate backtesting code via LLM."""

from __future__ import annotations

from src.config.llm_client import LLMClient, create_llm_client
from src.tools._helpers import fail, ok
from src.tools.types import ToolResult

TOOL_NAME = "generate_backtest_code"

_DATA_SOURCE_RULES = (
    "\n\nIMPORTANT RULES:\n"
    "1. ALWAYS load data from 'data/sample/spy_daily.csv' using pandas.read_csv. "
    "Never use Yahoo Finance, Alpha Vantage, or any other online data source.\n"
    "2. The CSV has columns: Date,Open,High,Low,Close,Volume. "
    "Parse Date column with pd.to_datetime and set as index.\n"
    "3. For backtrader: use bt.feeds.PandasData(dataname=dataframe). "
    "For vectorbt: use the Close price series directly.\n"
    "4. Include a cerebro.run() call and print final portfolio value.\n"
    "5. Code must run in a restricted sandbox with no internet access.\n"
    "6. Output ONLY the Python code block, no explanation text."
)

_DEPTH_PROMPTS = {
    "backtrader": (
        "Generate complete, runnable Python backtesting code using the backtrader library. "
        "Include imports, a Strategy class, data loading, and cerebro setup.\n\n"
        "STRATEGY GUARD: Implement ONLY the strategy specified below. "
        "- If the strategy is ATR stop loss → implement ATR-based exits, NOT MA crossover. "
        "- If the strategy is walk-forward → implement rolling window optimization, NOT simple MA crossover. "
        "- If the strategy is buy and hold → implement buy at start and hold, NOT active trading. "
        "- If the strategy is MA crossover → implement the specific MA periods requested. "
        "Only implement MA crossover when MA crossover is explicitly requested."
        + _DATA_SOURCE_RULES
    ),
    "vectorbt": (
        "Generate complete, runnable Python backtesting code using the vectorbt library. "
        "Include imports, signal logic, and portfolio simulation.\n\n"
        "STRATEGY GUARD: Implement ONLY the strategy specified below. "
        "- If the strategy is buy and hold → implement buy and hold, NOT MA crossover. "
        "- If the strategy is monthly rebalance → implement rebalancing, NOT active trading. "
        "Only implement MA crossover when MA crossover is explicitly requested."
        + _DATA_SOURCE_RULES
    ),
}


def generate_backtest_code(
    strategy_description: str,
    framework: str = "backtrader",
    asset_class: str | None = None,
    timeframe: str | None = None,
    *,
    strategy_name: str = "",
    llm: LLMClient | None = None,
) -> ToolResult:
    client = llm or create_llm_client()
    system = _DEPTH_PROMPTS.get(framework, _DEPTH_PROMPTS["backtrader"])

    user_parts = [f"Strategy: {strategy_description}", f"Framework: {framework}"]
    if strategy_name:
        user_parts.insert(0, f"Strategy Name: {strategy_name}")
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
