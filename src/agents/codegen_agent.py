"""CodeGen Subgraph — parse, confirm (interrupt), generate, sandbox validate."""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from src.agents._utils import last_user_text
from src.agents.state import AgentState, SubgraphOutput
from src.config.llm_client import LLMClient
from src.sandbox.sandbox_runner import SandboxRunner
from src.tools.generate_backtest_code import generate_backtest_code

SAMPLE_DATA_PATH = "data/sample/spy_daily.csv"
SANDBOX_TIMEOUT_S = 30


class StrategySpecSchema(BaseModel):
    name: str = Field(description='Strategy name e.g. "RSI mean reversion"')
    description: str = Field(description="Original user description")
    framework: str = Field(description='"backtrader" or "vectorbt"')
    parameters: dict = Field(description="Strategy parameters e.g. rsi_period, buy_threshold")
    signal_logic: str = Field(description="Signal logic description for code generation")
    asset_class: str | None = Field(default=None, description='"equity" / "futures" / null')
    timeframe: str | None = Field(default=None, description='"daily" / "hourly" / null')


def _failure_detail(state: AgentState) -> str:
    sandbox = state.get("sandbox_result")
    if sandbox and sandbox.error:
        return sandbox.error
    code = state.get("generated_code")
    if not code:
        return "Code generation failed"
    return "Sandbox validation failed"


def compile_codegen_subgraph(
    llm_client: LLMClient,
    sandbox_runner: SandboxRunner | None = None,
    *,
    checkpointer=None,
) -> CompiledStateGraph:
    """Build and compile the CodeGen subgraph on AgentState."""
    runner = sandbox_runner or SandboxRunner()

    def parse_strategy(state: AgentState) -> dict:
        user_text = last_user_text(state)
        raw = llm_client.chat_structured(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a trading strategy parser. Extract the strategy the user described "
                        "into structured fields.\n\n"
                        "FOUR EXAMPLES:\n"
                        "1. User: '双均线20/60日交叉策略'\n"
                        "   → name='Dual MA 20/60 Crossover', framework='backtrader', "
                        "parameters={'ma_fast': 20, 'ma_slow': 60}, "
                        "signal_logic='Buy when MA(20) crosses above MA(60). Sell when MA(20) "
                        "crosses below MA(60). Use daily closing prices.'\n"
                        "2. User: 'ATR-based stop loss with 2x ATR multiplier on daily SPY'\n"
                        "   → name='ATR Stop Loss', framework='backtrader', "
                        "parameters={'atr_period': 14, 'atr_multiplier': 2.0}, "
                        "signal_logic='Calculate ATR(14). Exit when close crosses below "
                        "entry_price - 2*ATR. Trail stop upward.'\n"
                        "3. User: 'Buy and hold SPY with monthly rebalancing using vectorbt'\n"
                        "   → name='Buy and Hold Monthly Rebalance', framework='vectorbt', "
                        "parameters={'rebalance_freq': 'monthly'}, "
                        "signal_logic='Buy 100% SPY at start. Rebalance monthly. No exit signals.'\n"
                        "4. User: 'Walk-forward optimization for momentum with 36-month rolling window'\n"
                        "   → name='Walk-Forward Momentum', framework='backtrader', "
                        "parameters={'lookback': 12, 'rebalance_freq': 'monthly', 'rolling_window': 36}, "
                        "signal_logic='Each month, train on prior 36 months, pick top quintile "
                        "momentum stocks, hold 1 month. Walk forward each month.'\n\n"
                        "RULES:\n"
                        "1. PARSE WHAT THE USER SAID. Map their words exactly: "
                        "'ATR' → ATR, 'vectorbt' → vectorbt, '20/60' → ma_fast=20 ma_slow=60, "
                        "'walk-forward' → rolling window optimization. "
                        "Every concrete strategy request must produce concrete parameters.\n"
                        "2. Only output NEEDS_CLARIFICATION if the user's message contains NO "
                        "recognizable trading strategy (e.g. 'help me get rich').\n"
                        "3. Default framework to 'backtrader' only when none is mentioned.\n\n"
                        "Fields: name, description, framework, parameters, signal_logic, "
                        "asset_class (optional), timeframe (optional)."
                    ),
                },
                {"role": "user", "content": user_text},
            ],
            StrategySpecSchema,
            model=llm_client.strong_model,
        )
        spec = {
            "name": raw.get("name", "Unspecified"),
            "description": user_text,
            "framework": raw.get("framework", "backtrader"),
            "parameters": raw.get("parameters") or {},
            "signal_logic": raw.get("signal_logic", ""),
            "asset_class": raw.get("asset_class"),
            "timeframe": raw.get("timeframe"),
        }
        return {"strategy_spec": spec}

    def confirm_with_user(state: AgentState) -> dict:
        spec = state.get("strategy_spec")
        confirmed_spec = interrupt(spec)
        return {"strategy_spec": confirmed_spec}

    def generate_code(state: AgentState) -> dict:
        spec = state.get("strategy_spec") or {}
        result = generate_backtest_code(
            strategy_description=spec.get("signal_logic", ""),
            framework=spec.get("framework", "backtrader"),
            asset_class=spec.get("asset_class"),
            timeframe=spec.get("timeframe"),
            strategy_name=spec.get("name", ""),
            llm=llm_client,
        )
        code = result.data.get("code") if result.ok else None
        return {"generated_code": code}

    def execute_sandbox(state: AgentState) -> dict:
        code = state.get("generated_code") or ""
        sandbox_result = runner.run(
            code=code,
            sample_data_path=SAMPLE_DATA_PATH,
            timeout_s=SANDBOX_TIMEOUT_S,
        )
        return {"sandbox_result": sandbox_result}

    def route_after_sandbox(
        state: AgentState,
    ) -> Literal["format_success", "handle_failure"]:
        sandbox = state.get("sandbox_result")
        if sandbox and sandbox.success:
            return "format_success"
        return "handle_failure"

    def format_success(state: AgentState) -> dict:
        spec = state.get("strategy_spec") or {}
        code = state.get("generated_code") or ""
        sandbox = state.get("sandbox_result")
        stdout = sandbox.stdout if sandbox else ""
        name = spec.get("name", "Strategy")
        final = (
            f"Strategy: {name}\n\n"
            f"Generated code:\n```python\n{code}\n```\n\n"
            f"Sandbox output:\n{stdout}"
        )
        subgraph_output: SubgraphOutput = {
            "mode": "codegen",
            "result": final,
            "citations": [],
            "error": None,
        }
        return {
            "final_response": final,
            "subgraph_outputs": {"codegen": subgraph_output},
        }

    def handle_failure(state: AgentState) -> dict:
        current = state.get("retry_counts", {}).get("codegen", 0)
        new_count = current + 1
        updates: dict = {"retry_counts": {"codegen": new_count}}
        if new_count > 1:
            detail = _failure_detail(state)
            final = f"Code generation failed after retry.\n\nError: {detail}"
            subgraph_output: SubgraphOutput = {
                "mode": "codegen",
                "result": final,
                "citations": [],
                "error": "sandbox_failed_after_retry",
            }
            updates["final_response"] = final
            updates["subgraph_outputs"] = {"codegen": subgraph_output}
        return updates

    def route_after_failure(
        state: AgentState,
    ) -> Literal["generate_code", "__end__"]:
        retries = state.get("retry_counts", {}).get("codegen", 0)
        if retries <= 1:
            return "generate_code"
        return END

    builder = StateGraph(AgentState)
    builder.add_node("parse_strategy", parse_strategy)
    builder.add_node("confirm_with_user", confirm_with_user)
    builder.add_node("generate_code", generate_code)
    builder.add_node("execute_sandbox", execute_sandbox)
    builder.add_node("format_success", format_success)
    builder.add_node("handle_failure", handle_failure)

    builder.add_edge(START, "parse_strategy")
    builder.add_edge("parse_strategy", "confirm_with_user")
    builder.add_edge("confirm_with_user", "generate_code")
    builder.add_edge("generate_code", "execute_sandbox")
    builder.add_conditional_edges("execute_sandbox", route_after_sandbox)
    builder.add_edge("format_success", END)
    builder.add_conditional_edges("handle_failure", route_after_failure)

    return builder.compile(checkpointer=checkpointer)
