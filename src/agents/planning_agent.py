"""Planning Subgraph — parse goal, generate research plan, format output."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from src.agents._utils import last_user_text
from src.agents.state import AgentState, SubgraphOutput
from src.config.llm_client import LLMClient
from src.memory.user_memory import UserMemory
from src.tools.create_research_plan import create_research_plan


class GoalParseSchema(BaseModel):
    goal_text: str = Field(description="The user's learning or research goal")
    current_level: str | None = Field(default=None, description="beginner/intermediate/advanced")
    num_steps: int = Field(default=5, description="Number of plan steps to generate")


def _format_plan_steps(steps: list[dict], existing: bool) -> str:
    header = "研究学习计划"
    if existing:
        header += "（已有计划）"
    lines = [header, ""]
    for i, step in enumerate(steps, 1):
        lines.append(
            f"{i}. [{step.get('step_id', f's{i}')}] {step.get('title', '')} "
            f"({step.get('status', 'pending')})"
        )
        desc = step.get("description", "")
        if desc:
            lines.append(f"   {desc}")
    return "\n".join(lines)


def compile_planning_subgraph(
    llm_client: LLMClient,
    memory: UserMemory | None = None,
    *,
    checkpointer=None,
) -> CompiledStateGraph:
    """Build and compile the Planning subgraph on AgentState."""
    mem = memory or UserMemory()
    _ctx: dict = {}

    def parse_goal(state: AgentState) -> dict:
        _ctx.clear()
        user_text = last_user_text(state)
        try:
            raw = llm_client.chat_structured(
                [
                    {
                        "role": "system",
                        "content": (
                            "Extract a structured research learning goal from the user message. "
                            "Return goal_text, optional current_level, and num_steps (default 5)."
                        ),
                    },
                    {"role": "user", "content": user_text},
                ],
                GoalParseSchema,
            )
            _ctx["goal"] = raw
        except Exception as exc:  # noqa: BLE001
            _ctx["error"] = str(exc)
        return {}

    def generate_plan(state: AgentState) -> dict:
        if _ctx.get("error"):
            return {}
        goal = _ctx.get("goal") or {}
        result = create_research_plan(
            user_id=state.get("user_id", "eval_user"),
            goal=goal.get("goal_text", ""),
            current_level=goal.get("current_level"),
            num_steps=goal.get("num_steps", 5),
            memory=mem,
            llm=llm_client,
        )
        if not result.ok:
            _ctx["error"] = result.error or "Failed to create research plan"
            return {}
        _ctx["plan"] = result.data
        return {}

    def format_plan(state: AgentState) -> dict:
        error = _ctx.get("error")
        if error:
            final = f"无法生成研究计划：{error}"
            subgraph_output: SubgraphOutput = {
                "mode": "planning",
                "result": final,
                "citations": [],
                "error": error,
            }
            return {
                "final_response": final,
                "subgraph_outputs": {"planning": subgraph_output},
            }

        plan = _ctx.get("plan") or {}
        steps = plan.get("steps") or []
        existing = bool(plan.get("existing"))
        final = _format_plan_steps(steps, existing)
        subgraph_output = {
            "mode": "planning",
            "result": final,
            "citations": [],
            "error": None,
        }
        return {
            "final_response": final,
            "subgraph_outputs": {"planning": subgraph_output},
        }

    builder = StateGraph(AgentState)
    builder.add_node("parse_goal", parse_goal)
    builder.add_node("generate_plan", generate_plan)
    builder.add_node("format_plan", format_plan)

    builder.add_edge(START, "parse_goal")
    builder.add_edge("parse_goal", "generate_plan")
    builder.add_edge("generate_plan", "format_plan")
    builder.add_edge("format_plan", END)

    return builder.compile(checkpointer=checkpointer)
