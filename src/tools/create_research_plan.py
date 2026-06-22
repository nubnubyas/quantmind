"""T6: Create an idempotent multi-step research plan."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from src.config.llm_client import LLMClient, create_llm_client
from src.memory.user_memory import UserMemory
from src.tools._helpers import fail, make_plan_id, ok
from src.tools.types import ToolResult

TOOL_NAME = "create_research_plan"


class PlanStep(BaseModel):
    step_id: str
    title: str
    description: str
    status: str = "pending"


class ResearchPlanSchema(BaseModel):
    steps: list[PlanStep] = Field(default_factory=list)


def _find_existing_plan(plans: list[dict], plan_id: str, goal: str) -> dict | None:
    for plan in plans:
        if plan.get("plan_id") == plan_id or plan.get("goal") == goal:
            return plan
    return None


def _store_plan(
    memory: UserMemory,
    user_id: str,
    plan_id: str,
    goal: str,
    current_level: str | None,
    steps: list[dict],
) -> None:
    plan = {
        "plan_id": plan_id,
        "goal": goal,
        "current_level": current_level,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "steps": steps,
    }
    memory.store_research_plan(user_id, plan_id, plan)


def create_research_plan(
    user_id: str,
    goal: str,
    current_level: str | None = None,
    num_steps: int = 5,
    idempotency_key: str | None = None,
    *,
    memory: UserMemory | None = None,
    llm: LLMClient | None = None,
) -> ToolResult:
    mem = memory or UserMemory()
    plan_id = make_plan_id(user_id, goal, idempotency_key)

    existing_plans = mem.get_research_plans(user_id)
    existing = _find_existing_plan(existing_plans, plan_id, goal)
    if existing:
        return ok(
            TOOL_NAME,
            "research_plan",
            plan_id=existing.get("plan_id", plan_id),
            goal=existing.get("goal", goal),
            steps=existing.get("steps", []),
            existing=True,
        )

    client = llm or create_llm_client()
    level_hint = f" at {current_level} level" if current_level else ""

    try:
        parsed = client.chat_structured(
            [
                {
                    "role": "system",
                    "content": (
                        "Decompose a learning goal into a sequenced research plan. "
                        "Each step needs step_id, title, description, and status (pending)."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Goal: {goal}\n"
                        f"Number of steps: {num_steps}\n"
                        f"Learner level{level_hint}"
                    ),
                },
            ],
            ResearchPlanSchema,
        )
    except Exception as exc:  # noqa: BLE001
        return fail(TOOL_NAME, str(exc), "LLM_ERROR", retryable=True)

    now = datetime.now(timezone.utc).isoformat()
    steps = []
    for raw in parsed.get("steps", []):
        step = dict(raw)
        step.setdefault("status", "pending")
        step["updated_at"] = now
        steps.append(step)

    _store_plan(mem, user_id, plan_id, goal, current_level, steps)

    return ok(
        TOOL_NAME,
        "research_plan",
        plan_id=plan_id,
        goal=goal,
        steps=steps,
        existing=False,
    )
