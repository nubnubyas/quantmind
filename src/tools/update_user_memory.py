"""T7: Update user profile, bookmarks, or plan progress in LangGraph Store."""

from __future__ import annotations

from dataclasses import asdict
from typing import Literal

from src.memory.user_memory import UserMemory
from src.tools._helpers import fail, ok
from src.tools.types import ToolResult

TOOL_NAME = "update_user_memory"

MemoryType = Literal["profile", "bookmark", "plan"]


def update_user_memory(
    user_id: str,
    memory_type: MemoryType,
    content: dict,
    *,
    memory: UserMemory | None = None,
) -> ToolResult:
    mem = memory or UserMemory()

    if memory_type == "profile":
        mem.update_profile(user_id, content)
        profile = mem.get_profile(user_id)
        stored = asdict(profile) if profile else content
        return ok(TOOL_NAME, "memory_update", memory_type=memory_type, stored=stored)

    if memory_type == "bookmark":
        paper_id = content.get("paper_id")
        note = content.get("note", "")
        if not paper_id:
            return fail(TOOL_NAME, "bookmark requires content.paper_id", "VALIDATION_ERROR")
        mem.add_bookmark(user_id, paper_id, note)
        return ok(
            TOOL_NAME,
            "memory_update",
            memory_type=memory_type,
            stored={"paper_id": paper_id, "note": note},
        )

    if memory_type == "plan":
        plan_id = content.get("plan_id")
        steps = content.get("steps")
        if not plan_id or steps is None:
            return fail(
                TOOL_NAME,
                "plan requires content.plan_id and content.steps",
                "VALIDATION_ERROR",
            )
        mem.upsert_plan_progress(user_id, plan_id, steps)
        return ok(
            TOOL_NAME,
            "memory_update",
            memory_type=memory_type,
            stored={"plan_id": plan_id, "steps": steps},
        )

    return fail(TOOL_NAME, f"Unknown memory_type: {memory_type}", "VALIDATION_ERROR")
