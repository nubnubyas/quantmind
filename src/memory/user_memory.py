from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from langgraph.config import get_store


@dataclass
class Profile:
    research_interests: list[str] = field(default_factory=list)
    target_roles: list[str] = field(default_factory=list)
    skill_level: str | None = None
    preferences: dict = field(default_factory=dict)


class UserMemory:
    PROFILE_KEY = "profile"

    def __init__(self, store: Any | None = None) -> None:
        self._store = store

    def _resolve_store(self):
        if self._store is not None:
            return self._store
        try:
            store = get_store()
            if store is not None:
                return store
        except Exception:
            pass
        from langgraph.store.memory import InMemoryStore

        self._store = InMemoryStore()
        return self._store

    @staticmethod
    def _profile_ns(user_id: str) -> tuple[str, ...]:
        return ("user", user_id, "profile")

    @staticmethod
    def _bookmarks_ns(user_id: str) -> tuple[str, ...]:
        return ("user", user_id, "bookmarks")

    @staticmethod
    def _plans_ns(user_id: str) -> tuple[str, ...]:
        return ("user", user_id, "plans")

    def get_profile(self, user_id: str) -> Profile | None:
        store = self._resolve_store()
        if store is None:
            return None
        item = store.get(self._profile_ns(user_id), self.PROFILE_KEY)
        if item is None:
            return None
        data = item.value
        return Profile(
            research_interests=data.get("research_interests", []),
            target_roles=data.get("target_roles", []),
            skill_level=data.get("skill_level"),
            preferences=data.get("preferences", {}),
        )

    def update_profile(self, user_id: str, patch: dict) -> None:
        store = self._resolve_store()
        current = self.get_profile(user_id) or Profile()
        data = asdict(current)
        for key, value in patch.items():
            if key in data:
                data[key] = value
        store.put(self._profile_ns(user_id), self.PROFILE_KEY, data)

    def add_bookmark(self, user_id: str, paper_id: str, note: str) -> None:
        store = self._resolve_store()
        store.put(
            self._bookmarks_ns(user_id),
            paper_id,
            {
                "paper_id": paper_id,
                "note": note,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def list_bookmarks(self, user_id: str) -> list[dict]:
        store = self._resolve_store()
        if store is None:
            return []
        try:
            items = store.search(self._bookmarks_ns(user_id))
            return [item.value for item in items]
        except Exception:
            return []

    def delete_bookmark(self, user_id: str, paper_id: str) -> None:
        store = self._resolve_store()
        store.delete(self._bookmarks_ns(user_id), paper_id)

    def get_research_plans(self, user_id: str) -> list[dict]:
        store = self._resolve_store()
        if store is None:
            return []
        try:
            items = store.search(self._plans_ns(user_id))
            return [item.value for item in items]
        except Exception:
            return []

    def upsert_plan_progress(self, user_id: str, plan_id: str, steps: list[dict]) -> None:
        store = self._resolve_store()
        ns = self._plans_ns(user_id)
        existing = store.get(ns, plan_id)
        plan = dict(existing.value) if existing else {"plan_id": plan_id}
        plan["steps"] = steps
        store.put(ns, plan_id, plan)

    def store_research_plan(self, user_id: str, plan_id: str, plan: dict) -> None:
        """Store a complete research plan dict under (user, user_id, plans)."""
        store = self._resolve_store()
        ns = self._plans_ns(user_id)
        store.put(ns, plan_id, plan)

    def remove_research_plan(self, user_id: str, plan_id: str) -> None:
        store = self._resolve_store()
        store.delete(self._plans_ns(user_id), plan_id)
