from datetime import datetime, timezone

import pytest
from langgraph.store.memory import InMemoryStore

from src.memory import Profile, UserMemory


@pytest.fixture
def store():
    return InMemoryStore()


@pytest.fixture
def memory(store):
    return UserMemory(store=store)


def test_profile_persistence_across_instances(store):
    mem1 = UserMemory(store=store)
    mem1.update_profile("user_alice", {
        "research_interests": ["momentum", "stat_arb"],
        "skill_level": "intermediate",
    })

    mem2 = UserMemory(store=store)
    profile = mem2.get_profile("user_alice")

    assert profile is not None
    assert profile.research_interests == ["momentum", "stat_arb"]
    assert profile.skill_level == "intermediate"


def test_get_profile_returns_none_for_new_user(memory):
    assert memory.get_profile("new_user") is None


def test_bookmark_lifecycle(memory):
    memory.add_bookmark("user1", "paper-123", "Important momentum paper")

    bookmarks = memory.list_bookmarks("user1")
    assert len(bookmarks) == 1
    assert bookmarks[0]["paper_id"] == "paper-123"
    assert bookmarks[0]["note"] == "Important momentum paper"

    memory.delete_bookmark("user1", "paper-123")
    assert memory.list_bookmarks("user1") == []


def test_research_plan_upsert_and_remove(memory):
    steps = [
        {
            "step_id": "s1",
            "status": "completed",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
        {
            "step_id": "s2",
            "status": "in_progress",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    ]
    memory.upsert_plan_progress("user1", "plan-abc", steps)

    plans = memory.get_research_plans("user1")
    assert len(plans) == 1
    assert plans[0]["plan_id"] == "plan-abc"
    assert len(plans[0]["steps"]) == 2

    memory.remove_research_plan("user1", "plan-abc")
    assert memory.get_research_plans("user1") == []


def test_profile_is_dataclass():
    profile = Profile()
    assert profile.research_interests == []
    assert profile.target_roles == []
    assert profile.skill_level is None
    assert profile.preferences == {}


def test_get_profile_no_langgraph_context_returns_none():
    """get_profile() should return None (not crash) when no LangGraph runtime."""
    mem = UserMemory()  # no store injected
    result = mem.get_profile("test_user_123")
    assert result is None


def test_get_research_plans_no_langgraph_context_returns_empty():
    """get_research_plans() should return [] (not crash) when no LangGraph runtime."""
    mem = UserMemory()  # no store injected
    result = mem.get_research_plans("test_user_123")
    assert result == []


def test_list_bookmarks_no_langgraph_context_returns_empty():
    """list_bookmarks() should return [] (not crash) when no LangGraph runtime."""
    mem = UserMemory()  # no store injected
    result = mem.list_bookmarks("test_user_123")
    assert result == []


def test_update_profile_no_langgraph_context_does_not_crash():
    """update_profile() should not crash when no LangGraph runtime."""
    mem = UserMemory()
    # Should not raise
    mem.update_profile("test_user_123", {"skill_level": "beginner"})


def test_add_and_get_bookmark_no_langgraph_context():
    """Bookmark CRUD should work with InMemoryStore fallback."""
    mem = UserMemory()
    mem.add_bookmark("test_user_123", "paper_001", "Good paper")
    bookmarks = mem.list_bookmarks("test_user_123")
    assert len(bookmarks) == 1
    assert bookmarks[0]["paper_id"] == "paper_001"
