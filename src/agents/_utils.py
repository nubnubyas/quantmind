"""Shared utilities for agent subgraphs."""
from __future__ import annotations

from langchain_core.messages import HumanMessage

from src.agents.state import AgentState


def last_user_text(state: AgentState) -> str:
    """Extract the text content of the last user message from AgentState.

    Handles HumanMessage objects, dict messages, and messages with
    role="user" / type="human" attributes.
    """
    messages = state.get("messages") or []
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content", ""))
        role = getattr(msg, "type", None) or getattr(msg, "role", None)
        if role in ("human", "user"):
            content = getattr(msg, "content", "")
            return content if isinstance(content, str) else str(content)
    return ""
