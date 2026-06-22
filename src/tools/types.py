"""§C2 Tool shared types (interface contract v1.0)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ToolResult:
    ok: bool
    tool_name: str
    data: dict
    error: str | None = None
    error_code: str | None = None
    retryable: bool = False
