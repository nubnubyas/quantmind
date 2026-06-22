"""Internal helpers for tool implementations (not exported)."""

from __future__ import annotations

import uuid
from dataclasses import asdict

from src.db.exceptions import InvalidTransition, NotFoundError, StaleStatusError
from src.tools.types import ToolResult
from src.vector_store.types import (
    BackendTimeout,
    CollectionNotFound,
    SearchResult,
    VectorStoreUnavailable,
)


def ok(tool_name: str, result_type: str, **fields: object) -> ToolResult:
    return ToolResult(
        ok=True,
        tool_name=tool_name,
        data={"result_type": result_type, **fields},
    )


def fail(
    tool_name: str,
    error: str,
    error_code: str,
    *,
    retryable: bool = False,
) -> ToolResult:
    return ToolResult(
        ok=False,
        tool_name=tool_name,
        data={"result_type": "error"},
        error=error,
        error_code=error_code,
        retryable=retryable,
    )


def search_result_to_dict(result: SearchResult) -> dict:
    return asdict(result)


def map_vector_store_error(tool_name: str, exc: Exception) -> ToolResult:
    if isinstance(exc, BackendTimeout):
        return fail(tool_name, str(exc), "TIMEOUT", retryable=True)
    if isinstance(exc, VectorStoreUnavailable):
        return fail(tool_name, str(exc), "SERVICE_UNAVAILABLE", retryable=True)
    if isinstance(exc, CollectionNotFound):
        return fail(tool_name, str(exc), "NOT_FOUND")
    return fail(tool_name, str(exc), "INTERNAL_ERROR")


def map_db_error(tool_name: str, exc: Exception) -> ToolResult:
    if isinstance(exc, (InvalidTransition, StaleStatusError)):
        return fail(tool_name, str(exc), "VALIDATION_ERROR")
    if isinstance(exc, NotFoundError):
        return fail(tool_name, str(exc), "NOT_FOUND")
    return fail(tool_name, str(exc), "INTERNAL_ERROR")


def make_plan_id(user_id: str, goal: str, idempotency_key: str | None) -> str:
    if idempotency_key:
        return idempotency_key
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{user_id}:{goal}"))


def find_application(
    company: str,
    position: str,
    idempotency_key: str | None,
    apps: list[dict],
) -> dict | None:
    del idempotency_key
    for app in apps:
        if app.get("company") == company and app.get("position") == position:
            return app
    return None
