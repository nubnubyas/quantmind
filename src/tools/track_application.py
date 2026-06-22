"""T8: Track job applications with idempotent create."""

from __future__ import annotations

from sqlalchemy.orm import Session

from src.db import ApplicationRepo, get_session
from src.tools._helpers import fail, find_application, map_db_error, ok
from src.tools.types import ToolResult

TOOL_NAME = "track_application"


def _resolve_app(
    repo: ApplicationRepo,
    user_id: str,
    company: str,
    position: str,
    idempotency_key: str | None,
) -> dict | None:
    apps = repo.list_by_user(user_id)
    return find_application(company, position, idempotency_key, apps)


def _track_with_repo(
    repo: ApplicationRepo,
    session: Session,
    user_id: str,
    company: str,
    position: str,
    action: str,
    status: str | None,
    notes: str | None,
    idempotency_key: str | None,
) -> ToolResult:
    if action == "create":
        existing = _resolve_app(repo, user_id, company, position, idempotency_key)
        if existing:
            return ok(
                TOOL_NAME,
                "application",
                app_id=existing["id"],
                action=action,
                company=company,
                position=position,
                status=existing.get("status"),
                existing=True,
            )

        initial_status = status or "researching"
        app_id = repo.create(user_id, company, position, status=initial_status)
        if notes:
            repo.add_notes(app_id, notes)

        return ok(
            TOOL_NAME,
            "application",
            app_id=app_id,
            action=action,
            company=company,
            position=position,
            status=initial_status,
            existing=False,
        )

    existing = _resolve_app(repo, user_id, company, position, idempotency_key)
    if not existing:
        return fail(
            TOOL_NAME,
            f"No application found for {company} / {position}",
            "NOT_FOUND",
        )

    app_id = existing["id"]

    try:
        if action == "update_status":
            if not status:
                return fail(TOOL_NAME, "update_status requires status", "VALIDATION_ERROR")
            repo.update_status(app_id, status)
        elif action == "add_notes":
            if not notes:
                return fail(TOOL_NAME, "add_notes requires notes", "VALIDATION_ERROR")
            repo.add_notes(app_id, notes)
        else:
            return fail(TOOL_NAME, f"Unknown action: {action}", "VALIDATION_ERROR")
    except Exception as exc:  # noqa: BLE001
        return map_db_error(TOOL_NAME, exc)

    apps = repo.list_by_user(user_id)
    updated = next((a for a in apps if a["id"] == app_id), existing)

    return ok(
        TOOL_NAME,
        "application",
        app_id=app_id,
        action=action,
        company=company,
        position=position,
        status=updated.get("status"),
        existing=False,
    )


def track_application(
    user_id: str,
    company: str,
    position: str,
    action: str,
    status: str | None = None,
    notes: str | None = None,
    idempotency_key: str | None = None,
    *,
    repo: ApplicationRepo | None = None,
    session: Session | None = None,
) -> ToolResult:
    if repo is not None and session is not None:
        return _track_with_repo(
            repo, session, user_id, company, position, action, status, notes, idempotency_key
        )

    with get_session() as sess:
        application_repo = repo or ApplicationRepo(sess)
        return _track_with_repo(
            application_repo,
            sess,
            user_id,
            company,
            position,
            action,
            status,
            notes,
            idempotency_key,
        )
