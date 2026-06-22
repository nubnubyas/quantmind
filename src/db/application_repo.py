from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.base import model_to_dict
from src.db.exceptions import InvalidTransition, NotFoundError, StaleStatusError
from src.db.models import Application

VALID_TRANSITIONS: dict[str, set[str]] = {
    "researching": {"applied"},
    "applied": {"phone_screen", "rejected"},
    "phone_screen": {"technical", "rejected"},
    "technical": {"offer", "rejected"},
}


class ApplicationRepo:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        user_id: str,
        company: str,
        position: str,
        status: str = "researching",
    ) -> str:
        app = Application(
            user_id=user_id,
            company=company,
            position=position,
            status=status,
        )
        self._session.add(app)
        self._session.flush()
        return str(app.id)

    def update_status(
        self,
        app_id: str,
        new_status: str,
        expected_status: str | None = None,
    ) -> None:
        app = self._session.get(Application, uuid.UUID(app_id))
        if app is None:
            raise NotFoundError("Application", app_id)

        if expected_status is not None and app.status != expected_status:
            raise StaleStatusError(expected_status, app.status)

        allowed = VALID_TRANSITIONS.get(app.status, set())
        if new_status not in allowed:
            raise InvalidTransition(app.status, new_status)

        app.status = new_status
        app.updated_at = datetime.now(timezone.utc)
        self._session.flush()

    def add_notes(self, app_id: str, notes: str) -> None:
        app = self._session.get(Application, uuid.UUID(app_id))
        if app is None:
            raise NotFoundError("Application", app_id)

        if app.notes:
            app.notes = f"{app.notes}\n{notes}"
        else:
            app.notes = notes
        app.updated_at = datetime.now(timezone.utc)
        self._session.flush()

    def list_by_user(self, user_id: str) -> list[dict]:
        stmt = select(Application).where(Application.user_id == user_id)
        return [model_to_dict(a) for a in self._session.scalars(stmt).all()]
