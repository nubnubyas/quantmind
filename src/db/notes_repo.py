from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.base import model_to_dict
from src.db.models import ResearchNote


class NotesRepo:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        user_id: str,
        thread_id: str,
        title: str,
        content: str,
        sources: list[dict],
    ) -> str:
        note = ResearchNote(
            user_id=user_id,
            thread_id=thread_id,
            title=title,
            content=content,
            sources=sources,
        )
        self._session.add(note)
        self._session.flush()
        return str(note.id)

    def get(self, note_id: str) -> dict | None:
        note = self._session.get(ResearchNote, uuid.UUID(note_id))
        if note is None:
            return None
        return model_to_dict(note)

    def list_by_user(self, user_id: str) -> list[dict]:
        stmt = select(ResearchNote).where(ResearchNote.user_id == user_id)
        return [model_to_dict(n) for n in self._session.scalars(stmt).all()]

    def list_by_thread(self, thread_id: str) -> list[dict]:
        stmt = select(ResearchNote).where(ResearchNote.thread_id == thread_id)
        return [model_to_dict(n) for n in self._session.scalars(stmt).all()]
