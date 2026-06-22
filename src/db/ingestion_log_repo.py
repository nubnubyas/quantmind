from __future__ import annotations

from sqlalchemy.orm import Session

from src.db.models import IngestionLog


class IngestionLogRepo:
    def __init__(self, session: Session) -> None:
        self._session = session

    def record(
        self,
        source_type: str,
        source_id: str,
        collection: str,
        status: str,
        chunk_count: int,
        doc_id: str | None = None,
        error_message: str | None = None,
        checksum: str | None = None,
    ) -> str:
        log = IngestionLog(
            source_type=source_type,
            source_id=source_id,
            collection=collection,
            status=status,
            chunk_count=chunk_count,
            doc_id=doc_id,
            error_message=error_message,
            checksum=checksum,
        )
        self._session.add(log)
        self._session.flush()
        return str(log.id)
