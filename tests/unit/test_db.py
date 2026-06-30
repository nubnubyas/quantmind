import pytest
from sqlalchemy import text

from src.db import (
    ApplicationRepo,
    IngestionLogRepo,
    InvalidTransition,
    NotesRepo,
    StaleStatusError,
    create_all,
    get_session,
)


@pytest.fixture(scope="module")
def setup_db(requires_db):
    """Create all tables once per module. Skipped when PostgreSQL unreachable."""
    create_all()
    yield


@pytest.fixture
def db_session(setup_db):
    with get_session() as session:
        yield session
        session.execute(text("DELETE FROM research_notes"))
        session.execute(text("DELETE FROM applications"))
        session.execute(text("DELETE FROM ingestion_log"))


def test_invalid_transition_researching_to_offer(db_session):
    repo = ApplicationRepo(db_session)
    app_id = repo.create("user1", "Two Sigma", "Quant Researcher")

    with pytest.raises(InvalidTransition):
        repo.update_status(app_id, "offer")


def test_valid_transition_chain(db_session):
    repo = ApplicationRepo(db_session)
    app_id = repo.create("user1", "Citadel", "Quant Dev")

    repo.update_status(app_id, "applied")
    repo.update_status(app_id, "phone_screen", expected_status="applied")

    apps = repo.list_by_user("user1")
    assert len(apps) == 1
    assert apps[0]["status"] == "phone_screen"


def test_expected_status_optimistic_lock(db_session):
    repo = ApplicationRepo(db_session)
    app_id = repo.create("user1", "Jane Street", "Researcher")

    with pytest.raises(StaleStatusError):
        repo.update_status(app_id, "applied", expected_status="phone_screen")


def test_notes_repo_crud(db_session):
    repo = NotesRepo(db_session)
    note_id = repo.create(
        "user1",
        "thread-abc",
        "Momentum Notes",
        "Content here",
        [{"paper_id": "p1", "title": "Paper 1"}],
    )

    note = repo.get(note_id)
    assert note is not None
    assert note["title"] == "Momentum Notes"
    assert note["user_id"] == "user1"

    by_user = repo.list_by_user("user1")
    assert len(by_user) == 1

    by_thread = repo.list_by_thread("thread-abc")
    assert len(by_thread) == 1


def test_ingestion_log_repo_record(db_session):
    repo = IngestionLogRepo(db_session)
    log_id = repo.record(
        source_type="arxiv",
        source_id="2301.12345",
        collection="papers",
        status="done",
        chunk_count=42,
        doc_id="doc-1",
        checksum="abc123",
    )
    assert log_id is not None
    assert len(log_id) == 36
