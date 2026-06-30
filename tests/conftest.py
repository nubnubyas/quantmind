import os
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _pg_reachable() -> bool:
    """Check if PostgreSQL is reachable without importing src.db (avoids side effects)."""
    url = os.environ.get("POSTGRES_URL")
    if not url:
        return False
    try:
        engine = create_engine(url, connect_args={"connect_timeout": 3})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except OperationalError:
        return False


def _qdrant_reachable() -> bool:
    """Check if Qdrant is reachable."""
    import socket

    host = os.environ.get("QDRANT_HOST", "localhost")
    port = int(os.environ.get("QDRANT_PORT", "6333"))
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def requires_db():
    """Skip tests that need PostgreSQL if it's not running."""
    if not _pg_reachable():
        pytest.skip(
            "PostgreSQL not reachable. Start it with: "
            "docker compose -f docker-compose.yml up -d postgres"
        )


@pytest.fixture(scope="session")
def requires_qdrant():
    """Skip tests that need Qdrant if it's not running."""
    if not _qdrant_reachable():
        pytest.skip(
            "Qdrant not reachable. Start it with: "
            "docker compose -f docker-compose.yml up -d qdrant"
        )


@pytest.fixture(scope="session")
def requires_llm():
    """Skip tests that need a real LLM API key."""
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY not set")
