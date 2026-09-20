"""
Database engine and session management.

Kept separate from db_models.py so that anything needing a session
(FastAPI endpoints, scripts, tests) imports from here without pulling
in table definitions it doesn't need, and vice versa.
"""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# SQLite for now. Per the master plan, swapping to Postgres later means
# changing this URL (e.g. to a postgresql:// DSN pulled from an env var)
# -- the rest of the app's session-handling code does not change.
DATABASE_URL = "sqlite:///./civicinbox.db"

# check_same_thread=False is SQLite-specific: SQLite by default only
# allows the thread that created a connection to use it. FastAPI can
# serve requests from different threads, so this is required for the
# API to work at all. This flag has no meaning (and isn't needed) once
# you're on Postgres.
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency: yields a session, guarantees it's closed after
    the request finishes (even if the request raised an exception).
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
