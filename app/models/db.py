"""
Database engine and session management.

Kept separate from db_models.py so that anything needing a session
(FastAPI endpoints, scripts, tests) imports from here without pulling
in table definitions it doesn't need, and vice versa.
"""

from dotenv import load_dotenv

load_dotenv()
import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///civicinbox.db")


connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)

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
