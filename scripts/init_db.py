"""
Create all tables in the real database (civicinbox.db) if they don't
already exist. Run once before starting the app locally, or after
pulling schema changes. Idempotent — create_all() skips tables that
already exist, so re-running this is always safe.
"""

from app.models.db import engine
from app.models.db_models import Base


def main() -> None:
    Base.metadata.create_all(engine)
    print(f"Tables created (or already present) at {engine.url}")


if __name__ == "__main__":
    main()
