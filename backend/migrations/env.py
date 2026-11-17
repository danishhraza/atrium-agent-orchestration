"""Alembic environment. The database URL comes from the app's settings (ATRIUM_DATABASE_URL), never from alembic.ini.

Usage from backend/:
  uv run alembic upgrade head                       apply migrations (the app also does this on startup)
  uv run alembic revision --autogenerate -m "..."   after changing a model in app/db.py
"""
from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlmodel import SQLModel

from app import db  # noqa: F401  registers every table on SQLModel.metadata
from app.config import settings

config = context.config
target_metadata = SQLModel.metadata


def run_migrations_offline() -> None:
    context.configure(url=settings.db_url, target_metadata=target_metadata, literal_binds=True,
                      render_as_batch=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")  # set when the app runs migrations itself
    if connection is None:
        engine = engine_from_config({"sqlalchemy.url": settings.db_url}, prefix="sqlalchemy.", poolclass=pool.NullPool)
        with engine.connect() as connection:
            _run(connection)
    else:
        _run(connection)


def _run(connection) -> None:
    # render_as_batch: SQLite cannot ALTER most things in place; batch mode rebuilds the table instead.
    context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
