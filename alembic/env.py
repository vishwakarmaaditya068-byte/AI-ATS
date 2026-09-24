"""
Alembic migration environment for AI-ATS.

Runs synchronously via psycopg3 using PostgresSettings.sync_uri.
Imports all SQL models so autogenerate sees the full metadata graph.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from src.data.sql.base import SQLBase

from src.data.sql.models import (  # noqa: F401
    AuditRecord,
    JobRecord,
    MatchRecord,
    Workspace,
)
from src.utils.config import get_settings

# --- Alembic config ---------------------------------------------------------
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Inject the real DB URL from our settings layer (no hardcoded credentials).
config.set_main_option("sqlalchemy.url", get_settings().postgres.sync_uri)

target_metadata = SQLBase.metadata


def run_migrations_offline() -> None:
    """Generate SQL without a live DB connection — writes to stdout."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live DB connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
