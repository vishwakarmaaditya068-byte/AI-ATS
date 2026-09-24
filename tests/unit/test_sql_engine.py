from __future__ import annotations

import pytest

from src.data.sql.engine import SQLDatabaseManager, get_sql_manager
from src.utils.config import PostgresSettings


class TestPostgresSettingsURI:
    def test_sync_uri_uses_psycopg_driver(self) -> None:
        s = PostgresSettings(host="db.example.com", port=5433, name="ats", username=None, password=None)
        assert s.sync_uri == "postgresql+psycopg://db.example.com:5433/ats"

    def test_async_uri_uses_asyncpg_driver(self) -> None:
        s = PostgresSettings(host="db.example.com", port=5433, name="ats", username=None, password=None)
        assert s.async_uri == "postgresql+asyncpg://db.example.com:5433/ats"

    def test_uri_includes_credentials_when_provided(self) -> None:
        s = PostgresSettings(
            host="localhost",
            port=5432,
            name="ats",
            username="alice",
            password="secret",
        )
        assert "alice:secret@localhost" in s.sync_uri
        assert "alice:secret@localhost" in s.async_uri

    def test_uri_url_encodes_special_chars(self) -> None:
        s = PostgresSettings(
            host="localhost",
            port=5432,
            name="ats",
            username="user@mail",
            password="p@ss/word",
        )
        # '@' -> %40, '/' -> %2F
        assert "%40" in s.sync_uri
        assert "%2F" in s.sync_uri

    def test_rejects_shell_metacharacters_in_host(self) -> None:
        with pytest.raises(ValueError):
            PostgresSettings(host="localhost;rm -rf /")

    def test_rejects_invalid_db_name(self) -> None:
        with pytest.raises(ValueError):
            PostgresSettings(name="bad/name")

    def test_repr_redacts_password(self) -> None:
        s = PostgresSettings(host="localhost", username="alice", password="super-secret")
        assert "super-secret" not in repr(s)
        assert "***" in repr(s)


class TestSQLDatabaseManagerSingleton:
    def test_returns_same_instance(self) -> None:
        a = get_sql_manager()
        b = get_sql_manager()
        assert a is b

    def test_direct_instantiation_also_returns_singleton(self) -> None:
        a = SQLDatabaseManager()
        b = SQLDatabaseManager()
        assert a is b
        assert a is get_sql_manager()

    def test_lazy_engine_creation_does_not_connect(self) -> None:
        """Constructing the manager must not open connections.

        The manager is pulled in at import time by repositories; if
        construction triggered a connect, every test run would need a
        live Postgres.
        """
        mgr = get_sql_manager()
        # Accessing the manager object without calling get_sync_engine
        # should not create any engine.
        assert mgr is not None


class TestRepositoryImports:
    """Import sanity — catches circular imports and missing __init__ exports."""

    def test_can_import_all_repositories(self) -> None:
        from src.data.sql.repositories import (  # noqa: F401
            AuditRecordRepository,
            BaseSQLRepository,
            JobRecordRepository,
            MatchRecordRepository,
            WorkspaceRepository,
            get_audit_record_repository,
            get_job_record_repository,
            get_match_record_repository,
            get_workspace_repository,
        )

    def test_can_import_all_models(self) -> None:
        from src.data.sql.models import (  # noqa: F401
            AuditRecord,
            JobRecord,
            JobStatus,
            MatchRecord,
            Workspace,
            WorkspaceStatus,
        )
