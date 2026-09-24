"""PostgreSQL repositories (SQLAlchemy 2.0)."""
from src.data.sql.repositories.audit_record_repo import (
    AuditRecordRepository,
    get_audit_record_repository,
)
from src.data.sql.repositories.base import BaseSQLRepository
from src.data.sql.repositories.job_record_repo import (
    JobRecordRepository,
    get_job_record_repository,
)
from src.data.sql.repositories.match_record_repo import (
    MatchRecordRepository,
    get_match_record_repository,
)
from src.data.sql.repositories.workspace_repo import (
    WorkspaceRepository,
    get_workspace_repository,
)

__all__ = [
    "AuditRecordRepository",
    "BaseSQLRepository",
    "JobRecordRepository",
    "MatchRecordRepository",
    "WorkspaceRepository",
    "get_audit_record_repository",
    "get_job_record_repository",
    "get_match_record_repository",
    "get_workspace_repository",
]
