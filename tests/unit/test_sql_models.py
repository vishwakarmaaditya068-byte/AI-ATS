from __future__ import annotations

import pytest

from src.data.sql.base import SQLBase
from src.data.sql.models import (
    AuditRecord,
    JobRecord,
    JobStatus,
    MatchRecord,
    Workspace,
    WorkspaceStatus,
)


class TestMetadataGraph:
    def test_all_tables_registered(self) -> None:
        tables = set(SQLBase.metadata.tables.keys())
        assert {"workspaces", "jobs", "matches", "audit_logs"} <= tables

    def test_workspace_has_timestamps(self) -> None:
        cols = Workspace.__table__.columns
        assert "created_at" in cols
        assert "updated_at" in cols
        assert cols["created_at"].nullable is False
        assert cols["updated_at"].nullable is False

    def test_workspace_status_enum_values(self) -> None:
        assert {s.value for s in WorkspaceStatus} == {
            "active",
            "archived",
            "purged",
        }

    def test_job_status_enum_values(self) -> None:
        assert {s.value for s in JobStatus} == {
            "draft",
            "open",
            "paused",
            "closed",
            "filled",
        }


class TestWorkspaceModel:
    def test_has_lifecycle_columns(self) -> None:
        cols = Workspace.__table__.columns
        assert "status" in cols
        assert "archived_at" in cols
        assert "last_opened_at" in cols
        assert cols["archived_at"].nullable is True

    def test_cascades_to_jobs(self) -> None:
        ws_jobs_fk = [
            fk for fk in JobRecord.__table__.foreign_keys if fk.column.table.name == "workspaces"
        ]
        assert len(ws_jobs_fk) == 1
        assert ws_jobs_fk[0].ondelete == "CASCADE"


class TestJobRecordModel:
    def test_workspace_fk_not_null(self) -> None:
        cols = JobRecord.__table__.columns
        assert cols["workspace_id"].nullable is False

    def test_mongo_doc_id_index(self) -> None:
        index_names = {idx.name for idx in JobRecord.__table__.indexes}
        assert "ix_jobs_workspace_id_status" in index_names

    def test_cascades_to_matches(self) -> None:
        job_matches_fk = [
            fk for fk in MatchRecord.__table__.foreign_keys if fk.column.table.name == "jobs"
        ]
        assert len(job_matches_fk) == 1
        assert job_matches_fk[0].ondelete == "CASCADE"


class TestMatchRecordModel:
    def test_unique_job_candidate_constraint(self) -> None:
        constraint_names = {c.name for c in MatchRecord.__table__.constraints if c.name}
        assert "uq_match_job_candidate" in constraint_names

    def test_score_range_check_constraints(self) -> None:
        constraint_names = {c.name for c in MatchRecord.__table__.constraints if c.name}
        assert "ck_match_overall_score_range" in constraint_names
        assert "ck_match_override_score_range" in constraint_names

    def test_effective_score_falls_back_to_overall(self) -> None:
        # MatchRecord instances don't need a session for property computation.
        match = MatchRecord(
            job_id="00000000-0000-0000-0000-000000000000",
            candidate_mongo_id="507f1f77bcf86cd799439011",
            overall_score=0.75,
        )
        assert match.effective_score == pytest.approx(0.75)

    def test_effective_score_uses_override_when_present(self) -> None:
        match = MatchRecord(
            job_id="00000000-0000-0000-0000-000000000000",
            candidate_mongo_id="507f1f77bcf86cd799439011",
            overall_score=0.75,
            manual_score_override=0.90,
        )
        assert match.effective_score == pytest.approx(0.90)


class TestAuditRecordModel:
    def test_workspace_fk_nullable_set_null(self) -> None:
        """Audit survives workspace purge — FK is SET NULL, not CASCADE."""
        fks = [
            fk for fk in AuditRecord.__table__.foreign_keys if fk.column.table.name == "workspaces"
        ]
        assert len(fks) == 1
        assert fks[0].ondelete == "SET NULL"

    def test_required_columns(self) -> None:
        cols = AuditRecord.__table__.columns
        assert cols["action"].nullable is False
        assert cols["action_description"].nullable is False
        assert cols["occurred_at"].nullable is False

    def test_has_indexes_for_common_queries(self) -> None:
        index_names = {idx.name for idx in AuditRecord.__table__.indexes}
        expected = {
            "ix_audit_workspace_occurred",
            "ix_audit_action_occurred",
            "ix_audit_compliance",
            "ix_audit_related_match",
        }
        assert expected <= index_names
