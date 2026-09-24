from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001_baseline"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- workspaces ------------------------------------------------------
    workspace_status = postgresql.ENUM("active", "archived", "purged", name="workspace_status")
    workspace_status.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "workspaces",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "status",
            postgresql.ENUM(
                "active",
                "archived",
                "purged",
                name="workspace_status",
                create_type=False,
            ),
            nullable=False,
            server_default="active",
        ),
        sa.Column("created_by", sa.String(200), nullable=True),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_opened_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_workspaces_status", "workspaces", ["status"])
    op.create_index("ix_workspaces_last_opened_at", "workspaces", ["last_opened_at"])

    # --- jobs ------------------------------------------------------------
    job_status = postgresql.ENUM("draft", "open", "paused", "closed", "filled", name="job_status")
    job_status.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "workspace_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("mongo_doc_id", sa.String(24), nullable=True),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("company_name", sa.String(300), nullable=True),
        sa.Column("description_snippet", sa.Text(), nullable=True),
        sa.Column(
            "status",
            postgresql.ENUM(
                "draft",
                "open",
                "paused",
                "closed",
                "filled",
                name="job_status",
                create_type=False,
            ),
            nullable=False,
            server_default="open",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_jobs_mongo_doc_id", "jobs", ["mongo_doc_id"])
    op.create_index("ix_jobs_workspace_id_status", "jobs", ["workspace_id", "status"])
    op.create_index("ix_jobs_created_at", "jobs", ["created_at"])

    # --- matches ---------------------------------------------------------
    op.create_table(
        "matches",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("candidate_mongo_id", sa.String(24), nullable=False),
        sa.Column("overall_score", sa.Numeric(5, 4), nullable=False),
        sa.Column("skills_score", sa.Numeric(5, 4), nullable=True),
        sa.Column("experience_score", sa.Numeric(5, 4), nullable=True),
        sa.Column("education_score", sa.Numeric(5, 4), nullable=True),
        sa.Column("semantic_score", sa.Numeric(5, 4), nullable=True),
        sa.Column("keyword_score", sa.Numeric(5, 4), nullable=True),
        sa.Column("score_level", sa.String(32), nullable=True),
        sa.Column("manual_score_override", sa.Numeric(5, 4), nullable=True),
        sa.Column("override_reason", sa.Text(), nullable=True),
        sa.Column("overridden_by", sa.String(200), nullable=True),
        sa.Column(
            "explanation",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "bias_check",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "score_breakdown",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "scoring_model_version",
            sa.String(32),
            nullable=False,
            server_default="1.0",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("job_id", "candidate_mongo_id", name="uq_match_job_candidate"),
        sa.CheckConstraint(
            "overall_score >= 0 AND overall_score <= 1",
            name="ck_match_overall_score_range",
        ),
        sa.CheckConstraint(
            "manual_score_override IS NULL OR "
            "(manual_score_override >= 0 AND manual_score_override <= 1)",
            name="ck_match_override_score_range",
        ),
    )
    op.create_index("ix_matches_job_id_overall_score", "matches", ["job_id", "overall_score"])
    op.create_index("ix_matches_candidate_mongo_id", "matches", ["candidate_mongo_id"])

    # --- audit_logs ------------------------------------------------------
    op.create_table(
        "audit_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "workspace_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("workspaces.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("action", sa.String(64), nullable=False),
        sa.Column("action_description", sa.Text(), nullable=False),
        sa.Column(
            "actor",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "resource",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "changes",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "ai_decision",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "bias_audit",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "context",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("related_candidate_mongo_id", sa.String(24), nullable=True),
        sa.Column("related_job_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("related_match_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "compliance_relevant",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "retention_days",
            sa.Integer(),
            nullable=False,
            server_default="2555",
        ),
        sa.Column("session_id", sa.String(64), nullable=True),
        sa.Column("request_id", sa.String(64), nullable=True),
        sa.Column(
            "occurred_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_audit_workspace_occurred", "audit_logs", ["workspace_id", "occurred_at"])
    op.create_index("ix_audit_action_occurred", "audit_logs", ["action", "occurred_at"])
    op.create_index("ix_audit_compliance", "audit_logs", ["compliance_relevant", "occurred_at"])
    op.create_index("ix_audit_related_candidate", "audit_logs", ["related_candidate_mongo_id"])
    op.create_index("ix_audit_related_job", "audit_logs", ["related_job_id"])
    op.create_index("ix_audit_related_match", "audit_logs", ["related_match_id"])


def downgrade() -> None:
    op.drop_index("ix_audit_related_match", table_name="audit_logs")
    op.drop_index("ix_audit_related_job", table_name="audit_logs")
    op.drop_index("ix_audit_related_candidate", table_name="audit_logs")
    op.drop_index("ix_audit_compliance", table_name="audit_logs")
    op.drop_index("ix_audit_action_occurred", table_name="audit_logs")
    op.drop_index("ix_audit_workspace_occurred", table_name="audit_logs")
    op.drop_table("audit_logs")

    op.drop_index("ix_matches_candidate_mongo_id", table_name="matches")
    op.drop_index("ix_matches_job_id_overall_score", table_name="matches")
    op.drop_table("matches")

    op.drop_index("ix_jobs_created_at", table_name="jobs")
    op.drop_index("ix_jobs_workspace_id_status", table_name="jobs")
    op.drop_index("ix_jobs_mongo_doc_id", table_name="jobs")
    op.drop_table("jobs")
    op.execute("DROP TYPE IF EXISTS job_status")

    op.drop_index("ix_workspaces_last_opened_at", table_name="workspaces")
    op.drop_index("ix_workspaces_status", table_name="workspaces")
    op.drop_table("workspaces")
    op.execute("DROP TYPE IF EXISTS workspace_status")
