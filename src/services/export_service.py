"""Workspace export — ZIP bundle, SQLite file, and training JSONL."""
from __future__ import annotations

import enum
import json
import sqlite3
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Serialize a SQLAlchemy model instance to a plain dict.

    UUID → str, datetime → ISO-8601, Enum → .value, everything else as-is.
    """
    result: dict[str, Any] = {}
    for col in row.__table__.columns:
        val = getattr(row, col.name, None)
        if isinstance(val, uuid.UUID):
            val = str(val)
        elif isinstance(val, datetime):
            val = val.isoformat()
        elif isinstance(val, enum.Enum):
            val = val.value
        result[col.name] = val
    return result


class WorkspaceExportService:
    """Exports a single workspace and all its data to various formats."""

    def __init__(self) -> None:
        from src.data.sql.repositories import (
            get_audit_record_repository,
            get_job_record_repository,
            get_match_record_repository,
            get_workspace_repository,
        )
        self._ws_repo = get_workspace_repository()
        self._job_repo = get_job_record_repository()
        self._match_repo = get_match_record_repository()
        self._audit_repo = get_audit_record_repository()

    # ── Public API ─────────────────────────────────────────────────────────────

    def export_to_zip(
        self,
        workspace_id: uuid.UUID | str,
        output_path: Path,
    ) -> Path:
        """Bundle workspace + jobs + matches + audit logs into a ZIP file."""
        ws_id = self._coerce(workspace_id)
        ws, jobs, matches, audits = self._load_all(ws_id)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in (ws.name if ws else "workspace")  # type: ignore[attr-defined]
        )
        folder = f"workspace_{safe_name}_{timestamp}"

        manifest = {
            "version": "1.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "workspace_id": str(ws_id),
            "workspace_name": ws.name if ws else None,  # type: ignore[attr-defined]
            "counts": {
                "jobs": len(jobs),
                "matches": len(matches),
                "audit_logs": len(audits),
            },
        }

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(
                f"{folder}/manifest.json",
                json.dumps(manifest, indent=2, default=str),
            )
            if ws:
                zf.writestr(
                    f"{folder}/workspace.json",
                    json.dumps(_row_to_dict(ws), indent=2, default=str),
                )
            zf.writestr(
                f"{folder}/jobs.json",
                json.dumps([_row_to_dict(j) for j in jobs], indent=2, default=str),
            )
            zf.writestr(
                f"{folder}/matches.json",
                json.dumps([_row_to_dict(m) for m in matches], indent=2, default=str),
            )
            zf.writestr(
                f"{folder}/audit_logs.json",
                json.dumps([_row_to_dict(a) for a in audits], indent=2, default=str),
            )

        return output_path

    def export_to_sqlite(
        self,
        workspace_id: uuid.UUID | str,
        output_path: Path,
    ) -> Path:
        """Write workspace data to a portable SQLite file."""
        ws_id = self._coerce(workspace_id)
        _, jobs, matches, audits = self._load_all(ws_id)

        ws_row = self._ws_repo.get(ws_id)
        if ws_row is None:
            raise ValueError(f"Workspace {ws_id} not found")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(output_path))
        try:
            con.execute("PRAGMA journal_mode=WAL")
            self._create_sqlite_tables(con)

            # Workspace
            d = _row_to_dict(ws_row)
            self._sqlite_insert(con, "workspaces", d)

            # Jobs
            for j in jobs:
                self._sqlite_insert(con, "jobs", _row_to_dict(j))

            # Matches
            for m in matches:
                self._sqlite_insert(con, "matches", _row_to_dict(m))

            # Audit
            for a in audits:
                row = _row_to_dict(a)
                # JSONB fields are already dicts; store as JSON text
                for key in ("actor", "resource", "changes", "ai_decision", "bias_audit", "context"):
                    if isinstance(row.get(key), (dict, list)):
                        row[key] = json.dumps(row[key], default=str)
                self._sqlite_insert(con, "audit_logs", row)

            con.commit()
        finally:
            con.close()

        return output_path

    def export_training_jsonl(
        self,
        workspace_id: uuid.UUID | str,
        output_path: Path,
    ) -> Path:
        """Write one JSON line per match: AI score + manual override + reason."""
        ws_id = self._coerce(workspace_id)
        jobs = self._job_repo.list_for_workspace(ws_id, limit=10_000)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with output_path.open("w", encoding="utf-8") as fh:
            for job in jobs:
                matches = self._match_repo.list_for_job(job.id, limit=10_000)
                for m in matches:
                    record: dict[str, Any] = {
                        "job_id": str(m.job_id),  # type: ignore[attr-defined]
                        "candidate_mongo_id": m.candidate_mongo_id,  # type: ignore[attr-defined]
                        "ai_score": float(m.overall_score),  # type: ignore[attr-defined]
                        "skills_score": float(m.skills_score) if m.skills_score is not None else None,  # type: ignore[attr-defined]
                        "experience_score": float(m.experience_score) if m.experience_score is not None else None,  # type: ignore[attr-defined]
                        "education_score": float(m.education_score) if m.education_score is not None else None,  # type: ignore[attr-defined]
                        "semantic_score": float(m.semantic_score) if m.semantic_score is not None else None,  # type: ignore[attr-defined]
                        "score_level": m.score_level,  # type: ignore[attr-defined]
                        "manual_score": float(m.manual_score_override) if m.manual_score_override is not None else None,  # type: ignore[attr-defined]
                        "override_reason": m.override_reason,  # type: ignore[attr-defined]
                        "overridden_by": m.overridden_by,  # type: ignore[attr-defined]
                        "scoring_model_version": m.scoring_model_version,  # type: ignore[attr-defined]
                    }
                    fh.write(json.dumps(record, default=str) + "\n")
                    written += 1

        return output_path

    # ── Private helpers ────────────────────────────────────────────────────────

    def _load_all(
        self, ws_id: uuid.UUID
    ) -> tuple[Any, list[Any], list[Any], list[Any]]:
        ws = self._ws_repo.get(ws_id)
        jobs = self._job_repo.list_for_workspace(ws_id, limit=10_000)
        matches: list[Any] = []
        for job in jobs:
            matches.extend(self._match_repo.list_for_job(job.id, limit=10_000))
        audits = self._audit_repo.list_for_workspace(ws_id, limit=50_000)
        return ws, jobs, matches, audits

    @staticmethod
    def _coerce(value: uuid.UUID | str) -> uuid.UUID:
        return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))

    @staticmethod
    def _create_sqlite_tables(con: sqlite3.Connection) -> None:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS workspaces (
                id TEXT PRIMARY KEY, name TEXT, description TEXT,
                status TEXT, created_by TEXT,
                archived_at TEXT, last_opened_at TEXT,
                created_at TEXT, updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, workspace_id TEXT, mongo_doc_id TEXT,
                title TEXT, company_name TEXT, description_snippet TEXT,
                status TEXT, created_at TEXT, updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS matches (
                id TEXT PRIMARY KEY, job_id TEXT, candidate_mongo_id TEXT,
                overall_score REAL, skills_score REAL, experience_score REAL,
                education_score REAL, semantic_score REAL, keyword_score REAL,
                score_level TEXT, manual_score_override REAL, override_reason TEXT,
                overridden_by TEXT, explanation TEXT, bias_check TEXT,
                score_breakdown TEXT, scoring_model_version TEXT,
                created_at TEXT, updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS audit_logs (
                id TEXT PRIMARY KEY, workspace_id TEXT,
                action TEXT, action_description TEXT,
                actor TEXT, resource TEXT, changes TEXT,
                ai_decision TEXT, bias_audit TEXT, context TEXT,
                related_candidate_mongo_id TEXT, related_job_id TEXT,
                related_match_id TEXT, compliance_relevant INTEGER,
                retention_days INTEGER, session_id TEXT, request_id TEXT,
                occurred_at TEXT, created_at TEXT, updated_at TEXT
            );
        """)

    @staticmethod
    def _sqlite_insert(
        con: sqlite3.Connection, table: str, row: dict[str, Any]
    ) -> None:
        cols = ", ".join(row.keys())
        placeholders = ", ".join("?" for _ in row)
        con.execute(
            f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({placeholders})",
            list(row.values()),
        )


_export_service: Optional[WorkspaceExportService] = None


def get_export_service() -> WorkspaceExportService:
    global _export_service
    if _export_service is None:
        _export_service = WorkspaceExportService()
    return _export_service
