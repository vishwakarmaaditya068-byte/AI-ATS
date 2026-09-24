"""Workspace import — reverse of export_service: ZIP bundle or SQLite file."""
from __future__ import annotations

import json
import sqlite3
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class WorkspaceImportService:
    """Imports a workspace export file and creates a new workspace with fresh IDs."""

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

    def import_from_zip(self, zip_path: Path) -> Any:
        """Read a ZIP export and create a new workspace with remapped IDs."""
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()

            # Locate data files inside whichever top-level folder the ZIP has
            def _read(filename: str) -> Any:
                candidates = [n for n in names if n.endswith(f"/{filename}")]
                if not candidates:
                    return None
                return json.loads(zf.read(candidates[0]).decode())

            manifest = _read("manifest.json")
            ws_data: Optional[dict[str, Any]] = _read("workspace.json")
            jobs_data: list[dict[str, Any]] = _read("jobs.json") or []
            matches_data: list[dict[str, Any]] = _read("matches.json") or []
            audits_data: list[dict[str, Any]] = _read("audit_logs.json") or []

        if ws_data is None:
            raise ValueError("Invalid export: workspace.json not found")

        return self._rebuild(ws_data, jobs_data, matches_data, audits_data)

    def import_from_sqlite(self, sqlite_path: Path) -> Any:
        """Read a SQLite export and create a new workspace with remapped IDs."""
        con = sqlite3.connect(str(sqlite_path))
        con.row_factory = sqlite3.Row
        try:
            ws_rows = list(con.execute("SELECT * FROM workspaces").fetchall())
            if not ws_rows:
                raise ValueError("Invalid export: workspaces table is empty")
            ws_data = dict(ws_rows[0])

            jobs_data = [dict(r) for r in con.execute("SELECT * FROM jobs").fetchall()]
            matches_data = [
                dict(r) for r in con.execute("SELECT * FROM matches").fetchall()
            ]
            audits_raw = [
                dict(r) for r in con.execute("SELECT * FROM audit_logs").fetchall()
            ]
        finally:
            con.close()

        # SQLite stores JSONB fields as text strings — deserialise them back
        json_cols = {"actor", "resource", "changes", "ai_decision", "bias_audit", "context"}
        audits_data: list[dict[str, Any]] = []
        for row in audits_raw:
            for col in json_cols:
                if isinstance(row.get(col), str):
                    try:
                        row[col] = json.loads(row[col])
                    except (json.JSONDecodeError, TypeError):
                        pass
            audits_data.append(row)

        return self._rebuild(ws_data, jobs_data, matches_data, audits_data)

    # ── Core rebuild logic ─────────────────────────────────────────────────────

    def _rebuild(
        self,
        ws_data: dict[str, Any],
        jobs_data: list[dict[str, Any]],
        matches_data: list[dict[str, Any]],
        audits_data: list[dict[str, Any]],
    ) -> Any:
        """Create all rows with new UUIDs, remapping FK references."""
        id_map: dict[str, str] = {}  # old_id_str -> new_id_str

        # ── Workspace ─────────────────────────────────────────────────────────
        old_ws_id = ws_data.get("id", "")
        new_ws_id = str(uuid.uuid4())
        id_map[old_ws_id] = new_ws_id

        base_name: str = ws_data.get("name", "Imported Workspace")
        import_name = self._unique_name(base_name)

        from src.data.sql.models.workspace import WorkspaceStatus
        from src.data.sql.repositories import get_workspace_repository
        workspace = get_workspace_repository().create_workspace(
            name=import_name,
            description=ws_data.get("description"),
            created_by=ws_data.get("created_by"),
        )
        id_map[old_ws_id] = str(workspace.id)

        # ── Jobs ──────────────────────────────────────────────────────────────
        from src.data.sql.models.job_record import JobRecord, JobStatus
        for jd in jobs_data:
            old_job_id = jd.get("id", "")
            new_job_id = str(uuid.uuid4())
            id_map[old_job_id] = new_job_id

            job = JobRecord(
                id=uuid.UUID(new_job_id),
                workspace_id=workspace.id,
                mongo_doc_id=jd.get("mongo_doc_id"),
                title=jd.get("title", ""),
                company_name=jd.get("company_name"),
                description_snippet=jd.get("description_snippet"),
                status=JobStatus(jd["status"]) if jd.get("status") else JobStatus.OPEN,
            )
            self._job_repo.create(job)

        # ── Matches ───────────────────────────────────────────────────────────
        from src.data.sql.models.match_record import MatchRecord
        for md in matches_data:
            old_match_id = md.get("id", "")
            new_match_id = str(uuid.uuid4())
            id_map[old_match_id] = new_match_id

            old_job_id = md.get("job_id", "")
            new_job_id_for_match = id_map.get(old_job_id, old_job_id)

            match = MatchRecord(
                id=uuid.UUID(new_match_id),
                job_id=uuid.UUID(new_job_id_for_match),
                candidate_mongo_id=md.get("candidate_mongo_id", ""),
                overall_score=float(md.get("overall_score", 0.0)),
                skills_score=_opt_float(md.get("skills_score")),
                experience_score=_opt_float(md.get("experience_score")),
                education_score=_opt_float(md.get("education_score")),
                semantic_score=_opt_float(md.get("semantic_score")),
                keyword_score=_opt_float(md.get("keyword_score")),
                score_level=md.get("score_level"),
                manual_score_override=_opt_float(md.get("manual_score_override")),
                override_reason=md.get("override_reason"),
                overridden_by=md.get("overridden_by"),
                scoring_model_version=md.get("scoring_model_version"),
            )
            self._match_repo.create(match)

        # ── Audit logs ────────────────────────────────────────────────────────
        for ad in audits_data:
            old_match_id = ad.get("related_match_id")
            old_job_id_a = ad.get("related_job_id")

            new_related_match = (
                uuid.UUID(id_map[old_match_id])
                if old_match_id and old_match_id in id_map
                else None
            )
            new_related_job = (
                uuid.UUID(id_map[old_job_id_a])
                if old_job_id_a and old_job_id_a in id_map
                else None
            )

            self._audit_repo.log(
                action=ad.get("action", "imported"),
                action_description=ad.get("action_description", "Imported record"),
                workspace_id=workspace.id,
                actor=ad.get("actor"),
                resource=ad.get("resource"),
                changes=ad.get("changes"),
                ai_decision=ad.get("ai_decision"),
                bias_audit=ad.get("bias_audit"),
                context=ad.get("context"),
                related_candidate_mongo_id=ad.get("related_candidate_mongo_id"),
                related_job_id=new_related_job,
                related_match_id=new_related_match,
                compliance_relevant=bool(ad.get("compliance_relevant", False)),
                session_id=ad.get("session_id"),
                request_id=ad.get("request_id"),
            )

        return workspace

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _unique_name(self, base_name: str) -> str:
        """Append date suffix until the name is not taken."""
        candidate = f"{base_name} (Imported {datetime.now(timezone.utc).strftime('%Y-%m-%d')})"
        existing = self._ws_repo.find_by_name(candidate)
        if existing is None:
            return candidate
        # Last resort: add a short UUID fragment
        return f"{candidate} {uuid.uuid4().hex[:6]}"


def _opt_float(val: Any) -> Optional[float]:
    return float(val) if val is not None else None


_import_service: Optional[WorkspaceImportService] = None


def get_import_service() -> WorkspaceImportService:
    global _import_service
    if _import_service is None:
        _import_service = WorkspaceImportService()
    return _import_service
