from __future__ import annotations

import os
import uuid
from typing import Any, Optional

from src.data.sql.models.audit_record import AuditRecord
from src.data.sql.repositories.audit_record_repo import (
    AuditRecordRepository,
    get_audit_record_repository,
)
from src.utils.constants import AuditAction
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _resolve_actor(actor_id: Optional[str], actor_name: Optional[str]) -> dict[str, Any]:
    """Build an actor payload — user if provided, else OS-login fallback."""
    if actor_id or actor_name:
        return {
            "actor_type": "user",
            "actor_id": actor_id,
            "actor_name": actor_name,
        }
    try:
        login_name = os.getlogin()
    except OSError:
        login_name = os.environ.get("USER") or "unknown"
    return {
        "actor_type": "user",
        "actor_id": login_name,
        "actor_name": login_name,
    }


class AuditService:
    """Thin, fail-soft wrapper over AuditRecordRepository."""

    def __init__(self, repo: Optional[AuditRecordRepository] = None) -> None:
        self._repo: AuditRecordRepository = repo or get_audit_record_repository()

    # Ingestion
    def emit_candidate_added(
        self,
        *,
        candidate_mongo_id: str,
        candidate_name: str,
        source: Optional[str] = None,
        workspace_id: Optional[uuid.UUID | str] = None,
        actor_id: Optional[str] = None,
        actor_name: Optional[str] = None,
    ) -> Optional[AuditRecord]:
        return self._safe_log(
            action=AuditAction.CANDIDATE_ADDED.value,
            action_description=f"Candidate '{candidate_name}' ingested",
            workspace_id=workspace_id,
            actor=_resolve_actor(actor_id, actor_name),
            resource={
                "resource_type": "candidate",
                "resource_id": candidate_mongo_id,
                "resource_name": candidate_name,
            },
            context={"source": source} if source else {},
            related_candidate_mongo_id=candidate_mongo_id,
        )

    # Matching
    def emit_candidate_scored(
        self,
        *,
        workspace_id: uuid.UUID | str,
        job_id: uuid.UUID | str,
        match_id: uuid.UUID | str,
        candidate_mongo_id: str,
        candidate_name: str,
        job_title: str,
        score: float,
        model_version: str = "1.0",
        explanation_summary: Optional[str] = None,
    ) -> Optional[AuditRecord]:
        return self._safe_log(
            action=AuditAction.CANDIDATE_SCORED.value,
            action_description=(
                f"Candidate '{candidate_name}' scored {score:.2f} for job '{job_title}'"
            ),
            workspace_id=workspace_id,
            actor={"actor_type": "system", "actor_name": "matching_engine"},
            resource={
                "resource_type": "match",
                "resource_id": str(match_id),
            },
            ai_decision={
                "model_name": "matching_engine",
                "model_version": model_version,
                "confidence_score": round(float(score), 4),
                "explanation": explanation_summary,
            },
            context={"job_title": job_title, "candidate_name": candidate_name},
            related_candidate_mongo_id=candidate_mongo_id,
            related_job_id=job_id,
            related_match_id=match_id,
            compliance_relevant=True,
        )

    # Bias detection
    def emit_bias_detected(
        self,
        *,
        workspace_id: Optional[uuid.UUID | str],
        bias_types: list[str],
        detection_method: str,
        affected_count: int,
        fairness_metrics: Optional[dict[str, float]] = None,
        job_id: Optional[uuid.UUID | str] = None,
        match_id: Optional[uuid.UUID | str] = None,
        candidate_mongo_id: Optional[str] = None,
        remediation: Optional[str] = None,
    ) -> Optional[AuditRecord]:
        bias_label = ", ".join(bias_types) if bias_types else "unspecified"
        return self._safe_log(
            action=AuditAction.BIAS_DETECTED.value,
            action_description=(
                f"Potential {bias_label} bias indicators detected "
                f"(affecting {affected_count} candidate(s))"
            ),
            workspace_id=workspace_id,
            actor={"actor_type": "system", "actor_name": "bias_detector"},
            bias_audit={
                "bias_type": bias_label,
                "detection_method": detection_method,
                "affected_candidates": affected_count,
                "fairness_metrics": fairness_metrics or {},
                "remediation_action": remediation,
            },
            related_candidate_mongo_id=candidate_mongo_id,
            related_job_id=job_id,
            related_match_id=match_id,
            compliance_relevant=True,
        )

    # Manual override
    def emit_manual_override(
        self,
        *,
        workspace_id: uuid.UUID | str,
        job_id: uuid.UUID | str,
        match_id: uuid.UUID | str,
        candidate_mongo_id: str,
        candidate_name: str,
        original_score: float,
        new_score: float,
        reason: str,
        actor_id: Optional[str] = None,
        actor_name: Optional[str] = None,
    ) -> Optional[AuditRecord]:
        return self._safe_log(
            action=AuditAction.MANUAL_OVERRIDE.value,
            action_description=(
                f"Score for '{candidate_name}' overridden "
                f"from {original_score:.2f} to {new_score:.2f}"
            ),
            workspace_id=workspace_id,
            actor=_resolve_actor(actor_id, actor_name),
            resource={
                "resource_type": "match",
                "resource_id": str(match_id),
                "resource_name": candidate_name,
            },
            changes=[
                {
                    "field_name": "score",
                    "old_value": round(float(original_score), 4),
                    "new_value": round(float(new_score), 4),
                    "change_type": "update",
                }
            ],
            context={"reason": reason, "candidate_name": candidate_name},
            related_candidate_mongo_id=candidate_mongo_id,
            related_job_id=job_id,
            related_match_id=match_id,
            compliance_relevant=True,
        )

    def emit_workspace_event(
        self,
        *,
        workspace_id: uuid.UUID | str,
        action: str,
        description: str,
        actor_id: Optional[str] = None,
        actor_name: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> Optional[AuditRecord]:
        return self._safe_log(
            action=action,
            action_description=description,
            workspace_id=workspace_id,
            actor=_resolve_actor(actor_id, actor_name),
            context=context or {},
            compliance_relevant=True,
        )

    # Internal
    def _safe_log(self, **kwargs: Any) -> Optional[AuditRecord]:
        """Catch all repo errors — audit must never block the pipeline."""
        try:
            return self._repo.log(**kwargs)
        except Exception as exc:
            logger.warning(f"Audit emit failed (action={kwargs.get('action')!r}): {exc}")
            return None


_audit_service: Optional[AuditService] = None


def get_audit_service() -> AuditService:
    """Process-wide AuditService singleton."""
    global _audit_service
    if _audit_service is None:
        _audit_service = AuditService()
    return _audit_service
