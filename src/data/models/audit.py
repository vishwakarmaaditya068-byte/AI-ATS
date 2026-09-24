"""
Audit log data models for AI-ATS.

Defines the schema for audit logging to support compliance,
transparency, and system monitoring.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.utils.constants import AuditAction

from .base import BaseDocument, EmbeddedModel, PyObjectId


class ActorInfo(EmbeddedModel):
    """Information about the entity that performed the action."""

    actor_type: str = "system"  # "user", "system", "automated"
    actor_id: Optional[str] = None  # User ID if applicable
    actor_name: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class ResourceInfo(EmbeddedModel):
    """Information about the resource affected by the action."""

    resource_type: str  # "candidate", "job", "match", "resume"
    resource_id: str
    resource_name: Optional[str] = None  # Human-readable identifier


class ChangeRecord(EmbeddedModel):
    """Record of a specific field change."""

    field_name: str
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    change_type: str = "update"  # "create", "update", "delete"


class AIDecisionInfo(EmbeddedModel):
    """Information about AI-driven decisions for explainability."""

    model_name: str
    model_version: str
    input_summary: Optional[str] = None  # Summary of inputs (avoid storing PII)
    output_summary: Optional[str] = None
    confidence_score: Optional[float] = None
    explanation: Optional[str] = None
    factors: list[dict[str, Any]] = Field(default_factory=list)


class BiasAuditInfo(EmbeddedModel):
    """Information specific to bias detection events."""

    bias_type: str
    detection_method: str
    affected_candidates: int = 0
    fairness_metrics: dict[str, float] = Field(default_factory=dict)
    remediation_action: Optional[str] = None


class AuditLog(BaseDocument):
    """
    Main audit log document for compliance and transparency.

    Tracks all significant actions in the system including:
    - Candidate lifecycle events
    - Scoring and ranking decisions
    - Manual overrides
    - Bias detection alerts
    - Report generation
    """

    # Action Details
    action: AuditAction
    action_description: str  # Human-readable description

    # Actor Information
    actor: ActorInfo = Field(default_factory=ActorInfo)

    # Resource Information
    resource: Optional[ResourceInfo] = None

    # Changes Made
    changes: list[ChangeRecord] = Field(default_factory=list)

    # AI Decision Context (for AI-driven actions)
    ai_decision: Optional[AIDecisionInfo] = None

    # Bias Detection Context
    bias_audit: Optional[BiasAuditInfo] = None

    # Additional Context
    context: dict[str, Any] = Field(default_factory=dict)

    # Related Entities
    related_candidate_id: Optional[PyObjectId] = None
    related_job_id: Optional[PyObjectId] = None
    related_match_id: Optional[PyObjectId] = None

    # Compliance Fields
    compliance_relevant: bool = False  # Flag for compliance-critical events
    retention_period_days: int = 2555  # ~7 years default for compliance

    # Session Tracking
    session_id: Optional[str] = None
    request_id: Optional[str] = None

    @property
    def is_ai_action(self) -> bool:
        """Check if this was an AI-driven action."""
        return self.ai_decision is not None

    @property
    def is_manual_override(self) -> bool:
        """Check if this was a manual override action."""
        return self.action == AuditAction.MANUAL_OVERRIDE

    @property
    def involves_bias(self) -> bool:
        """Check if this audit involves bias detection."""
        return self.action == AuditAction.BIAS_DETECTED or self.bias_audit is not None

    class Settings:
        """MongoDB collection settings."""

        name = "audit_logs"
        indexes = [
            "action",
            "actor.actor_id",
            "resource.resource_type",
            "resource.resource_id",
            "related_candidate_id",
            "related_job_id",
            "compliance_relevant",
            "created_at",
        ]


class AuditLogCreate(BaseModel):
    """Schema for creating a new audit log entry."""

    action: AuditAction
    action_description: str
    actor: Optional[ActorInfo] = None
    resource: Optional[ResourceInfo] = None
    changes: list[ChangeRecord] = Field(default_factory=list)
    ai_decision: Optional[AIDecisionInfo] = None
    bias_audit: Optional[BiasAuditInfo] = None
    context: dict[str, Any] = Field(default_factory=dict)
    related_candidate_id: Optional[str] = None
    related_job_id: Optional[str] = None
    related_match_id: Optional[str] = None
    compliance_relevant: bool = False


class AuditLogQuery(BaseModel):
    """Query parameters for searching audit logs."""

    action: Optional[AuditAction] = None
    actor_id: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    candidate_id: Optional[str] = None
    job_id: Optional[str] = None
    compliance_only: bool = False
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = Field(default=100, le=1000)
    offset: int = 0


class AuditSummary(BaseModel):
    """Summary statistics for audit logs."""

    total_actions: int = 0
    actions_by_type: dict[str, int] = Field(default_factory=dict)
    ai_decisions_count: int = 0
    manual_overrides_count: int = 0
    bias_detections_count: int = 0
    compliance_events_count: int = 0
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None


# Utility functions for creating common audit entries

def create_candidate_added_audit(
    candidate_id: str,
    candidate_name: str,
    actor_id: Optional[str] = None,
    source: Optional[str] = None,
) -> AuditLogCreate:
    """Create an audit entry for a new candidate."""
    return AuditLogCreate(
        action=AuditAction.CANDIDATE_ADDED,
        action_description=f"Candidate '{candidate_name}' was added to the system",
        actor=ActorInfo(
            actor_type="user" if actor_id else "system",
            actor_id=actor_id,
        ),
        resource=ResourceInfo(
            resource_type="candidate",
            resource_id=candidate_id,
            resource_name=candidate_name,
        ),
        context={"source": source} if source else {},
        related_candidate_id=candidate_id,
    )


def create_candidate_scored_audit(
    candidate_id: str,
    candidate_name: str,
    job_id: str,
    job_title: str,
    match_id: str,
    score: float,
    model_name: str,
    model_version: str,
    explanation: Optional[str] = None,
) -> AuditLogCreate:
    """Create an audit entry for candidate scoring."""
    return AuditLogCreate(
        action=AuditAction.CANDIDATE_SCORED,
        action_description=f"Candidate '{candidate_name}' scored {score:.2f} for job '{job_title}'",
        actor=ActorInfo(actor_type="system"),
        resource=ResourceInfo(
            resource_type="match",
            resource_id=match_id,
        ),
        ai_decision=AIDecisionInfo(
            model_name=model_name,
            model_version=model_version,
            confidence_score=score,
            explanation=explanation,
        ),
        context={"job_title": job_title},
        related_candidate_id=candidate_id,
        related_job_id=job_id,
        related_match_id=match_id,
        compliance_relevant=True,
    )


def create_manual_override_audit(
    match_id: str,
    candidate_id: str,
    job_id: str,
    original_score: float,
    new_score: float,
    reason: str,
    actor_id: str,
    actor_name: Optional[str] = None,
) -> AuditLogCreate:
    """Create an audit entry for manual score override."""
    return AuditLogCreate(
        action=AuditAction.MANUAL_OVERRIDE,
        action_description=f"Score manually overridden from {original_score:.2f} to {new_score:.2f}",
        actor=ActorInfo(
            actor_type="user",
            actor_id=actor_id,
            actor_name=actor_name,
        ),
        resource=ResourceInfo(
            resource_type="match",
            resource_id=match_id,
        ),
        changes=[
            ChangeRecord(
                field_name="score",
                old_value=original_score,
                new_value=new_score,
                change_type="update",
            )
        ],
        context={"reason": reason},
        related_candidate_id=candidate_id,
        related_job_id=job_id,
        related_match_id=match_id,
        compliance_relevant=True,
    )


def create_bias_detected_audit(
    bias_type: str,
    detection_method: str,
    affected_count: int,
    fairness_metrics: dict[str, float],
    job_id: Optional[str] = None,
    remediation: Optional[str] = None,
) -> AuditLogCreate:
    """Create an audit entry for bias detection."""
    return AuditLogCreate(
        action=AuditAction.BIAS_DETECTED,
        action_description=f"Potential {bias_type} bias detected affecting {affected_count} candidates",
        actor=ActorInfo(actor_type="system"),
        bias_audit=BiasAuditInfo(
            bias_type=bias_type,
            detection_method=detection_method,
            affected_candidates=affected_count,
            fairness_metrics=fairness_metrics,
            remediation_action=remediation,
        ),
        related_job_id=job_id,
        compliance_relevant=True,
    )
