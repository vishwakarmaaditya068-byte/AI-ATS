from __future__ import annotations

from typing import Any, Optional

import pytest

from src.services.audit_service import AuditService
from src.utils.constants import AuditAction


class FakeAuditRepo:
    """In-memory stand-in for AuditRecordRepository."""

    def __init__(self, *, raise_on_log: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.raise_on_log: bool = raise_on_log

    def log(self, **kwargs: Any) -> Optional[object]:
        if self.raise_on_log:
            raise RuntimeError("boom")
        self.calls.append(kwargs)

        class _Record:
            action = kwargs.get("action")

        return _Record()


@pytest.fixture
def fake_repo() -> FakeAuditRepo:
    return FakeAuditRepo()


@pytest.fixture
def service(fake_repo: FakeAuditRepo) -> AuditService:
    # Type ignore — Fake is structural-compatible for the public surface used.
    return AuditService(repo=fake_repo)  # type: ignore[arg-type]


class TestCandidateAdded:
    def test_emits_correct_action(self, service: AuditService, fake_repo: FakeAuditRepo) -> None:
        service.emit_candidate_added(
            candidate_mongo_id="507f1f77bcf86cd799439011",
            candidate_name="Ada Lovelace",
            source="ada.pdf",
        )
        assert len(fake_repo.calls) == 1
        call = fake_repo.calls[0]
        assert call["action"] == AuditAction.CANDIDATE_ADDED.value

    def test_resource_payload_includes_candidate_metadata(
        self, service: AuditService, fake_repo: FakeAuditRepo
    ) -> None:
        service.emit_candidate_added(
            candidate_mongo_id="507f1f77bcf86cd799439011",
            candidate_name="Ada Lovelace",
        )
        resource = fake_repo.calls[0]["resource"]
        assert resource["resource_type"] == "candidate"
        assert resource["resource_id"] == "507f1f77bcf86cd799439011"
        assert resource["resource_name"] == "Ada Lovelace"


class TestCandidateScored:
    def test_marks_compliance_relevant(
        self, service: AuditService, fake_repo: FakeAuditRepo
    ) -> None:
        service.emit_candidate_scored(
            workspace_id="00000000-0000-0000-0000-000000000001",
            job_id="00000000-0000-0000-0000-000000000002",
            match_id="00000000-0000-0000-0000-000000000003",
            candidate_mongo_id="507f1f77bcf86cd799439011",
            candidate_name="Ada",
            job_title="Senior Dev",
            score=0.82,
        )
        assert fake_repo.calls[0]["compliance_relevant"] is True

    def test_ai_decision_carries_score_and_model(
        self, service: AuditService, fake_repo: FakeAuditRepo
    ) -> None:
        service.emit_candidate_scored(
            workspace_id="00000000-0000-0000-0000-000000000001",
            job_id="00000000-0000-0000-0000-000000000002",
            match_id="00000000-0000-0000-0000-000000000003",
            candidate_mongo_id="507f1f77bcf86cd799439011",
            candidate_name="Ada",
            job_title="Senior Dev",
            score=0.8234,
            model_version="2.1",
        )
        ai = fake_repo.calls[0]["ai_decision"]
        assert ai["model_version"] == "2.1"
        assert ai["confidence_score"] == pytest.approx(0.8234)


class TestManualOverride:
    def test_records_old_and_new_score_in_changes(
        self, service: AuditService, fake_repo: FakeAuditRepo
    ) -> None:
        service.emit_manual_override(
            workspace_id="00000000-0000-0000-0000-000000000001",
            job_id="00000000-0000-0000-0000-000000000002",
            match_id="00000000-0000-0000-0000-000000000003",
            candidate_mongo_id="507f1f77bcf86cd799439011",
            candidate_name="Ada",
            original_score=0.75,
            new_score=0.90,
            reason="Domain expertise not captured by AI",
            actor_id="alice",
            actor_name="Alice Recruiter",
        )
        changes = fake_repo.calls[0]["changes"]
        assert len(changes) == 1
        assert changes[0]["field_name"] == "score"
        assert changes[0]["old_value"] == pytest.approx(0.75)
        assert changes[0]["new_value"] == pytest.approx(0.90)

    def test_actor_uses_user_type_when_id_supplied(
        self, service: AuditService, fake_repo: FakeAuditRepo
    ) -> None:
        service.emit_manual_override(
            workspace_id="00000000-0000-0000-0000-000000000001",
            job_id="00000000-0000-0000-0000-000000000002",
            match_id="00000000-0000-0000-0000-000000000003",
            candidate_mongo_id="507f1f77bcf86cd799439011",
            candidate_name="Ada",
            original_score=0.75,
            new_score=0.90,
            reason="Domain expertise not captured by AI",
            actor_id="alice",
        )
        actor = fake_repo.calls[0]["actor"]
        assert actor["actor_type"] == "user"
        assert actor["actor_id"] == "alice"


class TestBiasDetected:
    def test_bias_audit_payload_shape(
        self, service: AuditService, fake_repo: FakeAuditRepo
    ) -> None:
        service.emit_bias_detected(
            workspace_id="00000000-0000-0000-0000-000000000001",
            bias_types=["gender", "age"],
            detection_method="protected_attribute_detector",
            affected_count=3,
            fairness_metrics={"disparate_impact": 0.72},
            match_id="00000000-0000-0000-0000-000000000003",
        )
        call = fake_repo.calls[0]
        assert call["action"] == AuditAction.BIAS_DETECTED.value
        assert call["bias_audit"]["affected_candidates"] == 3
        assert call["bias_audit"]["fairness_metrics"]["disparate_impact"] == pytest.approx(0.72)
        assert call["compliance_relevant"] is True


class TestFailSoftContract:
    def test_repo_exception_does_not_propagate(self) -> None:
        bad_repo = FakeAuditRepo(raise_on_log=True)
        svc = AuditService(repo=bad_repo)  # type: ignore[arg-type]
        # Should return None, not raise.
        result = svc.emit_candidate_added(
            candidate_mongo_id="507f1f77bcf86cd799439011",
            candidate_name="Ada",
        )
        assert result is None
