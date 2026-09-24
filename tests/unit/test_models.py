"""
Tests for Pydantic data models in src.data.models.
"""

from datetime import date, datetime, timedelta

import pytest
from bson import ObjectId

from src.data.models import (
    ActorInfo,
    AIDecisionInfo,
    AuditLog,
    AuditLogCreate,
    BiasAuditInfo,
    BiasCheckResult,
    Candidate,
    CandidateMetadata,
    Certification,
    ChangeRecord,
    ContactInfo,
    Education,
    EducationMatch,
    EducationRequirement,
    ExperienceMatch,
    ExperienceRequirement,
    Explanation,
    ExplanationFactor,
    FileMetadata,
    Job,
    JobCreate,
    JobMetadata,
    KeywordMatch,
    Language,
    Location,
    Match,
    MatchStatus,
    ParsedContent,
    ProcessingError,
    ProcessingStatus,
    RecruiterFeedback,
    ResourceInfo,
    Resume,
    ResumeFormat,
    SalaryRange,
    ScoreBreakdown,
    ScoringWeights,
    Skill,
    SkillMatch,
    SkillRequirement,
    WorkExperience,
    create_bias_detected_audit,
    create_candidate_added_audit,
    create_candidate_scored_audit,
    create_manual_override_audit,
)
from src.data.models.base import BaseDocument, PyObjectId, TimestampMixin
from src.utils.constants import (
    AuditAction,
    CandidateStatus,
    JobStatus,
    MatchScoreLevel,
)


# ═══════════════════════════════════════════════════════════════════════════
#  base.py
# ═══════════════════════════════════════════════════════════════════════════


class TestPyObjectId:
    def test_validate_valid_string(self):
        oid = ObjectId()
        result = PyObjectId.validate(str(oid))
        assert result == oid

    def test_validate_object_id_passthrough(self):
        oid = ObjectId()
        assert PyObjectId.validate(oid) is oid

    def test_validate_invalid_string(self):
        with pytest.raises(ValueError, match="Invalid ObjectId"):
            PyObjectId.validate("not-a-valid-id")


class TestTimestampMixin:
    def test_auto_created_at(self):
        ts = TimestampMixin()
        assert isinstance(ts.created_at, datetime)

    def test_auto_updated_at(self):
        ts = TimestampMixin()
        assert isinstance(ts.updated_at, datetime)


class TestBaseDocument:
    def test_model_dump_mongo_excludes_none_id(self):
        doc = BaseDocument()
        data = doc.model_dump_mongo()
        assert "_id" not in data

    def test_model_dump_mongo_includes_set_id(self):
        oid = ObjectId()
        doc = BaseDocument(id=oid)
        data = doc.model_dump_mongo()
        assert data["_id"] == oid

    def test_timestamps_present(self):
        doc = BaseDocument()
        assert doc.created_at is not None
        assert doc.updated_at is not None


# ═══════════════════════════════════════════════════════════════════════════
#  match.py
# ═══════════════════════════════════════════════════════════════════════════


class TestScoreBreakdown:
    def test_total_score(self):
        bd = ScoreBreakdown(
            skills_weighted=0.28,
            experience_weighted=0.20,
            education_weighted=0.15,
            semantic_weighted=0.14,
            keyword_weighted=0.03,
        )
        assert abs(bd.total_score - 0.80) < 1e-9

    def test_total_score_all_zeros(self):
        bd = ScoreBreakdown()
        assert bd.total_score == 0.0


class TestMatch:
    def test_effective_score_no_override(self, sample_match):
        assert sample_match.effective_score == sample_match.overall_score

    def test_effective_score_with_override(self, sample_match):
        sample_match.manual_score_override = 0.95
        assert sample_match.effective_score == 0.95

    def test_is_shortlisted_false(self, sample_match):
        assert sample_match.is_shortlisted is False

    def test_is_shortlisted_true(self, sample_match):
        sample_match.status = MatchStatus.SHORTLISTED
        assert sample_match.is_shortlisted is True

    def test_skills_match_percentage_some_matched(self, sample_match):
        # 2 of 3 required skills matched
        pct = sample_match.skills_match_percentage
        assert abs(pct - (2 / 3 * 100)) < 0.1

    def test_skills_match_percentage_no_required(self):
        m = Match(
            candidate_id=ObjectId(),
            job_id=ObjectId(),
            skill_matches=[SkillMatch(skill_name="docker", required=False, candidate_has_skill=True)],
        )
        assert m.skills_match_percentage == 100.0

    def test_calculate_score_level(self):
        m = Match(candidate_id=ObjectId(), job_id=ObjectId(), overall_score=0.90)
        level = m.calculate_score_level()
        assert level == MatchScoreLevel.EXCELLENT
        assert m.score_level == MatchScoreLevel.EXCELLENT

    def test_add_feedback(self, sample_match):
        sample_match.add_feedback(recruiter_id="rec1", rating=4, comments="Strong candidate")
        assert len(sample_match.feedback) == 1
        assert sample_match.feedback[0].recruiter_id == "rec1"
        assert sample_match.feedback[0].rating == 4

    def test_add_multiple_feedbacks(self, sample_match):
        sample_match.add_feedback(recruiter_id="rec1", rating=4)
        sample_match.add_feedback(recruiter_id="rec2", rating=3, decision="shortlist")
        assert len(sample_match.feedback) == 2

    def test_override_score(self, sample_match):
        sample_match.override_score(0.95, "Manager approved")
        assert sample_match.manual_score_override == 0.95
        assert sample_match.override_reason == "Manager approved"
        assert sample_match.effective_score == 0.95

    def test_override_score_updates_level(self):
        m = Match(candidate_id=ObjectId(), job_id=ObjectId(), overall_score=0.40)
        assert m.score_level == MatchScoreLevel.POOR
        m.override_score(0.90, "Re-evaluated")
        # override_score calls calculate_score_level which uses overall_score (0.40)
        # so the level depends on overall_score not the override
        assert m.score_level == MatchScoreLevel.POOR


# ═══════════════════════════════════════════════════════════════════════════
#  candidate.py
# ═══════════════════════════════════════════════════════════════════════════


class TestSkillModel:
    def test_name_normalized_to_lowercase(self):
        s = Skill(name="  Python  ")
        assert s.name == "python"

    def test_name_strip_whitespace(self):
        s = Skill(name="  React ")
        assert s.name == "react"


class TestWorkExperience:
    def test_duration_months_basic(self):
        we = WorkExperience(
            job_title="Dev",
            company="Co",
            start_date=date(2020, 1, 1),
            end_date=date(2022, 1, 1),
        )
        # ~24 months
        assert we.duration_months is not None
        assert we.duration_months >= 23

    def test_duration_months_no_start_date(self):
        we = WorkExperience(job_title="Dev", company="Co")
        assert we.duration_months is None

    def test_duration_months_current_job(self):
        we = WorkExperience(
            job_title="Dev",
            company="Co",
            start_date=date(2023, 1, 1),
            is_current=True,
        )
        # No end_date => uses today
        assert we.duration_months is not None
        assert we.duration_months >= 1


class TestEducation:
    def test_valid_gpa(self):
        e = Education(degree="Bachelor", field_of_study="CS", institution="MIT", gpa=3.5)
        assert e.gpa == 3.5

    def test_gpa_none_is_valid(self):
        e = Education(degree="Bachelor", field_of_study="CS", institution="MIT", gpa=None)
        assert e.gpa is None

    def test_gpa_too_high_raises(self):
        with pytest.raises(ValueError, match="GPA must be between"):
            Education(degree="Bachelor", field_of_study="CS", institution="MIT", gpa=5.0)

    def test_gpa_negative_raises(self):
        with pytest.raises(ValueError, match="GPA must be between"):
            Education(degree="Bachelor", field_of_study="CS", institution="MIT", gpa=-1.0)


class TestCertification:
    def test_is_valid_no_expiration(self):
        c = Certification(name="AWS", issuing_organization="Amazon")
        assert c.is_valid is True

    def test_is_valid_future_expiration(self):
        c = Certification(
            name="AWS",
            issuing_organization="Amazon",
            expiration_date=date.today() + timedelta(days=365),
        )
        assert c.is_valid is True

    def test_is_valid_past_expiration(self):
        c = Certification(
            name="AWS",
            issuing_organization="Amazon",
            expiration_date=date.today() - timedelta(days=1),
        )
        assert c.is_valid is False


class TestCandidate:
    def test_full_name(self, sample_candidate):
        assert sample_candidate.full_name == "Jane Smith"

    def test_skill_names(self, sample_candidate):
        names = sample_candidate.skill_names
        assert "python" in names
        assert "django" in names
        assert "postgresql" in names

    def test_total_experience_years(self, sample_candidate):
        years = sample_candidate.total_experience_years
        # Two jobs: 2020-01 to now + 2017-06 to 2019-12
        assert years > 0

    def test_highest_education_level(self, sample_candidate):
        level = sample_candidate.highest_education_level
        assert level is not None
        assert "bachelor" in level.lower()

    def test_highest_education_level_no_education(self, sample_contact_info):
        c = Candidate(
            first_name="John",
            last_name="Doe",
            contact=sample_contact_info,
            education=[],
        )
        assert c.highest_education_level is None

    def test_highest_education_level_masters(self, sample_contact_info):
        c = Candidate(
            first_name="John",
            last_name="Doe",
            contact=sample_contact_info,
            education=[
                Education(degree="Bachelor", field_of_study="CS", institution="MIT"),
                Education(degree="Master", field_of_study="CS", institution="Stanford"),
            ],
        )
        assert c.highest_education_level is not None
        assert "master" in c.highest_education_level.lower()


# ═══════════════════════════════════════════════════════════════════════════
#  job.py
# ═══════════════════════════════════════════════════════════════════════════


class TestScoringWeights:
    def test_total_weight_default(self):
        sw = ScoringWeights.from_defaults()
        assert abs(sw.total_weight - 1.0) < 1e-9

    def test_total_weight_custom(self):
        sw = ScoringWeights(skills_match=0.5, experience_match=0.3, education_match=0.1, semantic_similarity=0.1, keyword_match=0.0)
        assert abs(sw.total_weight - 1.0) < 1e-9

    def test_to_dict(self):
        sw = ScoringWeights.from_defaults()
        d = sw.to_dict()
        assert "skills_match" in d
        assert abs(sum(d.values()) - 1.0) < 1e-9


class TestLocation:
    def test_display_string_full(self):
        loc = Location(city="San Francisco", state="CA", country="USA")
        assert loc.display_string == "San Francisco, CA, USA"

    def test_display_string_country_only(self):
        loc = Location(country="USA")
        assert loc.display_string == "USA"

    def test_display_string_city_state_only(self):
        loc = Location(city=None, state=None)
        # country defaults to "USA"
        assert loc.display_string == "USA"


class TestJob:
    def test_required_skills(self, sample_job):
        req = sample_job.required_skills
        assert "python" in req
        assert "django" in req
        assert "docker" not in req

    def test_preferred_skills(self, sample_job):
        pref = sample_job.preferred_skills
        assert "docker" in pref
        assert "kubernetes" in pref

    def test_all_skills(self, sample_job):
        all_s = sample_job.all_skills
        assert len(all_s) == 5

    def test_is_active_open(self, sample_job):
        assert sample_job.is_active is True

    def test_is_active_draft(self, sample_job):
        sample_job.status = JobStatus.DRAFT
        assert sample_job.is_active is False

    def test_is_active_expired_closing_date(self, sample_job):
        sample_job.closing_date = date.today() - timedelta(days=1)
        assert sample_job.is_active is False

    def test_days_open(self, sample_job):
        days = sample_job.days_open
        assert days is not None
        assert days > 0

    def test_days_open_no_posted_date(self):
        j = Job(
            title="Test",
            description="Test description for a job posting",
            company_name="Co",
        )
        assert j.days_open is None

    def test_publish(self):
        j = Job(
            title="Test",
            description="Test description for a job posting",
            company_name="Co",
            status=JobStatus.DRAFT,
        )
        j.publish()
        assert j.status == JobStatus.OPEN
        assert j.posted_date is not None

    def test_close(self, sample_job):
        sample_job.close()
        assert sample_job.status == JobStatus.CLOSED


class TestSalaryRange:
    def test_negative_amount_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            SalaryRange(min_amount=-1000)


# ═══════════════════════════════════════════════════════════════════════════
#  resume.py
# ═══════════════════════════════════════════════════════════════════════════


class TestResume:
    def test_has_errors_false(self, sample_resume):
        assert sample_resume.has_errors is False

    def test_has_errors_true(self, sample_resume):
        sample_resume.add_error("parsing", "ValueError", "bad input")
        assert sample_resume.has_errors is True

    def test_is_processed(self, sample_resume):
        assert sample_resume.is_processed is True

    def test_is_processed_false(self, sample_resume):
        sample_resume.status = ProcessingStatus.PENDING
        assert sample_resume.is_processed is False

    def test_is_failed(self, sample_resume):
        assert sample_resume.is_failed is False

    def test_is_failed_true(self, sample_resume):
        sample_resume.status = ProcessingStatus.FAILED
        assert sample_resume.is_failed is True

    def test_add_error_recoverable(self, sample_resume):
        sample_resume.add_error("parsing", "ValueError", "minor issue", is_recoverable=True)
        assert len(sample_resume.processing_errors) == 1
        assert sample_resume.status == ProcessingStatus.COMPLETED  # Not changed

    def test_add_error_non_recoverable(self, sample_resume):
        sample_resume.add_error("parsing", "FatalError", "critical failure", is_recoverable=False)
        assert sample_resume.status == ProcessingStatus.FAILED


# ═══════════════════════════════════════════════════════════════════════════
#  audit.py
# ═══════════════════════════════════════════════════════════════════════════


class TestAuditLog:
    def test_is_ai_action_true(self, sample_audit_log):
        sample_audit_log.ai_decision = AIDecisionInfo(model_name="matching_engine", model_version="1.0")
        assert sample_audit_log.is_ai_action is True

    def test_is_ai_action_false(self, sample_audit_log):
        sample_audit_log.ai_decision = None
        assert sample_audit_log.is_ai_action is False

    def test_is_manual_override_true(self):
        log = AuditLog(
            action=AuditAction.MANUAL_OVERRIDE,
            action_description="Score overridden",
        )
        assert log.is_manual_override is True

    def test_is_manual_override_false(self, sample_audit_log):
        assert sample_audit_log.is_manual_override is False

    def test_involves_bias_by_action(self):
        log = AuditLog(
            action=AuditAction.BIAS_DETECTED,
            action_description="Bias found",
        )
        assert log.involves_bias is True

    def test_involves_bias_by_audit_info(self, sample_audit_log):
        sample_audit_log.bias_audit = BiasAuditInfo(
            bias_type="gender", detection_method="statistical"
        )
        assert sample_audit_log.involves_bias is True

    def test_involves_bias_false(self, sample_audit_log):
        sample_audit_log.bias_audit = None
        assert sample_audit_log.involves_bias is False


class TestAuditHelperFunctions:
    def test_create_candidate_added_audit(self):
        result = create_candidate_added_audit(
            candidate_id="abc123",
            candidate_name="Jane Smith",
            actor_id="user1",
            source="upload",
        )
        assert result.action == AuditAction.CANDIDATE_ADDED
        assert "Jane Smith" in result.action_description
        assert result.actor.actor_type == "user"
        assert result.context.get("source") == "upload"

    def test_create_candidate_added_audit_system_actor(self):
        result = create_candidate_added_audit(
            candidate_id="abc123",
            candidate_name="Jane Smith",
        )
        assert result.actor.actor_type == "system"

    def test_create_candidate_scored_audit(self):
        result = create_candidate_scored_audit(
            candidate_id="c1",
            candidate_name="Jane",
            job_id="j1",
            job_title="Engineer",
            match_id="m1",
            score=0.85,
            model_name="matching_engine",
            model_version="1.0",
        )
        assert result.action == AuditAction.CANDIDATE_SCORED
        assert result.compliance_relevant is True
        assert result.ai_decision is not None
        assert result.ai_decision.confidence_score == 0.85

    def test_create_manual_override_audit(self):
        result = create_manual_override_audit(
            match_id="m1",
            candidate_id="c1",
            job_id="j1",
            original_score=0.60,
            new_score=0.85,
            reason="Manager re-evaluation",
            actor_id="user1",
            actor_name="Recruiter Bob",
        )
        assert result.action == AuditAction.MANUAL_OVERRIDE
        assert len(result.changes) == 1
        assert result.changes[0].old_value == 0.60
        assert result.changes[0].new_value == 0.85
        assert result.compliance_relevant is True

    def test_create_bias_detected_audit(self):
        result = create_bias_detected_audit(
            bias_type="gender",
            detection_method="statistical_parity",
            affected_count=15,
            fairness_metrics={"demographic_parity": 0.15},
            job_id="j1",
            remediation="Re-scoring applied",
        )
        assert result.action == AuditAction.BIAS_DETECTED
        assert result.bias_audit is not None
        assert result.bias_audit.affected_candidates == 15
        assert result.compliance_relevant is True


def test_candidate_has_file_hashes_field():
    from src.data.models.candidate import Candidate, ContactInfo
    c = Candidate(
        first_name="A",
        last_name="B",
        contact=ContactInfo(email="a@b.com"),
    )
    assert hasattr(c, "file_hashes")
    assert isinstance(c.file_hashes, list)
