import os
os.environ.setdefault("APP_ENVIRONMENT", "testing")
os.environ.setdefault("DB_NAME", "ai_ats_test")

import pytest

from src.ml.ethics.protected_attributes import ProtectedAttributeDetector


def _detector() -> ProtectedAttributeDetector:
    return ProtectedAttributeDetector()


# ── "partner" false-positive suppression ──────────────────────────────────────

def test_domestic_partner_detected_despite_business_context() -> None:
    """'domestic partner' must not be suppressed even when 'business' appears nearby."""
    text = "I supported my domestic partner through a career transition in the business sector."
    result = _detector().detect(text)

    attr_types = result.attribute_types_found
    assert "marital_status" in attr_types, (
        "'domestic partner' should be detected as marital_status regardless of nearby 'business'"
    )


def test_life_partner_always_detected() -> None:
    """'life partner' is an unambiguous family-status indicator and must always be flagged."""
    text = "I relocated with my life partner to accept a new role."
    result = _detector().detect(text)

    assert "marital_status" in result.attribute_types_found


def test_civil_partner_always_detected() -> None:
    """'civil partner' is an unambiguous family-status indicator and must always be flagged."""
    text = "Married with a civil partner and two children."
    result = _detector().detect(text)

    assert "marital_status" in result.attribute_types_found


def test_technology_partner_not_flagged_as_family_status() -> None:
    """'technology partner' is a business role, not a family relationship."""
    text = "Worked as a technology partner managing enterprise integrations."
    result = _detector().detect(text)

    assert "marital_status" not in result.attribute_types_found, (
        "'technology partner' should not trigger marital_status detection"
    )


def test_business_partner_not_flagged_as_family_status() -> None:
    """'business partner' is a professional title, not a family indicator."""
    text = "Served as senior business partner for EMEA region."
    result = _detector().detect(text)

    assert "marital_status" not in result.attribute_types_found, (
        "'business partner' should not trigger marital_status detection"
    )


# ── existing suppressions still work ──────────────────────────────────────────

def test_foreign_key_not_flagged_as_nationality() -> None:
    """'foreign key' is a database term and must not flag nationality."""
    text = "Designed schema with foreign key constraints for referential integrity."
    result = _detector().detect(text)

    assert "nationality" not in result.attribute_types_found


def test_single_threaded_not_flagged_as_family_status() -> None:
    """'single' in 'single-threaded' must not trigger family-status detection."""
    text = "Optimised single-threaded event loop to handle 10k concurrent connections."
    result = _detector().detect(text)

    assert "marital_status" not in result.attribute_types_found


def test_blind_review_not_flagged_as_disability() -> None:
    """'blind' in 'blind review' must not trigger disability detection."""
    text = "Conducted blind review of 50 technical assessments to reduce evaluator bias."
    result = _detector().detect(text)

    assert "disability" not in result.attribute_types_found
