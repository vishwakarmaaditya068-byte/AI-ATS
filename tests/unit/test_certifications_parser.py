"""
Tests for src.ml.nlp.parsers.certifications_parser — CertificationsParser.
"""

import pytest
from datetime import date

from src.ml.nlp.parsers.certifications_parser import CertificationsParser, CertificationsParseResult


@pytest.fixture
def parser():
    return CertificationsParser()


class TestCertificationsParser:
    def test_single_cert_name_extraction(self, parser):
        text = "AWS Certified Solutions Architect\nAmazon Web Services\nIssued January 2022"
        result = parser.parse(text)
        assert len(result.certifications) == 1
        assert result.certifications[0].name == "AWS Certified Solutions Architect"

    def test_known_issuer_detection_aws(self, parser):
        text = "AWS Certified Developer\nAmazon Web Services\nIssued March 2021"
        result = parser.parse(text)
        assert result.certifications[0].issuer is not None
        assert "Amazon" in result.certifications[0].issuer or "AWS" in result.certifications[0].issuer

    def test_known_issuer_detection_google(self, parser):
        text = "Google Cloud Professional Data Engineer\nGoogle\nIssued June 2023"
        result = parser.parse(text)
        assert result.certifications[0].issuer == "Google"

    def test_issue_date_extraction(self, parser):
        text = "CompTIA Security+\nIssued January 2022"
        result = parser.parse(text)
        assert result.certifications[0].issue_date is not None
        assert result.certifications[0].issue_date == date(2022, 1, 1)

    def test_expiry_date_extraction(self, parser):
        text = "Cisco CCNA\nIssued January 2021\nExpires January 2024"
        result = parser.parse(text)
        assert result.certifications[0].expiry_date is not None
        assert result.certifications[0].expiry_date == date(2024, 1, 1)

    def test_credential_id_extraction(self, parser):
        text = "AWS Certified Solutions Architect\nCredential ID: ABC-123456"
        result = parser.parse(text)
        assert result.certifications[0].credential_id == "ABC-123456"

    def test_credential_url_extraction(self, parser):
        text = "AWS Certified Developer\nhttps://aws.amazon.com/verify/ABC123XYZ"
        result = parser.parse(text)
        assert result.certifications[0].credential_url is not None
        assert "verify" in result.certifications[0].credential_url

    def test_empty_input_returns_empty_result(self, parser):
        result = parser.parse("")
        assert result.certifications == []
        assert result.confidence == 0.0

    def test_whitespace_only_returns_empty_result(self, parser):
        result = parser.parse("   \n   ")
        assert result.certifications == []

    def test_two_certs_split_by_blank_line(self, parser):
        text = (
            "AWS Certified Solutions Architect\nAmazon\nIssued January 2022\n\n"
            "Google Cloud Engineer\nGoogle\nIssued March 2023"
        )
        result = parser.parse(text)
        assert len(result.certifications) == 2

    def test_confidence_higher_with_more_fields(self, parser):
        name_only = "Some Certification"
        full_cert = "AWS Certified Developer\nAmazon\nIssued January 2022\nCredential ID: XYZ123"
        r_name = parser.parse(name_only)
        r_full = parser.parse(full_cert)
        assert r_full.certifications[0].confidence > r_name.certifications[0].confidence

    def test_name_only_confidence(self, parser):
        text = "Certified Kubernetes Administrator"
        result = parser.parse(text)
        assert result.certifications[0].confidence == pytest.approx(0.4)

    def test_average_confidence_in_result(self, parser):
        text = (
            "AWS Certified Solutions Architect\nAmazon\nIssued January 2022\n\n"
            "Google Cloud Engineer\nGoogle\nIssued March 2023"
        )
        result = parser.parse(text)
        individual_avg = sum(c.confidence for c in result.certifications) / len(result.certifications)
        assert result.confidence == pytest.approx(individual_avg, abs=0.01)
