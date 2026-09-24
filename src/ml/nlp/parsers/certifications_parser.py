"""
Certifications parser for resumes.

Extracts certification names, issuers, dates, and credential identifiers
from a certifications section of a resume.
"""

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractedCertification:
    """A certification extracted from a resume."""

    name: Optional[str] = None
    issuer: Optional[str] = None
    issue_date: Optional[date] = None
    expiry_date: Optional[date] = None
    credential_id: Optional[str] = None
    credential_url: Optional[str] = None
    raw_text: str = ""
    confidence: float = 0.0


@dataclass
class CertificationsParseResult:
    """Result of certifications parsing."""

    certifications: list[ExtractedCertification] = field(default_factory=list)
    confidence: float = 0.0


class CertificationsParser:
    """Parser for extracting certifications from resume text."""

    KNOWN_PROVIDERS = [
        "AWS", "Amazon", "Google", "Microsoft", "Azure",
        "Cisco", "CompTIA", "PMI", "Salesforce", "Oracle",
        "Red Hat", "HashiCorp", "CNCF", "ISC2", "ISACA",
        "EC-Council", "Scrum Alliance", "Coursera", "Udacity",
        "LinkedIn Learning", "edX", "Pluralsight", "SANS",
    ]

    CREDENTIAL_ID_PATTERN = re.compile(
        r"(?:credential\s*(?:id|#)?|cert(?:ificate)?\s*(?:id|#))\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-]{4,28})",
        re.IGNORECASE,
    )

    CREDENTIAL_URL_PATTERN = re.compile(
        r"https?://[\w\-./?=#&%+]+(?:verify|credential|certificate)[\w\-./?=#&%+]*",
        re.IGNORECASE,
    )

    EXPIRY_MARKERS = re.compile(
        r"(?:expir(?:es?|y)|valid\s*(?:until|through)|renewal\s*(?:date)?)\s*[:\-]?\s*",
        re.IGNORECASE,
    )

    ISSUE_MARKERS = re.compile(
        r"(?:issued?|obtained|earned|awarded|completed)\s*[:\-]?\s*",
        re.IGNORECASE,
    )

    DATE_PATTERN = re.compile(
        r"(?:"
        r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
        r"dec(?:ember)?)[,.\s]*\d{4}"
        r"|\d{1,2}[/\-]\d{4}"
        r"|\d{4}"
        r")",
        re.IGNORECASE,
    )

    MONTH_MAP = {
        "jan": 1, "january": 1, "feb": 2, "february": 2,
        "mar": 3, "march": 3, "apr": 4, "april": 4,
        "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
        "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10, "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }

    def parse(self, section_text: str) -> CertificationsParseResult:
        """Parse certifications from a section of resume text."""
        if not section_text or not section_text.strip():
            return CertificationsParseResult()

        blocks = self._split_into_blocks(section_text)
        certs = []
        for block in blocks:
            cert = self._parse_cert_block(block)
            if cert and cert.name:
                certs.append(cert)

        if certs:
            avg_conf = sum(c.confidence for c in certs) / len(certs)
        else:
            avg_conf = 0.0

        return CertificationsParseResult(certifications=certs, confidence=avg_conf)

    def _split_into_blocks(self, text: str) -> list[str]:
        """Split text into certification blocks by blank lines."""
        blocks = re.split(r"\n\s*\n", text.strip())
        return [b.strip() for b in blocks if b.strip()]

    def _parse_cert_block(self, block: str) -> Optional[ExtractedCertification]:
        """Extract a single certification from a text block."""
        if not block:
            return None

        cert = ExtractedCertification(raw_text=block)
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]

        if not lines:
            return None

        # Name: first non-empty line that is reasonably short
        first_line = lines[0]
        # Strip leading bullets/numbers
        first_line = re.sub(r"^\s*[\d.)\-•*]\s*", "", first_line).strip()
        if first_line and len(first_line) <= 80:
            cert.name = first_line

        # Issuer: scan all lines for known providers
        cert.issuer = self._detect_issuer(block)

        # Dates: look for expiry markers first, then issue markers
        cert.expiry_date = self._extract_marked_date(block, self.EXPIRY_MARKERS)
        cert.issue_date = self._extract_marked_date(block, self.ISSUE_MARKERS)

        # If no marked dates found, try to extract dates from remaining lines
        if cert.issue_date is None and cert.expiry_date is None:
            all_dates = self.DATE_PATTERN.findall(block)
            if all_dates:
                cert.issue_date = self._parse_date(all_dates[0])
                if len(all_dates) > 1:
                    cert.expiry_date = self._parse_date(all_dates[1])

        # Credential ID
        id_match = self.CREDENTIAL_ID_PATTERN.search(block)
        if id_match:
            cert.credential_id = id_match.group(1)

        # Credential URL
        url_match = self.CREDENTIAL_URL_PATTERN.search(block)
        if url_match:
            cert.credential_url = url_match.group(0)

        # Calculate confidence
        cert.confidence = self._calculate_confidence(cert)
        return cert

    def _extract_marked_date(self, block: str, marker_pattern: re.Pattern) -> Optional[date]:
        """Extract a date that follows a specific marker (issued/expires)."""
        match = marker_pattern.search(block)
        if not match:
            return None
        after_marker = block[match.end():]
        date_match = self.DATE_PATTERN.search(after_marker)
        if date_match:
            return self._parse_date(date_match.group(0))
        return None

    def _detect_issuer(self, block: str) -> Optional[str]:
        """Scan block for known provider names."""
        for provider in self.KNOWN_PROVIDERS:
            if re.search(rf"\b{re.escape(provider)}\b", block, re.IGNORECASE):
                return provider
        return None

    def _parse_date(self, date_str: str) -> Optional[date]:
        """Parse a date string into a datetime.date object."""
        date_str = date_str.strip().rstrip(".,")

        # "Month YYYY" format
        month_year = re.match(
            r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
            r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
            r"dec(?:ember)?)[,.\s]*(\d{4})",
            date_str, re.IGNORECASE,
        )
        if month_year:
            month = self.MONTH_MAP.get(month_year.group(1).lower()[:3], 1)
            year = int(month_year.group(2))
            try:
                return date(year, month, 1)
            except ValueError:
                return None

        # "MM/YYYY" or "MM-YYYY" format
        num_format = re.match(r"(\d{1,2})[/\-](\d{4})", date_str)
        if num_format:
            try:
                return date(int(num_format.group(2)), int(num_format.group(1)), 1)
            except ValueError:
                return None

        # Year only
        year_only = re.match(r"^(\d{4})$", date_str)
        if year_only:
            try:
                return date(int(year_only.group(1)), 1, 1)
            except ValueError:
                return None

        return None

    def _calculate_confidence(self, cert: ExtractedCertification) -> float:
        """Calculate confidence score for a certification."""
        score = 0.0
        if cert.name:
            score += 0.4
        if cert.issuer:
            score += 0.3
        if cert.issue_date:
            score += 0.2
        if cert.credential_id or cert.credential_url:
            score += 0.1
        return round(score, 2)
