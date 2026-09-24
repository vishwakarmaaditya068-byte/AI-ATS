"""
Contact information parser for resumes.

Extracts names, emails, phone numbers, LinkedIn URLs, etc.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ContactInfo:
    """Extracted contact information from a resume."""

    first_name: Optional[str] = None
    last_name: Optional[str] = None
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None
    portfolio_url: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    postal_code: Optional[str] = None
    confidence: float = 0.0
    raw_matches: dict = field(default_factory=dict)


class ContactParser:
    """Parser for extracting contact information from resume text."""

    # Email pattern
    EMAIL_PATTERN = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    )

    # Phone patterns (various formats)
    PHONE_PATTERNS = [
        # US formats
        re.compile(r"\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"),
        # International
        re.compile(r"\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}"),
        # General
        re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    ]

    # LinkedIn URL pattern
    LINKEDIN_PATTERN = re.compile(
        r"(?:https?://)?(?:www\.)?linkedin\.com/in/[\w\-]+/?",
        re.IGNORECASE,
    )

    # GitHub URL pattern
    GITHUB_PATTERN = re.compile(
        r"(?:https?://)?(?:www\.)?github\.com/[\w\-]+/?",
        re.IGNORECASE,
    )

    # Portfolio/Website patterns
    WEBSITE_PATTERN = re.compile(
        r"(?:https?://)?(?:www\.)?[\w\-]+\.(?:com|io|dev|me|org|net|co)(?:/[\w\-./]*)?",
        re.IGNORECASE,
    )

    # US ZIP code pattern
    ZIP_PATTERN = re.compile(r"\b\d{5}(?:-\d{4})?\b")

    # Common US states
    US_STATES = {
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
        "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
        "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
        "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
        "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
        "DC",
    }

    # Common name prefixes/suffixes to filter out
    NAME_STOPWORDS = {
        "resume", "cv", "curriculum", "vitae", "page", "of",
        "phone", "email", "address", "linkedin", "github",
        "objective", "summary", "experience", "education", "skills",
        "references", "available", "upon", "request",
    }

    # Maximum text length to process (prevent ReDoS)
    MAX_TEXT_LENGTH = 100_000  # 100KB

    def parse(self, text: str, focus_on_header: bool = True) -> ContactInfo:
        """
        Parse contact information from resume text.

        Args:
            text: Full resume text or contact section
            focus_on_header: If True, prioritize first 20 lines

        Returns:
            ContactInfo with extracted details

        Security:
            - Limits input text length to prevent ReDoS attacks
            - Truncates overly long lines before regex processing
        """
        result = ContactInfo()
        raw_matches = {}

        # Security: Limit input length to prevent ReDoS
        if len(text) > self.MAX_TEXT_LENGTH:
            logger.warning(f"Input text truncated from {len(text)} to {self.MAX_TEXT_LENGTH} chars")
            text = text[:self.MAX_TEXT_LENGTH]

        # Focus on header for contact info (usually first 15-20 lines)
        if focus_on_header:
            lines = text.split("\n")
            header_text = "\n".join(lines[:20])
        else:
            header_text = text

        # Extract email
        email = self._extract_email(text)
        if email:
            result.email = email
            raw_matches["email"] = email

        # Extract phone
        phone = self._extract_phone(text)
        if phone:
            result.phone = phone
            raw_matches["phone"] = phone

        # Extract LinkedIn
        linkedin = self._extract_linkedin(text)
        if linkedin:
            result.linkedin_url = linkedin
            raw_matches["linkedin"] = linkedin

        # Extract GitHub
        github = self._extract_github(text)
        if github:
            result.github_url = github
            raw_matches["github"] = github

        # Extract other website/portfolio
        portfolio = self._extract_portfolio(text, exclude=[linkedin, github])
        if portfolio:
            result.portfolio_url = portfolio
            raw_matches["portfolio"] = portfolio

        # Extract name (from header)
        name_info = self._extract_name(header_text)
        if name_info:
            result.full_name = name_info.get("full_name")
            result.first_name = name_info.get("first_name")
            result.last_name = name_info.get("last_name")
            raw_matches["name"] = name_info

        # Extract location
        location_info = self._extract_location(header_text)
        if location_info:
            result.city = location_info.get("city")
            result.state = location_info.get("state")
            result.country = location_info.get("country")
            result.postal_code = location_info.get("postal_code")
            result.address = location_info.get("address")
            raw_matches["location"] = location_info

        # Calculate confidence
        result.confidence = self._calculate_confidence(result)
        result.raw_matches = raw_matches

        return result

    def _extract_email(self, text: str) -> Optional[str]:
        """Extract email address from text."""
        matches = self.EMAIL_PATTERN.findall(text)
        if matches:
            # Return the first valid-looking email
            for email in matches:
                email = email.lower()
                # Filter out obviously fake emails
                if not any(x in email for x in ["example", "test", "sample"]):
                    return email
        return None

    def _extract_phone(self, text: str) -> Optional[str]:
        """Extract phone number from text."""
        for pattern in self.PHONE_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                # Clean and return first match
                phone = matches[0]
                # Normalize format
                digits = re.sub(r"\D", "", phone)
                if len(digits) >= 10:
                    return phone
        return None

    def _extract_linkedin(self, text: str) -> Optional[str]:
        """Extract LinkedIn URL from text."""
        match = self.LINKEDIN_PATTERN.search(text)
        if match:
            url = match.group(0)
            # Ensure https prefix
            if not url.startswith("http"):
                url = "https://" + url
            return url
        return None

    def _extract_github(self, text: str) -> Optional[str]:
        """Extract GitHub URL from text."""
        match = self.GITHUB_PATTERN.search(text)
        if match:
            url = match.group(0)
            if not url.startswith("http"):
                url = "https://" + url
            return url
        return None

    def _extract_portfolio(
        self, text: str, exclude: list[Optional[str]] = None
    ) -> Optional[str]:
        """Extract portfolio/personal website URL."""
        exclude = [u for u in (exclude or []) if u]

        matches = self.WEBSITE_PATTERN.findall(text)
        for url in matches:
            # Skip LinkedIn and GitHub
            if "linkedin.com" in url.lower() or "github.com" in url.lower():
                continue
            # Skip excluded URLs
            if any(url in ex for ex in exclude if ex):
                continue
            # Skip common non-portfolio sites
            skip_domains = ["google.com", "facebook.com", "twitter.com", "indeed.com"]
            if any(domain in url.lower() for domain in skip_domains):
                continue

            if not url.startswith("http"):
                url = "https://" + url
            return url

        return None

    def _extract_name(self, text: str) -> Optional[dict]:
        """
        Extract name from resume header.

        Uses heuristics since name is typically the first prominent line.
        """
        lines = text.split("\n")

        for line in lines[:10]:  # Focus on first 10 lines
            line = line.strip()

            if not line or len(line) < 3:
                continue

            # Skip lines that look like section headers or contain common keywords
            line_lower = line.lower()
            if any(word in line_lower for word in self.NAME_STOPWORDS):
                continue

            # Skip lines with email or phone patterns
            if self.EMAIL_PATTERN.search(line) or any(
                p.search(line) for p in self.PHONE_PATTERNS
            ):
                continue

            # Skip lines with URLs
            if "http" in line.lower() or ".com" in line.lower():
                continue

            # Skip lines that are too long (probably not a name)
            if len(line) > 50:
                continue

            # Skip single-word all-uppercase lines (section headers like "SKILLS")
            # but allow two-word ALL-CAPS names like "JOHN SMITH"
            if line.isupper() and len(line.split()) <= 1:
                continue

            # Check if line looks like a name (2-4 words, mostly letters)
            words = line.split()
            if 1 <= len(words) <= 4:
                # Check if words look like name parts.
                # Handles:
                #   - Normal caps: "John Smith"
                #   - All-caps: "JOHN SMITH"
                #   - Middle initials: "John A. Smith" (single letter + optional period)
                #   - Hyphenated: "Mary-Jane Watson"
                #   - Apostrophe names: "O'Brien"
                def _is_name_word(w: str) -> bool:
                    clean = w.replace("-", "").replace("'", "").rstrip(".")
                    # Single uppercase letter (middle initial like "A.")
                    if len(clean) == 1 and clean.isupper():
                        return True
                    # Must start with uppercase (title-case or all-caps)
                    if not w[0].isupper():
                        return False
                    return clean.isalpha()

                name_like = all(_is_name_word(word) for word in words if word)

                if name_like:
                    # Normalize all-caps names to title case ("JOHN SMITH" → "John Smith")
                    normalized = line.title() if line.isupper() else line
                    norm_words = normalized.split()
                    first_name = norm_words[0] if norm_words else None
                    last_name = norm_words[-1] if len(norm_words) > 1 else None

                    return {
                        "full_name": normalized,
                        "first_name": first_name,
                        "last_name": last_name,
                    }

        return None

    def _extract_location(self, text: str) -> Optional[dict]:
        """Extract location information from text."""
        result = {}

        # Look for ZIP code
        zip_match = self.ZIP_PATTERN.search(text)
        if zip_match:
            result["postal_code"] = zip_match.group(0)

        # Look for state abbreviations
        for state in self.US_STATES:
            # Match state abbreviation with word boundaries
            pattern = rf"\b{state}\b"
            if re.search(pattern, text):
                result["state"] = state
                result["country"] = "USA"
                break

        # Try to extract city from common patterns
        # Pattern: City, ST or City, State
        city_state_pattern = re.compile(
            r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s*,\s*([A-Z]{2})\b"
        )
        match = city_state_pattern.search(text)
        if match:
            result["city"] = match.group(1)
            result["state"] = match.group(2)
            result["country"] = "USA"

        # Build address from context if we found location markers
        if result:
            # Try to find the full address line
            lines = text.split("\n")
            for line in lines[:15]:
                line = line.strip()
                if result.get("state") and result["state"] in line:
                    result["address"] = line
                    break
                if result.get("postal_code") and result["postal_code"] in line:
                    result["address"] = line
                    break

        return result if result else None

    def _calculate_confidence(self, info: ContactInfo) -> float:
        """Calculate confidence score for extracted contact info."""
        score = 0.0
        max_score = 5.0

        if info.email:
            score += 1.5  # Email is very important

        if info.phone:
            score += 1.0

        if info.full_name or (info.first_name and info.last_name):
            score += 1.5  # Name is very important

        if info.linkedin_url:
            score += 0.5

        if info.city or info.state:
            score += 0.5

        return min(score / max_score, 1.0)
