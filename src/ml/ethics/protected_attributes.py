"""
Protected attribute detection for bias analysis.

Detects potential protected attributes in resume text that could
lead to biased decision-making in the hiring process.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from src.utils.constants import PROTECTED_ATTRIBUTES
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DetectedAttribute:
    """A detected protected attribute in text."""

    attribute_type: str  # e.g., "gender", "age", "ethnicity"
    indicator: str  # The text that triggered detection
    confidence: float  # 0-1 confidence score
    location: Optional[tuple[int, int]] = None  # Start, end position in text
    context: Optional[str] = None  # Surrounding text for context


@dataclass
class AttributeDetectionResult:
    """Result of protected attribute detection."""

    has_protected_attributes: bool = False
    detected_attributes: list[DetectedAttribute] = field(default_factory=list)
    risk_level: str = "low"  # "low", "medium", "high"
    recommendations: list[str] = field(default_factory=list)

    @property
    def attribute_types_found(self) -> list[str]:
        """Get unique attribute types found."""
        return list(set(attr.attribute_type for attr in self.detected_attributes))


class ProtectedAttributeDetector:
    """
    Detects protected attributes in text content.

    This detector identifies potential indicators of protected attributes
    such as gender, age, ethnicity, religion, etc. that should not
    influence hiring decisions.
    """

    def __init__(self):
        """Initialize the protected attribute detector."""
        self._setup_patterns()

    def _setup_patterns(self):
        """Set up detection patterns for each protected attribute."""

        # Gender indicators
        self.gender_patterns = {
            "pronouns": re.compile(
                r"\b(he|she|him|her|his|hers|himself|herself)\b",
                re.IGNORECASE
            ),
            "titles": re.compile(
                r"\b(mr\.?|mrs\.?|ms\.?|miss|sir|madam)\b",
                re.IGNORECASE
            ),
            "gendered_terms": re.compile(
                r"\b(wife|husband|mother|father|son|daughter|"
                r"boyfriend|girlfriend|maternity|paternity)\b",
                re.IGNORECASE
            ),
        }

        # Age indicators
        self.age_patterns = {
            "birth_year": re.compile(
                r"\b(born\s+(?:in\s+)?(?:19[4-9]\d|20[0-2]\d)|"
                r"(?:19[4-9]\d|20[0-2]\d)\s*[-–]\s*(?:present|current|now))\b",
                re.IGNORECASE
            ),
            "graduation_year": re.compile(
                r"\b(?:class\s+of|graduated?(?:\s+in)?)\s*['\"]?(\d{2,4})\b",
                re.IGNORECASE
            ),
            "age_explicit": re.compile(
                r"\b(\d{2})\s*(?:years?\s+old|y\.?o\.?|yrs?\s+old)\b",
                re.IGNORECASE
            ),
            "generational": re.compile(
                r"\b(baby\s+boomer|gen\s*[xyz]|millennial|zoomer)\b",
                re.IGNORECASE
            ),
        }

        # Ethnicity/Race indicators
        self.ethnicity_patterns = {
            "explicit": re.compile(
                r"\b(african|asian|caucasian|hispanic|latino|latina|latinx|"
                r"native\s+american|pacific\s+islander|middle\s+eastern)\b",
                re.IGNORECASE
            ),
            "nationality_origin": re.compile(
                r"\b(native\s+of|originally\s+from|born\s+in)\s+"
                r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
                re.IGNORECASE
            ),
        }

        # Religion indicators
        self.religion_patterns = {
            "religious_terms": re.compile(
                r"\b(christian|muslim|jewish|hindu|buddhist|sikh|"
                r"catholic|protestant|orthodox|evangelical|"
                r"church|mosque|synagogue|temple|"
                r"pastor|rabbi|imam|priest|minister)\b",
                re.IGNORECASE
            ),
            "religious_schools": re.compile(
                r"\b(catholic\s+(?:university|college|school)|"
                r"(?:university|college)\s+of\s+(?:notre\s+dame|"
                r"brigham\s+young|liberty))\b",
                re.IGNORECASE
            ),
        }

        # Disability indicators
        self.disability_patterns = {
            "explicit": re.compile(
                r"\b(disabled?|disability|handicapped?|impaired?|"
                r"wheelchair|blind|deaf|autism|adhd|dyslexia|"
                r"chronic\s+(?:illness|condition|disease))\b",
                re.IGNORECASE
            ),
            "accommodations": re.compile(
                r"\b(reasonable\s+accommodation|ada\s+compliant|"
                r"accessibility\s+needs)\b",
                re.IGNORECASE
            ),
        }

        # Marital/Family status indicators
        self.family_patterns = {
            "marital": re.compile(
                r"\b(married|single|divorced|widowed|engaged|"
                r"spouse|partner|domestic\s+partner)\b",
                re.IGNORECASE
            ),
            "children": re.compile(
                r"\b(children|kids|parent|mother|father|"
                r"pregnant|expecting|parental\s+leave)\b",
                re.IGNORECASE
            ),
        }

        # Nationality/Citizenship indicators
        self.nationality_patterns = {
            "citizenship": re.compile(
                r"\b(citizen(?:ship)?|permanent\s+resident|green\s+card|"
                r"visa\s+(?:status|holder|required)|work\s+authorization|"
                r"h-?1b|opt|cpt)\b",
                re.IGNORECASE
            ),
            "national_origin": re.compile(
                r"\b(foreign|immigrant|expat|expatriate|"
                r"international\s+(?:student|worker))\b",
                re.IGNORECASE
            ),
        }

    def detect(self, text: str) -> AttributeDetectionResult:
        """
        Detect protected attributes in text.

        Args:
            text: The text to analyze (resume content).

        Returns:
            AttributeDetectionResult with detected attributes and recommendations.
        """
        if not text:
            return AttributeDetectionResult()

        detected = []

        # Check each category
        detected.extend(self._detect_gender(text))
        detected.extend(self._detect_age(text))
        detected.extend(self._detect_ethnicity(text))
        detected.extend(self._detect_religion(text))
        detected.extend(self._detect_disability(text))
        detected.extend(self._detect_family_status(text))
        detected.extend(self._detect_nationality(text))

        # Calculate risk level
        risk_level = self._calculate_risk_level(detected)

        # Generate recommendations
        recommendations = self._generate_recommendations(detected)

        result = AttributeDetectionResult(
            has_protected_attributes=len(detected) > 0,
            detected_attributes=detected,
            risk_level=risk_level,
            recommendations=recommendations,
        )

        if detected:
            logger.debug(
                f"Detected {len(detected)} protected attribute indicators "
                f"(risk: {risk_level})"
            )

        return result

    def _detect_gender(self, text: str) -> list[DetectedAttribute]:
        """Detect gender-related indicators."""
        detected = []

        for pattern_name, pattern in self.gender_patterns.items():
            for match in pattern.finditer(text):
                # Skip common non-gendered uses
                if pattern_name == "pronouns" and self._is_quote_context(text, match):
                    continue

                detected.append(DetectedAttribute(
                    attribute_type="gender",
                    indicator=match.group(),
                    confidence=0.7 if pattern_name == "pronouns" else 0.9,
                    location=(match.start(), match.end()),
                    context=self._get_context(text, match),
                ))

        return detected

    def _detect_age(self, text: str) -> list[DetectedAttribute]:
        """Detect age-related indicators."""
        detected = []

        for pattern_name, pattern in self.age_patterns.items():
            for match in pattern.finditer(text):
                confidence = 0.95 if pattern_name == "age_explicit" else 0.8

                detected.append(DetectedAttribute(
                    attribute_type="age",
                    indicator=match.group(),
                    confidence=confidence,
                    location=(match.start(), match.end()),
                    context=self._get_context(text, match),
                ))

        return detected

    def _detect_ethnicity(self, text: str) -> list[DetectedAttribute]:
        """Detect ethnicity/race-related indicators."""
        detected = []

        for pattern_name, pattern in self.ethnicity_patterns.items():
            for match in pattern.finditer(text):
                detected.append(DetectedAttribute(
                    attribute_type="ethnicity",
                    indicator=match.group(),
                    confidence=0.85,
                    location=(match.start(), match.end()),
                    context=self._get_context(text, match),
                ))

        return detected

    def _detect_religion(self, text: str) -> list[DetectedAttribute]:
        """Detect religion-related indicators."""
        detected = []

        for pattern_name, pattern in self.religion_patterns.items():
            for match in pattern.finditer(text):
                # Lower confidence for schools (could be secular)
                confidence = 0.6 if pattern_name == "religious_schools" else 0.8

                detected.append(DetectedAttribute(
                    attribute_type="religion",
                    indicator=match.group(),
                    confidence=confidence,
                    location=(match.start(), match.end()),
                    context=self._get_context(text, match),
                ))

        return detected

    def _detect_disability(self, text: str) -> list[DetectedAttribute]:
        """Detect disability-related indicators."""
        detected = []

        for pattern_name, pattern in self.disability_patterns.items():
            for match in pattern.finditer(text):
                # "blind" fires on "blind review", "single-blind study", etc.
                if match.group().lower() == "blind" and self._is_technical_context(text, match):
                    continue

                detected.append(DetectedAttribute(
                    attribute_type="disability",
                    indicator=match.group(),
                    confidence=0.85,
                    location=(match.start(), match.end()),
                    context=self._get_context(text, match),
                ))

        return detected

    def _detect_family_status(self, text: str) -> list[DetectedAttribute]:
        """Detect marital/family status indicators."""
        detected = []

        # Terms that commonly appear in technical contexts and produce false positives
        _technical_prone = {"single", "partner", "parent", "children"}

        for pattern_name, pattern in self.family_patterns.items():
            for match in pattern.finditer(text):
                if match.group().lower() in _technical_prone and self._is_technical_context(text, match):
                    continue

                detected.append(DetectedAttribute(
                    attribute_type="marital_status",
                    indicator=match.group(),
                    confidence=0.75,
                    location=(match.start(), match.end()),
                    context=self._get_context(text, match),
                ))

        return detected

    def _detect_nationality(self, text: str) -> list[DetectedAttribute]:
        """Detect nationality/citizenship indicators."""
        detected = []

        for pattern_name, pattern in self.nationality_patterns.items():
            for match in pattern.finditer(text):
                # "foreign" fires on "foreign key", "foreign function interface", etc.
                if match.group().lower() == "foreign" and self._is_technical_context(text, match):
                    continue

                detected.append(DetectedAttribute(
                    attribute_type="nationality",
                    indicator=match.group(),
                    confidence=0.8,
                    location=(match.start(), match.end()),
                    context=self._get_context(text, match),
                ))

        return detected

    def _is_quote_context(self, text: str, match: re.Match) -> bool:
        """Check if match is within a quote context (testimonial, etc.)."""
        context = self._get_context(text, match, window=50)
        quote_indicators = ['"', "'", "said", "stated", "according to"]
        return any(ind in context.lower() for ind in quote_indicators)

    def _is_technical_context(self, text: str, match: re.Match) -> bool:
        """
        Check if a match sits inside a technical/programming context.

        Used to suppress false positives for short common words that also
        appear in software engineering vocabulary, e.g.:
          - "single" in "single-threaded" or "single page application"
          - "parent" in "parent class" or "parent component"
          - "partner" in "technology partner" or "channel partner"
          - "blind"  in "blind review" or "double-blind study"
          - "foreign" in "foreign key" or "foreign function interface"
        """
        context = self._get_context(text, match, window=40).lower()
        technical_indicators = [
            # OOP / programming
            "class", "method", "function", "interface", "module",
            "component", "process", "element", "node", "thread",
            "threaded", "instance", "object", "library", "framework",
            "api", "sdk", "algorithm", "protocol", "dependency",
            # Data / DB
            "key", "index", "constraint", "query", "database", "table",
            "schema", "foreign key", "primary key",
            # Research / testing
            "review", "study", "test", "fold", "assessment", "trial",
            # File system / OS
            "directory", "folder", "path", "file",
            # Business / channel (for "partner")
            "technology", "business", "strategic", "channel", "ecosystem",
            "integration", "program", "track",
            # Web / UI
            "application", "page", "source",
        ]
        return any(ind in context for ind in technical_indicators)

    def _get_context(
        self,
        text: str,
        match: re.Match,
        window: int = 30,
    ) -> str:
        """Get surrounding context for a match."""
        start = max(0, match.start() - window)
        end = min(len(text), match.end() + window)
        return text[start:end]

    def _calculate_risk_level(
        self,
        detected: list[DetectedAttribute],
    ) -> str:
        """Calculate overall risk level based on detected attributes."""
        if not detected:
            return "low"

        # High-risk attributes
        high_risk_types = {"ethnicity", "religion", "disability"}
        high_risk_count = sum(
            1 for attr in detected if attr.attribute_type in high_risk_types
        )

        # Count unique attribute types
        unique_types = len(set(attr.attribute_type for attr in detected))

        # Average confidence
        avg_confidence = sum(attr.confidence for attr in detected) / len(detected)

        if high_risk_count >= 2 or unique_types >= 4 or avg_confidence > 0.9:
            return "high"
        elif high_risk_count >= 1 or unique_types >= 2 or avg_confidence > 0.7:
            return "medium"
        return "low"

    def _generate_recommendations(
        self,
        detected: list[DetectedAttribute],
    ) -> list[str]:
        """Generate recommendations based on detected attributes."""
        recommendations = []
        types_found = set(attr.attribute_type for attr in detected)

        if "gender" in types_found:
            recommendations.append(
                "Consider removing gender-specific pronouns or titles from evaluation"
            )

        if "age" in types_found:
            recommendations.append(
                "Age-related information detected. Focus on qualifications and "
                "experience rather than years since graduation"
            )

        if "ethnicity" in types_found:
            recommendations.append(
                "Ethnicity/race indicators found. Ensure evaluation is based "
                "solely on job-relevant qualifications"
            )

        if "religion" in types_found:
            recommendations.append(
                "Religious affiliation indicators detected. This should not "
                "factor into hiring decisions"
            )

        if "disability" in types_found:
            recommendations.append(
                "Disability-related information found. Focus on candidate's "
                "ability to perform essential job functions"
            )

        if "marital_status" in types_found:
            recommendations.append(
                "Family/marital status indicators detected. These should not "
                "influence hiring decisions"
            )

        if "nationality" in types_found:
            recommendations.append(
                "Nationality/citizenship indicators found. Verify only legal "
                "work authorization requirements"
            )

        if not recommendations and detected:
            recommendations.append(
                "Protected attributes detected. Review to ensure fair evaluation"
            )

        return recommendations


# Singleton instance
_detector: Optional[ProtectedAttributeDetector] = None


def get_attribute_detector() -> ProtectedAttributeDetector:
    """Get the protected attribute detector singleton instance."""
    global _detector
    if _detector is None:
        _detector = ProtectedAttributeDetector()
    return _detector
