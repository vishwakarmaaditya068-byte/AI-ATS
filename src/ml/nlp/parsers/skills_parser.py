"""
Skills parser for resumes.

Extracts technical and soft skills from resume text.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from src.utils.constants import SKILL_CATEGORIES
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractedSkill:
    """A skill extracted from a resume."""

    name: str
    category: Optional[str] = None
    proficiency: Optional[str] = None  # beginner, intermediate, advanced, expert
    years: Optional[float] = None
    confidence: float = 1.0
    source: str = "text"  # "text", "section", "inferred"


@dataclass
class SkillsParseResult:
    """Result of skills parsing."""

    skills: list[ExtractedSkill] = field(default_factory=list)
    raw_skill_text: Optional[str] = None
    confidence: float = 0.0


class SkillsParser:
    """Parser for extracting skills from resume text."""

    # Extended skill database (in addition to SKILL_CATEGORIES)
    ADDITIONAL_SKILLS = {
        "programming_languages": [
            "html", "css", "bash", "shell", "powershell", "perl", "lua",
            "groovy", "objective-c", "assembly", "fortran", "cobol",
            "haskell", "erlang", "elixir", "clojure", "f#", "vb.net",
            "dart", "julia", "solidity", "vhdl", "verilog",
        ],
        "frameworks": [
            "next.js", "nuxt.js", "svelte", "nestjs", "fastify", "koa",
            "gin", "echo", "fiber", "actix", "rocket", "phoenix",
            "asp.net", "blazor", "wpf", "winforms", "qt", "gtk",
            "electron", "tauri", "flutter", "react native", "ionic",
            "xamarin", "maui", "unity", "unreal", "godot",
            "spring boot", "hibernate", "mybatis", "dropwizard",
            "micronaut", "quarkus", "vert.x",
            "celery", "airflow", "luigi", "prefect",
            "huggingface", "langchain", "llamaindex",
        ],
        "databases": [
            "mariadb", "cockroachdb", "timescaledb", "clickhouse",
            "neo4j", "arangodb", "couchdb", "rethinkdb",
            "memcached", "etcd", "consul", "zookeeper",
            "pinecone", "weaviate", "milvus", "qdrant", "chromadb",
            "snowflake", "databricks", "bigquery", "redshift", "athena",
        ],
        "cloud_platforms": [
            "cloudflare", "vercel", "netlify", "railway", "render",
            "lambda", "ec2", "s3", "rds", "eks", "ecs", "fargate",
            "cloud functions", "cloud run", "gke", "app engine",
            "azure functions", "aks", "cosmos db",
            "openshift", "rancher", "nomad", "consul", "vault",
            "prometheus", "grafana", "datadog", "splunk", "elk",
            "jenkins", "gitlab ci", "github actions", "circleci",
            "argocd", "flux", "spinnaker", "tekton",
        ],
        "data_science": [
            "pandas", "numpy", "scipy", "matplotlib", "seaborn",
            "plotly", "bokeh", "altair", "streamlit", "gradio",
            "jupyter", "notebook", "colab",
            "spark", "hadoop", "hive", "pig", "flink", "kafka",
            "dbt", "great expectations", "mlflow", "kubeflow",
            "sagemaker", "vertex ai", "azure ml",
            "opencv", "pillow", "imageio",
            "nltk", "spacy", "gensim", "transformers",
            "xgboost", "lightgbm", "catboost",
        ],
        "devops": [
            "ci/cd", "continuous integration", "continuous deployment",
            "infrastructure as code", "iac", "gitops",
            "linux", "unix", "windows server", "macos",
            "nginx", "apache", "caddy", "traefik", "haproxy",
            "systemd", "supervisor", "pm2",
            "vagrant", "packer", "pulumi", "cdk",
        ],
        "testing": [
            "jest", "mocha", "jasmine", "cypress", "playwright",
            "selenium", "puppeteer", "testcafe",
            "pytest", "unittest", "nose", "robot framework",
            "junit", "testng", "mockito", "wiremock",
            "postman", "insomnia", "soapui",
            "jmeter", "gatling", "locust", "k6",
            "tdd", "bdd", "unit testing", "integration testing",
            "e2e testing", "load testing", "performance testing",
        ],
        "soft_skills": [
            "agile", "scrum", "kanban", "waterfall",
            "project management", "product management",
            "technical writing", "documentation",
            "mentoring", "coaching", "training",
            "stakeholder management", "client facing",
            "cross-functional", "collaboration",
            "critical thinking", "decision making",
            "time management", "prioritization",
            "conflict resolution", "negotiation",
            "presentation", "public speaking",
        ],
        "security": [
            "oauth", "jwt", "saml", "openid", "ldap",
            "ssl", "tls", "https", "encryption",
            "penetration testing", "vulnerability assessment",
            "soc2", "gdpr", "hipaa", "pci-dss",
            "owasp", "security best practices",
        ],
        "methodologies": [
            "rest", "graphql", "grpc", "soap", "websocket",
            "microservices", "monolith", "serverless",
            "event-driven", "cqrs", "event sourcing",
            "domain-driven design", "ddd",
            "clean architecture", "hexagonal architecture",
            "solid", "dry", "kiss", "yagni",
            "design patterns", "gang of four",
        ],
    }

    # Proficiency indicators
    PROFICIENCY_PATTERNS = {
        "expert": [
            "expert", "advanced", "proficient", "senior", "lead",
            "extensive experience", "deep knowledge", "mastery",
        ],
        "advanced": [
            "strong", "solid", "significant", "substantial",
            "considerable", "thorough",
        ],
        "intermediate": [
            "intermediate", "moderate", "good", "working knowledge",
            "familiar", "competent",
        ],
        "beginner": [
            "beginner", "basic", "fundamental", "learning",
            "exposure", "some experience", "entry",
        ],
    }

    def __init__(self):
        """Initialize the skills parser with combined skill lists."""
        self._build_skill_index()

    def _build_skill_index(self) -> None:
        """Build an index of all known skills for quick lookup."""
        self.skill_to_category: dict[str, str] = {}

        # Add skills from constants
        for category, skills in SKILL_CATEGORIES.items():
            for skill in skills:
                self.skill_to_category[skill.lower()] = category

        # Add additional skills
        for category, skills in self.ADDITIONAL_SKILLS.items():
            for skill in skills:
                self.skill_to_category[skill.lower()] = category

    def parse(
        self,
        text: str,
        skills_section: Optional[str] = None,
    ) -> SkillsParseResult:
        """
        Parse skills from resume text.

        Args:
            text: Full resume text
            skills_section: Optional dedicated skills section text

        Returns:
            SkillsParseResult with extracted skills
        """
        skills: list[ExtractedSkill] = []
        seen_skills: set[str] = set()

        # Parse skills section first (higher confidence)
        if skills_section:
            section_skills = self._parse_skills_section(skills_section)
            for skill in section_skills:
                if skill.name.lower() not in seen_skills:
                    skill.source = "section"
                    skills.append(skill)
                    seen_skills.add(skill.name.lower())

        # Parse from full text (find mentions of known skills)
        text_skills = self._extract_known_skills(text)
        for skill in text_skills:
            if skill.name.lower() not in seen_skills:
                skill.source = "text"
                skill.confidence *= 0.8  # Lower confidence for text mentions
                skills.append(skill)
                seen_skills.add(skill.name.lower())

        # Calculate overall confidence
        if skills:
            avg_confidence = sum(s.confidence for s in skills) / len(skills)
        else:
            avg_confidence = 0.0

        return SkillsParseResult(
            skills=skills,
            raw_skill_text=skills_section,
            confidence=avg_confidence,
        )

    def _parse_skills_section(self, section_text: str) -> list[ExtractedSkill]:
        """Parse skills from a dedicated skills section."""
        skills = []

        # Expand "Category Label: skill1, skill2" lines — strip the label,
        # keeping only the items so they can be tokenised correctly below.
        expanded_lines = []
        for line in section_text.split("\n"):
            colon_match = re.match(
                r"^([A-Za-z][A-Za-z\s&/()]{0,30}):\s*(.+)$", line.strip()
            )
            if colon_match:
                label, items = colon_match.group(1).strip(), colon_match.group(2)
                # Only strip the label when it looks like a category header
                # (short phrase, not a skill name itself)
                if len(label.split()) <= 4:
                    expanded_lines.append(items)
                    continue
            expanded_lines.append(line)
        section_text = "\n".join(expanded_lines)

        # Common separators in skills sections
        # Split by common delimiters
        skill_candidates = re.split(
            r"[,;|•·\-\n]|\s{2,}",
            section_text,
        )

        for candidate in skill_candidates:
            candidate = candidate.strip()

            # Skip empty or too short
            if not candidate or len(candidate) < 2:
                continue

            # Skip if too long (probably not a single skill)
            if len(candidate) > 50:
                # Try to extract skills from this longer text
                sub_skills = self._extract_known_skills(candidate)
                skills.extend(sub_skills)
                continue

            # Clean up the candidate
            clean_skill = self._clean_skill_name(candidate)

            if clean_skill:
                category = self._get_skill_category(clean_skill)
                proficiency = self._detect_proficiency(candidate)

                skills.append(
                    ExtractedSkill(
                        name=clean_skill,
                        category=category,
                        proficiency=proficiency,
                        confidence=0.9 if category else 0.7,
                    )
                )

        return skills

    def _extract_known_skills(self, text: str) -> list[ExtractedSkill]:
        """Extract mentions of known skills from text."""
        skills = []
        text_lower = text.lower()

        for skill_name, category in self.skill_to_category.items():
            # Use word boundary matching for accuracy
            pattern = rf"\b{re.escape(skill_name)}\b"

            if re.search(pattern, text_lower, re.IGNORECASE):
                # Find the original case version in text
                match = re.search(pattern, text, re.IGNORECASE)
                original_name = match.group(0) if match else skill_name

                skills.append(
                    ExtractedSkill(
                        name=original_name,
                        category=category,
                        confidence=0.85,
                    )
                )

        return skills

    def _clean_skill_name(self, name: str) -> Optional[str]:
        """Clean and validate a skill name."""
        # Remove leading/trailing punctuation and whitespace
        name = re.sub(r"^[\s\W]+|[\s\W]+$", "", name)

        # Skip if empty or too short
        if not name or len(name) < 2:
            return None

        # Skip if it's just numbers
        if name.isdigit():
            return None

        # Skip common non-skill words
        skip_words = {
            "and", "or", "the", "with", "using", "including",
            "etc", "years", "year", "experience", "knowledge",
            "understanding", "familiar", "proficient",
        }
        if name.lower() in skip_words:
            return None

        return name

    def _get_skill_category(self, skill_name: str) -> Optional[str]:
        """Get the category for a skill."""
        return self.skill_to_category.get(skill_name.lower())

    def _detect_proficiency(self, context: str) -> Optional[str]:
        """Detect proficiency level from context."""
        context_lower = context.lower()

        for level, indicators in self.PROFICIENCY_PATTERNS.items():
            for indicator in indicators:
                if indicator in context_lower:
                    return level

        return None

    def categorize_skills(
        self, skills: list[ExtractedSkill]
    ) -> dict[str, list[ExtractedSkill]]:
        """Group skills by category."""
        categorized: dict[str, list[ExtractedSkill]] = {}

        for skill in skills:
            category = skill.category or "other"
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(skill)

        return categorized
