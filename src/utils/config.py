from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Base paths
ROOT_DIR = Path(__file__).parent.parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"
CONFIGS_DIR = ROOT_DIR / "configs"
RESOURCES_DIR = ROOT_DIR / "resources"


class DatabaseSettings(BaseSettings):
    """MongoDB database configuration."""

    model_config = SettingsConfigDict(env_prefix="DB_", env_file=".env", env_file_encoding="utf-8", extra="ignore")

    host: str = "localhost"
    port: int = 27017
    name: str = "ai_ats"
    username: str | None = None
    password: str | None = None

    @field_validator("host")
    @classmethod
    def validate_host(cls, v: str) -> str:
        """Validate database host to prevent injection."""
        v = v.strip()
        # Prevent shell injection characters
        if any(c in v for c in [";", "&", "|", "$", "`", "\n", "\r"]):
            raise ValueError("Invalid characters in database host")
        return v

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate database name."""
        v = v.strip()
        # MongoDB database names have restrictions
        if not v or any(c in v for c in ["/", "\\", ".", '"', "$", "\0"]):
            raise ValueError("Invalid database name")
        return v

    @property
    def connection_string(self) -> str:
        """Generate MongoDB connection string."""
        from urllib.parse import quote_plus

        if self.username and self.password:
            # URL-encode credentials for safety
            return f"mongodb://{quote_plus(self.username)}:{quote_plus(self.password)}@{self.host}:{self.port}"
        return f"mongodb://{self.host}:{self.port}"

    def __repr__(self) -> str:
        """Safe repr that doesn't expose password."""
        return f"DatabaseSettings(host={self.host!r}, port={self.port}, name={self.name!r}, username={self.username!r}, password='***')"


class PostgresSettings(BaseSettings):
    """PostgreSQL relational-store configuration.

    Path A of the dual-store architecture: Postgres owns workspaces, jobs
    metadata, matches, and audit logs. MongoDB owns candidate/resume docs.
    """

    model_config = SettingsConfigDict(env_prefix="POSTGRES_", env_file=".env", env_file_encoding="utf-8", extra="ignore")

    host: str = "localhost"
    port: int = 5432
    name: str = "ai_ats"
    username: str | None = None
    password: str | None = None
    pool_size: int = 10
    max_overflow: int = 20

    @field_validator("host")
    @classmethod
    def validate_host(cls, v: str) -> str:
        v = v.strip()
        if not v or any(c in v for c in [";", "&", "|", "$", "`", "\n", "\r"]):
            raise ValueError("Invalid characters in postgres host")
        return v

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip()
        if not v or any(c in v for c in ["/", "\\", '"', "$", "\0", " "]):
            raise ValueError("Invalid postgres database name")
        return v

    @property
    def sync_uri(self) -> str:
        """psycopg3 driver URI used by alembic and sync sessions."""
        from urllib.parse import quote_plus

        auth = ""
        if self.username:
            if self.password:
                auth = f"{quote_plus(self.username)}:{quote_plus(self.password)}@"
            else:
                auth = f"{quote_plus(self.username)}@"
        return f"postgresql+psycopg://{auth}{self.host}:{self.port}/{self.name}"

    @property
    def async_uri(self) -> str:
        """asyncpg driver URI used by async sessions at runtime."""
        from urllib.parse import quote_plus

        auth = ""
        if self.username:
            if self.password:
                auth = f"{quote_plus(self.username)}:{quote_plus(self.password)}@"
            else:
                auth = f"{quote_plus(self.username)}@"
        return f"postgresql+asyncpg://{auth}{self.host}:{self.port}/{self.name}"

    def __repr__(self) -> str:
        return (
            f"PostgresSettings(host={self.host!r}, port={self.port}, "
            f"name={self.name!r}, username={self.username!r}, password='***')"
        )


class VectorStoreSettings(BaseSettings):
    """Vector database configuration for embeddings."""

    model_config = SettingsConfigDict(env_prefix="VECTOR_")

    provider: Literal["chromadb", "faiss"] = "chromadb"
    persist_directory: Path = DATA_DIR / "vectors"
    collection_name: str = "resume_embeddings"


class MLSettings(BaseSettings):
    """Machine Learning model configuration."""

    model_config = SettingsConfigDict(env_prefix="ML_")

    # Embedding model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # spaCy model
    spacy_model: str = "en_core_web_trf"

    # Device settings
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto"

    # Batch processing
    batch_size: int = 32

    # Model paths
    models_directory: Path = DATA_DIR / "models"

    resume_success_threshold: float = 0.3

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Auto-detect device if set to auto."""
        if v == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
                elif torch.backends.mps.is_available():
                    return "mps"
            except ImportError:
                pass
            return "cpu"
        return v


class UISettings(BaseSettings):
    """User interface configuration."""

    model_config = SettingsConfigDict(env_prefix="UI_")

    theme: Literal["light", "dark", "system"] = "system"
    language: str = "en"
    window_width: int = 1400
    window_height: int = 900
    font_size: int = 12


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    model_config = SettingsConfigDict(env_prefix="LOG_")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
    file_path: Path = ROOT_DIR / "logs" / "ai_ats.log"
    rotation: str = "10 MB"
    retention: str = "30 days"
    console_output: bool = True


class AppSettings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application metadata
    name: str = "AI-ATS"
    version: str = "0.1.0"
    description: str = "AI-powered Applicant Tracking System"
    debug: bool = False

    # Environment
    environment: Literal["development", "production", "testing"] = "development"

    # Nested settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    ui: UISettings = Field(default_factory=UISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


# Global settings instance (singleton pattern)
_settings: AppSettings | None = None


def get_settings() -> AppSettings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = AppSettings()
    return _settings


def reload_settings() -> AppSettings:
    """Force reload settings from environment."""
    global _settings
    _settings = AppSettings()
    return _settings


def write_env_settings(updates: dict[str, str], env_file: Path | None = None) -> None:
    import re

    target: Path = env_file or (ROOT_DIR / ".env")

    # Read existing lines (create empty file if missing)
    if target.exists():
        lines: list[str] = target.read_text(encoding="utf-8").splitlines(keepends=True)
    else:
        lines = []

    remaining: dict[str, str] = dict(updates)
    new_lines: list[str] = []

    for line in lines:
        # Match uncommented KEY=value lines
        m = re.match(r"^([A-Z][A-Z0-9_]*)=", line)
        if m:
            key = m.group(1)
            if key in remaining:
                # Replace value, preserve trailing newline
                eol = "\n" if not line.endswith("\n") else ""
                new_lines.append(f"{key}={remaining.pop(key)}{eol}\n".rstrip("\n\n") + "\n")
                continue
        new_lines.append(line)

    # Append keys that had no existing line
    if remaining:
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines.append("\n")
        for key, value in remaining.items():
            new_lines.append(f"{key}={value}\n")

    # Atomic write via temp file in the same directory
    tmp = target.with_suffix(".env.tmp")
    tmp.write_text("".join(new_lines), encoding="utf-8")
    tmp.replace(target)


# Convenience exports
settings = get_settings()
