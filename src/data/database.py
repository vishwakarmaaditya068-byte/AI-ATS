"""
Database connection manager for AI-ATS.

Provides MongoDB connection management with both synchronous (PyMongo)
and asynchronous (Motor) client support.
"""

from contextlib import asynccontextmanager, contextmanager
from typing import Any, Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    """
    Manages MongoDB database connections.

    Supports both synchronous and asynchronous operations.
    Implements singleton pattern for connection reuse.
    """

    _instance: Optional["DatabaseManager"] = None
    _sync_client: Optional[MongoClient] = None
    _async_client: Optional[AsyncIOMotorClient] = None

    def __new__(cls) -> "DatabaseManager":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize database manager with settings."""
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._settings = get_settings()
        self._db_name = self._settings.database.name
        self._uri = self._build_uri()
        self._initialized = True

    def _build_uri(self) -> str:
        """
        Build MongoDB connection URI from settings.

        Security: URL-encodes credentials to prevent injection attacks.
        """
        from urllib.parse import quote_plus

        db_settings = self._settings.database

        # Validate host to prevent injection
        host = db_settings.host.strip()
        if not host or any(c in host for c in [";", "&", "|", "$", "`"]):
            raise ValueError(f"Invalid database host: {host}")

        # Build URI from components with URL-encoded credentials
        auth = ""
        if db_settings.username and db_settings.password:
            # URL-encode credentials to handle special characters safely
            encoded_user = quote_plus(db_settings.username)
            encoded_pass = quote_plus(db_settings.password)
            auth = f"{encoded_user}:{encoded_pass}@"

        return f"mongodb://{auth}{host}:{db_settings.port}"

    # -------------------------------------------------------------------------
    # Synchronous Client
    # -------------------------------------------------------------------------

    def get_sync_client(self) -> MongoClient:
        """Get or create synchronous MongoDB client."""
        if self._sync_client is None:
            logger.info("Creating synchronous MongoDB client")
            try:
                self._sync_client = MongoClient(
                    self._uri,
                    serverSelectionTimeoutMS=5000,
                    connectTimeoutMS=5000,
                    maxPoolSize=50,
                    minPoolSize=5,
                )
            except Exception as e:
                self._sync_client = None
                logger.error(f"Failed to create sync client: {e}")
                raise
        return self._sync_client

    def get_sync_database(self) -> Database:
        """Get synchronous database instance."""
        return self.get_sync_client()[self._db_name]

    def get_sync_collection(self, collection_name: str) -> Any:
        """Get a synchronous collection by name."""
        return self.get_sync_database()[collection_name]

    @contextmanager
    def sync_session(self):
        """Context manager for synchronous database session."""
        client = self.get_sync_client()
        session = client.start_session()
        try:
            yield session
        finally:
            session.end_session()

    def check_sync_connection(self) -> bool:
        """Check if synchronous connection is healthy."""
        try:
            self.get_sync_client().admin.command("ping")
            return True
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Sync connection check failed: {e}")
            self._sync_client = None
            return False
        except Exception as e:
            logger.error(f"Unexpected sync connection error: {e}")
            self._sync_client = None
            return False

    # -------------------------------------------------------------------------
    # Asynchronous Client
    # -------------------------------------------------------------------------

    def get_async_client(self) -> AsyncIOMotorClient:
        """Get or create asynchronous MongoDB client."""
        if self._async_client is None:
            logger.info("Creating asynchronous MongoDB client")
            self._async_client = AsyncIOMotorClient(
                self._uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                maxPoolSize=50,
                minPoolSize=5,
            )
        return self._async_client

    def get_async_database(self) -> AsyncIOMotorDatabase:
        """Get asynchronous database instance."""
        return self.get_async_client()[self._db_name]

    def get_async_collection(self, collection_name: str) -> Any:
        """Get an asynchronous collection by name."""
        return self.get_async_database()[collection_name]

    @asynccontextmanager
    async def async_session(self):
        """Context manager for asynchronous database session."""
        client = self.get_async_client()
        session = await client.start_session()
        try:
            yield session
        finally:
            await session.end_session()

    async def check_async_connection(self) -> bool:
        """Check if asynchronous connection is healthy."""
        try:
            await self.get_async_client().admin.command("ping")
            return True
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Async connection check failed: {e}")
            return False

    # -------------------------------------------------------------------------
    # Lifecycle Management
    # -------------------------------------------------------------------------

    def close_sync(self) -> None:
        """Close synchronous client connection."""
        if self._sync_client:
            logger.info("Closing synchronous MongoDB client")
            self._sync_client.close()
            self._sync_client = None

    def close_async(self) -> None:
        """Close asynchronous client connection."""
        if self._async_client:
            logger.info("Closing asynchronous MongoDB client")
            self._async_client.close()
            self._async_client = None

    def close_all(self) -> None:
        """Close all database connections."""
        self.close_sync()
        self.close_async()

    # -------------------------------------------------------------------------
    # Index Management
    # -------------------------------------------------------------------------

    async def ensure_indexes(self) -> None:
        """Create indexes for all collections."""
        logger.info("Ensuring database indexes")

        # Candidates collection indexes
        candidates = self.get_async_collection("candidates")
        await candidates.create_index("contact.email", unique=True)
        await candidates.create_index("status")
        await candidates.create_index("metadata.tags")
        await candidates.create_index("created_at")

        # Resumes collection indexes
        resumes = self.get_async_collection("resumes")
        await resumes.create_index("candidate_id")
        await resumes.create_index("status")
        await resumes.create_index("file.file_hash", unique=True)
        await resumes.create_index("created_at")

        # Jobs collection indexes
        jobs = self.get_async_collection("jobs")
        await jobs.create_index("status")
        await jobs.create_index("company_name")
        await jobs.create_index("employment_type")
        await jobs.create_index("experience_level")
        await jobs.create_index("metadata.tags")
        await jobs.create_index("posted_date")
        await jobs.create_index("created_at")

        # Matches collection indexes
        matches = self.get_async_collection("matches")
        await matches.create_index(
            [("candidate_id", 1), ("job_id", 1)], unique=True
        )
        await matches.create_index("job_id")
        await matches.create_index("overall_score")
        await matches.create_index("score_level")
        await matches.create_index("status")
        await matches.create_index("rank")
        await matches.create_index("created_at")

        # Audit logs collection indexes
        audit_logs = self.get_async_collection("audit_logs")
        await audit_logs.create_index("action")
        await audit_logs.create_index("actor.actor_id")
        await audit_logs.create_index("resource.resource_type")
        await audit_logs.create_index("resource.resource_id")
        await audit_logs.create_index("related_candidate_id")
        await audit_logs.create_index("related_job_id")
        await audit_logs.create_index("compliance_relevant")
        await audit_logs.create_index("created_at")

        logger.info("Database indexes created successfully")


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def get_sync_db() -> Database:
    """Convenience function to get synchronous database."""
    return get_database_manager().get_sync_database()


def get_async_db() -> AsyncIOMotorDatabase:
    """Convenience function to get asynchronous database."""
    return get_database_manager().get_async_database()
