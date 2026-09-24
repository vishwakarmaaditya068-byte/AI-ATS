from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from typing import AsyncIterator, Iterator, Optional

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SQLDatabaseManager:
    """
    Manages PostgreSQL connections using SQLAlchemy 2.0.

    Provides lazily-initialised sync and async engines and session factories.
    Singleton: one instance per process, connections pooled internally.
    """

    _instance: Optional["SQLDatabaseManager"] = None
    _sync_engine: Optional[Engine] = None
    _async_engine: Optional[AsyncEngine] = None
    _sync_session_factory: Optional[sessionmaker[Session]] = None
    _async_session_factory: Optional[async_sessionmaker[AsyncSession]] = None

    def __new__(cls) -> "SQLDatabaseManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._settings = get_settings()
        self._initialized: bool = True

    # ------------------------------------------------------------------
    # Engine factories
    # ------------------------------------------------------------------

    def get_sync_engine(self) -> Engine:
        """Get or create the synchronous psycopg3 engine."""
        if self._sync_engine is None:
            pg = self._settings.postgres
            logger.info("Creating synchronous PostgreSQL engine")
            self._sync_engine = create_engine(
                pg.sync_uri,
                pool_size=pg.pool_size,
                max_overflow=pg.max_overflow,
                pool_pre_ping=True,
                future=True,
            )
        return self._sync_engine

    def get_async_engine(self) -> AsyncEngine:
        """Get or create the asynchronous asyncpg engine."""
        if self._async_engine is None:
            pg = self._settings.postgres
            logger.info("Creating asynchronous PostgreSQL engine")
            self._async_engine = create_async_engine(
                pg.async_uri,
                pool_size=pg.pool_size,
                max_overflow=pg.max_overflow,
                pool_pre_ping=True,
                future=True,
            )
        return self._async_engine

    # ------------------------------------------------------------------
    # Session factories
    # ------------------------------------------------------------------

    def get_sync_session_factory(self) -> sessionmaker[Session]:
        if self._sync_session_factory is None:
            self._sync_session_factory = sessionmaker(
                bind=self.get_sync_engine(),
                autoflush=False,
                autocommit=False,
                expire_on_commit=False,
            )
        return self._sync_session_factory

    def get_async_session_factory(self) -> async_sessionmaker[AsyncSession]:
        if self._async_session_factory is None:
            self._async_session_factory = async_sessionmaker(
                bind=self.get_async_engine(),
                autoflush=False,
                autocommit=False,
                expire_on_commit=False,
            )
        return self._async_session_factory

    # ------------------------------------------------------------------
    # Session context managers — commit on success, rollback on error
    # ------------------------------------------------------------------

    @contextmanager
    def sync_session(self) -> Iterator[Session]:
        factory = self.get_sync_session_factory()
        session: Session = factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @asynccontextmanager
    async def async_session(self) -> AsyncIterator[AsyncSession]:
        factory = self.get_async_session_factory()
        async with factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    def check_sync_connection(self) -> bool:
        try:
            with self.get_sync_engine().connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as exc:
            logger.error(f"Sync postgres check failed: {exc}")
            return False

    async def check_async_connection(self) -> bool:
        try:
            async with self.get_async_engine().connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception as exc:
            logger.error(f"Async postgres check failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close_sync(self) -> None:
        if self._sync_engine is not None:
            logger.info("Disposing synchronous PostgreSQL engine")
            self._sync_engine.dispose()
            self._sync_engine = None
            self._sync_session_factory = None

    async def close_async(self) -> None:
        if self._async_engine is not None:
            logger.info("Disposing asynchronous PostgreSQL engine")
            await self._async_engine.dispose()
            self._async_engine = None
            self._async_session_factory = None


_sql_manager: Optional[SQLDatabaseManager] = None


def get_sql_manager() -> SQLDatabaseManager:
    """Get the process-wide SQLDatabaseManager singleton."""
    global _sql_manager
    if _sql_manager is None:
        _sql_manager = SQLDatabaseManager()
    return _sql_manager
