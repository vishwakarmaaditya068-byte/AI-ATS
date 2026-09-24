from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, Sequence, TypeVar

from sqlalchemy import delete as sa_delete, select, update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from src.data.sql.base import SQLBase
from src.data.sql.engine import get_sql_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=SQLBase)


class BaseSQLRepository(ABC, Generic[T]):
    """CRUD base for a single SQLAlchemy model."""

    @property
    @abstractmethod
    def model_class(self) -> type[T]:
        """The SQLAlchemy model this repository manages."""

    def __init__(self) -> None:
        self._sql = get_sql_manager()

    # Create
    def create(self, entity: T) -> T:
        with self._sql.sync_session() as session:
            session.add(entity)
            session.flush()
            session.refresh(entity)
            session.expunge(entity)
            return entity

    async def create_async(self, entity: T) -> T:
        async with self._sql.async_session() as session:
            session.add(entity)
            await session.flush()
            await session.refresh(entity)
            session.expunge(entity)
            return entity

    # Read
    def get(self, id_value: uuid.UUID | str) -> Optional[T]:
        pk = self._coerce_uuid(id_value)
        with self._sql.sync_session() as session:
            entity: Optional[T] = session.get(self.model_class, pk)
            if entity is not None:
                session.expunge(entity)
            return entity

    async def get_async(self, id_value: uuid.UUID | str) -> Optional[T]:
        pk = self._coerce_uuid(id_value)
        async with self._sql.async_session() as session:
            entity: Optional[T] = await session.get(self.model_class, pk)
            if entity is not None:
                session.expunge(entity)
            return entity

    def list_all(self, limit: int = 100, offset: int = 0) -> list[T]:
        with self._sql.sync_session() as session:
            stmt = select(self.model_class).limit(limit).offset(offset)
            result: Sequence[T] = session.execute(stmt).scalars().all()
            for entity in result:
                session.expunge(entity)
            return list(result)

    # Update (partial via kwargs)
    def update(self, id_value: uuid.UUID | str, **values: Any) -> Optional[T]:
        if not values:
            return self.get(id_value)
        pk = self._coerce_uuid(id_value)
        with self._sql.sync_session() as session:
            stmt = (
                sa_update(self.model_class)
                .where(self.model_class.id == pk)  # type: ignore[attr-defined]
                .values(**values)
                .returning(self.model_class)
            )
            entity: Optional[T] = session.execute(stmt).scalars().first()
            if entity is not None:
                session.flush()
                session.refresh(entity)
                session.expunge(entity)
            return entity

    async def update_async(self, id_value: uuid.UUID | str, **values: Any) -> Optional[T]:
        if not values:
            return await self.get_async(id_value)
        pk = self._coerce_uuid(id_value)
        async with self._sql.async_session() as session:
            stmt = (
                sa_update(self.model_class)
                .where(self.model_class.id == pk)  # type: ignore[attr-defined]
                .values(**values)
                .returning(self.model_class)
            )
            result = await session.execute(stmt)
            entity: Optional[T] = result.scalars().first()
            if entity is not None:
                await session.flush()
                await session.refresh(entity)
                session.expunge(entity)
            return entity

    # Delete
    def delete(self, id_value: uuid.UUID | str) -> bool:
        pk = self._coerce_uuid(id_value)
        with self._sql.sync_session() as session:
            stmt = sa_delete(self.model_class).where(
                self.model_class.id == pk  # type: ignore[attr-defined]
            )
            result = session.execute(stmt)
            return result.rowcount > 0

    async def delete_async(self, id_value: uuid.UUID | str) -> bool:
        pk = self._coerce_uuid(id_value)
        async with self._sql.async_session() as session:
            stmt = sa_delete(self.model_class).where(
                self.model_class.id == pk  # type: ignore[attr-defined]
            )
            result = await session.execute(stmt)
            return result.rowcount > 0

    # Helpers
    @staticmethod
    def _coerce_uuid(value: uuid.UUID | str) -> uuid.UUID:
        if isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(str(value))

    def _in_session(self, session: Session, entity: T) -> T:
        """Attach an external entity to a session for custom workflows."""
        session.add(entity)
        return entity

    async def _in_session_async(self, session: AsyncSession, entity: T) -> T:
        session.add(entity)
        return entity
