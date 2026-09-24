"""
Base repository class providing common CRUD operations.

All entity-specific repositories inherit from this base class.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Generic, Optional, TypeVar

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo.collection import Collection
from pymongo.results import DeleteResult, InsertOneResult, UpdateResult

from src.data.database import get_database_manager
from src.data.models.base import BaseDocument
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Type variable for document models
T = TypeVar("T", bound=BaseDocument)


class BaseRepository(ABC, Generic[T]):
    """
    Abstract base repository providing common database operations.

    Implements both synchronous and asynchronous CRUD operations.
    Subclasses must define the collection name and model class.
    """

    @property
    @abstractmethod
    def collection_name(self) -> str:
        """Name of the MongoDB collection."""
        pass

    @property
    @abstractmethod
    def model_class(self) -> type[T]:
        """Pydantic model class for this repository."""
        pass

    def __init__(self) -> None:
        """Initialize repository with database connection."""
        self._db_manager = get_database_manager()

    # -------------------------------------------------------------------------
    # Collection Access
    # -------------------------------------------------------------------------

    def _get_sync_collection(self) -> Collection:
        """Get synchronous collection instance."""
        return self._db_manager.get_sync_collection(self.collection_name)

    def _get_async_collection(self) -> AsyncIOMotorCollection:
        """Get asynchronous collection instance."""
        return self._db_manager.get_async_collection(self.collection_name)

    # -------------------------------------------------------------------------
    # Document Conversion
    # -------------------------------------------------------------------------

    def _to_model(self, document: Optional[dict[str, Any]]) -> Optional[T]:
        """Convert MongoDB document to Pydantic model."""
        if document is None:
            return None
        return self.model_class.model_validate(document)

    def _to_models(self, documents: list[dict[str, Any]]) -> list[T]:
        """Convert list of MongoDB documents to Pydantic models."""
        return [self._to_model(doc) for doc in documents if doc is not None]

    def _to_document(self, model: T) -> dict[str, Any]:
        """Convert Pydantic model to MongoDB document."""
        return model.model_dump_mongo()

    @staticmethod
    def _to_object_id(id_value: str | ObjectId) -> ObjectId:
        """
        Convert string to ObjectId if needed.

        Security: Validates ObjectId format to prevent injection and errors.

        Raises:
            ValueError: If the id_value is not a valid ObjectId format.
        """
        if isinstance(id_value, ObjectId):
            return id_value

        # Validate ObjectId format (24 hex characters)
        if not isinstance(id_value, str):
            raise ValueError(f"Invalid ID type: expected str or ObjectId, got {type(id_value).__name__}")

        id_value = id_value.strip()
        if not id_value:
            raise ValueError("ID cannot be empty")

        if not ObjectId.is_valid(id_value):
            raise ValueError(f"Invalid ObjectId format: {id_value[:50]}")

        return ObjectId(id_value)

    # -------------------------------------------------------------------------
    # Synchronous CRUD Operations
    # -------------------------------------------------------------------------

    def create(self, model: T) -> T:
        """Create a new document."""
        collection = self._get_sync_collection()
        document = self._to_document(model)
        document["created_at"] = datetime.now(timezone.utc)
        document["updated_at"] = datetime.now(timezone.utc)

        result: InsertOneResult = collection.insert_one(document)
        model.id = result.inserted_id
        logger.debug(f"Created {self.collection_name} document: {result.inserted_id}")
        return model

    def get_by_id(self, id_value: str | ObjectId) -> Optional[T]:
        """Get a document by its ID."""
        collection = self._get_sync_collection()
        document = collection.find_one({"_id": self._to_object_id(id_value)})
        return self._to_model(document)

    def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        sort_by: Optional[str] = None,
        sort_order: int = -1,
    ) -> list[T]:
        """Get all documents with pagination."""
        collection = self._get_sync_collection()
        cursor = collection.find().skip(skip).limit(limit)

        if sort_by:
            cursor = cursor.sort(sort_by, sort_order)
        else:
            cursor = cursor.sort("created_at", -1)

        return self._to_models(list(cursor))

    def find(
        self,
        query: dict[str, Any],
        skip: int = 0,
        limit: int = 100,
        sort_by: Optional[str] = None,
        sort_order: int = -1,
    ) -> list[T]:
        """Find documents matching a query."""
        collection = self._get_sync_collection()
        cursor = collection.find(query).skip(skip).limit(limit)

        if sort_by:
            cursor = cursor.sort(sort_by, sort_order)
        else:
            cursor = cursor.sort("created_at", -1)

        return self._to_models(list(cursor))

    def find_one(self, query: dict[str, Any]) -> Optional[T]:
        """Find a single document matching a query."""
        collection = self._get_sync_collection()
        document = collection.find_one(query)
        return self._to_model(document)

    def update(self, id_value: str | ObjectId, update_data: dict[str, Any]) -> Optional[T]:
        """Update a document by ID."""
        collection = self._get_sync_collection()
        update_data["updated_at"] = datetime.now(timezone.utc)

        result: UpdateResult = collection.update_one(
            {"_id": self._to_object_id(id_value)},
            {"$set": update_data},
        )

        if result.modified_count > 0:
            logger.debug(f"Updated {self.collection_name} document: {id_value}")
            return self.get_by_id(id_value)
        return None

    def replace(self, id_value: str | ObjectId, model: T) -> Optional[T]:
        """Replace an entire document."""
        collection = self._get_sync_collection()
        document = self._to_document(model)
        document["updated_at"] = datetime.now(timezone.utc)
        document.pop("_id", None)  # Remove _id to avoid duplication

        result: UpdateResult = collection.replace_one(
            {"_id": self._to_object_id(id_value)},
            document,
        )

        if result.modified_count > 0:
            logger.debug(f"Replaced {self.collection_name} document: {id_value}")
            return self.get_by_id(id_value)
        return None

    def delete(self, id_value: str | ObjectId) -> bool:
        """Delete a document by ID."""
        collection = self._get_sync_collection()
        result: DeleteResult = collection.delete_one(
            {"_id": self._to_object_id(id_value)}
        )
        if result.deleted_count > 0:
            logger.debug(f"Deleted {self.collection_name} document: {id_value}")
            return True
        return False

    def count(self, query: Optional[dict[str, Any]] = None) -> int:
        """Count documents matching a query."""
        collection = self._get_sync_collection()
        if query:
            return collection.count_documents(query)
        return collection.count_documents({})

    def exists(self, query: dict[str, Any]) -> bool:
        """Check if any document matches the query."""
        collection = self._get_sync_collection()
        return collection.count_documents(query, limit=1) > 0

    # -------------------------------------------------------------------------
    # Asynchronous CRUD Operations
    # -------------------------------------------------------------------------

    async def create_async(self, model: T) -> T:
        """Create a new document asynchronously."""
        collection = self._get_async_collection()
        document = self._to_document(model)
        document["created_at"] = datetime.now(timezone.utc)
        document["updated_at"] = datetime.now(timezone.utc)

        result: InsertOneResult = await collection.insert_one(document)
        model.id = result.inserted_id
        logger.debug(f"Created {self.collection_name} document: {result.inserted_id}")
        return model

    async def get_by_id_async(self, id_value: str | ObjectId) -> Optional[T]:
        """Get a document by its ID asynchronously."""
        collection = self._get_async_collection()
        document = await collection.find_one({"_id": self._to_object_id(id_value)})
        return self._to_model(document)

    async def get_all_async(
        self,
        skip: int = 0,
        limit: int = 100,
        sort_by: Optional[str] = None,
        sort_order: int = -1,
    ) -> list[T]:
        """Get all documents with pagination asynchronously."""
        collection = self._get_async_collection()
        cursor = collection.find().skip(skip).limit(limit)

        if sort_by:
            cursor = cursor.sort(sort_by, sort_order)
        else:
            cursor = cursor.sort("created_at", -1)

        documents = await cursor.to_list(length=limit)
        return self._to_models(documents)

    async def find_async(
        self,
        query: dict[str, Any],
        skip: int = 0,
        limit: int = 100,
        sort_by: Optional[str] = None,
        sort_order: int = -1,
    ) -> list[T]:
        """Find documents matching a query asynchronously."""
        collection = self._get_async_collection()
        cursor = collection.find(query).skip(skip).limit(limit)

        if sort_by:
            cursor = cursor.sort(sort_by, sort_order)
        else:
            cursor = cursor.sort("created_at", -1)

        documents = await cursor.to_list(length=limit)
        return self._to_models(documents)

    async def find_one_async(self, query: dict[str, Any]) -> Optional[T]:
        """Find a single document matching a query asynchronously."""
        collection = self._get_async_collection()
        document = await collection.find_one(query)
        return self._to_model(document)

    async def update_async(
        self, id_value: str | ObjectId, update_data: dict[str, Any]
    ) -> Optional[T]:
        """Update a document by ID asynchronously."""
        collection = self._get_async_collection()
        update_data["updated_at"] = datetime.now(timezone.utc)

        result: UpdateResult = await collection.update_one(
            {"_id": self._to_object_id(id_value)},
            {"$set": update_data},
        )

        if result.modified_count > 0:
            logger.debug(f"Updated {self.collection_name} document: {id_value}")
            return await self.get_by_id_async(id_value)
        return None

    async def replace_async(self, id_value: str | ObjectId, model: T) -> Optional[T]:
        """Replace an entire document asynchronously."""
        collection = self._get_async_collection()
        document = self._to_document(model)
        document["updated_at"] = datetime.now(timezone.utc)
        document.pop("_id", None)

        result: UpdateResult = await collection.replace_one(
            {"_id": self._to_object_id(id_value)},
            document,
        )

        if result.modified_count > 0:
            logger.debug(f"Replaced {self.collection_name} document: {id_value}")
            return await self.get_by_id_async(id_value)
        return None

    async def delete_async(self, id_value: str | ObjectId) -> bool:
        """Delete a document by ID asynchronously."""
        collection = self._get_async_collection()
        result: DeleteResult = await collection.delete_one(
            {"_id": self._to_object_id(id_value)}
        )
        if result.deleted_count > 0:
            logger.debug(f"Deleted {self.collection_name} document: {id_value}")
            return True
        return False

    async def count_async(self, query: Optional[dict[str, Any]] = None) -> int:
        """Count documents matching a query asynchronously."""
        collection = self._get_async_collection()
        if query:
            return await collection.count_documents(query)
        return await collection.count_documents({})

    async def exists_async(self, query: dict[str, Any]) -> bool:
        """Check if any document matches the query asynchronously."""
        collection = self._get_async_collection()
        count = await collection.count_documents(query, limit=1)
        return count > 0

    # -------------------------------------------------------------------------
    # Bulk Operations
    # -------------------------------------------------------------------------

    def bulk_create(self, models: list[T]) -> list[T]:
        """Create multiple documents at once."""
        if not models:
            return []

        collection = self._get_sync_collection()
        now = datetime.now(timezone.utc)
        documents = []

        for model in models:
            doc = self._to_document(model)
            doc["created_at"] = now
            doc["updated_at"] = now
            documents.append(doc)

        result = collection.insert_many(documents)

        for model, inserted_id in zip(models, result.inserted_ids):
            model.id = inserted_id

        logger.debug(f"Bulk created {len(models)} {self.collection_name} documents")
        return models

    async def bulk_create_async(self, models: list[T]) -> list[T]:
        """Create multiple documents at once asynchronously."""
        if not models:
            return []

        collection = self._get_async_collection()
        now = datetime.now(timezone.utc)
        documents = []

        for model in models:
            doc = self._to_document(model)
            doc["created_at"] = now
            doc["updated_at"] = now
            documents.append(doc)

        result = await collection.insert_many(documents)

        for model, inserted_id in zip(models, result.inserted_ids):
            model.id = inserted_id

        logger.debug(f"Bulk created {len(models)} {self.collection_name} documents")
        return models

    def bulk_delete(self, ids: list[str | ObjectId]) -> int:
        """Delete multiple documents by IDs."""
        if not ids:
            return 0

        collection = self._get_sync_collection()
        object_ids = [self._to_object_id(id_val) for id_val in ids]
        result = collection.delete_many({"_id": {"$in": object_ids}})
        logger.debug(f"Bulk deleted {result.deleted_count} {self.collection_name} documents")
        return result.deleted_count

    async def bulk_delete_async(self, ids: list[str | ObjectId]) -> int:
        """Delete multiple documents by IDs asynchronously."""
        if not ids:
            return 0

        collection = self._get_async_collection()
        object_ids = [self._to_object_id(id_val) for id_val in ids]
        result = await collection.delete_many({"_id": {"$in": object_ids}})
        logger.debug(f"Bulk deleted {result.deleted_count} {self.collection_name} documents")
        return result.deleted_count
