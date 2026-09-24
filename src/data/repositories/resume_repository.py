"""
Resume repository for AI-ATS.

Provides data access operations for resume documents,
including file tracking and processing status management.
"""

from datetime import datetime, timezone
from typing import Any, Optional

from bson import ObjectId

from src.data.models.resume import (
    ParsedContent,
    ProcessingMetrics,
    ProcessingStatus,
    Resume,
)
from src.utils.logger import get_logger

from .base import BaseRepository

logger = get_logger(__name__)


class ResumeRepository(BaseRepository[Resume]):
    """Repository for resume document operations."""

    @property
    def collection_name(self) -> str:
        return "resumes"

    @property
    def model_class(self) -> type[Resume]:
        return Resume

    # -------------------------------------------------------------------------
    # Query Operations
    # -------------------------------------------------------------------------

    def get_by_file_hash(self, file_hash: str) -> Optional[Resume]:
        """Get a resume by its file hash (for deduplication)."""
        return self.find_one({"file.file_hash": file_hash})

    async def get_by_file_hash_async(self, file_hash: str) -> Optional[Resume]:
        """Get a resume by its file hash asynchronously."""
        return await self.find_one_async({"file.file_hash": file_hash})

    def file_hash_exists(self, file_hash: str) -> bool:
        """Check if a resume with the given file hash exists."""
        return self.exists({"file.file_hash": file_hash})

    async def file_hash_exists_async(self, file_hash: str) -> bool:
        """Check if a resume with the given file hash exists asynchronously."""
        return await self.exists_async({"file.file_hash": file_hash})

    def get_by_candidate(
        self,
        candidate_id: str | ObjectId,
        skip: int = 0,
        limit: int = 10,
    ) -> list[Resume]:
        """Get all resumes for a candidate."""
        return self.find(
            {"candidate_id": self._to_object_id(candidate_id)},
            skip=skip,
            limit=limit,
        )

    async def get_by_candidate_async(
        self,
        candidate_id: str | ObjectId,
        skip: int = 0,
        limit: int = 10,
    ) -> list[Resume]:
        """Get all resumes for a candidate asynchronously."""
        return await self.find_async(
            {"candidate_id": self._to_object_id(candidate_id)},
            skip=skip,
            limit=limit,
        )

    def get_by_status(
        self,
        status: ProcessingStatus,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Resume]:
        """Get resumes by processing status."""
        return self.find({"status": status.value}, skip=skip, limit=limit)

    async def get_by_status_async(
        self,
        status: ProcessingStatus,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Resume]:
        """Get resumes by processing status asynchronously."""
        return await self.find_async({"status": status.value}, skip=skip, limit=limit)

    def get_pending_resumes(self, limit: int = 50) -> list[Resume]:
        """Get resumes pending processing."""
        return self.find(
            {"status": ProcessingStatus.PENDING.value},
            limit=limit,
            sort_by="created_at",
            sort_order=1,  # Oldest first
        )

    async def get_pending_resumes_async(self, limit: int = 50) -> list[Resume]:
        """Get resumes pending processing asynchronously."""
        return await self.find_async(
            {"status": ProcessingStatus.PENDING.value},
            limit=limit,
            sort_by="created_at",
            sort_order=1,
        )

    def get_failed_resumes(self, limit: int = 50) -> list[Resume]:
        """Get resumes that failed processing."""
        return self.find(
            {"status": ProcessingStatus.FAILED.value},
            limit=limit,
            sort_by="created_at",
            sort_order=-1,
        )

    async def get_failed_resumes_async(self, limit: int = 50) -> list[Resume]:
        """Get resumes that failed processing asynchronously."""
        return await self.find_async(
            {"status": ProcessingStatus.FAILED.value},
            limit=limit,
            sort_by="created_at",
            sort_order=-1,
        )

    # -------------------------------------------------------------------------
    # Status Update Operations
    # -------------------------------------------------------------------------

    def update_status(
        self, id_value: str | ObjectId, status: ProcessingStatus
    ) -> Optional[Resume]:
        """Update resume processing status."""
        return self.update(id_value, {"status": status.value})

    async def update_status_async(
        self, id_value: str | ObjectId, status: ProcessingStatus
    ) -> Optional[Resume]:
        """Update resume processing status asynchronously."""
        return await self.update_async(id_value, {"status": status.value})

    def mark_processing(self, id_value: str | ObjectId) -> Optional[Resume]:
        """Mark resume as currently processing."""
        return self.update_status(id_value, ProcessingStatus.PROCESSING)

    async def mark_processing_async(self, id_value: str | ObjectId) -> Optional[Resume]:
        """Mark resume as currently processing asynchronously."""
        return await self.update_status_async(id_value, ProcessingStatus.PROCESSING)

    def mark_parsed(
        self,
        id_value: str | ObjectId,
        parsed_content: ParsedContent,
        metrics: Optional[ProcessingMetrics] = None,
    ) -> Optional[Resume]:
        """Mark resume as parsed and store parsed content."""
        update_data: dict[str, Any] = {
            "status": ProcessingStatus.PARSED.value,
            "parsed_content": parsed_content.model_dump(),
        }
        if metrics:
            update_data["processing_metrics"] = metrics.model_dump()
        return self.update(id_value, update_data)

    async def mark_parsed_async(
        self,
        id_value: str | ObjectId,
        parsed_content: ParsedContent,
        metrics: Optional[ProcessingMetrics] = None,
    ) -> Optional[Resume]:
        """Mark resume as parsed and store parsed content asynchronously."""
        update_data: dict[str, Any] = {
            "status": ProcessingStatus.PARSED.value,
            "parsed_content": parsed_content.model_dump(),
        }
        if metrics:
            update_data["processing_metrics"] = metrics.model_dump()
        return await self.update_async(id_value, update_data)

    def mark_embedded(
        self,
        id_value: str | ObjectId,
        vector_ids: list[str],
    ) -> Optional[Resume]:
        """Mark resume as embedded and store vector IDs."""
        return self.update(
            id_value,
            {
                "status": ProcessingStatus.EMBEDDED.value,
                "vector_ids": vector_ids,
            },
        )

    async def mark_embedded_async(
        self,
        id_value: str | ObjectId,
        vector_ids: list[str],
    ) -> Optional[Resume]:
        """Mark resume as embedded and store vector IDs asynchronously."""
        return await self.update_async(
            id_value,
            {
                "status": ProcessingStatus.EMBEDDED.value,
                "vector_ids": vector_ids,
            },
        )

    def mark_completed(
        self,
        id_value: str | ObjectId,
        quality_score: Optional[float] = None,
    ) -> Optional[Resume]:
        """Mark resume processing as completed."""
        update_data: dict[str, Any] = {"status": ProcessingStatus.COMPLETED.value}
        if quality_score is not None:
            update_data["parse_quality_score"] = quality_score
        return self.update(id_value, update_data)

    async def mark_completed_async(
        self,
        id_value: str | ObjectId,
        quality_score: Optional[float] = None,
    ) -> Optional[Resume]:
        """Mark resume processing as completed asynchronously."""
        update_data: dict[str, Any] = {"status": ProcessingStatus.COMPLETED.value}
        if quality_score is not None:
            update_data["parse_quality_score"] = quality_score
        return await self.update_async(id_value, update_data)

    def mark_failed(
        self,
        id_value: str | ObjectId,
        error_stage: str,
        error_type: str,
        error_message: str,
    ) -> Optional[Resume]:
        """Mark resume as failed with error details."""
        collection = self._get_sync_collection()
        error = {
            "stage": error_stage,
            "error_type": error_type,
            "error_message": error_message,
            "occurred_at": datetime.now(timezone.utc),
            "is_recoverable": False,
        }
        collection.update_one(
            {"_id": self._to_object_id(id_value)},
            {
                "$set": {
                    "status": ProcessingStatus.FAILED.value,
                    "updated_at": datetime.now(timezone.utc),
                },
                "$push": {"processing_errors": error},
            },
        )
        return self.get_by_id(id_value)

    async def mark_failed_async(
        self,
        id_value: str | ObjectId,
        error_stage: str,
        error_type: str,
        error_message: str,
    ) -> Optional[Resume]:
        """Mark resume as failed with error details asynchronously."""
        collection = self._get_async_collection()
        error = {
            "stage": error_stage,
            "error_type": error_type,
            "error_message": error_message,
            "occurred_at": datetime.now(timezone.utc),
            "is_recoverable": False,
        }
        await collection.update_one(
            {"_id": self._to_object_id(id_value)},
            {
                "$set": {
                    "status": ProcessingStatus.FAILED.value,
                    "updated_at": datetime.now(timezone.utc),
                },
                "$push": {"processing_errors": error},
            },
        )
        return await self.get_by_id_async(id_value)

    # -------------------------------------------------------------------------
    # Candidate Linking
    # -------------------------------------------------------------------------

    def link_candidate(
        self, resume_id: str | ObjectId, candidate_id: str | ObjectId
    ) -> Optional[Resume]:
        """Link a resume to a candidate."""
        return self.update(
            resume_id, {"candidate_id": self._to_object_id(candidate_id)}
        )

    async def link_candidate_async(
        self, resume_id: str | ObjectId, candidate_id: str | ObjectId
    ) -> Optional[Resume]:
        """Link a resume to a candidate asynchronously."""
        return await self.update_async(
            resume_id, {"candidate_id": self._to_object_id(candidate_id)}
        )

    # -------------------------------------------------------------------------
    # Aggregation Operations
    # -------------------------------------------------------------------------

    def get_status_counts(self) -> dict[str, int]:
        """Get count of resumes by processing status."""
        collection = self._get_sync_collection()
        pipeline = [
            {"$group": {"_id": "$status", "count": {"$sum": 1}}},
        ]
        results = list(collection.aggregate(pipeline))
        return {r["_id"]: r["count"] for r in results}

    async def get_status_counts_async(self) -> dict[str, int]:
        """Get count of resumes by processing status asynchronously."""
        collection = self._get_async_collection()
        pipeline = [
            {"$group": {"_id": "$status", "count": {"$sum": 1}}},
        ]
        results = await collection.aggregate(pipeline).to_list(length=None)
        return {r["_id"]: r["count"] for r in results}

    def get_format_distribution(self) -> dict[str, int]:
        """Get distribution of resume file formats."""
        collection = self._get_sync_collection()
        pipeline = [
            {"$group": {"_id": "$file.file_format", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]
        results = list(collection.aggregate(pipeline))
        return {r["_id"]: r["count"] for r in results}

    async def get_format_distribution_async(self) -> dict[str, int]:
        """Get distribution of resume file formats asynchronously."""
        collection = self._get_async_collection()
        pipeline = [
            {"$group": {"_id": "$file.file_format", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]
        results = await collection.aggregate(pipeline).to_list(length=None)
        return {r["_id"]: r["count"] for r in results}

    def get_processing_stats(self) -> dict[str, Any]:
        """Get processing statistics for resumes."""
        collection = self._get_sync_collection()
        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "total": {"$sum": 1},
                    "completed": {
                        "$sum": {
                            "$cond": [
                                {"$eq": ["$status", ProcessingStatus.COMPLETED.value]},
                                1,
                                0,
                            ]
                        }
                    },
                    "failed": {
                        "$sum": {
                            "$cond": [
                                {"$eq": ["$status", ProcessingStatus.FAILED.value]},
                                1,
                                0,
                            ]
                        }
                    },
                    "pending": {
                        "$sum": {
                            "$cond": [
                                {"$eq": ["$status", ProcessingStatus.PENDING.value]},
                                1,
                                0,
                            ]
                        }
                    },
                    "avg_quality_score": {"$avg": "$parse_quality_score"},
                }
            }
        ]
        results = list(collection.aggregate(pipeline))
        if results:
            return results[0]
        return {"total": 0, "completed": 0, "failed": 0, "pending": 0}

    async def get_processing_stats_async(self) -> dict[str, Any]:
        """Get processing statistics for resumes asynchronously."""
        collection = self._get_async_collection()
        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "total": {"$sum": 1},
                    "completed": {
                        "$sum": {
                            "$cond": [
                                {"$eq": ["$status", ProcessingStatus.COMPLETED.value]},
                                1,
                                0,
                            ]
                        }
                    },
                    "failed": {
                        "$sum": {
                            "$cond": [
                                {"$eq": ["$status", ProcessingStatus.FAILED.value]},
                                1,
                                0,
                            ]
                        }
                    },
                    "pending": {
                        "$sum": {
                            "$cond": [
                                {"$eq": ["$status", ProcessingStatus.PENDING.value]},
                                1,
                                0,
                            ]
                        }
                    },
                    "avg_quality_score": {"$avg": "$parse_quality_score"},
                }
            }
        ]
        results = await collection.aggregate(pipeline).to_list(length=1)
        if results:
            return results[0]
        return {"total": 0, "completed": 0, "failed": 0, "pending": 0}


# Singleton instance
_resume_repository: Optional[ResumeRepository] = None


def get_resume_repository() -> ResumeRepository:
    """Get the resume repository singleton instance."""
    global _resume_repository
    if _resume_repository is None:
        _resume_repository = ResumeRepository()
    return _resume_repository
