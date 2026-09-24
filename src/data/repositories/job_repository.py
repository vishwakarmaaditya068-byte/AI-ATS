"""
Job repository for AI-ATS.

Provides data access operations for job posting documents,
including search, filtering, and analytics capabilities.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Optional

from bson import ObjectId

from src.data.models.job import (
    EmploymentType,
    ExperienceLevel,
    Job,
    JobCreate,
    JobMetadata,
    JobUpdate,
    Location,
    ScoringWeights,
    WorkLocation,
)
from src.utils.constants import JobStatus
from src.utils.logger import get_logger

from .base import BaseRepository

logger = get_logger(__name__)


class JobRepository(BaseRepository[Job]):
    """Repository for job posting document operations."""

    @property
    def collection_name(self) -> str:
        return "jobs"

    @property
    def model_class(self) -> type[Job]:
        return Job

    # -------------------------------------------------------------------------
    # Create Operations
    # -------------------------------------------------------------------------

    def create_from_schema(self, data: "JobCreate") -> "Job":
        """Create a job from a create schema and auto-embed it (non-fatal)."""
        job: Job = Job(
            title=data.title,
            description=data.description,
            responsibilities=data.responsibilities,
            benefits=data.benefits,
            company_name=data.company_name,
            company_description=data.company_description,
            employment_type=data.employment_type,
            work_location=data.work_location,
            location=data.location or Location(),
            salary=data.salary,
            experience_level=data.experience_level,
            skill_requirements=data.skill_requirements,
            education_requirement=data.education_requirement,
            experience_requirement=data.experience_requirement,
            certifications_required=data.certifications_required,
            languages_required=data.languages_required,
            closing_date=data.closing_date,
            target_hire_date=data.target_hire_date,
            positions_available=data.positions_available,
            scoring_weights=data.scoring_weights or ScoringWeights.from_defaults(),
            metadata=data.metadata or JobMetadata(),
        )
        created_job: Job = self.create(job)

        # Auto-embed (non-fatal — never blocks job creation)
        try:
            from src.ml.embeddings.embedding_service import EmbeddingService
            emb_svc: EmbeddingService = EmbeddingService(repo=self)
            emb_svc.embed_job(str(created_job.id), created_job)
        except Exception as exc:
            logger.warning(f"Job embedding failed (non-fatal) for '{created_job.title}': {exc}")

        return created_job

    async def create_from_schema_async(self, data: JobCreate) -> Job:
        """Create a job from a create schema asynchronously."""
        job = Job(
            title=data.title,
            description=data.description,
            responsibilities=data.responsibilities,
            benefits=data.benefits,
            company_name=data.company_name,
            company_description=data.company_description,
            employment_type=data.employment_type,
            work_location=data.work_location,
            location=data.location or Location(),
            salary=data.salary,
            experience_level=data.experience_level,
            skill_requirements=data.skill_requirements,
            education_requirement=data.education_requirement,
            experience_requirement=data.experience_requirement,
            certifications_required=data.certifications_required,
            languages_required=data.languages_required,
            closing_date=data.closing_date,
            target_hire_date=data.target_hire_date,
            positions_available=data.positions_available,
            scoring_weights=data.scoring_weights or ScoringWeights.from_defaults(),
            metadata=data.metadata or JobMetadata(),
        )
        return await self.create_async(job)

    # -------------------------------------------------------------------------
    # Update Operations
    # -------------------------------------------------------------------------

    def update_from_schema(
        self, id_value: str | ObjectId, data: JobUpdate
    ) -> Optional[Job]:
        """Update a job from an update schema."""
        update_data = data.model_dump(exclude_unset=True, exclude_none=True)
        if not update_data:
            return self.get_by_id(id_value)
        return self.update(id_value, update_data)

    async def update_from_schema_async(
        self, id_value: str | ObjectId, data: JobUpdate
    ) -> Optional[Job]:
        """Update a job from an update schema asynchronously."""
        update_data = data.model_dump(exclude_unset=True, exclude_none=True)
        if not update_data:
            return await self.get_by_id_async(id_value)
        return await self.update_async(id_value, update_data)

    def update_status(
        self, id_value: str | ObjectId, status: JobStatus
    ) -> Optional[Job]:
        """Update job status."""
        return self.update(id_value, {"status": status.value})

    async def update_status_async(
        self, id_value: str | ObjectId, status: JobStatus
    ) -> Optional[Job]:
        """Update job status asynchronously."""
        return await self.update_async(id_value, {"status": status.value})

    def publish(self, id_value: str | ObjectId) -> Optional[Job]:
        """Publish a job posting (set to OPEN status)."""
        return self.update(
            id_value,
            {
                "status": JobStatus.OPEN.value,
                "posted_date": datetime.now(timezone.utc),
            },
        )

    async def publish_async(self, id_value: str | ObjectId) -> Optional[Job]:
        """Publish a job posting asynchronously."""
        return await self.update_async(
            id_value,
            {
                "status": JobStatus.OPEN.value,
                "posted_date": datetime.now(timezone.utc),
            },
        )

    def close(self, id_value: str | ObjectId) -> Optional[Job]:
        """Close a job posting."""
        return self.update_status(id_value, JobStatus.CLOSED)

    async def close_async(self, id_value: str | ObjectId) -> Optional[Job]:
        """Close a job posting asynchronously."""
        return await self.update_status_async(id_value, JobStatus.CLOSED)

    def mark_filled(self, id_value: str | ObjectId) -> Optional[Job]:
        """Mark a job as filled."""
        return self.update_status(id_value, JobStatus.FILLED)

    async def mark_filled_async(self, id_value: str | ObjectId) -> Optional[Job]:
        """Mark a job as filled asynchronously."""
        return await self.update_status_async(id_value, JobStatus.FILLED)

    # -------------------------------------------------------------------------
    # Query Operations
    # -------------------------------------------------------------------------

    def get_by_status(
        self,
        status: JobStatus,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Job]:
        """Get jobs by status."""
        return self.find({"status": status.value}, skip=skip, limit=limit)

    async def get_by_status_async(
        self,
        status: JobStatus,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Job]:
        """Get jobs by status asynchronously."""
        return await self.find_async({"status": status.value}, skip=skip, limit=limit)

    def get_open_jobs(self, skip: int = 0, limit: int = 100) -> list[Job]:
        """Get all open job postings."""
        return self.find(
            {
                "status": JobStatus.OPEN.value,
                "$or": [
                    {"closing_date": None},
                    {"closing_date": {"$gte": date.today().isoformat()}},
                ],
            },
            skip=skip,
            limit=limit,
            sort_by="posted_date",
            sort_order=-1,
        )

    async def get_open_jobs_async(self, skip: int = 0, limit: int = 100) -> list[Job]:
        """Get all open job postings asynchronously."""
        return await self.find_async(
            {
                "status": JobStatus.OPEN.value,
                "$or": [
                    {"closing_date": None},
                    {"closing_date": {"$gte": date.today().isoformat()}},
                ],
            },
            skip=skip,
            limit=limit,
            sort_by="posted_date",
            sort_order=-1,
        )

    def get_by_company(
        self,
        company_name: str,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Job]:
        """Get jobs by company name."""
        return self.find(
            {"company_name": {"$regex": company_name, "$options": "i"}},
            skip=skip,
            limit=limit,
        )

    async def get_by_company_async(
        self,
        company_name: str,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Job]:
        """Get jobs by company name asynchronously."""
        return await self.find_async(
            {"company_name": {"$regex": company_name, "$options": "i"}},
            skip=skip,
            limit=limit,
        )

    def get_by_tags(
        self,
        tags: list[str],
        match_all: bool = False,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Job]:
        """Get jobs by metadata tags."""
        if match_all:
            query = {"metadata.tags": {"$all": tags}}
        else:
            query = {"metadata.tags": {"$in": tags}}
        return self.find(query, skip=skip, limit=limit)

    async def get_by_tags_async(
        self,
        tags: list[str],
        match_all: bool = False,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Job]:
        """Get jobs by metadata tags asynchronously."""
        if match_all:
            query = {"metadata.tags": {"$all": tags}}
        else:
            query = {"metadata.tags": {"$in": tags}}
        return await self.find_async(query, skip=skip, limit=limit)

    # -------------------------------------------------------------------------
    # Search Operations
    # -------------------------------------------------------------------------

    def search(
        self,
        query_text: Optional[str] = None,
        status: Optional[JobStatus] = None,
        company_name: Optional[str] = None,
        employment_type: Optional[EmploymentType] = None,
        work_location: Optional[WorkLocation] = None,
        experience_level: Optional[ExperienceLevel] = None,
        skills: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Job]:
        """
        Search jobs with multiple filters.

        Args:
            query_text: Text to search in title, description
            status: Filter by job status
            company_name: Filter by company name (partial match)
            employment_type: Filter by employment type
            work_location: Filter by work location type
            experience_level: Filter by required experience level
            skills: Filter by required skills (matches any)
            tags: Filter by metadata tags (matches any)
            skip: Number of documents to skip
            limit: Maximum documents to return

        Returns:
            List of matching jobs
        """
        query: dict[str, Any] = {}

        if query_text:
            query["$or"] = [
                {"title": {"$regex": query_text, "$options": "i"}},
                {"description": {"$regex": query_text, "$options": "i"}},
            ]

        if status:
            query["status"] = status.value

        if company_name:
            query["company_name"] = {"$regex": company_name, "$options": "i"}

        if employment_type:
            query["employment_type"] = employment_type.value

        if work_location:
            query["work_location"] = work_location.value

        if experience_level:
            query["experience_level"] = experience_level.value

        if skills:
            normalized_skills = [s.lower() for s in skills]
            query["skill_requirements.name"] = {"$in": normalized_skills}

        if tags:
            query["metadata.tags"] = {"$in": tags}

        return self.find(query, skip=skip, limit=limit)

    async def search_async(
        self,
        query_text: Optional[str] = None,
        status: Optional[JobStatus] = None,
        company_name: Optional[str] = None,
        employment_type: Optional[EmploymentType] = None,
        work_location: Optional[WorkLocation] = None,
        experience_level: Optional[ExperienceLevel] = None,
        skills: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Job]:
        """Search jobs with multiple filters asynchronously."""
        query: dict[str, Any] = {}

        if query_text:
            query["$or"] = [
                {"title": {"$regex": query_text, "$options": "i"}},
                {"description": {"$regex": query_text, "$options": "i"}},
            ]

        if status:
            query["status"] = status.value

        if company_name:
            query["company_name"] = {"$regex": company_name, "$options": "i"}

        if employment_type:
            query["employment_type"] = employment_type.value

        if work_location:
            query["work_location"] = work_location.value

        if experience_level:
            query["experience_level"] = experience_level.value

        if skills:
            normalized_skills = [s.lower() for s in skills]
            query["skill_requirements.name"] = {"$in": normalized_skills}

        if tags:
            query["metadata.tags"] = {"$in": tags}

        return await self.find_async(query, skip=skip, limit=limit)

    # -------------------------------------------------------------------------
    # Aggregation Operations
    # -------------------------------------------------------------------------

    def get_status_counts(self) -> dict[str, int]:
        """Get count of jobs by status."""
        collection = self._get_sync_collection()
        pipeline = [
            {"$group": {"_id": "$status", "count": {"$sum": 1}}},
        ]
        results = list(collection.aggregate(pipeline))
        return {r["_id"]: r["count"] for r in results}

    async def get_status_counts_async(self) -> dict[str, int]:
        """Get count of jobs by status asynchronously."""
        collection = self._get_async_collection()
        pipeline = [
            {"$group": {"_id": "$status", "count": {"$sum": 1}}},
        ]
        results = await collection.aggregate(pipeline).to_list(length=None)
        return {r["_id"]: r["count"] for r in results}

    def get_company_distribution(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get job count by company."""
        collection = self._get_sync_collection()
        pipeline = [
            {"$group": {"_id": "$company_name", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": limit},
        ]
        return list(collection.aggregate(pipeline))

    async def get_company_distribution_async(
        self, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Get job count by company asynchronously."""
        collection = self._get_async_collection()
        pipeline = [
            {"$group": {"_id": "$company_name", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": limit},
        ]
        return await collection.aggregate(pipeline).to_list(length=limit)

    def get_skill_demand(self, limit: int = 30) -> list[dict[str, Any]]:
        """Get most in-demand skills across all job postings."""
        collection = self._get_sync_collection()
        pipeline = [
            {"$match": {"status": JobStatus.OPEN.value}},
            {"$unwind": "$skill_requirements"},
            {
                "$group": {
                    "_id": "$skill_requirements.name",
                    "count": {"$sum": 1},
                    "required_count": {
                        "$sum": {
                            "$cond": ["$skill_requirements.is_required", 1, 0]
                        }
                    },
                }
            },
            {"$sort": {"count": -1}},
            {"$limit": limit},
        ]
        return list(collection.aggregate(pipeline))

    async def get_skill_demand_async(self, limit: int = 30) -> list[dict[str, Any]]:
        """Get most in-demand skills across all job postings asynchronously."""
        collection = self._get_async_collection()
        pipeline = [
            {"$match": {"status": JobStatus.OPEN.value}},
            {"$unwind": "$skill_requirements"},
            {
                "$group": {
                    "_id": "$skill_requirements.name",
                    "count": {"$sum": 1},
                    "required_count": {
                        "$sum": {
                            "$cond": ["$skill_requirements.is_required", 1, 0]
                        }
                    },
                }
            },
            {"$sort": {"count": -1}},
            {"$limit": limit},
        ]
        return await collection.aggregate(pipeline).to_list(length=limit)

    def get_experience_level_distribution(self) -> dict[str, int]:
        """Get distribution of jobs by experience level."""
        collection = self._get_sync_collection()
        pipeline = [
            {"$group": {"_id": "$experience_level", "count": {"$sum": 1}}},
        ]
        results = list(collection.aggregate(pipeline))
        return {r["_id"]: r["count"] for r in results}

    async def get_experience_level_distribution_async(self) -> dict[str, int]:
        """Get distribution of jobs by experience level asynchronously."""
        collection = self._get_async_collection()
        pipeline = [
            {"$group": {"_id": "$experience_level", "count": {"$sum": 1}}},
        ]
        results = await collection.aggregate(pipeline).to_list(length=None)
        return {r["_id"]: r["count"] for r in results}

    # -------------------------------------------------------------------------
    # Metrics Operations
    # -------------------------------------------------------------------------

    def increment_views(self, id_value: str | ObjectId) -> None:
        """Increment the view count for a job."""
        collection = self._get_sync_collection()
        collection.update_one(
            {"_id": self._to_object_id(id_value)},
            {"$inc": {"metadata.views_count": 1}},
        )

    async def increment_views_async(self, id_value: str | ObjectId) -> None:
        """Increment the view count for a job asynchronously."""
        collection = self._get_async_collection()
        await collection.update_one(
            {"_id": self._to_object_id(id_value)},
            {"$inc": {"metadata.views_count": 1}},
        )

    def increment_applications(self, id_value: str | ObjectId) -> None:
        """Increment the application count for a job."""
        collection = self._get_sync_collection()
        collection.update_one(
            {"_id": self._to_object_id(id_value)},
            {"$inc": {"metadata.applications_count": 1}},
        )

    async def increment_applications_async(self, id_value: str | ObjectId) -> None:
        """Increment the application count for a job asynchronously."""
        collection = self._get_async_collection()
        await collection.update_one(
            {"_id": self._to_object_id(id_value)},
            {"$inc": {"metadata.applications_count": 1}},
        )

    def set_embedding_id(
        self, id_value: str | ObjectId, embedding_id: str
    ) -> Optional[Job]:
        """Set the vector embedding ID for a job."""
        return self.update(id_value, {"embedding_id": embedding_id})

    async def set_embedding_id_async(
        self, id_value: str | ObjectId, embedding_id: str
    ) -> Optional[Job]:
        """Set the vector embedding ID for a job asynchronously."""
        return await self.update_async(id_value, {"embedding_id": embedding_id})


# Singleton instance
_job_repository: Optional[JobRepository] = None


def get_job_repository() -> JobRepository:
    """Get the job repository singleton instance."""
    global _job_repository
    if _job_repository is None:
        _job_repository = JobRepository()
    return _job_repository
