"""
Candidate repository for AI-ATS.

Provides data access operations for candidate documents,
including search and filtering capabilities.
"""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.ml.nlp.accurate_resume_parser import ParsedResume

from bson import ObjectId

from src.data.models.candidate import (
    Candidate,
    CandidateCreate,
    CandidateMetadata,
    CandidateUpdate,
)
from src.utils.constants import CandidateStatus
from src.utils.logger import get_logger

from .base import BaseRepository

logger = get_logger(__name__)


class CandidateRepository(BaseRepository[Candidate]):
    """Repository for candidate document operations."""

    @property
    def collection_name(self) -> str:
        return "candidates"

    @property
    def model_class(self) -> type[Candidate]:
        return Candidate

    # -------------------------------------------------------------------------
    # Create Operations
    # -------------------------------------------------------------------------

    def create_from_schema(self, data: CandidateCreate) -> Candidate:
        """Create a candidate from a create schema."""
        candidate = Candidate(
            first_name=data.first_name,
            last_name=data.last_name,
            contact=data.contact,
            headline=data.headline,
            summary=data.summary,
            skills=data.skills,
            work_experience=data.work_experience,
            education=data.education,
            certifications=data.certifications,
            languages=data.languages,
            metadata=data.metadata or CandidateMetadata(),
        )
        return self.create(candidate)

    async def create_from_schema_async(self, data: CandidateCreate) -> Candidate:
        """Create a candidate from a create schema asynchronously."""
        candidate = Candidate(
            first_name=data.first_name,
            last_name=data.last_name,
            contact=data.contact,
            headline=data.headline,
            summary=data.summary,
            skills=data.skills,
            work_experience=data.work_experience,
            education=data.education,
            certifications=data.certifications,
            languages=data.languages,
            metadata=data.metadata or CandidateMetadata(),
        )
        return await self.create_async(candidate)

    # -------------------------------------------------------------------------
    # Update Operations
    # -------------------------------------------------------------------------

    def update_from_schema(
        self, id_value: str | ObjectId, data: CandidateUpdate
    ) -> Optional[Candidate]:
        """Update a candidate from an update schema."""
        update_data = data.model_dump(exclude_unset=True, exclude_none=True)
        if not update_data:
            return self.get_by_id(id_value)
        return self.update(id_value, update_data)

    async def update_from_schema_async(
        self, id_value: str | ObjectId, data: CandidateUpdate
    ) -> Optional[Candidate]:
        """Update a candidate from an update schema asynchronously."""
        update_data = data.model_dump(exclude_unset=True, exclude_none=True)
        if not update_data:
            return await self.get_by_id_async(id_value)
        return await self.update_async(id_value, update_data)

    def update_status(
        self, id_value: str | ObjectId, status: CandidateStatus
    ) -> Optional[Candidate]:
        """Update candidate status."""
        return self.update(id_value, {"status": status.value})

    async def update_status_async(
        self, id_value: str | ObjectId, status: CandidateStatus
    ) -> Optional[Candidate]:
        """Update candidate status asynchronously."""
        return await self.update_async(id_value, {"status": status.value})

    # -------------------------------------------------------------------------
    # Query Operations
    # -------------------------------------------------------------------------

    def get_by_email(self, email: str) -> Optional[Candidate]:
        """Get a candidate by email address."""
        return self.find_one({"contact.email": email.lower()})

    async def get_by_email_async(self, email: str) -> Optional[Candidate]:
        """Get a candidate by email address asynchronously."""
        return await self.find_one_async({"contact.email": email.lower()})

    def email_exists(self, email: str) -> bool:
        """Check if a candidate with the given email exists."""
        return self.exists({"contact.email": email.lower()})

    async def email_exists_async(self, email: str) -> bool:
        """Check if a candidate with the given email exists asynchronously."""
        return await self.exists_async({"contact.email": email.lower()})

    def get_by_status(
        self,
        status: CandidateStatus,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Candidate]:
        """Get candidates by status."""
        return self.find({"status": status.value}, skip=skip, limit=limit)

    async def get_by_status_async(
        self,
        status: CandidateStatus,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Candidate]:
        """Get candidates by status asynchronously."""
        return await self.find_async({"status": status.value}, skip=skip, limit=limit)

    def get_by_tags(
        self,
        tags: list[str],
        match_all: bool = False,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Candidate]:
        """Get candidates by metadata tags."""
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
    ) -> list[Candidate]:
        """Get candidates by metadata tags asynchronously."""
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
        status: Optional[CandidateStatus] = None,
        skills: Optional[list[str]] = None,
        min_experience_years: Optional[float] = None,
        tags: Optional[list[str]] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Candidate]:
        """
        Search candidates with multiple filters.

        Args:
            query_text: Text to search in name, headline, summary
            status: Filter by candidate status
            skills: Filter by skill names (matches any)
            min_experience_years: Minimum years of experience
            tags: Filter by metadata tags (matches any)
            skip: Number of documents to skip
            limit: Maximum documents to return

        Returns:
            List of matching candidates
        """
        query: dict[str, Any] = {}

        if query_text:
            query["$or"] = [
                {"first_name": {"$regex": query_text, "$options": "i"}},
                {"last_name": {"$regex": query_text, "$options": "i"}},
                {"headline": {"$regex": query_text, "$options": "i"}},
                {"summary": {"$regex": query_text, "$options": "i"}},
            ]

        if status:
            query["status"] = status.value

        if skills:
            normalized_skills = [s.lower() for s in skills]
            query["skills.name"] = {"$in": normalized_skills}

        if tags:
            query["metadata.tags"] = {"$in": tags}

        # Fast path: no experience filter — simple indexed find with correct pagination.
        if min_experience_years is None:
            return self.find(query, skip=skip, limit=limit)

        # Aggregation path: compute experience server-side so that $skip/$limit
        # are applied AFTER filtering, giving correct page sizes.
        #
        # Logic mirrors the Python property total_experience_years:
        #   sum(max(1, (end - start).days // 30) for each entry with a start_date)
        #
        # $subtract on two BSON Date values returns milliseconds.
        # 2_592_000_000 ms = 30 days (matches the Python `delta.days // 30` divisor).
        collection = self._get_sync_collection()
        pipeline: list[dict[str, Any]] = []

        if query:
            pipeline.append({"$match": query})

        pipeline.append({
            "$addFields": {
                "_total_exp_months": {
                    "$reduce": {
                        "input": "$work_experience",
                        "initialValue": 0,
                        "in": {
                            "$add": [
                                "$$value",
                                {
                                    "$cond": {
                                        "if": {"$gt": ["$$this.start_date", None]},
                                        "then": {
                                            "$max": [
                                                1,
                                                {
                                                    "$floor": {
                                                        "$divide": [
                                                            {
                                                                "$subtract": [
                                                                    {"$ifNull": ["$$this.end_date", "$$NOW"]},
                                                                    "$$this.start_date",
                                                                ]
                                                            },
                                                            2_592_000_000,  # ms in 30 days
                                                        ]
                                                    }
                                                },
                                            ]
                                        },
                                        "else": 0,
                                    }
                                },
                            ]
                        },
                    }
                }
            }
        })

        pipeline.append({
            "$match": {"_total_exp_months": {"$gte": min_experience_years * 12}}
        })
        pipeline.append({"$sort": {"created_at": -1}})
        pipeline.append({"$unset": "_total_exp_months"})
        if skip:
            pipeline.append({"$skip": skip})
        pipeline.append({"$limit": limit})

        docs = list(collection.aggregate(pipeline))
        return [Candidate.model_validate(doc) for doc in docs]

    async def search_async(
        self,
        query_text: Optional[str] = None,
        status: Optional[CandidateStatus] = None,
        skills: Optional[list[str]] = None,
        min_experience_years: Optional[float] = None,
        tags: Optional[list[str]] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Candidate]:
        """Search candidates with multiple filters asynchronously."""
        query: dict[str, Any] = {}

        if query_text:
            query["$or"] = [
                {"first_name": {"$regex": query_text, "$options": "i"}},
                {"last_name": {"$regex": query_text, "$options": "i"}},
                {"headline": {"$regex": query_text, "$options": "i"}},
                {"summary": {"$regex": query_text, "$options": "i"}},
            ]

        if status:
            query["status"] = status.value

        if skills:
            normalized_skills = [s.lower() for s in skills]
            query["skills.name"] = {"$in": normalized_skills}

        if tags:
            query["metadata.tags"] = {"$in": tags}

        if min_experience_years is None:
            return await self.find_async(query, skip=skip, limit=limit)

        collection = self._get_async_collection()
        pipeline: list[dict[str, Any]] = []

        if query:
            pipeline.append({"$match": query})

        pipeline.append({
            "$addFields": {
                "_total_exp_months": {
                    "$reduce": {
                        "input": "$work_experience",
                        "initialValue": 0,
                        "in": {
                            "$add": [
                                "$$value",
                                {
                                    "$cond": {
                                        "if": {"$gt": ["$$this.start_date", None]},
                                        "then": {
                                            "$max": [
                                                1,
                                                {
                                                    "$floor": {
                                                        "$divide": [
                                                            {
                                                                "$subtract": [
                                                                    {"$ifNull": ["$$this.end_date", "$$NOW"]},
                                                                    "$$this.start_date",
                                                                ]
                                                            },
                                                            2_592_000_000,
                                                        ]
                                                    }
                                                },
                                            ]
                                        },
                                        "else": 0,
                                    }
                                },
                            ]
                        },
                    }
                }
            }
        })

        pipeline.append({
            "$match": {"_total_exp_months": {"$gte": min_experience_years * 12}}
        })
        pipeline.append({"$sort": {"created_at": -1}})
        pipeline.append({"$unset": "_total_exp_months"})
        if skip:
            pipeline.append({"$skip": skip})
        pipeline.append({"$limit": limit})

        docs = await collection.aggregate(pipeline).to_list(length=limit)
        return [Candidate.model_validate(doc) for doc in docs]

    # -------------------------------------------------------------------------
    # Aggregation Operations
    # -------------------------------------------------------------------------

    def get_status_counts(self) -> dict[str, int]:
        """Get count of candidates by status."""
        collection = self._get_sync_collection()
        pipeline = [
            {"$group": {"_id": "$status", "count": {"$sum": 1}}},
        ]
        results = list(collection.aggregate(pipeline))
        return {r["_id"]: r["count"] for r in results}

    async def get_status_counts_async(self) -> dict[str, int]:
        """Get count of candidates by status asynchronously."""
        collection = self._get_async_collection()
        pipeline = [
            {"$group": {"_id": "$status", "count": {"$sum": 1}}},
        ]
        results = await collection.aggregate(pipeline).to_list(length=None)
        return {r["_id"]: r["count"] for r in results}

    def get_skill_distribution(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get most common skills across all candidates."""
        collection = self._get_sync_collection()
        pipeline = [
            {"$unwind": "$skills"},
            {"$group": {"_id": "$skills.name", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": limit},
        ]
        return list(collection.aggregate(pipeline))

    async def get_skill_distribution_async(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get most common skills across all candidates asynchronously."""
        collection = self._get_async_collection()
        pipeline = [
            {"$unwind": "$skills"},
            {"$group": {"_id": "$skills.name", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": limit},
        ]
        return await collection.aggregate(pipeline).to_list(length=limit)

    # -------------------------------------------------------------------------
    # Resume Linking
    # -------------------------------------------------------------------------

    def link_resume(
        self, candidate_id: str | ObjectId, resume_id: str | ObjectId
    ) -> Optional[Candidate]:
        """Link a resume to a candidate."""
        return self.update(candidate_id, {"resume_id": self._to_object_id(resume_id)})

    async def link_resume_async(
        self, candidate_id: str | ObjectId, resume_id: str | ObjectId
    ) -> Optional[Candidate]:
        """Link a resume to a candidate asynchronously."""
        return await self.update_async(
            candidate_id, {"resume_id": self._to_object_id(resume_id)}
        )

    def set_embedding_id(
        self, candidate_id: str | ObjectId, embedding_id: str
    ) -> Optional[Candidate]:
        """Set the vector embedding ID for a candidate."""
        return self.update(candidate_id, {"embedding_id": embedding_id})

    async def set_embedding_id_async(
        self, candidate_id: str | ObjectId, embedding_id: str
    ) -> Optional[Candidate]:
        """Set the vector embedding ID for a candidate asynchronously."""
        return await self.update_async(candidate_id, {"embedding_id": embedding_id})

    # -------------------------------------------------------------------------
    # Ingestion helpers
    # -------------------------------------------------------------------------

    def hash_exists(self, file_hash: str) -> bool:
        """Return True if a candidate record carrying this SHA-256 hash already exists."""
        return self.exists({"file_hashes": file_hash})

    def upsert_by_email(
        self,
        parsed: "ParsedResume",
        file_hash: str,
        filename: str,
    ) -> Optional[Candidate]:
        """
        Insert or update a candidate from a ParsedResume.

        - If a candidate with the same email exists, refresh their skills /
          experience / education and append the new file hash.
        - Otherwise, create a fresh Candidate document.

        Returns the resulting Candidate (new or updated), or None on failure.
        """
        from src.ml.nlp.accurate_resume_parser import ParsedResume  # local import avoids circular
        from src.data.models.candidate import (
            ContactInfo as CandidateContactInfo,
            Skill,
            WorkExperience,
            Education,
        )

        contact = parsed.contact

        # ---- map parsed skills ------------------------------------------------
        skills: list[Skill] = [
            Skill(name=s, category=cat.category)
            for cat in parsed.skills
            for s in cat.skills
            if s.strip()
        ]

        # ---- map parsed work experience --------------------------------------
        work_exp: list[WorkExperience] = [
            WorkExperience(
                job_title=e.title or "Unknown",
                company=e.company or "Unknown",
                responsibilities=e.bullets,
            )
            for e in parsed.experience
        ]

        # ---- map parsed education --------------------------------------------
        education: list[Education] = [
            Education(
                degree=e.degree or "Unknown",
                field_of_study="",
                institution=e.institution or "Unknown",
            )
            for e in parsed.education
        ]

        # ---- build contact info (email is required by Pydantic EmailStr) -----
        raw_email: str = (contact.email or "").strip().lower()
        if not raw_email:
            raw_email = f"unknown_{file_hash[:8]}@imported.local"

        contact_info = CandidateContactInfo(
            email=raw_email,
            phone=contact.phone or None,
            linkedin_url=contact.linkedin or None,
            github_url=contact.github or None,
            portfolio_url=contact.portfolio or None,
        )

        # ---- update existing record if email already present -----------------
        existing: Optional[Candidate] = self.get_by_email(raw_email)
        if existing is not None:
            collection = self._get_sync_collection()
            collection.update_one(
                {"_id": existing.id},
                {
                    "$set": {
                        "skills": [s.model_dump() for s in skills],
                        "work_experience": [w.model_dump() for w in work_exp],
                        "education": [e.model_dump() for e in education],
                    },
                    "$addToSet": {"file_hashes": file_hash},
                },
            )
            return self.get_by_id(existing.id)

        # ---- create new candidate --------------------------------------------
        name_parts: list[str] = contact.name.split(None, 1) if contact.name else []
        first_name: str = name_parts[0] if name_parts else "Unknown"
        last_name: str = name_parts[1] if len(name_parts) > 1 else "Candidate"

        candidate = Candidate(
            first_name=first_name,
            last_name=last_name,
            contact=contact_info,
            summary=parsed.summary or "",
            skills=skills,
            work_experience=work_exp,
            education=education,
            metadata=CandidateMetadata(source=filename),
            file_hashes=[file_hash],
        )
        return self.create(candidate)


# Singleton instance
_candidate_repository: Optional[CandidateRepository] = None


def get_candidate_repository() -> CandidateRepository:
    """Get the candidate repository singleton instance."""
    global _candidate_repository
    if _candidate_repository is None:
        _candidate_repository = CandidateRepository()
    return _candidate_repository
