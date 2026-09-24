"""
Match repository for AI-ATS.

Provides data access operations for candidate-job match documents,
including ranking, scoring, and analytics capabilities.
"""

from typing import Any, Optional

from bson import ObjectId

from src.data.models.match import (
    Match,
    MatchStatus,
    MatchSummary,
    RecruiterFeedback,
)
from src.utils.constants import MatchScoreLevel
from src.utils.logger import get_logger

from .base import BaseRepository

logger = get_logger(__name__)


class MatchRepository(BaseRepository[Match]):
    """Repository for candidate-job match document operations."""

    @property
    def collection_name(self) -> str:
        return "matches"

    @property
    def model_class(self) -> type[Match]:
        return Match

    # -------------------------------------------------------------------------
    # Query Operations
    # -------------------------------------------------------------------------

    def get_by_candidate_and_job(
        self,
        candidate_id: str | ObjectId,
        job_id: str | ObjectId,
    ) -> Optional[Match]:
        """Get a specific match by candidate and job IDs."""
        return self.find_one(
            {
                "candidate_id": self._to_object_id(candidate_id),
                "job_id": self._to_object_id(job_id),
            }
        )

    async def get_by_candidate_and_job_async(
        self,
        candidate_id: str | ObjectId,
        job_id: str | ObjectId,
    ) -> Optional[Match]:
        """Get a specific match by candidate and job IDs asynchronously."""
        return await self.find_one_async(
            {
                "candidate_id": self._to_object_id(candidate_id),
                "job_id": self._to_object_id(job_id),
            }
        )

    def match_exists(
        self,
        candidate_id: str | ObjectId,
        job_id: str | ObjectId,
    ) -> bool:
        """Check if a match already exists."""
        return self.exists(
            {
                "candidate_id": self._to_object_id(candidate_id),
                "job_id": self._to_object_id(job_id),
            }
        )

    async def match_exists_async(
        self,
        candidate_id: str | ObjectId,
        job_id: str | ObjectId,
    ) -> bool:
        """Check if a match already exists asynchronously."""
        return await self.exists_async(
            {
                "candidate_id": self._to_object_id(candidate_id),
                "job_id": self._to_object_id(job_id),
            }
        )

    def get_by_job(
        self,
        job_id: str | ObjectId,
        skip: int = 0,
        limit: int = 100,
        sort_by_score: bool = True,
    ) -> list[Match]:
        """Get all matches for a job, optionally sorted by score."""
        query = {"job_id": self._to_object_id(job_id)}
        sort_field = "overall_score" if sort_by_score else "created_at"
        return self.find(
            query,
            skip=skip,
            limit=limit,
            sort_by=sort_field,
            sort_order=-1,
        )

    async def get_by_job_async(
        self,
        job_id: str | ObjectId,
        skip: int = 0,
        limit: int = 100,
        sort_by_score: bool = True,
    ) -> list[Match]:
        """Get all matches for a job asynchronously."""
        query = {"job_id": self._to_object_id(job_id)}
        sort_field = "overall_score" if sort_by_score else "created_at"
        return await self.find_async(
            query,
            skip=skip,
            limit=limit,
            sort_by=sort_field,
            sort_order=-1,
        )

    def get_by_candidate(
        self,
        candidate_id: str | ObjectId,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Match]:
        """Get all matches for a candidate."""
        return self.find(
            {"candidate_id": self._to_object_id(candidate_id)},
            skip=skip,
            limit=limit,
            sort_by="overall_score",
            sort_order=-1,
        )

    async def get_by_candidate_async(
        self,
        candidate_id: str | ObjectId,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Match]:
        """Get all matches for a candidate asynchronously."""
        return await self.find_async(
            {"candidate_id": self._to_object_id(candidate_id)},
            skip=skip,
            limit=limit,
            sort_by="overall_score",
            sort_order=-1,
        )

    def get_by_status(
        self,
        status: MatchStatus,
        job_id: Optional[str | ObjectId] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Match]:
        """Get matches by status, optionally filtered by job."""
        query: dict[str, Any] = {"status": status.value}
        if job_id:
            query["job_id"] = self._to_object_id(job_id)
        return self.find(query, skip=skip, limit=limit)

    async def get_by_status_async(
        self,
        status: MatchStatus,
        job_id: Optional[str | ObjectId] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Match]:
        """Get matches by status asynchronously."""
        query: dict[str, Any] = {"status": status.value}
        if job_id:
            query["job_id"] = self._to_object_id(job_id)
        return await self.find_async(query, skip=skip, limit=limit)

    def get_by_score_level(
        self,
        score_level: MatchScoreLevel,
        job_id: Optional[str | ObjectId] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Match]:
        """Get matches by score level."""
        query: dict[str, Any] = {"score_level": score_level.value}
        if job_id:
            query["job_id"] = self._to_object_id(job_id)
        return self.find(
            query,
            skip=skip,
            limit=limit,
            sort_by="overall_score",
            sort_order=-1,
        )

    async def get_by_score_level_async(
        self,
        score_level: MatchScoreLevel,
        job_id: Optional[str | ObjectId] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Match]:
        """Get matches by score level asynchronously."""
        query: dict[str, Any] = {"score_level": score_level.value}
        if job_id:
            query["job_id"] = self._to_object_id(job_id)
        return await self.find_async(
            query,
            skip=skip,
            limit=limit,
            sort_by="overall_score",
            sort_order=-1,
        )

    def get_top_matches(
        self,
        job_id: str | ObjectId,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[Match]:
        """Get top N matches for a job above a minimum score."""
        query = {
            "job_id": self._to_object_id(job_id),
            "overall_score": {"$gte": min_score},
        }
        return self.find(
            query,
            limit=limit,
            sort_by="overall_score",
            sort_order=-1,
        )

    async def get_top_matches_async(
        self,
        job_id: str | ObjectId,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[Match]:
        """Get top N matches for a job asynchronously."""
        query = {
            "job_id": self._to_object_id(job_id),
            "overall_score": {"$gte": min_score},
        }
        return await self.find_async(
            query,
            limit=limit,
            sort_by="overall_score",
            sort_order=-1,
        )

    def get_shortlisted(
        self,
        job_id: str | ObjectId,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Match]:
        """Get shortlisted matches for a job."""
        return self.get_by_status(MatchStatus.SHORTLISTED, job_id, skip, limit)

    async def get_shortlisted_async(
        self,
        job_id: str | ObjectId,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Match]:
        """Get shortlisted matches for a job asynchronously."""
        return await self.get_by_status_async(
            MatchStatus.SHORTLISTED, job_id, skip, limit
        )

    # -------------------------------------------------------------------------
    # Update Operations
    # -------------------------------------------------------------------------

    def update_status(
        self, id_value: str | ObjectId, status: MatchStatus
    ) -> Optional[Match]:
        """Update match status."""
        return self.update(id_value, {"status": status.value})

    async def update_status_async(
        self, id_value: str | ObjectId, status: MatchStatus
    ) -> Optional[Match]:
        """Update match status asynchronously."""
        return await self.update_async(id_value, {"status": status.value})

    def shortlist(self, id_value: str | ObjectId) -> Optional[Match]:
        """Shortlist a match."""
        return self.update_status(id_value, MatchStatus.SHORTLISTED)

    async def shortlist_async(self, id_value: str | ObjectId) -> Optional[Match]:
        """Shortlist a match asynchronously."""
        return await self.update_status_async(id_value, MatchStatus.SHORTLISTED)

    def reject(self, id_value: str | ObjectId) -> Optional[Match]:
        """Reject a match."""
        return self.update_status(id_value, MatchStatus.REJECTED)

    async def reject_async(self, id_value: str | ObjectId) -> Optional[Match]:
        """Reject a match asynchronously."""
        return await self.update_status_async(id_value, MatchStatus.REJECTED)

    def override_score(
        self,
        id_value: str | ObjectId,
        new_score: float,
        reason: str,
    ) -> Optional[Match]:
        """Manually override the match score."""
        return self.update(
            id_value,
            {
                "manual_score_override": new_score,
                "override_reason": reason,
            },
        )

    async def override_score_async(
        self,
        id_value: str | ObjectId,
        new_score: float,
        reason: str,
    ) -> Optional[Match]:
        """Manually override the match score asynchronously."""
        return await self.update_async(
            id_value,
            {
                "manual_score_override": new_score,
                "override_reason": reason,
            },
        )

    def add_feedback(
        self,
        id_value: str | ObjectId,
        recruiter_id: str,
        rating: Optional[int] = None,
        comments: Optional[str] = None,
        decision: Optional[str] = None,
    ) -> Optional[Match]:
        """Add recruiter feedback to a match."""
        collection = self._get_sync_collection()
        feedback = RecruiterFeedback(
            recruiter_id=recruiter_id,
            rating=rating,
            comments=comments,
            decision=decision,
        )
        collection.update_one(
            {"_id": self._to_object_id(id_value)},
            {"$push": {"feedback": feedback.model_dump()}},
        )
        return self.get_by_id(id_value)

    async def add_feedback_async(
        self,
        id_value: str | ObjectId,
        recruiter_id: str,
        rating: Optional[int] = None,
        comments: Optional[str] = None,
        decision: Optional[str] = None,
    ) -> Optional[Match]:
        """Add recruiter feedback to a match asynchronously."""
        collection = self._get_async_collection()
        feedback = RecruiterFeedback(
            recruiter_id=recruiter_id,
            rating=rating,
            comments=comments,
            decision=decision,
        )
        await collection.update_one(
            {"_id": self._to_object_id(id_value)},
            {"$push": {"feedback": feedback.model_dump()}},
        )
        return await self.get_by_id_async(id_value)

    # -------------------------------------------------------------------------
    # Ranking Operations
    # -------------------------------------------------------------------------

    def update_ranks_for_job(self, job_id: str | ObjectId) -> int:
        """
        Update ranks for all matches of a job based on scores.

        Returns the number of matches updated.
        """
        collection = self._get_sync_collection()
        job_oid = self._to_object_id(job_id)

        # Get all matches sorted by score
        matches = list(
            collection.find({"job_id": job_oid}).sort("overall_score", -1)
        )

        # Update ranks
        for rank, match in enumerate(matches, start=1):
            collection.update_one({"_id": match["_id"]}, {"$set": {"rank": rank}})

        logger.debug(f"Updated ranks for {len(matches)} matches for job {job_id}")
        return len(matches)

    async def update_ranks_for_job_async(self, job_id: str | ObjectId) -> int:
        """Update ranks for all matches of a job asynchronously."""
        collection = self._get_async_collection()
        job_oid = self._to_object_id(job_id)

        # Get all matches sorted by score
        matches = await collection.find({"job_id": job_oid}).sort(
            "overall_score", -1
        ).to_list(length=None)

        # Update ranks
        for rank, match in enumerate(matches, start=1):
            await collection.update_one(
                {"_id": match["_id"]}, {"$set": {"rank": rank}}
            )

        logger.debug(f"Updated ranks for {len(matches)} matches for job {job_id}")
        return len(matches)

    # -------------------------------------------------------------------------
    # Aggregation Operations
    # -------------------------------------------------------------------------

    def get_status_counts_for_job(self, job_id: str | ObjectId) -> dict[str, int]:
        """Get count of matches by status for a specific job."""
        collection = self._get_sync_collection()
        pipeline = [
            {"$match": {"job_id": self._to_object_id(job_id)}},
            {"$group": {"_id": "$status", "count": {"$sum": 1}}},
        ]
        results = list(collection.aggregate(pipeline))
        return {r["_id"]: r["count"] for r in results}

    async def get_status_counts_for_job_async(
        self, job_id: str | ObjectId
    ) -> dict[str, int]:
        """Get count of matches by status for a specific job asynchronously."""
        collection = self._get_async_collection()
        pipeline = [
            {"$match": {"job_id": self._to_object_id(job_id)}},
            {"$group": {"_id": "$status", "count": {"$sum": 1}}},
        ]
        results = await collection.aggregate(pipeline).to_list(length=None)
        return {r["_id"]: r["count"] for r in results}

    def get_score_distribution_for_job(
        self, job_id: str | ObjectId
    ) -> dict[str, int]:
        """Get distribution of score levels for a job."""
        collection = self._get_sync_collection()
        pipeline = [
            {"$match": {"job_id": self._to_object_id(job_id)}},
            {"$group": {"_id": "$score_level", "count": {"$sum": 1}}},
        ]
        results = list(collection.aggregate(pipeline))
        return {r["_id"]: r["count"] for r in results}

    async def get_score_distribution_for_job_async(
        self, job_id: str | ObjectId
    ) -> dict[str, int]:
        """Get distribution of score levels for a job asynchronously."""
        collection = self._get_async_collection()
        pipeline = [
            {"$match": {"job_id": self._to_object_id(job_id)}},
            {"$group": {"_id": "$score_level", "count": {"$sum": 1}}},
        ]
        results = await collection.aggregate(pipeline).to_list(length=None)
        return {r["_id"]: r["count"] for r in results}

    def get_score_stats_for_job(self, job_id: str | ObjectId) -> dict[str, Any]:
        """Get score statistics for a job."""
        collection = self._get_sync_collection()
        pipeline = [
            {"$match": {"job_id": self._to_object_id(job_id)}},
            {
                "$group": {
                    "_id": None,
                    "count": {"$sum": 1},
                    "avg_score": {"$avg": "$overall_score"},
                    "max_score": {"$max": "$overall_score"},
                    "min_score": {"$min": "$overall_score"},
                    "shortlisted": {
                        "$sum": {
                            "$cond": [
                                {"$eq": ["$status", MatchStatus.SHORTLISTED.value]},
                                1,
                                0,
                            ]
                        }
                    },
                }
            },
        ]
        results = list(collection.aggregate(pipeline))
        if results:
            result = results[0]
            result.pop("_id", None)
            return result
        return {
            "count": 0,
            "avg_score": 0,
            "max_score": 0,
            "min_score": 0,
            "shortlisted": 0,
        }

    async def get_score_stats_for_job_async(
        self, job_id: str | ObjectId
    ) -> dict[str, Any]:
        """Get score statistics for a job asynchronously."""
        collection = self._get_async_collection()
        pipeline = [
            {"$match": {"job_id": self._to_object_id(job_id)}},
            {
                "$group": {
                    "_id": None,
                    "count": {"$sum": 1},
                    "avg_score": {"$avg": "$overall_score"},
                    "max_score": {"$max": "$overall_score"},
                    "min_score": {"$min": "$overall_score"},
                    "shortlisted": {
                        "$sum": {
                            "$cond": [
                                {"$eq": ["$status", MatchStatus.SHORTLISTED.value]},
                                1,
                                0,
                            ]
                        }
                    },
                }
            },
        ]
        results = await collection.aggregate(pipeline).to_list(length=1)
        if results:
            result = results[0]
            result.pop("_id", None)
            return result
        return {
            "count": 0,
            "avg_score": 0,
            "max_score": 0,
            "min_score": 0,
            "shortlisted": 0,
        }

    # -------------------------------------------------------------------------
    # Bulk Operations
    # -------------------------------------------------------------------------

    def delete_by_job(self, job_id: str | ObjectId) -> int:
        """Delete all matches for a job."""
        collection = self._get_sync_collection()
        result = collection.delete_many({"job_id": self._to_object_id(job_id)})
        logger.debug(f"Deleted {result.deleted_count} matches for job {job_id}")
        return result.deleted_count

    async def delete_by_job_async(self, job_id: str | ObjectId) -> int:
        """Delete all matches for a job asynchronously."""
        collection = self._get_async_collection()
        result = await collection.delete_many({"job_id": self._to_object_id(job_id)})
        logger.debug(f"Deleted {result.deleted_count} matches for job {job_id}")
        return result.deleted_count

    def delete_by_candidate(self, candidate_id: str | ObjectId) -> int:
        """Delete all matches for a candidate."""
        collection = self._get_sync_collection()
        result = collection.delete_many(
            {"candidate_id": self._to_object_id(candidate_id)}
        )
        logger.debug(
            f"Deleted {result.deleted_count} matches for candidate {candidate_id}"
        )
        return result.deleted_count

    async def delete_by_candidate_async(self, candidate_id: str | ObjectId) -> int:
        """Delete all matches for a candidate asynchronously."""
        collection = self._get_async_collection()
        result = await collection.delete_many(
            {"candidate_id": self._to_object_id(candidate_id)}
        )
        logger.debug(
            f"Deleted {result.deleted_count} matches for candidate {candidate_id}"
        )
        return result.deleted_count


# Singleton instance
_match_repository: Optional[MatchRepository] = None


def get_match_repository() -> MatchRepository:
    """Get the match repository singleton instance."""
    global _match_repository
    if _match_repository is None:
        _match_repository = MatchRepository()
    return _match_repository
