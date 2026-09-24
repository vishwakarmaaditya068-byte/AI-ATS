"""
Audit log repository for AI-ATS.

Provides data access operations for audit log documents,
supporting compliance, transparency, and system monitoring.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from bson import ObjectId

from src.data.models.audit import (
    AuditLog,
    AuditLogCreate,
    AuditLogQuery,
    AuditSummary,
)
from src.utils.constants import AuditAction
from src.utils.logger import get_logger

from .base import BaseRepository

logger = get_logger(__name__)


class AuditRepository(BaseRepository[AuditLog]):
    """Repository for audit log document operations."""

    @property
    def collection_name(self) -> str:
        return "audit_logs"

    @property
    def model_class(self) -> type[AuditLog]:
        return AuditLog

    # -------------------------------------------------------------------------
    # Create Operations
    # -------------------------------------------------------------------------

    def log(self, data: AuditLogCreate) -> AuditLog:
        """Create an audit log entry from a create schema."""
        audit_log = AuditLog(
            action=data.action,
            action_description=data.action_description,
            actor=data.actor if data.actor else None,
            resource=data.resource,
            changes=data.changes,
            ai_decision=data.ai_decision,
            bias_audit=data.bias_audit,
            context=data.context,
            related_candidate_id=(
                self._to_object_id(data.related_candidate_id)
                if data.related_candidate_id
                else None
            ),
            related_job_id=(
                self._to_object_id(data.related_job_id)
                if data.related_job_id
                else None
            ),
            related_match_id=(
                self._to_object_id(data.related_match_id)
                if data.related_match_id
                else None
            ),
            compliance_relevant=data.compliance_relevant,
        )
        return self.create(audit_log)

    async def log_async(self, data: AuditLogCreate) -> AuditLog:
        """Create an audit log entry from a create schema asynchronously."""
        audit_log = AuditLog(
            action=data.action,
            action_description=data.action_description,
            actor=data.actor if data.actor else None,
            resource=data.resource,
            changes=data.changes,
            ai_decision=data.ai_decision,
            bias_audit=data.bias_audit,
            context=data.context,
            related_candidate_id=(
                self._to_object_id(data.related_candidate_id)
                if data.related_candidate_id
                else None
            ),
            related_job_id=(
                self._to_object_id(data.related_job_id)
                if data.related_job_id
                else None
            ),
            related_match_id=(
                self._to_object_id(data.related_match_id)
                if data.related_match_id
                else None
            ),
            compliance_relevant=data.compliance_relevant,
        )
        return await self.create_async(audit_log)

    # -------------------------------------------------------------------------
    # Query Operations
    # -------------------------------------------------------------------------

    def get_by_action(
        self,
        action: AuditAction,
        skip: int = 0,
        limit: int = 100,
    ) -> list[AuditLog]:
        """Get audit logs by action type."""
        return self.find(
            {"action": action.value},
            skip=skip,
            limit=limit,
            sort_by="created_at",
            sort_order=-1,
        )

    async def get_by_action_async(
        self,
        action: AuditAction,
        skip: int = 0,
        limit: int = 100,
    ) -> list[AuditLog]:
        """Get audit logs by action type asynchronously."""
        return await self.find_async(
            {"action": action.value},
            skip=skip,
            limit=limit,
            sort_by="created_at",
            sort_order=-1,
        )

    def get_by_actor(
        self,
        actor_id: str,
        skip: int = 0,
        limit: int = 100,
    ) -> list[AuditLog]:
        """Get audit logs by actor ID."""
        return self.find(
            {"actor.actor_id": actor_id},
            skip=skip,
            limit=limit,
            sort_by="created_at",
            sort_order=-1,
        )

    async def get_by_actor_async(
        self,
        actor_id: str,
        skip: int = 0,
        limit: int = 100,
    ) -> list[AuditLog]:
        """Get audit logs by actor ID asynchronously."""
        return await self.find_async(
            {"actor.actor_id": actor_id},
            skip=skip,
            limit=limit,
            sort_by="created_at",
            sort_order=-1,
        )

    def get_by_resource(
        self,
        resource_type: str,
        resource_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[AuditLog]:
        """Get audit logs by resource type and optionally ID."""
        query: dict[str, Any] = {"resource.resource_type": resource_type}
        if resource_id:
            query["resource.resource_id"] = resource_id
        return self.find(
            query,
            skip=skip,
            limit=limit,
            sort_by="created_at",
            sort_order=-1,
        )

    async def get_by_resource_async(
        self,
        resource_type: str,
        resource_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[AuditLog]:
        """Get audit logs by resource type and optionally ID asynchronously."""
        query: dict[str, Any] = {"resource.resource_type": resource_type}
        if resource_id:
            query["resource.resource_id"] = resource_id
        return await self.find_async(
            query,
            skip=skip,
            limit=limit,
            sort_by="created_at",
            sort_order=-1,
        )

    def get_for_candidate(
        self,
        candidate_id: str | ObjectId,
        skip: int = 0,
        limit: int = 100,
    ) -> list[AuditLog]:
        """Get all audit logs related to a candidate."""
        return self.find(
            {"related_candidate_id": self._to_object_id(candidate_id)},
            skip=skip,
            limit=limit,
            sort_by="created_at",
            sort_order=-1,
        )

    async def get_for_candidate_async(
        self,
        candidate_id: str | ObjectId,
        skip: int = 0,
        limit: int = 100,
    ) -> list[AuditLog]:
        """Get all audit logs related to a candidate asynchronously."""
        return await self.find_async(
            {"related_candidate_id": self._to_object_id(candidate_id)},
            skip=skip,
            limit=limit,
            sort_by="created_at",
            sort_order=-1,
        )

    def get_for_job(
        self,
        job_id: str | ObjectId,
        skip: int = 0,
        limit: int = 100,
    ) -> list[AuditLog]:
        """Get all audit logs related to a job."""
        return self.find(
            {"related_job_id": self._to_object_id(job_id)},
            skip=skip,
            limit=limit,
            sort_by="created_at",
            sort_order=-1,
        )

    async def get_for_job_async(
        self,
        job_id: str | ObjectId,
        skip: int = 0,
        limit: int = 100,
    ) -> list[AuditLog]:
        """Get all audit logs related to a job asynchronously."""
        return await self.find_async(
            {"related_job_id": self._to_object_id(job_id)},
            skip=skip,
            limit=limit,
            sort_by="created_at",
            sort_order=-1,
        )

    def get_compliance_logs(
        self,
        skip: int = 0,
        limit: int = 100,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> list[AuditLog]:
        """Get compliance-relevant audit logs."""
        query: dict[str, Any] = {"compliance_relevant": True}

        if start_date or end_date:
            query["created_at"] = {}
            if start_date:
                query["created_at"]["$gte"] = start_date
            if end_date:
                query["created_at"]["$lte"] = end_date

        return self.find(
            query,
            skip=skip,
            limit=limit,
            sort_by="created_at",
            sort_order=-1,
        )

    async def get_compliance_logs_async(
        self,
        skip: int = 0,
        limit: int = 100,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> list[AuditLog]:
        """Get compliance-relevant audit logs asynchronously."""
        query: dict[str, Any] = {"compliance_relevant": True}

        if start_date or end_date:
            query["created_at"] = {}
            if start_date:
                query["created_at"]["$gte"] = start_date
            if end_date:
                query["created_at"]["$lte"] = end_date

        return await self.find_async(
            query,
            skip=skip,
            limit=limit,
            sort_by="created_at",
            sort_order=-1,
        )

    def get_bias_detections(
        self,
        skip: int = 0,
        limit: int = 100,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> list[AuditLog]:
        """Get bias detection audit logs."""
        query: dict[str, Any] = {"action": AuditAction.BIAS_DETECTED.value}

        if start_date or end_date:
            query["created_at"] = {}
            if start_date:
                query["created_at"]["$gte"] = start_date
            if end_date:
                query["created_at"]["$lte"] = end_date

        return self.find(
            query,
            skip=skip,
            limit=limit,
            sort_by="created_at",
            sort_order=-1,
        )

    async def get_bias_detections_async(
        self,
        skip: int = 0,
        limit: int = 100,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> list[AuditLog]:
        """Get bias detection audit logs asynchronously."""
        query: dict[str, Any] = {"action": AuditAction.BIAS_DETECTED.value}

        if start_date or end_date:
            query["created_at"] = {}
            if start_date:
                query["created_at"]["$gte"] = start_date
            if end_date:
                query["created_at"]["$lte"] = end_date

        return await self.find_async(
            query,
            skip=skip,
            limit=limit,
            sort_by="created_at",
            sort_order=-1,
        )

    def search(self, query_params: AuditLogQuery) -> list[AuditLog]:
        """Search audit logs with multiple filters."""
        query: dict[str, Any] = {}

        if query_params.action:
            query["action"] = query_params.action.value

        if query_params.actor_id:
            query["actor.actor_id"] = query_params.actor_id

        if query_params.resource_type:
            query["resource.resource_type"] = query_params.resource_type

        if query_params.resource_id:
            query["resource.resource_id"] = query_params.resource_id

        if query_params.candidate_id:
            query["related_candidate_id"] = self._to_object_id(
                query_params.candidate_id
            )

        if query_params.job_id:
            query["related_job_id"] = self._to_object_id(query_params.job_id)

        if query_params.compliance_only:
            query["compliance_relevant"] = True

        if query_params.start_date or query_params.end_date:
            query["created_at"] = {}
            if query_params.start_date:
                query["created_at"]["$gte"] = query_params.start_date
            if query_params.end_date:
                query["created_at"]["$lte"] = query_params.end_date

        return self.find(
            query,
            skip=query_params.offset,
            limit=query_params.limit,
            sort_by="created_at",
            sort_order=-1,
        )

    async def search_async(self, query_params: AuditLogQuery) -> list[AuditLog]:
        """Search audit logs with multiple filters asynchronously."""
        query: dict[str, Any] = {}

        if query_params.action:
            query["action"] = query_params.action.value

        if query_params.actor_id:
            query["actor.actor_id"] = query_params.actor_id

        if query_params.resource_type:
            query["resource.resource_type"] = query_params.resource_type

        if query_params.resource_id:
            query["resource.resource_id"] = query_params.resource_id

        if query_params.candidate_id:
            query["related_candidate_id"] = self._to_object_id(
                query_params.candidate_id
            )

        if query_params.job_id:
            query["related_job_id"] = self._to_object_id(query_params.job_id)

        if query_params.compliance_only:
            query["compliance_relevant"] = True

        if query_params.start_date or query_params.end_date:
            query["created_at"] = {}
            if query_params.start_date:
                query["created_at"]["$gte"] = query_params.start_date
            if query_params.end_date:
                query["created_at"]["$lte"] = query_params.end_date

        return await self.find_async(
            query,
            skip=query_params.offset,
            limit=query_params.limit,
            sort_by="created_at",
            sort_order=-1,
        )

    # -------------------------------------------------------------------------
    # Aggregation Operations
    # -------------------------------------------------------------------------

    def get_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> AuditSummary:
        """Get summary statistics for audit logs."""
        collection = self._get_sync_collection()

        match_stage: dict[str, Any] = {}
        if start_date or end_date:
            match_stage["created_at"] = {}
            if start_date:
                match_stage["created_at"]["$gte"] = start_date
            if end_date:
                match_stage["created_at"]["$lte"] = end_date

        pipeline = []
        if match_stage:
            pipeline.append({"$match": match_stage})

        pipeline.extend(
            [
                {
                    "$group": {
                        "_id": None,
                        "total_actions": {"$sum": 1},
                        "ai_decisions_count": {
                            "$sum": {"$cond": [{"$ne": ["$ai_decision", None]}, 1, 0]}
                        },
                        "manual_overrides_count": {
                            "$sum": {
                                "$cond": [
                                    {
                                        "$eq": [
                                            "$action",
                                            AuditAction.MANUAL_OVERRIDE.value,
                                        ]
                                    },
                                    1,
                                    0,
                                ]
                            }
                        },
                        "bias_detections_count": {
                            "$sum": {
                                "$cond": [
                                    {
                                        "$eq": [
                                            "$action",
                                            AuditAction.BIAS_DETECTED.value,
                                        ]
                                    },
                                    1,
                                    0,
                                ]
                            }
                        },
                        "compliance_events_count": {
                            "$sum": {"$cond": ["$compliance_relevant", 1, 0]}
                        },
                    }
                }
            ]
        )

        results = list(collection.aggregate(pipeline))

        # Get action counts
        action_pipeline = []
        if match_stage:
            action_pipeline.append({"$match": match_stage})
        action_pipeline.append({"$group": {"_id": "$action", "count": {"$sum": 1}}})

        action_results = list(collection.aggregate(action_pipeline))
        actions_by_type = {r["_id"]: r["count"] for r in action_results}

        if results:
            result = results[0]
            return AuditSummary(
                total_actions=result["total_actions"],
                actions_by_type=actions_by_type,
                ai_decisions_count=result["ai_decisions_count"],
                manual_overrides_count=result["manual_overrides_count"],
                bias_detections_count=result["bias_detections_count"],
                compliance_events_count=result["compliance_events_count"],
                period_start=start_date,
                period_end=end_date,
            )

        return AuditSummary(period_start=start_date, period_end=end_date)

    async def get_summary_async(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> AuditSummary:
        """Get summary statistics for audit logs asynchronously."""
        collection = self._get_async_collection()

        match_stage: dict[str, Any] = {}
        if start_date or end_date:
            match_stage["created_at"] = {}
            if start_date:
                match_stage["created_at"]["$gte"] = start_date
            if end_date:
                match_stage["created_at"]["$lte"] = end_date

        pipeline = []
        if match_stage:
            pipeline.append({"$match": match_stage})

        pipeline.extend(
            [
                {
                    "$group": {
                        "_id": None,
                        "total_actions": {"$sum": 1},
                        "ai_decisions_count": {
                            "$sum": {"$cond": [{"$ne": ["$ai_decision", None]}, 1, 0]}
                        },
                        "manual_overrides_count": {
                            "$sum": {
                                "$cond": [
                                    {
                                        "$eq": [
                                            "$action",
                                            AuditAction.MANUAL_OVERRIDE.value,
                                        ]
                                    },
                                    1,
                                    0,
                                ]
                            }
                        },
                        "bias_detections_count": {
                            "$sum": {
                                "$cond": [
                                    {
                                        "$eq": [
                                            "$action",
                                            AuditAction.BIAS_DETECTED.value,
                                        ]
                                    },
                                    1,
                                    0,
                                ]
                            }
                        },
                        "compliance_events_count": {
                            "$sum": {"$cond": ["$compliance_relevant", 1, 0]}
                        },
                    }
                }
            ]
        )

        results = await collection.aggregate(pipeline).to_list(length=1)

        # Get action counts
        action_pipeline = []
        if match_stage:
            action_pipeline.append({"$match": match_stage})
        action_pipeline.append({"$group": {"_id": "$action", "count": {"$sum": 1}}})

        action_results = await collection.aggregate(action_pipeline).to_list(
            length=None
        )
        actions_by_type = {r["_id"]: r["count"] for r in action_results}

        if results:
            result = results[0]
            return AuditSummary(
                total_actions=result["total_actions"],
                actions_by_type=actions_by_type,
                ai_decisions_count=result["ai_decisions_count"],
                manual_overrides_count=result["manual_overrides_count"],
                bias_detections_count=result["bias_detections_count"],
                compliance_events_count=result["compliance_events_count"],
                period_start=start_date,
                period_end=end_date,
            )

        return AuditSummary(period_start=start_date, period_end=end_date)

    # -------------------------------------------------------------------------
    # Cleanup Operations
    # -------------------------------------------------------------------------

    def cleanup_expired(self) -> int:
        """
        Delete audit logs past their retention period.

        Returns the number of deleted documents.
        """
        collection = self._get_sync_collection()

        # Calculate cutoff date based on retention period
        # Default is 7 years (2555 days)
        pipeline = [
            {
                "$match": {
                    "$expr": {
                        "$lt": [
                            "$created_at",
                            {
                                "$subtract": [
                                    datetime.now(timezone.utc),
                                    {"$multiply": ["$retention_period_days", 86400000]},
                                ]
                            },
                        ]
                    }
                }
            },
            {"$project": {"_id": 1}},
        ]

        expired_ids = [doc["_id"] for doc in collection.aggregate(pipeline)]

        if expired_ids:
            result = collection.delete_many({"_id": {"$in": expired_ids}})
            logger.info(f"Cleaned up {result.deleted_count} expired audit logs")
            return result.deleted_count

        return 0

    async def cleanup_expired_async(self) -> int:
        """Delete audit logs past their retention period asynchronously."""
        collection = self._get_async_collection()

        pipeline = [
            {
                "$match": {
                    "$expr": {
                        "$lt": [
                            "$created_at",
                            {
                                "$subtract": [
                                    datetime.now(timezone.utc),
                                    {"$multiply": ["$retention_period_days", 86400000]},
                                ]
                            },
                        ]
                    }
                }
            },
            {"$project": {"_id": 1}},
        ]

        expired_docs = await collection.aggregate(pipeline).to_list(length=None)
        expired_ids = [doc["_id"] for doc in expired_docs]

        if expired_ids:
            result = await collection.delete_many({"_id": {"$in": expired_ids}})
            logger.info(f"Cleaned up {result.deleted_count} expired audit logs")
            return result.deleted_count

        return 0


# Singleton instance
_audit_repository: Optional[AuditRepository] = None


def get_audit_repository() -> AuditRepository:
    """Get the audit repository singleton instance."""
    global _audit_repository
    if _audit_repository is None:
        _audit_repository = AuditRepository()
    return _audit_repository
