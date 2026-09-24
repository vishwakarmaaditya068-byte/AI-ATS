"""
Integration tests for MongoDB data persistence.

Tests Candidate and Job round-trips through their repositories across
re-instantiation — verifying that data written via one repo instance is
retrievable by a fresh instance (as happens between app restarts).

All tests are skipped automatically when MongoDB is unreachable, so the
CI suite stays green without a live database.

Run these manually while the Podman container is active:
    podman start ats-mongo   (or the systemd user service)
    pytest tests/integration/test_data_persistence.py -v
"""

from __future__ import annotations

import os

os.environ.setdefault("APP_ENVIRONMENT", "testing")
os.environ.setdefault("DB_NAME", "ai_ats_test")

from datetime import date
from typing import Optional

import pytest
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from src.utils.config import get_settings


# ---------------------------------------------------------------------------
# DB reachability probe (runs once at collection time)
# ---------------------------------------------------------------------------


def _mongo_is_available() -> bool:
    """Return True if MongoDB is reachable within 500 ms."""
    try:
        settings = get_settings()
        client: MongoClient = MongoClient(
            settings.database.connection_string,
            serverSelectionTimeoutMS=500,
        )
        client.admin.command("ping")
        client.close()
        return True
    except (ConnectionFailure, ServerSelectionTimeoutError, Exception):
        return False


_DB_AVAILABLE: bool = _mongo_is_available()

requires_mongo = pytest.mark.skipif(
    not _DB_AVAILABLE,
    reason="MongoDB not reachable — start the ats-mongo container to run these tests",
)


# ---------------------------------------------------------------------------
# Shared cleanup helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def candidate_repo():
    """Fresh CandidateRepository instance (resets singleton for each test)."""
    from src.data.repositories.candidate_repository import CandidateRepository
    import src.data.repositories.candidate_repository as _mod

    repo = CandidateRepository()
    _mod._candidate_repository = None  # ensure next call creates fresh instance
    yield repo
    _mod._candidate_repository = None


@pytest.fixture()
def job_repo():
    """Fresh JobRepository instance (resets singleton for each test)."""
    from src.data.repositories.job_repository import JobRepository
    import src.data.repositories.job_repository as _mod

    repo = JobRepository()
    _mod._job_repository = None
    yield repo
    _mod._job_repository = None


# ---------------------------------------------------------------------------
# Candidate persistence tests
# ---------------------------------------------------------------------------


@requires_mongo
class TestCandidatePersistence:
    """Verify Candidate CRUD survives repository re-instantiation."""

    def test_create_and_retrieve_by_id(self, candidate_repo) -> None:
        """A candidate created with one repo instance is readable by a fresh instance."""
        from src.data.models.candidate import Candidate, ContactInfo
        from src.data.repositories.candidate_repository import CandidateRepository

        contact = ContactInfo(
            email="persist.test.candidate@example.com",
            city="Test City",
        )
        candidate = Candidate(
            first_name="Persist",
            last_name="Test",
            contact=contact,
        )

        created: Candidate = candidate_repo.create(candidate)
        assert created.id is not None

        try:
            # Fresh repository instance — simulates app restart
            fresh_repo = CandidateRepository()
            retrieved: Optional[Candidate] = fresh_repo.get_by_id(created.id)

            assert retrieved is not None
            assert retrieved.first_name == "Persist"
            assert retrieved.last_name == "Test"
            assert retrieved.contact.email == "persist.test.candidate@example.com"
        finally:
            candidate_repo.delete(created.id)

    def test_create_and_retrieve_by_email(self, candidate_repo) -> None:
        """get_by_email works across repository re-instantiation."""
        from src.data.models.candidate import Candidate, ContactInfo
        from src.data.repositories.candidate_repository import CandidateRepository

        email = "email.lookup.test@example.com"
        contact = ContactInfo(email=email)
        candidate = Candidate(first_name="Email", last_name="Lookup", contact=contact)

        created: Candidate = candidate_repo.create(candidate)

        try:
            fresh_repo = CandidateRepository()
            found: Optional[Candidate] = fresh_repo.get_by_email(email)

            assert found is not None
            assert found.first_name == "Email"
        finally:
            candidate_repo.delete(created.id)

    def test_update_persists_across_restart(self, candidate_repo) -> None:
        """An update written by one repo instance is visible to a fresh instance."""
        from src.data.models.candidate import Candidate, ContactInfo, CandidateUpdate
        from src.data.repositories.candidate_repository import CandidateRepository
        from src.utils.constants import CandidateStatus

        contact = ContactInfo(email="update.persist.test@example.com")
        candidate = Candidate(first_name="Update", last_name="Persist", contact=contact)

        created: Candidate = candidate_repo.create(candidate)

        try:
            # Update status using the first repo instance
            candidate_repo.update_status(created.id, CandidateStatus.SHORTLISTED)

            # Read back using a fresh repo instance
            fresh_repo = CandidateRepository()
            updated: Optional[Candidate] = fresh_repo.get_by_id(created.id)

            assert updated is not None
            assert updated.status == CandidateStatus.SHORTLISTED
        finally:
            candidate_repo.delete(created.id)

    def test_delete_removes_from_db(self, candidate_repo) -> None:
        """Deleting a candidate means it is not findable by any subsequent repo."""
        from src.data.models.candidate import Candidate, ContactInfo
        from src.data.repositories.candidate_repository import CandidateRepository

        contact = ContactInfo(email="delete.persist.test@example.com")
        candidate = Candidate(first_name="Delete", last_name="Me", contact=contact)

        created: Candidate = candidate_repo.create(candidate)
        candidate_repo.delete(created.id)

        fresh_repo = CandidateRepository()
        gone: Optional[Candidate] = fresh_repo.get_by_id(created.id)
        assert gone is None

    def test_email_exists_reflects_persisted_state(self, candidate_repo) -> None:
        """email_exists() returns True for a candidate saved in a previous repo instance."""
        from src.data.models.candidate import Candidate, ContactInfo
        from src.data.repositories.candidate_repository import CandidateRepository

        email = "exists.test.persist@example.com"
        contact = ContactInfo(email=email)
        candidate = Candidate(first_name="Exists", last_name="Check", contact=contact)

        created: Candidate = candidate_repo.create(candidate)

        try:
            fresh_repo = CandidateRepository()
            assert fresh_repo.email_exists(email) is True
        finally:
            candidate_repo.delete(created.id)


# ---------------------------------------------------------------------------
# Job persistence tests
# ---------------------------------------------------------------------------


@requires_mongo
class TestJobPersistence:
    """Verify Job CRUD survives repository re-instantiation."""

    def test_create_and_retrieve_job_by_id(self, job_repo) -> None:
        """A job created with one repo instance is readable by a fresh instance."""
        from src.data.models.job import Job
        from src.data.repositories.job_repository import JobRepository

        job = Job(
            title="Persistence Test Engineer",
            description="Testing persistence across restarts.",
            company_name="TestCorp",
        )

        created: Job = job_repo.create(job)
        assert created.id is not None

        try:
            fresh_repo = JobRepository()
            retrieved: Optional[Job] = fresh_repo.get_by_id(created.id)

            assert retrieved is not None
            assert retrieved.title == "Persistence Test Engineer"
            assert retrieved.company_name == "TestCorp"
        finally:
            job_repo.delete(created.id)

    def test_job_update_persists(self, job_repo) -> None:
        """A status update written by one repo is visible to a fresh instance."""
        from src.data.models.job import Job
        from src.data.repositories.job_repository import JobRepository
        from src.utils.constants import JobStatus

        job = Job(
            title="Status Update Test",
            description="Job for status persistence test.",
            company_name="AcmeCorp",
        )

        created: Job = job_repo.create(job)

        try:
            job_repo.update_status(created.id, JobStatus.CLOSED)

            fresh_repo = JobRepository()
            updated: Optional[Job] = fresh_repo.get_by_id(created.id)

            assert updated is not None
            assert updated.status == JobStatus.CLOSED
        finally:
            job_repo.delete(created.id)

    def test_job_delete_removes_from_db(self, job_repo) -> None:
        """Deleting a job means it is not findable by any subsequent repo."""
        from src.data.models.job import Job
        from src.data.repositories.job_repository import JobRepository

        job = Job(
            title="To Be Deleted",
            description="This job will be deleted.",
            company_name="DeleteCorp",
        )

        created: Job = job_repo.create(job)
        job_repo.delete(created.id)

        fresh_repo = JobRepository()
        gone: Optional[Job] = fresh_repo.get_by_id(created.id)
        assert gone is None

    def test_job_search_by_company_name(self, job_repo) -> None:
        """Search by company name returns a job persisted by a prior instance."""
        from src.data.models.job import Job
        from src.data.repositories.job_repository import JobRepository

        job = Job(
            title="Search Target Role",
            description="Unique job for search persistence test.",
            company_name="UniquePersistCorp",
        )

        created: Job = job_repo.create(job)

        try:
            fresh_repo = JobRepository()
            results: list[Job] = fresh_repo.get_by_company("UniquePersistCorp")

            assert len(results) >= 1
            titles = [j.title for j in results]
            assert "Search Target Role" in titles
        finally:
            job_repo.delete(created.id)

    def test_job_status_counts_reflect_created_jobs(self, job_repo) -> None:
        """get_status_counts() aggregation reflects jobs created earlier."""
        from src.data.models.job import Job
        from src.data.repositories.job_repository import JobRepository
        from src.utils.constants import JobStatus

        job = Job(
            title="Count Me",
            description="For status count test.",
            company_name="CountCorp",
        )

        created: Job = job_repo.create(job)

        try:
            fresh_repo = JobRepository()
            counts: dict[str, int] = fresh_repo.get_status_counts()

            # Default status is DRAFT — it should appear in counts
            draft_status = JobStatus.DRAFT.value
            assert draft_status in counts
            assert counts[draft_status] >= 1
        finally:
            job_repo.delete(created.id)


# ---------------------------------------------------------------------------
# Cross-entity: Candidate count after bulk insert
# ---------------------------------------------------------------------------


@requires_mongo
class TestBulkCandidatePersistence:
    def test_count_reflects_multiple_inserts(self, candidate_repo) -> None:
        """count_all() on a fresh repo equals the number of documents inserted."""
        from src.data.models.candidate import Candidate, ContactInfo
        from src.data.repositories.candidate_repository import CandidateRepository

        inserted_ids: list = []

        for i in range(3):
            contact = ContactInfo(email=f"bulk.test.{i}@example.com")
            candidate = Candidate(
                first_name=f"Bulk{i}",
                last_name="Candidate",
                contact=contact,
            )
            created: Candidate = candidate_repo.create(candidate)
            inserted_ids.append(created.id)

        try:
            fresh_repo = CandidateRepository()
            total: int = fresh_repo.count({})
            assert total >= 3
        finally:
            for oid in inserted_ids:
                candidate_repo.delete(oid)
