from __future__ import annotations

import uuid
from typing import Any, Optional

import pytest

from src.data.sql.models.job_record import JobRecord, JobStatus
from src.data.sql.models.workspace import Workspace, WorkspaceStatus
from src.services.match_persistence import (
    DEFAULT_WORKSPACE_NAME,
    MatchPersistenceService,
)


# Fakes
class FakeWorkspaceRepo:
    def __init__(self) -> None:
        self.by_name: dict[str, Workspace] = {}
        self.created: list[Workspace] = []

    def find_by_name(self, name: str) -> Optional[Workspace]:
        return self.by_name.get(name)

    def create_workspace(
        self,
        name: str,
        description: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> Workspace:
        ws = Workspace(
            id=uuid.uuid4(),
            name=name,
            description=description,
            created_by=created_by,
            status=WorkspaceStatus.ACTIVE,
        )
        self.by_name[name] = ws
        self.created.append(ws)
        return ws

    @staticmethod
    def _coerce_uuid(value: uuid.UUID | str) -> uuid.UUID:
        return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


class FakeJobRecordRepo:
    def __init__(self) -> None:
        self.by_mongo_id: dict[str, JobRecord] = {}
        self.created: list[JobRecord] = []

    def find_by_mongo_id(self, mongo_doc_id: str) -> Optional[JobRecord]:
        return self.by_mongo_id.get(mongo_doc_id)

    def create(self, job: JobRecord) -> JobRecord:
        if getattr(job, "id", None) is None:
            job.id = uuid.uuid4()
        if job.mongo_doc_id:
            self.by_mongo_id[job.mongo_doc_id] = job
        self.created.append(job)
        return job


class FakeMatchRecordRepo:
    def __init__(self) -> None:
        self.upserts: list[dict[str, Any]] = []

    def upsert(self, **kwargs: Any) -> Any:
        self.upserts.append(kwargs)

        class _Match:
            id = uuid.uuid4()
            job_id = kwargs["job_id"]
            candidate_mongo_id = kwargs["candidate_mongo_id"]
            overall_score = kwargs["overall_score"]
            manual_score_override = None
            effective_score = float(kwargs["overall_score"])

        return _Match()


# Fixtures
@pytest.fixture
def workspace_repo() -> FakeWorkspaceRepo:
    return FakeWorkspaceRepo()


@pytest.fixture
def job_record_repo() -> FakeJobRecordRepo:
    return FakeJobRecordRepo()


@pytest.fixture
def match_record_repo() -> FakeMatchRecordRepo:
    return FakeMatchRecordRepo()


@pytest.fixture
def service(
    workspace_repo: FakeWorkspaceRepo,
    job_record_repo: FakeJobRecordRepo,
    match_record_repo: FakeMatchRecordRepo,
) -> MatchPersistenceService:
    return MatchPersistenceService(
        workspace_repo=workspace_repo,  # type: ignore[arg-type]
        job_record_repo=job_record_repo,  # type: ignore[arg-type]
        match_record_repo=match_record_repo,  # type: ignore[arg-type]
    )


# Tests
class TestEnsureDefaultWorkspace:
    def test_creates_on_first_call(
        self, service: MatchPersistenceService, workspace_repo: FakeWorkspaceRepo
    ) -> None:
        ws = service.ensure_default_workspace()
        assert ws.name == DEFAULT_WORKSPACE_NAME
        assert len(workspace_repo.created) == 1

    def test_reuses_on_second_call(
        self, service: MatchPersistenceService, workspace_repo: FakeWorkspaceRepo
    ) -> None:
        a = service.ensure_default_workspace()
        b = service.ensure_default_workspace()
        assert a.id == b.id
        assert len(workspace_repo.created) == 1


class TestEnsureJobRecord:
    def test_creates_new_when_not_linked(
        self, service: MatchPersistenceService, job_record_repo: FakeJobRecordRepo
    ) -> None:
        ws = service.ensure_default_workspace()
        job = service.ensure_job_record(
            workspace_id=ws.id,
            title="Senior Go Engineer",
            company_name="Acme",
            description="We are hiring...",
            mongo_doc_id="507f1f77bcf86cd799439011",
        )
        assert job.title == "Senior Go Engineer"
        assert job.status == JobStatus.OPEN
        assert len(job_record_repo.created) == 1

    def test_reuses_by_mongo_id(
        self, service: MatchPersistenceService, job_record_repo: FakeJobRecordRepo
    ) -> None:
        ws = service.ensure_default_workspace()
        a = service.ensure_job_record(
            workspace_id=ws.id,
            title="A",
            mongo_doc_id="507f1f77bcf86cd799439011",
        )
        b = service.ensure_job_record(
            workspace_id=ws.id,
            title="Different title — should still return original",
            mongo_doc_id="507f1f77bcf86cd799439011",
        )
        assert a.id == b.id
        assert len(job_record_repo.created) == 1

    def test_description_snippet_capped(
        self, service: MatchPersistenceService, job_record_repo: FakeJobRecordRepo
    ) -> None:
        ws = service.ensure_default_workspace()
        long_desc: str = "x" * 2000
        job = service.ensure_job_record(
            workspace_id=ws.id,
            title="Test",
            description=long_desc,
        )
        assert len(job.description_snippet or "") == 500


class TestPersistMatch:
    def test_passes_through_to_repo(
        self,
        service: MatchPersistenceService,
        match_record_repo: FakeMatchRecordRepo,
    ) -> None:
        job_id = uuid.uuid4()
        match = service.persist_match(
            job_id=job_id,
            candidate_mongo_id="507f1f77bcf86cd799439011",
            overall_score=0.75,
            skills_score=0.8,
            bias_check={"potential_bias_detected": False},
        )
        assert match.overall_score == pytest.approx(0.75)
        assert len(match_record_repo.upserts) == 1
        assert match_record_repo.upserts[0]["skills_score"] == pytest.approx(0.8)

    def test_defaults_empty_payloads(
        self,
        service: MatchPersistenceService,
        match_record_repo: FakeMatchRecordRepo,
    ) -> None:
        service.persist_match(
            job_id=uuid.uuid4(),
            candidate_mongo_id="507f1f77bcf86cd799439011",
            overall_score=0.5,
        )
        call = match_record_repo.upserts[0]
        assert call["explanation"] is None
        assert call["bias_check"] is None
