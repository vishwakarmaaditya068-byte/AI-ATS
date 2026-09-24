"""
Unit tests for the AI-ATS CLI commands.

Uses Typer's CliRunner (wraps Click's test runner) to invoke commands
in-process, capturing stdout and exit codes without spawning subprocesses.

All external dependencies (database, ML models) are mocked so these tests
run fully offline and do not require MongoDB or GPU.
"""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from src.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------


def test_version_shows_app_name_and_version() -> None:
    """version command prints the app name and version string."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0, result.output
    assert "AI-ATS" in result.output
    assert "0.1.0" in result.output


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


def test_info_prints_configuration_table() -> None:
    """info command renders a settings table without error."""
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0, result.output
    # The Rich table contains these column headers
    assert "Setting" in result.output
    assert "Value" in result.output


# ---------------------------------------------------------------------------
# init-db
# ---------------------------------------------------------------------------


def test_init_db_succeeds_when_mongo_available() -> None:
    """init-db exits 0 when MongoDB is reachable and indexes are created."""
    mock_db = MagicMock()
    mock_db.check_sync_connection.return_value = True

    import asyncio

    with (
        patch("src.data.database.get_database_manager", return_value=mock_db),
        patch("asyncio.run") as mock_run,
    ):
        result = runner.invoke(app, ["init-db"])

    assert result.exit_code == 0, result.output
    assert "successfully" in result.output.lower()


def test_init_db_exits_1_when_mongo_unavailable() -> None:
    """init-db exits 1 and prints an error when MongoDB is not reachable."""
    mock_db = MagicMock()
    mock_db.check_sync_connection.return_value = False

    with patch("src.data.database.get_database_manager", return_value=mock_db):
        result = runner.invoke(app, ["init-db"])

    assert result.exit_code == 1
    assert "MongoDB" in result.output


# ---------------------------------------------------------------------------
# import-resumes
# ---------------------------------------------------------------------------


def test_import_resumes_exits_1_for_nonexistent_path(tmp_path: pytest.fixture) -> None:
    """import-resumes exits 1 when the supplied path does not exist."""
    result = runner.invoke(app, ["import-resumes", str(tmp_path / "nonexistent")])
    assert result.exit_code == 1
    assert "not exist" in result.output.lower() or "error" in result.output.lower()


def test_import_resumes_skips_unsupported_extension(tmp_path: pytest.fixture) -> None:
    """import-resumes exits 1 when a single file has an unsupported extension."""
    bad_file = tmp_path / "resume.xyz"
    bad_file.write_text("content")

    mock_db = MagicMock()
    mock_db.check_sync_connection.return_value = True

    with patch("src.data.database.get_database_manager", return_value=mock_db):
        result = runner.invoke(app, ["import-resumes", str(bad_file)])

    assert result.exit_code == 1
    assert "unsupported" in result.output.lower()


def test_import_resumes_reports_no_files_in_empty_dir(tmp_path: pytest.fixture) -> None:
    """import-resumes exits 0 with a 'no files' notice for an empty directory."""
    mock_db = MagicMock()
    mock_db.check_sync_connection.return_value = True

    with patch("src.data.database.get_database_manager", return_value=mock_db):
        result = runner.invoke(app, ["import-resumes", str(tmp_path)])

    assert result.exit_code == 0
    assert "no resume files found" in result.output.lower()


# ---------------------------------------------------------------------------
# list-jobs
# ---------------------------------------------------------------------------


def test_list_jobs_exits_1_when_mongo_unavailable() -> None:
    """list-jobs exits 1 when MongoDB is not reachable."""
    mock_db = MagicMock()
    mock_db.check_sync_connection.return_value = False

    with patch("src.data.database.get_database_manager", return_value=mock_db):
        result = runner.invoke(app, ["list-jobs"])

    assert result.exit_code == 1


def test_list_jobs_shows_empty_message_when_no_jobs() -> None:
    """list-jobs exits 0 and says 'no jobs' when the repo returns an empty list."""
    mock_db = MagicMock()
    mock_db.check_sync_connection.return_value = True

    mock_repo = MagicMock()
    mock_repo.find.return_value = []

    with (
        patch("src.data.database.get_database_manager", return_value=mock_db),
        patch("src.data.repositories.get_job_repository", return_value=mock_repo),
    ):
        result = runner.invoke(app, ["list-jobs"])

    assert result.exit_code == 0
    assert "no jobs" in result.output.lower()


# ---------------------------------------------------------------------------
# health-check
# ---------------------------------------------------------------------------


def test_health_check_reports_mongo_connected() -> None:
    """health-check shows 'MongoDB connected' when DB is reachable."""
    mock_db = MagicMock()
    mock_db.check_sync_connection.return_value = True

    with patch("src.data.database.get_database_manager", return_value=mock_db):
        result = runner.invoke(app, ["health-check"])

    assert result.exit_code == 0
    assert "MongoDB connected" in result.output


def test_health_check_reports_mongo_not_connected() -> None:
    """health-check shows MongoDB failure when DB is unreachable."""
    mock_db = MagicMock()
    mock_db.check_sync_connection.return_value = False

    with patch("src.data.database.get_database_manager", return_value=mock_db):
        result = runner.invoke(app, ["health-check"])

    assert result.exit_code == 0  # health-check always exits 0 (soft-fail probe)
    assert "not connected" in result.output.lower()


# ---------------------------------------------------------------------------
# match
# ---------------------------------------------------------------------------


def test_match_exits_1_when_mongo_unavailable() -> None:
    """match exits 1 when MongoDB is not reachable."""
    mock_db = MagicMock()
    mock_db.check_sync_connection.return_value = False

    with patch("src.data.database.get_database_manager", return_value=mock_db):
        result = runner.invoke(app, ["match", "fake-job-id"])

    assert result.exit_code == 1


def test_match_exits_1_when_job_not_found() -> None:
    """match exits 1 when the requested job ID does not exist."""
    mock_db = MagicMock()
    mock_db.check_sync_connection.return_value = True

    mock_job_repo = MagicMock()
    mock_job_repo.get_by_id.return_value = None

    mock_candidate_repo = MagicMock()
    mock_match_repo = MagicMock()

    with (
        patch("src.data.database.get_database_manager", return_value=mock_db),
        patch("src.data.repositories.get_job_repository", return_value=mock_job_repo),
        patch("src.data.repositories.get_candidate_repository", return_value=mock_candidate_repo),
        patch("src.data.repositories.get_match_repository", return_value=mock_match_repo),
    ):
        result = runner.invoke(app, ["match", "nonexistent-job-id"])

    assert result.exit_code == 1
    assert "not found" in result.output.lower()
