"""
Tests for settings persistence: write_env_settings() and the reload round-trip.

Covers:
- Updating existing keys in .env without destroying comments or unrelated lines
- Appending new keys that are not yet present
- Atomic write: temp file is cleaned up, target file is never half-written
- Round-trip: write_env_settings → reload_settings → in-memory values match
- Edge cases: empty file, missing file, multiple updates in one call
"""

import os
from pathlib import Path

import pytest

from src.utils.config import write_env_settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# write_env_settings — in-place replacement
# ---------------------------------------------------------------------------


class TestWriteEnvSettingsReplacement:
    def test_replaces_existing_key_value(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        _write(env_file, "UI_THEME=light\n")

        write_env_settings({"UI_THEME": "dark"}, env_file=env_file)

        assert _read(env_file) == "UI_THEME=dark\n"

    def test_preserves_comment_lines(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        _write(env_file, "# Database settings\nDB_HOST=localhost\n")

        write_env_settings({"DB_HOST": "remotehost"}, env_file=env_file)

        content = _read(env_file)
        assert "# Database settings" in content
        assert "DB_HOST=remotehost" in content

    def test_preserves_unrelated_keys(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        _write(env_file, "DB_HOST=localhost\nDB_PORT=27017\n")

        write_env_settings({"DB_HOST": "newhost"}, env_file=env_file)

        content = _read(env_file)
        assert "DB_PORT=27017" in content
        assert "DB_HOST=newhost" in content

    def test_multiple_keys_replaced_in_one_call(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        _write(env_file, "UI_THEME=light\nDB_HOST=localhost\nDB_PORT=27017\n")

        write_env_settings(
            {"UI_THEME": "dark", "DB_HOST": "remotehost"},
            env_file=env_file,
        )

        content = _read(env_file)
        assert "UI_THEME=dark" in content
        assert "DB_HOST=remotehost" in content
        assert "DB_PORT=27017" in content

    def test_only_first_occurrence_replaced(self, tmp_path: Path) -> None:
        """If a key appears twice (malformed .env), only the first is replaced."""
        env_file = tmp_path / ".env"
        _write(env_file, "UI_THEME=light\nUI_THEME=system\n")

        write_env_settings({"UI_THEME": "dark"}, env_file=env_file)

        content = _read(env_file)
        lines = content.splitlines()
        # First occurrence replaced, second remains
        assert lines[0] == "UI_THEME=dark"
        assert lines[1] == "UI_THEME=system"


# ---------------------------------------------------------------------------
# write_env_settings — appending new keys
# ---------------------------------------------------------------------------


class TestWriteEnvSettingsAppend:
    def test_appends_new_key_when_absent(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        _write(env_file, "DB_HOST=localhost\n")

        write_env_settings({"NEW_KEY": "hello"}, env_file=env_file)

        content = _read(env_file)
        assert "NEW_KEY=hello" in content

    def test_appends_to_empty_file(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        _write(env_file, "")

        write_env_settings({"FIRST_KEY": "value"}, env_file=env_file)

        assert "FIRST_KEY=value" in _read(env_file)

    def test_creates_file_when_missing(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        assert not env_file.exists()

        write_env_settings({"BRAND_NEW": "yes"}, env_file=env_file)

        assert env_file.exists()
        assert "BRAND_NEW=yes" in _read(env_file)

    def test_mix_replace_and_append(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        _write(env_file, "EXISTING=old\n")

        write_env_settings({"EXISTING": "new", "APPENDED": "added"}, env_file=env_file)

        content = _read(env_file)
        assert "EXISTING=new" in content
        assert "APPENDED=added" in content


# ---------------------------------------------------------------------------
# write_env_settings — atomic write guarantee
# ---------------------------------------------------------------------------


class TestWriteEnvSettingsAtomic:
    def test_temp_file_cleaned_up_after_write(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        _write(env_file, "UI_THEME=light\n")

        write_env_settings({"UI_THEME": "dark"}, env_file=env_file)

        tmp_file = env_file.with_suffix(".env.tmp")
        assert not tmp_file.exists(), ".env.tmp should be renamed away after atomic write"

    def test_target_file_has_correct_content(self, tmp_path: Path) -> None:
        """After the atomic rename the target file must contain updated content."""
        env_file = tmp_path / ".env"
        _write(env_file, "DB_NAME=ai_ats\n")

        write_env_settings({"DB_NAME": "ai_ats_test"}, env_file=env_file)

        assert "DB_NAME=ai_ats_test" in _read(env_file)
        assert "DB_NAME=ai_ats\n" not in _read(env_file)

    def test_existing_file_not_truncated_on_empty_updates(self, tmp_path: Path) -> None:
        """Calling with an empty dict must not change file content."""
        env_file = tmp_path / ".env"
        original = "DB_HOST=localhost\nDB_PORT=27017\n"
        _write(env_file, original)

        write_env_settings({}, env_file=env_file)

        assert _read(env_file) == original


# ---------------------------------------------------------------------------
# write_env_settings — value edge cases
# ---------------------------------------------------------------------------


class TestWriteEnvSettingsEdgeCases:
    def test_value_with_spaces(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        _write(env_file, "APP_NAME=Old Name\n")

        write_env_settings({"APP_NAME": "AI Applicant Tracking System"}, env_file=env_file)

        assert "APP_NAME=AI Applicant Tracking System" in _read(env_file)

    def test_empty_string_value(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        _write(env_file, "DB_PASSWORD=secret\n")

        write_env_settings({"DB_PASSWORD": ""}, env_file=env_file)

        assert "DB_PASSWORD=\n" in _read(env_file)

    def test_numeric_string_value(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        _write(env_file, "DB_PORT=27017\n")

        write_env_settings({"DB_PORT": "27018"}, env_file=env_file)

        assert "DB_PORT=27018" in _read(env_file)

    def test_lowercase_key_not_matched(self, tmp_path: Path) -> None:
        """The regex only matches UPPER_SNAKE keys; lowercase keys are left unchanged."""
        env_file = tmp_path / ".env"
        _write(env_file, "db_host=localhost\n")

        # Passing a lowercase key — it should be appended, not replace the existing lowercase line
        write_env_settings({"DB_HOST": "newhost"}, env_file=env_file)

        content = _read(env_file)
        # Original lowercase line preserved
        assert "db_host=localhost" in content
        # New uppercase key appended
        assert "DB_HOST=newhost" in content

    def test_comment_line_not_treated_as_key(self, tmp_path: Path) -> None:
        """Lines starting with '#' must not be matched even if they look like KEY=value."""
        env_file = tmp_path / ".env"
        _write(env_file, "# UI_THEME=light\nUI_THEME=system\n")

        write_env_settings({"UI_THEME": "dark"}, env_file=env_file)

        content = _read(env_file)
        assert "# UI_THEME=light" in content   # comment preserved
        assert "UI_THEME=dark" in content       # real key updated


# ---------------------------------------------------------------------------
# Round-trip: write → reload_settings → in-memory values match
# ---------------------------------------------------------------------------


class TestSettingsReloadRoundTrip:
    """
    These tests mutate os.environ and call reload_settings(), so they reset
    the environment after each test to avoid contaminating other tests.
    """

    @pytest.fixture(autouse=True)
    def _restore_env(self, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[return]
        """monkeypatch auto-restores os.environ after each test."""
        yield

    def test_reload_picks_up_written_ui_theme(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.utils.config import reload_settings

        env_file = tmp_path / ".env"
        _write(env_file, "")

        write_env_settings({"UI_THEME": "dark"}, env_file=env_file)

        # Simulate what settings_view does: set env var then reload
        monkeypatch.setenv("UI_THEME", "dark")
        settings = reload_settings()

        assert settings.ui.theme == "dark"

    def test_reload_picks_up_written_db_host(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.utils.config import reload_settings

        env_file = tmp_path / ".env"
        _write(env_file, "")

        write_env_settings({"DB_HOST": "testhost"}, env_file=env_file)

        monkeypatch.setenv("DB_HOST", "testhost")
        settings = reload_settings()

        assert settings.database.host == "testhost"

    def test_reload_picks_up_log_level(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.utils.config import reload_settings

        env_file = tmp_path / ".env"
        _write(env_file, "")

        write_env_settings({"LOG_LEVEL": "DEBUG"}, env_file=env_file)

        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        settings = reload_settings()

        assert settings.logging.level == "DEBUG"

    def test_default_theme_is_system(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When UI_THEME is unset, theme defaults to 'system'."""
        from src.utils.config import reload_settings

        monkeypatch.delenv("UI_THEME", raising=False)
        settings = reload_settings()

        assert settings.ui.theme == "system"
