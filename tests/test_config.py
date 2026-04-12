"""Tests for the configuration module."""

import sys
from pathlib import Path

from arxiv_mcp_server.config import Settings


def test_storage_path_default(monkeypatch, tmp_path):
    """Default storage path should fall back under the user's home directory."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()

    monkeypatch.setattr(Path, "home", lambda: fake_home)
    monkeypatch.setattr(sys, "argv", ["program"])
    monkeypatch.delenv("ARXIV_STORAGE_PATH", raising=False)

    settings = Settings()
    expected_path = fake_home / ".arxiv-mcp-server" / "papers"

    assert settings.STORAGE_PATH == expected_path.resolve()
    assert expected_path.is_dir()


def test_storage_path_from_env(monkeypatch, tmp_path):
    """ARXIV_STORAGE_PATH should be honored when CLI args are absent."""
    env_path = tmp_path / "env-storage"

    monkeypatch.setattr(sys, "argv", ["program"])
    monkeypatch.setenv("ARXIV_STORAGE_PATH", str(env_path))

    settings = Settings()

    assert settings.STORAGE_PATH == env_path.resolve()
    assert env_path.is_dir()


def test_storage_path_cli_takes_precedence_over_env(monkeypatch, tmp_path):
    """--storage-path should override ARXIV_STORAGE_PATH."""
    cli_path = tmp_path / "cli-storage"
    env_path = tmp_path / "env-storage"

    monkeypatch.setenv("ARXIV_STORAGE_PATH", str(env_path))
    monkeypatch.setattr(sys, "argv", ["program", "--storage-path", str(cli_path)])

    settings = Settings()

    assert settings.STORAGE_PATH == cli_path.resolve()
    assert cli_path.is_dir()
    assert not env_path.exists()


def test_storage_path_creates_missing_directory(monkeypatch, tmp_path):
    """Resolved storage paths should be created automatically."""
    nested_path = tmp_path / "deeply" / "nested" / "directory" / "structure"

    monkeypatch.setattr(sys, "argv", ["program", "--storage-path", str(nested_path)])
    monkeypatch.delenv("ARXIV_STORAGE_PATH", raising=False)

    settings = Settings()

    assert settings.STORAGE_PATH == nested_path.resolve()
    assert nested_path.is_dir()
