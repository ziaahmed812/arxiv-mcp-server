"""Tests for bundle-backed paper download functionality."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import arxiv
import pytest

from arxiv_mcp_server.paper_store import get_bundle_paths, older_files_path
from arxiv_mcp_server.tools import download as download_module
from arxiv_mcp_server.tools.download import (
    ArtifactDownloadError,
    PaperNotFoundError,
    handle_download,
)


def _write_artifact(destination: Path, content: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)


@pytest.fixture
def download_test_env(monkeypatch, temp_storage_args):
    """Route download module storage into a temp directory and disable indexing."""
    monkeypatch.setattr(download_module, "_semantic_search_available", False)
    return temp_storage_args


def _artifact_side_effect(source_error: Exception | None = None):
    """Build a side-effect function for artifact downloads."""

    def _download(url, destination, label, retries=2):
        del url, retries
        if label == "PDF":
            _write_artifact(destination, b"%PDF-1.7")
            return "downloaded"
        if source_error is not None:
            raise source_error
        _write_artifact(destination, b"\x1f\x8bsource")
        return "downloaded"

    return _download


@pytest.mark.asyncio
async def test_cached_markdown_backfills_missing_sidecars(
    download_test_env, mocker, mock_paper
):
    """Cache hits should return immediately and backfill missing PDF/source files."""
    paper_id = "2103.12345v1"
    paths = get_bundle_paths(paper_id)
    paths["bundle_dir"].mkdir(parents=True, exist_ok=True)
    paths["markdown"].write_text("# Cached bundle\ncontent", encoding="utf-8")

    mocker.patch(
        "arxiv_mcp_server.tools.download._fetch_paper_result",
        return_value=mock_paper,
    )
    mock_fetch_html = mocker.patch(
        "arxiv_mcp_server.tools.download._fetch_html_content"
    )
    mock_convert = mocker.patch(
        "arxiv_mcp_server.tools.download._convert_pdf_to_markdown"
    )
    mocker.patch(
        "arxiv_mcp_server.tools.download._download_binary_artifact",
        side_effect=_artifact_side_effect(),
    )

    response = await handle_download({"paper_id": "2103.12345"})
    result = json.loads(response[0].text)

    assert result["status"] == "success"
    assert result["source"] == "cache"
    assert result["paper_id"] == paper_id
    assert result["message"] == (
        "Paper already available (returned from cache and backfilled artifacts)"
    )
    assert result["artifacts"]["markdown"]["status"] == "cached"
    assert result["artifacts"]["pdf"]["status"] == "downloaded"
    assert result["artifacts"]["source"]["status"] == "downloaded"
    assert paths["pdf"].exists()
    assert paths["source"].exists()
    mock_fetch_html.assert_not_called()
    mock_convert.assert_not_called()


@pytest.mark.asyncio
async def test_html_download_creates_bundle_with_sidecars(
    download_test_env, mocker, mock_paper
):
    """HTML-first downloads should keep markdown, PDF, and source together."""
    html_text = "Title of the Paper\nAbstract content goes here."

    mocker.patch(
        "arxiv_mcp_server.tools.download._fetch_paper_result",
        return_value=mock_paper,
    )
    mocker.patch(
        "arxiv_mcp_server.tools.download._fetch_html_content",
        return_value=html_text,
    )
    mock_convert = mocker.patch(
        "arxiv_mcp_server.tools.download._convert_pdf_to_markdown"
    )
    mocker.patch(
        "arxiv_mcp_server.tools.download._download_binary_artifact",
        side_effect=_artifact_side_effect(),
    )

    response = await handle_download({"paper_id": "2103.12345"})
    result = json.loads(response[0].text)
    paths = get_bundle_paths("2103.12345v1")

    assert result["status"] == "success"
    assert result["source"] == "html"
    assert result["paper_id"] == "2103.12345v1"
    assert result["requested_paper_id"] == "2103.12345"
    assert result["storage_dir"] == str(paths["bundle_dir"])
    assert result["artifacts"]["markdown"]["status"] == "downloaded_from_html"
    assert result["artifacts"]["pdf"]["status"] == "downloaded"
    assert result["artifacts"]["source"]["status"] == "downloaded"
    assert result["warnings"] == []
    assert paths["markdown"].read_text(encoding="utf-8") == html_text
    assert paths["pdf"].exists()
    assert paths["source"].exists()
    mock_convert.assert_not_called()


@pytest.mark.asyncio
async def test_pdf_fallback_creates_bundle_and_retains_pdf(
    download_test_env, mocker, mock_paper
):
    """PDF fallback should generate markdown and keep the PDF on disk."""
    pdf_markdown = "# PDF Paper\nConverted from PDF."

    mocker.patch(
        "arxiv_mcp_server.tools.download._fetch_paper_result",
        return_value=mock_paper,
    )
    mocker.patch(
        "arxiv_mcp_server.tools.download._fetch_html_content",
        return_value=None,
    )
    mocker.patch(
        "arxiv_mcp_server.tools.download._convert_pdf_to_markdown",
        return_value=pdf_markdown,
    )
    mocker.patch(
        "arxiv_mcp_server.tools.download._download_binary_artifact",
        side_effect=_artifact_side_effect(),
    )

    response = await handle_download({"paper_id": "2103.12345v1"})
    result = json.loads(response[0].text)
    paths = get_bundle_paths("2103.12345v1")

    assert result["status"] == "success"
    assert result["source"] == "pdf"
    assert result["artifacts"]["markdown"]["status"] == "generated_from_pdf"
    assert result["artifacts"]["pdf"]["status"] == "downloaded"
    assert result["artifacts"]["source"]["status"] == "downloaded"
    assert paths["markdown"].read_text(encoding="utf-8") == pdf_markdown
    assert paths["pdf"].exists()
    assert paths["source"].exists()


@pytest.mark.asyncio
async def test_partial_success_returns_warning_when_source_download_fails(
    download_test_env, mocker, mock_paper
):
    """Markdown success plus source failure should return success with warnings."""
    source_error = ArtifactDownloadError(
        "Source archive download failed after retry: boom"
    )

    mocker.patch(
        "arxiv_mcp_server.tools.download._fetch_paper_result",
        return_value=mock_paper,
    )
    mocker.patch(
        "arxiv_mcp_server.tools.download._fetch_html_content",
        return_value="HTML body",
    )
    mocker.patch(
        "arxiv_mcp_server.tools.download._download_binary_artifact",
        side_effect=_artifact_side_effect(source_error=source_error),
    )

    response = await handle_download({"paper_id": "2103.12345"})
    result = json.loads(response[0].text)

    assert result["status"] == "success"
    assert result["source"] == "html"
    assert result["message"] == "Paper fetched from arXiv HTML endpoint with warnings"
    assert result["artifacts"]["pdf"]["status"] == "downloaded"
    assert result["artifacts"]["source"]["status"] == "failed"
    assert "Source archive download failed after retry: boom" in result["warnings"][0]


@pytest.mark.asyncio
async def test_legacy_flat_files_are_archived_before_download(
    download_test_env, mocker, mock_paper
):
    """Legacy flat files should be moved into older-files without touching folders."""
    legacy_file = download_test_env / "1999.12345v1.md"
    legacy_file.write_text("old flat content", encoding="utf-8")
    dated_folder = download_test_env / "2026-04-12"
    dated_folder.mkdir(parents=True, exist_ok=True)
    (dated_folder / "notes.md").write_text("leave me alone", encoding="utf-8")

    mocker.patch(
        "arxiv_mcp_server.tools.download._fetch_paper_result",
        return_value=mock_paper,
    )
    mocker.patch(
        "arxiv_mcp_server.tools.download._fetch_html_content",
        return_value="new content",
    )
    mocker.patch(
        "arxiv_mcp_server.tools.download._download_binary_artifact",
        side_effect=_artifact_side_effect(),
    )

    await handle_download({"paper_id": "2103.12345"})

    archived_file = older_files_path() / legacy_file.name
    assert not legacy_file.exists()
    assert archived_file.exists()
    assert (dated_folder / "notes.md").exists()


@pytest.mark.asyncio
async def test_paper_not_found_returns_error(download_test_env, mocker):
    """Unknown arXiv IDs should return a clean error payload."""
    paper_id = "invalid.00000"

    mocker.patch(
        "arxiv_mcp_server.tools.download._fetch_paper_result",
        side_effect=PaperNotFoundError(f"Paper {paper_id} not found on arXiv"),
    )

    response = await handle_download({"paper_id": paper_id})
    result = json.loads(response[0].text)

    assert result["status"] == "error"
    assert "not found on arXiv" in result["message"]
