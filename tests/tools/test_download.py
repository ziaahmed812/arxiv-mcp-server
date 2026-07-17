"""Tests for bundle-backed paper downloads and their MCP receipts."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import arxiv
import pytest

from arxiv_mcp_server.paper_store import get_bundle_paths
from arxiv_mcp_server.tools import download as download_module
from arxiv_mcp_server.tools.download import (
    ArtifactDownloadError,
    PaperNotFoundError,
    _download_arxiv_pdf_to_path,
    _html_to_text,
    handle_download,
)


def _write_artifact(destination: Path, content: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)


def _artifact_side_effect(source_error: Exception | None = None):
    """Return a deterministic replacement for sidecar network downloads."""

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


@pytest.fixture
def download_test_env(monkeypatch, temp_storage_args):
    """Route storage to a temporary directory and disable semantic indexing."""
    monkeypatch.setattr(download_module, "_semantic_search_available", False)
    return temp_storage_args


def test_download_arxiv_pdf_streams_via_httpx(temp_storage_path, mocker):
    """PDF downloads use bounded HTTP chunks and retain the complete body."""
    stream_response = MagicMock()
    stream_response.raise_for_status = MagicMock()
    stream_response.iter_bytes.return_value = [b"chunk-one", b"chunk-two"]

    stream_context = MagicMock()
    stream_context.__enter__.return_value = stream_response
    stream_context.__exit__.return_value = False

    client = MagicMock()
    client.stream.return_value = stream_context
    client.__enter__.return_value = client
    client.__exit__.return_value = False
    mocker.patch.object(download_module.httpx, "Client", return_value=client)

    paper = MagicMock(spec=arxiv.Result)
    paper.pdf_url = "https://arxiv.org/pdf/2103.00000.pdf"
    destination = temp_storage_path / "paper.pdf"

    _download_arxiv_pdf_to_path(paper, destination)

    assert destination.read_bytes() == b"chunk-onechunk-two"
    client.stream.assert_called_once_with("GET", paper.pdf_url)
    stream_response.iter_bytes.assert_called_once_with(
        chunk_size=download_module._DOWNLOAD_CHUNK_SIZE
    )


def test_download_arxiv_pdf_requires_pdf_url(temp_storage_path):
    """A missing PDF URL fails before creating an artifact."""
    paper = MagicMock(spec=arxiv.Result)
    paper.pdf_url = None

    with pytest.raises(ArtifactDownloadError, match="No pdf URL available"):
        _download_arxiv_pdf_to_path(paper, temp_storage_path / "paper.pdf")


def test_failed_stream_removes_partial_file(temp_storage_path, mocker):
    """Failed downloads never expose a partial artifact as complete."""
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = False
    client.stream.side_effect = RuntimeError("network failure")
    mocker.patch.object(download_module.httpx, "Client", return_value=client)
    mocker.patch.object(download_module.time, "sleep")

    destination = temp_storage_path / "paper.pdf"
    with pytest.raises(ArtifactDownloadError, match="after retry"):
        download_module._download_binary_artifact(
            "https://arxiv.org/pdf/test", destination, "PDF"
        )

    assert not destination.exists()
    assert not (temp_storage_path / ".paper.pdf.part").exists()


def test_html_to_text_strips_noncontent_elements():
    html = (
        "<html><head><style>body{color:red}</style></head><body>"
        "<nav>Navigation</nav><script>alert(1)</script>"
        "<article><h1>Title</h1><p>Abstract here.</p></article>"
        "<footer>Footer</footer></body></html>"
    )

    text = _html_to_text(html)

    assert "Title" in text
    assert "Abstract here" in text
    assert "Navigation" not in text
    assert "alert" not in text
    assert "color" not in text
    assert "Footer" not in text


def test_empty_html_is_not_treated_as_stored_content(mocker):
    """A 200 response containing no readable paper text triggers PDF fallback."""
    response = MagicMock(status_code=200, text="<script>nothing useful</script>")
    mocker.patch.object(download_module.httpx, "get", return_value=response)

    assert download_module._fetch_html_content("2103.12345v1") is None


@pytest.mark.asyncio
async def test_cached_markdown_backfills_sidecars_and_returns_receipt(
    download_test_env, mocker, mock_paper
):
    """Cache hits confirm storage and backfill missing retained artifacts."""
    paper_id = "2103.12345v1"
    paths = get_bundle_paths(paper_id)
    paths["bundle_dir"].mkdir(parents=True, exist_ok=True)
    content = "# Cached bundle\ncontent"
    paths["markdown"].write_text(content, encoding="utf-8")

    mocker.patch.object(download_module, "_fetch_paper_result", return_value=mock_paper)
    html_fetch = mocker.patch.object(download_module, "_fetch_html_content")
    convert = mocker.patch.object(download_module, "_convert_pdf_to_markdown")
    mocker.patch.object(
        download_module,
        "_download_binary_artifact",
        side_effect=_artifact_side_effect(),
    )

    response = await handle_download({"paper_id": "2103.12345"})
    result = json.loads(response[0].text)

    assert result["status"] == "success"
    assert result["stored_locally"] is True
    assert result["source"] == "cache"
    assert result["paper_id"] == paper_id
    assert result["storage_dir"] == str(paths["bundle_dir"])
    assert result["resource_uri"] == f"arxiv://{paper_id}"
    assert "read_paper" in result["suggested_next_action"]
    assert "already stored locally" in result["message"]
    assert result["artifacts"]["markdown"]["status"] == "cached"
    assert result["artifacts"]["pdf"]["status"] == "downloaded"
    assert result["artifacts"]["source"]["status"] == "downloaded"
    assert result["content_length"] == len(content)
    assert result["next_start"] is None
    assert result["is_truncated"] is False
    assert paths["pdf"].exists()
    assert paths["source"].exists()
    html_fetch.assert_not_called()
    convert.assert_not_called()


@pytest.mark.asyncio
async def test_download_receipt_precedes_paginated_content(
    download_test_env, mocker, mock_paper
):
    """Large cached papers expose storage state alongside a bounded text page."""
    paper_id = "2103.12345v1"
    paths = get_bundle_paths(paper_id)
    paths["bundle_dir"].mkdir(parents=True, exist_ok=True)
    content = "abcdefghijklmnopqrstuvwxyz"
    paths["markdown"].write_text(content, encoding="utf-8")
    _write_artifact(paths["pdf"], b"%PDF")
    _write_artifact(paths["source"], b"source")

    mocker.patch.object(download_module, "_fetch_paper_result", return_value=mock_paper)

    response = await handle_download(
        {"paper_id": "2103.12345", "start": 10, "max_chars": 5}
    )
    result = json.loads(response[0].text)

    assert result["stored_locally"] is True
    assert result["content_length"] == len(content)
    assert result["start"] == 10
    assert result["returned_chars"] == 5
    assert result["next_start"] == 15
    assert result["is_truncated"] is True
    assert result["content"].split("\n\n", 1)[1] == "klmno"


@pytest.mark.asyncio
async def test_html_download_creates_complete_bundle(
    download_test_env, mocker, mock_paper
):
    """HTML-first downloads retain markdown, PDF, and source together."""
    html_text = "Title of the Paper\nAbstract content goes here."
    mocker.patch.object(download_module, "_fetch_paper_result", return_value=mock_paper)
    mocker.patch.object(download_module, "_fetch_html_content", return_value=html_text)
    convert = mocker.patch.object(download_module, "_convert_pdf_to_markdown")
    mocker.patch.object(
        download_module,
        "_download_binary_artifact",
        side_effect=_artifact_side_effect(),
    )

    response = await handle_download({"paper_id": "2103.12345"})
    result = json.loads(response[0].text)
    paths = get_bundle_paths("2103.12345v1")

    assert result["status"] == "success"
    assert result["stored_locally"] is True
    assert result["source"] == "html"
    assert result["paper_id"] == "2103.12345v1"
    assert result["requested_paper_id"] == "2103.12345"
    assert "downloaded, stored locally" in result["message"]
    assert result["artifacts"]["markdown"]["status"] == "downloaded_from_html"
    assert result["artifacts"]["pdf"]["status"] == "downloaded"
    assert result["artifacts"]["source"]["status"] == "downloaded"
    assert result["warnings"] == []
    assert paths["markdown"].read_text(encoding="utf-8") == html_text
    assert paths["pdf"].exists()
    assert paths["source"].exists()
    convert.assert_not_called()


@pytest.mark.asyncio
async def test_pdf_fallback_creates_bundle_and_retains_pdf(
    download_test_env, mocker, mock_paper
):
    """PDF fallback generates markdown without deleting the retained PDF."""
    markdown = "# PDF Paper\nConverted from PDF."
    mocker.patch.object(download_module, "_fetch_paper_result", return_value=mock_paper)
    mocker.patch.object(download_module, "_fetch_html_content", return_value=None)
    mocker.patch.object(
        download_module, "_convert_pdf_to_markdown", return_value=markdown
    )
    mocker.patch.object(
        download_module,
        "_download_binary_artifact",
        side_effect=_artifact_side_effect(),
    )

    response = await handle_download({"paper_id": "2103.12345v1"})
    result = json.loads(response[0].text)
    paths = get_bundle_paths("2103.12345v1")

    assert result["status"] == "success"
    assert result["stored_locally"] is True
    assert result["source"] == "pdf"
    assert result["artifacts"]["markdown"]["status"] == "generated_from_pdf"
    assert paths["markdown"].read_text(encoding="utf-8") == markdown
    assert paths["pdf"].exists()
    assert paths["source"].exists()


@pytest.mark.asyncio
async def test_sidecar_failure_keeps_markdown_success(
    download_test_env, mocker, mock_paper
):
    """A source failure is reported without hiding usable stored markdown."""
    source_error = ArtifactDownloadError(
        "Source archive download failed after retry: boom"
    )
    mocker.patch.object(download_module, "_fetch_paper_result", return_value=mock_paper)
    mocker.patch.object(
        download_module, "_fetch_html_content", return_value="HTML body"
    )
    mocker.patch.object(
        download_module,
        "_download_binary_artifact",
        side_effect=_artifact_side_effect(source_error=source_error),
    )

    response = await handle_download({"paper_id": "2103.12345"})
    result = json.loads(response[0].text)

    assert result["status"] == "success"
    assert result["stored_locally"] is True
    assert "sidecar warnings" in result["message"]
    assert result["artifacts"]["pdf"]["status"] == "downloaded"
    assert result["artifacts"]["source"]["status"] == "failed"
    assert "Source archive download failed" in result["warnings"][0]


@pytest.mark.asyncio
async def test_top_level_flat_files_are_ignored(download_test_env, mocker, mock_paper):
    """Flat legacy files are neither migrated nor treated as active bundles."""
    legacy_file = download_test_env / "1999.12345v1.md"
    legacy_file.write_text("old flat content", encoding="utf-8")
    dated_folder = download_test_env / "2026-04-12"
    dated_folder.mkdir(parents=True, exist_ok=True)
    notes = dated_folder / "notes.md"
    notes.write_text("leave me alone", encoding="utf-8")

    mocker.patch.object(download_module, "_fetch_paper_result", return_value=mock_paper)
    mocker.patch.object(
        download_module, "_fetch_html_content", return_value="new content"
    )
    mocker.patch.object(
        download_module,
        "_download_binary_artifact",
        side_effect=_artifact_side_effect(),
    )

    await handle_download({"paper_id": "2103.12345"})

    assert legacy_file.exists()
    assert notes.exists()


@pytest.mark.asyncio
async def test_paper_not_found_returns_error(download_test_env, mocker):
    """Unknown arXiv IDs return an MCP-recognizable error payload."""
    paper_id = "invalid.00000"
    mocker.patch.object(
        download_module,
        "_fetch_paper_result",
        side_effect=PaperNotFoundError(f"Paper {paper_id} not found on arXiv"),
    )

    response = await handle_download({"paper_id": paper_id})
    result = json.loads(response[0].text)

    assert result["status"] == "error"
    assert "not found on arXiv" in result["message"]


@pytest.mark.asyncio
async def test_invalid_canonical_id_never_becomes_a_path(
    download_test_env, mocker, mock_paper
):
    """Unexpected metadata cannot escape the configured storage directory."""
    mock_paper.get_short_id.return_value = "../../outside"
    mocker.patch.object(download_module, "_fetch_paper_result", return_value=mock_paper)

    response = await handle_download({"paper_id": "2103.12345"})
    result = json.loads(response[0].text)

    assert result["status"] == "error"
    assert "invalid canonical ID" in result["message"]
    assert not any(download_test_env.iterdir())


@pytest.mark.asyncio
async def test_unexpected_error_returns_error_payload(download_test_env, mocker):
    """Unexpected failures are reported without leaking a partial success."""
    mocker.patch.object(
        download_module,
        "_fetch_paper_result",
        side_effect=RuntimeError("network exploded"),
    )

    response = await handle_download({"paper_id": "2103.44444"})
    result = json.loads(response[0].text)

    assert result["status"] == "error"
    assert "network exploded" in result["message"]
