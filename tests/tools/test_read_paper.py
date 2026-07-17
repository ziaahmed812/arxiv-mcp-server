"""Tests for reading bundle-backed downloaded papers."""

import json

import pytest

from arxiv_mcp_server.paper_store import get_bundle_paths
from arxiv_mcp_server.tools.read_paper import handle_read_paper


def _store_markdown(paper_id: str, content: str) -> None:
    paths = get_bundle_paths(paper_id)
    paths["bundle_dir"].mkdir(parents=True, exist_ok=True)
    paths["markdown"].write_text(content, encoding="utf-8")


@pytest.mark.asyncio
async def test_read_paper_supports_content_pagination(temp_storage_args):
    """Large papers can be retrieved in bounded chunks."""
    paper_id = "2505.13525v2"
    content = "abcdefghijklmnopqrstuvwxyz"
    _store_markdown(paper_id, content)

    response = await handle_read_paper(
        {"paper_id": "2505.13525", "start": 5, "max_chars": 10}
    )
    result = json.loads(response[0].text)

    assert result["status"] == "success"
    assert result["stored_locally"] is True
    assert result["paper_id"] == paper_id
    assert result["requested_paper_id"] == "2505.13525"
    assert result["storage_dir"] == str(get_bundle_paths(paper_id)["bundle_dir"])
    assert result["content_length"] == len(content)
    assert result["start"] == 5
    assert result["returned_chars"] == 10
    assert result["next_start"] == 15
    assert result["is_truncated"] is True
    assert result["content"].split("\n\n", 1)[1] == "fghijklmno"


@pytest.mark.asyncio
async def test_read_paper_reports_final_chunk(temp_storage_args):
    """The final page clearly reports that no continuation remains."""
    paper_id = "2505.13525"
    content = "abcdefghijklmnopqrstuvwxyz"
    _store_markdown(paper_id, content)

    response = await handle_read_paper(
        {"paper_id": paper_id, "start": 20, "max_chars": 20}
    )
    result = json.loads(response[0].text)

    assert result["status"] == "success"
    assert result["returned_chars"] == 6
    assert result["next_start"] is None
    assert result["is_truncated"] is False
    assert result["content"].endswith("uvwxyz")


@pytest.mark.asyncio
async def test_read_paper_missing_bundle_returns_error(temp_storage_args):
    """A missing local bundle tells the agent to download first."""
    response = await handle_read_paper({"paper_id": "9999.99999"})
    result = json.loads(response[0].text)

    assert result["status"] == "error"
    assert "download it first" in result["message"]
