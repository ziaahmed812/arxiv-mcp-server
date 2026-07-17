"""Tests for bundle-backed resource management."""

import json

import pytest

from arxiv_mcp_server.paper_store import get_bundle_paths
from arxiv_mcp_server.resources.papers import PaperManager


@pytest.mark.asyncio
async def test_store_paper_uses_download_tool_flow(temp_storage_args, mocker):
    """PaperManager.store_paper should delegate to the bundle-aware download tool."""
    mocker.patch(
        "arxiv_mcp_server.resources.papers.handle_download",
        return_value=[
            type("TextContentLike", (), {"text": json.dumps({"status": "success"})})()
        ],
    )

    manager = PaperManager()

    assert await manager.store_paper("2103.12345", "https://arxiv.org/pdf/2103.12345v1")


@pytest.mark.asyncio
async def test_list_resources_uses_bundle_markdown_path(
    temp_storage_args, mocker, mock_paper
):
    """Paper resources should point at the bundle markdown file."""
    bundle = get_bundle_paths("2103.12345v1")
    bundle["bundle_dir"].mkdir(parents=True, exist_ok=True)
    bundle["markdown"].write_text("paper body", encoding="utf-8")

    manager = PaperManager()
    manager.client.results = mocker.MagicMock(return_value=[mock_paper])

    resources = await manager.list_resources()

    assert len(resources) == 1
    assert str(resources[0].uri) == bundle["markdown"].as_uri()
    assert resources[0].name == "Test Paper"


@pytest.mark.asyncio
async def test_get_paper_content_resolves_bare_id(temp_storage_args):
    """PaperManager should read the highest bundled version for a bare ID."""
    older = get_bundle_paths("2501.17913v1")
    newer = get_bundle_paths("2501.17913v2")
    older["bundle_dir"].mkdir(parents=True, exist_ok=True)
    newer["bundle_dir"].mkdir(parents=True, exist_ok=True)
    older["markdown"].write_text("older", encoding="utf-8")
    newer["markdown"].write_text("newer", encoding="utf-8")

    manager = PaperManager()
    content = await manager.get_paper_content("2501.17913")

    assert content == "newer"
