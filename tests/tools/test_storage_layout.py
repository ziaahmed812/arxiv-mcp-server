"""Tests for bundle-backed list/read behavior."""

import json

import pytest

from arxiv_mcp_server.paper_store import get_bundle_paths
from arxiv_mcp_server.tools.list_papers import handle_list_papers
from arxiv_mcp_server.tools.read_paper import handle_read_paper


@pytest.mark.asyncio
async def test_list_papers_only_returns_active_bundles(temp_storage_args):
    """list_papers should ignore flat files and unrelated folders."""
    bundle_paths = get_bundle_paths("2501.00001v2")
    bundle_paths["bundle_dir"].mkdir(parents=True, exist_ok=True)
    bundle_paths["markdown"].write_text("paper body", encoding="utf-8")

    (temp_storage_args / "2401.00001v1.md").write_text("old paper", encoding="utf-8")
    dated_folder = temp_storage_args / "2026-04-12"
    dated_folder.mkdir(parents=True, exist_ok=True)
    (dated_folder / "README.md").write_text("reading pack", encoding="utf-8")

    response = await handle_list_papers()
    payload = json.loads(response[0].text)

    assert payload["total_papers"] == 1
    assert payload["papers"] == ["2501.00001v2"]


@pytest.mark.asyncio
async def test_read_paper_resolves_bare_id_to_latest_bundle(temp_storage_args):
    """read_paper should resolve bare IDs to the highest local bundled version."""
    older = get_bundle_paths("2603.23432v1")
    newer = get_bundle_paths("2603.23432v3")
    older["bundle_dir"].mkdir(parents=True, exist_ok=True)
    newer["bundle_dir"].mkdir(parents=True, exist_ok=True)
    older["markdown"].write_text("older version", encoding="utf-8")
    newer["markdown"].write_text("newer version", encoding="utf-8")

    response = await handle_read_paper({"paper_id": "2603.23432"})
    payload = json.loads(response[0].text)

    assert payload["status"] == "success"
    assert payload["paper_id"] == "2603.23432v3"
    assert payload["content"].endswith("newer version")


@pytest.mark.asyncio
async def test_read_paper_missing_returns_error(temp_storage_args):
    """Missing bundled papers should return a clear error message."""
    response = await handle_read_paper({"paper_id": "9999.99999"})
    payload = json.loads(response[0].text)

    assert payload["status"] == "error"
    assert "download it first" in payload["message"]
