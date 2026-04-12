"""Resource management and storage for arXiv papers."""

import json
import logging
from pathlib import Path
from typing import List

import aiofiles
import arxiv
import mcp.types as types
from pydantic import AnyUrl

from ..config import Settings
from ..paper_store import (
    ensure_storage_layout_prepared,
    get_bundle_paths,
    list_active_paper_ids,
    resolve_local_paper_id,
)
from ..tools.download import handle_download

logger = logging.getLogger("arxiv-mcp-server")


class PaperManager:
    """Manages the storage, retrieval, and resource handling of arXiv papers."""

    def __init__(self):
        """Initialize the paper management system."""
        settings = Settings()
        self.storage_path = Path(settings.STORAGE_PATH)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        ensure_storage_layout_prepared()
        self.client = arxiv.Client()

    def _get_paper_path(self, paper_id: str) -> Path:
        """Get the markdown path for a stored paper bundle."""
        return get_bundle_paths(paper_id)["markdown"]

    async def store_paper(self, paper_id: str, pdf_url: str) -> bool:
        """Download and store a paper bundle from arXiv."""
        del pdf_url  # Canonical metadata lookup happens inside download_paper.

        response = await handle_download({"paper_id": paper_id})
        payload = json.loads(response[0].text)
        if payload.get("status") == "success":
            return True

        raise ValueError(payload.get("message", f"Failed to store paper {paper_id}."))

    async def has_paper(self, paper_id: str) -> bool:
        """Check if a paper bundle is available in storage."""
        return resolve_local_paper_id(paper_id) is not None

    async def list_papers(self) -> list[str]:
        """List all stored paper IDs."""
        logger.info("Listing papers in %s", self.storage_path)
        paper_ids = list_active_paper_ids()
        logger.info("Found %s papers", len(paper_ids))
        return paper_ids

    async def list_resources(self) -> List[types.Resource]:
        """List all papers as MCP resources with metadata."""
        paper_ids = await self.list_papers()
        resources = []

        for paper_id in paper_ids:
            search = arxiv.Search(id_list=[paper_id])
            papers = list(self.client.results(search))
            paper_path = self._get_paper_path(paper_id)

            if papers:
                paper = papers[0]
                resources.append(
                    types.Resource(
                        uri=AnyUrl(f"file://{str(paper_path)}"),
                        name=paper.title,
                        description=paper.summary,
                        mimeType="text/markdown",
                    )
                )
            else:
                resources.append(
                    types.Resource(
                        uri=AnyUrl(f"file://{str(paper_path)}"),
                        name=paper_id,
                        description="Downloaded arXiv paper",
                        mimeType="text/markdown",
                    )
                )

        logger.info("Found %s resources", len(resources))
        return resources

    async def get_paper_content(self, paper_id: str) -> str:
        """Get the markdown content of a stored paper."""
        resolved_paper_id = resolve_local_paper_id(paper_id)
        if resolved_paper_id is None:
            raise ValueError(f"Paper {paper_id} not found in storage")

        paper_path = self._get_paper_path(resolved_paper_id)
        if not paper_path.exists():
            raise ValueError(f"Paper {paper_id} not found in storage")

        async with aiofiles.open(paper_path, "r", encoding="utf-8") as f:
            return await f.read()
