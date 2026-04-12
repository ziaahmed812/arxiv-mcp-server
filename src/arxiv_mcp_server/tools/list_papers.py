"""List functionality for the arXiv MCP server."""

import json
from typing import Any, Dict, List, Optional

import mcp.types as types
from mcp.types import ToolAnnotations

from ..paper_store import is_valid_arxiv_id as _is_valid_arxiv_id, list_active_paper_ids


def is_valid_arxiv_id(stem: str) -> bool:
    """Backward-compatible wrapper for arXiv ID validation."""
    return _is_valid_arxiv_id(stem)


list_tool = types.Tool(
    name="list_papers",
    annotations=ToolAnnotations(readOnlyHint=True),
    description=(
        "List all papers that have been downloaded and stored locally via download_paper. "
        "Returns arXiv IDs only — use read_paper to access content. "
        "Returns an empty list if no papers have been downloaded yet. "
        "Workflow: search_papers -> download_paper -> list_papers -> read_paper."
    ),
    inputSchema={
        "type": "object",
        "properties": {},
        "required": [],
    },
)


def list_papers() -> list[str]:
    """List all active bundle-backed paper IDs."""
    return list_active_paper_ids()


async def handle_list_papers(
    arguments: Optional[Dict[str, Any]] = None,
) -> List[types.TextContent]:
    """Handle requests to list all stored papers."""
    try:
        papers = list_papers()

        if not papers:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps({"total_papers": 0, "papers": []}, indent=2),
                )
            ]

        response_data = {
            "total_papers": len(papers),
            "papers": papers,
        }

        return [
            types.TextContent(type="text", text=json.dumps(response_data, indent=2))
        ]
    except Exception as e:
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]
