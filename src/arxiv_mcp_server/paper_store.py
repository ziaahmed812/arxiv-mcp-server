"""Shared storage helpers for bundle-backed arXiv paper downloads."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, Optional

from .config import Settings

settings = Settings()

MARKDOWN_FILENAME = "paper.md"
PDF_FILENAME = "paper.pdf"
SOURCE_FILENAME = "source.tar.gz"
_SLASH_TOKEN = "__slash__"

_ARXIV_ID_RE = re.compile(
    r"^(\d{4}\.\d{4,5}(v\d+)?|[a-z\-]+(/[a-z\-]+)?/\d{7}(v\d+)?)$",
    re.IGNORECASE,
)


def storage_path() -> Path:
    """Return the configured storage path, ensuring it exists."""
    path = Path(settings.STORAGE_PATH)
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_valid_arxiv_id(paper_id: str) -> bool:
    """Return True if *paper_id* looks like a valid arXiv identifier."""
    return bool(_ARXIV_ID_RE.match(paper_id))


def paper_id_to_bundle_name(paper_id: str) -> str:
    """Map an arXiv ID to a filesystem-safe bundle directory name."""
    return paper_id.replace("/", _SLASH_TOKEN)


def bundle_name_to_paper_id(bundle_name: str) -> str:
    """Recover the original arXiv ID from a bundle directory name."""
    return bundle_name.replace(_SLASH_TOKEN, "/")


def split_version(paper_id: str) -> tuple[str, Optional[int]]:
    """Split a paper ID into its base identifier and numeric version."""
    match = re.search(r"v(\d+)$", paper_id)
    if match is None:
        return paper_id, None
    return paper_id[: match.start()], int(match.group(1))


def get_bundle_paths(paper_id: str) -> dict[str, Path]:
    """Return the bundle directory plus canonical artifact paths for a paper."""
    bundle_dir = storage_path() / paper_id_to_bundle_name(paper_id)
    return {
        "bundle_dir": bundle_dir,
        "markdown": bundle_dir / MARKDOWN_FILENAME,
        "pdf": bundle_dir / PDF_FILENAME,
        "source": bundle_dir / SOURCE_FILENAME,
    }


def has_content(path: Path) -> bool:
    """Return True when a path exists and is non-empty."""
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


def iter_active_bundles() -> Iterator[tuple[str, Path]]:
    """Yield active paper bundles that contain a stored markdown file."""
    root = storage_path()

    for item in sorted(root.iterdir(), key=lambda path: path.name):
        if not item.is_dir():
            continue

        paper_id = bundle_name_to_paper_id(item.name)
        if not is_valid_arxiv_id(paper_id):
            continue

        if has_content(item / MARKDOWN_FILENAME):
            yield paper_id, item


def list_active_paper_ids() -> list[str]:
    """List all active bundle-backed paper IDs."""
    return [paper_id for paper_id, _ in iter_active_bundles()]


def resolve_local_paper_id(requested_paper_id: str) -> Optional[str]:
    """Resolve a local paper ID, preferring the highest bundled version."""
    if not is_valid_arxiv_id(requested_paper_id):
        return None

    exact_markdown_path = get_bundle_paths(requested_paper_id)["markdown"]
    if has_content(exact_markdown_path):
        return requested_paper_id

    requested_base, requested_version = split_version(requested_paper_id)
    if requested_version is not None:
        return None

    best_match: Optional[str] = None
    best_version = -1

    for paper_id in list_active_paper_ids():
        base_id, version = split_version(paper_id)
        if base_id != requested_base:
            continue

        numeric_version = version or 0
        if numeric_version > best_version:
            best_match = paper_id
            best_version = numeric_version

    return best_match
