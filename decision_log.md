# Decision Log

## 2026-07-17 01:38 BST - Upstream v0.5.0 synchronization
Decision: Merge `upstream/main` into the fork without rebasing, preserving bundle-backed storage while adopting upstream transport, schema, streaming, and pagination changes.
Reason/impact: Existing paper bundles and fork history remain compatible; `download_paper` will explicitly confirm local storage and provide an optional `read_paper` nudge before returned content.

## 2026-07-17 01:59 BST - Cross-platform CI correction
Decision: Generate expected file URIs with `Path.as_uri()` and use the Node 24 GitHub Actions majors; keep the test matrix running after an individual failure.
Reason/impact: Windows paths are validated with the same URI rules as production, deprecation warnings are removed, and future failures retain diagnostics from every supported platform.
