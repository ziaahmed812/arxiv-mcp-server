# Decision Log

## 2026-07-17 01:38 BST - Upstream v0.5.0 synchronization
Decision: Merge `upstream/main` into the fork without rebasing, preserving bundle-backed storage while adopting upstream transport, schema, streaming, and pagination changes.
Reason/impact: Existing paper bundles and fork history remain compatible; `download_paper` will explicitly confirm local storage and provide an optional `read_paper` nudge before returned content.
