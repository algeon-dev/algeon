"""Shared metadata readers for saved MerRec verification results."""

from __future__ import annotations

import json
from pathlib import Path


# Every capture saved under website/static/data is derived from the MerRec
# dataset and carries its terms, not the repository's Apache-2.0 license. The
# field stays first so a copied file shows its terms before any result row.
MERREC_LICENSE = (
    "CC-BY-NC-4.0; derived from the MerRec dataset, Copyright Mercari, Inc.; "
    "see LICENSE-MERREC.md in this directory"
)


def query_record(query_records: Path, root: Path, name: str) -> dict[str, object]:
    path = query_records / f"{name}.json"
    if not path.is_file():
        return {"path": str(path.relative_to(root)), "status": "missing"}
    record = json.loads(path.read_text())
    return {
        "path": str(path.relative_to(root)),
        "status": record.get("status"),
        "query_id": record.get("query_id"),
        "rows": record.get("rows"),
        "error": record.get("error"),
    }
