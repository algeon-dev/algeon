"""Shared metadata readers for saved MerRec verification results."""

from __future__ import annotations

import json
from pathlib import Path


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
