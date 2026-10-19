"""Compare each label's replay capture with its live capture; exit 1 on the first difference.

    python3 cases/b_bridge_purification/compare_captures.py all sep_oct

Timestamps, timings, query ids and run provenance are ignored. A live run
calls the backend and a replay is served from the registry, so the per-batch
decision summaries and totals are left out, and `registry_hits` and
`api_calls` are ignored wherever they appear, including the per-pass records.
Each recorded response is compared on its entity key, model id, answers and
input tokens; `source`, the raw `response`, `output_tokens` and `cost_usd` are
null for a registry-served decision and present for a fetched one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

CASE = Path(__file__).resolve().parent.name
CAPTURES_DIR = Path(__file__).resolve().parents[3] / "captures"
IGNORED = {"decided_at", "started_at", "finished_at", "elapsed_s", "wall_s", "admission_wait_ms", "query_id",
           "environment", "mode", "replay_sources", "registry_hits", "api_calls"}
IGNORED_PREFIXES = ("latency_",)
UNCOMPARED = {"decisions", "responses", "totals", "model"}
BATCH_FIELDS = ("name", "entities", "distinct_states", "model_ids")
RESPONSE_FIELDS = ("entity_key", "model_id", "answers", "input_tokens")


def ignored(key: str) -> bool:
    return key in IGNORED or key.startswith(IGNORED_PREFIXES)


def first_difference(left, right, path: str = "$") -> str | None:
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            if ignored(key):
                continue
            if key not in left or key not in right:
                return f"{path}.{key}: present only in {'live' if key in left else 'replay'}"
            found = first_difference(left[key], right[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return f"{path}: length {len(left)} vs {len(right)}"
        for index, (a, b) in enumerate(zip(left, right)):
            found = first_difference(a, b, f"{path}[{index}]")
            if found:
                return found
        return None
    if left != right:
        return f"{path}: {json.dumps(left, default=str)[:200]} vs {json.dumps(right, default=str)[:200]}"
    return None


def comparable(capture: dict) -> dict:
    view = {key: value for key, value in capture.items() if key not in UNCOMPARED}
    view["decision_batches"] = [{key: batch.get(key) for key in BATCH_FIELDS} for batch in capture.get("decisions", [])]
    view["responses"] = {key: {field: record.get(field) for field in RESPONSE_FIELDS}
                         for key, record in (capture.get("responses") or {}).items()}
    return view


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("labels", nargs="+", help="capture labels, for example all sep_oct")
    parser.add_argument("--captures-dir", type=Path, default=CAPTURES_DIR)
    args = parser.parse_args()
    status = 0
    for label in args.labels:
        paths = [args.captures_dir / f"{CASE}_{label}_{mode}.json" for mode in ("live", "replay")]
        missing = [path.name for path in paths if not path.exists()]
        if missing:
            print(f"{label}: missing {', '.join(missing)}")
            status = 1
            continue
        live, replay = (comparable(json.loads(path.read_text(encoding="utf-8"))) for path in paths)
        difference = first_difference(live, replay)
        print(f"{label}: {'identical' if difference is None else 'differs at ' + difference}")
        status |= difference is not None
    return status


if __name__ == "__main__":
    sys.exit(main())
