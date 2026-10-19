#!/usr/bin/env python3
"""Compare a live capture of case F with its replay; exit 1 on the first difference.

Ignored keys: timestamps, timings, query ids, the environment block, the mode
and the replay sources. Keys ending in `elapsed_s` or `wall_s` and keys starting
with `latency_` are timings. The decision batches and totals depend on how
many answers came from the registry, so their counters (`registry_hits`,
`hit_rate`, `backend_calls`, `api_calls`, `backend`, `output_tokens`,
`cost_usd`) are not compared; every replay batch must report zero API calls.
Each recorded response is compared on its entity key, model id, answers and
input tokens, as in cases A, B and C: `source`, the raw `response`,
`output_tokens` and `cost_usd` are null for a registry-served decision. The
top-level `model` is skipped because the replay backend reports no model name.

Usage: compare_captures.py LIVE.json REPLAY.json
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

IGNORED = {"decided_at", "started_at", "finished_at", "admission_wait_ms", "query_id", "environment", "mode",
           "replay_sources"}
TIMING_SUFFIXES = ("elapsed_s", "wall_s")
COUNTERS = {"registry_hits", "hit_rate", "backend_calls", "api_calls", "backend", "input_tokens", "output_tokens",
            "cost_usd"}
RESPONSE_UNCOMPARED = {"source", "response", "output_tokens", "cost_usd"}


def ignored(key: str, path: tuple) -> bool:
    if key in IGNORED or key.startswith("latency_") or key.endswith(TIMING_SUFFIXES):
        return True
    if key == "model":
        return path == ()
    if path[:1] in (("decisions",), ("totals",)):
        return key in COUNTERS
    return len(path) == 2 and path[0] == "responses" and key in RESPONSE_UNCOMPARED


def first_difference(live, replay, path: tuple = ()) -> str | None:
    if isinstance(live, dict) and isinstance(replay, dict):
        for key in sorted(set(live) | set(replay)):
            if ignored(key, path):
                continue
            if key not in live or key not in replay:
                return f"{'/'.join(map(str, path + (key,)))}: present only in {'live' if key in live else 'replay'}"
            difference = first_difference(live[key], replay[key], path + (key,))
            if difference:
                return difference
        return None
    if isinstance(live, list) and isinstance(replay, list):
        if len(live) != len(replay):
            return f"{'/'.join(map(str, path))}: length {len(live)} != {len(replay)}"
        for index, (left, right) in enumerate(zip(live, replay)):
            difference = first_difference(left, right, path + (index,))
            if difference:
                return difference
        return None
    if live != replay:
        return f"{'/'.join(map(str, path))}: {live!r} != {replay!r}"
    return None


def replay_problems(replay: dict) -> str | None:
    for batch in replay.get("decisions", []):
        if batch.get("api_calls") != 0:
            return f"replay batch {batch.get('name')!r} reports api_calls={batch.get('api_calls')}"
    if (replay.get("totals") or {}).get("api_calls") != 0:
        return f"replay totals report api_calls={replay['totals'].get('api_calls')}"
    return None


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    live, replay = (json.loads(Path(name).read_text(encoding="utf-8")) for name in sys.argv[1:])
    if live.get("mode") != "live" or replay.get("mode") != "replay":
        print(f"expected a live and a replay capture, got {live.get('mode')!r} and {replay.get('mode')!r}")
        return 2
    problem = replay_problems(replay) or first_difference(live, replay)
    if problem:
        print(f"captures differ: {problem}")
        return 1
    print(f"captures match: {len(live.get('responses', {}))} responses, {len(live.get('stages', []))} stages, "
          f"{len(live.get('results', {}))} result entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
