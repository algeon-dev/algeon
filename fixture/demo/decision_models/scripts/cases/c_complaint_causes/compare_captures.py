#!/usr/bin/env python3
"""Compare a live and a replay capture of case C; exit 1 at the first difference.

Usage: python3 compare_captures.py <live.json> <replay.json>

Ignored: timestamps, timings, query ids, the environment, the mode and the
replay sources. The top-level requested `model` is skipped (replay has none;
the answered `model_ids` are still compared). The decision batches' counters
(`backend`, `api_calls`, `registry_hits`, `hit_rate`, `backend_calls`,
`input_tokens`, `output_tokens`, `cost_usd`) depend on how many answers the registry served,
so they are not compared, as in cases A and B; every replay batch must still
report zero API calls. Each recorded response is compared on its entity key,
model id, answers and input tokens; `source`, the raw `response`,
`output_tokens` and `cost_usd` are null for a registry-served decision.

Stages are matched by name. A stage missing from one capture is accepted only
when that capture lists it in `results.skipped_stages`, which `--reuse-clusters`
does for the kmeans probe.
"""

from __future__ import annotations

import json
import sys

IGNORED_KEYS = {
    "decided_at", "started_at", "finished_at", "elapsed_s", "wall_s", "admission_wait_ms", "query_id",
    "environment", "mode", "replay_sources",
}
MODE_KEYS = {("model",)}
# A live run counts registry hits, backend calls, tokens and cost against the backend it used; a replay
# counts them against the replay backend and its own fresh registry, so the batch counters are not compared.
COUNTER_KEYS = {"backend", "api_calls", "registry_hits", "hit_rate", "backend_calls", "input_tokens", "output_tokens", "cost_usd"}
RESPONSE_UNCOMPARED = {"source", "response", "output_tokens", "cost_usd"}


class Difference(Exception):
    pass


def ignored(path: tuple, key: str) -> bool:
    if key in IGNORED_KEYS or key.startswith("latency_") or key.endswith(("_elapsed_s", "_wall_s")):
        return True
    if len(path) == 2 and path[0] == "responses" and key in RESPONSE_UNCOMPARED:
        return True
    if path[:1] in (("decisions",), ("totals",)) and key in COUNTER_KEYS:
        return True
    generic = tuple("*" if isinstance(part, int) else part for part in (*path, key))
    return generic in MODE_KEYS


def compare(left, right, path: tuple = ()) -> None:
    where = "/".join(map(str, path)) or "<root>"
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            if ignored(path, key):
                continue
            if key not in left or key not in right:
                raise Difference(f"{where}/{key}: present only in the {'live' if key in left else 'replay'} capture")
            compare(left[key], right[key], (*path, key))
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise Difference(f"{where}: {len(left)} items in live, {len(right)} in replay")
        for index, (a, b) in enumerate(zip(left, right)):
            compare(a, b, (*path, index))
    elif left != right:
        raise Difference(f"{where}: live {left!r} != replay {right!r}")


def compare_stages(live: dict, replay: dict) -> list[str]:
    left = {stage["name"]: stage for stage in live["stages"]}
    right = {stage["name"]: stage for stage in replay["stages"]}
    notes = []
    for name in sorted(set(left) | set(right)):
        if name not in right:
            if name not in (replay.get("results") or {}).get("skipped_stages", []):
                raise Difference(f"stages/{name}: missing from the replay capture")
            notes.append(f"stage {name} skipped in replay")
        elif name not in left:
            if name not in (live.get("results") or {}).get("skipped_stages", []):
                raise Difference(f"stages/{name}: missing from the live capture")
            notes.append(f"stage {name} skipped in live")
        else:
            compare(left[name], right[name], ("stages", name))
    return notes


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.splitlines()[2], file=sys.stderr)
        return 2
    with open(sys.argv[1], encoding="utf-8") as handle:
        live = json.load(handle)
    with open(sys.argv[2], encoding="utf-8") as handle:
        replay = json.load(handle)
    try:
        for batch in replay.get("decisions", []):
            if batch.get("api_calls") != 0:
                raise Difference(f"replay batch {batch.get('name')!r} reports api_calls={batch.get('api_calls')}")
        notes = compare_stages(live, replay)
        rest = lambda capture: {key: value for key, value in capture.items() if key != "stages"}
        live_rest, replay_rest = rest(live), rest(replay)
        for capture in (live_rest, replay_rest):
            capture["results"] = {key: value for key, value in capture["results"].items() if key != "skipped_stages"}
        compare(live_rest, replay_rest)
    except Difference as difference:
        print(f"DIFFERENT {difference}")
        return 1
    print(f"SAME stages={len(replay['stages'])} decisions={sum(batch['entities'] for batch in replay['decisions'])} "
          f"results={len(replay['results'])} " + "; ".join(notes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
