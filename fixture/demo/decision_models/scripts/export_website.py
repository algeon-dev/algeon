"""Export the website data files for cases A and B from their local captures.

The website pages and the blog post read `merrec-decision-bridges.json` and
`merrec-decision-twins.json` from `website/static/data`. Each file holds the
question sets, the policy, the executed SQL, the GPU stage metrics, the
decision batch counters, every judgment and every result row of the recorded
runs, under the MerRec CC BY-NC terms like the other `merrec-*.json` files
there.

The public files name no decision backend or model: the capture's `backend`,
`model_ids` and per-entity `model_id` fields are dropped, and so is the
`content_hash` registry key. `check_no_backend_fields` enforces that.

Usage, from fixture/demo/decision_models/scripts:

    python3 export_website.py            # writes both files
    python3 export_website.py --check    # verifies the files on disk match
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from client.algeon import ROOT  # noqa: E402

DEMO_DIR = Path(__file__).resolve().parents[1]
CASES_DIR = DEMO_DIR / "scripts" / "cases"
CAPTURES_DIR = DEMO_DIR / "captures"
DATA_DIR = ROOT / "website" / "static" / "data"
LICENSE = ("CC-BY-NC-4.0; derived from the MerRec dataset, Copyright Mercari, Inc.; "
           "see LICENSE-MERREC.md in this directory")
BACKEND_FIELDS = frozenset({"backend", "model_ids", "model_id", "content_hash", "replay_sources"})
BATCH_FIELDS = ("name", "entities", "distinct_states", "registry_hits", "hit_rate", "api_calls", "wall_s",
                "input_tokens", "output_tokens", "cost_usd", "latency_ms_p50", "latency_ms_p95", "latency_ms_max")
TOTAL_FIELDS = ("gpu_elapsed_s", "decide_wall_s", "entities", "registry_hits", "api_calls", "input_tokens",
                "output_tokens", "cost_usd", "max_host_boundaries")
B_WINDOWS = {"all": ("2023-05-01", "2023-11-01"), "sep_oct": ("2023-09-01", "2023-11-01")}


def load_capture(name: str) -> dict:
    return json.loads((CAPTURES_DIR / f"{name}.json").read_text(encoding="utf-8"))


def load_json(case: str, name: str):
    return json.loads((CASES_DIR / case / name).read_text(encoding="utf-8"))


def load_sql(case: str, name: str, **values: str) -> str:
    sql = (CASES_DIR / case / name).read_text(encoding="utf-8")
    for key, value in values.items():
        sql = sql.replace("{" + key + "}", value)
    return sql


def stage_record(stage: dict) -> dict:
    execution = stage.get("execution") or {}
    return {
        "name": stage["name"],
        "rows": stage["rows"],
        "elapsed_s": stage["elapsed_s"],
        "query_id": stage.get("query_id"),
        "native_fragments": execution.get("native_plan_fragment_count"),
        "host_boundaries": execution.get("host_datafusion_boundary_count"),
        "allocator_peak_bytes": execution.get("allocator_high_water_bytes"),
        "query_limit_bytes": execution.get("query_grant_bytes"),
    }


def batch_record(batch: dict) -> dict:
    return {key: batch[key] for key in BATCH_FIELDS if key in batch}


def totals_record(totals: dict) -> dict:
    return {key: totals[key] for key in TOTAL_FIELDS if key in totals}


def strip_backend_fields(value):
    if isinstance(value, dict):
        return {key: strip_backend_fields(item) for key, item in value.items() if key not in BACKEND_FIELDS}
    if isinstance(value, list):
        return [strip_backend_fields(item) for item in value]
    return value


def check_no_backend_fields(value, path: str = "") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in BACKEND_FIELDS:
                raise ValueError(f"backend field {key!r} at {path or '<root>'}")
            check_no_backend_fields(item, f"{path}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            check_no_backend_fields(item, f"{path}[{index}]")


def bridges_window(capture: dict, *, canonical: dict | None = None) -> dict:
    results = capture["results"]
    record = {
        "window": results["window"],
        "stages": [stage_record(stage) for stage in capture["stages"]],
        "decisions": [batch_record(batch) for batch in capture["decisions"]],
        "totals": totals_record(capture["totals"]),
        "passes": [{key: item[key] for key in ("pass", "new_entries", "new_band_counts", "registry_hits", "api_calls")}
                   for item in results["passes"]],
        "band_counts": results["band_counts"],
        "kind_counts": results["kind_counts"],
        "ranking": results["ranking"],
        "review_queue": results["review_queue"],
        "judgments": results["decisions"],
        "denylist_vs_model": results["denylist_vs_model"],
    }
    if canonical is not None:
        record["canonical_ranking"] = canonical["results"]["canonical_ranking"]
        record["canonical_run"] = {
            "stages": [stage_record(stage) for stage in canonical["stages"]],
            "totals": totals_record(canonical["totals"]),
        }
    return record


def export_bridges() -> dict:
    case = "b_bridge_purification"
    all_window = load_capture(f"{case}_all_live")
    canonical = load_capture(f"{case}_all_canonical_live")
    sep_oct = load_capture(f"{case}_sep_oct_live")
    start, end = B_WINDOWS["all"]
    return {
        "license": LICENSE,
        "name": case,
        "description": "World bridges recomputed without the catalog entries a decision model rejects as "
                       "standalone brands; the published betweenness ranking, the post-filtered ranking and "
                       "the purified ranking over the same window, with a denylist control and a second window.",
        "captures": [f"{case}_all_live.json", f"{case}_all_canonical_live.json", f"{case}_sep_oct_live.json"],
        "server_binary_sha256": all_window["environment"]["server_binary_sha256"],
        "questions": load_json(case, "questions.json"),
        "policy": load_json(case, "policy.json"),
        "control_denylist": load_json(case, "control_denylist.json"),
        "sql": {
            "probe": load_sql(case, "01_probe.sql", window_start=start, window_end=end),
            "context": load_sql(case, "02_context.sql"),
            "recompute": load_sql(case, "03_recompute.sql", ontology_table="brand_ontology",
                                  window_start=start, window_end=end),
            "recompute_canonical": load_sql(case, "04_recompute_canonical.sql", ontology_table="brand_ontology",
                                            window_start=start, window_end=end),
        },
        "windows": {
            "all": bridges_window(all_window, canonical=canonical),
            "sep_oct": bridges_window(sep_oct),
        },
    }


def export_twins() -> dict:
    case = "a_brand_resolution"
    capture = load_capture(f"{case}_live")
    results = capture["results"]
    return {
        "license": LICENSE,
        "name": case,
        "description": "Brand twins resolved by a decision model: the 120 highest fan-Jaccard pairs among the "
                       "2,000 most liked accepted brands, the pairs accepted as one brand, the exact fan counts "
                       "of the merged brands and a trigram control over all 20,000 names.",
        "captures": [f"{case}_live.json"],
        "server_binary_sha256": capture["environment"]["server_binary_sha256"],
        "questions": load_json(case, "questions.json"),
        "policy": load_json(case, "policy.json"),
        "ontology": results["ontology"],
        "sql": {
            "probe": load_sql(case, "01_probe.sql", max_entities="120"),
            "context": load_sql(case, "02_context.sql"),
            "canonical": load_sql(case, "03a_canonical.sql"),
            "dedup": load_sql(case, "03_recompute.sql"),
        },
        "stages": [stage_record(stage) for stage in capture["stages"]],
        "decisions": [batch_record(batch) for batch in capture["decisions"]],
        "totals": totals_record(capture["totals"]),
        "band_counts": results["band_counts"],
        "relationship_counts": results["relationship_counts"],
        "pairs": results["pairs"],
        "review_queue": results["review_queue"],
        "clusters": results["clusters"],
        "trigram_control": results["trigram_control"],
        "full_pairwise_cost_estimate": results["full_pairwise_cost_estimate"],
    }


EXPORTS = {
    "merrec-decision-bridges.json": export_bridges,
    "merrec-decision-twins.json": export_twins,
}


def render(name: str) -> str:
    data = strip_backend_fields(EXPORTS[name]())
    check_no_backend_fields(data)
    return json.dumps(data, indent=1, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare with the files on disk instead of writing")
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    status = 0
    for name in EXPORTS:
        text = render(name)
        path = args.out_dir / name
        if args.check:
            current = path.read_text(encoding="utf-8") if path.exists() else None
            state = "matches" if current == text else "differs"
            status |= state == "differs"
            print(f"{path}: {state}")
        else:
            path.write_text(text, encoding="utf-8")
            print(f"{path}: {len(text):,} bytes")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
