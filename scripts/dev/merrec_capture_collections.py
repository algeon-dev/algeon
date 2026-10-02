#!/usr/bin/env python3
"""Rebuild the public MerRec market-collection capture from saved research inputs."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import re

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "docs/research/merrec"
SQL = REPORT / "sql/24_market_collections.sql"
RESULT = REPORT / "results/24_market_collections_final.json"
RESULT_PARQUET = ROOT / "target/merrec-exploration/results/24_market_collections_final.parquet"
STATS = REPORT / "results/24_market_collections_final.stats.txt.gz"
VALIDATION = REPORT / "community_validation.json"
OUTPUT = ROOT / "website/static/data/merrec-community-capture.json"

SOURCE_PATHS = {
    "query_sql": SQL,
    "graph_preparation_sql": REPORT / "sql/01_graph_views.sql",
    "dataset_manifest": ROOT / "fixture/demo/merrec/manifest.json",
    "graph_validation": REPORT / "graph_validation.json",
    "result_record": RESULT,
    "result_parquet": RESULT_PARQUET,
    "query_stats": STATS,
    "nodes_snapshot": ROOT / "target/merrec-exploration/results/prepare_nodes.parquet",
    "directed_edges_snapshot": ROOT / "target/merrec-exploration/results/prepare_edges.parquet",
    "undirected_edges_snapshot": ROOT / "target/merrec-exploration/results/prepare_undirected.parquet",
    "full_pagerank_snapshot": ROOT / "target/merrec-exploration/results/11_pagerank_full.parquet",
    "cpu_validation": VALIDATION,
    "capture_generator": Path(__file__).resolve(),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stats_section(text: str, name: str) -> dict[str, str]:
    marker = f"[{name}]\n"
    if marker not in text:
        raise ValueError(f"Query stats have no [{name}] section")
    section = text.split(marker, 1)[1].split("\n\n", 1)[0]
    lines = section.splitlines()
    if len(lines) < 2:
        raise ValueError(f"Query stats [{name}] section has no data row")
    columns = lines[0].split("\t")
    values = lines[1].split("\t")
    if len(columns) != len(values):
        raise ValueError(f"Query stats [{name}] row does not match its header")
    return dict(zip(columns, values, strict=True))


def algorithm_summary(graph_runtime_rows: str) -> list[dict[str, str]]:
    results = []
    for source in graph_runtime_rows.split("|"):
        match = re.search(
            r"algorithm=([^,]+),runtime_outcome=([^,]+).*?"
            r"input_boundary=([^,]+),output_boundary=([^,]+)",
            source,
        )
        if match:
            results.append(
                {
                    "algorithm": match.group(1),
                    "runtime_outcome": match.group(2),
                    "input_boundary": match.group(3),
                    "output_boundary": match.group(4),
                }
            )
    return results


def main() -> None:
    missing = [str(path.relative_to(ROOT)) for path in SOURCE_PATHS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required saved collection inputs are missing: {missing}")

    record = json.loads(RESULT.read_text())
    sql = SQL.read_text()
    if record["status"] != "ok" or record["sql"] != sql:
        raise ValueError("The final result record does not match the executed SQL capture")

    table = pq.read_table(RESULT_PARQUET)
    rows = table.to_pylist()
    if len(rows) != record["rows"] or len(rows) != 110:
        raise ValueError(f"Expected 110 result rows, found {len(rows)}")

    stats_text = gzip.decompress(STATS.read_bytes()).decode("utf-8")
    planning = stats_section(stats_text, "planning")
    execution = stats_section(stats_text, "execution")
    if (
        planning["native_plan_fragment_count"] != "2"
        or planning["host_datafusion_boundary_count"] != "15"
    ):
        raise ValueError("The saved query stats do not match the documented mixed execution")
    server_elapsed = re.search(r"^# elapsed_ms=(\d+)$", stats_text, re.MULTILINE)
    if server_elapsed is None:
        raise ValueError("Query stats have no elapsed_ms observation")

    validation = json.loads(VALIDATION.read_text())
    composition = validation["market_collections_composition"]
    expected_checks = {
        "representative_vertices_are_known_and_unique",
        "market_metadata_matches_nodes",
        "pagerank_matches_saved_full_snapshot",
        "five_representatives_per_returned_community",
        "community_sizes_are_consistent_and_at_least_ten",
        "representative_rank_matches_pagerank_order",
    }
    if composition["status"] != "assessed" or set(composition["checks"]) != expected_checks:
        raise ValueError("The saved CPU collection assessment is incomplete")

    capture = {
        "query": {
            "name": record["name"],
            "query_id": record["query_id"],
            "status": record["status"],
            "rows": record["rows"],
            "schema": record["schema"],
            "sql": sql,
        },
        "execution": {
            "elapsed_s": record["elapsed_s"],
            "server_stats_elapsed_ms": int(server_elapsed.group(1)),
            "observation_count": 1,
            "timing_boundary": (
                "One Flight elapsed observation including planning and complete result receipt; "
                "graph preparation excluded. The server was reused after many previous queries."
            ),
            "native_fragment_count": int(planning["native_plan_fragment_count"]),
            "host_datafusion_boundary_count": int(planning["host_datafusion_boundary_count"]),
            "execution_path": execution["post_native_execution_path"],
            "host_execution_note": (
                "Leiden and PageRank ran as native GPU fragments; joins, repartitioning, "
                "and ranked-window processing used host DataFusion boundaries."
            ),
            "algorithms": algorithm_summary(execution["graph_runtime_rows"]),
            "performance_claim": False,
        },
        "cpu_validation": {
            "report": str(VALIDATION.relative_to(ROOT)),
            "status": composition["status"],
            "checks": composition["checks"],
            "pagerank_max_absolute_error": composition["pagerank_max_absolute_error"],
            "full_leiden_partition_comparison": composition["full_leiden_partition_comparison"],
            "community_coverage_limit": composition["community_coverage_limit"],
        },
        "source_sha256": {
            key: sha256_file(path)
            for key, path in SOURCE_PATHS.items()
        },
        "records": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(capture, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {OUTPUT.relative_to(ROOT)} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
