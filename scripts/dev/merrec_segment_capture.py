#!/usr/bin/env python3
"""Regenerate the fixed MerRec KMeans and Top-K result capture."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "docs/research/merrec"
RESULTS = ROOT / "target/merrec-exploration/results"
SQL_PATH = RESEARCH / "sql/21_segment_themes.sql"
RESULT_PATH = RESEARCH / "results/21_segment_themes.json"
PARQUET_PATH = RESULTS / "21_segment_themes.parquet"
STATS_PATH = RESEARCH / "results/21_segment_themes.stats.txt.gz"
FEATURES_PATH = RESEARCH / "features.json"
COMPOSITION_VALIDATION_PATH = RESEARCH / "composition_validation.json"
TASTES_PATH = RESULTS / "prepare_tastes.parquet"
KMEANS_PATH = RESULTS / "cuvs_kmeans.parquet"
OUTPUT_PATH = ROOT / "website/static/data/merrec-segment-capture.json"
EXPECTED_QUERY_ID = "2cbb5a71-fef2-41fd-8b33-6b030a101ee9"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_stats() -> tuple[dict[str, str], dict[str, str]]:
    with gzip.open(STATS_PATH, "rt", encoding="utf-8") as source:
        lines = source.readlines()
    header_values = {}
    for line in lines:
        if not line.startswith("# "):
            break
        key, value = line[2:].rstrip("\n").split("=", 1)
        header_values[key] = value
    planning_index = lines.index("[planning]\n")
    columns = lines[planning_index + 1].rstrip("\n").split("\t")
    values = lines[planning_index + 2].rstrip("\n").split("\t")
    return header_values, dict(zip(columns, values, strict=True))


def main() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    sql = SQL_PATH.read_text(encoding="utf-8")
    if sql != result["sql"]:
        raise ValueError("source SQL differs from the exact SQL saved in the result JSON")
    features = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    composition_validation = json.loads(
        COMPOSITION_VALIDATION_PATH.read_text(encoding="utf-8")
    )
    table = pq.read_table(PARQUET_PATH)
    rows = table.to_pylist()
    if result["query_id"] != EXPECTED_QUERY_ID or result["status"] != "ok":
        raise ValueError("query 21 result metadata does not match the captured query")
    if len(rows) != 24 or result["rows"] != len(rows):
        raise ValueError(f"expected 24 result rows, got {len(rows)}")
    if len(features["columns"]) != 17 or len(features["categories"]) != 17:
        raise ValueError("features.json must define 17 ordered category dimensions")
    tastes_rows = pq.ParquetFile(TASTES_PATH).metadata.num_rows
    if tastes_rows != 1919:
        raise ValueError(f"expected 1,919 prepared users, got {tastes_rows}")

    expected_keys = [(cluster_id, rank) for cluster_id in range(8) for rank in range(1, 4)]
    actual_keys = [(int(row["cluster_id"]), int(row["rank"])) for row in rows]
    if actual_keys != expected_keys:
        raise ValueError("query 21 output must contain ordered ranks 1-3 for clusters 0-7")

    cross_fit = next(
        check
        for check in composition_validation["checks"]
        if check["name"] == "segment_themes_saved_kmeans_cross_fit_consistency"
    )
    if cross_fit["status"] != "passed":
        raise ValueError("saved-profile cross-fit consistency check did not pass")

    stats_header, planning = read_stats()
    if stats_header.get("query_id") != EXPECTED_QUERY_ID:
        raise ValueError("query statistics belong to a different query ID")

    source_paths = (
        SQL_PATH,
        RESULT_PATH,
        PARQUET_PATH,
        STATS_PATH,
        FEATURES_PATH,
        COMPOSITION_VALIDATION_PATH,
        TASTES_PATH,
        KMEANS_PATH,
    )
    capture = {
        "query": "21_segment_themes",
        "queryId": EXPECTED_QUERY_ID,
        "sql": sql,
        "schema": {
            "text": str(table.schema),
            "fields": [
                {"name": field.name, "type": str(field.type), "nullable": field.nullable}
                for field in table.schema
            ],
        },
        "population": {
            "users": tastes_rows,
            "minimumItemViewEvents": 20,
            "eventType": "item_view",
            "cutoffExclusive": features["cutoff_exclusive"],
            "featureTransform": features["transform"],
            "featureColumns": features["columns"],
            "categories": features["categories"],
        },
        "method": {
            "algorithm": "cuvs_kmeans",
            "clusters": 8,
            "nInit": 5,
            "profileAggregation": "SQL squares each stored square-root share, then averages across users in each query-local cluster",
            "selection": "cuvs_top_k with k=3 and select=max",
            "shareMeaning": "Each mean_view_share is an equal-user mean of that user's share of item_view events in the category. It is neither a share of all cluster events nor a proportion of users.",
        },
        "execution": {
            "outcome": stats_header.get("outcome"),
            "disposition": stats_header.get("attempt_disposition"),
            "serverReportedElapsedMs": int(stats_header["elapsed_ms"]),
            "serverReportedElapsedBoundary": "Server query statistics field elapsed_ms.",
            "clientFlightElapsedSeconds": result["elapsed_s"],
            "clientFlightElapsedBoundary": "Client Flight elapsed_s includes planning and receipt of the full result; preparation is excluded.",
            "timingContext": "Both values are single observations from a shared server. Concurrent server activity was not controlled; neither value is an isolated benchmark.",
            "nativePlanFragments": int(planning["native_plan_fragment_count"]),
            "hostDataFusionBoundaries": int(planning["host_datafusion_boundary_count"]),
            "postNativeExecutionPath": planning["post_native_execution_path"],
        },
        "crossFitConsistency": {
            "status": cross_fit["status"],
            "outputClusterToSavedCluster": cross_fit["details"]["output_cluster_to_saved_cluster"],
            "maxTopThreeMeanAbsoluteError": cross_fit["details"]["max_top_three_mean_absolute_error"],
            "comparisonScope": cross_fit["details"]["comparison_scope"],
            "doesNotAssertPerUserAssignmentIdentity": cross_fit["details"]["does_not_assert_per_user_assignment_identity"],
            "interpretation": cross_fit["details"]["interpretation"],
        },
        "rows": rows,
        "sourceSha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in source_paths
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(capture, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
