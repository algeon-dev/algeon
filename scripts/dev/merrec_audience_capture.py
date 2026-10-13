#!/usr/bin/env python3
"""Write the fixed MerRec audience map capture from saved Parquet results."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pyarrow.parquet as pq

from merrec_verify_common import MERREC_LICENSE


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "target/merrec-exploration/results"
RESEARCH = ROOT / "docs/research/merrec"
FEATURES_PATH = RESEARCH / "features.json"
VALIDATION_PATH = RESEARCH / "vector_validation.json"
OUTPUT = ROOT / "website/static/data/merrec-audience-capture.json"
NAMES = ("prepare_tastes", "cuml_umap", "cuml_hdbscan")
EXPECTED_ROWS = 1919
EXPECTED_CLUSTERS = {-1: 1316, 0: 65, 1: 230, 2: 68, 3: 192, 4: 48}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def six_significant_digits(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"capture has a non-finite number: {value}")
    return float(format(value, ".6g"))


def rows_by_id(path: Path) -> dict[int, dict[str, object]]:
    rows = pq.read_table(path).to_pylist()
    result = {int(row["id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"{path} contains duplicate ids")
    return result


def main() -> None:
    feature_spec = json.loads(FEATURES_PATH.read_text())
    categories = feature_spec["categories"]
    columns = feature_spec["columns"]
    if len(categories) != 17 or len(columns) != 17:
        raise ValueError("features.json must define the ordered 17-category vector")

    taste_path = RESULTS / "prepare_tastes.parquet"
    umap_path = RESULTS / "cuml_umap.parquet"
    hdbscan_path = RESULTS / "cuml_hdbscan.parquet"
    source_rows = rows_by_id(taste_path)
    umap_rows = rows_by_id(umap_path)
    cluster_rows = rows_by_id(hdbscan_path)
    expected_ids = set(source_rows)
    if len(expected_ids) != EXPECTED_ROWS or set(umap_rows) != expected_ids or set(cluster_rows) != expected_ids:
        raise ValueError("taste, UMAP, and HDBSCAN results must cover the same 1,919 IDs")

    cluster_sizes: dict[int, int] = {}
    for row in cluster_rows.values():
        cluster_id = int(row["cluster_id"])
        cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1
    if cluster_sizes != EXPECTED_CLUSTERS:
        raise ValueError(f"unexpected HDBSCAN cluster sizes: {cluster_sizes}")

    validation = json.loads(VALIDATION_PATH.read_text())
    if not validation["overall_pass"]:
        raise ValueError("vector_validation.json contains failed checks")
    trust = validation["results"]["cuml_umap"]["trustworthiness"]
    if trust["input_dimensions"] != 17 or trust["n_neighbors"] != 15:
        raise ValueError(f"unexpected UMAP trustworthiness definition: {trust}")
    cluster_validation = validation["results"]["cuml_hdbscan"]["clusters_including_noise"]
    validation_by_cluster = {
        int(cluster["cluster_id"]): cluster for cluster in cluster_validation
    }

    points = []
    for user_id in sorted(expected_ids):
        taste = source_rows[user_id]
        umap = umap_rows[user_id]
        cluster = cluster_rows[user_id]
        shares = [
            six_significant_digits(float(taste[column]) ** 2)
            for column in columns
        ]
        points.append(
            {
                "id": user_id,
                "x": six_significant_digits(float(umap["dim_0"])),
                "y": six_significant_digits(float(umap["dim_1"])),
                "clusterId": int(cluster["cluster_id"]),
                "probability": six_significant_digits(float(cluster["probability"])),
                "views": int(taste["views"]),
                "shares": shares,
            }
        )

    cluster_names = {
        0: "Electronics",
        1: "Toys & Collectibles",
        2: "Kids",
        3: "Women",
        4: "Men",
    }
    clusters = []
    for cluster_id in sorted(cluster_sizes):
        row = validation_by_cluster[cluster_id]
        shares_by_column = row["mean_squared_feature_shares"]
        dominant_column = max(columns, key=lambda column: shares_by_column[column])
        clusters.append(
            {
                "clusterId": cluster_id,
                "label": "Noise" if cluster_id == -1 else f"Cluster {cluster_id}",
                "size": cluster_sizes[cluster_id],
                "theme": None if cluster_id == -1 else cluster_names[cluster_id],
                "dominantCategory": dominant_column,
                "meanSquaredFeatureShare": six_significant_digits(
                    float(shares_by_column[dominant_column])
                ),
                "meanSquaredFeatureShares": [
                    six_significant_digits(float(shares_by_column[column]))
                    for column in columns
                ],
            }
        )

    query_records = {
        name: json.loads((RESEARCH / "results" / f"{name}.json").read_text())
        for name in NAMES
    }
    for name, result in query_records.items():
        if result["status"] != "ok" or int(result["rows"]) != EXPECTED_ROWS:
            raise ValueError(f"{name} result record is not a complete successful capture")
    query_provenance = {
        name: {
            "queryId": query_records[name]["query_id"],
            "resultJson": f"docs/research/merrec/results/{name}.json",
            "sql": query_records[name]["sql"],
        }
        for name in NAMES
    }

    input_paths = [
        FEATURES_PATH,
        VALIDATION_PATH,
        taste_path,
        umap_path,
        hdbscan_path,
        *(RESEARCH / "results" / f"{name}.json" for name in NAMES),
        *(RESEARCH / "sql" / "probes" / f"{name}.sql" for name in ("cuml_umap", "cuml_hdbscan")),
    ]
    capture = {
        "license": MERREC_LICENSE,
        "source": "MerRec captured audience and category-share results",
        "rows": EXPECTED_ROWS,
        "joinKey": "id",
        "categories": [
            {"column": column, "name": category}
            for column, category in zip(columns, categories, strict=True)
        ],
        "featureTransform": feature_spec["transform"],
        "featureShareFormula": "per-user squared Float32 sqrt category share, rounded to 6 significant digits",
        "queries": query_provenance,
        "trustworthiness": {
            "value": six_significant_digits(float(trust["value"])),
            "nNeighbors": trust["n_neighbors"],
            "inputDimensions": trust["input_dimensions"],
            "interpretation": "descriptive UMAP neighborhood preservation against the original 17-dimensional inputs",
        },
        "noiseRows": cluster_sizes[-1],
        "assignedRows": EXPECTED_ROWS - cluster_sizes[-1],
        "clusterCount": len(EXPECTED_CLUSTERS) - 1,
        "clusters": clusters,
        "sourceSha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [*input_paths, Path(__file__).resolve()]
        },
        "points": points,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(capture, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(OUTPUT.relative_to(ROOT)),
                "rows": len(points),
                "noise_rows": cluster_sizes[-1],
                "assigned_rows": EXPECTED_ROWS - cluster_sizes[-1],
                "cluster_sizes": {str(key): value for key, value in sorted(cluster_sizes.items())},
                "trustworthiness": capture["trustworthiness"]["value"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
