#!/usr/bin/env python3
"""Validate EMBER2024 challenge projections and clustering outputs on CPU."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.manifold import trustworthiness
from sklearn.metrics import adjusted_rand_score


ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "target/ember2024-research"
REPORT = ROOT / "docs/research/ember2024/results/geometry_check.json"
GRAM_TOLERANCE = 1e-4
VARIANTS = ("", "_r2", "_r3")
RESULTS = {
    "kmeans": "40_challenge_kmeans_shared",
    "pca": "41_challenge_pca_shared",
    "tsvd": "42_challenge_tsvd_shared",
    "umap": "43_challenge_umap_shared",
    "hdbscan": "44_challenge_hdbscan_shared",
    "pca_hdbscan": "45_challenge_pca_hdbscan_shared",
}
COORDINATES = {"pca": ("pc_0", "pc_1"), "tsvd": ("pc_0", "pc_1"), "umap": ("dim_0", "dim_1")}
CLUSTERING = {"kmeans": "kmeans", "hdbscan": "hdbscan", "pca_hdbscan": "hdbscan"}


def read_win32_histograms() -> tuple[np.ndarray, np.ndarray, list[str]]:
    metadata = pq.read_table(DATA / "input_metadata_challenge.parquet", columns=["id", "file_type", "family"])
    metadata = metadata.filter(pc.equal(metadata["file_type"], "Win32")).sort_by([("id", "ascending")])
    ids = metadata["id"].to_numpy()
    families = metadata["family"].to_pylist()

    vectors = pq.read_table(DATA / "input_research_vectors_challenge.parquet", columns=["id", "histogram"])
    vectors = vectors.sort_by([("id", "ascending")])
    vector_ids = vectors["id"].to_numpy()
    positions = np.searchsorted(vector_ids, ids)
    if np.any(positions >= len(vector_ids)) or not np.array_equal(vector_ids[positions], ids):
        raise ValueError("Win32 metadata ids do not all have matching challenge histograms")
    histogram = vectors["histogram"].combine_chunks()
    dimension = histogram.type.list_size
    all_values = histogram.values.to_numpy().reshape(vectors.num_rows, dimension)
    source = np.asarray(all_values[positions], dtype=np.float64)
    if source.ndim != 2 or source.shape[0] != len(ids) or not np.isfinite(source).all():
        raise ValueError("Win32 challenge histogram matrix has invalid shape or non-finite values")
    return ids, source, families


def load_result(stem: str, expected_ids: np.ndarray, violations: list[str]) -> dict | None:
    path = DATA / f"{stem}.parquet"
    if not path.is_file():
        return None
    table = pq.read_table(path)
    if "id" not in table.column_names:
        violations.append(f"{stem}: missing id column")
        return {"table": table, "ids_match": False, "path": path}

    ids = table["id"].to_numpy()
    ids_match = np.array_equal(ids, expected_ids)
    if not ids_match:
        violations.append(f"{stem}: output ids are not sorted and equal to source ids")
    for field, column in zip(table.schema, table.columns, strict=True):
        if pa.types.is_floating(field.type):
            values = column.to_numpy()
            if not np.isfinite(values).all():
                violations.append(f"{stem}: {field.name} contains non-finite values")
    return {"table": table, "ids_match": ids_match, "path": path}


def column_values(result: dict, name: str) -> np.ndarray:
    return result["table"][name].to_numpy()


def coordinates(result: dict, columns: tuple[str, str]) -> np.ndarray:
    return np.column_stack([column_values(result, column) for column in columns]).astype(np.float64)


def relative_gram_error(reference: np.ndarray, observed: np.ndarray) -> float:
    numerator = 0.0
    denominator = 0.0
    for start in range(0, reference.shape[0], 256):
        stop = min(start + 256, reference.shape[0])
        reference_gram = reference[start:stop] @ reference.T
        observed_gram = observed[start:stop] @ observed.T
        difference = reference_gram - observed_gram
        numerator += float(np.sum(difference * difference, dtype=np.float64))
        denominator += float(np.sum(reference_gram * reference_gram, dtype=np.float64))
    if denominator == 0:
        return 0.0 if numerator == 0 else float("inf")
    return float(np.sqrt(numerator / denominator))


def cluster_summary(table: pa.Table) -> list[dict]:
    cluster_ids = table["cluster_id"].to_numpy()
    families = table["family"].to_pylist()
    counts: dict[int, Counter] = {}
    sizes = Counter(int(cluster_id) for cluster_id in cluster_ids)
    for cluster_id, family in zip(cluster_ids, families, strict=True):
        if family:
            counts.setdefault(int(cluster_id), Counter())[family] += 1
    summary = []
    for cluster_id in sorted(sizes):
        family_counts = counts.get(cluster_id, Counter())
        most_common = family_counts.most_common(1)
        summary.append(
            {
                "cluster_id": cluster_id,
                "size": sizes[cluster_id],
                "known_family_count": sum(family_counts.values()),
                "most_common_family": most_common[0][0] if most_common else None,
                "most_common_family_count": most_common[0][1] if most_common else 0,
            }
        )
    return summary


def main() -> int:
    expected_ids, source, _ = read_win32_histograms()
    violations: list[str] = []
    source_centered = source - source.mean(axis=0, keepdims=True)
    centered_covariance = source_centered.T @ source_centered
    centered_values, centered_vectors = np.linalg.eigh(centered_covariance)
    centered_order = np.argsort(centered_values)[::-1]
    centered_values = np.maximum(centered_values[centered_order], 0.0)
    centered_vectors = centered_vectors[:, centered_order]
    pca_projection = source_centered @ centered_vectors[:, :2]

    uncentered_covariance = source.T @ source
    uncentered_values, uncentered_vectors = np.linalg.eigh(uncentered_covariance)
    uncentered_order = np.argsort(uncentered_values)[::-1]
    uncentered_values = np.maximum(uncentered_values[uncentered_order], 0.0)
    uncentered_vectors = uncentered_vectors[:, uncentered_order]
    tsvd_projection = source @ uncentered_vectors[:, :2]

    projection_metrics = {
        "centered_pca_variance_ratio_2": float(centered_values[:2].sum() / centered_values.sum()),
        "centered_pca_variance_ratio_16": float(centered_values[:16].sum() / centered_values.sum()),
        "uncentered_tsvd_energy_ratio_2": float(uncentered_values[:2].sum() / uncentered_values.sum()),
    }

    variants: dict[str, dict[str, dict | None]] = {}
    for key, stem in RESULTS.items():
        variants[key] = {}
        for suffix in VARIANTS:
            variant_name = "base" if not suffix else suffix.removeprefix("_")
            result = load_result(f"{stem}{suffix}", expected_ids, violations)
            if not suffix and result is None:
                raise FileNotFoundError(DATA / f"{stem}.parquet")
            variants[key][variant_name] = result

    clustering_metrics: dict[str, dict] = {}
    for key, family in CLUSTERING.items():
        clustering_metrics[key] = {}
        for variant_name, result in variants[key].items():
            if result is None:
                continue
            table = result["table"]
            cluster_ids = column_values(result, "cluster_id")
            if family == "kmeans":
                if not np.all((cluster_ids >= 0) & (cluster_ids < 12)):
                    violations.append(f"{RESULTS[key]}_{variant_name}: KMeans cluster_id is outside [0, 12)")
            else:
                if not np.all(cluster_ids >= -1):
                    violations.append(f"{RESULTS[key]}_{variant_name}: HDBSCAN cluster_id is below -1")
                probabilities = column_values(result, "probability")
                if not np.all((probabilities >= 0) & (probabilities <= 1)):
                    violations.append(f"{RESULTS[key]}_{variant_name}: HDBSCAN probability is outside [0, 1]")
            clustering_metrics[key][variant_name] = cluster_summary(table)

    pair_metrics: dict[str, dict] = {key: {} for key in (*CLUSTERING.keys(), *COORDINATES.keys())}
    for key in pair_metrics:
        available = [(name, result) for name, result in variants[key].items() if result is not None]
        for (left_name, left), (right_name, right) in combinations(available, 2):
            pair_name = f"{left_name}_vs_{right_name}"
            if not left["ids_match"] or not right["ids_match"]:
                continue
            if key in CLUSTERING:
                pair_metrics[key][pair_name] = {
                    "adjusted_rand_index": float(
                        adjusted_rand_score(column_values(left, "cluster_id"), column_values(right, "cluster_id"))
                    )
                }
            else:
                left_coordinates = coordinates(left, COORDINATES[key])
                right_coordinates = coordinates(right, COORDINATES[key])
                pair_metrics[key][pair_name] = {
                    "max_absolute_coordinate_difference": float(
                        np.max(np.abs(left_coordinates - right_coordinates))
                    )
                }

    gram_metrics = {}
    cpu_projections = {"pca": pca_projection, "tsvd": tsvd_projection}
    for key, reference in cpu_projections.items():
        gram_metrics[key] = {}
        for variant_name, result in variants[key].items():
            if result is None or not result["ids_match"]:
                continue
            observed = coordinates(result, COORDINATES[key])
            relative_error = relative_gram_error(reference, observed)
            passed = relative_error <= GRAM_TOLERANCE
            gram_metrics[key][variant_name] = {
                "relative_frobenius_error": relative_error,
                "tolerance": GRAM_TOLERANCE,
                "passed": passed,
            }
            if not passed:
                violations.append(f"{RESULTS[key]}_{variant_name}: projection Gram relative error exceeds {GRAM_TOLERANCE}")

    umap_metrics = {}
    for variant_name, result in variants["umap"].items():
        if result is None or not result["ids_match"]:
            continue
        embedding = coordinates(result, COORDINATES["umap"])
        umap_metrics[variant_name] = float(trustworthiness(source, embedding, n_neighbors=15))

    report = {
        "source": {
            "split": "challenge",
            "file_type": "Win32",
            "rows": int(len(expected_ids)),
            "id_min": int(expected_ids.min()),
            "id_max": int(expected_ids.max()),
            "histogram_dimension": int(source.shape[1]),
        },
        "projection_metrics": projection_metrics,
        "projection_gram_comparison": gram_metrics,
        "umap_trustworthiness_n_neighbors_15": umap_metrics,
        "cluster_sizes_and_known_family_counts": clustering_metrics,
        "sibling_comparisons": pair_metrics,
        "violations": violations,
        "status": "failed" if violations else "passed",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "violations": violations}, sort_keys=True))
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
