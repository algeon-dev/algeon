#!/usr/bin/env python3
"""Verify saved MerRec segment-summary outputs against CPU references."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "target/merrec-exploration/results"
REPORT_PATH = ROOT / "docs/research/merrec/composition_validation.json"
FEATURES_PATH = ROOT / "docs/research/merrec/features.json"
VECTOR_REPORT_PATH = ROOT / "docs/research/merrec/vector_validation.json"
INPUT_PATH = RESULTS / "prepare_tastes.parquet"
KMEANS_PATH = RESULTS / "cuvs_kmeans.parquet"
HDBSCAN_PATH = RESULTS / "cuml_hdbscan.parquet"
SHARE_TOLERANCE = 5e-6
PROBABILITY_TOLERANCE = 5e-6
DISTANCE_TOLERANCE = 5e-6
EXPECTED_USERS = 1919
EXPECTED_KMEANS_CLUSTERS = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def status_for(checks: list[dict[str, Any]]) -> str:
    statuses = {check["status"] for check in checks}
    if "failed" in statuses:
        return "failed"
    if "pending" in statuses:
        return "pending"
    if statuses & {"not_comparable", "not_verifiable"}:
        return "partial"
    return "passed"


def add_check(
    checks: list[dict[str, Any]], name: str, status: str, details: dict[str, Any]
) -> None:
    checks.append({"name": name, "status": status, "details": details})


def read_vector_matrix(path: Path, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(path)
    ids = table["id"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    values = [
        table[column].fill_null(0).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        for column in columns
    ]
    matrix = np.column_stack(values).astype(np.float32, copy=False)
    order = np.argsort(ids)
    return ids[order], matrix[order]


def profiles_from_assignments(
    ids: np.ndarray,
    matrix: np.ndarray,
    assignment_ids: np.ndarray,
    labels: np.ndarray,
) -> dict[int, dict[str, Any]]:
    labels_by_id = {int(user_id): int(label) for user_id, label in zip(assignment_ids, labels)}
    aligned_labels = np.asarray([labels_by_id[int(user_id)] for user_id in ids], dtype=np.int64)
    squared = np.multiply(matrix, matrix, dtype=np.float32).astype(np.float64)
    profiles: dict[int, dict[str, Any]] = {}
    for cluster_id in sorted(set(int(value) for value in aligned_labels)):
        mask = aligned_labels == cluster_id
        means = np.mean(squared[mask], axis=0).astype(np.float32).astype(np.float64)
        profiles[cluster_id] = {"user_count": int(mask.sum()), "means": means}
    return profiles


def top_three_profiles(
    profiles: dict[int, dict[str, Any]], category_names: list[str]
) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = {}
    for cluster_id, profile in profiles.items():
        positions = sorted(
            range(len(category_names)),
            key=lambda index: (-float(profile["means"][index]), index),
        )[:3]
        result[cluster_id] = [
            {"category": category_names[index], "mean_view_share": float(profile["means"][index])}
            for index in positions
        ]
    return result


def match_top_three_profiles(
    expected: dict[int, list[dict[str, Any]]],
    actual: dict[int, list[dict[str, Any]]],
) -> tuple[bool, dict[str, Any]]:
    if len(expected) != len(actual):
        return False, {"reason": "cluster counts differ across fits"}

    def match_profile(reference: list[dict[str, Any]], observed: list[dict[str, Any]]) -> bool:
        return len(reference) == len(observed) and all(
            left["category"] == right["category"]
            and abs(left["mean_view_share"] - right["mean_view_share"]) <= SHARE_TOLERANCE
            for left, right in zip(reference, observed)
        )

    output_ids = sorted(actual)
    saved_ids = sorted(expected)
    for permutation in itertools.permutations(saved_ids):
        mapping = dict(zip(output_ids, permutation))
        if all(match_profile(expected[mapping[key]], actual[key]) for key in output_ids):
            max_error = max(
                abs(reference["mean_view_share"] - observed["mean_view_share"])
                for output_id in output_ids
                for reference, observed in zip(expected[mapping[output_id]], actual[output_id])
            )
            return True, {
                "output_cluster_to_saved_cluster": {
                    str(key): value for key, value in sorted(mapping.items())
                },
                "max_top_three_mean_absolute_error": max_error,
                "matching_saved_profile_permutation": True,
            }
    return False, {"reason": "no saved-label permutation matched all top-three category means"}


def hdbscan_reference(
    ids: np.ndarray,
    matrix: np.ndarray,
    assignment_table: Any,
    columns: list[str],
) -> dict[int, dict[str, Any]]:
    assignment_ids = assignment_table["id"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    labels = assignment_table["cluster_id"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    probabilities = assignment_table["probability"].to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    probability_by_id = {int(user_id): float(value) for user_id, value in zip(assignment_ids, probabilities)}
    labels_by_id = {int(user_id): int(label) for user_id, label in zip(assignment_ids, labels)}
    aligned_labels = np.asarray([labels_by_id[int(user_id)] for user_id in ids], dtype=np.int64)
    aligned_probabilities = np.asarray([probability_by_id[int(user_id)] for user_id in ids], dtype=np.float64)
    squared = np.multiply(matrix, matrix, dtype=np.float32).astype(np.float64)
    profiles: dict[int, dict[str, Any]] = {}
    for cluster_id in sorted(set(int(value) for value in aligned_labels)):
        mask = aligned_labels == cluster_id
        means = np.mean(squared[mask], axis=0).astype(np.float32).astype(np.float64)
        profiles[cluster_id] = {
            "user_count": int(mask.sum()),
            "mean_probability": float(np.mean(aligned_probabilities[mask])),
            "means": means,
        }
    return profiles


def output_hdbscan_summaries(
    path: Path, feature_columns: list[str]
) -> dict[int, dict[str, Any]]:
    table = pq.read_table(path)
    records = table.to_pylist()
    summaries: dict[int, dict[str, Any]] = {}
    for row in records:
        cluster_id = int(row["cluster_id"])
        summaries[cluster_id] = {
            "user_count": int(row["user_count"]),
            "mean_probability": float(row["mean_probability"]),
            "means": np.asarray(
                [row[f"{column}_mean_view_share"] for column in feature_columns],
                dtype=np.float64,
            ),
        }
    return summaries


def hdbscan_summary_match(
    expected: dict[int, dict[str, Any]],
    actual: dict[int, dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    def matches(reference: dict[str, Any], observed: dict[str, Any]) -> bool:
        return (
            reference["user_count"] == observed["user_count"]
            and abs(reference["mean_probability"] - observed["mean_probability"])
            <= PROBABILITY_TOLERANCE
            and float(np.max(np.abs(reference["means"] - observed["means"])))
            <= SHARE_TOLERANCE
        )

    if set(expected) == set(actual) and all(
        matches(expected[key], actual[key]) for key in expected
    ):
        errors = {
            "cluster_id_mapping": {str(key): key for key in sorted(expected)},
            "max_probability_absolute_error": max(
                abs(expected[key]["mean_probability"] - actual[key]["mean_probability"])
                for key in expected
            ),
            "max_share_absolute_error": max(
                float(np.max(np.abs(expected[key]["means"] - actual[key]["means"])))
                for key in expected
            ),
        }
        return True, errors

    expected_positive = sorted(key for key in expected if key >= 0)
    actual_positive = sorted(key for key in actual if key >= 0)
    if -1 not in expected or -1 not in actual or len(expected_positive) != len(actual_positive):
        return False, {"reason": "cluster labels or group counts differ"}
    if not matches(expected[-1], actual[-1]):
        return False, {"reason": "noise summary differs"}

    candidates: list[dict[int, int]] = []
    for permutation in itertools.permutations(expected_positive):
        mapping = dict(zip(actual_positive, permutation))
        if all(matches(expected[mapping[key]], actual[key]) for key in actual_positive):
            candidates.append(mapping)
            if len(candidates) > 1:
                break
    if len(candidates) == 1:
        mapping = candidates[0]
        complete_mapping = {-1: -1, **mapping}
        return True, {
            "cluster_id_mapping": {str(key): value for key, value in sorted(mapping.items())},
            "noise_cluster_id_mapping": {"-1": -1},
            "max_probability_absolute_error": max(
                abs(expected[complete_mapping[key]]["mean_probability"] - actual[key]["mean_probability"])
                for key in actual
            ),
            "max_share_absolute_error": max(
                float(np.max(np.abs(expected[complete_mapping[key]]["means"] - actual[key]["means"])))
                for key in actual
            ),
        }
    return False, {"reason": "summaries did not yield a unique aggregate cluster mapping"}


def verify_segment_themes(
    output_path: Path,
    checks: list[dict[str, Any]],
    category_names: list[str],
    saved_kmeans_sizes: dict[int, int],
    saved_kmeans_top_three: dict[int, list[dict[str, Any]]],
    input_rows: int,
) -> dict[str, Any]:
    if not output_path.is_file():
        add_check(checks, "segment_themes_output", "pending", {"missing_file": str(output_path.relative_to(ROOT))})
        add_check(
            checks,
            "segment_themes_cluster_counts",
            "pending",
            {"reason": "output is missing; the SQL output schema also omits assignments and counts"},
        )
        add_check(
            checks,
            "segment_themes_saved_kmeans_cross_fit_consistency",
            "pending",
            {"reason": "output is missing; saved KMeans profiles are available for label-permutation-invariant comparison"},
        )
        return {"status": "pending", "path": str(output_path.relative_to(ROOT))}

    table = pq.read_table(output_path)
    required = {"cluster_id", "rank", "category", "mean_view_share"}
    missing_columns = sorted(required - set(table.column_names))
    if missing_columns:
        add_check(checks, "segment_themes_schema", "failed", {"missing_columns": missing_columns})
        return {"status": "failed", "missing_columns": missing_columns}

    rows = table.to_pylist()
    clusters = sorted({int(row["cluster_id"]) for row in rows})
    by_cluster: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_cluster.setdefault(int(row["cluster_id"]), []).append(row)
    shape_ok = (
        len(rows) == EXPECTED_KMEANS_CLUSTERS * 3
        and clusters == list(range(EXPECTED_KMEANS_CLUSTERS))
        and all(sorted(int(row["rank"]) for row in group) == [1, 2, 3] for group in by_cluster.values())
    )
    order_ok = True
    shares_ok = True
    category_ok = True
    max_share = 0.0
    for cluster_id, group in by_cluster.items():
        ordered = sorted(group, key=lambda row: int(row["rank"]))
        values = [float(row["mean_view_share"]) for row in ordered]
        categories = [row["category"] for row in ordered]
        category_ok &= len(set(categories)) == len(categories) and all(
            category in category_names for category in categories
        )
        shares_ok &= all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in values)
        max_share = max(max_share, *values)
        order_ok &= all(values[index] + SHARE_TOLERANCE >= values[index + 1] for index in range(len(values) - 1))
    add_check(
        checks,
        "segment_themes_top_three_shape_and_order",
        "passed" if shape_ok and order_ok and category_ok and shares_ok else "failed",
        {
            "rows": len(rows),
            "cluster_ids": clusters,
            "expected_cluster_ids": list(range(EXPECTED_KMEANS_CLUSTERS)),
            "three_ranked_categories_per_cluster": shape_ok,
            "descending_mean_share_order": order_ok,
            "categories_from_features_contract": category_ok,
            "share_range_0_to_1": shares_ok,
            "share_tolerance": SHARE_TOLERANCE,
            "max_returned_share": max_share,
        },
    )
    add_check(
        checks,
        "segment_themes_cluster_counts",
        "not_verifiable",
        {
            "source_population_rows": input_rows,
            "saved_kmeans_counts_from_separate_fit": {
                str(key): value for key, value in sorted(saved_kmeans_sizes.items())
            },
            "reason": "21_segment_themes.sql fits KMeans again and returns only top-three category rows, without user assignments or counts; saved cuvs_kmeans labels are not a valid count reference",
        },
    )
    add_check(
        checks,
        "segment_themes_share_scope",
        "passed" if shares_ok else "failed",
        {
            "all_returned_mean_squared_shares_in_unit_interval": shares_ok,
            "source_population_rows": input_rows,
            "per_user_assignment_identity": "not asserted; cross-fit consistency is recorded separately",
        },
    )
    actual_top_three = {
        cluster_id: [
            {"category": row["category"], "mean_view_share": float(row["mean_view_share"])}
            for row in sorted(group, key=lambda item: int(item["rank"]))
        ]
        for cluster_id, group in by_cluster.items()
    }
    cross_fit_matches, cross_fit_details = match_top_three_profiles(
        saved_kmeans_top_three, actual_top_three
    )
    add_check(
        checks,
        "segment_themes_saved_kmeans_cross_fit_consistency",
        "passed" if cross_fit_matches else "not_comparable",
        {
            **cross_fit_details,
            "comparison_scope": "cross-fit consistency against saved cuvs_kmeans assignments, invariant to cluster-label permutation",
            "share_tolerance": SHARE_TOLERANCE,
            "does_not_assert_per_user_assignment_identity": True,
            "interpretation": "a match supports the captured profile table only",
        },
    )
    return {
        "status": "partial" if shape_ok and order_ok and category_ok and shares_ok else "failed",
        "rows": len(rows),
        "cluster_ids": clusters,
        "saved_kmeans_cross_fit_consistency": cross_fit_matches,
    }


def verify_dense_audiences(
    output_path: Path,
    checks: list[dict[str, Any]],
    expected: dict[int, dict[str, Any]],
    feature_columns: list[str],
    input_rows: int,
) -> dict[str, Any]:
    if not output_path.is_file():
        add_check(checks, "dense_audiences_output", "pending", {"missing_file": str(output_path.relative_to(ROOT))})
        return {"status": "pending", "path": str(output_path.relative_to(ROOT))}

    table = pq.read_table(output_path)
    required = {
        "cluster_id", "user_count", "mean_probability",
        *(f"{column}_mean_view_share" for column in feature_columns),
    }
    missing_columns = sorted(required - set(table.column_names))
    if missing_columns:
        add_check(checks, "dense_audiences_schema", "failed", {"missing_columns": missing_columns})
        return {"status": "failed", "missing_columns": missing_columns}

    output_rows = table.to_pylist()
    actual = output_hdbscan_summaries(output_path, feature_columns)
    unique_cluster_ids = len(actual) == len(output_rows)
    counts_valid = all(row["user_count"] > 0 for row in actual.values())
    population = sum(row["user_count"] for row in actual.values())
    labels_valid = all(cluster_id >= -1 for cluster_id in actual)
    probabilities_valid = all(
        np.isfinite(row["mean_probability"]) and 0.0 <= row["mean_probability"] <= 1.0
        for row in actual.values()
    )
    shares_valid = all(
        np.isfinite(value) and 0.0 <= value <= 1.0
        for row in actual.values() for value in row["means"]
    )
    add_check(
        checks,
        "dense_audiences_summary_shape",
        "passed" if unique_cluster_ids and labels_valid and counts_valid and population == input_rows and probabilities_valid and shares_valid else "failed",
        {
            "cluster_ids": sorted(actual),
            "cluster_count": len(actual),
            "unique_cluster_ids": unique_cluster_ids,
            "labels_at_least_minus_one": labels_valid,
            "user_count_sum": population,
            "expected_source_rows": input_rows,
            "probability_means_finite_in_0_to_1": probabilities_valid,
            "mean_squared_shares_finite_in_0_to_1": shares_valid,
        },
    )
    parity, parity_details = hdbscan_summary_match(expected, actual)
    add_check(
        checks,
        "dense_audiences_saved_hdbscan_aggregate_parity",
        "passed" if parity else "not_comparable",
        {
            **parity_details,
            "share_tolerance": SHARE_TOLERANCE,
            "mean_probability_tolerance": PROBABILITY_TOLERANCE,
            "scope_note": "comparison uses only aggregate rows; matching counts, probability means, and category means do not prove per-user assignment identity",
        },
    )
    return {
        "status": "passed" if parity and unique_cluster_ids and labels_valid and counts_valid and population == input_rows and probabilities_valid and shares_valid else "partial",
        "rows": len(actual),
        "user_count_sum": population,
        "aggregate_parity": parity,
    }


def expected_segment_distances(
    profiles: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    sizes = {cluster_id: profile["user_count"] for cluster_id, profile in profiles.items()}
    rows: list[dict[str, Any]] = []
    cluster_ids = sorted(profiles)
    for index, x_id in enumerate(cluster_ids):
        for y_id in cluster_ids[index + 1:]:
            delta = profiles[x_id]["means"] - profiles[y_id]["means"]
            distance = float(np.dot(delta, delta))
            rows.append({
                "cluster_id": x_id,
                "other_cluster_id": y_id,
                "distance": distance,
                "sort_distance": float(np.float32(distance)),
            })
    rows.sort(key=lambda row: (row["sort_distance"], row["cluster_id"], row["other_cluster_id"]))
    return rows, sizes


def verify_segment_distances(
    output_path: Path,
    checks: list[dict[str, Any]],
    expected_pairs: list[dict[str, Any]],
    saved_kmeans_sizes: dict[int, int],
) -> dict[str, Any]:
    if not output_path.is_file():
        add_check(checks, "segment_distances_output", "pending", {"missing_file": str(output_path.relative_to(ROOT))})
        return {"status": "pending", "path": str(output_path.relative_to(ROOT))}

    table = pq.read_table(output_path)
    required = {"cluster_id", "other_cluster_id", "distance"}
    missing_columns = sorted(required - set(table.column_names))
    if missing_columns:
        add_check(checks, "segment_distances_schema", "failed", {"missing_columns": missing_columns})
        return {"status": "failed", "missing_columns": missing_columns}

    rows = table.to_pylist()
    expected_top = expected_pairs[:10]
    actual_pairs = [(int(row["cluster_id"]), int(row["other_cluster_id"])) for row in rows]
    expected_order = [(row["cluster_id"], row["other_cluster_id"]) for row in expected_top]
    by_pair = {(row["cluster_id"], row["other_cluster_id"]): row for row in expected_pairs}
    distance_errors = []
    for row in rows:
        pair = (int(row["cluster_id"]), int(row["other_cluster_id"]))
        if pair in by_pair:
            distance_errors.append(abs(float(row["distance"]) - by_pair[pair]["distance"]))
        else:
            distance_errors.append(float("inf"))
    max_error = max(distance_errors, default=float("inf"))
    unique_pairs = len(actual_pairs) == len(set(actual_pairs))
    ids_valid = all(x_id < y_id and x_id in saved_kmeans_sizes and y_id in saved_kmeans_sizes for x_id, y_id in actual_pairs)
    ordered = all(
        float(rows[index]["distance"]) <= float(rows[index + 1]["distance"])
        for index in range(len(rows) - 1)
    )
    matched = (
        len(rows) == 10
        and actual_pairs == expected_order
        and unique_pairs
        and ids_valid
        and ordered
        and max_error <= DISTANCE_TOLERANCE
    )
    add_check(
        checks,
        "segment_distances_numpy_top_ten_and_order",
        "passed" if matched else "failed",
        {
            "rows": len(rows),
            "expected_rows": 10,
            "actual_pair_order": [list(pair) for pair in actual_pairs],
            "numpy_pair_order": [list(pair) for pair in expected_order],
            "max_distance_absolute_error": max_error,
            "distance_tolerance": DISTANCE_TOLERANCE,
            "unique_pairs": unique_pairs,
            "ids_are_ordered_saved_clusters": ids_valid,
            "returned_distances_sorted": ordered,
            "numpy_reference_pairs": len(expected_pairs),
        },
    )
    return {
        "status": "passed" if matched else "failed",
        "rows": len(rows),
        "max_distance_absolute_error": max_error,
    }


def main() -> int:
    checks: list[dict[str, Any]] = []
    output_paths = {
        "21_segment_themes": RESULTS / "21_segment_themes.parquet",
        "22_dense_audiences": RESULTS / "22_dense_audiences.parquet",
        "23_segment_distances": RESULTS / "23_segment_distances.parquet",
    }
    required_sources = [INPUT_PATH, KMEANS_PATH, HDBSCAN_PATH, FEATURES_PATH]
    missing_sources = [str(path.relative_to(ROOT)) for path in required_sources if not path.is_file()]
    if missing_sources:
        add_check(checks, "cpu_reference_inputs", "pending", {"missing_files": missing_sources})
        report = {
            "schema_version": 1,
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "validator": "scripts/dev/merrec_compositions_verify.py",
            "overall_status": "pending",
            "checks": checks,
            "pending_outputs": [str(path.relative_to(ROOT)) for path in output_paths.values()],
        }
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0

    features = json.loads(FEATURES_PATH.read_text())
    columns = list(features["columns"])
    categories = list(features["categories"])
    ids, matrix = read_vector_matrix(INPUT_PATH, columns)
    input_rows = len(ids)
    unique_ids = len(set(int(value) for value in ids)) == input_rows
    values_finite = bool(np.isfinite(matrix).all())
    add_check(
        checks,
        "source_tastes_contract",
        "passed" if input_rows == EXPECTED_USERS and len(columns) == 17 and len(categories) == 17 and unique_ids and values_finite else "failed",
        {
            "rows": input_rows,
            "expected_rows": EXPECTED_USERS,
            "feature_columns": columns,
            "feature_count": len(columns),
            "category_count": len(categories),
            "unique_ids": unique_ids,
            "finite_after_null_coalesce": values_finite,
        },
    )

    kmeans = pq.read_table(KMEANS_PATH)
    km_ids = kmeans["id"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    km_labels = kmeans["cluster_id"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    km_coverage = len(km_ids) == input_rows and set(int(value) for value in km_ids) == set(int(value) for value in ids)
    kmeans_sizes = {int(key): int(value) for key, value in zip(*np.unique(km_labels, return_counts=True))}
    kmeans_profiles = profiles_from_assignments(ids, matrix, km_ids, km_labels)
    kmeans_top_three = top_three_profiles(kmeans_profiles, categories)
    add_check(
        checks,
        "saved_kmeans_assignment_coverage",
        "passed" if km_coverage and sorted(kmeans_sizes) == list(range(EXPECTED_KMEANS_CLUSTERS)) else "failed",
        {"rows": len(km_ids), "coverage_matches_tastes": km_coverage,
         "cluster_counts": {str(key): value for key, value in sorted(kmeans_sizes.items())}},
    )

    hdbscan = pq.read_table(HDBSCAN_PATH)
    hb_ids = hdbscan["id"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    hb_coverage = len(hb_ids) == input_rows and set(int(value) for value in hb_ids) == set(int(value) for value in ids)
    hdbscan_reference_rows = hdbscan.to_pylist()
    hdbscan_sizes: dict[int, int] = {}
    for row in hdbscan_reference_rows:
        key = int(row["cluster_id"])
        hdbscan_sizes[key] = hdbscan_sizes.get(key, 0) + 1
    add_check(
        checks,
        "saved_hdbscan_assignment_coverage",
        "passed" if hb_coverage and sum(hdbscan_sizes.values()) == input_rows else "failed",
        {"rows": len(hb_ids), "coverage_matches_tastes": hb_coverage,
         "cluster_counts": {str(key): value for key, value in sorted(hdbscan_sizes.items())}},
    )

    hdbscan_profiles = hdbscan_reference(ids, matrix, hdbscan, columns)
    segment_pairs, distance_cluster_sizes = expected_segment_distances(kmeans_profiles)
    add_check(
        checks,
        "segment_distance_cpu_reference_ready",
        "passed" if len(segment_pairs) == 28 and distance_cluster_sizes == kmeans_sizes else "failed",
        {
            "saved_kmeans_cluster_counts": {str(key): value for key, value in sorted(distance_cluster_sizes.items())},
            "profile_pairs": len(segment_pairs),
            "metric": "squared L2 over the 17 mean squared Float32 category shares, computed by NumPy Float64 and rounded to Float32 for SQL ordering",
        "expected_top_10": [
                {"cluster_id": row["cluster_id"], "other_cluster_id": row["other_cluster_id"], "distance": row["distance"]}
                for row in segment_pairs[:10]
            ],
        },
    )

    add_check(
        checks,
        "saved_kmeans_top_three_cross_fit_reference",
        "passed" if len(kmeans_top_three) == EXPECTED_KMEANS_CLUSTERS else "failed",
        {
            "comparison_scope": "independent profiles from existing cuvs_kmeans assignments, used only for cross-fit consistency with query 21",
            "does_not_assert_per_user_assignment_identity": True,
            "profiles": [
                {
                    "cluster_id": cluster_id,
                    "top_three": kmeans_top_three[cluster_id],
                }
                for cluster_id in sorted(kmeans_top_three)
            ],
        },
    )

    outcomes = {
        "21_segment_themes": verify_segment_themes(
            output_paths["21_segment_themes"],
            checks,
            categories,
            kmeans_sizes,
            kmeans_top_three,
            input_rows,
        ),
        "22_dense_audiences": verify_dense_audiences(
            output_paths["22_dense_audiences"], checks, hdbscan_profiles, columns, input_rows
        ),
        "23_segment_distances": verify_segment_distances(
            output_paths["23_segment_distances"], checks, segment_pairs, kmeans_sizes
        ),
    }
    sql_paths = [
        ROOT / "docs/research/merrec/sql/21_segment_themes.sql",
        ROOT / "docs/research/merrec/sql/22_dense_audiences.sql",
        ROOT / "docs/research/merrec/sql/23_segment_distances.sql",
    ]
    hashes = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in [INPUT_PATH, KMEANS_PATH, HDBSCAN_PATH, FEATURES_PATH, *sql_paths, VECTOR_REPORT_PATH]
        if path.is_file()
    }
    missing_outputs = [
        str(path.relative_to(ROOT)) for path in output_paths.values() if not path.is_file()
    ]
    report = {
        "schema_version": 1,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "validator": "scripts/dev/merrec_compositions_verify.py",
        "validator_sha256": sha256_file(Path(__file__).resolve()),
        "overall_status": status_for(checks),
        "input_contract": {
            "rows": input_rows,
            "feature_columns": columns,
            "feature_count": len(columns),
            "cutoff_exclusive": features.get("cutoff_exclusive"),
        },
        "sql_hashes": {
            str(path.relative_to(ROOT)): hashes.get(str(path.relative_to(ROOT))) for path in sql_paths
        },
        "source_hashes": {
            key: value for key, value in hashes.items() if key not in {str(path.relative_to(ROOT)) for path in sql_paths}
        },
        "outputs": outcomes,
        "pending_outputs": missing_outputs,
        "tolerances": {
            "mean_squared_share_absolute": SHARE_TOLERANCE,
            "mean_probability_absolute": PROBABILITY_TOLERANCE,
            "pair_distance_absolute": DISTANCE_TOLERANCE,
        },
        "checks": checks,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 1 if report["overall_status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
