#!/usr/bin/env python3
"""Validate MerRec vector SQL snapshots with independent CPU calculations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import sklearn
from sklearn.cluster import KMeans
from sklearn.manifold import trustworthiness
from sklearn.metrics import adjusted_rand_score


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "target/merrec-exploration/results"
DEFAULT_FEATURES = ROOT / "docs/research/merrec/features.json"
DEFAULT_OUTPUT = ROOT / "docs/research/merrec/vector_validation.json"
SEED_ID = 65484
K = 10
TOP_K = 3
EXACT_DISTANCE_TOLERANCE = 5e-7
SHARE_TOLERANCE = 5e-7
DISTANCE_TIE_TOLERANCE = 1e-12
MIN_ANN_RECALL_AT_10 = 0.9
MAX_PROJECTION_GRAM_RELATIVE_ERROR = 5e-4
MAX_SUBSPACE_ANGLE_DEGREES = 0.1
ROW_ORDINAL_COLUMN = "row_ordinal"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Validation:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.results: dict[str, Any] = {}

    def check(self, name: str, passed: bool, details: dict[str, Any]) -> None:
        self.checks.append({"name": name, "passed": bool(passed), "details": details})

    @property
    def passed(self) -> bool:
        return all(check["passed"] for check in self.checks)


def to_matrix(table: Any, feature_columns: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    null_counts: dict[str, int] = {}
    vectors = []
    for column in feature_columns:
        values = table[column]
        null_counts[column] = values.null_count
        vectors.append(values.fill_null(0).to_numpy(zero_copy_only=False))
    return np.column_stack(vectors).astype(np.float32, copy=False), null_counts


def table_records(path: Path) -> tuple[Any, list[dict[str, Any]]]:
    table = pq.read_table(path)
    return table, table.to_pylist()


def id_coverage(
    records: list[dict[str, Any]], expected_ids: np.ndarray, *, name: str
) -> dict[str, Any]:
    ids = [int(row["id"]) for row in records]
    unique_ids = set(ids)
    expected = set(int(value) for value in expected_ids)
    return {
        "name": name,
        "rows": len(records),
        "duplicate_ids": len(ids) - len(unique_ids),
        "missing_ids": len(expected - unique_ids),
        "unexpected_ids": len(unique_ids - expected),
        "row_ids_match": len(records) == len(expected) and unique_ids == expected,
    }


def check_id_coverage(
    validation: Validation,
    records: list[dict[str, Any]],
    expected_ids: np.ndarray,
    *,
    name: str,
) -> dict[str, Any]:
    coverage = id_coverage(records, expected_ids, name=name)
    validation.check(
        f"{name}_row_coverage",
        coverage["row_ids_match"] and coverage["duplicate_ids"] == 0,
        {key: value for key, value in coverage.items() if key != "name"},
    )
    return coverage


def ordinal_check(
    validation: Validation,
    records: list[dict[str, Any]],
    sorted_ids: np.ndarray,
    *,
    name: str,
) -> None:
    if not records or ROW_ORDINAL_COLUMN not in records[0]:
        return
    id_to_ordinal = {int(user_id): index for index, user_id in enumerate(sorted_ids)}
    mismatches = sum(
        int(row[ROW_ORDINAL_COLUMN]) != id_to_ordinal.get(int(row["id"]), -1)
        for row in records
    )
    validation.check(
        f"{name}_row_ordinal",
        mismatches == 0,
        {"mismatched_ordinals": mismatches, "expected_order": "ascending id"},
    )


def nearest_recall(
    returned_ids: list[int],
    distances_by_id: dict[int, float],
    ordered_ids: np.ndarray,
    ordered_distances: np.ndarray,
    *,
    k: int,
) -> dict[str, Any]:
    cutoff = float(ordered_distances[k - 1])
    mandatory = {
        int(ordered_ids[index])
        for index, distance in enumerate(ordered_distances)
        if float(distance) < cutoff - DISTANCE_TIE_TOLERANCE
    }
    boundary = {
        int(ordered_ids[index])
        for index, distance in enumerate(ordered_distances)
        if abs(float(distance) - cutoff) <= DISTANCE_TIE_TOLERANCE
    }
    returned = set(returned_ids)
    mandatory_hits = len(returned & mandatory)
    boundary_hits = len(returned & boundary)
    boundary_slots = max(0, k - len(mandatory))
    tie_aware_hits = mandatory_hits + min(boundary_slots, boundary_hits)
    valid_returned = [
        user_id
        for user_id in returned_ids
        if user_id in distances_by_id
        and distances_by_id[user_id] <= cutoff + DISTANCE_TIE_TOLERANCE
    ]
    return {
        "k": k,
        "cutoff_squared_l2": cutoff,
        "mandatory_neighbors_before_tied_boundary": len(mandatory),
        "neighbors_tied_at_boundary": len(boundary),
        "tie_aware_hits": tie_aware_hits,
        "recall_at_10_tie_aware": tie_aware_hits / k,
        "returned_within_exact_cutoff": len(valid_returned),
    }


def compare_neighbor_metadata(
    records: list[dict[str, Any]],
    source_by_id: dict[int, dict[str, Any]],
    vector_by_id: dict[int, np.ndarray],
) -> dict[str, Any]:
    metadata_errors: dict[str, float] = {
        "toy_share": 0.0,
        "electronics_share": 0.0,
        "mean_viewed_price": 0.0,
    }
    metadata_mismatches = 0
    views_mismatches = 0
    for record in records:
        source = source_by_id[int(record["neighbor_id"])]
        vector = vector_by_id[int(record["neighbor_id"])]
        views_mismatches += int(int(record["views"]) != int(source["views"]))
        for column, feature_index in (("toy_share", 1), ("electronics_share", 5)):
            expected = float(vector[feature_index] * vector[feature_index])
            error = abs(float(record[column]) - expected)
            metadata_errors[column] = max(metadata_errors[column], error)
            metadata_mismatches += int(error > SHARE_TOLERANCE)
        expected_price = source["mean_viewed_price"]
        actual_price = record["mean_viewed_price"]
        if expected_price is None or actual_price is None:
            price_matches = expected_price is actual_price
            error = 0.0 if price_matches else None
        else:
            error = abs(float(actual_price) - float(expected_price))
            price_matches = error <= 1e-12
        if error is not None:
            metadata_errors["mean_viewed_price"] = max(
                metadata_errors["mean_viewed_price"], error
            )
        metadata_mismatches += int(not price_matches)
    return {
        "views_mismatches": views_mismatches,
        "projected_value_mismatches": metadata_mismatches,
        "max_absolute_error_by_column": metadata_errors,
        "tolerances": {
            "toy_share": SHARE_TOLERANCE,
            "electronics_share": SHARE_TOLERANCE,
            "mean_viewed_price": 1e-12,
            "views": "exact integer equality",
        },
    }


def validate_knn(
    validation: Validation,
    method: str,
    records: list[dict[str, Any]],
    ids: np.ndarray,
    exact_ids: np.ndarray,
    exact_distances: np.ndarray,
    exact_distances_by_id: dict[int, float],
    source_by_id: dict[int, dict[str, Any]],
    vector_by_id: dict[int, np.ndarray],
) -> dict[str, Any]:
    rows = sorted(records, key=lambda row: int(row["rank"]))
    ranks = [int(row["rank"]) for row in rows]
    neighbor_ids = [int(row["neighbor_id"]) for row in rows]
    distances_match = []
    output_distance_errors: list[float] = []
    for row in rows:
        user_id = int(row["neighbor_id"])
        error = abs(float(row["distance"]) - exact_distances_by_id[user_id])
        output_distance_errors.append(error)
        distances_match.append(error <= EXACT_DISTANCE_TOLERANCE)
    recall = nearest_recall(
        neighbor_ids,
        exact_distances_by_id,
        exact_ids,
        exact_distances,
        k=K,
    )
    unique_ids = len(set(neighbor_ids)) == len(neighbor_ids)
    correct_rows = (
        len(rows) == K
        and ranks == list(range(1, K + 1))
        and unique_ids
        and SEED_ID not in neighbor_ids
    )
    metadata = compare_neighbor_metadata(rows, source_by_id, vector_by_id)
    if method == "cuvs_brute_force_knn":
        neighbors_pass = recall["recall_at_10_tie_aware"] == 1.0
        distances_pass = all(distances_match)
        validation.check(
            "brute_force_knn_matches_numpy_float64_squared_l2",
            correct_rows and neighbors_pass and distances_pass,
            {
                "query_id": SEED_ID,
                "rows": len(rows),
                "rank_sequence_matches_1_to_10": ranks == list(range(1, K + 1)),
                "unique_neighbor_ids": unique_ids,
                "returned_seed_id": SEED_ID in neighbor_ids,
                **recall,
                "max_distance_absolute_error": max(output_distance_errors, default=0.0),
                "distance_tolerance": EXACT_DISTANCE_TOLERANCE,
                "expected_neighbor_ids": [int(value) for value in exact_ids[:K]],
                "actual_neighbor_ids": neighbor_ids,
            },
        )
    else:
        validation.check(
            f"{method}_tie_aware_recall_at_10",
            correct_rows and recall["recall_at_10_tie_aware"] >= MIN_ANN_RECALL_AT_10,
            {
                "query_id": SEED_ID,
                "rows": len(rows),
                "rank_sequence_matches_1_to_10": ranks == list(range(1, K + 1)),
                "unique_neighbor_ids": unique_ids,
                **recall,
                "minimum_recall_at_10": MIN_ANN_RECALL_AT_10,
                "exact_neighbor_ids": [int(value) for value in exact_ids[:K]],
                "actual_neighbor_ids": neighbor_ids,
                "max_returned_distance_error_vs_exact_id": max(
                    output_distance_errors, default=0.0
                ),
            },
        )
    validation.check(
        f"{method}_neighbor_metadata",
        metadata["views_mismatches"] == 0
        and metadata["projected_value_mismatches"] == 0,
        metadata,
    )
    return {
        "query_id": SEED_ID,
        "ranked_neighbor_ids": neighbor_ids,
        "ranked_squared_l2_distances": [float(row["distance"]) for row in rows],
        **recall,
        "max_distance_absolute_error_vs_numpy": max(output_distance_errors, default=0.0),
        "neighbor_metadata": metadata,
    }


def validate_top_k(
    validation: Validation,
    records: list[dict[str, Any]],
    vector_by_id: dict[int, np.ndarray],
    feature_columns: list[str],
) -> dict[str, Any]:
    by_id: dict[int, list[dict[str, Any]]] = {}
    for row in records:
        by_id.setdefault(int(row["id"]), []).append(row)
    bad_groups = []
    value_errors = []
    for user_id, rows in sorted(by_id.items()):
        rows.sort(key=lambda row: int(row["rank"]))
        vector = vector_by_id[user_id]
        ranked_values = np.sort(vector.astype(np.float64) ** 2)[::-1]
        cutoff = float(ranked_values[TOP_K - 1])
        mandatory_positions = {
            int(index)
            for index, value in enumerate(vector.astype(np.float64) ** 2)
            if float(value) > cutoff + SHARE_TOLERANCE
        }
        tie_positions = {
            int(index)
            for index, value in enumerate(vector.astype(np.float64) ** 2)
            if abs(float(value) - cutoff) <= SHARE_TOLERANCE
        }
        returned_positions = [int(row["position"]) for row in rows]
        actual_by_position = set(returned_positions)
        selected_mandatory = mandatory_positions <= actual_by_position
        selected_boundary = len(actual_by_position & tie_positions) >= max(
            0, TOP_K - len(mandatory_positions)
        )
        bad_position = any(position < 0 or position >= len(feature_columns) for position in returned_positions)
        duplicate_position = len(actual_by_position) != len(returned_positions)
        bad_rank = [int(row["rank"]) for row in rows] != list(range(1, TOP_K + 1))
        for row in rows:
            position = int(row["position"])
            if 0 <= position < len(feature_columns):
                expected = float(vector[position]) ** 2
                value_errors.append(abs(float(row["view_share"]) - expected))
        if (
            len(rows) != TOP_K
            or not selected_mandatory
            or not selected_boundary
            or bad_position
            or duplicate_position
            or bad_rank
        ):
            bad_groups.append(
                {
                    "id": user_id,
                    "row_count": len(rows),
                    "positions": returned_positions,
                    "mandatory_positions_missing": sorted(mandatory_positions - actual_by_position),
                    "boundary_tie_positions": sorted(tie_positions),
                    "invalid_position": bad_position,
                    "duplicate_position": duplicate_position,
                    "rank_sequence": [int(row["rank"]) for row in rows],
                }
            )
    max_error = max(value_errors, default=0.0)
    passed = not bad_groups and max_error <= SHARE_TOLERANCE
    validation.check(
        "top_k_positions_and_squared_feature_shares",
        passed,
        {
            "users": len(by_id),
            "expected_users": [8981, 52253, 65484],
            "rows_per_user": TOP_K,
            "position_convention": "zero-based index into features.json columns",
            "selected_share_formula": "input feature value squared",
            "max_share_absolute_error": max_error,
            "share_tolerance": SHARE_TOLERANCE,
            "bad_groups": bad_groups,
        },
    )
    return {
        "user_count": len(by_id),
        "positions_are_zero_based": True,
        "max_share_absolute_error": max_error,
        "rows": [
            {
                "id": int(row["id"]),
                "rank": int(row["rank"]),
                "position": int(row["position"]),
                "feature": feature_columns[int(row["position"])],
                "view_share": float(row["view_share"]),
            }
            for row in sorted(records, key=lambda row: (int(row["id"]), int(row["rank"])))
        ],
    }


def orthonormal_basis(values: np.ndarray) -> np.ndarray:
    q, r = np.linalg.qr(values.astype(np.float64), mode="reduced")
    if np.any(np.abs(np.diag(r)) <= np.finfo(np.float64).eps):
        raise ValueError("projection output has rank below the component count")
    return q


def validate_projection(
    validation: Validation,
    method: str,
    records: list[dict[str, Any]],
    matrix: np.ndarray,
    *,
    centered: bool,
) -> dict[str, Any]:
    ordered = sorted(records, key=lambda row: int(row["id"]))
    columns = [f"pc_{index}" for index in range(3)]
    observed = np.array([[float(row[column]) for column in columns] for row in ordered])
    reference_input = matrix.astype(np.float64)
    if centered:
        reference_input = reference_input - reference_input.mean(axis=0, keepdims=True)
    u, singular_values, _ = np.linalg.svd(reference_input, full_matrices=False)
    reference = u[:, :3] * singular_values[:3]
    observed_basis = orthonormal_basis(observed)
    reference_basis = orthonormal_basis(reference)
    cosines = np.linalg.svd(observed_basis.T @ reference_basis, compute_uv=False)
    cosines = np.clip(cosines, 0.0, 1.0)
    max_angle = math.degrees(math.acos(float(cosines.min())))
    observed_gram = observed @ observed.T
    reference_gram = reference @ reference.T
    gram_error = observed_gram - reference_gram
    gram_relative = float(np.linalg.norm(gram_error, ord="fro") / np.linalg.norm(reference_gram, ord="fro"))
    gram_max_abs = float(np.max(np.abs(gram_error)))
    total_spectral_energy = float(np.sum(singular_values ** 2))
    retained_spectral_energy = float(np.sum(singular_values[:3] ** 2))
    retained_spectral_fraction = retained_spectral_energy / total_spectral_energy
    observed_score_energy = float(np.sum(observed ** 2))
    observed_score_fraction = observed_score_energy / total_spectral_energy
    projection_trustworthiness = float(trustworthiness(matrix, observed, n_neighbors=15))
    observed_projector = observed_basis @ observed_basis.T
    reference_projector = reference_basis @ reference_basis.T
    projector_relative = float(
        np.linalg.norm(observed_projector - reference_projector, ord="fro")
        / np.linalg.norm(reference_projector, ord="fro")
    )
    passed = (
        max_angle <= MAX_SUBSPACE_ANGLE_DEGREES
        and gram_relative <= MAX_PROJECTION_GRAM_RELATIVE_ERROR
    )
    validation.check(
        f"{method}_numpy_subspace_and_score_gram",
        passed,
        {
            "input_centered": centered,
            "components": 3,
            "max_principal_angle_degrees": max_angle,
            "principal_cosines": [float(value) for value in cosines],
            "subspace_projector_relative_frobenius_error": projector_relative,
            "score_gram_relative_frobenius_error": gram_relative,
            "score_gram_max_absolute_error": gram_max_abs,
            "numpy_singular_values": [float(value) for value in singular_values[:3]],
            "retained_spectral_fraction": retained_spectral_fraction,
            "observed_score_energy_fraction": observed_score_fraction,
            "trustworthiness_at_15_neighbors": projection_trustworthiness,
            "max_subspace_angle_degrees": MAX_SUBSPACE_ANGLE_DEGREES,
            "max_score_gram_relative_error": MAX_PROJECTION_GRAM_RELATIVE_ERROR,
            "sign_comparison": "not used; score Gram and subspace angles are sign-invariant",
        },
    )
    return {
        "input_centered": centered,
        "components": 3,
        "max_principal_angle_degrees": max_angle,
        "principal_cosines": [float(value) for value in cosines],
        "subspace_projector_relative_frobenius_error": projector_relative,
        "score_gram_relative_frobenius_error": gram_relative,
        "score_gram_max_absolute_error": gram_max_abs,
        "numpy_singular_values": [float(value) for value in singular_values[:3]],
        "spectral_retention": {
            "metric": (
                "fraction_of_centered_variance"
                if centered
                else "fraction_of_uncentered_squared_frobenius_energy"
            ),
            "computed_from": "full Float64 input SVD",
            "retained_squared_singular_value_energy": retained_spectral_energy,
            "total_squared_singular_value_energy": total_spectral_energy,
            "fraction_retained": retained_spectral_fraction,
            "observed_projection_score_energy_fraction": observed_score_fraction,
            "interpretation": "descriptive compression measure, not business quality",
        },
        "trustworthiness": {
            "value": projection_trustworthiness,
            "n_neighbors": 15,
            "input_dimensions": int(matrix.shape[1]),
            "projection_dimensions": 3,
            "interpretation": "descriptive neighborhood preservation metric, no acceptance threshold",
        },
    }


def cluster_shares(
    labels: np.ndarray,
    matrix: np.ndarray,
    feature_columns: list[str],
) -> list[dict[str, Any]]:
    summaries = []
    for label in sorted(int(value) for value in np.unique(labels)):
        member_matrix = matrix[labels == label].astype(np.float64)
        means = np.mean(member_matrix * member_matrix, axis=0)
        summaries.append(
            {
                "cluster_id": label,
                "size": int(len(member_matrix)),
                "mean_squared_feature_shares": {
                    feature: float(value)
                    for feature, value in zip(feature_columns, means, strict=True)
                },
            }
        )
    return summaries


def validate_hdbscan(
    validation: Validation,
    records: list[dict[str, Any]],
    matrix: np.ndarray,
    feature_columns: list[str],
) -> dict[str, Any]:
    ordered = sorted(records, key=lambda row: int(row["id"]))
    labels = np.array([int(row["cluster_id"]) for row in ordered], dtype=np.int64)
    probabilities = np.array([float(row["probability"]) for row in ordered], dtype=np.float64)
    labels_valid = bool(np.all(labels >= -1))
    probabilities_finite = bool(np.all(np.isfinite(probabilities)))
    probabilities_in_range = bool(np.all((probabilities >= 0.0) & (probabilities <= 1.0)))
    noise = labels == -1
    noise_probability_max = float(np.max(probabilities[noise])) if np.any(noise) else None
    noise_probability_zero = noise_probability_max is None or noise_probability_max <= 1e-7
    validation.check(
        "hdbscan_label_probability_and_noise_semantics",
        labels_valid and probabilities_finite and probabilities_in_range and noise_probability_zero,
        {
            "labels_at_least_minus_one": labels_valid,
            "probabilities_finite": probabilities_finite,
            "probability_range": [float(np.min(probabilities)), float(np.max(probabilities))],
            "probabilities_within_zero_one": probabilities_in_range,
            "noise_rows": int(noise.sum()),
            "noise_fraction": float(noise.mean()),
            "maximum_noise_probability": noise_probability_max,
            "noise_probability_tolerance": 1e-7,
        },
    )
    clusters = cluster_shares(labels, matrix, feature_columns)
    return {
        "row_count": len(records),
        "noise_rows": int(noise.sum()),
        "noise_fraction": float(noise.mean()),
        "non_noise_cluster_count": int(len(set(labels.tolist()) - {-1})),
        "probability_min": float(np.min(probabilities)),
        "probability_mean": float(np.mean(probabilities)),
        "probability_max": float(np.max(probabilities)),
        "clusters_including_noise": clusters,
    }


def validate_kmeans(
    validation: Validation,
    records: list[dict[str, Any]],
    sorted_ids: np.ndarray,
    matrix: np.ndarray,
    feature_columns: list[str],
) -> dict[str, Any]:
    ordered = sorted(records, key=lambda row: int(row["id"]))
    labels = np.array([int(row["cluster_id"]) for row in ordered], dtype=np.int64)
    labels_valid = bool(np.all((labels >= 0) & (labels < 8)))
    cluster_ids, counts = np.unique(labels, return_counts=True)
    sizes = {int(label): int(count) for label, count in zip(cluster_ids, counts, strict=True)}
    centers = np.zeros((8, matrix.shape[1]), dtype=np.float64)
    inertia = 0.0
    for label in range(8):
        members = matrix[labels == label].astype(np.float64)
        if len(members):
            centers[label] = members.mean(axis=0)
            inertia += float(np.sum((members - centers[label]) ** 2))
    validation.check(
        "kmeans_labels_and_inertia_summary",
        labels_valid and len(records) == len(sorted_ids),
        {
            "rows": len(records),
            "cluster_id_range": [0, 7],
            "labels_in_range": labels_valid,
            "clusters_observed": len(cluster_ids),
            "empty_cluster_ids": sorted(set(range(8)) - set(sizes)),
            "cluster_sizes": sizes,
            "squared_l2_inertia_from_returned_labels": inertia,
        },
    )
    sklearn_model = KMeans(n_clusters=8, n_init=5, random_state=42, algorithm="lloyd")
    sklearn_labels = sklearn_model.fit_predict(matrix)
    sklearn_sizes = {
        int(label): int(count)
        for label, count in zip(*np.unique(sklearn_labels, return_counts=True), strict=True)
    }
    ari = float(adjusted_rand_score(labels, sklearn_labels))
    summaries = cluster_shares(labels, matrix, feature_columns)
    return {
        "row_count": len(records),
        "requested_cluster_count": 8,
        "clusters_observed": len(cluster_ids),
        "cluster_sizes": sizes,
        "squared_l2_inertia_from_returned_labels": inertia,
        "clusters": summaries,
        "sklearn_comparison": {
            "purpose": "descriptive only; cluster labels are compared permutation-invariantly",
            "n_clusters": 8,
            "n_init": 5,
            "random_state": 42,
            "algorithm": "lloyd",
            "inertia": float(sklearn_model.inertia_),
            "cluster_sizes": sklearn_sizes,
            "adjusted_rand_index": ari,
            "inertia_absolute_difference": abs(inertia - float(sklearn_model.inertia_)),
        },
    }


def validate_umap(
    validation: Validation,
    records: list[dict[str, Any]],
    matrix: np.ndarray,
) -> dict[str, Any]:
    ordered = sorted(records, key=lambda row: int(row["id"]))
    embedding = np.array(
        [[float(row["dim_0"]), float(row["dim_1"])] for row in ordered],
        dtype=np.float64,
    )
    finite = bool(np.all(np.isfinite(embedding)))
    shape_valid = embedding.shape == (len(matrix), 2)
    score = (
        float(trustworthiness(matrix, embedding, n_neighbors=15))
        if finite and shape_valid
        else None
    )
    validation.check(
        "umap_rows_dimensions_and_finite_coordinates",
        finite and shape_valid,
        {
            "rows": int(embedding.shape[0]),
            "dimensions": int(embedding.shape[1]) if embedding.ndim == 2 else None,
            "expected_rows": int(len(matrix)),
            "expected_dimensions": 2,
            "all_coordinates_finite": finite,
        },
    )
    return {
        "row_count": int(embedding.shape[0]),
        "dimensions": int(embedding.shape[1]),
        "all_coordinates_finite": finite,
        "trustworthiness": {
            "value": score,
            "n_neighbors": 15,
            "input_dimensions": int(matrix.shape[1]),
            "interpretation": "descriptive embedding quality metric, no acceptance threshold",
        },
    }


def validate_pairwise(
    validation: Validation,
    records: list[dict[str, Any]],
    vector_by_id: dict[int, np.ndarray],
) -> dict[str, Any]:
    expected_ids = {8981, 52253, 65484}
    row_keys = [(int(row["x_id"]), int(row["y_id"])) for row in records]
    expected_pairs = {(left, right) for left in expected_ids for right in expected_ids}
    errors = []
    for row in records:
        left = vector_by_id[int(row["x_id"])].astype(np.float64)
        right = vector_by_id[int(row["y_id"])].astype(np.float64)
        direct = float(np.sum((left - right) ** 2))
        errors.append(abs(float(row["distance"]) - direct))
    unique_pairs = set(row_keys)
    ids_valid = unique_pairs == expected_pairs and len(row_keys) == len(expected_pairs)
    ordinal_pairs: dict[tuple[str, int], int] = {}
    ordinal_consistent = True
    for row in records:
        for axis, ordinal_key, id_key in (
            ("x", "x_ordinal", "x_id"),
            ("y", "y_ordinal", "y_id"),
        ):
            key = (axis, int(row[ordinal_key]))
            user_id = int(row[id_key])
            if key in ordinal_pairs and ordinal_pairs[key] != user_id:
                ordinal_consistent = False
            ordinal_pairs[key] = user_id
    diagonal = [row for row in records if int(row["x_id"]) == int(row["y_id"])]
    diagonal_zero = all(float(row["distance"]) == 0.0 for row in diagonal)
    max_error = max(errors, default=0.0)
    passed = (
        ids_valid
        and ordinal_consistent
        and len(diagonal) == 3
        and diagonal_zero
        and max_error <= EXACT_DISTANCE_TOLERANCE
    )
    validation.check(
        "pairwise_distance_direct_squared_l2_and_diagonal",
        passed,
        {
            "rows": len(records),
            "expected_rows": 9,
            "ids_match_selected_users": ids_valid,
            "ordinal_to_id_consistent": ordinal_consistent,
            "diagonal_rows": len(diagonal),
            "diagonal_distances_exactly_zero": diagonal_zero,
            "max_absolute_error_vs_numpy": max_error,
            "distance_tolerance": EXACT_DISTANCE_TOLERANCE,
        },
    )
    return {
        "rows": len(records),
        "selected_ids": sorted(expected_ids),
        "diagonal_rows": len(diagonal),
        "max_absolute_error_vs_numpy": max_error,
        "pairs": [
            {
                "x_id": int(row["x_id"]),
                "y_id": int(row["y_id"]),
                "distance": float(row["distance"]),
            }
            for row in sorted(records, key=lambda row: (int(row["x_id"]), int(row["y_id"])))
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    tastes_path = args.results / "prepare_tastes.parquet"
    features_doc = json.loads(args.features.read_text())
    feature_columns = list(features_doc["columns"])
    validation = Validation()

    tastes_table, source_records = table_records(tastes_path)
    source_records.sort(key=lambda row: int(row["id"]))
    ids = np.array([int(row["id"]) for row in source_records], dtype=np.int64)
    matrix, null_counts = to_matrix(tastes_table, feature_columns)
    row_order = np.argsort(np.asarray(tastes_table["id"].to_numpy()))
    matrix = matrix[row_order]
    vectors_finite = bool(np.all(np.isfinite(matrix)))
    shape_valid = len(source_records) == 1919 and matrix.shape == (1919, 17)
    ids_unique = len(np.unique(ids)) == len(ids)
    query_present = SEED_ID in set(ids.tolist())
    validation.check(
        "tastes_input_dimensions_and_values",
        shape_valid and ids_unique and query_present and vectors_finite,
        {
            "rows": int(len(ids)),
            "feature_count": len(feature_columns),
            "expected_rows": 1919,
            "expected_feature_count": 17,
            "feature_columns_match_features_json": feature_columns == features_doc["columns"],
            "unique_ids": ids_unique,
            "query_id_present": query_present,
            "feature_values_finite_after_sql_coalesce": vectors_finite,
            "null_feature_values_coalesced_to_zero": null_counts,
        },
    )
    if len(matrix) != len(ids):
        raise ValueError("input table feature matrix and ids differ in row count")
    source_by_id = {int(row["id"]): row for row in source_records}
    vector_by_id = {int(user_id): matrix[index] for index, user_id in enumerate(ids)}

    provenance_paths = [tastes_path, args.features]
    artifact_names = [
        "cuvs_brute_force_knn",
        "cuvs_cagra",
        "cuvs_ivf_flat",
        "cuvs_ivf_pq_valid",
        "cuvs_pairwise_distance",
        "cuvs_top_k",
        "cuvs_pca",
        "cuvs_kmeans",
        "cuml_tsvd",
        "cuml_hdbscan",
        "cuml_umap",
    ]
    artifact_paths = {name: args.results / f"{name}.parquet" for name in artifact_names}
    probe_paths = {
        name: ROOT / "docs/research/merrec/sql/probes" / f"{name}.sql"
        for name in artifact_names
    }
    provenance_paths.extend(artifact_paths.values())
    provenance_paths.extend(probe_paths.values())
    artifact_data: dict[str, tuple[Any, list[dict[str, Any]]]] = {
        name: table_records(path) for name, path in artifact_paths.items()
    }

    exact_matrix = matrix.astype(np.float64)
    query_vector = vector_by_id[SEED_ID].astype(np.float64)
    candidate_mask = ids != SEED_ID
    candidate_ids = ids[candidate_mask]
    candidate_vectors = exact_matrix[candidate_mask]
    exact_distances_all = np.sum((candidate_vectors - query_vector) ** 2, axis=1)
    exact_order = np.lexsort((candidate_ids, exact_distances_all))
    exact_ids = candidate_ids[exact_order]
    exact_distances = exact_distances_all[exact_order]
    exact_distances_by_id = {
        int(user_id): float(distance)
        for user_id, distance in zip(candidate_ids, exact_distances_all, strict=True)
    }
    validation.results["numpy_exact_knn_oracle"] = {
        "query_id": SEED_ID,
        "metric": "float64 squared L2 over float32 source values",
        "candidate_count": int(len(candidate_ids)),
        "top_10": [
            {"rank": rank, "neighbor_id": int(user_id), "distance": float(distance)}
            for rank, (user_id, distance) in enumerate(
                zip(exact_ids[:K], exact_distances[:K], strict=True), start=1
            )
        ],
        "kth_distance": float(exact_distances[K - 1]),
        "tie_tolerance": DISTANCE_TIE_TOLERANCE,
        "boundary_tie_count": int(
            np.sum(np.abs(exact_distances - exact_distances[K - 1]) <= DISTANCE_TIE_TOLERANCE)
        ),
    }
    for name in artifact_names:
        records = artifact_data[name][1]
        if name in {"cuvs_brute_force_knn", "cuvs_cagra", "cuvs_ivf_flat", "cuvs_ivf_pq_valid"}:
            validation.results[name] = validate_knn(
                validation,
                name,
                records,
                ids,
                exact_ids,
                exact_distances,
                exact_distances_by_id,
                source_by_id,
                vector_by_id,
            )
        elif name == "cuvs_pairwise_distance":
            validation.results[name] = validate_pairwise(validation, records, vector_by_id)
        else:
            if name == "cuvs_top_k":
                observed_ids = sorted({int(row["id"]) for row in records})
                expected_ids = [8981, 52253, 65484]
                validation.check(
                    "top_k_selected_user_coverage",
                    observed_ids == expected_ids
                    and len(records) == len(expected_ids) * TOP_K,
                    {
                        "expected_user_ids": expected_ids,
                        "actual_user_ids": observed_ids,
                        "rows": len(records),
                    },
                )
                validation.results[name] = validate_top_k(
                    validation, records, vector_by_id, feature_columns
                )
            else:
                check_id_coverage(validation, records, ids, name=name)
                ordinal_check(validation, records, ids, name=name)
            if name == "cuvs_pca":
                validation.results[name] = validate_projection(
                    validation, name, records, matrix, centered=True
                )
            elif name == "cuml_tsvd":
                validation.results[name] = validate_projection(
                    validation, name, records, matrix, centered=False
                )
            elif name == "cuml_hdbscan":
                validation.results[name] = validate_hdbscan(
                    validation, records, matrix, feature_columns
                )
            elif name == "cuvs_kmeans":
                validation.results[name] = validate_kmeans(
                    validation, records, ids, matrix, feature_columns
                )
            elif name == "cuml_umap":
                validation.results[name] = validate_umap(validation, records, matrix)

    output_hashes = {
        str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): sha256_file(path)
        for path in provenance_paths
    }
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "validator": str(Path(__file__).relative_to(ROOT)),
        "validator_sha256": sha256_file(Path(__file__).resolve()),
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pyarrow": __import__("pyarrow").__version__,
            "scikit_learn": sklearn.__version__,
            "pandas_used": False,
            "gpu_or_server_used": False,
        },
        "inputs": {
            "tastes_parquet": str(tastes_path.relative_to(ROOT)) if tastes_path.is_relative_to(ROOT) else str(tastes_path),
            "features_json": str(args.features.relative_to(ROOT)) if args.features.is_relative_to(ROOT) else str(args.features),
            "probe_sql": {
                name: str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
                for name, path in probe_paths.items()
            },
            "rows": int(len(ids)),
            "feature_columns": feature_columns,
            "hashes_sha256": output_hashes,
        },
        "tolerances": {
            "exact_squared_l2_absolute": EXACT_DISTANCE_TOLERANCE,
            "squared_feature_share_absolute": SHARE_TOLERANCE,
            "knn_boundary_tie_squared_l2": DISTANCE_TIE_TOLERANCE,
            "minimum_ann_recall_at_10": MIN_ANN_RECALL_AT_10,
            "projection_score_gram_relative_frobenius": MAX_PROJECTION_GRAM_RELATIVE_ERROR,
            "projection_max_principal_angle_degrees": MAX_SUBSPACE_ANGLE_DEGREES,
        },
        "checks": validation.checks,
        "results": validation.results,
        "failure_count": sum(not check["passed"] for check in validation.checks),
        "overall_pass": validation.passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "overall_pass": report["overall_pass"],
                "checks_passed": len(validation.checks) - report["failure_count"],
                "checks_failed": report["failure_count"],
                "output": str(args.output),
                "ann_recall_at_10": {
                    name: validation.results[name]["recall_at_10_tie_aware"]
                    for name in ("cuvs_cagra", "cuvs_ivf_flat", "cuvs_ivf_pq_valid")
                },
                "umap_trustworthiness": validation.results["cuml_umap"]["trustworthiness"]["value"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if validation.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
