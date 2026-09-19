#!/usr/bin/env python3
"""Audit exact challenge neighbors and grouped attribution Top-K outputs on CPU."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "target/ember2024-research"
REPORTS = ROOT / "docs/research/ember2024/results"
ID_BASE = 20_000_000
K = 6
DISTANCE_ATOL = 1e-6
TOPK_ATOL = 1e-6
ATTRIBUTION_K = 3


def source_win32() -> tuple[np.ndarray, np.ndarray, list[str]]:
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
    histograms = vectors["histogram"].combine_chunks()
    dimension = histograms.type.list_size
    values = histograms.values.to_numpy().reshape(vectors.num_rows, dimension)
    selected = np.asarray(values[positions], dtype=np.float64)
    if not np.isfinite(selected).all():
        raise ValueError("Win32 challenge histograms contain non-finite values")
    return ids, selected, families


def exact_distance_matrix(vectors: np.ndarray) -> np.ndarray:
    distances = np.empty((vectors.shape[0], vectors.shape[0]), dtype=np.float64)
    for query_index, query in enumerate(vectors):
        difference = vectors - query
        distances[query_index] = np.sum(difference * difference, axis=1, dtype=np.float64)
    return distances


def check_neighbors(ids: np.ndarray, vectors: np.ndarray, families: list[str]) -> tuple[dict, list[str]]:
    violations: list[str] = []
    path = DATA / "47_challenge_exact_neighbors_shared.parquet"
    result = pq.read_table(path)
    required = ("query_id", "rank", "neighbor_id", "distance")
    missing = [column for column in required if column not in result.column_names]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    query_ids = result["query_id"].to_numpy()
    ranks = result["rank"].to_numpy()
    neighbor_ids = result["neighbor_id"].to_numpy()
    distances = result["distance"].to_numpy().astype(np.float64, copy=False)
    query_counts = Counter(int(value) for value in query_ids)
    rows_per_query_correct = (
        result.num_rows == len(ids) * K
        and set(query_counts) == set(int(value) for value in ids)
        and all(count == K for count in query_counts.values())
    )
    if not rows_per_query_correct:
        violations.append("neighbor output must contain six rows for every source query id")

    ranks_correct = True
    unique_neighbors_correct = True
    for query_id in query_counts:
        row_mask = query_ids == query_id
        query_ranks = ranks[row_mask]
        query_neighbors = neighbor_ids[row_mask]
        if not np.array_equal(np.sort(query_ranks), np.arange(1, K + 1, dtype=query_ranks.dtype)):
            ranks_correct = False
        if len(np.unique(query_neighbors)) != K:
            unique_neighbors_correct = False
    if not ranks_correct:
        violations.append("each query must have ranks 1 through 6")
    if not unique_neighbors_correct:
        violations.append("neighbor ids must be unique within each query")

    all_distances_finite = bool(np.isfinite(distances).all())
    all_distances_nonnegative = bool(all_distances_finite and np.all(distances >= 0))
    if not all_distances_finite:
        violations.append("returned neighbor distances must be finite")
    if all_distances_finite and not all_distances_nonnegative:
        violations.append("returned neighbor distances must be nonnegative")

    id_positions = np.searchsorted(ids, query_ids)
    neighbor_positions = np.searchsorted(ids, neighbor_ids)
    query_ids_in_source = bool(
        np.all(id_positions < len(ids)) and np.array_equal(ids[id_positions], query_ids)
    )
    neighbors_in_source = bool(
        np.all(neighbor_positions < len(ids)) and np.array_equal(ids[neighbor_positions], neighbor_ids)
    )
    if not query_ids_in_source:
        violations.append("query ids must match the sorted Win32 source ids")
    if not neighbors_in_source:
        violations.append("neighbor ids must belong to the Win32 source rows")

    self_pairs = query_ids == neighbor_ids
    self_zero_failures = int(np.count_nonzero(distances[self_pairs] != 0.0)) if all_distances_finite else None
    if self_zero_failures:
        violations.append("returned self-neighbor distances must be exactly zero")

    distance_matrix = exact_distance_matrix(vectors)
    exact_kth = np.partition(distance_matrix, K - 1, axis=1)[:, K - 1]
    pair_errors = None
    topk_excess = None
    topk_violations = None
    max_absolute_error = None
    p50_absolute_error = None
    p99_absolute_error = None
    max_above_kth = None
    strict_failure_examples = []
    if query_ids_in_source and neighbors_in_source:
        pair_difference = vectors[id_positions] - vectors[neighbor_positions]
        returned_exact_distances = np.sum(pair_difference * pair_difference, axis=1, dtype=np.float64)
        pair_errors = np.abs(distances - returned_exact_distances)
        topk_excess = returned_exact_distances - exact_kth[id_positions]
        topk_violations = int(np.count_nonzero(topk_excess > TOPK_ATOL))
        if topk_violations:
            violations.append("returned exact neighbor distances exceed the exact sixth-neighbor threshold plus tolerance")
        finite_errors = pair_errors[np.isfinite(pair_errors)]
        if finite_errors.size:
            max_absolute_error = float(finite_errors.max())
            p50_absolute_error = float(np.quantile(finite_errors, 0.50))
            p99_absolute_error = float(np.quantile(finite_errors, 0.99))
        max_above_kth = float(max(0.0, topk_excess.max()))
        strict_failure_examples = [
            {
                "query_id": int(query_ids[index]),
                "rank": int(ranks[index]),
                "neighbor_id": int(neighbor_ids[index]),
                "returned_distance": float(distances[index]),
                "exact_distance": float(returned_exact_distances[index]),
                "absolute_error": float(pair_errors[index]),
            }
            for index in np.flatnonzero(pair_errors > DISTANCE_ATOL)
        ]

    family_by_id = {int(sample_id): family for sample_id, family in zip(ids, families, strict=True)}
    known_family_nonself_pairs = 0
    same_family_nonself_pairs = 0
    query_equals_neighbor = int(np.count_nonzero(self_pairs))
    if query_ids_in_source and neighbors_in_source:
        for query_id, neighbor_id in zip(query_ids, neighbor_ids, strict=True):
            if query_id == neighbor_id:
                continue
            query_family = family_by_id[int(query_id)]
            neighbor_family = family_by_id[int(neighbor_id)]
            if query_family and neighbor_family:
                known_family_nonself_pairs += 1
                same_family_nonself_pairs += int(query_family == neighbor_family)
    same_family_fraction = (
        same_family_nonself_pairs / known_family_nonself_pairs if known_family_nonself_pairs else None
    )

    strict_failures = (
        int(np.count_nonzero(pair_errors > DISTANCE_ATOL)) if pair_errors is not None else None
    )
    strict_passed = strict_failures == 0 if strict_failures is not None else False
    report = {
        "rows": int(result.num_rows),
        "queries": int(len(ids)),
        "max_absolute_distance_error": max_absolute_error,
        "p50_absolute_distance_error": p50_absolute_error,
        "p99_absolute_distance_error": p99_absolute_error,
        "max_returned_distance_above_exact_kth": max_above_kth,
        "distance_atol": DISTANCE_ATOL,
        "topk_atol": TOPK_ATOL,
        "strict_distance_audit_failure_count": strict_failures,
        "strict_distance_failures": strict_failure_examples,
        "all_distances_finite": all_distances_finite,
        "all_distances_nonnegative": all_distances_nonnegative,
        "exact_distance_check_passed": strict_passed,
        "self_neighbor_pairs": query_equals_neighbor,
        "self_zero_failures": self_zero_failures,
        "topk_membership_violation_count": topk_violations,
        "topk_membership_with_ties_passed": topk_violations == 0 if topk_violations is not None else False,
        "rows_per_query_correct": rows_per_query_correct,
        "ranks_1_through_6_correct": ranks_correct,
        "unique_neighbor_ids_per_query": unique_neighbors_correct,
        "query_ids_match_sorted_source": query_ids_in_source,
        "neighbor_ids_in_source": neighbors_in_source,
        "known_family_nonself_pairs": known_family_nonself_pairs,
        "same_family_nonself_pairs": same_family_nonself_pairs,
        "known_pair_same_family_fraction": same_family_fraction,
        "family_pair_population": "non-self pairs with non-empty family labels for both query and neighbor",
        "distinct_histograms": int(np.unique(vectors, axis=0).shape[0]),
        "numerical_observation": {
            "classification": "numerical_observation",
            "status": "unresolved" if strict_failures is None or strict_failures else "within_tolerance",
            "confirmed_engine_bug": False,
            "tolerance_scope": "strict_research_audit_not_public_contract",
        },
        "exit_status_policy": "strict distance audit failures are reported and do not fail this script; structural, top-k, and attribution value violations do",
        "violations": violations,
    }
    return report, violations


def check_attribution_topk() -> tuple[dict, list[str]]:
    violations: list[str] = []
    attribution_path = DATA / "input_attributions.parquet"
    groups_path = DATA / "input_feature_groups.parquet"
    result_path = DATA / "28_top_k_attributions_shared.parquet"
    source = pq.read_table(attribution_path, columns=["id", "vector"]).sort_by([("id", "ascending")])
    groups = pq.read_table(groups_path, columns=["group_index", "group_name"]).sort_by(
        [("group_index", "ascending")]
    )
    result = pq.read_table(result_path)

    ids = source["id"].to_numpy()
    vectors = source["vector"].combine_chunks()
    source_values = vectors.values.to_numpy().reshape(source.num_rows, vectors.type.list_size)
    group_names = groups["group_name"].to_pylist()
    group_indices = groups["group_index"].to_numpy()
    group_index_by_name = {name: int(index) for name, index in zip(group_names, group_indices, strict=True)}
    if len(group_index_by_name) != groups.num_rows:
        violations.append("feature group names must be unique")
    if vectors.type.list_size != groups.num_rows:
        violations.append("attribution vector length must match the feature group count")
    if not np.array_equal(group_indices, np.arange(groups.num_rows, dtype=group_indices.dtype)):
        violations.append("feature group indices must match attribution vector positions")
    if not np.isfinite(source_values).all():
        violations.append("source attribution vectors must be finite")

    required = ("id", "rank", "value", "group_name")
    missing = [column for column in required if column not in result.column_names]
    if missing:
        raise ValueError(f"{result_path} missing required columns: {missing}")
    if result.schema.field("value").type != pa.float32():
        violations.append("top-k attribution value must be Float32")
    returned_ids = result["id"].to_numpy()
    ranks = result["rank"].to_numpy()
    values = result["value"].to_numpy().astype(np.float32, copy=False)
    names = result["group_name"].to_pylist()
    if not np.isfinite(values).all():
        violations.append("top-k attribution values must be finite")
    positions = np.searchsorted(ids, returned_ids)
    returned_ids_valid = bool(
        np.all(positions < len(ids)) and np.array_equal(ids[positions], returned_ids)
    )
    if not returned_ids_valid:
        violations.append("top-k attribution ids must belong to the source attribution rows")

    source_positive = np.any(source_values > 0, axis=1)
    expected_ids = set(int(value) for value in ids[source_positive])
    returned_id_set = set(int(value) for value in returned_ids)
    missing_ids = sorted(expected_ids - returned_id_set)
    unexpected_ids = sorted(returned_id_set - expected_ids)
    if missing_ids:
        violations.append("top-k output is missing source ids with positive group contributions")
    if unexpected_ids:
        violations.append("top-k output contains source ids without positive group contributions")

    rows_by_id: dict[int, list[int]] = defaultdict(list)
    for row_index, sample_id in enumerate(returned_ids):
        rows_by_id[int(sample_id)].append(row_index)
    rank_violations = 0
    source_value_mismatches = 0
    maximum_value_order_violations = 0
    max_abs_value_error = 0.0
    for sample_index, sample_id in enumerate(ids):
        sample_id = int(sample_id)
        positive_values = source_values[sample_index][source_values[sample_index] > 0]
        expected_count = min(ATTRIBUTION_K, int(positive_values.size))
        row_indices = rows_by_id.get(sample_id, [])
        sample_ranks = ranks[row_indices]
        expected_ranks = np.arange(1, expected_count + 1, dtype=sample_ranks.dtype)
        if len(row_indices) != expected_count or not np.array_equal(np.sort(sample_ranks), expected_ranks):
            rank_violations += 1
        if not row_indices:
            continue

        expected_values = np.sort(positive_values.astype(np.float32))[::-1][:expected_count]
        ranked = sorted(row_indices, key=lambda row: int(ranks[row]))
        actual_ranked_values = values[ranked]
        if not np.array_equal(actual_ranked_values, expected_values):
            maximum_value_order_violations += 1

        source_row = source_values[sample_index]
        for row_index in row_indices:
            group_name = names[row_index]
            group_index = group_index_by_name.get(group_name)
            if group_index is None:
                source_value_mismatches += 1
                continue
            source_value = np.float32(source_row[group_index])
            result_value = np.float32(values[row_index])
            error = abs(float(result_value) - float(source_value))
            max_abs_value_error = max(max_abs_value_error, error)
            if result_value != source_value:
                source_value_mismatches += 1

    if rank_violations:
        violations.append("top-k output ranks must be 1 through min(3, positive group count) for every source id")
    if source_value_mismatches:
        violations.append("top-k group names and Float32 values must match the source attribution vectors exactly")
    if maximum_value_order_violations:
        violations.append("top-k values must be the largest positive source group values, with ties allowed")

    report = {
        "rows": int(result.num_rows),
        "source_ids": int(source.num_rows),
        "ids_with_positive_group_values": len(expected_ids),
        "ids_returned": len(returned_id_set),
        "missing_positive_id_count": len(missing_ids),
        "missing_positive_id_examples": missing_ids[:20],
        "unexpected_id_count": len(unexpected_ids),
        "rank_violation_count": rank_violations,
        "source_value_mismatch_count": source_value_mismatches,
        "maximum_value_order_violation_count": maximum_value_order_violations,
        "max_abs_source_value_error": max_abs_value_error,
        "maximum_positive_values_checked": ATTRIBUTION_K,
        "ties_allowed": True,
        "passed": not violations,
        "violations": violations,
    }
    return report, violations


def main() -> int:
    ids, vectors, families = source_win32()
    neighbor_report, neighbor_violations = check_neighbors(ids, vectors, families)
    attribution_report, attribution_violations = check_attribution_topk()
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "challenge_neighbor_check.json").write_text(
        json.dumps(neighbor_report, indent=2, sort_keys=True) + "\n"
    )
    (REPORTS / "attribution_topk_check.json").write_text(
        json.dumps(attribution_report, indent=2, sort_keys=True) + "\n"
    )
    failures = neighbor_violations + attribution_violations
    print(
        json.dumps(
            {
                "neighbor_structural_topk_violations": len(neighbor_violations),
                "strict_distance_audit_failures": neighbor_report["strict_distance_audit_failure_count"],
                "attribution_topk_violations": len(attribution_violations),
            },
            sort_keys=True,
        )
    )
    if failures:
        for violation in failures:
            print(violation, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
