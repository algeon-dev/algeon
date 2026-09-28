#!/usr/bin/env python3
"""Check EMBER2024 cuVS neighbor outputs against a bounded Float64 CPU oracle."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path
import resource
import statistics
import sys
import time
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "target/ember2024-research"
REPORT_DIR = ROOT / "docs/research/ember2024/results"
RESULTS = ROOT / "docs/research/ember2024/results"
DIMENSIONS = 512
K = 10
QUERY_BATCH_SIZE = 64
TIE_ABS_TOLERANCE = 1e-12
TIE_REL_TOLERANCE = 1e-12
ALGORITHMS = {
    "cuvs_brute_force_knn": "20_cuvs_brute_force_knn",
    "cuvs_cagra": "21_cuvs_cagra",
    "cuvs_ivf_flat": "22_cuvs_ivf_flat",
    "cuvs_ivf_pq": "23_cuvs_ivf_pq",
}
RUN_SUFFIXES = ("", "_r2", "_r3")


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * 1024 if sys.platform.startswith("linux") else value)


def read_vector_rows(split: str) -> tuple[np.ndarray, np.ndarray, int]:
    path = DATA / f"input_research_vectors_{split}.parquet"
    table = pq.read_table(path, columns=["id", "split", "histogram"])
    ids = table.column("id").combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    splits = table.column("split").combine_chunks().to_pylist()
    if any(value != split for value in splits):
        raise ValueError(f"{path} contains a row from another split")
    histogram_type = table.schema.field("histogram").type
    if not pa.types.is_fixed_size_list(histogram_type) or histogram_type.list_size != DIMENSIONS:
        raise ValueError(f"{path} has unexpected histogram type: {histogram_type}")
    if histogram_type.value_type != pa.float32():
        raise ValueError(f"{path} histogram elements are not Float32")
    vectors = table.column("histogram").combine_chunks()
    matrix = vectors.values.to_numpy(zero_copy_only=False).reshape(len(ids), DIMENSIONS)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{path} has non-finite histogram values")
    order = np.argsort(ids, kind="stable")
    return ids[order], matrix[order], pq.ParquetFile(path).metadata.num_rows


def read_metadata_for_ids(split: str, wanted_ids: np.ndarray) -> tuple[dict[int, dict[str, Any]], int]:
    path = DATA / f"input_metadata_{split}.parquet"
    wanted = set(int(value) for value in wanted_ids)
    value_set = pa.array(sorted(wanted), type=pa.int64())
    columns = ["id", "file_type", "week_id", "sha256", "family", "label"]
    parquet = pq.ParquetFile(path)
    found: dict[int, dict[str, Any]] = {}
    for batch in parquet.iter_batches(columns=columns, batch_size=131_072):
        selected = batch.filter(pc.is_in(batch.column("id"), value_set=value_set))
        for row in selected.to_pylist():
            identifier = int(row["id"])
            if identifier in found:
                raise ValueError(f"duplicate metadata ID {identifier} in {path}")
            found[identifier] = row
    missing = wanted.difference(found)
    if missing:
        raise ValueError(f"{len(missing)} vector IDs are absent from {path}")
    return found, parquet.metadata.num_rows


def load_cohort(split: str) -> dict[str, Any]:
    vector_ids, vectors, vector_rows = read_vector_rows(split)
    metadata, metadata_rows = read_metadata_for_ids(split, vector_ids)
    selected = np.fromiter(
        (metadata[int(identifier)]["file_type"] == "Win32" for identifier in vector_ids),
        dtype=np.bool_,
        count=len(vector_ids),
    )
    ids = vector_ids[selected]
    features = vectors[selected].astype(np.float64, copy=True)
    rows = [metadata[int(identifier)] for identifier in ids]
    if not len(ids):
        raise ValueError(f"no Win32 rows in the sampled {split} vector relation")
    return {
        "split": split,
        "ids": ids,
        "features": features,
        "metadata": {int(row["id"]): row for row in rows},
        "vector_rows": vector_rows,
        "metadata_rows": metadata_rows,
    }


def cohort_summary(cohort: dict[str, Any]) -> dict[str, Any]:
    ids = cohort["ids"]
    metadata = cohort["metadata"]
    rows = list(metadata.values())
    hashes = [row["sha256"] for row in rows]
    weeks = [int(row["week_id"]) for row in rows]
    return {
        "sampled_vector_relation_rows": cohort["vector_rows"],
        "metadata_relation_rows": cohort["metadata_rows"],
        "win32_cohort_rows": len(ids),
        "id_range": [int(ids.min()), int(ids.max())],
        "week_id_range": [min(weeks), max(weeks)],
        "sha256_unique": len(set(hashes)) == len(hashes),
        "unique_sha256_rows": len(set(hashes)),
        "known_family_rows": sum(row["family"] != "" for row in rows),
        "label_counts": {str(label): sum(int(row["label"]) == label for row in rows) for label in sorted({int(row["label"]) for row in rows})},
    }


def direct_squared_l2(corpus: np.ndarray, query: np.ndarray) -> np.ndarray:
    # One corpus-by-feature difference buffer bounds peak memory independently of query count.
    delta = corpus - query
    return np.einsum("ij,ij->i", delta, delta, dtype=np.float64, optimize=False)


def exact_neighbors(
    corpus_ids: np.ndarray,
    corpus: np.ndarray,
    queries: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    result_ids = np.empty((len(queries), K), dtype=np.int64)
    result_cutoffs = np.empty(len(queries), dtype=np.float64)
    result_tie_sizes = np.empty(len(queries), dtype=np.int64)
    started = time.perf_counter()
    for query_index in range(len(queries)):
        distances = direct_squared_l2(corpus, queries[query_index])
        cutoff = float(np.partition(distances, K - 1)[K - 1])
        candidates = np.flatnonzero(distances <= cutoff)
        order = np.lexsort((corpus_ids[candidates], distances[candidates]))
        result_ids[query_index] = corpus_ids[candidates[order[:K]]]
        result_cutoffs[query_index] = cutoff
        result_tie_sizes[query_index] = int(np.count_nonzero(distances == cutoff))
    return result_ids, result_cutoffs, result_tie_sizes, time.perf_counter() - started


def read_json(name: str) -> dict[str, Any]:
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf8"))


def read_result_columns(name: str) -> tuple[dict[str, Any], dict[str, np.ndarray | list[str]]]:
    record = read_json(name)
    if record.get("status") != "ok":
        raise ValueError(f"{name} status is {record.get('status')!r}")
    table = pq.read_table(DATA / f"{name}.parquet")
    if table.num_rows != 36_180:
        raise ValueError(f"{name} contains {table.num_rows} rows, expected 36,180")
    columns: dict[str, np.ndarray | list[str]] = {}
    for name in (
        "query_ordinal", "query_id", "query_week", "query_label", "rank",
        "neighbor_id", "neighbor_week", "neighbor_label", "distance",
    ):
        columns[name] = table.column(name).combine_chunks().to_numpy(zero_copy_only=False)
    for name in ("neighbor_sha256", "neighbor_family"):
        columns[name] = table.column(name).combine_chunks().to_pylist()
    return record, columns


def metadata_contracts(
    columns: dict[str, np.ndarray | list[str]],
    train: dict[str, Any],
    test: dict[str, Any],
    algorithm: str,
    run: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordinals = np.asarray(columns["query_ordinal"], dtype=np.int64)
    query_ids = np.asarray(columns["query_id"], dtype=np.int64)
    ranks = np.asarray(columns["rank"], dtype=np.int64)
    neighbor_ids = np.asarray(columns["neighbor_id"], dtype=np.int64)
    distances = np.asarray(columns["distance"], dtype=np.float64)
    expected_ordinals = np.repeat(np.arange(len(test["ids"]), dtype=np.int64), K)
    expected_ranks = np.tile(np.arange(1, K + 1, dtype=np.int64), len(test["ids"]))
    expected_query_ids = np.repeat(test["ids"], K)
    if not np.array_equal(ordinals, expected_ordinals):
        raise ValueError(f"{algorithm} {run}: query_ordinal does not enumerate the ordered test cohort")
    if not np.array_equal(ranks, expected_ranks):
        raise ValueError(f"{algorithm} {run}: rank is not contiguous 1..10 within each query")
    if not np.array_equal(query_ids, expected_query_ids):
        raise ValueError(f"{algorithm} {run}: query_id does not match query_ordinal and ORDER BY id")
    if len(np.unique(neighbor_ids)) > len(train["ids"]):
        raise ValueError(f"{algorithm} {run}: neighbor IDs exceed the training corpus cardinality")
    train_positions = np.searchsorted(train["ids"], neighbor_ids)
    if np.any(train_positions >= len(train["ids"])) or not np.array_equal(train["ids"][train_positions], neighbor_ids):
        raise ValueError(f"{algorithm} {run}: a neighbor_id is not in the sampled Win32 train corpus")
    if np.intersect1d(query_ids, neighbor_ids).size:
        raise ValueError(f"{algorithm} {run}: query and neighbor IDs overlap")
    if not np.isfinite(distances).all() or np.any(distances < 0):
        raise ValueError(f"{algorithm} {run}: distance is not finite and nonnegative")
    shaped_distances = distances.reshape(len(test["ids"]), K)
    if np.any(np.diff(shaped_distances, axis=1) < -1e-7):
        raise ValueError(f"{algorithm} {run}: ranks are not ordered by nondecreasing distance")

    qweek = np.asarray(columns["query_week"], dtype=np.int64)
    qlabel = np.asarray(columns["query_label"], dtype=np.int64)
    nweek = np.asarray(columns["neighbor_week"], dtype=np.int64)
    nlabel = np.asarray(columns["neighbor_label"], dtype=np.int64)
    nsha = columns["neighbor_sha256"]
    nfamily = columns["neighbor_family"]
    for index in range(len(query_ids)):
        query_meta = test["metadata"][int(query_ids[index])]
        neighbor_meta = train["metadata"][int(neighbor_ids[index])]
        if qweek[index] != int(query_meta["week_id"]) or qlabel[index] != int(query_meta["label"]):
            raise ValueError(f"{algorithm} {run}: query metadata join changed ID {query_ids[index]}")
        if (
            nweek[index] != int(neighbor_meta["week_id"])
            or nlabel[index] != int(neighbor_meta["label"])
            or nsha[index] != neighbor_meta["sha256"]
            or nfamily[index] != neighbor_meta["family"]
        ):
            raise ValueError(f"{algorithm} {run}: neighbor metadata join changed ID {neighbor_ids[index]}")
    return ordinals, query_ids, neighbor_ids


def returned_distances(
    columns: dict[str, np.ndarray | list[str]],
    train: dict[str, Any],
    test: dict[str, Any],
) -> np.ndarray:
    ordinals = np.asarray(columns["query_ordinal"], dtype=np.int64)
    neighbor_ids = np.asarray(columns["neighbor_id"], dtype=np.int64)
    query_ids = np.asarray(columns["query_id"], dtype=np.int64)
    ranks = np.asarray(columns["rank"], dtype=np.int64)
    returned = np.empty(len(neighbor_ids), dtype=np.float64)
    train_positions = np.searchsorted(train["ids"], neighbor_ids)
    for batch_start in range(0, len(test["ids"]), QUERY_BATCH_SIZE):
        batch_stop = min(batch_start + QUERY_BATCH_SIZE, len(test["ids"]))
        row_start, row_stop = batch_start * K, batch_stop * K
        expected = np.arange(batch_start, batch_stop, dtype=np.int64)
        if not np.array_equal(np.unique(ordinals[row_start:row_stop]), expected):
            raise ValueError("result rows are not grouped by query ordinal")
        qindices = ordinals[row_start:row_stop]
        if not np.array_equal(query_ids[row_start:row_stop], np.repeat(test["ids"][expected], K)):
            raise ValueError("query IDs changed while computing returned-neighbor distances")
        differences = train["features"][train_positions[row_start:row_stop]] - test["features"][qindices]
        returned[row_start:row_stop] = np.einsum("ij,ij->i", differences, differences, dtype=np.float64, optimize=False)
    return returned


def family_rates(
    columns: dict[str, np.ndarray | list[str]],
    train: dict[str, Any],
    test: dict[str, Any],
) -> dict[str, Any]:
    query_families = [test["metadata"][int(identifier)]["family"] for identifier in test["ids"]]
    known_queries = [family for family in query_families if family != ""]
    if not known_queries:
        raise ValueError("test cohort has no known family labels for post-hoc analysis")
    train_families = [row["family"] for row in train["metadata"].values()]
    counts = Counter(family for family in train_families if family != "")
    random_probabilities = [counts[family] / len(train_families) for family in known_queries]
    random_baseline = float(np.mean(random_probabilities))

    neighbor_families = columns["neighbor_family"]
    observed: dict[str, Any] = {}
    q_known_pairs = 0
    n_known_pairs = 0
    for k in (1, K):
        matches = 0
        denominator = 0
        known_reference_count = 0
        for query_index, query_family in enumerate(query_families):
            if query_family == "":
                continue
            first = query_index * K
            for family in neighbor_families[first : first + k]:
                denominator += 1
                known_reference_count += family != ""
                matches += family != "" and family == query_family
        if k == K:
            q_known_pairs = denominator
            n_known_pairs = known_reference_count
        observed[str(k)] = {
            "same_known_family_rate": matches / denominator if denominator else None,
            "matching_pairs": matches,
            "denominator_query_known_family_pairs": denominator,
            "known_neighbor_family_coverage": known_reference_count / denominator if denominator else None,
            "expected_uniform_random_train_reference_rate": random_baseline,
        }
    observed["known_test_queries"] = len(known_queries)
    observed["train_family_labeled_rows"] = sum(family != "" for family in train_families)
    return observed


def assess_algorithm(
    algorithm: str,
    stem: str,
    train: dict[str, Any],
    test: dict[str, Any],
    exact_ids: np.ndarray,
    exact_cutoffs: np.ndarray,
    tie_sizes: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    run_results: list[dict[str, Any]] = []
    outputs_by_run: dict[str, np.ndarray] = {}
    canonical: dict[str, Any] | None = None
    canonical_nearest: np.ndarray | None = None
    for suffix in RUN_SUFFIXES:
        name = stem + suffix
        record, columns = read_result_columns(name)
        ordinals, query_ids, neighbor_ids = metadata_contracts(columns, train, test, algorithm, suffix or "base")
        neighbor_matrix = neighbor_ids.reshape(len(test["ids"]), K)
        outputs_by_run[suffix or "base"] = neighbor_matrix
        oracle_distances = returned_distances(columns, train, test)
        recorded = np.asarray(columns["distance"], dtype=np.float64)
        distance_errors = np.abs(recorded - oracle_distances)
        run_summary = {
            "capture": name,
            "query_id": record["query_id"],
            "rows": record["rows"],
            "status": record["status"],
            "csv_export_status": record.get("csv_export_status"),
            "elapsed_seconds": float(record["elapsed_s"]),
            "max_absolute_distance_error": float(distance_errors.max(initial=0.0)),
            "id_matrix_equal_to_base": None,
        }
        run_results.append(run_summary)
        if suffix == "_r3":
            canonical = {"record": record, "columns": columns, "distances": oracle_distances}
            canonical_nearest = neighbor_matrix

    assert canonical is not None and canonical_nearest is not None
    base_matrix = outputs_by_run["base"]
    for run in run_results:
        key = run["capture"][len(stem):] or "base"
        changed = int(np.count_nonzero(outputs_by_run[key] != base_matrix))
        run["id_matrix_equal_to_base"] = changed == 0
        run["neighbor_id_slots_changed_from_base"] = changed

    r3_columns = canonical["columns"]
    flat_r3_ids = canonical_nearest.reshape(-1)
    strict_recall_counts = np.fromiter(
        (len(set(map(int, actual)).intersection(map(int, truth))) for actual, truth in zip(canonical_nearest, exact_ids)),
        dtype=np.int64,
        count=len(test["ids"]),
    )
    tie_tolerance = np.maximum(TIE_ABS_TOLERANCE, np.abs(exact_cutoffs) * TIE_REL_TOLERANCE)
    tie_boundary = exact_cutoffs[:, None] + tie_tolerance[:, None]
    tied_distance_valid = canonical["distances"].reshape(len(test["ids"]), K) <= tie_boundary
    same_family = family_rates(r3_columns, train, test)
    r3_reported = np.asarray(r3_columns["distance"], dtype=np.float64)
    r3_errors = np.abs(r3_reported - canonical["distances"])
    worst_rows = np.argsort(r3_errors, kind="stable")[-10:][::-1]
    top_distance_error_pairs = [
        {
            "query_id": int(np.asarray(r3_columns["query_id"])[index]),
            "neighbor_id": int(np.asarray(r3_columns["neighbor_id"])[index]),
            "rank": int(np.asarray(r3_columns["rank"])[index]),
            "reported_distance": float(r3_reported[index]),
            "float64_direct_squared_l2": float(canonical["distances"][index]),
            "absolute_error": float(r3_errors[index]),
        }
        for index in worst_rows
    ]
    all_elapsed = [run["elapsed_seconds"] for run in run_results]
    summary = {
        "function": algorithm,
        "result_rows": 36_180,
        "k": K,
        "run_captures": run_results,
        "median_full_statement_seconds": statistics.median(all_elapsed),
        "strict_id_recall_at_10": float(strict_recall_counts.sum() / (len(test["ids"]) * K)),
        "tie_aware_recall_at_10": float(tied_distance_valid.sum() / (len(test["ids"]) * K)),
        "queries_with_boundary_ties": int(np.count_nonzero(tie_sizes > 1)),
        "max_absolute_returned_distance_error": float(np.abs(np.asarray(r3_columns["distance"], dtype=np.float64) - canonical["distances"]).max(initial=0.0)),
        "mean_absolute_returned_distance_error": float(np.mean(np.abs(np.asarray(r3_columns["distance"], dtype=np.float64) - canonical["distances"]))),
        "posthoc_same_known_family_rates": same_family,
        "r3_query_id_contract_ok": True,
        "r3_rank_contract_ok": True,
        "r3_id_contract_ok": True,
        "r3_output_neighbor_ids": canonical_nearest,
    }
    if algorithm == "cuvs_ivf_pq":
        summary["r3_worst_distance_error_pairs"] = top_distance_error_pairs
    return summary, canonical_nearest


def pairwise_check(train: dict[str, Any], test: dict[str, Any]) -> dict[str, Any]:
    name = "24_cuvs_pairwise_distance"
    record = read_json(name)
    table = pq.read_table(DATA / f"{name}.parquet")
    if record.get("status") != "ok" or table.num_rows != 2_000:
        raise ValueError(f"{name} is not a successful 2,000-row pairwise capture")
    sql = (ROOT / "docs/research/ember2024/sql/24_cuvs_pairwise_distance.sql").read_text()
    if record["sql"] != sql:
        raise ValueError("pairwise capture SQL no longer matches SQL 24")
    x_ids = test["ids"][:100]
    y_ids = train["ids"][:20]
    x_ord = table.column("x_ordinal").to_numpy(zero_copy_only=False).astype(np.int64)
    y_ord = table.column("y_ordinal").to_numpy(zero_copy_only=False).astype(np.int64)
    actual_x = table.column("x_id").to_numpy(zero_copy_only=False).astype(np.int64)
    actual_y = table.column("y_id").to_numpy(zero_copy_only=False).astype(np.int64)
    distances = table.column("distance").to_numpy(zero_copy_only=False).astype(np.float64)
    expected_x_ord = np.repeat(np.arange(100, dtype=np.int64), 20)
    expected_y_ord = np.tile(np.arange(20, dtype=np.int64), 100)
    expected_x = x_ids[expected_x_ord]
    expected_y = y_ids[expected_y_ord]
    if not np.array_equal(x_ord, expected_x_ord) or not np.array_equal(y_ord, expected_y_ord):
        raise ValueError("pairwise row ordinals do not match the ordered 100-by-20 cohorts")
    if not np.array_equal(actual_x, expected_x) or not np.array_equal(actual_y, expected_y):
        raise ValueError("pairwise x_id/y_id do not match their row ordinals")
    x_positions = np.searchsorted(test["ids"], actual_x)
    y_positions = np.searchsorted(train["ids"], actual_y)
    delta = test["features"][x_positions] - train["features"][y_positions]
    exact = np.einsum("ij,ij->i", delta, delta, dtype=np.float64, optimize=False)
    error = np.abs(distances - exact)
    return {
        "capture": name,
        "query_id": record["query_id"],
        "rows": table.num_rows,
        "x_rows": 100,
        "y_rows": 20,
        "ordinal_and_id_contracts_ok": True,
        "metric": "squared L2, independent Float64 direct-difference oracle",
        "max_absolute_distance_error": float(error.max(initial=0.0)),
        "mean_absolute_distance_error": float(error.mean()),
    }


def write_markdown(report: dict[str, Any]) -> None:
    cohort = report["cohort"]
    algorithms = report["algorithms"]
    lines = [
        "# EMBER2024 near-neighbor validation",
        "",
        "The benchmark uses the one-percent SHA-256 sampled Win32 train and test",
        "cohorts with the 512-element normalized histogram (`f7..f518`). Search",
        "inputs contain only `id` and `vector`; labels, family, week, and SHA-256 are",
        "joined after each UDTF for output checks and post-hoc analysis.",
        "",
        "## Cohorts and identity",
        "",
        "| Split | Sampled Win32 rows | ID range | Week range | Known-family rows | SHA-256 unique |",
        "| --- | ---: | --- | --- | ---: | --- |",
    ]
    for split in ("train", "test"):
        item = cohort[split]
        lines.append(
            f"| {split} | {item['win32_cohort_rows']:,} | {item['id_range'][0]:,}–{item['id_range'][1]:,} | "
            f"{item['week_id_range'][0]}–{item['week_id_range'][1]} | {item['known_family_rows']:,} | {item['sha256_unique']} |"
        )
    disjoint = cohort["sha256_disjoint"]
    lines += [
        "",
        f"The sampled cohorts have **{disjoint['intersection_rows']:,} shared SHA-256 values**; SHA-256 disjointness is `{disjoint['disjoint']}`. "
        f"The temporal split remains train before test. The sampled train and test vectors contain "
        f"{cohort['train']['sampled_vector_relation_rows']:,} and {cohort['test']['sampled_vector_relation_rows']:,} rows across all file types; the search filters these to Win32.",
        "",
        "The independent CPU oracle computes each query-to-corpus squared L2 distance",
        "from Float32 inputs promoted to Float64, one query at a time, using one",
        "corpus-by-512 difference buffer. It does not retain a query-by-corpus distance",
        "matrix. The recorded process peak RSS was",
        f"{report['oracle']['peak_rss_mib']:.1f} MiB; the largest single difference buffer was "
        f"{report['oracle']['largest_difference_buffer_mib']:.1f} MiB. Oracle computation took "
        f"{report['oracle']['elapsed_seconds']:.1f} s on CPU.",
        "",
        "## Recall and distance checks",
        "",
        "All times are runner-recorded whole-statement elapsed seconds, including each",
        "statement's per-query index construction and result retrieval. Base, r2, and r3",
        "used the dedicated EMBER server at `grpc://127.0.0.1:50062`. Three captures do",
        "not form a controlled performance study. Approximate methods did not show a",
        "lower median than brute force, so no speedup claim is made.",
        "",
        "| Search function | r3 query ID | Whole SQL times (s) | Median (s) | ID recall@10 | Tie-aware recall@10 | Max distance error | Top-1 same family | Top-10 same family | Random-reference baseline | Changed neighbor slots r2/r3 vs base |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for algorithm, item in algorithms.items():
        run_times = ", ".join(f"{run['elapsed_seconds']:.3f}" for run in item["run_captures"])
        family = item["posthoc_same_known_family_rates"]
        lines.append(
            f"| `{algorithm}` | `{item['run_captures'][-1]['query_id']}` | {run_times} | "
            f"{item['median_full_statement_seconds']:.3f} | {item['strict_id_recall_at_10']:.4f} | "
            f"{item['tie_aware_recall_at_10']:.4f} | {item['max_absolute_returned_distance_error']:.3g} | "
            f"{family['1']['same_known_family_rate']:.4f} | {family['10']['same_known_family_rate']:.4f} | "
            f"{family['10']['expected_uniform_random_train_reference_rate']:.4f} | "
            f"{item['run_captures'][1]['neighbor_id_slots_changed_from_base']:,} / "
            f"{item['run_captures'][2]['neighbor_id_slots_changed_from_base']:,} |"
        )
    lines += [
        "",
        "Strict ID recall uses the exact Float64 top ten sorted by `(distance, train_id)`.",
        "Tie-aware recall accepts any returned neighbor whose exact distance is at most",
        "the exact rank-10 cutoff plus `1e-12` absolute/relative tolerance. Maximum distance",
        "error compares each r3 reported distance with a direct Float64 recomputation for",
        "that returned query/neighbor pair.",
        "",
        "The post-hoc family rates use metadata only after search. The denominator includes",
        "all top-k references for queries with a known test family; an empty neighbor family",
        "does not count as a match. The baseline is the expected same-family rate from a",
        "uniform random training reference, averaged across known-family test queries.",
        "The query does not use family or label in its search inputs.",
        "",
        "IVF-PQ's compressed distance values differ from direct Float64 squared L2.",
        "The ten largest r3 returned-pair errors are preserved below and in the JSON",
        "report. Direct distance recomputation uses the returned query/reference pairs.",
        "",
        "| Query ID | Neighbor ID | Rank | Reported distance | Direct Float64 distance | Absolute error |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pair in report["algorithms"]["cuvs_ivf_pq"]["r3_worst_distance_error_pairs"]:
        lines.append(
            f"| {pair['query_id']} | {pair['neighbor_id']} | {pair['rank']} | "
            f"{pair['reported_distance']:.9g} | {pair['float64_direct_squared_l2']:.12g} | {pair['absolute_error']:.9g} |"
        )
    lines += [
        "",
        "## Pairwise distance oracle",
        "",
        f"SQL 24 returned {report['pairwise']['rows']:,} ordered distances for the first 100 test Win32 rows "
        f"against the first 20 train Win32 rows. Its row-ordinal and ID mappings matched the ordered cohorts; "
        f"the maximum absolute distance error versus direct Float64 squared L2 was "
        f"`{report['pairwise']['max_absolute_distance_error']:.3g}`.",
        "",
        "Exact CPU distances validate the selected metric and returned IDs. Family agreement is a descriptive",
        "post-hoc measure of family-label agreement. This search evaluation uses temporal test queries.",
        "",
    ]
    (ROOT / "docs/research/ember2024/ANN.md").write_text("\n".join(lines), encoding="utf8")


def main() -> None:
    started = time.perf_counter()
    train = load_cohort("train")
    test = load_cohort("test")
    if len(test["ids"]) * K != 36_180:
        raise ValueError(f"sampled Win32 test cohort has {len(test['ids'])} rows, expected 3,618")
    if train["features"].shape[1] != DIMENSIONS or test["features"].shape[1] != DIMENSIONS:
        raise ValueError("search histogram dimension differs from 512")

    train_hashes = {row["sha256"] for row in train["metadata"].values()}
    test_hashes = {row["sha256"] for row in test["metadata"].values()}
    overlap = sorted(train_hashes.intersection(test_hashes))
    cohort = {
        "train": cohort_summary(train),
        "test": cohort_summary(test),
        "feature_input": {
            "column": "histogram",
            "dimension": DIMENSIONS,
            "source_features": "f7..f518",
            "element_type": "Float32, promoted to Float64 only for the CPU oracle",
            "search_projection": ["id", "vector"],
            "labels_or_family_in_search_input": False,
        },
        "sha256_disjoint": {
            "intersection_rows": len(overlap),
            "disjoint": len(overlap) == 0,
            "first_overlapping_hashes": overlap[:10],
        },
    }

    estimated_buffer_bytes = len(train["ids"]) * DIMENSIONS * np.dtype(np.float64).itemsize
    oracle_started = time.perf_counter()
    exact_ids, exact_cutoffs, tie_sizes, oracle_seconds = exact_neighbors(
        train["ids"], train["features"], test["features"]
    )
    del oracle_started

    algorithm_results: dict[str, Any] = {}
    output_matrices: dict[str, np.ndarray] = {}
    for algorithm, stem in ALGORITHMS.items():
        summary, ids = assess_algorithm(algorithm, stem, train, test, exact_ids, exact_cutoffs, tie_sizes)
        output_matrices[algorithm] = ids
        algorithm_results[algorithm] = summary
        del summary["r3_output_neighbor_ids"]

    pairwise = pairwise_check(train, test)
    # SQL query text supplies only ID and vector to the search functions; labels and family join afterwards.
    for algorithm, stem in ALGORITHMS.items():
        sql = (ROOT / "docs/research/ember2024/sql" / f"{stem}.sql").read_text()
        call_region = sql.split("FROM cuvs_", 1)[1].split(") n", 1)[0]
        if "SELECT id, vector" not in call_region or "label" in call_region or "family" in call_region:
            raise ValueError(f"{stem} includes post-hoc metadata in a search input")

    memory_peak = peak_rss_bytes()
    report = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset": "EMBER2024",
            "query_cohort": "one-percent SHA-256 sampled test Win32 rows",
            "corpus_cohort": "one-percent SHA-256 sampled train Win32 rows",
            "sql_files": [f"docs/research/ember2024/sql/{stem}.sql" for stem in ALGORITHMS.values()],
            "cpu_script": "fixture/demo/ember2024/scripts/research_ann_check.py",
        },
        "cohort": cohort,
        "ordering_and_identity_contracts": {
            "query_order": "ORDER BY id in the cuVS queries subquery; query_ordinal is zero-based and maps to the same sorted test ID on all runs",
            "neighbor_identity": "Every neighbor_id is present in the sampled Win32 train corpus and resolves to the matching post-hoc metadata row",
            "rank": "Each query has ranks 1 through 10 once, ordered by nondecreasing returned distance",
            "search_inputs": "Only id and histogram-as-vector enter the cuVS search functions; labels, family, SHA-256, and week are joined after search",
            "all_runs_contracts_passed": True,
        },
        "oracle": {
            "method": "direct squared L2 from (Float64(train_feature) - Float64(query_feature)) differences, with no square root",
            "source_feature_dtype": "Float32",
            "compute_dtype": "Float64",
            "queries": len(test["ids"]),
            "corpus_rows": len(train["ids"]),
            "dimensions": DIMENSIONS,
            "k": K,
            "query_batch_size": 1,
            "largest_difference_buffer_bytes": estimated_buffer_bytes,
            "largest_difference_buffer_mib": estimated_buffer_bytes / (1024 * 1024),
            "peak_rss_bytes": memory_peak,
            "peak_rss_mib": memory_peak / (1024 * 1024),
            "elapsed_seconds": oracle_seconds,
            "tie_tolerance": {
                "absolute": TIE_ABS_TOLERANCE,
                "relative": TIE_REL_TOLERANCE,
            },
            "queries_with_rank10_boundary_ties": int(np.count_nonzero(tie_sizes > 1)),
            "maximum_rank10_tie_set_size": int(tie_sizes.max(initial=0)),
        },
        "algorithms": algorithm_results,
        "pairwise": pairwise,
        "statement_timing_notes": {
            "elapsed_scope": "Each elapsed_s is the complete research.py statement capture, including planning, scans, per-call index construction where used, and fetching the complete result before export.",
            "run_contexts": {
                "base": "dedicated EMBER server at grpc://127.0.0.1:50062",
                "_r2": "dedicated EMBER server at grpc://127.0.0.1:50062",
                "_r3": "dedicated EMBER server on grpc://127.0.0.1:50062",
            },
            "performance_conclusion": "The three approximate variants did not have a lower whole-statement median than brute force in these captures. The three captures are not a controlled performance study, so no speedup claim is made.",
        },
        "cpu_runtime_seconds_total": time.perf_counter() - started,
    }
    out = REPORT_DIR / "ann_check.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf8")
    write_markdown(report)
    print(json.dumps({"report": str(out), "markdown": str(ROOT / "docs/research/ember2024/ANN.md"), "oracle_s": oracle_seconds, "max_rss_mib": report["oracle"]["peak_rss_mib"], "algorithms": {name: {"recall": value["strict_id_recall_at_10"], "tie_aware": value["tie_aware_recall_at_10"], "median_s": value["median_full_statement_seconds"]} for name, value in algorithm_results.items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
