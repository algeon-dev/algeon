#!/usr/bin/env python3
"""Validate EMBER2024 supervised query results against their source data."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "target/ember2024-research"
REPORT = ROOT / "docs/research/ember2024"
MODEL = ROOT / "fixture/demo/ember2024/models/EMBER2024_all.model"
THRESHOLDS = (0.5, 0.9, 0.99, 0.999)
WILSON_Z = 1.959963984540054


def array(table: pa.Table, name: str, dtype: np.dtype | type | None = None) -> np.ndarray:
    result = table.column(name).to_numpy(zero_copy_only=False)
    if dtype is not None:
        result = result.astype(dtype, copy=False)
    return result


def sorted_table(path: Path) -> pa.Table:
    table = pq.read_table(path)
    if "id" not in table.column_names:
        raise ValueError(f"{path.name} has no id column")
    for index, field in enumerate(table.schema):
        if pa.types.is_string_view(field.type):
            string_field = pa.field(field.name, pa.string(), nullable=field.nullable, metadata=field.metadata)
            table = table.set_column(index, string_field, table.column(index).cast(pa.string()))
    return table.sort_by([("id", "ascending")])


def require_equal(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if not np.array_equal(actual, expected):
        raise ValueError(f"{name} does not match the source rows")


def wilson_interval(errors: int, trials: int) -> dict[str, float] | None:
    if trials == 0:
        return None
    z2 = WILSON_Z * WILSON_Z
    rate = errors / trials
    denominator = 1 + z2 / trials
    center = (rate + z2 / (2 * trials)) / denominator
    radius = WILSON_Z * np.sqrt(rate * (1 - rate) / trials + z2 / (4 * trials * trials)) / denominator
    return {"lower": float(max(0.0, center - radius)), "upper": float(min(1.0, center + radius))}


def confusion_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int | dict | None]:
    labels = labels.astype(np.int8, copy=False)
    predictions = predictions.astype(np.int8, copy=False)
    tp = int(np.count_nonzero((labels == 1) & (predictions == 1)))
    fp = int(np.count_nonzero((labels == 0) & (predictions == 1)))
    tn = int(np.count_nonzero((labels == 0) & (predictions == 0)))
    fn = int(np.count_nonzero((labels == 1) & (predictions == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else None
    return {
        "rows": int(labels.size),
        "positive_rows": int(tp + fn),
        "negative_rows": int(tn + fp),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "accuracy": float((tp + tn) / labels.size) if labels.size else None,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "false_positive_rate": float(fpr) if fpr is not None else None,
        "false_positive_rate_wilson_95": wilson_interval(fp, fp + tn),
    }


def ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int | None]:
    positives = int(np.count_nonzero(labels == 1))
    negatives = int(np.count_nonzero(labels == 0))
    if not positives or not negatives:
        return {
            "rows": int(labels.size),
            "positive_rows": positives,
            "negative_rows": negatives,
            "roc_auc": None,
            "average_precision": None,
        }
    return {
        "rows": int(labels.size),
        "positive_rows": positives,
        "negative_rows": negatives,
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def aligned_positions(ids: np.ndarray, source_ids: np.ndarray, name: str) -> np.ndarray:
    positions = np.searchsorted(source_ids, ids)
    if positions.size and (np.any(positions >= source_ids.size) or not np.array_equal(source_ids[positions], ids)):
        raise ValueError(f"{name} contains an id absent from sampled test metadata")
    return positions


def verify_result_metadata(
    name: str,
    result: pa.Table,
    metadata: pa.Table,
    expected_ids: np.ndarray,
    columns: tuple[str, ...],
) -> None:
    ids = array(result, "id", np.int64)
    require_equal(f"{name} ids", ids, expected_ids)
    source_ids = array(metadata, "id", np.int64)
    positions = aligned_positions(ids, source_ids, name)
    for column in columns:
        actual = result.column(column).to_pylist()
        expected = metadata.column(column).take(pa.array(positions, type=pa.int64())).to_pylist()
        if actual != expected:
            raise ValueError(f"{name}.{column} differs from source metadata")


def read_source_sha_disjointness() -> dict[str, int | bool]:
    train_sha = pq.read_table(DATA / "input_metadata_train.parquet", columns=["sha256"]).column("sha256")
    test_sha = pq.read_table(DATA / "input_metadata_test.parquet", columns=["sha256"]).column("sha256")
    if train_sha.null_count or test_sha.null_count:
        raise ValueError("source metadata contains null SHA-256 values")
    overlaps = pc.is_in(train_sha, value_set=test_sha)
    overlap_rows = int(pc.sum(pc.cast(overlaps, pa.int64())).as_py() or 0)
    if overlap_rows:
        raise ValueError(f"source train and test metadata share {overlap_rows} SHA-256 values")
    return {
        "train_metadata_rows": len(train_sha),
        "test_metadata_rows": len(test_sha),
        "overlapping_sha256_rows": overlap_rows,
        "disjoint": True,
    }


def supervised_classification(result: pa.Table, group_values: np.ndarray) -> dict:
    labels = array(result, "label", np.int32)
    predictions = array(result, "prediction", np.int32)
    if not np.isin(labels, (0, 1)).all() or not np.isin(predictions, (0, 1)).all():
        raise ValueError(f"{result.column_names} has non-binary labels or predictions")
    per_week = {}
    for week in np.unique(group_values):
        mask = group_values == week
        per_week[str(int(week))] = confusion_metrics(labels[mask], predictions[mask])
    return {"overall": confusion_metrics(labels, predictions), "by_week_id": per_week}


def forest_threshold_report(result: pa.Table, labels: np.ndarray, file_types: np.ndarray) -> dict:
    scores = array(result, "prediction", np.float64)
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("15_forest_temporal_test predictions must be finite probabilities in [0, 1]")
    by_type = {}
    for file_type in np.unique(file_types):
        mask = file_types == file_type
        by_type[str(file_type)] = ranking_metrics(labels[mask], scores[mask])
    thresholds = {}
    for threshold in THRESHOLDS:
        predicted = scores >= threshold
        thresholds[str(threshold)] = {
            "threshold": threshold,
            "overall": confusion_metrics(labels, predicted),
            "by_file_type": {
                str(file_type): confusion_metrics(labels[file_types == file_type], predicted[file_types == file_type])
                for file_type in np.unique(file_types)
            },
        }
    return {
        "continuous_score_metrics": {
            "overall": ranking_metrics(labels, scores),
            "by_file_type": by_type,
        },
        "thresholds": thresholds,
    }


def main() -> int:
    started = time.monotonic()
    test_metadata_source_sha = read_source_sha_disjointness()
    vectors = sorted_table(DATA / "input_research_vectors_test.parquet")
    vector_ids = array(vectors, "id", np.int64)
    vector_labels = array(vectors, "label", np.int32)
    if vector_ids.size != np.unique(vector_ids).size:
        raise ValueError("sampled test vector ids are not unique")
    if not np.isin(vector_labels, (0, 1)).all():
        raise ValueError("sampled test vectors have non-binary labels")

    metadata_all = pq.read_table(
        DATA / "input_metadata_test.parquet",
        columns=["id", "source_id", "split", "sha256", "file_type", "week_id", "label", "family"],
    )
    selected = metadata_all.filter(pc.is_in(metadata_all.column("id"), value_set=pa.array(vector_ids)))
    metadata = selected.sort_by([("id", "ascending")])
    metadata_ids = array(metadata, "id", np.int64)
    require_equal("sampled test vector ids versus metadata", metadata_ids, vector_ids)
    require_equal("sampled test vector labels versus metadata", vector_labels, array(metadata, "label", np.int32))
    if not np.all(array(metadata, "split").astype(str) == "test"):
        raise ValueError("sampled test metadata includes rows outside the test split")
    source_ids = array(metadata, "source_id", np.int64)
    require_equal("test id offset", metadata_ids, 10_000_000 + source_ids)

    labels = array(metadata, "label", np.int32)
    weeks = array(metadata, "week_id", np.int32)
    file_types = np.asarray(metadata.column("file_type").to_pylist(), dtype=object)
    win32_ids = metadata_ids[file_types == "Win32"]
    if win32_ids.size == 0:
        raise ValueError("sampled test metadata has no Win32 rows")

    expected = {
        "13_temporal_logistic": win32_ids,
        "14_temporal_random_forest": win32_ids,
        "15_forest_temporal_test": vector_ids,
    }
    output_tables = {name: sorted_table(DATA / f"{name}.parquet") for name in expected}
    verify_result_metadata(
        "13_temporal_logistic",
        output_tables["13_temporal_logistic"],
        metadata,
        expected["13_temporal_logistic"],
        ("label", "week_id", "sha256", "family"),
    )
    verify_result_metadata(
        "14_temporal_random_forest",
        output_tables["14_temporal_random_forest"],
        metadata,
        expected["14_temporal_random_forest"],
        ("label", "week_id", "sha256", "family"),
    )
    verify_result_metadata(
        "15_forest_temporal_test",
        output_tables["15_forest_temporal_test"],
        metadata,
        expected["15_forest_temporal_test"],
        ("label", "file_type", "week_id", "sha256"),
    )

    supervised = {}
    for name in ("13_temporal_logistic", "14_temporal_random_forest"):
        result = output_tables[name]
        result_ids = array(result, "id", np.int64)
        positions = aligned_positions(result_ids, metadata_ids, name)
        result_weeks = weeks[positions]
        supervised[name] = supervised_classification(result, result_weeks)
    baseline_labels = array(output_tables["13_temporal_logistic"], "label", np.int32)
    baseline = {
        "prediction": "all_negative",
        "overall": confusion_metrics(baseline_labels, np.zeros(baseline_labels.size, dtype=np.int8)),
        "by_week_id": {},
    }
    logistic = output_tables["13_temporal_logistic"]
    logistic_weeks = array(logistic, "week_id", np.int32)
    for week in np.unique(logistic_weeks):
        mask = logistic_weeks == week
        baseline["by_week_id"][str(int(week))] = confusion_metrics(
            array(logistic, "label", np.int32)[mask], np.zeros(int(mask.sum()), dtype=np.int8)
        )

    full_labels = array(output_tables["15_forest_temporal_test"], "label", np.int32)
    full_file_types = np.asarray(output_tables["15_forest_temporal_test"].column("file_type").to_pylist(), dtype=object)
    full_forest = forest_threshold_report(output_tables["15_forest_temporal_test"], full_labels, full_file_types)

    win32_positions_in_all = np.flatnonzero(full_file_types == "Win32")
    forest_win32_ids = array(output_tables["15_forest_temporal_test"], "id", np.int64)[win32_positions_in_all]
    require_equal("Win32 supervised cohort ids", forest_win32_ids, win32_ids)
    common_cohort = {"rows": int(win32_ids.size), "ids_match_across_all_three_models": True}
    common_cohort["label_match_across_all_three_models"] = bool(
        np.array_equal(
            array(output_tables["13_temporal_logistic"], "label", np.int32),
            array(output_tables["15_forest_temporal_test"], "label", np.int32)[win32_positions_in_all],
        )
    )
    common_cohort["model_cohorts"] = {
        "13_temporal_logistic": "sampled Win32 test rows; sampled train split; 512 histogram features",
        "14_temporal_random_forest": "sampled Win32 test rows; sampled train split; 512 histogram features",
        "15_forest_temporal_test": "same sampled Win32 test rows; pretrained EMBER2024_all model; 2568 features and full training split",
    }
    if not common_cohort["label_match_across_all_three_models"]:
        raise ValueError("Win32 model labels differ on the shared sampled test cohort")

    vector_array = vectors.column("vector").combine_chunks()
    dimensions = int(vector_array.type.list_size)
    feature_matrix = vector_array.values.to_numpy(zero_copy_only=False).reshape(vector_ids.size, dimensions)
    if dimensions != 2568 or not np.isfinite(feature_matrix).all():
        raise ValueError(f"test feature matrix is invalid: shape={feature_matrix.shape}")
    booster = lgb.Booster(model_file=str(MODEL))
    if int(booster.num_feature()) != dimensions:
        raise ValueError(f"model feature count {booster.num_feature()} differs from sampled vectors {dimensions}")
    cpu_scores = np.asarray(booster.predict(feature_matrix, num_threads=8), dtype=np.float64)
    gpu_scores = array(output_tables["15_forest_temporal_test"], "prediction", np.float64)
    if cpu_scores.shape != gpu_scores.shape or not np.isfinite(cpu_scores).all():
        raise ValueError("CPU reference scores are invalid or differ in row count from the query output")
    score_error = np.abs(cpu_scores - gpu_scores)
    cpu_gpu_comparison = {
        "rows": int(cpu_scores.size),
        "atol": 1e-6,
        "rtol": 0.0,
        "max_abs_error": float(score_error.max()),
        "p50_abs_error": float(np.quantile(score_error, 0.5)),
        "p99_abs_error": float(np.quantile(score_error, 0.99)),
        "matched_within_tolerance": bool(np.allclose(cpu_scores, gpu_scores, rtol=0.0, atol=1e-6)),
    }
    if not cpu_gpu_comparison["matched_within_tolerance"]:
        raise ValueError(f"GPU forest scores differ from CPU LightGBM beyond 1e-6: {cpu_gpu_comparison}")

    report = {
        "sampled_test": {
            "rows": int(vector_ids.size),
            "win32_rows": int(win32_ids.size),
            "label_counts": {
                "negative": int(np.count_nonzero(labels == 0)),
                "positive": int(np.count_nonzero(labels == 1)),
            },
            "file_type_counts": {str(value): int(np.count_nonzero(file_types == value)) for value in np.unique(file_types)},
        },
        "source_sha256_disjointness": test_metadata_source_sha,
        "win32_models_same_test_cohort": common_cohort,
        "baseline_all_negative_win32": baseline,
        "temporal_histogram_classifiers": supervised,
        "pretrained_forest_full_test": full_forest,
        "cpu_lightgbm_score_comparison": cpu_gpu_comparison,
        "model_feature_count": dimensions,
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "lightgbm": lgb.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "runtime_seconds": time.monotonic() - started,
    }
    results_dir = REPORT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "supervised_check.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    logistic_metrics = supervised["13_temporal_logistic"]["overall"]
    rf_metrics = supervised["14_temporal_random_forest"]["overall"]
    forest_metrics = full_forest["thresholds"]["0.5"]["by_file_type"]["Win32"]
    all_forest_metrics = full_forest["thresholds"]["0.5"]["overall"]

    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{100 * value:.2f}%"

    def metric_line(name: str, metric: dict) -> str:
        matrix = metric["confusion_matrix"]
        return (
            f"| {name} | {metric['rows']:,} | {matrix['tp']:,} | {matrix['fp']:,} | "
            f"{pct(metric['accuracy'])} | {pct(metric['precision'])} | {pct(metric['recall'])} | "
            f"{pct(metric['f1'])} | {pct(metric['false_positive_rate'])} |"
        )

    win_auc = full_forest["continuous_score_metrics"]["by_file_type"].get("Win32", {})
    all_auc = full_forest["continuous_score_metrics"]["overall"]
    document = "\n".join(
        [
            "# EMBER2024 supervised results",
            "",
            f"The sampled test projection contains {vector_ids.size:,} rows, including {win32_ids.size:,} Win32 rows. The train and test source metadata contain {test_metadata_source_sha['train_metadata_rows']:,} and {test_metadata_source_sha['test_metadata_rows']:,} unique SHA-256 rows respectively, with no cross-split overlap.",
            "",
            "The 13 and 14 classifiers use the same sampled Win32 test cohort as the Win32 slice of 15. They train on sampled train rows with 512 histogram features. Query 15 applies the pretrained `EMBER2024_all.model` to 2568 features and uses the full training split. Their scores therefore describe a shared evaluation cohort with different training data and features.",
            "",
            "At threshold 0.5, the Win32 classification results are:",
            "",
            "| Model | Rows | TP | FP | Accuracy | Precision | Recall | F1 | FPR |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            metric_line("All-negative baseline", baseline["overall"]),
            metric_line("13 temporal logistic", logistic_metrics),
            metric_line("14 temporal random forest", rf_metrics),
            metric_line("15 pretrained LightGBM, Win32", forest_metrics),
            "",
            f"Across all {vector_ids.size:,} sampled test rows, query 15 at threshold 0.5 has accuracy {pct(all_forest_metrics['accuracy'])}, recall {pct(all_forest_metrics['recall'])}, and FPR {pct(all_forest_metrics['false_positive_rate'])}. Its continuous-score ROC AUC and average precision are {all_auc['roc_auc']:.4f} and {all_auc['average_precision']:.4f}; for Win32 they are {win_auc['roc_auc']:.4f} and {win_auc['average_precision']:.4f}.",
            "",
            f"CPU LightGBM predictions match query 15 within absolute tolerance 1e-6 on {cpu_scores.size:,} rows; maximum absolute error is {score_error.max():.3g}. The JSON report includes confusion metrics and Wilson 95% intervals per week for queries 13 and 14, plus per-file-type threshold metrics for query 15.",
            "",
            "The challenge split contains only positive examples. Its detection rates are reported separately and are not test accuracy or false-positive-rate estimates.",
            "",
        ]
    )
    (REPORT / "SUPERVISED.md").write_text(document)
    print(json.dumps({"rows": int(vector_ids.size), "win32_rows": int(win32_ids.size), "cpu_gpu": cpu_gpu_comparison}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
