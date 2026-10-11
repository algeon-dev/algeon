#!/usr/bin/env python3
"""Evaluate saved MerRec cuML predictions against the fixed CPU ground truth."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)

from merrec_verify_common import query_record


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "target/merrec-exploration/results"
TRAIN = ROOT / "target/merrec-exploration/supervised_train.parquet"
TEST = ROOT / "target/merrec-exploration/supervised_test.parquet"
CPU_FOREST = ROOT / "target/merrec-exploration/forest_cpu_predictions.parquet"
REPORT = ROOT / "docs/research/merrec/supervised_validation.json"
QUERY_RECORDS = ROOT / "docs/research/merrec/results"
TARGET_CLASS = "future_has_like"
TARGET_REGRESSION = "future_view_count_log1p"
FOREST_TOLERANCE = 2e-6

GPU_RESULTS = {
    "cuml_logistic_regression": "cuml_logistic_regression_ordered.parquet",
    "cuml_random_forest_classifier": "cuml_random_forest_classifier_ordered.parquet",
    "cuml_random_forest_regressor": "cuml_random_forest_regressor_ordered.parquet",
    "cuml_forest_predict": "cuml_forest_predict_ordered.parquet",
}


def query_provenance(model: str, result_filename: str) -> dict[str, object]:
    corrected_name = f"{model}_ordered"
    return {
        "initial_attempt": {
            "sql": f"docs/research/merrec/sql/probes/{model}.sql",
            **query_record(QUERY_RECORDS, ROOT, model),
        },
        "verified_attempt": {
            "sql": f"docs/research/merrec/sql/probes/{corrected_name}.sql",
            "parquet": str((RESULTS / result_filename).relative_to(ROOT)),
            **query_record(QUERY_RECORDS, ROOT, corrected_name),
        },
    }


def require_columns(table: pa.Table, names: list[str], path: Path) -> None:
    missing = [name for name in names if name not in table.column_names]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    nulls = [name for name in names if table[name].null_count]
    if nulls:
        raise ValueError(f"{path} has null values in required columns: {nulls}")


def column_array(table: pa.Table, name: str, dtype: type, path: Path) -> np.ndarray:
    try:
        values = table[name].to_numpy(zero_copy_only=False).astype(dtype, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}.{name} has an unsupported type") from error
    return values


def align_predictions(path: Path, expected_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(path)
    require_columns(table, ["id", "prediction"], path)
    ids = column_array(table, "id", np.int64, path)
    predictions = column_array(table, "prediction", np.float64, path)
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{path} contains duplicate ids")
    order = np.argsort(ids)
    if not np.array_equal(ids[order], expected_ids):
        raise ValueError(f"{path} ids do not exactly match supervised_test.parquet")
    predictions = predictions[order]
    if not np.isfinite(predictions).all():
        raise ValueError(f"{path}.prediction contains non-finite values")
    return ids[order], predictions


def classification_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    if not np.isin(predicted, [0, 1]).all():
        raise ValueError("classification predictions must contain only labels 0 and 1")
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "confusion_counts": {
            "true_negative": int(matrix[0, 0]),
            "false_positive": int(matrix[0, 1]),
            "false_negative": int(matrix[1, 0]),
            "true_positive": int(matrix[1, 1]),
        },
    }


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae_log_view": float(mean_absolute_error(actual, predicted)),
        "rmse_log_view": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def metric_deltas(
    actual: np.ndarray,
    predictions: np.ndarray,
    baselines: dict[str, np.ndarray],
    metric_function,
) -> dict[str, dict[str, float]]:
    model_metrics = metric_function(actual, predictions)
    deltas: dict[str, dict[str, float]] = {}
    for name, baseline_predictions in baselines.items():
        baseline_metrics = metric_function(actual, baseline_predictions)
        deltas[name] = {
            metric: float(model_metrics[metric] - baseline_metrics[metric])
            for metric in model_metrics
            if isinstance(model_metrics[metric], (int, float))
        }
    return deltas


def main() -> int:
    for path in (TRAIN, TEST):
        if not path.is_file():
            raise SystemExit(f"Required supervised input does not exist: {path}")

    train = pq.read_table(TRAIN)
    test = pq.read_table(TEST)
    require_columns(train, [TARGET_CLASS, TARGET_REGRESSION], TRAIN)
    require_columns(
        test,
        [
            "user_id",
            "baseline_likes",
            "view_count_log1p",
            TARGET_CLASS,
            TARGET_REGRESSION,
        ],
        TEST,
    )
    test_ids = column_array(test, "user_id", np.int64, TEST)
    if len(np.unique(test_ids)) != len(test_ids):
        raise ValueError(f"{TEST} contains duplicate user_id values")
    order = np.argsort(test_ids)
    test_ids = test_ids[order]
    actual_class = column_array(test, TARGET_CLASS, np.int32, TEST)[order]
    actual_views = column_array(test, TARGET_REGRESSION, np.float64, TEST)[order]
    prior_likes = column_array(test, "baseline_likes", np.int64, TEST)[order]
    prior_views = column_array(test, "view_count_log1p", np.float64, TEST)[order]
    train_class = column_array(train, TARGET_CLASS, np.int32, TRAIN)
    train_views = column_array(train, TARGET_REGRESSION, np.float64, TRAIN)
    if not np.isin(actual_class, [0, 1]).all() or not np.isin(train_class, [0, 1]).all():
        raise ValueError("supervised_test and supervised_train labels must be binary")
    if not np.isfinite(actual_views).all() or not np.isfinite(train_views).all():
        raise ValueError("log-view labels contain non-finite values")

    majority_label = int(np.mean(train_class) >= 0.5)
    class_baselines = {
        "training_majority_class": np.full(len(actual_class), majority_label, dtype=np.int32),
        "prior_likes": (prior_likes > 0).astype(np.int32),
    }
    classification_baselines = {
        name: {
            "definition": (
                f"Always predict training majority label {majority_label}"
                if name == "training_majority_class"
                else "Predict positive when baseline_likes > 0"
            ),
            "metrics": classification_metrics(actual_class, predictions),
        }
        for name, predictions in class_baselines.items()
    }

    training_view_mean = float(np.mean(train_views))
    regression_baselines = {
        "prior_view_count": {
            "definition": "Use feature-window view_count_log1p as the next-window prediction",
            "metrics": regression_metrics(actual_views, prior_views),
        },
        "training_label_mean": {
            "definition": "Use mean supervised_train.future_view_count_log1p for every test row",
            "prediction_log_view_mean": training_view_mean,
            "metrics": regression_metrics(
                actual_views,
                np.full(len(actual_views), training_view_mean, dtype=np.float64),
            ),
        },
    }

    model_results: dict[str, dict[str, object]] = {}
    forest_reference: dict[str, object] | None = None
    query_provenance_by_model = {
        model: query_provenance(model, filename)
        for model, filename in GPU_RESULTS.items()
    }
    if CPU_FOREST.is_file():
        cpu_forest = pq.read_table(CPU_FOREST)
        require_columns(cpu_forest, ["id", "prediction", "predicted_label", "actual_label"], CPU_FOREST)
        cpu_ids = column_array(cpu_forest, "id", np.int64, CPU_FOREST)
        cpu_order = np.argsort(cpu_ids)
        if len(np.unique(cpu_ids)) != len(cpu_ids) or not np.array_equal(cpu_ids[cpu_order], test_ids):
            raise ValueError(f"{CPU_FOREST} ids do not exactly match supervised_test.parquet")
        cpu_probabilities = column_array(cpu_forest, "prediction", np.float64, CPU_FOREST)[cpu_order]
        cpu_labels = column_array(cpu_forest, "predicted_label", np.int32, CPU_FOREST)[cpu_order]
        cpu_actual = column_array(cpu_forest, "actual_label", np.int32, CPU_FOREST)[cpu_order]
        if not np.isfinite(cpu_probabilities).all() or not np.allclose(cpu_actual, actual_class):
            raise ValueError(f"{CPU_FOREST} predictions do not match the test ground truth")
        forest_reference = {
            "path": str(CPU_FOREST.relative_to(ROOT)),
            "metrics": classification_metrics(actual_class, cpu_labels),
        }

    for model, filename in GPU_RESULTS.items():
        path = RESULTS / filename
        if not path.is_file():
            model_results[model] = {
                "status": "not_executed",
                "path": str(path.relative_to(ROOT)),
                "reason": "Saved GPU result file is absent.",
            }
            continue

        _, predictions = align_predictions(path, test_ids)
        entry: dict[str, object] = {
            "status": "evaluated",
            "path": str(path.relative_to(ROOT)),
            "rows": len(predictions),
            "query_id": query_provenance_by_model[model]["verified_attempt"]["query_id"],
            "initial_attempt_result": query_provenance_by_model[model]["initial_attempt"],
        }
        if model in {"cuml_logistic_regression", "cuml_random_forest_classifier"}:
            if not np.isin(predictions, [0.0, 1.0]).all():
                raise ValueError(
                    f"{path}.prediction must contain returned class labels 0 or 1"
                )
            class_predictions = predictions.astype(np.int32)
            entry["prediction_semantics"] = "Returned class labels"
            entry["metrics"] = classification_metrics(actual_class, class_predictions)
            entry["metric_deltas_vs_baselines"] = metric_deltas(
                actual_class,
                class_predictions,
                class_baselines,
                classification_metrics,
            )
        elif model == "cuml_random_forest_regressor":
            entry["prediction_semantics"] = "Predicted future_view_count_log1p"
            entry["metrics"] = regression_metrics(actual_views, predictions)
            entry["metric_deltas_vs_baselines"] = metric_deltas(
                actual_views,
                predictions,
                {
                    "prior_view_count": prior_views,
                    "training_label_mean": np.full(
                        len(actual_views), training_view_mean, dtype=np.float64
                    ),
                },
                regression_metrics,
            )
        else:
            class_predictions = (predictions >= 0.5).astype(np.int32)
            entry["prediction_semantics"] = "Positive-class probability from the saved forest model"
            entry["classification_threshold"] = 0.5
            entry["metrics"] = classification_metrics(actual_class, class_predictions)
            entry["metric_deltas_vs_baselines"] = metric_deltas(
                actual_class,
                class_predictions,
                class_baselines,
                classification_metrics,
            )
            if forest_reference is None:
                entry["cpu_forest_comparison"] = {
                    "status": "not_executed",
                    "reason": "CPU forest prediction reference file is absent.",
                }
            else:
                cpu_table = pq.read_table(CPU_FOREST, columns=["id", "prediction"])
                cpu_ids = column_array(cpu_table, "id", np.int64, CPU_FOREST)
                cpu_order = np.argsort(cpu_ids)
                cpu_probabilities = column_array(
                    cpu_table, "prediction", np.float64, CPU_FOREST
                )[cpu_order]
                max_error = float(np.max(np.abs(predictions - cpu_probabilities)))
                matches = max_error <= FOREST_TOLERANCE
                entry["cpu_forest_comparison"] = {
                    "status": "matched" if matches else "mismatch",
                    "reference_path": str(CPU_FOREST.relative_to(ROOT)),
                    "tolerance": FOREST_TOLERANCE,
                    "max_absolute_error": max_error,
                    "matches_cpu_predictions": matches,
                }
        model_results[model] = entry

    report = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "runtime": {
            "cpu_file_evaluation": True,
            "gpu_or_server_used": False,
            "pandas_used": False,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "evaluation": {
            "test_path": str(TEST.relative_to(ROOT)),
            "training_path": str(TRAIN.relative_to(ROOT)),
            "test_rows": len(actual_class),
            "classification_target": TARGET_CLASS,
            "regression_target": TARGET_REGRESSION,
            "classification_baselines": classification_baselines,
            "regression_baselines": regression_baselines,
            "cpu_forest_reference": forest_reference,
            "models": model_results,
            "classification_metrics": ["accuracy", "precision", "recall", "confusion_counts"],
            "classification_predictions_are_labels": True,
            "query_provenance": query_provenance_by_model,
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(REPORT.relative_to(ROOT))
    for model, result in model_results.items():
        print(f"{model}: {result['status']}")
    return int(
        any(
            result.get("cpu_forest_comparison", {}).get("status") == "mismatch"
            for result in model_results.values()
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
