#!/usr/bin/env python3
"""Train and evaluate the fixed MerRec temporal engagement experiment on CPU."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
import xgboost as xgb
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "target/merrec-exploration"
PREPARATION = ROOT / "docs/research/merrec/supervised_preparation.json"
TRAIN = RESULTS / "supervised_train.parquet"
TEST = RESULTS / "supervised_test.parquet"
MODEL = ROOT / "docs/research/merrec/models/engagement.json"
PREDICTIONS = RESULTS / "forest_cpu_predictions.parquet"
PROVENANCE = ROOT / "docs/research/merrec/forest_model.json"
SQL_PROBE = ROOT / "docs/research/merrec/sql/probes/cuml_forest_predict.sql"
SQL_PROBE_ORDERED = ROOT / "docs/research/merrec/sql/probes/cuml_forest_predict_ordered.sql"
LABEL = "future_has_like"
PARAMETERS = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "random_state": 42,
    "n_jobs": 4,
    "tree_method": "hist",
    "device": "cpu",
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "verbosity": 0,
}
EXPORTED_SCORE_TOLERANCE = 1e-7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_split(path: Path, features: list[str]) -> tuple[pa.Table, np.ndarray, np.ndarray]:
    table = pq.read_table(path)
    required = ["user_id", *features, LABEL]
    missing = [name for name in required if name not in table.column_names]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    nulls = [name for name in required if table[name].null_count]
    if nulls:
        raise ValueError(f"{path} has null values in required columns: {nulls}")
    wrong_types = {
        name: str(table.schema.field(name).type)
        for name in features
        if not pa.types.is_float32(table.schema.field(name).type)
    }
    if wrong_types:
        raise ValueError(f"model features must be Float32: {wrong_types}")
    table = table.sort_by([("user_id", "ascending")])
    matrix = np.column_stack(
        [table[name].to_numpy(zero_copy_only=False) for name in features]
    ).astype(np.float32, copy=False)
    labels = table[LABEL].to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
    if not np.isfinite(matrix).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{path} contains non-finite features or a non-binary label")
    return table, matrix, labels


def label_counts(labels: np.ndarray) -> dict[str, int]:
    return {
        "negative": int(np.sum(labels == 0)),
        "positive": int(np.sum(labels == 1)),
    }


def metrics(labels: np.ndarray, scores: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    precision_curve, recall_curve, _ = precision_recall_curve(labels, scores)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "pr_auc_average_precision": float(average_precision_score(labels, scores)),
        "pr_auc_trapezoidal": float(auc(recall_curve, precision_curve)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "confusion_counts": {
            "true_positive": int(np.sum((labels == 1) & (predictions == 1))),
            "true_negative": int(np.sum((labels == 0) & (predictions == 0))),
            "false_positive": int(np.sum((labels == 0) & (predictions == 1))),
            "false_negative": int(np.sum((labels == 1) & (predictions == 0))),
        },
    }


def forest_predict_sql(select_features: str) -> str:
    return f"""-- Register supervised_test.parquet as supervised_test.
-- Model training and CPU predictions are produced by scripts/dev/merrec_forest_prepare.py.
SELECT row_ordinal, id, prediction
FROM cuml_forest_predict(
    input => (
        SELECT
            user_id AS id,
            {select_features}
        FROM supervised_test
        ORDER BY id
    ),
    model => 'engagement.json',
    model_format => 'xgboost_json'
)
ORDER BY row_ordinal;
"""


def main() -> int:
    preparation = json.loads(PREPARATION.read_text())
    features = list(preparation["features"]["columns"])
    if len(features) != 21 or len(set(features)) != 21:
        raise ValueError("supervised_preparation.json must list 21 unique ordered features")

    train_table, train_x, train_y = read_split(TRAIN, features)
    test_table, test_x, test_y = read_split(TEST, features)
    train_ids = train_table["user_id"].to_numpy(zero_copy_only=False).astype(np.int64)
    test_ids = test_table["user_id"].to_numpy(zero_copy_only=False).astype(np.int64)
    if len(train_y) != 1232 or len(test_y) != 1242:
        raise ValueError(f"unexpected temporal split sizes: train={len(train_y)}, test={len(test_y)}")
    if label_counts(train_y) != {"negative": 361, "positive": 871}:
        raise ValueError(f"unexpected training label counts: {label_counts(train_y)}")
    if label_counts(test_y) != {"negative": 359, "positive": 883}:
        raise ValueError(f"unexpected test label counts: {label_counts(test_y)}")
    if len(np.unique(train_ids)) != len(train_ids) or len(np.unique(test_ids)) != len(test_ids):
        raise ValueError("each temporal split must contain one row per user_id")

    classifier = XGBClassifier(**PARAMETERS)
    classifier.fit(train_x, train_y)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    classifier.save_model(MODEL)

    classifier_scores = classifier.predict_proba(test_x)[:, 1]
    exported_model = xgb.Booster()
    exported_model.load_model(MODEL)
    exported_model.set_param({"nthread": 4, "device": "cpu"})
    scores = np.asarray(exported_model.predict(xgb.DMatrix(test_x)), dtype=np.float32)
    export_error = float(np.max(np.abs(classifier_scores - scores)))
    predicted = (scores >= 0.5).astype(np.int32)

    train_prevalence = float(np.mean(train_y))
    majority_class = int(train_prevalence >= 0.5)
    majority_predictions = np.full(len(test_y), majority_class, dtype=np.int32)
    majority_scores = np.full(len(test_y), train_prevalence, dtype=np.float64)
    prior_likes = test_table["baseline_likes"].to_numpy(zero_copy_only=False).astype(np.int64)
    prior_predictions = (prior_likes > 0).astype(np.int32)
    prior_scores = prior_likes.astype(np.float64)

    xgboost_metrics = metrics(test_y, scores, predicted)
    majority_metrics = metrics(test_y, majority_scores, majority_predictions)
    majority_metrics["constant_ranking_score"] = train_prevalence
    prior_metrics = metrics(test_y, prior_scores, prior_predictions)
    metric_names = (
        "accuracy", "precision", "recall", "pr_auc_average_precision",
        "pr_auc_trapezoidal", "roc_auc",
    )
    comparison = {
        "xgboost": xgboost_metrics,
        "majority_class": {
            "definition": f"always predict training majority class {majority_class}",
            **majority_metrics,
        },
        "prior_likes": {
            "definition": "predict positive when baseline_likes > 0; rank by baseline_likes count",
            **prior_metrics,
        },
        "xgboost_delta_vs_majority_class": {
            name: float(xgboost_metrics[name] - majority_metrics[name]) for name in metric_names
        },
        "xgboost_delta_vs_prior_likes": {
            name: float(xgboost_metrics[name] - prior_metrics[name]) for name in metric_names
        },
    }

    prediction_table = pa.table(
        {
            "row_ordinal": pa.array(np.arange(len(test_ids), dtype=np.int64)),
            "id": pa.array(test_ids, type=pa.int64()),
            "prediction": pa.array(scores, type=pa.float32()),
            "predicted_label": pa.array(predicted, type=pa.int32()),
            "actual_label": pa.array(test_y, type=pa.int32()),
        }
    )
    PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(prediction_table, PREDICTIONS, compression="snappy", version="2.6")

    select_features = ",\n            ".join(features)
    SQL_PROBE.parent.mkdir(parents=True, exist_ok=True)
    SQL_PROBE_ORDERED.write_text(forest_predict_sql(select_features))

    model_matches = export_error <= EXPORTED_SCORE_TOLERANCE
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": "predict at least one like in the next chronological window",
        "target": {
            "column": LABEL,
            "positive_class": 1,
            "definition": preparation["labels"][LABEL]["definition"],
        },
        "model": {
            "estimator": "xgboost.XGBClassifier",
            "format": "xgboost_json",
            "path": str(MODEL.relative_to(ROOT)),
            "parameters": PARAMETERS,
            "feature_count": len(features),
            "feature_order": features,
            "boosted_rounds": int(classifier.get_booster().num_boosted_rounds()),
            "prediction_is_positive_class_probability": True,
            "classification_threshold": 0.5,
            "saved_json_prediction_max_absolute_error_vs_classifier": export_error,
            "saved_json_prediction_tolerance": EXPORTED_SCORE_TOLERANCE,
            "saved_json_predictions_match_classifier": model_matches,
        },
        "split": {
            "policy": preparation["split_policy"],
            "train": {
                "path": str(TRAIN.relative_to(ROOT)),
                "rows": len(train_y),
                "distinct_users": len(np.unique(train_ids)),
                "label_counts": label_counts(train_y),
                "positive_prevalence": train_prevalence,
            },
            "test": {
                "path": str(TEST.relative_to(ROOT)),
                "rows": len(test_y),
                "distinct_users": len(np.unique(test_ids)),
                "label_counts": label_counts(test_y),
                "positive_prevalence": float(np.mean(test_y)),
            },
            "user_ids_shared_between_splits": len(set(train_ids.tolist()) & set(test_ids.tolist())),
            "test_used_for_fitting_or_parameter_selection": False,
        },
        "test_metrics": comparison,
        "metric_definitions": {
            "accuracy_precision_recall": "XGBoost probability threshold 0.5; each baseline uses its stated class rule",
            "pr_auc_average_precision": "scikit-learn average_precision_score",
            "pr_auc_trapezoidal": "trapezoidal area under precision_recall_curve",
            "roc_auc": "scikit-learn roc_auc_score over each model's ranking score",
        },
        "cpu_predictions": {
            "path": str(PREDICTIONS.relative_to(ROOT)),
            "rows": prediction_table.num_rows,
            "columns": prediction_table.column_names,
            "prediction_column": "positive-class probability from the reloaded JSON model",
        },
        "sql_probe": {
            "path": str(SQL_PROBE_ORDERED.relative_to(ROOT)),
            "validation_failure_path": str(SQL_PROBE.relative_to(ROOT)),
            "model_reference": "engagement.json under the configured models root",
            "feature_order": features,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "xgboost": xgb.__version__,
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "training_device": "cpu",
            "inference_device": "cpu",
            "pandas_used": False,
            "gpu_or_server_used": False,
        },
        "sha256": {
            "inputs": {
                str(path.relative_to(ROOT)): sha256(path)
                for path in (TRAIN, TEST, PREPARATION, Path(__file__).resolve())
            },
            "outputs": {
                str(path.relative_to(ROOT)): sha256(path)
                for path in (MODEL, PREDICTIONS, SQL_PROBE_ORDERED)
            },
        },
        "limitations": [
            "The evaluation covers one captured October temporal split and predicts future likes, not purchases.",
            "The prior-likes baseline uses baseline_likes from the test feature window; baseline_likes is excluded from model features.",
        ],
    }
    PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"test_metrics": comparison, "saved_model_prediction_max_error": export_error, "provenance": str(PROVENANCE.relative_to(ROOT))}, indent=2, sort_keys=True))
    return 0 if model_matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
