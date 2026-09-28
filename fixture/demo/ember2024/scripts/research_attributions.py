#!/usr/bin/env python3
"""Aggregate LightGBM challenge SHAP values by thrember feature block."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from thrember import PEFeatureExtractor


ROOT = Path(__file__).resolve().parents[4]
FIXTURE = ROOT / "fixture/demo/ember2024"
DATA = ROOT / "target/ember2024-research"
REPORT = ROOT / "docs/research/ember2024/results/attributions.json"
ID_BASE = 20_000_000
EXPECTED_DIMENSION = 2568
ADDITIVITY_ATOL = 1e-7
SIGMOID_ATOL = 1e-10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    nonnegative = values >= 0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exp_values = np.exp(values[~nonnegative])
    result[~nonnegative] = exp_values / (1.0 + exp_values)
    return result


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def feature_groups(extractor: PEFeatureExtractor, dimension: int) -> list[dict[str, int | str]]:
    groups = []
    start = 0
    for group_index, block in enumerate(extractor.features):
        count = int(block.dim)
        groups.append(
            {
                "group_index": group_index,
                "group_name": str(block.name),
                "start_feature": start,
                "feature_count": count,
            }
        )
        start += count
    if start != EXPECTED_DIMENSION or start != dimension:
        raise ValueError(f"thrember feature blocks total {start}, model dimension is {dimension}")
    return groups


def main() -> int:
    model_path = FIXTURE / "models/EMBER2024_all.model"
    x_path = FIXTURE / "raw/X_challenge.dat"
    y_path = FIXTURE / "raw/y_challenge.dat"
    metadata_path = DATA / "input_metadata_challenge.parquet"

    booster = lgb.Booster(model_file=str(model_path))
    dimension = int(booster.num_feature())
    y_bytes = y_path.stat().st_size
    if y_bytes % np.dtype(np.int32).itemsize:
        raise ValueError(f"{y_path} size is not aligned to int32 labels")
    rows = y_bytes // np.dtype(np.int32).itemsize
    if x_path.stat().st_size != rows * dimension * np.dtype(np.float32).itemsize:
        raise ValueError(f"{x_path} byte size does not match {rows} rows x {dimension} model features")

    labels = np.memmap(y_path, dtype=np.int32, mode="r", shape=(rows,))
    if not np.all(labels == 1):
        raise ValueError("challenge split labels must all be 1")
    features = np.memmap(x_path, dtype=np.float32, mode="r", shape=(rows, dimension))
    groups = feature_groups(PEFeatureExtractor(), dimension)

    predict_started = time.perf_counter()
    contributions_with_bias = np.asarray(
        booster.predict(features, raw_score=True, pred_contrib=True, num_threads=8), dtype=np.float64
    )
    contribution_runtime_s = time.perf_counter() - predict_started
    if contributions_with_bias.shape != (rows, dimension + 1):
        raise ValueError(f"pred_contrib returned unexpected shape {contributions_with_bias.shape}")
    if not np.isfinite(contributions_with_bias).all():
        raise ValueError("LightGBM contributions contain non-finite values")

    margin_started = time.perf_counter()
    raw_margin = np.asarray(booster.predict(features, raw_score=True, num_threads=8), dtype=np.float64)
    cpu_scores = np.asarray(booster.predict(features, num_threads=8), dtype=np.float64)
    score_runtime_s = time.perf_counter() - margin_started
    if raw_margin.shape != (rows,) or cpu_scores.shape != (rows,):
        raise ValueError("LightGBM returned a margin or score array with an unexpected shape")
    if not np.isfinite(raw_margin).all() or not np.isfinite(cpu_scores).all():
        raise ValueError("LightGBM margins or scores contain non-finite values")

    reconstructed_margin = contributions_with_bias.sum(axis=1, dtype=np.float64)
    additivity_error = np.abs(reconstructed_margin - raw_margin)
    sigmoid_error = np.abs(sigmoid(raw_margin) - cpu_scores)
    max_additivity_error = float(additivity_error.max())
    max_sigmoid_error = float(sigmoid_error.max())
    if max_additivity_error > ADDITIVITY_ATOL:
        raise ValueError(f"SHAP contributions do not reconstruct raw margin: max error {max_additivity_error}")
    if max_sigmoid_error > SIGMOID_ATOL:
        raise ValueError(f"sigmoid(raw margin) differs from CPU scores: max error {max_sigmoid_error}")

    feature_contributions = contributions_with_bias[:, :dimension]
    grouped64 = np.empty((rows, len(groups)), dtype=np.float64)
    for group in groups:
        start = int(group["start_feature"])
        stop = start + int(group["feature_count"])
        grouped64[:, int(group["group_index"])] = feature_contributions[:, start:stop].sum(
            axis=1, dtype=np.float64
        )
    grouped32 = grouped64.astype(np.float32)
    if not np.isfinite(grouped64).all() or not np.isfinite(grouped32).all():
        raise ValueError("grouped feature contributions contain non-finite values")

    feature_sum = feature_contributions.sum(axis=1, dtype=np.float64)
    grouped_sum = grouped64.sum(axis=1, dtype=np.float64)
    grouped_margin = grouped32.astype(np.float64).sum(axis=1, dtype=np.float64) + contributions_with_bias[:, -1]
    grouped_margin_error = np.abs(grouped_margin - raw_margin)
    grouped_reduction_error = np.abs(grouped_sum - feature_sum)
    half_ulp_sum = np.abs(np.spacing(grouped32)).astype(np.float64).sum(axis=1) / 2.0
    float64_sum_bound = (
        np.finfo(np.float64).eps
        * 32.0
        * (np.abs(contributions_with_bias[:, -1]) + np.abs(grouped32.astype(np.float64)).sum(axis=1))
    )
    grouped_rounding_bound = additivity_error + grouped_reduction_error + half_ulp_sum + float64_sum_bound
    if np.any(grouped_margin_error > grouped_rounding_bound):
        raise ValueError("Float32 grouped contributions exceed the recorded additive rounding bound")

    group_count = len(groups)
    vector_type = pa.list_(pa.field("element", pa.float32(), nullable=False), group_count)
    ids = np.arange(rows, dtype=np.int64) + ID_BASE
    vector = pa.FixedSizeListArray.from_arrays(
        pa.array(grouped32.reshape(-1), type=pa.float32()), type=vector_type
    )
    attribution_schema = pa.schema(
        [pa.field("id", pa.int64(), nullable=False), pa.field("vector", vector_type, nullable=False)]
    )
    attribution_table = pa.Table.from_arrays([pa.array(ids), vector], schema=attribution_schema)

    group_schema = pa.schema(
        [
            pa.field("group_index", pa.int32(), nullable=False),
            pa.field("group_name", pa.string(), nullable=False),
            pa.field("start_feature", pa.int32(), nullable=False),
            pa.field("feature_count", pa.int32(), nullable=False),
        ]
    )
    group_table = pa.Table.from_pylist(groups, schema=group_schema)

    DATA.mkdir(parents=True, exist_ok=True)
    pq.write_table(attribution_table, DATA / "input_attributions.parquet", compression="snappy")
    pq.write_table(group_table, DATA / "input_feature_groups.parquet", compression="snappy")

    report = {
        "model_sha256": sha256(model_path),
        "source_sha256": {
            "X_challenge.dat": sha256(x_path),
            "y_challenge.dat": sha256(y_path),
            "input_metadata_challenge.parquet": sha256(metadata_path),
        },
        "rows": rows,
        "dimension": dimension,
        "model_feature_count": dimension,
        "group_mapping": groups,
        "contribution_units": "signed SHAP contributions in raw-margin (log-odds) units; bias excluded",
        "interpretation": "These model attributions do not establish causal risk effects.",
        "negative_only_rows": int(np.count_nonzero(np.all(grouped32 <= 0, axis=1) & np.any(grouped32 < 0, axis=1))),
        "validation_error_maxima": {
            "shap_additivity_to_raw_margin": max_additivity_error,
            "sigmoid_margin_to_cpu_score": max_sigmoid_error,
            "float32_group_sum_to_raw_margin": float(grouped_margin_error.max()),
            "float32_group_sum_rounding_bound": float(grouped_rounding_bound.max()),
        },
        "validation_tolerances": {
            "shap_additivity_atol": ADDITIVITY_ATOL,
            "sigmoid_margin_atol": SIGMOID_ATOL,
            "float32_group_sum": "per-row bound from SHAP additivity error, Float64 group reduction, half-ULP Float32 rounding, and Float64 summation",
        },
        "runtime_seconds": {
            "pred_contrib": contribution_runtime_s,
            "raw_margin_and_score": score_runtime_s,
            "total_prediction": contribution_runtime_s + score_runtime_s,
        },
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "lightgbm": lgb.__version__,
            "thrember": package_version("thrember"),
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": rows, "groups": group_count, "validation_error_maxima": report["validation_error_maxima"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
