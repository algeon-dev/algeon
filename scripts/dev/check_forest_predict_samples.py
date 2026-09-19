#!/usr/bin/env python3
"""Compare `forest_predict` bench samples against the CPU source predictor and GTIL.

The `forest_predict` bench binary writes `cases/<case>.json` and
`samples/<case>.csv` (`id,prediction`) per case into its output directory.
This script rebuilds the Float32 feature rows for every sampled `id`, scores
them with the source predictor (LightGBM or XGBoost) and Treelite GTIL at a
fixed thread count, and writes `correctness.tsv` next to the samples.

Feature reconstruction mirrors the bench SQL:
- EMBER2024: `id` is the row ordinal of the split; `prepare.py` exported the
  Parquet from `X_<split>.dat`, so the memmap row equals the Parquet row.
- lineitem: `id = l_orderkey * 8 + l_linenumber`; each feature is cast to
  Float32 (decimals through Float64, as DataFusion's decimal cast does).

Run with the stage 0 venv:
    ~/.venvs/forest/bin/python scripts/dev/check_forest_predict_samples.py \
        --bench-dir docs/bench/forest_predict_20260925_stage4
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import treelite
import treelite.gtil
import xgboost as xgb

LINEITEM_FEATURES = ["l_orderkey", "l_partkey", "l_suppkey", "l_linenumber", "l_quantity", "l_discount", "l_tax"]


def read_sample(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ids, predictions = [], []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            ids.append(int(row["id"]))
            predictions.append(float(row["prediction"]))
    return np.asarray(ids, dtype=np.int64), np.asarray(predictions, dtype=np.float32)


def ember_features(ember_dir: Path, split: str, feature_count: int, ids: np.ndarray) -> np.ndarray:
    matrix = np.memmap(ember_dir / "raw" / f"X_{split}.dat", dtype=np.float32, mode="r").reshape(-1, feature_count)
    return np.asarray(matrix[ids], dtype=np.float32)


def lineitem_features(tpch_root: Path, ids: np.ndarray) -> np.ndarray:
    table = pq.read_table(tpch_root / "lineitem.parquet", columns=LINEITEM_FEATURES)
    row_ids = pc.add(pc.multiply(table["l_orderkey"], 8), pc.cast(table["l_linenumber"], pa.int64()))
    table = table.filter(pc.is_in(row_ids, value_set=pa.array(ids)))
    row_ids = pc.add(pc.multiply(table["l_orderkey"], 8), pc.cast(table["l_linenumber"], pa.int64())).to_numpy()
    columns = []
    for name in LINEITEM_FEATURES:
        column = table[name]
        if pa.types.is_decimal(column.type):
            column = pc.cast(column, pa.float64())
        columns.append(column.to_numpy().astype(np.float32))
    features = np.column_stack(columns)
    order = {row_id: index for index, row_id in enumerate(row_ids)}
    missing = [row_id for row_id in ids if row_id not in order]
    if missing:
        raise SystemExit(f"lineitem ids not found: {missing[:5]}")
    return features[[order[row_id] for row_id in ids]]


def source_predict(model_path: Path, model_format: str, features: np.ndarray, threads: int) -> np.ndarray:
    if model_format == "lightgbm":
        return np.asarray(lgb.Booster(model_file=str(model_path)).predict(features, num_threads=threads))
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    booster.set_param({"nthread": threads})
    return np.asarray(booster.predict(xgb.DMatrix(features)))


def gtil_predict(model_path: Path, model_format: str, features: np.ndarray, threads: int) -> np.ndarray:
    if model_format == "lightgbm":
        model = treelite.frontend.load_lightgbm_model(str(model_path))
    else:
        model = treelite.frontend.load_xgboost_model(str(model_path))
    return np.asarray(treelite.gtil.predict(model, features, nthread=threads)).reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench-dir", type=Path, required=True)
    parser.add_argument("--ember-dir", type=Path, default=Path("fixture/demo/ember2024"))
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    rows = []
    for path in sorted((args.bench_dir / "cases").glob("*.json")):
        document = json.loads(path.read_text())
        case = document["case"]
        if case["failure"] is not None:
            print(f"skipping {case['case']}: {case['failure']}", flush=True)
            continue
        ids, predictions = read_sample(args.bench_dir / case["sample_file"])
        input_relation = case["input"]
        if input_relation.startswith("ember2024_"):
            features = ember_features(args.ember_dir, input_relation.removeprefix("ember2024_"), case["features"], ids)
        else:
            features = lineitem_features(Path(document["dataset"]["tpch_root"]), ids)
        model_path = Path(case["model_path"])
        source = source_predict(model_path, case["model_format"], features, args.threads).reshape(-1)
        gtil = gtil_predict(model_path, case["model_format"], features, args.threads)
        algeon = predictions.astype(np.float64)
        scale = np.maximum(np.abs(source), 1.0)
        rows.append(
            {
                "case": case["case"],
                "sampled_rows": len(ids),
                "max_abs_vs_source": float(np.max(np.abs(algeon - source))),
                "max_abs_vs_gtil": float(np.max(np.abs(algeon - gtil))),
                "max_rel_vs_source": float(np.max(np.abs(algeon - source) / scale)),
                "mean_abs_vs_source": float(np.mean(np.abs(algeon - source))),
                "max_abs_gtil_vs_source": float(np.max(np.abs(gtil - source))),
            }
        )
        print(rows[-1], flush=True)

    columns = list(rows[0].keys())
    with (args.bench_dir / "correctness.tsv").open("w") as out:
        out.write("\t".join(columns) + "\n")
        for row in rows:
            out.write("\t".join(f"{row[c]:.3e}" if isinstance(row[c], float) else str(row[c]) for c in columns) + "\n")


if __name__ == "__main__":
    main()
