#!/usr/bin/env python3
"""Prepare the EMBER2024 forest-inference fixture.

Downloads the 14 pretrained LightGBM benchmark models and the raw weekly
JSONL archives via `thrember`, vectorizes them into the `X_*.dat` / `y_*.dat`
memmaps `thrember` uses natively, then re-exports each requested split as
Parquet with columns `id` (row ordinal within the split), `f0..f{d-1}`
(feature vector, d taken from the vectorized files), and `label` (EMBER's
raw int32 label).

Run with the venv described in fixture/ember2024/README.md:
    ~/.venvs/forest/bin/python fixture/ember2024/prepare.py --split test
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SPLITS = ("train", "test", "challenge")
EXPECTED_RAW_JSONL_COUNT = 448
EXPECTED_MODEL_COUNT = 14
DEFAULT_ROW_GROUP_ROWS = 100_000


def raw_jsonl_present(root: Path) -> bool:
    return sum(1 for _ in root.glob("*.jsonl")) >= EXPECTED_RAW_JSONL_COUNT


def models_present(models_dir: Path) -> bool:
    return models_dir.is_dir() and sum(1 for _ in models_dir.glob("*.model")) >= EXPECTED_MODEL_COUNT


def vectorized_present(root: Path) -> bool:
    return all(
        (root / f"{prefix}_{split}.dat").is_file() and (root / f"{prefix}_{split}.dat").stat().st_size > 0
        for prefix in ("X", "y")
        for split in SPLITS
    )


def run_in_dir(fn, *args) -> None:
    """thrember's download_* helpers os.chdir() into their target directory; restore cwd after."""
    cwd = os.getcwd()
    try:
        fn(*args)
    finally:
        os.chdir(cwd)


def ensure_models(root: Path, force: bool) -> None:
    models_dir = root / "models"
    if not force and models_present(models_dir):
        print(f"models: {sum(1 for _ in models_dir.glob('*.model'))} .model files present, skipping download")
        return
    import thrember

    models_dir.mkdir(parents=True, exist_ok=True)
    print("models: downloading 14 pretrained benchmark models via thrember.download_models")
    run_in_dir(thrember.download_models, str(models_dir))


def ensure_raw_data(root: Path, force: bool) -> None:
    if not force and raw_jsonl_present(root):
        print(f"raw data: {sum(1 for _ in root.glob('*.jsonl'))} weekly JSONL files present, skipping download")
        return
    import thrember

    root.mkdir(parents=True, exist_ok=True)
    print("raw data: downloading full EMBER2024 dataset (train + test + challenge, ~24.3 GB compressed)")
    run_in_dir(thrember.download_dataset, str(root), "all", "all")


def ensure_vectorized(root: Path, force: bool) -> None:
    if not force and vectorized_present(root):
        print("vectorized features: X_*.dat / y_*.dat present for all splits, skipping vectorization")
        return
    import thrember

    print("vectorizing raw features into X_*.dat / y_*.dat (thrember.create_vectorized_features)")
    thrember.create_vectorized_features(str(root))


def split_shape(root: Path, split: str) -> tuple[int, int]:
    """Derive (rows, feature_dim) from file sizes rather than hardcoding d,
    so a thrember feature-version bump is caught by the byte-alignment
    assertion instead of silently misreading the memmap."""
    x_path = root / f"X_{split}.dat"
    y_path = root / f"y_{split}.dat"
    if not x_path.is_file() or not y_path.is_file():
        raise FileNotFoundError(f"missing vectorized files for split {split!r}: {x_path}, {y_path}")

    y_bytes = y_path.stat().st_size
    if y_bytes % 4 != 0:
        raise ValueError(f"{y_path} size {y_bytes} is not a multiple of int32 width")
    rows = y_bytes // 4

    x_bytes = x_path.stat().st_size
    if rows == 0 or x_bytes % (rows * 4) != 0:
        raise ValueError(f"{x_path} size {x_bytes} does not divide evenly into {rows} rows of float32 features")
    ndim = x_bytes // 4 // rows
    return rows, ndim


def write_split_parquet(
    root: Path,
    out_dir: Path,
    split: str,
    *,
    row_limit: int | None,
    row_group_rows: int,
    force: bool,
) -> None:
    out_path = out_dir / f"{split}.parquet"
    if not force and out_path.is_file():
        print(f"{split}: {out_path} already present, skipping (use --force to regenerate)")
        return

    rows_total, ndim = split_shape(root, split)
    rows = rows_total if row_limit is None else min(rows_total, row_limit)

    x = np.memmap(root / f"X_{split}.dat", dtype=np.float32, mode="r", shape=(rows_total, ndim))
    y = np.memmap(root / f"y_{split}.dat", dtype=np.int32, mode="r", shape=(rows_total,))

    feature_names = [f"f{i}" for i in range(ndim)]
    schema = pa.schema(
        [pa.field("id", pa.int64(), nullable=False)]
        + [pa.field(name, pa.float32(), nullable=False) for name in feature_names]
        + [pa.field("label", pa.int32(), nullable=False)]
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".parquet.tmp")

    nonfinite_total = 0
    t0 = time.time()
    with pq.ParquetWriter(tmp_path, schema, compression="snappy") as writer:
        for start in range(0, rows, row_group_rows):
            end = min(start + row_group_rows, rows)
            x_block = np.asarray(x[start:end])
            y_block = np.asarray(y[start:end])

            nonfinite = int((~np.isfinite(x_block)).sum())
            nonfinite_total += nonfinite

            columns = [pa.array(np.arange(start, end, dtype=np.int64))]
            columns.extend(pa.array(x_block[:, i], type=pa.float32()) for i in range(ndim))
            columns.append(pa.array(y_block, type=pa.int32()))
            writer.write_table(pa.Table.from_arrays(columns, schema=schema))
            print(f"{split}: wrote rows [{start}, {end}) of {rows}", flush=True)

    if nonfinite_total > 0:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(
            f"{split}: found {nonfinite_total} non-finite feature values (NaN or Inf) across {rows} rows; "
            "the cuML forest predict SQL function requires finite input (see docs/plan/"
            "2026-09-24-nvforest-forest-inference-evaluation-and-plan.md D3). Refusing to write "
            f"{out_path}."
        )

    tmp_path.replace(out_path)
    elapsed = time.time() - t0
    size_bytes = out_path.stat().st_size
    print(
        f"{split}: wrote {rows} rows ({rows_total} available) x {ndim} features to {out_path} "
        f"({size_bytes / (1024**3):.2f} GiB) in {elapsed:.1f}s, 0 non-finite values"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--split",
        action="append",
        choices=SPLITS,
        help="Split to export as Parquet; repeat for multiple splits",
    )
    parser.add_argument("--rows", type=int, help="Cap the number of rows written per split (first N rows)")
    parser.add_argument("--force", action="store_true", help="Redownload / revectorize even if outputs exist")
    parser.add_argument(
        "--row-group-rows",
        type=int,
        default=DEFAULT_ROW_GROUP_ROWS,
        help=f"Rows per Parquet row group / memmap chunk (default: {DEFAULT_ROW_GROUP_ROWS})",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Fixture root containing raw JSONL, models/, and X_*.dat / y_*.dat (default: this directory)",
    )
    args = parser.parse_args()

    if not args.split:
        parser.error("--split is required; choose one or more of: " + ", ".join(SPLITS))
    if args.rows is not None and args.rows <= 0:
        parser.error("--rows must be positive")
    if args.row_group_rows <= 0:
        parser.error("--row-group-rows must be positive")

    root: Path = args.root
    out_dir = root / "parquet"

    ensure_models(root, args.force)
    ensure_raw_data(root, args.force)
    ensure_vectorized(root, args.force)

    for split in args.split:
        write_split_parquet(
            root,
            out_dir,
            split,
            row_limit=args.rows,
            row_group_rows=args.row_group_rows,
            force=args.force,
        )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
