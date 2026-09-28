#!/usr/bin/env python3
"""Build chronological MerRec supervised feature tables with the DuckDB CLI."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "fixture/demo/merrec/raw/sample/20231001_000000000000.parquet"
FEATURE_SPEC = ROOT / "docs/research/merrec/features.json"
DEFAULT_OUTPUT_DIR = ROOT / "target/merrec-exploration"
DEFAULT_PROVENANCE = ROOT / "docs/research/merrec/supervised_preparation.json"

WINDOWS = {
    "train": {
        "feature_start": "2023-10-01",
        "feature_end": "2023-10-11",
        "outcome_start": "2023-10-11",
        "outcome_end": "2023-10-21",
    },
    "test": {
        "feature_start": "2023-10-11",
        "feature_end": "2023-10-21",
        "outcome_start": "2023-10-21",
        "outcome_end": "2023-10-31",
    },
}


def sql_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def load_feature_spec(path: Path) -> tuple[list[str], list[str]]:
    spec = json.loads(path.read_text())
    categories = spec["categories"]
    columns = spec["columns"]
    if len(categories) != 17 or len(columns) != 17:
        raise ValueError("features.json must define the 17 category share columns")
    return categories, columns


def feature_query(input_path: Path, window: dict[str, str], categories: list[str], columns: list[str]) -> str:
    category_aggregates = ",\n        ".join(
        f"count(*) FILTER (WHERE event_id = 'item_view' AND c0_name = {sql_string(category)}) AS {column}_view_count"
        for category, column in zip(categories, columns, strict=True)
    )
    category_features = ",\n    ".join(
        f"CAST(sqrt(CAST(f.{column}_view_count AS DOUBLE) / f.baseline_views) AS FLOAT) AS {column}"
        for column in columns
    )
    return f"""WITH source AS (
    SELECT user_id, stime, event_id, item_id, c0_name, price
    FROM read_parquet({sql_string(input_path)})
), features AS (
    SELECT
        user_id,
        count(*) FILTER (WHERE event_id = 'item_view') AS baseline_views,
        count(*) FILTER (WHERE event_id = 'item_like') AS baseline_likes,
        count(DISTINCT item_id) FILTER (WHERE event_id = 'item_view') AS unique_viewed_items,
        avg(price) FILTER (WHERE event_id = 'item_view') AS mean_viewed_price,
        {category_aggregates}
    FROM source
    WHERE stime >= TIMESTAMP {sql_string(window['feature_start'] + ' 00:00:00')}
      AND stime < TIMESTAMP {sql_string(window['feature_end'] + ' 00:00:00')}
    GROUP BY user_id
    HAVING count(*) FILTER (WHERE event_id = 'item_view') >= 20
), outcomes AS (
    SELECT
        user_id,
        count(*) FILTER (WHERE event_id = 'item_like') AS future_likes,
        count(*) FILTER (WHERE event_id = 'item_view') AS future_views
    FROM source
    WHERE stime >= TIMESTAMP {sql_string(window['outcome_start'] + ' 00:00:00')}
      AND stime < TIMESTAMP {sql_string(window['outcome_end'] + ' 00:00:00')}
    GROUP BY user_id
)
SELECT
    f.user_id,
    f.baseline_views,
    f.baseline_likes,
    CAST(ln(1.0 + f.baseline_views) AS FLOAT) AS view_count_log1p,
    CAST(ln(1.0 + f.baseline_likes) AS FLOAT) AS item_like_count_log1p,
    {category_features},
    CAST(ln(1.0 + coalesce(f.mean_viewed_price, 0.0)) AS FLOAT) AS mean_viewed_price_log1p,
    CAST(CAST(f.unique_viewed_items AS DOUBLE) / f.baseline_views AS FLOAT) AS unique_viewed_items_per_view,
    CAST(CASE WHEN coalesce(o.future_likes, 0) >= 1 THEN 1 ELSE 0 END AS INTEGER) AS future_has_like,
    CAST(ln(1.0 + coalesce(o.future_views, 0)) AS FLOAT) AS future_view_count_log1p
FROM features f
LEFT JOIN outcomes o USING (user_id)
ORDER BY f.user_id"""


def run_json(duckdb: str, sql: str) -> list[dict[str, object]]:
    result = subprocess.run(
        [duckdb, "-json", "-c", sql],
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(result.stdout)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rewrite_required_parquet(source: Path, destination: Path) -> None:
    table = pq.read_table(source)
    if any(column.null_count for column in table.columns):
        raise ValueError(f"DuckDB produced nulls in required supervised fields: {source}")
    schema = pa.schema(
        [pa.field(field.name, field.type, nullable=False) for field in table.schema]
    )
    required_table = pa.Table.from_arrays(table.columns, schema=schema)
    pq.write_table(
        required_table,
        destination,
        compression="snappy",
        version="2.6",
        data_page_version="2.0",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--duckdb", default=shutil.which("duckdb"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.duckdb:
        raise SystemExit("DuckDB CLI not found. Install it or pass --duckdb.")
    if not args.input_parquet.is_file():
        raise SystemExit(f"Input Parquet file does not exist: {args.input_parquet}")
    categories, category_columns = load_feature_spec(FEATURE_SPEC)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.provenance.parent.mkdir(parents=True, exist_ok=True)

    generated_sql: dict[str, str] = {}
    output_paths: dict[str, Path] = {}
    for split, window in WINDOWS.items():
        query = feature_query(args.input_parquet, window, categories, category_columns)
        output_path = args.output_dir / f"supervised_{split}.parquet"
        temporary_path = output_path.with_suffix(".parquet.tmp")
        temporary_path.unlink(missing_ok=True)
        copy_sql = f"COPY ({query}) TO {sql_string(temporary_path)} (FORMAT PARQUET, COMPRESSION SNAPPY)"
        subprocess.run([args.duckdb, "-bail", "-c", copy_sql], check=True)
        required_path = output_path.with_suffix(".parquet.required.tmp")
        required_path.unlink(missing_ok=True)
        rewrite_required_parquet(temporary_path, required_path)
        temporary_path.unlink()
        required_path.replace(output_path)
        generated_sql[split] = query
        output_paths[split] = output_path

    summaries: dict[str, dict[str, object]] = {}
    for split, output_path in output_paths.items():
        summaries[split] = run_json(
            args.duckdb,
            f"""SELECT
                count(*) AS rows,
                count(DISTINCT user_id) AS distinct_users,
                count(*) FILTER (WHERE future_has_like = 1) AS positive_future_like_rows,
                count(*) FILTER (WHERE future_has_like = 0) AS negative_future_like_rows,
                sum(baseline_views) AS baseline_views_sum,
                sum(baseline_likes) AS baseline_likes_sum,
                count(*) FILTER (WHERE future_view_count_log1p > 0) AS nonzero_future_view_rows
            FROM read_parquet({sql_string(output_path)})""",
        )[0]

    source_coverage = run_json(
        args.duckdb,
        f"""SELECT
            count(*) AS source_rows,
            min(stime)::VARCHAR AS min_event_time,
            max(stime)::VARCHAR AS max_event_time,
            count(*) FILTER (WHERE event_id = 'item_view') AS item_view_rows,
            count(*) FILTER (WHERE event_id = 'item_like') AS item_like_rows,
            count(*) FILTER (WHERE event_id = 'item_view' AND price IS NULL) AS null_view_price_rows
        FROM read_parquet({sql_string(args.input_parquet)})""",
    )[0]

    feature_columns = [
        "view_count_log1p",
        "item_like_count_log1p",
        *category_columns,
        "mean_viewed_price_log1p",
        "unique_viewed_items_per_view",
    ]
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source": {
            "path": str(args.input_parquet.resolve()),
            "sha256": sha256(args.input_parquet),
            "duckdb_cli": run_json(args.duckdb, "SELECT version() AS version")[0]["version"],
            "coverage": source_coverage,
        },
        "split_policy": {
            "train": WINDOWS["train"],
            "test": WINDOWS["test"],
            "minimum_feature_window_item_views": 20,
            "users_can_appear_in_both_splits": True,
            "all_eligible_users_retained_when_outcome_is_absent": True,
        },
        "features": {
            "columns": feature_columns,
            "type": "Float32",
            "view_count_log1p": "ln(1 + item_view event count in feature window)",
            "item_like_count_log1p": "ln(1 + item_like event count in feature window)",
            "category_share_columns": {
                "source_spec": "docs/research/merrec/features.json",
                "transform": "sqrt(feature-window item_view count for c0_name / total feature-window item_view count)",
                "categories": categories,
            },
            "mean_viewed_price_log1p": "ln(1 + mean feature-window item_view price); null mean is replaced with 0 before transform",
            "price_source_rationale": "The original interaction Parquet carries price on event rows, while the prepared items catalog keeps each item's latest observed state. Event-row prices keep this feature tied to the feature window and avoid using a later listing price.",
            "unique_viewed_items_per_view": "distinct feature-window item_view item_id count / feature-window item_view event count",
            "metadata_columns_excluded_from_model_features": ["user_id", "baseline_views", "baseline_likes"],
        },
        "labels": {
            "future_has_like": {
                "type": "Int32",
                "definition": "1 when next-window item_like count is at least one, otherwise 0, including users without outcome rows",
            },
            "future_view_count_log1p": {
                "type": "Float32",
                "definition": "ln(1 + next-window item_view event count), with an absent outcome treated as zero views",
            },
        },
        "outputs": {
            split: {"path": str(path.resolve()), "summary": summaries[split]}
            for split, path in output_paths.items()
        },
        "source_sql": generated_sql,
        "limitations": [
            "This split evaluates next-window engagement on one captured October shard; it does not measure purchases.",
            "The test outcome window is October 21 through October 30 in the source capture, with October 31 excluded.",
        ],
    }
    args.provenance.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"outputs": summaries, "source_coverage": source_coverage, "provenance": str(args.provenance)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
