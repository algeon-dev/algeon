#!/usr/bin/env python3
"""Convert one Amazon Reviews 2023 category from JSONL into GPU-scan-friendly Parquet.

Inputs are the raw McAuley-Lab/Amazon-Reviews-2023 files (default category
``Electronics``):

- ``raw/review_categories/<Category>.jsonl``   one review per line
- ``raw/meta_categories/meta_<Category>.jsonl`` one product (``parent_asin``) per line

Output under ``--out-dir`` (default: this directory):

- ``parquet/products.parquet`` one row per ``parent_asin``, sorted by ``product_id``
- ``parquet/reviews.parquet``  one row per review, sorted by ``(review_time, review_id)``
- ``manifest.json``            source identity, counts, and per-table Parquet layout

DuckDB (the ``duckdb`` CLI, as in ``fixture/tpc``) parses the JSONL, assigns keys,
checks the join invariants, and sorts with external-memory spilling. PyArrow then
rewrites the sorted intermediates with the final Parquet options, which DuckDB
cannot set per column.

The layout targets the native GPU scan path:

- Every column has a type the native engine can filter, group, and join on. Nested
  source fields (lists, structs, the ``details`` object) become strings or counts,
  because Struct and Map columns send the whole scan to the CPU and List columns
  can only be carried. ``review_time`` is a timezone-naive millisecond timestamp
  holding UTC wall time; timezone-aware timestamps cannot be compared on the GPU.
- ``product_id`` and ``user_key`` are dense Int32 surrogate keys, so hash joins and
  group-bys avoid string hashing and a cuGraph user->product edge list can use
  them as vertex ids directly. The source string keys stay alongside them.
- Rows are physically sorted so row-group min/max statistics prune time ranges on
  ``reviews`` and key ranges on ``products``. As in ``fixture/tpc``, the files do
  not declare ``sorting_columns``: the order serves pruning, not as an
  output-ordering promise to DataFusion.
- Snappy compression, format 2.6, page indexes, dictionary encoding only on
  low-cardinality columns, statistics on every non-free-text column, and Bloom
  filters on the equality-lookup keys.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


DATASET_DIR = Path(__file__).resolve().parents[1]
GENERATOR = "fixture/demo/amazon_reviews_2023/scripts/prepare.py"
SOURCE = "McAuley-Lab/Amazon-Reviews-2023 (https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)"
DEFAULT_CATEGORY = "Electronics"
DEFAULT_REVIEW_ROW_GROUP_ROWS = 1_048_576
# Product rows carry long feature/description/details text; smaller groups keep
# one decoded row group near the size of a review row group.
DEFAULT_PRODUCT_ROW_GROUP_ROWS = 131_072
BLOOM_FPP = 0.01

PRODUCT_SCHEMA = pa.schema(
    [
        pa.field("product_id", pa.int32(), nullable=False),
        pa.field("parent_asin", pa.string(), nullable=False),
        pa.field("title", pa.string()),
        pa.field("main_category", pa.string()),
        pa.field("store", pa.string()),
        pa.field("brand", pa.string()),
        pa.field("category_l1", pa.string()),
        pa.field("category_l2", pa.string()),
        pa.field("category_l3", pa.string()),
        pa.field("category_path", pa.string()),
        pa.field("average_rating", pa.float32()),
        pa.field("rating_number", pa.int32()),
        pa.field("price", pa.float64()),
        pa.field("date_first_available", pa.date32()),
        pa.field("image_count", pa.int16(), nullable=False),
        pa.field("video_count", pa.int16(), nullable=False),
        pa.field("features_text", pa.string()),
        pa.field("description_text", pa.string()),
        pa.field("details_json", pa.string()),
    ]
)

REVIEW_SCHEMA = pa.schema(
    [
        pa.field("review_id", pa.int64(), nullable=False),
        pa.field("review_time", pa.timestamp("ms"), nullable=False),
        pa.field("product_id", pa.int32(), nullable=False),
        pa.field("user_key", pa.int32(), nullable=False),
        pa.field("rating", pa.int8(), nullable=False),
        pa.field("helpful_vote", pa.int32(), nullable=False),
        pa.field("verified_purchase", pa.bool_(), nullable=False),
        pa.field("image_count", pa.int16(), nullable=False),
        pa.field("parent_asin", pa.string(), nullable=False),
        pa.field("asin", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("title", pa.string()),
        pa.field("text", pa.string()),
    ]
)


@dataclass(frozen=True)
class TableLayout:
    name: str
    schema: pa.Schema
    sort_keys: tuple[str, ...]
    dictionary_columns: tuple[str, ...]
    bloom_columns: tuple[str, ...]
    free_text_columns: tuple[str, ...]

    def statistics_columns(self) -> list[str]:
        # Min/max of free text is never a useful pruning bound and bloats the footer.
        return [name for name in self.schema.names if name not in self.free_text_columns]


PRODUCTS = TableLayout(
    name="products",
    schema=PRODUCT_SCHEMA,
    sort_keys=("product_id",),
    dictionary_columns=("main_category", "store", "brand", "category_l1", "category_l2", "category_l3", "category_path"),
    bloom_columns=("store", "brand"),
    free_text_columns=("title", "features_text", "description_text", "details_json"),
)
REVIEWS = TableLayout(
    name="reviews",
    schema=REVIEW_SCHEMA,
    sort_keys=("review_time", "review_id"),
    dictionary_columns=("rating", "helpful_vote", "image_count"),
    bloom_columns=("product_id", "user_key", "parent_asin", "user_id"),
    free_text_columns=("title", "text"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", default=DEFAULT_CATEGORY, help=f"Source category (default: {DEFAULT_CATEGORY})")
    parser.add_argument("--reviews-jsonl", type=Path, help="Review JSONL (default: raw/review_categories/<category>.jsonl)")
    parser.add_argument("--meta-jsonl", type=Path, help="Metadata JSONL (default: raw/meta_categories/meta_<category>.jsonl)")
    parser.add_argument("--out-dir", type=Path, default=DATASET_DIR, help="Directory receiving parquet/ and manifest.json")
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Parent of the per-run intermediate and DuckDB spill directory (default: <out-dir>/work)",
    )
    parser.add_argument("--review-row-group-rows", type=int, default=DEFAULT_REVIEW_ROW_GROUP_ROWS)
    parser.add_argument("--product-row-group-rows", type=int, default=DEFAULT_PRODUCT_ROW_GROUP_ROWS)
    parser.add_argument("--duckdb-memory-limit", help="DuckDB memory_limit, e.g. 64GB (default: DuckDB's own default)")
    parser.add_argument("--duckdb-threads", type=int, help="DuckDB threads (default: DuckDB's own default)")
    parser.add_argument("--skip-sha256", action="store_true", help="Do not hash the source JSONL files")
    parser.add_argument("--keep-work", action="store_true", help="Keep the per-run intermediate directory")
    parser.add_argument("--force", action="store_true", help="Replace existing output Parquet and manifest")
    args = parser.parse_args()

    for name in ("review_row_group_rows", "product_row_group_rows"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    if args.duckdb_threads is not None and args.duckdb_threads < 1:
        parser.error("--duckdb-threads must be >= 1")
    args.reviews_jsonl = args.reviews_jsonl or DATASET_DIR / "raw" / "review_categories" / f"{args.category}.jsonl"
    args.meta_jsonl = args.meta_jsonl or DATASET_DIR / "raw" / "meta_categories" / f"meta_{args.category}.jsonl"
    args.work_dir = args.work_dir or args.out_dir / "work"
    return args


def log(message: str) -> None:
    print(f"[amazon-reviews] {message}", file=sys.stderr, flush=True)


def sql_string(value: Path | str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def duckdb_script(args: argparse.Namespace, products_path: Path, reviews_path: Path, stats_path: Path) -> str:
    # Invariant checks are SELECTs that call error(); trash mode hides their empty results.
    settings = [".mode trash", f"SET temp_directory = {sql_string(args.work_dir / 'duckdb-tmp')};"]
    # review_id is the 0-based source line: rowid of a table created from an
    # insertion-ordered scan follows file order.
    settings.append("SET preserve_insertion_order = true;")
    if args.duckdb_memory_limit:
        settings.append(f"SET memory_limit = {sql_string(args.duckdb_memory_limit)};")
    if args.duckdb_threads:
        settings.append(f"SET threads = {args.duckdb_threads};")
    date_first_available = "json_extract_string(details, '$.\"Date First Available\"')"
    return "\n".join(
        settings
        + [
            ".print load metadata",
            f"""
CREATE TABLE meta_src AS
SELECT * FROM read_json({sql_string(args.meta_jsonl)},
    format = 'newline_delimited',
    maximum_object_size = 268435456,
    columns = {{
        parent_asin: 'VARCHAR', title: 'VARCHAR', main_category: 'VARCHAR', store: 'VARCHAR',
        categories: 'VARCHAR[]', average_rating: 'DOUBLE', rating_number: 'BIGINT', price: 'JSON',
        features: 'VARCHAR[]', description: 'VARCHAR[]', details: 'JSON', images: 'JSON', videos: 'JSON'
    }});
SELECT CASE WHEN count(*) FILTER (parent_asin IS NULL) > 0
            THEN error('metadata has rows without parent_asin')
            WHEN count(*) <> count(DISTINCT parent_asin)
            THEN error('metadata parent_asin is not unique') END
FROM meta_src;
""",
            ".print build products",
            f"""
CREATE TABLE products AS
SELECT
    (row_number() OVER (ORDER BY parent_asin) - 1)::INTEGER AS product_id,
    parent_asin,
    nullif(trim(title), '') AS title,
    main_category,
    store,
    details->>'$.Brand' AS brand,
    categories[1] AS category_l1,
    categories[2] AS category_l2,
    categories[3] AS category_l3,
    nullif(array_to_string(categories, ' > '), '') AS category_path,
    average_rating::FLOAT AS average_rating,
    rating_number::INTEGER AS rating_number,
    -- A few prices are strings such as 'from 19.95' or a dash; they become NULL.
    CASE WHEN json_type(price) IN ('DOUBLE', 'BIGINT', 'UBIGINT') THEN price::DOUBLE END AS price,
    try_strptime({date_first_available}, '%B %d, %Y')::DATE AS date_first_available,
    coalesce(json_array_length(images), 0)::SMALLINT AS image_count,
    coalesce(json_array_length(videos), 0)::SMALLINT AS video_count,
    nullif(array_to_string(features, chr(10)), '') AS features_text,
    nullif(array_to_string(description, chr(10)), '') AS description_text,
    details::VARCHAR AS details_json
FROM meta_src;
DROP TABLE meta_src;
""",
            ".print load reviews",
            f"""
CREATE TABLE reviews_src AS
SELECT * FROM read_json({sql_string(args.reviews_jsonl)},
    format = 'newline_delimited',
    maximum_object_size = 268435456,
    columns = {{
        rating: 'DOUBLE', title: 'VARCHAR', text: 'VARCHAR', images: 'JSON', asin: 'VARCHAR',
        parent_asin: 'VARCHAR', user_id: 'VARCHAR', timestamp: 'BIGINT', helpful_vote: 'INTEGER',
        verified_purchase: 'BOOLEAN'
    }});
SELECT CASE WHEN count(*) FILTER (parent_asin IS NULL OR asin IS NULL OR user_id IS NULL OR "timestamp" IS NULL) > 0
            THEN error('reviews have rows without parent_asin, asin, user_id, or timestamp')
            WHEN count(*) FILTER (rating IS NULL OR rating <> round(rating) OR rating NOT BETWEEN 0 AND 5) > 0
            THEN error('reviews have ratings outside the integers 0..5') END
FROM reviews_src;
SELECT CASE WHEN count(*) > 0 THEN error('reviews reference parent_asin values missing from metadata') END
FROM reviews_src r ANTI JOIN products p USING (parent_asin);
""",
            ".print build users",
            """
CREATE TABLE users AS
SELECT user_id, (row_number() OVER (ORDER BY user_id) - 1)::INTEGER AS user_key
FROM (SELECT DISTINCT user_id FROM reviews_src);
""",
            ".print write sorted products",
            f"COPY (SELECT * FROM products ORDER BY product_id) TO {sql_string(products_path)} (FORMAT parquet, COMPRESSION snappy);",
            ".print write sorted reviews",
            f"""
COPY (
    SELECT
        r.rowid AS review_id,
        epoch_ms(r."timestamp")::TIMESTAMP_MS AS review_time,
        p.product_id,
        u.user_key,
        r.rating::TINYINT AS rating,
        coalesce(r.helpful_vote, 0) AS helpful_vote,
        coalesce(r.verified_purchase, false) AS verified_purchase,
        coalesce(json_array_length(r.images), 0)::SMALLINT AS image_count,
        r.parent_asin,
        r.asin,
        r.user_id,
        r.title,
        r.text
    FROM reviews_src r
    JOIN products p USING (parent_asin)
    JOIN users u USING (user_id)
    ORDER BY review_time, review_id
) TO {sql_string(reviews_path)} (FORMAT parquet, COMPRESSION snappy);
""",
            ".print collect counts",
            f"""
COPY (
    SELECT
        (SELECT count(*) FROM reviews_src) AS reviews,
        (SELECT count(*) FROM products) AS products,
        (SELECT count(*) FROM users) AS users,
        (SELECT count(*) FROM products p ANTI JOIN reviews_src r USING (parent_asin)) AS products_without_reviews,
        (SELECT count(*) FROM products WHERE price IS NOT NULL) AS products_with_price,
        (SELECT count(*) FROM reviews_src WHERE verified_purchase) AS verified_reviews,
        (SELECT min("timestamp") FROM reviews_src) AS min_timestamp_ms,
        (SELECT max("timestamp") FROM reviews_src) AS max_timestamp_ms
) TO {sql_string(stats_path)} (FORMAT json);
""",
        ]
    )


def run_duckdb(script: str) -> None:
    duckdb = shutil.which("duckdb")
    if duckdb is None:
        raise SystemExit("duckdb executable not found; install the DuckDB CLI")
    result = subprocess.run([duckdb, "-bail", ":memory:"], input=script, text=True)
    if result.returncode != 0:
        raise SystemExit(f"duckdb exited with status {result.returncode}; see the error above")


def bloom_options(layout: TableLayout, row_group_rows: int, table_rows: int) -> dict[str, dict[str, Any]]:
    # One filter per column per row group, so size it to one group's rows.
    ndv = max(1024, min(row_group_rows, table_rows))
    return {column: {"ndv": ndv, "fpp": BLOOM_FPP} for column in layout.bloom_columns}


def rewrite(layout: TableLayout, source: Path, destination: Path, row_group_rows: int) -> None:
    """Stream DuckDB's sorted intermediate into the final Parquet layout."""
    reader = pq.ParquetFile(source)
    if reader.schema_arrow.names != layout.schema.names:
        raise SystemExit(f"{source}: columns {reader.schema_arrow.names} do not match {layout.schema.names}")
    tmp_path = destination.with_suffix(".parquet.tmp")
    writer = pq.ParquetWriter(
        tmp_path,
        layout.schema,
        compression="snappy",
        version="2.6",
        write_statistics=layout.statistics_columns(),
        write_page_index=True,
        use_dictionary=list(layout.dictionary_columns),
        bloom_filter_options=bloom_options(layout, row_group_rows, reader.metadata.num_rows),
    )
    written = 0
    try:
        for batch in reader.iter_batches(batch_size=row_group_rows):
            table = pa.Table.from_batches([batch]).cast(layout.schema)
            writer.write_table(table, row_group_size=row_group_rows)
            written += table.num_rows
            log(f"{layout.name}: wrote {written:,}/{reader.metadata.num_rows:,} rows")
    except BaseException:
        writer.close()
        tmp_path.unlink(missing_ok=True)
        raise
    writer.close()
    tmp_path.replace(destination)


def verify(layout: TableLayout, path: Path, expected_rows: int) -> dict[str, Any]:
    parquet_file = pq.ParquetFile(path)
    metadata = parquet_file.metadata
    if parquet_file.schema_arrow != layout.schema:
        raise SystemExit(f"{path}: schema {parquet_file.schema_arrow} does not match {layout.schema}")
    if metadata.num_rows != expected_rows:
        raise SystemExit(f"{path}: {metadata.num_rows} rows, expected {expected_rows}")
    key_index = layout.schema.get_field_index(layout.sort_keys[0])
    previous_max = None
    for index in range(metadata.num_row_groups):
        row_group = metadata.row_group(index)
        for column in range(row_group.num_columns):
            compression = row_group.column(column).compression
            if compression != "SNAPPY":
                raise SystemExit(f"{path}: row group {index} column {column} uses {compression}, expected SNAPPY")
        stats = row_group.column(key_index).statistics
        if stats is None or not stats.has_min_max:
            raise SystemExit(f"{path}: row group {index} lacks min/max statistics for {layout.sort_keys[0]}")
        if previous_max is not None and stats.min < previous_max:
            raise SystemExit(f"{path}: row group {index} is not sorted by {layout.sort_keys[0]}")
        previous_max = stats.max
    return {
        "path": f"parquet/{path.name}",
        "bytes": path.stat().st_size,
        "num_rows": metadata.num_rows,
        "num_row_groups": metadata.num_row_groups,
        "schema": layout.schema.names,
        "sort_keys": list(layout.sort_keys),
        "dictionary_columns": list(layout.dictionary_columns),
        "bloom_filter_columns": list(layout.bloom_columns),
        "free_text_columns_without_statistics": list(layout.free_text_columns),
    }


def source_entry(path: Path, skip_sha256: bool) -> dict[str, Any]:
    digest = None
    if not skip_sha256:
        log(f"hash {path}")
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    for path in (args.reviews_jsonl, args.meta_jsonl):
        if not path.is_file():
            raise SystemExit(f"source JSONL not found: {path}; see fixture/demo/amazon_reviews_2023/README.md")

    parquet_dir = args.out_dir / "parquet"
    manifest_path = args.out_dir / "manifest.json"
    outputs = [parquet_dir / "products.parquet", parquet_dir / "reviews.parquet", manifest_path]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.force:
        raise SystemExit(f"output exists: {', '.join(map(str, existing))}; pass --force to replace it")
    for path in existing:
        path.unlink()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    # A private run directory, so cleanup never touches a caller-supplied --work-dir's contents.
    args.work_dir = Path(tempfile.mkdtemp(prefix="amazon-reviews-", dir=args.work_dir))
    parquet_dir.mkdir(parents=True, exist_ok=True)
    sorted_products = args.work_dir / "products.sorted.parquet"
    sorted_reviews = args.work_dir / "reviews.sorted.parquet"
    stats_path = args.work_dir / "stats.json"
    try:
        log(f"DuckDB: {args.meta_jsonl} + {args.reviews_jsonl}")
        run_duckdb(duckdb_script(args, sorted_products, sorted_reviews, stats_path))
        counts = json.loads(stats_path.read_text(encoding="utf-8").splitlines()[0])
        tables = {}
        for layout, source, row_group_rows, expected_rows in (
            (PRODUCTS, sorted_products, args.product_row_group_rows, counts["products"]),
            (REVIEWS, sorted_reviews, args.review_row_group_rows, counts["reviews"]),
        ):
            destination = parquet_dir / f"{layout.name}.parquet"
            rewrite(layout, source, destination, row_group_rows)
            tables[layout.name] = verify(layout, destination, expected_rows)
            tables[layout.name]["row_group_rows"] = row_group_rows
    finally:
        if not args.keep_work:
            shutil.rmtree(args.work_dir, ignore_errors=True)

    manifest = {
        "generator": GENERATOR,
        "generated_at_unix": int(time.time()),
        "dataset": {"source": SOURCE, "category": args.category},
        "sources": {
            "reviews": source_entry(args.reviews_jsonl, args.skip_sha256),
            "metadata": source_entry(args.meta_jsonl, args.skip_sha256),
        },
        "counts": counts,
        "layout": {"compression": "SNAPPY", "format_version": "2.6", "sorting_columns_metadata": False},
        "tables": tables,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"complete: {counts['reviews']:,} reviews, {counts['products']:,} products -> {parquet_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
