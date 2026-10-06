#!/usr/bin/env python3
"""Convert MerRec recommendation interaction data into GPU-scan-friendly Parquet.

Inputs are raw parquet shards from mercari-us/merrec (default partition 20231001).

Output under --out-dir (default: this directory):

- parquet/items.parquet   one row per item_id, sorted by item_id
- parquet/events.parquet  one row per interaction event, sorted by (event_time, event_id)
- manifest.json           source identity, counts, and per-table Parquet layout

DuckDB parses the source parquet, rejects invalid rows, checks join and key
invariants, deduplicates item attributes to their latest state, and sorts with
external-memory spilling. PyArrow then streams the sorted intermediates into
the final Parquet layout with native GPU scan options.

The layout targets the native GPU scan path:

- Every column uses a type the native engine can filter, group, and join on.
  event_time is a timezone-naive millisecond timestamp holding UTC wall time.
  Category hierarchy is flattened to individual levels and a category_path string.
- item_id and user_id are native 64-bit integer keys. Hash joins, group-bys,
  and cuGraph bipartite graph construction operate directly on integers without
  string hashing.
- Rows are physically sorted so row-group min/max statistics prune time ranges on
  events and key ranges on items. The files omit sorting_columns in the footer
  because order serves pruning, not an output-ordering promise to DataFusion.
- Snappy compression, format 2.6, page indexes, dictionary encoding on
  low-cardinality columns, statistics on non-free-text columns, and Bloom
  filters on equality-lookup keys.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


DATASET_DIR = Path(__file__).resolve().parents[1]
GENERATOR = "fixture/demo/merrec/scripts/prepare.py"
SOURCE = "mercari-us/merrec (https://huggingface.co/datasets/mercari-us/merrec)"
DEFAULT_PARTITION = "20231001"
DEFAULT_EVENT_ROW_GROUP_ROWS = 1_048_576
DEFAULT_ITEM_ROW_GROUP_ROWS = 131_072
BLOOM_FPP = 0.01
SOURCE_FILES = json.loads((Path(__file__).with_name("source_files.json")).read_text())
SOURCE_REVISION = SOURCE_FILES["revision"]
PARTITIONS = SOURCE_FILES["partitions"]

ITEM_SCHEMA = pa.schema(
    [
        pa.field("item_id", pa.int64(), nullable=False),
        pa.field("product_id", pa.string()),
        pa.field("name", pa.string()),
        pa.field("price", pa.float64()),
        pa.field("c0_id", pa.int32()),
        pa.field("c0_name", pa.string()),
        pa.field("c1_id", pa.int32()),
        pa.field("c1_name", pa.string()),
        pa.field("c2_id", pa.int32()),
        pa.field("c2_name", pa.string()),
        pa.field("category_path", pa.string()),
        pa.field("brand_id", pa.int32()),
        pa.field("brand_name", pa.string()),
        pa.field("item_condition_id", pa.int16()),
        pa.field("item_condition_name", pa.string()),
        pa.field("size_id", pa.int32()),
        pa.field("size_name", pa.string()),
        pa.field("color", pa.string()),
        pa.field("shipper_id", pa.int16()),
        pa.field("shipper_name", pa.string()),
    ]
)

EVENT_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.int64(), nullable=False),
        pa.field("event_time", pa.timestamp("ms"), nullable=False),
        pa.field("user_id", pa.int64(), nullable=False),
        pa.field("item_id", pa.int64(), nullable=False),
        pa.field("session_id", pa.string(), nullable=False),
        pa.field("sequence_id", pa.string(), nullable=False),
        pa.field("sequence_length", pa.int32(), nullable=False),
        pa.field("event_type", pa.string(), nullable=False),
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
        return [name for name in self.schema.names if name not in self.free_text_columns]


ITEMS = TableLayout(
    name="items",
    schema=ITEM_SCHEMA,
    sort_keys=("item_id",),
    dictionary_columns=(
        "c0_name",
        "c1_name",
        "c2_name",
        "category_path",
        "item_condition_name",
        "size_name",
        "color",
        "shipper_name",
    ),
    bloom_columns=("item_id", "brand_name", "c0_name"),
    free_text_columns=("name",),
)

EVENTS = TableLayout(
    name="events",
    schema=EVENT_SCHEMA,
    sort_keys=("event_time", "event_id"),
    dictionary_columns=("event_type",),
    bloom_columns=("user_id", "item_id", "session_id", "sequence_id"),
    free_text_columns=(),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--partition",
        default=DEFAULT_PARTITION,
        choices=[*PARTITIONS, "all"],
        help=f"Complete monthly partition or all six months (default: {DEFAULT_PARTITION})",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        help="Source directory (default: beside the physical parquet directory)",
    )
    parser.add_argument(
        "--shards",
        type=int,
        help="Use the first N complete shards of one month for scale validation",
    )
    parser.add_argument(
        "--input-parquet",
        help="Input Parquet file, directory, or glob pattern (default: raw/<partition>/*.parquet)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DATASET_DIR,
        help="Directory receiving parquet/ and manifest.json (default: this directory)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="DuckDB scratch parent (default: beside the physical parquet directory)",
    )
    parser.add_argument(
        "--event-row-group-rows",
        type=int,
        default=DEFAULT_EVENT_ROW_GROUP_ROWS,
        help=f"Row group size for events (default: {DEFAULT_EVENT_ROW_GROUP_ROWS})",
    )
    parser.add_argument(
        "--item-row-group-rows",
        type=int,
        default=DEFAULT_ITEM_ROW_GROUP_ROWS,
        help=f"Row group size for items (default: {DEFAULT_ITEM_ROW_GROUP_ROWS})",
    )
    parser.add_argument(
        "--duckdb-memory-limit",
        help="DuckDB memory_limit (e.g. 32GB)",
    )
    parser.add_argument(
        "--duckdb-threads",
        type=int,
        help="DuckDB worker threads",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum input rows to read from source Parquet",
    )
    acquisition = parser.add_mutually_exclusive_group()
    acquisition.add_argument(
        "--download",
        action="store_true",
        help="Download missing shards for the selected partition or all six months",
    )
    acquisition.add_argument(
        "--download-sample",
        action="store_true",
        help="Explicitly select and download only the first shard of one month",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=8,
        help="Concurrent shard downloads (default: 8)",
    )
    parser.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Do not compute SHA-256 digests over source Parquet files",
    )
    parser.add_argument(
        "--keep-work",
        action="store_true",
        help="Keep the per-run temporary intermediate directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing output Parquet and manifest",
    )
    args = parser.parse_args(argv)

    for name in ("event_row_group_rows", "item_row_group_rows"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    if args.duckdb_threads is not None and args.duckdb_threads < 1:
        parser.error("--duckdb-threads must be >= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.download_workers < 1:
        parser.error("--download-workers must be >= 1")
    if args.shards is not None and args.shards < 1:
        parser.error("--shards must be >= 1")
    if args.partition == "all" and (args.shards is not None or args.download_sample):
        parser.error("--shards and --download-sample require one monthly partition")
    if args.input_parquet and (args.download or args.download_sample or args.shards is not None):
        parser.error("--input-parquet cannot be combined with --download, --download-sample, or --shards")
    if args.shards is not None and args.shards > len(PARTITIONS[args.partition]):
        parser.error(f"partition {args.partition} has only {len(PARTITIONS[args.partition])} shards")
    if args.download_sample and args.shards is not None:
        parser.error("--download-sample cannot be combined with --shards")
    if args.limit is not None and not (args.input_parquet or args.download_sample or args.shards is not None):
        parser.error("--limit requires an explicit subset (--input-parquet, --download-sample, or --shards)")
    args.raw_dir = args.raw_dir or (args.out_dir / "parquet").resolve().parent / "raw"
    # Follow the output link so DuckDB spills share the generated data's filesystem.
    args.work_dir = args.work_dir or (args.out_dir / "parquet").resolve().parent / "work"
    return args


def log(message: str) -> None:
    print(f"[merrec] {message}", file=sys.stderr, flush=True)


def sql_string(value: Path | str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def download_shard(partition: str, index: int, dest: Path, metadata: dict[str, Any]) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/mercari-us/merrec/resolve/{SOURCE_REVISION}/{partition}/{index:012d}.parquet"
    log(f"downloading {partition}/{index:012d}.parquet -> {dest}")
    tmp_dest = dest.with_suffix(".parquet.tmp")
    req = urllib.request.Request(url, headers={"User-Agent": "Algeon-Fixture-Downloader/1.0"})
    hasher = hashlib.sha256()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, tmp_dest.open("wb") as out:
            while chunk := resp.read(8 * 1024 * 1024):
                out.write(chunk)
                hasher.update(chunk)
        if tmp_dest.stat().st_size != metadata["bytes"] or hasher.hexdigest() != metadata["sha256"]:
            raise SystemExit(f"source integrity check failed: {dest}")
        pq.ParquetFile(tmp_dest)
    except BaseException:
        tmp_dest.unlink(missing_ok=True)
        raise
    tmp_dest.replace(dest)
    log(f"downloaded {dest} ({dest.stat().st_size:,} bytes)")
    return dest


def find_input_files(args: argparse.Namespace) -> list[Path]:
    if args.input_parquet:
        input_str = str(args.input_parquet)
        if "*" in input_str or "?" in input_str:
            matched = sorted(Path(p) for p in glob.glob(input_str, recursive=True))
            files = [p for p in matched if p.is_file() and p.name.endswith(".parquet")]
            if files:
                return files
            raise SystemExit(f"no parquet files matched pattern: {args.input_parquet}")
        path = Path(input_str)
        if path.is_file():
            return [path]
        if path.is_dir():
            files = sorted(path.rglob("*.parquet"))
            if files:
                return files
            raise SystemExit(f"directory contains no parquet files: {path}")
        raise SystemExit(f"specified --input-parquet path not found: {args.input_parquet}")

    partitions = list(PARTITIONS) if args.partition == "all" else [args.partition]
    selected = []
    for partition in partitions:
        count = 1 if args.download_sample else (args.shards or len(PARTITIONS[partition]))
        for index, metadata in enumerate(PARTITIONS[partition][:count]):
            path = args.raw_dir / partition / f"{index:012d}.parquet"
            if args.download_sample:
                path = args.raw_dir / "sample" / f"{partition}_{index:012d}.parquet"
            selected.append((partition, index, path, metadata))

    missing = [entry for entry in selected if not entry[2].is_file() or entry[2].stat().st_size != entry[3]["bytes"]]
    if missing and not (args.download or args.download_sample):
        raise SystemExit(
            f"incomplete MerRec source: {len(missing)} of {len(selected)} selected shards missing or wrong size; "
            "pass --download to acquire the selected scope, or --download-sample for an explicit sample"
        )
    if missing:
        with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
            list(executor.map(lambda entry: download_shard(*entry), missing))
    return [entry[2] for entry in selected]


def duckdb_settings(args: argparse.Namespace) -> list[str]:
    settings = [".mode trash", f"SET temp_directory = {sql_string(args.work_dir / 'duckdb-tmp')};"]
    settings.append("SET preserve_insertion_order = true;")
    if args.duckdb_memory_limit:
        settings.append(f"SET memory_limit = {sql_string(args.duckdb_memory_limit)};")
    if args.duckdb_threads:
        settings.append(f"SET threads = {args.duckdb_threads};")
    return settings


def source_sql(args: argparse.Namespace, input_files: list[Path]) -> str:
    file_list = ", ".join(sql_string(p) for p in input_files)
    limit_clause = f" LIMIT {args.limit}" if args.limit is not None else ""
    # A limited selection is materialized once so repeated scans retain identical rows.
    source_kind = "TABLE" if args.limit is not None else "VIEW"
    return f"CREATE {source_kind} src AS SELECT * FROM read_parquet([{file_list}]){limit_clause};"


def partition_items(args: argparse.Namespace, input_files: list[Path]) -> list[Path]:
    fields = ", ".join(name for name in ITEM_SCHEMA.names if name != "category_path")
    partition_dir = args.work_dir / "item-source"
    # Equal item IDs share a bucket, keeping latest-row aggregation bounded across months.
    run_duckdb("\n".join(duckdb_settings(args) + [
        source_sql(args, input_files),
        """
SELECT CASE WHEN count(*) FILTER (item_id IS NULL OR user_id IS NULL OR stime IS NULL OR session_id IS NULL OR sequence_id IS NULL OR event_id IS NULL) > 0
            THEN error('merrec source has rows without item_id, user_id, stime, session_id, sequence_id, or event_id')
            WHEN count(*) FILTER (price IS NOT NULL AND price < 0) > 0
            THEN error('merrec source has negative prices') END
FROM src;
""",
        *([f"COPY src TO {sql_string(args.work_dir / 'selected.parquet')} (FORMAT parquet, COMPRESSION snappy);"]
          if args.limit is not None else []),
        ".print partition item attributes",
        f"COPY (SELECT {fields}, stime, hash(item_id) % 64 AS bucket FROM src) "
        f"TO {sql_string(partition_dir)} (FORMAT parquet, COMPRESSION snappy, PARTITION_BY (bucket));",
    ]))
    partitions = sorted(partition_dir.glob("bucket=*"))
    if not partitions:
        empty_partition = partition_dir / "bucket=0"
        empty_partition.mkdir(parents=True, exist_ok=True)
        run_duckdb("\n".join(duckdb_settings(args) + [
            source_sql(args, input_files),
            f"COPY (SELECT {fields}, stime FROM src WHERE false) TO "
            f"{sql_string(empty_partition / 'empty.parquet')} (FORMAT parquet, COMPRESSION snappy);",
        ]))
        partitions = [empty_partition]
    return partitions


def duckdb_script(
    args: argparse.Namespace,
    input_files: list[Path],
    item_partitions: list[Path],
    items_path: Path,
    events_path: Path,
    stats_path: Path,
) -> str:
    item_fields = ", ".join(f"{name} := {name}" for name in ITEM_SCHEMA.names if name != "category_path")
    item_parts = args.work_dir / "item-latest"
    item_parts.mkdir()
    commands = duckdb_settings(args) + [source_sql(args, input_files)]
    for index, partition in enumerate(item_partitions):
        commands.extend([
            f".print deduplicate items {index + 1}/{len(item_partitions)}",
            f"COPY (SELECT latest.* FROM (SELECT arg_max(struct_pack({item_fields}), stime) AS latest "
            f"FROM read_parquet({sql_string(partition / '*.parquet')}) GROUP BY item_id)) "
            f"TO {sql_string(item_parts / f'{index:02d}.parquet')} (FORMAT parquet, COMPRESSION snappy);",
        ])
    return "\n".join(
        commands + [
            ".print build items",
            f"""
CREATE VIEW items AS
SELECT
    item_id::BIGINT AS item_id,
    nullif(trim(product_id), '') AS product_id,
    nullif(trim(name), '') AS name,
    price::DOUBLE AS price,
    c0_id::INTEGER AS c0_id,
    nullif(trim(c0_name), '') AS c0_name,
    c1_id::INTEGER AS c1_id,
    nullif(trim(c1_name), '') AS c1_name,
    c2_id::INTEGER AS c2_id,
    nullif(trim(c2_name), '') AS c2_name,
    nullif(concat_ws(' > ', nullif(trim(c0_name), ''), nullif(trim(c1_name), ''), nullif(trim(c2_name), '')), '') AS category_path,
    brand_id::INTEGER AS brand_id,
    nullif(trim(brand_name), '') AS brand_name,
    item_condition_id::SMALLINT AS item_condition_id,
    nullif(trim(item_condition_name), '') AS item_condition_name,
    size_id::INTEGER AS size_id,
    nullif(trim(size_name), '') AS size_name,
    nullif(trim(color), '') AS color,
    shipper_id::SMALLINT AS shipper_id,
    nullif(trim(shipper_name), '') AS shipper_name
FROM read_parquet({sql_string(item_parts / '*.parquet')});
""",
            ".print build events",
            """
CREATE VIEW events AS
SELECT
    stime::TIMESTAMP_MS AS event_time,
    user_id::BIGINT AS user_id,
    item_id::BIGINT AS item_id,
    session_id,
    sequence_id,
    sequence_length::INTEGER AS sequence_length,
    event_id AS event_type
FROM src;
SELECT CASE WHEN count(*) > 0 THEN error('events reference item_id missing from items') END
FROM src e ANTI JOIN items i USING (item_id);
""",
            ".print write sorted items",
            f"COPY (SELECT * FROM items ORDER BY item_id) TO {sql_string(items_path)} (FORMAT parquet, COMPRESSION snappy);",
            ".print write sorted events",
            f"COPY (SELECT * FROM events ORDER BY event_time, item_id, user_id) TO {sql_string(events_path)} (FORMAT parquet, COMPRESSION snappy);",
            f"CREATE OR REPLACE VIEW events AS SELECT * FROM read_parquet({sql_string(events_path)});",
            ".print collect counts",
            f"""
COPY (
    SELECT
        (SELECT count(*) FROM events) AS events,
        (SELECT count(*) FROM items) AS items,
        (SELECT count(DISTINCT user_id) FROM events) AS users,
        (SELECT count(DISTINCT session_id) FROM events) AS sessions,
        (SELECT count(DISTINCT sequence_id) FROM events) AS sequences,
        (SELECT count(*) FROM events WHERE event_type = 'buy_comp') AS buy_events,
        (SELECT min(event_time)::VARCHAR FROM events) AS min_event_time,
        (SELECT max(event_time)::VARCHAR FROM events) AS max_event_time
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
    ndv = max(1024, min(row_group_rows, table_rows))
    return {column: {"ndv": ndv, "fpp": BLOOM_FPP} for column in layout.bloom_columns}


def rewrite(layout: TableLayout, source: Path, destination: Path, row_group_rows: int) -> None:
    """Stream DuckDB sorted intermediate into final Parquet layout with GPU options."""
    reader = pq.ParquetFile(source)
    source_columns = layout.schema.names[1:] if layout == EVENTS else layout.schema.names
    if reader.schema_arrow.names != source_columns:
        raise SystemExit(f"{source}: columns {reader.schema_arrow.names} do not match {source_columns}")
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
        for table in output_row_groups(reader, row_group_rows):
            if layout == EVENTS:
                # Assign dense IDs after sorting, avoiding a window followed by a second global sort.
                event_ids = pa.array(range(written, written + table.num_rows), type=pa.int64())
                table = table.add_column(0, layout.schema.field("event_id"), event_ids)
            table = table.cast(layout.schema)
            writer.write_table(table, row_group_size=row_group_rows)
            written += table.num_rows
            log(f"{layout.name}: wrote {written:,}/{reader.metadata.num_rows:,} rows")
    except BaseException:
        writer.close()
        tmp_path.unlink(missing_ok=True)
        raise
    writer.close()
    tmp_path.replace(destination)


def output_row_groups(reader: pq.ParquetFile, row_group_rows: int) -> Iterator[pa.Table]:
    pending = []
    rows = 0
    for index in range(reader.metadata.num_row_groups):
        # A cross-row-group Arrow iterator retains decoded dictionary buffers.
        for batch in reader.iter_batches(batch_size=row_group_rows, row_groups=[index]):
            offset = 0
            while offset < batch.num_rows:
                count = min(row_group_rows - rows, batch.num_rows - offset)
                pending.append(batch.slice(offset, count))
                rows += count
                offset += count
                if rows == row_group_rows:
                    yield pa.Table.from_batches(pending)
                    pending = []
                    rows = 0
    if pending:
        yield pa.Table.from_batches(pending)


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


def file_digest(path: Path, skip_sha256: bool) -> dict[str, Any]:
    digest = None
    if not skip_sha256:
        log(f"hash {path}")
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    parquet_dir = args.out_dir / "parquet"
    manifest_path = args.out_dir / "manifest.json"
    outputs = [parquet_dir / "items.parquet", parquet_dir / "events.parquet", manifest_path]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.force:
        raise SystemExit(f"output exists: {', '.join(map(str, existing))}; pass --force to replace it")

    input_files = find_input_files(args)
    sources = [file_digest(p, args.skip_sha256) for p in input_files]
    input_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in input_files)
    if not args.input_parquet and not args.skip_sha256:
        for path, source in zip(input_files, sources, strict=True):
            partition = path.name.split("_")[0] if args.download_sample else path.parent.name
            index = int(path.stem.split("_")[-1])
            if source["sha256"] != PARTITIONS[partition][index]["sha256"]:
                raise SystemExit(f"source integrity check failed: {path}")
    expected_events = min(input_rows, args.limit) if args.limit is not None else input_rows
    selection = (
        "explicit" if args.input_parquet else
        "sample" if args.download_sample else
        "shards" if args.shards is not None else
        "all" if args.partition == "all" else "partition"
    )
    log(f"scope={selection}, partition={args.partition}, {len(input_files)} shards, {expected_events:,} events")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir = Path(tempfile.mkdtemp(prefix="merrec-", dir=args.work_dir))
    parquet_dir.mkdir(parents=True, exist_ok=True)
    # Keep the usable tables until both replacements pass validation.
    staging_dir = Path(tempfile.mkdtemp(prefix=".merrec-", dir=parquet_dir.resolve().parent))
    sorted_items = args.work_dir / "items.sorted.parquet"
    sorted_events = args.work_dir / "events.sorted.parquet"
    stats_path = args.work_dir / "stats.json"

    try:
        log(f"DuckDB: reading {len(input_files)} source files")
        item_partitions = partition_items(args, input_files)
        conversion_inputs = [args.work_dir / "selected.parquet"] if args.limit is not None else input_files
        run_duckdb(duckdb_script(args, conversion_inputs, item_partitions, sorted_items, sorted_events, stats_path))
        counts = json.loads(stats_path.read_text(encoding="utf-8").splitlines()[0])
        if counts["events"] != expected_events:
            raise SystemExit(f"event count mismatch: {counts['events']}, expected {expected_events}")
        tables = {}
        for layout, source, row_group_rows, expected_rows in (
            (ITEMS, sorted_items, args.item_row_group_rows, counts["items"]),
            (EVENTS, sorted_events, args.event_row_group_rows, counts["events"]),
        ):
            destination = staging_dir / f"{layout.name}.parquet"
            rewrite(layout, source, destination, row_group_rows)
            tables[layout.name] = verify(layout, destination, expected_rows)
            tables[layout.name]["row_group_rows"] = row_group_rows
        manifest = {
            "generator": GENERATOR,
            "generated_at_unix": int(time.time()),
            "dataset": {
                "source": SOURCE,
                "partition": args.partition,
                "selection": selection,
                "revision": SOURCE_REVISION if not args.input_parquet else None,
                "input_rows": input_rows,
                "row_limit": args.limit,
            },
            "sources": sources,
            "counts": counts,
            "layout": {"compression": "SNAPPY", "format_version": "2.6", "sorting_columns_metadata": False},
            "tables": tables,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        }
        manifest_target = manifest_path.resolve()
        staged_manifest = manifest_target.with_suffix(".json.tmp")
        staged_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for layout in (ITEMS, EVENTS):
            (staging_dir / f"{layout.name}.parquet").replace(parquet_dir / f"{layout.name}.parquet")
        # Preserve the repository's manifest symlink when installing the new file.
        staged_manifest.replace(manifest_target)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
        if not args.keep_work:
            shutil.rmtree(args.work_dir, ignore_errors=True)
    log(f"complete: {counts['events']:,} events, {counts['items']:,} items -> {parquet_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
