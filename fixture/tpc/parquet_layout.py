#!/usr/bin/env python3
"""Shared Parquet layout helpers for TPC fixture generation."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Literal, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


DEFAULT_ROW_GROUP_BYTES = 8 * 1024 * 1024
# SF300 lineitem needed about 291 GiB of DuckDB memory plus temp space to sort 82 GB of
# raw Snappy Parquet, the densest TPC-H table. Other tables need less per source byte.
SORT_WORKING_SET_PER_SOURCE_BYTE = 3.8
# Share of the memory the host reports as available when a table's sort starts that
# DuckDB may use. Its default limit is 80% of total RAM and its resident size can pass
# the limit, so next to a 120 GiB co-tenant process the kernel OOM killer stopped an
# SF1000 slice sort. With a lower limit DuckDB spills to temp space instead.
SORT_MEMORY_FRACTION = 0.5
# A slice's estimated working set stays at this share of the DuckDB limit, leaving room
# for sort-key copies and merge buffers.
SORT_SLICE_MEMORY_SHARE = 0.5
# "spark" sorts with DuckDB and writes the final file with parquet-java through Spark.
OptimizeEngine = Literal["duckdb", "pyarrow", "spark"]

# The Spark writer uses the Spark and parquet-java defaults below, set explicitly so
# a site configuration cannot change them; only the single output file is non-default.
SPARK_VERSION = "4.0.1"
SPARK_PARQUET_BLOCK_SIZE = 128 * 1024 * 1024
SPARK_PARQUET_PAGE_SIZE = 1024 * 1024
SPARK_PARQUET_PAGE_ROW_COUNT_LIMIT = 20_000
SPARK_SUPPORTED_JAVA_MAJORS = (17, 21)
SPARK_DRIVER_MEMORY = "16g"
SPARK_INSTALL_HINT = (
    f"install it with `pip install pyspark=={SPARK_VERSION}` (Java 17 or 21 on PATH or in JAVA_HOME) "
    "and pass --spark-python or set ALGEON_SPARK_PYTHON to that interpreter"
)
SPARK_WRITE_SCRIPT = Path(__file__).resolve().with_name("spark_write.py")
# parquet 59.3, which DataFusion 55 uses, stores each row-group ordinal as a Thrift
# i16 while decoding the footer and rejects a file with more row groups.
MAX_ROW_GROUPS_PER_FILE = 32_767


@dataclass(frozen=True)
class TableLayout:
    name: str
    sort_keys: tuple[str, ...]
    field_renames: tuple[tuple[str, str], ...] = ()
    field_casts: tuple[tuple[str, str], ...] = ()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def scale_label(scale: float) -> str:
    return f"sf{int(scale)}" if scale.is_integer() else f"sf{str(scale).replace('.', '_')}"


def parse_scales(values: Sequence[str] | None) -> list[float]:
    scales = [parse_scale(value) for value in values] if values else [1.0]
    for scale in scales:
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
    return scales


def parse_scale(value: str) -> float:
    normalized = value.strip().lower()
    if normalized.startswith("sf"):
        normalized = normalized[2:]
    return float(normalized)


def resolve_output_dir(*, workload: str, scale: float, out: Path | None, out_root: Path | None) -> Path:
    if out is not None:
        return out
    fixture_dir = {"tpch": "tpc-h", "tpcds": "tpc-ds"}.get(workload)
    if fixture_dir is None:
        raise ValueError(f"unsupported TPC fixture workload: {workload}")
    root = out_root if out_root is not None else repo_root() / "fixture" / fixture_dir / "data"
    return root / scale_label(scale) / "parquet"


def ensure_safe_replace_dir(path: Path) -> None:
    resolved = path.resolve()
    root = repo_root().resolve()
    protected = {
        Path("/").resolve(),
        Path.home().resolve(),
        root,
        root / "fixture",
        root / "fixture" / "tpc-h" / "data",
        root / "fixture" / "tpc-ds" / "data",
        root / "target",
        root / "target" / "fixture-staging",
    }
    if resolved in protected:
        raise ValueError(f"refusing to replace protected directory: {path}")


def replace_with_empty_dir(path: Path, *, force: bool) -> None:
    ensure_safe_replace_dir(path)
    if path.exists():
        if not force and any(path.iterdir()):
            raise FileExistsError(f"{path} already exists and is not empty; pass --force to replace it")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def run(cmd: Sequence[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(cmd), file=sys.stderr, flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def normalize_for_pyarrow(table: pa.Table) -> pa.Table:
    fields = []
    changed = False
    for field in table.schema:
        if hasattr(pa, "string_view") and field.type == pa.string_view():
            fields.append(pa.field(field.name, pa.string(), field.nullable, field.metadata))
            changed = True
        else:
            fields.append(field)
    return table.cast(pa.schema(fields, metadata=table.schema.metadata)) if changed else table


def read_table(source: Path) -> pa.Table:
    if source.is_dir():
        files = sorted(source.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no parquet files under {source}")
        table = pa.concat_tables([pq.read_table(file) for file in files], promote_options="default")
    else:
        table = pq.read_table(source)
    return normalize_for_pyarrow(table).combine_chunks()


def sort_table(table: pa.Table, keys: Sequence[str]) -> pa.Table:
    existing = [key for key in keys if key in table.schema.names]
    if not existing or table.num_rows <= 1:
        return table
    indices = pc.sort_indices(table, sort_keys=[(key, "ascending") for key in existing], null_placement="at_end")
    return table.take(indices)


def rename_table_fields(table: pa.Table, renames: Sequence[tuple[str, str]]) -> pa.Table:
    if not renames:
        return table
    rename_map = dict(renames)
    names = [rename_map.get(name, name) for name in table.schema.names]
    if len(names) != len(set(names)):
        raise ValueError(f"field renames would create duplicate columns: {renames}")
    return table.rename_columns(names)


def cast_table_fields(table: pa.Table, casts: Sequence[tuple[str, str]]) -> pa.Table:
    if not casts:
        return table
    cast_map = dict(casts)
    fields = []
    changed = False
    for field in table.schema:
        cast_type = cast_map.get(field.name)
        if cast_type is None:
            fields.append(field)
            continue
        target_type = parse_cast_type(cast_type)
        fields.append(pa.field(field.name, target_type, field.nullable, field.metadata))
        changed = True
    unknown = sorted(set(cast_map) - set(table.schema.names))
    if unknown:
        raise ValueError(f"field casts reference missing columns: {unknown}")
    return table.cast(pa.schema(fields, metadata=table.schema.metadata)) if changed else table


def parse_cast_type(name: str) -> pa.DataType:
    if name == "int64":
        return pa.int64()
    raise ValueError(f"unsupported fixture cast type: {name}")


def rows_per_group(table: pa.Table, target_bytes: int) -> int:
    if table.num_rows == 0:
        return 1
    bytes_per_row = max(1.0, table.nbytes / table.num_rows)
    return max(1, min(1_000_000, int(target_bytes / bytes_per_row)))


def source_files(source: Path) -> list[Path]:
    if source.is_dir():
        files = sorted(source.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no parquet files under {source}")
        return files
    return [source]


def rows_per_group_from_metadata(source: Path, target_bytes: int) -> int:
    total_rows = 0
    total_bytes = 0
    for file in source_files(source):
        metadata = pq.ParquetFile(file).metadata
        total_rows += metadata.num_rows
        for rg_index in range(metadata.num_row_groups):
            row_group = metadata.row_group(rg_index)
            total_bytes += max(0, row_group.total_byte_size)
    if total_rows <= 0 or total_bytes <= 0:
        return 1
    bytes_per_row = max(1.0, total_bytes / total_rows)
    return max(1, min(1_000_000, int(target_bytes / bytes_per_row)))


def write_optimized_table(
    *,
    table: pa.Table,
    output_file: Path,
    sort_keys: Sequence[str],
    row_group_bytes: int,
    compression: str,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        output_file,
        compression=compression.lower(),
        row_group_size=rows_per_group(table, row_group_bytes),
        write_statistics=True,
    )


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def duckdb_read_parquet_source(source: Path) -> str:
    files = source_files(source)
    if len(files) == 1:
        return f"read_parquet({sql_string(str(files[0]))})"
    paths = ", ".join(sql_string(str(path)) for path in files)
    return f"read_parquet([{paths}])"


def duckdb_select_list(source: Path, layout: TableLayout) -> str:
    schema = pq.read_schema(source_files(source)[0])
    rename_map = dict(layout.field_renames)
    cast_map = dict(layout.field_casts)
    fields = []
    for source_name in schema.names:
        final_name = rename_map.get(source_name, source_name)
        expr = f'"{source_name}"'
        cast_type = cast_map.get(final_name)
        if cast_type is not None:
            expr = f"CAST({expr} AS {duckdb_cast_type(cast_type)})"
        if final_name != source_name or cast_type is not None:
            expr = f'{expr} AS "{final_name}"'
        fields.append(expr)
    unknown = sorted(set(cast_map) - {rename_map.get(name, name) for name in schema.names})
    if unknown:
        raise ValueError(f"field casts reference missing columns: {unknown}")
    return ", ".join(fields)


def duckdb_cast_type(name: str) -> str:
    if name == "int64":
        return "BIGINT"
    raise ValueError(f"unsupported fixture cast type: {name}")


def available_memory_bytes() -> int:
    with open("/proc/meminfo") as meminfo:
        for line in meminfo:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/meminfo has no MemAvailable line")


def sort_slice_count(source_bytes: int, slice_bytes: int) -> int:
    return max(1, math.ceil(source_bytes * SORT_WORKING_SET_PER_SOURCE_BYTE / slice_bytes))


def duckdb_query_rows(duckdb: str, sql: str, tag: str) -> list[list[str]]:
    """Return the columns after `tag` for each CSV row whose first column is `tag`.

    The CLI prints a notice on stdout in agent mode, so rows are matched by tag.
    """
    result = subprocess.run([duckdb, "-csv", "-noheader", "-c", sql], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"duckdb exited with {result.returncode}: {result.stderr.strip()}")
    return [row[1:] for row in csv.reader(io.StringIO(result.stdout)) if row and row[0] == tag]


def sort_slice_predicates(*, duckdb: str, subquery: str, key: str, slices: int) -> list[str]:
    """Predicates on `key` that split every row into disjoint, ordered value ranges.

    Each range holds all rows of its key values, so sorting the ranges one by one and
    concatenating them in order equals one global sort. NULL keys sort last and land in
    the final range, matching DuckDB's default NULL order.
    """
    quoted = f'"{key}"'
    positions = ", ".join(repr(index / slices) for index in range(1, slices))
    rows = duckdb_query_rows(
        duckdb,
        "SELECT 'BOUND', typeof(b), b::VARCHAR FROM ("
        f"SELECT DISTINCT b FROM (SELECT unnest(quantile_disc({quoted}, [{positions}])) AS b FROM {subquery})"
        ") ORDER BY b",
        "BOUND",
    )
    bounds = [f"CAST({sql_string(value)} AS {type_name})" for type_name, value in rows]
    if not bounds:
        return []
    predicates = [f"{quoted} < {bounds[0]}"]
    predicates += [f"{quoted} >= {low} AND {quoted} < {high}" for low, high in zip(bounds, bounds[1:])]
    predicates.append(f"{quoted} >= {bounds[-1]} OR {quoted} IS NULL")
    return predicates


def duckdb_copy_sql(
    *,
    query: str,
    output_file: Path,
    temp_directory: Path,
    memory_limit_bytes: int,
    compression: str,
    row_group_rows: int,
    keep_order: bool,
) -> str:
    return "\n".join(
        [
            f"SET memory_limit='{max(1, memory_limit_bytes // 2**20)}MiB';",
            f"SET temp_directory={sql_string(str(temp_directory))};",
            f"SET preserve_insertion_order={'true' if keep_order else 'false'};",
            f"COPY ({query})",
            (
                f"TO {sql_string(str(output_file))} "
                f"(FORMAT PARQUET, COMPRESSION {compression.upper()}, ROW_GROUP_SIZE {row_group_rows});"
            ),
        ]
    )


def find_duckdb() -> str:
    duckdb = shutil.which("duckdb")
    if duckdb is None:
        raise FileNotFoundError("duckdb executable not found; install duckdb or pass --optimize-engine pyarrow")
    return duckdb


def run_duckdb_copy(
    *,
    duckdb: str,
    query: str,
    destination: Path,
    temp_directory: Path,
    memory_limit_bytes: int,
    compression: str,
    row_group_rows: int,
    keep_order: bool,
) -> None:
    sql = duckdb_copy_sql(
        query=query,
        output_file=destination,
        temp_directory=temp_directory,
        memory_limit_bytes=memory_limit_bytes,
        compression=compression,
        row_group_rows=row_group_rows,
        keep_order=keep_order,
    )
    run([duckdb, "-c", sql])


def sort_table_with_duckdb(
    *,
    source: Path,
    output_dir: Path,
    layout: TableLayout,
    row_group_bytes: int,
    compression: str,
    sort_slice_bytes: int | None = None,
) -> list[Path]:
    """Sort `source` by `layout.sort_keys` into files under `output_dir`.

    The returned files, concatenated in list order, are the sorted table. DuckDB runs
    with `SORT_MEMORY_FRACTION` of the memory available when the sort starts. A table
    whose estimated sort working set exceeds `sort_slice_bytes` (default:
    `SORT_SLICE_MEMORY_SHARE` of that limit) is sorted in value ranges of its first
    sort key, one file per range. One global sort of SF750 lineitem needed more than
    388 GiB of DuckDB temp space and failed; a range fits in memory.
    """
    duckdb = find_duckdb()
    output_dir.mkdir(parents=True, exist_ok=True)
    select_list = duckdb_select_list(source, layout)
    subquery = f"(SELECT {select_list} FROM {duckdb_read_parquet_source(source)})"
    order_by = ", ".join(f'"{key}"' for key in layout.sort_keys)
    order_clause = f" ORDER BY {order_by}" if order_by else ""
    memory_limit_bytes = int(SORT_MEMORY_FRACTION * available_memory_bytes())
    copy_args = {
        "duckdb": duckdb,
        "temp_directory": output_dir / ".duckdb-tmp",
        "memory_limit_bytes": memory_limit_bytes,
        "compression": compression,
        "row_group_rows": rows_per_group_from_metadata(source, row_group_bytes),
        "keep_order": False,
    }
    try:
        slices = 1
        if layout.sort_keys:
            source_bytes = sum(file.stat().st_size for file in source_files(source))
            slice_bytes = sort_slice_bytes or int(SORT_SLICE_MEMORY_SHARE * memory_limit_bytes)
            slices = sort_slice_count(source_bytes, slice_bytes)
        predicates = (
            sort_slice_predicates(duckdb=duckdb, subquery=subquery, key=layout.sort_keys[0], slices=slices)
            if slices > 1
            else []
        )
        if len(predicates) < 2:
            sorted_file = output_dir / "00000.parquet"
            run_duckdb_copy(query=f"SELECT * FROM {subquery}{order_clause}", destination=sorted_file, **copy_args)
            return [sorted_file]

        slice_files = []
        for index, predicate in enumerate(predicates):
            print(f"  sorting {layout.name} slice {index + 1}/{len(predicates)}", file=sys.stderr, flush=True)
            slice_file = output_dir / f"{index:05d}.parquet"
            query = f"SELECT * FROM {subquery} WHERE {predicate}{order_clause}"
            run_duckdb_copy(query=query, destination=slice_file, **copy_args)
            slice_files.append(slice_file)
        return slice_files
    finally:
        shutil.rmtree(copy_args["temp_directory"], ignore_errors=True)


def check_written_rows(*, layout: TableLayout, inputs: Sequence[Path], output_file: Path, step: str) -> None:
    input_rows = sum(pq.ParquetFile(file).metadata.num_rows for file in inputs)
    written_rows = pq.ParquetFile(output_file).metadata.num_rows
    if written_rows != input_rows:
        raise RuntimeError(f"{step} of {layout.name} wrote {written_rows} rows from {input_rows} input rows")


def optimize_table_with_duckdb(
    *,
    source: Path,
    output_file: Path,
    layout: TableLayout,
    row_group_bytes: int,
    compression: str,
    sort_slice_bytes: int | None = None,
) -> None:
    """Rewrite `source` as one file sorted by `layout.sort_keys`.

    Sorted value ranges from `sort_table_with_duckdb` are concatenated in order.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    slice_directory = output_file.parent / ".sort-slices"
    try:
        slice_files = sort_table_with_duckdb(
            source=source,
            output_dir=slice_directory,
            layout=layout,
            row_group_bytes=row_group_bytes,
            compression=compression,
            sort_slice_bytes=sort_slice_bytes,
        )
        if len(slice_files) == 1:
            os.replace(slice_files[0], output_file)
            return
        paths = ", ".join(sql_string(str(file)) for file in slice_files)
        run_duckdb_copy(
            duckdb=find_duckdb(),
            query=f"SELECT * FROM read_parquet([{paths}])",
            destination=output_file,
            temp_directory=slice_directory / ".duckdb-tmp",
            memory_limit_bytes=int(SORT_MEMORY_FRACTION * available_memory_bytes()),
            compression=compression,
            row_group_rows=rows_per_group_from_metadata(source, row_group_bytes),
            keep_order=True,
        )
        check_written_rows(layout=layout, inputs=source_files(source), output_file=output_file, step="sliced sort")
    finally:
        shutil.rmtree(slice_directory, ignore_errors=True)


def check_spark_writer(spark_python: str) -> None:
    """Fail before any generation when `spark_python` cannot run the pinned Spark writer."""
    try:
        probe = subprocess.run(
            [spark_python, "-c", "import pyspark; print(pyspark.__version__)"], capture_output=True, text=True
        )
    except OSError as error:
        raise RuntimeError(f"cannot run Spark writer interpreter {spark_python}: {error}; {SPARK_INSTALL_HINT}") from error
    if probe.returncode != 0:
        raise RuntimeError(f"the Spark writer needs PySpark {SPARK_VERSION}, which {spark_python} cannot import; {SPARK_INSTALL_HINT}")
    version = probe.stdout.strip().splitlines()[-1] if probe.stdout.strip() else ""
    if version != SPARK_VERSION:
        raise RuntimeError(f"{spark_python} has PySpark {version or 'unknown'}; the fixture pins {SPARK_VERSION}; {SPARK_INSTALL_HINT}")

    java_home = os.environ.get("JAVA_HOME")
    java = str(Path(java_home) / "bin" / "java") if java_home else shutil.which("java")
    if java is None or not Path(java).exists():
        raise RuntimeError(f"the Spark writer needs Java 17 or 21 and found none; {SPARK_INSTALL_HINT}")
    banner = subprocess.run([java, "-version"], capture_output=True, text=True).stderr
    major = java_major_version(banner)
    if major not in SPARK_SUPPORTED_JAVA_MAJORS:
        first_line = banner.strip().splitlines()[0] if banner.strip() else "no version"
        raise RuntimeError(f"Spark {SPARK_VERSION} runs on Java 17 or 21; {java} reports {first_line!r}")


def java_major_version(banner: str) -> int | None:
    for line in banner.splitlines():
        if ' version "' in line:
            version = line.split(' version "', 1)[1].split('"', 1)[0]
            parts = version.split(".")
            return int(parts[1] if parts[0] == "1" else parts[0].split("-")[0])
    return None


def parquet_java_version(file: Path) -> str:
    created_by = pq.ParquetFile(file).metadata.created_by or ""
    prefix = "parquet-mr version "
    if not created_by.startswith(prefix):
        raise RuntimeError(f"{file} was written by {created_by!r}, expected parquet-mr")
    return created_by[len(prefix) :].split(" ", 1)[0]


def write_table_with_spark(
    *,
    source: Path,
    output_file: Path,
    layout: TableLayout,
    sort_rows: bool,
    row_group_bytes: int,
    compression: str,
    spark_python: str,
    sort_slice_bytes: int | None = None,
) -> str:
    """Write `source` as one parquet-java file through Spark and return the Spark version.

    With `sort_rows`, DuckDB first sorts the table into ordered value-range files and
    Spark reads them in that order; otherwise Spark reads the generator's files as
    they are. The work directory holding the sorted files and Spark's output
    directory is removed before returning.
    """
    work_dir = output_file.parent / f".{layout.name}.spark-work"
    work_dir.mkdir(parents=True)
    try:
        if sort_rows:
            inputs = sort_table_with_duckdb(
                source=source,
                output_dir=work_dir / "sorted",
                layout=layout,
                row_group_bytes=row_group_bytes,
                compression=compression,
                sort_slice_bytes=sort_slice_bytes,
            )
        else:
            if layout.field_renames or layout.field_casts:
                raise ValueError(f"unsorted Spark output cannot apply the field renames or casts of {layout.name}")
            inputs = source_files(source)
        spark_dir = work_dir / "spark"
        job = {
            "inputs": [str(path) for path in inputs],
            "output_dir": str(spark_dir),
            "result_file": str(work_dir / "result.json"),
            "local_dir": str(work_dir / "spark-local"),
            "driver_memory": SPARK_DRIVER_MEMORY,
            "compression": compression.lower(),
            "parquet_block_size": SPARK_PARQUET_BLOCK_SIZE,
            "parquet_page_size": SPARK_PARQUET_PAGE_SIZE,
            "parquet_page_row_count_limit": SPARK_PARQUET_PAGE_ROW_COUNT_LIMIT,
        }
        job_file = work_dir / "job.json"
        job_file.write_text(json.dumps(job, indent=2) + "\n")
        run([spark_python, str(SPARK_WRITE_SCRIPT), str(job_file)])
        parts = sorted(spark_dir.glob("part-*.parquet"))
        if len(parts) != 1:
            raise RuntimeError(f"Spark wrote {len(parts)} part files for {layout.name}, expected 1")
        os.replace(parts[0], output_file)
        check_written_rows(layout=layout, inputs=source_files(source), output_file=output_file, step="Spark write")
        return json.loads((work_dir / "result.json").read_text())["spark_version"]
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def optimize_layout(
    *,
    raw_dir: Path,
    output_dir: Path,
    layouts: Sequence[TableLayout],
    row_group_bytes: int,
    compression: str,
    force: bool,
    scale: float,
    engine: OptimizeEngine = "duckdb",
    sort_rows: bool = True,
    spark_python: str | None = None,
    sort_slice_bytes: int | None = None,
) -> None:
    """Write one Parquet file per table under `output_dir` with a `_generation.json` manifest.

    `sort_rows=False` keeps the generator's row order and needs the Spark writer.
    """
    if not sort_rows and engine != "spark":
        raise ValueError(f"unsorted output needs the spark engine, got {engine}")
    if engine == "spark" and spark_python is None:
        raise ValueError("the spark engine needs spark_python")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    spark_versions: set[str] = set()
    temp_dir: Path | None = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp.", dir=output_dir.parent))
    try:
        for layout in layouts:
            assert temp_dir is not None
            source_file = raw_dir / f"{layout.name}.parquet"
            source_dir = raw_dir / layout.name
            if source_file.exists():
                source = source_file
            elif source_dir.exists():
                source = source_dir
            else:
                raise FileNotFoundError(f"missing generated table {layout.name} under {raw_dir}")
            print(f"optimizing {layout.name}", file=sys.stderr, flush=True)
            output_file = temp_dir / f"{layout.name}.parquet"
            if engine == "duckdb":
                optimize_table_with_duckdb(
                    source=source,
                    output_file=output_file,
                    layout=layout,
                    row_group_bytes=row_group_bytes,
                    compression=compression,
                    sort_slice_bytes=sort_slice_bytes,
                )
            elif engine == "spark":
                assert spark_python is not None
                spark_versions.add(
                    write_table_with_spark(
                        source=source,
                        output_file=output_file,
                        layout=layout,
                        sort_rows=sort_rows,
                        row_group_bytes=row_group_bytes,
                        compression=compression,
                        spark_python=spark_python,
                        sort_slice_bytes=sort_slice_bytes,
                    )
                )
            elif engine == "pyarrow":
                table = rename_table_fields(read_table(source), layout.field_renames)
                table = cast_table_fields(table, layout.field_casts)
                table = sort_table(table, layout.sort_keys)
                write_optimized_table(
                    table=table,
                    output_file=output_file,
                    sort_keys=layout.sort_keys,
                    row_group_bytes=row_group_bytes,
                    compression=compression,
                )
            else:
                raise ValueError(f"unsupported optimize engine: {engine}")

        assert temp_dir is not None
        if engine == "spark":
            java_versions = {parquet_java_version(temp_dir / f"{layout.name}.parquet") for layout in layouts}
            if len(spark_versions) != 1 or len(java_versions) != 1:
                raise RuntimeError(f"tables were written by Spark {sorted(spark_versions)} and parquet-java {sorted(java_versions)}")
            writer_settings: dict[str, object] = {
                "spark_version": spark_versions.pop(),
                "parquet_java_version": java_versions.pop(),
                "parquet_block_size": SPARK_PARQUET_BLOCK_SIZE,
                "parquet_page_size": SPARK_PARQUET_PAGE_SIZE,
                "parquet_page_row_count_limit": SPARK_PARQUET_PAGE_ROW_COUNT_LIMIT,
            }
        else:
            writer_settings = {"row_group_bytes": row_group_bytes}
        manifest = {
            "layout": "single_file_per_table_sorted_row_groups" if sort_rows else "single_file_per_table_unsorted",
            "scale_factor": int(scale) if scale.is_integer() else scale,
            "writer": engine,
            **writer_settings,
            "compression": compression.upper(),
            "parquet_sorting_columns_metadata": False,
            "tables": [
                {
                    "name": layout.name,
                    "sort_keys": list(layout.sort_keys) if sort_rows else [],
                    "field_renames": list(layout.field_renames),
                    "field_casts": list(layout.field_casts),
                }
                for layout in layouts
            ],
        }
        (temp_dir / "_generation.json").write_text(json.dumps(manifest, indent=2) + "\n")
        replace_with_empty_dir(output_dir, force=force)
        shutil.rmtree(output_dir)
        os.replace(temp_dir, output_dir)
        temp_dir = None
    finally:
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir)


def verify_layout(
    *,
    output_dir: Path,
    layouts: Sequence[TableLayout],
    compression: str,
    require_sorted: bool,
    expected_rows: dict[str, int] | None = None,
    writer: OptimizeEngine | None = None,
) -> None:
    """Check the files under `output_dir`; `writer` adds the checks of the writer that made them."""
    expected = {layout.name for layout in layouts}
    actual = {path.stem for path in output_dir.glob("*.parquet")}
    if actual != expected:
        raise RuntimeError(f"unexpected parquet table set in {output_dir}; missing={sorted(expected - actual)} extra={sorted(actual - expected)}")
    total_rows = 0
    total_row_groups = 0
    for layout in layouts:
        file = output_dir / f"{layout.name}.parquet"
        metadata = pq.ParquetFile(file).metadata
        if metadata.num_rows <= 0 or metadata.num_row_groups <= 0:
            raise RuntimeError(f"{file} has rows={metadata.num_rows} row_groups={metadata.num_row_groups}")
        if expected_rows is not None:
            expected_table_rows = expected_rows[layout.name]
            if metadata.num_rows != expected_table_rows:
                raise RuntimeError(f"{file} has rows={metadata.num_rows}, expected={expected_table_rows}")
        if metadata.num_row_groups > MAX_ROW_GROUPS_PER_FILE:
            raise RuntimeError(
                f"{file} has {metadata.num_row_groups} row groups; the parquet 59 footer decoder that DataFusion 55 "
                f"links reads at most {MAX_ROW_GROUPS_PER_FILE}"
            )
        total_rows += metadata.num_rows
        total_row_groups += metadata.num_row_groups
        verify_compression(file, metadata, compression)
        if writer == "spark":
            parquet_java_version(file)
            verify_page_row_limit(file, SPARK_PARQUET_PAGE_ROW_COUNT_LIMIT)
        if require_sorted:
            verify_sorted_row_groups(file, metadata, layout.sort_keys)
    print(f"verified {len(layouts)} tables, {total_rows} rows, {total_row_groups} row groups in {output_dir}", file=sys.stderr)


def verify_compression(file: Path, metadata: pq.FileMetaData, compression: str) -> None:
    expected = compression.upper()
    for rg_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(rg_index)
        for col_index in range(row_group.num_columns):
            actual = row_group.column(col_index).compression.upper()
            if actual != expected:
                raise RuntimeError(f"{file} row_group={rg_index} column={col_index} compression={actual}, expected={expected}")


def thrift_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def thrift_value(buf: bytes, pos: int, kind: int) -> tuple[object, int]:
    """Decode one Thrift compact-protocol value of type `kind` starting at `pos`."""
    if kind in (1, 2):
        return kind == 1, pos
    if kind == 3:
        return buf[pos], pos + 1
    if kind in (4, 5, 6):
        raw, pos = thrift_varint(buf, pos)
        return (raw >> 1) ^ -(raw & 1), pos
    if kind == 7:
        return buf[pos : pos + 8], pos + 8
    if kind == 8:
        size, pos = thrift_varint(buf, pos)
        return buf[pos : pos + size], pos + size
    if kind in (9, 10):
        header = buf[pos]
        pos += 1
        size, element = header >> 4, header & 0x0F
        if size == 15:
            size, pos = thrift_varint(buf, pos)
        items = []
        for _ in range(size):
            if element in (1, 2):
                items.append(buf[pos] == 1)
                pos += 1
            else:
                item, pos = thrift_value(buf, pos, element)
                items.append(item)
        return items, pos
    if kind == 11:
        size, pos = thrift_varint(buf, pos)
        if size == 0:
            return [], pos
        key_kind, value_kind = buf[pos] >> 4, buf[pos] & 0x0F
        pos += 1
        pairs = []
        for _ in range(size):
            key, pos = thrift_value(buf, pos, key_kind)
            value, pos = thrift_value(buf, pos, value_kind)
            pairs.append((key, value))
        return pairs, pos
    if kind == 12:
        return thrift_struct(buf, pos)
    raise ValueError(f"unknown Thrift compact type {kind} at byte {pos}")


def thrift_struct(buf: bytes, pos: int) -> tuple[dict[int, object], int]:
    """Decode a compact-protocol struct into a map from field id to value."""
    fields: dict[int, object] = {}
    field_id = 0
    while True:
        header = buf[pos]
        pos += 1
        if header == 0:
            return fields, pos
        delta, kind = header >> 4, header & 0x0F
        if delta:
            field_id += delta
        else:
            raw, pos = thrift_varint(buf, pos)
            field_id = (raw >> 1) ^ -(raw & 1)
        fields[field_id], pos = thrift_value(buf, pos, kind)


def read_footer(handle: BinaryIO) -> dict[int, object]:
    """Decode the footer's FileMetaData struct; field ids follow parquet.thrift (row_groups=4)."""
    handle.seek(-8, os.SEEK_END)
    tail = handle.read(8)
    if tail[4:] != b"PAR1":
        raise RuntimeError(f"{handle.name} does not end with the Parquet magic")
    footer_length = int.from_bytes(tail[:4], "little")
    handle.seek(-8 - footer_length, os.SEEK_END)
    footer, _ = thrift_struct(handle.read(footer_length), 0)
    return footer


def offset_index_page_rows(handle: BinaryIO, chunk: dict[int, object], rg_rows: int) -> list[int] | None:
    """Rows of each data page in a column chunk, or None when the chunk has no OffsetIndex.

    ColumnChunk.offset_index_offset=4 and offset_index_length=5, OffsetIndex.page_locations=1,
    PageLocation.first_row_index=3.
    """
    if 4 not in chunk or 5 not in chunk:
        return None
    handle.seek(chunk[4])
    offset_index, _ = thrift_struct(handle.read(chunk[5]), 0)
    first_rows = [location[3] for location in offset_index[1]] + [rg_rows]
    return [end - start for start, end in zip(first_rows, first_rows[1:])]


def page_header_rows(handle: BinaryIO, meta: dict[int, object]) -> list[int]:
    """Rows of each data page in a flat column chunk, read from the page headers.

    ColumnMetaData.total_compressed_size=7, data_page_offset=9 and dictionary_page_offset=11;
    PageHeader.compressed_page_size=3, data_page_header=5 (num_values=1) and
    data_page_header_v2=8 (num_rows=3). A dictionary or index page has neither header.
    """
    pos = min(offset for offset in (meta.get(9), meta.get(11)) if offset)
    end = pos + meta[7]
    rows: list[int] = []
    while pos < end:
        read = 1024
        while True:
            handle.seek(pos)
            try:
                header, header_length = thrift_struct(handle.read(read), 0)
                break
            except IndexError:
                read *= 4
        if 5 in header:
            rows.append(header[5][1])
        elif 8 in header:
            rows.append(header[8][3])
        pos += header_length + header[3]
    return rows


def verify_page_row_limit(file: Path, limit: int) -> None:
    """Check from the offset index that every data page holds at most `limit` rows.

    pyarrow does not expose the page index, so the footer and each column chunk's
    OffsetIndex are decoded here. Field ids follow parquet.thrift: FileMetaData.row_groups=4,
    RowGroup.columns=1 and num_rows=3, ColumnChunk.meta_data=3, ColumnMetaData.path_in_schema=3.
    """
    with open(file, "rb") as handle:
        footer = read_footer(handle)
        for rg_index, row_group in enumerate(footer[4]):
            for chunk in row_group[1]:
                column = ".".join(part.decode() for part in chunk[3][3])
                page_rows = offset_index_page_rows(handle, chunk, row_group[3])
                if page_rows is None:
                    raise RuntimeError(f"{file} row_group={rg_index} column={column} has no offset index")
                if not page_rows or max(page_rows) > limit or min(page_rows) <= 0:
                    raise RuntimeError(
                        f"{file} row_group={rg_index} column={column} has {len(page_rows)} data pages of "
                        f"{min(page_rows, default=0)} to {max(page_rows, default=0)} rows; each must hold 1 to {limit} rows"
                    )


def verify_sorted_row_groups(file: Path, metadata: pq.FileMetaData, sort_keys: Sequence[str]) -> None:
    if metadata.num_row_groups <= 1 or not sort_keys:
        return
    schema = pq.read_schema(file)
    missing = [candidate for candidate in sort_keys if candidate not in schema.names]
    if missing:
        raise RuntimeError(f"{file} is missing declared sort keys: {missing}")
    key = sort_keys[0]
    key_index = schema.names.index(key)
    previous_max = None
    seen_null_tail = False
    for rg_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(rg_index)
        stats = row_group.column(key_index).statistics
        if stats is None or not stats.has_min_max or stats.min is None or stats.max is None:
            if stats is not None and stats.null_count == row_group.num_rows:
                seen_null_tail = True
                continue
            raise RuntimeError(f"{file} row_group={rg_index} lacks min/max statistics for sort key {key}")
        if seen_null_tail:
            raise RuntimeError(f"{file} has non-null sort key stats after all-null tail row groups for {key}")
        if previous_max is not None and previous_max > stats.min:
            raise RuntimeError(f"{file} row groups are not monotonic on {key}: {previous_max!r} > {stats.min!r}")
        previous_max = stats.max


def distribution(values: Iterable[int]) -> dict[str, int]:
    """Minimum, lower median and maximum of `values`; the median is an observed value."""
    ordered = sorted(values)
    return {"min": ordered[0], "median": ordered[(len(ordered) - 1) // 2], "max": ordered[-1]}


def histogram_median_max(histogram: dict[int, int]) -> dict[str, int]:
    """Lower median and maximum of the values counted in `histogram`."""
    target = (sum(histogram.values()) - 1) // 2
    seen = 0
    for value in sorted(histogram):
        seen += histogram[value]
        if seen > target:
            return {"median": value, "max": max(histogram)}
    raise ValueError("empty histogram")


def non_overlapping_columns(metadata: pq.FileMetaData) -> list[str]:
    """Columns whose row-group [min, max] ranges are pairwise disjoint apart from shared bounds.

    A column that holds one value in the whole file, or lacks statistics in any row group
    that has non-null rows, is not listed.
    """
    if metadata.num_row_groups < 2:
        return []
    names = metadata.schema.to_arrow_schema().names
    columns = []
    for index, name in enumerate(names):
        ranges = []
        for rg_index in range(metadata.num_row_groups):
            row_group = metadata.row_group(rg_index)
            stats = row_group.column(index).statistics
            if stats is not None and stats.has_min_max:
                ranges.append((stats.min, stats.max))
            elif stats is None or stats.null_count != row_group.num_rows:
                ranges = []
                break
        if len(ranges) < 2 or min(low for low, _ in ranges) == max(high for _, high in ranges):
            continue
        ranges.sort()
        if all(previous[1] <= following[0] for previous, following in zip(ranges, ranges[1:])):
            columns.append(name)
    return columns


def describe_table_file(file: Path) -> dict[str, object]:
    """Measure one Parquet file from its footer, OffsetIndex and page headers."""
    metadata = pq.ParquetFile(file).metadata
    page_rows_histogram: dict[int, int] = {}
    pages_from_offset_index = pages_from_headers = 0
    dictionary_chunks = column_chunks = 0
    codecs: set[str] = set()
    with open(file, "rb") as handle:
        footer = read_footer(handle)
        for rg_index, row_group in enumerate(footer[4]):
            for col_index, chunk in enumerate(row_group[1]):
                column_chunks += 1
                column = metadata.row_group(rg_index).column(col_index)
                codecs.add(column.compression.upper())
                dictionary_chunks += column.has_dictionary_page
                page_rows = offset_index_page_rows(handle, chunk, row_group[3])
                if page_rows is not None:
                    pages_from_offset_index += len(page_rows)
                else:
                    page_rows = page_header_rows(handle, chunk[3])
                    pages_from_headers += len(page_rows)
                if sum(page_rows) != row_group[3]:
                    raise RuntimeError(f"{file} row_group={rg_index} column={col_index} pages hold {sum(page_rows)} rows, expected {row_group[3]}")
                for rows in page_rows:
                    page_rows_histogram[rows] = page_rows_histogram.get(rows, 0) + 1
    row_groups = [metadata.row_group(index) for index in range(metadata.num_row_groups)]
    if pages_from_offset_index and pages_from_headers:
        page_source = "mixed"
    else:
        page_source = "offset_index" if pages_from_offset_index else "page_headers"
    return {
        "file": file.name,
        "bytes": file.stat().st_size,
        "rows": metadata.num_rows,
        "columns": metadata.num_columns,
        "createdBy": metadata.created_by,
        "rowGroups": metadata.num_row_groups,
        "rowGroupRows": distribution(group.num_rows for group in row_groups),
        "rowGroupUncompressedBytes": distribution(group.total_byte_size for group in row_groups),
        "dataPages": pages_from_offset_index + pages_from_headers,
        "rowsPerPage": histogram_median_max(page_rows_histogram),
        "pageSource": page_source,
        "codecs": sorted(codecs),
        "dictionaryChunks": dictionary_chunks,
        "columnChunks": column_chunks,
        "nonOverlappingColumns": non_overlapping_columns(metadata),
    }


def describe_layout(parquet_dir: Path) -> dict[str, object]:
    """Describe a fixture directory written by `optimize_layout` and check it against its manifest.

    The result carries the manifest text of `_generation.json`, the partitioning derived
    from the directory contents, and the measurements of each table file. A mismatch
    between the manifest and the footers (table set, files per table, codec, writer
    version, page row limit, sort order) raises RuntimeError.
    """
    manifest = json.loads((parquet_dir / "_generation.json").read_text())
    expected_tables = [table["name"] for table in manifest["tables"]]
    entries = sorted(path for path in parquet_dir.iterdir())
    files = [path for path in entries if path.is_file() and path.suffix == ".parquet"]
    directories = [path.name for path in entries if path.is_dir()]
    if sorted(path.stem for path in files) != sorted(expected_tables) or len(files) != len(set(path.stem for path in files)):
        raise RuntimeError(f"{parquet_dir} holds {[path.name for path in files]}; the manifest lists one file for each of {expected_tables}")
    if directories or not str(manifest["layout"]).startswith("single_file_per_table"):
        raise RuntimeError(f"{parquet_dir} is not one file per table without subdirectories: layout={manifest['layout']} directories={directories}")
    tables = {path.stem: describe_table_file(path) for path in files}
    sorted_layout = manifest["layout"].endswith("sorted_row_groups")
    for table in manifest["tables"]:
        measured = tables[table["name"]]
        file = parquet_dir / measured["file"]
        if measured["codecs"] != [str(manifest["compression"]).upper()]:
            raise RuntimeError(f"{file} codecs {measured['codecs']} differ from manifest compression {manifest['compression']}")
        if manifest.get("writer") == "spark":
            if parquet_java_version(file) != manifest["parquet_java_version"]:
                raise RuntimeError(f"{file} was written by parquet-java {parquet_java_version(file)}, manifest says {manifest['parquet_java_version']}")
            limit = manifest["parquet_page_row_count_limit"]
            if measured["pageSource"] != "offset_index" or measured["rowsPerPage"]["max"] > limit:
                raise RuntimeError(f"{file} has pages of up to {measured['rowsPerPage']['max']} rows, manifest limit is {limit}")
        if sorted_layout:
            verify_sorted_row_groups(file, pq.ParquetFile(file).metadata, table["sort_keys"])
    return {
        "generation": manifest,
        "partitioning": {"kind": "single_file_per_table", "fileCount": len(files), "hiveKeys": []},
        "tables": tables,
    }


def print_summary(output_dirs: Iterable[Path]) -> None:
    for output_dir in output_dirs:
        parquet_files = sorted(output_dir.glob("*.parquet"))
        bytes_total = sum(path.stat().st_size for path in parquet_files)
        print(f"{output_dir}: {len(parquet_files)} parquet files, {bytes_total} bytes", file=sys.stderr)
