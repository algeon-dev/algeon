#!/usr/bin/env python3
"""Shared Parquet layout helpers for TPC fixture generation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


DEFAULT_ROW_GROUP_BYTES = 8 * 1024 * 1024


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
    scales = [float(value) for value in values] if values else [1.0]
    for scale in scales:
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
    return scales


def resolve_output_dir(*, workload: str, scale: float, out: Path | None, out_root: Path | None) -> Path:
    if out is not None:
        return out
    root = out_root if out_root is not None else repo_root() / "fixture" / "data" / workload
    return root / scale_label(scale) / "parquet"


def ensure_safe_replace_dir(path: Path) -> None:
    resolved = path.resolve()
    root = repo_root().resolve()
    protected = {
        Path("/").resolve(),
        Path.home().resolve(),
        root,
        root / "fixture",
        root / "fixture" / "data",
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


def optimize_layout(
    *,
    raw_dir: Path,
    output_dir: Path,
    layouts: Sequence[TableLayout],
    row_group_bytes: int,
    compression: str,
    force: bool,
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
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
            table = rename_table_fields(read_table(source), layout.field_renames)
            table = cast_table_fields(table, layout.field_casts)
            table = sort_table(table, layout.sort_keys)
            write_optimized_table(
                table=table,
                output_file=temp_dir / f"{layout.name}.parquet",
                sort_keys=layout.sort_keys,
                row_group_bytes=row_group_bytes,
                compression=compression,
            )

        assert temp_dir is not None
        manifest = {
            "layout": "single_file_per_table_sorted_row_groups",
            "row_group_bytes": row_group_bytes,
            "compression": compression.upper(),
            "parquet_sorting_columns_metadata": False,
            "tables": [
                {
                    "name": layout.name,
                    "sort_keys": list(layout.sort_keys),
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
) -> None:
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
        total_rows += metadata.num_rows
        total_row_groups += metadata.num_row_groups
        verify_compression(file, metadata, compression)
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


def print_summary(output_dirs: Iterable[Path]) -> None:
    for output_dir in output_dirs:
        parquet_files = sorted(output_dir.glob("*.parquet"))
        bytes_total = sum(path.stat().st_size for path in parquet_files)
        print(f"{output_dir}: {len(parquet_files)} parquet files, {bytes_total} bytes", file=sys.stderr)
