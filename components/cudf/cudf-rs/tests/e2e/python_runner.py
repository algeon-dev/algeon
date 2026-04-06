#!/usr/bin/env python3
"""Python cuDF reference executor for E2E conformance tests.

Invoked by the Rust integration-test harness as:
    python python_runner.py --fixture <path> --case <case_id>

Reads the fixture JSON from disk, locates the referenced dataset and case,
executes the operation through Python cuDF, and writes a single normalized
JSON result object to stdout. Exits non-zero on any failure; diagnostics
go to stderr.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from typing import Any


# ---------------------------------------------------------------------------
# Dtype canonicalization
# ---------------------------------------------------------------------------

# Python cuDF dtype string -> manifest-canonical name
DTYPE_CANONICAL: dict[str, str] = {
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "uint16": "uint16",
    "uint32": "uint32",
    "uint64": "uint64",
    "float32": "float32",
    "float64": "float64",
    "bool": "bool8",
    "boolean": "bool8",
    "bool8": "bool8",
    "object": "string",
    "str": "string",
    "string": "string",
    "category": "dictionary32",
}


def canonicalize_dtype(dtype_str: str) -> str:
    """Map a Python cuDF dtype string to the manifest-canonical name."""
    normalized = dtype_str.strip().lower()
    canon = DTYPE_CANONICAL.get(normalized)
    if canon is None:
        if normalized.startswith("list") or "listdtype" in normalized:
            return "list"
        if normalized.startswith("struct") or "structdtype" in normalized:
            return "struct"
        if normalized.startswith("categorical") or normalized == "category":
            return "dictionary32"
        raise ValueError(f"Unmapped Python cuDF dtype: {dtype_str!r}")
    return canon


# Manifest-canonical dtype -> Python cuDF dtype constructor string
MANIFEST_TO_PYTHON_DTYPE: dict[str, str] = {
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "uint16": "uint16",
    "uint32": "uint32",
    "uint64": "uint64",
    "float32": "float32",
    "float64": "float64",
    "bool8": "bool",
    "string": "str",
}


# ---------------------------------------------------------------------------
# Payload / device-allocated byte computation
# (mirrors cudf-sys C++ shim; adapted from info_control.py)
# ---------------------------------------------------------------------------


def bitmask_payload_bytes(num_rows: int) -> int:
    """num_bitmask_words(size) * sizeof(bitmask_type) -- cudf uses uint32_t."""
    return ((num_rows + 31) // 32) * 4


def _dtype_size(dtype) -> int | None:
    """Byte size of a fixed-width dtype, or None for variable-width."""
    sizes = {
        "int8": 1, "int16": 2, "int32": 4, "int64": 8,
        "uint8": 1, "uint16": 2, "uint32": 4, "uint64": 8,
        "float32": 4, "float64": 8,
        "bool": 1,
    }
    return sizes.get(str(dtype))


def _string_chars_size(col) -> int:
    if col.data is not None:
        return col.data.size
    return 0


def _col_payload(col) -> int:
    """Recursive payload computation on an internal cudf column object."""
    total = 0
    dtype = col.dtype
    size = col.size

    if str(dtype) == "object":
        total += _string_chars_size(col)
    else:
        elem_size = _dtype_size(dtype)
        if elem_size is not None:
            total += size * elem_size

    if col.mask is not None:
        total += bitmask_payload_bytes(size)

    if hasattr(col, "children"):
        for child in col.children:
            total += _col_payload(child)

    return total


def column_payload_bytes(series: cudf.Series) -> int:
    return _col_payload(series._column)


def _col_device_allocated(col) -> int:
    total = 0
    if col.data is not None:
        total += col.data.size
    if col.mask is not None:
        total += col.mask.size
    if hasattr(col, "children"):
        for child in col.children:
            total += _col_device_allocated(child)
    return total


def column_device_allocated_bytes(series: cudf.Series) -> int:
    return _col_device_allocated(series._column)


def _import_cudf():
    import cudf

    return cudf


def _is_na(val: Any) -> bool:
    try:
        import pandas as pd

        return bool(pd.isna(val))
    except ImportError:
        return False
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def build_dataframe(dataset: dict) -> cudf.DataFrame:
    """Construct a cudf.DataFrame from a manifest dataset definition."""
    cudf = _import_cudf()
    columns = dataset.get("columns", [])
    json_lines = dataset.get("json_lines", [])

    if columns:
        data: dict[str, cudf.Series] = {}
        for col_def in columns:
            name = col_def["name"]
            manifest_dtype = col_def["dtype"]
            values = col_def["values"]

            py_dtype = MANIFEST_TO_PYTHON_DTYPE.get(manifest_dtype)
            if py_dtype is None:
                raise ValueError(f"Unsupported manifest dtype: {manifest_dtype!r}")

            if py_dtype == "str":
                data[name] = cudf.Series(values, dtype="str")
            elif py_dtype == "bool":
                data[name] = cudf.Series(values, dtype="bool")
            else:
                data[name] = cudf.Series(values, dtype=py_dtype)

        df = cudf.DataFrame(data)
    elif json_lines:
        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
                temp_path = handle.name
                for line in json_lines:
                    handle.write(line)
                    handle.write("\n")
            df = cudf.read_json(temp_path, lines=True)
        finally:
            if temp_path is not None and os.path.exists(temp_path):
                os.unlink(temp_path)
    else:
        raise ValueError("Dataset must define either 'columns' or 'json_lines'")

    for column_name in dataset.get("dictionary_columns", []):
        df[column_name] = df[column_name].astype("category")

    return df


# ---------------------------------------------------------------------------
# Value normalization for JSON output
# ---------------------------------------------------------------------------


def _normalize_value(val: Any) -> Any:
    """Convert a single cell value to a JSON-compatible Python type."""
    if val is None:
        return None
    # cudf NA / pandas NA
    if _is_na(val):
        return None
    # numpy / cudf scalar types
    t = type(val).__name__
    if "int" in t.lower():
        return int(val)
    if "float" in t.lower() or "double" in t.lower():
        f = float(val)
        if math.isnan(f):
            return None
        return f
    if "bool" in t.lower() and not isinstance(val, str):
        return bool(val)
    return str(val)


def normalize_dataframe(df: cudf.DataFrame) -> dict:
    """Convert a cudf.DataFrame to normalized result-schema JSON."""
    columns = []
    for col_name in df.columns:
        columns.append({
            "name": str(col_name),
            "dtype": canonicalize_dtype(str(df[col_name].dtype)),
        })

    rows: list[list[Any]] = []
    for i in range(len(df)):
        row: list[Any] = []
        for col_name in df.columns:
            series = df[col_name]
            # Check null first via the column's null mask
            if series.iloc[i] is None:
                row.append(None)
                continue
            if _is_na(series.iloc[i]):
                row.append(None)
                continue
            row.append(_normalize_value(series.iloc[i]))
        rows.append(row)

    return {
        "columns": columns,
        "row_count": len(df),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Operation dispatch
# ---------------------------------------------------------------------------


def execute_info(df: cudf.DataFrame, operation: dict) -> dict:
    """Execute an info operation and return an info_snapshot result."""
    raw_info = build_raw_info_snapshot(df)
    return project_info_detail(raw_info, operation.get("detail", []))


def build_raw_info_snapshot(df: cudf.DataFrame) -> dict:
    """Collect the full info snapshot before projecting requested detail fields."""
    col_names = list(df.columns)
    row_count = len(df)
    column_count = len(col_names)

    # Dtype counts (array of {"dtype": ..., "count": N} to match Rust Vec<DtypeCount>)
    dtype_count_map: dict[str, int] = {}
    for col_name in col_names:
        dt = canonicalize_dtype(str(df[col_name].dtype))
        dtype_count_map[dt] = dtype_count_map.get(dt, 0) + 1
    dtype_counts = [{"dtype": dt, "count": cnt} for dt, cnt in dtype_count_map.items()]

    # Per-column metadata (matches Rust NormalizedColumnInfo)
    columns_meta: list[dict] = []
    total_payload = 0
    total_device_allocated = 0

    for i, col_name in enumerate(col_names):
        series = df[col_name]
        nullable = series._column.mask is not None
        null_count = series.null_count
        non_null_count = row_count - null_count
        payload = column_payload_bytes(series)
        device_alloc = column_device_allocated_bytes(series)
        total_payload += payload
        total_device_allocated += device_alloc

        columns_meta.append({
            "name": str(col_name),
            "dtype": canonicalize_dtype(str(series.dtype)),
            "nullable": nullable,
            "null_count": null_count,
            "non_null_count": non_null_count,
            "payload_bytes": payload,
            "device_allocated_bytes": device_alloc,
        })

    return {
        "row_count": row_count,
        "column_count": column_count,
        "dtype_counts": dtype_counts,
        "columns": columns_meta,
        "payload_bytes": total_payload,
        "device_allocated_bytes": total_device_allocated,
    }


def project_info_detail(raw_info: dict, detail_flags: list[str]) -> dict:
    """Project a full info snapshot to match Rust `info_with(detail)` output shape."""
    detail_set = set(detail_flags)
    include_columns = "columns" in detail_set
    include_null_counts = "null_counts" in detail_set
    include_payload = "payload_bytes" in detail_set
    include_device = "device_allocated" in detail_set

    projected_columns: list[dict] = []
    if include_columns:
        for col in raw_info["columns"]:
            projected_columns.append({
                "name": col["name"],
                "dtype": col["dtype"],
                "nullable": col["nullable"],
                "null_count": col["null_count"] if include_null_counts else None,
                "non_null_count": col["non_null_count"] if include_null_counts else None,
                "payload_bytes": col["payload_bytes"] if include_payload else None,
                "device_allocated_bytes": col["device_allocated_bytes"] if include_device else None,
            })

    return {
        "result_kind": "info_snapshot",
        "row_count": raw_info["row_count"],
        "column_count": raw_info["column_count"],
        "dtype_counts": raw_info["dtype_counts"],
        "columns": projected_columns,
        "payload_bytes": raw_info["payload_bytes"] if include_payload else None,
        "device_allocated_bytes": raw_info["device_allocated_bytes"] if include_device else None,
    }


def execute_schema(df: cudf.DataFrame, operation: dict) -> dict:
    """Execute a schema operation and return a column_list result."""
    entries = []
    for col_name in df.columns:
        entries.append({
            "name": str(col_name),
            "dtype": canonicalize_dtype(str(df[col_name].dtype)),
        })
    return {
        "result_kind": "column_list",
        "entries": entries,
    }


def execute_dtypes(df: cudf.DataFrame, operation: dict) -> dict:
    """Execute a dtypes operation and return a column_list result."""
    # Same shape as schema
    return execute_schema(df, operation)


def execute_select(df: cudf.DataFrame, operation: dict) -> dict:
    """Execute a select operation and return a dataframe result."""
    columns = operation["columns"]
    result_df = df[columns]
    return {
        "result_kind": "dataframe",
        **normalize_dataframe(result_df),
    }


def execute_rename(df: cudf.DataFrame, operation: dict) -> dict:
    """Execute a rename operation and return a dataframe result."""
    mapping = operation["mapping"]
    result_df = df.rename(columns=mapping)
    return {
        "result_kind": "dataframe",
        **normalize_dataframe(result_df),
    }


def execute_sort_values(df: cudf.DataFrame, operation: dict) -> dict:
    """Execute a sort_values operation and return a dataframe result."""
    by = operation["by"]
    ascending = operation.get("ascending", [True] * len(by))
    na_position = operation.get("null_position", "last")
    result_df = df.sort_values(by=by, ascending=ascending, na_position=na_position)
    result_df = result_df.reset_index(drop=True)
    return {
        "result_kind": "dataframe",
        **normalize_dataframe(result_df),
    }


def execute_filter_eq(df: cudf.DataFrame, operation: dict) -> dict:
    """Execute a filter_eq operation and return a dataframe result."""
    col = operation["column"]
    value = operation["value"]
    value_dtype = operation.get("value_dtype")

    # Cast the comparison value to the appropriate type
    if value_dtype == "string" or isinstance(value, str):
        cmp_value = str(value)
    elif value_dtype in ("float32", "float64"):
        cmp_value = float(value)
    elif value_dtype in ("bool8", "bool", "boolean"):
        cmp_value = bool(value)
    else:
        # integer types
        cmp_value = int(value) if value is not None else value

    mask = df[col] == cmp_value
    result_df = df[mask].reset_index(drop=True)
    return {
        "result_kind": "dataframe",
        **normalize_dataframe(result_df),
    }


def execute_drop_nulls(df: cudf.DataFrame, operation: dict) -> dict:
    """Execute a drop_nulls operation and return a dataframe result."""
    subset = operation.get("subset")
    result_df = df.dropna(subset=subset).reset_index(drop=True)
    return {
        "result_kind": "dataframe",
        **normalize_dataframe(result_df),
    }


def execute_head(df: cudf.DataFrame, operation: dict) -> dict:
    """Execute a head operation and return a dataframe result."""
    n = operation["n"]
    result_df = df.head(n)
    return {
        "result_kind": "dataframe",
        **normalize_dataframe(result_df),
    }


def execute_tail(df: cudf.DataFrame, operation: dict) -> dict:
    """Execute a tail operation and return a dataframe result."""
    n = operation["n"]
    result_df = df.tail(n).reset_index(drop=True)
    return {
        "result_kind": "dataframe",
        **normalize_dataframe(result_df),
    }


# Operation kind -> executor function
OPERATION_DISPATCH: dict[str, Any] = {
    "info": execute_info,
    "schema": execute_schema,
    "dtypes": execute_dtypes,
    "select": execute_select,
    "rename": execute_rename,
    "sort_values": execute_sort_values,
    "filter_eq": execute_filter_eq,
    "drop_nulls": execute_drop_nulls,
    "head": execute_head,
    "tail": execute_tail,
}


def execute_operation(df: cudf.DataFrame, operation: dict) -> dict:
    """Dispatch an operation and return a normalized result dict."""
    kind = operation["kind"]
    executor = OPERATION_DISPATCH.get(kind)
    if executor is None:
        raise ValueError(f"Unsupported operation kind: {kind!r}")
    return executor(df, operation)


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


def classify_error(exc: Exception) -> str:
    """Map a Python exception to a stable error kind string."""
    name = type(exc).__name__
    msg = str(exc).lower()

    if isinstance(exc, KeyError) or "not found" in msg or "not in" in msg:
        return "column_not_found"
    if isinstance(exc, (TypeError, ValueError)):
        return "invalid_argument"
    if isinstance(exc, IndexError):
        return "index_out_of_bounds"
    return "unknown"


def _sample_raw_info_snapshot() -> dict:
    return {
        "row_count": 4,
        "column_count": 1,
        "dtype_counts": [{"dtype": "int32", "count": 1}],
        "columns": [{
            "name": "a",
            "dtype": "int32",
            "nullable": True,
            "null_count": 1,
            "non_null_count": 3,
            "payload_bytes": 16,
            "device_allocated_bytes": 64,
        }],
        "payload_bytes": 16,
        "device_allocated_bytes": 64,
    }


def run_selftest(args: argparse.Namespace) -> int:
    if args.selftest == "classify-error":
        error_name = args.error or "runtime_error"
        if error_name == "key_error":
            exc = KeyError("missing")
        elif error_name == "type_error":
            exc = TypeError("bad type")
        elif error_name == "index_error":
            exc = IndexError("bad index")
        else:
            exc = RuntimeError("boom")
        json.dump({"kind": classify_error(exc)}, sys.stdout, allow_nan=False)
        sys.stdout.write("\n")
        return 0

    if args.selftest == "info-detail":
        detail_flags = [] if args.detail is None else [part for part in args.detail.split(",") if part]
        json.dump(project_info_detail(_sample_raw_info_snapshot(), detail_flags), sys.stdout, allow_nan=False)
        sys.stdout.write("\n")
        return 0

    print(f"Unsupported self-test: {args.selftest!r}", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Python cuDF reference executor for E2E conformance tests."
    )
    parser.add_argument("--selftest", help="Run a Python-runner self-test that does not require Python cuDF")
    parser.add_argument("--detail", help="Comma-separated info detail flags for --selftest info-detail")
    parser.add_argument("--error", help="Named error kind for --selftest classify-error")
    parser.add_argument("--fixture", help="Path to fixture JSON file")
    parser.add_argument("--case", help="Case ID to execute")
    args = parser.parse_args()

    if args.selftest is not None:
        sys.exit(run_selftest(args))

    if args.fixture is None or args.case is None:
        parser.error("--fixture and --case are required unless --selftest is used")

    # Load fixture
    try:
        with open(args.fixture) as f:
            fixture = json.load(f)
    except Exception as exc:
        print(f"Failed to load fixture: {exc}", file=sys.stderr)
        sys.exit(1)

    # Build dataset lookup
    datasets = {ds["id"]: ds for ds in fixture.get("datasets", [])}

    # Find case
    cases = fixture.get("cases", [])
    case = None
    for c in cases:
        if c["id"] == args.case:
            case = c
            break

    if case is None:
        print(f"Case not found: {args.case!r}", file=sys.stderr)
        sys.exit(1)

    # Find dataset
    dataset_id = case["dataset"]
    dataset = datasets.get(dataset_id)
    if dataset is None:
        print(f"Dataset not found: {dataset_id!r}", file=sys.stderr)
        sys.exit(1)

    # Build DataFrame
    try:
        df = build_dataframe(dataset)
    except Exception as exc:
        print(f"Failed to build DataFrame: {exc}", file=sys.stderr)
        sys.exit(1)

    # Execute operation
    operation = case["operation"]
    expect = case.get("expect", {})
    result_kind = expect.get("result_kind")

    if result_kind == "error":
        # We expect an error from this operation
        try:
            execute_operation(df, operation)
            # If no error raised, report that
            result = {
                "result_kind": "error",
                "kind": "expected_error_not_raised",
                "operation": operation["kind"],
                "context": [],
            }
        except Exception as exc:
            result = {
                "result_kind": "error",
                "kind": classify_error(exc),
                "operation": operation["kind"],
                "context": [],
            }
    else:
        # Normal execution
        try:
            result = execute_operation(df, operation)
        except Exception as exc:
            print(
                f"Operation failed unexpectedly: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Write result to stdout
    json.dump(result, sys.stdout, allow_nan=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
