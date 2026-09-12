#!/usr/bin/env python3
"""Summarize CUDA activity in one Nsight Systems SQLite export.

All timestamps are reported in nanoseconds.  GPU activity for the activity
span and idle-window calculation is the union of kernel and memcpy records;
CUDA memset records are intentionally not included in that definition.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path


class TraceError(RuntimeError):
    """Raised when an export does not contain the data needed to summarize it."""


MEMCPY_KIND_LABELS = ("H2D", "D2H", "D2D", "other")
MEMCPY_KIND_NAMES = {
    "CUDA_MEMCPY_KIND_HTOD": "H2D",
    "CUDA_MEMCPY_KIND_UVM_HTOD": "H2D",
    "CUDA_MEMCPY_KIND_DTOH": "D2H",
    "CUDA_MEMCPY_KIND_UVM_DTOH": "D2H",
    "CUDA_MEMCPY_KIND_DTOD": "D2D",
    "CUDA_MEMCPY_KIND_UVM_DTOD": "D2D",
}


def table_exists(connection: sqlite3.Connection, name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND lower(name) = lower(?)",
        (name,),
    ).fetchone()
    return row is not None


def table_columns(connection: sqlite3.Connection, name: str) -> set[str]:
    return {row[1] for row in connection.execute(f'PRAGMA table_info("{name}")')}


def column_name(columns: set[str], wanted: str) -> str | None:
    wanted_lower = wanted.lower()
    return next((column for column in columns if column.lower() == wanted_lower), None)


def require_columns(connection: sqlite3.Connection, name: str, columns: set[str]) -> None:
    if not table_exists(connection, name):
        raise TraceError(f"SQLite export is missing table {name}")
    available = table_columns(connection, name)
    missing = {column for column in columns if column_name(available, column) is None}
    if missing:
        names = ", ".join(sorted(missing))
        raise TraceError(f"table {name} is missing columns: {names}")


def load_gpu_events(connection: sqlite3.Connection, table: str) -> list[tuple[int, int, int, int | None]]:
    """Return (start, end, stream_id, correlation_id) rows from a GPU table."""

    if not table_exists(connection, table):
        return []
    columns = table_columns(connection, table)
    require = {"start", "end", "streamId"}
    missing = {column for column in require if column_name(columns, column) is None}
    if missing:
        names = ", ".join(sorted(missing))
        raise TraceError(f"table {table} is missing columns: {names}")
    correlation = ", correlationId" if column_name(columns, "correlationId") else ", NULL"
    query = f'SELECT start, end, streamId{correlation} FROM "{table}"'
    events = []
    for start, end, stream_id, correlation_id in connection.execute(query):
        events.append((int(start), int(end), int(stream_id), correlation_id))
    return events


def load_memcpy_events(
    connection: sqlite3.Connection,
) -> tuple[list[tuple[int, int, int, int | None, int | None, int | None]], dict[str, object]]:
    """Return memcpy rows and the optional kind/byte schema metadata.

    The kind metadata is kept separate from the existing GPU-event tuple so
    kernel and memcpy activity calculations retain their original inputs.
    """

    table = "CUPTI_ACTIVITY_KIND_MEMCPY"
    if not table_exists(connection, table):
        return [], {
            "kind_mapping_available": False,
            "kind_mapping_reason": f"table {table} is unavailable",
            "bytes_available": False,
        }
    columns = table_columns(connection, table)
    require_columns(connection, table, {"start", "end", "streamId"})
    has_correlation = column_name(columns, "correlationId") is not None
    has_copy_kind = column_name(columns, "copyKind") is not None
    has_bytes = column_name(columns, "bytes") is not None
    correlation = ", correlationId" if has_correlation else ", NULL"
    copy_kind = ", copyKind" if has_copy_kind else ", NULL"
    bytes_count = ", bytes" if has_bytes else ", NULL"
    query = f'SELECT start, end, streamId{correlation}{copy_kind}{bytes_count} FROM "{table}"'
    events = []
    for start, end, stream_id, correlation_id, copy_kind_value, bytes_value in connection.execute(query):
        events.append(
            (
                int(start),
                int(end),
                int(stream_id),
                correlation_id,
                None if copy_kind_value is None else int(copy_kind_value),
                None if bytes_value is None else int(bytes_value),
            )
        )

    enum_table = "ENUM_CUDA_MEMCPY_OPER"
    if not has_copy_kind:
        kind_mapping_available = False
        kind_mapping_reason = f"table {table} has no copyKind column"
    elif not table_exists(connection, enum_table):
        kind_mapping_available = False
        kind_mapping_reason = f"table {enum_table} is unavailable"
    else:
        enum_columns = table_columns(connection, enum_table)
        has_enum_id = column_name(enum_columns, "id") is not None
        has_enum_name = column_name(enum_columns, "name") is not None
        if not has_enum_id or not has_enum_name:
            missing = [name for name, present in (("id", has_enum_id), ("name", has_enum_name)) if not present]
            kind_mapping_available = False
            kind_mapping_reason = f"table {enum_table} is missing columns: {', '.join(missing)}"
        else:
            kind_mapping_available = True
            kind_mapping_reason = None

    metadata = {
        "kind_mapping_available": kind_mapping_available,
        "kind_mapping_reason": kind_mapping_reason,
        "bytes_available": has_bytes,
    }
    if kind_mapping_available:
        enum_columns = table_columns(connection, enum_table)
        enum_id = column_name(enum_columns, "id")
        enum_name = column_name(enum_columns, "name")
        assert enum_id is not None
        assert enum_name is not None
        metadata["kind_names"] = {
            int(kind_id): str(kind_name)
            for kind_id, kind_name in connection.execute(f'SELECT "{enum_id}", "{enum_name}" FROM "{enum_table}"')
        }
    return events, metadata


def load_runtime_events(connection: sqlite3.Connection) -> list[tuple[int, int, int | None, str]]:
    """Return (start, end, correlation_id, API name) runtime rows."""

    if not table_exists(connection, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        return []
    require_columns(connection, "CUPTI_ACTIVITY_KIND_RUNTIME", {"start", "end", "nameId"})
    has_correlation = column_name(table_columns(connection, "CUPTI_ACTIVITY_KIND_RUNTIME"), "correlationId") is not None
    correlation = "r.correlationId" if has_correlation else "NULL"
    name_join = "LEFT JOIN StringIds s ON s.id = r.nameId" if table_exists(connection, "StringIds") else ""
    name = "COALESCE(s.value, CAST(r.nameId AS TEXT))" if name_join else "CAST(r.nameId AS TEXT)"
    query = (
        f"SELECT r.start, r.end, {correlation}, {name} "
        f'FROM "CUPTI_ACTIVITY_KIND_RUNTIME" r {name_join}'
    )
    return [(int(start), int(end), correlation_id, str(api_name)) for start, end, correlation_id, api_name in connection.execute(query)]


def positive_intervals(events: list[tuple]) -> list[tuple[int, int]]:
    return [(event[0], event[1]) for event in events if event[1] > event[0]]


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def interval_total(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in intervals)


def span_summary(intervals: list[tuple[int, int]]) -> dict[str, int | None]:
    if not intervals:
        return {"first_start_ns": None, "last_end_ns": None, "span_ns": 0}
    first = min(start for start, _end in intervals)
    last = max(end for _start, end in intervals)
    return {"first_start_ns": first, "last_end_ns": last, "span_ns": last - first}


def stream_distribution(events: list[tuple]) -> dict[str, int]:
    counts = Counter(event[2] for event in events)
    return {str(stream_id): count for stream_id, count in sorted(counts.items())}


def interval_intersection(first: list[tuple[int, int]], second: list[tuple[int, int]]) -> int:
    total = 0
    first_index = second_index = 0
    while first_index < len(first) and second_index < len(second):
        start = max(first[first_index][0], second[second_index][0])
        end = min(first[first_index][1], second[second_index][1])
        if end > start:
            total += end - start
        if first[first_index][1] < second[second_index][1]:
            first_index += 1
        else:
            second_index += 1
    return total


def memcpy_kind_summary(
    events: list[tuple[int, int, int, int | None, int | None, int | None]],
    metadata: dict[str, object],
) -> dict[str, object]:
    if not metadata["kind_mapping_available"]:
        return {
            "available": False,
            "reason": metadata["kind_mapping_reason"],
            "by_kind": None,
        }

    kind_names = metadata["kind_names"]
    assert isinstance(kind_names, dict)
    events_by_kind: dict[str, list[tuple[int, int, int | None]]] = {label: [] for label in MEMCPY_KIND_LABELS}
    for start, end, _stream_id, _correlation_id, copy_kind, bytes_count in events:
        kind_name = kind_names.get(copy_kind)
        label = MEMCPY_KIND_NAMES.get(kind_name, "other")
        events_by_kind[label].append((start, end, bytes_count))

    by_kind = {}
    for label in MEMCPY_KIND_LABELS:
        kind_events = events_by_kind[label]
        item: dict[str, object] = {
            "count": len(kind_events),
            "busy_union_ns": interval_total(merge_intervals([(start, end) for start, end, _bytes in kind_events])),
        }
        if metadata["bytes_available"]:
            item["bytes"] = sum(bytes_count or 0 for _start, _end, bytes_count in kind_events)
        by_kind[label] = item

    return {
        "available": True,
        "source": "CUPTI_ACTIVITY_KIND_MEMCPY.copyKind + ENUM_CUDA_MEMCPY_OPER.name",
        "by_kind": by_kind,
    }


def idle_window_host_submission_summary(
    activity_events: list[tuple[int, int, int | None, str]],
    activity_union: list[tuple[int, int]],
    runtime_events: list[tuple[int, int, int | None, str]],
) -> dict[str, object]:
    """Summarize idle windows whose next GPU event has a matching CUDA API.

    A matching API is evidence about host submission timing.  It does not
    establish that the whole idle window is a host delay because the API can
    overlap the preceding GPU work or other causes can account for the gap.
    """

    runtime_by_correlation: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for start, end, correlation_id, name in runtime_events:
        if correlation_id is not None:
            runtime_by_correlation[int(correlation_id)].append((start, end, name))

    evidence = []
    positive_events = [event for event in activity_events if event[1] > event[0]]
    for previous, current in zip(activity_union, activity_union[1:]):
        previous_gpu_end = previous[1]
        next_gpu_start = current[0]
        if next_gpu_start <= previous_gpu_end:
            continue
        next_events = sorted(
            (event for event in positive_events if event[0] == next_gpu_start),
            key=lambda event: (event[1], event[3], -1 if event[2] is None else event[2]),
        )
        if not next_events:
            continue
        next_event_and_apis = [
            (event, runtime_by_correlation.get(int(event[2]), []))
            for event in next_events
            if event[2] is not None and runtime_by_correlation.get(int(event[2]), [])
        ]
        if not next_event_and_apis:
            continue
        next_event, api_candidates = next_event_and_apis[0]
        api_start, api_end, api_name = min(
            api_candidates,
            key=lambda api: (abs(next_gpu_start - api[1]), api[0], api[1], api[2]),
        )
        gap_ns = next_gpu_start - previous_gpu_end
        api_overlap_ns = max(0, min(api_end, next_gpu_start) - max(api_start, previous_gpu_end))
        evidence.append(
            {
                "gap_ns": gap_ns,
                "gap_start_ns": previous_gpu_end,
                "gap_end_ns": next_gpu_start,
                "next_activity_kind": next_event[3],
                "api_name": api_name,
                "api_start_ns": api_start,
                "api_end_ns": api_end,
                "api_start_after_previous_gpu_end_ns": api_start - previous_gpu_end,
                "api_end_before_next_gpu_start_ns": next_gpu_start - api_end,
                "api_overlap_ns": api_overlap_ns,
            }
        )

    largest = max(evidence, key=lambda item: (item["gap_ns"], -item["gap_start_ns"])) if evidence else None
    api_start_deltas = [item["api_start_after_previous_gpu_end_ns"] for item in evidence]
    return {
        "window_count": len(evidence),
        "total_ns": sum(item["gap_ns"] for item in evidence),
        "largest_gap_ns": None if largest is None else largest["gap_ns"],
        "largest_gap_start_ns": None if largest is None else largest["gap_start_ns"],
        "largest_gap_end_ns": None if largest is None else largest["gap_end_ns"],
        "largest_gap_next_activity_kind": None if largest is None else largest["next_activity_kind"],
        "largest_gap_api_name": None if largest is None else largest["api_name"],
        "largest_gap_api_start_after_previous_gpu_end_ns": (
            None if largest is None else largest["api_start_after_previous_gpu_end_ns"]
        ),
        "largest_gap_api_overlap_ns": None if largest is None else largest["api_overlap_ns"],
        "api_overlap_window_count": sum(item["api_overlap_ns"] > 0 for item in evidence),
        "api_overlap_total_ns": sum(item["api_overlap_ns"] for item in evidence),
        "api_start_after_previous_gpu_end_ns": number_summary(api_start_deltas),
        "interpretation": (
            "A correlated CUDA API identifies host submission evidence for the next GPU event; "
            "the idle window is not classified as a pure host delay because API overlap and other causes remain possible."
        ),
    }


def at_least_two_overlap(intervals: list[tuple[int, int]]) -> int:
    """Return time covered by at least two intervals."""

    points = defaultdict(int)
    for start, end in intervals:
        if end > start:
            points[start] += 1
            points[end] -= 1
    active = 0
    previous = None
    total = 0
    for timestamp in sorted(points):
        if previous is not None and active >= 2:
            total += timestamp - previous
        active += points[timestamp]
        previous = timestamp
    return total


def idle_summary(activity: list[tuple[int, int]]) -> dict[str, object]:
    merged = merge_intervals(activity)
    windows = [(previous[1], current[0]) for previous, current in zip(merged, merged[1:]) if current[0] > previous[1]]
    largest = max(windows, key=lambda window: window[1] - window[0], default=None)
    return {
        "window_count": len(windows),
        "total_ns": sum(end - start for start, end in windows),
        "largest_ns": None if largest is None else largest[1] - largest[0],
        "largest_start_ns": None if largest is None else largest[0],
        "largest_end_ns": None if largest is None else largest[1],
    }


def number_summary(values: list[int]) -> dict[str, int | None]:
    if not values:
        return {"count": 0, "min_ns": None, "max_ns": None, "sum_ns": 0, "negative_count": 0}
    return {
        "count": len(values),
        "min_ns": min(values),
        "max_ns": max(values),
        "sum_ns": sum(values),
        "negative_count": sum(value < 0 for value in values),
    }


def is_kernel_launch_api(name: str) -> bool:
    lowered = name.lower()
    return "launch" in lowered and "hostfunc" not in lowered and "launchhostfunc" not in lowered


def launch_evidence(
    runtime_events: list[tuple[int, int, int | None, str]],
    kernel_events: list[tuple[int, int, int, int | None]],
) -> dict[str, object]:
    launch_events = [event for event in runtime_events if is_kernel_launch_api(event[3])]
    kernels_by_correlation: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for start, end, _stream, correlation_id in kernel_events:
        if correlation_id is not None:
            kernels_by_correlation[int(correlation_id)].append((start, end))

    per_name: dict[str, dict[str, int]] = {}
    delays = []
    correlated_api_count = 0
    correlated_kernel_count = 0
    launch_correlation_ids = set()
    for start, end, correlation_id, name in launch_events:
        item = per_name.setdefault(name, {"call_count": 0, "api_time_ns": 0, "correlated_call_count": 0})
        item["call_count"] += 1
        item["api_time_ns"] += max(0, end - start)
        matches = kernels_by_correlation.get(int(correlation_id), []) if correlation_id is not None else []
        if matches:
            correlated_api_count += 1
            correlated_kernel_count += len(matches)
            launch_correlation_ids.add(int(correlation_id))
            item["correlated_call_count"] += 1
            delays.extend(kernel_start - end for kernel_start, _kernel_end in matches)

    kernel_count_with_launch = sum(
        len(kernels) for correlation_id, kernels in kernels_by_correlation.items() if correlation_id in launch_correlation_ids
    )
    unmatched_kernel_count = len(kernel_events) - kernel_count_with_launch
    return {
        "launch_api_count": len(launch_events),
        "launch_api_time_ns": sum(max(0, end - start) for start, end, _correlation, _name in launch_events),
        "launch_api_names": {name: per_name[name] for name in sorted(per_name)},
        "correlated_api_count": correlated_api_count,
        "correlated_kernel_count": correlated_kernel_count,
        "unmatched_launch_api_count": len(launch_events) - correlated_api_count,
        "unmatched_kernel_count": int(unmatched_kernel_count),
        "kernel_start_minus_api_end_ns": number_summary(delays),
        "interpretation": (
            "Evidence uses CUDA launch API timestamps joined by correlationId; "
            "GPU idle windows are not classified as host launch gaps."
        ),
    }


def synchronize_summary(runtime_events: list[tuple[int, int, int | None, str]]) -> dict[str, object]:
    sync_events = [event for event in runtime_events if re.search(r"stream.*synchroniz", event[3], re.IGNORECASE)]
    by_name: dict[str, dict[str, int]] = {}
    for _start, end, _correlation, name in sync_events:
        # The runtime table records a complete API interval, including any wait.
        item = by_name.setdefault(name, {"call_count": 0, "api_time_ns": 0})
        item["call_count"] += 1
        item["api_time_ns"] += max(0, end - _start)
    return {
        "api_call_count": len(sync_events),
        "api_time_ns": sum(item["api_time_ns"] for item in by_name.values()),
        "api_names": {name: by_name[name] for name in sorted(by_name)},
    }


def summarize(path: Path) -> dict[str, object]:
    try:
        connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    except sqlite3.Error as error:
        raise TraceError(f"cannot open SQLite export {path}: {error}") from error
    try:
        kernels = load_gpu_events(connection, "CUPTI_ACTIVITY_KIND_KERNEL")
        memcpy, memcpy_metadata = load_memcpy_events(connection)
        runtime = load_runtime_events(connection)
    finally:
        connection.close()

    kernel_intervals = positive_intervals(kernels)
    memcpy_intervals = positive_intervals(memcpy)
    kernel_union = merge_intervals(kernel_intervals)
    memcpy_union = merge_intervals(memcpy_intervals)
    activity_union = merge_intervals(kernel_intervals + memcpy_intervals)
    activity_span = span_summary(activity_union)
    activity_events = [
        (start, end, correlation_id, "kernel") for start, end, _stream_id, correlation_id in kernels
    ] + [
        (start, end, correlation_id, "memcpy") for start, end, _stream_id, correlation_id, _copy_kind, _bytes in memcpy
    ]
    idle_windows = idle_summary(activity_union)
    idle_windows["host_submission_evidence"] = idle_window_host_submission_summary(
        activity_events,
        activity_union,
        runtime,
    )
    memcpy_summary: dict[str, object] = {
        "count": len(memcpy),
        "stream_distribution": stream_distribution(memcpy),
        "busy_union_ns": interval_total(memcpy_union),
        **span_summary(memcpy_intervals),
        "kind_summary": memcpy_kind_summary(memcpy, memcpy_metadata),
    }
    if memcpy_metadata["bytes_available"]:
        memcpy_summary["bytes"] = sum(
            bytes_count or 0
            for _start, _end, _stream_id, _correlation_id, _copy_kind, bytes_count in memcpy
        )

    return {
        "trace": str(path),
        "time_unit": "ns",
        "activity_definition": ["CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY"],
        "kernels": {
            "count": len(kernels),
            "stream_distribution": stream_distribution(kernels),
            "busy_union_ns": interval_total(kernel_union),
            **span_summary(kernel_intervals),
        },
        "memcpy": memcpy_summary,
        "gpu_activity": {
            **activity_span,
            "busy_union_ns": interval_total(activity_union),
            "idle_windows": idle_windows,
        },
        "overlap": {
            "kernel_kernel_ns": at_least_two_overlap(kernel_intervals),
            "kernel_memcpy_ns": interval_intersection(kernel_union, memcpy_union),
        },
        "cuda_stream_synchronization": synchronize_summary(runtime),
        "host_launch_evidence": launch_evidence(runtime, kernels),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite_export", type=Path, help="Nsight Systems SQLite export")
    args = parser.parse_args(argv)
    try:
        report = summarize(args.sqlite_export)
    except (OSError, sqlite3.Error, TraceError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
