#!/usr/bin/env python3
"""Check the pinned TPC-H SF1 join prefetch OOM reproduction TSV."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_TPCH_TSV = Path("/tmp/bench-runs/tpch_sf1.tsv")
EXPECTED_MATCHED_COUNT = 11
EXPECTED_NATIVE_FAILED_COUNT = 11
EXPECTED_FAILED_QUERY_IDS = [
    "q03",
    "q05",
    "q07",
    "q08",
    "q09",
    "q10",
    "q12",
    "q13",
    "q14",
    "q16",
    "q17",
]
EXPECTED_MAX_SOURCE_READS_IN_FLIGHT = "2"

EXPECTED_FAILURE_FACTS = {
    "budget_boundary": "join_source_read_prefetch",
    "budget_enforcement": "hard_memory_resource_limit",
    "cudf_code": "OutOfMemory",
    "cudf_operation": "dataframe::read_parquet_source_chunk_with_budget",
    "native_operation": "cudf_query_engine.local_parquet_chunk.read",
    "pipeline_task_kind": "join_probe",
    "stage": "join_source_read",
}

REQUIRED_COLUMNS = {
    "query_id",
    "runtime_outcome",
    "bounded_status",
    "runtime_failure_code",
    "runtime_failure_kind",
    "runtime_failure_facts",
    "reservation_denied_count",
    "parity_status",
}


@dataclass(frozen=True)
class ReproSummary:
    matched_count: int
    native_failed_count: int
    failed_query_ids: list[str]
    representative_failure_query_id: str


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = check_repro_tsv(args.tpch_tsv)
    except ReproCheckError as error:
        print(error, file=sys.stderr)
        return 1

    print(
        "tpch_sf1_join_prefetch_repro "
        f"matched={summary.matched_count} "
        f"native_failed={summary.native_failed_count} "
        f"failed_queries={','.join(summary.failed_query_ids)} "
        f"max_source_reads_in_flight={EXPECTED_MAX_SOURCE_READS_IN_FLIGHT} "
        f"representative_failure={summary.representative_failure_query_id}"
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tpch-tsv",
        type=Path,
        default=DEFAULT_TPCH_TSV,
        help=f"Path to native_tpch_benchmark TSV output. Defaults to {DEFAULT_TPCH_TSV}.",
    )
    return parser.parse_args(argv)


def check_repro_tsv(path: Path) -> ReproSummary:
    comments, rows = read_benchmark_tsv(path)
    errors: list[str] = []
    if not rows:
        errors.append(f"{path}: no benchmark rows found")
        raise_if_errors(errors)

    columns = set(rows[0])
    missing_columns = sorted(REQUIRED_COLUMNS - columns)
    if missing_columns:
        errors.append(f"{path}: missing required columns: {','.join(missing_columns)}")
        raise_if_errors(errors)

    max_source_reads = comments.get("tpch_benchmark_native_max_source_reads_in_flight")
    if max_source_reads != EXPECTED_MAX_SOURCE_READS_IN_FLIGHT:
        errors.append(
            f"{path}: expected tpch_benchmark_native_max_source_reads_in_flight="
            f"{EXPECTED_MAX_SOURCE_READS_IN_FLIGHT}, found {display_value(max_source_reads)}"
        )

    matched_rows = [row for row in rows if row["parity_status"] == "matched"]
    native_failed_rows = [row for row in rows if row["parity_status"] == "native_failed"]
    failed_query_ids = [row["query_id"] for row in native_failed_rows]
    if len(matched_rows) != EXPECTED_MATCHED_COUNT:
        errors.append(f"{path}: expected matched={EXPECTED_MATCHED_COUNT}, found {len(matched_rows)}")
    if len(native_failed_rows) != EXPECTED_NATIVE_FAILED_COUNT:
        errors.append(f"{path}: expected native_failed={EXPECTED_NATIVE_FAILED_COUNT}, found {len(native_failed_rows)}")
    if failed_query_ids != EXPECTED_FAILED_QUERY_IDS:
        errors.append(
            f"{path}: expected failed query ids {','.join(EXPECTED_FAILED_QUERY_IDS)}, "
            f"found {','.join(failed_query_ids)}"
        )

    representative_failure_query_id = None
    for row in native_failed_rows:
        query_id = row["query_id"]
        require_equal(errors, path, query_id, "runtime_outcome", row["runtime_outcome"], "failed")
        require_equal(
            errors,
            path,
            query_id,
            "bounded_status",
            row["bounded_status"],
            "bounded_accepted_native_failed",
        )
        require_equal(
            errors,
            path,
            query_id,
            "runtime_failure_code",
            row["runtime_failure_code"],
            "cudf_resource_exhausted",
        )
        require_equal(
            errors,
            path,
            query_id,
            "runtime_failure_kind",
            row["runtime_failure_kind"],
            "resource_exhausted",
        )
        require_equal(errors, path, query_id, "reservation_denied_count", row["reservation_denied_count"], "0")
        facts = parse_facts(row["runtime_failure_facts"])
        for key, expected in EXPECTED_FAILURE_FACTS.items():
            require_equal(errors, path, query_id, f"runtime_failure_facts.{key}", facts.get(key), expected)
        if representative_failure_query_id is None and has_representative_budget_facts(facts):
            representative_failure_query_id = query_id

    if representative_failure_query_id is None:
        errors.append(
            f"{path}: no failing row had budget_current_bytes=0 and nonzero "
            "budget_largest_failed_allocation_bytes for join_source_read_prefetch"
        )

    raise_if_errors(errors)
    assert representative_failure_query_id is not None
    return ReproSummary(
        matched_count=len(matched_rows),
        native_failed_count=len(native_failed_rows),
        failed_query_ids=failed_query_ids,
        representative_failure_query_id=representative_failure_query_id,
    )


def read_benchmark_tsv(path: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    if not path.exists():
        raise ReproCheckError(f"{path}: file does not exist")
    comments: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        header: list[str] | None = None
        for line in handle:
            if line.startswith("#"):
                key, value = parse_comment(line)
                if key:
                    comments[key] = value
                continue
            header = line.rstrip("\n").split("\t")
            break
        if header is None:
            raise ReproCheckError(f"{path}: no TSV header found")
        rows = list(csv.DictReader(handle, fieldnames=header, delimiter="\t"))
    return comments, rows


def parse_comment(line: str) -> tuple[str | None, str]:
    stripped = line.removeprefix("#").strip()
    if "=" not in stripped:
        return None, ""
    key, value = stripped.split("=", 1)
    return key.strip(), value.strip()


def parse_facts(raw: str) -> dict[str, str]:
    facts: dict[str, str] = {}
    if raw in {"", "none"}:
        return facts
    for item in raw.split(";"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        facts[key] = value
    return facts


def has_representative_budget_facts(facts: dict[str, str]) -> bool:
    return (
        facts.get("budget_boundary") == "join_source_read_prefetch"
        and facts.get("pipeline_task_kind") == "join_probe"
        and facts.get("budget_current_bytes") == "0"
        and positive_integer(facts.get("budget_largest_failed_allocation_bytes"))
    )


def positive_integer(value: str | None) -> bool:
    try:
        return value is not None and int(value) > 0
    except ValueError:
        return False


def require_equal(
    errors: list[str],
    path: Path,
    query_id: str,
    field: str,
    actual: str | None,
    expected: str,
) -> None:
    if actual != expected:
        errors.append(f"{path}: {query_id}: expected {field}={expected}, found {display_value(actual)}")


def display_value(value: str | None) -> str:
    return "missing" if value is None else value


def raise_if_errors(errors: list[str]) -> None:
    if errors:
        raise ReproCheckError("\n".join(errors))


class ReproCheckError(Exception):
    pass


if __name__ == "__main__":
    raise SystemExit(main())
