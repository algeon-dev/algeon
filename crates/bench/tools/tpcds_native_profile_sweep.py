#!/usr/bin/env python3
"""Run a TPC-DS native GPU source-profile sweep.

The sweep drives `tpcds_gpu_cpu_bench` with a matrix of bounded native source
profiles, writes one raw TSV per candidate, and emits a compact summary TSV
using median measured wall time per query. Defaults encode the native-profile
analysis matrix.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import signal
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_QUERIES = ",".join(
    [
        "q44",
        "q96",
        "q82",
        "q88",
        "q23",
        "q14",
        "q24",
        "q58",
        "q64",
        "q03",
        "q34",
        "q68",
        "q73",
        "q31",
        "q74",
        "q95",
        "q72",
        "q09",
        "q22",
    ]
)
DEFAULT_ROW_GROUPS = "1,8,32,128,512,1024"
DEFAULT_SOURCE_CHUNK_BYTES = "64MiB,512MiB,2GiB,8GiB,32GiB"
DEFAULT_DEVICE_BUDGETS = "auto,32GiB,64GiB,95GiB"
SUMMARY_HEADER = [
    "candidate_id",
    "device_budget_label",
    "device_budget_bytes_env",
    "max_source_chunk_bytes",
    "max_row_groups_per_chunk",
    "source_chunk_planning_mode",
    "queries",
    "warmups",
    "iterations",
    "return_code",
    "outcome",
    "successful_queries",
    "partial_queries",
    "failed_queries",
    "failed_query_ids",
    "partial_query_ids",
    "total_median_wall_ms",
    "max_reservation_live_peak_bytes",
    "total_reservation_denied_count",
    "total_retry_count",
    "total_source_chunks_committed",
    "total_row_groups_committed",
    "raw_tsv",
    "stderr_log",
]
NATIVE_ENV_OWNED_BY_SWEEP = (
    "ALGEON_TPCDS_NATIVE_PROFILE",
    "ALGEON_TPCDS_NATIVE_DEVICE_BUDGET_BYTES",
    "ALGEON_TPCDS_NATIVE_MAX_SOURCE_CHUNK_BYTES",
    "ALGEON_TPCDS_NATIVE_MAX_ROW_GROUPS_PER_CHUNK",
    "ALGEON_TPCDS_NATIVE_SOURCE_CHUNK_PLANNING_MODE",
    "ALGEON_TPCDS_NATIVE_MAX_RETRY_ATTEMPTS",
    "ALGEON_TPCDS_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS",
)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    device_budget_label: str
    device_budget_bytes: int | None
    max_source_chunk_bytes: int
    max_row_groups_per_chunk: int


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = build_candidates(args)
    write_candidate_manifest(out_dir / "candidates.tsv", candidates)
    if args.dry_run:
        print(f"candidate_manifest={out_dir / 'candidates.tsv'}")
        return 0

    summary_rows = []
    for index, candidate in enumerate(candidates, start=1):
        print(f"[tpcds_native_profile_sweep] candidate {index}/{len(candidates)} {candidate.candidate_id}", file=sys.stderr)
        summary_rows.append(run_candidate(args, candidate, out_dir))
        write_summary(out_dir / "summary.tsv", summary_rows)
    print(f"summary={out_dir / 'summary.tsv'}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-bin", type=Path, default=Path("target/release/tpcds_gpu_cpu_bench"))
    parser.add_argument("--parquet-dir", type=Path, default=os.environ.get("ALGEON_PARQUET_DIR"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--queries", default=DEFAULT_QUERIES)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--row-groups", default=DEFAULT_ROW_GROUPS)
    parser.add_argument("--source-chunk-bytes", default=DEFAULT_SOURCE_CHUNK_BYTES)
    parser.add_argument("--device-budgets", default=DEFAULT_DEVICE_BUDGETS)
    parser.add_argument("--candidate-timeout-secs", type=float, default=900.0)
    parser.add_argument("--gpu-lock", type=Path, default=Path("/tmp/cudf-gpu.lock"))
    parser.add_argument("--no-gpu-lock", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-staleness-check", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.parquet_dir is None:
        parser.error("--parquet-dir is required when ALGEON_PARQUET_DIR is unset")
    if args.warmups < 0:
        parser.error("--warmups must be >= 0")
    if args.iterations <= 0:
        parser.error("--iterations must be > 0")
    return args


def build_candidates(args: argparse.Namespace) -> list[Candidate]:
    row_groups = parse_int_list(args.row_groups, "--row-groups")
    source_bytes = parse_size_list(args.source_chunk_bytes, "--source-chunk-bytes")
    budgets = parse_budget_list(args.device_budgets)
    candidates = []
    for budget_label, budget_bytes in budgets:
        for chunk_bytes in source_bytes:
            for row_group_count in row_groups:
                candidates.append(
                    Candidate(
                        candidate_id=(
                            f"budget{safe_label(budget_label)}_src{size_label(chunk_bytes)}_rg{row_group_count}"
                        ),
                        device_budget_label=budget_label,
                        device_budget_bytes=budget_bytes,
                        max_source_chunk_bytes=chunk_bytes,
                        max_row_groups_per_chunk=row_group_count,
                    )
                )
    return candidates


def run_candidate(args: argparse.Namespace, candidate: Candidate, out_dir: Path) -> dict[str, str]:
    raw_tsv = out_dir / f"{candidate.candidate_id}.tsv"
    stderr_log = out_dir / f"{candidate.candidate_id}.stderr"
    env = candidate_env(args, candidate)
    command = command_for_candidate(args)
    return_code = 0
    try:
        with raw_tsv.open("w", encoding="utf-8") as stdout, stderr_log.open("w", encoding="utf-8") as stderr:
            completed = run_with_process_group_timeout(
                command,
                env=env,
                stdout=stdout,
                stderr=stderr,
                timeout=args.candidate_timeout_secs,
            )
        return_code = completed.returncode
    except subprocess.TimeoutExpired:
        return_code = 124
        with stderr_log.open("a", encoding="utf-8") as stderr:
            stderr.write(f"\n[tpcds_native_profile_sweep] timeout_secs={args.candidate_timeout_secs}\n")
    return summarize_candidate(candidate, raw_tsv, stderr_log, return_code, args)


def candidate_env(args: argparse.Namespace, candidate: Candidate) -> dict[str, str]:
    env = os.environ.copy()
    for name in NATIVE_ENV_OWNED_BY_SWEEP:
        env.pop(name, None)
    env.update(
        {
            "ALGEON_PARQUET_DIR": str(args.parquet_dir),
            "ALGEON_TPCDS_BENCH_ENGINE": "native_gpu",
            "ALGEON_TPCDS_BENCH_QUERIES": args.queries,
            "ALGEON_TPCDS_BENCH_WARMUPS": str(args.warmups),
            "ALGEON_TPCDS_BENCH_ITERATIONS": str(args.iterations),
            "ALGEON_TPCDS_NATIVE_PROFILE": "conservative",
            "ALGEON_TPCDS_NATIVE_MAX_SOURCE_CHUNK_BYTES": str(candidate.max_source_chunk_bytes),
            "ALGEON_TPCDS_NATIVE_MAX_ROW_GROUPS_PER_CHUNK": str(candidate.max_row_groups_per_chunk),
            "ALGEON_TPCDS_NATIVE_SOURCE_CHUNK_PLANNING_MODE": "conservative",
        }
    )
    if args.skip_staleness_check:
        env["CUDF_RS_SKIP_STALENESS_CHECK"] = "1"
    if candidate.device_budget_bytes is None:
        env.pop("ALGEON_TPCDS_NATIVE_DEVICE_BUDGET_BYTES", None)
    else:
        env["ALGEON_TPCDS_NATIVE_DEVICE_BUDGET_BYTES"] = str(candidate.device_budget_bytes)
    return env


def command_for_candidate(args: argparse.Namespace) -> list[str]:
    bench = str(args.bench_bin)
    if args.no_gpu_lock:
        return [bench]
    return ["flock", str(args.gpu_lock), bench]


def run_with_process_group_timeout(
    command: list[str],
    env: dict[str, str],
    stdout,
    stderr,
    timeout: float,
) -> subprocess.CompletedProcess:
    process = subprocess.Popen(
        command,
        env=env,
        stdout=stdout,
        stderr=stderr,
        text=True,
        start_new_session=True,
    )
    try:
        return_code = process.wait(timeout=timeout)
        return subprocess.CompletedProcess(command, return_code)
    except subprocess.TimeoutExpired:
        terminate_process_group(process)
        raise


def terminate_process_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def summarize_candidate(
    candidate: Candidate,
    raw_tsv: Path,
    stderr_log: Path,
    return_code: int,
    args: argparse.Namespace,
) -> dict[str, str]:
    rows = read_benchmark_rows(raw_tsv)
    by_query: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row.get("warmup", "").lower() == "true":
            continue
        by_query.setdefault(row.get("query_id", ""), []).append(row)
    query_ids = parse_queries(args.queries)
    successful = 0
    partial_query_ids = []
    failed_query_ids = []
    total_median_wall = 0.0
    max_reservation = 0
    total_denied = 0
    total_retry = 0
    total_source_chunks = 0
    total_row_groups = 0
    for query_id in query_ids:
        query_rows = by_query.get(query_id, [])
        succeeded = [row for row in query_rows if row.get("outcome") == "succeeded" and parse_float(row.get("wall_ms")) is not None]
        if len(succeeded) != args.iterations:
            failed_query_ids.append(query_id)
            if succeeded:
                partial_query_ids.append(query_id)
            continue
        successful += 1
        total_median_wall += statistics.median(parse_float(row["wall_ms"]) for row in succeeded)
        max_reservation = max(max_reservation, max(parse_int(row.get("reservation_live_peak_bytes")) for row in succeeded))
        total_denied += sum(parse_int(row.get("reservation_denied_count")) for row in succeeded)
        total_retry += sum(parse_int(row.get("retry_count")) for row in succeeded)
        total_source_chunks += sum(parse_int(row.get("source_chunks_committed")) for row in succeeded)
        total_row_groups += sum(parse_int(row.get("row_groups_committed")) for row in succeeded)
    outcome = "succeeded" if return_code == 0 and not failed_query_ids and not partial_query_ids else "failed"
    return {
        "candidate_id": candidate.candidate_id,
        "device_budget_label": candidate.device_budget_label,
        "device_budget_bytes_env": "" if candidate.device_budget_bytes is None else str(candidate.device_budget_bytes),
        "max_source_chunk_bytes": str(candidate.max_source_chunk_bytes),
        "max_row_groups_per_chunk": str(candidate.max_row_groups_per_chunk),
        "source_chunk_planning_mode": "conservative",
        "queries": ",".join(query_ids),
        "warmups": str(args.warmups),
        "iterations": str(args.iterations),
        "return_code": str(return_code),
        "outcome": outcome,
        "successful_queries": str(successful),
        "partial_queries": str(len(partial_query_ids)),
        "failed_queries": str(len(failed_query_ids)),
        "failed_query_ids": ",".join(failed_query_ids),
        "partial_query_ids": ",".join(partial_query_ids),
        "total_median_wall_ms": f"{total_median_wall:.3f}" if successful else "",
        "max_reservation_live_peak_bytes": str(max_reservation),
        "total_reservation_denied_count": str(total_denied),
        "total_retry_count": str(total_retry),
        "total_source_chunks_committed": str(total_source_chunks),
        "total_row_groups_committed": str(total_row_groups),
        "raw_tsv": str(raw_tsv),
        "stderr_log": str(stderr_log),
    }


def read_benchmark_rows(path: Path) -> list[dict[str, str]]:
    rows = []
    header = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            cells = line.rstrip("\n").split("\t")
            if cells and cells[0] == "engine":
                header = cells
                continue
            if header is None:
                continue
            rows.append({column: cells[index] if index < len(cells) else "" for index, column in enumerate(header)})
    return rows


def write_candidate_manifest(path: Path, candidates: list[Candidate]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "candidate_id",
                "device_budget_label",
                "device_budget_bytes_env",
                "max_source_chunk_bytes",
                "max_row_groups_per_chunk",
                "source_chunk_planning_mode",
            ]
        )
        for candidate in candidates:
            writer.writerow(
                [
                    candidate.candidate_id,
                    candidate.device_budget_label,
                    "" if candidate.device_budget_bytes is None else candidate.device_budget_bytes,
                    candidate.max_source_chunk_bytes,
                    candidate.max_row_groups_per_chunk,
                    "conservative",
                ]
            )


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_HEADER, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_queries(value: str) -> list[str]:
    return [query.strip() for query in value.split(",") if query.strip()]


def parse_int_list(value: str, name: str) -> list[int]:
    out = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parsed = int(item)
        if parsed <= 0:
            raise ValueError(f"{name} values must be positive: {item}")
        out.append(parsed)
    if not out:
        raise ValueError(f"{name} must contain at least one value")
    return out


def parse_size_list(value: str, name: str) -> list[int]:
    out = [parse_size(item.strip()) for item in value.split(",") if item.strip()]
    if not out:
        raise ValueError(f"{name} must contain at least one value")
    return out


def parse_budget_list(value: str) -> list[tuple[str, int | None]]:
    out = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item.lower() == "auto":
            out.append(("auto", None))
        else:
            out.append((item, parse_size(item)))
    if not out:
        raise ValueError("--device-budgets must contain at least one value")
    return out


SIZE_RE = re.compile(r"^(\d+)([KMGT]i?B?)?$", re.IGNORECASE)
SIZE_MULTIPLIERS = {
    None: 1,
    "B": 1,
    "K": 1024,
    "KB": 1024,
    "KIB": 1024,
    "M": 1024**2,
    "MB": 1024**2,
    "MIB": 1024**2,
    "G": 1024**3,
    "GB": 1024**3,
    "GIB": 1024**3,
    "T": 1024**4,
    "TB": 1024**4,
    "TIB": 1024**4,
}


def parse_size(value: str) -> int:
    match = SIZE_RE.match(value)
    if not match:
        raise ValueError(f"invalid byte size {value!r}")
    number = int(match.group(1))
    suffix = match.group(2)
    multiplier = SIZE_MULTIPLIERS[suffix.upper() if suffix else None]
    parsed = number * multiplier
    if parsed <= 0:
        raise ValueError(f"byte size must be positive: {value}")
    return parsed


def size_label(value: int) -> str:
    for suffix, divisor in (("TiB", 1024**4), ("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)):
        if value % divisor == 0:
            return f"{value // divisor}{suffix}"
    return str(value)


def safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", value)


def parse_int(value: str | None) -> int:
    try:
        return int(value or 0)
    except ValueError:
        return 0


def parse_float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


if __name__ == "__main__":
    sys.exit(main())
