#!/usr/bin/env python3
"""Prepare ClickBench cross-engine raw outputs for nondeterministic parity triage.

The default query set is the 12 reason-coded parity-waiver queries plus q23.
Rows are written in two machine-readable forms:

* rows.tsv
* rows.jsonl

Each row represents one engine/query/iteration and points at the raw TSV output
used to compute the script's order-insensitive fingerprint.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_reference_REPO = REPO_ROOT.parent / "reference"
DEFAULT_reference_BIN = DEFAULT_reference_REPO / "build" / "release" / "duckdb"
GPU_LOCK = Path("/tmp/cudf-gpu.lock")

WAIVER_QUERY_IDS = (
    "q04",
    "q10",
    "q18",
    "q25",
    "q26",
    "q31",
    "q32",
    "q33",
    "q39",
    "q40",
    "q41",
    "q42",
)
DEFAULT_QUERY_IDS = (*WAIVER_QUERY_IDS, "q23")

OUTPUT_HEADER = [
    "query_id",
    "engine",
    "iteration",
    "status",
    "row_count",
    "fingerprint",
    "elapsed_ms",
    "raw_output_path",
    "native_rewritten",
    "error",
]


@dataclass
class Query:
    query_id: str
    sql: str


@dataclass
class OutputRow:
    query_id: str
    engine: str
    iteration: int
    status: str
    row_count: str
    fingerprint: str
    elapsed_ms: str
    raw_output_path: str
    native_rewritten: str = "n/a"
    error: str = "none"


def main() -> int:
    args = parse_args()
    query_ids = parse_query_ids(args.query_ids)
    queries = load_clickbench_queries(args.repo_root)
    unknown = [query_id for query_id in query_ids if query_id not in queries]
    if unknown:
        raise SystemExit(f"unknown ClickBench query ids: {','.join(unknown)}")

    engines = parse_engines(args.engines)
    fixture = discover_fixture(args.fixture, args.repo_root)
    out_dir = args.out_dir or default_out_dir()
    raw_dir = args.raw_dir or out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows: list[OutputRow] = []
    if "duckdb" in engines:
        rows.extend(run_duckdb_outputs(args, fixture, raw_dir, [queries[q] for q in query_ids]))
    if "reference" in engines:
        rows.extend(run_reference_outputs(args, fixture, out_dir, raw_dir, [queries[q] for q in query_ids]))

    datafusion_engines = [engine for engine in engines if engine in {"cpu_datafusion", "datafusion_nexus"}]
    if datafusion_engines:
        rows.extend(run_datafusion_outputs(args, fixture, out_dir, raw_dir, [queries[q] for q in query_ids], datafusion_engines))

    tsv_path = args.tsv or out_dir / "rows.tsv"
    jsonl_path = args.jsonl or out_dir / "rows.jsonl"
    write_tsv(tsv_path, rows)
    write_jsonl(jsonl_path, rows)
    write_manifest(out_dir / "manifest.json", args, fixture, query_ids, engines, tsv_path, jsonl_path)

    print(f"clickbench_cross_engine_output_tsv={tsv_path}")
    print(f"clickbench_cross_engine_output_jsonl={jsonl_path}")
    print(f"clickbench_cross_engine_raw_dir={raw_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--fixture", type=Path, help="ClickBench Parquet file or directory. Defaults to CUDF_CLICKBENCH_PARQUET_DIR or local/NVMe defaults.")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--tsv", type=Path)
    parser.add_argument("--jsonl", type=Path)
    parser.add_argument("--query-ids", default=",".join(DEFAULT_QUERY_IDS))
    parser.add_argument(
        "--engines",
        default="duckdb,reference,cpu_datafusion,datafusion_nexus",
        help="Comma-separated subset of duckdb,reference,cpu_datafusion,datafusion_nexus.",
    )
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--duckdb-bin", default=os.environ.get("DUCKDB", "duckdb"))
    parser.add_argument("--cargo-bin", default=os.environ.get("CARGO", "cargo"))
    parser.add_argument("--release", action="store_true", help="Use cargo run --release for DataFusion/datafusion-nexus.")
    parser.add_argument("--no-gpu-flock", action="store_true", help="Do not wrap native GPU runs with flock /tmp/cudf-gpu.lock.")
    parser.add_argument("--native-source-chunk-bytes", default=os.environ.get("CUDF_CLICKBENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES", "4294967296"))
    parser.add_argument("--native-row-groups-per-chunk", default=os.environ.get("CUDF_CLICKBENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK", "1024"))
    parser.add_argument("--reference-bin", type=Path, default=Path(os.environ.get("reference_DUCKDB_BIN", DEFAULT_reference_BIN)))
    parser.add_argument("--reference-db", type=Path, help="DuckDB database used by reference. Defaults under --out-dir.")
    parser.add_argument("--reference-load", action="store_true", help="Drop/recreate the reference hits table from the Parquet fixture before running.")
    parser.add_argument(
        "--reference-config-file",
        type=Path,
        default=Path(os.environ.get("reference_CONFIG_FILE", "")),
        help=(
            "Path to a reference YAML config file for transparent GPU execution "
            "(sets reference_CONFIG_FILE). "
            "The new reference API routes SQL through the GPU when this is set; "
            "the old gpu_buffer_init/gpu_processing API is no longer available."
        ),
    )
    # Legacy arguments kept for backward compatibility; ignored when --reference-config-file is set.
    parser.add_argument("--reference-caching-region-size", default="19 GB")
    parser.add_argument("--reference-processing-region-size", default="19 GB")
    parser.add_argument("--reference-pinned-memory-size")
    args = parser.parse_args()
    if args.iterations < 1:
        raise SystemExit("--iterations must be >= 1")
    return args


def parse_query_ids(raw: str) -> list[str]:
    if raw == "waivers_plus_q23":
        return list(DEFAULT_QUERY_IDS)
    query_ids = [part.strip() for part in raw.split(",") if part.strip()]
    return dedup_preserving_order(query_ids)


def parse_engines(raw: str) -> list[str]:
    valid = {"duckdb", "reference", "cpu_datafusion", "datafusion_nexus"}
    engines = dedup_preserving_order([part.strip() for part in raw.split(",") if part.strip()])
    unknown = [engine for engine in engines if engine not in valid]
    if unknown:
        raise SystemExit(f"unknown engines: {','.join(unknown)}")
    return engines


def dedup_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def load_clickbench_queries(repo_root: Path) -> dict[str, Query]:
    source = (
        repo_root / "crates" / "datafusion-nexus-bench" / "src" / "workload" / "clickbench_queries.rs"
    ).read_text(encoding="utf-8")
    pattern = re.compile(
        r'ClickbenchQuery\s*\{\s*id:\s*"(?P<id>q\d{2})",\s*sql:\s*r#"(?:\n)?(?P<sql>.*?)(?:\n)?"#\s*,?\s*\}',
        re.DOTALL,
    )
    queries: dict[str, Query] = {}
    for match in pattern.finditer(source):
        query_id = match.group("id")
        queries[query_id] = Query(query_id=query_id, sql=match.group("sql"))
    if len(queries) != 43:
        raise RuntimeError(f"expected to load 43 ClickBench queries, loaded {len(queries)}")
    return queries


def discover_fixture(explicit: Path | None, repo_root: Path) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    if env := os.environ.get("CUDF_CLICKBENCH_PARQUET_DIR"):
        candidates.append(Path(env))
    candidates.extend(
        [
            Path("/opt/dlami/nvme/clickbench/partitioned"),
            repo_root / "tests" / "data" / "clickbench" / "partitioned",
            repo_root / "tests" / "data" / "clickbench" / "hits.parquet",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    rendered = ", ".join(str(path) for path in candidates)
    raise SystemExit(f"no ClickBench fixture found; checked {rendered}")


def default_out_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("/tmp") / f"clickbench_cross_engine_outputs_{stamp}"


def run_duckdb_outputs(args: argparse.Namespace, fixture: Path, raw_dir: Path, queries: list[Query]) -> list[OutputRow]:
    rows: list[OutputRow] = []
    for query in queries:
        for iteration in range(1, args.iterations + 1):
            raw_path = raw_dir / f"{query.query_id}_duckdb_iter{iteration}.tsv"
            sql_path = raw_dir / f"{query.query_id}_duckdb_iter{iteration}.sql"
            sql_path.write_text(duckdb_copy_sql(fixture, query.sql, raw_path), encoding="utf-8")
            started = time.monotonic()
            proc = subprocess.run(
                [args.duckdb_bin, "-batch", "-bail", ":memory:", "-f", str(sql_path)],
                cwd=args.repo_root,
                text=True,
                capture_output=True,
                timeout=args.timeout_seconds,
            )
            elapsed_ms = monotonic_elapsed_ms(started)
            rows.append(row_from_process(query.query_id, "duckdb", iteration, proc, raw_path, elapsed_ms))
    return rows


def duckdb_copy_sql(fixture: Path, query_sql: str, raw_path: Path) -> str:
    return "\n".join(
        [
            duckdb_setup_sql(fixture),
            f"COPY ({query_sql}) TO '{quote_sql_literal(raw_path)}' (FORMAT CSV, HEADER true, DELIMITER '\\t', NULL 'NULL');",
            "",
        ]
    )


def duckdb_setup_sql(fixture: Path) -> str:
    source = duckdb_read_parquet_source(fixture)
    return "\n".join(
        [
            f"CREATE OR REPLACE VIEW hits_raw AS SELECT * FROM {source};",
            "CREATE OR REPLACE VIEW hits AS "
            "SELECT * EXCLUDE (EventDate), DATE '1970-01-01' + CAST(EventDate AS INTEGER) AS EventDate FROM hits_raw;",
        ]
    )


def duckdb_read_parquet_source(fixture: Path) -> str:
    if fixture.is_file():
        parquet_arg = quote_sql_literal(fixture)
    else:
        if (fixture / "hits.parquet").is_file():
            parquet_arg = quote_sql_literal(fixture / "hits.parquet")
        elif list(fixture.glob("hits_*.parquet")):
            parquet_arg = quote_sql_literal(fixture / "hits_*.parquet")
        elif list((fixture / "partitioned").glob("hits_*.parquet")):
            parquet_arg = quote_sql_literal(fixture / "partitioned" / "hits_*.parquet")
        else:
            raise SystemExit(f"no ClickBench hits parquet files under {fixture}")
    return f"read_parquet('{parquet_arg}', binary_as_string=true)"


def run_reference_outputs(
    args: argparse.Namespace,
    fixture: Path,
    out_dir: Path,
    raw_dir: Path,
    queries: list[Query],
) -> list[OutputRow]:
    reference_db = (args.reference_db or out_dir / "reference_clickbench.duckdb").resolve()
    if args.reference_load or not reference_db.exists():
        load_reference_database(args, fixture, reference_db, raw_dir)

    rows: list[OutputRow] = []
    for query in queries:
        for iteration in range(1, args.iterations + 1):
            raw_path = raw_dir / f"{query.query_id}_reference_iter{iteration}.tsv"
            sql_path = raw_dir / f"{query.query_id}_reference_iter{iteration}.sql"
            sql_path.write_text(reference_query_sql(args, query.sql, raw_path), encoding="utf-8")
            started = time.monotonic()
            proc = subprocess.run(
                [str(args.reference_bin), str(reference_db), "-batch", "-bail", "-f", str(sql_path)],
                cwd=args.repo_root,
                text=True,
                capture_output=True,
                timeout=args.timeout_seconds,
                env=reference_env(args),
            )
            elapsed_ms = monotonic_elapsed_ms(started)
            rows.append(row_from_process(query.query_id, "reference", iteration, proc, raw_path, elapsed_ms))
    return rows


def load_reference_database(args: argparse.Namespace, fixture: Path, reference_db: Path, raw_dir: Path) -> None:
    reference_db.parent.mkdir(parents=True, exist_ok=True)
    load_sql = "\n".join(
        [
            "DROP TABLE IF EXISTS hits;",
            f"CREATE TABLE hits AS {duckdb_setup_select_sql(fixture)};",
            "",
        ]
    )
    load_path = raw_dir / "reference_load_clickbench.sql"
    load_path.write_text(load_sql, encoding="utf-8")
    proc = subprocess.run(
        [str(args.reference_bin), str(reference_db), "-batch", "-bail", "-f", str(load_path)],
        cwd=args.repo_root,
        text=True,
        capture_output=True,
        timeout=args.timeout_seconds,
        env=reference_env(),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"failed to load reference ClickBench DB {reference_db}: {compact_process_error(proc)}")


def duckdb_setup_select_sql(fixture: Path) -> str:
    source = duckdb_read_parquet_source(fixture)
    return f"SELECT * EXCLUDE (EventDate), DATE '1970-01-01' + CAST(EventDate AS INTEGER) AS EventDate FROM {source}"


def reference_query_sql(args: argparse.Namespace, query_sql: str, raw_path: Path) -> str:
    if _reference_config_file(args):
        # New reference API: transparent GPU execution routed via reference_CONFIG_FILE.
        # Plain SQL — no gpu_buffer_init/gpu_processing calls needed.
        return "\n".join(
            [
                ".mode tabs",
                ".headers on",
                ".nullvalue NULL",
                f".once {raw_path}",
                f"{query_sql};",
                "",
            ]
        )
    # Legacy reference API (gpu_buffer_init + gpu_processing).
    pinned = ""
    if args.reference_pinned_memory_size:
        pinned = f", pinned_memory_size = '{quote_sql_literal(args.reference_pinned_memory_size)}'"
    return "\n".join(
        [
            ".mode tabs",
            ".headers on",
            ".nullvalue NULL",
            f"call gpu_buffer_init('{quote_sql_literal(args.reference_caching_region_size)}', '{quote_sql_literal(args.reference_processing_region_size)}'{pinned});",
            f".once {raw_path}",
            f"call gpu_processing('{quote_sql_literal(query_sql)}');",
            "",
        ]
    )


def _reference_config_file(args: argparse.Namespace) -> Path | None:
    p = getattr(args, "reference_config_file", None)
    if p is None or str(p) == "":
        return None
    return p


def reference_env(args: argparse.Namespace | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("reference_DISABLE", "1")
    if args is not None:
        config = _reference_config_file(args)
        if config is not None:
            env["reference_CONFIG_FILE"] = str(config)
    return env


def run_datafusion_outputs(
    args: argparse.Namespace,
    fixture: Path,
    out_dir: Path,
    raw_dir: Path,
    queries: list[Query],
    engines: list[str],
) -> list[OutputRow]:
    dump_dir = raw_dir / "datafusion_dump"
    dump_dir.mkdir(parents=True, exist_ok=True)
    query_ids = ",".join(query.query_id for query in queries)
    env = os.environ.copy()
    env.update(
        {
            "CUDF_CLICKBENCH_PARQUET_DIR": str(fixture),
            "CUDF_CLICKBENCH_BENCH_QUERY_IDS": query_ids,
            "CUDF_CLICKBENCH_BENCH_WARMUPS": "0",
            "CUDF_CLICKBENCH_BENCH_ITERATIONS": str(args.iterations),
            "CUDF_CLICKBENCH_BENCH_DUMP_ROWS_DIR": str(dump_dir),
            "CUDF_CLICKBENCH_BENCH_PROGRESS": "1",
            "CUDF_CLICKBENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES": str(args.native_source_chunk_bytes),
            "CUDF_CLICKBENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK": str(args.native_row_groups_per_chunk),
        }
    )
    if engines == ["cpu_datafusion"]:
        env["CUDF_CLICKBENCH_BENCH_SKIP_NATIVE"] = "1"
    elif engines == ["datafusion_nexus"]:
        env["CUDF_CLICKBENCH_BENCH_SKIP_CPU"] = "1"

    report_path = out_dir / "datafusion_clickbench_benchmark.tsv"
    command = [args.cargo_bin, "run"]
    if args.release:
        command.append("--release")
    command.extend(["--bin", "native_clickbench_benchmark", "--quiet"])
    if "datafusion_nexus" in engines and not args.no_gpu_flock:
        command = ["flock", str(GPU_LOCK), *command]

    proc = subprocess.run(
        command,
        cwd=args.repo_root,
        text=True,
        capture_output=True,
        timeout=args.timeout_seconds,
        env=env,
    )
    report_path.write_text(proc.stdout, encoding="utf-8")

    progress = parse_datafusion_progress(proc.stderr)
    rows: list[OutputRow] = []
    engine_map = {"cpu": "cpu_datafusion", "native": "datafusion_nexus"}
    for query in queries:
        for engine_short, engine_name in engine_map.items():
            if engine_name not in engines:
                continue
            for iteration in range(1, args.iterations + 1):
                raw_path = dump_dir / f"{query.query_id}_{engine_short}_iter{iteration}.tsv"
                event = progress.get((engine_short, query.query_id, iteration))
                elapsed_ms = event.get("elapsed_ms", "none") if event else "none"
                native_rewritten = event.get("native_rewritten", "n/a") if event else "n/a"
                if proc.returncode == 0 or raw_path.exists():
                    row = row_from_raw(query.query_id, engine_name, iteration, raw_path, str(elapsed_ms), native_rewritten)
                    if event and event.get("status") != "succeeded":
                        row.status = str(event["status"])
                    rows.append(row)
                else:
                    rows.append(
                        OutputRow(
                            query_id=query.query_id,
                            engine=engine_name,
                            iteration=iteration,
                            status="failed",
                            row_count="none",
                            fingerprint="none",
                            elapsed_ms=str(elapsed_ms),
                            raw_output_path=str(raw_path),
                            native_rewritten=str(native_rewritten),
                            error=compact_process_error(proc),
                        )
                    )
    return rows


def parse_datafusion_progress(stderr: str) -> dict[tuple[str, str, int], dict[str, str]]:
    pattern = re.compile(
        r"^\[bench (?P<engine>cpu|native)\] iter (?P<iteration>\d+)/\d+ q=(?P<query_id>q\d{2}) "
        r"\(\d+/\d+\) elapsed_ms=(?P<elapsed_ms>\d+) outcome=(?P<status>\w+) rewritten=(?P<rewritten>.*)$"
    )
    events: dict[tuple[str, str, int], dict[str, str]] = {}
    for line in stderr.splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        engine = match.group("engine")
        query_id = match.group("query_id")
        iteration = int(match.group("iteration"))
        rewritten = match.group("rewritten")
        native_rewritten = "n/a"
        if rewritten == "Some(true)":
            native_rewritten = "true"
        elif rewritten == "Some(false)":
            native_rewritten = "false"
        events[(engine, query_id, iteration)] = {
            "elapsed_ms": match.group("elapsed_ms"),
            "status": match.group("status"),
            "native_rewritten": native_rewritten,
        }
    return events


def row_from_process(
    query_id: str,
    engine: str,
    iteration: int,
    proc: subprocess.CompletedProcess[str],
    raw_path: Path,
    elapsed_ms: int,
) -> OutputRow:
    row = row_from_raw(query_id, engine, iteration, raw_path, str(elapsed_ms), "n/a")
    if proc.returncode != 0:
        row.status = "failed"
        row.error = compact_process_error(proc)
        if not raw_path.exists():
            row.row_count = "none"
            row.fingerprint = "none"
    return row


def row_from_raw(
    query_id: str,
    engine: str,
    iteration: int,
    raw_path: Path,
    elapsed_ms: str,
    native_rewritten: str,
) -> OutputRow:
    if raw_path.exists():
        row_count, fingerprint = fingerprint_raw_tsv(raw_path)
        status = "succeeded"
    else:
        row_count = "none"
        fingerprint = "none"
        status = "missing_raw_output"
    return OutputRow(
        query_id=query_id,
        engine=engine,
        iteration=iteration,
        status=status,
        row_count=str(row_count),
        fingerprint=fingerprint,
        elapsed_ms=elapsed_ms,
        raw_output_path=str(raw_path),
        native_rewritten=native_rewritten,
    )


def fingerprint_raw_tsv(path: Path) -> tuple[int, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            next(reader)
        except StopIteration:
            rows: list[str] = []
        else:
            rows = [encode_row(row) for row in reader]
    rows.sort()
    hasher = Fnv1a64()
    hasher.update(f"rows={len(rows)}\n".encode("utf-8"))
    for row in rows:
        hasher.update(row.encode("utf-8"))
        hasher.update(b"\n")
    return len(rows), f"rows-fnv1a64:{hasher.hexdigest()}"


def encode_row(row: list[str]) -> str:
    encoded: list[str] = []
    for value in row:
        if value == "NULL":
            encoded.append("\0")
        else:
            encoded.append("\1" + value)
    return "\x1f".join(encoded)


class Fnv1a64:
    def __init__(self) -> None:
        self._value = 0xCBF29CE484222325

    def update(self, data: bytes) -> None:
        for byte in data:
            self._value ^= byte
            self._value = (self._value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF

    def hexdigest(self) -> str:
        return f"{self._value:016x}"


def monotonic_elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def compact_process_error(proc: subprocess.CompletedProcess[str]) -> str:
    text = "\n".join(part.strip() for part in [proc.stderr, proc.stdout] if part.strip())
    text = re.sub(r"\s+", " ", text).strip()
    return text[:2000] if text else f"return_code={proc.returncode}"


def quote_sql_literal(value: object) -> str:
    return str(value).replace("'", "''")


def write_tsv(path: Path, rows: list[OutputRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_HEADER, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_jsonl(path: Path, rows: list[OutputRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), sort_keys=True, ensure_ascii=False))
            handle.write("\n")


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    fixture: Path,
    query_ids: list[str],
    engines: list[str],
    tsv_path: Path,
    jsonl_path: Path,
) -> None:
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "fixture": str(fixture),
        "query_ids": query_ids,
        "waiver_query_ids": list(WAIVER_QUERY_IDS),
        "engines": engines,
        "iterations": args.iterations,
        "tsv": str(tsv_path),
        "jsonl": str(jsonl_path),
        "duckdb_bin": args.duckdb_bin,
        "reference_bin": str(args.reference_bin),
        "reference_load": args.reference_load,
        "cargo_bin": args.cargo_bin,
        "cargo_release": args.release,
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
