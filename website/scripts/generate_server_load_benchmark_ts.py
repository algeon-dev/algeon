#!/usr/bin/env python3
"""Regenerate website/src/data/serverLoadBenchmark.ts from flight-sql-stress runs.

Inputs follow the "TPC-H SF100 streams" and "Overload" sections of
crates/tools/fixtures/stress/RUNBOOK.md. Each --profile is the server TOML of
one configuration; its file stem names the run directories:

  <streams-root>/<stem>/c<clients>/r<repeat>/**/summary.json
  <overload-root>/r<repeat>/<factor>C/**/summary.json

The generator refuses runs whose server configuration, statements, stream
settings or result references differ, any failed stream query, any result that
did not match the reference, grants still live after a run, and overload runs
that are off the expected rate or limited by the client.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import tomllib
from pathlib import Path

STRESS_SCHEMA = "algeon_datafusion.flight_sql_stress.run_artifact"
EXECUTION_MODES = ("native_required", "functions_only")
REJECTION_REASONS = ("admission_queue_full", "queue_wait_elapsed")
ENVIRONMENT_KEYS = (
    "instanceType",
    "cpu",
    "gpu",
    "dataFusionVersion",
    "measuredDate",
    "sourceRevision",
    "serverBinarySha256",
    "serverCpuThreads",
    "clientCpuThreads",
    "cpuPathSettings",
)


def fail(message: str) -> None:
    raise SystemExit(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one_summary(run_dir: Path) -> dict:
    found = sorted(run_dir.rglob("summary.json"))
    if len(found) != 1:
        fail(f"{run_dir}: expected one summary.json, found {len(found)}")
    summary = json.loads(found[0].read_text())
    if summary.get("schema") != STRESS_SCHEMA:
        fail(f"{found[0]}: schema {summary.get('schema')!r} is not {STRESS_SCHEMA}")
    summary["_path"] = found[0]
    return summary


def spread(values: list[float], digits: int) -> dict:
    return {
        "median": round(statistics.median(values), digits),
        "min": round(min(values), digits),
        "max": round(max(values), digits),
    }


class Invariants:
    """Settings every run must share, recorded from the first run seen."""

    def __init__(self) -> None:
        self.values: dict[str, tuple[object, Path]] = {}

    def check(self, key: str, value: object, path: Path) -> None:
        if key not in self.values:
            self.values[key] = (value, path)
        elif self.values[key][0] != value:
            fail(f"{key} differs: {path} has {value!r}, {self.values[key][1]} has {self.values[key][0]!r}")

    def get(self, key: str) -> object:
        return self.values[key][0]


def check_results(summary: dict, invariants: Invariants, rejections_expected: bool) -> int:
    path = summary["_path"]
    results = summary.get("results") or {}
    if results.get("mode") != "check":
        fail(f"{path}: results.mode is {results.get('mode')!r}, expected 'check'")
    checks = results["checks_all_phases"]
    if checks["mismatch"] or checks["error"]:
        fail(f"{path}: {checks['mismatch']} mismatched and {checks['error']} errored result checks")
    # Rejected attempts return no result to check; every success must have been checked.
    if results.get("unchecked_successes", 0) or (checks["not_checked"] and not rejections_expected):
        fail(f"{path}: {checks['not_checked']} attempts not checked, {results.get('unchecked_successes')} unchecked successes")
    references = {name: ref["sha256"] for name, ref in results["references"].items()}
    invariants.check("result references", references, path)
    invariants.check("float tolerances", (results["float_abs_tolerance"], results["float_rel_tolerance"]), path)
    return checks["match"]


def check_common(summary: dict, config_sha: str, invariants: Invariants) -> None:
    path = summary["_path"]
    config = summary["config"]
    if config["server_config_sha256"] != config_sha:
        fail(f"{path}: server_config_sha256 {config['server_config_sha256']} does not match its profile TOML")
    invariants.check("statements", config["statements"], path)
    invariants.check("seed", config["seed"], path)
    invariants.check("connection mode", config["mode"], path)
    admission = summary["admission"]
    if not admission.get("after_run_idle"):
        fail(f"{path}: grants were still live after the run: {admission.get('after_run')}")
    limits = admission["limits"]
    invariants.check(
        "queue limits", (limits["max_queued_attempts"], limits["max_queue_wait_ms"]), path
    )


def peak(median_qps: dict[int, float]) -> dict:
    streams, qps = max(median_qps.items(), key=lambda item: item[1])
    return {"streams": streams, "completedQps": round(qps, 4)}


def load_profile(path: Path) -> dict:
    config = tomllib.loads(path.read_text())
    mode = (config.get("native") or {}).get("execution_mode")
    if mode not in EXECUTION_MODES:
        fail(f"{path}: [native] execution_mode must be one of {EXECUTION_MODES}, found {mode!r}")
    return {
        "key": path.stem,
        "path": path,
        "sha": sha256(path),
        "executionMode": mode,
        # A profile without an [admission] table runs the server's derived defaults.
        "admissionDefaults": "admission" not in config,
    }


def stream_points(profile: dict, root: Path, invariants: Invariants) -> tuple[list[dict], dict[int, float], int, dict]:
    profile_dir = root / profile["key"]
    client_dirs = sorted(profile_dir.glob("c*"), key=lambda p: int(p.name[1:]))
    if not client_dirs:
        fail(f"{profile_dir}: no c<clients> directories")
    points, median_qps, matched = [], {}, 0
    device_limits = Invariants()
    for client_dir in client_dirs:
        clients = int(client_dir.name[1:])
        runs = [one_summary(run_dir) for run_dir in sorted(client_dir.glob("r*"))]
        invariants.check("repeats per point", len(runs), client_dir)
        for summary in runs:
            path = summary["_path"]
            check_common(summary, profile["sha"], invariants)
            config = summary["config"]
            if config["arrival_mode"] != "streams" or config["clients"] != clients:
                fail(f"{path}: expected {clients} streams, found {config['arrival_mode']} with {config['clients']} clients")
            invariants.check("passes", (config["passes"], config["warmup_passes"]), path)
            overall = summary["overall"]
            if overall["failed"] or overall["cancelled"]:
                fail(f"{path}: {overall['failed']} failed and {overall['cancelled']} cancelled queries")
            matched += check_results(summary, invariants, False)
            devices = summary["admission"]["limits"]["devices"]
            if len(devices) != 1:
                fail(f"{path}: expected one GPU, found {len(devices)}")
            device_limits.check(
                "device limits", (devices[0]["max_active_attempts"], devices[0]["default_query_limit_bytes"]), path
            )
        median_qps[clients] = statistics.median(s["overall"]["throughput"]["completed_qps"] for s in runs)
        points.append(
            {
                "streams": clients,
                "runs": len(runs),
                "measuredQueriesPerRun": runs[0]["overall"]["attempts"],
                "completedQps": spread([s["overall"]["throughput"]["completed_qps"] for s in runs], 4),
                "p50Seconds": spread([s["overall"]["latency_ns"]["p50"] / 1e9 for s in runs], 3),
                "p95Seconds": spread([s["overall"]["latency_ns"]["p95"] / 1e9 for s in runs], 3),
            }
        )
    active, query_limit = device_limits.get("device limits")
    return points, median_qps, matched, {"maxActiveAttempts": active, "queryLimitBytes": query_limit}


def overload_levels(root: Path, default_gpu: dict, capacity_qps: float, invariants: Invariants) -> tuple[list[dict], int]:
    by_level: dict[str, list[dict]] = {}
    for repeat_dir in sorted(root.glob("r*")):
        for level_dir in repeat_dir.iterdir():
            if level_dir.is_dir() and re.fullmatch(r"\d+(\.\d+)?C", level_dir.name):
                by_level.setdefault(level_dir.name, []).append(one_summary(level_dir))
    if not by_level:
        fail(f"{root}: no r<repeat>/<factor>C run directories")
    levels, matched = [], 0
    for level in sorted(by_level, key=lambda name: float(name[:-1])):
        runs = by_level[level]
        factor = float(level[:-1])
        invariants.check("overload repeats", len(runs), root / level)
        for summary in runs:
            path = summary["_path"]
            check_common(summary, default_gpu["sha"], invariants)
            config = summary["config"]
            if config["arrival_mode"] != "rate":
                fail(f"{path}: arrival_mode {config['arrival_mode']!r} is not 'rate'")
            expected_rate = factor * capacity_qps
            if abs(config["rate_per_second"] - expected_rate) > 1e-6 * expected_rate:
                fail(f"{path}: rate {config['rate_per_second']} is not {factor} x C = {expected_rate}")
            invariants.check(
                "overload window", (config["warmup_seconds"], config["measurement_seconds"]), path
            )
            if summary["open_loop"]["client_bound_reached"]:
                fail(f"{path}: the client's concurrency bound limited the offered rate")
            overall = summary["overall"]
            rejected = overall["errors"].get("service_overloaded", {})
            unexpected = set(rejected) - set(REJECTION_REASONS)
            if overall["cancelled"] or overall["failed"] != sum(rejected.values()) or unexpected:
                fail(
                    f"{path}: failures other than service_overloaded rejections: "
                    f"error_classes={overall['error_classes']}, errors={overall['errors']}"
                )
            matched += check_results(summary, invariants, True)
        rejection_p50 = {}
        for reason in REJECTION_REASONS:
            values = [
                s["overall"]["rejected_latency_p50_ns"][reason] / 1e9
                for s in runs
                if s["overall"].get("rejected_latency_p50_ns", {}).get(reason) is not None
            ]
            rejection_p50[reason] = spread(values, 3) if values else None
        levels.append(
            {
                "level": level,
                "factor": factor,
                "runs": len(runs),
                "configuredRateQps": round(factor * capacity_qps, 4),
                "offeredQps": spread([s["overall"]["throughput"]["offered_qps"] for s in runs], 4),
                "completedQps": spread([s["overall"]["throughput"]["completed_qps"] for s in runs], 4),
                "rejectedFraction": spread([s["overall"]["failed"] / s["overall"]["attempts"] for s in runs], 4),
                "rejectedCount": {
                    reason: sum(s["overall"]["errors"].get("service_overloaded", {}).get(reason, 0) for s in runs)
                    for reason in REJECTION_REASONS
                },
                "rejectionP50Seconds": {
                    "admissionQueueFull": rejection_p50["admission_queue_full"],
                    "queueWaitElapsed": rejection_p50["queue_wait_elapsed"],
                },
                "completedP95Seconds": spread([s["overall"]["latency_ns"]["p95"] / 1e9 for s in runs], 3),
                "attempts": sum(s["overall"]["attempts"] for s in runs),
            }
        )
    return levels, matched


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", type=Path, action="append", required=True, help="server TOML of one configuration")
    ap.add_argument("--streams-root", type=Path, required=True)
    ap.add_argument("--overload-root", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True, help="output directory of the reference-recording run")
    ap.add_argument("--environment", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    environment = json.loads(args.environment.read_text())
    missing = [key for key in ENVIRONMENT_KEYS if key not in environment]
    if missing:
        fail(f"{args.environment}: missing {missing}")

    profiles = [load_profile(path) for path in args.profile]
    defaults = [p for p in profiles if p["executionMode"] == "native_required" and p["admissionDefaults"]]
    cpu = [p for p in profiles if p["executionMode"] == "functions_only"]
    if len(defaults) != 1 or len(cpu) != 1:
        fail("expected exactly one GPU profile without an [admission] table and one functions_only profile")
    default_gpu, cpu_profile = defaults[0], cpu[0]

    invariants = Invariants()
    reference = one_summary(args.reference)
    if reference["results"].get("mode") != "record":
        fail(f"{reference['_path']}: results.mode is not 'record'")
    if reference["config"]["server_config_sha256"] != cpu_profile["sha"]:
        fail(f"{reference['_path']}: references were not recorded by the functions_only profile")
    invariants.check(
        "result references",
        {name: ref["sha256"] for name, ref in reference["results"]["references"].items()},
        reference["_path"],
    )
    invariants.check("statements", reference["config"]["statements"], reference["_path"])

    series, median_qps, gpu_matched, cpu_matched = [], {}, 0, 0
    for profile in profiles:
        points, median_qps[profile["key"]], matched, limits = stream_points(profile, args.streams_root, invariants)
        on_gpu = profile["executionMode"] == "native_required"
        if on_gpu:
            gpu_matched += matched
        else:
            cpu_matched += matched
        series.append(
            {
                "key": profile["key"],
                "executionMode": profile["executionMode"],
                "admissionDefaults": profile["admissionDefaults"],
                # functions_only plans take no admission slot, so the limits apply to GPU profiles only.
                "maxActiveAttempts": limits["maxActiveAttempts"] if on_gpu else None,
                "queryLimitBytes": limits["queryLimitBytes"] if on_gpu else None,
                "peak": peak(median_qps[profile["key"]]),
                "points": points,
            }
        )
    stream_counts = {tuple(p["streams"] for p in s["points"]) for s in series}
    if len(stream_counts) != 1:
        fail(f"profiles cover different stream counts: {sorted(stream_counts)}")
    series.sort(key=lambda s: (s["executionMode"] != "native_required", s["maxActiveAttempts"] or 0))

    # The overload rates are multiples of the default profile's highest median, unrounded.
    capacity_streams, capacity_qps = max(median_qps[default_gpu["key"]].items(), key=lambda item: item[1])
    levels, overload_matched = overload_levels(args.overload_root, default_gpu, capacity_qps, invariants)
    gpu_matched += overload_matched

    statements = invariants.get("statements")
    references = reference["results"]
    payload = {
        "environment": environment,
        "config": {
            "statements": [s["name"] for s in statements],
            "emptyReferenceStatements": references.get("empty_references", []),
            "seed": invariants.get("seed"),
            "passes": invariants.get("passes")[0],
            "warmupPasses": invariants.get("passes")[1],
            "repeats": invariants.get("repeats per point"),
            "streams": list(next(iter(stream_counts))),
            "connectionMode": invariants.get("connection mode"),
            "maxQueuedAttempts": invariants.get("queue limits")[0],
            "maxQueueWaitMs": invariants.get("queue limits")[1],
            "floatToleranceAbsolute": invariants.get("float tolerances")[0],
            "floatToleranceRelative": invariants.get("float tolerances")[1],
            "overloadWarmupSeconds": invariants.get("overload window")[0],
            "overloadMeasurementSeconds": invariants.get("overload window")[1],
            "overloadRepeats": invariants.get("overload repeats"),
        },
        "series": series,
        "overload": {"capacityQps": round(capacity_qps, 4), "capacityStreams": capacity_streams, "levels": levels},
        "summary": {
            "gpuResultsMatched": gpu_matched,
            "cpuResultsMatched": cpu_matched,
            "overloadAttempts": sum(level["attempts"] for level in levels),
        },
    }
    profile_list = ", ".join(path.as_posix() for path in args.profile)
    text = (
        f"// Generated by website/scripts/generate_server_load_benchmark_ts.py from\n"
        f"// {args.streams_root.as_posix()}, {args.overload_root.as_posix()} and {args.reference.as_posix()}.\n"
        f"// Profiles: {profile_list}.\n"
        f"// Environment: {args.environment.as_posix()}.\n"
        f"// Do not hand-edit benchmark values; regenerate from the run summaries.\n\n"
        "export type Spread = {median: number; min: number; max: number};\n\n"
        f"export const serverLoadBenchmark = {json.dumps(payload, indent=2)} as const;\n"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {args.out}")
    for s in series:
        line = " ".join(f"c{p['streams']}={p['completedQps']['median']:.3f}" for p in s["points"])
        print(f"{s['key']}: {line}")
    print(f"C={capacity_qps:.4f} QPS; GPU results matched {gpu_matched}, CPU {cpu_matched}")


if __name__ == "__main__":
    main()
