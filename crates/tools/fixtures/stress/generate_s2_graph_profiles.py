#!/usr/bin/env python3
"""Emit concrete S2 profiles from checked-in bases and small overlays.

The generated files deliberately live under an operator-chosen directory:
the sweep establishes a knee and the rate profiles then bind that measured
knee.  Do not edit generated profiles; adjust a base or an overlay here.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
BASES = {
    "sweep": ROOT / "profiles" / "s2-graph-sweep-base.json",
    "rate": ROOT / "profiles" / "s2-graph-rate-base.json",
    "relational_rate": ROOT / "profiles" / "s2-relational-rate-base.json",
}
ALGORITHMS: dict[str, dict[str, Any]] = {
    "g1-force-atlas2": {
        "query": "force_atlas2",
        "concurrency": (1, 2, 4, 8),
        "max_active_attempts": 1,
        "stream_pool_size": 3,
    },
    "g2-bfs-deep": {
        "query": "bfs_deep",
        "concurrency": (1, 2, 4, 8, 16),
        "max_active_attempts": 2,
        "stream_pool_size": 6,
    },
    "g3-pagerank": {
        "query": "pagerank",
        "concurrency": (1, 2, 4, 8, 16),
        "max_active_attempts": 2,
        "stream_pool_size": 6,
    },
    "g4-betweenness": {
        "query": "betweenness_centrality",
        "concurrency": (1, 2, 4, 8, 16),
        "max_active_attempts": 2,
        "stream_pool_size": 6,
    },
}
PROFILE_SCHEMA = "algeon_tools.single_node_multi_gpu.run_profile.v1"
RATE_ARRIVAL_BUDGET = 80_000


def read_base(kind: str) -> dict[str, Any]:
    try:
        loaded = json.loads(BASES[kind].read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {BASES[kind]}: {error}") from error
    if not isinstance(loaded, dict):
        raise ValueError(f"{BASES[kind]} must contain one JSON object")
    return loaded


def graph_workload(profile: dict[str, Any]) -> dict[str, Any]:
    matches = [workload for workload in profile.get("workloads", []) if workload.get("kind") == "dblp"]
    if len(matches) != 1:
        raise ValueError("base profile must contain exactly one DBLP workload")
    return matches[0]


def overlay_devices(profile: dict[str, Any], overlay: dict[str, Any]) -> None:
    devices = profile.get("backend", {}).get("device_profiles")
    if not isinstance(devices, list) or len(devices) != 4:
        raise ValueError("base profile must contain exactly four device profiles")
    for device in devices:
        device["max_active_attempts"] = overlay["max_active_attempts"]
        device["installation"]["stream_pool_size"] = overlay["stream_pool_size"]


def sweep_profile(algorithm: str) -> tuple[str, dict[str, Any]]:
    overlay = ALGORITHMS[algorithm]
    profile = copy.deepcopy(read_base("sweep"))
    profile["name"] = f"stress-s2-sweep-{algorithm}-sf10"
    graph_workload(profile)["queries"] = [overlay["query"]]
    overlay_devices(profile, overlay)
    profile["scheduling"]["steps"] = [
        {"concurrency": concurrency, "measurement_seconds": 60} for concurrency in overlay["concurrency"]
    ]
    return f"stress-s2-sweep-{algorithm}-sf10.json", profile


def rate_profile(algorithm: str, knee_qps: int, phase_seconds: int) -> tuple[str, dict[str, Any]]:
    overlay = ALGORITHMS[algorithm]
    profile = copy.deepcopy(read_base("rate"))
    profile["name"] = f"stress-s2-rate-{algorithm}-sf10"
    graph_workload(profile)["queries"] = [overlay["query"]]
    overlay_devices(profile, overlay)
    profile["scheduling"]["steps"] = rate_steps(knee_qps, phase_seconds)
    return f"stress-s2-rate-{algorithm}-sf10.json", profile


def rate_steps(knee_qps: int, phase_seconds: int) -> list[dict[str, Any]]:
    rates = (
        knee_qps // 2,
        knee_qps,
        knee_qps * 3 // 2,
        knee_qps * 2,
        knee_qps * 5,
        knee_qps * 10,
    )
    steps = [
        {"arrival_rate_per_second": rate, "measurement_seconds": phase_seconds, "rate_overrun": overrun}
        for overrun in ("delay", "drop")
        for rate in rates
    ]
    steps.append({"arrival_rate_per_second": rates[0], "measurement_seconds": phase_seconds, "rate_overrun": "delay"})
    return steps


def rate_phase_seconds(knee_qps: int) -> int:
    numerator = 2 * (RATE_ARRIVAL_BUDGET - 5 * knee_qps)
    denominator = 81 * knee_qps
    phase_seconds = min(60, numerator // denominator)
    if phase_seconds < 1:
        raise ValueError(
            "--knee-qps cannot fit the 80,000-arrival budget; the largest supported positive even knee is 1758"
        )
    return phase_seconds


def standard_rate_profile(profile_class: str, knee_qps: int, phase_seconds: int) -> tuple[str, dict[str, Any]]:
    if profile_class in {"aggregate", "scan"}:
        profile = copy.deepcopy(read_base("relational_rate"))
        query = "q01" if profile_class == "aggregate" else "q06"
    elif profile_class == "graph-io":
        profile = copy.deepcopy(read_base("rate"))
        profile.pop("graph_claim_calibration", None)
        query = "edge_scan"
    else:
        raise ValueError(f"unsupported standard rate class {profile_class}")
    profile["name"] = f"stress-s2-rate-{profile_class}-sf10"
    if profile_class == "graph-io":
        graph_workload(profile)["queries"] = [query]
    else:
        workloads = profile.get("workloads", [])
        if len(workloads) != 1 or workloads[0].get("kind") != "tpch":
            raise ValueError("relational rate base must contain exactly one TPC-H workload")
        workloads[0]["queries"] = [query]
        if profile_class == "aggregate":
            # q01's reservation-limit witness is 1,068,599,820 B live plus
            # 15,237,120 B requested. Keep the immutable floor above that
            # observed owner-side requirement in both rate phases.
            aggregate_floor = 1_342_177_280
            profile["backend"]["native_execution"]["device_budget_bytes"] = aggregate_floor
            for device in profile["backend"]["device_profiles"]:
                device["default_query_limit_bytes"] = aggregate_floor
    profile["scheduling"]["steps"] = rate_steps(knee_qps, phase_seconds)
    return f"stress-s2-rate-{profile_class}-sf10.json", profile


def validate_rate_steps(name: str, scheduling: dict[str, Any], knee_qps: int, phase_seconds: int) -> None:
    rates = [knee_qps // 2, knee_qps, knee_qps * 3 // 2, knee_qps * 2, knee_qps * 5, knee_qps * 10]
    expected_rates = rates + rates + [rates[0]]
    expected_overruns = ["delay"] * 6 + ["drop"] * 6 + ["delay"]
    steps = scheduling.get("steps")
    if scheduling.get("arrival_mode") != "rate" or not isinstance(steps, list) or len(steps) != 13:
        raise ValueError(f"{name}: invalid rate sweep shape")
    if [step.get("arrival_rate_per_second") for step in steps] != expected_rates:
        raise ValueError(f"{name}: invalid rate overlay")
    if [step.get("measurement_seconds") for step in steps] != [phase_seconds] * 13:
        raise ValueError(f"{name}: invalid rate phase duration")
    if [step.get("rate_overrun") for step in steps] != expected_overruns:
        raise ValueError(f"{name}: invalid rate overrun/recovery schedule")


def validate_profile(
    name: str,
    profile: dict[str, Any],
    *,
    kind: str,
    algorithm: str,
    knee_qps: int | None = None,
    phase_seconds: int | None = None,
) -> None:
    overlay = ALGORITHMS[algorithm]
    if profile.get("schema") != PROFILE_SCHEMA or profile.get("version") != 1:
        raise ValueError(f"{name}: invalid run-profile schema/version")
    if profile.get("name") != name.removesuffix(".json"):
        raise ValueError(f"{name}: profile name does not match file name")
    if profile.get("graph_claim_calibration") != {}:
        raise ValueError(f"{name}: graph claim calibration must remain enabled")
    if graph_workload(profile).get("queries") != [overlay["query"]]:
        raise ValueError(f"{name}: graph query overlay is not exact")
    devices = profile["backend"]["device_profiles"]
    if any(device["max_active_attempts"] != overlay["max_active_attempts"] for device in devices):
        raise ValueError(f"{name}: device active-attempt overlay is not exact")
    if any(device["installation"]["stream_pool_size"] != overlay["stream_pool_size"] for device in devices):
        raise ValueError(f"{name}: stream-pool overlay is not exact")
    scheduling = profile.get("scheduling", {})
    steps = scheduling.get("steps")
    if kind == "sweep":
        expected = list(overlay["concurrency"])
        actual = [step.get("concurrency") for step in steps]
        if scheduling.get("arrival_mode") != "concurrency" or actual != expected:
            raise ValueError(f"{name}: invalid concurrency sweep")
    else:
        if knee_qps is None or phase_seconds is None:
            raise ValueError(f"{name}: missing rate overlay")
        validate_rate_steps(name, scheduling, knee_qps, phase_seconds)
    encoded = json.dumps(profile, sort_keys=True)
    if "__" in encoded:
        raise ValueError(f"{name}: unresolved template token")


def validate_standard_rate(
    name: str, profile: dict[str, Any], profile_class: str, knee_qps: int, phase_seconds: int
) -> None:
    expected_query = {"aggregate": "q01", "scan": "q06", "graph-io": "edge_scan"}[profile_class]
    if profile.get("schema") != PROFILE_SCHEMA or profile.get("version") != 1:
        raise ValueError(f"{name}: invalid run-profile schema/version")
    if profile.get("name") != name.removesuffix(".json"):
        raise ValueError(f"{name}: profile name does not match file name")
    workloads = profile.get("workloads", [])
    if len(workloads) != 1 or workloads[0].get("queries") != [expected_query]:
        raise ValueError(f"{name}: workload overlay is not exact")
    if profile_class == "graph-io":
        if workloads[0].get("kind") != "dblp" or "graph_claim_calibration" in profile:
            raise ValueError(f"{name}: graph I/O must not claim an algorithm peak")
    elif workloads[0].get("kind") != "tpch":
        raise ValueError(f"{name}: relational class must use TPC-H")
    validate_rate_steps(name, profile.get("scheduling", {}), knee_qps, phase_seconds)
    if "__" in json.dumps(profile, sort_keys=True):
        raise ValueError(f"{name}: unresolved template token")


def write_or_check(output: Path, profile: dict[str, Any], check: bool) -> None:
    rendered = json.dumps(profile, indent=2) + "\n"
    if check:
        if not output.is_file() or output.read_text() != rendered:
            raise ValueError(f"{output} is not the deterministic generated profile")
        return
    output.write_text(rendered)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True, help="directory for concrete runnable profiles")
    parser.add_argument(
        "--knee-qps",
        type=int,
        help="measured S2 knee rounded up to a positive even integer; emits rate profiles when supplied",
    )
    parser.add_argument("--check", action="store_true", help="verify the emitted profiles match deterministic output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.knee_qps is not None and args.knee_qps <= 0:
        raise ValueError("--knee-qps must be greater than zero")
    if args.knee_qps is not None and args.knee_qps % 2 != 0:
        raise ValueError("--knee-qps must be a positive even integer after rounding the measured knee upward")
    if args.check and not args.out_dir.is_dir():
        raise ValueError(f"--out-dir {args.out_dir} must exist with --check")
    if not args.check:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    phase_seconds = rate_phase_seconds(args.knee_qps) if args.knee_qps is not None else None
    emitted: list[tuple[str, dict[str, Any], str, str]] = []
    for algorithm in ALGORITHMS:
        name, profile = sweep_profile(algorithm)
        emitted.append((name, profile, "sweep", algorithm))
        if args.knee_qps is not None:
            name, profile = rate_profile(algorithm, args.knee_qps, phase_seconds)
            emitted.append((name, profile, "rate", algorithm))
    if args.knee_qps is not None:
        for profile_class in ("aggregate", "scan", "graph-io"):
            name, profile = standard_rate_profile(profile_class, args.knee_qps, phase_seconds)
            emitted.append((name, profile, "standard_rate", profile_class))
    for name, profile, kind, algorithm in emitted:
        if kind == "standard_rate":
            validate_standard_rate(name, profile, algorithm, args.knee_qps, phase_seconds)
        else:
            validate_profile(
                name,
                profile,
                kind=kind,
                algorithm=algorithm,
                knee_qps=args.knee_qps,
                phase_seconds=phase_seconds,
            )
        write_or_check(args.out_dir / name, profile, args.check)
    print(f"validated {len(emitted)} concrete S2 profiles in {args.out_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as error:
        raise SystemExit(f"profile generation failed: {error}")
