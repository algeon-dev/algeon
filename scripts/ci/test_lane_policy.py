#!/usr/bin/env python3
"""Validate the test-lane exceptions, nextest inventory, and scheduling overrides."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = ROOT / ".config" / "policies" / "test-lanes.toml"
DEFAULT_NEXTEST = ROOT / ".config" / "nextest.toml"
FULL_ID = re.compile(r"^[a-z0-9_-]+(?:::(?:bin/)?[A-Za-z0-9_-]+)?$")
TARGET_KINDS = frozenset({"lib", "test", "bin"})

GPU_LIB_PACKAGES = frozenset({
    "cudf-nexus",
    "cugraph-nexus",
    "cuvs-nexus",
    "datafusion-nexus",
    "nexus-query-runtime",
    "nexus-server",
})


class LanePolicyError(RuntimeError):
    """The checked-in lane policy or live classification is invalid."""


class InventoryError(RuntimeError):
    """The nextest inventory command succeeded but its output is unusable."""


def target_kind_for_id(binary_id: str) -> str:
    if "::" not in binary_id:
        return "lib"
    target = binary_id.split("::", 1)[1]
    return "bin" if target.startswith("bin/") else "test"


@dataclass(frozen=True)
class ExceptionEntry:
    binary_id: str
    lane: str
    resource_class: str | None
    reason: str | None

    @property
    def package(self) -> str:
        return self.binary_id.split("::", 1)[0]

    @property
    def target_kind(self) -> str:
        return target_kind_for_id(self.binary_id)


@dataclass(frozen=True)
class InventoryBinary:
    binary_id: str
    binary_name: str
    target_kind: str

    @property
    def package(self) -> str:
        return self.binary_id.split("::", 1)[0]

    @property
    def target_key(self) -> tuple[str, str, str]:
        return (self.package, self.binary_name, self.target_kind)


def _load_toml(path: Path, description: str) -> dict[str, Any]:
    try:
        with path.open("rb") as source:
            value = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise LanePolicyError(f"cannot load {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise LanePolicyError(f"{description} {path} must contain a TOML table")
    return value


def load_catalog(path: Path) -> dict[str, ExceptionEntry]:
    catalog = _load_toml(path, "test-lane exception policy")
    if set(catalog) != {"exception"}:
        raise LanePolicyError("test-lane policy top-level keys must be exactly exception")
    raw_entries = catalog["exception"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise LanePolicyError("test-lane policy exception must be a non-empty array of tables")

    entries: dict[str, ExceptionEntry] = {}
    for index, raw in enumerate(raw_entries):
        location = f"exception[{index}]"
        if not isinstance(raw, dict):
            raise LanePolicyError(f"{location} must be a table")
        allowed = {"id", "lane", "resource_class", "reason"}
        unknown = set(raw) - allowed
        if unknown:
            raise LanePolicyError(f"{location} has unknown keys: {", ".join(sorted(unknown))}")
        for required in ("id", "lane"):
            if not isinstance(raw.get(required), str) or not raw[required].strip():
                raise LanePolicyError(f"{location}.{required} must be a non-empty string")

        binary_id = raw["id"]
        lane = raw["lane"]
        resource_class = raw.get("resource_class")
        reason = raw.get("reason")
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            raise LanePolicyError(f"{location}.reason must be a non-empty string")
        if not FULL_ID.fullmatch(binary_id):
            raise LanePolicyError(f"{location}.id is not a complete nextest binary id: {binary_id!r}")
        if binary_id in entries:
            raise LanePolicyError(f"duplicate catalog binary id: {binary_id}")
        if lane not in {"cpu", "gpu"}:
            raise LanePolicyError(f"{location}.lane must be cpu or gpu")
        if lane == "cpu" and resource_class is not None:
            raise LanePolicyError(f"CPU entry {binary_id} must not declare resource_class")
        if lane == "gpu" and resource_class not in {"shared", "exclusive"}:
            raise LanePolicyError(f"GPU entry {binary_id} requires resource_class=shared or exclusive")

        entries[binary_id] = ExceptionEntry(binary_id, lane, resource_class, reason)

    return entries


def infer_lane_from_name(binary_name: str) -> tuple[str, str | None] | None:
    if binary_name.endswith("_gpu_exclusive"):
        return ("gpu", "exclusive")
    if binary_name.endswith("_gpu"):
        return ("gpu", "shared")
    if binary_name.endswith("_cpu"):
        return ("cpu", None)
    return None


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise InventoryError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def load_inventory(path: Path) -> list[InventoryBinary]:
    try:
        with path.open(encoding="utf-8") as source:
            inventory = json.load(source, object_pairs_hook=_object_without_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError, InventoryError) as error:
        raise InventoryError(f"malformed nextest JSON inventory: {error}") from error
    if not isinstance(inventory, dict):
        raise InventoryError("malformed nextest JSON inventory: top-level value must be an object")
    binaries = inventory.get("rust-binaries")
    if not isinstance(binaries, dict):
        raise InventoryError("malformed nextest JSON inventory: rust-binaries must be an object")
    if not binaries:
        raise InventoryError("nextest JSON inventory contained no test binaries")

    records: list[InventoryBinary] = []
    seen_targets: dict[tuple[str, str, str], str] = {}
    for inventory_key, raw in binaries.items():
        if not isinstance(raw, dict):
            raise InventoryError(f"malformed nextest JSON inventory: rust-binaries[{inventory_key!r}] must be an object")
        binary_id = raw.get("binary-id")
        binary_name = raw.get("binary-name")
        if not isinstance(binary_id, str) or not binary_id or not FULL_ID.fullmatch(binary_id):
            raise InventoryError(f"malformed nextest JSON inventory: invalid binary-id for {inventory_key!r}")
        if inventory_key != binary_id:
            raise InventoryError(
                f"malformed nextest JSON inventory: key {inventory_key!r} does not match binary-id {binary_id!r}"
            )
        if not isinstance(binary_name, str) or not binary_name or any(char in binary_name for char in "\t\r\n"):
            raise InventoryError(f"malformed nextest JSON inventory: invalid binary-name for {binary_id!r}")
        target_kind = raw.get("kind")
        if not isinstance(target_kind, str) or target_kind not in TARGET_KINDS:
            raise InventoryError(f"malformed nextest JSON inventory: invalid target kind for {binary_id!r}")
        expected_kind = target_kind_for_id(binary_id)
        if target_kind != expected_kind:
            raise InventoryError(
                f"malformed nextest JSON inventory: target kind mismatch for {binary_id!r}: "
                f"expected {expected_kind!r}, got {target_kind!r}"
            )
        record = InventoryBinary(binary_id, binary_name, target_kind)
        previous = seen_targets.get(record.target_key)
        if previous is not None:
            raise LanePolicyError(
                "test-binary target collision "
                f"{record.package!r}/{binary_name!r} kind={target_kind}: {previous} and {binary_id}"
            )
        seen_targets[record.target_key] = binary_id
        records.append(record)
    return sorted(records, key=lambda record: record.binary_id)


def validate_inventory(
    catalog: dict[str, ExceptionEntry],
    inventory: list[InventoryBinary],
) -> dict[str, tuple[str, str | None]]:
    inventory_by_id = {record.binary_id: record for record in inventory}
    problems: list[str] = []

    stale = sorted(catalog.keys() - inventory_by_id.keys())
    if stale:
        for binary_id in stale:
            problems.append(f"stale exception entry: {binary_id}")

    assigned: dict[str, tuple[str, str | None]] = {}
    for binary in inventory:
        inferred = infer_lane_from_name(binary.binary_name)
        exception = catalog.get(binary.binary_id)

        if inferred is not None and exception is not None:
            exc_lane = (exception.lane, exception.resource_class)
            if inferred != exc_lane:
                problems.append(
                    f"conflicting lane for {binary.binary_id}: "
                    f"target suffix implies {inferred[0]}/{inferred[1]}, "
                    f"exception declares {exc_lane[0]}/{exc_lane[1]}"
                )
            else:
                problems.append(
                    f"redundant exception entry for {binary.binary_id}: "
                    f"target name already has canonical suffix"
                )
        elif inferred is None and exception is None:
            problems.append(f"unclassified test binary: {binary.binary_id}")
        elif exception is not None:
            assigned[binary.binary_id] = (exception.lane, exception.resource_class)
        else:
            assert inferred is not None
            assigned[binary.binary_id] = inferred

    if problems:
        raise LanePolicyError("\n".join(problems))
    return assigned


def check_nextest(path: Path) -> None:
    config = _load_toml(path, "nextest config")
    test_groups = config.get("test-groups", {})
    gpu_group = test_groups.get("gpu", {})
    if gpu_group.get("max-threads") != "num-cpus":
        raise LanePolicyError(
            f"nextest config test-groups.gpu.max-threads must be num-cpus, got {gpu_group.get(max-threads)!r}"
        )
    gpu_exclusive_group = test_groups.get("gpu-exclusive", {})
    if gpu_exclusive_group.get("max-threads") != 1:
        raise LanePolicyError(
            f"nextest config test-groups.gpu-exclusive.max-threads must be 1, got {gpu_exclusive_group.get(max-threads)!r}"
        )

    overrides = config.get("profile", {}).get("default", {}).get("overrides", [])
    if not isinstance(overrides, list):
        raise LanePolicyError("nextest config profile.default.overrides must be an array of tables")

    gpu_override = None
    exclusive_override = None
    for override in overrides:
        group = override.get("test-group")
        if group == "gpu":
            gpu_override = override
        elif group == "gpu-exclusive":
            exclusive_override = override

    if gpu_override is None:
        raise LanePolicyError("nextest config missing override for test-group gpu")
    if exclusive_override is None:
        raise LanePolicyError("nextest config missing override for test-group gpu-exclusive")

    gpu_filter = gpu_override.get("filter", "")
    if "/_gpu$/" not in gpu_filter:
        raise LanePolicyError("nextest gpu group filter must include binary(/_gpu$/)")
    for target in ("root_gpu_tests", "native_exec_tests"):
        if f"binary({target})" not in gpu_filter:
            raise LanePolicyError(f"nextest gpu group filter must include binary({target})")
    for pkg in GPU_LIB_PACKAGES:
        if f"package({pkg})" not in gpu_filter:
            raise LanePolicyError(f"nextest gpu group filter must include package({pkg})")

    exclusive_filter = exclusive_override.get("filter", "")
    if "/_gpu_exclusive$/" not in exclusive_filter:
        raise LanePolicyError("nextest gpu-exclusive filter must include binary(/_gpu_exclusive$/)")
    if exclusive_override.get("threads-required") != "num-test-threads":
        raise LanePolicyError("nextest gpu-exclusive override must declare threads-required = num-test-threads")


def command_check_catalog(args: argparse.Namespace) -> None:
    entries = load_catalog(args.catalog)
    cpu = sum(entry.lane == "cpu" for entry in entries.values())
    shared = sum(entry.resource_class == "shared" for entry in entries.values())
    exclusive = sum(entry.resource_class == "exclusive" for entry in entries.values())
    print(f"OK: exception catalog has {len(entries)} binaries (cpu={cpu}, gpu_shared={shared}, gpu_exclusive={exclusive}).")


def command_check_nextest(args: argparse.Namespace) -> None:
    check_nextest(args.nextest)
    print("OK: nextest configuration matches test-lane scheduling policy.")


def command_validate(args: argparse.Namespace) -> None:
    entries = load_catalog(args.catalog)
    check_nextest(args.nextest)
    inventory = load_inventory(args.inventory)
    assigned = validate_inventory(entries, inventory)
    cpu = sum(lane == "cpu" for lane, _ in assigned.values())
    shared = sum(res == "shared" for _, res in assigned.values())
    exclusive = sum(res == "exclusive" for _, res in assigned.values())
    print(f"OK: all {len(inventory)} live test binaries classified (cpu={cpu}, gpu_shared={shared}, gpu_exclusive={exclusive}).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--nextest", type=Path, default=DEFAULT_NEXTEST)
    subparsers = parser.add_subparsers(dest="command", required=True)

    catalog = subparsers.add_parser("check-catalog")
    catalog.set_defaults(function=command_check_catalog)

    nextest = subparsers.add_parser("check-nextest")
    nextest.set_defaults(function=command_check_nextest)

    validate = subparsers.add_parser("validate-inventory")
    validate.add_argument(
        "--inventory",
        type=Path,
        required=True,
        help="nextest binaries-only JSON inventory",
    )
    validate.set_defaults(function=command_validate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        args.function(args)
    except InventoryError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    except LanePolicyError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
