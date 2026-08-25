#!/usr/bin/env python3
"""Validate the test-lane catalog, nextest inventory, and generated overrides."""

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
GENERATED_BEGIN = "# BEGIN GENERATED TEST-LANE OVERRIDES — scripts/ci/test_lane_policy.py"
GENERATED_END = "# END GENERATED TEST-LANE OVERRIDES — scripts/ci/test_lane_policy.py"
FULL_ID = re.compile(r"^[a-z0-9_-]+(?:::(?:bin/)?[A-Za-z0-9_-]+)?$")
TARGET_KINDS = frozenset({"lib", "test", "bin"})


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
class LaneEntry:
    binary_id: str
    lane: str
    reason: str
    resource_class: str | None
    binary_name_override: str | None = None

    @property
    def package(self) -> str:
        return self.binary_id.split("::", 1)[0]

    @property
    def binary_name(self) -> str:
        if self.binary_name_override is not None:
            return self.binary_name_override
        if "::" not in self.binary_id:
            return self.package.replace("-", "_")
        target = self.binary_id.split("::", 1)[1]
        return target.removeprefix("bin/")

    @property
    def target_kind(self) -> str:
        return target_kind_for_id(self.binary_id)

    @property
    def target_key(self) -> tuple[str, str, str]:
        return (self.package, self.binary_name, self.target_kind)


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


def load_catalog(path: Path) -> list[LaneEntry]:
    catalog = _load_toml(path, "test-lane catalog")
    if set(catalog) != {"binary"}:
        raise LanePolicyError("test-lane catalog top-level keys must be exactly binary")
    raw_entries = catalog["binary"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise LanePolicyError("test-lane catalog binary must be a non-empty array of tables")

    entries: list[LaneEntry] = []
    seen_ids: set[str] = set()
    seen_targets: dict[tuple[str, str, str], str] = {}
    for index, raw in enumerate(raw_entries):
        location = f"binary[{index}]"
        if not isinstance(raw, dict):
            raise LanePolicyError(f"{location} must be a table")
        allowed = {"id", "lane", "reason", "resource_class", "binary_name"}
        unknown = set(raw) - allowed
        if unknown:
            raise LanePolicyError(f"{location} has unknown keys: {', '.join(sorted(unknown))}")
        for required in ("id", "lane", "reason"):
            if not isinstance(raw.get(required), str) or not raw[required].strip():
                raise LanePolicyError(f"{location}.{required} must be a non-empty string")

        binary_id = raw["id"]
        lane = raw["lane"]
        reason = raw["reason"].strip()
        resource_class = raw.get("resource_class")
        binary_name_override = raw.get("binary_name")
        if not FULL_ID.fullmatch(binary_id):
            raise LanePolicyError(f"{location}.id is not a complete nextest binary id: {binary_id!r}")
        if binary_id in seen_ids:
            raise LanePolicyError(f"duplicate catalog binary id: {binary_id}")
        if lane not in {"cpu", "gpu"}:
            raise LanePolicyError(f"{location}.lane must be cpu or gpu")
        if lane == "cpu" and resource_class is not None:
            raise LanePolicyError(f"CPU entry {binary_id} must not declare resource_class")
        if lane == "gpu" and resource_class not in {"shared", "exclusive"}:
            raise LanePolicyError(f"GPU entry {binary_id} requires resource_class=shared or exclusive")
        if binary_name_override is not None and (
            not isinstance(binary_name_override, str)
            or not binary_name_override.strip()
            or any(char in binary_name_override for char in "\t\r\n")
        ):
            raise LanePolicyError(f"{location}.binary_name must be a non-empty single-line string")

        entry = LaneEntry(binary_id, lane, reason, resource_class, binary_name_override)
        previous = seen_targets.get(entry.target_key)
        if previous is not None:
            raise LanePolicyError(
                "catalog target collision "
                f"{entry.package!r}/{entry.binary_name!r} kind={entry.target_kind}: {previous} and {binary_id}"
            )
        seen_ids.add(binary_id)
        seen_targets[entry.target_key] = binary_id
        entries.append(entry)

    if not any(entry.resource_class == "exclusive" for entry in entries):
        raise LanePolicyError("catalog must declare at least one exclusive GPU binary")
    return sorted(entries, key=lambda entry: entry.binary_id)


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


def merge_inventories(inventories: list[list[InventoryBinary]]) -> list[InventoryBinary]:
    """Union one or more nextest inventories.

    Matching binary-id + binary-name + target-kind triples from distinct inventories coalesce so
    the harness can validate default-member and explicit bench surfaces both
    before and after bench leaves default-members. Conflicting names fail closed.
    """
    if not inventories:
        raise InventoryError("at least one nextest JSON inventory is required")

    by_id: dict[str, InventoryBinary] = {}
    seen_targets: dict[tuple[str, str, str], str] = {}
    for records in inventories:
        for record in records:
            existing = by_id.get(record.binary_id)
            if existing is not None:
                if (existing.binary_name, existing.target_kind) != (record.binary_name, record.target_kind):
                    raise LanePolicyError(
                        "conflicting binary metadata for "
                        f"{record.binary_id}: "
                        f"{existing.binary_name!r}/{existing.target_kind} vs "
                        f"{record.binary_name!r}/{record.target_kind}"
                    )
                continue
            previous = seen_targets.get(record.target_key)
            if previous is not None:
                raise LanePolicyError(
                    "test-binary target collision "
                    f"{record.package!r}/{record.binary_name!r} kind={record.target_kind}: "
                    f"{previous} and {record.binary_id}"
                )
            by_id[record.binary_id] = record
            seen_targets[record.target_key] = record.binary_id

    if not by_id:
        raise InventoryError("nextest JSON inventory contained no test binaries")
    return sorted(by_id.values(), key=lambda record: record.binary_id)


def filter_expression(entry: LaneEntry) -> str:
    if "::" not in entry.binary_id:
        return f"(package({entry.package}) & kind(lib))"
    return f"(package({entry.package}) & kind({entry.target_kind}) & binary({entry.binary_name}))"


def filter_block(entries: list[LaneEntry]) -> str:
    expressions = [filter_expression(entry) for entry in entries]
    return "\n| ".join(expressions)


def generated_overrides(entries: list[LaneEntry]) -> str:
    shared = [entry for entry in entries if entry.lane == "gpu" and entry.resource_class == "shared"]
    exclusive = [entry for entry in entries if entry.lane == "gpu" and entry.resource_class == "exclusive"]
    if not shared or not exclusive:
        raise LanePolicyError("generated overrides require both shared and exclusive GPU entries")

    # Exclusive tests must also not overlap the shared GPU group when the
    # whole suite runs in one invocation (e.g. a bare `cargo nextest run`).
    # A test-group cap only serializes within the group, so the exclusive
    # override reserves the entire test-thread pool — the only nextest
    # mechanism that excludes across groups.
    blocks = [
        ("default", "gpu", shared, []),
        ("default", "gpu-exclusive", exclusive, ['threads-required = "num-test-threads"']),
    ]
    lines = [
        GENERATED_BEGIN,
        "# Source: .config/policies/test-lanes.toml. Do not edit this section by hand.",
        "# Regenerate: python3 scripts/ci/test_lane_policy.py generate-nextest --write",
    ]
    for profile, group, group_entries, extra_lines in blocks:
        lines.extend(
            [
                "",
                f"[[profile.{profile}.overrides]]",
                "filter = '''",
                filter_block(group_entries),
                "'''",
                f'test-group = "{group}"',
                *extra_lines,
            ]
        )
    lines.extend([GENERATED_END, ""])
    return "\n".join(lines)


def generated_section(text: str) -> str:
    if text.count(GENERATED_BEGIN) != 1 or text.count(GENERATED_END) != 1:
        raise LanePolicyError("nextest config must contain exactly one generated override marker pair")
    start = text.index(GENERATED_BEGIN)
    end = text.index(GENERATED_END, start) + len(GENERATED_END)
    if end < len(text) and text[end] == "\n":
        end += 1
    return text[start:end]


def check_nextest(catalog: list[LaneEntry], path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise LanePolicyError(f"cannot read nextest config {path}: {error}") from error
    _load_toml(path, "nextest config")
    expected = generated_overrides(catalog)
    actual = generated_section(text)
    if actual != expected:
        raise LanePolicyError(
            "generated nextest overrides drifted from .config/policies/test-lanes.toml; run "
            "`python3 scripts/ci/test_lane_policy.py generate-nextest --write`"
        )


def write_nextest(catalog: list[LaneEntry], path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise LanePolicyError(f"cannot read nextest config {path}: {error}") from error
    old = generated_section(text)
    new = generated_overrides(catalog)
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def validate_inventory(catalog: list[LaneEntry], inventory: list[InventoryBinary], *, narrow: bool) -> list[str]:
    catalog_by_id = {entry.binary_id: entry for entry in catalog}
    inventory_by_id = {record.binary_id: record for record in inventory}
    problems: list[str] = []
    for binary_id in sorted(inventory_by_id.keys() - catalog_by_id.keys()):
        problems.append(f"unclassified test binary: {binary_id}")
    stale = sorted(catalog_by_id.keys() - inventory_by_id.keys())
    if stale and not narrow:
        problems.extend(f"stale all-feature catalog entry: {binary_id}" for binary_id in stale)
    for binary_id in sorted(catalog_by_id.keys() & inventory_by_id.keys()):
        expected_name = catalog_by_id[binary_id].binary_name
        actual_name = inventory_by_id[binary_id].binary_name
        if expected_name != actual_name:
            problems.append(
                f"binary-name mismatch for {binary_id}: catalog derives {expected_name!r}, inventory reports {actual_name!r}"
            )
        expected_kind = catalog_by_id[binary_id].target_kind
        actual_kind = inventory_by_id[binary_id].target_kind
        if expected_kind != actual_kind:
            problems.append(
                f"target-kind mismatch for {binary_id}: catalog derives {expected_kind!r}, inventory reports {actual_kind!r}"
            )
    if problems:
        raise LanePolicyError("\n".join(problems))
    return stale


def command_check_catalog(args: argparse.Namespace) -> None:
    entries = load_catalog(args.catalog)
    cpu = sum(entry.lane == "cpu" for entry in entries)
    shared = sum(entry.resource_class == "shared" for entry in entries)
    exclusive = sum(entry.resource_class == "exclusive" for entry in entries)
    print(f"OK: catalog has {len(entries)} binaries (cpu={cpu}, gpu_shared={shared}, gpu_exclusive={exclusive}).")


def command_validate(args: argparse.Namespace) -> None:
    entries = load_catalog(args.catalog)
    check_nextest(entries, args.nextest)
    inventory = merge_inventories([load_inventory(path) for path in args.inventory])
    stale = validate_inventory(entries, inventory, narrow=args.feature_mode == "narrow")
    if stale:
        print("WARN: catalog binaries absent from explicit narrower feature inventory:", file=sys.stderr)
        for binary_id in stale:
            print(f"  - {binary_id}", file=sys.stderr)
    print(f"OK: all {len(inventory)} live test binaries are cataloged exactly once.")


def command_generate(args: argparse.Namespace) -> None:
    entries = load_catalog(args.catalog)
    if args.check:
        check_nextest(entries, args.nextest)
        print("OK: generated nextest overrides match .config/policies/test-lanes.toml.")
    else:
        write_nextest(entries, args.nextest)
        check_nextest(entries, args.nextest)
        print(f"WROTE: regenerated test-lane overrides in {args.nextest}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--nextest", type=Path, default=DEFAULT_NEXTEST)
    subparsers = parser.add_subparsers(dest="command", required=True)

    catalog = subparsers.add_parser("check-catalog")
    catalog.set_defaults(function=command_check_catalog)

    validate = subparsers.add_parser("validate-inventory")
    validate.add_argument(
        "--inventory",
        type=Path,
        required=True,
        action="append",
        help="nextest binaries-only JSON inventory; repeat to union multiple package surfaces",
    )
    validate.add_argument("--feature-mode", choices=("all", "narrow"), required=True)
    validate.set_defaults(function=command_validate)

    generate = subparsers.add_parser("generate-nextest")
    mode = generate.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    generate.set_defaults(function=command_generate)
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
