#!/usr/bin/env python3
"""Reject known pass-as-skip mechanisms from canonical Rust test roots."""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALLOWLIST = ROOT / ".config" / "policies" / "pass-as-skip-allowlist.toml"
CANONICAL_TEST_ROOTS = (ROOT / "tests", ROOT / "crates", ROOT / "src")
RULES = {
    "optional_sf1_fixture": re.compile(r"\bsf1_fixture_root_or_skip\s*\("),
    "optional_gpu_runtime": re.compile(
        r"let\s+Ok\s*\([^)]*runtime[^)]*\)\s*=\s*cudf::Runtime::new\s*\(\s*\)\s*else\s*\{\s*return\s*;\s*\}",
        re.DOTALL,
    ),
    "device_count_skip": re.compile(
        r"(?:if\s+)?(?:cudf::)?DeviceId::count\s*\(\)(?:\s*\.unwrap_or\s*\(\s*0\s*\))?\s*==\s*0\s*\{\s*return\s*;\s*\}",
        re.DOTALL,
    ),
    "optional_python_oracle": re.compile(
        r"handle_cugraph_python_oracle_unavailable\s*\([^;]*\)\s*;\s*return\s+None\s*;",
        re.DOTALL,
    ),
    "sibling_source_skip": re.compile(r"skipping\s+cugraph\s+crate\s+drift\s+test", re.IGNORECASE),
    "sibling_source_scrape": re.compile(r"cugraph-nexus[^\"']*src/graph\.rs"),
}


def load_allowlist(path: Path) -> set[tuple[str, str]]:
    try:
        with path.open("rb") as source:
            data = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"cannot load pass-as-skip allowlist {path}: {error}") from error
    entries = data.get("allow", [])
    if set(data) not in (set(), {"allow"}) or not isinstance(entries, list):
        raise ValueError("pass-as-skip allowlist must contain only [[allow]] entries")
    allowed: set[tuple[str, str]] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {"path", "rule", "owner", "reason"}:
            raise ValueError(f"allow[{index}] must contain path, rule, owner, and reason")
        if not all(isinstance(entry[key], str) and entry[key].strip() for key in entry):
            raise ValueError(f"allow[{index}] values must be non-empty strings")
        if entry["rule"] not in RULES:
            raise ValueError(f"allow[{index}].rule is unknown: {entry['rule']}")
        allowed.add((entry["path"], entry["rule"]))
    return allowed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    parser.add_argument("--root", type=Path, action="append", default=[])
    args = parser.parse_args()
    try:
        allowed = load_allowlist(args.allowlist)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    roots = args.root or list(CANONICAL_TEST_ROOTS)
    violations: list[str] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.rs")):
            relative = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()
            text = path.read_text(encoding="utf-8")
            for rule, pattern in RULES.items():
                if pattern.search(text) and (relative, rule) not in allowed:
                    violations.append(f"{relative}: {rule}")
    if violations:
        print("ERROR: canonical Rust tests contain pass-as-skip mechanisms:", file=sys.stderr)
        print("\n".join(f"  {violation}" for violation in violations), file=sys.stderr)
        return 1
    print("pass-as-skip policy: no unallowlisted mechanisms found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
