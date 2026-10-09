#!/usr/bin/env python3
"""Reject known pass-as-skip mechanisms from canonical Rust test roots."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_TEST_ROOTS = (ROOT / "crates", ROOT / "components")
RULES = {
    "optional_gpu_runtime": re.compile(
        r"let\s+Ok\s*\([^)]*runtime[^)]*\)\s*=\s*cudf::Runtime::new\s*\(\s*\)\s*else\s*\{\s*return\s*;\s*\}",
        re.DOTALL,
    ),
    "device_count_skip": re.compile(
        r"(?:if\s+)?(?:cudf::)?DeviceId::count\s*\(\)(?:\s*\.unwrap_or\s*\(\s*0\s*\))?\s*(?:==|!=|<=|>=|<|>)\s*\d+\s*\{\s*return\s*;\s*\}",
        re.DOTALL,
    ),
    "bound_device_count_skip": re.compile(
        r"let\s+(?P<count>[A-Za-z_][A-Za-z_0-9]*)\s*=\s*(?:cudf::)?DeviceId::count\s*\(\s*\)"
        r"\s*\.\s*(?:expect\s*\(\s*\"(?:[^\"\\]|\\.)*\"\s*\)|unwrap\s*\(\s*\)|unwrap_or\s*\(\s*\d+\s*\))"
        r"\s*;\s*if\s+(?P=count)\s*(?:==|!=|<=|>=|<|>)\s*\d+\s*\{\s*return\s*;\s*\}",
        re.DOTALL,
    ),
    "optional_python_oracle": re.compile(
        r"handle_cugraph_python_oracle_unavailable\s*\([^;]*\)\s*;\s*return\s+None\s*;",
        re.DOTALL,
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, action="append", default=[])
    args = parser.parse_args()

    roots = args.root or list(CANONICAL_TEST_ROOTS)
    violations: list[str] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.rs")):
            relative = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()
            text = path.read_text(encoding="utf-8")
            for rule, pattern in RULES.items():
                if pattern.search(text):
                    violations.append(f"{relative}: {rule}")
    if violations:
        print("ERROR: canonical Rust tests contain pass-as-skip mechanisms:", file=sys.stderr)
        print("\n".join(f"  {violation}" for violation in violations), file=sys.stderr)
        return 1
    print("pass-as-skip policy: no pass-as-skip mechanisms found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
