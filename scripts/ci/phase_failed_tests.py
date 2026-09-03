#!/usr/bin/env python3
"""Extract failed test names from a check_all.sh phase log for the final summary."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NEXTEST_FAILURE = re.compile(r"^FAIL\s+\[[^\]]*\]\s+(.+)$")


def failed_nextest(text: str) -> list[str]:
    names: list[str] = []
    for line in text.splitlines():
        match = NEXTEST_FAILURE.match(ANSI_ESCAPE.sub("", line).strip())
        if match is None:
            continue
        name = match.group(1).strip()
        if name and name not in names:
            names.append(name)
    return names


def failed_cargo(text: str) -> list[str]:
    names: list[str] = []
    in_failures = False
    for raw_line in text.splitlines():
        line = ANSI_ESCAPE.sub("", raw_line).rstrip()
        if line.strip() == "failures:":
            in_failures = True
            continue
        if not in_failures:
            continue
        if not line.strip():
            continue
        if line[0] in " \t":
            name = line.strip()
            if name not in names:
                names.append(name)
        else:
            in_failures = False
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("nextest", "cargo"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    try:
        text = args.input.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    names = failed_nextest(text) if args.kind == "nextest" else failed_cargo(text)
    for name in names:
        print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
