#!/usr/bin/env python3
"""Extract fail-closed CI test counts from nextest or Cargo harness output."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NEXTEST_SUMMARY = re.compile(r"\bSummary\s+\[[^]]+]\s+([0-9]+)\s+tests?\s+run(?::\s*(.*))?$")
NEXTEST_CATEGORY = re.compile(r"([0-9]+)\s+(passed|failed|skipped)\b")
CARGO_RESULT = re.compile(
    r"^test result: (?:ok|FAILED)\. "
    r"([0-9]+) passed; ([0-9]+) failed; ([0-9]+) ignored; "
    r"([0-9]+) measured; ([0-9]+) filtered out; finished in .+$"
)


class CountError(RuntimeError):
    """The test output did not contain one trustworthy count summary."""


def clean_lines(text: str) -> list[str]:
    return [ANSI_ESCAPE.sub("", line).strip() for line in text.splitlines()]


def summarize_nextest(text: str) -> str:
    summaries: list[tuple[int, dict[str, int]]] = []
    for line in clean_lines(text):
        match = NEXTEST_SUMMARY.search(line)
        if match is None:
            continue
        total = int(match.group(1))
        categories = {"passed": 0, "failed": 0, "skipped": 0}
        for value, name in NEXTEST_CATEGORY.findall(match.group(2) or ""):
            categories[name] = int(value)
        if categories["passed"] + categories["failed"] != total:
            raise CountError(
                "nextest summary run count does not equal passed + failed: "
                f"run={total} passed={categories['passed']} failed={categories['failed']}"
            )
        summaries.append((total, categories))
    if len(summaries) != 1:
        raise CountError(f"expected exactly one nextest Summary line, found {len(summaries)}")
    total, categories = summaries[0]
    return (
        f"run={total} passed={categories['passed']} failed={categories['failed']} "
        f"skipped={categories['skipped']}"
    )


def summarize_cargo(text: str) -> str:
    rows = [CARGO_RESULT.fullmatch(line) for line in clean_lines(text)]
    parsed = [row for row in rows if row is not None]
    if not parsed:
        raise CountError("expected at least one Cargo test result line, found 0")
    passed = sum(int(row.group(1)) for row in parsed)
    failed = sum(int(row.group(2)) for row in parsed)
    ignored = sum(int(row.group(3)) for row in parsed)
    measured = sum(int(row.group(4)) for row in parsed)
    filtered = sum(int(row.group(5)) for row in parsed)
    executed = passed + failed + measured
    return (
        f"run={executed} passed={passed} failed={failed} ignored={ignored} "
        f"measured={measured} filtered={filtered} harnesses={len(parsed)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("nextest", "cargo"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    try:
        text = args.input.read_text(encoding="utf-8")
        summary = summarize_nextest(text) if args.kind == "nextest" else summarize_cargo(text)
    except (OSError, UnicodeError, CountError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
