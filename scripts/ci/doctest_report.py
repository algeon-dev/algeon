#!/usr/bin/env python3
"""Parse and render Cargo/rustdoc doctest accounting for the CI harness."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


HARNESS = re.compile(r"^Doc-tests (.+)$")
RESULT = re.compile(
    r"^test result: (ok|FAILED)\. "
    r"([0-9]+) passed; ([0-9]+) failed; ([0-9]+) ignored; "
    r"([0-9]+) measured; ([0-9]+) filtered out; finished in .+$"
)


class DoctestReportError(RuntimeError):
    """Rustdoc output could not be accounted for reliably."""


@dataclass(frozen=True)
class HarnessResult:
    name: str
    outcome: str
    passed: int
    failed: int
    ignored: int
    measured: int
    filtered: int

    @property
    def executed(self) -> int:
        return self.passed + self.failed + self.measured


def parse_doctest_output(text: str) -> list[HarnessResult]:
    results: list[HarnessResult] = []
    pending: str | None = None
    seen: set[str] = set()
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        heading = HARNESS.fullmatch(line)
        if heading:
            if pending is not None:
                raise DoctestReportError(
                    f"harness {pending!r} has no result before the next Doc-tests heading at line {line_number}"
                )
            pending = heading.group(1)
            if pending in seen:
                raise DoctestReportError(f"duplicate Doc-tests harness heading at line {line_number}: {pending}")
            seen.add(pending)
            continue
        if line.startswith("test result:"):
            match = RESULT.fullmatch(line)
            if match is None:
                raise DoctestReportError(f"malformed doctest result line {line_number}: {line}")
            if pending is None:
                raise DoctestReportError(f"doctest result at line {line_number} has no preceding Doc-tests heading")
            outcome = match.group(1)
            counts = tuple(int(match.group(index)) for index in range(2, 7))
            passed, failed, ignored, measured, filtered = counts
            if (outcome == "ok") != (failed == 0):
                raise DoctestReportError(
                    f"doctest result for {pending!r} has inconsistent outcome={outcome} and failed={failed}"
                )
            results.append(HarnessResult(pending, outcome, passed, failed, ignored, measured, filtered))
            pending = None
    if pending is not None:
        raise DoctestReportError(f"harness {pending!r} has no result before end of output")
    if not results:
        raise DoctestReportError("no Doc-tests harness results found")
    return results


def totals(results: list[HarnessResult]) -> dict[str, int]:
    return {
        "harnesses": len(results),
        "executed": sum(result.executed for result in results),
        "passed": sum(result.passed for result in results),
        "failed": sum(result.failed for result in results),
        "ignored": sum(result.ignored for result in results),
        "measured": sum(result.measured for result in results),
        "filtered": sum(result.filtered for result in results),
    }


def report_document(results: list[HarnessResult], command_status: int) -> dict[str, object]:
    return {
        "schema_version": 1,
        "command_status": command_status,
        "parse_status": "ok",
        "harnesses": [asdict(result) | {"executed": result.executed} for result in results],
        "totals": totals(results),
    }


def error_document(command_status: int, error: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "command_status": command_status,
        "parse_status": "error",
        "error": error,
        "harnesses": [],
        "totals": totals([]),
    }


def write_json(path: Path, document: dict[str, object]) -> None:
    path.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DoctestReportError(f"cannot load doctest summary {path}: {error}") from error
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise DoctestReportError(f"doctest summary {path} has an unsupported schema")
    return value


def render(document: dict[str, object]) -> str:
    totals_value = document.get("totals")
    if not isinstance(totals_value, dict):
        raise DoctestReportError("doctest summary totals must be an object")
    required = ("harnesses", "executed", "passed", "failed", "ignored", "measured", "filtered")
    if any(not isinstance(totals_value.get(key), int) for key in required):
        raise DoctestReportError("doctest summary totals are malformed")
    command_status = document.get("command_status")
    parse_status = document.get("parse_status")
    if not isinstance(command_status, int) or not isinstance(parse_status, str):
        raise DoctestReportError("doctest summary status fields are malformed")
    return (
        f"command_status={command_status} parse_status={parse_status} "
        + " ".join(f"{key}={totals_value[key]}" for key in required)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--input", type=Path, required=True)
    summarize.add_argument("--command-status", type=int, required=True)
    summarize.add_argument("--json-out", type=Path, required=True)

    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--summary", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "render":
        try:
            print(render(load_json(args.summary)))
        except DoctestReportError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 2
        return 0

    if not 0 <= args.command_status <= 255:
        print("ERROR: --command-status must be between 0 and 255", file=sys.stderr)
        return 2
    try:
        output = args.input.read_text(encoding="utf-8")
        harnesses = parse_doctest_output(output)
        document = report_document(harnesses, args.command_status)
    except (OSError, UnicodeError, DoctestReportError) as error:
        document = error_document(args.command_status, str(error))
        write_json(args.json_out, document)
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    write_json(args.json_out, document)
    for harness in harnesses:
        print(
            f"DOCTEST HARNESS: name={harness.name} outcome={harness.outcome} executed={harness.executed} "
            f"passed={harness.passed} failed={harness.failed} ignored={harness.ignored} "
            f"measured={harness.measured} filtered={harness.filtered}"
        )
    print(f"DOCTEST TOTAL: {render(document)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
