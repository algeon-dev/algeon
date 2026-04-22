#!/usr/bin/env python3
"""Isolated parser/status tests for doctest accounting."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "doctest_report.py"


class DoctestReportTests(unittest.TestCase):
    def run_summary(self, output: str, status: int = 0) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
        with tempfile.TemporaryDirectory(prefix="doctest-report-") as temporary:
            root = Path(temporary)
            output_path = root / "output.txt"
            summary_path = root / "summary.json"
            output_path.write_text(output, encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "summarize",
                    "--input",
                    str(output_path),
                    "--command-status",
                    str(status),
                    "--json-out",
                    str(summary_path),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            document = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
            return result, document

    def test_multiple_crates_are_summarized_per_harness_and_in_total(self) -> None:
        result, document = self.run_summary(
            """
   Doc-tests first_crate

running 3 tests
test result: ok. 2 passed; 0 failed; 1 ignored; 0 measured; 4 filtered out; finished in 0.01s
   Doc-tests second_crate

running 4 tests
test result: FAILED. 2 passed; 1 failed; 0 ignored; 1 measured; 3 filtered out; finished in 0.02s
""",
            status=101,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(document["command_status"], 101)
        self.assertEqual(document["parse_status"], "ok")
        self.assertEqual([row["name"] for row in document["harnesses"]], ["first_crate", "second_crate"])
        self.assertEqual(
            document["totals"],
            {"harnesses": 2, "executed": 6, "passed": 4, "failed": 1, "ignored": 1, "measured": 1, "filtered": 7},
        )

    def test_zero_test_harness_reports_actual_executed_zero(self) -> None:
        result, document = self.run_summary(
            """Doc-tests empty_crate
running 0 tests
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
"""
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(document["totals"]["executed"], 0)
        self.assertIn("executed=0", result.stdout)

    def test_no_harness_fails_closed_instead_of_assuming_zero_tests(self) -> None:
        result, document = self.run_summary("Finished test profile\n")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(document["parse_status"], "error")
        self.assertEqual(document["totals"]["harnesses"], 0)
        self.assertEqual(document["totals"]["executed"], 0)
        self.assertIn("no Doc-tests harness results found", result.stderr)

    def test_failure_status_is_preserved_without_becoming_parser_status(self) -> None:
        result, document = self.run_summary(
            """Doc-tests failing_crate
test result: FAILED. 0 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
""",
            status=37,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(document["command_status"], 37)
        self.assertEqual(document["totals"]["failed"], 2)

    def test_malformed_result_and_incomplete_harness_fail_closed(self) -> None:
        cases = (
            "Doc-tests broken\ntest result: maybe\n",
            "Doc-tests broken\nrunning 1 test\n",
            "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n",
        )
        for output in cases:
            with self.subTest(output=output):
                result, document = self.run_summary(output, status=55)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(document["command_status"], 55)
                self.assertEqual(document["parse_status"], "error")
                self.assertIn("ERROR:", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
