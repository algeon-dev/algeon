#!/usr/bin/env python3
"""Unit tests for phase-table nextest and Cargo count extraction."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase_test_counts import CountError, summarize_cargo, summarize_nextest


class PhaseTestCountsTests(unittest.TestCase):
    def test_nextest_pass_and_failure_summaries(self) -> None:
        self.assertEqual(
            summarize_nextest("Summary [ 0.100s] 3 tests run: 3 passed, 8 skipped\n"),
            "run=3 passed=3 failed=0 skipped=8",
        )
        self.assertEqual(
            summarize_nextest("Summary [ 0.200s] 4 tests run: 3 passed, 1 failed, 2 skipped\n"),
            "run=4 passed=3 failed=1 skipped=2",
        )

    def test_cargo_sums_multiple_harnesses(self) -> None:
        output = """
test result: ok. 2 passed; 0 failed; 1 ignored; 0 measured; 3 filtered out; finished in 0.01s
test result: FAILED. 1 passed; 1 failed; 0 ignored; 1 measured; 0 filtered out; finished in 0.02s
"""
        self.assertEqual(
            summarize_cargo(output),
            "run=5 passed=3 failed=1 ignored=1 measured=1 filtered=3 harnesses=2",
        )

    def test_missing_or_inconsistent_summaries_fail_closed(self) -> None:
        with self.assertRaises(CountError):
            summarize_nextest("Finished test profile\n")
        with self.assertRaises(CountError):
            summarize_nextest("Summary [ 0.100s] 3 tests run: 2 passed, 0 failed\n")
        with self.assertRaises(CountError):
            summarize_cargo("Finished test profile\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
