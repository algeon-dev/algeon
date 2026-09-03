#!/usr/bin/env python3
"""Unit tests for phase-log failed test name extraction."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase_failed_tests import failed_cargo, failed_nextest


class PhaseFailedTestsTests(unittest.TestCase):
    def test_nextest_collects_unique_names_in_order(self) -> None:
        output = (
            "PASS [   0.001s] datafusion_nexus::cpu_ok\n"
            "\x1b[31mFAIL [   0.002s] datafusion_nexus::cpu_broke\x1b[0m\n"
            "Summary [ 0.200s] 3 tests run: 2 passed, 1 failed, 5 skipped\n"
            "    FAIL [   0.002s] datafusion_nexus::cpu_broke\n"
            "error: test run failed\n"
        )
        self.assertEqual(failed_nextest(output), ["datafusion_nexus::cpu_broke"])

    def test_nextest_ignores_non_failure_lines(self) -> None:
        output = "Summary [ 0.100s] 3 tests run: 3 passed, 5 skipped\nFAILURE\n"
        self.assertEqual(failed_nextest(output), [])

    def test_cargo_collects_failures_sections_across_harnesses(self) -> None:
        output = """
Doc-tests fake_one
failures:

    src/lib.rs - doctest_broke (line 10)

test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s

---- src/lib.rs - doctest_broke (line 10) stdout ----
panicked
Doc-tests fake_two
failures:

    unit_broke

test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s
"""
        self.assertEqual(
            failed_cargo(output),
            ["src/lib.rs - doctest_broke (line 10)", "unit_broke"],
        )

    def test_cargo_without_failures_section_is_empty(self) -> None:
        output = "test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s\n"
        self.assertEqual(failed_cargo(output), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
