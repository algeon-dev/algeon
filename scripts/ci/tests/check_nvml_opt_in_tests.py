#!/usr/bin/env python3
"""Exercise the NVML opt-in graph policy with a fake resolver."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/ci"))
import check_nvml_opt_in as policy


PACKAGES = {
    "bench": ["cugraph", "nvml"],
    "tools": ["nvml", "single-node-multi-gpu-validate-run"],
    "datafusion-facade-consumer": [],
}
REQUIRED = (("bench", "nvml"), ("tools", "single-node-multi-gpu-validate-run,nvml"))


def resolver(nvml_selections: set[tuple[str, str]]) -> policy.Resolver:
    return lambda package, features: (package, features) in nvml_selections


class NvmlOptInTests(unittest.TestCase):
    def test_accepts_nvml_only_for_explicit_selections(self):
        selected = {("bench", "nvml"), ("tools", "single-node-multi-gpu-validate-run,nvml"),
                    ("datafusion-facade-consumer", "")}
        self.assertEqual(policy.violations(PACKAGES, resolver(selected), REQUIRED), [])

    def test_rejects_a_run_feature_that_enables_nvml(self):
        selected = {("bench", "nvml"), ("tools", "single-node-multi-gpu-validate-run,nvml"),
                    ("tools", "single-node-multi-gpu-validate-run")}
        self.assertEqual(policy.violations(PACKAGES, resolver(selected), REQUIRED), [
            "tools [single-node-multi-gpu-validate-run] resolves nvml-wrapper without selecting nvml",
        ])

    def test_rejects_an_unconditional_dependency_edge(self):
        selected = {("bench", ""), ("bench", "cugraph"), ("bench", "nvml"),
                    ("tools", "single-node-multi-gpu-validate-run,nvml")}
        self.assertEqual(policy.violations(PACKAGES, resolver(selected), REQUIRED), [
            "bench [no features] resolves nvml-wrapper without selecting nvml",
            "bench [cugraph] resolves nvml-wrapper without selecting nvml",
        ])

    def test_rejects_broken_forwarding_and_unknown_required_packages(self):
        selected = {("bench", "nvml")}
        self.assertEqual(policy.violations(PACKAGES, resolver(selected), (*REQUIRED, ("server", "nvml"))), [
            "tools [single-node-multi-gpu-validate-run,nvml] does not resolve nvml-wrapper",
            "required NVML selection names unknown package server",
        ])


if __name__ == "__main__":
    unittest.main()
