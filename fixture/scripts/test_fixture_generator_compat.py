#!/usr/bin/env python3
"""Compatibility tests for the public TPC-H/TPC-DS fixture entrypoints.

The tests use /bin/true and /bin/false as generator backends so they validate
argument routing, legacy defaults, and verification behavior without generating
large benchmark datasets.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_SH = REPO_ROOT / "fixture" / "scripts" / "fixture.sh"
GEN_TPCH = REPO_ROOT / "fixture" / "scripts" / "gen_tpch.py"
GEN_TPCDS = REPO_ROOT / "fixture" / "scripts" / "gen_tpcds.py"
LEGACY_TPCH = REPO_ROOT / "fixture" / "scripts" / "workloads" / "tpch" / "generate.sh"
LEGACY_TPCDS = REPO_ROOT / "fixture" / "scripts" / "workloads" / "tpcds" / "generate.sh"


def run_cmd(args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        args,
        cwd=REPO_ROOT,
        env=merged_env,
        text=True,
        capture_output=True,
        check=False,
    )


def load_script_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FixtureGeneratorCompatibilityTests(unittest.TestCase):
    def test_generators_default_to_repo_sf1_roots(self) -> None:
        tpch = load_script_module("gen_tpch_under_test", GEN_TPCH)
        tpcds = load_script_module("gen_tpcds_under_test", GEN_TPCDS)

        tpch_args = tpch.parse_args([])
        tpcds_args = tpcds.parse_args([])

        self.assertEqual(tpch_args.out_root, REPO_ROOT / "fixture" / "data" / "tpch")
        self.assertEqual(tpcds_args.out_root, REPO_ROOT / "fixture" / "data" / "tpcds")
        self.assertEqual(tpch_args.scale or tpch_args.positional_scale or ["1"], ["1"])
        self.assertEqual(tpcds_args.scale or tpcds_args.positional_scale or ["1"], ["1"])

    def test_legacy_workload_wrappers_generate_raw_without_verification_dependency(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nexus_fixture_compat.") as tmp:
            root = Path(tmp)

            tpch = run_cmd(
                [
                    str(LEGACY_TPCH),
                    "--scale",
                    "sf1",
                    "--out",
                    str(root / "tpch_out"),
                    "--tpchgen-bin",
                    "/bin/true",
                ]
            )
            self.assertEqual(tpch.returncode, 0, tpch.stderr)
            self.assertIn("generating raw sf1", tpch.stdout)

            tpcds = run_cmd(
                [
                    str(LEGACY_TPCDS),
                    "sf1",
                    "--out",
                    str(root / "tpcds_out"),
                    "--duckdb",
                    "/bin/true",
                ]
            )
            self.assertEqual(tpcds.returncode, 0, tpcds.stderr)
            self.assertIn("generating sf1 (raw)", tpcds.stdout)

    def test_fixture_router_legacy_calls_generate_raw_without_verification_dependency(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nexus_fixture_router.") as tmp:
            root = Path(tmp)
            tpch = run_cmd(
                [
                    str(FIXTURE_SH),
                    "tpch",
                    "generate",
                    "sf1",
                    "--out",
                    str(root / "router_tpch"),
                    "--tpchgen-bin",
                    "/bin/true",
                ]
            )
            self.assertEqual(tpch.returncode, 0, tpch.stderr)
            self.assertIn("generating raw sf1", tpch.stdout)

            tpcds = run_cmd(
                [
                    str(FIXTURE_SH),
                    "tpcds",
                    "generate",
                    "sf1",
                    "--out",
                    str(root / "router_tpcds"),
                    "--duckdb",
                    "/bin/true",
                ]
            )
            self.assertEqual(tpcds.returncode, 0, tpcds.stderr)
            self.assertIn("generating sf1 (raw)", tpcds.stdout)

    def test_force_env_preserves_legacy_overwrite_behavior(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nexus_fixture_force.") as tmp:
            out_dir = Path(tmp) / "tpch_force"
            out_dir.mkdir()
            stale = out_dir / "stale.parquet"
            stale.write_bytes(b"stale")

            result = run_cmd(
                [
                    str(LEGACY_TPCH),
                    "--out",
                    str(out_dir),
                    "--tpchgen-bin",
                    "/bin/true",
                ],
                env={"FORCE": "1"},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(stale.exists())

    def test_new_generator_entrypoints_remain_strict_by_default(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nexus_fixture_strict.") as tmp:
            root = Path(tmp)

            tpch = run_cmd(
                [
                    sys.executable,
                    str(GEN_TPCH),
                    "--scale",
                    "1",
                    "--variant",
                    "raw",
                    "--out-root",
                    str(root / "tpch"),
                    "--tpchgen-bin",
                    "/bin/true",
                ]
            )
            self.assertNotEqual(tpch.returncode, 0)

            tpcds = run_cmd(
                [
                    sys.executable,
                    str(GEN_TPCDS),
                    "--scale",
                    "1",
                    "--variant",
                    "raw",
                    "--out-root",
                    str(root / "tpcds"),
                    "--duckdb",
                    "/bin/true",
                ]
            )
            self.assertNotEqual(tpcds.returncode, 0)

    def test_new_optimized_variant_reaches_backend_with_out_root(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nexus_fixture_optimized.") as tmp:
            root = Path(tmp)

            tpch = run_cmd(
                [
                    sys.executable,
                    str(GEN_TPCH),
                    "--scale",
                    "1",
                    "--variant",
                    "optimized",
                    "--out-root",
                    str(root / "tpch"),
                    "--tpchgen-bin",
                    "/bin/false",
                    "--no-verify",
                ]
            )
            self.assertNotEqual(tpch.returncode, 0)
            self.assertIn("generating optimized sf1", tpch.stdout)

            tpcds = run_cmd(
                [
                    sys.executable,
                    str(GEN_TPCDS),
                    "--scale",
                    "1",
                    "--variant",
                    "optimized",
                    "--out-root",
                    str(root / "tpcds"),
                    "--duckdb",
                    "/bin/false",
                    "--no-verify",
                ]
            )
            self.assertNotEqual(tpcds.returncode, 0)
            self.assertIn("generating sf1 (optimized)", tpcds.stdout)

    def test_fixture_router_new_optimized_variant_reaches_backend_with_out_root(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nexus_fixture_router_optimized.") as tmp:
            root = Path(tmp)

            tpch = run_cmd(
                [
                    str(FIXTURE_SH),
                    "tpch",
                    "generate",
                    "--scale",
                    "1",
                    "--variant",
                    "optimized",
                    "--out-root",
                    str(root / "tpch"),
                    "--tpchgen-bin",
                    "/bin/false",
                    "--no-verify",
                ]
            )
            self.assertNotEqual(tpch.returncode, 0)
            self.assertIn("generating optimized sf1", tpch.stdout)

            tpcds = run_cmd(
                [
                    str(FIXTURE_SH),
                    "tpcds",
                    "generate",
                    "--scale",
                    "1",
                    "--variant",
                    "optimized",
                    "--out-root",
                    str(root / "tpcds"),
                    "--duckdb",
                    "/bin/false",
                    "--no-verify",
                ]
            )
            self.assertNotEqual(tpcds.returncode, 0)
            self.assertIn("generating sf1 (optimized)", tpcds.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
