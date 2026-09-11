from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "tpcds_native_profile_sweep.py"


def load_module():
    spec = importlib.util.spec_from_file_location("tpcds_native_profile_sweep", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TpcdsNativeProfileSweepTest(unittest.TestCase):
    def test_build_candidates_crosses_budget_chunk_and_row_group_dimensions(self) -> None:
        module = load_module()
        args = SimpleNamespace(row_groups="1,8", source_chunk_bytes="64MiB,2GiB", device_budgets="auto,32GiB")

        candidates = module.build_candidates(args)

        self.assertEqual(len(candidates), 8)
        self.assertEqual(candidates[0].candidate_id, "budgetauto_src64MiB_rg1")
        self.assertIsNone(candidates[0].device_budget_bytes)
        self.assertEqual(candidates[-1].device_budget_bytes, 32 * 1024**3)
        self.assertEqual(candidates[-1].max_source_chunk_bytes, 2 * 1024**3)
        self.assertEqual(candidates[-1].max_row_groups_per_chunk, 8)

    def test_candidate_env_omits_auto_budget_and_sets_source_knobs(self) -> None:
        module = load_module()
        args = SimpleNamespace(
            parquet_dir=Path("/data/tpcds"),
            queries="q44,q96",
            warmups=1,
            iterations=3,
            skip_staleness_check=True,
        )
        candidate = module.Candidate(
            candidate_id="budgetauto_src32GiB_rg1024",
            device_budget_label="auto",
            device_budget_bytes=None,
            max_source_chunk_bytes=32 * 1024**3,
            max_row_groups_per_chunk=1024,
        )

        env = module.candidate_env(args, candidate)

        self.assertEqual(env["ALGEON_TPCDS_BENCH_ENGINE"], "native_gpu")
        self.assertEqual(env["ALGEON_TPCDS_BENCH_QUERIES"], "q44,q96")
        self.assertEqual(env["ALGEON_TPCDS_NATIVE_PROFILE"], "conservative")
        self.assertEqual(env["ALGEON_TPCDS_NATIVE_MAX_SOURCE_CHUNK_BYTES"], str(32 * 1024**3))
        self.assertEqual(env["ALGEON_TPCDS_NATIVE_MAX_ROW_GROUPS_PER_CHUNK"], "1024")
        self.assertEqual(env["ALGEON_TPCDS_NATIVE_SOURCE_CHUNK_PLANNING_MODE"], "conservative")
        self.assertNotIn("ALGEON_TPCDS_NATIVE_DEVICE_BUDGET_BYTES", env)

    def test_candidate_env_clears_ambient_profile_for_auto_budget(self) -> None:
        module = load_module()
        args = SimpleNamespace(
            parquet_dir=Path("/data/tpcds"),
            queries="q44",
            warmups=1,
            iterations=3,
            skip_staleness_check=False,
        )
        candidate = module.Candidate(
            candidate_id="budgetauto_src64MiB_rg1",
            device_budget_label="auto",
            device_budget_bytes=None,
            max_source_chunk_bytes=64 * 1024**2,
            max_row_groups_per_chunk=1,
        )

        old_profile = os.environ.get("ALGEON_TPCDS_NATIVE_PROFILE")
        old_budget = os.environ.get("ALGEON_TPCDS_NATIVE_DEVICE_BUDGET_BYTES")
        old_mode = os.environ.get("ALGEON_TPCDS_NATIVE_SOURCE_CHUNK_PLANNING_MODE")
        try:
            os.environ["ALGEON_TPCDS_NATIVE_PROFILE"] = "sf100_gpu_throughput"
            os.environ["ALGEON_TPCDS_NATIVE_DEVICE_BUDGET_BYTES"] = "102005473280"
            os.environ["ALGEON_TPCDS_NATIVE_SOURCE_CHUNK_PLANNING_MODE"] = "local_parquet_throughput"
            env = module.candidate_env(args, candidate)
        finally:
            restore_env("ALGEON_TPCDS_NATIVE_PROFILE", old_profile)
            restore_env("ALGEON_TPCDS_NATIVE_DEVICE_BUDGET_BYTES", old_budget)
            restore_env("ALGEON_TPCDS_NATIVE_SOURCE_CHUNK_PLANNING_MODE", old_mode)

        self.assertEqual(env["ALGEON_TPCDS_NATIVE_PROFILE"], "conservative")
        self.assertEqual(env["ALGEON_TPCDS_NATIVE_SOURCE_CHUNK_PLANNING_MODE"], "conservative")
        self.assertNotIn("ALGEON_TPCDS_NATIVE_DEVICE_BUDGET_BYTES", env)

    def test_summarize_candidate_requires_all_measured_iterations_to_succeed(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.tsv"
            err = Path(tmp) / "candidate.stderr"
            err.write_text("", encoding="utf-8")
            path.write_text(
                "\n".join(
                    [
                        "# tpcds_benchmark_native_profile_name=custom",
                        (
                            "engine\tquery_id\titeration\twarmup\twall_ms\toutcome\treservation_live_peak_bytes"
                            "\treservation_denied_count\tretry_count\tsource_chunks_committed\trow_groups_committed"
                        ),
                        "native_gpu\tq44\t0\ttrue\t999.0\tsucceeded\t1\t0\t0\t1\t1",
                        "native_gpu\tq44\t0\tfalse\t30.0\tsucceeded\t100\t0\t1\t7\t4690",
                        "native_gpu\tq44\t1\tfalse\t10.0\tsucceeded\t120\t0\t0\t7\t4690",
                        "native_gpu\tq44\t2\tfalse\t20.0\tsucceeded\t110\t1\t0\t7\t4690",
                        "native_gpu\tq96\t0\tfalse\t\texecution_failed\t\t\t\t\t",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            args = SimpleNamespace(queries="q44,q96", warmups=1, iterations=3)
            candidate = module.Candidate("c0", "auto", None, 64 * 1024**2, 1)

            row = module.summarize_candidate(candidate, path, err, 0, args)

        self.assertEqual(row["outcome"], "failed")
        self.assertEqual(row["successful_queries"], "1")
        self.assertEqual(row["failed_query_ids"], "q96")
        self.assertEqual(row["partial_query_ids"], "")
        self.assertEqual(row["total_median_wall_ms"], "20.000")
        self.assertEqual(row["max_reservation_live_peak_bytes"], "120")
        self.assertEqual(row["total_reservation_denied_count"], "1")
        self.assertEqual(row["total_retry_count"], "1")
        self.assertEqual(row["total_source_chunks_committed"], "21")
        self.assertEqual(row["total_row_groups_committed"], str(4690 * 3))

    def test_summarize_candidate_marks_partial_success_as_failed(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.tsv"
            err = Path(tmp) / "candidate.stderr"
            err.write_text("", encoding="utf-8")
            path.write_text(
                "\n".join(
                    [
                        "engine\tquery_id\titeration\twarmup\twall_ms\toutcome\treservation_live_peak_bytes",
                        "native_gpu\tq09\t0\tfalse\t10.0\tsucceeded\t100",
                        "native_gpu\tq09\t1\tfalse\t\texecution_failed\t",
                        "native_gpu\tq09\t2\tfalse\t12.0\tsucceeded\t100",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            args = SimpleNamespace(queries="q09", warmups=1, iterations=3)
            candidate = module.Candidate("c0", "auto", None, 64 * 1024**2, 1)

            row = module.summarize_candidate(candidate, path, err, 0, args)

        self.assertEqual(row["outcome"], "failed")
        self.assertEqual(row["successful_queries"], "0")
        self.assertEqual(row["partial_queries"], "1")
        self.assertEqual(row["failed_query_ids"], "q09")
        self.assertEqual(row["partial_query_ids"], "q09")


if __name__ == "__main__":
    unittest.main()


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
