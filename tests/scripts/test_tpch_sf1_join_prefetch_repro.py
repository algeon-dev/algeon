from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_tpch_sf1_join_prefetch_repro.py"


def load_module():
    spec = importlib.util.spec_from_file_location("check_tpch_sf1_join_prefetch_repro", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TpchSf1JoinPrefetchReproTest(unittest.TestCase):
    def test_checker_accepts_pinned_failure_shape(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            tsv = Path(tmp) / "tpch_sf1.tsv"
            write_tpch_tsv(tsv, max_source_reads_in_flight="2")

            summary = module.check_repro_tsv(tsv)

            self.assertEqual(summary.matched_count, 11)
            self.assertEqual(summary.native_failed_count, 11)
            self.assertEqual(summary.failed_query_ids, module.EXPECTED_FAILED_QUERY_IDS)
            self.assertEqual(summary.representative_failure_query_id, "q03")

    def test_checker_rejects_changed_in_flight_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tsv = Path(tmp) / "tpch_sf1.tsv"
            write_tpch_tsv(tsv, max_source_reads_in_flight="1")

            proc = run_checker(tsv)

            self.assertEqual(proc.returncode, 1)
            self.assertIn("tpch_benchmark_native_max_source_reads_in_flight=2", proc.stderr)

    def test_checker_rejects_changed_failure_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tsv = Path(tmp) / "tpch_sf1.tsv"
            write_tpch_tsv(tsv, failed_query_ids=["q03"])

            proc = run_checker(tsv)

            self.assertEqual(proc.returncode, 1)
            self.assertIn("expected native_failed=11", proc.stderr)
            self.assertIn("expected failed query ids q03,q05,q07,q08,q09,q10,q12,q13,q14,q16,q17", proc.stderr)

    def test_checker_rejects_missing_representative_budget_fact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tsv = Path(tmp) / "tpch_sf1.tsv"
            write_tpch_tsv(tsv, largest_failed_allocation_bytes="0")

            proc = run_checker(tsv)

            self.assertEqual(proc.returncode, 1)
            self.assertIn("no failing row had budget_current_bytes=0", proc.stderr)


def run_checker(tsv: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--tpch-tsv", str(tsv)],
        text=True,
        capture_output=True,
        check=False,
    )


def write_tpch_tsv(
    path: Path,
    *,
    max_source_reads_in_flight: str = "2",
    failed_query_ids: list[str] | None = None,
    largest_failed_allocation_bytes: str = "1230296",
) -> None:
    module = load_module()
    failed = failed_query_ids if failed_query_ids is not None else module.EXPECTED_FAILED_QUERY_IDS
    query_ids = [f"q{index:02d}" for index in range(1, 23)]
    header = [
        "query_id",
        "runtime_outcome",
        "bounded_status",
        "runtime_failure_code",
        "runtime_failure_kind",
        "runtime_failure_facts",
        "reservation_denied_count",
        "parity_status",
    ]
    rows = []
    for query_id in query_ids:
        if query_id in failed:
            rows.append(
                [
                    query_id,
                    "failed",
                    "bounded_accepted_native_failed",
                    "cudf_resource_exhausted",
                    "resource_exhausted",
                    failure_facts(largest_failed_allocation_bytes),
                    "0",
                    "native_failed",
                ]
            )
        else:
            rows.append(
                [
                    query_id,
                    "succeeded",
                    "bounded_accepted_parity_matched",
                    "none",
                    "none",
                    "none",
                    "0",
                    "matched",
                ]
            )

    lines = [
        "# native_tpch_benchmark_tsv_schema_version=18",
        f"# tpch_benchmark_native_max_source_reads_in_flight={max_source_reads_in_flight}",
        "\t".join(header),
    ]
    lines.extend("\t".join(row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def failure_facts(largest_failed_allocation_bytes: str) -> str:
    return ";".join(
        [
            "attempt=1",
            "budget_boundary=join_source_read_prefetch",
            "budget_bytes=1746095",
            "budget_current_bytes=0",
            "budget_enforcement=hard_memory_resource_limit",
            "budget_failed_allocation_count=1",
            f"budget_largest_failed_allocation_bytes={largest_failed_allocation_bytes}",
            "budget_peak_bytes=1183407",
            "cudf_code=OutOfMemory",
            "cudf_kind=ResourceExhausted",
            "cudf_operation=dataframe::read_parquet_source_chunk_with_budget",
            "native_operation=cudf_query_engine.local_parquet_chunk.read",
            "pipeline_task_kind=join_probe",
            "stage=join_source_read",
        ]
    )


if __name__ == "__main__":
    unittest.main()
