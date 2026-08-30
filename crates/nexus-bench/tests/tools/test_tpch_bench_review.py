from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "tpch_bench_review.py"

BASE_COLUMNS = [
    "query_id", "planning_outcome", "reason_code", "unsupported_category",
    "post_native_execution_path", "rejected_before_execution",
    "cpu_runtime_started", "native_runtime_started", "runtime_outcome",
    "cpu_failure_code", "runtime_failure_code", "runtime_failure_kind",
    "runtime_failure_operation", "runtime_failure_facts", "parity_status",
    "cpu_wall_ms", "native_wall_ms", "speedup_vs_cpu",
    "resident_dataframe_peak_bytes", "allocator_high_water_bytes",
    "max_task_observed_peak_bytes", "max_task_retained_bytes",
    "runtime_operator_class", "retry_count", "retry_resource_exhausted_count",
    "source_read_time_ms", "operator_time_ms", "final_merge_time_ms",
    "final_sort_time_ms", "materialization_time_ms",
]

DEFAULTS = {
    "reason_code": "none", "unsupported_category": "none",
    "rejected_before_execution": "false", "cpu_runtime_started": "true",
    "native_runtime_started": "true", "cpu_failure_code": "none",
    "runtime_failure_code": "none", "runtime_failure_kind": "none",
    "runtime_failure_operation": "none", "runtime_failure_facts": "none",
    "resident_dataframe_peak_bytes": "none", "allocator_high_water_bytes": "none",
    "max_task_observed_peak_bytes": "none", "max_task_retained_bytes": "none",
    "runtime_operator_class": "source", "retry_count": "0",
    "retry_resource_exhausted_count": "0", "source_read_time_ms": "none",
    "operator_time_ms": "none", "final_merge_time_ms": "none",
    "final_sort_time_ms": "none", "materialization_time_ms": "none",
    "planning_outcome": "selected", "post_native_execution_path": "native",
    "runtime_outcome": "succeeded", "parity_status": "matched",
    "cpu_wall_ms": "not_run", "native_wall_ms": "not_run", "speedup_vs_cpu": "not_run",
}


def load_module():
    spec = importlib.util.spec_from_file_location("tpch_bench_review", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_row(**overrides):
    row = dict(DEFAULTS)
    row.update(overrides)
    return row


def write_tsv(path, rows):
    lines = ["\t".join(BASE_COLUMNS)]
    for row in rows:
        lines.append("\t".join(row.get(c, "") for c in BASE_COLUMNS))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_tool(tsv, *extra_args):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(tsv), *extra_args],
        text=True,
        capture_output=True,
        check=False,
    )
    return result


class TpchBenchReviewTest(unittest.TestCase):
    def test_reads_final_dispositions_from_frozen_tsv_column(self) -> None:
        module = load_module()
        self.assertEqual(module.FINAL_DISPOSITION_TSV_COLUMN, "post_native_execution_path")

        with tempfile.TemporaryDirectory() as tmp:
            tsv = Path(tmp) / "tpch.tsv"
            write_tsv(tsv, [
                make_row(query_id="q01", post_native_execution_path="native",
                         parity_status="matched", cpu_wall_ms="20", native_wall_ms="5",
                         speedup_vs_cpu="4"),
                make_row(query_id="q02", post_native_execution_path="datafusion",
                         parity_status="not_run", cpu_wall_ms="20", native_wall_ms="not_run",
                         speedup_vs_cpu="not_run", runtime_operator_class="not_run",
                         native_runtime_started="false", runtime_outcome="not_started"),
            ])

            result = run_tool(tsv, "--pretty", "never", "--width", "250")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("disposition", result.stdout)
        self.assertIn("q01", result.stdout)
        self.assertIn("native", result.stdout)
        self.assertIn("q02", result.stdout)
        self.assertIn("datafusion", result.stdout)
        self.assertIn("disposition=datafusion", result.stdout)

    def test_unknown_memory_is_reported_unknown_not_zero(self) -> None:
        module = load_module()
        rows = [
            make_row(query_id="q01", allocator_high_water_bytes="none",
                     max_task_observed_peak_bytes="6193336787",
                     parity_status="matched", cpu_wall_ms="20", native_wall_ms="5",
                     speedup_vs_cpu="4"),
        ]
        meta = {}
        records, summary = module.summarize(meta, rows, 2.0)
        self.assertIsNone(records[0]["alloc_hw"])
        self.assertEqual(records[0]["task_peak"], 6193336787.0)

        with tempfile.TemporaryDirectory() as tmp:
            tsv = Path(tmp) / "tpch.tsv"
            write_tsv(tsv, rows)
            result = run_tool(tsv, "--pretty", "never", "--width", "250")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("alloc_hw=n/a", result.stdout)
        self.assertNotIn("peak_rsv", result.stdout)
        self.assertNotIn("0.00GiB", result.stdout)

    def test_totals_use_only_paired_matched_queries(self) -> None:
        module = load_module()
        rows = [
            make_row(query_id="q01", parity_status="matched",
                     cpu_wall_ms="20", native_wall_ms="5", speedup_vs_cpu="4"),
            make_row(query_id="q02", parity_status="native_failed",
                     runtime_outcome="failed", cpu_wall_ms="100", native_wall_ms="not_run",
                     speedup_vs_cpu="not_run", runtime_failure_code="query_memory_limit_exceeded",
                     runtime_operator_class="not_run"),
            make_row(query_id="q03", parity_status="mismatched",
                     cpu_wall_ms="10", native_wall_ms="10", speedup_vs_cpu="1"),
        ]
        records, summary = module.summarize({}, rows, 2.0)
        self.assertEqual(summary["cpu"], 20)
        self.assertEqual(summary["gpu"], 5)
        self.assertEqual(summary["overall"], 4.0)
        self.assertEqual(summary["paired"], 1)
        self.assertEqual(summary["total"], 3)

        with tempfile.TemporaryDirectory() as tmp:
            tsv = Path(tmp) / "tpch.tsv"
            write_tsv(tsv, rows)
            result = run_tool(tsv, "--pretty", "never", "--width", "250")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("paired matched 1/3", result.stdout)
        self.assertIn("full-suite speedup: n/a", result.stdout)

    def test_failure_classes_are_counted_separately(self) -> None:
        module = load_module()
        rows = [
            make_row(query_id="q15", planning_outcome="none",
                     rejected_before_execution="true", reason_code="physical_planning_failed",
                     unsupported_category="planning", cpu_runtime_started="false",
                     native_runtime_started="false", runtime_outcome="not_started",
                     runtime_failure_code="final_plan_requirement_unsatisfied",
                     runtime_failure_facts="capability_reason=native_ordered_output_unsupported",
                     post_native_execution_path="none", parity_status="not_run",
                     runtime_operator_class="not_run"),
            make_row(query_id="q04", runtime_outcome="failed", parity_status="native_failed",
                     runtime_failure_code="cudf_dependency_failure",
                     runtime_failure_facts=(
                         "cudf_code=IoFailure;cudf_message=Unable to open file. Linux "
                         "system/library function call error at: /some/very/long/path/"
                         "file_utils.cpp:160: Too many open files"
                     ),
                     cpu_wall_ms="500", native_wall_ms="not_run", speedup_vs_cpu="not_run",
                     runtime_operator_class="not_run"),
            make_row(query_id="q05", runtime_outcome="failed", parity_status="native_failed",
                     runtime_failure_code="query_memory_limit_exceeded",
                     runtime_failure_facts="failure_source=RESERVATION_LIMIT;budget_boundary=source_adjacent_aggregate_hash_partition",
                     cpu_wall_ms="700", native_wall_ms="not_run", speedup_vs_cpu="not_run",
                     runtime_operator_class="not_run"),
            make_row(query_id="q06", runtime_outcome="failed", parity_status="native_failed",
                     runtime_failure_code="device_out_of_memory",
                     cpu_wall_ms="300", native_wall_ms="not_run", speedup_vs_cpu="not_run",
                     runtime_operator_class="not_run"),
            make_row(query_id="q07", parity_status="cpu_failed", cpu_failure_code="datafusion_panic",
                     cpu_wall_ms="not_run", native_wall_ms="not_run", speedup_vs_cpu="not_run",
                     runtime_operator_class="not_run"),
            make_row(query_id="q08", parity_status="matched", cpu_wall_ms="100",
                     native_wall_ms="200", speedup_vs_cpu="0.5"),
        ]
        records, summary = module.summarize({}, rows, 2.0)
        by_q = {rec["q"]: rec for rec in records}

        self.assertEqual(by_q["q15"]["cls"], "planning_rejected")
        self.assertEqual(by_q["q04"]["cls"], "runtime_failed:cudf_dependency_failure")
        self.assertEqual(by_q["q05"]["cls"], "runtime_failed:query_memory_limit_exceeded")
        self.assertEqual(by_q["q06"]["cls"], "runtime_failed:device_out_of_memory")
        self.assertEqual(by_q["q07"]["cls"], "cpu_failed")
        self.assertEqual(by_q["q08"]["cls"], "matched")

        for q in ["q15", "q04", "q05", "q06", "q07"]:
            self.assertTrue(by_q[q]["flags"].startswith("FAIL:"), by_q[q]["flags"])
            self.assertNotIn("SLOW", by_q[q]["flags"])
            self.assertNotIn("PAR", by_q[q]["flags"])
        self.assertIn("SLOW", by_q["q08"]["flags"])

        self.assertIn("Too many open files", by_q["q04"]["detail"])
        self.assertIn("RESERVATION_LIMIT", by_q["q05"]["detail"])

        coverage_text = "\n".join(summary["coverage"])
        self.assertIn("planning_rejected: 1", coverage_text)
        self.assertIn("runtime_failed:cudf_dependency_failure: 1", coverage_text)
        self.assertIn("runtime_failed:query_memory_limit_exceeded: 1", coverage_text)
        self.assertIn("runtime_failed:device_out_of_memory: 1", coverage_text)
        self.assertIn("cpu_failed: 1", coverage_text)
        self.assertIn("parity_matched: 1", coverage_text)

    def test_failed_rows_show_runtime_started_not_operator_class(self) -> None:
        module = load_module()
        rows = [
            make_row(query_id="q01", runtime_outcome="failed", parity_status="native_failed",
                     native_runtime_started="true", runtime_failure_code="query_memory_limit_exceeded",
                     runtime_operator_class="not_run",
                     cpu_wall_ms="500", native_wall_ms="not_run", speedup_vs_cpu="not_run"),
        ]
        records, summary = module.summarize({}, rows, 2.0)
        self.assertEqual(records[0]["rt"], "started")


if __name__ == "__main__":
    unittest.main()
