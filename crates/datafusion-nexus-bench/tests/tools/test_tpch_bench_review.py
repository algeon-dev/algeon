from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "tpch_bench_review.py"


def load_module():
    spec = importlib.util.spec_from_file_location("tpch_bench_review", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TpchBenchReviewTest(unittest.TestCase):
    def test_reads_final_dispositions_from_frozen_tsv_column(self) -> None:
        module = load_module()
        self.assertEqual(module.FINAL_DISPOSITION_TSV_COLUMN, "post_native_execution_path")

        with tempfile.TemporaryDirectory() as tmp:
            tsv = Path(tmp) / "tpch.tsv"
            tsv.write_text(
                "\n".join(
                    [
                        "query_id\tpost_native_execution_path\tparity_status\tcpu_wall_ms\tnative_wall_ms\tspeedup_vs_cpu\truntime_operator_class\treservation_live_peak_bytes\tretry_count\treservation_denied_count\tsource_read_time_ms\toperator_time_ms\tfinal_merge_time_ms\tfinal_sort_time_ms\tmaterialization_time_ms",
                        "q01\tnative\tmatched\t20\t5\t4\tjoin\t1073741824\t0\t0\t1\t2\t0\t0\t1",
                        "q02\tdatafusion\tnot_run\t20\tnone\tnone\tnone\t0\t0\t0\tnone\tnone\tnone\tnone\tnone",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(tsv)],
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("disposition", result.stdout)
        self.assertIn("q01", result.stdout)
        self.assertIn("native", result.stdout)
        self.assertIn("q02", result.stdout)
        self.assertIn("datafusion", result.stdout)
        self.assertIn("disposition=datafusion", result.stdout)


if __name__ == "__main__":
    unittest.main()
