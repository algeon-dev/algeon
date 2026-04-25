from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[2] / "tools"


def load_module(name: str):
    script = TOOLS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TpcdsProfileMetadataTest(unittest.TestCase):
    def test_cpu_gpu_compare_reads_native_profile_comments(self) -> None:
        module = load_module("tpcds_cpu_gpu_compare")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gpu.tsv"
            path.write_text(
                "\n".join(
                    [
                        "# tpcds_benchmark_native_profile_name=sf100_gpu_throughput",
                        "# tpcds_benchmark_native_profile_source=env_profile",
                        "# tpcds_benchmark_native_device_budget_bytes=102005473280",
                        "# tpcds_benchmark_native_max_source_chunk_bytes=34359738368",
                        "# tpcds_benchmark_native_max_row_groups_per_chunk=1024",
                        "# tpcds_benchmark_native_source_chunk_planning_mode=local_parquet_throughput",
                        "engine\tquery_id\titeration\twarmup\twall_ms\toutcome\toutput_rows\tnative_runtime_started",
                        "native_gpu\tq44\t0\tfalse\t640.0\tsucceeded\t1\ttrue",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            rows, metadata = module.load(path)
            profile = module.native_profile_summary(metadata)

        self.assertEqual(rows["q44"]["measured"], [640.0])
        self.assertEqual(profile["name"], "sf100_gpu_throughput")
        self.assertEqual(profile["row_groups"], "1024")
        self.assertEqual(profile["source_chunk_planning_mode"], "local_parquet_throughput")
        self.assertTrue(profile["known"])

if __name__ == "__main__":
    unittest.main()
