import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


DEV = Path(__file__).parents[1]
sys.path.insert(0, str(DEV))
SPEC = importlib.util.spec_from_file_location("relational_benchmarks_driver", DEV / "relational_benchmarks_driver.py")
DRIVER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(DRIVER)

SORTED_LAYOUT = "single_file_per_table_sorted_row_groups"
UNSORTED_LAYOUT = "single_file_per_table_unsorted"


def tpch_tsv(query, *, parity="matched", samples=5, fingerprint="output-fnv1a64:1", rows=10):
    preamble = {
        "tpch_benchmark_warmups": 1,
        "tpch_benchmark_iterations": 5,
        "tpch_benchmark_wall_time_fields": "per_measured_iteration_median",
        "tpch_benchmark_wall_time_boundary": "sql_planning_and_collect",
    }
    header = ["query_id", "planning_outcome", "runtime_outcome", "parity_status", "cpu_wall_ms_samples",
              "native_wall_ms_samples", "output_fingerprint", "cpu_output_rows", "native_output_rows"]
    series = ",".join(["1.000"] * samples)
    row = [query, "selected", "succeeded", parity, series, series, fingerprint, str(rows), str(rows)]
    lines = [f"# {key}={value}" for key, value in preamble.items()] + ["\t".join(header), "\t".join(row)]
    return "\n".join(lines) + "\n"


class CheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        DRIVER.OUT = Path(self.tmp.name)

    def write_dataset(self, dataset_id, *, layout, scale, **overrides):
        directory = DRIVER.stage_dir("tpch", dataset_id)
        directory.mkdir(parents=True)
        for query in DRIVER.TPCH_QUERIES:
            (directory / f"{query}.tsv").write_text(tpch_tsv(query, **overrides.get(query, {})))
        (directory / "layout.json").write_text(json.dumps({"generation": {"layout": layout, "scale_factor": scale}}))
        return directory

    def test_a_clean_dataset_passes(self):
        self.write_dataset("sf100_sorted", layout=SORTED_LAYOUT, scale=100)
        reasons, rows = DRIVER.check_tpch_dataset("sf100_sorted")
        self.assertEqual(reasons, [])
        self.assertEqual(len(rows), 22)

    def test_a_mismatched_row_blocks_publication(self):
        self.write_dataset("sf100_sorted", layout=SORTED_LAYOUT, scale=100, q09={"parity": "mismatched"})
        reasons, _ = DRIVER.check_tpch_dataset("sf100_sorted")
        self.assertEqual(reasons, ["q09: parity_status=mismatched, expected matched"])

    def test_a_sample_count_other_than_five_blocks_publication(self):
        self.write_dataset("sf100_sorted", layout=SORTED_LAYOUT, scale=100, q03={"samples": 4})
        reasons, _ = DRIVER.check_tpch_dataset("sf100_sorted")
        self.assertEqual(
            reasons,
            ["q03: cpu_wall_ms_samples holds 4 samples, expected 5", "q03: native_wall_ms_samples holds 4 samples, expected 5"],
        )

    def test_a_bench_binary_without_the_planning_boundary_blocks_publication(self):
        directory = self.write_dataset("sf100_sorted", layout=SORTED_LAYOUT, scale=100)
        text = (directory / "q01.tsv").read_text().replace("sql_planning_and_collect", "collect")
        (directory / "q01.tsv").write_text(text)
        reasons, _ = DRIVER.check_tpch_dataset("sf100_sorted")
        self.assertEqual(len(reasons), 1)
        self.assertIn("predates the planning boundary", reasons[0])

    def test_a_manifest_that_is_not_the_expected_layout_blocks_publication(self):
        self.write_dataset("sf100_dbgen_order", layout=SORTED_LAYOUT, scale=100)
        reasons, _ = DRIVER.check_tpch_dataset("sf100_dbgen_order")
        self.assertEqual(len(reasons), 1)
        self.assertIn(f"expected {UNSORTED_LAYOUT}", reasons[0])

    def test_different_fingerprints_between_sorted_and_dbgen_order_block_publication(self):
        self.write_dataset("sf100_sorted", layout=SORTED_LAYOUT, scale=100)
        self.write_dataset("sf100_dbgen_order", layout=UNSORTED_LAYOUT, scale=100,
                           q18={"fingerprint": "output-fnv1a64:2", "rows": 11})
        _, sorted_rows = DRIVER.check_tpch_dataset("sf100_sorted")
        _, dbgen_rows = DRIVER.check_tpch_dataset("sf100_dbgen_order")
        reasons = DRIVER.check_dbgen_order_matches_sorted(sorted_rows, dbgen_rows)
        self.assertEqual(sorted(reason.split(":")[1].split()[0] for reason in reasons),
                         ["cpu_output_rows", "native_output_rows", "output_fingerprint"])
        self.assertTrue(all(reason.startswith("q18:") for reason in reasons))
        self.assertEqual(DRIVER.check_dbgen_order_matches_sorted(sorted_rows, sorted_rows), [])


class HostFactsTests(unittest.TestCase):
    def test_host_facts_may_not_carry_keys_the_driver_owns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "facts.json"
            facts = {key: 1 for key in DRIVER.REQUIRED_HOST_FACTS}
            path.write_text(json.dumps(facts))
            self.assertEqual(DRIVER.load_host_facts(path), facts)
            for key in ("sourceRevision", "cpuTuning"):
                path.write_text(json.dumps({**facts, key: 1}))
                with self.assertRaisesRegex(SystemExit, key):
                    DRIVER.load_host_facts(path)


if __name__ == "__main__":
    unittest.main()
