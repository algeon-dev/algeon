from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "clickbench_cross_engine_outputs.py"


def load_module():
    spec = importlib.util.spec_from_file_location("clickbench_cross_engine_outputs", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ClickbenchCrossEngineOutputsTest(unittest.TestCase):
    def test_default_query_ids_are_waivers_plus_q23(self) -> None:
        module = load_module()

        self.assertEqual(
            module.DEFAULT_QUERY_IDS,
            (
                "q04",
                "q10",
                "q18",
                "q25",
                "q26",
                "q31",
                "q32",
                "q33",
                "q39",
                "q40",
                "q41",
                "q42",
                "q23",
            ),
        )

    def test_loads_vendored_clickbench_queries(self) -> None:
        module = load_module()

        queries = module.load_clickbench_queries(REPO_ROOT)

        self.assertEqual(len(queries), 43)
        self.assertIn('COUNT(DISTINCT "UserID")', queries["q23"].sql)
        self.assertNotIn(";", queries["q04"].sql)

    def test_duckdb_setup_casts_event_date_from_epoch_days(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "partitioned"
            fixture.mkdir()
            (fixture / "hits_0.parquet").write_text("", encoding="utf-8")

            sql = module.duckdb_setup_sql(fixture)

        self.assertIn(f"read_parquet('{fixture}/hits_*.parquet', binary_as_string=true)", sql)
        self.assertIn("DATE '1970-01-01' + CAST(EventDate AS INTEGER) AS EventDate", sql)
        self.assertIn("CREATE OR REPLACE VIEW hits", sql)

    def test_datafusion_progress_parser_maps_native_rewrite_state(self) -> None:
        module = load_module()
        stderr = "\n".join(
            [
                "[bench cpu] iter 1/2 q=q04 (1/13) elapsed_ms=11 outcome=succeeded rewritten=None",
                "[bench native] iter 2/2 q=q23 (13/13) elapsed_ms=22 outcome=succeeded rewritten=Some(true)",
            ]
        )

        events = module.parse_datafusion_progress(stderr)

        self.assertEqual(events[("cpu", "q04", 1)]["elapsed_ms"], "11")
        self.assertEqual(events[("cpu", "q04", 1)]["native_rewritten"], "n/a")
        self.assertEqual(events[("native", "q23", 2)]["native_rewritten"], "true")

    def test_raw_tsv_fingerprint_is_order_insensitive(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.tsv"
            second = Path(tmp) / "second.tsv"
            first.write_text("a\tb\nx\t1\nNULL\t2\n", encoding="utf-8")
            second.write_text("a\tb\nNULL\t2\nx\t1\n", encoding="utf-8")

            self.assertEqual(module.fingerprint_raw_tsv(first), module.fingerprint_raw_tsv(second))


if __name__ == "__main__":
    unittest.main()
