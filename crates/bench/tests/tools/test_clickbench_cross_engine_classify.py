from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "clickbench_cross_engine_classify.py"


def load_module():
    spec = importlib.util.spec_from_file_location("clickbench_cross_engine_classify", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_row(
    query_id: str,
    engine: str,
    iteration: int,
    fingerprint: str,
    row_count: str = "10",
    status: str = "succeeded",
) -> dict:
    return {
        "query_id": query_id,
        "engine": engine,
        "iteration": str(iteration),
        "status": status,
        "row_count": row_count,
        "fingerprint": fingerprint,
        "elapsed_ms": "100",
        "raw_output_path": "/tmp/test.tsv",
        "native_rewritten": "n/a",
        "error": "none",
    }


class ClassifyQueryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.m = load_module()

    # --- all_engines_same ---

    def test_all_engines_same_when_all_fingerprints_match(self) -> None:
        rows = [
            make_row("q18", "duckdb", 1, "fp-abc"),
            make_row("q18", "cpu_datafusion", 1, "fp-abc"),
            make_row("q18", "algeon_datafusion", 1, "fp-abc"),
        ]
        result = self.m.classify_query("q18", rows)
        self.assertEqual(result.classification, "all_engines_same")
        self.assertEqual(result.engine_instability, "false")
        self.assertEqual(result.fingerprint_agreement, "true")
        self.assertEqual(result.mismatched_engines, "none")

    def test_all_engines_same_with_multiple_iterations(self) -> None:
        rows = [
            make_row("q25", "duckdb", 1, "fp-x"),
            make_row("q25", "duckdb", 2, "fp-x"),
            make_row("q25", "cpu_datafusion", 1, "fp-x"),
            make_row("q25", "algeon_datafusion", 1, "fp-x"),
        ]
        result = self.m.classify_query("q25", rows)
        self.assertEqual(result.classification, "all_engines_same")
        self.assertEqual(result.engine_instability, "false")

    # --- tie_divergence ---

    def test_tie_divergence_for_waiver_query_different_fingerprints_same_count(self) -> None:
        rows = [
            make_row("q18", "duckdb", 1, "fp-duckdb"),
            make_row("q18", "cpu_datafusion", 1, "fp-cpu"),
            make_row("q18", "algeon_datafusion", 1, "fp-native"),
        ]
        result = self.m.classify_query("q18", rows)
        self.assertEqual(result.classification, "tie_divergence")
        self.assertEqual(result.row_count_agreement, "true")
        self.assertEqual(result.engine_instability, "false")

    def test_tie_divergence_for_all_ten_waiver_queries(self) -> None:
        m = self.m
        tie_queries = ["q18", "q25", "q26", "q31", "q32", "q33", "q39", "q40", "q41", "q42"]
        for qid in tie_queries:
            with self.subTest(query_id=qid):
                rows = [
                    make_row(qid, "cpu_datafusion", 1, "fp-cpu"),
                    make_row(qid, "algeon_datafusion", 1, "fp-native"),
                ]
                result = m.classify_query(qid, rows)
                self.assertEqual(result.classification, "tie_divergence", f"{qid} should be tie_divergence")

    def test_unexpected_mismatch_when_tie_query_row_counts_differ(self) -> None:
        rows = [
            make_row("q39", "cpu_datafusion", 1, "fp-cpu", row_count="10"),
            make_row("q39", "algeon_datafusion", 1, "fp-native", row_count="9"),
        ]
        result = self.m.classify_query("q39", rows)
        self.assertEqual(result.classification, "unexpected_mismatch")
        self.assertEqual(result.row_count_agreement, "false")

    # --- float_variance ---

    def test_float_variance_q04_same_row_count_different_fingerprints(self) -> None:
        rows = [
            make_row("q04", "duckdb", 1, "fp-duckdb", row_count="1"),
            make_row("q04", "cpu_datafusion", 1, "fp-cpu", row_count="1"),
            make_row("q04", "algeon_datafusion", 1, "fp-native", row_count="1"),
        ]
        result = self.m.classify_query("q04", rows)
        self.assertEqual(result.classification, "float_variance")
        self.assertEqual(result.row_count_agreement, "true")

    def test_q10_is_no_longer_a_float_variance_waiver(self) -> None:
        # q10 was removed from the float-variance waiver set: a mismatch must
        # surface as unexpected_mismatch instead of being masked, mirroring the
        # Rust PARITY_WAIVERS contract (q04 only).
        rows = [
            make_row("q10", "cpu_datafusion", 1, "fp-cpu", row_count="10"),
            make_row("q10", "algeon_datafusion", 1, "fp-native", row_count="10"),
        ]
        result = self.m.classify_query("q10", rows)
        self.assertEqual(result.classification, "unexpected_mismatch")

    def test_unexpected_mismatch_when_float_query_row_counts_differ(self) -> None:
        rows = [
            make_row("q04", "cpu_datafusion", 1, "fp-cpu", row_count="1"),
            make_row("q04", "algeon_datafusion", 1, "fp-native", row_count="2"),
        ]
        result = self.m.classify_query("q04", rows)
        self.assertEqual(result.classification, "unexpected_mismatch")

    # --- engine_instability ---

    def test_engine_instability_when_engine_returns_different_fingerprints(self) -> None:
        rows = [
            make_row("q39", "duckdb", 1, "fp-a"),
            make_row("q39", "duckdb", 2, "fp-b"),  # different — unstable
            make_row("q39", "cpu_datafusion", 1, "fp-stable"),
        ]
        result = self.m.classify_query("q39", rows)
        self.assertEqual(result.classification, "engine_instability")
        self.assertEqual(result.engine_instability, "true")
        self.assertIn("duckdb", result.unstable_engines)

    def test_engine_instability_takes_priority_over_tie_divergence(self) -> None:
        rows = [
            make_row("q31", "duckdb", 1, "fp-a"),
            make_row("q31", "duckdb", 2, "fp-b"),  # unstable
            make_row("q31", "cpu_datafusion", 1, "fp-cpu"),
            make_row("q31", "algeon_datafusion", 1, "fp-native"),
        ]
        result = self.m.classify_query("q31", rows)
        self.assertEqual(result.classification, "engine_instability")

    def test_stable_engine_with_two_identical_iterations_is_not_unstable(self) -> None:
        rows = [
            make_row("q26", "duckdb", 1, "fp-same"),
            make_row("q26", "duckdb", 2, "fp-same"),  # same — stable
            make_row("q26", "cpu_datafusion", 1, "fp-other"),
        ]
        result = self.m.classify_query("q26", rows)
        self.assertNotEqual(result.classification, "engine_instability")
        self.assertEqual(result.engine_instability, "false")

    # --- correctness_bug (q23) ---

    def test_correctness_bug_when_reference_engines_agree_and_native_differs(self) -> None:
        rows = [
            make_row("q23", "duckdb", 1, "fp-ref"),
            make_row("q23", "cpu_datafusion", 1, "fp-ref"),
            make_row("q23", "algeon_datafusion", 1, "fp-native-wrong"),
        ]
        result = self.m.classify_query("q23", rows)
        self.assertEqual(result.classification, "correctness_bug")
        self.assertEqual(result.reference_fingerprint, "fp-ref")
        self.assertEqual(result.algeon_datafusion_fingerprint, "fp-native-wrong")
        self.assertEqual(result.engine_instability, "false")

    def test_correctness_bug_when_only_cpu_datafusion_is_reference(self) -> None:
        rows = [
            make_row("q23", "cpu_datafusion", 1, "fp-cpu"),
            make_row("q23", "algeon_datafusion", 1, "fp-native-wrong"),
        ]
        result = self.m.classify_query("q23", rows)
        self.assertEqual(result.classification, "correctness_bug")
        self.assertEqual(result.reference_fingerprint, "fp-cpu")

    def test_all_engines_same_q23_when_all_fingerprints_match(self) -> None:
        rows = [
            make_row("q23", "duckdb", 1, "fp-match"),
            make_row("q23", "cpu_datafusion", 1, "fp-match"),
            make_row("q23", "algeon_datafusion", 1, "fp-match"),
        ]
        result = self.m.classify_query("q23", rows)
        self.assertEqual(result.classification, "all_engines_same")

    def test_unexpected_mismatch_q23_when_reference_engines_disagree(self) -> None:
        rows = [
            make_row("q23", "duckdb", 1, "fp-duckdb"),
            make_row("q23", "cpu_datafusion", 1, "fp-cpu"),
            make_row("q23", "algeon_datafusion", 1, "fp-native"),
        ]
        result = self.m.classify_query("q23", rows)
        self.assertEqual(result.classification, "unexpected_mismatch")
        self.assertEqual(result.reference_fingerprint, "disagreement")

    def test_insufficient_data_q23_when_native_failed_and_single_reference(self) -> None:
        # Only one engine succeeded — can't compare, so insufficient_data regardless of q23 identity.
        rows = [
            make_row("q23", "cpu_datafusion", 1, "fp-cpu"),
            make_row("q23", "algeon_datafusion", 1, "fp-ignored", status="failed"),
        ]
        result = self.m.classify_query("q23", rows)
        self.assertEqual(result.classification, "insufficient_data")
        self.assertEqual(result.algeon_datafusion_fingerprint, "none")

    # --- insufficient_data ---

    def test_insufficient_data_when_only_one_engine_succeeded(self) -> None:
        rows = [make_row("q18", "duckdb", 1, "fp-only")]
        result = self.m.classify_query("q18", rows)
        self.assertEqual(result.classification, "insufficient_data")
        self.assertEqual(result.fingerprint_agreement, "n/a")
        self.assertEqual(result.row_count_agreement, "n/a")

    def test_insufficient_data_when_all_engines_failed(self) -> None:
        rows = [
            make_row("q18", "duckdb", 1, "fp-ignored", status="failed"),
            make_row("q18", "cpu_datafusion", 1, "fp-ignored", status="failed"),
        ]
        result = self.m.classify_query("q18", rows)
        self.assertEqual(result.classification, "insufficient_data")

    # --- failed iteration exclusion ---

    def test_failed_iteration_not_counted_when_other_succeeds(self) -> None:
        rows = [
            make_row("q18", "duckdb", 1, "fp-good"),
            make_row("q18", "cpu_datafusion", 1, "fp-good"),
        ]
        result = self.m.classify_query("q18", rows)
        self.assertEqual(result.classification, "all_engines_same")

    def test_unstable_engine_detection_ignores_failed_iterations(self) -> None:
        # iteration 1 failed, iteration 2 succeeded — only one successful fp, so stable
        rows = [
            make_row("q40", "duckdb", 1, "fp-ignored", status="failed"),
            make_row("q40", "duckdb", 2, "fp-good"),
            make_row("q40", "cpu_datafusion", 1, "fp-other"),
        ]
        result = self.m.classify_query("q40", rows)
        self.assertNotEqual(result.classification, "engine_instability")
        self.assertEqual(result.engine_instability, "false")

    # --- mismatched_engines tracking ---

    def test_mismatched_engines_identifies_outlier(self) -> None:
        rows = [
            make_row("q32", "duckdb", 1, "fp-majority"),
            make_row("q32", "cpu_datafusion", 1, "fp-outlier"),
            make_row("q32", "algeon_datafusion", 1, "fp-majority"),
        ]
        result = self.m.classify_query("q32", rows)
        self.assertEqual(result.classification, "tie_divergence")
        self.assertIn("cpu_datafusion", result.mismatched_engines)
        self.assertNotIn("duckdb", result.mismatched_engines)

    def test_mismatched_engines_empty_when_all_agree(self) -> None:
        rows = [
            make_row("q33", "cpu_datafusion", 1, "fp-same"),
            make_row("q33", "algeon_datafusion", 1, "fp-same"),
        ]
        result = self.m.classify_query("q33", rows)
        self.assertEqual(result.mismatched_engines, "none")

    # --- reference_fingerprint field ---

    def test_reference_fingerprint_is_consensus_when_refs_agree(self) -> None:
        rows = [
            make_row("q23", "duckdb", 1, "fp-ref-consensus"),
            make_row("q23", "cpu_datafusion", 1, "fp-ref-consensus"),
            make_row("q23", "algeon_datafusion", 1, "fp-native"),
        ]
        result = self.m.classify_query("q23", rows)
        self.assertEqual(result.reference_fingerprint, "fp-ref-consensus")

    def test_reference_fingerprint_is_disagreement_when_refs_differ(self) -> None:
        rows = [
            make_row("q23", "duckdb", 1, "fp-a"),
            make_row("q23", "cpu_datafusion", 1, "fp-b"),
            make_row("q23", "algeon_datafusion", 1, "fp-c"),
        ]
        result = self.m.classify_query("q23", rows)
        self.assertEqual(result.reference_fingerprint, "disagreement")

    def test_reference_fingerprint_is_none_when_no_reference_engines_present(self) -> None:
        rows = [
            make_row("q18", "algeon_datafusion", 1, "fp-native"),
        ]
        result = self.m.classify_query("q18", rows)
        self.assertEqual(result.reference_fingerprint, "none")

    # --- TSV/JSONL round-trip ---

    def test_write_and_read_tsv(self) -> None:
        m = self.m
        rows = [
            make_row("q04", "cpu_datafusion", 1, "fp-cpu", row_count="1"),
            make_row("q04", "algeon_datafusion", 1, "fp-native", row_count="1"),
        ]
        classification = m.classify_query("q04", rows)

        with tempfile.TemporaryDirectory() as tmp:
            tsv_path = Path(tmp) / "classification.tsv"
            m.write_tsv(tsv_path, [classification])

            with tsv_path.open(newline="", encoding="utf-8") as fh:
                import csv as csv_mod
                reader = csv_mod.DictReader(fh, delimiter="\t")
                loaded = list(reader)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["query_id"], "q04")
        self.assertEqual(loaded[0]["classification"], "float_variance")

    def test_write_and_read_jsonl(self) -> None:
        m = self.m
        rows_input = [
            make_row("q23", "duckdb", 1, "fp-ref"),
            make_row("q23", "cpu_datafusion", 1, "fp-ref"),
            make_row("q23", "algeon_datafusion", 1, "fp-wrong"),
        ]
        classification = m.classify_query("q23", rows_input)

        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = Path(tmp) / "classification.jsonl"
            m.write_jsonl(jsonl_path, [classification])

            with jsonl_path.open(encoding="utf-8") as fh:
                loaded = [json.loads(line) for line in fh if line.strip()]

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["query_id"], "q23")
        self.assertEqual(loaded[0]["classification"], "correctness_bug")

    def test_output_header_covers_all_dataclass_fields(self) -> None:
        m = load_module()
        from dataclasses import fields
        field_names = [f.name for f in fields(m.ClassificationRow)]
        self.assertEqual(field_names, m.OUTPUT_HEADER)

    # --- load_input ---

    def test_load_tsv_input(self) -> None:
        m = self.m
        import argparse
        with tempfile.TemporaryDirectory() as tmp:
            tsv_path = Path(tmp) / "rows.tsv"
            tsv_path.write_text(
                "query_id\tengine\titeration\tstatus\trow_count\tfingerprint\telapsed_ms\traw_output_path\tnative_rewritten\terror\n"
                "q04\tcpu_datafusion\t1\tsucceeded\t1\tfp-cpu\t100\t/tmp/f.tsv\tn/a\tnone\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(input=tsv_path, out_dir=None, tsv=None, jsonl=None)
            rows = m.load_input(args)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["query_id"], "q04")

    def test_load_jsonl_input(self) -> None:
        m = self.m
        import argparse
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = Path(tmp) / "rows.jsonl"
            jsonl_path.write_text(
                json.dumps(make_row("q10", "duckdb", 1, "fp-x")) + "\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(input=jsonl_path, out_dir=None, tsv=None, jsonl=None)
            rows = m.load_input(args)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["query_id"], "q10")


if __name__ == "__main__":
    unittest.main()
