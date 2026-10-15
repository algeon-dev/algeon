import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("merrec_benchmark", Path(__file__).parents[1] / "merrec_benchmark.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RuntimeSummaryTests(unittest.TestCase):
    @staticmethod
    def event(plan, span, time_ns, state, kind="vector_native"):
        return {"record_type": "timeline_event", "data": {
            "plan_execution_id": plan, "span_id": span, "monotonic_ns": time_ns,
            "payload": {"kind": "operator_stage", "stage_kind": kind,
                        "lifecycle": {"state": state}}}}

    def summary(self, events, complete=True):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.runtime.jsonl.gz"
            with gzip.open(path, "wt") as output:
                for record in events + [{"record_type": "footer", "data": {"complete": complete}}]:
                    output.write(json.dumps(record) + "\n")
            return MODULE.runtime_summary(path)

    def test_fragment_local_spans_and_overlapping_invocations(self):
        summary = self.summary([
            self.event("plan-a", 1, 100, "started"),
            self.event("plan-b", 1, 150, "started"),
            self.event("plan-a", 1, 200, "finished"),
            self.event("plan-b", 1, 250, "finished"),
        ])
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["stages"]["vector_native"]["invocations"], 2)
        self.assertAlmostEqual(summary["native_host_span_union_s"], 150 / 1e9)
        self.assertAlmostEqual(summary["stages"]["vector_native"]["host_span_sum_s"], 200 / 1e9)

    def test_incomplete_capture_and_unpaired_events_have_no_fraction(self):
        for events, complete in (([self.event("plan", 1, 100, "started")], True),
                                 ([self.event("plan", 1, 200, "finished")], True),
                                 ([self.event("plan", 1, 100, "started"),
                                   self.event("plan", 1, 200, "finished")], False)):
            with self.subTest(events=events, complete=complete):
                summary = self.summary(events, complete)
                self.assertFalse(summary["complete"])
                self.assertIsNone(summary["native_host_span_union_s"])

    def test_forest_import_is_reported_separately_from_native_compute(self):
        begin = self.event("plan", 1, 100, "started", "ml_native")
        finish = self.event("plan", 1, 200, "finished", "ml_native")
        for event in (begin, finish):
            event["data"]["payload"]["strategy_label"] = "forest_import:model:float32"
        summary = self.summary([begin, finish])
        self.assertEqual(summary["stages"]["ml_native:forest_import"]["invocations"], 1)
        self.assertIsNone(summary["native_host_span_union_s"])


class KnnValidationTests(unittest.TestCase):
    def test_equal_distance_ids_can_differ_but_duplicate_or_far_neighbors_fail(self):
        import numpy as np
        import pyarrow as pa

        ids = np.array([1, 2, 3, 4])
        matrix = np.array([[0.0], [1.0], [-1.0], [2.0]], dtype=np.float32)
        distances = np.array([[0.0, 1.0]], dtype=np.float32)
        for neighbors, values, matches in (([1, 3], [0.0, 1.0], True),
                                           ([1, 1], [0.0, 0.0], False),
                                           ([1, 4], [0.0, 4.0], False),
                                           ([1, 3], [0.0, 0.5], False)):
            with self.subTest(neighbors=neighbors, values=values):
                table = pa.table({"query_id": [1, 1], "neighbor_id": neighbors, "distance": values})
                result = MODULE.validate_knn(table, ids, matrix, 1, distances, 2, 1e-6)
                self.assertEqual(result["matches"], matches)

    def test_cpu_reported_noise_does_not_become_the_gpu_candidate_reference(self):
        import numpy as np
        import pyarrow as pa

        ids = np.array([1, 2])
        matrix = np.array([[0.0], [1.0]], dtype=np.float32)
        neighbors = np.array([[1, 2]])
        expected = MODULE.recompute_candidate_distances(ids, matrix, 1, neighbors)
        cpu = pa.table({"query_id": [1, 1], "neighbor_id": [1, 2], "distance": [0.0, 1.0000025]})
        gpu = pa.table({"query_id": [1, 1], "neighbor_id": [1, 2], "distance": [0.0, 1.0]})
        cpu_result = MODULE.validate_knn(cpu, ids, matrix, 1, expected, 2, 1e-6, neighbors)
        gpu_result = MODULE.validate_knn(gpu, ids, matrix, 1, expected, 2, 1e-6, neighbors)
        self.assertFalse(cpu_result["matches"])
        self.assertFalse(cpu_result["reported_distance_matches"])
        self.assertTrue(cpu_result["candidate_distance_matches"])
        self.assertTrue(gpu_result["matches"])

    def test_candidate_equivalence_and_reported_accuracy_have_independent_error_bounds(self):
        import numpy as np
        import pyarrow as pa

        ids = np.array([1, 2, 3, 4])
        matrix = np.array([[0.0], [1.0], [1.00000036], [2.0]], dtype=np.float32)
        neighbors = np.array([[1, 2]])
        expected = MODULE.recompute_candidate_distances(ids, matrix, 1, neighbors)
        near_distance = float(matrix[2, 0]) ** 2
        for neighbor, reported, candidate_match, reported_match in (
            (3, np.float32(near_distance + 8e-7), True, True),
            (3, np.float32(near_distance + 2e-6), True, False),
            (4, np.float32(4.0), False, True),
        ):
            with self.subTest(neighbor=neighbor, reported=reported):
                gpu = pa.table({"query_id": [1, 1], "neighbor_id": [1, neighbor],
                                "distance": np.array([0.0, reported], dtype=np.float32)})
                result = MODULE.validate_knn(gpu, ids, matrix, 1, expected, 2, 1e-6, neighbors)
                self.assertEqual(result["candidate_distance_matches"], candidate_match)
                self.assertEqual(result["reported_distance_matches"], reported_match)
                self.assertEqual(result["matches"], candidate_match and reported_match)


if __name__ == "__main__":
    unittest.main()
