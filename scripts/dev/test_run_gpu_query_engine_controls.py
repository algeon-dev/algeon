#!/usr/bin/env python3
"""Focused regression tests for the GPU goodput control validator."""

from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).with_name("run_gpu_query_engine_controls.py")
SPEC = importlib.util.spec_from_file_location("run_gpu_query_engine_controls", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {SCRIPT}")
CONTROLS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTROLS)


class ControlValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.output_root = Path(self.tempdir.name)
        self.old_output_root = CONTROLS.OUTPUT_ROOT
        self.old_binary = CONTROLS.GOODPUT_BINARY
        CONTROLS.OUTPUT_ROOT = self.output_root
        CONTROLS.GOODPUT_BINARY = self.output_root / "query_service_goodput"
        CONTROLS.GOODPUT_BINARY.write_bytes(b"benchmark binary")
        self.native_library = self.output_root / "libcudf_rust.so"
        self.native_library.write_bytes(b"first library")
        self.core_library = self.output_root / "libcudf.so"
        self.core_library.write_bytes(b"core library")
        self.run_id = "control-identity"
        self.item_id = "adhoc.independent_union"
        self.identity = {
            "identity_version": 1,
            "identity_sha256": "test-identity",
            "binary": {"sha256": CONTROLS.sha256_file(CONTROLS.GOODPUT_BINARY)},
        }
        self.run_dir = self.output_root / self.run_id
        self.run_dir.mkdir()
        (self.run_dir / CONTROLS.IDENTITY_FILE).write_text(json.dumps(self.identity), encoding="utf-8")
        self._write_library_snapshot()
        self._write_result()

    def tearDown(self) -> None:
        CONTROLS.OUTPUT_ROOT = self.old_output_root
        CONTROLS.GOODPUT_BINARY = self.old_binary
        self.tempdir.cleanup()

    def _file_snapshot(self, path: Path) -> dict[str, object]:
        stat = path.stat()
        return {
            "path": str(path),
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "size_bytes": stat.st_size,
            "sha256": CONTROLS.sha256_file(path),
        }

    def _write_library_snapshot(self) -> None:
        libraries = []
        for path in (self.native_library, self.core_library):
            libraries.append({**self._file_snapshot(path), "mapped_path": str(path), "mapped_device": "00:00"})
        snapshot = {
            "schema_version": 2,
            "source": "/proc/self/maps",
            "binary": self._file_snapshot(CONTROLS.GOODPUT_BINARY),
            "libraries": libraries,
        }
        (self.run_dir / CONTROLS.LOADED_LIBRARIES_FILE).write_text(json.dumps(snapshot), encoding="utf-8")

    def _write_result(self, bad_window: int | None = None, missing_window: int | None = None) -> None:
        summary_fields = [
            "run_id",
            "group",
            "mix",
            "arrival",
            "warmup_seconds",
            "window_seconds",
            "max_active_attempts",
            "device_capacity_bytes",
            "query_cap_bytes",
            "source_identity",
            "memory_failure_ratio",
            "offered",
            "completed",
            "succeeded",
            "slo_met",
            "attempts_begun",
            "reports_ready",
            *CONTROLS.FAILURE_SUMMARY_COLUMNS,
        ]
        summary = {
            "run_id": self.run_id,
            "group": self.run_id,
            "mix": f"{self.item_id}@1",
            "arrival": "wave",
            "warmup_seconds": "2.000",
            "window_seconds": "12.000",
            "max_active_attempts": "1",
            "device_capacity_bytes": str(CONTROLS.CAP_BYTES),
            "query_cap_bytes": str(CONTROLS.CAP_BYTES),
            "source_identity": self.identity["identity_sha256"],
            "memory_failure_ratio": "0.000000",
            "offered": "12",
            "completed": "12",
            "succeeded": "12",
            "slo_met": "12",
            "attempts_begun": "13",
            "reports_ready": "13",
            **{name: "0" for name in CONTROLS.FAILURE_SUMMARY_COLUMNS},
        }
        with (self.run_dir / "summary.tsv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=summary_fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerow(summary)

        attempt_fields = [
            "sequence",
            "phase",
            "item_id",
            "disposition",
            "terminal_outcome",
            "outcome_class",
            "fingerprint_match",
            "harness_timeout",
            "cleanup_disposition",
            "capacity_disposition",
            "error_message",
            "failure_code",
            "failure_status",
            "failure_operation",
            "failure_reason",
            "output_fingerprint",
            "slo_met",
        ]
        rows = []
        for sequence in range(CONTROLS.EXPECTED_TOTAL_ATTEMPTS):
            row = {
                "sequence": str(sequence),
                "phase": "warmup" if sequence == 0 else "window",
                "item_id": self.item_id,
                "disposition": "native",
                "terminal_outcome": "succeeded",
                "outcome_class": "succeeded",
                "fingerprint_match": "true",
                "harness_timeout": "false",
                "cleanup_disposition": "completed",
                "capacity_disposition": "released",
                "error_message": "none",
                "failure_code": "none",
                "failure_status": "none",
                "failure_operation": "none",
                "failure_reason": "none",
                "output_fingerprint": "output-fnv1a64:test",
                "slo_met": "true",
            }
            if sequence == bad_window:
                row["terminal_outcome"] = "failed"
            if sequence != missing_window:
                rows.append(row)
        with (self.run_dir / "attempts.tsv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=attempt_fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def test_accepts_complete_clean_window(self) -> None:
        self.assertEqual(CONTROLS.valid_output(self.run_id, self.item_id, self.identity), 12)

    def test_rejects_one_failed_attempt_even_with_eleven_successes(self) -> None:
        self._write_result(bad_window=1)
        self.assertIsNone(CONTROLS.valid_output(self.run_id, self.item_id, self.identity))

    def test_rejects_one_missing_attempt(self) -> None:
        self._write_result(missing_window=1)
        self.assertIsNone(CONTROLS.valid_output(self.run_id, self.item_id, self.identity))

    def test_same_library_path_with_changed_bytes_cannot_reuse_run(self) -> None:
        self.assertEqual(CONTROLS.valid_output(self.run_id, self.item_id, self.identity), 12)
        original_inode = self.native_library.stat().st_ino
        self.native_library.write_bytes(b"other library")
        self.assertEqual(self.native_library.stat().st_ino, original_inode)
        with (
            mock.patch.object(CONTROLS, "control_identity", return_value=(self.identity, "test-identity")),
            mock.patch.object(CONTROLS, "effective_run_id", return_value=self.run_id),
            self.assertRaisesRegex(RuntimeError, "incomplete or invalid evidence"),
        ):
            CONTROLS.run_control(self.run_id, self.item_id, "unused_slo.tsv", "tpch")

    def test_missing_library_snapshot_cannot_reuse_run(self) -> None:
        (self.run_dir / CONTROLS.LOADED_LIBRARIES_FILE).unlink()
        self.assertIsNone(CONTROLS.valid_output(self.run_id, self.item_id, self.identity))

    def test_identity_changes_effective_run_directory(self) -> None:
        self.assertNotEqual(
            CONTROLS.effective_run_id("control", "a" * 64),
            CONTROLS.effective_run_id("control", "b" * 64),
        )

    def test_same_binary_bytes_at_another_path_cannot_reuse_run(self) -> None:
        old_binary = CONTROLS.GOODPUT_BINARY
        old_artifact_root = CONTROLS.ARTIFACT_ROOT
        CONTROLS.ARTIFACT_ROOT = self.output_root
        first = self.output_root / "goodput-a"
        second = self.output_root / "goodput-b"
        first.write_bytes(b"same executable")
        second.write_bytes(b"same executable")
        (self.output_root / "karate-edges.parquet").write_bytes(b"edges")
        calibration = self.output_root / "graph_slo.tsv"
        calibration.write_text("item_id\ngraph.pagerank\n", encoding="utf-8")
        try:
            environment = CONTROLS.environment("graph.pagerank", calibration, "pagerank", "graph")
            CONTROLS.GOODPUT_BINARY = first
            _, first_identity = CONTROLS.control_identity("pagerank", "graph.pagerank", calibration, "graph", environment)
            CONTROLS.GOODPUT_BINARY = second
            _, second_identity = CONTROLS.control_identity("pagerank", "graph.pagerank", calibration, "graph", environment)
            self.assertNotEqual(first_identity, second_identity)
        finally:
            CONTROLS.GOODPUT_BINARY = old_binary
            CONTROLS.ARTIFACT_ROOT = old_artifact_root


if __name__ == "__main__":
    unittest.main()
