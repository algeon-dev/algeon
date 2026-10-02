import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "capture_gpu_query_engine_baseline.py"
SPEC = importlib.util.spec_from_file_location("capture_gpu_query_engine_baseline", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class LoadedLibrarySnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.binary = self.root / "native_tpch_benchmark"
        self.binary.write_bytes(b"benchmark")
        self.cudf_rust = self.root / "libcudf_rust.so"
        self.cudf = self.root / "libcudf.so"
        self.cudf_rust.write_bytes(b"rust")
        self.cudf.write_bytes(b"cudf")

    def tearDown(self):
        self.tempdir.cleanup()

    @staticmethod
    def entry(path):
        stat = path.stat()
        return {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": stat.st_size,
            "device": stat.st_dev,
            "inode": stat.st_ino,
        }

    def write_snapshot(self, libraries):
        snapshot = self.root / "loaded-native-libraries.json"
        snapshot.write_text(
            json.dumps({"source": "/proc/self/maps", "binary": self.entry(self.binary), "libraries": libraries}) + "\n",
            encoding="utf-8",
        )
        return snapshot

    def test_accepts_runtime_paths_and_hashes(self):
        snapshot = self.write_snapshot([self.entry(self.cudf_rust), self.entry(self.cudf)])

        result = MODULE.loaded_library_snapshot(snapshot, self.root, self.binary)

        self.assertEqual(result["library_count"], 2)
        self.assertEqual(result["libraries"][0]["path"], str(self.cudf_rust))
        self.assertTrue(result["libraries"][0]["verified"])

    def test_rejects_replaced_library(self):
        snapshot = self.write_snapshot([self.entry(self.cudf_rust), self.entry(self.cudf)])
        self.cudf.write_bytes(b"replaced")

        with self.assertRaisesRegex(MODULE.ManifestError, "identity changed|hash changed|size changed"):
            MODULE.loaded_library_snapshot(snapshot, self.root, self.binary)

    def test_rejects_missing_core_library(self):
        snapshot = self.write_snapshot([self.entry(self.cudf_rust)])

        with self.assertRaisesRegex(MODULE.ManifestError, "missing required dependency libcudf.so"):
            MODULE.loaded_library_snapshot(snapshot, self.root, self.binary)

    def test_rejects_binary_from_another_run(self):
        snapshot = self.write_snapshot([self.entry(self.cudf_rust), self.entry(self.cudf)])
        self.binary.write_bytes(b"different benchmark")

        with self.assertRaisesRegex(MODULE.ManifestError, "archived benchmark binary does not match"):
            MODULE.loaded_library_snapshot(snapshot, self.root, self.binary)


if __name__ == "__main__":
    unittest.main()
