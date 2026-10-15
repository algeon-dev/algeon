#!/usr/bin/env python3
"""Exercise release consumer rejection paths without Cargo or GPU work."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/ci"))
import registry_consumer as consumer


class RegistryConsumerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="algeon-registry-policy-")
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)
        self.app = self.output / "app"
        self.archive = self.output / "sources/algeon-datafusion-0.1.0"
        self.app.mkdir()
        self.archive.mkdir(parents=True)
        self.packages = [{"name": "algeon-datafusion", "path": str(self.archive)}]
        self.metadata = {
            "packages": [
                {"name": "registry-consumer", "source": None, "manifest_path": str(self.app / "Cargo.toml")},
                {"name": "algeon-datafusion", "source": None, "manifest_path": str(self.archive / "Cargo.toml")},
            ],
            "resolve": {"nodes": [{"id": "path+file:///archives/algeon-datafusion#0.1.0", "features": []}]},
        }

    def test_rejects_workspace_source_instead_of_extracted_archive(self):
        consumer.verify_resolution(self.metadata, self.packages, self.app, None)
        self.metadata["packages"][1]["manifest_path"] = str(ROOT / "crates/algeon-datafusion/Cargo.toml")
        with self.assertRaisesRegex(RuntimeError, "outside the extracted release archives"):
            consumer.verify_resolution(self.metadata, self.packages, self.app, None)

    def test_rejects_extra_path_dependency(self):
        self.metadata["packages"].append({
            "name": "local-helper", "source": None,
            "manifest_path": str(self.output / "helper/Cargo.toml"),
        })
        with self.assertRaisesRegex(RuntimeError, "unexpected path dependency"):
            consumer.verify_resolution(self.metadata, self.packages, self.app, None)

    def test_locked_rejects_third_party_drift_while_fresh_accepts_new_resolution(self):
        release_lock = self.output / "release.lock"
        package = {
            "name": "chacha20", "version": "0.10.2",
            "source": "registry+https://github.com/rust-lang/crates.io-index", "checksum": "release-checksum",
        }

        def write_lock(path, entry):
            path.write_text("version = 4\n[[package]]\n" + "".join(
                f"{key} = {json.dumps(value)}\n" for key, value in entry.items()
            ))

        write_lock(release_lock, package)
        write_lock(self.app / "Cargo.lock", package)
        consumer.verify_resolution(self.metadata, self.packages, self.app, release_lock)
        for changed in (
            {**package, "version": "0.10.3"},
            {**package, "checksum": "different-checksum"},
            {**package, "source": "git+https://example.invalid/chacha20#other-source"},
        ):
            with self.subTest(changed=changed):
                write_lock(self.app / "Cargo.lock", changed)
                consumer.verify_resolution(self.metadata, self.packages, self.app, None)
                with self.assertRaisesRegex(RuntimeError, "release lock resolution changed"):
                    consumer.verify_resolution(self.metadata, self.packages, self.app, release_lock)

    def test_rejects_test_hooks_in_release_resolution(self):
        self.metadata["resolve"]["nodes"][0]["features"] = ["test-hooks"]
        with self.assertRaisesRegex(RuntimeError, "non-release feature enabled"):
            consumer.verify_resolution(self.metadata, self.packages, self.app, None)

    def test_nvml_packages_resolve_only_for_rows_that_select_nvml(self):
        for package in self.metadata["packages"]:
            package["id"] = f"path+file:///{package['name']}#0.1.0"
        nvml = [{"id": f"registry#{name}@1", "name": name} for name in consumer.NVML_PACKAGES]
        consumer.verify_nvml_selection(self.metadata, "")
        consumer.verify_nvml_selection(self.metadata, "cugraph,cuvs,cuml,server")
        with self.assertRaisesRegex(RuntimeError, "the nvml row resolved"):
            consumer.verify_nvml_selection(self.metadata, "nvml")
        self.metadata["packages"].extend(nvml)
        self.metadata["resolve"]["nodes"].extend({"id": package["id"], "features": []} for package in nvml)
        consumer.verify_nvml_selection(self.metadata, "nvml")
        consumer.verify_nvml_selection(self.metadata, "cugraph,cuvs,cuml,server,nvml")
        with self.assertRaisesRegex(RuntimeError, "without selecting nvml"):
            consumer.verify_nvml_selection(self.metadata, "server")

    def test_loader_accepts_soname_symlinks_and_rejects_other_native_install(self):
        native = self.output / "native"
        library = native / "cudf-install/lib"
        library.mkdir(parents=True)
        names = ("libcudf_rust.so", "libcudf.so", "librmm.so")
        for name in names:
            (library / f"{name}.26").write_bytes(b"native-library")
            (library / name).symlink_to(f"{name}.26")
        loader = "\n".join(f"\t{name}.26 => {library / (name + '.26')} (0x1)" for name in names)
        consumer.verify_loader(loader, native, "")
        other = self.output / "other-install/libcudf.so.26"
        other.parent.mkdir()
        other.write_bytes(b"other-native-library")
        wrong_loader = loader.replace(str(library / "libcudf.so.26"), str(other))
        with self.assertRaisesRegex(RuntimeError, "loader selected.*libcudf.so"):
            consumer.verify_loader(wrong_loader, native, "")
        with self.assertRaisesRegex(RuntimeError, "unresolved runtime dependency"):
            consumer.verify_loader(loader + "\nlibcuda.so.1 => not found\n", native, "")
        with self.assertRaisesRegex(RuntimeError, "loader selected.*librmm.so"):
            consumer.verify_loader("\n".join(loader.splitlines()[:-1]), native, "")

    def test_quickstart_rejects_drift_and_missing_snippet(self):
        fixture = self.output / "fixture"
        (fixture / "src").mkdir(parents=True)
        (fixture / "Cargo.toml").write_text('[dependencies]\nalgeon-datafusion = "=0.1.0"\n')
        (fixture / "src/main.rs").write_text("fn main() {}\n")
        document = self.output / "website/docs/build/rust-dependency.md"
        document.parent.mkdir(parents=True)
        matching = "\n".join(
            "{/* registry-consumer:" + filename + " */}\n```text\n" +
            (fixture / filename).read_text().strip() + "\n```"
            for filename in ("Cargo.toml", "src/main.rs")
        )
        with patch.object(consumer, "ROOT", self.output), patch.object(consumer, "FIXTURE", fixture):
            document.write_text(matching)
            consumer.check_quickstart()
            for changed in (
                matching.replace('"=0.1.0"', '"=0.2.0"'),
                matching.replace("registry-consumer:src/main.rs", "untracked-example:src/main.rs"),
            ):
                document.write_text(changed)
                with self.assertRaisesRegex(RuntimeError, "differs from the tested registry consumer"):
                    consumer.check_quickstart()


if __name__ == "__main__":
    unittest.main()
