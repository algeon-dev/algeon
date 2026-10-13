#!/usr/bin/env python3
"""Exercise archive and docs.rs isolation contracts without native builds."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/ci"))
import release_archives as release


class ReleaseArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="algeon-release-policy-")
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.packages = []
        for name in sorted(release.PUBLISHED):
            path = self.output / "sources" / f"{name}-0.1.0"
            path.mkdir(parents=True)
            features = ["server"] if name == "algeon-datafusion" else []
            manifest = (
                f'[package]\nname = "{name}"\nversion = "0.1.0"\npublish = ["crates-io"]\n'
                '[package.metadata.docs.rs]\ndefault-target = "x86_64-unknown-linux-gnu"\ntargets = []\n'
                f'features = {json.dumps(features)}\n[features]\n'
                + ''.join(f'{feature} = []\n' for feature in features)
            )
            (path / "Cargo.toml").write_text(manifest)
            self.packages.append(dict(name=name, version="0.1.0", path=str(path), docs_features=features))
        self.build_inputs = {name: [] for name in release.PUBLISHED}
        (self.output / "release.json").write_text(json.dumps(dict(packages=self.packages, build_inputs=self.build_inputs)))
        (self.output / "patches.toml").write_text('[patch.crates-io]\n' + ''.join(
            f'{package["name"]} = {{ path = {json.dumps(package["path"])} }}\n' for package in self.packages
        ))

    def test_rejects_checkout_dependency_and_inexact_release_version(self):
        manifest = Path(self.packages[0]["path"]) / "Cargo.toml"
        original = manifest.read_text()
        for dependency in (
            'version = "=0.1.0", path = "/checkout/components/algeon-interop"',
            'version = "0.1.0"',
            'version = "=0.1.0", git = "https://example.invalid/algeon"',
        ):
            with self.subTest(dependency=dependency):
                manifest.write_text(original + '\n[dependencies]\nalgeon-interop = { ' + dependency + ' }\n')
                with self.assertRaises(ValueError):
                    release.validate_archives(self.output)
        manifest.write_text(original + '\n[dependencies]\nalgeon-interop = { version = "=0.1.0" }\n')
        release.validate_archives(self.output)

    def test_rejects_drifted_archive_set_and_docs_features(self):
        manifest = Path(self.packages[-1]["path"]) / "Cargo.toml"
        manifest.write_text(manifest.read_text() + '\ntest-hooks = []\ncugraph = []\n')
        with self.assertRaisesRegex(ValueError, "docs.rs features"):
            release.validate_archives(self.output)
        (self.output / "release.json").write_text(json.dumps(dict(packages=self.packages[:-1])))
        with self.assertRaisesRegex(ValueError, "publication allowlist"):
            release.validate_archives(self.output)

    def test_rejects_dirty_release_before_packaging(self):
        checkout = self.output / "checkout"
        checkout.mkdir()
        subprocess.run(["git", "init", "--quiet", str(checkout)], check=True)
        (checkout / "untracked").write_text("dirty release")
        with patch.object(release, "ROOT", checkout):
            with self.assertRaisesRegex(ValueError, "clean checkout"):
                release.prepare(self.output / "release", require_clean=True)
        self.assertFalse((self.output / "release").exists())

    def test_rejects_archive_missing_native_input(self):
        package = next(package for package in self.packages if package["name"] == "algeon-cuvs-sys")
        self.build_inputs[package["name"]] = ["native/abi_probe.c"]
        (self.output / "release.json").write_text(json.dumps(dict(packages=self.packages, build_inputs=self.build_inputs)))
        with self.assertRaisesRegex(ValueError, "native/abi_probe.c"):
            release.validate_archives(self.output)
        source = Path(package["path"]) / "native/abi_probe.c"
        source.parent.mkdir()
        source.write_text("int probe(void) { return 0; }\n")
        release.validate_archives(self.output)

    def test_docs_fetches_before_offline_builds_with_no_native_or_config_selectors(self):
        bindir = self.output / "bin"
        bindir.mkdir()
        cargo = bindir / "cargo"
        cargo.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
with open(os.environ["ARCHIVE_TEST_EVENTS"], "a") as stream:
    stream.write(json.dumps({"args": sys.argv[1:], "cwd": os.getcwd(), "env": dict(os.environ)}) + "\\n")
assert not (Path(os.environ["CARGO_HOME"]) / "config.toml").exists()
if sys.argv[1] == "doc":
    manifest = Path(sys.argv[sys.argv.index("--manifest-path") + 1])
    name = manifest.parent.name.removesuffix("-0.1.0")
    if name.endswith("-sys"):
        build = Path(os.environ["CARGO_TARGET_DIR"]) / name
        (build / "out").mkdir(parents=True, exist_ok=True)
        (build / "output").write_text("cargo:rerun-if-env-changed=DOCS_RS\\n")
        print(json.dumps({"reason": "build-script-executed", "package_id": f"path+{manifest.parent.as_uri()}#{name}@0.1.0", "out_dir": str(build / "out"), "linked_libs": [], "linked_paths": []}))
''')
        cargo.chmod(0o755)
        original_home = self.output / "original-home"
        original_home.mkdir()
        (original_home / "registry").mkdir()
        (original_home / "bin").mkdir()
        (original_home / "config.toml").write_text('[env]\nCUDA_HOME = { value = "/forced/cuda", force = true }\n')
        events_path = self.output / "events.jsonl"
        env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", CARGO_HOME=str(original_home),
                   ALGEON_NATIVE_DIR="/ambient/native", CUDF_INSTALL_DIR="/ambient/cudf", CUGRAPH_SYS_AUTO_BUILD="1",
                   CUDA_HOME="/ambient/cuda", CUDAToolkit_ROOT="/ambient/cuda", LD_LIBRARY_PATH="/ambient/libs",
                   RUSTFLAGS="--cfg ambient", CARGO_ENCODED_RUSTFLAGS="ambient", ARCHIVE_TEST_EVENTS=str(events_path))
        ambient_native = ("DLPACK_INCLUDE_DIR", "LIBCUVS_USE_PYTHON", "CONDA_PREFIX", "VIRTUAL_ENV", "INSTALL_PREFIX",
                          "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS")
        env.update({selector: "/ambient/override" for selector in ambient_native})
        with patch.dict(os.environ, env, clear=True):
            isolated = release.clean_environment(self.output)
        self.assertEqual((Path(isolated["CARGO_HOME"]) / "bin").resolve(), (original_home / "bin").resolve())
        for selector in ambient_native:
            self.assertNotIn(selector, isolated)
        result = subprocess.run(["bash", str(ROOT / "scripts/ci/check_docsrs_native_free.sh"), "--archives", str(self.output)],
                                cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        self.assertEqual(len(events), 2 * len(release.PUBLISHED))
        self.assertEqual([event["args"][0] for event in events], ["fetch"] * len(release.PUBLISHED) + ["doc"] * len(release.PUBLISHED))
        for event in events:
            self.assertEqual(event["cwd"], str(self.output))
            for selector in ("CUGRAPH_SYS_AUTO_BUILD", "LD_LIBRARY_PATH", "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS",
                             "LIBCUVS_USE_PYTHON", "INSTALL_PREFIX", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS"):
                self.assertNotIn(selector, event["env"])
            for selector in ("ALGEON_NATIVE_DIR", "CUDF_INSTALL_DIR", "CUDA_HOME", "CUDAToolkit_ROOT", "CUDACXX"):
                self.assertNotIn("ambient", event["env"][selector])
                self.assertFalse(Path(event["env"][selector]).exists())
            self.assertEqual(event["env"]["DOCS_RS"], "1")
            if event["args"][0] == "doc":
                self.assertIn("--offline", event["args"])
                self.assertIn("--locked", event["args"])
                self.assertEqual(event["env"]["CARGO_NET_OFFLINE"], "true")
                self.assertEqual(event["env"]["RUSTDOCFLAGS"], "-D warnings")
        adapter = next(event for event in events if event["args"][0] == "doc" and "algeon-datafusion-" in " ".join(event["args"]))
        self.assertEqual(adapter["args"][adapter["args"].index("--features") + 1], "server")
        self.assertTrue((self.output / "cargo-home/registry").is_symlink())
        evidence = json.loads((self.output / "logs/docsrs/native-free-evidence.json").read_text())
        self.assertEqual(set(evidence["sys_build_scripts"]), {name for name in release.PUBLISHED if name.endswith("-sys")})

    def test_docs_build_failure_reaches_caller(self):
        bindir = self.output / "bin"
        bindir.mkdir()
        cargo = bindir / "cargo"
        cargo.write_text('#!/usr/bin/env bash\nif [[ $1 == doc ]]; then echo "injected rustdoc failure"; exit 17; fi\n')
        cargo.chmod(0o755)
        env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}")
        result = subprocess.run(["bash", str(ROOT / "scripts/ci/check_docsrs_native_free.sh"), "--archives", str(self.output)],
                                cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("injected rustdoc failure", result.stderr)

    def test_docs_rejects_native_linking_in_cargo_messages_or_build_output(self):
        bindir = self.output / "bin"
        bindir.mkdir()
        cargo = bindir / "cargo"
        cargo.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
if sys.argv[1] == "doc":
    package = Path(os.environ["ARCHIVE_TEST_PACKAGE"])
    build = Path(os.environ["CARGO_TARGET_DIR"]) / "native-link-regression"
    (build / "out").mkdir(parents=True, exist_ok=True)
    directive = "cargo::rustc-link-arg=-Wl,-rpath,/ambient/native\\n" if os.environ["ARCHIVE_TEST_LINK"] == "output" else ""
    (build / "output").write_text(directive)
    libraries = ["dylib=cudf_rust"] if os.environ["ARCHIVE_TEST_LINK"] == "message" else []
    print(json.dumps({"reason": "build-script-executed", "package_id": f"path+{package.as_uri()}#algeon-cudf-sys@0.1.0", "out_dir": str(build / "out"), "linked_libs": libraries, "linked_paths": []}))
''')
        cargo.chmod(0o755)
        package = next(package for package in self.packages if package["name"] == "algeon-cudf-sys")
        for kind in ("message", "output"):
            with self.subTest(kind=kind):
                env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}",
                           ARCHIVE_TEST_PACKAGE=package["path"], ARCHIVE_TEST_LINK=kind)
                result = subprocess.run(["bash", str(ROOT / "scripts/ci/check_docsrs_native_free.sh"), "--archives", str(self.output)],
                                        cwd=ROOT, env=env, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("DOCS_RS emitted native linker directives", result.stderr)


if __name__ == "__main__":
    unittest.main()
