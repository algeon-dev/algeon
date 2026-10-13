#!/usr/bin/env python3
"""Exercise gate ordering and environment propagation without Cargo builds."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import select
import shutil
import signal
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("toolchain_environment", ROOT / "scripts/ci/cargo_toolchain_environment.py")
ENVIRONMENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENVIRONMENT)


def package(name: str, links: str | None = None, deps: tuple[tuple[str, str | None], ...] = ()) -> tuple[dict, dict]:
    return ({"id": name, "name": name, "links": links},
            {"id": name, "deps": [{"pkg": dep, "dep_kinds": [{"kind": kind}]} for dep, kind in deps]})


# `consumer` builds against `lib` normally, so `lib`'s dev-dependency on the
# native member never reaches it; `vendored` links a library outside the workspace.
NATIVE_FREE_GRAPH = [
    package("sys", links="native"),
    package("vendored", links="zstd"),
    package("plain", deps=(("vendored", None),)),
    package("direct", deps=(("sys", None),)),
    package("builder", deps=(("sys", "build"),)),
    package("lib", deps=(("sys", "dev"),)),
    package("consumer", deps=(("lib", None),)),
]
NATIVE_FREE_METADATA = {
    "packages": [node[0] for node in NATIVE_FREE_GRAPH],
    "workspace_members": [node[0]["id"] for node in NATIVE_FREE_GRAPH if node[0]["id"] != "vendored"],
    "resolve": {"nodes": [node[1] for node in NATIVE_FREE_GRAPH]},
}


class GateTests(unittest.TestCase):
    def run_gate(self, build_status: int = 0, test_status: int = 0, metadata_status: int = 0, mode: str | None = None, cargo_env: bool = True) -> tuple[list[dict], str, dict[str, str]]:
        with tempfile.TemporaryDirectory(prefix="algeon-check-all-") as temporary:
            root = Path(temporary)
            script = (ROOT / "scripts/check_all.sh").read_text()
            (root / "scripts/ci/tests").mkdir(parents=True)
            (root / "scripts/check_all.sh").write_text(script)
            shutil.copy(ROOT / "scripts/ci/cargo_toolchain_environment.py", root / "scripts/ci")
            shutil.copy(ROOT / "scripts/ci/native_free_packages.py", root / "scripts/ci")
            for name in re.findall(r"scripts/[\w/.-]+\.(?:sh|py)", script):
                path = root / name
                if path.exists():
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("exit 0\n" if name.endswith(".sh") else "pass\n")
            wrapper = root / "scripts/ci/nextest_gpu_lock.sh"
            wrapper.write_text('#!/usr/bin/env bash\nshift\nexec "$@"\n')
            wrapper.chmod(0o755)
            bindir = root / "bin"
            bindir.mkdir()
            git = bindir / "git"
            git.write_text('#!/usr/bin/env bash\n[[ "$*" == *"rev-parse HEAD"* ]] && echo test-head\nexit 0\n')
            git.chmod(0o755)
            packages = re.search(r"FEATURE_ISOLATION_SHARDS=\((.*?)\n\)", script, re.DOTALL)[1]
            packages = " ".join(re.findall(r'"([^"]+)"', packages)).split()
            mock = bindir / "cargo"
            mock.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
def event(kind):
    with open(os.environ["GATE_EVENTS"], "a") as log:
        log.write(json.dumps({"kind": kind, "args": args, "loader": os.environ.get("LD_LIBRARY_PATH"), "nice": os.getpriority(os.PRIO_PROCESS, 0),
                              "docs_rs": os.environ.get("DOCS_RS"), "target_dir": os.environ.get("CARGO_TARGET_DIR")}) + "\\n")
event("command")
if args == ["nextest", "--version"]:
    print("cargo-nextest test-version")
elif args == ["hack", "--version"]:
    print("cargo-hack 0.6.45")
elif "config" in args and os.environ["GATE_CARGO_ENV"] == "0":
    # Like Cargo on a host whose configuration has no [env] table.
    if "env" in args:
        print("error: config value `env` is not set", file=sys.stderr)
        sys.exit(101)
    print('alias.nt = ["nextest"] # /repo/.cargo/config.toml')
elif "config" in args:
    print('env.CUDA_HOME.value = "/forced/cuda" # /machine/.cargo/config.toml')
    print('env.CUDA_HOME.force = true # /machine/.cargo/config.toml')
elif "--print-command-list" in args:
    packages = json.loads(os.environ["GATE_PACKAGES"]) if "--workspace" in args else [args[i + 1] for i, arg in enumerate(args) if arg == "-p"]
    print("\\n".join(packages))
elif args[:1] == ["metadata"] and "GATE_NATIVE_FREE_METADATA" in os.environ:
    print(os.environ["GATE_NATIVE_FREE_METADATA"])
elif args[:1] == ["metadata"]:
    print(json.dumps({"mock_cargo_metadata": True}))
    sys.exit(int(os.environ["GATE_METADATA_STATUS"]))
elif args[:2] == ["nextest", "list"]:
    metadata = Path(args[args.index("--cargo-metadata") + 1])
    assert json.loads(metadata.read_text()) == {"mock_cargo_metadata": True}
    with open(os.environ["GATE_STARTED"], "w") as started:
        started.write("started\\n")
    with open(os.environ["GATE_RELEASE"]) as release:
        release.read(1)
    event("build-finished")
    print(json.dumps({"mock_binaries_metadata": True}))
    sys.exit(int(os.environ["GATE_BUILD_STATUS"]))
elif args[:2] == ["nextest", "run"] and "-p" in args:
    sys.exit(int(os.environ["GATE_TEST_STATUS"]))
elif args[:2] == ["nextest", "run"]:
    for flag, key in [("--cargo-metadata", "mock_cargo_metadata"), ("--binaries-metadata", "mock_binaries_metadata")]:
        metadata = Path(args[args.index(flag) + 1])
        assert metadata.parent == Path("target/check-all/latest").resolve()
        assert json.loads(metadata.read_text()) == {key: True}
    assert "--workspace" not in args and "--all-features" not in args
    event("metadata-loaded")
    sys.exit(int(os.environ["GATE_TEST_STATUS"]))
elif args[:1] == ["nt"]:
    raise AssertionError("test execution must reuse the foreground build")
''')
            mock.chmod(0o755)
            events_path = root / "events.jsonl"
            started, release = root / "started", root / "release"
            os.mkfifo(started)
            os.mkfifo(release)
            started_fd = os.open(started, os.O_RDWR | os.O_NONBLOCK)
            release_fd = os.open(release, os.O_RDWR | os.O_NONBLOCK)
            environ = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", CUDA_HOME="/shell/cuda", LD_LIBRARY_PATH="polluted:polluted", GATE_EVENTS=str(events_path), GATE_STARTED=str(started), GATE_RELEASE=str(release), GATE_PACKAGES=json.dumps(packages), GATE_BUILD_STATUS=str(build_status), GATE_TEST_STATUS=str(test_status), GATE_METADATA_STATUS=str(metadata_status), GATE_CARGO_ENV=str(int(cargo_env)))
            # An enclosing gate's settings would redirect these mock runs into
            # its own log directory and Cargo target.
            for name in ("CHECK_ALL_PHASE_LOG_DIR", "DOCS_RS", "CARGO_TARGET_DIR"):
                environ.pop(name, None)
            if mode == "--native-free":
                environ["GATE_NATIVE_FREE_METADATA"] = json.dumps(NATIVE_FREE_METADATA)
            try:
                process = subprocess.Popen(["bash", "scripts/check_all.sh", *([mode] if mode else [])], cwd=root, env=environ, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
                try:
                    if not metadata_status and not mode:
                        self.assertTrue(select.select([started_fd], [], [], 10)[0], "gate did not reach the compile barrier")
                        os.read(started_fd, 1024)
                        before = [json.loads(line) for line in events_path.read_text().splitlines()]
                        self.assertFalse(any(event["args"][0] in ("clippy", "check", "doc") or event["args"][:2] == ["hack", "check"] for event in before))
                        os.write(release_fd, b"x")
                    output, _ = process.communicate(timeout=20)
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.communicate()
            finally:
                os.close(started_fd)
                os.close(release_fd)
            self.assertEqual(process.returncode, 1 if build_status or test_status or metadata_status else 0, output)
            events = [json.loads(line) for line in events_path.read_text().splitlines()]
            rows = {}
            for row in (root / "target/check-all/latest/summary.tsv").read_text().splitlines()[1:]:
                name, status, *_ = row.split("\t")
                rows[name] = status
            canonical = ":".join(str(root / "target/native" / name) for name in ("cudf-install/lib", "cuvs-install/lib", "cugraph-install/lib", "cuml-install/lib", "cugraph-build"))
            self.assertEqual({event["loader"] for event in events}, {canonical})
            self.assertIn(f"check_all: Cargo CUDA_HOME={'/forced/cuda' if cargo_env else '/shell/cuda'}", output)
            if not mode:
                self.assertTrue(any(event["args"][:1] == ["test"] for event in events), "doctests did not run")
            for event in events:
                if event["args"][0] in ("clippy", "check", "doc") or event["args"][:2] == ["hack", "check"] and "--print-command-list" not in event["args"]:
                    self.assertEqual(event["nice"], min(os.getpriority(os.PRIO_PROCESS, 0) + 10, 19), event)
            return events, output, rows

    def test_compile_barrier_loader_and_low_priority_checks(self):
        events, _, rows = self.run_gate()
        self.assertEqual(rows["test-build"], "PASS")
        self.assertEqual(rows["tests"], "PASS")
        build_index = next(index for index, event in enumerate(events) if event["kind"] == "build-finished")
        self.assertTrue(all(index > build_index for index, event in enumerate(events) if event["args"][0] in ("clippy", "check", "doc")))
        builds = [event for event in events if event["kind"] == "command" and event["args"][:2] == ["nextest", "list"]]
        self.assertEqual(len(builds), 1)
        self.assertIn("--workspace", builds[0]["args"])
        self.assertIn("--all-features", builds[0]["args"])
        self.assertIn("binaries-only", builds[0]["args"])
        self.assertTrue(any(event["kind"] == "metadata-loaded" for event in events))

    def test_compile_failure_skips_execution_and_continues_checks(self):
        events, _, rows = self.run_gate(build_status=7)
        self.assertEqual(rows["test-build"], "FAIL(7)")
        self.assertEqual(rows["tests"], "SKIP(test-build-failed)")
        self.assertFalse(any(event["args"][:2] == ["nextest", "run"] for event in events))
        self.assertEqual(rows["feature-isolation"], "PASS")

    def test_metadata_failure_skips_build_and_execution(self):
        events, _, rows = self.run_gate(metadata_status=6)
        self.assertEqual(rows["test-build"], "FAIL(6)")
        self.assertEqual(rows["tests"], "SKIP(test-build-failed)")
        self.assertFalse(any(event["args"][:2] in (["nextest", "list"], ["nextest", "run"]) for event in events))
        self.assertEqual(rows["feature-isolation"], "PASS")

    def test_execution_failure_still_runs_doctests(self):
        _, _, rows = self.run_gate(test_status=9)
        self.assertEqual(rows["tests"], "FAIL(9)")
        self.assertEqual(rows["doctests"], "PASS")

    def test_native_free_mode_runs_docs_rs_checks_and_native_free_tests(self):
        events, output, rows = self.run_gate(mode="--native-free")
        self.assertEqual(set(rows), {"gate", "rustfmt", "native-sources", "policy-scripts", "clippy", "orphan-sources", "rustdoc",
                                     "native-free-tests", "shipped-clippy", "msrv", "docsrs-native-free",
                                     "public-api-consumers", "publication-policy"})
        self.assertEqual(set(rows.values()), {"PASS"}, output)
        self.assertIn("PARTIAL", output)
        cargo = [event for event in events if event["args"][0] in ("clippy", "doc", "metadata", "nextest", "hack")
                 and "--version" not in event["args"]]
        self.assertTrue(cargo)
        for event in cargo:
            self.assertEqual(event["docs_rs"], "1", event)
            self.assertTrue(event["target_dir"].endswith("/target/native-free"), event)
        self.assertFalse(any(event["args"][:1] == ["test"] or event["args"][:2] == ["nextest", "list"] for event in events))
        runs = [event["args"] for event in events if event["args"][:2] == ["nextest", "run"]]
        self.assertEqual(len(runs), 1)
        selected = [runs[0][index + 1] for index, arg in enumerate(runs[0]) if arg == "-p"]
        self.assertEqual(selected, ["consumer", "plain"])

    def test_native_free_mode_runs_without_cargo_env_config(self):
        _, output, rows = self.run_gate(mode="--native-free", cargo_env=False)
        self.assertEqual(rows["gate"], "PASS", output)

    def test_native_free_test_failure_fails_the_mode(self):
        _, output, rows = self.run_gate(test_status=9, mode="--native-free")
        self.assertEqual(rows["native-free-tests"], "FAIL(9)")
        self.assertEqual(rows["msrv"], "PASS")
        self.assertIn("FAIL: native-free checks reported problems.", output)

    def test_toolchain_diagnostics_apply_cargo_precedence_and_relative_paths(self):
        values = ENVIRONMENT.effective_environment('''
env.CC = "/config/cc"
env.CXX.value = "/config/cxx"
env.CXX.force = true
env.CUDACXX.value = "cuda/bin/nvcc" # /workspace/.cargo/config.toml
env.CUDACXX.relative = true # /workspace/.cargo/config.toml
''', {"CC": "/shell/cc", "CXX": "/shell/cxx", "CUDA_HOME": "/shell/cuda"})
        self.assertEqual(values["CC"], "/shell/cc")
        self.assertEqual(values["CXX"], "/config/cxx")
        self.assertEqual(values["CUDACXX"], "/workspace/cuda/bin/nvcc")
        self.assertEqual(values["CUDA_HOME"], "/shell/cuda")
        self.assertIsNone(values["CUDA_PATH"])


if __name__ == "__main__":
    unittest.main()
