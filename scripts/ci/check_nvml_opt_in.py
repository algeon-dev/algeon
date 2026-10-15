#!/usr/bin/env python3
"""Check that NVML enters a resolved build graph only through an explicit `nvml` selection.

For every workspace member, the check resolves the no-feature build and each
non-default feature on its own with `cargo tree`, matching the feature-isolation
selections. `nvml-wrapper` must be absent unless the selection is `nvml`.
Each REQUIRED_NVML_SELECTIONS row must resolve it, which shows that the
forwarding still reaches the runtime. Cargo merges features across one
invocation, so `--workspace --all-features` resolving NVML is expected and out of scope.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Callable, Iterable

NVML_PACKAGE = "nvml-wrapper"
NVML_FEATURE = "nvml"
# Fixtures that enable every capability feature on purpose; the public API
# facade check compiles them with NVML on.
ALWAYS_NVML_PACKAGES = frozenset({"datafusion-facade-consumer"})
# Product selections that must reach the NVML runtime. `tools/nvml` alone
# forwards weakly and adds no backend, so it is paired with a run feature.
REQUIRED_NVML_SELECTIONS = (
    ("algeon-datafusion", "nvml"),
    ("algeon-server", "nvml"),
    ("bench", "nvml"),
    ("tools", "single-node-multi-gpu-validate-run,nvml"),
    ("tools", "clickbench-stage-probe,nvml"),
)

Resolver = Callable[[str, str], bool]


def cargo() -> str:
    return os.environ.get("CARGO", "cargo")


def workspace_features() -> dict[str, list[str]]:
    result = subprocess.run(
        [cargo(), "metadata", "--locked", "--no-deps", "--format-version", "1"],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(result.stdout)
    members = set(metadata["workspace_members"])
    return {
        package["name"]: sorted(feature for feature in package["features"] if feature != "default")
        for package in metadata["packages"]
        if package["id"] in members
    }


def resolves_nvml(package: str, features: str) -> bool:
    command = [cargo(), "tree", "--locked", "-p", package, "--no-default-features", "-e", "normal,build",
               "--prefix", "none", "-f", "{p}"]
    if features:
        command += ["--features", features]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"cargo tree failed for {package} [{features}]: {result.stderr.strip()}")
    return any(line.split(" ", 1)[0] == NVML_PACKAGE for line in result.stdout.splitlines())


def violations(packages: dict[str, list[str]], resolver: Resolver,
               required: Iterable[tuple[str, str]] = REQUIRED_NVML_SELECTIONS) -> list[str]:
    found = []
    for package, features in sorted(packages.items()):
        if package in ALWAYS_NVML_PACKAGES:
            continue
        for selection in ["", *features]:
            if selection != NVML_FEATURE and resolver(package, selection):
                found.append(f"{package} [{selection or 'no features'}] resolves {NVML_PACKAGE} without selecting nvml")
    for package, selection in required:
        if package not in packages:
            found.append(f"required NVML selection names unknown package {package}")
        elif not resolver(package, selection):
            found.append(f"{package} [{selection}] does not resolve {NVML_PACKAGE}")
    return found


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    try:
        found = violations(workspace_features(), resolves_nvml)
    except (OSError, RuntimeError, subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2
    for violation in found:
        print(f"FAIL: {violation}", file=sys.stderr)
    if found:
        return 1
    print("OK: NVML resolves only for explicit nvml selections")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
