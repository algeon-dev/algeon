#!/usr/bin/env python3
"""Enforce the workspace publication policy.

Every member declares ``publish`` directly: the release closure as
``publish = ["crates-io"]`` and everything else as ``publish = false``.  The
root sets no ``[workspace.package].publish`` at all, so nothing can inherit a
publication status by accident.

That explicitness is the point.  The workspace previously inherited a root
``publish = false`` through ``publish.workspace = true``, and a later refactor
that moved keys into ``[workspace.package]`` silently reverted the whole
release closure to unpublishable -- ``cargo package`` failed on a contract no
manifest appeared to state.  An inherited publication status hides which
packages ship, so this check requires each manifest to say so itself and
pins the closure to the list below.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

# The approved release closure. Publishing to crates.io requires every normal
# and build dependency to be published too, so this set must stay closed under
# the dependency relation -- a check below enforces that.
PUBLISHED = frozenset(
    {
        "algeon-datafusion",
        "algeon-query-engine",
        "algeon-query-planning",
        "algeon-query-runtime",
        "algeon-cudf",
        "algeon-cudf-sys",
        "algeon-cugraph",
        "algeon-cugraph-sys",
        "algeon-cuvs",
        "algeon-cuvs-sys",
        "algeon-build-support",
        "algeon-interop",
    }
)


def fail(message: str) -> None:
    print(f"FAIL: workspace publication policy: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    root_manifest_path = ROOT / "Cargo.toml"
    with root_manifest_path.open("rb") as source:
        root_manifest = tomllib.load(source)
    workspace_package = root_manifest.get("workspace", {}).get("package", {})
    if "publish" in workspace_package:
        fail("root [workspace.package] must not set publish; each member declares it directly")

    result = subprocess.run(
        [
            "cargo",
            "metadata",
            "--locked",
            "--all-features",
            "--no-deps",
            "--format-version",
            "1",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(result.stdout)
    packages_by_id = {package["id"]: package for package in metadata["packages"]}
    workspace_ids = metadata["workspace_members"]
    if not workspace_ids:
        fail("cargo metadata reported no workspace members")

    members = {}
    for package_id in workspace_ids:
        package = packages_by_id.get(package_id)
        if package is None:
            fail(f"workspace member {package_id!r} is missing from metadata packages")
        name = package["name"]
        members[name] = package
        expected_published = name in PUBLISHED

        manifest_path = Path(package["manifest_path"])
        with manifest_path.open("rb") as source:
            manifest = tomllib.load(source)
        declared = manifest.get("package", {}).get("publish")
        if declared is None:
            fail(f"{name} must declare [package] publish directly; found none")
        if declared == {"workspace": True}:
            fail(f"{name} must not inherit publish from the workspace; declare it directly")

        if expected_published:
            if declared != ["crates-io"]:
                fail(f'{name} is in the release closure and must set publish = ["crates-io"]; found {declared!r}')
        elif declared is not False:
            fail(f"{name} is not in the release closure and must set publish = false; found {declared!r}")

    unknown = PUBLISHED - set(members)
    if unknown:
        fail(f"release closure names packages that are not workspace members: {', '.join(sorted(unknown))}")

    # crates.io rejects a package whose normal or build dependency is
    # unpublished, so catch that here rather than mid-release.
    for name in sorted(PUBLISHED):
        for dependency in members[name]["dependencies"]:
            if dependency.get("kind") == "dev":
                continue
            target = dependency["name"]
            if target in members and target not in PUBLISHED:
                fail(f"{name} is published but depends on unpublished workspace package {target}")

    print(
        "PASS: workspace publication policy: "
        f"{len(PUBLISHED)} published, {len(members) - len(PUBLISHED)} private, "
        "all declared directly"
    )


if __name__ == "__main__":
    main()
