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

The README each published crate shows on crates.io is checked too. It must
not name workspace-only paths or commands, and no two crates may open with the
same paragraph.
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
# the dependency relation; a check below enforces that.
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
        "algeon-cuml",
        "algeon-cuml-sys",
        "algeon-build-support",
        "algeon-interop",
    }
)

# Tokens that resolve only inside this checkout.
README_INTERNAL_TOKENS = ("scripts/", "cargo nt", "target/native", "check_all.sh")


def fail(message: str) -> None:
    print(f"FAIL: workspace publication policy: {message}", file=sys.stderr)
    raise SystemExit(1)


def readme_path(package: dict, manifest: dict) -> Path | None:
    """The README file Cargo packages with ``package``; None when the manifest disables it."""
    package_dir = Path(package["manifest_path"]).parent
    declared = manifest.get("package", {}).get("readme")
    if declared is False:
        return None
    if declared is None:
        # Cargo's auto-detection order when the field is absent.
        for candidate in ("README.md", "README.txt", "README"):
            if (package_dir / candidate).is_file():
                return package_dir / candidate
        return None
    return package_dir / declared


def first_paragraph(text: str) -> str:
    """Skips leading headings and blank lines, then joins the following lines up to
    the next blank line with spaces."""
    lines = text.splitlines()
    index = 0
    while index < len(lines) and (lines[index].startswith("#") or not lines[index].strip()):
        index += 1
    paragraph = []
    while index < len(lines) and lines[index].strip():
        paragraph.append(lines[index].strip())
        index += 1
    return " ".join(paragraph)


def check_readmes(members: dict) -> None:
    openings: dict[str, tuple[str, Path]] = {}
    for name in sorted(PUBLISHED):
        package = members[name]
        with Path(package["manifest_path"]).open("rb") as source:
            manifest = tomllib.load(source)
        path = readme_path(package, manifest)
        if path is None or not path.is_file():
            fail(f"{name} is published but has no README to show on crates.io")
        text = path.read_text(encoding="utf-8")
        display = path.resolve().relative_to(ROOT)
        for line_number, line in enumerate(text.splitlines(), start=1):
            hits = [token for token in README_INTERNAL_TOKENS if token in line]
            if hits:
                fail(f"{name}: {display}:{line_number} names workspace-only {', '.join(repr(hit) for hit in hits)}")
        opening = first_paragraph(text)
        if not opening:
            fail(f"{name}: {display} has no opening paragraph after its title")
        if opening in openings:
            other, other_path = openings[opening]
            fail(f"{name}: {display} opens with the same paragraph as {other} ({other_path.resolve().relative_to(ROOT)})")
        openings[opening] = (name, path)


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
    # unpublished. This check reports it before the release.
    for name in sorted(PUBLISHED):
        for dependency in members[name]["dependencies"]:
            if dependency.get("kind") == "dev":
                continue
            target = dependency["name"]
            if target in members and target not in PUBLISHED:
                fail(f"{name} is published but depends on unpublished workspace package {target}")

    check_readmes(members)

    print(
        "PASS: workspace publication policy: "
        f"{len(PUBLISHED)} published, {len(members) - len(PUBLISHED)} private, "
        "all declared directly, READMEs distinct and free of workspace paths"
    )


if __name__ == "__main__":
    main()
