#!/usr/bin/env bash
# Fail closed if a publishable package sharing the workspace root directory
# omits `include`, which otherwise sweeps the whole worktree into its archive.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "${ROOT}"

metadata=$(mktemp "${TMPDIR:-/tmp}/nexus-workspace-root-include-metadata.XXXXXX.json")
trap 'rm -f "${metadata}"' EXIT
cargo metadata --locked --all-features --format-version 1 >"${metadata}"

python3 - "${metadata}" <<'PY'
import json
import os
import sys
import tomllib

metadata_path = sys.argv[1]
with open(metadata_path, encoding="utf-8") as source:
    metadata = json.load(source)

workspace_root = metadata["workspace_root"]

for package in metadata["packages"]:
    if package["publish"] is not None:
        continue
    manifest_path = package["manifest_path"]
    if os.path.dirname(manifest_path) != workspace_root:
        continue
    # `cargo metadata` does not surface a manifest's `include`/`exclude` keys,
    # so read the manifest directly for the one fact it omits.
    with open(manifest_path, "rb") as manifest_file:
        manifest = tomllib.load(manifest_file)
    include = manifest.get("package", {}).get("include")
    if include:
        continue
    raise SystemExit(
        "ERROR: publishable package "
        + package["name"]
        + " shares its manifest directory with the workspace root ("
        + workspace_root
        + ") but declares no `include`. Without one, `cargo package` sweeps in the"
        + " entire worktree — including the vendored RAPIDS submodules under"
        + " components/, website/, and fixture/ — instead of just the crate's"
        + " sources."
    )

print("OK: every publishable package at the workspace root declares `include`.")
PY
