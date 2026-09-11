#!/usr/bin/env bash
# Fail closed if the DataFusion-free Iceberg leaf reaches an adapter, engine,
# GPU, or DataFusion dependency through any enabled feature edge.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "${ROOT}"

metadata=$(mktemp "${TMPDIR:-/tmp}/iceberg-catalog-metadata.XXXXXX.json")
trap 'rm -f "${metadata}"' EXIT
cargo metadata --locked --all-features --format-version 1 >"${metadata}"

python3 - "${metadata}" <<'PY'
import json
import sys
from collections import deque

metadata_path = sys.argv[1]
with open(metadata_path, encoding="utf-8") as source:
    metadata = json.load(source)

packages = {package["id"]: package["name"] for package in metadata["packages"]}
leaf_ids = [package_id for package_id, name in packages.items() if name == "iceberg-catalog"]
if len(leaf_ids) != 1:
    raise SystemExit(f"ERROR: expected exactly one iceberg-catalog package, found {len(leaf_ids)}")

nodes = {node["id"]: node for node in metadata.get("resolve", {}).get("nodes", [])}
leaf_id = leaf_ids[0]
if leaf_id not in nodes:
    raise SystemExit("ERROR: Cargo metadata did not resolve iceberg-catalog")

banned = {"algeon-datafusion", "query-engine", "datafusion", "cudf", "cudf-sys", "cugraph", "cugraph-sys"}
parents = {leaf_id: None}
queue = deque([leaf_id])
while queue:
    current = queue.popleft()
    for dependency in nodes[current].get("deps", []):
        child = dependency["pkg"]
        if child not in parents:
            parents[child] = current
            queue.append(child)

for package_id in sorted(parents, key=lambda item: packages.get(item, item)):
    if package_id == leaf_id or packages.get(package_id) not in banned:
        continue
    path = []
    cursor = package_id
    while cursor is not None:
        path.append(packages.get(cursor, cursor))
        cursor = parents[cursor]
    path.reverse()
    raise SystemExit("FAIL: iceberg-catalog reaches prohibited dependency: " + " -> ".join(path))

print("OK: iceberg-catalog has no DataFusion, Algeon engine, or GPU dependency path.")
PY
