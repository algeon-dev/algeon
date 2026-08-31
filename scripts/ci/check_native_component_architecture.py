#!/usr/bin/env python3

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / ".config/policies/native-component-surfaces.toml"
COMPONENTS = {
    "cudf-nexus": ROOT / "components/cudf/cudf-nexus",
    "cudf-nexus-sys": ROOT / "components/cudf/cudf-nexus-sys",
    "cugraph-nexus": ROOT / "components/cugraph/cugraph-nexus",
    "cugraph-nexus-sys": ROOT / "components/cugraph/cugraph-nexus-sys",
}
SNAPSHOT_CRATES = {"cudf-nexus": "cudf", "cugraph-nexus": "cugraph"}
ALLOWED_BASES = {"nexus_production", "cross_component", "safety_abi_semantic"}


def fail(message: str) -> None:
    raise ValueError(message)


def manifest_contracts() -> None:
    def dependency_names(table: dict[str, object]) -> list[str]:
        names: list[str] = []
        for key, value in table.items():
            if (key == "dependencies" or key.endswith("-dependencies")) and isinstance(value, dict):
                names.extend(value)
            elif isinstance(value, dict):
                names.extend(dependency_names(value))
        return names

    for crate, directory in COMPONENTS.items():
        manifest_path = directory / "Cargo.toml"
        manifest = tomllib.loads(manifest_path.read_text())
        if manifest.get("package", {}).get("publish") is not False:
            fail(f"{manifest_path.relative_to(ROOT)} must set package.publish = false")
        for dependency in dependency_names(manifest):
            if dependency == "datafusion" or dependency.startswith("datafusion-"):
                fail(f"{manifest_path.relative_to(ROOT)} must not depend on {dependency}")


def source_contracts() -> None:
    forbidden = {
        ROOT / "components/cudf/cudf-nexus/src": [
            r"\bpub\s+mod\s+query\b",
            r"\bQueryAggregateSpec\b",
            r"\bQuerySortSpec\b",
            r"\bNamedExpr\b",
            r"\bMeltOptions\b",
            r"\bLookupTable\b",
            r"\bCsv(?:Table)?(?:Read|Write)Options\b",
            r"\bpub\s+mod\s+csv\b",
        ],
        ROOT / "components/cugraph/cugraph-nexus/src": [
            r"\bgraph_from_dataframe\b",
            r"\bDataFrameGraphOptions\b",
            r"\bgraph_from_series\b",
            r"\bSeriesGraphOptions\b",
            r"\bgraph_from_parquet\b",
            r"\bParquetGraphOptions\b",
            r"\bLoadedGraph\b",
            r"\bgraph_from_table_view\b",
            r"\bgraph_from_borrowed(?:_columns|_table_view)\b",
            r"\bBorrowedInputOrderingPlan\b",
        ],
        ROOT / "crates/nexus-query-engine/src": [
            r"\b(?:cudf_sys|cugraph_sys)::",
            r"\brun_with_cugraph_graph_edge_ids\b",
            r"\bbuild_graph_from_edge_id_columns\b",
            r"\bgraph_device_(?:slice|view)\b",
        ],
    }
    for directory, patterns in forbidden.items():
        compiled = [(pattern, re.compile(pattern)) for pattern in patterns]
        for path in directory.rglob("*.rs"):
            text = path.read_text()
            for pattern, regex in compiled:
                if regex.search(text):
                    fail(f"{path.relative_to(ROOT)} reintroduces forbidden architecture token {pattern}")


def surface_inventory(snapshot_dir: Path) -> None:
    policy = tomllib.loads(POLICY.read_text())
    if policy.get("version") != 1:
        fail(f"{POLICY.relative_to(ROOT)} must declare version = 1")

    inventory: dict[tuple[str, str], dict[str, str]] = {}
    for entry in policy.get("surface", []):
        crate = entry.get("crate")
        namespace = entry.get("namespace")
        key = (crate, namespace)
        if key in inventory:
            fail(f"duplicate surface inventory entry: {crate}:{namespace}")
        if entry.get("decision") != "keep":
            fail(f"settled surface inventory entries must be keep: {crate}:{namespace}")
        if entry.get("basis") not in ALLOWED_BASES:
            fail(f"invalid surface basis for {crate}:{namespace}")
        caller = ROOT / entry.get("caller", "")
        if not caller.exists():
            fail(f"surface caller evidence does not exist for {crate}:{namespace}: {caller}")
        inventory[key] = entry

    for crate, rust_name in SNAPSHOT_CRATES.items():
        snapshot = snapshot_dir / f"{crate}.txt"
        if not snapshot.is_file():
            fail(f"missing component public API snapshot: {snapshot}")
        seen: set[str] = set()
        path_pattern = re.compile(rf"\b{rust_name}(?:::(?P<namespace>[A-Za-z_][A-Za-z0-9_]*))?")
        for line_number, line in enumerate(snapshot.read_text().splitlines(), 1):
            match = path_pattern.search(line)
            if match is None:
                fail(f"{snapshot}:{line_number} has no {rust_name} API path")
            candidate = match.group("namespace") or "root"
            if line.startswith(f"pub mod {rust_name}::") or (crate, candidate) in inventory:
                namespace = candidate
            else:
                namespace = "root"
            if (crate, namespace) not in inventory:
                fail(f"{snapshot}:{line_number} has unclassified namespace {crate}:{namespace}")
            seen.add(namespace)
        declared = {namespace for declared_crate, namespace in inventory if declared_crate == crate}
        missing = sorted(declared - seen)
        if missing:
            fail(f"surface inventory has empty namespaces for {crate}: {', '.join(missing)}")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_native_component_architecture.py PUBLIC_API_SNAPSHOT_DIR", file=sys.stderr)
        return 2
    try:
        manifest_contracts()
        source_contracts()
        surface_inventory(Path(sys.argv[1]))
    except (OSError, tomllib.TOMLDecodeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("native component architecture contracts OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
