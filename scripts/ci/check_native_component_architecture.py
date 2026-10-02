#!/usr/bin/env python3

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / ".config/policies/native-component-surfaces.toml"
COMPONENTS = {
    "algeon-cudf": ROOT / "components/cudf/algeon-cudf",
    "algeon-cudf-sys": ROOT / "components/cudf/algeon-cudf-sys",
    "algeon-cugraph": ROOT / "components/cugraph/algeon-cugraph",
    "algeon-cugraph-sys": ROOT / "components/cugraph/algeon-cugraph-sys",
}
# Package name -> `[lib] name`. Snapshots are filed under the package name,
# while the `pub mod` lines inside them carry the bare library identity the
# `algeon-` prefixed packages deliberately keep.
SNAPSHOT_CRATES = {"algeon-cudf": "cudf", "algeon-cugraph": "cugraph"}


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

    for directory in COMPONENTS.values():
        manifest_path = directory / "Cargo.toml"
        manifest = tomllib.loads(manifest_path.read_text())
        for dependency in dependency_names(manifest):
            if dependency == "datafusion" or dependency.startswith("datafusion-"):
                fail(f"{manifest_path.relative_to(ROOT)} must not depend on {dependency}")


def source_contracts() -> None:
    forbidden = {
        ROOT / "components/cudf/algeon-cudf/src": [
            r"\bpub\s+mod\s+query\b",
            r"\bQueryAggregateSpec\b",
            r"\bQuerySortSpec\b",
            r"\bNamedExpr\b",
            r"\bMeltOptions\b",
            r"\bLookupTable\b",
            r"\bCsv(?:Table)?(?:Read|Write)Options\b",
            r"\bpub\s+mod\s+csv\b",
        ],
        ROOT / "components/cugraph/algeon-cugraph/src": [
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
        ROOT / "crates/algeon-query-engine/src": [
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


def surface_inventory(snapshot_dir: Path, policy_path: Path = POLICY) -> None:
    policy = tomllib.loads(policy_path.read_text())

    surfaces = policy.get("surfaces")
    if not isinstance(surfaces, dict):
        fail(f"{policy_path} must define a [surfaces] table")

    unknown_crates = set(surfaces.keys()) - set(SNAPSHOT_CRATES.keys())
    if unknown_crates:
        fail(f"unknown crates in surface policy: {', '.join(sorted(unknown_crates))}")

    missing_crates = set(SNAPSHOT_CRATES.keys()) - set(surfaces.keys())
    if missing_crates:
        fail(f"missing crates in surface policy: {', '.join(sorted(missing_crates))}")

    mod_pattern = re.compile(r"^pub mod ([A-Za-z_][A-Za-z0-9_]*)::([A-Za-z_][A-Za-z0-9_]*)")

    for crate, rust_name in SNAPSHOT_CRATES.items():
        declared = surfaces.get(crate)
        if not isinstance(declared, list) or not all(isinstance(x, str) for x in declared):
            fail(f"{crate} namespaces must be a list of strings")

        seen_declared = set()
        for ns in declared:
            if ns in seen_declared:
                fail(f"duplicate approved namespace in {crate}: {ns}")
            seen_declared.add(ns)

        snapshot = snapshot_dir / f"{crate}.txt"
        if not snapshot.is_file():
            fail(f"missing component public API snapshot: {snapshot}")

        actual: set[str] = set()
        for line in snapshot.read_text().splitlines():
            match = mod_pattern.match(line)
            if match and match.group(1) == rust_name:
                actual.add(match.group(2))

        unapproved = sorted(actual - seen_declared)
        if unapproved:
            fail(f"unapproved top-level namespaces for {crate}: {', '.join(unapproved)}")

        stale = sorted(seen_declared - actual)
        if stale:
            fail(f"stale approved top-level namespaces for {crate}: {', '.join(stale)}")


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print("usage: check_native_component_architecture.py PUBLIC_API_SNAPSHOT_DIR [POLICY_FILE]", file=sys.stderr)
        return 2
    try:
        manifest_contracts()
        source_contracts()
        policy_path = Path(sys.argv[2]) if len(sys.argv) == 3 else POLICY
        surface_inventory(Path(sys.argv[1]), policy_path)
    except (OSError, tomllib.TOMLDecodeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("native component architecture contracts OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
