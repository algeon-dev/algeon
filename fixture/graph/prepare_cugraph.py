#!/usr/bin/env python3
"""Prepare local cuGraph parity fixtures from the sibling cuGraph checkout.

The source of truth for cuGraph dataset bytes is the cuGraph repository, not
this repository. This script copies the small upstream datasets and reference
outputs needed by the datafusion-nexus parity harness into the versioned graph
fixture root, then writes a manifest with SHA-256 checksums for the copied
bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CUGRAPH_REPO = REPO_ROOT.parent / "cugraph-nexus" / "cugraph"
DEFAULT_DATA_ROOT = REPO_ROOT / "fixture" / "graph" / "cugraph"


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    relpath: str


@dataclass(frozen=True)
class ReferenceSpec:
    ref_id: str
    relpath: str
    dataset_id: str


DATASETS = [
    DatasetSpec("karate", "karate.csv"),
    DatasetSpec("dolphins", "dolphins.csv"),
    DatasetSpec("polbooks", "polbooks.csv"),
    DatasetSpec("toy_graph", "toy_graph.csv"),
]

REFERENCES = [
    ReferenceSpec("katz/karate", "ref/katz/karate.csv", "karate"),
    ReferenceSpec("katz/dolphins", "ref/katz/dolphins.csv", "dolphins"),
    ReferenceSpec("katz/polbooks", "ref/katz/polbooks.csv", "polbooks"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy cuGraph parity datasets from ../cugraph-nexus/cugraph and generate fixture/graph/cugraph/manifest.json.",
    )
    parser.add_argument(
        "--cugraph-repo",
        type=Path,
        default=DEFAULT_CUGRAPH_REPO,
        help="Path to the cuGraph source checkout. Defaults to ../cugraph-nexus/cugraph.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.getenv("NEXUS_CUGRAPH_DATA_ROOT", DEFAULT_DATA_ROOT)),
        help="Local fixture root. Defaults to NEXUS_CUGRAPH_DATA_ROOT or fixture/graph/cugraph.",
    )
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="After copying datasets, regenerate all local oracle JSON with refresh_cugraph_oracle.",
    )
    parser.add_argument(
        "--oracle-check",
        action="store_true",
        help="After copying datasets, check existing local oracle JSON against Python cuGraph without writing.",
    )
    parser.add_argument(
        "--force-oracle",
        action="store_true",
        help="Overwrite drifted local oracle JSON when --oracle is used.",
    )
    parser.add_argument(
        "--release",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use cargo run --release for oracle regeneration. Enabled by default.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source_root: Path, out_root: Path, relpath: str) -> Path:
    source = source_root / "datasets" / relpath
    if not source.is_file():
        raise SystemExit(f"cuGraph source dataset missing: {source}")
    target = out_root / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def dataset_entry(path: Path, spec: DatasetSpec) -> dict[str, Any]:
    vertices: set[int] = set()
    edges = 0
    weighted = False
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 3:
                raise SystemExit(f"{path} line {edges + 1} is not a 'dst src weight' triple: {stripped!r}")
            dst = int(parts[0])
            src = int(parts[1])
            weight = float(parts[2])
            vertices.add(dst)
            vertices.add(src)
            edges += 1
            if weight != 1.0:
                weighted = True
    return {
        "id": spec.dataset_id,
        "path": spec.relpath,
        "format": "space_separated_edges",
        "edge_columns": ["dst", "src", "weight"],
        "vertices": len(vertices),
        "edges": edges,
        "directed": False,
        "weighted": weighted,
        "sha256": sha256(path),
    }


def reference_entry(path: Path, spec: ReferenceSpec) -> dict[str, Any]:
    return {
        "id": spec.ref_id,
        "path": spec.relpath,
        "format": "vertex_rank",
        "columns": ["vertex"],
        "dataset": spec.dataset_id,
        "sha256": sha256(path),
    }


def write_manifest(cugraph_repo: Path, out_root: Path) -> None:
    datasets = []
    for spec in DATASETS:
        path = copy_file(cugraph_repo, out_root, spec.relpath)
        datasets.append(dataset_entry(path, spec))

    references = []
    for spec in REFERENCES:
        path = copy_file(cugraph_repo, out_root, spec.relpath)
        references.append(reference_entry(path, spec))

    manifest = {
        "schema_version": 1,
        "description": (
            "Local cuGraph datasets and reference outputs prepared from the sibling cuGraph checkout. "
            "Do not commit these generated bytes."
        ),
        "source_repo": "rapidsai/cugraph via local sibling checkout",
        "source_repo_path": str(cugraph_repo),
        "license": "Apache-2.0",
        "datasets": datasets,
        "references": references,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def run_oracle(args: argparse.Namespace, out_root: Path) -> None:
    command = ["cargo", "run"]
    if args.release:
        command.append("--release")
    command.extend([
        "-p",
        "nexus-tools",
        "--features",
        "cugraph-oracle",
        "--bin",
        "refresh_cugraph_oracle",
        "--",
    ])
    command.extend(["--all", "--workspace-root", str(REPO_ROOT)])
    if args.oracle_check:
        command.append("--check")
    if args.force_oracle:
        command.append("--force")
    print("running:", " ".join(command), file=sys.stderr)
    env = os.environ.copy()
    env["NEXUS_CUGRAPH_DATA_ROOT"] = str(out_root)
    env.setdefault("NEXUS_CUGRAPH_PYTHON", sys.executable)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def main() -> int:
    args = parse_args()
    cugraph_repo = args.cugraph_repo.resolve()
    out_root = args.out_dir.resolve()
    if args.oracle and args.oracle_check:
        raise SystemExit("--oracle and --oracle-check are mutually exclusive")
    if args.oracle_check and args.force_oracle:
        raise SystemExit("--oracle-check and --force-oracle cannot be combined")
    if not (cugraph_repo / "datasets").is_dir():
        raise SystemExit(f"cuGraph datasets directory missing: {cugraph_repo / 'datasets'}")
    write_manifest(cugraph_repo, out_root)
    print(f"prepared cuGraph fixture data under {out_root}", file=sys.stderr)
    if args.oracle or args.oracle_check:
        run_oracle(args, out_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
