#!/usr/bin/env python3
"""Generate Graph500-style R-MAT edge-list Parquet fixtures with cuGraph.

The default is intentionally small so the generator can be smoke-tested on
developer machines. Set GRAPH500_RMAT_SCALE=25, or pass --scale 25, for the
Graph500 Scale 25 vertex domain. Full Scale 25 with edgefactor 16 requests
536,870,912 edges and is too large to treat as a casual test on a 16 GiB GPU.

Output layout:
  fixture/data/graph500/rmat_scale<scale>_ef<edgefactor>/
    metadata.json
    parquet/
      edges_00000.parquet
      ...

The Parquet schema is:
  edge_id: int64
  src:     int64
  dst:     int64
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCALE = 12
DEFAULT_EDGEFACTOR = 16
DEFAULT_SEED = 20260601
GRAPH500_A = 0.57
GRAPH500_B = 0.19
GRAPH500_C = 0.19


@dataclass(frozen=True)
class ShardResult:
    path: str
    seed: int
    requested_edges: int
    written_edges: int
    elapsed_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Graph500-style R-MAT Parquet edge-list fixture using cugraph.generators.rmat.",
    )
    parser.add_argument("--scale", type=int, default=int(os.getenv("GRAPH500_RMAT_SCALE", DEFAULT_SCALE)))
    parser.add_argument(
        "--edgefactor",
        type=int,
        default=int(os.getenv("GRAPH500_RMAT_EDGEFACTOR", DEFAULT_EDGEFACTOR)),
        help="Edges per vertex. Graph500 commonly uses 16.",
    )
    parser.add_argument("--seed", type=int, default=int(os.getenv("GRAPH500_RMAT_SEED", DEFAULT_SEED)))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Fixture root. Defaults to fixture/data/graph500/rmat_scale<scale>_ef<edgefactor>.",
    )
    parser.add_argument(
        "--edges-per-shard",
        type=int,
        default=int(os.getenv("GRAPH500_RMAT_EDGES_PER_SHARD", "0")),
        help="Generate multiple Parquet shards with at most this many requested edges per shard. 0 means one shard.",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=int(os.getenv("GRAPH500_RMAT_MAX_EDGES", "0")),
        help="Cap total requested edges for smoke/limit probing. 0 means scale * edgefactor exactly.",
    )
    parser.add_argument(
        "--no-scramble",
        action="store_true",
        help="Disable Graph500-style vertex-id scrambling. Scrambling is enabled by default.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=os.getenv("GRAPH500_RMAT_OVERWRITE", "0") == "1",
        help="Replace an existing fixture root.",
    )
    parser.add_argument(
        "--source-vertex",
        type=int,
        default=None,
        help="BFS source vertex written to metadata. Defaults to the first generated src.",
    )
    return parser.parse_args()


def load_rapids_modules() -> tuple[Any, Any, Any]:
    try:
        import cudf
        import cugraph
        from cugraph.generators import rmat
    except Exception as exc:
        print(
            "RAPIDS Python stack unavailable. Run with the repo .venv or set "
            f"CUDF_CUGRAPH_PYTHON to a Python that imports cudf/cugraph: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(77) from exc
    return cudf, cugraph, rmat


def default_out_dir(scale: int, edgefactor: int, max_edges: int) -> Path:
    full_edges = (1 << scale) * edgefactor
    suffix = f"_edges{max_edges}" if 0 < max_edges < full_edges else ""
    return REPO_ROOT / "fixture" / "data" / "graph500" / f"rmat_scale{scale}_ef{edgefactor}{suffix}"


def requested_edge_count(scale: int, edgefactor: int, max_edges: int) -> int:
    count = (1 << scale) * edgefactor
    if max_edges > 0:
        count = min(count, max_edges)
    return count


def shard_sizes(total_edges: int, edges_per_shard: int) -> list[int]:
    if edges_per_shard <= 0 or total_edges <= edges_per_shard:
        return [total_edges]
    sizes = []
    remaining = total_edges
    while remaining > 0:
        size = min(edges_per_shard, remaining)
        sizes.append(size)
        remaining -= size
    return sizes


def prepare_output(root: Path, overwrite: bool) -> Path:
    parquet_dir = root / "parquet"
    if root.exists():
        if not overwrite:
            raise SystemExit(f"output root already exists: {root} (pass --overwrite to replace it)")
        shutil.rmtree(root)
    parquet_dir.mkdir(parents=True, exist_ok=True)
    return parquet_dir


def normalize_edges(df: Any, cudf_mod: Any, edge_id_offset: int) -> tuple[Any, int | None]:
    rename_map = {}
    if "source" in df.columns and "src" not in df.columns:
        rename_map["source"] = "src"
    if "destination" in df.columns and "dst" not in df.columns:
        rename_map["destination"] = "dst"
    if rename_map:
        df = df.rename(columns=rename_map)
    missing = {"src", "dst"} - set(df.columns)
    if missing:
        raise RuntimeError(f"rmat returned columns {list(df.columns)!r}; missing {sorted(missing)!r}")

    df = df[["src", "dst"]]
    df["src"] = df["src"].astype("int64")
    df["dst"] = df["dst"].astype("int64")
    df.insert(0, "edge_id", cudf_mod.Series(range(edge_id_offset, edge_id_offset + len(df)), dtype="int64"))
    first_src = None
    if len(df) > 0:
        first_src = int(df["src"].head(1).to_pandas().iloc[0])
    return df, first_src


def write_fixture(args: argparse.Namespace) -> dict[str, Any]:
    cudf_mod, cugraph_mod, rmat = load_rapids_modules()
    root = args.out_dir or default_out_dir(args.scale, args.edgefactor, args.max_edges)
    root = root.resolve()
    parquet_dir = prepare_output(root, args.overwrite)
    total_edges = requested_edge_count(args.scale, args.edgefactor, args.max_edges)
    sizes = shard_sizes(total_edges, args.edges_per_shard)

    shards: list[ShardResult] = []
    edge_id_offset = 0
    source_vertex = args.source_vertex
    started = time.perf_counter()
    for shard_index, shard_edges in enumerate(sizes):
        shard_seed = args.seed + shard_index
        shard_started = time.perf_counter()
        df = rmat(
            args.scale,
            shard_edges,
            GRAPH500_A,
            GRAPH500_B,
            GRAPH500_C,
            seed=shard_seed,
            clip_and_flip=False,
            scramble_vertex_ids=not args.no_scramble,
            create_using=None,
            mg=False,
        )
        df, first_src = normalize_edges(df, cudf_mod, edge_id_offset)
        if source_vertex is None and first_src is not None:
            source_vertex = first_src
        path = parquet_dir / f"edges_{shard_index:05d}.parquet"
        df.to_parquet(str(path), index=False)
        written_edges = int(len(df))
        edge_id_offset += written_edges
        shards.append(
            ShardResult(
                path=str(path.relative_to(root)),
                seed=shard_seed,
                requested_edges=shard_edges,
                written_edges=written_edges,
                elapsed_ms=(time.perf_counter() - shard_started) * 1000.0,
            )
        )
        print(
            f"[graph500-rmat] shard={shard_index + 1}/{len(sizes)} "
            f"edges={written_edges} path={path}",
            flush=True,
        )

    if source_vertex is None:
        source_vertex = 0
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "generator": "fixture/scripts/gen_graph500_rmat.py",
        "rmat_api": "cugraph.generators.rmat",
        "versions": {
            "cudf": getattr(cudf_mod, "__version__", "unknown"),
            "cugraph": getattr(cugraph_mod, "__version__", "unknown"),
        },
        "scale": args.scale,
        "edgefactor": args.edgefactor,
        "num_vertices": 1 << args.scale,
        "num_edges_requested": total_edges,
        "num_edges_written": edge_id_offset,
        "a": GRAPH500_A,
        "b": GRAPH500_B,
        "c": GRAPH500_C,
        "seed": args.seed,
        "scramble_vertex_ids": not args.no_scramble,
        "source_vertex": source_vertex,
        "parquet_dir": "parquet",
        "columns": ["edge_id", "src", "dst"],
        "shards": [asdict(shard) for shard in shards],
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata | {"root": str(root)}


def main() -> int:
    args = parse_args()
    if args.scale < 1:
        raise SystemExit("--scale must be >= 1")
    if args.edgefactor < 1:
        raise SystemExit("--edgefactor must be >= 1")
    metadata = write_fixture(args)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
