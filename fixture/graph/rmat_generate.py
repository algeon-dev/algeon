#!/usr/bin/env python3
"""Deterministic synthetic R-MAT graph fixture generator.

Generates a directed R-MAT (recursive matrix) edge list as Parquet, sized by
``--scale`` (vertex count = ``2**scale``) and ``--edge-factor`` (edge count =
``edge_factor * 2**scale``). Edges are drawn quadrant-by-quadrant per the
classic R-MAT recursion: each of ``scale`` bit levels assigns one bit of
``src`` and one bit of ``dst`` by sampling one of four quadrants (a, b, c, d)
with skew-dependent probabilities (Graph500 ``skewed`` convention, or a flat
``uniform`` quadrant split). Self-loops and duplicate ``(src, dst)`` pairs are
kept as-is; their counts are recorded in the manifest instead of being
filtered out.

Output layout under ``<out-root>/rmat_s<scale>_e<edge_factor>_<skew>/``:

- ``edges.parquet``      ``src`` int64, ``dst`` int64, ``weight`` float64
- ``_generation.json``   generation manifest (command line, sizes, sha256,
                          degree statistics)

Generation is a single ``numpy.random.default_rng(seed)`` stream consumed in
fixed-size edge chunks (``--chunk-edges``, default 4,000,000): each chunk
draws its R-MAT bit levels then its weights, and is written as exactly one
Parquet row group. This keeps peak host memory low regardless of total edge
count and makes output byte-identical for identical arguments, since the
chunk boundaries (and therefore the RNG draw grouping) are themselves an
input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


EDGE_SCHEMA = pa.schema(
    [
        pa.field("src", pa.int64(), nullable=False),
        pa.field("dst", pa.int64(), nullable=False),
        pa.field("weight", pa.float64(), nullable=False),
    ]
)

# Quadrant probabilities (a, b, c, d) for src-bit/dst-bit = (0,0)/(0,1)/(1,0)/(1,1).
# "skewed" is the Graph500 R-MAT convention; "uniform" is Erdos-Renyi-like (no degree skew).
SKEW_PROBABILITIES = {
    "skewed": (0.57, 0.19, 0.19, 0.05),
    "uniform": (0.25, 0.25, 0.25, 0.25),
}

DEFAULT_CHUNK_EDGES = 4_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a deterministic synthetic R-MAT graph fixture as Parquet.")
    parser.add_argument("--scale", type=int, required=True, help="Vertex count is 2**scale.")
    parser.add_argument("--edge-factor", type=int, required=True, help="Edge count is edge_factor * 2**scale.")
    parser.add_argument("--skew", choices=sorted(SKEW_PROBABILITIES), required=True)
    parser.add_argument("--seed", type=int, required=True, help="numpy default_rng seed.")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--chunk-edges", type=int, default=DEFAULT_CHUNK_EDGES, help="Edges generated/written per Parquet row group.")
    parser.add_argument("--force", action="store_true", help="Replace an existing output directory.")
    return parser.parse_args()


def require_positive(name: str, value: int) -> None:
    if value < 1:
        raise SystemExit(f"{name} must be >= 1")


def output_dir_name(scale: int, edge_factor: int, skew: str) -> str:
    return f"rmat_s{scale}_e{edge_factor}_{skew}"


def rmat_bits(
    rng: np.random.Generator, n_edges: int, scale: int, probs: tuple[float, float, float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized R-MAT recursion: `scale` quadrant draws per edge, one per bit level."""
    a, b, c, _d = probs
    ab = a + b
    abc = ab + c
    src = np.zeros(n_edges, dtype=np.uint64)
    dst = np.zeros(n_edges, dtype=np.uint64)
    for _level in range(scale):
        u = rng.random(n_edges)
        src_bit = u >= ab  # quadrants c, d
        dst_bit = ((u >= a) & (u < ab)) | (u >= abc)  # quadrants b, d
        src = (src << np.uint64(1)) | src_bit.astype(np.uint64)
        dst = (dst << np.uint64(1)) | dst_bit.astype(np.uint64)
    return src, dst


def write_edges(
    edges_path: Path,
    *,
    seed: int,
    scale: int,
    probs: tuple[float, float, float, float],
    edge_count: int,
    chunk_edges: int,
) -> tuple[int, float]:
    rng = np.random.default_rng(seed)
    writer = pq.ParquetWriter(edges_path, EDGE_SCHEMA, compression="snappy")
    started = time.perf_counter()
    remaining = edge_count
    written = 0
    row_group_count = 0
    try:
        while remaining > 0:
            n = min(chunk_edges, remaining)
            src, dst = rmat_bits(rng, n, scale, probs)
            weight = rng.uniform(0.5, 1.5, size=n)
            table = pa.table(
                {
                    "src": pa.array(src.astype(np.int64), type=pa.int64()),
                    "dst": pa.array(dst.astype(np.int64), type=pa.int64()),
                    "weight": pa.array(weight, type=pa.float64()),
                },
                schema=EDGE_SCHEMA,
            )
            writer.write_table(table)
            row_group_count += 1
            written += n
            remaining -= n
            print(f"[rmat-generate] wrote chunk rows={n:,} written={written:,}/{edge_count:,}", file=sys.stderr, flush=True)
    finally:
        writer.close()
    return row_group_count, time.perf_counter() - started


def compute_degree_stats(edges_path: Path, *, vertex_count: int, edge_count: int, chunk_edges: int) -> tuple[dict, float]:
    """Streaming pass over the written Parquet: bincount degree accumulation, no per-edge Python objects.

    Duplicate-pair detection needs global (src, dst) uniqueness, so this encodes each pair as one
    int64 key (`src * vertex_count + dst`, safe from overflow at the scales this generator targets)
    into a preallocated numpy array and sorts it -- vectorized, not a Python dict of edges.
    """
    started = time.perf_counter()
    out_degree = np.zeros(vertex_count, dtype=np.int64)
    in_degree = np.zeros(vertex_count, dtype=np.int64)
    self_loops = 0
    pair_keys = np.empty(edge_count, dtype=np.int64)
    offset = 0
    parquet_file = pq.ParquetFile(edges_path)
    for batch in parquet_file.iter_batches(batch_size=chunk_edges, columns=["src", "dst"]):
        src = batch.column("src").to_numpy(zero_copy_only=False)
        dst = batch.column("dst").to_numpy(zero_copy_only=False)
        n = len(src)
        out_degree += np.bincount(src, minlength=vertex_count).astype(np.int64)
        in_degree += np.bincount(dst, minlength=vertex_count).astype(np.int64)
        self_loops += int(np.count_nonzero(src == dst))
        pair_keys[offset : offset + n] = src.astype(np.int64) * vertex_count + dst.astype(np.int64)
        offset += n
    assert offset == edge_count, f"read {offset} edges, expected {edge_count}"

    pair_keys.sort()
    unique_count = 0 if edge_count == 0 else 1 + int(np.count_nonzero(np.diff(pair_keys) != 0))
    duplicate_pair_count = edge_count - unique_count

    p50, p99, p999 = (np.percentile(out_degree, [50, 99, 99.9]).tolist() if vertex_count > 0 else (0.0, 0.0, 0.0))
    stats = {
        "max_out_degree": int(out_degree.max()) if vertex_count else 0,
        "max_in_degree": int(in_degree.max()) if vertex_count else 0,
        "out_degree_p50": p50,
        "out_degree_p99": p99,
        "out_degree_p999": p999,
        "self_loop_count": self_loops,
        "duplicate_pair_count": duplicate_pair_count,
        "zero_out_degree_vertex_count": int(np.count_nonzero(out_degree == 0)),
    }
    return stats, time.perf_counter() - started


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    require_positive("--scale", args.scale)
    require_positive("--edge-factor", args.edge_factor)
    require_positive("--chunk-edges", args.chunk_edges)
    if args.seed < 0:
        raise SystemExit("--seed must be >= 0")

    probs = SKEW_PROBABILITIES[args.skew]
    vertex_count = 1 << args.scale
    edge_count = args.edge_factor * vertex_count
    # Duplicate-pair encoding packs (src, dst) into one int64 key; guard the assumption this relies on.
    if vertex_count * vertex_count > np.iinfo(np.int64).max:
        raise SystemExit(f"--scale {args.scale} is too large for int64 pair-key encoding")

    out_dir = args.out_root / output_dir_name(args.scale, args.edge_factor, args.skew)
    if out_dir.exists() and not args.force:
        raise SystemExit(f"{out_dir} already exists; pass --force to overwrite")

    args.out_root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.tmp.", dir=args.out_root))
    try:
        edges_path = temp_dir / "edges.parquet"
        row_group_count, generation_seconds = write_edges(
            edges_path,
            seed=args.seed,
            scale=args.scale,
            probs=probs,
            edge_count=edge_count,
            chunk_edges=args.chunk_edges,
        )
        stats, stats_seconds = compute_degree_stats(
            edges_path, vertex_count=vertex_count, edge_count=edge_count, chunk_edges=args.chunk_edges
        )
        file_size = edges_path.stat().st_size
        digest = sha256_file(edges_path)

        manifest = {
            "command": shlex.join(sys.argv),
            "scale": args.scale,
            "edge_factor": args.edge_factor,
            "vertex_count": vertex_count,
            "edge_count": edge_count,
            "skew": {"label": args.skew, "a": probs[0], "b": probs[1], "c": probs[2], "d": probs[3]},
            "seed": args.seed,
            "chunk_edges": args.chunk_edges,
            "parquet_file_size_bytes": file_size,
            "parquet_sha256": digest,
            "row_group_count": row_group_count,
            "degree_stats": stats,
            "generation_seconds": generation_seconds,
            "stats_seconds": stats_seconds,
        }
        (temp_dir / "_generation.json").write_text(json.dumps(manifest, indent=2) + "\n")

        if out_dir.exists():
            shutil.rmtree(out_dir)
        os.replace(temp_dir, out_dir)
        temp_dir = None
    finally:
        if temp_dir is not None and Path(temp_dir).exists():
            shutil.rmtree(temp_dir)

    print(
        f"[rmat-generate] {out_dir}/edges.parquet "
        f"vertices={vertex_count:,} edges={edge_count:,} skew={args.skew} "
        f"size_bytes={file_size:,} row_groups={row_group_count} "
        f"self_loops={stats['self_loop_count']:,} duplicate_pairs={stats['duplicate_pair_count']:,} "
        f"max_out_degree={stats['max_out_degree']:,} max_in_degree={stats['max_in_degree']:,} "
        f"zero_out_degree_vertices={stats['zero_out_degree_vertex_count']:,} "
        f"sha256={digest} "
        f"generation_seconds={generation_seconds:.1f} stats_seconds={stats_seconds:.1f}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
