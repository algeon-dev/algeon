#!/usr/bin/env python3
"""Convert the DBLP / AMiner V12 citation network JSON into pruning-friendly Parquet.

The upstream file (``dblp.v12.json``, ~12.5 GB) is a single JSON *array* with one
paper object per line: line one is ``[``, every following line is ``,{...}``.  This
script streams that file once and writes the Algeon graph-fixture layout so
cuGraph SQL UDFs can read only the necessary bytes into the GPU before the showcase
enrichment JOIN back to rich paper metadata.

Output layout under ``<out-root>/citation_network/v1``:

- ``parquet/edges_by_src.parquet``   directed citation edges clustered by ``(src, dst, edge_id)``
- ``parquet/edges_by_dst.parquet``   same edges clustered by ``(dst, src, edge_id)``
- ``parquet/profiles_by_paper_id.parquet``  flattened scalar paper metadata, sorted by ``paper_id``
- ``parquet/paper_authors.parquet``  normalized author side table, sorted by ``(paper_id, author_ordinal)``
- ``parquet/paper_fos.parquet``      normalized field-of-study side table, sorted by ``(paper_id, fos_ordinal)``
- ``manifest.json`` + ``source.json``

Vertex IDs stay as the raw AMiner Int64 paper ids (no dense remap / codebook), matching
the cuGraph SQL fixture convention; cuGraph consumes Int64 ``src``/``dst`` directly and
the enrichment JOIN keys on the same ids.

Parquet is written with Snappy compression (fastest libcudf/nvCOMP GPU decode; the edge
tables are the bytes streamed onto the device for cuGraph), format v2.6, per-column
statistics, page indexes, dictionary encoding for low-cardinality categoricals, Bloom
filters on the high-cardinality equality-filter columns the cuGraph story touches
(``src``/``dst`` on edges; ``paper_id``/``venue``/``primary_fos`` on profiles; keys on the
side tables), and declared ``sorting_columns``. Row-group pruning reads the min/max statistics
that the clustering makes selective; ``sorting_columns`` lets the scan report its output ordering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq


DATASET_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = DATASET_DIR.parents[2]
DEFAULT_OUT_ROOT = DATASET_DIR.parent
DEFAULT_SOURCE_JSON = DATASET_DIR / "raw" / "dblp.v12.json"
GENERATOR = "fixture/demo/citation_network/scripts/prepare.py"
DEFAULT_ROW_GROUP_SIZE = 1_048_576
DEFAULT_PROFILE_ROW_GROUP_SIZE = 131_072
DEFAULT_BUCKET_WIDTH = 100_000_000  # AMiner ids reach ~3e9; keep bucket ids compact for pruning demos.

DATASET_ID = "citation-network"
DISPLAY_NAME = "DBLP / AMiner Citation Network V12"
GRAPH_VERTEX_UPPER_BOUND_METADATA_KEY = "algeon.graph.vertex_upper_bound"
_GRAPH_VERTEX_UPPER_BOUND_METADATA_KEY_BYTES = GRAPH_VERTEX_UPPER_BOUND_METADATA_KEY.encode("ascii")
GRAPH_SIMPLE_DIRECTED_EDGE_TOPOLOGY_METADATA_KEY = "algeon.graph.simple_directed_edge_topology"
_GRAPH_SIMPLE_DIRECTED_EDGE_TOPOLOGY_METADATA_KEY_BYTES = GRAPH_SIMPLE_DIRECTED_EDGE_TOPOLOGY_METADATA_KEY.encode("ascii")
MULTI_GRAPH_SLICE_PARTITION = "source_hash_splitmix64"
_U64_MASK = (1 << 64) - 1

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

EDGE_SCHEMA = pa.schema(
    [
        pa.field("edge_id", pa.int64(), nullable=False),
        pa.field("src", pa.int64(), nullable=False),
        pa.field("dst", pa.int64(), nullable=False),
        pa.field("weight", pa.float64(), nullable=False),
        pa.field("src_bucket", pa.int32(), nullable=False),
        pa.field("dst_bucket", pa.int32(), nullable=False),
    ]
)

PROFILE_SCHEMA = pa.schema(
    [
        pa.field("paper_id", pa.int64(), nullable=False),
        pa.field("title", pa.utf8(), nullable=True),
        pa.field("abstract_text", pa.utf8(), nullable=True),
        pa.field("venue", pa.utf8(), nullable=True),
        pa.field("venue_id", pa.int64(), nullable=True),
        pa.field("venue_type", pa.utf8(), nullable=True),
        pa.field("year", pa.int16(), nullable=True),
        pa.field("year_bucket", pa.int16(), nullable=True),
        pa.field("n_citation", pa.int32(), nullable=True),
        pa.field("doc_type", pa.utf8(), nullable=True),
        pa.field("publisher", pa.utf8(), nullable=True),
        pa.field("n_authors", pa.int16(), nullable=True),
        pa.field("n_references", pa.int32(), nullable=True),
        pa.field("primary_fos", pa.utf8(), nullable=True),
    ]
)

AUTHOR_SCHEMA = pa.schema(
    [
        pa.field("paper_id", pa.int64(), nullable=False),
        pa.field("author_ordinal", pa.int32(), nullable=False),
        pa.field("author_id", pa.int64(), nullable=True),
        pa.field("author_name", pa.utf8(), nullable=True),
        pa.field("author_org", pa.utf8(), nullable=True),
    ]
)

FOS_SCHEMA = pa.schema(
    [
        pa.field("paper_id", pa.int64(), nullable=False),
        pa.field("fos_ordinal", pa.int32(), nullable=False),
        pa.field("fos_name", pa.utf8(), nullable=True),
        pa.field("fos_weight", pa.float32(), nullable=True),
    ]
)

# Bloom filters help point/equality lookups on HIGH-cardinality columns the cuGraph
# subgraph-filter and enrichment-JOIN story touches. They are per-column *per row group*,
# so NDV is sized to the row-group row count (see bloom_options), not the dataset. Very
# low-cardinality columns (e.g. doc_type) are deliberately excluded: dictionary encoding
# already prunes them and a Bloom filter there is pure overhead.
EDGE_BLOOM_COLUMNS = ("src", "dst")
PROFILE_BLOOM_COLUMNS = ("paper_id", "venue", "primary_fos")
AUTHOR_BLOOM_COLUMNS = ("paper_id", "author_id")
FOS_BLOOM_COLUMNS = ("paper_id", "fos_name")

# Dictionary encoding shrinks projected reads for low-cardinality categoricals.
PROFILE_DICTIONARY_COLUMNS = ("venue", "venue_type", "doc_type", "publisher", "primary_fos")


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str = DATASET_ID
    display_name: str = DISPLAY_NAME
    source: str = "DBLP / AMiner Citation Network Dataset V12 (Kaggle mathurinache/citation-network-dataset)"
    directed: bool = True
    graph_kind: str = "directed_citation"
    expected_papers: int = 4_894_081
    expected_edges: int = 45_564_149
    notes: str = (
        "Directed paper->reference citation graph. Vertex ids are raw AMiner Int64 paper ids "
        "(no dense remap). Rich non-INT node metadata (title/abstract/venue/year/doc_type/fos) "
        "for cuGraph enrichment-join demos."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DBLP/AMiner V12 citation JSON into Algeon pruning-friendly Parquet.",
    )
    parser.add_argument(
        "--source-json",
        type=Path,
        default=Path(os.getenv("DBLP_SOURCE_JSON", DEFAULT_SOURCE_JSON)),
        help="Path to dblp.v12.json.",
    )
    parser.add_argument("--out-root", type=Path, default=Path(os.getenv("DBLP_OUT_ROOT", DEFAULT_OUT_ROOT)))
    parser.add_argument("--row-group-size", type=int, default=DEFAULT_ROW_GROUP_SIZE)
    parser.add_argument("--profile-row-group-size", type=int, default=DEFAULT_PROFILE_ROW_GROUP_SIZE)
    parser.add_argument("--bucket-width", type=int, default=DEFAULT_BUCKET_WIDTH)
    parser.add_argument(
        "--max-papers",
        type=int,
        default=0,
        help="Smoke-test cap. Uses only the first N paper objects and writes under a sample-suffixed root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated Parquet for the selected root.",
    )
    parser.add_argument(
        "--sort-buffer-rows",
        type=int,
        default=4_000_000,
        help=(
            "Rows buffered per external-sort run (per sorter). Larger uses more RAM but produces "
            "fewer spill runs. Edges use two sorters (src+dst) plus author/fos sorters, so peak RAM "
            "is roughly this many rows times the live sorters."
        ),
    )
    parser.add_argument(
        "--merge-fan-in",
        type=int,
        default=64,
        help=(
            "Maximum spill-run files opened at once during the k-way merge. Very small "
            "--sort-buffer-rows on full data can create thousands of runs; capping fan-in "
            "merges them in cascading passes so the process cannot exceed the OS file-descriptor limit."
        ),
    )
    parser.add_argument(
        "--sort-temp-dir",
        type=Path,
        default=None,
        help="Temporary directory for external sort spill files. Defaults to <root>/work.",
    )
    parser.add_argument(
        "--keep-work",
        action="store_true",
        help="Keep intermediate spill / work files after conversion.",
    )
    parser.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Skip hashing the (large) source JSON when writing source.json.",
    )
    parser.add_argument(
        "--edge-slice-count",
        type=int,
        default=1,
        help=(
            "Emit this many additional independently discoverable DBLP graph fixtures under "
            "<dataset-root>/slices/. Every edge and paper-owned metadata row is assigned to one "
            "source-id hash slice. The canonical full fixture is always retained."
        ),
    )
    parser.add_argument(
        "--edge-slice-seed",
        type=int,
        default=0,
        help="Unsigned 64-bit seed for deterministic source-id edge slicing (default: 0).",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=os.getenv("DBLP_PROGRESS_INTERVAL_SECONDS", "5"),
        help=(
            "Minimum seconds between repeated progress lines. Defaults to "
            "DBLP_PROGRESS_INTERVAL_SECONDS or 5."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logs; the final JSON summary is still printed.",
    )
    return parser.parse_args()


def require_positive(name: str, value: int) -> None:
    if value < 1:
        raise SystemExit(f"{name} must be >= 1")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class ProgressLogger:
    """Rate-limited stderr progress for long local fixture generation."""

    def __init__(self, enabled: bool, interval_seconds: float) -> None:
        self.enabled = enabled
        self.interval_seconds = max(0.0, interval_seconds)
        self.started = time.perf_counter()
        self._last_by_key: dict[str, float] = {}

    def log(self, message: str) -> None:
        if not self.enabled:
            return
        elapsed = time.perf_counter() - self.started
        print(f"[dblp-ingest +{elapsed:8.1f}s] {message}", file=sys.stderr, flush=True)

    def maybe(self, key: str, message: str) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        last = self._last_by_key.get(key, self.started - self.interval_seconds)
        if self.interval_seconds == 0.0 or now - last >= self.interval_seconds:
            self._last_by_key[key] = now
            elapsed = now - self.started
            print(f"[dblp-ingest +{elapsed:8.1f}s] {message}", file=sys.stderr, flush=True)


def _rate(rows: int, elapsed_seconds: float, unit: str) -> str:
    if elapsed_seconds <= 0.0:
        return f"0 {unit}/s"
    return f"{rows / elapsed_seconds:,.0f} {unit}/s"


def _percent(done: int, total: int) -> str:
    if total <= 0:
        return "n/a"
    return f"{(done / total) * 100.0:.1f}%"


def sha256(path: Path, logger: ProgressLogger | None = None) -> str:
    digest = hashlib.sha256()
    total_bytes = path.stat().st_size
    processed_bytes = 0
    started = time.perf_counter()
    if logger is not None:
        logger.log(f"source hash: start sha256 path={path} bytes={total_bytes:,}")
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            processed_bytes += len(chunk)
            if logger is not None:
                logger.maybe(
                    "source.sha256",
                    (
                        f"source hash: processed={processed_bytes:,}/{total_bytes:,} bytes "
                        f"({_percent(processed_bytes, total_bytes)}) "
                        f"rate={_rate(processed_bytes, time.perf_counter() - started, 'B')}"
                    ),
                )
    if logger is not None:
        logger.log(f"source hash: complete bytes={processed_bytes:,}")
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dataset_root(out_root: Path, max_papers: int) -> Path:
    suffix = f"_sample_papers{max_papers}" if max_papers > 0 else ""
    return out_root / DATASET_ID.replace("-", "_") / f"v1{suffix}"


def edge_slice_index(source_paper_id: int, slice_count: int, seed: int) -> int:
    """Return the deterministic owner for one source paper's outgoing edges.

    The fixed SplitMix64 finalizer gives a stable, well-distributed partition
    without relying on Python's process-randomized ``hash``. Partitioning by
    source keeps every paper's outgoing adjacency list in one graph fixture;
    the edge rowsets are disjoint and the paper-owned metadata tables use the
    same ownership rule.
    """
    if slice_count < 1:
        raise ValueError("edge slice count must be positive")
    value = (int(source_paper_id) & _U64_MASK) ^ (int(seed) & _U64_MASK)
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & _U64_MASK
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & _U64_MASK
    value ^= value >> 31
    return value % slice_count


def multi_graph_slice_root(root: Path, index: int) -> Path:
    return root / "slices" / f"source-hash-{index:02d}"


def nullable_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def reconstruct_abstract(indexed_abstract: Any) -> str | None:
    """Rebuild plain abstract text from AMiner's ``{IndexLength, InvertedIndex: {word: [pos...]}}``."""
    if not isinstance(indexed_abstract, dict):
        return None
    inverted = indexed_abstract.get("InvertedIndex")
    if not isinstance(inverted, dict) or not inverted:
        return None
    length = indexed_abstract.get("IndexLength")
    if not isinstance(length, int) or length <= 0:
        # Derive length from the highest position, tolerating malformed (non-list / non-int) entries.
        max_position = -1
        for positions in inverted.values():
            if not isinstance(positions, list):
                continue
            for position in positions:
                if isinstance(position, int) and position > max_position:
                    max_position = position
        length = max_position + 1
    if length <= 0:
        return None
    tokens: list[str | None] = [None] * length
    for word, positions in inverted.items():
        if not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int) and 0 <= position < length:
                tokens[position] = word
    text = " ".join(token for token in tokens if token is not None)
    return text or None


def iter_papers(source_json: Path, max_papers: int) -> Iterator[dict[str, Any]]:
    """Yield paper objects from the AMiner V12 JSON-array-per-line file."""
    emitted = 0
    with source_json.open("r", encoding="utf-8") as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            stripped = raw_line.strip()
            if not stripped or stripped in ("[", "]"):
                continue
            if stripped.startswith(","):
                stripped = stripped[1:]
            if stripped.endswith(","):
                stripped = stripped[:-1]
            if not stripped:
                continue
            try:
                paper = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise SystemExit(f"{source_json}:{line_number}: could not parse paper JSON: {error}") from error
            yield paper
            emitted += 1
            if max_papers > 0 and emitted >= max_papers:
                return


# ---------------------------------------------------------------------------
# External merge sort for the edge tables (edges do not fit in RAM at full scale).
# ---------------------------------------------------------------------------


def _read_edge_run(path: Path) -> Iterator[tuple[int, int, int]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            src_s, dst_s, edge_s = line.rstrip("\n").split("\t")
            yield (int(src_s), int(dst_s), int(edge_s))


class EdgeSorter:
    """Buffer (src, dst, edge_id) rows, spill sorted runs, then cascade-merge them.

    The merge fan-in is bounded so that a very small ``--sort-buffer-rows`` on full-scale data
    (which produces many spill runs) can never open more files at once than the OS descriptor
    limit allows: excess runs are merged in cascading passes down to ``merge_fan_in`` survivors.
    """

    def __init__(
        self,
        work_dir: Path,
        buffer_rows: int,
        order: str,
        merge_fan_in: int = 64,
        logger: ProgressLogger | None = None,
        run_id: str = "",
    ) -> None:
        self.work_dir = work_dir
        self.buffer_rows = buffer_rows
        self.order = order
        self.merge_fan_in = max(2, merge_fan_in)
        self.logger = logger
        self.run_id = run_id
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._buffer: list[tuple[int, int, int]] = []  # (src, dst, edge_id)
        self._run_paths: list[Path] = []
        self._run_seq = 0
        self.rows_added = 0

    def _sort_key(self, row: tuple[int, int, int]) -> tuple[int, int, int]:
        src, dst, edge_id = row
        if self.order == "src":
            return (src, dst, edge_id)
        return (dst, src, edge_id)

    def add(self, src: int, dst: int, edge_id: int) -> None:
        self._buffer.append((src, dst, edge_id))
        self.rows_added += 1
        if len(self._buffer) >= self.buffer_rows:
            self._spill()

    def _next_run_path(self) -> Path:
        prefix = f"{self.run_id}_" if self.run_id else ""
        path = self.work_dir / f"{prefix}edges_{self.order}_run_{self._run_seq:05d}.tsv"
        self._run_seq += 1
        return path

    def _spill(self) -> None:
        if not self._buffer:
            return
        run_rows = len(self._buffer)
        if self.logger is not None:
            self.logger.log(
                f"edge sorter {self.order}: spill run={self._run_seq} rows={run_rows:,} "
                f"total_rows={self.rows_added:,}"
            )
        self._buffer.sort(key=self._sort_key)
        run_path = self._next_run_path()
        try:
            with run_path.open("w", encoding="utf-8") as fh:
                for src, dst, edge_id in self._buffer:
                    fh.write(f"{src}\t{dst}\t{edge_id}\n")
        except BaseException:
            if run_path.exists():
                run_path.unlink()
            raise
        self._run_paths.append(run_path)
        self._buffer.clear()
        if self.logger is not None:
            self.logger.log(f"edge sorter {self.order}: wrote spill run={run_path.name}")

    def _merge_iters(self, iterators: list[Iterator[tuple[int, int, int]]]) -> Iterator[tuple[int, int, int]]:
        import heapq

        merged = heapq.merge(
            *(((self._sort_key(row), row) for row in it) for it in iterators),
            key=lambda item: item[0],
        )
        for _key, row in merged:
            yield row

    def _reduce_runs(self) -> None:
        """Cascade-merge spill runs until at most ``merge_fan_in`` remain."""
        pass_index = 0
        while len(self._run_paths) > self.merge_fan_in:
            pass_index += 1
            if self.logger is not None:
                self.logger.log(
                    f"edge sorter {self.order}: reduce pass={pass_index} runs={len(self._run_paths):,} "
                    f"fan_in={self.merge_fan_in}"
                )
            survivors: list[Path] = []
            for start in range(0, len(self._run_paths), self.merge_fan_in):
                group = self._run_paths[start : start + self.merge_fan_in]
                if len(group) == 1:
                    survivors.append(group[0])
                    continue
                if self.logger is not None:
                    self.logger.log(
                        f"edge sorter {self.order}: merge spill group start={start} count={len(group)}"
                    )
                out_path = self._next_run_path()
                try:
                    with out_path.open("w", encoding="utf-8") as fh:
                        for src, dst, edge_id in self._merge_iters([_read_edge_run(p) for p in group]):
                            fh.write(f"{src}\t{dst}\t{edge_id}\n")
                except BaseException:
                    if out_path.exists():
                        out_path.unlink()
                    raise
                for p in group:
                    p.unlink()
                survivors.append(out_path)
                if self.logger is not None:
                    self.logger.log(f"edge sorter {self.order}: wrote reduced run={out_path.name}")
            self._run_paths = survivors

    def merged_rows(self) -> Iterator[tuple[int, int, int]]:
        """Yield (src, dst, edge_id) in fully sorted order for this layout."""
        # Purely in-memory fast path when nothing spilled.
        if not self._run_paths:
            if self.logger is not None:
                self.logger.log(f"edge sorter {self.order}: final in-memory sort rows={len(self._buffer):,}")
            self._buffer.sort(key=self._sort_key)
            yield from self._buffer
            self._buffer.clear()
            return
        self._spill()
        self._reduce_runs()
        if self.logger is not None:
            self.logger.log(f"edge sorter {self.order}: stream merge runs={len(self._run_paths):,}")
        yield from self._merge_iters([_read_edge_run(path) for path in self._run_paths])
        if self.logger is not None:
            self.logger.log(f"edge sorter {self.order}: stream merge complete")

    def cleanup(self) -> None:
        for path in self._run_paths:
            if path.exists():
                path.unlink()
        self._run_paths.clear()


class RowSorter:
    """Buffer wide node rows, spill sorted-by-key Parquet runs, then k-way merge to a sorted stream.

    Used for the profile / author / fos tables so the declared ``sorting_columns`` are honest:
    the AMiner JSON is *not* ordered by ``paper_id``, so a stable global sort by the table key is
    required before writing, otherwise readers would trust a false clustering claim.
    """

    def __init__(
        self,
        work_dir: Path,
        name: str,
        schema: pa.Schema,
        sort_columns: list[str],
        buffer_rows: int,
        merge_fan_in: int = 64,
        logger: ProgressLogger | None = None,
        run_id: str = "",
    ) -> None:
        self.work_dir = work_dir
        self.name = name
        self.schema = schema
        self.sort_columns = sort_columns
        self.buffer_rows = buffer_rows
        self.merge_fan_in = max(2, merge_fan_in)
        self.logger = logger
        self.run_id = run_id
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._buffer: dict[str, list[Any]] = {field.name: [] for field in schema}
        self._rows = 0
        self._run_paths: list[Path] = []
        self._run_seq = 0
        self.rows_added = 0

    def add_columns(self, columns: dict[str, list[Any]]) -> None:
        """Absorb a batch of column buffers (they are cleared by the caller afterwards)."""
        added = len(columns[self.schema.field(0).name])
        if added == 0:
            return
        for name, values in columns.items():
            self._buffer[name].extend(values)
        self._rows += added
        self.rows_added += added
        if self._rows >= self.buffer_rows:
            self._spill()

    def _sorted_indices(self) -> list[int]:
        keys = list(zip(*[self._buffer[name] for name in self.sort_columns]))
        return sorted(range(len(keys)), key=lambda i: keys[i])

    def _next_run_path(self) -> Path:
        prefix = f"{self.run_id}_" if self.run_id else ""
        path = self.work_dir / f"{prefix}{self.name}_run_{self._run_seq:05d}.parquet"
        self._run_seq += 1
        return path

    def _spill(self) -> None:
        if self._rows == 0:
            return
        if self.logger is not None:
            self.logger.log(
                f"{self.name}: spill run={self._run_seq} rows={self._rows:,} total_rows={self.rows_added:,}"
            )
        order = self._sorted_indices()
        table = pa.table(
            {field.name: pa.array([self._buffer[field.name][i] for i in order], type=field.type) for field in self.schema},
            schema=self.schema,
        )
        run_path = self._next_run_path()
        try:
            pq.write_table(table, run_path, compression="snappy", version="2.6")
        except BaseException:
            if run_path.exists():
                run_path.unlink()
            raise
        self._run_paths.append(run_path)
        self._buffer = {field.name: [] for field in self.schema}
        self._rows = 0
        if self.logger is not None:
            self.logger.log(f"{self.name}: wrote spill run={run_path.name}")

    def _in_memory_sorted_table(self) -> pa.Table:
        order = self._sorted_indices()
        table = pa.table(
            {field.name: pa.array([self._buffer[field.name][i] for i in order], type=field.type) for field in self.schema},
            schema=self.schema,
        )
        self._buffer = {field.name: [] for field in self.schema}
        self._rows = 0
        return table

    def _run_row_iter(self, path: Path) -> Iterator[tuple[tuple, dict[str, Any]]]:
        reader = pq.ParquetFile(path)
        for batch in reader.iter_batches(batch_size=65_536):
            cols = {name: batch.column(idx).to_pylist() for idx, name in enumerate(batch.schema.names)}
            for i in range(batch.num_rows):
                row = {name: cols[name][i] for name in cols}
                key = tuple(row[name] for name in self.sort_columns)
                yield key, row

    def _merge_paths(self, paths: list[Path]) -> Iterator[dict[str, Any]]:
        import heapq

        merged = heapq.merge(*(self._run_row_iter(path) for path in paths), key=lambda item: item[0])
        for _key, row in merged:
            yield row

    def _reduce_runs(self) -> None:
        """Cascade-merge spill runs until at most ``merge_fan_in`` remain (bounds open fds)."""
        pass_index = 0
        while len(self._run_paths) > self.merge_fan_in:
            pass_index += 1
            if self.logger is not None:
                self.logger.log(
                    f"{self.name}: reduce pass={pass_index} runs={len(self._run_paths):,} "
                    f"fan_in={self.merge_fan_in}"
                )
            survivors: list[Path] = []
            for start in range(0, len(self._run_paths), self.merge_fan_in):
                group = self._run_paths[start : start + self.merge_fan_in]
                if len(group) == 1:
                    survivors.append(group[0])
                    continue
                if self.logger is not None:
                    self.logger.log(f"{self.name}: merge spill group start={start} count={len(group)}")
                out_path = self._next_run_path()
                try:
                    writer = pq.ParquetWriter(out_path, self.schema, compression="snappy", version="2.6")
                    batch: dict[str, list[Any]] = {field.name: [] for field in self.schema}
                    pending = 0
                    merged_rows = 0
                    try:
                        for row in self._merge_paths(group):
                            for name in batch:
                                batch[name].append(row[name])
                            pending += 1
                            if pending >= 131_072:
                                writer.write_table(
                                    pa.table(
                                        {f.name: pa.array(batch[f.name], type=f.type) for f in self.schema},
                                        schema=self.schema,
                                    )
                                )
                                batch = {field.name: [] for field in self.schema}
                                merged_rows += pending
                                pending = 0
                        if pending:
                            writer.write_table(
                                pa.table(
                                    {f.name: pa.array(batch[f.name], type=f.type) for f in self.schema},
                                    schema=self.schema,
                                )
                            )
                            merged_rows += pending
                    finally:
                        writer.close()
                except BaseException:
                    if out_path.exists():
                        out_path.unlink()
                    raise
                for path in group:
                    path.unlink()
                survivors.append(out_path)
                if self.logger is not None:
                    self.logger.log(f"{self.name}: wrote reduced run={out_path.name} rows={merged_rows:,}")
            self._run_paths = survivors

    def write_sorted(self, writer: pq.ParquetWriter, row_group_size: int) -> None:
        """Emit all buffered rows to ``writer`` in globally sorted order."""
        if not self._run_paths:
            if self.logger is not None:
                self.logger.log(f"{self.name}: final in-memory sort rows={self._rows:,}")
            table = self._in_memory_sorted_table()
            if table.num_rows:
                writer.write_table(table, row_group_size=row_group_size)
            if self.logger is not None:
                self.logger.log(f"{self.name}: wrote sorted rows={table.num_rows:,}")
            return
        self._spill()
        self._reduce_runs()

        out = {field.name: [] for field in self.schema}
        pending = 0
        written = 0
        started = time.perf_counter()
        if self.logger is not None:
            self.logger.log(
                f"{self.name}: stream merge/write runs={len(self._run_paths):,} row_group_size={row_group_size:,}"
            )
        for row in self._merge_paths(self._run_paths):
            for name in out:
                out[name].append(row[name])
            pending += 1
            if pending >= row_group_size:
                writer.write_table(
                    pa.table({f.name: pa.array(out[f.name], type=f.type) for f in self.schema}, schema=self.schema),
                    row_group_size=row_group_size,
                )
                written += pending
                if self.logger is not None:
                    self.logger.maybe(
                        f"{self.name}.write",
                        f"{self.name}: wrote rows={written:,}/{self.rows_added:,} "
                        f"({_percent(written, self.rows_added)}) rate={_rate(written, time.perf_counter() - started, 'rows')}",
                    )
                out = {field.name: [] for field in self.schema}
                pending = 0
        if pending:
            writer.write_table(
                pa.table({f.name: pa.array(out[f.name], type=f.type) for f in self.schema}, schema=self.schema),
                row_group_size=row_group_size,
            )
            written += pending
        if self.logger is not None:
            self.logger.log(f"{self.name}: write complete rows={written:,}")

    def cleanup(self) -> None:
        for path in self._run_paths:
            if path.exists():
                path.unlink()
        self._run_paths.clear()


# ---------------------------------------------------------------------------
# Parquet writers
# ---------------------------------------------------------------------------


def bloom_options(columns: tuple[str, ...], row_group_size: int, dataset_ndv: int) -> dict[str, Any]:
    """Size Bloom filters to per-row-group cardinality.

    A Bloom filter is written once per column *per row group*, so its NDV must reflect the
    distinct values in one row group -- at most ``row_group_size`` -- not the whole dataset.
    Sizing to the dataset count would bloat every row group's filter (e.g. a 131k-row profile
    group carrying a filter sized for ~4.9M values), which grows file size and hurts the
    GPU-read-minimization goal. We also cap by the dataset cardinality for small tables.
    """
    ndv = max(1024, min(row_group_size, dataset_ndv, 1 << 26))
    return {column: {"ndv": ndv, "fpp": 0.01} for column in columns}


def sorting_columns(schema: pa.Schema, names: list[str]) -> list[pq.SortingColumn]:
    return [pq.SortingColumn(schema.get_field_index(name)) for name in names]


def edge_footer_metadata(vertex_upper_bound: int | None) -> dict[bytes, bytes]:
    """Return edge-rowset contracts bound to the Parquet footer.

    Every generated edge file is a simple directed relation: ``src``/``dst``
    are non-null and unique as an ordered pair. The writer verifies that
    invariant while it streams the sorted rows below. The topology certificate
    lets a consumer prove that Python-compatible null filtering and directed
    deduplication would be no-ops without trusting a detached manifest.

    The optional vertex bound is narrower: it is emitted only for the complete
    canonical rowset, because a sample can contain endpoints outside its paper
    prefix, and the edge writer fails if any destination is not a scanned
    paper id.
    """
    if vertex_upper_bound is not None and vertex_upper_bound < 0:
        raise ValueError("graph vertex upper bound must not be negative")
    metadata = {
        _GRAPH_SIMPLE_DIRECTED_EDGE_TOPOLOGY_METADATA_KEY_BYTES: json.dumps(
            {
                "source_column_index": EDGE_SCHEMA.get_field_index("src"),
                "destination_column_index": EDGE_SCHEMA.get_field_index("dst"),
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8"),
    }
    if vertex_upper_bound is not None:
        metadata[_GRAPH_VERTEX_UPPER_BOUND_METADATA_KEY_BYTES] = str(vertex_upper_bound).encode("ascii")
    return metadata


def open_edge_writer(
    path: Path,
    order: str,
    row_group_size: int,
    dataset_ndv: int,
    graph_vertex_upper_bound: int | None,
) -> pq.ParquetWriter:
    sort_names = ["src", "dst", "edge_id"] if order == "src" else ["dst", "src", "edge_id"]
    edge_schema = EDGE_SCHEMA.with_metadata(edge_footer_metadata(graph_vertex_upper_bound))
    return pq.ParquetWriter(
        path,
        edge_schema,
        compression="snappy",
        version="2.6",
        write_statistics=True,
        write_page_index=True,
        sorting_columns=sorting_columns(EDGE_SCHEMA, sort_names),
        bloom_filter_options=bloom_options(EDGE_BLOOM_COLUMNS, row_group_size, dataset_ndv),
    )


def edge_batch(src: list[int], dst: list[int], edge_id: list[int], bucket_width: int) -> pa.Table:
    return pa.table(
        {
            "edge_id": pa.array(edge_id, type=pa.int64()),
            "src": pa.array(src, type=pa.int64()),
            "dst": pa.array(dst, type=pa.int64()),
            "weight": pa.array([1.0] * len(src), type=pa.float64()),
            "src_bucket": pa.array([value // bucket_width for value in src], type=pa.int32()),
            "dst_bucket": pa.array([value // bucket_width for value in dst], type=pa.int32()),
        },
        schema=EDGE_SCHEMA,
    )


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def parquet_metadata(path: Path) -> dict[str, Any]:
    parquet_file = pq.ParquetFile(path)
    meta = parquet_file.metadata
    column_names = parquet_file.schema_arrow.names
    row_groups: list[dict[str, Any]] = []
    stat_columns = ("src", "dst", "paper_id", "year", "n_citation", "author_id", "fos_ordinal")
    for index in range(min(meta.num_row_groups, 8)):
        row_group = meta.row_group(index)
        entry: dict[str, Any] = {"index": index, "rows": row_group.num_rows}
        for column in stat_columns:
            if column not in column_names:
                continue
            stats = row_group.column(column_names.index(column)).statistics
            if stats is not None and stats.has_min_max:
                entry[f"{column}_min"] = _stat_scalar(stats.min)
                entry[f"{column}_max"] = _stat_scalar(stats.max)
        row_groups.append(entry)
    result = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "num_rows": meta.num_rows,
        "num_row_groups": meta.num_row_groups,
        "schema": column_names,
        "sample_row_groups": row_groups,
    }
    footer_metadata = {}
    for key, encoding in (
        (_GRAPH_VERTEX_UPPER_BOUND_METADATA_KEY_BYTES, "ascii"),
        (_GRAPH_SIMPLE_DIRECTED_EDGE_TOPOLOGY_METADATA_KEY_BYTES, "utf-8"),
    ):
        value = (meta.metadata or {}).get(key)
        if value is not None:
            footer_metadata[key.decode("ascii")] = value.decode(encoding)
    if footer_metadata:
        result["footer_metadata"] = footer_metadata
    return result


def _stat_scalar(value: Any) -> Any:
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def validate_full_counts(spec: DatasetSpec, counters: dict[str, Any], max_papers: int) -> None:
    if max_papers > 0:
        return
    papers = int(counters["papers"])
    edges = int(counters["edges"])
    if papers != spec.expected_papers:
        raise SystemExit(
            f"expected {spec.expected_papers:,} papers, got {papers:,}. The source may be truncated "
            f"or not the canonical DBLP/AMiner V12 file; pass --max-papers for an intentional sample."
        )
    if edges != spec.expected_edges:
        raise SystemExit(f"expected {spec.expected_edges:,} citation edges, got {edges:,}.")


def full_edge_rowset_vertex_upper_bound(counters: dict[str, Any], max_papers: int) -> int | None:
    """Return a bound only when the edge files contain the complete source rowset.

    A sampled prefix can include references to papers outside that prefix, so its
    paper counter is not an upper bound for the vertices in its emitted edges.
    """
    if max_papers > 0:
        return None
    return int(counters["papers"])


def _open_multi_graph_slice_writer(
    table: str,
    path: Path,
    args: argparse.Namespace,
    dataset_ndv: int,
) -> pq.ParquetWriter:
    if table.startswith("edges_by_"):
        # A slice is a complete fixture of its own, but not the complete V12
        # edge rowset. It must not inherit the full-graph footer bound.
        return open_edge_writer(
            path,
            table.removeprefix("edges_by_"),
            args.row_group_size,
            dataset_ndv,
            graph_vertex_upper_bound=None,
        )

    if table == "profiles_by_paper_id":
        schema = PROFILE_SCHEMA
        sort_names = ["paper_id"]
        dictionary_columns = list(PROFILE_DICTIONARY_COLUMNS)
        bloom = bloom_options(PROFILE_BLOOM_COLUMNS, args.profile_row_group_size, dataset_ndv)
    elif table == "paper_authors":
        schema = AUTHOR_SCHEMA
        sort_names = ["paper_id", "author_ordinal"]
        dictionary_columns = ["author_org"]
        bloom = bloom_options(AUTHOR_BLOOM_COLUMNS, args.row_group_size, dataset_ndv * 3)
    elif table == "paper_fos":
        schema = FOS_SCHEMA
        sort_names = ["paper_id", "fos_ordinal"]
        dictionary_columns = ["fos_name"]
        bloom = bloom_options(FOS_BLOOM_COLUMNS, args.row_group_size, dataset_ndv * 5)
    else:
        raise ValueError(f"unknown DBLP table {table}")

    return pq.ParquetWriter(
        path,
        schema,
        compression="snappy",
        version="2.6",
        write_statistics=True,
        write_page_index=True,
        use_dictionary=dictionary_columns,
        sorting_columns=sorting_columns(schema, sort_names),
        bloom_filter_options=bloom,
    )


def _new_multi_graph_slice_counters(slice_count: int) -> list[dict[str, Any]]:
    return [
        {
            "papers": 0,
            "edges": 0,
            "papers_with_abstract": 0,
            "author_rows": 0,
            "fos_rows": 0,
            "min_paper_id": math.inf,
            "max_paper_id": -1,
        }
        for _ in range(slice_count)
    ]


def _record_slice_rows(
    table: str,
    slice_index: int,
    rows: pa.Table,
    counters: list[dict[str, Any]],
    rowset_hashers: list[hashlib._Hash],
) -> None:
    counter = counters[slice_index]
    if table == "edges_by_src":
        counter["edges"] += rows.num_rows
        edge_ids = rows.column("edge_id").to_pylist()
        srcs = rows.column("src").to_pylist()
        dsts = rows.column("dst").to_pylist()
        for edge_id, src, dst in zip(edge_ids, srcs, dsts, strict=True):
            # The digest is over the logical, source-sorted edge rowset rather
            # than Parquet bytes, so it stays stable across writer versions.
            rowset_hashers[slice_index].update(struct.pack(">qqq", int(edge_id), int(src), int(dst)))
        return
    if table == "edges_by_dst":
        # The same logical rows were counted and fingerprinted from the
        # source-sorted layout. This layout is its independently sorted copy.
        return
    if table == "profiles_by_paper_id":
        counter["papers"] += rows.num_rows
        counter["papers_with_abstract"] += rows.num_rows - rows.column("abstract_text").null_count
        paper_ids = [int(value) for value in rows.column("paper_id").to_pylist()]
        if paper_ids:
            counter["min_paper_id"] = min(counter["min_paper_id"], paper_ids[0])
            counter["max_paper_id"] = max(counter["max_paper_id"], paper_ids[-1])
        return
    if table == "paper_authors":
        counter["author_rows"] += rows.num_rows
        return
    if table == "paper_fos":
        counter["fos_rows"] += rows.num_rows
        return
    raise ValueError(f"unknown DBLP table {table}")


def _write_multi_graph_slice_table(
    source_path: Path,
    table: str,
    slice_roots: list[Path],
    args: argparse.Namespace,
    counters: list[dict[str, Any]],
    rowset_hashers: list[hashlib._Hash],
) -> None:
    slice_count = len(slice_roots)
    partition_column = "src" if table.startswith("edges_by_") else "paper_id"
    row_group_size = args.row_group_size if table != "profiles_by_paper_id" else args.profile_row_group_size
    dataset_ndv = max(1, (args.max_papers if args.max_papers > 0 else DatasetSpec().expected_papers) // slice_count)
    writers = [
        _open_multi_graph_slice_writer(table, root / "parquet" / f"{table}.parquet", args, dataset_ndv)
        for root in slice_roots
    ]
    try:
        parquet_file = pq.ParquetFile(source_path)
        for batch in parquet_file.iter_batches(batch_size=row_group_size):
            rows = pa.Table.from_batches([batch])
            positions = [[] for _ in range(slice_count)]
            for row_index, source_paper_id in enumerate(rows.column(partition_column).to_pylist()):
                positions[edge_slice_index(int(source_paper_id), slice_count, args.edge_slice_seed)].append(row_index)
            for index, selected_positions in enumerate(positions):
                if not selected_positions:
                    continue
                selected = rows.take(pa.array(selected_positions, type=pa.int64()))
                writers[index].write_table(selected, row_group_size=row_group_size)
                _record_slice_rows(table, index, selected, counters, rowset_hashers)
    finally:
        for writer in writers:
            writer.close()


def _slice_sql_registration(spec: DatasetSpec, parquet_dir: Path) -> dict[str, str]:
    table_prefix = spec.dataset_id.replace("-", "_")
    return {
        "edges_by_src": (
            f"CREATE EXTERNAL TABLE {table_prefix}_edges_by_src "
            f"STORED AS PARQUET LOCATION '{parquet_dir / 'edges_by_src.parquet'}'"
        ),
        "edges_by_dst": (
            f"CREATE EXTERNAL TABLE {table_prefix}_edges_by_dst "
            f"STORED AS PARQUET LOCATION '{parquet_dir / 'edges_by_dst.parquet'}'"
        ),
        "profiles_by_paper_id": (
            f"CREATE EXTERNAL TABLE {table_prefix}_profiles "
            f"STORED AS PARQUET LOCATION '{parquet_dir / 'profiles_by_paper_id.parquet'}'"
        ),
        "paper_authors": (
            f"CREATE EXTERNAL TABLE {table_prefix}_authors "
            f"STORED AS PARQUET LOCATION '{parquet_dir / 'paper_authors.parquet'}'"
        ),
        "paper_fos": (
            f"CREATE EXTERNAL TABLE {table_prefix}_fos "
            f"STORED AS PARQUET LOCATION '{parquet_dir / 'paper_fos.parquet'}'"
        ),
    }


def write_multi_graph_slices(
    spec: DatasetSpec,
    args: argparse.Namespace,
    root: Path,
    source_payload: dict[str, Any],
    logger: ProgressLogger,
) -> list[dict[str, Any]]:
    """Emit independently discoverable source-hash graph fixtures.

    Each slice receives all five fixture tables, its own manifest/source
    provenance, and a logical edge-rowset SHA-256. The source-hash ownership
    is applied to both outgoing edge rows and paper-owned metadata rows; no
    slice borrows Parquet data from another slice at read time.
    """
    slice_count = args.edge_slice_count
    if slice_count <= 1:
        return []

    slices_root = root / "slices"
    slice_roots = [multi_graph_slice_root(root, index) for index in range(slice_count)]
    for slice_root in slice_roots:
        (slice_root / "parquet").mkdir(parents=True, exist_ok=True)

    counters = _new_multi_graph_slice_counters(slice_count)
    rowset_hashers = [hashlib.sha256() for _ in range(slice_count)]
    for index, rowset_hasher in enumerate(rowset_hashers):
        rowset_hasher.update(b"algeon-datafusion.dblp.source-hash-edge-rowset.v1\\0")
        rowset_hasher.update(struct.pack(">QQQ", slice_count, index, args.edge_slice_seed))

    parquet_dir = root / "parquet"
    logger.log(f"multi-graph slices: write {slice_count} source-hash fixtures")
    for table in ("edges_by_src", "edges_by_dst", "profiles_by_paper_id", "paper_authors", "paper_fos"):
        logger.log(f"multi-graph slices: partition {table}")
        _write_multi_graph_slice_table(
            parquet_dir / f"{table}.parquet",
            table,
            slice_roots,
            args,
            counters,
            rowset_hashers,
        )

    if any(counter["edges"] == 0 for counter in counters):
        empty = [index for index, counter in enumerate(counters) if counter["edges"] == 0]
        raise SystemExit(
            f"--edge-slice-count={slice_count} produced empty edge slices {empty}; reduce the count or choose another seed"
        )

    entries: list[dict[str, Any]] = []
    for index, (slice_root, counter, rowset_hasher) in enumerate(zip(slice_roots, counters, rowset_hashers, strict=True)):
        slice_parquet_dir = slice_root / "parquet"
        edge_layouts = {}
        for order in ("src", "dst"):
            edge_meta = parquet_metadata(slice_parquet_dir / f"edges_by_{order}.parquet")
            edge_layouts[order] = {
                "sort_order": ["src", "dst", "edge_id"] if order == "src" else ["dst", "src", "edge_id"],
                "clustered_for": f"{order}-range filters and row-group / bloom pruning",
                "footer_metadata": edge_meta.get("footer_metadata", {}),
                "parquet": edge_meta,
            }
        rowset_sha256 = rowset_hasher.hexdigest()
        slice_metadata = {
            "partition": MULTI_GRAPH_SLICE_PARTITION,
            "slice_count": slice_count,
            "slice_index": index,
            "hash_seed": args.edge_slice_seed,
            "edge_rowset_sha256": rowset_sha256,
        }
        manifest = {
            "generator": GENERATOR,
            "generated_at_unix": int(time.time()),
            "repo_root": str(REPO_ROOT),
            "dataset": {
                "dataset_id": spec.dataset_id,
                "display_name": spec.display_name,
                "source": spec.source,
                "directed": spec.directed,
                "graph_kind": spec.graph_kind,
                "expected_papers": spec.expected_papers,
                "expected_edges": spec.expected_edges,
                "notes": spec.notes,
            },
            "root": str(slice_root),
            "max_papers": None if args.max_papers <= 0 else args.max_papers,
            "row_group_size": args.row_group_size,
            "profile_row_group_size": args.profile_row_group_size,
            "bucket_width": args.bucket_width,
            "multi_graph_slice": slice_metadata,
            "bloom_filter_columns": {
                "edges": list(EDGE_BLOOM_COLUMNS),
                "profiles": list(PROFILE_BLOOM_COLUMNS),
                "paper_authors": list(AUTHOR_BLOOM_COLUMNS),
                "paper_fos": list(FOS_BLOOM_COLUMNS),
            },
            "counts": {
                "papers": counter["papers"],
                "edges": counter["edges"],
                "papers_with_abstract": counter["papers_with_abstract"],
                "author_rows": counter["author_rows"],
                "fos_rows": counter["fos_rows"],
                "min_paper_id": None if counter["papers"] == 0 else int(counter["min_paper_id"]),
                "max_paper_id": None if counter["papers"] == 0 else int(counter["max_paper_id"]),
            },
            "edge_layouts": edge_layouts,
            "profiles": parquet_metadata(slice_parquet_dir / "profiles_by_paper_id.parquet"),
            "paper_authors": parquet_metadata(slice_parquet_dir / "paper_authors.parquet"),
            "paper_fos": parquet_metadata(slice_parquet_dir / "paper_fos.parquet"),
            "sql_registration": _slice_sql_registration(spec, slice_parquet_dir),
        }
        slice_source_payload = json.loads(json.dumps(source_payload))
        slice_source_payload["derived_fixture"] = slice_metadata
        write_json(slice_root / "manifest.json", manifest)
        write_json(slice_root / "source.json", slice_source_payload)
        entries.append({"index": index, "root": str(slice_root), "edge_rowset_sha256": rowset_sha256})

    logger.log(f"multi-graph slices: complete root={slices_root}")
    return entries


def convert(spec: DatasetSpec, args: argparse.Namespace) -> dict[str, Any]:
    edge_slice_count = getattr(args, "edge_slice_count", 1)
    edge_slice_seed = getattr(args, "edge_slice_seed", 0)
    if edge_slice_count < 1:
        raise SystemExit("--edge-slice-count must be >= 1")
    if edge_slice_seed < 0 or edge_slice_seed > _U64_MASK:
        raise SystemExit("--edge-slice-seed must be an unsigned 64-bit integer")
    # Unit tests construct a minimal Namespace directly; normalize it to the
    # public CLI shape before helpers consume the optional slice settings.
    args.edge_slice_count = edge_slice_count
    args.edge_slice_seed = edge_slice_seed
    root = dataset_root(args.out_root.resolve(), args.max_papers)
    parquet_dir = root / "parquet"
    work_dir = (args.sort_temp_dir or (root / "work")).resolve()
    run_id = f"run_{os.getpid()}_{time.time_ns()}"
    logger = ProgressLogger(not args.quiet, args.progress_interval)
    source_bytes = args.source_json.stat().st_size
    expected_scan_papers = args.max_papers if args.max_papers > 0 else spec.expected_papers
    logger.log(
        f"dataset={spec.dataset_id} root={root} source={args.source_json} "
        f"source_bytes={source_bytes:,} overwrite={args.overwrite} max_papers={args.max_papers or 'all'} "
        f"row_group_size={args.row_group_size:,} profile_row_group_size={args.profile_row_group_size:,} "
        f"sort_buffer_rows={args.sort_buffer_rows:,} merge_fan_in={args.merge_fan_in} "
        f"edge_slice_count={edge_slice_count} edge_slice_seed={edge_slice_seed}"
    )

    if parquet_dir.exists() and any(parquet_dir.glob("*.parquet")) and not args.overwrite:
        raise SystemExit(f"{parquet_dir}: existing Parquet found; pass --overwrite to regenerate")
    slices_root = root / "slices"
    if args.overwrite and slices_root.exists():
        # Slice roots are generated children of this selected dataset root;
        # retaining an old count would advertise stale graph fixtures.
        shutil.rmtree(slices_root)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    # Bloom NDV targets the number of *distinct values* in the filtered column. For src/dst that is
    # the distinct-vertex count (~ paper count, since papers both cite and are cited); for the node
    # tables the key is paper_id, whose distinct count is exactly the paper count.
    ndv_papers = args.max_papers if args.max_papers > 0 else spec.expected_papers
    ndv_edges = ndv_papers

    # Node tables are sorted by paper_id before writing so the declared sorting_columns are honest
    # (the AMiner JSON is not ordered by id). Edges use their own dedicated integer sorter.
    fan_in = args.merge_fan_in
    profile_sorter = RowSorter(
        work_dir,
        "profiles",
        PROFILE_SCHEMA,
        ["paper_id"],
        args.profile_row_group_size * 8,
        fan_in,
        logger,
        run_id,
    )
    author_sorter = RowSorter(
        work_dir,
        "authors",
        AUTHOR_SCHEMA,
        ["paper_id", "author_ordinal"],
        args.sort_buffer_rows,
        fan_in,
        logger,
        run_id,
    )
    fos_sorter = RowSorter(
        work_dir,
        "fos",
        FOS_SCHEMA,
        ["paper_id", "fos_ordinal"],
        args.sort_buffer_rows,
        fan_in,
        logger,
        run_id,
    )

    edge_src_sorter = EdgeSorter(work_dir, args.sort_buffer_rows, "src", fan_in, logger, run_id)
    edge_dst_sorter = EdgeSorter(work_dir, args.sort_buffer_rows, "dst", fan_in, logger, run_id)

    started = time.perf_counter()
    counters = {
        "papers": 0,
        "edges": 0,
        "papers_with_abstract": 0,
        "author_rows": 0,
        "fos_rows": 0,
        "min_paper_id": math.inf,
        "max_paper_id": -1,
    }

    profile_columns = _empty_profile_columns()
    author_columns = _empty_author_columns()
    fos_columns = _empty_fos_columns()
    edge_id = 0
    # The footer vertex bound (paper count) holds only if every citation
    # target is itself a scanned paper, so full runs keep the id set to check.
    paper_ids: set[int] | None = set() if args.max_papers <= 0 else None

    try:
        logger.log("scan source JSON: start")
        for paper in iter_papers(args.source_json, args.max_papers):
            paper_id = paper.get("id")
            if paper_id is None:
                continue
            paper_id = int(paper_id)
            if paper_ids is not None:
                paper_ids.add(paper_id)
            counters["min_paper_id"] = min(counters["min_paper_id"], paper_id)
            counters["max_paper_id"] = max(counters["max_paper_id"], paper_id)

            references = paper.get("references")
            n_references = 0
            if isinstance(references, list):
                for reference in references:
                    if not isinstance(reference, int):
                        continue  # skip malformed reference ids rather than crash mid-stream
                    edge_src_sorter.add(paper_id, reference, edge_id)
                    edge_dst_sorter.add(paper_id, reference, edge_id)
                    edge_id += 1
                    n_references += 1
            counters["edges"] += n_references

            authors = paper.get("authors")
            authors = authors if isinstance(authors, list) else []
            for ordinal, author in enumerate(authors):
                if not isinstance(author, dict):
                    continue
                author_columns["paper_id"].append(paper_id)
                author_columns["author_ordinal"].append(ordinal)
                author_id = author.get("id")
                author_columns["author_id"].append(int(author_id) if author_id is not None else None)
                author_columns["author_name"].append(nullable_text(author.get("name")))
                author_columns["author_org"].append(nullable_text(author.get("org")))
                counters["author_rows"] += 1

            fos_list = paper.get("fos")
            fos_list = fos_list if isinstance(fos_list, list) else []
            primary_fos = None
            best_weight = -math.inf
            for ordinal, field in enumerate(fos_list):
                if not isinstance(field, dict):
                    continue
                name = nullable_text(field.get("name"))
                weight = field.get("w")
                weight = float(weight) if isinstance(weight, (int, float)) else None
                fos_columns["paper_id"].append(paper_id)
                fos_columns["fos_ordinal"].append(ordinal)
                fos_columns["fos_name"].append(name)
                fos_columns["fos_weight"].append(weight)
                counters["fos_rows"] += 1
                if name is not None and weight is not None and weight > best_weight:
                    best_weight = weight
                    primary_fos = name
            if primary_fos is None and fos_list:
                first = fos_list[0]
                if isinstance(first, dict):
                    primary_fos = nullable_text(first.get("name"))

            venue = paper.get("venue")
            venue_raw = venue_id = venue_type = None
            if isinstance(venue, dict):
                venue_raw = nullable_text(venue.get("raw"))
                raw_venue_id = venue.get("id")
                venue_id = int(raw_venue_id) if raw_venue_id is not None else None
                venue_type = nullable_text(venue.get("type"))
            elif venue is not None:
                venue_raw = nullable_text(venue)

            abstract_text = reconstruct_abstract(paper.get("indexed_abstract"))
            if abstract_text is not None:
                counters["papers_with_abstract"] += 1

            year = paper.get("year")
            year = int(year) if isinstance(year, int) else None
            n_citation = paper.get("n_citation")
            n_citation = int(n_citation) if isinstance(n_citation, int) else None

            _append_profile_row(
                profile_columns,
                paper_id=paper_id,
                title=nullable_text(paper.get("title")),
                abstract_text=abstract_text,
                venue=venue_raw,
                venue_id=venue_id,
                venue_type=venue_type,
                year=year,
                n_citation=n_citation,
                doc_type=nullable_text(paper.get("doc_type")),
                publisher=nullable_text(paper.get("publisher")),
                n_authors=len(authors),
                n_references=n_references,
                primary_fos=primary_fos,
            )

            counters["papers"] += 1
            if counters["papers"] % 65_536 == 0:
                elapsed = time.perf_counter() - started
                logger.maybe(
                    "scan.source",
                    (
                        f"scan source JSON: papers={counters['papers']:,}/{expected_scan_papers:,} "
                        f"({_percent(counters['papers'], expected_scan_papers)}) "
                        f"edges={counters['edges']:,} authors={counters['author_rows']:,} "
                        f"fos={counters['fos_rows']:,} rate={_rate(counters['papers'], elapsed, 'papers')}"
                    ),
                )
            # Drain accumulated column batches into the per-table sorters in chunks (the sorters
            # own the spill-to-disk buffering); the caller just avoids per-row Python-call overhead.
            if len(profile_columns["paper_id"]) >= 65_536:
                profile_sorter.add_columns(profile_columns)
                _reset_columns(profile_columns)
            if len(author_columns["paper_id"]) >= 65_536:
                author_sorter.add_columns(author_columns)
                _reset_columns(author_columns)
            if len(fos_columns["paper_id"]) >= 65_536:
                fos_sorter.add_columns(fos_columns)
                _reset_columns(fos_columns)

        # Drain the tail of each column buffer into its sorter.
        logger.log(
            f"scan source JSON: complete papers={counters['papers']:,} edges={counters['edges']:,} "
            f"authors={counters['author_rows']:,} fos={counters['fos_rows']:,}"
        )
        validate_full_counts(spec, counters, args.max_papers)
        profile_sorter.add_columns(profile_columns)
        author_sorter.add_columns(author_columns)
        fos_sorter.add_columns(fos_columns)

        logger.log("write node Parquet: profiles_by_paper_id.parquet")
        _write_sorted_node_table(
            profile_sorter,
            parquet_dir / "profiles_by_paper_id.parquet",
            sorting_columns(PROFILE_SCHEMA, ["paper_id"]),
            use_dictionary=list(PROFILE_DICTIONARY_COLUMNS),
            bloom=bloom_options(PROFILE_BLOOM_COLUMNS, args.profile_row_group_size, ndv_papers),
            row_group_size=args.profile_row_group_size,
            keep_work=args.keep_work,
        )
        logger.log("write node Parquet: paper_authors.parquet")
        _write_sorted_node_table(
            author_sorter,
            parquet_dir / "paper_authors.parquet",
            sorting_columns(AUTHOR_SCHEMA, ["paper_id", "author_ordinal"]),
            use_dictionary=["author_org"],
            bloom=bloom_options(AUTHOR_BLOOM_COLUMNS, args.row_group_size, ndv_papers * 3),
            row_group_size=args.row_group_size,
            keep_work=args.keep_work,
        )
        logger.log("write node Parquet: paper_fos.parquet")
        _write_sorted_node_table(
            fos_sorter,
            parquet_dir / "paper_fos.parquet",
            sorting_columns(FOS_SCHEMA, ["paper_id", "fos_ordinal"]),
            use_dictionary=["fos_name"],
            bloom=bloom_options(FOS_BLOOM_COLUMNS, args.row_group_size, ndv_papers * 5),
            row_group_size=args.row_group_size,
            keep_work=args.keep_work,
        )
    except BaseException:
        # Best-effort spill cleanup on failure so a crashed run does not leak temp files.
        if not args.keep_work:
            for sorter in (profile_sorter, author_sorter, fos_sorter, edge_src_sorter, edge_dst_sorter):
                sorter.cleanup()
        raise

    edge_layouts: dict[str, Any] = {}
    graph_vertex_upper_bound = full_edge_rowset_vertex_upper_bound(counters, args.max_papers)
    try:
        for order, sorter in (("src", edge_src_sorter), ("dst", edge_dst_sorter)):
            logger.log(f"write edge Parquet: edges_by_{order}.parquet")
            edge_meta = _write_sorted_edges(
                sorter,
                parquet_dir,
                order,
                ndv_edges,
                graph_vertex_upper_bound,
                paper_ids,
                args,
            )
            edge_layouts[order] = {
                "sort_order": ["src", "dst", "edge_id"] if order == "src" else ["dst", "src", "edge_id"],
                "clustered_for": f"{order}-range filters and row-group / bloom pruning",
                "footer_metadata": edge_meta.get("footer_metadata", {}),
                "parquet": edge_meta,
            }
    except BaseException:
        if not args.keep_work:
            for sorter in (edge_src_sorter, edge_dst_sorter):
                sorter.cleanup()
        raise

    logger.log("read Parquet metadata: start")
    profile_meta = parquet_metadata(parquet_dir / "profiles_by_paper_id.parquet")
    author_meta = parquet_metadata(parquet_dir / "paper_authors.parquet")
    fos_meta = parquet_metadata(parquet_dir / "paper_fos.parquet")
    logger.log("read Parquet metadata: complete")

    if not args.keep_work and work_dir.exists() and not any(work_dir.iterdir()):
        work_dir.rmdir()

    manifest = {
        "generator": GENERATOR,
        "generated_at_unix": int(time.time()),
        "repo_root": str(REPO_ROOT),
        "dataset": {
            "dataset_id": spec.dataset_id,
            "display_name": spec.display_name,
            "source": spec.source,
            "directed": spec.directed,
            "graph_kind": spec.graph_kind,
            "expected_papers": spec.expected_papers,
            "expected_edges": spec.expected_edges,
            "notes": spec.notes,
        },
        "root": str(root),
        "max_papers": None if args.max_papers <= 0 else args.max_papers,
        "row_group_size": args.row_group_size,
        "profile_row_group_size": args.profile_row_group_size,
        "bucket_width": args.bucket_width,
        "bloom_filter_columns": {
            "edges": list(EDGE_BLOOM_COLUMNS),
            "profiles": list(PROFILE_BLOOM_COLUMNS),
            "paper_authors": list(AUTHOR_BLOOM_COLUMNS),
            "paper_fos": list(FOS_BLOOM_COLUMNS),
        },
        "counts": {
            "papers": counters["papers"],
            "edges": counters["edges"],
            "papers_with_abstract": counters["papers_with_abstract"],
            "author_rows": counters["author_rows"],
            "fos_rows": counters["fos_rows"],
            "min_paper_id": None if counters["papers"] == 0 else int(counters["min_paper_id"]),
            "max_paper_id": None if counters["papers"] == 0 else int(counters["max_paper_id"]),
        },
        "edge_layouts": edge_layouts,
        "profiles": profile_meta,
        "paper_authors": author_meta,
        "paper_fos": fos_meta,
        "sql_registration": _slice_sql_registration(spec, parquet_dir),
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }

    if args.skip_sha256:
        logger.log("source hash: skipped (--skip-sha256)")
        source_hash = None
    else:
        source_hash = sha256(args.source_json, logger)
    source_payload = {
        "source": spec.source,
        "retrieved_at_unix": int(time.time()),
        "source_json": {
            "path": str(args.source_json),
            "bytes": args.source_json.stat().st_size,
            "sha256": source_hash,
        },
        "dataset_id": spec.dataset_id,
    }
    logger.log(f"write source metadata: {root / 'source.json'}")
    write_json(root / "source.json", source_payload)
    if edge_slice_count > 1:
        slice_entries = write_multi_graph_slices(spec, args, root, source_payload, logger)
        manifest["multi_graph_slices"] = {
            "partition": MULTI_GRAPH_SLICE_PARTITION,
            "slice_count": edge_slice_count,
            "hash_seed": edge_slice_seed,
            "slices": slice_entries,
        }
    logger.log(f"write manifest: {root / 'manifest.json'}")
    write_json(root / "manifest.json", manifest)
    logger.log(f"complete elapsed_ms={manifest['elapsed_ms']:.3f}")

    return manifest


def _write_sorted_edges(
    sorter: EdgeSorter,
    parquet_dir: Path,
    order: str,
    ndv: int,
    graph_vertex_upper_bound: int | None,
    paper_ids: set[int] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    path = parquet_dir / f"edges_by_{order}.parquet"
    writer = open_edge_writer(path, order, args.row_group_size, ndv, graph_vertex_upper_bound)
    src_buf: list[int] = []
    dst_buf: list[int] = []
    id_buf: list[int] = []
    written = 0
    previous_endpoint_pair: tuple[int, int] | None = None
    started = time.perf_counter()
    if sorter.logger is not None:
        sorter.logger.log(f"edges_by_{order}: stream merge/write start path={path}")
    try:
        for src, dst, edge in sorter.merged_rows():
            endpoint_pair = (src, dst)
            if endpoint_pair == previous_endpoint_pair:
                raise SystemExit(
                    f"edges_by_{order} contains duplicate directed edge ({src}, {dst}); "
                    "cannot emit a simple-directed topology certificate"
                )
            previous_endpoint_pair = endpoint_pair
            if graph_vertex_upper_bound is not None and dst not in paper_ids:
                raise SystemExit(
                    f"edges_by_{order} references paper {dst}, which is not in the source; "
                    f"cannot emit {GRAPH_VERTEX_UPPER_BOUND_METADATA_KEY}={graph_vertex_upper_bound}"
                )
            src_buf.append(src)
            dst_buf.append(dst)
            id_buf.append(edge)
            if len(src_buf) >= args.row_group_size:
                batch_rows = len(src_buf)
                writer.write_table(
                    edge_batch(src_buf, dst_buf, id_buf, args.bucket_width),
                    row_group_size=args.row_group_size,
                )
                written += batch_rows
                if sorter.logger is not None:
                    sorter.logger.maybe(
                        f"edges.{order}.write",
                        f"edges_by_{order}: wrote rows={written:,}/{sorter.rows_added:,} "
                        f"({_percent(written, sorter.rows_added)}) "
                        f"rate={_rate(written, time.perf_counter() - started, 'rows')}",
                    )
                src_buf.clear()
                dst_buf.clear()
                id_buf.clear()
        if src_buf:
            batch_rows = len(src_buf)
            writer.write_table(
                edge_batch(src_buf, dst_buf, id_buf, args.bucket_width),
                row_group_size=args.row_group_size,
            )
            written += batch_rows
    finally:
        writer.close()
        if not args.keep_work:
            sorter.cleanup()
    if sorter.logger is not None:
        sorter.logger.log(f"edges_by_{order}: write complete rows={written:,}")
    return parquet_metadata(path)


# ---- column buffers ----


def _empty_profile_columns() -> dict[str, list[Any]]:
    return {field.name: [] for field in PROFILE_SCHEMA}


def _empty_author_columns() -> dict[str, list[Any]]:
    return {field.name: [] for field in AUTHOR_SCHEMA}


def _empty_fos_columns() -> dict[str, list[Any]]:
    return {field.name: [] for field in FOS_SCHEMA}


def _append_profile_row(columns: dict[str, list[Any]], **values: Any) -> None:
    year = values.get("year")
    values["year_bucket"] = (year // 10) * 10 if isinstance(year, int) else None
    for field in PROFILE_SCHEMA:
        columns[field.name].append(values.get(field.name))


def _reset_columns(columns: dict[str, list[Any]]) -> None:
    for values in columns.values():
        values.clear()


def _write_sorted_node_table(
    sorter: RowSorter,
    path: Path,
    sort_cols: list[pq.SortingColumn],
    use_dictionary: list[str],
    bloom: dict[str, Any],
    row_group_size: int,
    keep_work: bool,
) -> None:
    if sorter.logger is not None:
        sorter.logger.log(f"{sorter.name}: open Parquet writer path={path}")
    writer = pq.ParquetWriter(
        path,
        sorter.schema,
        compression="snappy",
        version="2.6",
        write_statistics=True,
        write_page_index=True,
        use_dictionary=use_dictionary,
        sorting_columns=sort_cols,
        bloom_filter_options=bloom,
    )
    try:
        sorter.write_sorted(writer, row_group_size)
    finally:
        writer.close()
        if not keep_work:
            sorter.cleanup()
    if sorter.logger is not None:
        sorter.logger.log(f"{sorter.name}: closed Parquet writer path={path}")


def main() -> int:
    args = parse_args()
    require_positive("--row-group-size", args.row_group_size)
    require_positive("--profile-row-group-size", args.profile_row_group_size)
    require_positive("--bucket-width", args.bucket_width)
    require_positive("--sort-buffer-rows", args.sort_buffer_rows)
    require_positive("--edge-slice-count", args.edge_slice_count)
    if args.max_papers < 0:
        raise SystemExit("--max-papers must be >= 0")
    if args.edge_slice_seed < 0 or args.edge_slice_seed > _U64_MASK:
        raise SystemExit("--edge-slice-seed must be an unsigned 64-bit integer")
    if args.progress_interval < 0.0:
        raise SystemExit("--progress-interval must be >= 0")
    if not args.source_json.exists():
        raise SystemExit(f"source JSON not found: {args.source_json}")

    if args.merge_fan_in < 2:
        raise SystemExit("--merge-fan-in must be >= 2")

    spec = DatasetSpec()
    started = time.perf_counter()
    manifest = convert(spec, args)

    summary = {
        "dataset": spec.dataset_id,
        "root": manifest.get("root"),
        "papers": manifest["counts"]["papers"],
        "edges": manifest["counts"]["edges"],
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
