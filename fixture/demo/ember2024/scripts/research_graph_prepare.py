#!/usr/bin/env python3
"""Prepare deterministic, label-independent EMBER2024 graph inputs."""

from __future__ import annotations

import importlib.metadata
import json
import os
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = WORKSPACE_ROOT / "target" / "ember2024-research"
PROVENANCE_PATH = WORKSPACE_ROOT / "docs" / "research" / "ember2024" / "results" / "graph_preparation.json"
SYMBOL_ID_BASE = 100_000_000
MIN_SYMBOL_SAMPLES = 3
MAX_SYMBOL_SAMPLES = 100
SAMPLE_ID_MIN = 20_000_000
SAMPLE_ID_MAX = 30_000_000


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def atomic_parquet(name: str, table: pa.Table) -> dict[str, Any]:
    path = OUTPUT_DIR / name
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=OUTPUT_DIR)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        pq.write_table(table, tmp_path, compression="zstd", row_group_size=100_000)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return {"path": path.relative_to(WORKSPACE_ROOT).as_posix(), "rows": table.num_rows, "size_bytes": path.stat().st_size}


def arrow_table(columns: dict[str, list[Any]], fields: list[pa.Field]) -> pa.Table:
    arrays = [pa.array(columns[field.name], type=field.type) for field in fields]
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def distribution(values: list[int]) -> dict[str, int | float]:
    if not values:
        return {"count": 0}
    counts = Counter(values)
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": ordered[0],
        "p25": float(np.quantile(ordered, 0.25)),
        "median": float(np.quantile(ordered, 0.5)),
        "p75": float(np.quantile(ordered, 0.75)),
        "max": ordered[-1],
        "mean": float(np.mean(ordered)),
        "zero_count": counts.get(0, 0),
    }


def graph_profile(graph: nx.Graph | nx.DiGraph) -> dict[str, Any]:
    degrees = [int(degree) for _, degree in graph.degree()]
    components = list(nx.weakly_connected_components(graph)) if graph.is_directed() else list(nx.connected_components(graph))
    component_sizes = sorted((len(component) for component in components), reverse=True)
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "components": len(component_sizes),
        "largest_component_nodes": component_sizes[0] if component_sizes else 0,
        "component_size_distribution": distribution(component_sizes),
        "degree_distribution": distribution(degrees),
        "density": float(nx.density(graph)) if graph.number_of_nodes() > 1 else 0.0,
    }


def prepare_import_graph() -> tuple[dict[str, Any], dict[int, int], dict[int, str]]:
    metadata_path = OUTPUT_DIR / "input_metadata_challenge.parquet"
    imports_path = OUTPUT_DIR / "input_imports_challenge.parquet"
    metadata = pq.read_table(metadata_path, columns=["id", "sha256", "file_type"]).to_pydict()
    win32_sha = {
        int(sample_id): str(sha256)
        for sample_id, sha256, file_type in zip(metadata["id"], metadata["sha256"], metadata["file_type"])
        if file_type == "Win32"
    }
    if len(win32_sha) != 3_225:
        raise ValueError(f"expected 3,225 Win32 challenge rows, found {len(win32_sha)}")
    sample_symbols: dict[int, set[str]] = {sample_id: set() for sample_id in win32_sha}
    parquet = pq.ParquetFile(imports_path)
    for batch in parquet.iter_batches(columns=["id", "symbol"], batch_size=100_000):
        ids = batch.column("id").to_pylist()
        symbols = batch.column("symbol").to_pylist()
        for sample_id, symbol in zip(ids, symbols):
            sample_id = int(sample_id)
            if sample_id in sample_symbols:
                sample_symbols[sample_id].add(str(symbol))

    symbol_support: Counter[str] = Counter()
    for symbols in sample_symbols.values():
        symbol_support.update(symbols)
    retained_symbols = sorted(
        symbol for symbol, support in symbol_support.items() if MIN_SYMBOL_SAMPLES <= support <= MAX_SYMBOL_SAMPLES
    )
    symbol_ids = {symbol: SYMBOL_ID_BASE + offset for offset, symbol in enumerate(retained_symbols)}
    if symbol_ids and max(symbol_ids.values()) > np.iinfo(np.int32).max:
        raise ValueError("symbol IDs exceed int32")

    sample_ids = sorted(win32_sha)
    unfiltered_counts = {sample_id: len(sample_symbols[sample_id]) for sample_id in sample_ids}
    retained_counts = {
        sample_id: sum(symbol in symbol_ids for symbol in sample_symbols[sample_id]) for sample_id in sample_ids
    }
    edge_src: list[int] = []
    edge_dst: list[int] = []
    for sample_id in sample_ids:
        for symbol in sorted(sample_symbols[sample_id]):
            symbol_id = symbol_ids.get(symbol)
            if symbol_id is not None:
                edge_src.append(sample_id)
                edge_dst.append(symbol_id)

    symbol_rows = [
        (symbol_ids[symbol], symbol, int(symbol_support[symbol])) for symbol in retained_symbols
    ]
    exclusions = [
        (symbol, int(support), "below_minimum_sample_support" if support < MIN_SYMBOL_SAMPLES else "above_maximum_sample_support")
        for symbol, support in sorted(symbol_support.items())
        if symbol not in symbol_ids
    ]

    outputs = {
        "edges": atomic_parquet(
            "input_48_challenge_import_edges.parquet",
            arrow_table(
                {"src_id": edge_src, "dst_id": edge_dst},
                [pa.field("src_id", pa.int32(), nullable=False), pa.field("dst_id", pa.int32(), nullable=False)],
            ),
        ),
        "symbols": atomic_parquet(
            "input_48_challenge_import_symbols.parquet",
            arrow_table(
                {
                    "symbol_id": [row[0] for row in symbol_rows],
                    "symbol": [row[1] for row in symbol_rows],
                    "distinct_sample_count": [row[2] for row in symbol_rows],
                },
                [
                    pa.field("symbol_id", pa.int32(), nullable=False),
                    pa.field("symbol", pa.string(), nullable=False),
                    pa.field("distinct_sample_count", pa.int32(), nullable=False),
                ],
            ),
        ),
        "sample_counts": atomic_parquet(
            "input_48_challenge_import_sample_counts.parquet",
            arrow_table(
                {
                    "id": sample_ids,
                    "unfiltered_import_count": [unfiltered_counts[sample_id] for sample_id in sample_ids],
                    "retained_import_count": [retained_counts[sample_id] for sample_id in sample_ids],
                },
                [
                    pa.field("id", pa.int32(), nullable=False),
                    pa.field("unfiltered_import_count", pa.int32(), nullable=False),
                    pa.field("retained_import_count", pa.int32(), nullable=False),
                ],
            ),
        ),
        "symbol_counts": atomic_parquet(
            "input_48_challenge_import_symbol_counts.parquet",
            arrow_table(
                {"symbol": sorted(symbol_support), "distinct_sample_count": [int(symbol_support[s]) for s in sorted(symbol_support)]},
                [pa.field("symbol", pa.string(), nullable=False), pa.field("distinct_sample_count", pa.int32(), nullable=False)],
            ),
        ),
        "exclusions": atomic_parquet(
            "input_48_challenge_import_exclusions.parquet",
            arrow_table(
                {"symbol": [row[0] for row in exclusions], "distinct_sample_count": [row[1] for row in exclusions], "reason": [row[2] for row in exclusions]},
                [
                    pa.field("symbol", pa.string(), nullable=False),
                    pa.field("distinct_sample_count", pa.int32(), nullable=False),
                    pa.field("reason", pa.string(), nullable=False),
                ],
            ),
        ),
    }

    graph = nx.Graph()
    graph.add_nodes_from(sample_ids, bipartite=0)
    graph.add_nodes_from(symbol_ids.values(), bipartite=1)
    graph.add_edges_from(zip(edge_src, edge_dst))
    retained_per_sample = [retained_counts[sample_id] for sample_id in sample_ids]
    selected_seed_candidates = [
        sample_id for sample_id in sample_ids if 10 <= retained_counts[sample_id] <= 100
    ]
    return (
        {
            "outputs": outputs,
            "counts": {
                "win32_challenge_samples": len(sample_ids),
                "input_import_rows_for_win32_samples": sum(unfiltered_counts.values()),
                "samples_with_zero_imports": sum(count == 0 for count in unfiltered_counts.values()),
                "distinct_symbols_before_filter": len(symbol_support),
                "retained_symbols": len(retained_symbols),
                "excluded_symbols": len(exclusions),
                "excluded_below_minimum": sum(reason == "below_minimum_sample_support" for _, _, reason in exclusions),
                "excluded_above_maximum": sum(reason == "above_maximum_sample_support" for _, _, reason in exclusions),
                "retained_bipartite_edges": len(edge_src),
            },
            "distributions": {
                "unfiltered_imports_per_sample": distribution([unfiltered_counts[s] for s in sample_ids]),
                "retained_imports_per_sample": distribution(retained_per_sample),
                "samples_per_symbol_before_filter": distribution(list(symbol_support.values())),
                "samples_per_retained_symbol": distribution([symbol_support[s] for s in retained_symbols]),
            },
            "networkx": graph_profile(graph),
            "private": {
                "sample_sha256": win32_sha,
                "retained_counts": retained_counts,
                "candidate_ids_for_seed": selected_seed_candidates,
            },
        },
        retained_counts,
        win32_sha,
    )


def prepare_knn_graph(retained_counts: dict[int, int], sample_sha256: dict[int, str]) -> dict[str, Any]:
    neighbor_path = OUTPUT_DIR / "47_challenge_exact_neighbors.parquet"
    neighbors = pq.read_table(neighbor_path, columns=["query_id", "rank", "neighbor_id", "distance"]).to_pydict()
    by_query: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    for query_id, rank, neighbor_id, distance in zip(
        neighbors["query_id"], neighbors["rank"], neighbors["neighbor_id"], neighbors["distance"]
    ):
        by_query[int(query_id)].append((int(rank), int(neighbor_id), float(distance)))

    directed_rows: list[tuple[int, int, int, float]] = []
    all_neighbor_nodes: set[int] = set()
    rank_counts: Counter[int] = Counter()
    self_rows_removed = 0
    nonself_rows_capped = 0
    for query_id in sorted(by_query):
        rows = sorted(by_query[query_id])
        all_neighbor_nodes.add(query_id)
        if len(rows) != 6 or [row[0] for row in rows] != [1, 2, 3, 4, 5, 6]:
            raise ValueError(f"query {query_id} must have exactly one row for each rank 1 through 6")
        nonself_rows = []
        for rank, neighbor_id, distance in rows:
            rank_counts[rank] += 1
            all_neighbor_nodes.add(neighbor_id)
            if neighbor_id == query_id:
                self_rows_removed += 1
                continue
            if not (SAMPLE_ID_MIN <= query_id < SAMPLE_ID_MAX and SAMPLE_ID_MIN <= neighbor_id < SAMPLE_ID_MAX):
                raise ValueError("kNN node ID does not fit the reserved sample ID domain")
            nonself_rows.append((query_id, neighbor_id, rank, distance))
        if len(nonself_rows) > 5:
            nonself_rows_capped += len(nonself_rows) - 5
        directed_rows.extend(nonself_rows[:5])

    expected_nodes = set(sample_sha256)
    if all_neighbor_nodes != expected_nodes:
        raise ValueError(
            f"47 nodes do not match Win32 challenge nodes: neighbors={len(all_neighbor_nodes)}, metadata={len(expected_nodes)}"
        )
    if len(directed_rows) != 5 * len(by_query):
        raise ValueError(f"expected five nonself directed neighbors per query, got {len(directed_rows)}")

    min_distance_by_pair: dict[tuple[int, int], float] = {}
    directed = nx.DiGraph()
    directed.add_nodes_from(sorted(expected_nodes))
    directed.add_weighted_edges_from((src, dst, distance) for src, dst, _, distance in directed_rows)
    for src, dst, _, distance in directed_rows:
        pair = (min(src, dst), max(src, dst))
        clipped_distance = max(0.0, distance)
        previous = min_distance_by_pair.get(pair)
        if previous is None or clipped_distance < previous:
            min_distance_by_pair[pair] = clipped_distance

    pair_rows = sorted((src, dst, distance) for (src, dst), distance in min_distance_by_pair.items())
    directed_sorted = sorted(directed_rows, key=lambda row: (row[0], row[2], row[1]))
    undirected = nx.Graph()
    undirected.add_nodes_from(sorted(expected_nodes))
    undirected.add_weighted_edges_from((src, dst, distance) for src, dst, distance in pair_rows)

    outputs = {
        "directed": atomic_parquet(
            "input_49_challenge_knn_directed.parquet",
            arrow_table(
                {
                    "src_id": [row[0] for row in directed_sorted],
                    "dst_id": [row[1] for row in directed_sorted],
                    "rank": [row[2] for row in directed_sorted],
                    "distance": [row[3] for row in directed_sorted],
                },
                [
                    pa.field("src_id", pa.int32(), nullable=False),
                    pa.field("dst_id", pa.int32(), nullable=False),
                    pa.field("rank", pa.uint32(), nullable=False),
                    pa.field("distance", pa.float32(), nullable=False),
                ],
            ),
        ),
        "affinity": atomic_parquet(
            "input_49_challenge_knn_affinity.parquet",
            arrow_table(
                {
                    "src_id": [row[0] for row in pair_rows],
                    "dst_id": [row[1] for row in pair_rows],
                    "weight": [1.0 / (1.0 + row[2]) for row in pair_rows],
                },
                [
                    pa.field("src_id", pa.int32(), nullable=False),
                    pa.field("dst_id", pa.int32(), nullable=False),
                    pa.field("weight", pa.float32(), nullable=False),
                ],
            ),
        ),
        "distance": atomic_parquet(
            "input_49_challenge_knn_distance.parquet",
            arrow_table(
                {
                    "src_id": [row[0] for row in pair_rows],
                    "dst_id": [row[1] for row in pair_rows],
                    "distance": [row[2] for row in pair_rows],
                },
                [
                    pa.field("src_id", pa.int32(), nullable=False),
                    pa.field("dst_id", pa.int32(), nullable=False),
                    pa.field("distance", pa.float32(), nullable=False),
                ],
            ),
        ),
    }

    degrees = dict(undirected.degree())
    seed_candidates = [
        sample_id
        for sample_id in expected_nodes
        if 10 <= retained_counts[sample_id] <= 100 and degrees[sample_id] >= 5
    ]
    if not seed_candidates:
        raise ValueError("no graph seed candidate satisfies import count and kNN degree constraints")
    seed_id = min(seed_candidates, key=lambda sample_id: sample_sha256[sample_id])
    outputs["seed"] = atomic_parquet(
        "input_49_challenge_graph_seed.parquet",
        arrow_table(
            {
                "id": [seed_id],
                "sha256": [sample_sha256[seed_id]],
                "retained_import_count": [retained_counts[seed_id]],
                "knn_degree": [int(degrees[seed_id])],
            },
            [
                pa.field("id", pa.int32(), nullable=False),
                pa.field("sha256", pa.string(), nullable=False),
                pa.field("retained_import_count", pa.int32(), nullable=False),
                pa.field("knn_degree", pa.int32(), nullable=False),
            ],
        ),
    )
    return {
        "outputs": outputs,
        "counts": {
            "nodes": len(expected_nodes),
            "queries": len(by_query),
            "raw_neighbor_rows": len(neighbors["query_id"]),
            "rank_counts": {str(rank): rank_counts[rank] for rank in sorted(rank_counts)},
            "removed_self_rows": self_rows_removed,
            "nonself_rows_removed_by_k_cap": nonself_rows_capped,
            "directed_nonself_edges": len(directed_rows),
            "canonical_undirected_union_edges": len(pair_rows),
        },
        "distributions": {
            "directed_out_degree": distribution([int(degree) for _, degree in directed.out_degree()]),
            "undirected_degree": distribution([int(degree) for _, degree in undirected.degree()]),
        },
        "networkx": {
            "directed": graph_profile(directed),
            "undirected_union": graph_profile(undirected),
        },
        "deduplication": "canonical pair is (min(src,dst), max(src,dst)); duplicate directions collapse to one pair and distance is the minimum clipped directional distance",
        "affinity": "1 / (1 + canonical undirected distance)",
        "seed_selection": {
            "eligibility": "retained import count in [10, 100] and undirected kNN degree >= 5",
            "selection": "eligible row with lexicographically smallest SHA256",
            "eligible_candidate_count": len(seed_candidates),
            "selected_id": seed_id,
            "selected_sha256": sample_sha256[seed_id],
        },
    }


def main() -> None:
    started = time.monotonic()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    import_result, retained_counts, sample_sha256 = prepare_import_graph()
    print(
        f"import graph ready: {import_result['counts']['retained_bipartite_edges']:,} edges, "
        f"{import_result['counts']['retained_symbols']:,} symbols",
        flush=True,
    )
    knn_result = prepare_knn_graph(retained_counts, sample_sha256)
    print(
        f"kNN graph ready: {knn_result['counts']['directed_nonself_edges']:,} directed, "
        f"{knn_result['counts']['canonical_undirected_union_edges']:,} undirected edges",
        flush=True,
    )
    provenance = {
        "dataset": "EMBER2024",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workspace_root": str(WORKSPACE_ROOT),
        "packages": {"python": sys.version.split()[0], "numpy": version("numpy"), "pyarrow": version("pyarrow"), "networkx": version("networkx")},
        "label_use": "labels and families were excluded from graph edge construction and seed eligibility",
        "import_graph": {
            "sources": ["input_metadata_challenge.parquet (id,sha256,file_type)", "input_imports_challenge.parquet (id,symbol)"],
            "symbol_support_rule": f"retain symbols observed in {MIN_SYMBOL_SAMPLES}..{MAX_SYMBOL_SAMPLES} distinct Win32 challenge samples, inclusive",
            "symbol_id_rule": f"lexically sorted retained symbol strings map to int32 IDs starting at {SYMBOL_ID_BASE}",
            "edge_direction": "sample ID to import symbol ID",
            "symbol_support_is_per_sample": True,
            **import_result,
        },
        "knn_graph": {
            "source": "47_challenge_exact_neighbors.parquet, columns query_id,rank,neighbor_id,distance",
            "directed_rule": "remove any row whose neighbor_id equals query_id, then retain the first five remaining rows in original rank order; this handles cases where rank 1 is not self or no self row appears among the six stored neighbors",
            "undirected_pair_rule": "canonical min/max ID union across directed edges, exact pairs deduplicated",
            **knn_result,
        },
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    provenance["import_graph"].pop("private", None)
    provenance_tmp = PROVENANCE_PATH.with_suffix(".json.tmp")
    provenance_tmp.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf8")
    os.replace(provenance_tmp, PROVENANCE_PATH)
    print(f"provenance: {PROVENANCE_PATH}", flush=True)
    for graph_name, result in (("import", import_result), ("knn", knn_result)):
        for output_name, output in result["outputs"].items():
            print(f"{graph_name} {output_name}: {output['rows']:,} rows, {output['size_bytes']:,} bytes", flush=True)
    print(f"elapsed: {provenance['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
