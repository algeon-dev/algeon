#!/usr/bin/env python3
"""Summarize saved MerRec community results with CPU graph references."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform

import networkx as nx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
from sklearn.metrics import adjusted_rand_score


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "target/merrec-exploration/results"
NODES_PATH = RESULTS / "prepare_nodes.parquet"
UNDIRECTED_PATH = RESULTS / "prepare_undirected.parquet"
EDGES_PATH = RESULTS / "prepare_edges.parquet"
PAGERANK_PATH = RESULTS / "11_pagerank_full.parquet"
REPORT = ROOT / "docs/research/merrec/community_validation.json"
QUERY_RECORDS = ROOT / "docs/research/merrec/results"
PAGERANK_TOLERANCE = 1e-6

COMMUNITY_RESULTS = {
    "cugraph_louvain": RESULTS / "cugraph_louvain_full.parquet",
    "cugraph_leiden": RESULTS / "cugraph_leiden_full.parquet",
    "cugraph_ecg": RESULTS / "cugraph_ecg_full.parquet",
    "cugraph_spectral_modularity_maximization": (
        RESULTS / "cugraph_spectral_modularity_maximization_full.parquet"
    ),
}
COMPARISON_METHODS = ("cugraph_louvain", "cugraph_leiden", "cugraph_ecg")
COMPOSITION_PATH = RESULTS / "24_market_collections_final.parquet"


def query_reference(name: str) -> dict[str, object]:
    path = QUERY_RECORDS / f"{name}.json"
    if not path.is_file():
        return {"path": str(path.relative_to(ROOT)), "status": "missing"}
    record = json.loads(path.read_text())
    return {
        "path": str(path.relative_to(ROOT)),
        "status": record.get("status"),
        "query_id": record.get("query_id"),
        "rows": record.get("rows"),
        "error": record.get("error"),
    }


def require_columns(
    table: pa.Table,
    names: list[str],
    path: Path,
    non_null: list[str] | None = None,
) -> None:
    missing = [name for name in names if name not in table.column_names]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    nulls = [name for name in (names if non_null is None else non_null) if table[name].null_count]
    if nulls:
        raise ValueError(f"{path} has null values in required columns: {nulls}")


def records_by_id(
    table: pa.Table, key_column: str, path: Path
) -> dict[int, dict[str, object]]:
    output: dict[int, dict[str, object]] = {}
    for row in table.to_pylist():
        key = int(row[key_column])
        if key in output:
            raise ValueError(f"{path} contains duplicate {key_column} values")
        output[key] = row
    return output


def top_counts(values: list[str | None], denominator: int, limit: int = 3) -> list[dict[str, object]]:
    counts = Counter(value for value in values if value is not None)
    return [
        {
            "name": name,
            "market_count": count,
            "share_of_community": count / denominator if denominator else 0.0,
        }
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def community_summary(
    label: int,
    member_ids: list[int],
    nodes: dict[int, dict[str, object]],
    pagerank: dict[int, float],
    graph: nx.Graph,
) -> dict[str, object]:
    community_size = len(member_ids)
    member_nodes = [nodes[vertex] for vertex in member_ids]
    known_brands = [row["brand_name"] for row in member_nodes if row["brand_name"] is not None]
    induced_components = list(nx.connected_components(graph.subgraph(member_ids)))
    induced_component_sizes = sorted((len(part) for part in induced_components), reverse=True)
    representatives = sorted(
        member_ids,
        key=lambda vertex: (-pagerank[vertex], str(nodes[vertex]["product_id"])),
    )[:5]
    return {
        "community_label": label,
        "community_size": community_size,
        "leading_categories": top_counts(
            [row["category"] for row in member_nodes], community_size
        ),
        "leading_brands": top_counts(known_brands, community_size),
        "markets_with_brand": len(known_brands),
        "markets_without_brand": community_size - len(known_brands),
        "induced_connected_components": len(induced_components),
        "largest_induced_component_size": induced_component_sizes[0],
        "representative_markets_by_weighted_pagerank": [
            {
                "vertex": vertex,
                "product_id": nodes[vertex]["product_id"],
                "brand_name": nodes[vertex]["brand_name"],
                "category": nodes[vertex]["category"],
                "leaf_category": nodes[vertex]["leaf_category"],
                "viewers": int(nodes[vertex]["viewers"]),
                "pagerank": pagerank[vertex],
            }
            for vertex in representatives
        ],
    }


def load_graph_inputs() -> tuple[
    nx.Graph,
    dict[int, dict[str, object]],
    dict[int, float],
    dict[str, object],
]:
    required_paths = (NODES_PATH, UNDIRECTED_PATH, EDGES_PATH, PAGERANK_PATH)
    missing = [str(path.relative_to(ROOT)) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required saved graph inputs are missing: {missing}")

    nodes_table = pq.read_table(NODES_PATH)
    require_columns(
        nodes_table,
        ["id", "product_id", "brand_name", "category", "leaf_category", "viewers"],
        NODES_PATH,
        non_null=["id", "product_id", "viewers"],
    )
    nodes = records_by_id(nodes_table, "id", NODES_PATH)
    vertex_ids = set(nodes)

    undirected = pq.read_table(UNDIRECTED_PATH, columns=["src", "dst"])
    require_columns(undirected, ["src", "dst"], UNDIRECTED_PATH)
    graph = nx.Graph()
    graph.add_nodes_from(vertex_ids)
    undirected_edge_count = 0
    for row in undirected.to_pylist():
        source = int(row["src"])
        destination = int(row["dst"])
        if source not in vertex_ids or destination not in vertex_ids:
            raise ValueError(f"{UNDIRECTED_PATH} references an unknown node")
        graph.add_edge(source, destination)
        undirected_edge_count += 1

    directed = pq.read_table(EDGES_PATH, columns=["src", "dst", "weight"])
    require_columns(directed, ["src", "dst", "weight"], EDGES_PATH)
    for row in directed.to_pylist():
        if int(row["src"]) not in vertex_ids or int(row["dst"]) not in vertex_ids:
            raise ValueError(f"{EDGES_PATH} references an unknown node")

    pagerank_table = pq.read_table(PAGERANK_PATH)
    require_columns(pagerank_table, ["vertex", "value"], PAGERANK_PATH)
    pagerank_rows = records_by_id(pagerank_table, "vertex", PAGERANK_PATH)
    pagerank = {vertex: float(row["value"]) for vertex, row in pagerank_rows.items()}
    if set(pagerank) != vertex_ids or not all(math.isfinite(value) for value in pagerank.values()):
        raise ValueError(f"{PAGERANK_PATH} must contain one finite score per node")

    graph_components = sorted(
        (len(part) for part in nx.connected_components(graph)), reverse=True
    )
    source_summary = {
        "nodes_path": str(NODES_PATH.relative_to(ROOT)),
        "node_count": len(nodes),
        "directed_edges_path": str(EDGES_PATH.relative_to(ROOT)),
        "directed_edge_count": directed.num_rows,
        "undirected_edges_path": str(UNDIRECTED_PATH.relative_to(ROOT)),
        "undirected_edge_rows": undirected_edge_count,
        "undirected_graph_unique_edges": graph.number_of_edges(),
        "undirected_component_count": len(graph_components),
        "undirected_component_sizes_top_10": graph_components[:10],
        "pagerank_reference_path": str(PAGERANK_PATH.relative_to(ROOT)),
        "pagerank_reference_rows": len(pagerank),
    }
    return graph, nodes, pagerank, source_summary


def assess_partition(
    name: str,
    path: Path,
    graph: nx.Graph,
    expected_vertices: set[int],
    nodes: dict[int, dict[str, object]],
    pagerank: dict[int, float],
) -> tuple[dict[str, object], dict[int, int] | None]:
    if not path.is_file():
        return (
            {
                "status": "pending",
                "path": str(path.relative_to(ROOT)),
                "reason": "Full community assignment output is absent.",
            },
            None,
        )

    table = pq.read_table(path)
    require_columns(table, ["vertex", "community", "community_size"], path)
    rows = table.to_pylist()
    vertex_ids = [int(row["vertex"]) for row in rows]
    labels = [int(row["community"]) for row in rows]
    reported_sizes = [int(row["community_size"]) for row in rows]
    duplicate_vertices = len(vertex_ids) - len(set(vertex_ids))
    actual_vertices = set(vertex_ids)
    missing_vertices = sorted(expected_vertices - actual_vertices)
    unexpected_vertices = sorted(actual_vertices - expected_vertices)
    assignments = dict(zip(vertex_ids, labels, strict=True))
    members: dict[int, list[int]] = defaultdict(list)
    for vertex, label in assignments.items():
        members[label].append(vertex)
    computed_sizes = {label: len(member_ids) for label, member_ids in members.items()}
    reported_size_mismatches = sum(
        computed_sizes.get(label) != size
        for label, size in zip(labels, reported_sizes, strict=True)
    )
    coverage_matches = (
        len(rows) == len(expected_vertices)
        and duplicate_vertices == 0
        and not missing_vertices
        and not unexpected_vertices
    )
    sizes_match = reported_size_mismatches == 0
    result: dict[str, object] = {
        "status": "assessed" if coverage_matches and sizes_match else "failed",
        "path": str(path.relative_to(ROOT)),
        "rows": len(rows),
        "complete_vertex_coverage": {
            "expected_vertices": len(expected_vertices),
            "actual_rows": len(rows),
            "unique_vertices": len(actual_vertices),
            "duplicate_rows": duplicate_vertices,
            "missing_vertex_count": len(missing_vertices),
            "unexpected_vertex_count": len(unexpected_vertices),
            "missing_vertex_sample": missing_vertices[:10],
            "unexpected_vertex_sample": unexpected_vertices[:10],
            "matches": coverage_matches,
        },
        "community_size_consistency": {
            "community_count": len(computed_sizes),
            "reported_size_mismatch_rows": reported_size_mismatches,
            "matches": sizes_match,
        },
        "partition_sizes_descending": sorted(computed_sizes.values(), reverse=True),
        "partition_quality": None,
        "communities": [],
        "correctness_scope": (
            "Completeness, size fields, modularity, and induced connectivity are assessed. "
            "A single stochastic fit does not establish run-to-run stability or business utility."
        ),
    }
    if coverage_matches and sizes_match:
        partition = [set(member_ids) for member_ids in members.values()]
        modularity = float(nx.algorithms.community.modularity(graph, partition, weight=None))
        summaries = [
            community_summary(label, member_ids, nodes, pagerank, graph)
            for label, member_ids in members.items()
        ]
        summaries.sort(key=lambda row: (-int(row["community_size"]), int(row["community_label"])))
        disconnected = [
            row for row in summaries if int(row["induced_connected_components"]) > 1
        ]
        result["partition_quality"] = {
            "networkx_unweighted_modularity": modularity,
            "modularity_weighting": "unweighted, matching community probe projections of src and dst",
            "induced_community_connectedness": {
                "connected_community_count": len(summaries) - len(disconnected),
                "disconnected_community_count": len(disconnected),
                "extra_induced_components": sum(
                    int(row["induced_connected_components"]) - 1 for row in disconnected
                ),
            },
        }
        result["communities"] = summaries
        return result, assignments
    return result, None


def compare_partitions(
    partitions: dict[str, dict[int, int] | None], expected_vertices: set[int]
) -> dict[str, object]:
    unavailable = [
        name for name in COMPARISON_METHODS
        if partitions.get(name) is None
    ]
    if unavailable:
        return {
            "status": "pending",
            "methods": list(COMPARISON_METHODS),
            "unavailable_methods": unavailable,
            "comparisons": [],
            "interpretation": (
                "Adjusted Rand index compares these single fits across methods. "
                "It does not estimate run-to-run stability."
            ),
        }

    ordered_vertices = sorted(expected_vertices)
    comparisons = []
    for index, left_name in enumerate(COMPARISON_METHODS):
        for right_name in COMPARISON_METHODS[index + 1:]:
            left = partitions[left_name]
            right = partitions[right_name]
            assert left is not None and right is not None
            score = float(
                adjusted_rand_score(
                    [left[vertex] for vertex in ordered_vertices],
                    [right[vertex] for vertex in ordered_vertices],
                )
            )
            comparisons.append(
                {"left_method": left_name, "right_method": right_name, "adjusted_rand_index": score}
            )
    return {
        "status": "assessed",
        "methods": list(COMPARISON_METHODS),
        "vertex_count": len(ordered_vertices),
        "comparisons": comparisons,
        "interpretation": (
            "Adjusted Rand index compares these single fits across methods. "
            "It does not estimate run-to-run stability."
        ),
    }


def assess_composition(
    path: Path,
    nodes: dict[int, dict[str, object]],
    pagerank: dict[int, float],
) -> dict[str, object]:
    if not path.is_file():
        return {
            "status": "pending",
            "path": str(path.relative_to(ROOT)),
            "reason": "The saved 24_market_collections_final query output is absent.",
            "full_leiden_partition_comparison": "not attempted; query fits Leiden independently",
        }

    table = pq.read_table(path)
    columns = [
        "vertex", "community", "community_size", "representative_rank", "product_id",
        "brand_name", "category", "leaf_category", "viewers", "pagerank",
    ]
    require_columns(
        table,
        columns,
        path,
        non_null=[
            "vertex", "community", "community_size", "representative_rank", "product_id",
            "viewers", "pagerank",
        ],
    )
    rows = table.to_pylist()
    ids = [int(row["vertex"]) for row in rows]
    duplicate_representatives = len(ids) - len(set(ids))

    metadata_mismatches: list[dict[str, object]] = []
    pagerank_errors: list[float] = []
    pagerank_nonfinite_rows = 0
    unknown_vertices = 0
    groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        vertex = int(row["vertex"])
        if vertex not in nodes:
            unknown_vertices += 1
            metadata_mismatches.append({"vertex": vertex, "column": "vertex", "reason": "unknown vertex"})
            continue
        node = nodes[vertex]
        for column in ("product_id", "brand_name", "category", "leaf_category", "viewers"):
            expected = int(node[column]) if column == "viewers" else node[column]
            actual = int(row[column]) if column == "viewers" else row[column]
            if actual != expected:
                metadata_mismatches.append(
                    {"vertex": vertex, "column": column, "expected": expected, "actual": actual}
                )
        observed_pagerank = float(row["pagerank"])
        if not math.isfinite(observed_pagerank):
            pagerank_nonfinite_rows += 1
        else:
            pagerank_errors.append(abs(observed_pagerank - pagerank[vertex]))
        groups[int(row["community"])].append(row)

    rank_order_mismatches = 0
    group_shape_mismatches = 0
    community_size_inconsistency_count = 0
    for community, group_rows in groups.items():
        group_rows.sort(key=lambda row: int(row["representative_rank"]))
        ranks = [int(row["representative_rank"]) for row in group_rows]
        if ranks != [1, 2, 3, 4, 5]:
            group_shape_mismatches += 1
        sizes = {int(row["community_size"]) for row in group_rows}
        if len(sizes) != 1 or min(sizes, default=0) < 10:
            community_size_inconsistency_count += 1
        expected_order = sorted(
            group_rows,
            key=lambda row: (-float(row["pagerank"]), str(row["product_id"])),
        )
        if [int(row["vertex"]) for row in group_rows] != [
            int(row["vertex"]) for row in expected_order
        ]:
            rank_order_mismatches += 1

    checks = {
        "representative_vertices_are_known_and_unique": (
            unknown_vertices == 0 and duplicate_representatives == 0
        ),
        "market_metadata_matches_nodes": len(metadata_mismatches) == 0,
        "pagerank_matches_saved_full_snapshot": (
            pagerank_nonfinite_rows == 0
            and bool(pagerank_errors)
            and max(pagerank_errors) <= PAGERANK_TOLERANCE
        ),
        "five_representatives_per_returned_community": group_shape_mismatches == 0,
        "community_sizes_are_consistent_and_at_least_ten": community_size_inconsistency_count == 0,
        "representative_rank_matches_pagerank_order": rank_order_mismatches == 0,
    }
    return {
        "status": "assessed" if all(checks.values()) else "failed",
        "path": str(path.relative_to(ROOT)),
        "rows": len(rows),
        "returned_community_count": len(groups),
        "community_sizes_reported": [
            {"community": label, "community_size": int(group[0]["community_size"])}
            for label, group in sorted(groups.items())
        ],
        "checks": checks,
        "pagerank_reference_path": str(PAGERANK_PATH.relative_to(ROOT)),
        "pagerank_tolerance": PAGERANK_TOLERANCE,
        "pagerank_max_absolute_error": max(pagerank_errors, default=None),
        "pagerank_nonfinite_rows": pagerank_nonfinite_rows,
        "duplicate_representative_rows": duplicate_representatives,
        "metadata_mismatch_count": len(metadata_mismatches),
        "metadata_mismatch_sample": metadata_mismatches[:10],
        "representative_rank_order_mismatch_communities": rank_order_mismatches,
        "full_leiden_partition_comparison": (
            "not attempted; query fits Leiden independently and numeric community labels are local"
        ),
        "community_coverage_limit": (
            "This output contains only top representatives. It does not expose all assignments, "
            "so the set of omitted communities cannot be checked against the separate full fit."
        ),
    }


def main() -> int:
    graph, nodes, pagerank, source_summary = load_graph_inputs()
    expected_vertices = set(nodes)
    partition_results: dict[str, dict[str, object]] = {}
    partitions: dict[str, dict[int, int] | None] = {}
    for name, path in COMMUNITY_RESULTS.items():
        result, assignments = assess_partition(
            name, path, graph, expected_vertices, nodes, pagerank
        )
        partition_results[name] = result
        partitions[name] = assignments

    composition = assess_composition(COMPOSITION_PATH, nodes, pagerank)
    composition["query_provenance"] = {
        "verified_final_capture": {
            "parquet": str(COMPOSITION_PATH.relative_to(ROOT)),
            **query_reference("24_market_collections_final"),
        },
        "initial_failed_attempt": {
            "parquet": str((RESULTS / "24_market_collections.parquet").relative_to(ROOT)),
            **query_reference("24_market_collections"),
        },
    }
    pending = [
        name for name, result in partition_results.items() if result["status"] == "pending"
    ]
    if composition["status"] == "pending":
        pending.append("24_market_collections")
    failed = [
        name for name, result in partition_results.items() if result["status"] == "failed"
    ]
    if composition["status"] == "failed":
        failed.append("24_market_collections")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "runtime": {
            "cpu_only_assessment": True,
            "gpu_or_server_used": False,
            "pandas_used": False,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "networkx": nx.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "assessment_status": "failed" if failed else "pending" if pending else "assessed",
        "source_graph": source_summary,
        "community_results": partition_results,
        "cross_method_partition_comparison": compare_partitions(partitions, expected_vertices),
        "market_collections_composition": composition,
        "pending_outputs": pending,
        "failed_outputs": failed,
        "interpretation": (
            "Modularity and induced connectivity describe each saved partition on the unweighted "
            "undirected graph. Community labels are query-local. Cross-method ARI compares one fit "
            "per method and does not establish run-to-run stability, campaign response, or sales lift."
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(REPORT.relative_to(ROOT))
    print(f"assessment_status: {report['assessment_status']}")
    print(f"pending_outputs: {', '.join(pending) if pending else 'none'}")
    print(f"failed_outputs: {', '.join(failed) if failed else 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
