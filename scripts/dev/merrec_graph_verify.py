#!/usr/bin/env python3
"""Build an independent CPU oracle for MerRec graph SQL and compare snapshots."""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import duckdb
import networkx as nx
import pyarrow as pa
import pyarrow.parquet as pq

from merrec_graph_probe_verify import validate_graph_probes


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "fixture/demo/merrec"
DEFAULT_SQL = ROOT / "docs/research/merrec/sql/01_graph_views.sql"
DEFAULT_RESULTS = ROOT / "target/merrec-exploration/results"
DEFAULT_GRAPH = ROOT / "target/merrec-exploration/cpu_transitions.parquet"
DEFAULT_OUTPUT = ROOT / "docs/research/merrec/graph_validation.json"
TASTE_SHARE_COLUMNS = (
    "women", "toys", "kids", "men", "home", "electronics", "beauty",
    "vintage", "books", "sports", "other", "handmade", "garden", "arts",
    "pets", "office", "tools",
)


def read_arrow(connection: duckdb.DuckDBPyConnection, sql: str) -> pa.Table:
    return connection.execute(sql).to_arrow_table()


def plain(value: Any) -> Any:
    if hasattr(value, "as_py"):
        return value.as_py()
    return value


def json_value(value: Any) -> Any:
    value = plain(value)
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def records(table: pa.Table) -> list[dict[str, Any]]:
    return [
        {key: json_value(value) for key, value in row.items()}
        for row in table.to_pylist()
    ]


def value_equal(expected: Any, actual: Any, tolerance: float) -> tuple[bool, float | None]:
    expected = plain(expected)
    actual = plain(actual)
    if expected is None or actual is None:
        return expected is actual, None
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        difference = abs(float(expected) - float(actual))
        return difference <= tolerance, difference
    return expected == actual, None


def compare_tables(
    expected: pa.Table,
    actual: pa.Table,
    key_columns: tuple[str, ...],
    tolerance: float,
) -> dict[str, Any]:
    expected_rows = records(expected)
    actual_rows = records(actual)
    expected_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    actual_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    duplicate_expected = 0
    duplicate_actual = 0
    for row, target in ((row, expected_by_key) for row in expected_rows):
        key = tuple(row[column] for column in key_columns)
        if key in target:
            duplicate_expected += 1
        target[key] = row
    for row, target in ((row, actual_by_key) for row in actual_rows):
        key = tuple(row[column] for column in key_columns)
        if key in target:
            duplicate_actual += 1
        target[key] = row

    expected_keys = set(expected_by_key)
    actual_keys = set(actual_by_key)
    missing_columns = sorted(set(expected.column_names) - set(actual.column_names))
    mismatches: dict[str, dict[str, Any]] = {}
    for key in expected_keys & actual_keys:
        expected_row = expected_by_key[key]
        actual_row = actual_by_key[key]
        for column in expected.column_names:
            if column in key_columns or column not in actual_row:
                continue
            equal, difference = value_equal(
                expected_row.get(column), actual_row.get(column), tolerance
            )
            if not equal:
                entry = mismatches.setdefault(
                    column, {"count": 0, "max_absolute_error": 0.0}
                )
                entry["count"] += 1
                if difference is not None:
                    entry["max_absolute_error"] = max(
                        entry["max_absolute_error"], difference
                    )

    return {
        "expected_rows": len(expected_rows),
        "actual_rows": len(actual_rows),
        "missing_keys": len(expected_keys - actual_keys),
        "unexpected_keys": len(actual_keys - expected_keys),
        "duplicate_expected_keys": duplicate_expected,
        "duplicate_actual_keys": duplicate_actual,
        "missing_columns": missing_columns,
        "value_mismatches_by_column": mismatches,
        "tolerance": tolerance,
        "matches": not (
            expected_keys - actual_keys
            or actual_keys - expected_keys
            or duplicate_expected
            or duplicate_actual
            or missing_columns
            or mismatches
        ),
    }


def components_summary(graph: nx.Graph | nx.DiGraph) -> dict[str, Any]:
    if graph.is_directed():
        weak = list(nx.weakly_connected_components(graph))
        strong = list(nx.strongly_connected_components(graph))
    else:
        weak = list(nx.connected_components(graph))
        strong = weak
    weak_sizes = sorted((len(part) for part in weak), reverse=True)
    strong_sizes = sorted((len(part) for part in strong), reverse=True)
    return {
        "weak_component_count": len(weak),
        "weak_component_sizes_top_10": weak_sizes[:10],
        "largest_weak_component_vertices": weak_sizes[0] if weak_sizes else 0,
        "strong_component_count": len(strong),
        "strong_component_sizes_top_10": strong_sizes[:10],
        "largest_strong_component_vertices": strong_sizes[0] if strong_sizes else 0,
    }


def degree_summary(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "median": 0, "mean": 0, "max": 0}
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    return {
        "min": ordered[0],
        "median": median,
        "mean": sum(ordered) / len(ordered),
        "max": ordered[-1],
    }


def spearman(left: dict[int, float], right: dict[int, float]) -> float | None:
    keys = sorted(set(left) & set(right))
    if len(keys) < 2:
        return None

    def ranks(values: dict[int, float]) -> dict[int, float]:
        ordered = sorted(keys, key=lambda key: values[key])
        output: dict[int, float] = {}
        start = 0
        while start < len(ordered):
            end = start + 1
            while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
                end += 1
            rank = (start + 1 + end) / 2
            for index in range(start, end):
                output[ordered[index]] = rank
            start = end
        return output

    left_ranks = ranks(left)
    right_ranks = ranks(right)
    left_mean = sum(left_ranks.values()) / len(keys)
    right_mean = sum(right_ranks.values()) / len(keys)
    numerator = sum(
        (left_ranks[key] - left_mean) * (right_ranks[key] - right_mean)
        for key in keys
    )
    left_norm = math.sqrt(sum((value - left_mean) ** 2 for value in left_ranks.values()))
    right_norm = math.sqrt(sum((value - right_mean) ** 2 for value in right_ranks.values()))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else None


def rank_comparison(
    expected: dict[int, float], actual_table: pa.Table, tolerance: float
) -> dict[str, Any]:
    columns = actual_table.column_names
    vertex_column = next((name for name in ("vertex", "id", "src") if name in columns), None)
    value_column = next((name for name in ("value", "pagerank", "rank") if name in columns), None)
    if vertex_column is None or value_column is None:
        return {"matches": False, "error": "expected vertex/id and value/pagerank/rank columns"}
    actual = {
        int(row[vertex_column]): float(row[value_column])
        for row in actual_table.to_pylist()
    }
    shared = set(expected) & set(actual)
    missing = set(expected) - set(actual)
    unexpected = set(actual) - set(expected)
    errors = [abs(expected[key] - actual[key]) for key in shared]
    expected_top = [key for key, _ in sorted(expected.items(), key=lambda pair: (-pair[1], pair[0]))[:20]]
    actual_top = [key for key, _ in sorted(actual.items(), key=lambda pair: (-pair[1], pair[0]))[:20]]
    return {
        "expected_vertices": len(expected),
        "actual_vertices": len(actual),
        "missing_vertices": len(missing),
        "unexpected_vertices": len(unexpected),
        "max_absolute_error": max(errors, default=0.0),
        "mean_absolute_error": sum(errors) / len(errors) if errors else None,
        "spearman_rank_correlation": spearman(expected, actual),
        "top_10_overlap": len(set(expected_top[:10]) & set(actual_top[:10])),
        "top_20_overlap": len(set(expected_top) & set(actual_top)),
        "tolerance": tolerance,
        "matches": (
            not missing and not unexpected and bool(errors)
            and max(errors, default=0.0) <= tolerance
        ),
    }


def top_rank_comparison(
    expected: dict[int, float],
    actual_table: pa.Table | None,
    exclude_vertices: set[int],
    top_k: int,
    tolerance: float,
) -> dict[str, Any]:
    if actual_table is None:
        return {"matches": None, "error": "personalized PageRank snapshot missing"}
    columns = actual_table.column_names
    vertex_column = next((name for name in ("vertex", "id", "src") if name in columns), None)
    value_column = next((name for name in ("value", "pagerank", "rank") if name in columns), None)
    if vertex_column is None or value_column is None:
        return {"matches": False, "error": "expected vertex/id and value/pagerank/rank columns"}
    expected_top = [
        (int(vertex), float(value))
        for vertex, value in sorted(expected.items(), key=lambda pair: (-pair[1], pair[0]))
        if int(vertex) not in exclude_vertices
    ][:top_k]
    actual_rows = [
        (int(row[vertex_column]), float(row[value_column]))
        for row in actual_table.to_pylist()
    ]
    expected_by_vertex = dict(expected_top)
    actual_by_vertex = dict(actual_rows)
    errors = [
        abs(expected_by_vertex[vertex] - actual_by_vertex[vertex])
        for vertex in set(expected_by_vertex) & set(actual_by_vertex)
    ]
    expected_ids = [vertex for vertex, _ in expected_top]
    actual_ids = [vertex for vertex, _ in actual_rows]
    return {
        "top_k": top_k,
        "expected_rows": len(expected_top),
        "actual_rows": len(actual_rows),
        "expected_top_vertex_ids": expected_ids,
        "actual_top_vertex_ids": actual_ids,
        "top_order_matches": expected_ids == actual_ids,
        "missing_expected_top_vertices": len(set(expected_ids) - set(actual_ids)),
        "unexpected_actual_vertices": len(set(actual_ids) - set(expected_ids)),
        "max_absolute_error_on_shared_vertices": max(errors, default=0.0),
        "mean_absolute_error_on_shared_vertices": sum(errors) / len(errors) if errors else None,
        "tolerance": tolerance,
        "matches": (
            expected_ids == actual_ids and bool(errors)
            and max(errors, default=0.0) <= tolerance
        ),
    }


def edge_comparison(
    expected: pa.Table,
    actual: pa.Table,
    product_to_id: dict[str, int],
    tolerance: float,
) -> dict[str, Any]:
    actual_columns = actual.column_names
    source_column = next(
        (name for name in ("src", "source", "src_id", "source_id", "from") if name in actual_columns),
        None,
    )
    target_column = next(
        (name for name in ("dst", "target", "dst_id", "target_id", "to") if name in actual_columns),
        None,
    )
    weight_column = next(
        (name for name in ("weight", "edge_weight", "value") if name in actual_columns),
        None,
    )
    if source_column is None or target_column is None or weight_column is None:
        return {"matches": False, "error": "expected source, target, and weight columns"}

    def endpoint(value: Any) -> int:
        if isinstance(value, str):
            return product_to_id[value] if value in product_to_id else int(value)
        return int(value)

    expected_rows = records(expected)
    actual_rows = records(actual)
    expected_edges = {
        (int(row["src"]), int(row["dst"])): float(row["weight"])
        for row in expected_rows
    }
    actual_edges: dict[tuple[int, int], float] = {}
    duplicate_edges = 0
    for row in actual_rows:
        edge = (endpoint(row[source_column]), endpoint(row[target_column]))
        if edge in actual_edges:
            duplicate_edges += 1
        actual_edges[edge] = actual_edges.get(edge, 0.0) + float(row[weight_column])
    expected_keys = set(expected_edges)
    actual_keys = set(actual_edges)
    shared = expected_keys & actual_keys
    differences = [abs(expected_edges[key] - actual_edges[key]) for key in shared]
    return {
        "expected_edges": len(expected_edges),
        "actual_edges": len(actual_rows),
        "missing_edges": len(expected_keys - actual_keys),
        "unexpected_edges": len(actual_keys - expected_keys),
        "duplicate_actual_edges_collapsed": duplicate_edges,
        "weight_mismatch_edges": sum(
            1 for difference in differences if difference > tolerance
        ),
        "max_absolute_weight_error": max(differences, default=0.0),
        "tolerance": tolerance,
        "matches": (
            not (expected_keys - actual_keys or actual_keys - expected_keys)
            and not any(difference > tolerance for difference in differences)
        ),
    }


def graph_metrics(
    nodes: pa.Table,
    edges: pa.Table,
    undirected_edges: pa.Table,
    pagerank_snapshot: pa.Table | None,
    personalized_pagerank_snapshot: pa.Table | None,
    compare_edges: pa.Table | None,
    compare_pagerank: pa.Table | None,
    tolerance: float,
    pagerank_snapshot_tolerance: float,
    alpha: float,
) -> tuple[dict[str, Any], dict[str, int], dict[int, float]]:
    node_rows = records(nodes)
    product_to_id = {str(row["product_id"]): int(row["id"]) for row in node_rows}
    id_to_node = {int(row["id"]): row for row in node_rows}
    graph = nx.DiGraph()
    graph.add_nodes_from(id_to_node)
    edge_rows = records(edges)
    for row in edge_rows:
        graph.add_edge(int(row["src"]), int(row["dst"]), weight=float(row["weight"]))

    directed_weight_sum = sum(float(row["weight"]) for row in edge_rows)
    directed_weights = [float(row["weight"]) for row in edge_rows]
    indegrees = [degree for _, degree in graph.in_degree()]
    outdegrees = [degree for _, degree in graph.out_degree()]
    affinity = nx.Graph()
    affinity.add_nodes_from(id_to_node)
    for row in records(undirected_edges):
        affinity.add_edge(int(row["src"]), int(row["dst"]), weight=float(row["weight"]))

    condensation = nx.condensation(graph)
    condensation_dag = {
        "scc_vertices": condensation.number_of_nodes(),
        "scc_edges": condensation.number_of_edges(),
        "source_sccs": sum(1 for node in condensation if condensation.in_degree(node) == 0),
        "sink_sccs": sum(1 for node in condensation if condensation.out_degree(node) == 0),
        "longest_path_edges": nx.dag_longest_path_length(condensation),
    }
    try:
        cycle = [list(edge[:2]) for edge in nx.find_cycle(graph, orientation="original")]
    except nx.NetworkXNoCycle:
        cycle = []

    pagerank = nx.pagerank(
        graph, alpha=alpha, weight="weight", tol=1e-10, max_iter=1000
    )
    seed_product_id = "15928_1315"
    seed_id = product_to_id.get(seed_product_id)
    personalized_pagerank: dict[int, float] | None = None
    if seed_id is not None:
        personalized_pagerank = nx.pagerank(
            graph,
            alpha=alpha,
            personalization={seed_id: 1.0},
            weight="weight",
            tol=1e-10,
            max_iter=1000,
        )
    ranked = sorted(pagerank.items(), key=lambda pair: (-pair[1], pair[0]))
    top = [
        {
            "id": node_id,
            "product_id": id_to_node[node_id]["product_id"],
            "pagerank": value,
            "category": id_to_node[node_id]["category"],
            "leaf_category": id_to_node[node_id]["leaf_category"],
            "viewers": id_to_node[node_id]["viewers"],
        }
        for node_id, value in ranked[:20]
    ]
    metrics: dict[str, Any] = {
        "directed": {
            "vertices": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "self_loops": nx.number_of_selfloops(graph),
            "density": nx.density(graph),
            "reciprocity": nx.reciprocity(graph),
            "sum_edge_weights": directed_weight_sum,
            "edge_weight_min": min(directed_weights, default=None),
            "edge_weight_median": sorted(directed_weights)[len(directed_weights) // 2]
                if directed_weights else None,
            "edge_weight_max": max(directed_weights, default=None),
            "in_degree": degree_summary(indegrees),
            "out_degree": degree_summary(outdegrees),
            **components_summary(graph),
        },
        "undirected_affinity": {
            "vertices": affinity.number_of_nodes(),
            "edges": affinity.number_of_edges(),
            "density": nx.density(affinity),
            **components_summary(affinity),
        },
        "dag": {
            "is_directed_acyclic_graph": nx.is_directed_acyclic_graph(graph),
            "cycle_witness_ids": cycle[:20],
            "condensation": condensation_dag,
        },
        "pagerank": {
            "algorithm": "networkx.pagerank",
            "alpha": alpha,
            "weight_column": "weight",
            "tolerance": 1e-10,
            "max_iterations": 1000,
            "vertex_count": len(pagerank),
            "sum": sum(pagerank.values()),
            "top_20": top,
        },
    }
    if pagerank_snapshot is not None:
        metrics["pagerank"]["saved_algeon_snapshot_comparison"] = rank_comparison(
            pagerank, pagerank_snapshot, tolerance
        )
    if personalized_pagerank is not None:
        personalized_ordered = sorted(
            personalized_pagerank.items(), key=lambda pair: (-pair[1], pair[0])
        )
        personalized_top = [
            {
                "vertex": node_id,
                "product_id": id_to_node[node_id]["product_id"],
                "value": value,
                "brand_name": id_to_node[node_id]["brand_name"],
                "category": id_to_node[node_id]["category"],
                "leaf_category": id_to_node[node_id]["leaf_category"],
                "viewers": id_to_node[node_id]["viewers"],
            }
            for node_id, value in personalized_ordered
            if node_id != seed_id
        ][:15]
        metrics["personalized_pagerank_nintendo"] = {
            "seed_product_id": seed_product_id,
            "seed_vertex_id": seed_id,
            "seed_node": id_to_node[seed_id],
            "alpha": alpha,
            "weight_column": "weight",
            "tolerance": 1e-10,
            "max_iterations": 1000,
            "rank_sum_all_vertices": sum(personalized_pagerank.values()),
            "top_15_excluding_seed": personalized_top,
            "saved_algeon_snapshot_comparison": top_rank_comparison(
                personalized_pagerank, personalized_pagerank_snapshot,
                {seed_id}, 15, pagerank_snapshot_tolerance
            ),
        }
    else:
        metrics["personalized_pagerank_nintendo"] = {
            "seed_product_id": seed_product_id,
            "error": "seed product is not a node in the transition graph",
        }
    if compare_pagerank is not None:
        metrics["pagerank"]["additional_algeon_comparison"] = rank_comparison(
            pagerank, compare_pagerank, tolerance
        )
    if compare_edges is not None:
        metrics["additional_algeon_edge_comparison"] = edge_comparison(
            edges, compare_edges, product_to_id, tolerance
        )
    return metrics, product_to_id, pagerank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--sql", type=Path, default=DEFAULT_SQL)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--graph-output", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--compare-edges", type=Path,
                        help="Optional later Algeon edge result Parquet to compare")
    parser.add_argument("--compare-pagerank", type=Path,
                        help="Optional later Algeon PageRank Parquet to compare")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--pagerank-snapshot-tolerance", type=float, default=1e-5)
    parser.add_argument("--alpha", type=float, default=0.85)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    events_path = data_dir / "parquet/events.parquet"
    items_path = data_dir / "parquet/items.parquet"
    graph_sql_path = args.sql.resolve()
    results_dir = args.results_dir.resolve()
    connection = duckdb.connect(database=":memory:")
    connection.execute(f"SET threads = {max(1, args.threads)}")
    connection.execute("SET preserve_insertion_order = false")
    connection.read_parquet(str(events_path)).create_view("merrec_events")
    connection.read_parquet(str(items_path)).create_view("merrec_items")

    graph_sql = graph_sql_path.read_text()
    for statement in graph_sql.split(";"):
        if statement.strip():
            connection.execute(statement)

    expected_markets = read_arrow(
        connection,
        """
        SELECT m.*, r.viewers
        FROM merrec_markets m JOIN merrec_market_reach r USING (product_id)
        ORDER BY product_id
        """,
    )
    expected_transitions = read_arrow(
        connection, "SELECT src, dst, weight FROM merrec_transitions ORDER BY src, dst"
    )
    expected_nodes = read_arrow(
        connection,
        """
        WITH incident_products AS (
            SELECT src AS product_id FROM merrec_transitions
            UNION SELECT dst AS product_id FROM merrec_transitions
        )
        SELECT CAST(ROW_NUMBER() OVER (ORDER BY m.product_id) AS INT) AS id,
               m.product_id, m.brand_name, m.category, m.leaf_category, m.items,
               m.mean_price, r.viewers
        FROM merrec_markets m
        JOIN merrec_market_reach r USING (product_id)
        JOIN incident_products p USING (product_id)
        ORDER BY id
        """,
    )
    expected_edges = read_arrow(
        connection,
        """
        SELECT src_node.id AS src, dst_node.id AS dst, t.weight
        FROM merrec_transitions t
        JOIN (
            SELECT CAST(ROW_NUMBER() OVER (ORDER BY product_id) AS INT) AS id,
                   product_id
            FROM (
                SELECT src AS product_id FROM merrec_transitions
                UNION SELECT dst AS product_id FROM merrec_transitions
            ) incident_products
        ) src_node ON src_node.product_id = t.src
        JOIN (
            SELECT CAST(ROW_NUMBER() OVER (ORDER BY product_id) AS INT) AS id,
                   product_id
            FROM (
                SELECT src AS product_id FROM merrec_transitions
                UNION SELECT dst AS product_id FROM merrec_transitions
            ) incident_products
        ) dst_node ON dst_node.product_id = t.dst
        ORDER BY src, dst
        """,
    )
    expected_undirected = read_arrow(
        connection,
        """
        SELECT LEAST(src, dst) AS src, GREATEST(src, dst) AS dst,
               SUM(weight) AS weight
        FROM (
            SELECT src_node.id AS src, dst_node.id AS dst, t.weight
            FROM merrec_transitions t
            JOIN (
                SELECT CAST(ROW_NUMBER() OVER (ORDER BY product_id) AS INT) AS id,
                       product_id
                FROM (
                    SELECT src AS product_id FROM merrec_transitions
                    UNION SELECT dst AS product_id FROM merrec_transitions
                ) incident_products
            ) src_node ON src_node.product_id = t.src
            JOIN (
                SELECT CAST(ROW_NUMBER() OVER (ORDER BY product_id) AS INT) AS id,
                       product_id
                FROM (
                    SELECT src AS product_id FROM merrec_transitions
                    UNION SELECT dst AS product_id FROM merrec_transitions
                ) incident_products
            ) dst_node ON dst_node.product_id = t.dst
        ) directed_edges
        GROUP BY LEAST(src, dst), GREATEST(src, dst)
        ORDER BY src, dst
        """,
    )
    expected_audience = read_arrow(
        connection,
        """
        SELECT DISTINCT n.id AS src, CAST(-a.user_id AS INT) AS dst
        FROM merrec_market_audience a
        JOIN (
            SELECT CAST(ROW_NUMBER() OVER (ORDER BY product_id) AS INT) AS id,
                   product_id
            FROM (
                SELECT src AS product_id FROM merrec_transitions
                UNION SELECT dst AS product_id FROM merrec_transitions
            ) incident_products
        ) n ON n.product_id = a.product_id
        ORDER BY src, dst
        """,
    )
    transitions_with_ids = read_arrow(
        connection,
        """
        SELECT a.id AS src, b.id AS dst, t.weight,
               t.src AS src_product_id, t.dst AS dst_product_id
        FROM merrec_transitions t
        JOIN (
            SELECT CAST(ROW_NUMBER() OVER (ORDER BY product_id) AS INT) AS id,
                   product_id
            FROM (
                SELECT src AS product_id FROM merrec_transitions
                UNION SELECT dst AS product_id FROM merrec_transitions
            ) incident_products
        ) a ON a.product_id = t.src
        JOIN (
            SELECT CAST(ROW_NUMBER() OVER (ORDER BY product_id) AS INT) AS id,
                   product_id
            FROM (
                SELECT src AS product_id FROM merrec_transitions
                UNION SELECT dst AS product_id FROM merrec_transitions
            ) incident_products
        ) b ON b.product_id = t.dst
        ORDER BY src, dst
        """,
    )

    args.graph_output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(transitions_with_ids, args.graph_output)

    def read_snapshot(name: str) -> pa.Table | None:
        path = results_dir / f"{name}.parquet"
        return pq.read_table(path) if path.exists() else None

    snapshot_pairs = {
        "markets": (expected_markets, read_snapshot("prepare_markets"), ("product_id",)),
        "transitions": (expected_transitions, read_snapshot("prepare_transitions"), ("src", "dst")),
        "nodes": (expected_nodes, read_snapshot("prepare_nodes"), ("id",)),
        "edges": (expected_edges, read_snapshot("prepare_edges"), ("src", "dst")),
        "undirected": (expected_undirected, read_snapshot("prepare_undirected"), ("src", "dst")),
        "audience": (expected_audience, read_snapshot("prepare_audience"), ("src", "dst")),
    }
    snapshot_comparisons: dict[str, Any] = {}
    for name, (expected, actual, key_columns) in snapshot_pairs.items():
        snapshot_comparisons[name] = (
            compare_tables(expected, actual, key_columns, args.tolerance)
            if actual is not None
            else {"matches": None, "error": "snapshot missing"}
        )

    # This count breakdown shows which adjacent-view rows the time and support filters remove.
    tie_counts = records(
        read_arrow(
            connection,
            """
            WITH candidates AS (
                SELECT s.user_id, s.event_time, s.next_time,
                       s.product_id AS src, s.next_product_id AS dst
                FROM merrec_view_steps s
                JOIN merrec_market_reach a ON a.product_id = s.product_id
                JOIN merrec_market_reach b ON b.product_id = s.next_product_id
                WHERE s.next_product_id IS NOT NULL
                  AND s.product_id <> s.next_product_id
                  AND a.viewers >= 10 AND b.viewers >= 10
            ), support AS (
                SELECT src, dst, count(DISTINCT user_id) AS users,
                       count(*) AS event_steps
                FROM candidates
                WHERE next_time > event_time
                  AND next_time <= event_time + INTERVAL '30 minutes'
                GROUP BY src, dst
            )
            SELECT count(*) AS popular_distinct_product_neighbor_rows,
                   count(*) FILTER (WHERE next_time = event_time)
                       AS equal_time_rows_excluded,
                   count(*) FILTER (WHERE next_time > event_time
                       AND next_time <= event_time + INTERVAL '30 minutes')
                       AS positive_gap_rows_within_30m,
                   count(*) FILTER (WHERE next_time > event_time + INTERVAL '30 minutes')
                       AS rows_over_30m,
                   count(*) FILTER (WHERE next_time < event_time) AS negative_gap_rows,
                   (SELECT count(*) FROM support) AS edge_count_before_user_threshold,
                   (SELECT count(*) FROM support WHERE users >= 3)
                       AS edge_count_after_user_threshold,
                   (SELECT coalesce(sum(event_steps), 0) FROM support WHERE users >= 3)
                       AS event_steps_on_kept_edges,
                   (SELECT coalesce(sum(users), 0) FROM support WHERE users >= 3)
                       AS sum_distinct_users_per_kept_edge
            FROM candidates
            """,
        )
    )[0]

    taste_path = results_dir / "prepare_tastes.parquet"
    tastes_sql_path = graph_sql_path.parent / "prepare_tastes.sql"
    tastes_comparison: dict[str, Any] | None = None
    if taste_path.exists() and tastes_sql_path.exists():
        expected_tastes = read_arrow(connection, tastes_sql_path.read_text())
        actual_tastes = pq.read_table(taste_path)
        tastes_comparison = compare_tables(
            expected_tastes, actual_tastes, ("id",), args.tolerance
        )
        taste_rows = records(expected_tastes)
        norms = [
            math.sqrt(sum(float(row[column]) ** 2 for column in TASTE_SHARE_COLUMNS))
            for row in taste_rows
        ]
        tastes_comparison.update(
            {
                "user_id_min": min((int(row["id"]) for row in taste_rows), default=None),
                "user_id_max": max((int(row["id"]) for row in taste_rows), default=None),
                "user_id_median": (
                    sorted(int(row["id"]) for row in taste_rows)[len(taste_rows) // 2]
                    if taste_rows else None
                ),
                "category_share_columns": list(TASTE_SHARE_COLUMNS),
                "l2_norm_min": min(norms, default=None),
                "l2_norm_max": max(norms, default=None),
                "l2_norm_max_absolute_error_from_one": max(
                    (abs(value - 1.0) for value in norms), default=None
                ),
                "unit_l2_within_tolerance_count": sum(
                    abs(value - 1.0) <= args.tolerance for value in norms
                ),
            }
        )
    else:
        tastes_comparison = {
            "matches": None,
            "error": "prepare_tastes.sql or prepare_tastes.parquet missing",
        }

    saved_pagerank = read_snapshot("11_pagerank_full")
    saved_personalized_pagerank = read_snapshot("12_nintendo_discovery")
    extra_edges = pq.read_table(args.compare_edges) if args.compare_edges else None
    extra_pagerank = pq.read_table(args.compare_pagerank) if args.compare_pagerank else None
    graph, product_to_id, pagerank = graph_metrics(
        expected_nodes,
        expected_edges,
        expected_undirected,
        saved_pagerank,
        saved_personalized_pagerank,
        extra_edges,
        extra_pagerank,
        args.tolerance,
        args.pagerank_snapshot_tolerance,
        args.alpha,
    )
    graph_probe_validation = validate_graph_probes(
        expected_nodes,
        expected_edges,
        expected_undirected,
        expected_audience,
        results_dir,
        args.tolerance,
    )

    source_hash = hashlib.sha256(graph_sql_path.read_bytes()).hexdigest()
    report = {
        "source": {
            "dataset": "fixture/demo/merrec, October 2023 shard",
            "events_parquet": str(events_path),
            "items_parquet": str(items_path),
            "graph_sql": str(graph_sql_path),
            "graph_sql_sha256": source_hash,
            "duckdb_version": duckdb.__version__,
            "networkx_version": nx.__version__,
            "cpu_threads": max(1, args.threads),
            "pagerank_alpha": args.alpha,
            "comparison_tolerance": args.tolerance,
            "pagerank_snapshot_tolerance": args.pagerank_snapshot_tolerance,
        },
        "graph_sql_contract": {
            "edge_semantics": (
                "LEAD over item_view rows within (user_id, session_id) ordered by "
                "(event_time, event_id); product_id transitions only; endpoint "
                "products each have at least 10 distinct viewers; next timestamp "
                "must be strictly greater and no more than 30 minutes later; edge "
                "weight counts distinct users and edges require at least 3 users."
            ),
            "tie_and_filter_counts": tie_counts,
            "expected_snapshot_rows": {
                "markets": expected_markets.num_rows,
                "transitions": expected_transitions.num_rows,
                "nodes": expected_nodes.num_rows,
                "edges": expected_edges.num_rows,
                "undirected_edges": expected_undirected.num_rows,
                "audience_edges": expected_audience.num_rows,
            },
        },
        "snapshot_comparisons": snapshot_comparisons,
        "networkx_graph": graph,
        "graph_probe_validation": graph_probe_validation,
        "taste_vectors": tastes_comparison,
        "artifacts": {
            "cpu_transition_parquet": str(args.graph_output.resolve()),
            "cpu_transition_schema": str(transitions_with_ids.schema),
            "pagerank_full_snapshot": str(results_dir / "11_pagerank_full.parquet"),
            "additional_edge_comparison": str(args.compare_edges.resolve())
                if args.compare_edges else None,
            "additional_pagerank_comparison": str(args.compare_pagerank.resolve())
                if args.compare_pagerank else None,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    compact = {
        "output": str(args.output.resolve()),
        "cpu_transitions": str(args.graph_output.resolve()),
        "transition_rows": expected_transitions.num_rows,
        "directed_edges": graph["directed"]["edges"],
        "vertices": graph["directed"]["vertices"],
        "snapshots_match": {
            name: result["matches"] for name, result in snapshot_comparisons.items()
        },
        "pagerank_snapshot": graph["pagerank"].get("saved_algeon_snapshot_comparison"),
        "personalized_pagerank_snapshot": graph[
            "personalized_pagerank_nintendo"
        ].get("saved_algeon_snapshot_comparison"),
        "taste_snapshot_matches": tastes_comparison.get("matches"),
        "graph_probes": {
            name: result.get("matches")
            for name, result in graph_probe_validation["similarity"].items()
        },
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))
    connection.close()


if __name__ == "__main__":
    main()
