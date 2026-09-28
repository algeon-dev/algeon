#!/usr/bin/env python3
"""CPU references for the saved MerRec cuGraph probe results."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import networkx as nx
import pyarrow as pa
import pyarrow.parquet as pq


STOCHASTIC_PROBES = {
    "ecg": "community assignment may vary; exact partition equality is not checked",
    "force_atlas2": "layout coordinates may vary; exact coordinate equality is not checked",
    "leiden": "community assignment may vary; exact partition equality is not checked",
    "louvain": "community assignment may vary; exact partition equality is not checked",
    "spectral_modularity_maximization": (
        "community assignment may vary; exact partition equality is not checked"
    ),
}

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def table_records(table: pa.Table) -> list[dict[str, Any]]:
    return table.to_pylist()


def load_probe(results_dir: Path, name: str) -> tuple[pa.Table | None, dict[str, Any]]:
    parquet_path = results_dir / f"cugraph_{name}.parquet"
    metadata_path = (
        REPOSITORY_ROOT / "docs/research/merrec/results" / f"cugraph_{name}.json"
    )
    if not parquet_path.is_file():
        return None, {"name": name, "status": "missing", "rows": None}
    table = pq.read_table(parquet_path)
    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
    return table, {
        "name": name,
        "status": metadata.get("status", "snapshot_present"),
        "rows": table.num_rows,
        "query_sha256": __import__("hashlib").sha256(
            metadata.get("sql", "").encode()
        ).hexdigest() if metadata.get("sql") else None,
        "metadata_path": str(metadata_path),
    }


def scalar_check(
    name: str,
    actual: float | int,
    expected: float | int,
    absolute_tolerance: float,
    relative_tolerance: float = 1e-5,
) -> dict[str, Any]:
    absolute_error = abs(float(actual) - float(expected))
    allowed_error = max(absolute_tolerance, relative_tolerance * abs(float(expected)))
    return {
        "actual": actual,
        "expected": expected,
        "absolute_error": absolute_error,
        "allowed_error": allowed_error,
        "matches": absolute_error <= allowed_error,
    }


def compare_node_top(
    table: pa.Table,
    value_columns: tuple[str, ...],
    cpu_values: dict[int, dict[str, float | int]],
    ordered_vertex_ids: list[int],
    product_to_id: dict[str, int],
    absolute_tolerance: float,
    relative_tolerance: float = 1e-5,
) -> dict[str, Any]:
    actual_rows = table_records(table)
    actual_vertex_ids = [product_to_id[row["product_id"]] for row in actual_rows]
    mismatches: list[dict[str, Any]] = []
    max_abs_error: dict[str, float] = {column: 0.0 for column in value_columns}
    for row, vertex_id in zip(actual_rows, actual_vertex_ids):
        for column in value_columns:
            expected = cpu_values[vertex_id][column]
            actual = row[column]
            check = scalar_check(
                column, actual, expected, absolute_tolerance, relative_tolerance
            )
            max_abs_error[column] = max(max_abs_error[column], check["absolute_error"])
            if not check["matches"]:
                mismatches.append({"vertex_id": vertex_id, **check, "column": column})
    return {
        "scope": "saved top 15 rows only",
        "actual_vertex_ids": actual_vertex_ids,
        "cpu_expected_top_vertex_ids": ordered_vertex_ids[: len(actual_rows)],
        "top_order_matches": actual_vertex_ids == ordered_vertex_ids[: len(actual_rows)],
        "max_absolute_error_by_column": max_abs_error,
        "value_mismatches": mismatches[:20],
        "mismatch_count": len(mismatches),
        "tolerance": {"absolute": absolute_tolerance, "relative": relative_tolerance},
        "matches": (
            actual_vertex_ids == ordered_vertex_ids[: len(actual_rows)] and not mismatches
        ),
    }


def validate_similarity(
    name: str,
    table: pa.Table,
    node_rows: list[dict[str, Any]],
    audience_by_item: dict[int, set[int]],
    product_to_id: dict[str, int],
    seed_id: int,
    tolerance: float,
) -> dict[str, Any]:
    seed_users = audience_by_item[seed_id]
    candidate_scores: list[dict[str, Any]] = []
    for node in node_rows:
        candidate_id = int(node["id"])
        if candidate_id == seed_id or int(node["viewers"]) < 10:
            continue
        candidate_users = audience_by_item[candidate_id]
        intersection = len(seed_users & candidate_users)
        seed_count = len(seed_users)
        candidate_count = len(candidate_users)
        if name == "jaccard":
            score = intersection / len(seed_users | candidate_users)
        elif name == "overlap":
            score = intersection / min(seed_count, candidate_count)
        elif name == "sorensen":
            score = 2 * intersection / (seed_count + candidate_count)
        elif name == "cosine":
            score = intersection / math.sqrt(seed_count * candidate_count)
        else:
            raise ValueError(f"unknown similarity metric: {name}")
        candidate_scores.append(
            {
                "id": candidate_id,
                "product_id": node["product_id"],
                "viewers": int(node["viewers"]),
                "intersection_users": intersection,
                "seed_audience_size": seed_count,
                "candidate_audience_size": candidate_count,
                "score": score,
            }
        )

    expected = sorted(
        candidate_scores,
        key=lambda row: (-row["score"], -row["viewers"], row["product_id"], row["id"]),
    )[: table.num_rows]
    actual_rows = table_records(table)
    actual_ids = [product_to_id[row["candidate_product"]] for row in actual_rows]
    expected_by_id = {row["id"]: row for row in candidate_scores}
    errors = []
    detail_rows = []
    for row, candidate_id in zip(actual_rows, actual_ids):
        cpu = expected_by_id[candidate_id]
        difference = abs(float(row["similarity"]) - cpu["score"])
        errors.append(difference)
        detail_rows.append(
            {
                "candidate_product_id": row["candidate_product"],
                "intersection_users": cpu["intersection_users"],
                "seed_audience_size": cpu["seed_audience_size"],
                "candidate_audience_size": cpu["candidate_audience_size"],
                "expected_score": cpu["score"],
                "native_score": float(row["similarity"]),
                "absolute_error": difference,
            }
        )
    return {
        "query_file": f"cugraph_{name}.json",
        "seed_product_id": "15928_1315",
        "seed_vertex_id": seed_id,
        "seed_audience_size": len(seed_users),
        "candidate_count": len(candidate_scores),
        "formula": {
            "jaccard": "|A intersect B| / |A union B|",
            "overlap": "|A intersect B| / min(|A|, |B|)",
            "sorensen": "2 * |A intersect B| / (|A| + |B|)",
            "cosine": "|A intersect B| / sqrt(|A| * |B|)",
        }[name],
        "cpu_expected_top_products": [
            {key: value for key, value in row.items() if key != "id"}
            for row in expected
        ],
        "native_top_products": [
            {"candidate_product_id": row["candidate_product"],
             "viewers": int(row["viewers"]),
             "similarity": float(row["similarity"])}
            for row in actual_rows
        ],
        "score_comparison_for_native_rows": detail_rows,
        "top_order_matches": actual_ids == [row["id"] for row in expected],
        "max_absolute_score_error": max(errors, default=0.0),
        "mean_absolute_score_error": sum(errors) / len(errors) if errors else None,
        "scores_within_tolerance": all(error <= tolerance for error in errors),
        "tolerance": tolerance,
        "matches": (
            actual_ids == [row["id"] for row in expected]
            and all(error <= tolerance for error in errors)
        ),
    }


def validate_graph_probes(
    nodes: pa.Table,
    edges: pa.Table,
    undirected_edges: pa.Table,
    audience: pa.Table,
    results_dir: Path,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    node_rows = table_records(nodes)
    product_to_id = {row["product_id"]: int(row["id"]) for row in node_rows}
    id_to_node = {int(row["id"]): row for row in node_rows}
    audience_by_item: dict[int, set[int]] = {int(node["id"]): set() for node in node_rows}
    for row in table_records(audience):
        audience_by_item[int(row["src"])].add(int(row["dst"]))
    audience_size_mismatches = [
        node["product_id"]
        for node in node_rows
        if len(audience_by_item[int(node["id"])]) != int(node["viewers"])
    ]

    directed = nx.DiGraph()
    directed.add_nodes_from(id_to_node)
    edge_rows = table_records(edges)
    for row in edge_rows:
        directed.add_edge(int(row["src"]), int(row["dst"]), weight=float(row["weight"]))
    directed_unweighted = nx.DiGraph()
    directed_unweighted.add_nodes_from(id_to_node)
    directed_unweighted.add_edges_from(
        (int(row["src"]), int(row["dst"])) for row in edge_rows
    )
    undirected = nx.Graph()
    undirected.add_nodes_from(id_to_node)
    for row in table_records(undirected_edges):
        undirected.add_edge(int(row["src"]), int(row["dst"]), weight=float(row["weight"]))

    seed_id = product_to_id["15928_1315"]
    probes: dict[str, Any] = {}
    similarity_probes: dict[str, Any] = {}
    inventory: list[dict[str, Any]] = []
    actual_tables: dict[str, pa.Table] = {}
    for parquet_path in sorted(results_dir.glob("cugraph_*.parquet")):
        name = parquet_path.stem.removeprefix("cugraph_")
        table, inventory_row = load_probe(results_dir, name)
        if table is not None:
            actual_tables[name] = table
        inventory.append(inventory_row)

    for name in ("jaccard", "overlap", "sorensen", "cosine"):
        table = actual_tables.get(name)
        if table is not None:
            similarity_probes[name] = validate_similarity(
                name, table, node_rows, audience_by_item, product_to_id, seed_id, tolerance
            )

    degree_maps = {
        "in_degree": dict(directed.in_degree()),
        "out_degree": dict(directed.out_degree()),
        "core_number": nx.core_number(undirected),
        "triangle_count": nx.triangles(undirected),
    }
    degree_specs = {
        "degrees_all": {
            "columns": ("in_degree", "out_degree"),
            "sort": lambda row: (-directed.out_degree(int(row["id"])),
                                 -directed.in_degree(int(row["id"])),
                                 row["product_id"], int(row["id"])),
        },
        "in_degrees_all": {
            "columns": ("in_degree",),
            "sort": lambda row: (-directed.in_degree(int(row["id"])),
                                 -int(row["viewers"]), row["product_id"], int(row["id"])),
        },
        "out_degrees_all": {
            "columns": ("out_degree",),
            "sort": lambda row: (-directed.out_degree(int(row["id"])),
                                 -int(row["viewers"]), row["product_id"], int(row["id"])),
        },
        "core_number": {
            "columns": ("core_number",),
            "sort": lambda row: (-degree_maps["core_number"][int(row["id"])],
                                 -int(row["viewers"]), row["product_id"], int(row["id"])),
        },
        "triangle_count_all": {
            "columns": ("triangle_count",),
            "sort": lambda row: (-degree_maps["triangle_count"][int(row["id"])],
                                 -int(row["viewers"]), row["product_id"], int(row["id"])),
        },
    }
    for name, spec in degree_specs.items():
        table = actual_tables.get(name)
        if table is None:
            continue
        ordered_nodes = [int(row["id"]) for row in sorted(node_rows, key=spec["sort"])]
        cpu_values = {
            int(node["id"]): {
                column: int(degree_maps[column][int(node["id"])])
                for column in spec["columns"]
            }
            for node in node_rows
        }
        probes[name] = compare_node_top(
            table, spec["columns"], cpu_values, ordered_nodes, product_to_id,
            tolerance, 0.0,
        )

    bfs = nx.single_source_shortest_path_length(directed, seed_id, cutoff=4)
    bfs_order = sorted(bfs, key=lambda vertex: (bfs[vertex], id_to_node[vertex]["product_id"], vertex))
    bfs_table = actual_tables.get("bfs")
    if bfs_table is not None:
        bfs_rows = table_records(bfs_table)
        actual_ids = [product_to_id[row["product_id"]] for row in bfs_rows]
        probes["bfs"] = {
            "scope": "saved top 15 reachable rows only, depth limit 4",
            "seed_product_id": "15928_1315",
            "seed_vertex_id": seed_id,
            "actual_product_ids": [row["product_id"] for row in bfs_rows],
            "cpu_expected_product_ids": [id_to_node[vertex]["product_id"] for vertex in bfs_order[:len(bfs_rows)]],
            "actual_distances": [int(row["distance"]) for row in bfs_rows],
            "cpu_expected_distances": [int(bfs[vertex]) for vertex in bfs_order[:len(bfs_rows)]],
            "top_order_matches": actual_ids == bfs_order[:len(actual_ids)],
            "distance_mismatch_count": sum(
                int(row["distance"]) != bfs[vertex]
                for row, vertex in zip(bfs_rows, actual_ids)
            ),
        }
        probes["bfs"]["matches"] = (
            probes["bfs"]["top_order_matches"]
            and probes["bfs"]["distance_mismatch_count"] == 0
        )

    affinity_rows = table_records(undirected_edges)
    affinity_weight = {
        (int(row["src"]), int(row["dst"])): float(row["weight"])
        for row in affinity_rows
    }
    cost_graph = nx.Graph()
    cost_graph.add_nodes_from(id_to_node)
    for (src, dst), weight in affinity_weight.items():
        cost_graph.add_edge(src, dst, weight=1.0 / weight)
    sssp = nx.single_source_dijkstra_path_length(cost_graph, seed_id, weight="weight")
    sssp_order = sorted(sssp, key=lambda vertex: (sssp[vertex], id_to_node[vertex]["product_id"], vertex))
    sssp_table = actual_tables.get("sssp")
    if sssp_table is not None:
        sssp_rows = table_records(sssp_table)
        actual_ids = [product_to_id[row["product_id"]] for row in sssp_rows]
        distance_errors = [
            abs(float(row["distance"]) - float(sssp[vertex]))
            for row, vertex in zip(sssp_rows, actual_ids)
        ]
        probes["sssp"] = {
            "scope": "saved top 15 reachable rows only",
            "seed_product_id": "15928_1315",
            "seed_vertex_id": seed_id,
            "cost": "inverse undirected affinity, 1 / edge weight",
            "actual_product_ids": [row["product_id"] for row in sssp_rows],
            "cpu_expected_product_ids": [id_to_node[vertex]["product_id"] for vertex in sssp_order[:len(sssp_rows)]],
            "actual_distances": [float(row["distance"]) for row in sssp_rows],
            "cpu_expected_distances": [float(sssp[vertex]) for vertex in sssp_order[:len(sssp_rows)]],
            "top_order_matches": actual_ids == sssp_order[:len(actual_ids)],
            "max_absolute_distance_error": max(distance_errors, default=0.0),
            "tolerance": tolerance,
        }
        probes["sssp"]["matches"] = (
            probes["sssp"]["top_order_matches"]
            and max(distance_errors, default=0.0) <= tolerance
        )

    for name, graph, component_list in (
        ("strongly_connected_components", directed, list(nx.strongly_connected_components(directed))),
        ("weakly_connected_components", undirected, list(nx.connected_components(undirected))),
    ):
        table = actual_tables.get(name)
        if table is None:
            continue
        component_of: dict[int, int] = {}
        component_size: dict[int, int] = {}
        for index, component in enumerate(component_list):
            size = len(component)
            for vertex in component:
                component_of[vertex] = index
                component_size[vertex] = size
        rows = table_records(table)
        actual_ids = [product_to_id[row["product_id"]] for row in rows]
        label_column = "component_label" if "component_label" in table.column_names else "label"
        labels = [row[label_column] for row in rows]
        size_column = "component_size"
        size_matches = [
            int(row[size_column]) == component_size[vertex]
            for row, vertex in zip(rows, actual_ids)
        ]
        partition_consistent = all(
            (labels[left] == labels[right])
            == (component_of[actual_ids[left]] == component_of[actual_ids[right]])
            for left in range(len(rows))
            for right in range(len(rows))
        )
        probes[name] = {
            "scope": "saved top 15 component rows only; full CPU partitions computed",
            "cpu_component_count": len(component_list),
            "cpu_largest_component_size": max(map(len, component_list), default=0),
            "actual_component_sizes": [int(row[size_column]) for row in rows],
            "cpu_expected_component_sizes": [component_size[vertex] for vertex in actual_ids],
            "component_size_matches": all(size_matches),
            "partition_consistency_within_returned_rows": partition_consistent,
        }
        probes[name]["matches"] = all(size_matches) and partition_consistent

    kcore_table = actual_tables.get("k_core")
    if kcore_table is not None:
        kcore_graph = nx.k_core(undirected, k=3, core_number=degree_maps["core_number"])
        full_kcore_edges = {
            orientation
            for left, right in kcore_graph.edges()
            for orientation in ((left, right), (right, left))
        }
        expected_kcore = sorted(
            (id_to_node[src]["product_id"], id_to_node[dst]["product_id"])
            for src, dst in full_kcore_edges
        )
        expected_kcore_top = expected_kcore[: kcore_table.num_rows]
        actual_kcore = [
            (row["source_product"], row["destination_product"])
            for row in table_records(kcore_table)
        ]
        probes["k_core"] = {
            "k": 3,
            "scope": "saved top 15 edge rows only",
            "actual_product_edges": [list(edge) for edge in actual_kcore],
            "cpu_expected_top_product_edges": [list(edge) for edge in expected_kcore_top],
            "cpu_core_directed_edge_count": len(expected_kcore),
            "top_order_matches": actual_kcore == expected_kcore_top,
            "actual_edges_in_cpu_core": all(
                edge in set(expected_kcore) for edge in actual_kcore
            ),
        }
        probes["k_core"]["matches"] = (
            probes["k_core"]["top_order_matches"]
            and probes["k_core"]["actual_edges_in_cpu_core"]
        )

    mst_table = actual_tables.get("minimum_spanning_tree")
    if mst_table is not None:
        cpu_mst = nx.minimum_spanning_tree(cost_graph, weight="weight", algorithm="kruskal")
        cpu_mst_pairs = {
            (min(src, dst), max(src, dst)) for src, dst in cpu_mst.edges()
        }
        expected_native_rows = []
        for src, dst in cpu_mst_pairs:
            affinity = affinity_weight[(src, dst)]
            cost = 1.0 / affinity
            expected_native_rows.extend(
                [
                    {"src": src, "dst": dst, "cost": cost, "affinity": affinity},
                    {"src": dst, "dst": src, "cost": cost, "affinity": affinity},
                ]
            )
        expected_native_rows.sort(key=lambda row: (row["cost"], row["src"], row["dst"]))
        actual_rows = table_records(mst_table)
        actual_edges = [
            (product_to_id[row["source_product"]], product_to_id[row["destination_product"]])
            for row in actual_rows
        ]
        actual_pairs = [(min(src, dst), max(src, dst)) for src, dst in actual_edges]
        actual_costs = [float(row["inverse_affinity_cost"]) for row in actual_rows]
        cpu_mst_membership = [pair in cpu_mst_pairs for pair in actual_pairs]
        cost_errors = [
            abs(float(row["inverse_affinity_cost"]) - 1.0 / float(row["affinity_weight"]))
            for row in actual_rows
        ]
        pair_counts: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for src, dst in actual_edges:
            pair_counts.setdefault((min(src, dst), max(src, dst)), []).append((src, dst))
        reverse_duplicate_pairs = sum(
            1 for pair_rows in pair_counts.values()
            if len(pair_rows) > 1 and set(pair_rows) == {(pair_rows[0][0], pair_rows[0][1]), (pair_rows[0][1], pair_rows[0][0])}
        )
        expected_top_rows = expected_native_rows[: len(actual_rows)]
        probes["minimum_spanning_tree"] = {
            "scope": "saved top 15 forest rows only",
            "cpu_spanning_forest_edges": cpu_mst.number_of_edges(),
            "cpu_component_count": nx.number_connected_components(cost_graph),
            "expected_forest_edge_count": cost_graph.number_of_nodes() - nx.number_connected_components(cost_graph),
            "native_inverse_costs_non_decreasing": actual_costs == sorted(actual_costs),
            "all_returned_undirected_pairs_in_cpu_mst": all(cpu_mst_membership),
            "top_oriented_rows_match_cpu_mst": actual_edges == [
                (row["src"], row["dst"]) for row in expected_top_rows
            ],
            "actual_top_oriented_edges": [
                {"source": row["source_product"], "destination": row["destination_product"],
                 "inverse_affinity_cost": float(row["inverse_affinity_cost"])}
                for row in actual_rows
            ],
            "reverse_duplicate_pairs_in_top_rows": reverse_duplicate_pairs,
            "max_inverse_cost_formula_error": max(cost_errors, default=0.0),
            "note": (
                "The saved native MST probe contains reversed rows for undirected "
                "forest edges. The SQL retains both orientations returned by cuGraph; "
                "the CPU reference uses one canonical edge per forest edge."
            ),
        }
        probes["minimum_spanning_tree"]["matches"] = (
            probes["minimum_spanning_tree"]["native_inverse_costs_non_decreasing"]
            and all(cpu_mst_membership)
            and max(cost_errors, default=0.0) <= tolerance
        )

    pagerank = nx.pagerank(directed, alpha=0.85, weight="weight", tol=1e-10, max_iter=1000)
    for name, values, column in (
        ("pagerank", pagerank, "pagerank"),
        ("personalized_pagerank", nx.pagerank(
            directed, alpha=0.85, personalization={seed_id: 1.0},
            weight="weight", tol=1e-10, max_iter=1000,
        ), "personalized_pagerank"),
    ):
        table = actual_tables.get(name)
        if table is None:
            continue
        ordered_ids = sorted(
            values,
            key=lambda vertex: (-values[vertex], -int(id_to_node[vertex]["viewers"]),
                                id_to_node[vertex]["product_id"], vertex),
        )
        rows = table_records(table)
        actual_ids = [product_to_id[row["product_id"]] for row in rows]
        errors = [abs(float(row[column]) - values[vertex]) for row, vertex in zip(rows, actual_ids)]
        score_tolerance = 1e-5 if name == "personalized_pagerank" else tolerance
        probes[name] = {
            "scope": "saved top 15 rows only",
            "cpu_expected_top_product_ids": [id_to_node[vertex]["product_id"] for vertex in ordered_ids[:len(rows)]],
            "actual_top_product_ids": [row["product_id"] for row in rows],
            "top_order_matches": actual_ids == ordered_ids[:len(rows)],
            "max_absolute_score_error": max(errors, default=0.0),
            "mean_absolute_score_error": sum(errors) / len(errors) if errors else None,
            "tolerance": score_tolerance,
            "tolerance_basis": (
                "SQL default epsilon 1e-5" if name == "personalized_pagerank"
                else "configured graph probe tolerance"
            ),
            "matches": (
                actual_ids == ordered_ids[:len(rows)]
                and max(errors, default=0.0) <= score_tolerance
            ),
        }

    # These SQL calls project src,dst only, so their CPU references are unweighted.
    iterative_specs = (
        ("eigenvector_centrality", "eigenvector_score", lambda: nx.eigenvector_centrality(
            directed_unweighted, max_iter=100, tol=1e-5, weight=None
        )),
        ("katz_centrality", "katz_score", lambda: nx.katz_centrality(
            directed_unweighted, alpha=0.001, beta=1.0, max_iter=100, tol=1e-5,
            normalized=True, weight=None,
        )),
    )
    for name, column, calculate in iterative_specs:
        table = actual_tables.get(name)
        if table is None:
            continue
        try:
            values = calculate()
        except Exception as error:
            probes[name] = {"matches": None, "cpu_error": str(error)}
            continue
        ordered_ids = sorted(
            values,
            key=lambda vertex: (-values[vertex], -int(id_to_node[vertex]["viewers"]),
                                id_to_node[vertex]["product_id"], vertex),
        )
        probes[name] = compare_node_top(
            table, (column,),
            {vertex: {column: float(value)} for vertex, value in values.items()},
            ordered_ids, product_to_id, tolerance, 1e-5,
        )
        probes[name]["semantics"] = (
            "NetworkX directed unweighted reference, matching SQL input projection "
            "SELECT src, dst."
        )

    hits_table = actual_tables.get("hits")
    if hits_table is not None:
        try:
            hubs, authorities = nx.hits(
                directed_unweighted, max_iter=100, tol=1e-5, normalized=True
            )
            ordered_ids = sorted(
                authorities,
                key=lambda vertex: (-authorities[vertex], -hubs[vertex],
                                    id_to_node[vertex]["product_id"], vertex),
            )
            probes["hits"] = compare_node_top(
                hits_table, ("hub_score", "authority_score"),
                {vertex: {"hub_score": hubs[vertex], "authority_score": authorities[vertex]}
                 for vertex in hubs},
                ordered_ids, product_to_id, tolerance, 1e-5,
            )
            probes["hits"]["semantics"] = (
                "NetworkX HITS on an unweighted directed graph, matching SQL input "
                "projection SELECT src, dst; epsilon 1e-5, 100 iterations, normalized."
            )
        except Exception as error:
            probes["hits"] = {"matches": None, "cpu_error": str(error)}

    bc_table = actual_tables.get("betweenness_centrality")
    if bc_table is not None:
        bc = nx.betweenness_centrality(directed, normalized=False, weight=None)
        ordered_ids = sorted(
            bc,
            key=lambda vertex: (-bc[vertex], id_to_node[vertex]["product_id"], vertex),
        )
        probes["betweenness_centrality"] = compare_node_top(
            bc_table, ("betweenness",),
            {vertex: {"betweenness": value} for vertex, value in bc.items()},
            ordered_ids, product_to_id, tolerance, 1e-6,
        )
        probes["betweenness_centrality"]["semantics"] = (
            "NetworkX directed, unweighted, exact betweenness with normalized=false, "
            "matching the SQL parameter."
        )
        rows = table_records(bc_table)
        relative_errors = [
            abs(
                float(row["betweenness"])
                - bc[product_to_id[row["product_id"]]]
            ) / abs(bc[product_to_id[row["product_id"]]])
            for row in rows
            if bc[product_to_id[row["product_id"]]] != 0
        ]
        probes["betweenness_centrality"]["max_relative_error_on_saved_rows"] = max(
            relative_errors, default=0.0
        )
        probes["betweenness_centrality"]["precision_note"] = (
            "Keep the configured score tolerance unchanged. Report relative error "
            "separately from the exact top-row order because the saved values have "
            "float32-like precision."
        )

    edge_bc_table = actual_tables.get("edge_betweenness_centrality")
    if edge_bc_table is not None:
        edge_bc = nx.edge_betweenness_centrality(directed, normalized=True, weight=None)
        ordered_edges = sorted(
            edge_bc,
            key=lambda pair: (-edge_bc[pair], id_to_node[pair[0]]["product_id"],
                              id_to_node[pair[1]]["product_id"]),
        )
        rows = table_records(edge_bc_table)
        actual_edges = [
            (product_to_id[row["source_product"]], product_to_id[row["destination_product"]])
            for row in rows
        ]
        errors = [
            abs(float(row["edge_betweenness"]) - edge_bc[edge])
            for row, edge in zip(rows, actual_edges)
        ]
        probes["edge_betweenness_centrality"] = {
            "scope": "saved top 15 directed edge rows only",
            "cpu_reference": "NetworkX exact directed, unweighted, normalized edge betweenness",
            "cpu_expected_top_edges": [
                [id_to_node[src]["product_id"], id_to_node[dst]["product_id"]]
                for src, dst in ordered_edges[:len(rows)]
            ],
            "actual_top_edges": [
                [row["source_product"], row["destination_product"]] for row in rows
            ],
            "top_order_matches": actual_edges == ordered_edges[:len(rows)],
            "max_absolute_score_error": max(errors, default=0.0),
            "mean_absolute_score_error": sum(errors) / len(errors) if errors else None,
            "tolerance": tolerance,
        }
        probes["edge_betweenness_centrality"]["matches"] = (
            probes["edge_betweenness_centrality"]["top_order_matches"]
            and max(errors, default=0.0) <= tolerance
        )

    composition_path = (
        REPOSITORY_ROOT / "docs/research/merrec/results/20_cross_category_discovery.json"
    )
    composition_result: dict[str, Any] = {
        "status": "not_checked",
        "query_path": "docs/research/merrec/sql/20_cross_category_discovery.sql",
    }
    if composition_path.is_file():
        query_record = json.loads(composition_path.read_text())
        composition_ppr = nx.pagerank(
            directed, alpha=0.85, personalization={seed_id: 1.0},
            weight="weight", tol=1e-10, max_iter=1000,
        )
        ppr_order = sorted(composition_ppr, key=lambda vertex: (-composition_ppr[vertex], vertex))
        ppr_candidates = [vertex for vertex in ppr_order if vertex != seed_id][:100]
        seed_users = audience_by_item[seed_id]
        candidate_rows = []
        for vertex in ppr_candidates:
            node = id_to_node[vertex]
            if node["category"] != "Toys & Collectibles" or int(node["viewers"]) < 30:
                continue
            candidate_users = audience_by_item[vertex]
            union_size = len(seed_users | candidate_users)
            score = len(seed_users & candidate_users) / union_size if union_size else 0.0
            candidate_rows.append(
                {
                    "product_id": node["product_id"],
                    "brand_name": node["brand_name"],
                    "category": node["category"],
                    "leaf_category": node["leaf_category"],
                    "viewers": int(node["viewers"]),
                    "audience_jaccard": score,
                    "ppr": composition_ppr[vertex],
                    "vertex": vertex,
                }
            )
        candidate_rows.sort(
            key=lambda row: (-row["audience_jaccard"], row["product_id"])
        )
        cpu_top = candidate_rows[:10]
        saved_rows = query_record.get("sample", [])
        saved_ids = [row["product_id"] for row in saved_rows]
        cpu_ids = [row["product_id"] for row in cpu_top]
        cpu_by_id = {row["product_id"]: row for row in candidate_rows}
        score_errors = [
            abs(
                float(row["audience_jaccard"])
                - cpu_by_id[row["product_id"]]["audience_jaccard"]
            )
            for row in saved_rows if row["product_id"] in cpu_by_id
        ]
        composition_result = {
            "status": query_record.get("status", "unknown"),
            "query_path": "docs/research/merrec/sql/20_cross_category_discovery.sql",
            "query_sha256": __import__("hashlib").sha256(
                query_record.get("sql", "").encode()
            ).hexdigest() if query_record.get("sql") else None,
            "cpu_reference": (
                "NetworkX weighted personalized PageRank from seed 672; select top 100 "
                "excluding seed by value desc, vertex; compute audience Jaccard; retain "
                "Toys & Collectibles with at least 30 viewers; sort Jaccard desc, product_id."
            ),
            "candidate_limit": 100,
            "cpu_top100_last_pagerank": composition_ppr[ppr_candidates[-1]] if ppr_candidates else None,
            "qualified_candidates_after_category_and_viewer_filter": len(candidate_rows),
            "cpu_top10": [
                {key: value for key, value in row.items() if key not in ("ppr", "vertex")}
                for row in cpu_top
            ],
            "saved_top10": saved_rows,
            "top_product_order_matches": saved_ids == cpu_ids,
            "max_absolute_jaccard_error_on_saved_rows": max(score_errors, default=0.0),
            "saved_rows_have_cpu_candidate": len(score_errors) == len(saved_rows),
            "score_tolerance": tolerance,
            "matches": (
                query_record.get("status") == "ok"
                and saved_ids == cpu_ids
                and len(score_errors) == len(saved_rows)
                and max(score_errors, default=0.0) <= tolerance
            ),
        }

    stochastic = {
        name: {
            "status": "skipped_exact_comparison",
            "reason": reason,
            "rows": actual_tables[name].num_rows if name in actual_tables else None,
        }
        for name, reason in STOCHASTIC_PROBES.items()
    }
    return {
        "reference_graph": {
            "vertices": directed.number_of_nodes(),
            "directed_edges": directed.number_of_edges(),
            "undirected_edges": undirected.number_of_edges(),
            "audience_membership_edges": audience.num_rows,
            "seed_product_id": "15928_1315",
            "seed_vertex_id": seed_id,
            "audience_sizes_match_node_viewer_counts": not audience_size_mismatches,
            "audience_size_mismatch_products": audience_size_mismatches,
        },
        "similarity": similarity_probes,
        "deterministic": {
            name: result for name, result in probes.items()
        },
        "composition20_cross_category_discovery": composition_result,
        "stochastic": stochastic,
        "probe_inventory": inventory,
        "executed_sql_location": "docs/research/merrec/results/cugraph_<name>.json",
        "numeric_tolerance": tolerance,
    }
