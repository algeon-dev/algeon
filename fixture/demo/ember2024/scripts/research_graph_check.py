#!/usr/bin/env python3
"""Validate EMBER2024 cuGraph probes against CPU graph definitions."""

from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import sys

import networkx as nx
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "target/ember2024-research"
REPORT = ROOT / "docs/research/ember2024"
RESULTS = REPORT / "results"


class Checks:
    def __init__(self) -> None:
        self.values: dict[str, dict] = {}
        self.failures: list[dict] = []

    def add(self, name: str, passed: bool, **details) -> None:
        self.values[name] = {"passed": bool(passed), **details}
        if not passed:
            self.failures.append({"check": name, **details})


def read_table(name: str) -> pa.Table:
    return pq.read_table(DATA / f"{name}.parquet")


def as_map(table: pa.Table, key: str, value: str) -> dict[int, object]:
    return {int(row[key]): row[value] for row in table.select([key, value]).to_pylist()}


def as_pair_set(table: pa.Table, source: str, destination: str) -> set[tuple[int, int]]:
    return {
        (int(row[source]), int(row[destination]))
        for row in table.select([source, destination]).to_pylist()
    }


def canonical_edges(edges) -> set[tuple[int, int]]:
    return {(min(int(u), int(v)), max(int(u), int(v))) for u, v in edges}


def group_partition(table: pa.Table, vertex_name: str, label_name: str) -> tuple[set[int], set[frozenset[int]]]:
    groups: dict[int, set[int]] = defaultdict(set)
    vertices: set[int] = set()
    for row in table.select([vertex_name, label_name]).to_pylist():
        vertex = int(row[vertex_name])
        label = int(row[label_name])
        vertices.add(vertex)
        groups[label].add(vertex)
    return vertices, {frozenset(group) for group in groups.values()}


def expected_partition(groups) -> set[frozenset[int]]:
    return {frozenset(int(vertex) for vertex in group) for group in groups}


def score_map(table: pa.Table, key: str, value: str) -> dict[int, float]:
    return {int(row[key]): float(row[value]) for row in table.select([key, value]).to_pylist()}


def max_abs_map_error(actual: dict, expected: dict) -> float:
    keys = set(actual) | set(expected)
    if not keys:
        return 0.0
    return max(abs(float(actual.get(key, math.nan)) - float(expected.get(key, math.nan))) for key in keys)


def sparse_left_product(graph: nx.DiGraph, vector: dict[int, float]) -> dict[int, float]:
    result = dict.fromkeys(graph.nodes, 0.0)
    for source, destination in graph.edges:
        result[destination] += vector[source]
    return result


def sparse_right_product(graph: nx.DiGraph, vector: dict[int, float]) -> dict[int, float]:
    result = dict.fromkeys(graph.nodes, 0.0)
    for source, destination in graph.edges:
        result[source] += vector[destination]
    return result


def norm(values) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def relative_eigen_residual(vector: dict[int, float], product: dict[int, float]) -> tuple[float, float]:
    numerator = sum(vector[node] * product[node] for node in vector)
    denominator = sum(vector[node] * vector[node] for node in vector)
    eigenvalue = numerator / denominator
    residual = norm(product[node] - eigenvalue * vector[node] for node in vector)
    scale = max(norm(product.values()), 1.0e-300)
    return eigenvalue, residual / scale


def output_path(name: str) -> Path:
    return DATA / f"{name}.parquet"


def run() -> dict:
    checks = Checks()
    observations: dict[str, object] = {}

    import_edges = read_table("input_48_challenge_import_edges")
    import_symbols = read_table("input_48_challenge_import_symbols")
    sample_counts = read_table("input_48_challenge_import_sample_counts")
    directed_edges = read_table("input_49_challenge_knn_directed")
    affinity_edges = read_table("input_49_challenge_knn_affinity")
    distance_edges = read_table("input_49_challenge_knn_distance")
    primary_seed_row = read_table("input_49_challenge_graph_seed").to_pylist()[0]
    primary_seed = int(primary_seed_row["id"])

    import_edge_rows = [(int(row["src_id"]), int(row["dst_id"])) for row in import_edges.to_pylist()]
    import_directed = nx.DiGraph()
    import_directed.add_edges_from(import_edge_rows)
    import_undirected = nx.Graph()
    import_undirected.add_edges_from(import_edge_rows)

    knn_edge_rows = [(int(row["src_id"]), int(row["dst_id"])) for row in directed_edges.to_pylist()]
    knn_directed = nx.DiGraph()
    knn_directed.add_edges_from(knn_edge_rows)
    knn_affinity = nx.Graph()
    knn_affinity.add_weighted_edges_from(
        (int(row["src_id"]), int(row["dst_id"]), float(row["weight"]))
        for row in affinity_edges.to_pylist()
    )
    knn_distance = nx.Graph()
    knn_distance.add_weighted_edges_from(
        (int(row["src_id"]), int(row["dst_id"]), float(row["distance"]))
        for row in distance_edges.to_pylist()
    )

    import_samples = {int(node) for node in import_directed if int(node) < 100_000_000}
    import_symbols_by_sample: dict[int, set[int]] = defaultdict(set)
    for source, symbol in import_edge_rows:
        import_symbols_by_sample[source].add(symbol)
    symbol_names = {
        int(row["symbol_id"]): row["symbol"] for row in import_symbols.to_pylist()
    }
    retained_count = {int(row["id"]): int(row["retained_import_count"]) for row in sample_counts.to_pylist()}

    checks.add(
        "input_graph_shapes",
        len(import_edge_rows) == 80_673
        and import_undirected.number_of_nodes() == 5_914
        and len(import_samples) == 926
        and len(symbol_names) == 4_988
        and directed_edges.num_rows == 16_125
        and knn_directed.number_of_nodes() == 3_225
        and affinity_edges.num_rows == 12_479
        and distance_edges.num_rows == 12_479,
        import_edges=len(import_edge_rows),
        import_edge_vertices=import_undirected.number_of_nodes(),
        import_samples_with_retained_edges=len(import_samples),
        import_symbols=len(symbol_names),
        knn_directed_edges=directed_edges.num_rows,
        knn_directed_vertices=knn_directed.number_of_nodes(),
        knn_undirected_pairs=affinity_edges.num_rows,
        distance_pairs=distance_edges.num_rows,
    )
    zero_import_samples = sum(count == 0 for count in retained_count.values())
    observations["graph_coverage"] = {
        "challenge_win32_rows": len(retained_count),
        "import_samples_with_retained_edges": len(import_samples),
        "import_samples_with_zero_retained_edges": zero_import_samples,
        "import_edge_only_vertex_count": import_undirected.number_of_nodes(),
        "import_symbol_count": len(symbol_names),
        "knn_vertex_count": knn_directed.number_of_nodes(),
        "knn_directed_nonself_edge_count": directed_edges.num_rows,
        "knn_nonself_outdegree_min": min(dict(knn_directed.out_degree()).values()),
        "knn_nonself_outdegree_max": max(dict(knn_directed.out_degree()).values()),
        "knn_undirected_pair_count": affinity_edges.num_rows,
        "knn_undirected_connected_components": nx.number_connected_components(knn_distance),
        "knn_core_number_values": sorted(set(nx.core_number(knn_affinity).values())),
        "import_primary_seed": primary_seed_row,
        "caveat": "Import-graph output omits 2,299 samples with zero retained imports; it is an edge-only vertex domain.",
    }

    metadata = pq.read_table(
        DATA / "input_metadata_challenge.parquet",
        columns=["id", "file_type", "sha256", "family", "label"],
    )
    metadata = metadata.filter(pc.equal(metadata["file_type"], "Win32"))
    metadata_by_id = {
        int(row["id"]): row for row in metadata.to_pylist()
    }
    graph_metadata = {node: metadata_by_id[node] for node in knn_directed.nodes}
    known_families = Counter(
        row["family"] for row in graph_metadata.values() if row["family"]
    )
    observations["known_family_coverage"] = {
        "win32_knn_samples": len(graph_metadata),
        "samples_with_known_family": sum(known_families.values()),
        "samples_with_blank_family": len(graph_metadata) - sum(known_families.values()),
        "known_family_count": len(known_families),
        "most_common_families": [
            {"family": family, "samples": count}
            for family, count in known_families.most_common(10)
        ],
    }

    # Explicit pair sets are sample-to-sample comparisons of retained imports.
    pair_rows: list[tuple[int, int]] = []
    for candidate, count in retained_count.items():
        if candidate != primary_seed and count >= 5:
            pair_rows.append((primary_seed, candidate))
    pair_set = set(pair_rows)
    neighbor_sets = import_symbols_by_sample
    expected_scores: dict[str, dict[tuple[int, int], float]] = {
        "cosine": {},
        "jaccard": {},
        "overlap": {},
        "sorensen": {},
    }
    conventional_cosine_scores: dict[tuple[int, int], float] = {}
    for first, second in pair_set:
        left = neighbor_sets[first]
        right = neighbor_sets[second]
        shared = len(left & right)
        union = len(left | right)
        expected_scores["cosine"][(first, second)] = 1.0 if shared else 0.0
        denominator = math.sqrt(len(left) * len(right))
        conventional_cosine_scores[(first, second)] = shared / denominator if denominator else 0.0
        expected_scores["jaccard"][(first, second)] = shared / union if union else 0.0
        expected_scores["overlap"][(first, second)] = shared / min(len(left), len(right))
        expected_scores["sorensen"][(first, second)] = 2.0 * shared / (len(left) + len(right))

    pair_check_results = {}
    for name in ["cosine", "jaccard", "overlap", "sorensen"]:
        result_name = {
            "cosine": "63_cugraph_cosine",
            "jaccard": "71_cugraph_jaccard",
            "overlap": "78_cugraph_overlap",
            "sorensen": "81_cugraph_sorensen",
        }[name]
        result = read_table(result_name)
        result_rows = result.to_pylist()
        actual_pairs = {(int(row["first"]), int(row["second"])) for row in result_rows}
        actual_scores = {
            (int(row["first"]), int(row["second"])): float(row["similarity"])
            for row in result_rows
        }
        max_error = max_abs_map_error(actual_scores, expected_scores[name])
        pair_matches = actual_pairs == pair_set
        passed = pair_matches and max_error <= 1.0e-6
        details = {
            "pairs_expected": len(pair_set),
            "pairs_returned": len(actual_pairs),
            "pair_set_exact": pair_matches,
            "max_abs_score_error": max_error,
            "absolute_tolerance": 1.0e-6,
        }
        check_name = (
            "similarity_cosine_native_unweighted_behavior"
            if name == "cosine"
            else f"similarity_{name}"
        )
        checks.add(check_name, passed, **details)
        pair_check_results[name] = details

    cosine_result = read_table("63_cugraph_cosine").to_pylist()
    cosine_actual = {
        (int(row["first"]), int(row["second"])): float(row["similarity"])
        for row in cosine_result
    }
    cosine_common_pairs = pair_set & set(cosine_actual)
    cosine_conventional_errors = np.asarray(
        [
            abs(cosine_actual[pair] - conventional_cosine_scores[pair])
            for pair in cosine_common_pairs
        ],
        dtype=np.float64,
    )
    cosine_expected_values = np.asarray(
        [conventional_cosine_scores[pair] for pair in cosine_common_pairs], dtype=np.float64
    )
    cosine_observed_values = np.asarray(
        [cosine_actual[pair] for pair in cosine_common_pairs], dtype=np.float64
    )
    cosine_example_pair = (primary_seed, 20_005_223)
    cosine_example_left = neighbor_sets[cosine_example_pair[0]]
    cosine_example_right = neighbor_sets[cosine_example_pair[1]]
    cosine_example_intersection = len(cosine_example_left & cosine_example_right)
    cosine_example_denominator = math.sqrt(len(cosine_example_left) * len(cosine_example_right))
    observations["cosine_public_semantic_gap"] = {
        "native_parity_behavior": "For unweighted explicit pairs, the current libcugraph path returns 1 when the neighbor intersection is nonempty and 0 otherwise. This is also the local hand-verified parity fixture behavior.",
        "documented_conventional_formula": "intersection size divided by sqrt(degree(first) * degree(second)) for binary adjacency vectors",
        "pair_count": len(pair_set),
        "native_pair_set_exact": set(cosine_actual) == pair_set,
        "pairs_missing_from_native_output": len(pair_set - set(cosine_actual)),
        "unexpected_native_pairs": len(set(cosine_actual) - pair_set),
        "pairs_with_nonempty_intersection": sum(
            bool(neighbor_sets[first] & neighbor_sets[second]) for first, second in pair_set
        ),
        "pairs_mismatching_conventional_formula_at_1e_6": int(
            np.count_nonzero(cosine_conventional_errors > 1.0e-6)
        ),
        "max_abs_error_vs_conventional_formula": float(
            cosine_conventional_errors.max(initial=0.0)
        ),
        "absolute_error_quantiles": {
            "p50": float(np.quantile(cosine_conventional_errors, 0.50)),
            "p90": float(np.quantile(cosine_conventional_errors, 0.90)),
            "p99": float(np.quantile(cosine_conventional_errors, 0.99)),
            "max": float(cosine_conventional_errors.max(initial=0.0)),
        },
        "conventional_score_range": [
            float(cosine_expected_values.min(initial=0.0)),
            float(cosine_expected_values.max(initial=0.0)),
        ],
        "native_score_values": sorted(float(value) for value in np.unique(cosine_observed_values)),
        "example_pair": {
            "first": cosine_example_pair[0],
            "second": cosine_example_pair[1],
            "first_degree": len(cosine_example_left),
            "second_degree": len(cosine_example_right),
            "intersection_size": cosine_example_intersection,
            "conventional_cosine": cosine_example_intersection / cosine_example_denominator,
            "observed_native_score": cosine_actual[cosine_example_pair],
        },
        "finding": "The SQL result matches the current native/parity behavior, while the NVIDIA Python API documentation and local SQL wording describe conventional cosine. Classify this as a public semantic gap; no engine-defect claim follows from this query.",
        "sources": {
            "local_public_sql_description": "website/docs/cugraph/udtfs/similarity/cugraph_cosine.mdx",
            "upstream_python_api_docstring": "components/cugraph/cugraph/python/cugraph/cugraph/link_prediction/cosine.py",
            "local_native_pairwise_path": "components/cugraph/cugraph/cpp/src/link_prediction/detail/similarity_impl.cuh",
            "local_parity_fixture": "crates/algeon-datafusion/src/cugraph_sql/parity.rs",
            "upstream_documentation_url": "https://docs.rapids.ai/api/cugraph/legacy/api_docs/api/cugraph/cugraph.link_prediction.cosine/",
        },
    }
    observations["similarity_pairs"] = {
        "primary_sample_id": primary_seed,
        "candidate_rule": "retained_import_count >= 5 and candidate_id != primary seed",
        "candidate_count": len(pair_set),
        "checks": pair_check_results,
    }
    observations["similarity_contracts"] = {
        "cosine": "Observed native behavior for unweighted explicit pairs: return 1 for any non-empty common-neighbor intersection and 0 otherwise. The conventional adjacency-cosine comparison is recorded separately under cosine_public_semantic_gap.",
        "jaccard": "intersection size divided by union size of retained import-symbol sets",
        "overlap": "intersection size divided by the smaller retained import-symbol set size",
        "sorensen": "twice the intersection size divided by the sum of retained import-symbol set sizes",
        "source": "crates/algeon-datafusion/src/cugraph_sql/parity.rs explicit-pair cosine hand-verified contract",
    }

    degree_result = read_table("64_cugraph_degrees_all")
    degree_by_vertex = {
        int(row["vertex"]): (int(row["in_degree"]), int(row["out_degree"]))
        for row in degree_result.to_pylist()
    }
    expected_degrees = {
        int(vertex): (int(in_degree), int(out_degree))
        for vertex, in_degree, out_degree in (
            (vertex, import_directed.in_degree(vertex), import_directed.out_degree(vertex))
            for vertex in import_directed.nodes
        )
    }
    degrees_exact = degree_by_vertex == expected_degrees
    checks.add(
        "directed_in_out_degrees",
        degrees_exact,
        vertex_count=len(degree_by_vertex),
        expected_vertex_count=import_directed.number_of_nodes(),
        all_degree_pairs_exact=degrees_exact,
    )
    degree_component_errors = {}
    for name, column, value_index, graph_direction in [
        ("70_cugraph_in_degrees_all", "in_degree", 0, "in"),
        ("77_cugraph_out_degrees_all", "out_degree", 1, "out"),
    ]:
        table = read_table(name)
        actual = {int(row["vertex"]): int(row[column]) for row in table.to_pylist()}
        expected = {
            vertex: pair[value_index] for vertex, pair in expected_degrees.items()
        }
        passed = actual == expected
        checks.add(
            f"{graph_direction}_degrees_all",
            passed,
            vertex_count=len(actual),
            exact=passed,
        )
        degree_component_errors[graph_direction] = passed
    observations["directed_degrees"] = {
        "input_edges": import_directed.number_of_edges(),
        "input_vertices": import_directed.number_of_nodes(),
        "isolated_import_sample_vertices_excluded": zero_import_samples,
        "all_degrees_exact": degrees_exact,
        "in_degrees_exact": degree_component_errors["in"],
        "out_degrees_exact": degree_component_errors["out"],
    }

    expected_core = nx.core_number(import_undirected)
    core_result = read_table("62_cugraph_core_number")
    actual_core = {int(row["vertex"]): int(row["core_number"]) for row in core_result.to_pylist()}
    core_error = max_abs_map_error(actual_core, expected_core)
    checks.add(
        "import_core_number",
        set(actual_core) == set(expected_core) and core_error == 0.0,
        vertices=len(actual_core),
        max_abs_error=core_error,
    )

    expected_kcore = nx.k_core(import_undirected, k=10)
    kcore_result = read_table("72_cugraph_k_core")
    kcore_output_edges = [(int(row["src"]), int(row["dst"])) for row in kcore_result.to_pylist()]
    expected_kcore_edges = canonical_edges(expected_kcore.edges)
    actual_kcore_edges = canonical_edges(kcore_output_edges)
    kcore_passed = actual_kcore_edges == expected_kcore_edges
    checks.add(
        "import_k_core_10",
        kcore_passed,
        directed_rows=len(kcore_output_edges),
        canonical_edges=len(actual_kcore_edges),
        expected_canonical_edges=len(expected_kcore_edges),
        canonical_edge_set_exact=kcore_passed,
        rows_per_canonical_edge=(len(kcore_output_edges) / len(actual_kcore_edges)) if actual_kcore_edges else None,
    )
    observations["import_core_structure"] = {
        "core_number_min": min(expected_core.values()),
        "core_number_max": max(expected_core.values()),
        "k_core_k": 10,
        "k_core_nodes": expected_kcore.number_of_nodes(),
        "k_core_canonical_edges": expected_kcore.number_of_edges(),
        "kNN_core_number_values": sorted(set(nx.core_number(knn_affinity).values())),
        "kNN_core_note": "The prepared kNN affinity graph has core number 5 for every vertex; k-core values <=5 retain all vertices and values >=6 are empty.",
    }

    expected_triangles = nx.triangles(knn_affinity)
    triangle_table = read_table("85_cugraph_triangle_count_all")
    actual_triangles = {int(row["vertex"]): int(row["triangle_count"]) for row in triangle_table.to_pylist()}
    triangle_error = max_abs_map_error(actual_triangles, expected_triangles)
    triangle_total = sum(expected_triangles.values()) // 3
    checks.add(
        "triangle_counts",
        set(actual_triangles) == set(expected_triangles) and triangle_error == 0.0 and triangle_total == 8_525,
        vertices=len(actual_triangles),
        max_abs_count_error=triangle_error,
        expected_graph_triangle_count=triangle_total,
        expected_triangle_count=8_525,
    )

    components = {}
    for result_name, label_col, expected_groups in [
        ("84_cugraph_strongly_connected_components", "component", nx.strongly_connected_components(knn_directed)),
        ("86_cugraph_weakly_connected_components", "component", nx.weakly_connected_components(knn_directed)),
    ]:
        table = read_table(result_name)
        result_vertices, result_groups = group_partition(table, "vertex", label_col)
        expected_groups_set = expected_partition(expected_groups)
        passed = result_vertices == set(knn_directed.nodes) and result_groups == expected_groups_set
        checks.add(
            result_name,
            passed,
            vertices=len(result_vertices),
            component_count=len(result_groups),
            expected_component_count=len(expected_groups_set),
            memberships_exact_ignoring_label_ids=passed,
        )
        components[result_name] = {
            "component_count": len(result_groups),
            "largest_component": max((len(group) for group in result_groups), default=0),
            "smallest_component": min((len(group) for group in result_groups), default=0),
            "membership_exact_ignoring_label_ids": passed,
        }
    observations["components"] = components

    bfs_table = read_table("61_cugraph_bfs")
    bfs_rows = bfs_table.to_pylist()
    bfs_expected = nx.single_source_shortest_path_length(import_undirected, primary_seed)
    bfs_actual = {int(row["vertex"]): row for row in bfs_rows}
    bfs_dist_errors = [
        abs(int(bfs_actual[vertex]["distance"]) - distance)
        for vertex, distance in bfs_expected.items()
        if vertex in bfs_actual and bfs_actual[vertex]["distance"] is not None
    ]
    bfs_pred_valid = True
    bfs_pred_violations = []
    for vertex, row in bfs_actual.items():
        predecessor = row["predecessor"]
        distance = row["distance"]
        if vertex == primary_seed:
            bfs_pred_valid &= predecessor is None
        elif predecessor is None or predecessor not in import_undirected or not import_undirected.has_edge(vertex, predecessor):
            bfs_pred_valid = False
            bfs_pred_violations.append(vertex)
        elif predecessor not in bfs_expected or distance is None or bfs_expected[predecessor] + 1 != distance:
            bfs_pred_valid = False
            bfs_pred_violations.append(vertex)
    bfs_reachability = {int(row["vertex"]) for row in bfs_rows if row["reachable"]}
    bfs_passed = (
        set(bfs_actual) == set(import_undirected.nodes)
        and bfs_reachability == set(bfs_expected)
        and len(bfs_dist_errors) == len(bfs_expected)
        and max(bfs_dist_errors, default=0) == 0
        and bfs_pred_valid
    )
    checks.add(
        "bfs_import_undirected",
        bfs_passed,
        vertices=len(bfs_actual),
        reachable=len(bfs_reachability),
        expected_reachable=len(bfs_expected),
        max_hop_distance_error=max(bfs_dist_errors, default=0),
        predecessor_valid=bfs_pred_valid,
        invalid_predecessor_count=len(bfs_pred_violations),
    )

    sssp_table = read_table("83_cugraph_sssp")
    sssp_rows = sssp_table.to_pylist()
    sssp_actual = {int(row["vertex"]): row for row in sssp_rows}
    sssp_expected = nx.single_source_dijkstra_path_length(knn_distance, primary_seed, weight="weight")
    sssp_distance_errors = []
    sssp_pred_valid = True
    sssp_pred_violations = []
    unreachable_values = Counter()
    sssp_tolerance = {"atol": 1.0e-5, "rtol": 1.0e-5}
    for vertex, row in sssp_actual.items():
        distance = float(row["distance"])
        predecessor = row["predecessor"]
        if vertex not in sssp_expected:
            unreachable_values[repr(distance)] += 1
            if predecessor != -1:
                sssp_pred_valid = False
                sssp_pred_violations.append(vertex)
            continue
        expected_distance = float(sssp_expected[vertex])
        sssp_distance_errors.append(abs(distance - expected_distance))
        if vertex == primary_seed:
            if predecessor != -1:
                sssp_pred_valid = False
                sssp_pred_violations.append(vertex)
        elif (
            predecessor not in knn_distance
            or not knn_distance.has_edge(vertex, predecessor)
            or predecessor not in sssp_expected
            or abs(
                float(sssp_expected[predecessor])
                + float(knn_distance[vertex][predecessor]["weight"])
                - expected_distance
            ) > sssp_tolerance["atol"]
        ):
            sssp_pred_valid = False
            sssp_pred_violations.append(vertex)
    sssp_max_abs = max(sssp_distance_errors, default=0.0)
    sssp_passed = (
        set(sssp_actual) == set(knn_distance.nodes)
        and sssp_pred_valid
        and all(
            math.isclose(float(sssp_actual[vertex]["distance"]), float(expected), rel_tol=sssp_tolerance["rtol"], abs_tol=sssp_tolerance["atol"])
            for vertex, expected in sssp_expected.items()
        )
    )
    checks.add(
        "sssp_weighted_distance",
        sssp_passed,
        vertices=len(sssp_actual),
        reachable=len(sssp_expected),
        unreachable=len(sssp_actual) - len(sssp_expected),
        max_abs_distance_error=sssp_max_abs,
        tolerance=sssp_tolerance,
        predecessor_valid=sssp_pred_valid,
        invalid_predecessor_count=len(sssp_pred_violations),
        unreachable_distance_values=dict(unreachable_values),
    )

    mst_table = read_table("76_cugraph_minimum_spanning_tree")
    mst_rows = [(int(row["src"]), int(row["dst"])) for row in mst_table.to_pylist()]
    mst_expected = nx.minimum_spanning_tree(knn_distance, weight="weight")
    mst_edges = canonical_edges(mst_rows)
    distance_by_canonical_edge = {
        (min(int(row["src_id"]), int(row["dst_id"])), max(int(row["src_id"]), int(row["dst_id"])))
        : float(row["distance"])
        for row in distance_edges.to_pylist()
    }
    mst_graph = nx.Graph()
    mst_graph.add_nodes_from(knn_distance.nodes)
    mst_graph.add_weighted_edges_from(
        (source, destination, distance_by_canonical_edge[(source, destination)])
        for source, destination in mst_edges
    )
    graph_partition = expected_partition(nx.connected_components(knn_distance))
    mst_partition = expected_partition(nx.connected_components(mst_graph))
    mst_reference_weight = float(mst_expected.size(weight="weight"))
    mst_returned_weight = float(mst_graph.size(weight="weight"))
    mst_weight_error = abs(mst_reference_weight - mst_returned_weight)
    mst_expected_edge_count = knn_distance.number_of_nodes() - nx.number_connected_components(knn_distance)
    mst_passed = (
        len(mst_edges) == mst_expected_edge_count
        and nx.is_forest(mst_graph)
        and mst_partition == graph_partition
        and mst_weight_error <= 1.0e-6
    )
    checks.add(
        "minimum_spanning_forest",
        mst_passed,
        output_rows=len(mst_rows),
        canonical_edges=len(mst_edges),
        expected_edges= mst_expected_edge_count,
        components=len(graph_partition),
        forest_acyclic=nx.is_forest(mst_graph),
        spans_each_input_component=mst_partition == graph_partition,
        returned_total_distance=mst_returned_weight,
        networkx_total_distance=mst_reference_weight,
        absolute_total_weight_error=mst_weight_error,
        ties_allow_different_edges=True,
    )

    pagerank_table = read_table("79_cugraph_pagerank")
    pagerank_gpu = score_map(pagerank_table, "vertex", "value")
    pagerank_cpu = nx.pagerank(knn_affinity, alpha=0.85, tol=1.0e-10, max_iter=1000, weight="weight")
    pagerank_errors = np.asarray([abs(pagerank_gpu[node] - pagerank_cpu[node]) for node in pagerank_cpu])
    pagerank_l1 = float(pagerank_errors.sum())
    pagerank_max = float(pagerank_errors.max(initial=0.0))
    pagerank_passed = (
        set(pagerank_gpu) == set(pagerank_cpu)
        and pagerank_l1 <= 1.0e-4
        and all(math.isfinite(value) and value >= 0.0 for value in pagerank_gpu.values())
    )
    checks.add(
        "pagerank_affinity",
        pagerank_passed,
        vertices=len(pagerank_gpu),
        cpu_tolerance=1.0e-10,
        l1_error=pagerank_l1,
        max_abs_error=pagerank_max,
        l1_tolerance=1.0e-4,
    )

    ppr_table = read_table("80_cugraph_personalized_pagerank")
    ppr_gpu = score_map(ppr_table, "vertex", "value")
    ppr_cpu = nx.pagerank(
        import_undirected,
        alpha=0.85,
        personalization={primary_seed: 1.0},
        tol=1.0e-10,
        max_iter=1000,
        weight=None,
    )
    ppr_errors = np.asarray([abs(ppr_gpu[node] - ppr_cpu[node]) for node in ppr_cpu])
    ppr_l1 = float(ppr_errors.sum())
    ppr_max = float(ppr_errors.max(initial=0.0))
    ppr_passed = (
        set(ppr_gpu) == set(ppr_cpu)
        and ppr_l1 <= 1.0e-4
        and all(math.isfinite(value) and value >= 0.0 for value in ppr_gpu.values())
    )
    checks.add(
        "personalized_pagerank_import_undirected",
        ppr_passed,
        vertices=len(ppr_gpu),
        cpu_tolerance=1.0e-10,
        l1_error=ppr_l1,
        max_abs_error=ppr_max,
        l1_tolerance=1.0e-4,
        personalization_vertex=primary_seed,
    )

    eigen_table = read_table("67_cugraph_eigenvector_centrality")
    eigen_gpu = score_map(eigen_table, "vertex", "value")
    eigen_product = sparse_left_product(knn_directed, eigen_gpu)
    eigenvalue, eigen_residual = relative_eigen_residual(eigen_gpu, eigen_product)
    eigen_l2 = norm(eigen_gpu.values())
    eigen_finite_and_normalized = (
        set(eigen_gpu) == set(knn_directed.nodes)
        and all(math.isfinite(value) and value >= 0.0 for value in eigen_gpu.values())
        and abs(eigen_l2 - 1.0) <= 1.0e-6
    )
    checks.add(
        "eigenvector_centrality_finite_and_normalized",
        eigen_finite_and_normalized,
        vertices=len(eigen_gpu),
        orientation="left eigenvector of directed adjacency, using incoming edges",
        l2_norm=eigen_l2,
        dominant_eigenvalue_estimate=eigenvalue,
        relative_residual=eigen_residual,
        residual_is_reported_without_a_separate_threshold=True,
    )

    katz_table = read_table("73_cugraph_katz_centrality")
    katz_gpu = score_map(katz_table, "vertex", "value")
    katz_product = sparse_left_product(knn_directed, katz_gpu)
    katz_residual_values = {
        node: katz_gpu[node] - 0.001 * katz_product[node]
        for node in katz_gpu
    }
    katz_beta_effective = float(np.mean(list(katz_residual_values.values())))
    katz_residual = max(abs(value - katz_beta_effective) for value in katz_residual_values.values())
    katz_l2 = norm(katz_gpu.values())
    katz_cpu = nx.katz_centrality(
        knn_directed,
        alpha=0.001,
        beta=1.0,
        max_iter=1000,
        tol=1.0e-10,
        normalized=True,
        weight=None,
    )
    katz_cpu_l2 = norm(katz_cpu.values())
    katz_cpu_normalized = {int(node): float(value) / katz_cpu_l2 for node, value in katz_cpu.items()}
    katz_cpu_max_error = max_abs_map_error(katz_gpu, katz_cpu_normalized)
    katz_reverse_cpu = nx.katz_centrality(
        knn_directed.reverse(copy=False),
        alpha=0.001,
        beta=1.0,
        max_iter=1000,
        tol=1.0e-10,
        normalized=True,
        weight=None,
    )
    katz_reverse_cpu_max_error = max_abs_map_error(katz_gpu, katz_reverse_cpu)
    katz_reverse_values = np.asarray(list(katz_reverse_cpu.values()), dtype=np.float64)
    katz_finite_and_normalized = (
        set(katz_gpu) == set(knn_directed.nodes)
        and all(math.isfinite(value) and value >= 0.0 for value in katz_gpu.values())
        and abs(katz_l2 - 1.0) <= 1.0e-6
    )
    checks.add(
        "katz_centrality_finite_and_normalized",
        katz_finite_and_normalized,
        vertices=len(katz_gpu),
        orientation="The native C++ iteration sums incoming source scores into each destination; the SQL result matches NetworkX on the input edge direction.",
        l2_norm=katz_l2,
        fitted_normalized_beta=katz_beta_effective,
        max_residual_after_normalization=katz_residual,
        max_abs_error_vs_networkx=katz_cpu_max_error,
        max_abs_error_vs_networkx_reversed=katz_reverse_cpu_max_error,
        networkx_tolerance=1.0e-10,
        incoming_edge_max_residual_after_normalization=katz_residual,
        outgoing_edge_networkx_comparison=katz_reverse_cpu_max_error,
        input_outdegree_values=sorted(set(dict(knn_directed.out_degree()).values())),
        reverse_networkx_score_range=[
            float(katz_reverse_values.min()),
            float(katz_reverse_values.max()),
        ],
        sql_score_range=[float(min(katz_gpu.values())), float(max(katz_gpu.values()))],
        upstream_python_documentation="The checked-out cuGraph Python API docstring describes out-edge Katz and says reversing the NetworkX graph should match.",
        residual_and_reference_error_are_reported_without_a_separate_threshold=True,
    )

    hits_table = read_table("69_cugraph_hits_1000")
    hits_rows = hits_table.to_pylist()
    hubs = {int(row["vertex"]): float(row["hub_score"]) for row in hits_rows}
    authorities = {int(row["vertex"]): float(row["authority_score"]) for row in hits_rows}
    hub_l1 = sum(hubs.values())
    authority_l1 = sum(authorities.values())
    expected_hubs = sparse_right_product(knn_directed, authorities)
    expected_authorities = sparse_left_product(knn_directed, hubs)
    expected_hub_sum = sum(expected_hubs.values())
    expected_authority_sum = sum(expected_authorities.values())
    hits_hub_residual = max(
        abs(hubs[node] - expected_hubs[node] / expected_hub_sum)
        for node in hubs
    )
    hits_authority_residual = max(
        abs(authorities[node] - expected_authorities[node] / expected_authority_sum)
        for node in authorities
    )
    hits_passed = (
        set(hubs) == set(knn_directed.nodes)
        and set(authorities) == set(knn_directed.nodes)
        and all(math.isfinite(value) and value >= 0.0 for value in hubs.values())
        and all(math.isfinite(value) and value >= 0.0 for value in authorities.values())
        and abs(hub_l1 - 1.0) <= 1.0e-6
        and abs(authority_l1 - 1.0) <= 1.0e-6
    )
    checks.add(
        "hits_1000_iteration_finite_and_normalized",
        hits_passed,
        vertices=len(hubs),
        hub_l1_norm=hub_l1,
        authority_l1_norm=authority_l1,
        max_hub_fixed_point_error=hits_hub_residual,
        max_authority_fixed_point_error=hits_authority_residual,
        residuals_are_reported_without_a_separate_threshold=True,
        original_100_iteration_status="failed to converge; native diagnostic is retained in graph_probe_index.json",
    )

    seed_record = json.loads((RESULTS / "graph_seed_selection.json").read_text())
    seed_ids = [int(vertex) for vertex in seed_record["selected_ids"]]
    seed_set = set(seed_ids)
    n_vertices = knn_directed.number_of_nodes()
    raw_vertex_bc = nx.betweenness_centrality_subset(
        knn_directed,
        sources=seed_ids,
        targets=list(knn_directed.nodes),
        normalized=False,
        weight=None,
    )
    expected_vertex_bc = {
        int(vertex): float(value)
        / ((len(seed_ids) - 1 if vertex in seed_set else len(seed_ids)) * (n_vertices - 2))
        for vertex, value in raw_vertex_bc.items()
    }
    vertex_bc_table = read_table("60_cugraph_betweenness_centrality")
    actual_vertex_bc = score_map(vertex_bc_table, "vertex", "value")
    vertex_bc_error = max_abs_map_error(actual_vertex_bc, expected_vertex_bc)
    vertex_bc_passed = set(actual_vertex_bc) == set(expected_vertex_bc) and vertex_bc_error <= 1.0e-6
    checks.add(
        "sampled_vertex_betweenness_64",
        vertex_bc_passed,
        vertices=len(actual_vertex_bc),
        sources=len(seed_ids),
        source_selection="sha256 ASC, id ASC among Win32 kNN vertices",
        max_abs_error=vertex_bc_error,
        tolerance=1.0e-6,
        normalization="NetworkX directed subset raw scores divided by k*(n-2), with sampled vertices divided by (k-1)*(n-2), matching the component implementation.",
    )

    raw_edge_bc = nx.edge_betweenness_centrality_subset(
        knn_directed,
        sources=seed_ids,
        targets=list(knn_directed.nodes),
        normalized=False,
        weight=None,
    )
    expected_edge_bc = {
        (int(source), int(destination)): float(value) / (len(seed_ids) * (n_vertices - 1))
        for (source, destination), value in raw_edge_bc.items()
    }
    edge_bc_table = read_table("66_cugraph_edge_betweenness_centrality")
    actual_edge_bc = {
        (int(row["src"]), int(row["dst"])): float(row["value"])
        for row in edge_bc_table.to_pylist()
    }
    edge_bc_error = max_abs_map_error(actual_edge_bc, expected_edge_bc)
    edge_bc_passed = set(actual_edge_bc) == set(expected_edge_bc) and edge_bc_error <= 1.0e-6
    checks.add(
        "sampled_edge_betweenness_64",
        edge_bc_passed,
        edges=len(actual_edge_bc),
        expected_edges=knn_directed.number_of_edges(),
        sources=len(seed_ids),
        source_selection="same 64 ids selected by sha256 ASC, id ASC",
        max_abs_error=edge_bc_error,
        tolerance=1.0e-6,
        normalization="NetworkX directed subset raw scores divided by k*(n-1), matching the component implementation.",
    )

    # The community labels are opaque; coverage and weighted modularity are reported,
    # while family composition remains post-hoc context only.
    community_results = {}
    for name in [
        "65_cugraph_ecg",
        "74_cugraph_leiden",
        "75_cugraph_louvain",
        "82_cugraph_spectral_modularity_maximization",
    ]:
        table = read_table(name)
        vertices, groups = group_partition(table, "vertex", "community")
        labels = {int(row["vertex"]): int(row["community"]) for row in table.to_pylist()}
        community_values = [set(group) for group in groups]
        modularity = float(nx.algorithms.community.modularity(knn_affinity, community_values, weight="weight"))
        community_sizes = sorted((len(group) for group in groups), reverse=True)
        family_context = []
        for group in sorted(groups, key=lambda group: (-len(group), min(group))):
            family_counts = Counter(
                graph_metadata[vertex]["family"]
                for vertex in group
                if graph_metadata[vertex]["family"]
            )
            if family_counts:
                family_context.append(
                    {
                        "community_size": len(group),
                        "known_family_samples": sum(family_counts.values()),
                        "family_count": len(family_counts),
                        "most_common_families": [
                            {"family": family, "samples": count}
                            for family, count in family_counts.most_common(5)
                        ],
                    }
                )
            if len(family_context) == 5:
                break
        coverage = vertices == set(knn_affinity.nodes) and sum(community_sizes) == len(knn_affinity.nodes)
        finite_modularity = math.isfinite(modularity)
        checks.add(
            f"{name}_partition_coverage",
            coverage and finite_modularity,
            vertices=len(vertices),
            expected_vertices=knn_affinity.number_of_nodes(),
            community_count=len(groups),
            weighted_modularity=modularity,
            finite_modularity=finite_modularity,
        )
        community_results[name] = {
            "community_count": len(groups),
            "largest_community_size": max(community_sizes, default=0),
            "smallest_community_size": min(community_sizes, default=0),
            "weighted_modularity": modularity,
            "family_composition_for_largest_communities": family_context,
            "family_semantics": "post-hoc counts only; not a label-quality or malware-family prediction test",
        }
    observations["communities"] = community_results

    force_table = read_table("68_cugraph_force_atlas2")
    force_x = np.asarray(force_table["x"].to_numpy(zero_copy_only=False), dtype=np.float64)
    force_y = np.asarray(force_table["y"].to_numpy(zero_copy_only=False), dtype=np.float64)
    force_finite = bool(np.isfinite(force_x).all() and np.isfinite(force_y).all())
    force_ranges = {"x": float(np.ptp(force_x)), "y": float(np.ptp(force_y))}
    force_cov_eigenvalues = np.linalg.eigvalsh(np.cov(np.column_stack([force_x, force_y]).T))
    force_non_degenerate = bool(force_ranges["x"] > 0.0 and force_ranges["y"] > 0.0 and force_cov_eigenvalues.min() > 0.0)
    checks.add(
        "force_atlas2_coordinates",
        force_finite and force_non_degenerate,
        vertices=len(force_x),
        finite=force_finite,
        coordinate_ranges=force_ranges,
        covariance_eigenvalues=[float(value) for value in force_cov_eigenvalues],
        non_degenerate=force_non_degenerate,
    )

    # Review-queue SQL is a 15-row convenience output from the same undirected
    # import graph. Validate the join metadata and ranking against saved PPR output.
    review_path = output_path("87_import_review_queue")
    if review_path.exists():
        review_table = pq.read_table(review_path)
        review_rows = review_table.to_pylist()
        ppr_candidates = [
            (int(vertex), float(value))
            for vertex, value in ppr_gpu.items()
            if int(vertex) != primary_seed and retained_count.get(int(vertex), 0) >= 5
        ]
        ppr_candidates.sort(key=lambda row: (-row[1], row[0]))
        ppr_top25 = [vertex for vertex, _ in ppr_candidates[:25]]
        boundary_tie = len(ppr_candidates) > 25 and ppr_candidates[24][1] == ppr_candidates[25][1]
        seed_imports = neighbor_sets[primary_seed]
        expected_review = []
        for candidate in ppr_top25:
            shared = seed_imports & neighbor_sets[candidate]
            if not shared:
                continue
            union = seed_imports | neighbor_sets[candidate]
            expected_review.append(
                {
                    "candidate_id": candidate,
                    "shared": len(shared),
                    "minimum_symbol": min(symbol_names[symbol] for symbol in shared),
                    "similarity": len(shared) / len(union),
                }
            )
        expected_review.sort(key=lambda row: (-row["similarity"], row["candidate_id"]))
        expected_review = expected_review[:15]
        actual_review_ids = [int(row["candidate_id"]) for row in review_rows]
        expected_review_ids = [row["candidate_id"] for row in expected_review]
        candidate_pool_subset = set(actual_review_ids) <= set(ppr_top25)
        review_details = []
        review_valid = actual_review_ids == expected_review_ids
        for row, expected in zip(review_rows, expected_review):
            candidate = int(row["candidate_id"])
            metadata_row = metadata_by_id.get(candidate)
            shared_symbols = seed_imports & neighbor_sets[candidate]
            computed_count = len(shared_symbols)
            computed_minimum = min(symbol_names[symbol] for symbol in shared_symbols) if shared_symbols else None
            computed_jaccard = computed_count / len(seed_imports | neighbor_sets[candidate]) if shared_symbols else 0.0
            metadata_match = metadata_row is not None and row["sha256"] == metadata_row["sha256"] and row["family"] == metadata_row["family"]
            row_match = (
                computed_count == int(row["shared_rare_imports"])
                and computed_minimum == row["example_shared_import"]
                and int(row["retained_import_count"]) == retained_count[candidate]
                and math.isclose(computed_jaccard, float(row["similarity"]), rel_tol=0.0, abs_tol=1.0e-6)
                and metadata_match
            )
            review_valid &= row_match
            review_details.append(
                {
                    "candidate_id": candidate,
                    "shared_import_count_matches": computed_count == int(row["shared_rare_imports"]),
                    "minimum_symbol_matches": computed_minimum == row["example_shared_import"],
                    "jaccard_abs_error": abs(computed_jaccard - float(row["similarity"])),
                    "retained_import_count_matches": int(row["retained_import_count"]) == retained_count[candidate],
                    "metadata_matches": metadata_match,
                    "family": row["family"],
                    "family_blank": not bool(row["family"]),
                }
            )
        review_valid &= candidate_pool_subset and len(review_rows) == 15
        checks.add(
            "import_review_queue_join_and_ranking",
            review_valid,
            rows=len(review_rows),
            ppr_candidate_pool_size=len(ppr_top25),
            returned_ids_within_ppr_top25=candidate_pool_subset,
            ppr_top25_boundary_value=ppr_candidates[24][1] if len(ppr_candidates) >= 25 else None,
            ppr_top25_boundary_tie_with_next=boundary_tie,
            expected_order_exact=actual_review_ids == expected_review_ids,
            jaccard_tolerance=1.0e-6,
            row_checks=review_details,
        )
        observations["import_review_queue"] = {
            "query_id": json.loads((RESULTS / "87_import_review_queue.json").read_text()).get("query_id"),
            "candidate_pool": "top 25 eligible sample vertices by undirected import personalized PageRank, value descending and id ascending for ties",
            "candidate_pool_ids": ppr_top25,
            "returned_rows": review_details,
            "all_families_blank": all(row["family_blank"] for row in review_details),
            "interpretation": "The output groups static import overlap for analyst review. It does not identify malware families or predict campaigns.",
        }

    attempts = {}
    for name in ["69_cugraph_hits", "69_cugraph_hits_1000"]:
        json_path = RESULTS / f"{name}.json"
        record = json.loads(json_path.read_text()) if json_path.exists() else {}
        stats_path = RESULTS / f"{name}.stats.txt.gz"
        stats_headers = {}
        runtime_failure = None
        structured_error = {}
        if stats_path.exists():
            with gzip.open(stats_path, "rt") as stats_file:
                stats_lines = stats_file.readlines()
                for line in stats_lines:
                    if not line.startswith("# ") or "=" not in line:
                        continue
                    key, value = line[2:].rstrip("\n").split("=", 1)
                    stats_headers[key] = value
                if "[execution]\n" in stats_lines:
                    section_index = stats_lines.index("[execution]\n")
                    header_index = section_index + 1
                    while header_index < len(stats_lines) and not stats_lines[header_index].startswith("query_id\t"):
                        header_index += 1
                    if header_index + 1 < len(stats_lines):
                        headers = stats_lines[header_index].rstrip("\n").split("\t")
                        values = stats_lines[header_index + 1].rstrip("\n").split("\t")
                        execution_record = dict(zip(headers, values))
                        try:
                            runtime_failure = json.loads(execution_record.get("runtime_failure", "null"))
                        except json.JSONDecodeError:
                            runtime_failure = execution_record.get("runtime_failure")
                if "[structured_error]\n" in stats_lines:
                    section_index = stats_lines.index("[structured_error]\n")
                    for line in stats_lines[section_index + 1 :]:
                        if line.startswith("["):
                            break
                        if "=" not in line:
                            continue
                        key, value = line.rstrip("\n").split("=", 1)
                        if key == "diagnostic":
                            value = value.split(" Obtained ", 1)[0]
                        structured_error[key] = value
        attempts[name] = {
            "status": record.get("status"),
            "error": record.get("error"),
            "elapsed_s": record.get("elapsed_s"),
            "query_id": record.get("query_id"),
            "native_stats": {
                key: stats_headers.get(key)
                for key in [
                    "attempt_cleanup_disposition",
                    "attempt_cleanup_failure_count",
                    "attempt_capacity_disposition",
                    "attempt_health_transition",
                    "attempt_native_fault_evidence",
                ]
                if key in stats_headers
            },
            "runtime_failure": runtime_failure,
            "structured_error": structured_error,
        }
    observations["hits_attempts"] = attempts

    report = {
        "status": "error" if checks.failures else "ok",
        "runtime": {
            "networkx": nx.__version__,
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "data_root": str(DATA.relative_to(ROOT)),
        },
        "checks": checks.values,
        "failure_count": len(checks.failures),
        "failures": checks.failures,
        "observations": observations,
    }
    (RESULTS / "graph_check.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> int:
    report = run()
    print(json.dumps({"status": report["status"], "failure_count": report["failure_count"], "failures": report["failures"]}, indent=2))
    return 1 if report["failure_count"] else 0


if __name__ == "__main__":
    sys.exit(main())
