#!/usr/bin/env python3
"""Compare captured MerRec HITS rows with Float32 and converged Float64 CPU runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import struct
from typing import Any

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "target/merrec-exploration/results"
REPORT = ROOT / "docs/research/merrec"
NODES = RESULTS / "prepare_nodes.parquet"
EDGES = RESULTS / "prepare_edges.parquet"
FLOAT64_EPSILON = 1e-14
FLOAT64_MAX_ITERATIONS = 100_000
FLOAT32 = struct.Struct("=f")


def float32(value: float) -> float:
    return FLOAT32.unpack(FLOAT32.pack(float(value)))[0]


def query_settings(path: Path) -> dict[str, Any]:
    sql = path.read_text()
    settings: dict[str, Any] = {}
    for name, conversion in (
        ("epsilon", float),
        ("max_iterations", int),
    ):
        match = re.search(rf"\b{name}\s*=>\s*([0-9.eE+-]+)", sql)
        if match is None:
            raise ValueError(f"{path} has no {name} argument")
        settings[name] = conversion(match.group(1))
    directed = re.search(r"\bdirected\s*=>\s*(true|false)", sql, re.IGNORECASE)
    normalize = re.search(r"\bnormalize\s*=>\s*(true|false)", sql, re.IGNORECASE)
    if directed is None or normalize is None:
        raise ValueError(f"{path} is missing directed or normalize settings")
    settings["directed"] = directed.group(1).lower() == "true"
    settings["normalize"] = normalize.group(1).lower() == "true"
    if not re.search(
        r"edges\s*=>\s*\(\s*SELECT\s+src\s*,\s*dst\s+FROM\s+mr_edges\s*\)",
        sql,
        re.IGNORECASE | re.DOTALL,
    ):
        raise ValueError(f"{path} must pass only src and dst from mr_edges")
    settings["edge_projection"] = ["src", "dst"]
    return settings


def sequential_sum(values: list[float], scalar_type: Any) -> float:
    total = scalar_type(0.0)
    for value in values:
        total = scalar_type(total + value)
    return total


def accumulate_edges(
    values: list[float],
    targets: list[int],
    sources: list[int],
    count: int,
    scalar_type: Any,
) -> list[float]:
    output = [scalar_type(0.0) for _ in range(count)]
    for target, source in zip(targets, sources, strict=True):
        output[target] = scalar_type(output[target] + values[source])
    return output


def iterate_hits(
    source: list[int],
    destination: list[int],
    vertex_count: int,
    *,
    arithmetic: str,
    epsilon: float,
    max_iterations: int,
    normalize_output: bool = True,
) -> dict[str, Any]:
    scalar_type = float32 if arithmetic == "float32" else float
    typed_epsilon = scalar_type(epsilon)
    tolerance = scalar_type(scalar_type(vertex_count) * typed_epsilon)
    initial_value = scalar_type(scalar_type(1.0) / scalar_type(vertex_count))
    hubs = [initial_value for _ in range(vertex_count)]
    authorities = [scalar_type(0.0) for _ in range(vertex_count)]
    stopping_difference = float("inf")

    for iteration in range(1, max_iterations + 1):
        authorities = accumulate_edges(hubs, destination, source, vertex_count, scalar_type)
        current_hubs = accumulate_edges(authorities, source, destination, vertex_count, scalar_type)

        hub_max = max(current_hubs)
        authority_max = max(authorities)
        if hub_max <= 0 or authority_max <= 0:
            raise ValueError("HITS max normalization requires positive values")
        current_hubs = [scalar_type(value / hub_max) for value in current_hubs]
        authorities = [scalar_type(value / authority_max) for value in authorities]

        differences = [
            scalar_type(abs(current - previous))
            for current, previous in zip(current_hubs, hubs, strict=True)
        ]
        stopping_difference = sequential_sum(differences, scalar_type)
        hubs = current_hubs
        if stopping_difference < tolerance:
            break
    else:
        raise RuntimeError(
            f"HITS did not converge within {max_iterations} iterations "
            f"(last L1 hub change {float(stopping_difference)})"
        )

    if normalize_output:
        hub_sum = sequential_sum(hubs, scalar_type)
        authority_sum = sequential_sum(authorities, scalar_type)
        if hub_sum <= 0 or authority_sum <= 0:
            raise ValueError("HITS sum normalization requires positive values")
        hubs = [scalar_type(value / hub_sum) for value in hubs]
        authorities = [scalar_type(value / authority_sum) for value in authorities]

    return {
        "hubs": hubs,
        "authorities": authorities,
        "iterations": iteration,
        "stopping_difference": float(stopping_difference),
        "tolerance": float(tolerance),
        "epsilon": float(typed_epsilon),
        "max_iterations": max_iterations,
        "normalized_output": normalize_output,
        "arithmetic": arithmetic,
        "edge_accumulation_order": "prepare_edges row order, sequential Float32/Float64 additions",
        "stopping_reduction_order": "vertex id order, sequential Float32/Float64 additions",
    }


def rank_vertices(
    hubs: list[float],
    authorities: list[float],
    vertex_ids: list[int],
    product_ids: list[str],
) -> tuple[list[int], dict[int, int]]:
    indices = sorted(
        range(len(vertex_ids)),
        key=lambda index: (
            -float(authorities[index]),
            -float(hubs[index]),
            product_ids[index],
            vertex_ids[index],
        ),
    )
    ordered_ids = [vertex_ids[index] for index in indices]
    return ordered_ids, {vertex_id: rank for rank, vertex_id in enumerate(ordered_ids, 1)}


def top_rows(
    ordered_ids: list[int],
    id_to_index: dict[int, int],
    product_ids_by_vertex: dict[int, str],
    hubs: list[float],
    authorities: list[float],
    limit: int = 15,
) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "vertex": vertex_id,
            "product_id": product_ids_by_vertex[vertex_id],
            "hub_score": float(hubs[id_to_index[vertex_id]]),
            "authority_score": float(authorities[id_to_index[vertex_id]]),
        }
        for rank, vertex_id in enumerate(ordered_ids[:limit], 1)
    ]


def error_metrics(captured: list[float], reference: list[float]) -> dict[str, float]:
    errors = [left - right for left, right in zip(captured, reference, strict=True)]
    absolute = [abs(value) for value in errors]
    return {
        "max_absolute_error": max(absolute, default=0.0),
        "mean_absolute_error": sum(absolute) / len(absolute) if absolute else 0.0,
        "rmse": (sum(value * value for value in errors) / len(errors)) ** 0.5 if errors else 0.0,
    }


def model_summary(
    model: dict[str, Any],
    vertex_ids: list[int],
    product_ids: list[str],
    id_to_index: dict[int, int],
    product_ids_by_vertex: dict[int, str],
    float64_reference: dict[str, Any],
) -> dict[str, Any]:
    ordered_ids, ranks = rank_vertices(
        model["hubs"], model["authorities"], vertex_ids, product_ids
    )
    reference_ids = float64_reference["ordered_ids"]
    reference_rank = float64_reference["ranks"]
    return {
        key: value
        for key, value in model.items()
        if key not in ("hubs", "authorities", "ordered_ids", "ranks")
    } | {
        "rank_order_product_ids_top15": [
            product_ids_by_vertex[vertex_id] for vertex_id in ordered_ids[:15]
        ],
        "top15": top_rows(
            ordered_ids,
            id_to_index,
            product_ids_by_vertex,
            model["hubs"],
            model["authorities"],
        ),
        "top15_order_matches_float64": ordered_ids[:15] == reference_ids[:15],
        "top15_overlap_with_float64": len(set(ordered_ids[:15]) & set(reference_ids[:15])),
        "all_vertices_max_absolute_score_error_vs_float64": {
            "hub": max(
                abs(left - right)
                for left, right in zip(model["hubs"], float64_reference["hubs"], strict=True)
            ),
            "authority": max(
                abs(left - right)
                for left, right in zip(
                    model["authorities"], float64_reference["authorities"], strict=True
                )
            ),
        },
        "max_top15_rank_displacement_vs_float64": max(
            abs(ranks[vertex_id] - reference_rank[vertex_id]) for vertex_id in ordered_ids[:15]
        ),
    }


def capture_comparison(
    name: str,
    results_path: Path,
    result_json_path: Path,
    settings: dict[str, Any],
    cpu_float32: dict[str, Any],
    float64_reference: dict[str, Any],
    vertex_by_product_id: dict[str, int],
    id_to_index: dict[int, int],
    product_ids_by_vertex: dict[int, str],
    float32_rank: dict[int, int],
    float64_rank: dict[int, int],
) -> dict[str, Any]:
    if not results_path.is_file():
        status = "absent"
        error = None
        if result_json_path.is_file():
            saved_record = json.loads(result_json_path.read_text())
            if saved_record.get("status") == "error":
                status = "captured_error"
                error = saved_record.get("error")
        return {
            "status": status,
            "result_path": str(results_path.relative_to(ROOT)),
            "record_path": str(result_json_path.relative_to(ROOT)),
            "error": error,
            "query_settings": settings,
        }

    table = pq.read_table(results_path)
    rows = table.to_pylist()
    captured_ids: list[int] = []
    per_row: list[dict[str, Any]] = []
    captured_hubs: list[float] = []
    captured_authorities: list[float] = []
    f32_hubs: list[float] = []
    f32_authorities: list[float] = []
    f64_hubs: list[float] = []
    f64_authorities: list[float] = []

    for rank, row in enumerate(rows, 1):
        product_id = row["product_id"]
        vertex_id = vertex_by_product_id.get(product_id)
        if vertex_id is None:
            raise ValueError(f"Captured product_id {product_id!r} is absent from the node snapshot")
        captured_ids.append(vertex_id)
        vertex_index = id_to_index[vertex_id]
        captured_hub = float(row["hub_score"])
        captured_authority = float(row["authority_score"])
        model_hub = float(cpu_float32["hubs"][vertex_index])
        model_authority = float(cpu_float32["authorities"][vertex_index])
        ref_hub = float(float64_reference["hubs"][vertex_index])
        ref_authority = float(float64_reference["authorities"][vertex_index])
        captured_hubs.append(captured_hub)
        captured_authorities.append(captured_authority)
        f32_hubs.append(model_hub)
        f32_authorities.append(model_authority)
        f64_hubs.append(ref_hub)
        f64_authorities.append(ref_authority)
        per_row.append(
            {
                "captured_rank": rank,
                "vertex": vertex_id,
                "product_id": product_id,
                "hub_score": captured_hub,
                "authority_score": captured_authority,
                "cpu_float32_rank": float32_rank[vertex_id],
                "cpu_float32_hub_score": model_hub,
                "cpu_float32_authority_score": model_authority,
                "float64_rank": float64_rank[vertex_id],
                "float64_hub_score": ref_hub,
                "float64_authority_score": ref_authority,
            }
        )

    capture_order = captured_ids
    f32_order = rank_vertices(
        cpu_float32["hubs"],
        cpu_float32["authorities"],
        sorted(id_to_index),
        [product_ids_by_vertex[vertex_id] for vertex_id in sorted(id_to_index)],
    )[0][: len(captured_ids)]
    f64_order = float64_reference["ordered_ids"][: len(captured_ids)]
    return {
        "status": "captured",
        "result_path": str(results_path.relative_to(ROOT)),
        "record_path": str(result_json_path.relative_to(ROOT)),
        "query_settings": settings,
        "rows": len(rows),
        "captured_rank_order_product_ids": [row["product_id"] for row in rows],
        "rank_order_matches_cpu_float32": capture_order == f32_order,
        "rank_order_matches_float64": capture_order == f64_order,
        "top15_overlap_with_cpu_float32": len(set(capture_order) & set(f32_order)),
        "top15_overlap_with_float64": len(set(capture_order) & set(f64_order)),
        "score_errors_vs_cpu_float32": {
            "hub": error_metrics(captured_hubs, f32_hubs),
            "authority": error_metrics(captured_authorities, f32_authorities),
        },
        "score_errors_vs_float64": {
            "hub": error_metrics(captured_hubs, f64_hubs),
            "authority": error_metrics(captured_authorities, f64_authorities),
        },
        "rows_detail": per_row,
    }


def main() -> int:
    nodes = pq.read_table(NODES).to_pylist()
    edges = pq.read_table(EDGES).to_pylist()
    nodes.sort(key=lambda row: int(row["id"]))
    vertex_ids = [int(row["id"]) for row in nodes]
    product_ids = [str(row["product_id"]) for row in nodes]
    vertex_by_product_id = {
        product_id: vertex_id
        for vertex_id, product_id in zip(vertex_ids, product_ids, strict=True)
    }
    product_ids_by_vertex = dict(zip(vertex_ids, product_ids, strict=True))
    if len(vertex_by_product_id) != len(nodes):
        raise ValueError("Node snapshot product_id values must be unique")
    id_to_index = {vertex_id: index for index, vertex_id in enumerate(vertex_ids)}

    edge_pairs: list[tuple[int, int]] = []
    for row in edges:
        src = int(row["src"])
        dst = int(row["dst"])
        if src not in id_to_index or dst not in id_to_index:
            raise ValueError(f"Edge endpoint ({src}, {dst}) is absent from the node snapshot")
        edge_pairs.append((id_to_index[src], id_to_index[dst]))
    if len(set(edge_pairs)) != len(edge_pairs):
        raise ValueError("The saved directed edge snapshot contains duplicate src/dst pairs")
    if any(source == destination for source, destination in edge_pairs):
        raise ValueError("The saved directed edge snapshot contains a self-loop")
    source = [pair[0] for pair in edge_pairs]
    destination = [pair[1] for pair in edge_pairs]
    vertex_count = len(nodes)

    default_sql = REPORT / "sql/probes/cugraph_hits.sql"
    strict_sql = REPORT / "sql/probes/cugraph_hits_strict.sql"
    default_settings = query_settings(default_sql)
    strict_settings = query_settings(strict_sql)
    float64_model = iterate_hits(
        source,
        destination,
        vertex_count,
        arithmetic="float64",
        epsilon=FLOAT64_EPSILON / vertex_count,
        max_iterations=FLOAT64_MAX_ITERATIONS,
    )
    float64_ordered_ids, float64_ranks = rank_vertices(
        float64_model["hubs"], float64_model["authorities"], vertex_ids, product_ids
    )
    float64_reference = float64_model | {
        "ordered_ids": float64_ordered_ids,
        "ranks": float64_ranks,
    }

    cpu_models: dict[str, dict[str, Any]] = {}
    cpu_summaries: dict[str, dict[str, Any]] = {}
    for name, settings in (("cugraph_hits", default_settings), ("cugraph_hits_strict", strict_settings)):
        model = iterate_hits(
            source,
            destination,
            vertex_count,
            arithmetic="float32",
            epsilon=settings["epsilon"],
            max_iterations=settings["max_iterations"],
            normalize_output=settings["normalize"],
        )
        ordered_ids, ranks = rank_vertices(model["hubs"], model["authorities"], vertex_ids, product_ids)
        model["ordered_ids"] = ordered_ids
        model["ranks"] = ranks
        cpu_models[name] = model
        cpu_summaries[name] = model_summary(
            model,
            vertex_ids,
            product_ids,
            id_to_index,
            product_ids_by_vertex,
            float64_reference,
        )

    graph_validation_path = REPORT / "graph_validation.json"
    graph_validation = json.loads(graph_validation_path.read_text())
    graph_contract = graph_validation["graph_sql_contract"]["expected_snapshot_rows"]
    graph_metadata = {
        "nodes_path": str(NODES.relative_to(ROOT)),
        "edges_path": str(EDGES.relative_to(ROOT)),
        "graph_validation_path": str(graph_validation_path.relative_to(ROOT)),
        "graph_sql_sha256": graph_validation["source"]["graph_sql_sha256"],
        "vertices": vertex_count,
        "directed_edges": len(edges),
        "vertex_id_min": min(vertex_ids),
        "vertex_id_max": max(vertex_ids),
        "edge_pairs_unique": True,
        "self_loops": 0,
        "stored_weight_column_present": "weight" in pq.read_schema(EDGES).names,
        "query_edge_projection": ["src", "dst"],
        "query_passes_edge_weights": False,
        "directed": True,
        "matches_saved_graph_validation_counts": {
            "nodes": vertex_count == graph_contract["nodes"],
            "edges": len(edges) == graph_contract["edges"],
        },
    }

    float64_summary = {
        key: value
        for key, value in float64_model.items()
        if key not in ("hubs", "authorities", "ordered_ids", "ranks")
    } | {
        "convergence_criterion": f"L1 change in max-normalized hubs < {FLOAT64_EPSILON}",
        "rank_order_product_ids_top15": [
            product_ids_by_vertex[vertex_id] for vertex_id in float64_ordered_ids[:15]
        ],
        "top15": top_rows(
            float64_ordered_ids,
            id_to_index,
            product_ids_by_vertex,
            float64_model["hubs"],
            float64_model["authorities"],
        ),
    }

    captures = {}
    for name, sql_path, settings in (
        ("cugraph_hits", default_sql, default_settings),
        ("cugraph_hits_strict", strict_sql, strict_settings),
    ):
        capture = capture_comparison(
            name,
            RESULTS / f"{name}.parquet",
            REPORT / "results" / f"{name}.json",
            settings,
            cpu_models[name],
            float64_reference,
            vertex_by_product_id,
            id_to_index,
            product_ids_by_vertex,
            cpu_models[name]["ranks"],
            float64_ranks,
        )
        capture["sql_path"] = str(sql_path.relative_to(ROOT))
        captures[name] = capture

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "CPU reproduction of saved unweighted HITS rows and native convergence semantics",
        "graph": graph_metadata,
        "native_source": {
            "path": "components/cugraph/cugraph/cpp/src/link_analysis/hits_impl.cuh",
            "lines": {
                "tolerance_is_vertex_count_times_epsilon": [65, 72],
                "uniform_initial_hubs": [99, 104],
                "authority_and_hub_edge_sums": [108, 135],
                "max_normalization": [137, 149],
                "l1_hub_stopping_check": [151, 172],
                "sum_normalization_at_output": [175, 177],
            },
            "behavior": [
                "Initialize hubs to 1/N.",
                "Compute authorities as incoming unweighted hub sums and hubs as outgoing authority sums.",
                "Max-normalize hubs and authorities each iteration.",
                "Stop when the L1 change in max-normalized hubs is below N * epsilon.",
                "Sum-normalize hubs and authorities at output when normalize is true.",
            ],
            "cpu_arithmetic_note": "CPU edge and stopping reductions use deterministic sequential accumulation in Float32 or Float64; native GPU reduction order can differ.",
        },
        "float64_reference": float64_summary,
        "cpu_float32_models": cpu_summaries,
        "captures": captures,
    }
    output = REPORT / "hits_validation.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(output.relative_to(ROOT)),
                "vertices": vertex_count,
                "directed_edges": len(edges),
                "float64_iterations": float64_model["iterations"],
                "captures": {name: value["status"] for name, value in captures.items()},
                "default_float32_iterations": cpu_models["cugraph_hits"]["iterations"],
                "strict_float32_iterations": cpu_models["cugraph_hits_strict"]["iterations"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
