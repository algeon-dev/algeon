#!/usr/bin/env python3
"""cuGraph oracle driver for the datafusion-nexus parity harness.

Produces deterministic JSON output for a single (dataset, algorithm,
params) triple by running RAPIDS Python cuGraph on a local CSV fixture
prepared from the sibling cuGraph checkout. The Rust parity harness reads
these JSON files from ``fixture/graph/cugraph/oracle/`` and compares the
native cuGraph execution path against them.

The script intentionally has a narrow interface so the Rust
``refresh_cugraph_oracle`` binary and the parity harness can both invoke
it. It exits with code ``77`` (the conventional "skip" code) when the
Python cuGraph stack is unavailable so the harness can degrade
gracefully on hosts without RAPIDS Python.

Output schema (stdout, one JSON document)::

    {
      "header": {
        "dataset": "karate",
        "dataset_sha256": "<hex>",
        "algorithm": "pagerank",
        "params": { ... },
        "seed": null,
        "versions": {
          "cugraph": "<ver>",
          "cudf": "<ver>",
          "pylibcugraph": "<ver>",
          "numpy": "<ver>"
        }
      },
      "result": { ... per-algorithm payload, sorted by vertex ... }
    }

The header is fully deterministic given identical inputs; the result
payload is sorted by vertex id so equality checks do not depend on
cuGraph's internal column ordering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_DATASET = 3
EXIT_DRIFT = 4
EXIT_UNAVAILABLE = 77


def _read_manifest(data_root: Path) -> Dict[str, Any]:
    manifest_path = data_root / "manifest.json"
    try:
        with manifest_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"cuGraph dataset manifest missing at {manifest_path}: {exc}"
        ) from exc


def _resolve_dataset(manifest: Dict[str, Any], dataset_id: str) -> Dict[str, Any]:
    for entry in manifest.get("datasets", []):
        if entry.get("id") == dataset_id:
            return entry
    raise SystemExit(
        f"cuGraph dataset id '{dataset_id}' not found in manifest "
        f"(known: {[e.get('id') for e in manifest.get('datasets', [])]})"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_dataset(data_root: Path, entry: Dict[str, Any]) -> Path:
    path = data_root / entry["path"]
    actual = _sha256(path)
    expected = entry["sha256"]
    if actual != expected:
        raise SystemExit(
            f"cuGraph dataset '{entry['id']}' SHA-256 mismatch: "
            f"expected {expected}, got {actual} at {path}"
        )
    return path


def _load_python_cugraph():
    try:
        import cudf  # noqa: F401
        import cugraph  # noqa: F401
        import pylibcugraph  # noqa: F401
        import numpy  # noqa: F401
    except ImportError as exc:
        # The shared RAPIDS venv documented in AGENTS.md is the expected
        # environment. When it is unavailable, exit with 77 so callers
        # can degrade gracefully.
        print(
            f"cuGraph Python stack unavailable: {exc}. "
            "Activate the shared ~/code/datafusion-nexus/.venv to refresh oracles.",
            file=sys.stderr,
        )
        sys.exit(EXIT_UNAVAILABLE)
    import cudf
    import cugraph
    import pylibcugraph
    import numpy

    return cudf, cugraph, pylibcugraph, numpy


def _versions(cudf_mod, cugraph_mod, pylibcugraph_mod, numpy_mod) -> Dict[str, str]:
    return {
        "cugraph": getattr(cugraph_mod, "__version__", "unknown"),
        "cudf": getattr(cudf_mod, "__version__", "unknown"),
        "pylibcugraph": getattr(pylibcugraph_mod, "__version__", "unknown"),
        "numpy": getattr(numpy_mod, "__version__", "unknown"),
    }


def _load_edges(cudf_mod, path: Path):
    """Load a cuGraph-style space-separated CSV.

    Each line is ``dst src weight`` (the cuGraph repository convention).
    The harness uses the same convention so the Rust path and the
    Python oracle see identical edge orientation.
    """
    return cudf_mod.read_csv(
        str(path),
        sep=" ",
        names=["dst", "src", "weight"],
        dtype=["int32", "int32", "float32"],
        header=None,
    )


def _build_graph(cudf_mod, cugraph_mod, edges, directed: bool, edge_attr: bool):
    graph = cugraph_mod.Graph(directed=directed)
    kwargs = {
        "source": "src",
        "destination": "dst",
        "renumber": True,
    }
    if edge_attr:
        kwargs["edge_attr"] = "weight"
    graph.from_cudf_edgelist(edges, **kwargs)
    return graph


def _allow_nonconvergence_dataframe(result):
    """Return the DataFrame from cuGraph's allow-nonconvergence result."""
    if isinstance(result, tuple):
        return result[0]
    return result


def _pagerank(cugraph_mod, graph, params: Dict[str, Any]) -> Dict[str, Any]:
    alpha = float(params.get("alpha", 0.85))
    epsilon = float(params.get("epsilon", 1.0e-5))
    max_iterations = int(params.get("max_iterations", 100))
    df = _allow_nonconvergence_dataframe(cugraph_mod.pagerank(
        graph,
        alpha=alpha,
        max_iter=max_iterations,
        tol=epsilon,
        fail_on_nonconvergence=False,
    )).sort_values("vertex").reset_index(drop=True)
    return {
        "vertex": [int(v) for v in df["vertex"].to_pandas().tolist()],
        "value": [float(v) for v in df["pagerank"].to_pandas().tolist()],
    }


def _personalized_pagerank(cugraph_mod, graph, params: Dict[str, Any]) -> Dict[str, Any]:
    import cudf as cudf_mod

    alpha = float(params.get("alpha", 0.85))
    epsilon = float(params.get("epsilon", 1.0e-5))
    max_iterations = int(params.get("max_iterations", 100))
    personalization_vertices = params.get("personalization_vertices", [0])
    personalization_values = params.get("personalization_values", [1.0])
    if len(personalization_vertices) != len(personalization_values):
        raise ValueError("personalization_vertices and personalization_values must have the same length")
    vertex_dtype = graph.edgelist.edgelist_df.dtypes.iloc[0]
    value_dtype = graph.edgelist.edgelist_df["weights"].dtype
    personalization = cudf_mod.DataFrame(
        {
            "vertex": cudf_mod.Series(
                [int(vertex) for vertex in personalization_vertices],
                dtype=vertex_dtype,
            ),
            "values": cudf_mod.Series(
                [float(value) for value in personalization_values],
                dtype=value_dtype,
            ),
        }
    )
    df = _allow_nonconvergence_dataframe(cugraph_mod.pagerank(
        graph,
        alpha=alpha,
        max_iter=max_iterations,
        tol=epsilon,
        personalization=personalization,
        fail_on_nonconvergence=False,
    )).sort_values("vertex").reset_index(drop=True)
    return {
        "vertex": [int(v) for v in df["vertex"].to_pandas().tolist()],
        "value": [float(v) for v in df["pagerank"].to_pandas().tolist()],
    }


def _bfs(cugraph_mod, graph, params: Dict[str, Any]) -> Dict[str, Any]:
    source_vertex = int(params["source_vertex"])
    depth_limit = params.get("depth_limit")
    kwargs: Dict[str, Any] = {"start": source_vertex}
    if depth_limit is not None and depth_limit != 0:
        kwargs["depth_limit"] = int(depth_limit)
    df = cugraph_mod.bfs(graph, **kwargs).sort_values("vertex").reset_index(drop=True)
    distances = df["distance"].to_pandas().tolist()
    predecessors_pd = df["predecessor"].to_pandas()
    predecessors = [
        None if (p is None or int(p) < 0) else int(p) for p in predecessors_pd.tolist()
    ]
    return {
        "vertex": [int(v) for v in df["vertex"].to_pandas().tolist()],
        "distance": [int(d) for d in distances],
        "predecessor": predecessors,
    }


def _louvain(cugraph_mod, graph, params: Dict[str, Any]) -> Dict[str, Any]:
    max_iter = int(params.get("max_level", 100))
    threshold = float(params.get("threshold", 1.0e-7))
    resolution = float(params.get("resolution", 1.0))
    df, modularity = cugraph_mod.louvain(
        graph, max_iter=max_iter, threshold=threshold, resolution=resolution
    )
    df = df.sort_values("vertex").reset_index(drop=True)
    return {
        "vertex": [int(v) for v in df["vertex"].to_pandas().tolist()],
        "partition": [int(v) for v in df["partition"].to_pandas().tolist()],
        "modularity": float(modularity),
    }


def _eigenvector_centrality(cugraph_mod, graph, params: Dict[str, Any]) -> Dict[str, Any]:
    """Run cuGraph eigenvector centrality with deterministic key ordering.

    Mirrors ``nexus_query_engine::GraphEigenvectorCentralityParams`` defaults
    (epsilon=1e-6, max_iterations=200) so the harness header round-trip
    matches the parity spec exactly.
    """
    tol = float(params.get("epsilon", 1.0e-6))
    max_iter = int(params.get("max_iterations", 200))
    df = cugraph_mod.eigenvector_centrality(graph, max_iter=max_iter, tol=tol)
    # cuGraph's eigenvector centrality returns a "eigenvector_centrality"
    # column in some versions and a vertex+value pair in others. Normalize
    # to a single "value" column so the JSON shape is stable across
    # RAPIDS versions.
    value_col_candidates = [c for c in df.columns if c != "vertex"]
    if not value_col_candidates:
        raise RuntimeError(
            "cugraph.eigenvector_centrality returned no centrality column",
        )
    value_col = value_col_candidates[0]
    df = df.rename(columns={value_col: "value"})
    df["vertex"] = df["vertex"].astype("int64")
    df["value"] = df["value"].astype("float64")
    df = df.sort_values("vertex").reset_index(drop=True)
    return {
        "vertex": [int(v) for v in df["vertex"].to_pandas().tolist()],
        "value": [float(v) for v in df["value"].to_pandas().tolist()],
    }


def _katz_centrality(cugraph_mod, graph, params: Dict[str, Any]) -> Dict[str, Any]:
    """Run cuGraph Katz centrality with deterministic key ordering.

    Mirrors ``nexus_query_engine::GraphKatzCentralityParams`` defaults
    (alpha=0.01, beta=1.0, epsilon=1e-6, max_iterations=1000) so the
    header round-trip matches the parity spec exactly.
    """
    alpha = float(params.get("alpha", 0.01))
    beta = float(params.get("beta", 1.0))
    tol = float(params.get("epsilon", 1.0e-6))
    max_iter = int(params.get("max_iterations", 1000))
    df = cugraph_mod.katz_centrality(graph, alpha=alpha, beta=beta, tol=tol, max_iter=max_iter)
    value_col_candidates = [c for c in df.columns if c != "vertex"]
    if not value_col_candidates:
        raise RuntimeError("cugraph.katz_centrality returned no centrality column")
    value_col = value_col_candidates[0]
    df = df.rename(columns={value_col: "value"})
    df["vertex"] = df["vertex"].astype("int64")
    df["value"] = df["value"].astype("float64")
    df = df.sort_values("vertex").reset_index(drop=True)
    return {
        "vertex": [int(v) for v in df["vertex"].to_pandas().tolist()],
        "value": [float(v) for v in df["value"].to_pandas().tolist()],
    }


def _hits(cugraph_mod, graph, params: Dict[str, Any]) -> Dict[str, Any]:
    """Run cuGraph HITS with deterministic key ordering.

    Mirrors ``nexus_query_engine::GraphHitsParams`` defaults
    (epsilon=1e-5, max_iterations=100, normalize=False) and normalizes
    the result column names (``hubs`` / ``authorities`` ->
    ``hub_score`` / ``authority_score``) so the JSON shape matches the
    stable adapter schema.
    """
    tol = float(params.get("epsilon", 1.0e-5))
    max_iter = int(params.get("max_iterations", 100))
    normalize = bool(params.get("normalize", False))
    df = cugraph_mod.hits(graph, tol=tol, max_iter=max_iter, normalized=normalize)
    rename_map: Dict[str, str] = {}
    if "hubs" in df.columns:
        rename_map["hubs"] = "hub_score"
    if "authorities" in df.columns:
        rename_map["authorities"] = "authority_score"
    if rename_map:
        df = df.rename(columns=rename_map)
    df["vertex"] = df["vertex"].astype("int64")
    df["hub_score"] = df["hub_score"].astype("float64")
    df["authority_score"] = df["authority_score"].astype("float64")
    df = df.sort_values("vertex").reset_index(drop=True)
    return {
        "vertex": [int(v) for v in df["vertex"].to_pandas().tolist()],
        "hub_score": [float(v) for v in df["hub_score"].to_pandas().tolist()],
        "authority_score": [float(v) for v in df["authority_score"].to_pandas().tolist()],
    }


def _betweenness_centrality(cugraph_mod, graph, params: Dict[str, Any]) -> Dict[str, Any]:
    seeds = params.get("seeds")
    k = None if seeds is None else [int(vertex) for vertex in seeds]
    normalized = bool(params.get("normalized", True))
    include_endpoints = bool(params.get("include_endpoints", False))
    df = cugraph_mod.betweenness_centrality(
        graph,
        k=k,
        normalized=normalized,
        endpoints=include_endpoints,
    )
    df = df.rename(columns={"betweenness_centrality": "value"})
    df["vertex"] = df["vertex"].astype("int64")
    df["value"] = df["value"].astype("float64")
    df = df.sort_values("vertex").reset_index(drop=True)
    return {
        "vertex": [int(v) for v in df["vertex"].to_pandas().tolist()],
        "value": [float(v) for v in df["value"].to_pandas().tolist()],
    }


def _edge_betweenness_centrality(cugraph_mod, graph, params: Dict[str, Any]) -> Dict[str, Any]:
    seeds = params.get("seeds")
    k = None if seeds is None else [int(vertex) for vertex in seeds]
    normalized = bool(params.get("normalized", True))
    df = cugraph_mod.edge_betweenness_centrality(
        graph,
        k=k,
        normalized=normalized,
    )
    df = df.rename(columns={"src": "source", "dst": "destination", "betweenness_centrality": "score"})
    df["source"] = df["source"].astype("int64")
    df["destination"] = df["destination"].astype("int64")
    df["score"] = df["score"].astype("float64")
    df = df.sort_values(["source", "destination"]).reset_index(drop=True)
    return {
        "source": [int(v) for v in df["source"].to_pandas().tolist()],
        "destination": [int(v) for v in df["destination"].to_pandas().tolist()],
        "score": [float(v) for v in df["score"].to_pandas().tolist()],
    }


def _sssp(cugraph_mod, graph, params: Dict[str, Any]) -> Dict[str, Any]:
    """Run cuGraph SSSP from a single source vertex.

    Returns vertex, distance (float64), and predecessor (int64) columns.
    Distance uses float64 matching the stable adapter schema.
    Predecessor is -1 for the source vertex (no predecessor) and for
    unreachable vertices.
    """
    source = int(params.get("source_vertex", 0))
    cutoff = params.get("cutoff", None)
    df = cugraph_mod.sssp(graph, source=source, cutoff=cutoff)
    df["vertex"] = df["vertex"].astype("int64")
    df["distance"] = df["distance"].astype("float64")
    df["predecessor"] = df["predecessor"].astype("int64")
    df = df.sort_values("vertex").reset_index(drop=True)
    preds = df["predecessor"].to_pandas().tolist()
    return {
        "vertex": [int(v) for v in df["vertex"].to_pandas().tolist()],
        "distance": [float(d) for d in df["distance"].to_pandas().tolist()],
        "predecessor": [None if p < 0 else int(p) for p in preds],
    }


def _degrees_all(cugraph_mod, graph, params: Dict[str, Any]) -> Dict[str, Any]:
    """Return per-vertex in-degree and out-degree.

    Uses the graph object's `degrees()` method which returns
    ``in_degree``, ``out_degree``, and ``vertex`` columns directly.
    Module-level ``cugraph.degree`` does not exist in cuGraph 26.x.
    """
    df = graph.degrees()
    df["vertex"] = df["vertex"].astype("int64")
    df["in_degree"] = df["in_degree"].astype("int64")
    df["out_degree"] = df["out_degree"].astype("int64")
    df = df.sort_values("vertex").reset_index(drop=True)
    return {
        "vertex": [int(v) for v in df["vertex"].to_pandas().tolist()],
        "in_degree": [int(d) for d in df["in_degree"].to_pandas().tolist()],
        "out_degree": [int(d) for d in df["out_degree"].to_pandas().tolist()],
    }


def _in_degrees_all(cugraph_mod, graph, params: Dict[str, Any]) -> Dict[str, Any]:
    """Return per-vertex in-degree only.

    Uses ``graph.in_degree()`` which returns a ``degree`` column;
    rename to ``in_degree`` to match the stable adapter schema.
    """
    df = graph.in_degree()
    df["vertex"] = df["vertex"].astype("int64")
    df = df.rename(columns={"degree": "in_degree"})
    df["in_degree"] = df["in_degree"].astype("int64")
    df = df.sort_values("vertex").reset_index(drop=True)
    return {
        "vertex": [int(v) for v in df["vertex"].to_pandas().tolist()],
        "in_degree": [int(d) for d in df["in_degree"].to_pandas().tolist()],
    }


def _out_degrees_all(cugraph_mod, graph, params: Dict[str, Any]) -> Dict[str, Any]:
    """Return per-vertex out-degree only.

    Uses ``graph.out_degree()`` which returns a ``degree`` column;
    rename to ``out_degree`` to match the stable adapter schema.
    """
    df = graph.out_degree()
    df["vertex"] = df["vertex"].astype("int64")
    df = df.rename(columns={"degree": "out_degree"})
    df["out_degree"] = df["out_degree"].astype("int64")
    df = df.sort_values("vertex").reset_index(drop=True)
    return {
        "vertex": [int(v) for v in df["vertex"].to_pandas().tolist()],
        "out_degree": [int(d) for d in df["out_degree"].to_pandas().tolist()],
    }


def _weakly_connected_components(cugraph_mod, graph, params: Dict[str, Any]) -> Dict[str, Any]:
    """Return weakly connected component labels per vertex.

    WCC requires an undirected (symmetric) graph. The oracle builds the
    graph with directed=False so reverse edges are included.
    Labels are arbitrary integers; parity comparison uses
    partition-equivalence, not label equality.
    """
    df = cugraph_mod.weakly_connected_components(graph)
    df["vertex"] = df["vertex"].astype("int64")
    df["labels"] = df["labels"].astype("int64")
    df = df.sort_values("vertex").reset_index(drop=True)
    return {
        "vertex": [int(v) for v in df["vertex"].to_pandas().tolist()],
        "label": [int(l) for l in df["labels"].to_pandas().tolist()],
    }


def _strongly_connected_components(cugraph_mod, graph, params: Dict[str, Any]) -> Dict[str, Any]:
    """Return strongly connected component labels per vertex."""
    df = cugraph_mod.strongly_connected_components(graph)
    df["vertex"] = df["vertex"].astype("int64")
    df["labels"] = df["labels"].astype("int64")
    df = df.sort_values("vertex").reset_index(drop=True)
    return {
        "vertex": [int(v) for v in df["vertex"].to_pandas().tolist()],
        "label": [int(l) for l in df["labels"].to_pandas().tolist()],
    }


def _triangle_count_all(cugraph_mod, graph, params: Dict[str, Any]) -> Dict[str, Any]:
    """Return per-vertex triangle counts."""
    df = cugraph_mod.triangle_count(graph)
    # cuGraph may name the count column "counts".
    count_col = "counts" if "counts" in df.columns else "triangle_count"
    df["vertex"] = df["vertex"].astype("int64")
    df[count_col] = df[count_col].astype("int64")
    df = df.sort_values("vertex").reset_index(drop=True)
    return {
        "vertex": [int(v) for v in df["vertex"].to_pandas().tolist()],
        "triangle_count": [int(c) for c in df[count_col].to_pandas().tolist()],
    }


def _core_number(cugraph_mod, graph, params: Dict[str, Any]) -> Dict[str, Any]:
    """Return core number for every vertex."""
    df = cugraph_mod.core_number(graph)
    df["vertex"] = df["vertex"].astype("int64")
    df["core_number"] = df["core_number"].astype("int64")
    df = df.sort_values("vertex").reset_index(drop=True)
    return {
        "vertex": [int(v) for v in df["vertex"].to_pandas().tolist()],
        "core_number": [int(c) for c in df["core_number"].to_pandas().tolist()],
    }


def _k_core(cugraph_mod, graph, params: Dict[str, Any]) -> Dict[str, Any]:
    """Return the k-core subgraph as an edge list (src, dst)."""
    k = int(params.get("k", 2))
    subgraph = cugraph_mod.k_core(graph, k=k)
    df = subgraph.view_edge_list()
    df = df.rename(columns={"src": "src", "dst": "dst"})
    # Normalize column names — cuGraph may use 0/1 integer column names.
    cols = list(df.columns)
    if len(cols) >= 2 and cols[0] != "src":
        df.columns = ["src", "dst"] + list(df.columns[2:])
    df["src"] = df["src"].astype("int64")
    df["dst"] = df["dst"].astype("int64")
    df = df.sort_values(["src", "dst"]).reset_index(drop=True)
    return {
        "src": [int(v) for v in df["src"].to_pandas().tolist()],
        "dst": [int(v) for v in df["dst"].to_pandas().tolist()],
    }


def _minimum_spanning_tree(cugraph_mod, graph, params: Dict[str, Any]) -> Dict[str, Any]:
    """Return the minimum spanning tree edge list (src, dst).

    MST is only meaningful on weighted graphs.  On an unweighted graph all
    edges have the same weight, so the MST is non-unique and any tie-breaking
    result would be oracle-fragile (libcugraph and Python cuGraph may break
    ties differently).  Raise an explicit error so refresh_cugraph_oracle
    reports python_error for unweighted fixtures instead of silently pinning
    an arbitrary spanning tree.
    """
    # Reject uniform-weight graphs: MST is non-unique when all edge weights
    # are equal (e.g. karate with all weights=1.0), so any tie-broken result
    # would be an oracle-fragile arbitrary spanning tree.
    edge_df = graph.view_edge_list().to_pandas()
    weight_col = next((c for c in ("weight", "weights") if c in edge_df.columns), None)
    if weight_col is not None and edge_df[weight_col].nunique() <= 1:
        raise ValueError(
            "MST oracle requires distinct edge weights for a unique result; "
            "this dataset has uniform weights so the MST is non-unique. "
            "Provide a dataset with varying edge weights instead."
        )
    subgraph = cugraph_mod.minimum_spanning_tree(graph)
    df = subgraph.view_edge_list()
    cols = list(df.columns)
    if len(cols) >= 2 and cols[0] != "src":
        df.columns = ["src", "dst"] + list(df.columns[2:])
    df["src"] = df["src"].astype("int64")
    df["dst"] = df["dst"].astype("int64")
    df = df.sort_values(["src", "dst"]).reset_index(drop=True)
    return {
        "src": [int(v) for v in df["src"].to_pandas().tolist()],
        "dst": [int(v) for v in df["dst"].to_pandas().tolist()],
    }


def _leiden(cugraph_mod, graph, params: Dict[str, Any]) -> Dict[str, Any]:
    # Python cuGraph uses `max_iter` (not `max_level`) and `random_state` (not `seed`).
    max_iter = int(params.get("max_level", 100))
    resolution = float(params.get("resolution", 1.0))
    theta = float(params.get("theta", 1.0))
    random_state = int(params.get("seed", 0))
    df, modularity = cugraph_mod.leiden(
        graph, max_iter=max_iter, resolution=resolution, theta=theta, random_state=random_state
    )
    df = df.sort_values("vertex").reset_index(drop=True)
    return {
        "vertex": [int(v) for v in df["vertex"].to_pandas().tolist()],
        "partition": [int(v) for v in df["partition"].to_pandas().tolist()],
    }


def _ecg(cugraph_mod, graph, params: Dict[str, Any]) -> Dict[str, Any]:
    # Python cuGraph uses `random_state` (not `seed`).
    min_weight = float(params.get("min_weight", 0.001))
    ensemble_size = int(params.get("ensemble_size", 10))
    max_level = int(params.get("max_level", 100))
    threshold = float(params.get("threshold", 1.0e-7))
    resolution = float(params.get("resolution", 1.0))
    random_state = int(params.get("seed", 0))
    df, modularity = cugraph_mod.ecg(
        graph,
        min_weight=min_weight,
        ensemble_size=ensemble_size,
        max_level=max_level,
        threshold=threshold,
        resolution=resolution,
        random_state=random_state,
    )
    df = df.sort_values("vertex").reset_index(drop=True)
    return {
        "vertex": [int(v) for v in df["vertex"].to_pandas().tolist()],
        "partition": [int(v) for v in df["partition"].to_pandas().tolist()],
    }


def _spectral_modularity_maximization(cugraph_mod, graph, params: Dict[str, Any]) -> Dict[str, Any]:
    # Python cuGraph uses `num_clusters` (positional), `num_eigen_vects`, `evs_max_iter`,
    # `kmean_max_iter`, and `random_state` (not `seed`). Requires int32 vertices.
    n_clusters = int(params.get("n_clusters", 2))
    n_eigenvectors = int(params.get("n_eigenvectors", 2))
    evs_tolerance = float(params.get("evs_tolerance", 0.01))
    evs_max_iterations = int(params.get("evs_max_iterations", 500))
    k_means_tolerance = float(params.get("k_means_tolerance", 0.001))
    k_means_max_iterations = int(params.get("k_means_max_iterations", 100))
    random_state = int(params.get("seed", 0))
    df = cugraph_mod.spectralModularityMaximizationClustering(
        graph,
        n_clusters,
        num_eigen_vects=n_eigenvectors,
        evs_tolerance=evs_tolerance,
        evs_max_iter=evs_max_iterations,
        kmean_tolerance=k_means_tolerance,
        kmean_max_iter=k_means_max_iterations,
        random_state=random_state,
    )
    df = df.sort_values("vertex").reset_index(drop=True)
    # Python cuGraph returns "cluster" column; stable schema uses "partition".
    if "cluster" in df.columns and "partition" not in df.columns:
        df = df.rename(columns={"cluster": "partition"})
    return {
        "vertex": [int(v) for v in df["vertex"].to_pandas().tolist()],
        "partition": [int(v) for v in df["partition"].to_pandas().tolist()],
    }


def _force_atlas2(cugraph_mod, graph, params: Dict[str, Any]) -> Dict[str, Any]:
    # Pass every option explicitly: Python cuGraph defaults differ from
    # the Rust `cugraph` crate for a few ForceAtlas2 knobs, while the adapter mirrors
    # libcugraph C API defaults.
    df = cugraph_mod.force_atlas2(
        graph,
        max_iter=int(params.get("max_iter", 500)),
        outbound_attraction_distribution=bool(
            params.get("outbound_attraction_distribution", False)
        ),
        lin_log_mode=bool(params.get("lin_log_mode", False)),
        prevent_overlapping=bool(params.get("prevent_overlapping", False)),
        overlap_scaling_ratio=float(params.get("overlap_scaling_ratio", 2.0)),
        edge_weight_influence=float(params.get("edge_weight_influence", 1.0)),
        jitter_tolerance=float(params.get("jitter_tolerance", 1.0)),
        barnes_hut_optimize=bool(params.get("barnes_hut_optimize", False)),
        barnes_hut_theta=float(params.get("barnes_hut_theta", 0.5)),
        scaling_ratio=float(params.get("scaling_ratio", 2.0)),
        strong_gravity_mode=bool(params.get("strong_gravity_mode", False)),
        gravity=float(params.get("gravity", 1.0)),
        verbose=bool(params.get("verbose", False)),
        random_state=int(params.get("seed", 0)),
    )
    df = df.sort_values("vertex").reset_index(drop=True)
    return {
        "vertex": [int(v) for v in df["vertex"].to_pandas().tolist()],
        "x": [float(v) for v in df["x"].to_pandas().tolist()],
        "y": [float(v) for v in df["y"].to_pandas().tolist()],
    }


ALGORITHMS = {
    "pagerank": {"layout": "transposed", "directed": True, "run": _pagerank},
    "bfs": {"layout": "forward", "directed": True, "run": _bfs},
    "louvain": {"layout": "forward", "directed": False, "run": _louvain},
    # Eigenvector centrality runs on the transposed layout in cuGraph
    # (same as PageRank); the Python `cugraph.Graph(directed=True)`
    # construction is what matches the native engine's TransposedView path.
    "eigenvector_centrality": {
        "layout": "transposed",
        "directed": True,
        "run": _eigenvector_centrality,
    },
    # Katz centrality also runs on the transposed layout. The parity row
    # uses the cuGraph-curated `ref/katz/karate.csv` as primary truth and
    # this oracle as a secondary cross-check on the full score table.
    "katz_centrality": {
        "layout": "transposed",
        "directed": True,
        "run": _katz_centrality,
    },
    # HITS runs on the transposed layout too. The result has both
    # hub and authority scores, renamed to `hub_score` /
    # `authority_score` to match the stable adapter schema.
    "hits": {
        "layout": "transposed",
        "directed": True,
        "run": _hits,
    },
    "betweenness_centrality": {
        "layout": "forward",
        "directed": True,
        "edge_attr": False,
        "run": _betweenness_centrality,
    },
    "edge_betweenness_centrality": {
        "layout": "forward",
        "directed": True,
        "edge_attr": False,
        "run": _edge_betweenness_centrality,
    },
    # SSSP: Forward layout, directed graph, requires a source_vertex param.
    "sssp": {
        "layout": "forward",
        "directed": True,
        "run": _sssp,
    },
    # Degree algorithms: Transposed layout matches the native engine adapter.
    "degrees_all": {
        "layout": "transposed",
        "directed": True,
        "run": _degrees_all,
    },
    "in_degrees_all": {
        "layout": "transposed",
        "directed": True,
        "run": _in_degrees_all,
    },
    # out_degrees is layout-agnostic; use Forward for consistency.
    "out_degrees_all": {
        "layout": "forward",
        "directed": True,
        "run": _out_degrees_all,
    },
    # WCC needs an undirected (symmetric) graph.
    "weakly_connected_components": {
        "layout": "forward",
        "directed": False,
        "run": _weakly_connected_components,
    },
    "strongly_connected_components": {
        "layout": "forward",
        "directed": True,
        "run": _strongly_connected_components,
    },
    # triangle_count, core_number, k_core require undirected graphs in
    # Python cuGraph.  The native adapter calls libcugraph directly and
    # accepts directed graphs; the oracle intentionally uses the undirected
    # version to match the documented algorithm semantics.
    "triangle_count_all": {
        "layout": "forward",
        "directed": False,
        "run": _triangle_count_all,
    },
    "core_number": {
        "layout": "forward",
        "directed": False,
        "run": _core_number,
    },
    "k_core": {
        "layout": "forward",
        "directed": False,
        "run": _k_core,
    },
    "minimum_spanning_tree": {
        "layout": "forward",
        "directed": False,
        "run": _minimum_spanning_tree,
    },
    "personalized_pagerank": {
        "layout": "transposed",
        "directed": True,
        "run": _personalized_pagerank,
    },
    # Leiden, ECG, and spectral modularity maximization require undirected graphs.
    "leiden": {
        "layout": "forward",
        "directed": False,
        "run": _leiden,
    },
    "ecg": {
        "layout": "forward",
        "directed": False,
        "run": _ecg,
    },
    "spectral_modularity_maximization": {
        "layout": "forward",
        "directed": False,
        "run": _spectral_modularity_maximization,
    },
    "force_atlas2": {
        "layout": "forward",
        "directed": False,
        "edge_attr": False,
        "run": _force_atlas2,
    },
}


def _run(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root).resolve()
    manifest = _read_manifest(data_root)
    entry = _resolve_dataset(manifest, args.dataset)
    dataset_path = _verify_dataset(data_root, entry)

    try:
        params = json.loads(args.params)
    except json.JSONDecodeError as exc:
        print(f"--params must be valid JSON: {exc}", file=sys.stderr)
        return EXIT_USAGE
    if not isinstance(params, dict):
        print("--params must be a JSON object", file=sys.stderr)
        return EXIT_USAGE

    spec = ALGORITHMS.get(args.algorithm)
    if spec is None:
        print(
            f"Unknown algorithm '{args.algorithm}' "
            f"(known: {sorted(ALGORITHMS)})",
            file=sys.stderr,
        )
        return EXIT_USAGE

    cudf_mod, cugraph_mod, pylibcugraph_mod, numpy_mod = _load_python_cugraph()
    edges = _load_edges(cudf_mod, dataset_path)
    graph = _build_graph(cudf_mod, cugraph_mod, edges, spec["directed"], spec.get("edge_attr", True))
    result = spec["run"](cugraph_mod, graph, params)

    document = {
        "header": {
            "dataset": entry["id"],
            "dataset_sha256": entry["sha256"],
            "algorithm": args.algorithm,
            "params": params,
            "seed": args.seed,
            "versions": _versions(
                cudf_mod, cugraph_mod, pylibcugraph_mod, numpy_mod
            ),
        },
        "result": result,
    }

    output = json.dumps(document, indent=2, sort_keys=True)
    if args.output is None:
        print(output)
    else:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cugraph_oracle",
        description=(
            "Generate a deterministic JSON oracle for one (dataset, "
            "algorithm, params) triple using Python cuGraph."
        ),
    )
    parser.add_argument(
        "--data-root",
        default=str(Path(__file__).resolve().parent.parent / "data" / "cugraph"),
        help="Directory holding manifest.json and local cuGraph datasets.",
    )
    parser.add_argument("--dataset", required=True, help="Dataset id (e.g. karate).")
    parser.add_argument(
        "--algorithm",
        required=True,
        choices=sorted(ALGORITHMS),
        help="Algorithm id.",
    )
    parser.add_argument(
        "--params",
        default="{}",
        help="JSON object of algorithm-specific parameters.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed; recorded in the header. Unused by deterministic algorithms.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path; defaults to stdout.",
    )
    args = parser.parse_args(argv)
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
