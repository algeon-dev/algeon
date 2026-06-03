use serde_json::{Value, json};

use super::{
    super::{CugraphAlgorithm, stable_result_schema},
    json_schema::data_type_label,
};

pub(super) fn argument_descriptions_json(algorithm: CugraphAlgorithm) -> String {
    let mut args = vec![
        json!({"name": "table_name", "type": "Utf8", "required": true}),
        json!({"name": "src_col", "type": "Utf8", "required": false, "default": "src"}),
        json!({"name": "dst_col", "type": "Utf8", "required": false, "default": "dst"}),
        json!({
            "name": "weight_col",
            "type": "Utf8|null",
            "required": algorithm.native_algorithm().weight_policy().requires_weight(),
            "weight_col_policy": algorithm.weight_col_policy(),
            "semantic_effect": weight_argument_semantic_effect(algorithm),
            "note": weight_argument_note(algorithm)
        }),
        json!({"name": "options_json", "type": "Utf8", "required": false}),
    ];
    if algorithm.requires_source_vertex() {
        let source_type = if algorithm == CugraphAlgorithm::Bfs { "Int64|Utf8" } else { "Int64" };
        args.insert(1, json!({"name": "source_vertex", "type": source_type, "required": true}));
    }
    Value::Array(args).to_string()
}

fn weight_argument_note(algorithm: CugraphAlgorithm) -> &'static str {
    match algorithm {
        CugraphAlgorithm::Bfs => "accepted as an edge-column binding but ignored by BFS traversal",
        CugraphAlgorithm::MinimumSpanningTree => "required edge weight column",
        _ => "optional edge weight column for graph construction when supported by the algorithm",
    }
}

fn weight_argument_semantic_effect(algorithm: CugraphAlgorithm) -> &'static str {
    match algorithm {
        CugraphAlgorithm::Bfs => "none for traversal; BFS distance is a hop count",
        CugraphAlgorithm::MinimumSpanningTree => "required edge weights define the spanning-tree objective",
        _ if algorithm.native_algorithm().weight_policy().consumes_weight() => {
            "edge weights affect algorithm results when provided"
        },
        _ => "none for this algorithm",
    }
}

pub(super) fn result_field_descriptions_json(algorithm: CugraphAlgorithm) -> String {
    let schema = stable_result_schema(algorithm);
    Value::Array(
        schema
            .fields()
            .iter()
            .map(|field| json!({"name": field.name(), "data_type": data_type_label(field.data_type())}))
            .collect(),
    )
    .to_string()
}

pub(super) fn limitations_json(algorithm: CugraphAlgorithm) -> String {
    let mut limitations = vec![
        "dry-run validates table resolution, column presence, static dtypes, and options only",
        "dry-run does not scan edge data, construct a graph, or prove source-vertex existence",
    ];
    if algorithm == CugraphAlgorithm::Bfs {
        limitations.extend([
            "BFS traversal ignores edge weights; distance is a hop count",
            "string logical vertex-domain BFS rejects edge_id_col, include_edge_ids_*, and include_edge_id_col",
            "path output requires exactly one target row at execution time",
            "options_schema_json is typed registry metadata; validate_call remains the authoritative call-specific checker",
        ]);
    }
    json!(limitations).to_string()
}

pub(super) fn summary(algorithm: CugraphAlgorithm) -> &'static str {
    match algorithm {
        CugraphAlgorithm::Bfs => {
            "Single-source breadth-first search; weight_col is accepted but traversal uses hop counts."
        },
        CugraphAlgorithm::PageRank | CugraphAlgorithm::PageRankStrict => "Compute PageRank scores.",
        CugraphAlgorithm::Louvain => "Detect communities with Louvain modularity optimization.",
        CugraphAlgorithm::EigenvectorCentrality => "Compute eigenvector centrality scores.",
        CugraphAlgorithm::KatzCentrality => "Compute Katz centrality scores.",
        CugraphAlgorithm::Hits => "Compute HITS hub and authority scores.",
        CugraphAlgorithm::Sssp => "Compute single-source shortest paths.",
        CugraphAlgorithm::DegreesAll => "Compute in-degree and out-degree for every vertex.",
        CugraphAlgorithm::InDegreesAll => "Compute in-degree for every vertex.",
        CugraphAlgorithm::OutDegreesAll => "Compute out-degree for every vertex.",
        CugraphAlgorithm::WeaklyConnectedComponents => "Compute weakly connected components.",
        CugraphAlgorithm::StronglyConnectedComponents => "Compute strongly connected components.",
        CugraphAlgorithm::TriangleCountAll => "Compute triangle counts for every vertex.",
        CugraphAlgorithm::CoreNumber => "Compute graph core numbers.",
        CugraphAlgorithm::KCore => "Return the graph k-core edge set.",
        CugraphAlgorithm::MinimumSpanningTree => "Compute a minimum spanning tree.",
        CugraphAlgorithm::Leiden => "Detect communities with Leiden optimization.",
        CugraphAlgorithm::Ecg => "Detect communities with ensemble clustering for graphs.",
        CugraphAlgorithm::SpectralModularityMaximization => "Cluster vertices with spectral modularity maximization.",
        CugraphAlgorithm::BalancedCutClustering => "Cluster vertices with balanced-cut spectral clustering.",
    }
}

pub(super) fn purpose(algorithm: CugraphAlgorithm) -> &'static str {
    match algorithm {
        CugraphAlgorithm::Bfs => {
            "Run BFS over an edge-list relation and optionally return normalized reachability, target metadata, or one reconstructed source-to-target path."
        },
        _ => "Run the named cuGraph algorithm over an edge-list relation using the SQL table-function contract.",
    }
}

pub(super) fn examples_json(algorithm: CugraphAlgorithm) -> String {
    if algorithm != CugraphAlgorithm::Bfs {
        return "[]".to_string();
    }
    json!([
        {
            "name": "basic_bfs",
            "sql": "SELECT * FROM cugraph_bfs('target_edges', 123, 'src', 'dst', NULL, '{\"depth_limit\":4}')",
            "call_json": {
                "source_vertex": 123,
                "src_col": "src",
                "dst_col": "dst",
                "options": {"depth_limit": 4}
            }
        },
        {
            "name": "normalized_bfs",
            "sql": "SELECT * FROM cugraph_bfs('target_edges', 123, 'src', 'dst', NULL, '{\"depth_limit\":4,\"output_mode\":\"normalized\"}')",
            "call_json": {
                "source_vertex": 123,
                "src_col": "src",
                "dst_col": "dst",
                "options": {"depth_limit": 4, "output_mode": "normalized"}
            }
        },
        {
            "name": "target_metadata",
            "sql": "SELECT * FROM cugraph_bfs('target_edges', 123, 'src', 'dst', NULL, '{\"output_mode\":\"normalized\",\"target_vertices_table\":\"targets\",\"target_vertex_col\":\"vertex_id\",\"return_target_info\":true}')",
            "call_json": {
                "source_vertex": 123,
                "src_col": "src",
                "dst_col": "dst",
                "options": {
                    "output_mode": "normalized",
                    "target_vertices_table": "targets",
                    "target_vertex_col": "vertex_id",
                    "return_target_info": true
                }
            }
        },
        {
            "name": "path_output",
            "sql": "SELECT * FROM cugraph_bfs('target_edges', 123, 'src', 'dst', NULL, '{\"output_mode\":\"path\",\"target_vertices_table\":\"targets\",\"target_vertex_col\":\"vertex_id\"}') ORDER BY path_index",
            "call_json": {
                "source_vertex": 123,
                "src_col": "src",
                "dst_col": "dst",
                "options": {
                    "output_mode": "path",
                    "target_vertices_table": "targets",
                    "target_vertex_col": "vertex_id"
                }
            }
        },
        {
            "name": "dry_run_path",
            "sql": "SELECT valid, error_code, message, output_schema_json, normalized_options_json FROM cugraph_validate_call('cugraph_bfs', 'target_edges', '{\"source_vertex\":123,\"src_col\":\"src\",\"dst_col\":\"dst\",\"options\":{\"output_mode\":\"path\",\"target_vertices_table\":\"targets\",\"target_vertex_col\":\"vertex_id\"}}')",
            "call_json": {
                "source_vertex": 123,
                "src_col": "src",
                "dst_col": "dst",
                "options": {
                    "output_mode": "path",
                    "target_vertices_table": "targets",
                    "target_vertex_col": "vertex_id"
                }
            }
        }
    ])
    .to_string()
}
