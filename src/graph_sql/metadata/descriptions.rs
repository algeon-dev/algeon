use serde_json::{Value, json};

use super::super::{CugraphAlgorithm, stable_result_schema};

pub(super) fn argument_descriptions_json(algorithm: CugraphAlgorithm) -> String {
    let mut args = vec![
        json!({"name": "table_name", "type": "Utf8", "required": true}),
        json!({"name": "src_col", "type": "Utf8", "required": false, "default": "src"}),
        json!({"name": "dst_col", "type": "Utf8", "required": false, "default": "dst"}),
        json!({
            "name": "weight_col",
            "type": "Utf8|null",
            "required": algorithm.native_algorithm().weight_policy().requires_weight(),
            "semantic_effect": weight_argument_semantic_effect(algorithm),
            "note": weight_argument_note(algorithm)
        }),
        json!({"name": "options_json", "type": "Utf8", "required": false}),
    ];
    if algorithm.requires_source_vertex() {
        let source = if algorithm == CugraphAlgorithm::Bfs {
            json!({
                "name": "source_vertex_or_vertices",
                "json_key": "source_vertex",
                "type": "Int64|Utf8|List<Int64>|List<Utf8>|null-with-source_vertices_table",
                "required": true,
                "selector_policy": "literal scalar/list source selector, or NULL with source_vertices_table/source_vertex_col options"
            })
        } else {
            json!({"name": "source_vertex", "type": "Int64", "required": true})
        };
        args.insert(1, source);
    }
    Value::Array(args).to_string()
}

fn weight_argument_note(algorithm: CugraphAlgorithm) -> &'static str {
    match algorithm {
        CugraphAlgorithm::Bfs => "accepted as an edge-column binding but ignored by BFS traversal",
        CugraphAlgorithm::MinimumSpanningTree => "required edge weight column",
        _ if !algorithm.native_algorithm().weight_policy().consumes_weight() => {
            "accepted as an edge-column binding; native algorithm execution does not consume weights"
        },
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
    if algorithm == CugraphAlgorithm::Bfs {
        return Value::Array(
            [
                "vertex",
                "distance",
                "predecessor",
                "reachable",
                "target_found",
                "target_distance",
                "path_index",
                "source",
                "target",
            ]
            .into_iter()
            .map(|name| {
                json!({
                    "name": name,
                    "description": result_field_description(algorithm, name)
                })
            })
            .collect(),
        )
        .to_string();
    }
    let schema = stable_result_schema(algorithm);
    Value::Array(
        schema
            .fields()
            .iter()
            .map(|field| {
                json!({
                    "name": field.name(),
                    "description": result_field_description(algorithm, field.name())
                })
            })
            .collect(),
    )
    .to_string()
}

fn result_field_description(algorithm: CugraphAlgorithm, field_name: &str) -> &'static str {
    match (algorithm, field_name) {
        (CugraphAlgorithm::PageRank | CugraphAlgorithm::PersonalizedPageRank, "vertex") => {
            "Vertex receiving the PageRank score."
        },
        (CugraphAlgorithm::PageRank | CugraphAlgorithm::PersonalizedPageRank, "value") => {
            "PageRank score for the vertex."
        },
        (CugraphAlgorithm::EigenvectorCentrality, "vertex") => "Vertex receiving the eigenvector centrality score.",
        (CugraphAlgorithm::EigenvectorCentrality, "value") => "Eigenvector centrality score for the vertex.",
        (CugraphAlgorithm::KatzCentrality, "vertex") => "Vertex receiving the Katz centrality score.",
        (CugraphAlgorithm::KatzCentrality, "value") => "Katz centrality score for the vertex.",
        (CugraphAlgorithm::Bfs, "vertex") => "Vertex reached or considered by the BFS traversal.",
        (CugraphAlgorithm::Bfs, "distance") => "Hop-count distance from the nearest selected BFS source vertex.",
        (CugraphAlgorithm::Bfs, "predecessor") => {
            "Previous vertex on the discovered BFS tree, null for selected source vertices or unreachable vertices depending on output mode."
        },
        (CugraphAlgorithm::Bfs, "reachable") => "Whether the vertex is reachable under normalized BFS output.",
        (CugraphAlgorithm::Bfs, "target_found") => {
            "Whether a requested target vertex was reached by the BFS traversal."
        },
        (CugraphAlgorithm::Bfs, "target_distance") => {
            "Hop-count distance to the requested target vertex, null when the target was not reached."
        },
        (CugraphAlgorithm::Bfs, "path_index") => "Zero-based row position in the reconstructed source-to-target path.",
        (CugraphAlgorithm::Bfs, "source") => "Source vertex for the reconstructed BFS path.",
        (CugraphAlgorithm::Bfs, "target") => "Target vertex for the reconstructed BFS path.",
        (CugraphAlgorithm::Sssp, "vertex") => "Vertex reached by the shortest-path traversal.",
        (CugraphAlgorithm::Sssp, "distance") => "Shortest weighted-path distance from the source vertex.",
        (CugraphAlgorithm::Sssp, "predecessor") => {
            "Previous vertex on the shortest-path tree, null for the source or unreachable vertices."
        },
        (CugraphAlgorithm::Louvain, "vertex") => "Vertex assigned to a Louvain community.",
        (CugraphAlgorithm::Louvain, "partition") => "Community identifier assigned by Louvain.",
        (CugraphAlgorithm::Leiden, "vertex") => "Vertex assigned to a Leiden community.",
        (CugraphAlgorithm::Leiden, "partition") => "Community identifier assigned by Leiden.",
        (CugraphAlgorithm::Ecg, "vertex") => "Vertex assigned to an ECG community.",
        (CugraphAlgorithm::Ecg, "partition") => "Community identifier assigned by ECG.",
        (CugraphAlgorithm::SpectralModularityMaximization | CugraphAlgorithm::BalancedCutClustering, "vertex") => {
            "Vertex assigned to a spectral clustering partition."
        },
        (CugraphAlgorithm::SpectralModularityMaximization, "partition") => {
            "Cluster identifier assigned by spectral modularity maximization."
        },
        (CugraphAlgorithm::BalancedCutClustering, "partition") => {
            "Cluster identifier assigned by balanced-cut spectral clustering."
        },
        (CugraphAlgorithm::ForceAtlas2, "vertex") => "Vertex receiving ForceAtlas2 layout coordinates.",
        (CugraphAlgorithm::ForceAtlas2, "x") => "X coordinate assigned by ForceAtlas2.",
        (CugraphAlgorithm::ForceAtlas2, "y") => "Y coordinate assigned by ForceAtlas2.",
        (CugraphAlgorithm::Hits, "vertex") => "Vertex receiving HITS scores.",
        (CugraphAlgorithm::Hits, "hub_score") => "HITS hub score for the vertex.",
        (CugraphAlgorithm::Hits, "authority_score") => "HITS authority score for the vertex.",
        (CugraphAlgorithm::DegreesAll, "vertex") => "Vertex whose degree counts are reported.",
        (CugraphAlgorithm::DegreesAll, "in_degree") => "Number of incoming edges for the vertex.",
        (CugraphAlgorithm::DegreesAll, "out_degree") => "Number of outgoing edges for the vertex.",
        (CugraphAlgorithm::InDegreesAll, "vertex") => "Vertex whose in-degree count is reported.",
        (CugraphAlgorithm::InDegreesAll, "in_degree") => "Number of incoming edges for the vertex.",
        (CugraphAlgorithm::OutDegreesAll, "vertex") => "Vertex whose out-degree count is reported.",
        (CugraphAlgorithm::OutDegreesAll, "out_degree") => "Number of outgoing edges for the vertex.",
        (CugraphAlgorithm::WeaklyConnectedComponents, "vertex") => "Vertex assigned to a weakly connected component.",
        (CugraphAlgorithm::WeaklyConnectedComponents, "label") => {
            "Weakly connected component identifier for the vertex."
        },
        (CugraphAlgorithm::StronglyConnectedComponents, "vertex") => {
            "Vertex assigned to a strongly connected component."
        },
        (CugraphAlgorithm::StronglyConnectedComponents, "label") => {
            "Strongly connected component identifier for the vertex."
        },
        (CugraphAlgorithm::TriangleCountAll, "vertex") => "Vertex whose triangle participation is reported.",
        (CugraphAlgorithm::TriangleCountAll, "triangle_count") => "Number of triangles incident to the vertex.",
        (CugraphAlgorithm::CoreNumber, "vertex") => "Vertex whose core number is reported.",
        (CugraphAlgorithm::CoreNumber, "core_number") => {
            "Largest k value for which the vertex belongs to the graph k-core."
        },
        (CugraphAlgorithm::KCore, "src") => "Source vertex of an edge retained in the k-core subgraph.",
        (CugraphAlgorithm::KCore, "dst") => "Destination vertex of an edge retained in the k-core subgraph.",
        (CugraphAlgorithm::MinimumSpanningTree, "src") => {
            "Source vertex of an edge selected for the minimum spanning tree."
        },
        (CugraphAlgorithm::MinimumSpanningTree, "dst") => {
            "Destination vertex of an edge selected for the minimum spanning tree."
        },
        _ => "Algorithm result column.",
    }
}

pub(super) fn limitations_json(algorithm: CugraphAlgorithm) -> String {
    let mut limitations = vec![
        "dry-run validates table resolution, column presence, static dtypes, and options only",
        "dry-run does not scan edge data, construct a graph, or prove source-vertex existence",
    ];
    if algorithm == CugraphAlgorithm::Bfs {
        limitations.extend([
            "source lists and source_vertices_table are multi-source BFS, not one BFS per source",
            "source lists and source_vertices_table must contain at most one source per connected component",
            "raw/normalized output does not expose origin source per vertex",
            "string logical vertex-domain BFS rejects edge_id_col, include_edge_ids_*, and include_edge_id_col",
            "path output requires exactly one target row at execution time",
            "options_schema_json is typed registry metadata; validate_call remains the authoritative call-specific checker",
        ]);
    }
    if algorithm == CugraphAlgorithm::ForceAtlas2 {
        limitations.push("prevent_overlapping=true is rejected until vertex_radius side inputs are exposed");
    }
    if matches!(
        algorithm.native_algorithm().vertex_id_type_support(),
        nexus_query_engine::GraphVertexIdTypeSupport::Int32OnlyLegacyEdgeType
    ) {
        limitations.push("legacy cuGraph implementation requires Int32 source and destination vertex columns");
    }
    if algorithm.native_algorithm().definition().required_graph_directedness() == Some(false) {
        limitations.push("cuGraph requires directed=false so the graph is constructed as an undirected/symmetric view");
    }
    json!(limitations).to_string()
}

pub(super) fn summary(algorithm: CugraphAlgorithm) -> &'static str {
    match algorithm {
        CugraphAlgorithm::Bfs => {
            "Multi-source-capable breadth-first search; source selectors must contain at most one source per connected component."
        },
        CugraphAlgorithm::PageRank => "Compute PageRank scores.",
        CugraphAlgorithm::PersonalizedPageRank => "Compute personalized PageRank scores.",
        CugraphAlgorithm::Louvain => "Detect communities with Louvain modularity optimization.",
        CugraphAlgorithm::EigenvectorCentrality => "Compute eigenvector centrality scores.",
        CugraphAlgorithm::KatzCentrality => "Compute Katz centrality scores.",
        CugraphAlgorithm::Hits => "Compute HITS hub and authority scores.",
        CugraphAlgorithm::BetweennessCentrality => {
            "Compute exact or explicitly seeded/k-sampled approximate betweenness centrality scores."
        },
        CugraphAlgorithm::EdgeBetweennessCentrality => {
            "Compute exact or explicitly seeded/k-sampled approximate edge betweenness centrality scores."
        },
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
        CugraphAlgorithm::ForceAtlas2 => "Compute ForceAtlas2 two-dimensional layout coordinates.",
    }
}

pub(super) fn primary_example_sql(algorithm: CugraphAlgorithm) -> String {
    let function_name = algorithm.function_name();
    match algorithm {
        CugraphAlgorithm::Bfs => {
            "SELECT * FROM cugraph_bfs('target_edges', 123, 'src', 'dst', NULL, '{\"depth_limit\":4}')".to_string()
        },
        CugraphAlgorithm::PersonalizedPageRank => {
            "SELECT * FROM cugraph_personalized_pagerank('target_edges', 'src', 'dst', NULL, '{\"personalization_table\":\"ppr_seeds\",\"personalization_vertex_col\":\"vertex\",\"personalization_value_col\":\"value\"}')".to_string()
        },
        CugraphAlgorithm::Sssp => {
            format!("SELECT * FROM {function_name}('target_edges', 123)")
        },
        CugraphAlgorithm::MinimumSpanningTree => {
            format!("SELECT * FROM {function_name}('target_edges', 'src', 'dst', 'weight')")
        },
        _ => format!("SELECT * FROM {function_name}('target_edges')"),
    }
}

pub(super) fn examples_json(algorithm: CugraphAlgorithm) -> String {
    match algorithm {
        CugraphAlgorithm::Bfs => bfs_examples_json(),
        CugraphAlgorithm::PersonalizedPageRank => personalized_pagerank_examples_json(),
        CugraphAlgorithm::Sssp => json!([
            source_vertex_example(algorithm, "basic_sssp"),
            weighted_source_vertex_example(algorithm, "weighted_sssp")
        ])
        .to_string(),
        CugraphAlgorithm::MinimumSpanningTree => json!([weighted_example(algorithm, "weighted_mst")]).to_string(),
        _ => json!([minimal_example(algorithm)]).to_string(),
    }
}

fn personalized_pagerank_examples_json() -> String {
    json!([
        {
            "name": "basic_personalized_pagerank",
            "function_name": "cugraph_personalized_pagerank",
            "call_json": {
                "src_col": "src",
                "dst_col": "dst",
                "options": {
                    "personalization_table": "ppr_seeds",
                    "personalization_vertex_col": "vertex",
                    "personalization_value_col": "value"
                }
            }
        }
    ])
    .to_string()
}

fn minimal_example(algorithm: CugraphAlgorithm) -> Value {
    let function_name = algorithm.function_name();
    json!({
        "name": format!("basic_{}", algorithm.native_algorithm().definition().name),
        "function_name": function_name,
        "call_json": {
            "src_col": "src",
            "dst_col": "dst"
        }
    })
}

fn source_vertex_example(algorithm: CugraphAlgorithm, name: &str) -> Value {
    let function_name = algorithm.function_name();
    json!({
        "name": name,
        "function_name": function_name,
        "call_json": {
            "source_vertex": 123,
            "src_col": "src",
            "dst_col": "dst"
        }
    })
}

fn weighted_example(algorithm: CugraphAlgorithm, name: &str) -> Value {
    let function_name = algorithm.function_name();
    json!({
        "name": name,
        "function_name": function_name,
        "call_json": {
            "src_col": "src",
            "dst_col": "dst",
            "weight_col": "weight"
        }
    })
}

fn weighted_source_vertex_example(algorithm: CugraphAlgorithm, name: &str) -> Value {
    let function_name = algorithm.function_name();
    json!({
        "name": name,
        "function_name": function_name,
        "call_json": {
            "source_vertex": 123,
            "src_col": "src",
            "dst_col": "dst",
            "weight_col": "weight"
        }
    })
}

fn bfs_examples_json() -> String {
    json!([
        {
            "name": "basic_bfs",
            "function_name": "cugraph_bfs",
            "call_json": {
                "source_vertex": 123,
                "src_col": "src",
                "dst_col": "dst",
                "options": {"depth_limit": 4}
            }
        },
        {
            "name": "normalized_bfs",
            "function_name": "cugraph_bfs",
            "call_json": {
                "source_vertex": 123,
                "src_col": "src",
                "dst_col": "dst",
                "options": {"depth_limit": 4, "output_mode": "normalized"}
            }
        },
        {
            "name": "multi_source_bfs_list",
            "function_name": "cugraph_bfs",
            "call_json": {
                "source_vertex": [123, 456],
                "src_col": "src",
                "dst_col": "dst",
                "options": {"output_mode": "normalized"}
            }
        },
        {
            "name": "multi_source_bfs_table",
            "function_name": "cugraph_bfs",
            "call_json": {
                "source_vertex": null,
                "src_col": "src",
                "dst_col": "dst",
                "options": {
                    "source_vertices_table": "bfs_sources",
                    "source_vertex_col": "vertex_id",
                    "output_mode": "normalized"
                }
            }
        },
        {
            "name": "target_metadata",
            "function_name": "cugraph_bfs",
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
            "function_name": "cugraph_bfs",
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
