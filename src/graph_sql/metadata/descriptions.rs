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
        let (source_name, source_type) = if algorithm == CugraphAlgorithm::Bfs {
            ("source_vertex_or_vertices", "Int64|Utf8|List<Int64>|List<Utf8>|null-with-source_vertices_table")
        } else {
            ("source_vertex", "Int64")
        };
        args.insert(1, json!({"name": source_name, "type": source_type, "required": true}));
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
            .map(|field| {
                json!({
                    "name": field.name(),
                    "data_type": data_type_label(field.data_type()),
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
            "BFS traversal ignores edge weights; distance is a hop count",
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

pub(super) fn purpose(algorithm: CugraphAlgorithm) -> &'static str {
    match algorithm {
        CugraphAlgorithm::Bfs => {
            "Run BFS over an edge-list relation and optionally return normalized reachability, target metadata, or one reconstructed source-to-target path."
        },
        CugraphAlgorithm::PageRank => {
            "Rank vertices by iterative link analysis so high-value vertices are those connected from other high-value vertices."
        },
        CugraphAlgorithm::PersonalizedPageRank => {
            "Rank vertices with PageRank while biasing the teleport distribution using a relation-backed personalization vector."
        },
        CugraphAlgorithm::Louvain => {
            "Assign vertices to communities by optimizing modularity over the edge-list graph."
        },
        CugraphAlgorithm::EigenvectorCentrality => {
            "Score vertices by their connection to other high-scoring vertices using eigenvector centrality."
        },
        CugraphAlgorithm::KatzCentrality => {
            "Score vertices by attenuated walks so direct and indirect neighbors both contribute to centrality."
        },
        CugraphAlgorithm::Hits => "Compute hub and authority scores for directed-link style graphs.",
        CugraphAlgorithm::BetweennessCentrality => {
            "Score vertices by how often they appear on shortest paths, using exact all-vertex traversal, an explicit seed list, or a deterministic k seed count for approximate results."
        },
        CugraphAlgorithm::EdgeBetweennessCentrality => {
            "Score edges by shortest-path participation, using exact all-vertex traversal, an explicit seed list, or a deterministic k seed count for approximate results."
        },
        CugraphAlgorithm::Sssp => {
            "Find shortest paths from one source vertex, using edge weights when a weight column is supplied."
        },
        CugraphAlgorithm::DegreesAll => "Return both incoming and outgoing edge counts for each vertex.",
        CugraphAlgorithm::InDegreesAll => "Return incoming edge counts for each vertex.",
        CugraphAlgorithm::OutDegreesAll => "Return outgoing edge counts for each vertex.",
        CugraphAlgorithm::WeaklyConnectedComponents => {
            "Label components that are connected when edge direction is ignored."
        },
        CugraphAlgorithm::StronglyConnectedComponents => {
            "Label components whose vertices are mutually reachable under directed edges."
        },
        CugraphAlgorithm::TriangleCountAll => {
            "Count the triangles incident to each vertex in an undirected graph view."
        },
        CugraphAlgorithm::CoreNumber => "Compute each vertex's core number to identify dense graph regions.",
        CugraphAlgorithm::KCore => "Return the edge set induced by the selected k-core subgraph.",
        CugraphAlgorithm::MinimumSpanningTree => {
            "Select a minimum-weight spanning tree edge set over an undirected graph view, requiring an explicit weight column."
        },
        CugraphAlgorithm::Leiden => {
            "Assign vertices to communities with Leiden refinement for modularity-style clustering."
        },
        CugraphAlgorithm::Ecg => "Assign vertices to communities using ensemble clustering for graphs.",
        CugraphAlgorithm::SpectralModularityMaximization => {
            "Cluster vertices by spectral modularity maximization over an undirected graph view."
        },
        CugraphAlgorithm::BalancedCutClustering => {
            "Cluster vertices by optimizing a balanced-cut spectral objective over an undirected graph view."
        },
        CugraphAlgorithm::ForceAtlas2 => {
            "Compute deterministic two-dimensional force-directed layout coordinates for graph visualization."
        },
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
            "sql": "SELECT * FROM cugraph_personalized_pagerank('target_edges', 'src', 'dst', NULL, '{\"personalization_table\":\"ppr_seeds\",\"personalization_vertex_col\":\"vertex\",\"personalization_value_col\":\"value\"}')",
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
        "sql": format!("SELECT * FROM {function_name}('target_edges')"),
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
        "sql": format!("SELECT * FROM {function_name}('target_edges', 123)"),
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
        "sql": format!("SELECT * FROM {function_name}('target_edges', 'src', 'dst', 'weight')"),
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
        "sql": format!("SELECT * FROM {function_name}('target_edges', 123, 'src', 'dst', 'weight')"),
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
            "name": "multi_source_bfs_list",
            "sql": "SELECT * FROM cugraph_bfs('target_edges', [123,456], 'src', 'dst', NULL, '{\"output_mode\":\"normalized\"}')",
            "call_json": {
                "source_vertex": [123, 456],
                "src_col": "src",
                "dst_col": "dst",
                "options": {"output_mode": "normalized"}
            }
        },
        {
            "name": "multi_source_bfs_table",
            "sql": "SELECT * FROM cugraph_bfs('target_edges', NULL, 'src', 'dst', NULL, '{\"source_vertices_table\":\"bfs_sources\",\"source_vertex_col\":\"vertex_id\",\"output_mode\":\"normalized\"}')",
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
