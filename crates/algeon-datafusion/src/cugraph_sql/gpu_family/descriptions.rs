use query_engine::plan::GraphVertexIdTypeSupport;
use serde_json::{Value, json};

use crate::cugraph_sql::{CugraphAlgorithm, stable_result_schema};

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
        json!({
            "name": "options_json",
            "type": "Utf8",
            "required": is_similarity(algorithm),
            "note": if is_similarity(algorithm) {
                "required: identifies the explicit candidate-pair relation with vertex_pairs_table, first_vertex_col, and second_vertex_col"
            } else {
                ""
            }
        }),
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
        } else if algorithm.definition().supports_logical_string_vertices() {
            json!({"name": "source_vertex", "type": "Int64|Utf8", "required": true})
        } else {
            json!({"name": "source_vertex", "type": "Int64", "required": true})
        };
        args.insert(1, source);
    }
    Value::Array(args).to_string()
}

fn is_similarity(algorithm: CugraphAlgorithm) -> bool {
    matches!(
        algorithm,
        CugraphAlgorithm::Jaccard | CugraphAlgorithm::Sorensen | CugraphAlgorithm::Overlap | CugraphAlgorithm::Cosine
    )
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
        (CugraphAlgorithm::SpectralModularityMaximization, "vertex") => {
            "Vertex assigned to a spectral clustering partition."
        },
        (CugraphAlgorithm::SpectralModularityMaximization, "partition") => {
            "Cluster identifier assigned by spectral modularity maximization."
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
            "Weakly connected component identifier for the vertex: the minimum vertex id in the component, stable across runs."
        },
        (CugraphAlgorithm::StronglyConnectedComponents, "vertex") => {
            "Vertex assigned to a strongly connected component."
        },
        (CugraphAlgorithm::StronglyConnectedComponents, "label") => {
            "Strongly connected component identifier for the vertex: the minimum vertex id in the component, stable across runs."
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
        (algorithm, "first") if is_similarity(algorithm) => "First vertex from the explicit candidate-pair relation.",
        (algorithm, "second") if is_similarity(algorithm) => "Second vertex from the explicit candidate-pair relation.",
        (algorithm, "similarity") if is_similarity(algorithm) => {
            "Similarity coefficient for the explicit candidate pair."
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
        limitations.extend([
            "vertex_attributes and initial_positions must contain exactly one non-null row for every graph vertex",
            "duplicate, missing, and extra ForceAtlas2 side-input vertices are rejected at execution",
            "radius, mobility, mass, x, and y side-input columns must be Float32",
            "prevent_overlapping=true requires a radius binding",
        ]);
    }
    if is_similarity(algorithm) {
        limitations.extend([
            "similarity is explicit-pair only; all-pairs candidate generation and all-pairs top-k search are not exposed",
            "options_json must name vertex_pairs_table, first_vertex_col, and second_vertex_col",
            "candidate-pair columns must match the graph vertex domain; logical string graphs accept Utf8, LargeUtf8, or Utf8View independently per column",
            "null candidate-pair values are rejected at execution and are never dropped",
            "candidate pairs are a multiset: duplicate, reversed, and self pairs remain distinct output rows",
            "result rows have no global ordering; use ORDER BY when order is required",
            "providing weight_col selects weighted similarity; omitting it selects unit-weight similarity",
        ]);
    }
    if algorithm == CugraphAlgorithm::Sssp {
        limitations.push("omitting weight_col assigns unit cost 1 to every edge");
    }
    if matches!(
        algorithm.native_algorithm().vertex_id_type_support(),
        GraphVertexIdTypeSupport::Int32OnlyLegacyEdgeType
    ) {
        limitations.push("cuGraph C API edge-type dispatch requires Int32 source and destination vertex columns");
    }
    if algorithm.native_algorithm().definition().required_graph_directedness() == Some(false) {
        limitations.push("cuGraph requires directed=false so the graph is constructed as an undirected/symmetric view");
    }
    json!(limitations).to_string()
}

pub(super) fn summary(algorithm: CugraphAlgorithm) -> &'static str {
    match algorithm {
        CugraphAlgorithm::Bfs => {
            "Visit reachable vertices in increasing unweighted hop distance from one or more sources, returning distances and optional predecessors."
        },
        CugraphAlgorithm::PageRank => {
            "Rank vertices by the stationary probability of a damped random walk that follows outgoing edges."
        },
        CugraphAlgorithm::PersonalizedPageRank => {
            "Rank vertices with PageRank while biasing random-walk restarts toward explicitly weighted personalization vertices."
        },
        CugraphAlgorithm::Louvain => {
            "Build a hierarchy of communities by greedily moving vertices and aggregating partitions to maximize modularity."
        },
        CugraphAlgorithm::EigenvectorCentrality => {
            "Score vertices by connections to other high-scoring vertices, using power iteration to find the dominant eigenvector."
        },
        CugraphAlgorithm::KatzCentrality => {
            "Score vertices from the number of walks that reach them, attenuating longer walks and adding a baseline contribution."
        },
        CugraphAlgorithm::Hits => {
            "Compute mutually reinforcing hub and authority scores: strong hubs point to strong authorities, and strong authorities are linked from strong hubs."
        },
        CugraphAlgorithm::BetweennessCentrality => {
            "Measure how often each vertex lies on shortest paths between other vertex pairs, exactly or from an explicit sample of source vertices."
        },
        CugraphAlgorithm::EdgeBetweennessCentrality => {
            "Measure how often each edge lies on shortest paths between vertex pairs, exactly or from an explicit sample of source vertices."
        },
        CugraphAlgorithm::Sssp => {
            "Compute minimum path distances and predecessors from one source vertex, using bound edge weights or unit cost 1 when weight_col is omitted."
        },
        CugraphAlgorithm::DegreesAll => "Count both incoming and outgoing edges for every vertex in the graph.",
        CugraphAlgorithm::InDegreesAll => "Count incoming edges for every vertex in the graph.",
        CugraphAlgorithm::OutDegreesAll => "Count outgoing edges for every vertex in the graph.",
        CugraphAlgorithm::WeaklyConnectedComponents => {
            "Label maximal connected subgraphs after treating directed edges as undirected."
        },
        CugraphAlgorithm::StronglyConnectedComponents => {
            "Label maximal directed subgraphs in which every vertex is reachable from every other vertex."
        },
        CugraphAlgorithm::TriangleCountAll => {
            "Count the number of three-vertex cycles incident to every vertex in the graph."
        },
        CugraphAlgorithm::CoreNumber => {
            "Assign each vertex the largest k for which it belongs to a k-core, where every vertex has degree at least k within that subgraph."
        },
        CugraphAlgorithm::KCore => {
            "Return the edges of the maximal subgraph whose vertices each have degree at least k within that subgraph."
        },
        CugraphAlgorithm::MinimumSpanningTree => {
            "Select a minimum-total-weight acyclic edge set, producing a spanning tree for a connected graph or a spanning forest otherwise."
        },
        CugraphAlgorithm::Leiden => {
            "Find modularity-based communities with refinement steps that improve the internal connectedness of the partitions."
        },
        CugraphAlgorithm::Ecg => {
            "Stabilize community assignments by combining an ensemble of randomized Louvain partitions into a final consensus clustering."
        },
        CugraphAlgorithm::SpectralModularityMaximization => {
            "Partition vertices by embedding the graph with leading modularity eigenvectors and clustering that embedding with k-means."
        },
        CugraphAlgorithm::ForceAtlas2 => {
            "Place vertices in two dimensions with a force-directed simulation that attracts connected vertices and repels vertices from one another."
        },
        CugraphAlgorithm::Jaccard => {
            "Compare explicit vertex pairs by dividing the size of their shared-neighbor intersection by the size of their neighbor union."
        },
        CugraphAlgorithm::Sorensen => {
            "Compare explicit vertex pairs as twice their shared-neighbor count divided by the sum of their neighbor counts."
        },
        CugraphAlgorithm::Overlap => {
            "Compare explicit vertex pairs by dividing their shared-neighbor count by the smaller of their two neighbor counts."
        },
        CugraphAlgorithm::Cosine => {
            "Compare the neighbor vectors of explicit vertex pairs using cosine similarity, with optional edge-weight contributions."
        },
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
        algorithm if is_similarity(algorithm) => format!(
            "SELECT * FROM {function_name}('target_edges', 'src', 'dst', NULL, '{{\"vertex_pairs_table\":\"candidate_pairs\",\"first_vertex_col\":\"first\",\"second_vertex_col\":\"second\"}}')"
        ),
        _ => format!("SELECT * FROM {function_name}('target_edges')"),
    }
}

pub(super) fn examples_json(algorithm: CugraphAlgorithm) -> String {
    let examples = match algorithm {
        CugraphAlgorithm::Bfs => bfs_examples(algorithm),
        CugraphAlgorithm::PersonalizedPageRank => vec![common_example(
            algorithm,
            "basic_personalized_pagerank",
            json!({
                "src_col": "src",
                "dst_col": "dst",
                "personalization_table": "ppr_seeds",
                "personalization_vertex_col": "vertex",
                "personalization_value_col": "value"
            }),
        )],
        CugraphAlgorithm::Sssp => vec![
            source_vertex_example(algorithm, "basic_sssp"),
            weighted_source_vertex_example(algorithm, "weighted_sssp"),
        ],
        CugraphAlgorithm::MinimumSpanningTree => vec![weighted_example(algorithm, "weighted_mst")],
        algorithm if is_similarity(algorithm) => vec![similarity_example(algorithm)],
        _ => vec![minimal_example(algorithm)],
    };
    Value::Array(examples).to_string()
}

fn common_example(algorithm: CugraphAlgorithm, name: impl Into<String>, options: Value) -> Value {
    json!({
        "name": name.into(),
        "function_name": algorithm.function_name(),
        "call_json": {
            "schema_version": 1,
            "relations": { "edges": { "table": "target_edges" } },
            "options": options,
        },
        "execution_sql": primary_example_sql(algorithm),
    })
}

fn similarity_example(algorithm: CugraphAlgorithm) -> Value {
    common_example(
        algorithm,
        format!("basic_{}", algorithm.native_algorithm().definition().name),
        json!({
            "src_col": "src",
            "dst_col": "dst",
            "vertex_pairs_table": "candidate_pairs",
            "first_vertex_col": "first",
            "second_vertex_col": "second"
        }),
    )
}

fn minimal_example(algorithm: CugraphAlgorithm) -> Value {
    common_example(
        algorithm,
        format!("basic_{}", algorithm.native_algorithm().definition().name),
        json!({
            "src_col": "src",
            "dst_col": "dst"
        }),
    )
}

fn source_vertex_example(algorithm: CugraphAlgorithm, name: &str) -> Value {
    common_example(
        algorithm,
        name,
        json!({
            "source_vertex": 123,
            "src_col": "src",
            "dst_col": "dst"
        }),
    )
}

fn weighted_example(algorithm: CugraphAlgorithm, name: &str) -> Value {
    common_example(
        algorithm,
        name,
        json!({
            "src_col": "src",
            "dst_col": "dst",
            "weight_col": "weight"
        }),
    )
}

fn weighted_source_vertex_example(algorithm: CugraphAlgorithm, name: &str) -> Value {
    common_example(
        algorithm,
        name,
        json!({
            "source_vertex": 123,
            "src_col": "src",
            "dst_col": "dst",
            "weight_col": "weight"
        }),
    )
}

fn bfs_examples(algorithm: CugraphAlgorithm) -> Vec<Value> {
    vec![
        common_example(
            algorithm,
            "basic_bfs",
            json!({
                "source_vertex": 123,
                "src_col": "src",
                "dst_col": "dst",
                "depth_limit": 4
            }),
        ),
        common_example(
            algorithm,
            "normalized_bfs",
            json!({
                "source_vertex": 123,
                "src_col": "src",
                "dst_col": "dst",
                "depth_limit": 4,
                "output_mode": "normalized"
            }),
        ),
        common_example(
            algorithm,
            "multi_source_bfs_list",
            json!({
                "source_vertex": [123, 456],
                "src_col": "src",
                "dst_col": "dst",
                "output_mode": "normalized"
            }),
        ),
        common_example(
            algorithm,
            "multi_source_bfs_table",
            json!({
                "source_vertex": null,
                "src_col": "src",
                "dst_col": "dst",
                "source_vertices_table": "bfs_sources",
                "source_vertex_col": "vertex_id",
                "output_mode": "normalized"
            }),
        ),
        common_example(
            algorithm,
            "target_metadata",
            json!({
                "source_vertex": 123,
                "src_col": "src",
                "dst_col": "dst",
                "output_mode": "normalized",
                "target_vertices_table": "targets",
                "target_vertex_col": "vertex_id",
                "return_target_info": true
            }),
        ),
        common_example(
            algorithm,
            "path_output",
            json!({
                "source_vertex": 123,
                "src_col": "src",
                "dst_col": "dst",
                "output_mode": "path",
                "target_vertices_table": "targets",
                "target_vertex_col": "vertex_id"
            }),
        ),
    ]
}
