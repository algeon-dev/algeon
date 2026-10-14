use query_planning::plan::GraphVertexIdTypeSupport;
use serde_json::{Value, json};

use crate::cugraph_sql::{CugraphAlgorithm, stable_result_schema};

fn is_similarity(algorithm: CugraphAlgorithm) -> bool {
    matches!(
        algorithm,
        CugraphAlgorithm::Jaccard | CugraphAlgorithm::Sorensen | CugraphAlgorithm::Overlap | CugraphAlgorithm::Cosine
    )
}

pub(super) fn result_field_descriptions_json(algorithm: CugraphAlgorithm) -> String {
    if algorithm == CugraphAlgorithm::Bfs {
        return Value::Array(
            ["vertex", "distance", "predecessor", "reachable", "path_index", "source", "target"]
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
        (CugraphAlgorithm::Bfs, "distance") => {
            "Hop-count distance from the nearest selected BFS source vertex within the predicates and depth limit; null when unreached in vertices output."
        },
        (CugraphAlgorithm::Bfs, "predecessor") => {
            "Previous vertex on the discovered BFS tree, null for selected sources and vertices not reached within the predicates and depth limit."
        },
        (CugraphAlgorithm::Bfs, "reachable") => {
            "Whether the vertex was reached within the predicates and depth limit; equivalent to distance IS NOT NULL."
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
        (CugraphAlgorithm::Louvain, "community") => "Community identifier assigned by Louvain.",
        (CugraphAlgorithm::Leiden, "vertex") => "Vertex assigned to a Leiden community.",
        (CugraphAlgorithm::Leiden, "community") => {
            "Community identifier assigned by Leiden; every community induces a connected subgraph."
        },
        (CugraphAlgorithm::Ecg, "vertex") => "Vertex assigned to an ECG community.",
        (CugraphAlgorithm::Ecg, "community") => "Community identifier assigned by ECG.",
        (CugraphAlgorithm::SpectralModularityMaximization, "vertex") => {
            "Vertex assigned to a spectral clustering partition."
        },
        (CugraphAlgorithm::SpectralModularityMaximization, "community") => {
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
        (CugraphAlgorithm::WeaklyConnectedComponents, "component") => {
            "Weakly connected component identifier for the vertex: the minimum vertex id in the component, stable across runs."
        },
        (CugraphAlgorithm::StronglyConnectedComponents, "vertex") => {
            "Vertex assigned to a strongly connected component."
        },
        (CugraphAlgorithm::StronglyConnectedComponents, "component") => {
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
        (CugraphAlgorithm::EdgeBetweennessCentrality, "src") => "Source vertex of the scored edge.",
        (CugraphAlgorithm::EdgeBetweennessCentrality, "dst") => "Destination vertex of the scored edge.",
        (CugraphAlgorithm::EdgeBetweennessCentrality, "value") => "Edge betweenness centrality score for the edge.",
        (algorithm, "first") if is_similarity(algorithm) => "First vertex from the explicit candidate-pair relation.",
        (algorithm, "second") if is_similarity(algorithm) => "Second vertex from the explicit candidate-pair relation.",
        (algorithm, "similarity") if is_similarity(algorithm) => {
            "Similarity coefficient for the explicit candidate pair."
        },
        _ => "Algorithm result column.",
    }
}

/// Run-to-run behavior on the same input, options and GPU, for algorithms whose
/// results can differ between runs or are not fixed by `seed`. The algorithms
/// that repeat are listed in website/docs/design/gpu-table-functions.mdx.
fn reproducibility_limitation(algorithm: CugraphAlgorithm) -> Option<&'static str> {
    match algorithm {
        CugraphAlgorithm::ForceAtlas2 => Some(
            "seed sets only the initial positions; coordinates differ between runs because force accumulation uses floating-point atomics",
        ),
        CugraphAlgorithm::SpectralModularityMaximization => Some(
            "seed sets the Lanczos start vector and the k-means initialization; the same seed can still yield a different partition because k-means centroid sums use floating-point atomics",
        ),
        CugraphAlgorithm::Ecg => Some(
            "seed sets the random initial community labels of each Louvain pass; the same seed does not guarantee identical communities across runs",
        ),
        CugraphAlgorithm::Leiden => Some(
            "seed sets the random choices in the refinement step; the same seed does not guarantee identical communities across runs",
        ),
        CugraphAlgorithm::Bfs => Some(
            "distances and reachability repeat between runs; when a vertex has several shortest-path parents, the returned predecessor can differ",
        ),
        CugraphAlgorithm::Hits => Some(
            "scores can differ in the last digits between runs because the hub update accumulates with floating-point atomics",
        ),
        CugraphAlgorithm::PageRank | CugraphAlgorithm::PersonalizedPageRank => Some(
            "calls without an edge weight column repeat between runs; with one, scores can differ in the last digits because out-weight sums accumulate with floating-point atomics",
        ),
        _ => None,
    }
}

pub(super) fn limitations_json(algorithm: CugraphAlgorithm) -> String {
    let mut limitations = vec![
        "dry-run validates table resolution, column presence, static dtypes, and options only",
        "dry-run does not scan edge data, construct a graph, or prove source-vertex existence",
    ];
    if algorithm == CugraphAlgorithm::Bfs {
        limitations.extend([
            "source_vertices and the sources relation are multi-source BFS, not one BFS per source",
            "source_vertices and the sources relation must contain at most one source per connected component",
            "vertices output does not expose origin source per vertex",
            "sources have distance 0 and null predecessor; unreached graph vertices remain in vertices output with null distance and predecessor",
            "reachable describes arrival within the predicates and depth limit; vertices beyond that limit and disconnected vertices both return false",
            "string logical vertex-domain BFS rejects edges.edge_id and edge-id predicate relations",
            "target_vertices is required for path output and rejected for vertices output; join target relations to vertices output for multiple target distances",
            "path output requires exactly one target row at execution time",
            "vertices output searches until the depth limit or frontier exhaustion; path output stops after completing the target's BFS depth",
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
            "vertex_pairs is a required relation with canonical first and second columns",
            "candidate-pair columns must match the graph vertex domain; logical string graphs accept Utf8, LargeUtf8, or Utf8View independently per column",
            "null candidate-pair values are rejected at execution and are never dropped",
            "candidate pairs are a multiset: duplicate, reversed, and self pairs remain distinct output rows",
            "result rows have no global ordering; use ORDER BY when order is required",
            "providing edges.weight selects weighted similarity; omitting it selects unit-weight similarity",
        ]);
    }
    if algorithm == CugraphAlgorithm::Sssp {
        limitations.push("omitting edges.weight assigns unit cost 1 to every edge");
    }
    limitations.extend(reproducibility_limitation(algorithm));
    if matches!(
        algorithm.native_algorithm().vertex_id_type_support(),
        GraphVertexIdTypeSupport::Int32OnlyLegacyEdgeType
    ) {
        limitations.extend([
            "the native graph type indexes Int32 vertices only; Int64 and string vertex columns run through a per-call codebook that fails when the distinct vertex count exceeds Int32::MAX",
            "edge_id columns are rejected when Int64 vertices are mapped to Int32 ids",
        ]);
    }
    if algorithm.native_algorithm().definition().required_graph_directedness() == Some(false) {
        limitations.push("cuGraph requires directed=false so the graph is constructed as an undirected/symmetric view");
    }
    json!(limitations).to_string()
}

pub(super) fn summary(algorithm: CugraphAlgorithm) -> &'static str {
    match algorithm {
        CugraphAlgorithm::Bfs => {
            "Run unweighted BFS from one or more sources, returning nullable vertex distances and predecessors with reachable, or one source-to-target path."
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
            "Compute minimum path distances and predecessors from one source vertex, using edges.weight or unit cost 1 when that field is omitted."
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
            "Find modularity-based communities with a refinement step, and split each community into the connected components of its internal edges, so every returned community induces a connected subgraph."
        },
        CugraphAlgorithm::Ecg => {
            "Stabilize community assignments by combining an ensemble of randomized Louvain partitions into a final consensus clustering."
        },
        CugraphAlgorithm::SpectralModularityMaximization => {
            "Partition vertices by running k-means on an embedding of the normalized graph Laplacian eigenvectors with the smallest eigenvalues. Modularity is not optimized directly."
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
            "SELECT * FROM cugraph_bfs(edges => (SELECT src, dst FROM target_edges), source_vertex => 123, depth_limit => 4)".to_string()
        },
        CugraphAlgorithm::PersonalizedPageRank => {
            "SELECT * FROM cugraph_personalized_pagerank(edges => (SELECT src, dst FROM target_edges), personalization => (SELECT vertex, value FROM ppr_seeds))".to_string()
        },
        CugraphAlgorithm::Sssp => {
            format!("SELECT * FROM {function_name}(edges => (SELECT src, dst, weight FROM target_edges), source_vertex => 123)")
        },
        CugraphAlgorithm::MinimumSpanningTree => {
            format!("SELECT * FROM {function_name}(edges => (SELECT src, dst, weight FROM target_edges))")
        },
        algorithm if is_similarity(algorithm) => format!(
            "SELECT * FROM {function_name}(edges => (SELECT src, dst FROM target_edges), vertex_pairs => (SELECT first, second FROM candidate_pairs))"
        ),
        _ => format!("SELECT * FROM {function_name}(edges => (SELECT src, dst FROM target_edges))"),
    }
}

pub(super) fn examples_json(algorithm: CugraphAlgorithm) -> String {
    let examples = match algorithm {
        CugraphAlgorithm::Bfs => bfs_examples(algorithm),
        CugraphAlgorithm::PersonalizedPageRank => {
            vec![common_example(algorithm, "basic_personalized_pagerank", json!({}))]
        },
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
    let mut relations = serde_json::Map::from_iter([("edges".to_string(), json!({"table": "target_edges"}))]);
    let options = options.as_object().cloned().expect("cuGraph examples use an object of named values");
    if algorithm == CugraphAlgorithm::PersonalizedPageRank {
        relations.insert("personalization".to_string(), json!({"table": "ppr_seeds"}));
    }
    if is_similarity(algorithm) {
        relations.insert("vertex_pairs".to_string(), json!({"table": "candidate_pairs"}));
    }
    json!({
        "name": name.into(),
        "function_name": algorithm.function_name(),
        "call_json": { "relations": relations, "options": options },
        "execution_sql": primary_example_sql(algorithm),
    })
}

fn similarity_example(algorithm: CugraphAlgorithm) -> Value {
    common_example(algorithm, format!("basic_{}", algorithm.native_algorithm().definition().name), json!({}))
}

fn minimal_example(algorithm: CugraphAlgorithm) -> Value {
    common_example(algorithm, format!("basic_{}", algorithm.native_algorithm().definition().name), json!({}))
}

fn source_vertex_example(algorithm: CugraphAlgorithm, name: &str) -> Value {
    common_example(algorithm, name, json!({"source_vertex": 123}))
}

fn weighted_example(algorithm: CugraphAlgorithm, name: &str) -> Value {
    common_example(algorithm, name, json!({}))
}

fn weighted_source_vertex_example(algorithm: CugraphAlgorithm, name: &str) -> Value {
    common_example(algorithm, name, json!({"source_vertex": 123}))
}

fn bfs_examples(_algorithm: CugraphAlgorithm) -> Vec<Value> {
    vec![
        bfs_example(
            "basic_bfs",
            json!({
                "source_vertex": 123,
                "depth_limit": 4
            }),
            &[],
            "SELECT * FROM cugraph_bfs(edges => (SELECT src, dst FROM target_edges), source_vertex => 123, depth_limit => 4)",
        ),
        bfs_example(
            "vertices_output",
            json!({
                "source_vertex": 123,
                "depth_limit": 4,
                "output_mode": "vertices"
            }),
            &[],
            "SELECT * FROM cugraph_bfs(edges => (SELECT src, dst FROM target_edges), source_vertex => 123, depth_limit => 4, output_mode => 'vertices')",
        ),
        bfs_example(
            "multi_source_bfs_list",
            json!({
                "source_vertices": [123, 456]
            }),
            &[],
            "SELECT * FROM cugraph_bfs(edges => (SELECT src, dst FROM target_edges), source_vertices => [123, 456])",
        ),
        bfs_example(
            "multi_source_bfs_table",
            json!({}),
            &[("sources", "bfs_sources")],
            "SELECT * FROM cugraph_bfs(edges => (SELECT src, dst FROM target_edges), sources => (SELECT vertex_id AS vertex FROM bfs_sources))",
        ),
        bfs_example(
            "target_distances",
            json!({
                "source_vertex": 123,
                "depth_limit": 4
            }),
            &[],
            "WITH bfs AS (SELECT vertex, distance FROM cugraph_bfs(edges => (SELECT src, dst FROM target_edges), source_vertex => 123, depth_limit => 4)) SELECT t.vertex_id AS target, b.distance, b.distance IS NOT NULL AS reachable FROM targets t LEFT JOIN bfs b ON b.vertex = t.vertex_id",
        ),
        bfs_example(
            "path_output",
            json!({
                "source_vertex": 123,
                "output_mode": "path"
            }),
            &[("target_vertices", "targets")],
            "SELECT * FROM cugraph_bfs(edges => (SELECT src, dst FROM target_edges), source_vertex => 123, target_vertices => (SELECT vertex_id AS vertex FROM targets), output_mode => 'path')",
        ),
    ]
}

fn bfs_example(name: &str, options: Value, extra_relations: &[(&str, &str)], execution_sql: &str) -> Value {
    let mut relations = serde_json::Map::from_iter([("edges".to_string(), json!({"table": "target_edges"}))]);
    for (role, table) in extra_relations {
        relations.insert((*role).to_string(), json!({"table": table}));
    }
    json!({
        "name": name,
        "function_name": CugraphAlgorithm::Bfs.function_name(),
        "call_json": {"relations": relations, "options": options},
        "execution_sql": execution_sql,
    })
}
