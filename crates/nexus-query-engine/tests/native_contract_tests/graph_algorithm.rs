use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use nexus_query_engine::{
    GRAPH_ALGORITHM_STRICT_BOUNDED_LOWER_LEVEL_CONTRACTS, GRAPH_ALGORITHM_STRICT_BOUNDED_MEMORY_BOUNDARY,
    GRAPH_ALGORITHM_STRICT_BOUNDED_REQUIRED_CONTRACT, GRAPH_OPERATION_REGISTRY, GpuResidentSource, GraphAlgorithm,
    GraphAlgorithmParams, GraphAlgorithmSpec, GraphBfsParams, GraphBfsPredicateInputs, GraphConstructionOptions,
    GraphConstructionPolicy, GraphEdgeColumnBindings, GraphLayoutRequirement, GraphPageRankParams, GraphPredicateInput,
    PlanNode, QueryPlan, QueryPlanBuilder, SourceNode, StrictBoundedContractReport, StrictBoundedExecutionMode,
    capability::{CapabilityDecision, ContractQuality, ContractStatus, NodeOperator},
    stable_graph_result_schema,
};

#[test]
fn graph_algorithm_builder_records_ir_contracts_and_display_summary() {
    let plan = graph_plan(
        GraphAlgorithm::PageRank,
        GraphAlgorithmParams::PageRank(GraphPageRankParams { alpha: 0.7, epsilon: 0.001, max_iterations: 25 }),
        edge_schema(DataType::Int64, DataType::Float64, false),
    );
    assert!(plan.contains_edge_normalize());

    let graph_node = plan.root_node();
    let PlanNode::GraphAlgorithm(graph) = &graph_node.kind else {
        panic!("expected graph algorithm root");
    };

    assert_eq!(graph.input.index(), 1);
    assert_eq!(graph.spec.algorithm, GraphAlgorithm::PageRank);
    assert_eq!(graph.spec.edge_columns.source, "src");
    assert_eq!(graph.spec.edge_columns.destination, "dst");
    assert_eq!(graph.spec.edge_columns.weight.as_deref(), Some("weight"));
    assert!(graph.spec.graph_options.directed);
    assert!(graph.spec.graph_options.renumber);
    assert_eq!(graph.spec.graph_options.policy, GraphConstructionPolicy::RawLibcugraph);
    assert_eq!(graph.output_schema().as_ref(), stable_graph_result_schema(GraphAlgorithm::PageRank).as_ref());

    let normalize_node = plan.node(graph.input).expect("graph input must exist");
    let PlanNode::EdgeNormalize(edge_normalize) = &normalize_node.kind else {
        panic!("expected edge normalize input");
    };
    assert_eq!(edge_normalize.input.index(), 0);
    assert!(!edge_normalize.spec.add_reverse);
    assert_eq!(edge_normalize.spec.dedup.as_str(), "keep_last");
    assert_eq!(edge_normalize.spec.weight_combiner.as_str(), "keep_last_input_order");

    let display = plan.to_string();
    assert!(display.contains("EdgeNormalize"));
    assert!(display.contains("dedup=keep_last"));
    assert!(display.contains("weight_combiner=keep_last_input_order"));
    assert!(display.contains("GraphAlgorithm"));
    assert!(display.contains("algorithm=pagerank"));
    assert!(display.contains("src_col=src"));
    assert!(display.contains("dst_col=dst"));
    assert!(display.contains("weight_col=weight"));
    assert!(display.contains("directed=true"));
    assert!(display.contains("renumber=true"));
    assert!(display.contains("construction_policy=raw_libcugraph"));
    assert!(display.contains("output_schema_fields=2"));
}

#[test]
fn undirected_weighted_python_graph_construction_uses_min_weight_normalization() {
    let mut builder = QueryPlanBuilder::new();
    let input = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
        "edges",
        edge_schema(DataType::Int64, DataType::Float64, false),
    )));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Louvain,
        GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
        GraphConstructionOptions { directed: false, ..Default::default() },
        GraphAlgorithmParams::Louvain(Default::default()),
    );
    let graph = builder.add_graph_algorithm(input, spec);
    let plan = builder.finish_with_host_sink(graph);

    let PlanNode::GraphAlgorithm(graph) = &plan.root_node().kind else {
        panic!("expected graph algorithm root");
    };
    let normalize_node = plan.node(graph.input).expect("graph input must exist");
    let PlanNode::EdgeNormalize(edge_normalize) = &normalize_node.kind else {
        panic!("expected edge normalize input");
    };
    assert!(edge_normalize.spec.add_reverse);
    assert_eq!(edge_normalize.spec.dedup.as_str(), "keep_first");
    assert_eq!(edge_normalize.spec.weight_combiner.as_str(), "min_weight");
}

#[test]
fn raw_graph_construction_policy_does_not_insert_edge_normalize() {
    let mut builder = QueryPlanBuilder::new();
    let input = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
        "edges",
        edge_schema(DataType::Int64, DataType::Float64, false),
    )));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::PageRank,
        GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
        GraphConstructionOptions { policy: GraphConstructionPolicy::RawLibcugraph, ..Default::default() },
        GraphAlgorithmParams::PageRank(GraphPageRankParams::default()),
    );
    let graph = builder.add_graph_algorithm(input, spec);
    let plan = builder.finish_with_host_sink(graph);

    assert!(!plan.contains_edge_normalize());
    assert_eq!(plan.node_count(), 2);
    let PlanNode::GraphAlgorithm(graph) = &plan.root_node().kind else {
        panic!("expected graph algorithm root");
    };
    assert_eq!(graph.input.index(), 0);
    assert_eq!(graph.spec.graph_options.policy, GraphConstructionPolicy::RawLibcugraph);
    assert!(!plan.to_string().contains("EdgeNormalize"));
}

#[test]
fn bfs_predicate_side_inputs_are_explicit_native_ir_edges() {
    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
        "edges",
        edge_schema(DataType::Int64, DataType::Float64, false),
    )));
    let targets = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
        "targets",
        Arc::new(Schema::new(vec![Field::new("vertex_id", DataType::Int64, false)])),
    )));
    let params = GraphBfsParams {
        source_vertex: 0,
        source_vertices: None,
        depth_limit: Some(4),
        predicates: GraphBfsPredicateInputs {
            target_vertices: Some(GraphPredicateInput::new(targets, "vertex_id")),
            ..Default::default()
        },
        return_target_info: false,
    };
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
        GraphConstructionOptions::default(),
        GraphAlgorithmParams::Bfs(Box::new(params)),
    );
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan = builder.finish_with_host_sink(graph);

    let PlanNode::GraphAlgorithm(graph) = &plan.root_node().kind else {
        panic!("expected graph algorithm root");
    };
    assert_eq!(graph.inputs, vec![graph.input, targets]);
    assert_eq!(plan.root_node().input_ids(), &[graph.input, targets]);

    let display = plan.to_string();
    assert!(display.contains("side_inputs=n1"));
    assert!(display.contains("predicates=target_vertices:n1:vertex_id"));
}

#[test]
fn bfs_source_vertex_side_input_precedes_predicate_side_inputs() {
    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
        "edges",
        edge_schema(DataType::Int32, DataType::Float32, false),
    )));
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
        "source",
        Arc::new(Schema::new(vec![Field::new("source_vertex", DataType::Int32, false)])),
    )));
    let targets = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
        "targets",
        Arc::new(Schema::new(vec![Field::new("vertex_id", DataType::Int32, false)])),
    )));
    let mut params = GraphBfsParams::new(0, None);
    params.source_vertices = Some(GraphPredicateInput::new(source, "source_vertex"));
    params.predicates.target_vertices = Some(GraphPredicateInput::new(targets, "vertex_id"));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
        GraphConstructionOptions::default(),
        GraphAlgorithmParams::Bfs(Box::new(params)),
    );
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan = builder.finish_with_host_sink(graph);

    let PlanNode::GraphAlgorithm(graph) = &plan.root_node().kind else {
        panic!("expected graph algorithm root");
    };
    assert_eq!(graph.inputs, vec![graph.input, source, targets]);
    assert_eq!(plan.root_node().input_ids(), &[graph.input, source, targets]);

    let display = plan.to_string();
    assert!(display.contains("side_inputs=n1,n2"));
    assert!(display.contains("params=source_vertex=0:source_vertices=n1:source_vertex:depth_limit=none"));
    assert!(display.contains("predicates=target_vertices:n2:vertex_id"));
}

#[test]
fn bfs_edge_id_predicates_require_primary_edge_id_binding() {
    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
        "edges",
        edge_schema(DataType::Int32, DataType::Float32, false),
    )));
    let allowed = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
        "allowed",
        Arc::new(Schema::new(vec![Field::new("edge_id", DataType::Int32, false)])),
    )));
    let params = GraphBfsParams {
        source_vertex: 0,
        source_vertices: None,
        depth_limit: None,
        predicates: GraphBfsPredicateInputs {
            include_edge_ids: Some(GraphPredicateInput::new(allowed, "edge_id")),
            ..Default::default()
        },
        return_target_info: false,
    };
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
        GraphConstructionOptions { policy: GraphConstructionPolicy::RawLibcugraph, ..Default::default() },
        GraphAlgorithmParams::Bfs(Box::new(params)),
    );
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan = builder.finish_with_host_sink(graph);

    let capability = plan.capability_analysis().nodes.into_iter().find(|node| node.node_id == plan.root_id()).unwrap();
    let CapabilityDecision::Rejected { facts, .. } = capability.executable else {
        panic!("expected graph capability rejection");
    };
    assert_eq!(facts.get("missing_binding").map(String::as_str), Some("edge_id_col"));
}

#[test]
fn edge_id_binding_is_preserved_through_python_edge_normalize() {
    let mut builder = QueryPlanBuilder::new();
    let input = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
        "edges",
        edge_schema_with_edge_id(DataType::Int32, DataType::Float32),
    )));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", Some("weight")).with_edge_id("edge_id"),
        GraphConstructionOptions::default(),
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
    );
    let graph = builder.add_graph_algorithm(input, spec);
    let plan = builder.finish_with_host_sink(graph);

    let PlanNode::GraphAlgorithm(graph) = &plan.root_node().kind else {
        panic!("expected graph algorithm root");
    };
    assert_eq!(graph.spec.edge_columns.edge_id.as_deref(), Some("edge_id"));
    let normalize_node = plan.node(graph.input).expect("graph input must exist");
    let PlanNode::EdgeNormalize(edge_normalize) = &normalize_node.kind else {
        panic!("expected edge normalize input");
    };
    assert_eq!(edge_normalize.spec.edge_columns.edge_id.as_deref(), Some("edge_id"));
    assert_eq!(
        normalize_node.output_schema().fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>(),
        vec!["src", "dst", "weight", "edge_id"]
    );

    let display = plan.to_string();
    assert!(display.contains("edge_id_col=edge_id"));
}

#[test]
fn graph_algorithm_capability_blocks_strict_bounded_on_lower_level_contract() {
    let plan = graph_plan(
        GraphAlgorithm::Bfs,
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, Some(3)))),
        edge_schema(DataType::Int32, DataType::Float32, false),
    );

    let analysis = plan.capability_analysis();
    let graph_capability = analysis.nodes.iter().find(|node| node.node_id == plan.root_id()).unwrap();
    assert_eq!(graph_capability.operator, NodeOperator::GraphAlgorithm { algorithm: GraphAlgorithm::Bfs });
    assert_eq!(graph_capability.executable, CapabilityDecision::Satisfied);
    assert_eq!(graph_capability.local_requirements.len(), 1);
    let requirement = &graph_capability.local_requirements[0];
    assert_eq!(requirement.memory_boundary, GRAPH_ALGORITHM_STRICT_BOUNDED_MEMORY_BOUNDARY);
    assert_eq!(requirement.required_contract, GRAPH_ALGORITHM_STRICT_BOUNDED_REQUIRED_CONTRACT);
    assert_eq!(requirement.status, ContractStatus::BlockedByLowerLevel);
    assert_eq!(requirement.quality, ContractQuality::ExactPreflight);

    let report = StrictBoundedContractReport::from_capability_analysis(&analysis).unwrap();
    assert_eq!(report.execution_mode(), StrictBoundedExecutionMode::UnsupportedStrictBounded);
    assert!(report.gate_rows().iter().any(|row| {
        row.memory_boundary == GRAPH_ALGORITHM_STRICT_BOUNDED_MEMORY_BOUNDARY
            && row.required_contract == GRAPH_ALGORITHM_STRICT_BOUNDED_REQUIRED_CONTRACT
            && row.contract_status == ContractStatus::BlockedByLowerLevel
    }));
}

#[test]
fn graph_algorithm_strict_bounded_contract_surface_lists_lower_level_gaps() {
    assert_eq!(GRAPH_ALGORITHM_STRICT_BOUNDED_MEMORY_BOUNDARY, "graph_algorithm");
    assert_eq!(GRAPH_ALGORITHM_STRICT_BOUNDED_REQUIRED_CONTRACT, "unsupported_graph_algorithm_reservation_contract");

    let gaps: Vec<_> =
        GRAPH_ALGORITHM_STRICT_BOUNDED_LOWER_LEVEL_CONTRACTS.iter().map(|contract| contract.name).collect();
    assert_eq!(
        gaps,
        vec![
            "graph_construction_memory",
            "renumbering_memory",
            "algorithm_workspace_memory",
            "result_table_memory",
            "retained_resource_lifetime_boundaries",
        ]
    );
    assert!(
        GRAPH_ALGORITHM_STRICT_BOUNDED_LOWER_LEVEL_CONTRACTS
            .iter()
            .all(|contract| !contract.required_surface.contains("estimate")),
        "graph strict bounded support must wait for lower-level contract surfaces, not adapter estimates"
    );
}

#[test]
fn graph_operation_registry_pins_baseline_inventory() {
    let inventory: Vec<_> =
        GRAPH_OPERATION_REGISTRY.iter().map(|def| (def.algorithm, def.name, def.layout_requirement)).collect();
    assert_eq!(
        inventory,
        vec![
            (GraphAlgorithm::PageRank, "pagerank", GraphLayoutRequirement::Transposed),
            (GraphAlgorithm::Bfs, "bfs", GraphLayoutRequirement::Forward),
            (GraphAlgorithm::Louvain, "louvain", GraphLayoutRequirement::Forward),
            (GraphAlgorithm::EigenvectorCentrality, "eigenvector_centrality", GraphLayoutRequirement::Transposed),
            (GraphAlgorithm::KatzCentrality, "katz_centrality", GraphLayoutRequirement::Transposed),
            (GraphAlgorithm::Hits, "hits", GraphLayoutRequirement::Transposed),
            (GraphAlgorithm::Sssp, "sssp", GraphLayoutRequirement::Forward),
            (GraphAlgorithm::DegreesAll, "degrees_all", GraphLayoutRequirement::Transposed),
            (GraphAlgorithm::InDegreesAll, "in_degrees_all", GraphLayoutRequirement::Transposed),
            (GraphAlgorithm::OutDegreesAll, "out_degrees_all", GraphLayoutRequirement::Any),
            (GraphAlgorithm::WeaklyConnectedComponents, "weakly_connected_components", GraphLayoutRequirement::Any),
            (GraphAlgorithm::StronglyConnectedComponents, "strongly_connected_components", GraphLayoutRequirement::Any),
            (GraphAlgorithm::TriangleCountAll, "triangle_count_all", GraphLayoutRequirement::Any),
            (GraphAlgorithm::CoreNumber, "core_number", GraphLayoutRequirement::Any),
            (GraphAlgorithm::KCore, "k_core", GraphLayoutRequirement::Any),
            (GraphAlgorithm::MinimumSpanningTree, "minimum_spanning_tree", GraphLayoutRequirement::Any),
            (GraphAlgorithm::PageRankStrict, "pagerank_strict", GraphLayoutRequirement::Transposed),
            (GraphAlgorithm::Leiden, "leiden", GraphLayoutRequirement::Forward),
            (GraphAlgorithm::Ecg, "ecg", GraphLayoutRequirement::Forward),
            (
                GraphAlgorithm::SpectralModularityMaximization,
                "spectral_modularity_maximization",
                GraphLayoutRequirement::Forward,
            ),
            (GraphAlgorithm::BalancedCutClustering, "balanced_cut_clustering", GraphLayoutRequirement::Forward),
        ],
        "graph operation registry order is part of the public surface; \
         adding an algorithm requires extending this baseline list",
    );

    for def in GRAPH_OPERATION_REGISTRY {
        assert_eq!(def.name, def.algorithm.as_str(), "registry name and GraphAlgorithm::as_str must agree");
        assert_eq!(def, def.algorithm.definition(), "GraphAlgorithm::definition must round-trip through the registry");
        assert_eq!(
            def.output_schema().as_ref(),
            stable_graph_result_schema(def.algorithm).as_ref(),
            "registry output_schema and stable_graph_result_schema must agree",
        );
    }

    let mut names = GRAPH_OPERATION_REGISTRY.iter().map(|def| def.name).collect::<Vec<_>>();
    names.sort_unstable();
    let unique_count = names.iter().collect::<std::collections::BTreeSet<_>>().len();
    assert_eq!(unique_count, names.len(), "registry algorithm names must be unique");

    let mut algorithms = GRAPH_OPERATION_REGISTRY.iter().map(|def| def.algorithm).collect::<Vec<_>>();
    algorithms.sort_unstable();
    let unique_algorithms = algorithms.iter().collect::<std::collections::BTreeSet<_>>().len();
    assert_eq!(unique_algorithms, algorithms.len(), "registry algorithm variants must be unique");

    assert_eq!(
        GraphAlgorithm::all().collect::<Vec<_>>(),
        GRAPH_OPERATION_REGISTRY.iter().map(|def| def.algorithm).collect::<Vec<_>>(),
        "GraphAlgorithm::all must iterate the registry order",
    );
}

#[test]
fn graph_layout_requirement_label_round_trips() {
    assert_eq!(GraphLayoutRequirement::Forward.as_str(), "forward");
    assert_eq!(GraphLayoutRequirement::Transposed.as_str(), "transposed");
    assert_eq!(GraphLayoutRequirement::Any.as_str(), "any");
}

#[test]
fn graph_construction_policy_label_round_trips() {
    assert_eq!(GraphConstructionPolicy::PythonCugraph.as_str(), "python_cugraph");
    assert_eq!(GraphConstructionPolicy::RawLibcugraph.as_str(), "raw_libcugraph");
}

#[test]
fn graph_algorithm_capability_validates_edge_column_types_and_nullability() {
    let valid = graph_plan(
        GraphAlgorithm::Louvain,
        GraphAlgorithmParams::Louvain(Default::default()),
        edge_schema(DataType::Int32, DataType::Float32, false),
    );
    let valid_capability =
        valid.capability_analysis().nodes.into_iter().find(|node| node.node_id == valid.root_id()).unwrap();
    assert_eq!(valid_capability.executable, CapabilityDecision::Satisfied);

    let invalid_vertex = graph_plan(
        GraphAlgorithm::Louvain,
        GraphAlgorithmParams::Louvain(Default::default()),
        edge_schema(DataType::Utf8, DataType::Float32, false),
    );
    assert_graph_rejection_contains(&invalid_vertex, "unsupported_vertex_type", "source");

    let nullable_weight = graph_plan(
        GraphAlgorithm::Louvain,
        GraphAlgorithmParams::Louvain(Default::default()),
        edge_schema(DataType::Int64, DataType::Float64, true),
    );
    assert_graph_rejection_contains(&nullable_weight, "nullable_edge_column", "weight");
}

fn assert_graph_rejection_contains(plan: &QueryPlan, reason: &str, role: &str) {
    let capability = plan.capability_analysis().nodes.into_iter().find(|node| node.node_id == plan.root_id()).unwrap();
    let CapabilityDecision::Rejected { facts, .. } = capability.executable else {
        panic!("expected graph capability rejection");
    };
    assert_eq!(facts.get("reason").map(String::as_str), Some(reason));
    assert_eq!(facts.get("edge_column_role").map(String::as_str), Some(role));
}

fn graph_plan(algorithm: GraphAlgorithm, params: GraphAlgorithmParams, input_schema: Arc<Schema>) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let input = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", input_schema)));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        algorithm,
        GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
        GraphConstructionOptions::default(),
        params,
    );
    let graph = builder.add_graph_algorithm(input, spec);
    builder.finish_with_host_sink(graph)
}

fn edge_schema(vertex_type: DataType, weight_type: DataType, weight_nullable: bool) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("src", vertex_type.clone(), false),
        Field::new("dst", vertex_type, false),
        Field::new("weight", weight_type, weight_nullable),
    ]))
}

fn edge_schema_with_edge_id(vertex_type: DataType, weight_type: DataType) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("src", vertex_type.clone(), false),
        Field::new("dst", vertex_type.clone(), false),
        Field::new("weight", weight_type, false),
        Field::new("edge_id", vertex_type, false),
    ]))
}

// ═══════════════════════════════════════════════════════════════════════════
// cugraph-rs inventory drift gate
// ═══════════════════════════════════════════════════════════════════════════

/// Mapping from cugraph-rs `Graph` method base names to the corresponding
/// `GraphAlgorithm::as_str()` registry name.
///
/// Only primary methods are listed here. Variant suffixes (`_with`,
/// `_strict`, `_strict_with`, `_checked`, `_all_checked`, `_all`) are
/// handled by separate registry entries or by Milestone 2/3 auxiliary-input
/// flow work; they are NOT separate primary methods for the purpose of this
/// drift test.
const CUGRAPH_RS_METHOD_TO_REGISTRY: &[(&str, &str)] = &[
    ("pagerank", "pagerank"),
    ("bfs", "bfs"),
    ("louvain", "louvain"),
    ("eigenvector_centrality", "eigenvector_centrality"),
    ("katz_centrality", "katz_centrality"),
    ("hits", "hits"),
    ("sssp", "sssp"),
    ("degrees_all", "degrees_all"),
    ("in_degrees_all", "in_degrees_all"),
    ("out_degrees_all", "out_degrees_all"),
    ("weakly_connected_components", "weakly_connected_components"),
    ("strongly_connected_components", "strongly_connected_components"),
    ("triangle_count_all", "triangle_count_all"),
    ("core_number", "core_number"),
    ("k_core", "k_core"),
    ("minimum_spanning_tree", "minimum_spanning_tree"),
    ("pagerank_strict", "pagerank_strict"),
    ("leiden", "leiden"),
    ("ecg", "ecg"),
    ("spectral_modularity_maximization", "spectral_modularity_maximization"),
    ("balanced_cut_clustering", "balanced_cut_clustering"),
];

/// cugraph-rs `Graph` method names that are intentionally absent from the
/// registry today, pinned by stage.
///
/// **Milestone 2** (need auxiliary inputs):
///   personalized_pagerank, hits_with_initial_hubs,
///   betweenness_centrality(_all), edge_betweenness_centrality(_all),
///   vertex_pairs, jaccard/sorensen/overlap/cosine (pair+all_pairs),
///   k_truss_subgraph, extract_ego, k_core_with_core_numbers,
///   analyze_clustering_*, degrees/in_degrees/out_degrees (with vertex selection),
///   select_random_vertices
///
/// **Milestone 3** (stochastic / sampling / layout):
///   uniform_random_walks, biased_random_walks, node2vec_random_walks,
///   homogeneous_*_neighbor_sample, homogeneous_*_temporal_neighbor_sample,
///   force_atlas2
///
/// When a new method is added to cugraph-rs and it is not yet implemented,
/// add it here with a comment indicating the planned stage. When it is
/// implemented, move it to `CUGRAPH_RS_METHOD_TO_REGISTRY` above and add
/// the registry entry.
const KNOWN_MISSING_CUGRAPH_RS_METHODS: &[&str] = &[
    // Milestone 2 — auxiliary inputs or variant forms of registered algorithms
    "pagerank_with",
    "pagerank_strict_with",
    "personalized_pagerank",
    "personalized_pagerank_with",
    "personalized_pagerank_strict",
    "personalized_pagerank_strict_with",
    "hits_with_initial_hubs",
    "degrees",
    "degrees_checked",
    "in_degrees",
    "in_degrees_checked",
    "out_degrees",
    "out_degrees_checked",
    "betweenness_centrality",
    "betweenness_centrality_all",
    "edge_betweenness_centrality",
    "edge_betweenness_centrality_all",
    "vertex_pairs",
    "jaccard_coefficients",
    "sorensen_coefficients",
    "overlap_coefficients",
    "cosine_similarity_coefficients",
    "all_pairs_jaccard_coefficients",
    "all_pairs_sorensen_coefficients",
    "all_pairs_overlap_coefficients",
    "all_pairs_cosine_similarity_coefficients",
    "k_truss_subgraph",
    "extract_ego",
    "k_core_with_core_numbers",
    "analyze_clustering_modularity",
    "analyze_clustering_edge_cut",
    "analyze_clustering_ratio_cut",
    "select_random_vertices",
    // Milestone 2 — vertex-selection variant of triangle_count_all
    "triangle_count",
    // Milestone 3 — stochastic / sampling / layout
    "uniform_random_walks",
    "biased_random_walks",
    "node2vec_random_walks",
    "homogeneous_uniform_neighbor_sample",
    "homogeneous_biased_neighbor_sample",
    "homogeneous_uniform_temporal_neighbor_sample",
    "homogeneous_biased_temporal_neighbor_sample",
    "force_atlas2",
    "force_atlas2_with",
];

/// Methods on `Graph<T>` that are deliberately excluded from the drift
/// check: generic graph queries, utility accessors, and multi-step
/// workflow helpers that are not standalone algorithm entry points.
const CUGRAPH_RS_NON_ALGORITHM_METHODS: &[&str] = &[
    // graph construction and property accessors
    "vertex_dtype",
    "weight_dtype",
    "properties",
    "edge_count",
    "edge_count_checked",
    "has_edge_ids",
    "has_weights",
    // registered algorithm variants represented by auxiliary-input IR, not separate registry entries
    "bfs_with_predicates",
    // vertex-selection entry points that are Milestone 2 variants of registered algorithms
    // (already listed in KNOWN_MISSING) - excluded here so they don't appear as unknown
];

#[test]
fn graph_operation_registry_covers_cugraph_rs_surface_without_unknown_gaps() {
    // Resolve the cugraph-rs graph.rs path relative to this crate's manifest
    // directory. `CARGO_MANIFEST_DIR` is set by cargo at compile time to the
    // directory containing Cargo.toml for nexus-query-engine.
    //
    // Path:  crates/nexus-query-engine/  →  ../../  →  workspace root
    //        workspace root             →  ../     →  ~/code/
    //        ~/code/                    →  cugraph-rs/cugraph-rs/src/graph.rs
    let graph_rs_path =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../cugraph-rs/cugraph-rs/src/graph.rs");

    // Skip if the sibling checkout is absent (CI or non-standard workspace).
    if !graph_rs_path.exists() {
        eprintln!("skipping cugraph-rs drift test: {} not found", graph_rs_path.display());
        return;
    }

    let source = std::fs::read_to_string(&graph_rs_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", graph_rs_path.display()));

    // Extract public method names from `impl Graph<*>` blocks only.
    //
    // Strategy: scan line-by-line with a simple state bit.  A line that
    // starts with `impl` and contains `Graph<` (but NOT `for Graph<` or
    // `GraphBuilder`) sets the bit; any other `impl` line clears it.
    // `    pub fn name(` lines inside an active block are collected.
    let mut in_graph_impl = false;
    let mut extracted = std::collections::BTreeSet::<String>::new();
    for line in source.lines() {
        if line.starts_with("impl") {
            // New impl block — decide if it belongs to Graph<T>.
            in_graph_impl = line.contains("Graph<") && !line.contains("GraphBuilder") && !line.contains(" for Graph<");
            continue;
        }
        if !in_graph_impl {
            continue;
        }
        // Extract `    pub fn <name>(` inside an active Graph impl block.
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("pub fn ")
            && let Some(name_end) = rest.find(|c: char| !c.is_alphanumeric() && c != '_')
        {
            extracted.insert(rest[..name_end].to_string());
        }
    }

    // Build the universe of known names: registered + known_missing + non_algorithm.
    let registered_methods: std::collections::BTreeSet<&str> =
        CUGRAPH_RS_METHOD_TO_REGISTRY.iter().map(|(method, _)| *method).collect();
    let known_missing: std::collections::BTreeSet<&str> = KNOWN_MISSING_CUGRAPH_RS_METHODS.iter().copied().collect();
    let non_algorithm: std::collections::BTreeSet<&str> = CUGRAPH_RS_NON_ALGORITHM_METHODS.iter().copied().collect();

    // Every name in the registered mapping must actually appear in graph.rs.
    // Failing this means cugraph-rs renamed or removed an algorithm.
    let mut stale_registry_entries: Vec<&str> = Vec::new();
    for (method, _registry_name) in CUGRAPH_RS_METHOD_TO_REGISTRY {
        if !extracted.contains(*method) {
            stale_registry_entries.push(method);
        }
    }
    assert!(
        stale_registry_entries.is_empty(),
        "CUGRAPH_RS_METHOD_TO_REGISTRY contains methods not found in graph.rs \
         (cugraph-rs may have renamed or removed them): {stale_registry_entries:?}",
    );

    // Every name in KNOWN_MISSING must appear in graph.rs.
    let mut stale_known_missing: Vec<&str> = Vec::new();
    for method in KNOWN_MISSING_CUGRAPH_RS_METHODS {
        if !extracted.contains(*method) {
            stale_known_missing.push(method);
        }
    }
    assert!(
        stale_known_missing.is_empty(),
        "KNOWN_MISSING_CUGRAPH_RS_METHODS contains methods not found in graph.rs \
         (cugraph-rs may have renamed or removed them): {stale_known_missing:?}",
    );

    // Every public Graph method in graph.rs must be accounted for in one of
    // the three lists.  An entry here that appears in none of them means
    // cugraph-rs added a new method that datafusion-nexus has not yet planned
    // for.  Add it to KNOWN_MISSING_CUGRAPH_RS_METHODS (with a stage comment)
    // or implement it in the registry.
    let mut unknown: Vec<&str> = Vec::new();
    for method in &extracted {
        let s = method.as_str();
        if !registered_methods.contains(s) && !known_missing.contains(s) && !non_algorithm.contains(s) {
            unknown.push(s);
        }
    }
    // Print extracted set for diagnosis when the assertion fires.
    if !unknown.is_empty() {
        eprintln!("All extracted Graph method names from graph.rs ({} total):", extracted.len());
        for name in &extracted {
            eprintln!("  {name}");
        }
    }
    assert!(
        unknown.is_empty(),
        "cugraph-rs graph.rs contains public Graph methods not accounted for in \
         CUGRAPH_RS_METHOD_TO_REGISTRY, KNOWN_MISSING_CUGRAPH_RS_METHODS, or \
         CUGRAPH_RS_NON_ALGORITHM_METHODS.\n\
         Add each new method to KNOWN_MISSING with a stage comment, or implement it.\n\
         Unknown: {unknown:?}",
    );

    // Verify the registry-side mapping is consistent: every registry name
    // listed in the mapping must exist in GRAPH_OPERATION_REGISTRY.
    let registry_names: std::collections::BTreeSet<&str> =
        GRAPH_OPERATION_REGISTRY.iter().map(|def| def.name).collect();
    let mut unmapped: Vec<(&str, &str)> = Vec::new();
    for (method, registry_name) in CUGRAPH_RS_METHOD_TO_REGISTRY {
        if !registry_names.contains(registry_name) {
            unmapped.push((method, registry_name));
        }
    }
    assert!(
        unmapped.is_empty(),
        "CUGRAPH_RS_METHOD_TO_REGISTRY maps to registry names not present in \
         GRAPH_OPERATION_REGISTRY: {unmapped:?}",
    );
}
