use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use cudf_query_engine::{
    GRAPH_ALGORITHM_STRICT_BOUNDED_LOWER_LEVEL_CONTRACTS, GRAPH_ALGORITHM_STRICT_BOUNDED_MEMORY_BOUNDARY,
    GRAPH_ALGORITHM_STRICT_BOUNDED_REQUIRED_CONTRACT, GpuResidentSource, GraphAlgorithm, GraphAlgorithmParams,
    GraphAlgorithmSpec, GraphBfsParams, GraphConstructionOptions, GraphConstructionPolicy, GraphEdgeColumnBindings,
    GraphPageRankParams, PlanNode, QueryPlan, QueryPlanBuilder, SourceNode, StrictBoundedContractReport,
    StrictBoundedExecutionMode,
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
    assert_eq!(edge_normalize.spec.dedup.as_str(), "keep_first");
    assert_eq!(edge_normalize.spec.weight_combiner.as_str(), "min_weight");

    let display = plan.to_string();
    assert!(display.contains("EdgeNormalize"));
    assert!(display.contains("dedup=keep_first"));
    assert!(display.contains("weight_combiner=min_weight"));
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
fn graph_algorithm_capability_blocks_strict_bounded_on_lower_level_contract() {
    let plan = graph_plan(
        GraphAlgorithm::Bfs,
        GraphAlgorithmParams::Bfs(GraphBfsParams { source_vertex: 0, depth_limit: Some(3) }),
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
