use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use nexus_query_engine::{
    GRAPH_ALGORITHM_STRICT_BOUNDED_LOWER_LEVEL_CONTRACTS, GRAPH_ALGORITHM_STRICT_BOUNDED_MEMORY_BOUNDARY,
    GRAPH_ALGORITHM_STRICT_BOUNDED_REQUIRED_CONTRACT, GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES,
    GRAPH_OPERATION_REGISTRY, GpuResidentSource, GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec,
    GraphBetweennessCentralityParams, GraphBetweennessSeeds, GraphBfsEndpointPolicy, GraphBfsParams,
    GraphBfsPredicateInputs, GraphCalibrationSource, GraphConstructionOptions, GraphConstructionPolicy,
    GraphEdgeBetweennessCentralityParams, GraphEdgeColumnBindings, GraphLayoutRequirement, GraphLouvainParams,
    GraphMemoryClass, GraphMemoryModel, GraphMemoryModelInput, GraphModelErrorRatio, GraphOutputSchemaKind,
    GraphPageRankParams, GraphPredicateInput, GraphReservationOutcome, GraphReservationReport, GraphSeedMode,
    GraphSourceVertexRequirement, GraphStatistics, GraphTriangleCountAllParams, GraphVertexEstimate,
    GraphVertexEstimateSource, GraphVertexIdTypeSupport, GraphWeightPolicy, PlanNode, QueryErrorCode, QueryPlan,
    QueryPlanBuilder, SourceNode, StrictBoundedContractReport, StrictBoundedExecutionMode,
    capability::{CapabilityDecision, ContractQuality, ContractStatus, NodeOperator},
    graph_memory_class, graph_memory_reservation_required, graph_runtime_statistics_probe_bytes,
    graph_vertex_estimate_probe_bytes, stable_graph_result_schema, triangle_count_peak_chunk_min_degree_pricing_term,
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
    assert!(graph.spec.edge_normalize_inserted);
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
fn louvain_ir_display_includes_all_cugraph_scalar_params() {
    let plan = graph_plan(
        GraphAlgorithm::Louvain,
        GraphAlgorithmParams::Louvain(GraphLouvainParams { max_level: 17, threshold: 0.001, resolution: 2.5 }),
        edge_schema(DataType::Int64, DataType::Float64, false),
    );

    let display = plan.to_string();
    assert!(display.contains("algorithm=louvain"));
    assert!(display.contains("params=max_level=17:threshold=0.001:resolution=2.5"));
}

#[test]
fn graph_algorithm_builder_rejects_missing_edge_normalize_columns() {
    let mut builder = QueryPlanBuilder::new();
    let input = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
        "edges",
        Arc::new(Schema::new(vec![
            Field::new("src", DataType::Int64, false),
            Field::new("dst", DataType::Int64, false),
        ])),
    )));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::PageRank,
        GraphEdgeColumnBindings::new("src", "dst", Some("missing_weight")),
        GraphConstructionOptions::default(),
        GraphAlgorithmParams::PageRank(GraphPageRankParams::default()),
    );

    let error = builder.try_add_graph_algorithm(input, spec).expect_err("missing weight column should not panic");

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.operation(), "nexus_query_engine.plan.graph_algorithm.edge_normalize_schema");
    assert_eq!(error.fact_value("column"), Some("missing_weight"));
    assert_eq!(error.fact_value("role"), Some("weight"));
    assert!(error.source_error().is_some());
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
    assert!(!graph.spec.edge_normalize_inserted);
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
        endpoint_policy: GraphBfsEndpointPolicy::Strict,
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
        endpoint_policy: GraphBfsEndpointPolicy::Strict,
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
fn graph_memory_classes_pin_phase_f3_algorithm_tiers() {
    assert_eq!(graph_memory_class(GraphAlgorithm::PageRank), GraphMemoryClass::OutputSized);
    assert_eq!(graph_memory_class(GraphAlgorithm::Bfs), GraphMemoryClass::OutputSized);
    assert_eq!(graph_memory_class(GraphAlgorithm::MinimumSpanningTree), GraphMemoryClass::OutputSized);
    assert_eq!(graph_memory_class(GraphAlgorithm::StronglyConnectedComponents), GraphMemoryClass::MultiArray);
    assert_eq!(graph_memory_class(GraphAlgorithm::Louvain), GraphMemoryClass::MultiArray);
    assert_eq!(graph_memory_class(GraphAlgorithm::TriangleCountAll), GraphMemoryClass::WorkspaceHeavyDataDependent);
    assert_eq!(graph_memory_class(GraphAlgorithm::ForceAtlas2), GraphMemoryClass::WorkspaceHeavyDataDependent);
    assert_eq!(
        graph_memory_class(GraphAlgorithm::BetweennessCentrality),
        GraphMemoryClass::WorkspaceHeavySeedDependent
    );
    assert_eq!(
        graph_memory_class(GraphAlgorithm::EdgeBetweennessCentrality),
        GraphMemoryClass::WorkspaceHeavySeedDependent
    );

    assert!(!graph_memory_reservation_required(GraphAlgorithm::Bfs));
    assert!(graph_memory_reservation_required(GraphAlgorithm::PageRank));
    assert!(graph_memory_reservation_required(GraphAlgorithm::MinimumSpanningTree));
    assert!(graph_memory_reservation_required(GraphAlgorithm::Louvain));
    assert!(graph_memory_reservation_required(GraphAlgorithm::TriangleCountAll));
    assert!(graph_memory_reservation_required(GraphAlgorithm::BetweennessCentrality));
    assert!(graph_memory_reservation_required(GraphAlgorithm::EdgeBetweennessCentrality));
}

#[test]
fn graph_memory_report_admits_and_denies_without_strict_contract_projection() {
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::PageRank,
        GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::PageRank(GraphPageRankParams::default()),
    );
    let estimate = GraphMemoryModel
        .estimate(
            &spec,
            GraphMemoryModelInput {
                raw_input_edges: None,
                input_edges: 100,
                vertex_width_bytes: 4,
                weight_width_bytes: Some(4),
                has_edge_id: false,
                device_total_memory_bytes: None,
                vertex_estimate: None,
                statistics: None,
                edge_normalize_inserted: false,
            },
        )
        .unwrap();
    assert_eq!(estimate.class, GraphMemoryClass::OutputSized);
    assert_eq!(estimate.vertex_upper_bound, 200);
    assert_eq!(estimate.vertex_estimate, 200);
    assert_eq!(estimate.vertex_estimate_source, GraphVertexEstimateSource::EdgeEndpointUpperBound);
    assert_eq!(estimate.requested_bytes, 12_800);
    assert!(estimate.requested_bytes > 0);

    let admitted = GraphReservationReport::from_estimate(
        "n2",
        estimate.clone(),
        Some(GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES),
    );
    assert_eq!(admitted.outcome, GraphReservationOutcome::Admitted);
    assert_eq!(admitted.reason, None);

    let denied = GraphReservationReport::from_estimate(
        "n2",
        estimate.clone(),
        Some(GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES - 1),
    );
    assert_eq!(denied.outcome, GraphReservationOutcome::Denied);
    assert_eq!(denied.reason, Some("graph_memory_estimate_exceeds_budget"));

    let strict = StrictBoundedContractReport::from_capability_analysis(
        &graph_plan(
            GraphAlgorithm::PageRank,
            GraphAlgorithmParams::PageRank(GraphPageRankParams::default()),
            edge_schema(DataType::Int32, DataType::Float32, false),
        )
        .capability_analysis(),
    )
    .unwrap();
    assert!(
        strict
            .gate_rows()
            .iter()
            .all(|row| { row.memory_boundary != "graph_memory" && !row.required_contract.contains("graph_memory") })
    );
}

#[test]
fn graph_memory_model_uses_refined_vertex_estimate_for_sparse_graph_admission() {
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Louvain,
        GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
        GraphConstructionOptions { directed: false, renumber: true, policy: GraphConstructionPolicy::PythonCugraph },
        GraphAlgorithmParams::Louvain(Default::default()),
    );
    let input = GraphMemoryModelInput {
        raw_input_edges: None,
        input_edges: 1_000_000,
        vertex_width_bytes: 8,
        weight_width_bytes: None,
        has_edge_id: false,
        device_total_memory_bytes: None,
        vertex_estimate: None,
        statistics: None,
        edge_normalize_inserted: false,
    };

    let fallback = GraphMemoryModel.estimate(&spec, input).unwrap();
    let refined = GraphMemoryModel
        .estimate(&spec, input.with_vertex_estimate(GraphVertexEstimate::source_statistics(10)))
        .unwrap();

    assert_eq!(fallback.vertex_upper_bound, 2_000_000);
    assert_eq!(fallback.vertex_estimate, 2_000_000);
    assert_eq!(fallback.vertex_estimate_source, GraphVertexEstimateSource::EdgeEndpointUpperBound);
    assert_eq!(refined.vertex_upper_bound, 2_000_000);
    assert_eq!(refined.vertex_estimate, 10);
    assert_eq!(refined.vertex_estimate_source, GraphVertexEstimateSource::SourceStatistics);
    assert!(
        refined.requested_bytes < fallback.requested_bytes,
        "sparse graph pricing must use the refined vertex count for vertex-sized Louvain terms"
    );

    let budget = refined.requested_bytes;
    let refined_report = GraphReservationReport::from_estimate("n2", refined, Some(budget));
    assert_eq!(refined_report.outcome, GraphReservationOutcome::Admitted);
    let fallback_report = GraphReservationReport::from_estimate("n2", fallback, Some(budget));
    assert_eq!(fallback_report.outcome, GraphReservationOutcome::Denied);
    assert_eq!(fallback_report.reason, Some("graph_memory_estimate_exceeds_budget"));
}

#[test]
fn graph_memory_model_clamps_refined_vertex_estimate_to_endpoint_upper_bound() {
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::PageRank,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::PageRank(GraphPageRankParams::default()),
    );
    let estimate = GraphMemoryModel
        .estimate(
            &spec,
            GraphMemoryModelInput {
                raw_input_edges: None,
                input_edges: 5,
                vertex_width_bytes: 4,
                weight_width_bytes: None,
                has_edge_id: false,
                device_total_memory_bytes: None,
                vertex_estimate: Some(GraphVertexEstimate::graph_metadata(999)),
                statistics: None,
                edge_normalize_inserted: false,
            },
        )
        .unwrap();

    assert_eq!(estimate.vertex_upper_bound, 10);
    assert_eq!(estimate.vertex_estimate, 10);
    assert_eq!(estimate.vertex_estimate_source, GraphVertexEstimateSource::GraphMetadata);
}

#[test]
fn graph_memory_model_keeps_triangle_no_stats_fallback_topology_independent() {
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::TriangleCountAll,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: false, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::TriangleCountAll(GraphTriangleCountAllParams),
    );
    let input = GraphMemoryModelInput {
        raw_input_edges: Some(45_564_149),
        input_edges: 91_128_298,
        vertex_width_bytes: 8,
        weight_width_bytes: None,
        has_edge_id: false,
        device_total_memory_bytes: None,
        vertex_estimate: Some(GraphVertexEstimate::runtime_exact_distinct(4_146_772)),
        statistics: None,
        edge_normalize_inserted: true,
    };

    let estimate = GraphMemoryModel.estimate(&spec, input).unwrap();

    assert_eq!(estimate.raw_input_edges, Some(45_564_149));
    assert_eq!(estimate.input_edges, 91_128_298);
    assert!(estimate.edge_normalize_inserted);
    assert_eq!(estimate.calibration_source, GraphCalibrationSource::RawInputEdges);
    assert_eq!(estimate.vertex_width_bytes, 8);
    assert_eq!(estimate.weight_width_bytes, None);
    assert_eq!(estimate.vertex_estimate, 4_146_772);
    assert!(
        estimate.requested_bytes >= 6_617_470_861,
        "topology-independent no-stats triangle-count fallback must cover the live-measured DBLP CUDA async-pool peak"
    );

    let report = GraphReservationReport::from_estimate("g0", estimate, Some(10_000_000_000));
    assert_eq!(report.raw_input_edges, Some(45_564_149));
    assert!(report.edge_normalize_inserted);
    assert_eq!(report.calibration_source, GraphCalibrationSource::RawInputEdges);
}

#[test]
fn graph_memory_model_prices_triangle_count_from_runtime_statistics_formula() {
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::TriangleCountAll,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: false, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::TriangleCountAll(GraphTriangleCountAllParams),
    );
    let statistics = GraphStatistics::runtime_exact_vertices(20)
        .with_sum_out_degree(200)
        .with_degree_distribution(8, 8)
        .with_triangle_min_degree(70, 50, 30);
    let input = GraphMemoryModelInput {
        raw_input_edges: Some(100),
        input_edges: 200,
        vertex_width_bytes: 8,
        weight_width_bytes: None,
        has_edge_id: false,
        device_total_memory_bytes: None,
        vertex_estimate: Some(GraphVertexEstimate::runtime_exact_distinct(20)),
        statistics: Some(statistics),
        edge_normalize_inserted: true,
    };

    let estimate = GraphMemoryModel.estimate(&spec, input).unwrap();

    assert_eq!(estimate.requested_bytes, 13_120);
}

#[test]
fn graph_memory_model_prices_dblp_triangle_from_runtime_statistics_above_measured_peak() {
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::TriangleCountAll,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: false, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::TriangleCountAll(GraphTriangleCountAllParams),
    );
    let peak_chunk_min_degree_sum =
        triangle_count_peak_chunk_min_degree_pricing_term(493_000_000, 45_476_753, Some(188));
    assert_eq!(peak_chunk_min_degree_sum, 77_031_250);
    let statistics = GraphStatistics::runtime_exact_vertices(4_146_772)
        .with_degree_distribution(128, 34)
        .with_triangle_min_degree(493_000_000, 45_476_753, peak_chunk_min_degree_sum);
    let input = GraphMemoryModelInput {
        raw_input_edges: Some(45_564_149),
        input_edges: 91_128_298,
        vertex_width_bytes: 8,
        weight_width_bytes: None,
        has_edge_id: false,
        device_total_memory_bytes: Some(101_449_662_464),
        vertex_estimate: Some(GraphVertexEstimate::runtime_exact_distinct(4_146_772)),
        statistics: Some(statistics),
        edge_normalize_inserted: true,
    };

    let estimate = GraphMemoryModel.estimate(&spec, input).unwrap();

    assert_eq!(estimate.requested_bytes, 7_360_481_984);
    assert!(
        estimate.requested_bytes >= 6_617_470_861,
        "runtime-statistics DBLP triangle ask must cover the measured CUDA async-pool peak"
    );
    assert!(
        estimate.requested_bytes < 12_681_207_808 - 2_190_000_000,
        "runtime-statistics DBLP triangle ask should fit the default min grant after the live edge reservation"
    );
}

#[test]
fn graph_memory_model_keeps_triangle_plan_time_fallback_below_whole_device_ticket() {
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::TriangleCountAll,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: false, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::TriangleCountAll(GraphTriangleCountAllParams),
    );
    let input = GraphMemoryModelInput {
        raw_input_edges: Some(45_564_149),
        input_edges: 91_128_298,
        vertex_width_bytes: 8,
        weight_width_bytes: None,
        has_edge_id: false,
        device_total_memory_bytes: Some(101_449_662_464),
        vertex_estimate: None,
        statistics: None,
        edge_normalize_inserted: true,
    };

    let estimate = GraphMemoryModel.estimate(&spec, input).unwrap();

    assert_eq!(estimate.vertex_estimate, 182_256_596);
    assert_eq!(estimate.requested_bytes, 33_535_213_664);
    assert!(
        estimate.requested_bytes < 101_449_662_464 / 2,
        "plan-time triangle fallback should remain a packing hint, not collapse to tenant want"
    );
}

#[test]
fn graph_runtime_statistics_probe_reserves_remaining_window_for_triangle_evidence() {
    let input = GraphMemoryModelInput {
        raw_input_edges: Some(45_564_149),
        input_edges: 45_564_149,
        vertex_width_bytes: 8,
        weight_width_bytes: None,
        has_edge_id: false,
        device_total_memory_bytes: Some(101_449_662_464),
        vertex_estimate: None,
        statistics: None,
        edge_normalize_inserted: false,
    };
    let default_min_grant = 101_449_662_464 / 8;
    let remaining_after_live_input = default_min_grant - 2_190_000_000;
    let exact_vertex_only_probe = graph_vertex_estimate_probe_bytes(input, remaining_after_live_input);

    assert_eq!(exact_vertex_only_probe, 6_561_237_456);
    assert_eq!(
        graph_runtime_statistics_probe_bytes(GraphAlgorithm::PageRank, input, remaining_after_live_input),
        exact_vertex_only_probe,
        "algorithms that only need exact vertices should keep the narrower probe"
    );
    assert_eq!(
        graph_runtime_statistics_probe_bytes(GraphAlgorithm::TriangleCountAll, input, remaining_after_live_input),
        remaining_after_live_input,
        "triangle runtime evidence should receive the full transient probe window"
    );
}

#[test]
fn graph_memory_pricing_constants_match_pinned_cugraph_sources() {
    let triangle_intersection =
        read_cugraph_source("cpp/include/cugraph/prims/transform_reduce_dst_nbr_intersection_of_e_endpoints_by_v.cuh");
    let triangle_intersection_compact = without_ascii_whitespace(&triangle_intersection);
    assert!(
        triangle_intersection_compact
            .contains("static_cast<size_t>(handle.get_device_properties().multiProcessorCount)*(1<<15)"),
        "triangle-count chunk pricing must track cuGraph's SM * 2^15 intersection chunk size"
    );
    assert!(
        triangle_intersection.contains("auto [intersection_offsets, intersection_indices]"),
        "triangle-count pricing assumes cuGraph materializes intersection offsets and indices"
    );
    assert!(
        triangle_intersection.contains("detail::nbr_intersection(handle"),
        "triangle-count pricing assumes the transform-reduce primitive delegates to nbr_intersection"
    );

    let nbr_intersection = read_cugraph_source("cpp/include/cugraph/prims/detail/nbr_intersection.cuh");
    let nbr_intersection_compact = without_ascii_whitespace(&nbr_intersection);
    assert!(
        nbr_intersection.contains("initially store minimum degrees (upper bound for intersection sizes)"),
        "triangle-count pricing uses min-degree as cuGraph's pre-compaction intersection upper bound"
    );
    assert!(
        nbr_intersection_compact
            .contains("nbr_intersection_indices.resize(nbr_intersection_offsets.back_element(handle.get_stream()),handle.get_stream())"),
        "triangle-count pricing assumes intersection indices are allocated from the min-degree offsets before compaction"
    );
    assert!(
        nbr_intersection_compact.contains("nbr_intersection_indices.resize(cuda::std::distance"),
        "triangle-count pricing assumes cuGraph compacts intersection indices after materialization"
    );

    let betweenness = read_cugraph_source("cpp/src/centrality/betweenness_centrality_impl.cuh");
    let betweenness_compact = without_ascii_whitespace(&betweenness);
    assert!(
        betweenness_compact.contains("autoconstexprmax_edge_tuple_data_ratio=0.25;"),
        "betweenness push-batch pricing must track cuGraph's /4 edge tuple memory cap"
    );
    assert!(
        betweenness_compact.contains("(sizeof(vertex_t)+sizeof(origin_t)+sizeof(edge_t))"),
        "betweenness push-batch pricing must track the cuGraph push tuple layout"
    );
    assert!(
        betweenness_compact.contains("usingorigin_t=uint16_t;")
            && betweenness_compact.contains("std::numeric_limits<origin_t>::max()"),
        "betweenness source batching must track cuGraph's uint16_t source-origin cap"
    );
    assert!(
        betweenness_compact.contains("autoconstexprmax_multisource_bfs_result_ratio=0.25;")
            && betweenness_compact.contains("std::numeric_limits<uint16_t>::max()"),
        "betweenness resident-source pricing must track cuGraph's /4 BFS result cap and uint16_t source limit"
    );
}

#[test]
fn graph_memory_model_is_dtype_sensitive_for_calibrated_sparse_keys() {
    let pagerank_spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::PageRank,
        GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::PageRank(GraphPageRankParams::default()),
    );
    let louvain_spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Louvain,
        GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
        GraphConstructionOptions { directed: false, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::Louvain(Default::default()),
    );
    let input = |vertex_width_bytes, weight_width_bytes| GraphMemoryModelInput {
        raw_input_edges: Some(1_000_000),
        input_edges: 1_000_000,
        vertex_width_bytes,
        weight_width_bytes: Some(weight_width_bytes),
        has_edge_id: false,
        device_total_memory_bytes: None,
        vertex_estimate: Some(GraphVertexEstimate::runtime_exact_distinct(100_000)),
        statistics: None,
        edge_normalize_inserted: false,
    };

    for spec in [&pagerank_spec, &louvain_spec] {
        let narrow = GraphMemoryModel.estimate(spec, input(4, 4)).unwrap();
        let wide = GraphMemoryModel.estimate(spec, input(8, 8)).unwrap();
        assert!(
            wide.requested_bytes > narrow.requested_bytes,
            "{} pricing must vary with vertex and weight dtype widths",
            spec.algorithm.as_str()
        );
    }

    let triangle_spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::TriangleCountAll,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: false, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::TriangleCountAll(GraphTriangleCountAllParams),
    );
    let triangle_input = |vertex_width_bytes| GraphMemoryModelInput {
        raw_input_edges: Some(1_000_000),
        input_edges: 1_000_000,
        vertex_width_bytes,
        weight_width_bytes: None,
        has_edge_id: false,
        device_total_memory_bytes: None,
        vertex_estimate: Some(GraphVertexEstimate::runtime_exact_distinct(100_000)),
        statistics: None,
        edge_normalize_inserted: false,
    };
    let narrow = GraphMemoryModel.estimate(&triangle_spec, triangle_input(4)).unwrap();
    let wide = GraphMemoryModel.estimate(&triangle_spec, triangle_input(8)).unwrap();
    assert!(wide.requested_bytes > narrow.requested_bytes, "triangle-count pricing must vary with vertex dtype width");
}

fn read_cugraph_source(relative: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../cugraph-rs/cugraph").join(relative);
    std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

fn without_ascii_whitespace(source: &str) -> String {
    source.chars().filter(|character| !character.is_ascii_whitespace()).collect()
}

#[test]
fn graph_memory_model_error_ratio_is_fixed_point_and_optional_on_zero_peak() {
    assert_eq!(GraphModelErrorRatio::from_requested_and_measured(4096, 2048).unwrap().as_decimal_string(), "2.000");
    assert_eq!(GraphModelErrorRatio::from_requested_and_measured(5000, 2000).unwrap().as_decimal_string(), "2.500");
    assert!(GraphModelErrorRatio::from_requested_and_measured(4096, 0).is_none());

    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::PageRank,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::PageRank(GraphPageRankParams::default()),
    );
    let estimate = GraphMemoryModel
        .estimate(
            &spec,
            GraphMemoryModelInput {
                raw_input_edges: None,
                input_edges: 100,
                vertex_width_bytes: 4,
                weight_width_bytes: None,
                has_edge_id: false,
                device_total_memory_bytes: None,
                vertex_estimate: None,
                statistics: None,
                edge_normalize_inserted: false,
            },
        )
        .unwrap();
    let requested_bytes = estimate.requested_bytes;
    assert!(requested_bytes < GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES);
    let mut report =
        GraphReservationReport::from_estimate("g0", estimate, Some(GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES));
    assert_eq!(report.model_requested_bytes, requested_bytes);
    assert_eq!(report.requested_bytes, GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES);
    report.record_model_error_ratio(requested_bytes);
    assert_eq!(
        report.graph_model_error_ratio.unwrap().as_decimal_string(),
        "1.000",
        "model error ratio must use the raw model ask, not the bounded reservation floor"
    );
}

#[test]
fn betweenness_seed_count_reports_requested_sources_and_prices_resident_source_state() {
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::BetweennessCentrality,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::BetweennessCentrality(GraphBetweennessCentralityParams {
            seeds: Some(GraphBetweennessSeeds::I64(vec![0, 2, 4])),
            ..Default::default()
        }),
    );
    let estimate = GraphMemoryModel
        .estimate(
            &spec,
            GraphMemoryModelInput {
                raw_input_edges: None,
                input_edges: 100,
                vertex_width_bytes: 4,
                weight_width_bytes: None,
                has_edge_id: false,
                device_total_memory_bytes: None,
                vertex_estimate: None,
                statistics: None,
                edge_normalize_inserted: false,
            },
        )
        .unwrap();
    assert_eq!(estimate.class, GraphMemoryClass::WorkspaceHeavySeedDependent);
    assert_eq!(estimate.seed_count, Some(3));
    assert_eq!(estimate.seed_mode, Some(GraphSeedMode::ExplicitSeedList));
    assert!(estimate.data_dependent);

    let k_spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::EdgeBetweennessCentrality,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::EdgeBetweennessCentrality(GraphEdgeBetweennessCentralityParams {
            k: Some(5),
            ..Default::default()
        }),
    );
    let k_estimate = GraphMemoryModel
        .estimate(
            &k_spec,
            GraphMemoryModelInput {
                raw_input_edges: None,
                input_edges: 100,
                vertex_width_bytes: 4,
                weight_width_bytes: None,
                has_edge_id: false,
                device_total_memory_bytes: None,
                vertex_estimate: None,
                statistics: None,
                edge_normalize_inserted: false,
            },
        )
        .unwrap();
    assert_eq!(k_estimate.seed_count, Some(5));
    assert_eq!(k_estimate.seed_mode, Some(GraphSeedMode::RuntimeKSelection));

    let all_vertex_estimate = GraphMemoryModel
        .estimate(
            &GraphAlgorithmSpec::with_stable_schema(
                GraphAlgorithm::BetweennessCentrality,
                GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
                GraphConstructionOptions {
                    directed: true,
                    renumber: true,
                    policy: GraphConstructionPolicy::RawLibcugraph,
                },
                GraphAlgorithmParams::BetweennessCentrality(GraphBetweennessCentralityParams::default()),
            ),
            GraphMemoryModelInput {
                raw_input_edges: None,
                input_edges: 100,
                vertex_width_bytes: 4,
                weight_width_bytes: None,
                has_edge_id: false,
                device_total_memory_bytes: None,
                vertex_estimate: Some(GraphVertexEstimate::runtime_exact_distinct(7)),
                statistics: None,
                edge_normalize_inserted: false,
            },
        )
        .unwrap();
    assert_eq!(all_vertex_estimate.vertex_upper_bound, 200);
    assert_eq!(all_vertex_estimate.vertex_estimate, 7);
    assert_eq!(all_vertex_estimate.seed_count, Some(7));
    assert_eq!(all_vertex_estimate.seed_mode, Some(GraphSeedMode::AllVertices));
}

#[test]
fn vertex_betweenness_source_workspace_uses_cugraph_physical_batch_cap() {
    let device_total_memory_bytes = 64 * 1024 * 1024 * 1024;
    let vertex_estimate = 1_000_000;
    let vertex_width_bytes = 8;
    let cugraph_batch_cap = device_total_memory_bytes / 4 / (vertex_estimate * vertex_width_bytes * 2);
    assert!(cugraph_batch_cap > 1);
    assert!(cugraph_batch_cap < u16::MAX as u64);

    let spec_for_k = |k| {
        GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::BetweennessCentrality,
            GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
            GraphAlgorithmParams::BetweennessCentrality(GraphBetweennessCentralityParams {
                k: Some(k),
                exact_vertex_threshold: vertex_estimate,
                ..Default::default()
            }),
        )
    };
    let input = GraphMemoryModelInput {
        raw_input_edges: None,
        input_edges: 1_000_000,
        vertex_width_bytes,
        weight_width_bytes: None,
        has_edge_id: false,
        device_total_memory_bytes: Some(device_total_memory_bytes),
        vertex_estimate: Some(GraphVertexEstimate::runtime_exact_distinct(vertex_estimate)),
        statistics: None,
        edge_normalize_inserted: false,
    };

    let capped = GraphMemoryModel.estimate(&spec_for_k(cugraph_batch_cap), input).unwrap();
    let oversized = GraphMemoryModel.estimate(&spec_for_k(cugraph_batch_cap * 10), input).unwrap();

    assert_eq!(capped.seed_count, Some(cugraph_batch_cap));
    assert_eq!(oversized.seed_count, Some(cugraph_batch_cap * 10));
    let expected_seed_list_delta = (oversized.seed_count.unwrap() - capped.seed_count.unwrap()) * vertex_width_bytes;
    assert_eq!(
        oversized.requested_bytes - capped.requested_bytes,
        expected_seed_list_delta,
        "k-only vertex betweenness should price additional requested sources as seed-list storage only"
    );
}

#[test]
fn vertex_betweenness_push_batch_workspace_scales_with_resident_sources() {
    let spec_for_seeds = |seeds| {
        GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::BetweennessCentrality,
            GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
            GraphAlgorithmParams::BetweennessCentrality(GraphBetweennessCentralityParams {
                seeds: Some(GraphBetweennessSeeds::I64(seeds)),
                ..Default::default()
            }),
        )
    };
    let input = GraphMemoryModelInput {
        raw_input_edges: None,
        input_edges: 100,
        vertex_width_bytes: 8,
        weight_width_bytes: None,
        has_edge_id: false,
        device_total_memory_bytes: None,
        vertex_estimate: Some(GraphVertexEstimate::runtime_exact_distinct(10)),
        statistics: Some(GraphStatistics::runtime_exact_vertices(10).with_sum_out_degree(100)),
        edge_normalize_inserted: false,
    };

    let one_source = GraphMemoryModel.estimate(&spec_for_seeds(vec![0]), input).unwrap();
    let three_sources = GraphMemoryModel.estimate(&spec_for_seeds(vec![0, 1, 2]), input).unwrap();

    assert_eq!(one_source.seed_count, Some(1));
    assert_eq!(three_sources.seed_count, Some(3));
    let per_source_vertex_state_width = 44;
    let expected_delta = 2 * 10 * per_source_vertex_state_width + 2 * 8;
    assert_eq!(
        three_sources.requested_bytes - one_source.requested_bytes,
        expected_delta,
        "multi-source vertex betweenness must not multiply an already resident-source push statistic"
    );
}

#[test]
fn vertex_betweenness_push_batch_workspace_caps_statistics_present_branch() {
    let device_total_memory_bytes = 1_024 * 1_024 * 1_024;
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::BetweennessCentrality,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::BetweennessCentrality(GraphBetweennessCentralityParams {
            seeds: Some(GraphBetweennessSeeds::I64(vec![0])),
            ..Default::default()
        }),
    );
    let input_for_sum = |sum_out_degree| GraphMemoryModelInput {
        raw_input_edges: None,
        input_edges: 1_000,
        vertex_width_bytes: 8,
        weight_width_bytes: None,
        has_edge_id: false,
        device_total_memory_bytes: Some(device_total_memory_bytes),
        vertex_estimate: Some(GraphVertexEstimate::runtime_exact_distinct(1_000)),
        statistics: Some(GraphStatistics::runtime_exact_vertices(1_000).with_sum_out_degree(sum_out_degree)),
        edge_normalize_inserted: false,
    };

    let uncapped = GraphMemoryModel.estimate(&spec, input_for_sum(1_000)).unwrap();
    let capped = GraphMemoryModel.estimate(&spec, input_for_sum(u64::MAX / 18)).unwrap();
    let tuple_width = 18;
    let push_cap_bytes = (device_total_memory_bytes / 4 / tuple_width) * tuple_width;

    assert_eq!(
        capped.requested_bytes - uncapped.requested_bytes,
        push_cap_bytes - 1_000 * tuple_width,
        "statistics-present betweenness push term must exercise cuGraph's total-memory / 4 cap"
    );
}

#[test]
fn vertex_betweenness_dblp_top64_statistics_fits_24gb_budget() {
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::BetweennessCentrality,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::BetweennessCentrality(GraphBetweennessCentralityParams {
            k: Some(64),
            exact_vertex_threshold: 4_146_772,
            ..Default::default()
        }),
    );
    let input = GraphMemoryModelInput {
        raw_input_edges: Some(45_564_149),
        input_edges: 45_564_149,
        vertex_width_bytes: 8,
        weight_width_bytes: None,
        has_edge_id: false,
        device_total_memory_bytes: Some(101_449_662_464),
        vertex_estimate: Some(GraphVertexEstimate::runtime_exact_distinct(4_146_772)),
        statistics: Some(GraphStatistics::runtime_exact_vertices(4_146_772).with_sum_out_degree(35_893)),
        edge_normalize_inserted: false,
    };

    let estimate = GraphMemoryModel.estimate(&spec, input).unwrap();

    assert_eq!(estimate.requested_bytes, 15_323_491_202);
    assert!(
        estimate.requested_bytes < 24 * 1024 * 1024 * 1024,
        "DBLP betweenness k=64 with resident-source degree statistics should fit a 24 GiB graph budget"
    );
}

#[test]
fn edge_betweenness_seed_count_does_not_price_all_requested_sources_as_resident() {
    let spec_for_k = |k| {
        GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::EdgeBetweennessCentrality,
            GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
            GraphAlgorithmParams::EdgeBetweennessCentrality(GraphEdgeBetweennessCentralityParams {
                k: Some(k),
                ..Default::default()
            }),
        )
    };
    let input = GraphMemoryModelInput {
        raw_input_edges: None,
        input_edges: 1_000,
        vertex_width_bytes: 8,
        weight_width_bytes: None,
        has_edge_id: false,
        device_total_memory_bytes: None,
        vertex_estimate: Some(GraphVertexEstimate::runtime_exact_distinct(1_000)),
        statistics: None,
        edge_normalize_inserted: false,
    };

    let small = GraphMemoryModel.estimate(&spec_for_k(10), input).unwrap();
    let large = GraphMemoryModel.estimate(&spec_for_k(10_000), input).unwrap();

    assert_eq!(small.seed_count, Some(10));
    assert_eq!(large.seed_count, Some(10_000));
    assert!(
        large.requested_bytes - small.requested_bytes < 10_000 * 8 * 2,
        "edge betweenness should account seed-list storage, not a live seeds-by-vertices matrix"
    );
}

#[test]
fn edge_betweenness_output_pricing_scales_with_edges() {
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::EdgeBetweennessCentrality,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::EdgeBetweennessCentrality(GraphEdgeBetweennessCentralityParams {
            k: Some(10),
            ..Default::default()
        }),
    );
    let input_for_edges = |input_edges| GraphMemoryModelInput {
        raw_input_edges: None,
        input_edges,
        vertex_width_bytes: 8,
        weight_width_bytes: None,
        has_edge_id: false,
        device_total_memory_bytes: None,
        vertex_estimate: Some(GraphVertexEstimate::runtime_exact_distinct(1_000)),
        statistics: None,
        edge_normalize_inserted: false,
    };

    let small = GraphMemoryModel.estimate(&spec, input_for_edges(1_000)).unwrap();
    let large = GraphMemoryModel.estimate(&spec, input_for_edges(10_000)).unwrap();

    assert!(
        large.requested_bytes - small.requested_bytes >= 9_000 * 24,
        "edge betweenness result memory should include source, destination, and score arrays per output edge"
    );
}

#[test]
fn betweenness_k_seed_selection_prices_endpoint_distinct_sort_transient() {
    let spec_for_algorithm = |algorithm| match algorithm {
        GraphAlgorithm::BetweennessCentrality => GraphAlgorithmSpec::with_stable_schema(
            algorithm,
            GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
            GraphAlgorithmParams::BetweennessCentrality(GraphBetweennessCentralityParams {
                k: Some(10),
                ..Default::default()
            }),
        ),
        GraphAlgorithm::EdgeBetweennessCentrality => GraphAlgorithmSpec::with_stable_schema(
            algorithm,
            GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
            GraphAlgorithmParams::EdgeBetweennessCentrality(GraphEdgeBetweennessCentralityParams {
                k: Some(10),
                ..Default::default()
            }),
        ),
        other => panic!("unexpected algorithm {other:?}"),
    };
    let input_for_edges = |input_edges| GraphMemoryModelInput {
        raw_input_edges: None,
        input_edges,
        vertex_width_bytes: 8,
        weight_width_bytes: None,
        has_edge_id: false,
        device_total_memory_bytes: None,
        vertex_estimate: Some(GraphVertexEstimate::runtime_exact_distinct(1_000)),
        statistics: None,
        edge_normalize_inserted: false,
    };

    for algorithm in [GraphAlgorithm::BetweennessCentrality, GraphAlgorithm::EdgeBetweennessCentrality] {
        let spec = spec_for_algorithm(algorithm);
        let small = GraphMemoryModel.estimate(&spec, input_for_edges(1_000)).unwrap();
        let large = GraphMemoryModel.estimate(&spec, input_for_edges(10_000)).unwrap();
        assert!(
            large.requested_bytes - small.requested_bytes >= 9_000 * 8 * 4,
            "{algorithm:?} k seed selection must price endpoint copy and concat transient"
        );
    }
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
            (GraphAlgorithm::BetweennessCentrality, "betweenness_centrality", GraphLayoutRequirement::Any),
            (GraphAlgorithm::EdgeBetweennessCentrality, "edge_betweenness_centrality", GraphLayoutRequirement::Any),
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
            (GraphAlgorithm::PersonalizedPageRank, "personalized_pagerank", GraphLayoutRequirement::Transposed),
            (GraphAlgorithm::Leiden, "leiden", GraphLayoutRequirement::Forward),
            (GraphAlgorithm::Ecg, "ecg", GraphLayoutRequirement::Forward),
            (
                GraphAlgorithm::SpectralModularityMaximization,
                "spectral_modularity_maximization",
                GraphLayoutRequirement::Forward,
            ),
            (GraphAlgorithm::BalancedCutClustering, "balanced_cut_clustering", GraphLayoutRequirement::Forward),
            (GraphAlgorithm::ForceAtlas2, "force_atlas2", GraphLayoutRequirement::Any),
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
        assert_eq!(
            def.base_result_schema().as_ref(),
            stable_graph_result_schema(def.algorithm).as_ref(),
            "descriptor base_result_schema and stable_graph_result_schema must agree",
        );
        assert_eq!(
            def.algorithm.requires_source_vertex(),
            def.source_vertex_requirement().requires_source_vertex(),
            "GraphAlgorithm source requirement helper must project the descriptor",
        );
        assert_eq!(
            def.algorithm.consumes_weight(),
            def.weight_policy().consumes_weight(),
            "GraphAlgorithm weight helper must project the descriptor",
        );
        assert_eq!(
            def.algorithm.output_schema_kind(),
            def.output_schema_kind(),
            "GraphAlgorithm output-schema helper must project the descriptor",
        );
        assert_eq!(
            def.default_params().algorithm(),
            def.algorithm,
            "descriptor default params must use the registry algorithm variant",
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
fn force_atlas2_result_schema_contract_is_exact() {
    assert_eq!(
        stable_graph_result_schema(GraphAlgorithm::ForceAtlas2)
            .fields()
            .iter()
            .map(|field| field.as_ref().clone())
            .collect::<Vec<_>>(),
        vec![
            Field::new("vertex", DataType::Int64, false),
            Field::new("x", DataType::Float32, false),
            Field::new("y", DataType::Float32, false),
        ],
        "ForceAtlas2 layout result schema is part of the native graph contract",
    );
}

#[test]
fn graph_operation_registry_pins_descriptor_contracts() {
    let descriptors: Vec<_> = GRAPH_OPERATION_REGISTRY
        .iter()
        .map(|def| {
            (
                def.algorithm,
                def.source_vertex_requirement(),
                def.weight_policy(),
                def.output_schema_kind(),
                def.default_graph_options(&GraphConstructionOptions::default()).directed,
            )
        })
        .collect();
    assert_eq!(
        descriptors,
        vec![
            (
                GraphAlgorithm::PageRank,
                GraphSourceVertexRequirement::None,
                GraphWeightPolicy::Optional,
                GraphOutputSchemaKind::Fixed,
                true,
            ),
            (
                GraphAlgorithm::Bfs,
                GraphSourceVertexRequirement::Required,
                GraphWeightPolicy::Ignored,
                GraphOutputSchemaKind::DefaultMode,
                true,
            ),
            (
                GraphAlgorithm::Louvain,
                GraphSourceVertexRequirement::None,
                GraphWeightPolicy::Ignored,
                GraphOutputSchemaKind::Fixed,
                true,
            ),
            (
                GraphAlgorithm::EigenvectorCentrality,
                GraphSourceVertexRequirement::None,
                GraphWeightPolicy::Ignored,
                GraphOutputSchemaKind::Fixed,
                true,
            ),
            (
                GraphAlgorithm::KatzCentrality,
                GraphSourceVertexRequirement::None,
                GraphWeightPolicy::Ignored,
                GraphOutputSchemaKind::Fixed,
                true,
            ),
            (
                GraphAlgorithm::Hits,
                GraphSourceVertexRequirement::None,
                GraphWeightPolicy::Ignored,
                GraphOutputSchemaKind::Fixed,
                true,
            ),
            (
                GraphAlgorithm::BetweennessCentrality,
                GraphSourceVertexRequirement::None,
                GraphWeightPolicy::Ignored,
                GraphOutputSchemaKind::Fixed,
                true,
            ),
            (
                GraphAlgorithm::EdgeBetweennessCentrality,
                GraphSourceVertexRequirement::None,
                GraphWeightPolicy::Ignored,
                GraphOutputSchemaKind::Fixed,
                true,
            ),
            (
                GraphAlgorithm::Sssp,
                GraphSourceVertexRequirement::Required,
                GraphWeightPolicy::Optional,
                GraphOutputSchemaKind::Fixed,
                true,
            ),
            (
                GraphAlgorithm::DegreesAll,
                GraphSourceVertexRequirement::None,
                GraphWeightPolicy::Ignored,
                GraphOutputSchemaKind::Fixed,
                true,
            ),
            (
                GraphAlgorithm::InDegreesAll,
                GraphSourceVertexRequirement::None,
                GraphWeightPolicy::Ignored,
                GraphOutputSchemaKind::Fixed,
                true,
            ),
            (
                GraphAlgorithm::OutDegreesAll,
                GraphSourceVertexRequirement::None,
                GraphWeightPolicy::Ignored,
                GraphOutputSchemaKind::Fixed,
                true,
            ),
            (
                GraphAlgorithm::WeaklyConnectedComponents,
                GraphSourceVertexRequirement::None,
                GraphWeightPolicy::Ignored,
                GraphOutputSchemaKind::Fixed,
                false,
            ),
            (
                GraphAlgorithm::StronglyConnectedComponents,
                GraphSourceVertexRequirement::None,
                GraphWeightPolicy::Ignored,
                GraphOutputSchemaKind::Fixed,
                true,
            ),
            (
                GraphAlgorithm::TriangleCountAll,
                GraphSourceVertexRequirement::None,
                GraphWeightPolicy::Ignored,
                GraphOutputSchemaKind::Fixed,
                false,
            ),
            (
                GraphAlgorithm::CoreNumber,
                GraphSourceVertexRequirement::None,
                GraphWeightPolicy::Ignored,
                GraphOutputSchemaKind::Fixed,
                false,
            ),
            (
                GraphAlgorithm::KCore,
                GraphSourceVertexRequirement::None,
                GraphWeightPolicy::Ignored,
                GraphOutputSchemaKind::Fixed,
                false,
            ),
            (
                GraphAlgorithm::MinimumSpanningTree,
                GraphSourceVertexRequirement::None,
                GraphWeightPolicy::Required,
                GraphOutputSchemaKind::Fixed,
                false,
            ),
            (
                GraphAlgorithm::PersonalizedPageRank,
                GraphSourceVertexRequirement::None,
                GraphWeightPolicy::Optional,
                GraphOutputSchemaKind::Fixed,
                true,
            ),
            (
                GraphAlgorithm::Leiden,
                GraphSourceVertexRequirement::None,
                GraphWeightPolicy::Ignored,
                GraphOutputSchemaKind::Fixed,
                true,
            ),
            (
                GraphAlgorithm::Ecg,
                GraphSourceVertexRequirement::None,
                GraphWeightPolicy::Ignored,
                GraphOutputSchemaKind::Fixed,
                true,
            ),
            (
                GraphAlgorithm::SpectralModularityMaximization,
                GraphSourceVertexRequirement::None,
                GraphWeightPolicy::Ignored,
                GraphOutputSchemaKind::Fixed,
                false,
            ),
            (
                GraphAlgorithm::BalancedCutClustering,
                GraphSourceVertexRequirement::None,
                GraphWeightPolicy::Ignored,
                GraphOutputSchemaKind::Fixed,
                false,
            ),
            (
                GraphAlgorithm::ForceAtlas2,
                GraphSourceVertexRequirement::None,
                GraphWeightPolicy::Optional,
                GraphOutputSchemaKind::Fixed,
                false,
            ),
        ],
        "native graph descriptor fields are part of the SQL/native graph contract",
    );

    let mut bfs_params = GraphBfsParams::new(0, None);
    bfs_params.return_target_info = true;
    let schema =
        GraphAlgorithm::Bfs.definition().result_schema_for_params(&GraphAlgorithmParams::Bfs(Box::new(bfs_params)));
    assert_eq!(
        schema.fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>(),
        vec!["vertex", "distance", "predecessor", "target_found", "target_distance"],
        "BFS target-info schema is native descriptor-owned",
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
fn graph_algorithm_capability_validates_edge_column_types_but_allows_nullable_schema() {
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
    let nullable_weight_capability = nullable_weight
        .capability_analysis()
        .nodes
        .into_iter()
        .find(|node| node.node_id == nullable_weight.root_id())
        .unwrap();
    assert_eq!(nullable_weight_capability.executable, CapabilityDecision::Satisfied);

    let nullable_vertices = graph_plan(
        GraphAlgorithm::Louvain,
        GraphAlgorithmParams::Louvain(Default::default()),
        edge_schema_with_vertex_nullability(DataType::Int64, DataType::Float64),
    );
    let nullable_vertices_capability = nullable_vertices
        .capability_analysis()
        .nodes
        .into_iter()
        .find(|node| node.node_id == nullable_vertices.root_id())
        .unwrap();
    assert_eq!(nullable_vertices_capability.executable, CapabilityDecision::Satisfied);

    let nullable_edge_id =
        graph_plan_with_edge_id(edge_schema_with_nullable_edge_id(DataType::Int64, DataType::Float64));
    let nullable_edge_id_capability = nullable_edge_id
        .capability_analysis()
        .nodes
        .into_iter()
        .find(|node| node.node_id == nullable_edge_id.root_id())
        .unwrap();
    assert_eq!(nullable_edge_id_capability.executable, CapabilityDecision::Satisfied);
}

#[test]
fn graph_algorithm_capability_rejects_int64_for_legacy_int32_only_algorithms() {
    for (algorithm, params) in [
        (GraphAlgorithm::MinimumSpanningTree, GraphAlgorithmParams::MinimumSpanningTree(Default::default())),
        (
            GraphAlgorithm::SpectralModularityMaximization,
            GraphAlgorithmParams::SpectralModularityMaximization(Default::default()),
        ),
        (GraphAlgorithm::BalancedCutClustering, GraphAlgorithmParams::BalancedCutClustering(Default::default())),
    ] {
        assert_eq!(algorithm.vertex_id_type_support(), GraphVertexIdTypeSupport::Int32OnlyLegacyEdgeType);
        let graph_options = algorithm.definition().default_graph_options(&GraphConstructionOptions::default());
        let int64_plan = graph_plan_with_options(
            algorithm,
            params.clone(),
            graph_options.clone(),
            edge_schema(DataType::Int64, DataType::Float64, false),
        );
        assert_graph_rejection_facts(
            &int64_plan,
            "unsupported_vertex_dtype",
            "source",
            &[
                ("algorithm", algorithm.as_str()),
                ("vertex_id_type_support", "int32_only_legacy_edge_type"),
                ("supported_vertex_dtypes", "Int32"),
            ],
        );

        let int32_plan = graph_plan_with_options(
            algorithm,
            params,
            graph_options,
            edge_schema(DataType::Int32, DataType::Float64, false),
        );
        let int32_capability = int32_plan
            .capability_analysis()
            .nodes
            .into_iter()
            .find(|node| node.node_id == int32_plan.root_id())
            .unwrap();
        assert_eq!(int32_capability.executable, CapabilityDecision::Satisfied);
    }
}

#[test]
fn graph_algorithm_capability_rejects_incompatible_directedness() {
    let scc = graph_plan_with_options(
        GraphAlgorithm::StronglyConnectedComponents,
        GraphAlgorithmParams::StronglyConnectedComponents(Default::default()),
        GraphConstructionOptions { directed: false, ..Default::default() },
        edge_schema(DataType::Int64, DataType::Float64, false),
    );
    assert_graph_directedness_rejection(&scc, "strongly_connected_components", false, true);

    let wcc = graph_plan_with_options(
        GraphAlgorithm::WeaklyConnectedComponents,
        GraphAlgorithmParams::WeaklyConnectedComponents(Default::default()),
        GraphConstructionOptions { directed: true, ..Default::default() },
        edge_schema(DataType::Int64, DataType::Float64, false),
    );
    assert_graph_directedness_rejection(&wcc, "weakly_connected_components", true, false);

    for (algorithm, params) in [
        (GraphAlgorithm::MinimumSpanningTree, GraphAlgorithmParams::MinimumSpanningTree(Default::default())),
        (
            GraphAlgorithm::SpectralModularityMaximization,
            GraphAlgorithmParams::SpectralModularityMaximization(Default::default()),
        ),
        (GraphAlgorithm::BalancedCutClustering, GraphAlgorithmParams::BalancedCutClustering(Default::default())),
    ] {
        let plan = graph_plan_with_options(
            algorithm,
            params,
            GraphConstructionOptions { directed: true, ..Default::default() },
            edge_schema(DataType::Int32, DataType::Float64, false),
        );
        assert_graph_directedness_rejection(&plan, algorithm.as_str(), true, false);
    }
}

fn assert_graph_rejection_contains(plan: &QueryPlan, reason: &str, role: &str) {
    assert_graph_rejection_facts(plan, reason, role, &[]);
}

fn assert_graph_rejection_facts(plan: &QueryPlan, reason: &str, role: &str, expected: &[(&str, &str)]) {
    let capability = plan.capability_analysis().nodes.into_iter().find(|node| node.node_id == plan.root_id()).unwrap();
    let CapabilityDecision::Rejected { facts, .. } = capability.executable else {
        panic!("expected graph capability rejection");
    };
    assert_eq!(facts.get("reason").map(String::as_str), Some(reason));
    assert_eq!(facts.get("edge_column_role").map(String::as_str), Some(role));
    for (key, value) in expected {
        assert_eq!(facts.get(*key).map(String::as_str), Some(*value), "unexpected fact {key}");
    }
}

fn assert_graph_directedness_rejection(plan: &QueryPlan, algorithm: &str, directed: bool, required_directed: bool) {
    let capability = plan.capability_analysis().nodes.into_iter().find(|node| node.node_id == plan.root_id()).unwrap();
    let CapabilityDecision::Rejected { facts, .. } = capability.executable else {
        panic!("expected graph capability rejection");
    };
    assert_eq!(facts.get("reason").map(String::as_str), Some("unsupported_graph_directedness"));
    assert_eq!(facts.get("algorithm").map(String::as_str), Some(algorithm));
    assert_eq!(facts.get("directed").map(String::as_str), Some(if directed { "true" } else { "false" }));
    assert_eq!(
        facts.get("required_directed").map(String::as_str),
        Some(if required_directed { "true" } else { "false" })
    );
}

fn graph_plan(algorithm: GraphAlgorithm, params: GraphAlgorithmParams, input_schema: Arc<Schema>) -> QueryPlan {
    graph_plan_with_options(algorithm, params, GraphConstructionOptions::default(), input_schema)
}

fn graph_plan_with_options(
    algorithm: GraphAlgorithm,
    params: GraphAlgorithmParams,
    graph_options: GraphConstructionOptions,
    input_schema: Arc<Schema>,
) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let input = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", input_schema)));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        algorithm,
        GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
        graph_options,
        params,
    );
    let graph = builder.add_graph_algorithm(input, spec);
    builder.finish_with_host_sink(graph)
}

fn graph_plan_with_edge_id(input_schema: Arc<Schema>) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let input = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", input_schema)));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", Some("weight")).with_edge_id("edge_id"),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
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

fn edge_schema_with_vertex_nullability(vertex_type: DataType, weight_type: DataType) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("src", vertex_type.clone(), true),
        Field::new("dst", vertex_type, true),
        Field::new("weight", weight_type, false),
    ]))
}

fn edge_schema_with_nullable_edge_id(vertex_type: DataType, weight_type: DataType) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("src", vertex_type.clone(), false),
        Field::new("dst", vertex_type.clone(), false),
        Field::new("weight", weight_type, false),
        Field::new("edge_id", vertex_type, true),
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
    ("personalized_pagerank", "personalized_pagerank"),
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
    ("leiden", "leiden"),
    ("ecg", "ecg"),
    ("spectral_modularity_maximization", "spectral_modularity_maximization"),
    ("balanced_cut_clustering", "balanced_cut_clustering"),
    ("force_atlas2", "force_atlas2"),
];

/// cugraph-rs `Graph` method names that are intentionally absent from the
/// registry today, pinned by stage.
///
/// **Milestone 2** (need auxiliary inputs):
///   hits_with_initial_hubs,
///   betweenness_centrality(_all), edge_betweenness_centrality(_all),
///   vertex_pairs, jaccard/sorensen/overlap/cosine (pair+all_pairs),
///   k_truss_subgraph, extract_ego, k_core_with_core_numbers,
///   analyze_clustering_*, degrees/in_degrees/out_degrees (with vertex selection),
///   select_random_vertices
///
/// **Milestone 3** (stochastic / sampling):
///   uniform_random_walks, biased_random_walks, node2vec_random_walks,
///   homogeneous_*_neighbor_sample, homogeneous_*_temporal_neighbor_sample
///
/// When a new method is added to cugraph-rs and it is not yet implemented,
/// add it here with a comment indicating the planned stage. When it is
/// implemented, move it to `CUGRAPH_RS_METHOD_TO_REGISTRY` above and add
/// the registry entry.
const KNOWN_MISSING_CUGRAPH_RS_METHODS: &[&str] = &[
    // Milestone 2 — auxiliary inputs or variant forms of registered algorithms
    "pagerank_with",
    "personalized_pagerank_with",
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
