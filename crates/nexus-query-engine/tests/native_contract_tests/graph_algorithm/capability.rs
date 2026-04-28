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

#[cfg(not(feature = "cugraph"))]
#[test]
fn feature_disabled_graph_execution_rejects_before_gpu_input_lookup() {
    let plan = graph_plan(
        GraphAlgorithm::PageRank,
        GraphAlgorithmParams::PageRank(Default::default()),
        edge_schema(DataType::Int64, DataType::Float64, false),
    );
    let context = ExecutionContext::builder().build().expect("execution context");

    let error = match execute_query_plan_with_gpu_inputs(&context, &plan, &GpuResidentInputMap::new()) {
        Ok(_) => panic!("feature-disabled graph execution must reject before resolving its input"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::UnsupportedOperator);
    assert_eq!(error.operation(), "nexus_query_engine.graph_algorithm.execute");
    assert_eq!(error.fact_value("reason"), Some("required_feature_disabled"));
    assert_eq!(error.fact_value("required_feature"), Some("cugraph"));
    assert_eq!(error.fact_value("host_fallback"), Some("disabled"));
}

#[cfg(not(feature = "cugraph"))]
#[test]
fn feature_disabled_bounded_graph_execution_rejects_before_gpu_input_lookup() {
    let plan = graph_plan(
        GraphAlgorithm::PageRank,
        GraphAlgorithmParams::PageRank(Default::default()),
        edge_schema(DataType::Int64, DataType::Float64, false),
    );
    let context = ExecutionContext::builder()
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 1,
            max_source_chunk_bytes: 1,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .expect("bounded execution context");

    let error = match execute_query_plan_with_gpu_inputs(&context, &plan, &GpuResidentInputMap::new()) {
        Ok(_) => panic!("feature-disabled bounded graph execution must reject before resolving its input"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::UnsupportedOperator);
    assert_eq!(error.operation(), "nexus_query_engine.graph_algorithm.execute");
    assert_eq!(error.fact_value("reason"), Some("required_feature_disabled"));
    assert_eq!(error.fact_value("required_feature"), Some("cugraph"));
    assert_eq!(error.fact_value("host_fallback"), Some("disabled"));
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
