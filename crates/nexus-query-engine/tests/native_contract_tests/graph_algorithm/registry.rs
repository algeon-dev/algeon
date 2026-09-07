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
        path_output: None,
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
fn similarity_ir_records_explicit_pair_side_input_and_fixed_schema() {
    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
        "edges",
        edge_schema(DataType::Int32, DataType::Float32, false),
    )));
    let pairs = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
        "pairs",
        Arc::new(Schema::new(vec![
            Field::new("first", DataType::Int32, false),
            Field::new("second", DataType::Int32, false),
        ])),
    )));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Jaccard,
        GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
        GraphConstructionOptions { directed: false, policy: GraphConstructionPolicy::RawLibcugraph, ..Default::default() },
        GraphAlgorithmParams::Similarity(GraphSimilarityParams::new(GraphVertexPairInput::new(
            pairs, "first", "second",
        ))),
    );
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan = builder.finish_with_host_sink(graph);

    let PlanNode::GraphAlgorithm(graph) = &plan.root_node().kind else {
        panic!("expected graph algorithm root");
    };
    assert_eq!(graph.inputs, vec![edges, pairs]);
    assert_eq!(plan.root_node().input_ids(), &[edges, pairs]);
    assert_eq!(
        graph.output_schema().fields().iter().map(|field| field.as_ref().clone()).collect::<Vec<_>>(),
        vec![
            Field::new("first", DataType::Int64, false),
            Field::new("second", DataType::Int64, false),
            Field::new("similarity", DataType::Float64, false),
        ],
    );
    let display = plan.to_string();
    assert!(display.contains("side_inputs=n1"));
    assert!(display.contains("params=vertex_pairs_node=n1:first_column=first:second_column=second"));
}

#[test]
fn similarity_registry_contract_is_shared_across_metrics() {
    for algorithm in [
        GraphAlgorithm::Jaccard,
        GraphAlgorithm::Sorensen,
        GraphAlgorithm::Overlap,
        GraphAlgorithm::Cosine,
    ] {
        let definition = algorithm.definition();
        assert!(algorithm.is_similarity());
        assert_eq!(definition.required_graph_directedness(), Some(false));
        assert_eq!(definition.weight_policy(), GraphWeightPolicy::Optional);
        assert!(definition.weight_policy().consumes_weight());
        assert_eq!(definition.empty_input_policy.edge_input.as_str(), "similarity_vertex_pairs");
        assert_eq!(definition.empty_input_policy.side_inputs.as_str(), "require_vertex_pairs");
        assert!(definition.default_params().supports_algorithm(algorithm));
        assert_eq!(
            stable_graph_result_schema(algorithm).fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>(),
            vec!["first", "second", "similarity"],
        );
    }
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
        path_output: None,
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

    let capability = plan.capability_analysis(nexus_query_engine::rapids_backend::rapids_analysis_context()).nodes.into_iter().find(|node| node.node_id == plan.root_id()).unwrap();
    let CapabilityDecision::Rejected { facts, .. } = capability.executable else {
        panic!("expected graph capability rejection");
    };
    assert_eq!(facts.get("missing_binding").map(String::as_str), Some("edge_id_col"));
}

#[test]
fn graph_operation_registry_pins_empty_input_policy_for_every_algorithm() {
    let policies = GRAPH_OPERATION_REGISTRY
        .iter()
        .map(|definition| {
            (
                definition.name,
                definition.empty_input_policy.edge_input.as_str(),
                definition.empty_input_policy.side_inputs.as_str(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        policies,
        vec![
            ("pagerank", "return_empty", "none"),
            ("bfs", "bfs_endpoint_policy", "bfs_predicates"),
            ("louvain", "return_empty", "none"),
            ("eigenvector_centrality", "return_empty", "none"),
            ("katz_centrality", "return_empty", "none"),
            ("hits", "return_empty", "none"),
            ("betweenness_centrality", "validate_selectors_then_return_empty", "none"),
            ("edge_betweenness_centrality", "validate_selectors_then_return_empty", "none"),
            ("sssp", "reject", "none"),
            ("degrees_all", "return_empty", "none"),
            ("in_degrees_all", "return_empty", "none"),
            ("out_degrees_all", "return_empty", "none"),
            ("weakly_connected_components", "return_empty", "none"),
            ("strongly_connected_components", "return_empty", "none"),
            ("triangle_count_all", "return_empty", "none"),
            ("core_number", "return_empty", "none"),
            ("k_core", "return_empty", "none"),
            ("minimum_spanning_tree", "return_empty", "none"),
            ("personalized_pagerank", "reject", "require_personalization"),
            ("leiden", "return_empty", "none"),
            ("ecg", "return_empty", "none"),
            ("spectral_modularity_maximization", "reject", "none"),
            ("force_atlas2", "return_empty", "none"),
            ("jaccard", "similarity_vertex_pairs", "require_vertex_pairs"),
            ("sorensen", "similarity_vertex_pairs", "require_vertex_pairs"),
            ("overlap", "similarity_vertex_pairs", "require_vertex_pairs"),
            ("cosine", "similarity_vertex_pairs", "require_vertex_pairs"),
        ]
    );
    assert_eq!(policies.len(), GraphAlgorithm::all().len());
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
            (GraphAlgorithm::ForceAtlas2, "force_atlas2", GraphLayoutRequirement::Any),
            (GraphAlgorithm::Jaccard, "jaccard", GraphLayoutRequirement::Any),
            (GraphAlgorithm::Sorensen, "sorensen", GraphLayoutRequirement::Any),
            (GraphAlgorithm::Overlap, "overlap", GraphLayoutRequirement::Any),
            (GraphAlgorithm::Cosine, "cosine", GraphLayoutRequirement::Any),
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
        assert!(
            def.default_params().supports_algorithm(def.algorithm),
            "descriptor default params must support the registry algorithm variant",
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
                false,
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
                false,
            ),
            (
                GraphAlgorithm::Ecg,
                GraphSourceVertexRequirement::None,
                GraphWeightPolicy::Ignored,
                GraphOutputSchemaKind::Fixed,
                false,
            ),
            (
                GraphAlgorithm::SpectralModularityMaximization,
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
            (
                GraphAlgorithm::Jaccard,
                GraphSourceVertexRequirement::None,
                GraphWeightPolicy::Optional,
                GraphOutputSchemaKind::Fixed,
                false,
            ),
            (
                GraphAlgorithm::Sorensen,
                GraphSourceVertexRequirement::None,
                GraphWeightPolicy::Optional,
                GraphOutputSchemaKind::Fixed,
                false,
            ),
            (
                GraphAlgorithm::Overlap,
                GraphSourceVertexRequirement::None,
                GraphWeightPolicy::Optional,
                GraphOutputSchemaKind::Fixed,
                false,
            ),
            (
                GraphAlgorithm::Cosine,
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

#[cfg(feature = "cugraph")]
#[test]
fn graph_operation_registry_has_cugraph_public_contracts() {
    // Keep the dependency's two graph orientations in this compile-time
    // contract. No source-tree layout outside this workspace is assumed.
    let _: Option<cugraph::Graph<cugraph::Forward>> = None;
    let _: Option<cugraph::Graph<cugraph::Transposed>> = None;

    // Method-item references are compile-time contracts against the local `cugraph` crate's
    // public API. They intentionally avoid parsing its implementation source,
    // while still failing if an execution adapter's entry point is
    // renamed or removed.
    let _ = cugraph::Graph::<cugraph::Transposed>::pagerank;
    let _ = cugraph::Graph::<cugraph::Transposed>::personalized_pagerank::<i32, f32>;
    let _ = cugraph::Graph::<cugraph::Forward>::louvain;
    let _ = cugraph::Graph::<cugraph::Transposed>::eigenvector_centrality;
    let _ = cugraph::Graph::<cugraph::Transposed>::katz_centrality;
    let _ = cugraph::Graph::<cugraph::Transposed>::hits;
    let _ = cugraph::Graph::<cugraph::Forward>::sssp::<i32>;
    let _ = cugraph::Graph::<cugraph::Transposed>::degrees_all;
    let _ = cugraph::Graph::<cugraph::Transposed>::in_degrees_all;
    let _ = cugraph::Graph::<cugraph::Forward>::out_degrees_all;
    let _ = cugraph::Graph::<cugraph::Forward>::weakly_connected_components;
    let _ = cugraph::Graph::<cugraph::Forward>::strongly_connected_components;
    let _ = cugraph::Graph::<cugraph::Forward>::triangle_count_all;
    let _ = cugraph::Graph::<cugraph::Forward>::core_number;
    let _ = cugraph::Graph::<cugraph::Forward>::k_core;
    let _ = cugraph::Graph::<cugraph::Forward>::minimum_spanning_tree;
    let _ = cugraph::Graph::<cugraph::Forward>::leiden;
    let _ = cugraph::Graph::<cugraph::Forward>::ecg;
    let _ = cugraph::Graph::<cugraph::Forward>::spectral_modularity_maximization;
    let _ = cugraph::Graph::<cugraph::Forward>::force_atlas2;
    let _ = cugraph::Graph::<cugraph::Forward>::jaccard_coefficients;
    let _ = cugraph::Graph::<cugraph::Forward>::sorensen_coefficients;
    let _ = cugraph::Graph::<cugraph::Forward>::overlap_coefficients;
    let _ = cugraph::Graph::<cugraph::Forward>::cosine_similarity_coefficients;

    let registry_names: std::collections::BTreeSet<&str> =
        GRAPH_OPERATION_REGISTRY.iter().map(|definition| definition.name).collect();
    let missing =
        CUGRAPH_REGISTRY_NAMES.iter().copied().filter(|name| !registry_names.contains(name)).collect::<Vec<_>>();
    assert!(missing.is_empty(), "cuGraph registry contract names missing from GRAPH_OPERATION_REGISTRY: {missing:?}");
}
