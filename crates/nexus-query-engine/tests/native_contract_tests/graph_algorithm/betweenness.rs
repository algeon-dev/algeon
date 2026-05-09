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
                construction_sort_workspace_bound_bytes: None,
                pair_rows: None,
                pair_vertex_width_bytes: None,
                pair_cardinality_source: None,
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
                construction_sort_workspace_bound_bytes: None,
                pair_rows: None,
                pair_vertex_width_bytes: None,
                pair_cardinality_source: None,
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
                construction_sort_workspace_bound_bytes: None,
                pair_rows: None,
                pair_vertex_width_bytes: None,
                pair_cardinality_source: None,
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
        construction_sort_workspace_bound_bytes: None,
        pair_rows: None,
        pair_vertex_width_bytes: None,
        pair_cardinality_source: None,
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
        construction_sort_workspace_bound_bytes: None,
        pair_rows: None,
        pair_vertex_width_bytes: None,
        pair_cardinality_source: None,
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
        construction_sort_workspace_bound_bytes: None,
        pair_rows: None,
        pair_vertex_width_bytes: None,
        pair_cardinality_source: None,
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
        construction_sort_workspace_bound_bytes: None,
        pair_rows: None,
        pair_vertex_width_bytes: None,
        pair_cardinality_source: None,
    };

    let estimate = GraphMemoryModel.estimate(&spec, input).unwrap();

    assert_eq!(estimate.requested_bytes, 16_052_517_586);
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
        construction_sort_workspace_bound_bytes: None,
        pair_rows: None,
        pair_vertex_width_bytes: None,
        pair_cardinality_source: None,
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
        construction_sort_workspace_bound_bytes: None,
        pair_rows: None,
        pair_vertex_width_bytes: None,
        pair_cardinality_source: None,
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
        construction_sort_workspace_bound_bytes: None,
        pair_rows: None,
        pair_vertex_width_bytes: None,
        pair_cardinality_source: None,
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
