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
