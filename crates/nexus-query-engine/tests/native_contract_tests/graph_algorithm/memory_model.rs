#[test]
fn graph_memory_classes_pin_phase_f3_algorithm_tiers() {
    assert_eq!(graph_memory_class(GraphAlgorithm::PageRank), GraphMemoryClass::OutputSized);
    assert_eq!(graph_memory_class(GraphAlgorithm::Bfs), GraphMemoryClass::OutputSized);
    assert_eq!(graph_memory_class(GraphAlgorithm::MinimumSpanningTree), GraphMemoryClass::OutputSized);
    for algorithm in
        [GraphAlgorithm::Jaccard, GraphAlgorithm::Sorensen, GraphAlgorithm::Overlap, GraphAlgorithm::Cosine]
    {
        assert_eq!(graph_memory_class(algorithm), GraphMemoryClass::PairOutputSized);
        assert!(graph_memory_reservation_required(algorithm));
    }
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
fn similarity_memory_model_requires_pair_evidence_and_prices_each_pair_relation_row() {
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Jaccard,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: false, renumber: true, policy: GraphConstructionPolicy::PythonCugraph },
        GraphAlgorithmParams::Similarity(GraphSimilarityParams::default()),
    );
    let input = GraphMemoryModelInput {
        raw_input_edges: None,
        input_edges: 20,
        vertex_width_bytes: 8,
        weight_width_bytes: None,
        has_edge_id: false,
        device_total_memory_bytes: None,
        vertex_estimate: None,
        statistics: None,
        edge_normalize_inserted: false,
        pair_rows: None,
        pair_vertex_width_bytes: None,
        pair_cardinality_source: None,
    };
    let missing = GraphMemoryModel.estimate(&spec, input).unwrap_err();
    assert_eq!(missing.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(missing.fact_value("reason"), Some("missing_pair_cardinality_evidence"));

    let estimate = GraphMemoryModel
        .estimate(&spec, input.with_similarity_pair_input(100, 8, GraphPairCardinalitySource::RuntimeMaterialized))
        .unwrap();
    assert_eq!(estimate.class, GraphMemoryClass::PairOutputSized);
    assert_eq!(estimate.pair_rows, Some(100));
    assert_eq!(estimate.pair_vertex_width_bytes, Some(8));
    assert_eq!(estimate.pair_cardinality_source, Some(GraphPairCardinalitySource::RuntimeMaterialized));
    assert_eq!(estimate.pair_cardinality_admission_mode, Some(GraphPairCardinalityAdmissionMode::Unbounded));
    // 100 rows * (two 8-byte pair columns + two native pair columns + an
    // 8-byte score + a 28-byte Int64 unweighted workspace term + normalized
    // `Int64, Int64, Float64` output) plus the calibrated fixed workspace.
    assert_eq!(estimate.execution_bytes, 120 * 1024 + 9_200);
    assert_eq!(estimate.requested_bytes, estimate.construction_bytes + estimate.execution_bytes);
}

#[test]
fn bounded_similarity_admission_accepts_a_frontend_proved_gpu_pair_upper_bound() {
    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(
        GpuResidentSource::new(
            "edges",
            Arc::new(Schema::new(vec![
                Field::new("src", DataType::Int64, false),
                Field::new("dst", DataType::Int64, false),
            ])),
        )
        .with_row_count_upper_bound(20),
    ));
    let pairs = builder.add_source(SourceNode::GpuResident(
        GpuResidentSource::new(
            "pairs",
            Arc::new(Schema::new(vec![
                Field::new("first", DataType::Int64, false),
                Field::new("second", DataType::Int64, false),
            ])),
        )
        .with_row_count_upper_bound(13),
    ));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Jaccard,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: false, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::Similarity(GraphSimilarityParams::new(GraphVertexPairInput::new(
            pairs, "first", "second",
        ))),
    );
    let graph = builder.add_graph_algorithm(edges, spec);
    let plan = builder.finish_with_host_sink(graph);
    let estimates = graph_memory_estimates_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None).unwrap();
    assert_eq!(estimates.len(), 1);
    assert_eq!(estimates[0].estimate.pair_rows, Some(13));
    assert_eq!(
        estimates[0].estimate.pair_cardinality_source,
        Some(GraphPairCardinalitySource::PlanHardUpperBound)
    );
    assert_eq!(
        estimates[0].estimate.pair_cardinality_admission_mode,
        Some(GraphPairCardinalityAdmissionMode::Bounded)
    );
}

#[test]
fn similarity_memory_model_pins_2026_07_25_medium_and_fixed_floor_cases() {
    // Captured with the same isolated sparse-uniform calibration command as
    // the large-case matrix. The 10k rows are a linear-regime check; the
    // 16-row case intentionally remains fixed-floor dominated.
    for (vertex_width_bytes, pair_rows, observed_peak_bytes, should_fit_band) in
        [(4, 10_000, 521_719, true), (8, 10_000, 841_887, true), (4, 16, 2_551, false)]
    {
        let spec = GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::Jaccard,
            GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            GraphConstructionOptions {
                directed: false,
                renumber: true,
                policy: GraphConstructionPolicy::RawLibcugraph,
            },
            GraphAlgorithmParams::Similarity(GraphSimilarityParams::default()),
        );
        let estimate = GraphMemoryModel
            .estimate(
                &spec,
                GraphMemoryModelInput {
                    raw_input_edges: None,
                    input_edges: 16,
                    vertex_width_bytes,
                    weight_width_bytes: None,
                    has_edge_id: false,
                    device_total_memory_bytes: None,
                    vertex_estimate: None,
                    statistics: None,
                    edge_normalize_inserted: false,
                    pair_rows: None,
                    pair_vertex_width_bytes: None,
                    pair_cardinality_source: None,
                }
                .with_similarity_pair_input(
                    pair_rows,
                    vertex_width_bytes,
                    GraphPairCardinalitySource::RuntimeMaterialized,
                ),
            )
            .unwrap();
        assert!(
            estimate.requested_bytes >= observed_peak_bytes,
            "width={vertex_width_bytes}, pairs={pair_rows}: ask={} must cover observed={observed_peak_bytes}",
            estimate.requested_bytes,
        );
        if should_fit_band {
            assert!(
                estimate.requested_bytes <= observed_peak_bytes.saturating_mul(3).saturating_div(2),
                "width={vertex_width_bytes}, pairs={pair_rows}: ask={} must stay within the calibrated 1.50 band over observed={observed_peak_bytes}",
                estimate.requested_bytes,
            );
            assert!(
                estimate.requested_bytes >= observed_peak_bytes.saturating_mul(11).saturating_div(10),
                "width={vertex_width_bytes}, pairs={pair_rows}: ask={} must retain the calibrated 1.10 safety band over observed={observed_peak_bytes}",
                estimate.requested_bytes,
            );
        } else {
            assert!(
                estimate.requested_bytes > observed_peak_bytes.saturating_mul(3).saturating_div(2),
                "the named fixed workspace floor is the only allowed small-case excess",
            );
        }
    }
}

#[test]
fn similarity_memory_model_pins_2026_07_25_runtime_calibration_bands() {
    // Captured with `nexus-tools cugraph-similarity-calibrate` on 2026-07-25
    // using 100,000 candidate pairs on the sparse-uniform topology:
    //
    //   --topology sparse-uniform --pair-rows 100000
    //
    // The companion hub-skewed cold/warm captures remain below these asks;
    // sparse is the observed high-water topology. These cases pin the
    // calibrated per-pair terms above: no observed peak may exceed the ask,
    // and medium/large asks stay inside the 1.10-1.50 admission band.
    for (vertex_width_bytes, weighted, observed_peak_bytes) in
        [(4, false, 5_201_975), (8, false, 8_402_143), (4, true, 10_414_167), (8, true, 13_614_303)]
    {
        let spec = GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::Jaccard,
            GraphEdgeColumnBindings::new("src", "dst", weighted.then_some("weight")),
            GraphConstructionOptions {
                directed: false,
                renumber: true,
                policy: GraphConstructionPolicy::RawLibcugraph,
            },
            GraphAlgorithmParams::Similarity(GraphSimilarityParams::default()),
        );
        let estimate = GraphMemoryModel
            .estimate(
                &spec,
                GraphMemoryModelInput {
                    raw_input_edges: None,
                    input_edges: 16,
                    vertex_width_bytes,
                    weight_width_bytes: weighted.then_some(8),
                    has_edge_id: false,
                    device_total_memory_bytes: None,
                    vertex_estimate: None,
                    statistics: None,
                    edge_normalize_inserted: false,
                    pair_rows: None,
                    pair_vertex_width_bytes: None,
                    pair_cardinality_source: None,
                }
                .with_similarity_pair_input(
                    100_000,
                    vertex_width_bytes,
                    GraphPairCardinalitySource::RuntimeMaterialized,
                ),
            )
            .unwrap();
        assert!(
            estimate.requested_bytes >= observed_peak_bytes,
            "width={vertex_width_bytes}, weighted={weighted}: ask={} must cover observed={observed_peak_bytes}",
            estimate.requested_bytes,
        );
        assert!(
            estimate.requested_bytes <= observed_peak_bytes.saturating_mul(3).saturating_div(2),
            "width={vertex_width_bytes}, weighted={weighted}: ask={} must stay within the calibrated 1.50 band over observed={observed_peak_bytes}",
            estimate.requested_bytes,
        );
        assert!(
            estimate.requested_bytes >= observed_peak_bytes.saturating_mul(11).saturating_div(10),
            "width={vertex_width_bytes}, weighted={weighted}: ask={} must retain the calibrated 1.10 safety band over observed={observed_peak_bytes}",
            estimate.requested_bytes,
        );
    }
}

#[test]
fn similarity_memory_model_pins_all_unweighted_metric_captures() {
    // Each case was captured through the shared native executor with
    // `nexus-tools cugraph-similarity-calibrate --pair-rows 100000` on
    // 2026-07-25. They deliberately span both supported vertex widths,
    // sparse/hub topology, and graph construction versus execution-only
    // estimates. The capture uses the construction-inclusive report because
    // this standalone calibration has no retained-graph admission path.
    for (algorithm, vertex_width_bytes, observed_peak_bytes) in [
        (GraphAlgorithm::Jaccard, 4, 5_201_975),
        (GraphAlgorithm::Sorensen, 8, 7_201_623),
        (GraphAlgorithm::Overlap, 4, 5_201_975),
        (GraphAlgorithm::Cosine, 8, 8_402_143),
    ] {
        let spec = GraphAlgorithmSpec::with_stable_schema(
            algorithm,
            GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            GraphConstructionOptions {
                directed: false,
                renumber: true,
                policy: GraphConstructionPolicy::RawLibcugraph,
            },
            GraphAlgorithmParams::Similarity(GraphSimilarityParams::default()),
        );
        let estimate = GraphMemoryModel
            .estimate(
                &spec,
                GraphMemoryModelInput {
                    raw_input_edges: None,
                    input_edges: 16,
                    vertex_width_bytes,
                    weight_width_bytes: None,
                    has_edge_id: false,
                    device_total_memory_bytes: None,
                    vertex_estimate: None,
                    statistics: None,
                    edge_normalize_inserted: false,
                    pair_rows: None,
                    pair_vertex_width_bytes: None,
                    pair_cardinality_source: None,
                }
                .with_similarity_pair_input(
                    100_000,
                    vertex_width_bytes,
                    GraphPairCardinalitySource::RuntimeMaterialized,
                ),
            )
            .unwrap();
        assert!(
            estimate.requested_bytes >= observed_peak_bytes,
            "algorithm={}: ask={} must cover observed={observed_peak_bytes}",
            algorithm.as_str(),
            estimate.requested_bytes,
        );
        assert!(
            estimate.requested_bytes <= observed_peak_bytes.saturating_mul(3).saturating_div(2),
            "algorithm={}: ask={} must stay within the calibrated 1.50 band over observed={observed_peak_bytes}",
            algorithm.as_str(),
            estimate.requested_bytes,
        );
        assert!(
            estimate.requested_bytes >= observed_peak_bytes.saturating_mul(11).saturating_div(10),
            "algorithm={}: ask={} must retain the calibrated 1.10 safety band over observed={observed_peak_bytes}",
            algorithm.as_str(),
            estimate.requested_bytes,
        );
    }
}

#[test]
fn similarity_memory_model_pins_representative_citation_fixture_capture() {
    // Captured on 2026-07-25 with the repository-owned citation-network
    // fixture, rather than a synthetic topology:
    //
    //   flock /tmp/cudf-gpu.lock cargo run -p nexus-tools \
    //     --features cugraph-similarity-calibrate -- \
    //     cugraph-similarity-calibrate --algorithm jaccard \
    //     --pair-rows 100000 --weighted --vertex-dtype int64 \
    //     --graph-cache-state cold \
    //     --edge-parquet fixture/graph/demo/citation_network/v1/parquet/edges_by_src.parquet \
    //     --edge-row-groups 1
    //
    // The selected row group contains 1,048,576 directed edges; calibration
    // supplies their reverse relation explicitly for raw-libcugraph, so the
    // native input has 2,097,152 edge rows. The runtime-observation peak was
    // 121,728,623 bytes. This pins a realistic local fixture in addition to
    // the synthetic topology matrix without folding its lower/higher graph
    // construction profile into an unexplained global multiplier.
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Jaccard,
        GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
        GraphConstructionOptions {
            directed: false,
            renumber: true,
            policy: GraphConstructionPolicy::RawLibcugraph,
        },
        GraphAlgorithmParams::Similarity(GraphSimilarityParams::default()),
    );
    let estimate = GraphMemoryModel
        .estimate(
            &spec,
            GraphMemoryModelInput {
                raw_input_edges: None,
                input_edges: 2_097_152,
                vertex_width_bytes: 8,
                weight_width_bytes: Some(8),
                has_edge_id: false,
                device_total_memory_bytes: None,
                vertex_estimate: None,
                statistics: None,
                edge_normalize_inserted: false,
                pair_rows: None,
                pair_vertex_width_bytes: None,
                pair_cardinality_source: None,
            }
            .with_similarity_pair_input(100_000, 8, GraphPairCardinalitySource::RuntimeMaterialized),
        )
        .unwrap();
    let observed_peak_bytes = 121_728_623;
    assert_eq!(estimate.requested_bytes, 166_017_824);
    assert!(estimate.requested_bytes >= observed_peak_bytes);
    assert!(
        estimate.requested_bytes <= observed_peak_bytes.saturating_mul(3).saturating_div(2),
        "representative fixture ask must stay within the 1.50 calibration band"
    );
    assert!(
        estimate.requested_bytes >= observed_peak_bytes.saturating_mul(11).saturating_div(10),
        "representative fixture ask must retain the 1.10 calibration safety band"
    );
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
                pair_rows: None,
                pair_vertex_width_bytes: None,
                pair_cardinality_source: None,
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

    let strict = CapabilityContractReport::from_capability_analysis(
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
        pair_rows: None,
        pair_vertex_width_bytes: None,
        pair_cardinality_source: None,
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
                pair_rows: None,
                pair_vertex_width_bytes: None,
                pair_cardinality_source: None,
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
        pair_rows: None,
        pair_vertex_width_bytes: None,
        pair_cardinality_source: None,
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
        pair_rows: None,
        pair_vertex_width_bytes: None,
        pair_cardinality_source: None,
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
        pair_rows: None,
        pair_vertex_width_bytes: None,
        pair_cardinality_source: None,
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
        pair_rows: None,
        pair_vertex_width_bytes: None,
        pair_cardinality_source: None,
    };

    let estimate = GraphMemoryModel.estimate(&spec, input).unwrap();

    assert_eq!(estimate.vertex_estimate, 182_256_596);
    assert_eq!(estimate.requested_bytes, 33_535_213_664);
    assert!(
        estimate.requested_bytes < 101_449_662_464 / 2,
        "plan-time triangle fallback should remain a packing hint, not become a caller-selected grant target"
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
        pair_rows: None,
        pair_vertex_width_bytes: None,
        pair_cardinality_source: None,
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
        pair_rows: None,
        pair_vertex_width_bytes: None,
        pair_cardinality_source: None,
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
        pair_rows: None,
        pair_vertex_width_bytes: None,
        pair_cardinality_source: None,
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
                pair_rows: None,
                pair_vertex_width_bytes: None,
                pair_cardinality_source: None,
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
    assert_eq!(report.measured_peak_bytes, Some(requested_bytes));
    assert_eq!(
        report.graph_model_error_ratio.unwrap().as_decimal_string(),
        "1.000",
        "model error ratio must use the raw model ask, not the bounded reservation floor"
    );
    let json = report.to_json();
    assert_eq!(json["measured_peak_bytes"], requested_bytes);
    assert_eq!(json["graph_model_error_ratio"], "1.000");
}
