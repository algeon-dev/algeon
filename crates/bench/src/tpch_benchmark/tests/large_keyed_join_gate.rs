use query_planning::execution_contracts::SourceChunkPlanningMode;

use super::*;

#[test]
fn large_keyed_join_strategy_gate_accepts_skip_cpu_sf100_matrix_report() {
    let report = sf100_matrix_report(vec![q07_large_partitioned_join_row()]);
    let text = report.to_tsv();
    let config = strategy_gate::LargeKeyedJoinStrategyGateConfig {
        enabled: true,
        query_ids: Some(["q03".to_string(), "q04".to_string(), "q07".to_string()].into_iter().collect()),
    };

    let summary = strategy_gate::validate_large_keyed_join_strategy_gate_tsv(&text, &config)
        .expect("large Q7 partitioned report should pass the regression gate without CPU parity");

    assert_eq!(summary.large_partitioned_sites, 1);
    assert_eq!(summary.large_fallback_one_shot_sites, 0);
    assert_eq!(summary.large_reusable_sites, 0);
    assert_eq!(summary.dynamic_filter_evidence.work_avoidance_statuses.get("unknown").copied(), Some(3));
    assert_eq!(summary.dynamic_filter_evidence.applied_without_work_avoidance_count, 0);
    assert_eq!(summary.query_ids, vec!["q03", "q04", "q07"]);
    assert!(summary.progress_line().contains("stage=large_keyed_join_strategy_gate outcome=passed"));
    assert!(summary.progress_line().contains("dynamic_filter_work_avoidance_statuses=unknown:3"));
    assert!(text.contains("\tcpu_failed\t"));
}

#[test]
fn large_keyed_join_strategy_gate_rejects_large_reusable_hash_site() {
    let mut join = q07_large_partitioned_join_row();
    join.join_strategy = "reusable_hash_equi".to_string();
    join.execution_strategy = "bounded_reusable_hash_join".to_string();
    join.hash_join_build_count = 1;
    join.hash_join_hash_table_bytes = 9 * 1024 * 1024 * 1024;
    join.partitioned_hash_join_count = 0;
    join.partition_count = None;
    join.partition_target_bytes = None;
    let report = sf100_matrix_report(vec![join]);
    let config = strategy_gate::LargeKeyedJoinStrategyGateConfig { enabled: true, query_ids: None };

    let error = strategy_gate::validate_large_keyed_join_strategy_gate_tsv(&report.to_tsv(), &config)
        .expect_err("large reusable hash builds must fail the regression gate");

    assert!(error.to_string().contains("large reusable_hash_equi sites"));
    assert!(error.to_string().contains("q07:n7:reusable_hash_equi"));
}

#[test]
fn large_keyed_join_strategy_gate_accepts_local_partitioned_fallback_reason() {
    let report = sf100_matrix_report(vec![q07_large_partitioned_join_row(), q07_large_residual_one_shot_join_row()]);
    let config = strategy_gate::LargeKeyedJoinStrategyGateConfig { enabled: true, query_ids: None };

    let summary = strategy_gate::validate_large_keyed_join_strategy_gate_tsv(&report.to_tsv(), &config)
        .expect("large one-shot fallbacks with local partitioned rejection facts should pass");

    assert_eq!(summary.large_partitioned_sites, 1);
    assert_eq!(summary.large_fallback_one_shot_sites, 1);
    assert_eq!(summary.large_reusable_sites, 0);
}

#[test]
fn large_keyed_join_strategy_gate_accepts_partitioned_mixed_residual_site() {
    let report = sf100_matrix_report(vec![q07_large_partitioned_join_row(), q07_large_partitioned_residual_join_row()]);
    let config = strategy_gate::LargeKeyedJoinStrategyGateConfig { enabled: true, query_ids: None };

    let summary = strategy_gate::validate_large_keyed_join_strategy_gate_tsv(&report.to_tsv(), &config)
        .expect("large partitioned residual sites should count as partitioned join coverage");

    assert_eq!(summary.large_partitioned_sites, 2);
    assert_eq!(summary.large_fallback_one_shot_sites, 0);
    assert_eq!(summary.large_reusable_sites, 0);
}

#[test]
fn large_keyed_join_strategy_gate_rejects_large_one_shot_without_partitioned_fallback_reason() {
    let mut join = q07_large_residual_one_shot_join_row();
    join.partitioned_hash_join_rejected_reason = None;
    let report = sf100_matrix_report(vec![join]);
    let config = strategy_gate::LargeKeyedJoinStrategyGateConfig { enabled: true, query_ids: None };

    let error = strategy_gate::validate_large_keyed_join_strategy_gate_tsv(&report.to_tsv(), &config)
        .expect_err("large one-shot joins need a local partitioned fallback reason");

    assert!(error.to_string().contains("without a local partitioned fallback reason"));
    assert!(error.to_string().contains("q07:n8:one_shot_equi"));
}

fn sf100_matrix_report(q07_join_strategy_rows: Vec<NativeJoinStrategyMetricRow>) -> NativeTpchBenchmarkReport {
    let config = NativeTpchBenchmarkConfig {
        query_ids: vec!["q03".to_string(), "q04".to_string(), "q07".to_string()],
        dataset_mode: NativeTpchDatasetMode::External(NativeTpchScaleFactor::Sf100),
        native_memory_policy: sf100_bounded_memory_policy(),
        ..NativeTpchBenchmarkConfig::default()
    };
    NativeTpchBenchmarkReport {
        metadata: benchmark_sf100_metadata(),
        config,
        devices: Vec::new(),
        rows: vec![
            skip_cpu_accepted_row("q03", NativeDataFusionExecutionMetrics::default()),
            skip_cpu_accepted_row("q04", q04_filtered_join_metrics()),
            skip_cpu_accepted_row(
                "q07",
                NativeDataFusionExecutionMetrics {
                    one_shot_join_count: q07_join_strategy_rows.iter().map(|row| row.one_shot_join_count).sum(),
                    hash_join_build_count: 1,
                    hash_join_hash_table_bytes: 16_779_199,
                    join_strategy_rows: q07_join_strategy_rows,
                    ..Default::default()
                },
            ),
        ],
    }
}

fn sf100_bounded_memory_policy() -> NativeTpchMemoryPolicy {
    NativeTpchMemoryPolicy::bounded_custom(
        NativeTpchDeviceBudget::Explicit { bytes: 94_489_280_512 },
        34_359_738_368,
        1024,
        DEFAULT_NATIVE_MAX_RETRY_ATTEMPTS,
        DEFAULT_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS,
        SourceChunkPlanningMode::LocalParquetThroughput,
    )
}

fn benchmark_sf100_metadata() -> TpchDatasetMetadata {
    benchmark_metadata_for_scale("sf100", 866_037_932, benchmark_sf100_tables())
}

fn benchmark_sf100_tables() -> Vec<TpchTableMetadata> {
    benchmark_tables(&[
        ("customer", 15_000_000),
        ("lineitem", 600_037_902),
        ("nation", 25),
        ("orders", 150_000_000),
        ("part", 20_000_000),
        ("partsupp", 80_000_000),
        ("region", 5),
        ("supplier", 1_000_000),
    ])
}

fn skip_cpu_accepted_row(query_id: &str, metrics: NativeDataFusionExecutionMetrics) -> NativeTpchBenchmarkRow {
    let mut planning = accepted_planning_row(query_id);
    planning.native_candidate_shape_family = Some("sort_hash_join_aggregate".to_string());
    let mut native = timed_native_run(&format!("output-fnv1a64:{:016x}", query_id.as_bytes()[2]));
    native.metrics = metrics;
    NativeTpchBenchmarkRow::accepted(
        query_id,
        planning,
        Err(execution::CpuRunFailure {
            runtime_started: false,
            message: "cpu_skipped_by_CUDF_TPCH_BENCH_SKIP_CPU".to_string(),
            details: Box::default(),
        }),
        Ok(native),
    )
}

fn q04_filtered_join_metrics() -> NativeDataFusionExecutionMetrics {
    NativeDataFusionExecutionMetrics {
        filtered_join_build_count: 1,
        join_strategy_rows: vec![NativeJoinStrategyMetricRow {
            node_id: "n4".to_string(),
            join_strategy: "filtered_join".to_string(),
            execution_strategy: "bounded_reusable_filtered_join".to_string(),
            join_type: "left_semi".to_string(),
            build_side: "left".to_string(),
            key_count: 1,
            estimated_build_rows: Some(600_037_902),
            observed_build_rows: Some(600_037_902),
            build_side_bytes: Some(27_145_989_519),
            filtered_join_build_count: 1,
            filtered_join_hash_table_bytes: 4096,
            ..Default::default()
        }],
        ..Default::default()
    }
}

fn q07_large_partitioned_join_row() -> NativeJoinStrategyMetricRow {
    NativeJoinStrategyMetricRow {
        node_id: "n7".to_string(),
        join_strategy: "partitioned_hash_equi".to_string(),
        execution_strategy: "bounded_partitioned_hash_equi_join".to_string(),
        join_type: "inner".to_string(),
        build_side: "right".to_string(),
        key_count: 1,
        estimated_build_rows: Some(600_037_902),
        observed_build_rows: Some(600_037_902),
        build_side_bytes: Some(27_145_989_519),
        probe_reuse_count: Some(1),
        partition_count: Some(14),
        partition_target_bytes: Some(2 * 1024 * 1024 * 1024),
        build_partition_batch_count: Some(14),
        probe_partition_batch_count: Some(14),
        build_partitioned_bytes: Some(27_145_989_519),
        probe_partitioned_bytes: Some(2_010_410_256),
        reusable_hash_join_rejected_reason: Some("build_side_too_large".to_string()),
        partitioned_hash_join_count: 1,
        join_gather_map_bytes: 4_496_849_296,
        join_output_reservation_bytes: 17_017_774_876,
        join_output_rows: 562_106_162,
        join_output_bytes: 17_017_774_876,
        ..Default::default()
    }
}

fn q07_large_residual_one_shot_join_row() -> NativeJoinStrategyMetricRow {
    NativeJoinStrategyMetricRow {
        node_id: "n8".to_string(),
        join_strategy: "one_shot_equi".to_string(),
        execution_strategy: "bounded_one_shot_equi_join".to_string(),
        join_type: "inner".to_string(),
        build_side: "right".to_string(),
        key_count: 1,
        estimated_build_rows: Some(600_037_902),
        observed_build_rows: Some(600_037_902),
        build_side_bytes: Some(27_145_989_519),
        probe_reuse_count: Some(1),
        reusable_hash_join_rejected_reason: Some("build_side_too_large".to_string()),
        partitioned_hash_join_rejected_reason: Some("partitioned_hash_equi_unsupported".to_string()),
        one_shot_join_count: 1,
        one_shot_join_index_bytes: 3_033_771_680,
        one_shot_join_materialization_bytes: 17_017_774_876,
        one_shot_join_output_bytes: 17_017_774_876,
        join_gather_map_bytes: 4_496_849_296,
        join_output_reservation_bytes: 17_017_774_876,
        join_output_rows: 562_106_162,
        join_output_bytes: 17_017_774_876,
        ..Default::default()
    }
}

fn q07_large_partitioned_residual_join_row() -> NativeJoinStrategyMetricRow {
    NativeJoinStrategyMetricRow {
        node_id: "n8".to_string(),
        join_strategy: "partitioned_mixed_keyed_residual".to_string(),
        execution_strategy: "bounded_partitioned_mixed_keyed_residual_join".to_string(),
        join_type: "inner".to_string(),
        build_side: "right".to_string(),
        key_count: 1,
        estimated_build_rows: Some(600_037_902),
        observed_build_rows: Some(600_037_902),
        build_side_bytes: Some(27_145_989_519),
        probe_reuse_count: Some(1),
        partition_count: Some(14),
        partition_target_bytes: Some(2 * 1024 * 1024 * 1024),
        build_partition_batch_count: Some(14),
        probe_partition_batch_count: Some(14),
        build_partitioned_bytes: Some(27_145_989_519),
        probe_partitioned_bytes: Some(2_010_410_256),
        reusable_hash_join_rejected_reason: Some("build_side_too_large".to_string()),
        partitioned_hash_join_count: 1,
        mixed_join_size_data_bytes: 3_033_771_680,
        join_gather_map_bytes: 4_496_849_296,
        join_output_reservation_bytes: 17_017_774_876,
        join_output_rows: 562_106_162,
        join_output_bytes: 17_017_774_876,
        ..Default::default()
    }
}
