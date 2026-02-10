use super::*;

#[test]
fn large_keyed_join_strategy_gate_accepts_skip_cpu_sf100_matrix_report() {
    let report = sf100_matrix_report(vec![q07_large_one_shot_join_row()]);
    let text = report.to_tsv();
    let config = strategy_gate::LargeKeyedJoinStrategyGateConfig {
        enabled: true,
        query_ids: Some(["q03".to_string(), "q04".to_string(), "q07".to_string()].into_iter().collect()),
    };

    let summary = strategy_gate::validate_large_keyed_join_strategy_gate_tsv(&text, &config)
        .expect("large Q7 one-shot report should pass the regression gate without CPU parity");

    assert_eq!(summary.large_one_shot_sites, 1);
    assert_eq!(summary.large_reusable_sites, 0);
    assert_eq!(summary.reservation_denied_count, 0);
    assert_eq!(summary.query_ids, vec!["q03", "q04", "q07"]);
    assert!(summary.progress_line().contains("phase=large_keyed_join_strategy_gate outcome=passed"));
    assert!(text.contains("\tcpu_failed\t"));
}

#[test]
fn large_keyed_join_strategy_gate_rejects_large_reusable_hash_site() {
    let mut join = q07_large_one_shot_join_row();
    join.join_strategy = "reusable_hash_equi".to_string();
    join.execution_strategy = "bounded_reusable_hash_join".to_string();
    join.hash_join_build_count = 1;
    join.hash_join_hash_table_bytes = 9 * 1024 * 1024 * 1024;
    join.one_shot_join_count = 0;
    join.one_shot_join_index_bytes = 0;
    let report = sf100_matrix_report(vec![join]);
    let config = strategy_gate::LargeKeyedJoinStrategyGateConfig { enabled: true, query_ids: None };

    let error = strategy_gate::validate_large_keyed_join_strategy_gate_tsv(&report.to_tsv(), &config)
        .expect_err("large reusable hash builds must fail the regression gate");

    assert!(error.to_string().contains("large reusable_hash_equi sites"));
    assert!(error.to_string().contains("q07:n7:reusable_hash_equi"));
}

#[test]
fn large_keyed_join_strategy_gate_rejects_reservation_denials() {
    let mut report = sf100_matrix_report(vec![q07_large_one_shot_join_row()]);
    let q07 = report.rows.iter_mut().find(|row| row.query_id == "q07").unwrap();
    q07.metrics.reservation_denied_count = 1;
    let config = strategy_gate::LargeKeyedJoinStrategyGateConfig { enabled: true, query_ids: None };

    let error = strategy_gate::validate_large_keyed_join_strategy_gate_tsv(&report.to_tsv(), &config)
        .expect_err("reservation denials must fail the regression gate");

    assert!(error.to_string().contains("reservation_denied_count=0"));
    assert!(error.to_string().contains("observed 1"));
}

#[test]
fn large_keyed_join_strategy_gate_can_select_q7_from_larger_report() {
    let report = sf100_matrix_report(vec![q07_large_one_shot_join_row()]);
    let config = strategy_gate::LargeKeyedJoinStrategyGateConfig {
        enabled: true,
        query_ids: Some(["q07".to_string()].into_iter().collect()),
    };

    let summary = strategy_gate::validate_large_keyed_join_strategy_gate_tsv(&report.to_tsv(), &config).unwrap();

    assert_eq!(summary.query_ids, vec!["q07"]);
    assert_eq!(summary.large_one_shot_sites, 1);
}

fn sf100_matrix_report(q07_join_strategy_rows: Vec<NativeJoinStrategyMetricRow>) -> NativeTpchBenchmarkReport {
    let config = NativeTpchBenchmarkConfig {
        query_ids: vec!["q03".to_string(), "q04".to_string(), "q07".to_string()],
        dataset_mode: NativeTpchDatasetMode::External(NativeTpchScaleFactor::Sf100),
        native_memory_policy: NativeTpchMemoryPolicy::Bounded {
            device_budget_bytes: 94_489_280_512,
            max_source_chunk_bytes: 34_359_738_368,
            max_row_groups_per_chunk: 1024,
            max_retry_attempts: DEFAULT_NATIVE_MAX_RETRY_ATTEMPTS,
            min_retry_chunk_row_groups: DEFAULT_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS,
        },
        ..NativeTpchBenchmarkConfig::default()
    };
    NativeTpchBenchmarkReport {
        metadata: benchmark_sf100_metadata(),
        config,
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
            details: execution::FailureDetails::default(),
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

fn q07_large_one_shot_join_row() -> NativeJoinStrategyMetricRow {
    NativeJoinStrategyMetricRow {
        node_id: "n7".to_string(),
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
        one_shot_join_count: 1,
        one_shot_join_index_bytes: 3_033_771_680,
        one_shot_join_materialization_bytes: 17_017_774_876,
        one_shot_join_output_bytes: 17_017_774_876,
        join_gather_map_bytes: 4_496_849_296,
        ..Default::default()
    }
}
