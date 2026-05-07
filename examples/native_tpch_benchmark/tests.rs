use super::*;

#[test]
fn default_config_selects_all_tpch_queries() {
    let config = NativeTpchBenchmarkConfig::default();
    assert_eq!(config.query_ids.len(), TPCH_QUERIES.len());
    assert_eq!(config.selected_queries().len(), TPCH_QUERIES.len());
    assert_eq!(config.iterations, 1);
    assert_eq!(config.native_memory_policy.as_str(), "bounded");
    config.validate().unwrap();
}

#[test]
fn benchmark_report_header_and_rejected_rows_are_stable() {
    let config =
        NativeTpchBenchmarkConfig { query_ids: vec!["q02".to_string()], ..NativeTpchBenchmarkConfig::default() };
    let metadata = TpchDatasetMetadata {
        root: PathBuf::from("/tmp/tpch"),
        scale_factor: "sf1".to_string(),
        provenance: "external_parquet_root".to_string(),
        generation_command: "dbgen".to_string(),
        table_count: 8,
        file_count: 8,
        total_rows: 1,
        total_bytes: 2,
        metadata_fingerprint: "metadata-fnv1a64:0000000000000001".to_string(),
        content_fingerprint: None,
        tables: Vec::new(),
    };
    let planning = NativeDataFusionPlanningRow {
        query_id: "q02".to_string(),
        physical_root: "SortExec".to_string(),
        physical_shape: "SortExec(HashJoinExec)".to_string(),
        native_candidate_shape: Some("SortExec(HashJoinExec)".to_string()),
        native_candidate_shape_family: Some("sort_hash_join_aggregate".to_string()),
        outcome: NativeDataFusionPlanningOutcome::RejectedBeforeExecution,
        reason_code: Some("native_bounded_join_unsupported".to_string()),
        unsupported_category: Some("operator_hashjoin".to_string()),
        bounded_join_reservation_facts: Default::default(),
        join_diagnostics: Vec::new(),
        source_diagnostics: Vec::new(),
        post_native_plan_root: "SortExec".to_string(),
        post_native_execution_path: NativeDataFusionPostNativeExecutionPath::Cpu,
        rejected_before_execution: true,
    };
    let report =
        NativeTpchBenchmarkReport { metadata, config, rows: vec![NativeTpchBenchmarkRow::rejected("q02", planning)] };

    let text = report.to_tsv();
    assert!(text.starts_with("# native_tpch_benchmark_tsv_schema_version=1\n"));
    assert!(text.contains(&format!("{NATIVE_TPCH_BENCHMARK_TSV_HEADER}\n")));
    assert!(text.contains("# tpch_dataset_scale_factor=sf1\n"));
    assert!(text.contains("# tpch_benchmark_query_ids=q02\n"));
    assert!(text.contains("q02\trejected_before_execution\tnative_bounded_join_unsupported\toperator_hashjoin"));
    assert!(text.contains("\tfalse\tfalse\tnot_started\tnone\tnone\tnot_run\t"));
}
