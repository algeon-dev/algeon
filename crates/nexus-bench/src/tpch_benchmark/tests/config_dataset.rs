use std::fs;

use super::*;

#[test]
fn default_config_selects_all_tpch_queries() {
    let config = NativeTpchBenchmarkConfig::default();
    assert_eq!(config.query_ids.len(), TPCH_QUERIES.len());
    assert_eq!(config.selected_queries().len(), TPCH_QUERIES.len());
    assert_eq!(config.iterations, 1);
    assert_eq!(config.dataset_mode.as_str(), "sf1_external");
    assert_eq!(NativeTpchDatasetMode::External(NativeTpchScaleFactor::Sf10).as_str(), "sf10_external");
    assert_eq!(config.native_memory_policy.as_str(), "bounded");
    assert_eq!(config.native_memory_policy.profile_name, TPCH_NATIVE_PROFILE_CONSERVATIVE);
    assert_eq!(config.native_memory_policy.profile_source, "default");
    assert_eq!(
        config.native_memory_policy.source_chunk_planning_mode,
        nexus_query_engine::runtime::SourceChunkPlanningMode::Conservative
    );
    assert_eq!(config.native_execution, crate::report_workload::RelationalExecutionSelection::strict_native());
    config.validate().unwrap();
}

#[test]
fn native_planning_uses_the_strict_final_plan_policy_frozen_in_benchmark_config() {
    let config = NativeTpchBenchmarkConfig::default();
    let resources = NativeTpchRuntimeResources::unavailable();

    let optimizer = execution::native_optimizer_config(&config, &resources);

    assert_eq!(
        optimizer.datafusion_cpu_execution_policy(),
        datafusion_nexus::planner::DataFusionCpuExecutionPolicy::Reject
    );
}

#[test]
fn native_profile_defaults_to_conservative() {
    let profile = tpch_native_profile_settings_from([]).unwrap();

    assert_eq!(profile.profile_name, TPCH_NATIVE_PROFILE_CONSERVATIVE);
    assert_eq!(profile.profile_source, "default");
    assert_eq!(profile.device_budget, NativeTpchDeviceBudget::Automatic);
    assert_eq!(profile.max_source_chunk_bytes, DEFAULT_NATIVE_MAX_SOURCE_CHUNK_BYTES);
    assert_eq!(profile.max_row_groups_per_chunk, DEFAULT_NATIVE_MAX_ROW_GROUPS_PER_CHUNK);
    assert_eq!(profile.max_retry_attempts, DEFAULT_NATIVE_MAX_RETRY_ATTEMPTS);
    assert_eq!(profile.min_retry_chunk_row_groups, DEFAULT_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS);
    assert_eq!(profile.source_chunk_planning_mode, nexus_query_engine::runtime::SourceChunkPlanningMode::Conservative);
}

#[test]
fn native_profile_resolves_sf100_gpu_throughput() {
    let profile =
        tpch_native_profile_settings_from([(NEXUS_TPCH_BENCH_NATIVE_PROFILE_ENV, "sf100_gpu_throughput")]).unwrap();

    assert_eq!(profile.profile_name, TPCH_NATIVE_PROFILE_SF100_GPU_THROUGHPUT);
    assert_eq!(profile.profile_source, "env_profile");
    assert_eq!(profile.device_budget, NativeTpchDeviceBudget::Automatic);
    assert_eq!(profile.max_source_chunk_bytes, TPCH_SF100_GPU_THROUGHPUT_MAX_SOURCE_CHUNK_BYTES);
    assert_eq!(profile.max_row_groups_per_chunk, TPCH_SF100_GPU_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK);
    assert_eq!(profile.max_retry_attempts, DEFAULT_NATIVE_MAX_RETRY_ATTEMPTS);
    assert_eq!(profile.min_retry_chunk_row_groups, DEFAULT_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS);
    assert_eq!(
        profile.source_chunk_planning_mode,
        nexus_query_engine::runtime::SourceChunkPlanningMode::LocalParquetThroughput
    );
}

#[test]
fn native_profile_source_chunk_planning_reaches_execution_config() {
    let profile =
        tpch_native_profile_settings_from([(NEXUS_TPCH_BENCH_NATIVE_PROFILE_ENV, "sf100_gpu_throughput")]).unwrap();
    let config = NativeTpchBenchmarkConfig {
        native_memory_policy: NativeTpchMemoryPolicy::bounded_from_profile_settings(profile),
        ..NativeTpchBenchmarkConfig::default()
    };
    let resources = NativeTpchRuntimeResources::unavailable();

    let execution = config.native_execution_config_with_resources(&resources);

    assert_eq!(
        execution.source_chunk_planning_mode(),
        nexus_query_engine::runtime::SourceChunkPlanningMode::LocalParquetThroughput
    );
}

#[test]
fn native_profile_exact_overrides_resolve_to_named_profile() {
    let profile = tpch_native_profile_settings_from([
        (NEXUS_TPCH_BENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, "34359738368"),
        (NEXUS_TPCH_BENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, "1024"),
        (NEXUS_TPCH_BENCH_NATIVE_MAX_RETRY_ATTEMPTS_ENV, "4"),
        (NEXUS_TPCH_BENCH_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV, "1"),
        (NEXUS_TPCH_BENCH_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV, "local_parquet_throughput"),
    ])
    .unwrap();

    assert_eq!(profile.profile_name, TPCH_NATIVE_PROFILE_SF100_GPU_THROUGHPUT);
    assert_eq!(profile.profile_source, "resolved_values");
}

#[test]
fn explicit_device_budget_override_is_custom_and_explicit() {
    let profile = tpch_native_profile_settings_from([
        (NEXUS_TPCH_BENCH_NATIVE_PROFILE_ENV, "sf100_gpu_throughput"),
        (NEXUS_TPCH_BENCH_NATIVE_DEVICE_BUDGET_BYTES_ENV, "15032385536"),
    ])
    .unwrap();

    assert_eq!(profile.profile_name, TPCH_NATIVE_PROFILE_CUSTOM);
    assert_eq!(profile.profile_source, "custom_overrides");
    assert_eq!(profile.device_budget, NativeTpchDeviceBudget::Explicit { bytes: 15_032_385_536 });
    assert_eq!(
        profile.device_budget.memory_limit(),
        nexus_query_engine::runtime::GpuDeviceMemoryLimit::Explicit { limit_bytes: 15_032_385_536 }
    );
    assert_eq!(profile.max_source_chunk_bytes, TPCH_SF100_GPU_THROUGHPUT_MAX_SOURCE_CHUNK_BYTES);
    assert_eq!(profile.max_row_groups_per_chunk, TPCH_SF100_GPU_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK);
    assert_eq!(profile.max_retry_attempts, DEFAULT_NATIVE_MAX_RETRY_ATTEMPTS);
    assert_eq!(profile.min_retry_chunk_row_groups, DEFAULT_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS);
    assert_eq!(
        profile.source_chunk_planning_mode,
        nexus_query_engine::runtime::SourceChunkPlanningMode::LocalParquetThroughput
    );

    assert_eq!(
        NativeTpchDeviceBudget::Automatic.memory_limit(),
        nexus_query_engine::runtime::GpuDeviceMemoryLimit::default()
    );
}

#[test]
fn native_profile_overrides_become_custom() {
    let profile = tpch_native_profile_settings_from([
        (NEXUS_TPCH_BENCH_NATIVE_PROFILE_ENV, "sf100_gpu_throughput"),
        (NEXUS_TPCH_BENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, "123456"),
        (NEXUS_TPCH_BENCH_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV, "conservative"),
    ])
    .unwrap();

    assert_eq!(profile.profile_name, TPCH_NATIVE_PROFILE_CUSTOM);
    assert_eq!(profile.profile_source, "custom_overrides");
    assert_eq!(profile.max_source_chunk_bytes, 123_456);
    assert_eq!(profile.source_chunk_planning_mode, nexus_query_engine::runtime::SourceChunkPlanningMode::Conservative);
}

#[test]
fn native_profile_rejects_zero_values() {
    for name in [
        NEXUS_TPCH_BENCH_NATIVE_DEVICE_BUDGET_BYTES_ENV,
        NEXUS_TPCH_BENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV,
        NEXUS_TPCH_BENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV,
        NEXUS_TPCH_BENCH_NATIVE_MAX_RETRY_ATTEMPTS_ENV,
        NEXUS_TPCH_BENCH_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV,
    ] {
        let error = tpch_native_profile_settings_from([(name, "0")])
            .expect_err("zero TPC-H native profile values must be rejected");
        assert!(error.to_string().contains("must be greater than zero"));
    }
}

#[test]
fn native_profile_rejects_invalid_profile_and_source_mode() {
    let error = tpch_native_profile_settings_from([(NEXUS_TPCH_BENCH_NATIVE_PROFILE_ENV, "fast")])
        .expect_err("unknown TPC-H native profiles must be rejected");
    assert!(error.to_string().contains("expected conservative or sf100_gpu_throughput"));

    let error = tpch_native_profile_settings_from([(NEXUS_TPCH_BENCH_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV, "wide")])
        .expect_err("unknown source chunk planning modes must be rejected");
    assert!(error.to_string().contains("expected conservative or local_parquet_throughput"));
}

#[test]
fn sf1_external_dataset_mode_rejects_tiny_or_unknown_metadata() {
    dataset::validate_sf1_external_metadata(&benchmark_metadata()).unwrap();
    dataset::validate_sf10_external_metadata(&benchmark_sf10_metadata()).unwrap();

    let tiny = TpchDatasetMetadata {
        scale_factor: "tiny".to_string(),
        provenance: "generated_tiny".to_string(),
        ..benchmark_metadata()
    };
    let error =
        dataset::validate_sf1_external_metadata(&tiny).expect_err("SF1 benchmark mode must not accept tiny metadata");
    assert!(error.to_string().contains("requires NEXUS_TPCH_SCALE_FACTOR=sf1"));

    let unknown = TpchDatasetMetadata {
        scale_factor: "unknown".to_string(),
        provenance: "external_parquet_root".to_string(),
        ..benchmark_metadata()
    };
    let error = dataset::validate_sf1_external_metadata(&unknown)
        .expect_err("SF1 benchmark mode must not accept an unknown scale factor");
    assert!(error.to_string().contains("observed unknown"));

    let without_content_fingerprint = TpchDatasetMetadata { content_fingerprint: None, ..benchmark_metadata() };
    let error = dataset::validate_sf1_external_metadata(&without_content_fingerprint)
        .expect_err("SF1 benchmark mode must preserve a content fingerprint");
    assert!(error.to_string().contains("requires a content fingerprint"));

    let unknown_generation =
        TpchDatasetMetadata { generation_command: "unknown_external".to_string(), ..benchmark_metadata() };
    let error = dataset::validate_sf1_external_metadata(&unknown_generation)
        .expect_err("SF1 benchmark mode must preserve reproducible generation metadata");
    assert!(error.to_string().contains("requires a reproducible NEXUS_TPCH_GENERATION_COMMAND"));

    let mut mislabeled_tiny = benchmark_metadata();
    mislabeled_tiny.tables.iter_mut().find(|table| table.table == "lineitem").unwrap().row_count = 3;
    let error = dataset::validate_sf1_external_metadata(&mislabeled_tiny)
        .expect_err("SF1 benchmark mode must reject mislabeled tiny data");
    assert!(error.to_string().contains("requires SF1 row count lineitem=6001215"));

    let error = dataset::validate_sf1_external_metadata(&benchmark_sf10_metadata())
        .expect_err("SF1 benchmark mode must reject SF10 metadata");
    assert!(error.to_string().contains("requires NEXUS_TPCH_SCALE_FACTOR=sf1"));

    let error = dataset::validate_sf10_external_metadata(&benchmark_metadata())
        .expect_err("SF10 benchmark mode must reject SF1 metadata");
    assert!(error.to_string().contains("requires NEXUS_TPCH_SCALE_FACTOR=sf10"));
}

#[test]
fn sf1_external_dataset_mode_rejects_inconsistent_row_group_metadata_lengths() {
    let mut rows_len_mismatch = benchmark_metadata();
    let table = rows_len_mismatch.tables.iter_mut().find(|table| table.table == "lineitem").unwrap();
    table.row_group_count = 2;
    table.row_group_rows = vec![3_000_000];
    table.row_group_bytes = vec![1, 1];
    let error = dataset::validate_sf1_external_metadata(&rows_len_mismatch)
        .expect_err("SF1 benchmark metadata must reject row group row length mismatches");
    assert!(error.to_string().contains("inconsistent row_group_rows length for table lineitem"));

    let mut bytes_len_mismatch = benchmark_metadata();
    let table = bytes_len_mismatch.tables.iter_mut().find(|table| table.table == "lineitem").unwrap();
    table.row_group_count = 2;
    table.row_group_rows = vec![3_000_000, 3_001_215];
    table.row_group_bytes = vec![1];
    let error = dataset::validate_sf1_external_metadata(&bytes_len_mismatch)
        .expect_err("SF1 benchmark metadata must reject row group byte length mismatches");
    assert!(error.to_string().contains("inconsistent row_group_bytes length for table lineitem"));

    let mut zero_row_group_bytes = benchmark_metadata();
    zero_row_group_bytes.tables.iter_mut().find(|table| table.table == "lineitem").unwrap().row_group_bytes = vec![0];
    let error = dataset::validate_sf1_external_metadata(&zero_row_group_bytes)
        .expect_err("SF1 benchmark metadata must reject zero-byte row groups");
    assert!(error.to_string().contains("zero-byte row groups for table lineitem"));

    let mut oversized_row_group_bytes = benchmark_metadata();
    let table = oversized_row_group_bytes.tables.iter_mut().find(|table| table.table == "lineitem").unwrap();
    table.byte_size = 1;
    table.row_group_bytes = vec![2];
    let error = dataset::validate_sf1_external_metadata(&oversized_row_group_bytes)
        .expect_err("SF1 benchmark metadata must reject row-group byte totals larger than table bytes");
    assert!(error.to_string().contains("row-group bytes greater than table bytes for table lineitem"));
}

#[test]
fn query_id_filter_preserves_configured_order_and_rejects_duplicates() {
    let config = NativeTpchBenchmarkConfig {
        query_ids: vec!["q03".to_string(), "q01".to_string()],
        ..NativeTpchBenchmarkConfig::default()
    };

    let selected = config.selected_queries().into_iter().map(|query| query.id).collect::<Vec<_>>();

    assert_eq!(selected, vec!["q03", "q01"]);

    let duplicate = NativeTpchBenchmarkConfig {
        query_ids: vec!["q01".to_string(), "q01".to_string()],
        ..NativeTpchBenchmarkConfig::default()
    };
    let error = duplicate.validate().expect_err("duplicate query ids must not silently collapse rows");
    assert!(error.to_string().contains("duplicate TPC-H query id q01"));
}

#[test]
fn direct_config_rejects_empty_query_list() {
    let empty = NativeTpchBenchmarkConfig { query_ids: Vec::new(), ..NativeTpchBenchmarkConfig::default() };
    let error = empty.validate().expect_err("empty direct query lists must not produce empty benchmark reports");
    assert!(error.to_string().contains("must name at least one query id"));
}

#[tokio::test]
async fn benchmark_runner_validates_direct_config_before_selecting_queries() {
    let config =
        NativeTpchBenchmarkConfig { query_ids: vec!["q99".to_string()], ..NativeTpchBenchmarkConfig::default() };
    let dir = unique_temp_dir("datafusion-nexus-native-tpch-invalid-config");
    let error = run_native_tpch_benchmark(&config, &dir).await.expect_err("invalid direct configs must not panic");

    assert!(error.to_string().contains("unknown TPC-H query id q99"));
    fs::remove_dir_all(dir).unwrap();
}
