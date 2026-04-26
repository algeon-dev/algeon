use std::{collections::HashMap, sync::Arc};

use arrow::{
    array::Int64Array,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion_nexus::{
    native_report::{
        FinalPlanDisposition, NativeDataFusionPlanningOutcome, NativeDataFusionRuntimeOutcome, ParityComparisonMode,
        ParityStatus,
    },
    table_format::{
        TableFormat, TableFormatDataFile, TableFormatDataFileFormat, TableFormatReadCapability, TableFormatScanPlan,
    },
};
use nexus_bench::{
    iceberg_smoke as native_iceberg_smoke_report,
    iceberg_smoke::{
        CpuBaselineSessionProfile, IcebergNativeSmokeReportRow, NativeExecutionProfile, ObjectStoreStagingMode,
        ObjectStoreStagingStatus, ObjectStoreStagingSummary,
    },
    iceberg_tpch_benchmark::{
        self as native_iceberg_tpch_benchmark, BenchmarkTimeoutCategory, IcebergTpchBenchmarkConfig,
        IcebergTpchBenchmarkReport, IcebergTpchBenchmarkRow, set_iteration_report_elapsed_ms,
        validate_full_sf1_staged_s3_gate_for_config, write_benchmark_tsv_header, write_benchmark_tsv_row,
    },
};
use nexus_query_engine::{
    mode::StatisticalAggregateExecutionMode,
    runtime::MemoryControlPolicy,
    source::{CredentialMode, LakehouseCacheStatus, ResolvedParquetSource},
};
use nexus_workloads::tpch::TPCH_QUERIES;

#[test]
fn benchmark_config_defaults_to_verified_tpch_subset_without_exposing_credentials() {
    let env = HashMap::from([("AWS_PROFILE", "local-dev"), ("NEXUS_ICEBERG_SMOKE_FIXTURE_MODE", "glue_s3")]);
    let config = IcebergTpchBenchmarkConfig::from_env_get(|name| env.get(name).map(|value| value.to_string())).unwrap();

    assert_eq!(config.query_ids, vec!["q01", "q03", "q06", "q10"]);
    assert_eq!(config.smoke_config.tpch_query_ids, config.query_ids);
    assert_eq!(config.warmup_iterations, 0);
    assert_eq!(config.iterations, 1);
    assert_eq!(config.query_timeout_secs, None);
    assert_eq!(config.iteration_timeout_secs, None);
    assert_eq!(config.smoke_config.cpu_baseline_timeout_secs, None);
    assert_eq!(config.smoke_config.fixture.credential_source(), "aws_profile_env");
}

#[test]
fn benchmark_config_accepts_iteration_query_and_staging_overrides() {
    let env = HashMap::from([
        ("AWS_PROFILE", "local-dev"),
        ("NEXUS_ICEBERG_SMOKE_FIXTURE_MODE", "glue_s3"),
        ("NEXUS_ICEBERG_TPCH_BENCH_QUERY_IDS", "q06,q01"),
        ("NEXUS_ICEBERG_TPCH_BENCH_WARMUPS", "2"),
        ("NEXUS_ICEBERG_TPCH_BENCH_ITERATIONS", "3"),
        ("NEXUS_ICEBERG_TPCH_BENCH_OBJECT_STORE_STAGING", "required"),
        ("NEXUS_ICEBERG_TPCH_BENCH_QUERY_TIMEOUT_SECS", "60"),
        ("NEXUS_ICEBERG_TPCH_BENCH_ITERATION_TIMEOUT_SECS", "600"),
    ]);
    let config = IcebergTpchBenchmarkConfig::from_env_get(|name| env.get(name).map(|value| value.to_string())).unwrap();

    assert_eq!(config.query_ids, vec!["q06", "q01"]);
    assert_eq!(config.warmup_iterations, 2);
    assert_eq!(config.iterations, 3);
    assert_eq!(config.query_timeout_secs, Some(60));
    assert_eq!(config.iteration_timeout_secs, Some(600));
    assert_eq!(config.smoke_config.cpu_baseline_timeout_secs, Some(60));
    assert_eq!(config.smoke_config.object_store_staging, ObjectStoreStagingMode::Required);
}

#[test]
fn benchmark_config_accepts_cpu_baseline_timeout_override() {
    let env = HashMap::from([
        ("AWS_PROFILE", "local-dev"),
        ("NEXUS_ICEBERG_SMOKE_FIXTURE_MODE", "glue_s3"),
        ("NEXUS_ICEBERG_TPCH_BENCH_QUERY_TIMEOUT_SECS", "90"),
        ("NEXUS_ICEBERG_TPCH_BENCH_CPU_BASELINE_TIMEOUT_SECS", "45"),
    ]);
    let config = IcebergTpchBenchmarkConfig::from_env_get(|name| env.get(name).map(|value| value.to_string())).unwrap();

    assert_eq!(config.query_timeout_secs, Some(90));
    assert_eq!(config.smoke_config.cpu_baseline_timeout_secs, Some(45));
}

#[test]
fn cpu_baseline_session_profiles_keep_native_smoke_and_tpch_cpu_configs_separate() {
    let smoke = native_iceberg_smoke_report::cpu_baseline_session_config(CpuBaselineSessionProfile::Smoke);
    let smoke_options = smoke.options();

    assert_eq!(smoke_options.execution.target_partitions, 1);
    assert!(!smoke_options.optimizer.repartition_file_scans);
    assert!(!smoke_options.optimizer.repartition_aggregations);
    assert!(!smoke_options.optimizer.repartition_sorts);
    assert!(!smoke_options.optimizer.enable_dynamic_filter_pushdown);

    let tpch = native_iceberg_smoke_report::cpu_baseline_session_config(CpuBaselineSessionProfile::TpchTuned);
    let tpch_options = tpch.options();
    let expected_parallelism = std::thread::available_parallelism().map(usize::from).unwrap_or(1);

    assert_eq!(tpch_options.execution.target_partitions, expected_parallelism);
    assert_eq!(tpch.batch_size(), 8192);
    assert!(tpch_options.optimizer.repartition_file_scans);
    assert!(tpch_options.optimizer.repartition_aggregations);
    assert!(tpch_options.optimizer.repartition_joins);
    assert!(tpch_options.optimizer.repartition_sorts);
    assert!(tpch_options.optimizer.enable_dynamic_filter_pushdown);
    assert!(!tpch_options.execution.parquet.pushdown_filters);
}

#[test]
fn iceberg_native_workload_session_profiles_preserve_native_scalar_subquery_contracts() {
    let smoke = native_iceberg_smoke_report::native_session_config_for_profile(NativeExecutionProfile::Smoke);
    assert!(
        smoke.options().optimizer.enable_physical_uncorrelated_scalar_subquery,
        "generic smoke profile keeps DataFusion's default physical scalar-subquery behavior"
    );

    let tpch = native_iceberg_smoke_report::native_session_config_for_profile(NativeExecutionProfile::TpchBenchmark);
    assert!(
        !tpch.options().optimizer.enable_physical_uncorrelated_scalar_subquery,
        "TPC-H native Iceberg profile must use relationalized scalar subqueries for native admission"
    );
    assert!(!tpch.options().optimizer.repartition_joins);
    assert_eq!(tpch.options().execution.target_partitions, 1);

    let tpcds = native_iceberg_smoke_report::native_session_config_for_profile(NativeExecutionProfile::TpcdsBenchmark);
    assert!(
        !tpcds.options().optimizer.enable_physical_uncorrelated_scalar_subquery,
        "TPC-DS native Iceberg profile must use relationalized scalar subqueries for native admission"
    );
    assert!(!tpcds.options().execution.collect_statistics);
    assert!(!tpcds.options().optimizer.repartition_joins);
    assert_eq!(tpcds.options().execution.target_partitions, 1);
}

#[test]
fn tpch_native_execution_profile_uses_decision_grade_source_chunk_defaults() {
    let config = native_iceberg_smoke_report::native_execution_config_for_profile_with_env(
        NativeExecutionProfile::TpchBenchmark,
        |_| None,
    )
    .unwrap();

    let MemoryControlPolicy::Bounded { max_source_chunk_bytes, max_row_groups_per_chunk, .. } =
        config.memory_control_policy()
    else {
        panic!("TPC-H Iceberg native profile must be bounded");
    };

    assert_eq!(max_source_chunk_bytes, 32 * 1024 * 1024 * 1024);
    assert_eq!(max_row_groups_per_chunk, 1024);
    assert_eq!(config.source_scheduler_policy().max_source_reads_in_flight(), 2);
}

#[test]
fn iceberg_benchmark_native_optimizer_allows_source_reuse_up_to_profile_source_concurrency_budget() {
    let config = native_iceberg_smoke_report::native_execution_config_for_profile_with_env(
        NativeExecutionProfile::TpchBenchmark,
        |_| None,
    )
    .unwrap();
    let MemoryControlPolicy::Bounded { max_source_chunk_bytes, .. } = config.memory_control_policy() else {
        panic!("TPC-H Iceberg native profile must be bounded");
    };
    let options = native_iceberg_smoke_report::native_optimizer_options_for_profile(
        NativeExecutionProfile::TpchBenchmark,
        &config,
    );

    let max_source_reads_in_flight = config.source_scheduler_policy().max_source_reads_in_flight() as u64;
    assert_eq!(
        options.local_parquet_source_reuse.max_variant_selected_bytes,
        max_source_chunk_bytes * max_source_reads_in_flight
    );

    let smoke_config = native_iceberg_smoke_report::native_execution_config_for_profile_with_env(
        NativeExecutionProfile::Smoke,
        |_| None,
    )
    .unwrap();
    let smoke_options =
        native_iceberg_smoke_report::native_optimizer_options_for_profile(NativeExecutionProfile::Smoke, &smoke_config);
    assert_eq!(
        smoke_options.local_parquet_source_reuse.max_variant_selected_bytes,
        nexus_query_engine::optimizer::LocalParquetSourceReuseOptions::default().max_variant_selected_bytes
    );
}

#[test]
fn tpcds_native_execution_profile_uses_analytics_statistical_aggregate_default() {
    let config = native_iceberg_smoke_report::native_execution_config_for_profile_with_env(
        NativeExecutionProfile::TpcdsBenchmark,
        |_| None,
    )
    .unwrap();

    assert_eq!(config.statistical_aggregate_execution_mode(), StatisticalAggregateExecutionMode::GpuNativeTolerant);
}

#[test]
fn tpcds_native_execution_profile_accepts_statistical_aggregate_env_override() {
    let env =
        HashMap::from([("NEXUS_ICEBERG_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE", "datafusion_exact_host_welford")]);
    let config = native_iceberg_smoke_report::native_execution_config_for_profile_with_env(
        NativeExecutionProfile::TpcdsBenchmark,
        |name| env.get(name).map(|value| value.to_string()),
    )
    .unwrap();

    assert_eq!(
        config.statistical_aggregate_execution_mode(),
        StatisticalAggregateExecutionMode::DataFusionExactHostWelford
    );
}

#[test]
fn smoke_parity_treats_partial_order_by_as_unordered_tie_candidate() {
    let q18 = TPCH_QUERIES.iter().find(|query| query.id == "q18").unwrap();
    let q18_schema = Schema::new(vec![
        Field::new("c_name", DataType::Utf8, true),
        Field::new("c_custkey", DataType::Int64, true),
        Field::new("o_orderkey", DataType::Int64, true),
        Field::new("o_orderdate", DataType::Date32, true),
        Field::new("o_totalprice", DataType::Decimal128(15, 2), true),
        Field::new("sum_quantity", DataType::Decimal128(38, 2), true),
    ]);

    assert!(!native_iceberg_smoke_report::smoke_parity_uses_ordered_rows(q18.sql, &q18_schema));

    let total_order_schema = Schema::new(vec![Field::new("a", DataType::Int64, true)]);
    assert!(native_iceberg_smoke_report::smoke_parity_uses_ordered_rows(
        "SELECT a FROM t ORDER BY a",
        &total_order_schema,
    ));
}

#[test]
fn smoke_output_fingerprint_uses_comparison_schema_not_arrow_field_names() {
    let cpu_schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
    let native_schema = Arc::new(Schema::new(vec![Field::new("native_a", DataType::Int64, false)]));
    let cpu_batch =
        RecordBatch::try_new(Arc::clone(&cpu_schema), vec![Arc::new(Int64Array::from(vec![1, 2]))]).unwrap();
    let native_batch =
        RecordBatch::try_new(Arc::clone(&native_schema), vec![Arc::new(Int64Array::from(vec![1, 2]))]).unwrap();

    let cpu_fingerprint =
        native_iceberg_smoke_report::fingerprint_output_batches(&cpu_schema, &[cpu_batch], "SELECT a FROM t");
    let native_fingerprint =
        native_iceberg_smoke_report::fingerprint_output_batches(&native_schema, &[native_batch], "SELECT a FROM t");

    assert_eq!(cpu_fingerprint.cpu_fingerprint, native_fingerprint.native_fingerprint);
}

#[test]
fn benchmark_report_tsv_exposes_runtime_timing_and_parity_fields() {
    let plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_snapshot_id("123")
        .with_catalog_type("glue")
        .with_data_file(
            TableFormatDataFile::new(
                "data-0001",
                "/tmp/iceberg/data-0001.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::local_file_system(),
            )
            .with_size_bytes(1024)
            .with_row_count(7),
        );
    let mut smoke_row = IcebergNativeSmokeReportRow::new(
        "q06".to_string(),
        "glue_s3".to_string(),
        "glue".to_string(),
        "tpch_sf1_all_tables".to_string(),
        "select 1".to_string(),
        plan.diagnostics(),
        Some(NativeDataFusionPlanningOutcome::Selected),
        None,
        FinalPlanDisposition::Native,
        None,
        ObjectStoreStagingSummary::new(ObjectStoreStagingMode::Auto, ObjectStoreStagingStatus::StagedLocal),
        "aws_profile_env".to_string(),
    );
    smoke_row.native_runtime_started = true;
    smoke_row.native_runtime_outcome = NativeDataFusionRuntimeOutcome::Succeeded;
    smoke_row.native_runtime_elapsed_ms = Some(42);
    smoke_row.cpu_runtime_started = true;
    smoke_row.cpu_runtime_outcome = NativeDataFusionRuntimeOutcome::Succeeded;
    smoke_row.cpu_runtime_elapsed_ms = Some(99);
    smoke_row.parity_status = ParityStatus::Matched;
    smoke_row.comparison_mode = ParityComparisonMode::UnorderedExact;

    let report = IcebergTpchBenchmarkReport {
        rows: vec![IcebergTpchBenchmarkRow {
            iteration: 1,
            tpch_scale_factor: "sf1".to_string(),
            warmup_iterations: 0,
            measured_iterations: 1,
            report_elapsed_ms: 123,
            timeout_category: None,
            smoke_row,
        }],
    };

    let tsv = report.to_tsv();
    assert!(tsv.starts_with(REPORT_HEADER_FRAGMENT));
    assert!(tsv.contains("\tnative_runtime_elapsed_ms\t"));
    assert!(tsv.contains("\tcpu_runtime_elapsed_ms\t"));
    assert!(tsv.contains("\tsource_read_strategy\t"));
    assert!(tsv.contains("\tcredential_mode\t"));
    assert!(tsv.contains("\tlakehouse_cache_status\t"));
    assert!(tsv.contains("\tlakehouse_cache_read_mode\t"));
    assert!(tsv.contains("\tlakehouse_cache_fill_mode\t"));
    assert!(tsv.contains("\tnative_source_read_runtime_rows\t"));
    assert!(tsv.contains("\tlakehouse_cache_fill_event_snapshot_rows\t"));
    assert!(tsv.contains("\ttimeout_category\t"));
    assert!(tsv.contains("\treport_elapsed_ms\t"));
    assert!(tsv.contains("q06\t1\tglue_s3\tglue\ttpch_sf1_all_tables\tsf1\ticeberg\t123\t"));
    assert!(tsv.contains("\tsucceeded\t42\t"));
    assert!(tsv.contains("\tsucceeded\t99\t"));
    assert!(tsv.contains("\tunordered_exact\tmatched\t"));
    assert!(tsv.contains("\tmatched\tnone\t"));
}

#[test]
fn benchmark_stream_writer_flushes_header_and_each_completed_query_row() {
    let row = staged_s3_matched_row("q01");
    let mut writer = FlushCountingWriter::default();

    write_benchmark_tsv_header(&mut writer).unwrap();
    write_benchmark_tsv_row(&mut writer, &row).unwrap();

    assert_eq!(writer.flushes, 2);
    let output = String::from_utf8(writer.bytes).unwrap();
    assert!(output.starts_with(REPORT_HEADER_FRAGMENT));
    assert_eq!(output.matches(REPORT_HEADER_FRAGMENT).count(), 1);
    assert!(output.contains("\nq01\t1\tglue_s3\tglue\ttpch_sf1_all_tables\tsf1\ticeberg\t123\t"));
}

#[test]
fn cpu_child_result_parser_extracts_cpu_fields_from_benchmark_tsv() {
    let mut row = staged_s3_matched_row("q18");
    row.smoke_row.cpu_output_rows = Some(100);
    row.smoke_row.cpu_output_batches = Some(1);
    row.smoke_row.cpu_output_fingerprint = Some("output-fnv1a64:cpu".to_string());
    row.smoke_row.comparison_mode = ParityComparisonMode::UnorderedExact;
    let tsv = IcebergTpchBenchmarkReport { rows: vec![row] }.to_tsv();

    let parsed = native_iceberg_tpch_benchmark::parse_cpu_child_result(&tsv).unwrap();

    assert_eq!(parsed.elapsed_ms, Some(20));
    assert_eq!(parsed.output_rows, Some(100));
    assert_eq!(parsed.output_batches, Some(1));
    assert_eq!(parsed.output_fingerprint.as_deref(), Some("output-fnv1a64:cpu"));
    assert_eq!(parsed.comparison_mode, ParityComparisonMode::UnorderedExact);
}

#[test]
fn cpu_child_failure_preserves_native_success_as_cpu_failed_row() {
    let mut row = staged_s3_matched_row("q18").smoke_row;
    row.cpu_runtime_started = false;
    row.cpu_runtime_outcome = NativeDataFusionRuntimeOutcome::NotStarted;
    row.cpu_runtime_elapsed_ms = None;
    row.cpu_output_rows = None;
    row.cpu_output_batches = None;
    row.cpu_output_fingerprint = None;
    row.native_runtime_outcome = NativeDataFusionRuntimeOutcome::Succeeded;
    row.native_output_fingerprint = Some("output-fnv1a64:native".to_string());

    native_iceberg_tpch_benchmark::apply_cpu_child_failure(
        &mut row,
        native_iceberg_tpch_benchmark::CpuChildFailure {
            elapsed_ms: 60_001,
            detail: "cpu_baseline_child_timeout_secs=60".to_string(),
            timeout: true,
        },
    );

    assert!(row.cpu_runtime_started);
    assert_eq!(row.cpu_runtime_outcome, NativeDataFusionRuntimeOutcome::Failed);
    assert_eq!(row.cpu_runtime_elapsed_ms, Some(60_001));
    assert_eq!(row.parity_status, ParityStatus::CpuFailed);
    assert_eq!(row.comparison_mode, ParityComparisonMode::NotApplicable);
    assert_eq!(
        row.parity_detail.as_deref(),
        Some("timeout_category=cpu_s3_baseline_timeout;cpu_baseline_child_timeout_secs=60")
    );
    assert_eq!(row.native_output_fingerprint.as_deref(), Some("output-fnv1a64:native"));
}

#[test]
fn cpu_baseline_prefers_existing_local_cache_reader_paths() {
    let dir = tempfile::tempdir().unwrap();
    let cache_path = dir.path().join("lineitem-cache.parquet");
    std::fs::write(&cache_path, b"not a real parquet file").unwrap();
    let cache_path_text = cache_path.display().to_string();
    let plan = TableFormatScanPlan::new(TableFormat::Iceberg).with_data_file(
        TableFormatDataFile::new(
            "lineitem-0001",
            "s3://lakehouse/warehouse/tpch_sf10/lineitem/data-0001.parquet",
            TableFormatDataFileFormat::Parquet,
            TableFormatReadCapability::ObjectStore { scheme: "s3".to_string() },
        )
        .with_resolved_parquet_source(ResolvedParquetSource::local_cache_kvikio(
            "s3://lakehouse/warehouse/tpch_sf10/lineitem/data-0001.parquet",
            cache_path_text.clone(),
            CredentialMode::SharedProcessEnv,
            LakehouseCacheStatus::Hit,
        )),
    );

    let urls = native_iceberg_smoke_report::cpu_baseline_listing_urls_from_scan_plan(&plan).unwrap();

    assert_eq!(urls.len(), 1);
    assert_eq!(urls[0].scheme(), "file");
    assert!(urls[0].as_str().ends_with("lineitem-cache.parquet"));
    assert_eq!(native_iceberg_smoke_report::cpu_baseline_listing_file_extension(&urls), ".parquet");
}

#[test]
fn cpu_baseline_uses_empty_listing_extension_for_extensionless_cache_files() {
    let url = datafusion::datasource::listing::ListingTableUrl::parse("/tmp/lakehouse-cache/content-key").unwrap();
    let urls = vec![url];

    assert_eq!(native_iceberg_smoke_report::cpu_baseline_listing_file_extension(&urls), "");
}

#[test]
fn full_sf1_staged_s3_gate_passes_for_all_queries_with_staged_native_parity() {
    let query_ids = TPCH_QUERIES.iter().map(|query| query.id).collect::<Vec<_>>().join(",");
    let env = HashMap::from([
        ("AWS_PROFILE", "local-dev"),
        ("NEXUS_ICEBERG_SMOKE_FIXTURE_MODE", "glue_s3"),
        ("NEXUS_ICEBERG_TPCH_BENCH_QUERY_IDS", query_ids.as_str()),
        ("NEXUS_ICEBERG_TPCH_BENCH_ITERATIONS", "1"),
    ]);
    let config = IcebergTpchBenchmarkConfig::from_env_get(|name| env.get(name).map(|value| value.to_string())).unwrap();
    let report =
        IcebergTpchBenchmarkReport { rows: TPCH_QUERIES.iter().map(|query| staged_s3_matched_row(query.id)).collect() };

    let summary = validate_full_sf1_staged_s3_gate_for_config(&report, &config).unwrap().unwrap();

    assert_eq!(summary.query_count, TPCH_QUERIES.len());
    assert_eq!(summary.measured_iterations, 1);
    assert_eq!(summary.native_runtime_elapsed_ms_total, TPCH_QUERIES.len() as u128 * 10);
    assert_eq!(summary.cpu_runtime_elapsed_ms_total, TPCH_QUERIES.len() as u128 * 20);
    assert_eq!(summary.report_elapsed_ms, 123);
}

#[test]
fn full_sf1_staged_s3_gate_fails_on_direct_or_unmatched_rows() {
    let query_ids = TPCH_QUERIES.iter().map(|query| query.id).collect::<Vec<_>>().join(",");
    let env = HashMap::from([
        ("AWS_PROFILE", "local-dev"),
        ("NEXUS_ICEBERG_SMOKE_FIXTURE_MODE", "glue_s3"),
        ("NEXUS_ICEBERG_TPCH_BENCH_QUERY_IDS", query_ids.as_str()),
    ]);
    let config = IcebergTpchBenchmarkConfig::from_env_get(|name| env.get(name).map(|value| value.to_string())).unwrap();
    let mut rows = TPCH_QUERIES.iter().map(|query| staged_s3_matched_row(query.id)).collect::<Vec<_>>();
    rows[0].smoke_row.staging_summary.status = ObjectStoreStagingStatus::DirectNative;
    rows[1].smoke_row.parity_status = ParityStatus::Mismatched;
    let report = IcebergTpchBenchmarkReport { rows };

    let error = validate_full_sf1_staged_s3_gate_for_config(&report, &config).unwrap_err().to_string();

    assert!(error.contains("q01:object_store_staging_status=direct_s3_native"));
    assert!(error.contains("q02:parity_status=mismatched"));
}

#[test]
fn full_sf1_staged_s3_gate_reports_timeout_failures_separately() {
    let query_ids = TPCH_QUERIES.iter().map(|query| query.id).collect::<Vec<_>>().join(",");
    let env = HashMap::from([
        ("AWS_PROFILE", "local-dev"),
        ("NEXUS_ICEBERG_SMOKE_FIXTURE_MODE", "glue_s3"),
        ("NEXUS_ICEBERG_TPCH_BENCH_QUERY_IDS", query_ids.as_str()),
    ]);
    let config = IcebergTpchBenchmarkConfig::from_env_get(|name| env.get(name).map(|value| value.to_string())).unwrap();
    let mut rows = TPCH_QUERIES.iter().map(|query| staged_s3_matched_row(query.id)).collect::<Vec<_>>();
    rows[0].timeout_category = Some(BenchmarkTimeoutCategory::NativeRuntime);
    rows[0].smoke_row.native_runtime_outcome = NativeDataFusionRuntimeOutcome::Failed;
    rows[0].smoke_row.parity_status = ParityStatus::NativeFailed;
    let report = IcebergTpchBenchmarkReport { rows };

    let error = validate_full_sf1_staged_s3_gate_for_config(&report, &config).unwrap_err().to_string();

    assert!(error.contains("full SF1 staged-S3 gate timeout failures=q01:native_runtime_timeout"));
    assert!(error.contains("semantic_failures=q01:native_runtime_outcome=failed,q01:parity_status=native_failed"));
}

#[test]
fn full_sf1_staged_s3_gate_does_not_apply_to_default_subset() {
    let env = HashMap::from([("AWS_PROFILE", "local-dev"), ("NEXUS_ICEBERG_SMOKE_FIXTURE_MODE", "glue_s3")]);
    let config = IcebergTpchBenchmarkConfig::from_env_get(|name| env.get(name).map(|value| value.to_string())).unwrap();
    let report = IcebergTpchBenchmarkReport { rows: vec![staged_s3_matched_row("q01")] };

    assert!(validate_full_sf1_staged_s3_gate_for_config(&report, &config).unwrap().is_none());
}

#[test]
fn timeout_enabled_rows_share_iteration_elapsed_observation() {
    let mut rows = vec![staged_s3_matched_row("q01"), staged_s3_matched_row("q02"), staged_s3_matched_row("q03")];
    rows[0].report_elapsed_ms = 11;
    rows[1].report_elapsed_ms = 22;
    rows[2].report_elapsed_ms = 33;

    set_iteration_report_elapsed_ms(&mut rows, 1, 777);

    assert_eq!(rows[0].report_elapsed_ms, 11);
    assert_eq!(rows[1].report_elapsed_ms, 777);
    assert_eq!(rows[2].report_elapsed_ms, 777);
}

fn staged_s3_matched_row(query_id: &str) -> IcebergTpchBenchmarkRow {
    let plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_snapshot_id("123")
        .with_catalog_type("glue")
        .with_data_file(
            TableFormatDataFile::new(
                "s3-data-0001",
                "/tmp/staged-data-0001.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::local_file_system(),
            )
            .with_size_bytes(1024)
            .with_row_count(7),
        );
    let mut staging_summary =
        ObjectStoreStagingSummary::new(ObjectStoreStagingMode::Required, ObjectStoreStagingStatus::StagedLocal);
    staging_summary.staged_data_file_count = 1;
    staging_summary.staged_bytes = 1024;
    let mut smoke_row = IcebergNativeSmokeReportRow::new(
        query_id.to_string(),
        "glue_s3".to_string(),
        "glue".to_string(),
        "tpch_sf1_all_tables".to_string(),
        "select 1".to_string(),
        plan.diagnostics(),
        Some(NativeDataFusionPlanningOutcome::Selected),
        None,
        FinalPlanDisposition::Native,
        None,
        staging_summary,
        "aws_profile_env".to_string(),
    );
    smoke_row.native_runtime_started = true;
    smoke_row.native_runtime_outcome = NativeDataFusionRuntimeOutcome::Succeeded;
    smoke_row.native_runtime_elapsed_ms = Some(10);
    smoke_row.cpu_runtime_started = true;
    smoke_row.cpu_runtime_outcome = NativeDataFusionRuntimeOutcome::Succeeded;
    smoke_row.cpu_runtime_elapsed_ms = Some(20);
    smoke_row.parity_status = ParityStatus::Matched;
    smoke_row.comparison_mode = ParityComparisonMode::UnorderedExact;

    IcebergTpchBenchmarkRow {
        iteration: 1,
        tpch_scale_factor: "sf1".to_string(),
        warmup_iterations: 0,
        measured_iterations: 1,
        report_elapsed_ms: 123,
        timeout_category: None,
        smoke_row,
    }
}

const REPORT_HEADER_FRAGMENT: &str = "query_id\titeration\tfixture_mode\tcatalog_type";

#[derive(Default)]
struct FlushCountingWriter {
    bytes: Vec<u8>,
    flushes: usize,
}

impl std::io::Write for FlushCountingWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.bytes.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.flushes += 1;
        Ok(())
    }
}
