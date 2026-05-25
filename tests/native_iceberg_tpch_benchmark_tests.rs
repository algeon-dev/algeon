#[allow(dead_code)]
#[path = "../src/bin/native_iceberg_tpch_benchmark.rs"]
mod native_iceberg_tpch_benchmark;

use std::collections::HashMap;

use datafusion_cudf::{
    NativeDataFusionPlanningOutcome, NativeDataFusionPostNativeExecutionPath, NativeDataFusionRuntimeOutcome,
    ParityComparisonMode, ParityStatus, TableFormat, TableFormatDataFile, TableFormatDataFileFormat,
    TableFormatReadCapability, TableFormatScanPlan,
};
use native_iceberg_tpch_benchmark::{
    BenchmarkTimeoutCategory, IcebergTpchBenchmarkConfig, IcebergTpchBenchmarkReport, IcebergTpchBenchmarkRow,
    native_iceberg_smoke_report::{
        IcebergNativeSmokeReportRow, ObjectStoreStagingMode, ObjectStoreStagingStatus, ObjectStoreStagingSummary,
    },
    set_iteration_report_elapsed_ms, validate_full_sf1_staged_s3_gate_for_config,
};

#[test]
fn benchmark_config_defaults_to_verified_tpch_subset_without_exposing_credentials() {
    let env = HashMap::from([("AWS_PROFILE", "xdr-int"), ("CUDF_ICEBERG_SMOKE_FIXTURE_MODE", "glue_s3")]);
    let config = IcebergTpchBenchmarkConfig::from_env_get(|name| env.get(name).map(|value| value.to_string())).unwrap();

    assert_eq!(config.query_ids, vec!["q01", "q03", "q06", "q10"]);
    assert_eq!(config.smoke_config.tpch_query_ids, config.query_ids);
    assert_eq!(config.warmup_iterations, 0);
    assert_eq!(config.iterations, 1);
    assert_eq!(config.query_timeout_secs, None);
    assert_eq!(config.iteration_timeout_secs, None);
    assert_eq!(config.smoke_config.fixture.credential_source(), "aws_profile_env");
}

#[test]
fn benchmark_config_accepts_iteration_query_and_staging_overrides() {
    let env = HashMap::from([
        ("AWS_PROFILE", "xdr-int"),
        ("CUDF_ICEBERG_SMOKE_FIXTURE_MODE", "glue_s3"),
        ("CUDF_ICEBERG_TPCH_BENCH_QUERY_IDS", "q06,q01"),
        ("CUDF_ICEBERG_TPCH_BENCH_WARMUPS", "2"),
        ("CUDF_ICEBERG_TPCH_BENCH_ITERATIONS", "3"),
        ("CUDF_ICEBERG_TPCH_BENCH_OBJECT_STORE_STAGING", "required"),
        ("CUDF_ICEBERG_TPCH_BENCH_QUERY_TIMEOUT_SECS", "60"),
        ("CUDF_ICEBERG_TPCH_BENCH_ITERATION_TIMEOUT_SECS", "600"),
    ]);
    let config = IcebergTpchBenchmarkConfig::from_env_get(|name| env.get(name).map(|value| value.to_string())).unwrap();

    assert_eq!(config.query_ids, vec!["q06", "q01"]);
    assert_eq!(config.warmup_iterations, 2);
    assert_eq!(config.iterations, 3);
    assert_eq!(config.query_timeout_secs, Some(60));
    assert_eq!(config.iteration_timeout_secs, Some(600));
    assert_eq!(config.smoke_config.object_store_staging, ObjectStoreStagingMode::Required);
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
        "demo_db_tpch_sf1_all_tables".to_string(),
        "select 1".to_string(),
        plan.diagnostics(),
        NativeDataFusionPlanningOutcome::AcceptedReplaced,
        None,
        NativeDataFusionPostNativeExecutionPath::Native,
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
    assert!(tsv.starts_with("# native_iceberg_tpch_benchmark_tsv_schema_version=2\n"));
    assert!(tsv.contains("\tnative_runtime_elapsed_ms\t"));
    assert!(tsv.contains("\tcpu_runtime_elapsed_ms\t"));
    assert!(tsv.contains("\ttimeout_category\t"));
    assert!(tsv.contains("\treport_elapsed_ms\t"));
    assert!(tsv.contains("q06\t1\tglue_s3\tglue\tdemo_db_tpch_sf1_all_tables\tsf1\ticeberg\t123\t"));
    assert!(tsv.contains("\tsucceeded\t42\t"));
    assert!(tsv.contains("\tsucceeded\t99\t"));
    assert!(tsv.contains("\tunordered_exact\tmatched\t"));
    assert!(tsv.contains("\tmatched\tnone\t"));
}

#[test]
fn full_sf1_staged_s3_gate_passes_for_all_queries_with_staged_native_parity() {
    let query_ids = datafusion_cudf::TPCH_QUERIES.iter().map(|query| query.id).collect::<Vec<_>>().join(",");
    let env = HashMap::from([
        ("AWS_PROFILE", "xdr-int"),
        ("CUDF_ICEBERG_SMOKE_FIXTURE_MODE", "glue_s3"),
        ("CUDF_ICEBERG_TPCH_BENCH_QUERY_IDS", query_ids.as_str()),
        ("CUDF_ICEBERG_TPCH_BENCH_ITERATIONS", "1"),
    ]);
    let config = IcebergTpchBenchmarkConfig::from_env_get(|name| env.get(name).map(|value| value.to_string())).unwrap();
    let report = IcebergTpchBenchmarkReport {
        rows: datafusion_cudf::TPCH_QUERIES.iter().map(|query| staged_s3_matched_row(query.id)).collect(),
    };

    let summary = validate_full_sf1_staged_s3_gate_for_config(&report, &config).unwrap().unwrap();

    assert_eq!(summary.query_count, datafusion_cudf::TPCH_QUERIES.len());
    assert_eq!(summary.measured_iterations, 1);
    assert_eq!(summary.native_runtime_elapsed_ms_total, datafusion_cudf::TPCH_QUERIES.len() as u128 * 10);
    assert_eq!(summary.cpu_runtime_elapsed_ms_total, datafusion_cudf::TPCH_QUERIES.len() as u128 * 20);
    assert_eq!(summary.report_elapsed_ms, 123);
}

#[test]
fn full_sf1_staged_s3_gate_fails_on_direct_or_unmatched_rows() {
    let query_ids = datafusion_cudf::TPCH_QUERIES.iter().map(|query| query.id).collect::<Vec<_>>().join(",");
    let env = HashMap::from([
        ("AWS_PROFILE", "xdr-int"),
        ("CUDF_ICEBERG_SMOKE_FIXTURE_MODE", "glue_s3"),
        ("CUDF_ICEBERG_TPCH_BENCH_QUERY_IDS", query_ids.as_str()),
    ]);
    let config = IcebergTpchBenchmarkConfig::from_env_get(|name| env.get(name).map(|value| value.to_string())).unwrap();
    let mut rows =
        datafusion_cudf::TPCH_QUERIES.iter().map(|query| staged_s3_matched_row(query.id)).collect::<Vec<_>>();
    rows[0].smoke_row.staging_summary.status = ObjectStoreStagingStatus::DirectNative;
    rows[1].smoke_row.parity_status = ParityStatus::Mismatched;
    let report = IcebergTpchBenchmarkReport { rows };

    let error = validate_full_sf1_staged_s3_gate_for_config(&report, &config).unwrap_err().to_string();

    assert!(error.contains("q01:object_store_staging_status=direct_s3_native"));
    assert!(error.contains("q02:parity_status=mismatched"));
}

#[test]
fn full_sf1_staged_s3_gate_reports_timeout_failures_separately() {
    let query_ids = datafusion_cudf::TPCH_QUERIES.iter().map(|query| query.id).collect::<Vec<_>>().join(",");
    let env = HashMap::from([
        ("AWS_PROFILE", "xdr-int"),
        ("CUDF_ICEBERG_SMOKE_FIXTURE_MODE", "glue_s3"),
        ("CUDF_ICEBERG_TPCH_BENCH_QUERY_IDS", query_ids.as_str()),
    ]);
    let config = IcebergTpchBenchmarkConfig::from_env_get(|name| env.get(name).map(|value| value.to_string())).unwrap();
    let mut rows =
        datafusion_cudf::TPCH_QUERIES.iter().map(|query| staged_s3_matched_row(query.id)).collect::<Vec<_>>();
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
    let env = HashMap::from([("AWS_PROFILE", "xdr-int"), ("CUDF_ICEBERG_SMOKE_FIXTURE_MODE", "glue_s3")]);
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
        "demo_db_tpch_sf1_all_tables".to_string(),
        "select 1".to_string(),
        plan.diagnostics(),
        NativeDataFusionPlanningOutcome::AcceptedReplaced,
        None,
        NativeDataFusionPostNativeExecutionPath::Native,
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
