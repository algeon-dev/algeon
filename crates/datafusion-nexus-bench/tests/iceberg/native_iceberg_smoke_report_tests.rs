#[allow(dead_code)]
#[path = "../../src/bin/native_iceberg_smoke_report.rs"]
mod native_iceberg_smoke_report;

use std::collections::HashMap;

use datafusion_nexus::{
    NativeDataFusionExecutionMetrics, NativeDataFusionPlanningOutcome, NativeDataFusionPostNativeExecutionPath,
    NativeLakehouseCacheFillEventMetricRow, NativeSourceReadRuntimeMetricRow, TableFormat, TableFormatDataFile,
    TableFormatDataFileFormat, TableFormatParquetReadStrategyConfig, TableFormatReadCapability, TableFormatScanPlan,
    TableFormatUnsupportedFeature, TableFormatUnsupportedFeatureCode, resolve_table_format_parquet_source,
};
use native_iceberg_smoke_report::{
    CustomWorkload, IcebergNativeSmokeConfig, IcebergNativeSmokeFixture, ObjectStoreStagingMode,
    ObjectStoreStagingStatus, ObjectStoreStagingSummary, SmokeQueryExecutionMode, StagedObjectStoreDataFile,
};

#[test]
fn glue_s3_defaults_use_aws_profile_without_exposing_credential_value_in_report_config() {
    let env = HashMap::from([("AWS_PROFILE", "local-dev"), ("NEXUS_ICEBERG_SMOKE_FIXTURE_MODE", "glue_s3")]);
    let config = IcebergNativeSmokeConfig::from_env_get(|name| env.get(name).map(|value| value.to_string())).unwrap();

    let IcebergNativeSmokeFixture::GlueS3 { profile, database, table, catalog_id, region, .. } = &config.fixture else {
        panic!("expected glue_s3 fixture");
    };
    assert_eq!(profile.as_deref(), Some("local-dev"));
    assert_eq!(database, "tpch_sf1");
    assert_eq!(table, "lineitem");
    assert_eq!(catalog_id.as_deref(), None);
    assert_eq!(region, "us-east-1");
    assert_eq!(config.table_alias, "lineitem");
    assert_eq!(config.sql, "select count(*) as row_count from lineitem");
    assert_eq!(config.fixture.credential_source(), "aws_profile_env");
    assert_eq!(config.object_store_staging, ObjectStoreStagingMode::Required);
    assert_eq!(config.tpch_query_ids, vec!["q01", "q03", "q06", "q10"]);
    assert_eq!(config.tpch_scale_factor, "sf1");
    assert!(config.tpcds_query_ids.is_empty());
    assert_eq!(config.tpcds_scale_factor, "sf1");
    assert_eq!(config.custom_workload, None);
    assert_eq!(config.custom_query_file, None);
    assert!(!config.stream_tsv_rows);
    assert!(!config.continue_on_query_error);
    assert_eq!(config.execution_mode, SmokeQueryExecutionMode::NativeAndCpu);
    assert_eq!(native_iceberg_smoke_report::tpch_glue_table_name("orders", &config.tpch_scale_factor), "orders");
}

#[test]
fn local_metadata_mode_requires_metadata_location_and_parses_table_ident() {
    let env = HashMap::from([
        ("NEXUS_ICEBERG_SMOKE_FIXTURE_MODE", "local_metadata"),
        ("NEXUS_ICEBERG_METADATA_LOCATION", "/tmp/table/metadata/v1.metadata.json"),
        ("NEXUS_ICEBERG_TABLE_IDENT", "local_db.local_table"),
    ]);
    let config = IcebergNativeSmokeConfig::from_env_get(|name| env.get(name).map(|value| value.to_string())).unwrap();

    let IcebergNativeSmokeFixture::LocalMetadata { metadata_location, table_ident } = &config.fixture else {
        panic!("expected local_metadata fixture");
    };
    assert_eq!(metadata_location, "/tmp/table/metadata/v1.metadata.json");
    assert_eq!(table_ident, &vec!["local_db".to_string(), "local_table".to_string()]);
    assert_eq!(config.table_alias, "local_table");
    assert!(config.tpch_query_ids.is_empty());
    assert!(config.tpcds_query_ids.is_empty());
}

#[test]
fn explicit_glue_table_keeps_single_table_count_smoke_by_default() {
    let env = HashMap::from([
        ("AWS_PROFILE", "local-dev"),
        ("NEXUS_ICEBERG_SMOKE_FIXTURE_MODE", "glue_s3"),
        ("NEXUS_ICEBERG_GLUE_TABLE", "lineitem"),
    ]);
    let config = IcebergNativeSmokeConfig::from_env_get(|name| env.get(name).map(|value| value.to_string())).unwrap();

    assert!(config.tpch_query_ids.is_empty());
    assert!(config.tpcds_query_ids.is_empty());
    assert_eq!(config.table_alias, "lineitem");
    assert_eq!(config.sql, "select count(*) as row_count from lineitem");
}

#[test]
fn glue_s3_fixture_accepts_local_moto_and_rustfs_endpoints() {
    let env = HashMap::from([
        ("NEXUS_ICEBERG_SMOKE_FIXTURE_MODE", "glue_s3"),
        ("NEXUS_ICEBERG_GLUE_URI", "http://localhost:5001"),
        ("NEXUS_ICEBERG_S3_ENDPOINT", "http://localhost:9000"),
        ("NEXUS_ICEBERG_S3_REGION", "us-east-1"),
        ("NEXUS_ICEBERG_S3_PATH_STYLE", "true"),
    ]);
    let config = IcebergNativeSmokeConfig::from_env_get(|name| env.get(name).map(|value| value.to_string())).unwrap();

    let IcebergNativeSmokeFixture::GlueS3 { uri, s3_endpoint, s3_region, s3_path_style_access, .. } = &config.fixture
    else {
        panic!("expected glue_s3 fixture");
    };
    assert_eq!(uri.as_deref(), Some("http://localhost:5001"));
    assert_eq!(s3_endpoint.as_deref(), Some("http://localhost:9000"));
    assert_eq!(s3_region.as_deref(), Some("us-east-1"));
    assert_eq!(*s3_path_style_access, Some(true));
}

#[test]
fn tpcds_query_ids_select_tpcds_all_table_smoke_mode() {
    let env = HashMap::from([
        ("NEXUS_ICEBERG_SMOKE_FIXTURE_MODE", "glue_s3"),
        ("NEXUS_ICEBERG_TPCDS_SCALE_FACTOR", "sf10"),
        ("NEXUS_ICEBERG_TPCDS_SMOKE_QUERY_IDS", "q03,q42"),
    ]);
    let config = IcebergNativeSmokeConfig::from_env_get(|name| env.get(name).map(|value| value.to_string())).unwrap();

    assert!(config.tpch_query_ids.is_empty());
    assert_eq!(config.tpcds_query_ids, vec!["q03", "q42"]);
    assert_eq!(config.tpcds_scale_factor, "sf10");
    assert_eq!(config.custom_workload, None);
    assert_eq!(config.custom_query_file, None);
    assert_eq!(
        native_iceberg_smoke_report::tpcds_glue_table_name("store_sales", &config.tpcds_scale_factor),
        "store_sales"
    );
    assert_eq!(config.fixture.tpcds_all_tables_ident_display(&config.tpcds_scale_factor), "tpcds_sf10_all_tables");
}

#[test]
fn rest_s3_tpcds_query_ids_rewrite_tables_within_rest_namespace() {
    let env = HashMap::from([
        ("NEXUS_ICEBERG_SMOKE_FIXTURE_MODE", "rest_s3"),
        ("NEXUS_ICEBERG_NAMESPACE", "tpcds_sf10"),
        ("NEXUS_ICEBERG_TPCDS_SCALE_FACTOR", "sf10"),
        ("NEXUS_ICEBERG_TPCDS_SMOKE_QUERY_IDS", "q03,q42"),
        ("NEXUS_ICEBERG_SMOKE_EXECUTION_MODE", "native_only"),
        ("NEXUS_ICEBERG_SMOKE_STREAM_TSV", "1"),
        ("NEXUS_ICEBERG_SMOKE_CONTINUE_ON_QUERY_ERROR", "true"),
    ]);
    let config = IcebergNativeSmokeConfig::from_env_get(|name| env.get(name).map(|value| value.to_string())).unwrap();

    assert!(config.tpch_query_ids.is_empty());
    assert_eq!(config.tpcds_query_ids, vec!["q03", "q42"]);
    assert_eq!(config.tpcds_scale_factor, "sf10");
    assert_eq!(config.execution_mode, SmokeQueryExecutionMode::NativeOnly);
    assert!(config.stream_tsv_rows);
    assert!(config.continue_on_query_error);
    let rewritten = config.fixture.with_tpcds_table("store_sales", &config.tpcds_scale_factor).unwrap();
    let IcebergNativeSmokeFixture::RestS3 { namespace, table, .. } = rewritten else {
        panic!("expected rest_s3 fixture");
    };
    assert_eq!(namespace, "tpcds_sf10");
    assert_eq!(table, "store_sales");
}

#[test]
fn tpch_and_tpcds_query_sets_are_mutually_exclusive() {
    let env = HashMap::from([
        ("NEXUS_ICEBERG_SMOKE_FIXTURE_MODE", "glue_s3"),
        ("NEXUS_ICEBERG_SMOKE_QUERY_IDS", "q06"),
        ("NEXUS_ICEBERG_TPCDS_SMOKE_QUERY_IDS", "q03"),
    ]);
    let error = IcebergNativeSmokeConfig::from_env_get(|name| env.get(name).map(|value| value.to_string()))
        .expect_err("mixed workload query ids should fail");

    assert!(error.to_string().contains("mutually exclusive"));
}

#[test]
fn custom_query_file_selects_all_table_custom_mode() {
    let env = HashMap::from([
        ("NEXUS_ICEBERG_SMOKE_FIXTURE_MODE", "glue_s3"),
        ("NEXUS_ICEBERG_CUSTOM_WORKLOAD", "tpcds"),
        ("NEXUS_ICEBERG_CUSTOM_QUERY_FILE", "docs/bench/custom_tpcds_probe.json"),
    ]);
    let config = IcebergNativeSmokeConfig::from_env_get(|name| env.get(name).map(|value| value.to_string())).unwrap();

    assert!(config.tpch_query_ids.is_empty());
    assert!(config.tpcds_query_ids.is_empty());
    assert_eq!(config.custom_workload, Some(CustomWorkload::Tpcds));
    assert_eq!(config.custom_query_file.as_deref(), Some(std::path::Path::new("docs/bench/custom_tpcds_probe.json")));
}

#[test]
fn custom_query_file_requires_workload_and_is_mutually_exclusive_with_builtins() {
    let missing_workload = HashMap::from([("NEXUS_ICEBERG_CUSTOM_QUERY_FILE", "docs/bench/custom_tpcds_probe.json")]);
    let error =
        IcebergNativeSmokeConfig::from_env_get(|name| missing_workload.get(name).map(|value| value.to_string()))
            .expect_err("custom query file without workload should fail");
    assert!(error.to_string().contains("NEXUS_ICEBERG_CUSTOM_WORKLOAD"));

    let mixed = HashMap::from([
        ("NEXUS_ICEBERG_SMOKE_QUERY_IDS", "q06"),
        ("NEXUS_ICEBERG_CUSTOM_WORKLOAD", "tpch"),
        ("NEXUS_ICEBERG_CUSTOM_QUERY_FILE", "docs/bench/custom_tpch_probe.json"),
    ]);
    let error = IcebergNativeSmokeConfig::from_env_get(|name| mixed.get(name).map(|value| value.to_string()))
        .expect_err("custom query file and built-in query ids should fail");
    assert!(error.to_string().contains("mutually exclusive"));
}

#[test]
fn object_store_staging_mode_env_accepts_disabled_and_required_aliases() {
    let disabled = HashMap::from([
        ("NEXUS_ICEBERG_SMOKE_FIXTURE_MODE", "glue_s3"),
        ("NEXUS_ICEBERG_SMOKE_OBJECT_STORE_STAGING", "disabled"),
    ]);
    let disabled_config =
        IcebergNativeSmokeConfig::from_env_get(|name| disabled.get(name).map(|value| value.to_string())).unwrap();
    assert_eq!(disabled_config.object_store_staging, ObjectStoreStagingMode::Disabled);

    let required = HashMap::from([
        ("NEXUS_ICEBERG_SMOKE_FIXTURE_MODE", "glue_s3"),
        ("NEXUS_ICEBERG_SMOKE_OBJECT_STORE_STAGING", "1"),
    ]);
    let required_config =
        IcebergNativeSmokeConfig::from_env_get(|name| required.get(name).map(|value| value.to_string())).unwrap();
    assert_eq!(required_config.object_store_staging, ObjectStoreStagingMode::Required);
}

#[test]
fn staged_object_store_files_rewrite_to_local_native_scan_without_dropping_other_rejections() {
    let plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_data_file(
            TableFormatDataFile::new(
                "s3-data-file",
                "s3://bucket/table/data-0001.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::object_store("s3"),
            )
            .with_size_bytes(2048),
        )
        .with_unsupported_feature(TableFormatUnsupportedFeature::for_file(
            TableFormatUnsupportedFeatureCode::IcebergObjectStoreReadUnsupported,
            "s3-data-file",
            "S3 source requires staging or native object-store read support",
        ))
        .with_unsupported_feature(TableFormatUnsupportedFeature::new(
            TableFormatUnsupportedFeatureCode::IcebergResidualPredicateUnsupported,
            "residual predicate remains unsupported",
        ));

    let staged = native_iceberg_smoke_report::rewrite_scan_plan_for_staged_object_store_files(
        plan,
        &[StagedObjectStoreDataFile {
            id: "s3-data-file".to_string(),
            local_path: "/tmp/staged-data-0001.parquet".into(),
            staged_bytes: 1234,
        }],
    );

    assert_eq!(staged.data_files[0].location, "/tmp/staged-data-0001.parquet");
    assert_eq!(staged.data_files[0].read_capability, TableFormatReadCapability::local_file_system());
    assert_eq!(staged.data_files[0].size_bytes, Some(1234));
    assert_eq!(staged.rejected_feature_codes(), vec!["iceberg_residual_predicate_unsupported"]);
    assert!(!staged.is_native_admissible());
}

#[test]
fn report_tsv_distinguishes_native_source_read_from_cpu_arrow_fallback() {
    let plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_snapshot_id("123")
        .with_catalog_type("local")
        .with_data_file(
            TableFormatDataFile::new(
                "data-0001",
                "/tmp/iceberg/data-0001.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::local_file_system(),
            )
            .with_size_bytes(1024)
            .with_row_count(7)
            .with_resolved_parquet_source(
                resolve_table_format_parquet_source(
                    "/tmp/iceberg/data-0001.parquet",
                    &TableFormatReadCapability::local_file_system(),
                    &TableFormatParquetReadStrategyConfig::default(),
                )
                .unwrap(),
            ),
        );
    let diagnostics = plan.diagnostics();
    let mut row = native_iceberg_smoke_report::IcebergNativeSmokeReportRow::new(
        "iceberg_smoke".to_string(),
        "local_metadata".to_string(),
        "local".to_string(),
        "local_db.local_table".to_string(),
        "select count(*) as row_count from local_table".to_string(),
        diagnostics,
        NativeDataFusionPlanningOutcome::AcceptedReplaced,
        None,
        NativeDataFusionPostNativeExecutionPath::Native,
        Some(7),
        ObjectStoreStagingSummary::new(ObjectStoreStagingMode::Auto, ObjectStoreStagingStatus::NotNeeded),
        "not_applicable".to_string(),
    );
    row.native_runtime_started = true;
    row.native_runtime_outcome = datafusion_nexus::NativeDataFusionRuntimeOutcome::Succeeded;
    row.native_output_rows = Some(1);
    row.native_output_batches = Some(1);
    row.native_count_value = Some(7);
    row.record_native_metrics(NativeDataFusionExecutionMetrics {
        source_fetch_bytes: 1024,
        source_fetch_count: 2,
        host_fetch_budget_wait_time_ns: Some(5_000_000),
        source_fetch_concurrency_wait_time_ns: Some(6_000_000),
        source_fetch_time_ns: Some(7_250_000),
        source_decode_time_ns: Some(8_500_000),
        source_read_runtime_rows: vec![NativeSourceReadRuntimeMetricRow {
            source_node_id: "source-0".to_string(),
            read_kind: "chunk".to_string(),
            read_strategy: "remote_direct".to_string(),
            parquet_read_input_kind: "remote_path".to_string(),
            remote_read_mode: "require_direct".to_string(),
            parquet_read_backend: "kvikio_remote".to_string(),
            remote_backend_build_status: "kvikio_remote_io_enabled".to_string(),
            remote_read_bytes: Some(2048),
            remote_read_range_count: Some(3),
            device_read_support: "supported".to_string(),
            ..Default::default()
        }],
        lakehouse_cache_fill_event_snapshot_rows: vec![NativeLakehouseCacheFillEventMetricRow {
            outcome: "completed".to_string(),
            bytes: Some(4096),
            elapsed_ns: 9_000_000,
            cache_path_hash: "cache123".to_string(),
            stable_id_hash: "stable456".to_string(),
            file_index: Some(0),
            error_code: "none".to_string(),
            error_kind: "none".to_string(),
            error_status: "none".to_string(),
            error_operation: "none".to_string(),
            ..Default::default()
        }],
        ..Default::default()
    });
    row.finalize_count_parity_if_needed();
    let report = native_iceberg_smoke_report::IcebergNativeSmokeReport { rows: vec![row.clone()] };

    let tsv = report.to_tsv();
    assert!(tsv.starts_with("# native_iceberg_smoke_report_tsv_schema_version=10\n"));
    assert!(tsv.contains("\tnative_source_read_capability\t"));
    assert!(tsv.contains("\tsource_read_strategy\t"));
    assert!(tsv.contains("\tsource_decode_input\t"));
    assert!(tsv.contains("\tcredential_mode\t"));
    assert!(tsv.contains("\tlakehouse_cache_status\t"));
    assert!(tsv.contains("\tlakehouse_cache_read_mode\t"));
    assert!(tsv.contains("\tlakehouse_cache_fill_mode\t"));
    assert!(tsv.contains("\tsource_fallback_reason\t"));
    assert!(tsv.contains("\tdirect_device_read_disposition\t"));
    assert!(tsv.contains("\tobject_store_staging_mode\t"));
    assert!(tsv.contains("\tobject_store_staging_status\t"));
    assert!(tsv.contains("\tcpu_arrow_reader_fallback\t"));
    assert!(tsv.contains("\tnative_runtime_elapsed_ms\t"));
    assert!(tsv.contains("\tcpu_runtime_started\t"));
    assert!(tsv.contains("\tcpu_runtime_elapsed_ms\t"));
    assert!(tsv.contains("\tcpu_output_fingerprint\t"));
    assert!(tsv.contains("\tcomparison_mode\t"));
    assert!(tsv.contains("\tparity_status\t"));
    assert!(tsv.contains("\tsource_fetch_bytes\tsource_fetch_count\t"));
    assert!(tsv.contains("\thost_fetch_budget_wait_time_ms\tsource_fetch_concurrency_wait_time_ms\t"));
    assert!(tsv.contains("\tnative_source_read_runtime_rows\t"));
    assert!(tsv.contains("\tlakehouse_cache_fill_event_snapshot_rows\t"));
    assert_eq!(tsv_field(&tsv, "source_fetch_bytes"), "1024");
    assert_eq!(tsv_field(&tsv, "source_fetch_count"), "2");
    assert!(tsv_field(&tsv, "native_source_read_runtime_rows").contains("remote_read_bytes=2048"));
    assert!(tsv_field(&tsv, "lakehouse_cache_fill_event_snapshot_rows").contains("outcome=completed"));
    assert!(tsv_field(&tsv, "lakehouse_cache_fill_event_snapshot_rows").contains("bytes=4096"));
    assert_eq!(tsv_field(&tsv, "host_fetch_budget_wait_time_ms"), "5.000");
    assert_eq!(tsv_field(&tsv, "source_fetch_concurrency_wait_time_ms"), "6.000");
    assert_eq!(tsv_field(&tsv, "source_fetch_time_ms"), "7.250");
    assert_eq!(tsv_field(&tsv, "source_decode_time_ms"), "8.500");
    assert_eq!(tsv_field(&tsv, "source_read_strategy"), "local_kvikio");
    assert_eq!(tsv_field(&tsv, "source_decode_input"), "filepath");
    assert_eq!(tsv_field(&tsv, "credential_mode"), "none");
    assert_eq!(tsv_field(&tsv, "lakehouse_cache_status"), "disabled");
    assert_eq!(tsv_field(&tsv, "lakehouse_cache_read_mode"), "disabled");
    assert_eq!(tsv_field(&tsv, "lakehouse_cache_fill_mode"), "disabled");
    assert_eq!(tsv_field(&tsv, "source_fallback_reason"), "none");
    assert_eq!(tsv_field(&tsv, "object_store_staging_mode"), "auto");
    assert_eq!(tsv_field(&tsv, "object_store_staging_status"), "not_needed");
    assert_eq!(tsv_field(&tsv, "native_planning_outcome"), "accepted_replaced");
    assert_eq!(tsv_field(&tsv, "post_native_execution_path"), "native");
    assert_eq!(tsv_field(&tsv, "cpu_runtime_started"), "false");
    assert_eq!(tsv_field(&tsv, "cpu_runtime_outcome"), "not_started");
    assert_eq!(tsv_field(&tsv, "comparison_mode"), "not_applicable");
    assert!(tsv.contains("\tmatched\t"));

    let mut writer = FlushCountingWriter::default();
    native_iceberg_smoke_report::write_smoke_tsv_header(&mut writer).unwrap();
    native_iceberg_smoke_report::write_smoke_tsv_row(&mut writer, &row).unwrap();
    assert_eq!(writer.flushes, 2);
    let streamed = String::from_utf8(writer.bytes).unwrap();
    assert_eq!(streamed, tsv);
}

#[test]
fn report_tsv_records_direct_s3_as_remote_direct_native_without_staging() {
    let plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_snapshot_id("123")
        .with_catalog_type("aws_glue")
        .with_data_file(
            TableFormatDataFile::new(
                "s3-data-0001",
                "s3://bucket/table/data-0001.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::object_store("s3"),
            )
            .with_size_bytes(1024)
            .with_resolved_parquet_source(
                resolve_table_format_parquet_source(
                    "s3://bucket/table/data-0001.parquet",
                    &TableFormatReadCapability::object_store("s3"),
                    &TableFormatParquetReadStrategyConfig::default(),
                )
                .unwrap(),
            ),
        );
    let row = native_iceberg_smoke_report::IcebergNativeSmokeReportRow::new(
        "iceberg_smoke".to_string(),
        "glue_s3".to_string(),
        "aws_glue".to_string(),
        "tpch_sf1.lineitem".to_string(),
        "select count(*) as row_count from lineitem".to_string(),
        plan.diagnostics(),
        NativeDataFusionPlanningOutcome::AcceptedReplaced,
        None,
        NativeDataFusionPostNativeExecutionPath::Native,
        None,
        ObjectStoreStagingSummary::new(ObjectStoreStagingMode::Disabled, ObjectStoreStagingStatus::DirectNative),
        "aws_profile_env".to_string(),
    );
    let report = native_iceberg_smoke_report::IcebergNativeSmokeReport { rows: vec![row] };

    let tsv = report.to_tsv();
    assert_eq!(tsv_field(&tsv, "native_source_read_capability"), "object_store:s3");
    assert_eq!(tsv_field(&tsv, "source_read_strategy"), "remote_direct");
    assert_eq!(tsv_field(&tsv, "source_decode_input"), "filepath");
    assert_eq!(tsv_field(&tsv, "object_store_staging_mode"), "disabled");
    assert_eq!(tsv_field(&tsv, "object_store_staging_status"), "direct_s3_native");
    assert_eq!(tsv_field(&tsv, "native_planning_outcome"), "accepted_replaced");
    assert!(!tsv.contains("staged_local"));
}

#[test]
fn direct_s3_scan_plan_produces_cpu_baseline_listing_url_without_staging() {
    let plan = TableFormatScanPlan::new(TableFormat::Iceberg)
        .with_data_file(
            TableFormatDataFile::new(
                "s3-data-0001",
                "s3://bucket/table/data-0001.parquet",
                TableFormatDataFileFormat::Parquet,
                TableFormatReadCapability::object_store("s3"),
            )
            .with_size_bytes(1024),
        )
        .with_data_file(TableFormatDataFile::new(
            "gs-data-0001",
            "gs://bucket/table/data-0001.parquet",
            TableFormatDataFileFormat::Parquet,
            TableFormatReadCapability::object_store("gs"),
        ));

    let urls = native_iceberg_smoke_report::cpu_baseline_listing_urls_from_scan_plan(&plan).unwrap();

    assert_eq!(urls.len(), 1);
    assert_eq!(urls[0].as_str(), "s3://bucket/table/data-0001.parquet");
    assert_eq!(urls[0].object_store().as_str(), "s3://bucket/");
}

fn tsv_field<'a>(tsv: &'a str, column: &str) -> &'a str {
    let mut lines = tsv.lines();
    let _schema_version = lines.next().expect("schema version line");
    let header = lines.next().expect("header line").split('\t').collect::<Vec<_>>();
    let row = lines.next().expect("data row").split('\t').collect::<Vec<_>>();
    let index = header.iter().position(|field| *field == column).expect("column present");
    row[index]
}

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
