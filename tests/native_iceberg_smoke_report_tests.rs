#[allow(dead_code)]
#[path = "../src/bin/native_iceberg_smoke_report.rs"]
mod native_iceberg_smoke_report;

use std::collections::HashMap;

use datafusion_cudf::{
    NativeDataFusionPlanningOutcome, NativeDataFusionPostNativeExecutionPath, TableFormat, TableFormatDataFile,
    TableFormatDataFileFormat, TableFormatReadCapability, TableFormatScanPlan, TableFormatUnsupportedFeature,
    TableFormatUnsupportedFeatureCode,
};
use native_iceberg_smoke_report::{
    IcebergNativeSmokeConfig, IcebergNativeSmokeFixture, ObjectStoreStagingMode, ObjectStoreStagingStatus,
    ObjectStoreStagingSummary, StagedObjectStoreDataFile,
};

#[test]
fn glue_s3_defaults_use_aws_profile_without_exposing_credential_value_in_report_config() {
    let env = HashMap::from([("AWS_PROFILE", "xdr-int"), ("CUDF_ICEBERG_SMOKE_FIXTURE_MODE", "glue_s3")]);
    let config = IcebergNativeSmokeConfig::from_env_get(|name| env.get(name).map(|value| value.to_string())).unwrap();

    let IcebergNativeSmokeFixture::GlueS3 { profile, database, table, catalog_id, region, .. } = &config.fixture else {
        panic!("expected glue_s3 fixture");
    };
    assert_eq!(profile.as_deref(), Some("xdr-int"));
    assert_eq!(database, "demo_db");
    assert_eq!(table, "tpch_sf1_lineitem");
    assert_eq!(catalog_id.as_deref(), Some("123456789012"));
    assert_eq!(region, "us-east-1");
    assert_eq!(config.table_alias, "tpch_sf1_lineitem");
    assert_eq!(config.sql, "select count(*) as row_count from tpch_sf1_lineitem");
    assert_eq!(config.fixture.credential_source(), "aws_profile_env");
    assert_eq!(config.object_store_staging, ObjectStoreStagingMode::Auto);
    assert_eq!(config.tpch_query_ids, vec!["q01", "q03", "q06", "q10"]);
    assert_eq!(config.tpch_scale_factor, "sf1");
    assert_eq!(
        native_iceberg_smoke_report::tpch_glue_table_name("orders", &config.tpch_scale_factor),
        "tpch_sf1_orders"
    );
}

#[test]
fn local_metadata_mode_requires_metadata_location_and_parses_table_ident() {
    let env = HashMap::from([
        ("CUDF_ICEBERG_SMOKE_FIXTURE_MODE", "local_metadata"),
        ("CUDF_ICEBERG_METADATA_LOCATION", "/tmp/table/metadata/v1.metadata.json"),
        ("CUDF_ICEBERG_TABLE_IDENT", "local_db.local_table"),
    ]);
    let config = IcebergNativeSmokeConfig::from_env_get(|name| env.get(name).map(|value| value.to_string())).unwrap();

    let IcebergNativeSmokeFixture::LocalMetadata { metadata_location, table_ident } = &config.fixture else {
        panic!("expected local_metadata fixture");
    };
    assert_eq!(metadata_location, "/tmp/table/metadata/v1.metadata.json");
    assert_eq!(table_ident, &vec!["local_db".to_string(), "local_table".to_string()]);
    assert_eq!(config.table_alias, "local_table");
    assert!(config.tpch_query_ids.is_empty());
}

#[test]
fn explicit_glue_table_keeps_single_table_count_smoke_by_default() {
    let env = HashMap::from([
        ("AWS_PROFILE", "xdr-int"),
        ("CUDF_ICEBERG_SMOKE_FIXTURE_MODE", "glue_s3"),
        ("CUDF_ICEBERG_GLUE_TABLE", "tpch_sf1_lineitem"),
    ]);
    let config = IcebergNativeSmokeConfig::from_env_get(|name| env.get(name).map(|value| value.to_string())).unwrap();

    assert!(config.tpch_query_ids.is_empty());
    assert_eq!(config.table_alias, "tpch_sf1_lineitem");
    assert_eq!(config.sql, "select count(*) as row_count from tpch_sf1_lineitem");
}

#[test]
fn object_store_staging_mode_env_accepts_disabled_and_required_aliases() {
    let disabled = HashMap::from([
        ("CUDF_ICEBERG_SMOKE_FIXTURE_MODE", "glue_s3"),
        ("CUDF_ICEBERG_SMOKE_OBJECT_STORE_STAGING", "disabled"),
    ]);
    let disabled_config =
        IcebergNativeSmokeConfig::from_env_get(|name| disabled.get(name).map(|value| value.to_string())).unwrap();
    assert_eq!(disabled_config.object_store_staging, ObjectStoreStagingMode::Disabled);

    let required = HashMap::from([
        ("CUDF_ICEBERG_SMOKE_FIXTURE_MODE", "glue_s3"),
        ("CUDF_ICEBERG_SMOKE_OBJECT_STORE_STAGING", "1"),
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
            .with_row_count(7),
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
    row.native_runtime_outcome = datafusion_cudf::NativeDataFusionRuntimeOutcome::Succeeded;
    row.native_output_rows = Some(1);
    row.native_output_batches = Some(1);
    row.native_count_value = Some(7);
    row.finalize_count_parity_if_needed();
    let report = native_iceberg_smoke_report::IcebergNativeSmokeReport { rows: vec![row] };

    let tsv = report.to_tsv();
    assert!(tsv.starts_with("# native_iceberg_smoke_report_tsv_schema_version=4\n"));
    assert!(tsv.contains("\tnative_source_read_capability\t"));
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
    assert!(
        tsv.contains("\tlocal_file_system\teligible\tauto\tnot_needed\t0\t0\taccepted_replaced\tnone\tnative\tfalse\ttrue\tsucceeded\t")
    );
    assert!(tsv.contains("\tfalse\tnot_started\tnone\tnone\tnone\tnone\tnone\tnot_applicable\tmatched\t"));
    assert!(tsv.contains("\tmatched\t"));
}

#[test]
fn report_tsv_can_prove_direct_s3_native_read_without_staging() {
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
            .with_size_bytes(1024),
        );
    let row = native_iceberg_smoke_report::IcebergNativeSmokeReportRow::new(
        "iceberg_smoke".to_string(),
        "glue_s3".to_string(),
        "aws_glue".to_string(),
        "demo_db.tpch_sf1_lineitem".to_string(),
        "select count(*) as row_count from tpch_sf1_lineitem".to_string(),
        plan.diagnostics(),
        NativeDataFusionPlanningOutcome::AcceptedReplaced,
        None,
        NativeDataFusionPostNativeExecutionPath::Native,
        None,
        ObjectStoreStagingSummary::new(ObjectStoreStagingMode::Auto, ObjectStoreStagingStatus::DirectNative),
        "aws_profile_env".to_string(),
    );
    let report = native_iceberg_smoke_report::IcebergNativeSmokeReport { rows: vec![row] };

    let tsv = report.to_tsv();
    assert!(tsv.contains("\tobject_store:s3\teligible\tauto\tdirect_s3_native\t0\t0\taccepted_replaced\t"));
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
