use std::{
    collections::{BTreeSet, HashMap, HashSet},
    error::Error as StdError,
    fmt, fs,
    io::{self, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Instant,
};

use arrow::{
    array::{Array, Int32Array, Int64Array, UInt32Array, UInt64Array},
    record_batch::RecordBatch,
};
use aws_config::BehaviorVersion;
use aws_credential_types::provider::ProvideCredentials;
use cudf_query_engine::{ParquetReadStrategy, SourceSchedulerPolicy};
use datafusion::{
    datasource::{
        file_format::parquet::ParquetFormat,
        listing::{ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl},
    },
    execution::SessionStateBuilder,
    physical_plan::{collect, metrics::MetricsSet},
    prelude::{SessionConfig, SessionContext},
};
use datafusion_cudf::{
    CudfNativePhysicalOptimizerRule, CudfOptimizerConfig, CudfSessionStateBuilderExt, Error, ErrorCode, FallbackReason,
    GpuFallbackPolicy, NativeCudfExecutionConfig, NativeDataFusionExecutionMetrics, NativeDataFusionPlanningOutcome,
    NativeDataFusionPostNativeExecutionPath, NativeDataFusionRuntimeOutcome, NativePipelineExecutionMode,
    ParityComparisonMode, ParityFloatTolerance, ParityStatus, StatisticalAggregateExecutionMode, TPCDS_QUERIES,
    TPCDS_TABLES, TPCH_QUERIES, TPCH_TABLES, TableFormatDataFileFormat, TableFormatDirectDeviceReadDisposition,
    TableFormatParquetReadStrategyConfig, TableFormatParquetSourceMetadata, TableFormatPlanner,
    TableFormatReadCapability, TableFormatScanDiagnostics, TableFormatScanPlan, TableFormatScanRequest,
    TableFormatUnsupportedFeatureCode, compare_record_batches,
    iceberg::{
        AWS_DEFAULT_REGION_ENV, AWS_PROFILE_ENV, AWS_REGION_ENV, CUDF_ICEBERG_GLUE_CATALOG_ID_ENV,
        CUDF_ICEBERG_GLUE_CATALOG_NAME_ENV, CUDF_ICEBERG_GLUE_DATABASE_ENV, CUDF_ICEBERG_GLUE_TABLE_ENV,
        CUDF_ICEBERG_GLUE_WAREHOUSE_ENV, CudfIcebergTableProvider, GlueIcebergCredentialSource, GlueIcebergTableConfig,
        IcebergTableFormatPlanner,
        catalog::{IcebergCatalogBackend, RestIcebergCatalogBackend, RestIcebergCatalogConfig},
        iceberg_schema_to_datafusion_arrow_schema, load_glue_table,
    },
    resolve_table_format_parquet_source_with_metadata, sql_has_top_level_order_by,
    sql_top_level_order_by_covers_output_columns, tpch_tuned_cpu_session_config,
};
use futures::future::BoxFuture;
use iceberg::{
    NamespaceIdent, TableIdent,
    io::FileIO,
    table::{StaticTable, Table},
};
use object_store::aws::AmazonS3Builder;

type SmokeResult<T> = Result<T, Box<dyn StdError + Send + Sync>>;

const REPORT_SCHEMA_VERSION: u32 = 6;
const REPORT_HEADER: &str = "query_id\tfixture_mode\tcatalog_type\ttable_ident\tsql\ttable_format\tsnapshot_id\tplanned_data_file_count\tplanned_delete_file_count\tplanned_bytes\tadmitted_native_source_file_count\trejected_feature_list\tnative_source_read_capability\tsource_read_strategy\tsource_decode_input\tcredential_mode\tlakehouse_cache_status\tdirect_device_read_disposition\tobject_store_staging_mode\tobject_store_staging_status\tstaged_data_file_count\tstaged_bytes\tnative_planning_outcome\tnative_reason_code\tpost_native_execution_path\tcpu_arrow_reader_fallback\tnative_runtime_started\tnative_runtime_outcome\tnative_runtime_elapsed_ms\tnative_output_rows\tnative_output_batches\tsource_fetch_bytes\tsource_fetch_count\thost_fetch_budget_wait_time_ms\tsource_fetch_concurrency_wait_time_ms\tsource_fetch_time_ms\tsource_decode_time_ms\texpected_count\tnative_count_value\tcpu_runtime_started\tcpu_runtime_outcome\tcpu_runtime_elapsed_ms\tcpu_output_rows\tcpu_output_batches\tcpu_output_fingerprint\tnative_output_fingerprint\tcomparison_mode\tparity_status\tparity_detail\tcredential_source";

const FIXTURE_MODE_ENV: &str = "CUDF_ICEBERG_SMOKE_FIXTURE_MODE";
const LOCAL_METADATA_LOCATION_ENV: &str = "CUDF_ICEBERG_METADATA_LOCATION";
const LOCAL_TABLE_IDENT_ENV: &str = "CUDF_ICEBERG_TABLE_IDENT";
const TABLE_ALIAS_ENV: &str = "CUDF_ICEBERG_TABLE_ALIAS";
const QUERY_ENV: &str = "CUDF_ICEBERG_SMOKE_QUERY";
const QUERY_IDS_ENV: &str = "CUDF_ICEBERG_SMOKE_QUERY_IDS";
const TPCDS_QUERY_IDS_ENV: &str = "CUDF_ICEBERG_TPCDS_SMOKE_QUERY_IDS";
const CUSTOM_WORKLOAD_ENV: &str = "CUDF_ICEBERG_CUSTOM_WORKLOAD";
const CUSTOM_QUERY_FILE_ENV: &str = "CUDF_ICEBERG_CUSTOM_QUERY_FILE";
const STREAM_TSV_ENV: &str = "CUDF_ICEBERG_SMOKE_STREAM_TSV";
const CONTINUE_ON_QUERY_ERROR_ENV: &str = "CUDF_ICEBERG_SMOKE_CONTINUE_ON_QUERY_ERROR";
const SMOKE_EXECUTION_MODE_ENV: &str = "CUDF_ICEBERG_SMOKE_EXECUTION_MODE";
const CPU_BASELINE_TIMEOUT_SECS_ENV: &str = "CUDF_ICEBERG_SMOKE_CPU_BASELINE_TIMEOUT_SECS";
const NATIVE_DEVICE_BUDGET_BYTES_ENV: &str = "CUDF_ICEBERG_NATIVE_DEVICE_BUDGET_BYTES";
const NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV: &str = "CUDF_ICEBERG_NATIVE_MAX_SOURCE_CHUNK_BYTES";
const NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV: &str = "CUDF_ICEBERG_NATIVE_MAX_ROW_GROUPS_PER_CHUNK";
const NATIVE_MAX_RETRY_ATTEMPTS_ENV: &str = "CUDF_ICEBERG_NATIVE_MAX_RETRY_ATTEMPTS";
const NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV: &str = "CUDF_ICEBERG_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS";
const NATIVE_MAX_SOURCE_READS_IN_FLIGHT_ENV: &str = "CUDF_ICEBERG_NATIVE_MAX_SOURCE_READS_IN_FLIGHT";
const NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV: &str =
    "CUDF_ICEBERG_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE";
const NATIVE_STDDEV_SAMP_EXECUTION_MODE_ENV: &str = "CUDF_ICEBERG_NATIVE_STDDEV_SAMP_EXECUTION_MODE";
const TPCH_SCALE_FACTOR_ENV: &str = "CUDF_ICEBERG_TPCH_SCALE_FACTOR";
const TPCDS_SCALE_FACTOR_ENV: &str = "CUDF_ICEBERG_TPCDS_SCALE_FACTOR";
const OBJECT_STORE_STAGING_ENV: &str = "CUDF_ICEBERG_SMOKE_OBJECT_STORE_STAGING";
const REST_URI_ENV: &str = "DATAFUSION_CUDF_ICEBERG_REST_URI";
const REST_WAREHOUSE_ENV: &str = "DATAFUSION_CUDF_ICEBERG_WAREHOUSE";
const REST_NAMESPACE_ENV: &str = "DATAFUSION_CUDF_ICEBERG_NAMESPACE";
const REST_TABLE_ENV: &str = "CUDF_ICEBERG_REST_TABLE";
const REST_S3_ENDPOINT_ENV: &str = "DATAFUSION_CUDF_ICEBERG_S3_ENDPOINT";
const REST_S3_REGION_ENV: &str = "DATAFUSION_CUDF_ICEBERG_S3_REGION";
const REST_S3_PATH_STYLE_ENV: &str = "DATAFUSION_CUDF_ICEBERG_S3_PATH_STYLE";
const REST_S3_ACCESS_KEY_ID_ENV: &str = "DATAFUSION_CUDF_ICEBERG_S3_ACCESS_KEY_ID";
const REST_S3_SECRET_ACCESS_KEY_ENV: &str = "DATAFUSION_CUDF_ICEBERG_S3_SECRET_ACCESS_KEY";
const REST_S3_SESSION_TOKEN_ENV: &str = "DATAFUSION_CUDF_ICEBERG_S3_SESSION_TOKEN";
const DEFAULT_TPCH_QUERY_IDS: &[&str] = &["q01", "q03", "q06", "q10"];
const DEFAULT_SMOKE_GLUE_CATALOG_NAME: &str = "glue";
const DEFAULT_SMOKE_GLUE_DATABASE: &str = "demo_db";
const DEFAULT_SMOKE_GLUE_TABLE: &str = "tpch_sf1_lineitem";
const DEFAULT_SMOKE_GLUE_WAREHOUSE: &str = "s3://example-warehouse/demo_db/";
const DEFAULT_SMOKE_GLUE_CATALOG_ID: &str = "123456789012";
const DEFAULT_SMOKE_GLUE_REGION: &str = "us-east-1";
const DEFAULT_REST_URI: &str = "http://localhost:8181";
const DEFAULT_REST_WAREHOUSE: &str = "s3://lakehouse/warehouse";
const DEFAULT_REST_NAMESPACE: &str = "tpch_sf1";
const DEFAULT_REST_TABLE: &str = "lineitem";
const TPCH_NATIVE_MAX_SOURCE_CHUNK_BYTES_DEFAULT: u64 = 32 * 1024 * 1024 * 1024;
const TPCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_DEFAULT: usize = 1024;
const TPCH_NATIVE_MAX_SOURCE_READS_IN_FLIGHT_DEFAULT: usize = 2;
#[tokio::main]
async fn main() {
    let config = match IcebergNativeSmokeConfig::from_env() {
        Ok(config) => config,
        Err(error) => {
            eprintln!("[native_iceberg_smoke_report] error={}", stable_tsv_field(&error.to_string()));
            std::process::exit(1);
        },
    };
    if config.stream_tsv_rows {
        let stdout = io::stdout();
        let mut stdout = stdout.lock();
        match run_report_streaming(config, &mut stdout).await {
            Ok(_report) => {},
            Err(error) => {
                eprintln!("[native_iceberg_smoke_report] error={}", stable_tsv_field(&error.to_string()));
                std::process::exit(1);
            },
        }
    } else {
        match run_report(config).await {
            Ok(report) => print!("{}", report.to_tsv()),
            Err(error) => {
                eprintln!("[native_iceberg_smoke_report] error={}", stable_tsv_field(&error.to_string()));
                std::process::exit(1);
            },
        }
    };
}

#[allow(dead_code)]
pub(crate) async fn run_from_env() -> SmokeResult<IcebergNativeSmokeReport> {
    run_report(IcebergNativeSmokeConfig::from_env()?).await
}

pub(crate) async fn run_report(config: IcebergNativeSmokeConfig) -> SmokeResult<IcebergNativeSmokeReport> {
    run_report_with_optional_stream(config, None).await
}

pub(crate) async fn run_report_streaming(
    config: IcebergNativeSmokeConfig,
    writer: &mut dyn Write,
) -> SmokeResult<IcebergNativeSmokeReport> {
    write_smoke_tsv_header(writer)?;
    run_report_with_optional_stream(config, Some(writer)).await
}

async fn run_report_with_optional_stream(
    config: IcebergNativeSmokeConfig,
    mut stream: Option<&mut dyn Write>,
) -> SmokeResult<IcebergNativeSmokeReport> {
    if config.custom_query_file.is_some() {
        return run_custom_subset_report(config, stream).await;
    }
    if !config.tpch_query_ids.is_empty() {
        let execution_mode = config.execution_mode;
        return run_tpch_subset_report_with_execution_mode(config, execution_mode, stream).await;
    }
    if !config.tpcds_query_ids.is_empty() {
        return run_tpcds_subset_report(config, stream).await;
    }
    eprintln!(
        "[native_iceberg_smoke_report] phase=iceberg_table_load_start fixture_mode={} table_ident={} credential_source={}",
        config.fixture.mode_code(),
        config.fixture.table_ident_display(),
        config.fixture.credential_source(),
    );
    let table = load_iceberg_table(&config.fixture).await?;
    eprintln!(
        "[native_iceberg_smoke_report] phase=iceberg_table_load_finish fixture_mode={} table_ident={}",
        config.fixture.mode_code(),
        config.fixture.table_ident_display(),
    );
    let file_io = table.file_io().clone();
    let catalog_type = config.fixture.catalog_type().to_string();
    let logical_schema = iceberg_table_arrow_schema(&table)?;
    let planner = IcebergTableFormatPlanner::new(table)
        .with_catalog_type(catalog_type.clone())
        .with_parquet_read_strategy_config(config.fixture.parquet_read_strategy_config());
    let staging_records = Arc::new(Mutex::new(Vec::new()));
    let staging_tempdir = Arc::new(tempfile::Builder::new().prefix("datafusion-cudf-iceberg-stage-").tempdir()?);
    let planner = Arc::new(StagingTableFormatPlanner::new(
        planner,
        file_io,
        config.object_store_staging,
        Arc::clone(&staging_tempdir),
        Arc::clone(&staging_records),
        config.table_alias.clone(),
    ));

    let provider = Arc::new(
        CudfIcebergTableProvider::from_table_format_planner(logical_schema, planner)
            .with_catalog_type(catalog_type.clone()),
    );
    let (ctx, native_rule) =
        native_session_context(config.fixture.source_storage_profile(), NativeExecutionProfile::Smoke)?;
    ctx.register_table(config.table_alias.as_str(), provider)?;
    let mut row = run_query_with_registered_native_tables(
        config.query_id.clone(),
        config.fixture.mode_code().to_string(),
        catalog_type,
        config.fixture.table_ident_display(),
        config.sql,
        &ctx,
        native_rule.as_ref(),
        Arc::clone(&staging_records),
        false,
        config.fixture.credential_source().to_string(),
        config.cpu_baseline_timeout_secs,
        CpuBaselineSessionProfile::Smoke,
        SmokeQueryExecutionMode::NativeOnly,
    )
    .await?;
    row.expected_count = staging_records_snapshot(&staging_records)
        .first()
        .and_then(|record| expected_count_from_scan_plan(&record.scan_plan));
    row.finalize_count_parity_if_needed();
    if let Some(writer) = stream.as_deref_mut() {
        write_smoke_tsv_row(writer, &row)?;
    }
    Ok(IcebergNativeSmokeReport { rows: vec![row] })
}

#[allow(dead_code)]
pub(crate) async fn run_single_tpch_query_report_with_execution_mode(
    mut config: IcebergNativeSmokeConfig,
    query_id: impl Into<String>,
    execution_mode: SmokeQueryExecutionMode,
) -> SmokeResult<IcebergNativeSmokeReportRow> {
    config.tpch_query_ids = vec![query_id.into()];
    let mut report = run_tpch_subset_report_with_execution_mode(config, execution_mode, None).await?;
    let row = report.rows.pop().ok_or_else(|| smoke_error("single TPC-H Iceberg report produced no rows"))?;
    if !report.rows.is_empty() {
        return Err(smoke_error("single TPC-H Iceberg report produced multiple rows"));
    }
    Ok(row)
}

async fn run_tpch_subset_report_with_execution_mode(
    config: IcebergNativeSmokeConfig,
    execution_mode: SmokeQueryExecutionMode,
    mut stream: Option<&mut dyn Write>,
) -> SmokeResult<IcebergNativeSmokeReport> {
    if !matches!(config.fixture, IcebergNativeSmokeFixture::GlueS3 { .. } | IcebergNativeSmokeFixture::RestS3 { .. }) {
        return Err(smoke_error(
            "TPC-H Iceberg smoke query ids require CUDF_ICEBERG_SMOKE_FIXTURE_MODE=glue_s3 or rest_s3",
        ));
    };

    let catalog_type = config.fixture.catalog_type().to_string();
    let staging_records = Arc::new(Mutex::new(Vec::new()));
    let staging_tempdir = Arc::new(tempfile::Builder::new().prefix("datafusion-cudf-iceberg-tpch-stage-").tempdir()?);
    let (ctx, native_rule) =
        native_session_context(config.fixture.source_storage_profile(), NativeExecutionProfile::TpchBenchmark)?;

    for table_name in TPCH_TABLES {
        let table_fixture = config.fixture.with_tpch_table(table_name, &config.tpch_scale_factor)?;
        eprintln!(
            "[native_iceberg_smoke_report] phase=iceberg_table_load_start fixture_mode={} table_ident={} credential_source={}",
            table_fixture.mode_code(),
            table_fixture.table_ident_display(),
            table_fixture.credential_source(),
        );
        let table = load_iceberg_table(&table_fixture).await?;
        eprintln!(
            "[native_iceberg_smoke_report] phase=iceberg_table_load_finish fixture_mode={} table_ident={}",
            table_fixture.mode_code(),
            table_fixture.table_ident_display(),
        );
        let logical_schema = iceberg_table_arrow_schema(&table)?;
        let file_io = table.file_io().clone();
        let planner = IcebergTableFormatPlanner::new(table)
            .with_catalog_type(catalog_type.clone())
            .with_parquet_read_strategy_config(table_fixture.parquet_read_strategy_config());
        let staging_planner = Arc::new(StagingTableFormatPlanner::new(
            planner,
            file_io,
            config.object_store_staging,
            Arc::clone(&staging_tempdir),
            Arc::clone(&staging_records),
            (*table_name).to_string(),
        ));
        let provider = Arc::new(
            CudfIcebergTableProvider::from_table_format_planner(logical_schema, staging_planner)
                .with_catalog_type(catalog_type.clone()),
        );
        ctx.register_table(*table_name, provider)?;
    }

    let mut rows = Vec::new();
    for query_id in &config.tpch_query_ids {
        clear_staging_records(&staging_records);
        let query = TPCH_QUERIES
            .iter()
            .find(|query| query.id == query_id)
            .ok_or_else(|| smoke_error(format!("unknown TPC-H query id {query_id:?}")))?;
        let table_ident = config.fixture.tpch_all_tables_ident_display(&config.tpch_scale_factor);
        let query_id = query.id.to_string();
        let sql = query.sql.to_string();
        let row = match run_query_with_registered_native_tables(
            query_id.clone(),
            config.fixture.mode_code().to_string(),
            catalog_type.clone(),
            table_ident.clone(),
            sql.clone(),
            &ctx,
            native_rule.as_ref(),
            Arc::clone(&staging_records),
            execution_mode.runs_cpu_baseline(),
            config.fixture.credential_source().to_string(),
            config.cpu_baseline_timeout_secs,
            CpuBaselineSessionProfile::TpchTuned,
            execution_mode,
        )
        .await
        {
            Ok(row) => row,
            Err(error) if config.continue_on_query_error => {
                eprintln!(
                    "[native_iceberg_smoke_report] phase=query_error query_id={} error={}",
                    query_id,
                    stable_tsv_field(&error.to_string()),
                );
                query_error_report_row(
                    query_id,
                    config.fixture.mode_code().to_string(),
                    catalog_type.clone(),
                    table_ident,
                    sql,
                    &staging_records,
                    config.fixture.credential_source().to_string(),
                    error.as_ref(),
                )
            },
            Err(error) => return Err(error),
        };
        if let Some(writer) = stream.as_deref_mut() {
            write_smoke_tsv_row(writer, &row)?;
        }
        rows.push(row);
    }
    Ok(IcebergNativeSmokeReport { rows })
}

async fn run_tpcds_subset_report(
    config: IcebergNativeSmokeConfig,
    mut stream: Option<&mut dyn Write>,
) -> SmokeResult<IcebergNativeSmokeReport> {
    if !matches!(config.fixture, IcebergNativeSmokeFixture::GlueS3 { .. } | IcebergNativeSmokeFixture::RestS3 { .. }) {
        return Err(smoke_error(
            "TPC-DS Iceberg smoke query ids require CUDF_ICEBERG_SMOKE_FIXTURE_MODE=glue_s3 or rest_s3",
        ));
    };

    let catalog_type = config.fixture.catalog_type().to_string();
    let staging_records = Arc::new(Mutex::new(Vec::new()));
    let staging_tempdir = Arc::new(tempfile::Builder::new().prefix("datafusion-cudf-iceberg-tpcds-stage-").tempdir()?);
    let (ctx, native_rule) =
        native_session_context(config.fixture.source_storage_profile(), NativeExecutionProfile::TpcdsBenchmark)?;

    for table_name in TPCDS_TABLES {
        let table_fixture = config.fixture.with_tpcds_table(table_name, &config.tpcds_scale_factor)?;
        eprintln!(
            "[native_iceberg_smoke_report] phase=iceberg_table_load_start fixture_mode={} table_ident={} credential_source={}",
            table_fixture.mode_code(),
            table_fixture.table_ident_display(),
            table_fixture.credential_source(),
        );
        let table = load_iceberg_table(&table_fixture).await?;
        eprintln!(
            "[native_iceberg_smoke_report] phase=iceberg_table_load_finish fixture_mode={} table_ident={}",
            table_fixture.mode_code(),
            table_fixture.table_ident_display(),
        );
        let logical_schema = iceberg_table_arrow_schema(&table)?;
        let file_io = table.file_io().clone();
        let planner = IcebergTableFormatPlanner::new(table)
            .with_catalog_type(catalog_type.clone())
            .with_parquet_read_strategy_config(table_fixture.parquet_read_strategy_config());
        let staging_planner = Arc::new(StagingTableFormatPlanner::new(
            planner,
            file_io,
            config.object_store_staging,
            Arc::clone(&staging_tempdir),
            Arc::clone(&staging_records),
            (*table_name).to_string(),
        ));
        let provider = Arc::new(
            CudfIcebergTableProvider::from_table_format_planner(logical_schema, staging_planner)
                .with_catalog_type(catalog_type.clone()),
        );
        ctx.register_table(*table_name, provider)?;
    }

    let mut rows = Vec::new();
    for query_id in &config.tpcds_query_ids {
        clear_staging_records(&staging_records);
        let query = TPCDS_QUERIES
            .iter()
            .find(|query| query.id == query_id)
            .ok_or_else(|| smoke_error(format!("unknown TPC-DS query id {query_id:?}")))?;
        let table_ident = config.fixture.tpcds_all_tables_ident_display(&config.tpcds_scale_factor);
        let query_id = query.id.to_string();
        let sql = query.sql.to_string();
        let row = match run_query_with_registered_native_tables(
            query_id.clone(),
            config.fixture.mode_code().to_string(),
            catalog_type.clone(),
            table_ident.clone(),
            sql.clone(),
            &ctx,
            native_rule.as_ref(),
            Arc::clone(&staging_records),
            config.execution_mode.runs_cpu_baseline(),
            config.fixture.credential_source().to_string(),
            config.cpu_baseline_timeout_secs,
            CpuBaselineSessionProfile::Smoke,
            config.execution_mode,
        )
        .await
        {
            Ok(row) => row,
            Err(error) if config.continue_on_query_error => {
                eprintln!(
                    "[native_iceberg_smoke_report] phase=query_error query_id={} error={}",
                    query_id,
                    stable_tsv_field(&error.to_string()),
                );
                query_error_report_row(
                    query_id,
                    config.fixture.mode_code().to_string(),
                    catalog_type.clone(),
                    table_ident,
                    sql,
                    &staging_records,
                    config.fixture.credential_source().to_string(),
                    error.as_ref(),
                )
            },
            Err(error) => return Err(error),
        };
        if let Some(writer) = stream.as_deref_mut() {
            write_smoke_tsv_row(writer, &row)?;
        }
        rows.push(row);
    }
    Ok(IcebergNativeSmokeReport { rows })
}

async fn run_custom_subset_report(
    config: IcebergNativeSmokeConfig,
    mut stream: Option<&mut dyn Write>,
) -> SmokeResult<IcebergNativeSmokeReport> {
    if !matches!(config.fixture, IcebergNativeSmokeFixture::GlueS3 { .. } | IcebergNativeSmokeFixture::RestS3 { .. }) {
        return Err(smoke_error(
            "custom Iceberg SQL probes require CUDF_ICEBERG_SMOKE_FIXTURE_MODE=glue_s3 or rest_s3",
        ));
    };
    let workload = config
        .custom_workload
        .ok_or_else(|| smoke_error(format!("{CUSTOM_WORKLOAD_ENV} is required with {CUSTOM_QUERY_FILE_ENV}")))?;
    let query_file =
        config.custom_query_file.as_ref().ok_or_else(|| smoke_error(format!("{CUSTOM_QUERY_FILE_ENV} is required")))?;
    let probes = load_custom_query_manifest(query_file)?;

    let catalog_type = config.fixture.catalog_type().to_string();
    let staging_records = Arc::new(Mutex::new(Vec::new()));
    let staging_tempdir = Arc::new(tempfile::Builder::new().prefix("datafusion-cudf-iceberg-custom-stage-").tempdir()?);
    let native_execution_profile = match workload {
        CustomWorkload::Tpch => NativeExecutionProfile::TpchBenchmark,
        CustomWorkload::Tpcds => NativeExecutionProfile::TpcdsBenchmark,
    };
    let (ctx, native_rule) = native_session_context(config.fixture.source_storage_profile(), native_execution_profile)?;

    match workload {
        CustomWorkload::Tpch => {
            for table_name in TPCH_TABLES {
                let table_fixture = config.fixture.with_tpch_table(table_name, &config.tpch_scale_factor)?;
                register_custom_probe_table(
                    &ctx,
                    &table_fixture,
                    table_name,
                    &catalog_type,
                    config.object_store_staging,
                    Arc::clone(&staging_tempdir),
                    Arc::clone(&staging_records),
                )
                .await?;
            }
        },
        CustomWorkload::Tpcds => {
            if !matches!(
                config.fixture,
                IcebergNativeSmokeFixture::GlueS3 { .. } | IcebergNativeSmokeFixture::RestS3 { .. }
            ) {
                return Err(smoke_error(
                    "custom TPC-DS probes require CUDF_ICEBERG_SMOKE_FIXTURE_MODE=glue_s3 or rest_s3",
                ));
            }
            for table_name in TPCDS_TABLES {
                let table_fixture = config.fixture.with_tpcds_table(table_name, &config.tpcds_scale_factor)?;
                register_custom_probe_table(
                    &ctx,
                    &table_fixture,
                    table_name,
                    &catalog_type,
                    config.object_store_staging,
                    Arc::clone(&staging_tempdir),
                    Arc::clone(&staging_records),
                )
                .await?;
            }
        },
    }

    let table_ident = match workload {
        CustomWorkload::Tpch => config.fixture.tpch_all_tables_ident_display(&config.tpch_scale_factor),
        CustomWorkload::Tpcds => config.fixture.tpcds_all_tables_ident_display(&config.tpcds_scale_factor),
    };
    let mut rows = Vec::new();
    for probe in probes {
        clear_staging_records(&staging_records);
        let query_id = probe.id;
        let sql = probe.sql;
        let row = match run_query_with_registered_native_tables(
            query_id.clone(),
            config.fixture.mode_code().to_string(),
            catalog_type.clone(),
            table_ident.clone(),
            sql.clone(),
            &ctx,
            native_rule.as_ref(),
            Arc::clone(&staging_records),
            config.execution_mode.runs_cpu_baseline(),
            config.fixture.credential_source().to_string(),
            config.cpu_baseline_timeout_secs,
            match workload {
                CustomWorkload::Tpch => CpuBaselineSessionProfile::TpchTuned,
                CustomWorkload::Tpcds => CpuBaselineSessionProfile::Smoke,
            },
            config.execution_mode,
        )
        .await
        {
            Ok(row) => row,
            Err(error) if config.continue_on_query_error => {
                eprintln!(
                    "[native_iceberg_smoke_report] phase=query_error query_id={} error={}",
                    query_id,
                    stable_tsv_field(&error.to_string()),
                );
                query_error_report_row(
                    query_id,
                    config.fixture.mode_code().to_string(),
                    catalog_type.clone(),
                    table_ident.clone(),
                    sql,
                    &staging_records,
                    config.fixture.credential_source().to_string(),
                    error.as_ref(),
                )
            },
            Err(error) => return Err(error),
        };
        if let Some(writer) = stream.as_deref_mut() {
            write_smoke_tsv_row(writer, &row)?;
        }
        rows.push(row);
    }
    Ok(IcebergNativeSmokeReport { rows })
}

async fn register_custom_probe_table(
    ctx: &SessionContext,
    table_fixture: &IcebergNativeSmokeFixture,
    table_name: &str,
    catalog_type: &str,
    object_store_staging: ObjectStoreStagingMode,
    staging_tempdir: Arc<tempfile::TempDir>,
    staging_records: Arc<Mutex<Vec<StagingRecord>>>,
) -> SmokeResult<()> {
    eprintln!(
        "[native_iceberg_smoke_report] phase=iceberg_table_load_start fixture_mode={} table_ident={} credential_source={}",
        table_fixture.mode_code(),
        table_fixture.table_ident_display(),
        table_fixture.credential_source(),
    );
    let table = load_iceberg_table(table_fixture).await?;
    eprintln!(
        "[native_iceberg_smoke_report] phase=iceberg_table_load_finish fixture_mode={} table_ident={}",
        table_fixture.mode_code(),
        table_fixture.table_ident_display(),
    );
    let logical_schema = iceberg_table_arrow_schema(&table)?;
    let file_io = table.file_io().clone();
    let planner = IcebergTableFormatPlanner::new(table)
        .with_catalog_type(catalog_type.to_string())
        .with_parquet_read_strategy_config(table_fixture.parquet_read_strategy_config());
    let staging_planner = Arc::new(StagingTableFormatPlanner::new(
        planner,
        file_io,
        object_store_staging,
        staging_tempdir,
        staging_records,
        table_name.to_string(),
    ));
    let provider = Arc::new(
        CudfIcebergTableProvider::from_table_format_planner(logical_schema, staging_planner)
            .with_catalog_type(catalog_type.to_string()),
    );
    ctx.register_table(table_name, provider)?;
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct IcebergNativeSmokeConfig {
    pub query_id: String,
    pub table_alias: String,
    pub sql: String,
    pub fixture: IcebergNativeSmokeFixture,
    pub object_store_staging: ObjectStoreStagingMode,
    pub tpch_query_ids: Vec<String>,
    pub tpch_scale_factor: String,
    pub tpcds_query_ids: Vec<String>,
    pub tpcds_scale_factor: String,
    pub custom_workload: Option<CustomWorkload>,
    pub custom_query_file: Option<PathBuf>,
    pub stream_tsv_rows: bool,
    pub continue_on_query_error: bool,
    pub execution_mode: SmokeQueryExecutionMode,
    pub cpu_baseline_timeout_secs: Option<u64>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SmokeQueryExecutionMode {
    NativeAndCpu,
    NativeOnly,
    CpuOnly,
}

impl SmokeQueryExecutionMode {
    fn from_env_value(value: &str) -> SmokeResult<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "native_and_cpu" | "native+cpu" | "all" => Ok(Self::NativeAndCpu),
            "native_only" | "native" => Ok(Self::NativeOnly),
            "cpu_only" | "cpu" => Ok(Self::CpuOnly),
            other => Err(smoke_error(format!(
                "unsupported {SMOKE_EXECUTION_MODE_ENV}={other}; expected native_and_cpu, native_only, or cpu_only"
            ))),
        }
    }

    const fn runs_native(self) -> bool {
        matches!(self, Self::NativeAndCpu | Self::NativeOnly)
    }

    const fn runs_cpu_baseline(self) -> bool {
        matches!(self, Self::NativeAndCpu | Self::CpuOnly)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CpuBaselineSessionProfile {
    Smoke,
    TpchTuned,
}

impl CpuBaselineSessionProfile {
    const fn code(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::TpchTuned => "tpch_tuned",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NativeExecutionProfile {
    Smoke,
    TpchBenchmark,
    TpcdsBenchmark,
}

impl IcebergNativeSmokeConfig {
    pub(crate) fn from_env() -> SmokeResult<Self> {
        Self::from_env_get(|name| std::env::var(name).ok())
    }

    pub(crate) fn from_env_get(mut get: impl FnMut(&str) -> Option<String>) -> SmokeResult<Self> {
        let mode = get(FIXTURE_MODE_ENV).unwrap_or_else(|| {
            if get(LOCAL_METADATA_LOCATION_ENV).is_some() {
                "local_metadata".to_string()
            } else {
                "glue_s3".to_string()
            }
        });
        let mut explicit_glue_table = false;
        let fixture = match mode.as_str() {
            "local_metadata" => {
                let metadata_location = required_env(&mut get, LOCAL_METADATA_LOCATION_ENV)?;
                let table_ident = parse_table_ident(
                    get(LOCAL_TABLE_IDENT_ENV).unwrap_or_else(|| "local.iceberg_t".to_string()).as_str(),
                )?;
                IcebergNativeSmokeFixture::LocalMetadata { metadata_location, table_ident }
            },
            "glue_s3" => {
                explicit_glue_table = get(CUDF_ICEBERG_GLUE_TABLE_ENV).is_some();
                let catalog_name = get(CUDF_ICEBERG_GLUE_CATALOG_NAME_ENV)
                    .unwrap_or_else(|| DEFAULT_SMOKE_GLUE_CATALOG_NAME.to_string());
                let database =
                    get(CUDF_ICEBERG_GLUE_DATABASE_ENV).unwrap_or_else(|| DEFAULT_SMOKE_GLUE_DATABASE.to_string());
                let table = get(CUDF_ICEBERG_GLUE_TABLE_ENV).unwrap_or_else(|| DEFAULT_SMOKE_GLUE_TABLE.to_string());
                let warehouse =
                    get(CUDF_ICEBERG_GLUE_WAREHOUSE_ENV).unwrap_or_else(|| DEFAULT_SMOKE_GLUE_WAREHOUSE.to_string());
                let region = get(AWS_REGION_ENV)
                    .or_else(|| get(AWS_DEFAULT_REGION_ENV))
                    .unwrap_or_else(|| DEFAULT_SMOKE_GLUE_REGION.to_string());
                let mut glue_config = GlueIcebergTableConfig::new(catalog_name, database, table, warehouse, region);
                if let Some(profile) = get(AWS_PROFILE_ENV).filter(|value| !value.trim().is_empty()) {
                    glue_config = glue_config.with_profile(profile);
                }
                glue_config.catalog_id = match get(CUDF_ICEBERG_GLUE_CATALOG_ID_ENV) {
                    Some(value) if value.trim().is_empty() => None,
                    Some(value) => Some(value),
                    None => Some(DEFAULT_SMOKE_GLUE_CATALOG_ID.to_string()),
                };
                let credential_source = glue_config.diagnostics_with_env(&mut get).credential_source;
                IcebergNativeSmokeFixture::GlueS3 {
                    catalog_name: glue_config.catalog_name,
                    database: glue_config.database,
                    table: glue_config.table,
                    warehouse: glue_config.warehouse,
                    catalog_id: glue_config.catalog_id,
                    region: glue_config.region,
                    profile: glue_config.profile,
                    credential_source,
                }
            },
            "rest_s3" => {
                explicit_glue_table = get(REST_TABLE_ENV).is_some();
                let catalog_handle = "lake".to_string();
                let uri = get(REST_URI_ENV).unwrap_or_else(|| DEFAULT_REST_URI.to_string());
                let warehouse = get(REST_WAREHOUSE_ENV).unwrap_or_else(|| DEFAULT_REST_WAREHOUSE.to_string());
                let namespace = get(REST_NAMESPACE_ENV).unwrap_or_else(|| DEFAULT_REST_NAMESPACE.to_string());
                let table = get(REST_TABLE_ENV).unwrap_or_else(|| DEFAULT_REST_TABLE.to_string());
                let s3_endpoint = get(REST_S3_ENDPOINT_ENV).filter(|value| !value.trim().is_empty());
                let s3_region = get(REST_S3_REGION_ENV).filter(|value| !value.trim().is_empty());
                let s3_path_style_access = get(REST_S3_PATH_STYLE_ENV)
                    .filter(|value| !value.trim().is_empty())
                    .map(|value| matches!(value.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"));
                let credential_source =
                    if get(REST_S3_ACCESS_KEY_ID_ENV).filter(|value| !value.trim().is_empty()).is_some()
                        && get(REST_S3_SECRET_ACCESS_KEY_ENV).filter(|value| !value.trim().is_empty()).is_some()
                    {
                        "static"
                    } else {
                        "default_chain"
                    };
                IcebergNativeSmokeFixture::RestS3 {
                    catalog_handle,
                    uri,
                    namespace,
                    table,
                    warehouse,
                    s3_endpoint,
                    s3_region,
                    s3_path_style_access,
                    credential_source,
                }
            },
            other => {
                return Err(smoke_error(format!(
                    "unsupported {FIXTURE_MODE_ENV}={other}; expected local_metadata, glue_s3, or rest_s3"
                )));
            },
        };
        let table_alias = get(TABLE_ALIAS_ENV).unwrap_or_else(|| fixture.default_table_alias());
        let query_env = get(QUERY_ENV);
        let custom_query_file = get(CUSTOM_QUERY_FILE_ENV).filter(|value| !value.trim().is_empty()).map(PathBuf::from);
        let custom_workload = get(CUSTOM_WORKLOAD_ENV)
            .filter(|value| !value.trim().is_empty())
            .map(|value| CustomWorkload::from_env_value(&value))
            .transpose()?;
        let sql = query_env.clone().unwrap_or_else(|| format!("select count(*) as row_count from {table_alias}"));
        let object_store_staging = match get(OBJECT_STORE_STAGING_ENV) {
            Some(value) => ObjectStoreStagingMode::from_env_value(Some(value.as_str()))?,
            None if matches!(fixture, IcebergNativeSmokeFixture::GlueS3 { .. }) => ObjectStoreStagingMode::Required,
            None => ObjectStoreStagingMode::Auto,
        };
        let tpch_scale_factor =
            normalize_tpch_scale_factor(get(TPCH_SCALE_FACTOR_ENV).unwrap_or_else(|| "sf1".to_string()).as_str())?;
        let tpcds_scale_factor =
            normalize_tpcds_scale_factor(get(TPCDS_SCALE_FACTOR_ENV).unwrap_or_else(|| "sf1".to_string()).as_str())?;
        let tpcds_query_ids_env = get(TPCDS_QUERY_IDS_ENV);
        let tpch_query_ids = match get(QUERY_IDS_ENV) {
            Some(value) => parse_query_ids(&value)?,
            None if query_env.is_none()
                && tpcds_query_ids_env.is_none()
                && custom_query_file.is_none()
                && !explicit_glue_table
                && matches!(fixture, IcebergNativeSmokeFixture::GlueS3 { .. }) =>
            {
                DEFAULT_TPCH_QUERY_IDS.iter().map(|query_id| (*query_id).to_string()).collect()
            },
            None => Vec::new(),
        };
        let tpcds_query_ids = match tpcds_query_ids_env {
            Some(value) => parse_tpcds_query_ids(&value)?,
            None => Vec::new(),
        };
        if !tpch_query_ids.is_empty() && !tpcds_query_ids.is_empty() {
            return Err(smoke_error(format!("{QUERY_IDS_ENV} and {TPCDS_QUERY_IDS_ENV} are mutually exclusive")));
        }
        let mode_count = usize::from(!tpch_query_ids.is_empty())
            + usize::from(!tpcds_query_ids.is_empty())
            + usize::from(custom_query_file.is_some());
        if mode_count > 1 {
            return Err(smoke_error(format!(
                "{QUERY_IDS_ENV}, {TPCDS_QUERY_IDS_ENV}, and {CUSTOM_QUERY_FILE_ENV} are mutually exclusive"
            )));
        }
        if custom_query_file.is_some() && custom_workload.is_none() {
            return Err(smoke_error(format!("{CUSTOM_WORKLOAD_ENV} is required with {CUSTOM_QUERY_FILE_ENV}")));
        }
        if custom_query_file.is_none() && custom_workload.is_some() {
            return Err(smoke_error(format!("{CUSTOM_QUERY_FILE_ENV} is required with {CUSTOM_WORKLOAD_ENV}")));
        }
        let execution_mode = get(SMOKE_EXECUTION_MODE_ENV)
            .filter(|value| !value.trim().is_empty())
            .map(|value| SmokeQueryExecutionMode::from_env_value(&value))
            .transpose()?
            .unwrap_or(SmokeQueryExecutionMode::NativeAndCpu);
        let stream_tsv_rows = parse_optional_bool(get(STREAM_TSV_ENV), STREAM_TSV_ENV)?;
        let continue_on_query_error =
            parse_optional_bool(get(CONTINUE_ON_QUERY_ERROR_ENV), CONTINUE_ON_QUERY_ERROR_ENV)?;
        let cpu_baseline_timeout_secs =
            parse_optional_timeout_secs(get(CPU_BASELINE_TIMEOUT_SECS_ENV), CPU_BASELINE_TIMEOUT_SECS_ENV)?;
        Ok(Self {
            query_id: "iceberg_smoke".to_string(),
            table_alias,
            sql,
            fixture,
            object_store_staging,
            tpch_query_ids,
            tpch_scale_factor,
            tpcds_query_ids,
            tpcds_scale_factor,
            custom_workload,
            custom_query_file,
            stream_tsv_rows,
            continue_on_query_error,
            execution_mode,
            cpu_baseline_timeout_secs,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CustomWorkload {
    Tpch,
    Tpcds,
}

impl CustomWorkload {
    fn from_env_value(value: &str) -> SmokeResult<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "tpch" | "tpc-h" => Ok(Self::Tpch),
            "tpcds" | "tpc-ds" => Ok(Self::Tpcds),
            other => Err(smoke_error(format!("unsupported {CUSTOM_WORKLOAD_ENV}={other}; expected tpch or tpcds"))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ObjectStoreStagingMode {
    Auto,
    Disabled,
    Required,
}

impl ObjectStoreStagingMode {
    fn from_env_value(value: Option<&str>) -> SmokeResult<Self> {
        match value.map(str::trim).filter(|value| !value.is_empty()) {
            None | Some("auto") => Ok(Self::Auto),
            Some("disabled" | "false" | "0" | "off") => Ok(Self::Disabled),
            Some("required" | "true" | "1" | "on") => Ok(Self::Required),
            Some(other) => Err(smoke_error(format!(
                "unsupported {OBJECT_STORE_STAGING_ENV}={other}; expected auto, disabled, or required"
            ))),
        }
    }

    const fn code(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Disabled => "disabled",
            Self::Required => "required",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum IcebergNativeSmokeFixture {
    LocalMetadata {
        metadata_location: String,
        table_ident: Vec<String>,
    },
    GlueS3 {
        catalog_name: String,
        database: String,
        table: String,
        warehouse: String,
        catalog_id: Option<String>,
        region: String,
        profile: Option<String>,
        credential_source: GlueIcebergCredentialSource,
    },
    RestS3 {
        catalog_handle: String,
        uri: String,
        namespace: String,
        table: String,
        warehouse: String,
        s3_endpoint: Option<String>,
        s3_region: Option<String>,
        s3_path_style_access: Option<bool>,
        credential_source: &'static str,
    },
}

impl IcebergNativeSmokeFixture {
    pub(crate) fn mode_code(&self) -> &'static str {
        match self {
            Self::LocalMetadata { .. } => "local_metadata",
            Self::GlueS3 { .. } => "glue_s3",
            Self::RestS3 { .. } => "rest_s3",
        }
    }

    pub(crate) fn catalog_type(&self) -> &'static str {
        match self {
            Self::LocalMetadata { .. } => "local",
            Self::GlueS3 { .. } => "glue",
            Self::RestS3 { .. } => "rest",
        }
    }

    fn default_table_alias(&self) -> String {
        match self {
            Self::LocalMetadata { table_ident, .. } => {
                table_ident.last().cloned().unwrap_or_else(|| "iceberg_t".to_string())
            },
            Self::GlueS3 { table, .. } => table.clone(),
            Self::RestS3 { table, .. } => table.clone(),
        }
    }

    fn table_ident_display(&self) -> String {
        match self {
            Self::LocalMetadata { table_ident, .. } => table_ident.join("."),
            Self::GlueS3 { database, table, .. } => format!("{database}.{table}"),
            Self::RestS3 { namespace, table, .. } => format!("{namespace}.{table}"),
        }
    }

    pub(crate) fn credential_source(&self) -> &'static str {
        match self {
            Self::LocalMetadata { .. } => "not_applicable",
            Self::GlueS3 { credential_source, .. } => credential_source.code(),
            Self::RestS3 { credential_source, .. } => credential_source,
        }
    }

    fn with_tpch_table(&self, table_name: &str, scale_factor: &str) -> SmokeResult<Self> {
        match self {
            Self::GlueS3 {
                catalog_name, database, warehouse, catalog_id, region, profile, credential_source, ..
            } => Ok(Self::GlueS3 {
                catalog_name: catalog_name.clone(),
                database: database.clone(),
                table: tpch_glue_table_name(table_name, scale_factor),
                warehouse: warehouse.clone(),
                catalog_id: catalog_id.clone(),
                region: region.clone(),
                profile: profile.clone(),
                credential_source: *credential_source,
            }),
            Self::RestS3 {
                catalog_handle,
                uri,
                namespace,
                warehouse,
                s3_endpoint,
                s3_region,
                s3_path_style_access,
                credential_source,
                ..
            } => Ok(Self::RestS3 {
                catalog_handle: catalog_handle.clone(),
                uri: uri.clone(),
                namespace: namespace.clone(),
                table: table_name.to_string(),
                warehouse: warehouse.clone(),
                s3_endpoint: s3_endpoint.clone(),
                s3_region: s3_region.clone(),
                s3_path_style_access: *s3_path_style_access,
                credential_source: *credential_source,
            }),
            Self::LocalMetadata { .. } => {
                Err(smoke_error("only glue_s3/rest_s3 fixtures can be rewritten to another TPC-H table"))
            },
        }
    }

    pub(crate) fn with_tpcds_table(&self, table_name: &str, scale_factor: &str) -> SmokeResult<Self> {
        match self {
            Self::GlueS3 {
                catalog_name, database, warehouse, catalog_id, region, profile, credential_source, ..
            } => Ok(Self::GlueS3 {
                catalog_name: catalog_name.clone(),
                database: database.clone(),
                table: tpcds_glue_table_name(table_name, scale_factor),
                warehouse: warehouse.clone(),
                catalog_id: catalog_id.clone(),
                region: region.clone(),
                profile: profile.clone(),
                credential_source: *credential_source,
            }),
            Self::RestS3 {
                catalog_handle,
                uri,
                namespace,
                warehouse,
                s3_endpoint,
                s3_region,
                s3_path_style_access,
                credential_source,
                ..
            } => Ok(Self::RestS3 {
                catalog_handle: catalog_handle.clone(),
                uri: uri.clone(),
                namespace: namespace.clone(),
                table: table_name.to_string(),
                warehouse: warehouse.clone(),
                s3_endpoint: s3_endpoint.clone(),
                s3_region: s3_region.clone(),
                s3_path_style_access: *s3_path_style_access,
                credential_source: *credential_source,
            }),
            Self::LocalMetadata { .. } => {
                Err(smoke_error("only glue_s3/rest_s3 fixtures can be rewritten to another TPC-DS table"))
            },
        }
    }

    pub(crate) fn tpch_all_tables_ident_display(&self, scale_factor: &str) -> String {
        match self {
            Self::GlueS3 { database, .. } => format!("{database}_tpch_{scale_factor}_all_tables"),
            Self::RestS3 { namespace, .. } => format!("{namespace}_tpch_{scale_factor}_all_tables"),
            Self::LocalMetadata { table_ident, .. } => table_ident.join("."),
        }
    }

    pub(crate) fn tpcds_all_tables_ident_display(&self, scale_factor: &str) -> String {
        match self {
            Self::GlueS3 { database, .. } => format!("{database}_tpcds_{scale_factor}_all_tables"),
            Self::RestS3 { namespace, .. } => format!("{namespace}_tpcds_{scale_factor}_all_tables"),
            Self::LocalMetadata { table_ident, .. } => table_ident.join("."),
        }
    }

    fn rest_config(&self) -> Option<RestIcebergCatalogConfig> {
        let Self::RestS3 { catalog_handle, uri, warehouse, s3_endpoint, s3_region, s3_path_style_access, .. } = self
        else {
            return None;
        };
        let mut config = RestIcebergCatalogConfig::new(catalog_handle.clone(), uri.clone(), warehouse.clone());
        if let Some(endpoint) = s3_endpoint.as_ref() {
            config = config.with_s3_endpoint(endpoint.clone());
        }
        if let Some(region) = s3_region.as_ref() {
            config = config.with_s3_region(region.clone());
        }
        if let Some(path_style) = s3_path_style_access {
            config = config.with_s3_path_style_access(*path_style);
        }
        Some(config)
    }

    fn parquet_read_strategy_config(&self) -> TableFormatParquetReadStrategyConfig {
        match self {
            Self::LocalMetadata { .. } => TableFormatParquetReadStrategyConfig::from_env(),
            Self::GlueS3 { catalog_name, warehouse, region, catalog_id, profile, .. } => {
                let mut config =
                    datafusion_cudf::iceberg::catalog::GlueIcebergCatalogConfig::new(catalog_name, warehouse, region);
                if let Some(catalog_id) = catalog_id.as_ref() {
                    config = config.with_catalog_id(catalog_id.clone());
                }
                if let Some(profile) = profile.as_ref() {
                    config = config.with_profile(profile.clone());
                }
                config.parquet_read_strategy_config()
            },
            Self::RestS3 { .. } => self.rest_config().expect("rest config").parquet_read_strategy_config(),
        }
    }

    fn source_storage_profile(&self) -> Option<datafusion_cudf::SourceStorageProfile> {
        match self {
            Self::LocalMetadata { .. } => None,
            Self::GlueS3 { catalog_name, warehouse, region, catalog_id, profile, .. } => {
                let mut config =
                    datafusion_cudf::iceberg::catalog::GlueIcebergCatalogConfig::new(catalog_name, warehouse, region);
                if let Some(catalog_id) = catalog_id.as_ref() {
                    config = config.with_catalog_id(catalog_id.clone());
                }
                if let Some(profile) = profile.as_ref() {
                    config = config.with_profile(profile.clone());
                }
                Some(config.source_storage_profile_from_env())
            },
            Self::RestS3 { .. } => Some(self.rest_config().expect("rest config").source_storage_profile_from_env()),
        }
    }
}

pub(crate) fn tpch_glue_table_name(table_name: &str, scale_factor: &str) -> String {
    format!("tpch_{scale_factor}_{table_name}")
}

pub(crate) fn tpcds_glue_table_name(table_name: &str, scale_factor: &str) -> String {
    format!("tpcds_{scale_factor}_{table_name}")
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct IcebergNativeSmokeReport {
    pub rows: Vec<IcebergNativeSmokeReportRow>,
}

impl IcebergNativeSmokeReport {
    pub(crate) fn to_tsv(&self) -> String {
        let rows = self.rows.iter().map(IcebergNativeSmokeReportRow::to_tsv_row).collect::<Vec<_>>().join("\n");
        format!("# native_iceberg_smoke_report_tsv_schema_version={REPORT_SCHEMA_VERSION}\n{REPORT_HEADER}\n{rows}\n",)
    }
}

pub(crate) fn write_smoke_tsv_header(writer: &mut dyn Write) -> SmokeResult<()> {
    writeln!(writer, "# native_iceberg_smoke_report_tsv_schema_version={REPORT_SCHEMA_VERSION}")?;
    writeln!(writer, "{REPORT_HEADER}")?;
    writer.flush()?;
    Ok(())
}

pub(crate) fn write_smoke_tsv_row(writer: &mut dyn Write, row: &IcebergNativeSmokeReportRow) -> SmokeResult<()> {
    writeln!(writer, "{}", row.to_tsv_row())?;
    writer.flush()?;
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct IcebergNativeSmokeReportRow {
    pub query_id: String,
    pub fixture_mode: String,
    pub catalog_type: String,
    pub table_ident: String,
    pub sql: String,
    pub diagnostics: TableFormatScanDiagnostics,
    pub native_planning_outcome: NativeDataFusionPlanningOutcome,
    pub native_reason_code: Option<String>,
    pub post_native_execution_path: NativeDataFusionPostNativeExecutionPath,
    pub staging_summary: ObjectStoreStagingSummary,
    pub cpu_arrow_reader_fallback: bool,
    pub native_runtime_started: bool,
    pub native_runtime_outcome: NativeDataFusionRuntimeOutcome,
    pub native_runtime_elapsed_ms: Option<u128>,
    pub native_output_rows: Option<usize>,
    pub native_output_batches: Option<usize>,
    pub source_fetch_bytes: usize,
    pub source_fetch_count: usize,
    pub host_fetch_budget_wait_time_ns: Option<usize>,
    pub source_fetch_concurrency_wait_time_ns: Option<usize>,
    pub source_fetch_time_ns: Option<usize>,
    pub source_decode_time_ns: Option<usize>,
    pub expected_count: Option<u64>,
    pub native_count_value: Option<u64>,
    pub cpu_runtime_started: bool,
    pub cpu_runtime_outcome: NativeDataFusionRuntimeOutcome,
    pub cpu_runtime_elapsed_ms: Option<u128>,
    pub cpu_output_rows: Option<usize>,
    pub cpu_output_batches: Option<usize>,
    pub cpu_output_fingerprint: Option<String>,
    pub native_output_fingerprint: Option<String>,
    pub comparison_mode: ParityComparisonMode,
    pub parity_status: ParityStatus,
    pub parity_detail: Option<String>,
    pub credential_source: String,
}

impl IcebergNativeSmokeReportRow {
    pub(crate) fn new(
        query_id: String,
        fixture_mode: String,
        catalog_type: String,
        table_ident: String,
        sql: String,
        diagnostics: TableFormatScanDiagnostics,
        native_planning_outcome: NativeDataFusionPlanningOutcome,
        native_reason_code: Option<String>,
        post_native_execution_path: NativeDataFusionPostNativeExecutionPath,
        expected_count: Option<u64>,
        staging_summary: ObjectStoreStagingSummary,
        credential_source: String,
    ) -> Self {
        Self {
            query_id,
            fixture_mode,
            catalog_type,
            table_ident,
            sql,
            diagnostics,
            native_planning_outcome,
            native_reason_code,
            post_native_execution_path,
            staging_summary,
            cpu_arrow_reader_fallback: false,
            native_runtime_started: false,
            native_runtime_outcome: NativeDataFusionRuntimeOutcome::NotStarted,
            native_runtime_elapsed_ms: None,
            native_output_rows: None,
            native_output_batches: None,
            source_fetch_bytes: 0,
            source_fetch_count: 0,
            host_fetch_budget_wait_time_ns: None,
            source_fetch_concurrency_wait_time_ns: None,
            source_fetch_time_ns: None,
            source_decode_time_ns: None,
            expected_count,
            native_count_value: None,
            cpu_runtime_started: false,
            cpu_runtime_outcome: NativeDataFusionRuntimeOutcome::NotStarted,
            cpu_runtime_elapsed_ms: None,
            cpu_output_rows: None,
            cpu_output_batches: None,
            cpu_output_fingerprint: None,
            native_output_fingerprint: None,
            comparison_mode: ParityComparisonMode::NotApplicable,
            parity_status: ParityStatus::NotCompared,
            parity_detail: None,
            credential_source,
        }
    }

    pub(crate) fn record_native_metrics(&mut self, metrics: NativeDataFusionExecutionMetrics) {
        self.source_fetch_bytes = metrics.source_fetch_bytes;
        self.source_fetch_count = metrics.source_fetch_count;
        self.host_fetch_budget_wait_time_ns = metrics.host_fetch_budget_wait_time_ns;
        self.source_fetch_concurrency_wait_time_ns = metrics.source_fetch_concurrency_wait_time_ns;
        self.source_fetch_time_ns = metrics.source_fetch_time_ns;
        self.source_decode_time_ns = metrics.source_decode_time_ns;
    }

    pub(crate) fn finalize_count_parity_if_needed(&mut self) {
        if self.cpu_runtime_started {
            return;
        }
        if self.native_planning_outcome != NativeDataFusionPlanningOutcome::AcceptedReplaced {
            self.parity_status = ParityStatus::NativeRejected;
            self.parity_detail.get_or_insert_with(|| "native planning rejected before execution".to_string());
            return;
        }
        if self.native_runtime_started && self.native_runtime_outcome == NativeDataFusionRuntimeOutcome::Failed {
            self.parity_status = ParityStatus::NativeFailed;
            return;
        }
        match (self.expected_count, self.native_count_value) {
            (Some(expected), Some(actual)) if expected == actual => {
                self.parity_status = ParityStatus::Matched;
                self.parity_detail = None;
            },
            (Some(expected), Some(actual)) => {
                self.parity_status = ParityStatus::Mismatched;
                self.parity_detail = Some(format!("expected_count={expected},native_count_value={actual}"));
            },
            _ => {
                self.parity_status = ParityStatus::NotCompared;
                self.parity_detail.get_or_insert_with(|| "count parity not available".to_string());
            },
        }
    }

    fn to_tsv_row(&self) -> String {
        [
            self.query_id.clone(),
            self.fixture_mode.clone(),
            self.catalog_type.clone(),
            self.table_ident.clone(),
            self.sql.clone(),
            self.diagnostics.table_format.clone(),
            self.diagnostics.snapshot_id.clone().unwrap_or_else(|| "none".to_string()),
            self.diagnostics.planned_data_file_count.to_string(),
            self.diagnostics.planned_delete_file_count.to_string(),
            optional_u64(self.diagnostics.planned_bytes),
            self.diagnostics.admitted_native_source_file_count.to_string(),
            list_or_none(&self.diagnostics.rejected_feature_list),
            empty_as_none(&self.diagnostics.native_source_read_capability),
            empty_as_none(&self.diagnostics.source_read_strategy),
            empty_as_none(&self.diagnostics.source_decode_input),
            empty_as_none(&self.diagnostics.credential_mode),
            empty_as_none(&self.diagnostics.lakehouse_cache_status),
            self.diagnostics.direct_device_read_disposition.code().to_string(),
            self.staging_summary.mode.code().to_string(),
            self.staging_summary.status.code().to_string(),
            self.staging_summary.staged_data_file_count.to_string(),
            self.staging_summary.staged_bytes.to_string(),
            self.native_planning_outcome.as_str().to_string(),
            self.native_reason_code.clone().unwrap_or_else(|| "none".to_string()),
            self.post_native_execution_path.as_str().to_string(),
            self.cpu_arrow_reader_fallback.to_string(),
            self.native_runtime_started.to_string(),
            self.native_runtime_outcome.as_str().to_string(),
            optional_u128(self.native_runtime_elapsed_ms),
            optional_usize(self.native_output_rows),
            optional_usize(self.native_output_batches),
            self.source_fetch_bytes.to_string(),
            self.source_fetch_count.to_string(),
            optional_duration_ms_text(self.host_fetch_budget_wait_time_ns),
            optional_duration_ms_text(self.source_fetch_concurrency_wait_time_ns),
            optional_duration_ms_text(self.source_fetch_time_ns),
            optional_duration_ms_text(self.source_decode_time_ns),
            optional_u64(self.expected_count),
            optional_u64(self.native_count_value),
            self.cpu_runtime_started.to_string(),
            self.cpu_runtime_outcome.as_str().to_string(),
            optional_u128(self.cpu_runtime_elapsed_ms),
            optional_usize(self.cpu_output_rows),
            optional_usize(self.cpu_output_batches),
            self.cpu_output_fingerprint.clone().unwrap_or_else(|| "none".to_string()),
            self.native_output_fingerprint.clone().unwrap_or_else(|| "none".to_string()),
            self.comparison_mode.as_str().to_string(),
            self.parity_status.as_str().to_string(),
            self.parity_detail.clone().unwrap_or_else(|| "none".to_string()),
            self.credential_source.clone(),
        ]
        .into_iter()
        .map(|field| stable_tsv_field(&field))
        .collect::<Vec<_>>()
        .join("\t")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ObjectStoreStagingSummary {
    pub mode: ObjectStoreStagingMode,
    pub status: ObjectStoreStagingStatus,
    pub staged_data_file_count: usize,
    pub staged_bytes: u64,
}

impl ObjectStoreStagingSummary {
    pub(crate) const fn new(mode: ObjectStoreStagingMode, status: ObjectStoreStagingStatus) -> Self {
        Self { mode, status, staged_data_file_count: 0, staged_bytes: 0 }
    }

    fn with_staged(mut self, staged_data_file_count: usize, staged_bytes: u64) -> Self {
        self.staged_data_file_count = staged_data_file_count;
        self.staged_bytes = staged_bytes;
        self
    }
}

impl Default for ObjectStoreStagingSummary {
    fn default() -> Self {
        Self::new(ObjectStoreStagingMode::Auto, ObjectStoreStagingStatus::NotNeeded)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ObjectStoreStagingStatus {
    NotNeeded,
    DirectNative,
    Disabled,
    StagedLocal,
}

impl ObjectStoreStagingStatus {
    const fn code(self) -> &'static str {
        match self {
            Self::NotNeeded => "not_needed",
            Self::DirectNative => "direct_s3_native",
            Self::Disabled => "disabled",
            Self::StagedLocal => "staged_local",
        }
    }
}

#[derive(Debug, Clone)]
struct StagingRecord {
    table_name: String,
    scan_plan: TableFormatScanPlan,
    summary: ObjectStoreStagingSummary,
}

#[derive(Debug, Clone)]
struct StagingTableFormatPlanner {
    inner: IcebergTableFormatPlanner,
    file_io: FileIO,
    mode: ObjectStoreStagingMode,
    tempdir: Arc<tempfile::TempDir>,
    records: Arc<Mutex<Vec<StagingRecord>>>,
    table_name: String,
}

impl StagingTableFormatPlanner {
    fn new(
        inner: IcebergTableFormatPlanner,
        file_io: FileIO,
        mode: ObjectStoreStagingMode,
        tempdir: Arc<tempfile::TempDir>,
        records: Arc<Mutex<Vec<StagingRecord>>>,
        table_name: String,
    ) -> Self {
        Self { inner, file_io, mode, tempdir, records, table_name }
    }
}

impl TableFormatPlanner for StagingTableFormatPlanner {
    fn plan_scan<'a>(
        &'a self,
        request: TableFormatScanRequest,
    ) -> BoxFuture<'a, datafusion_cudf::Result<TableFormatScanPlan>> {
        Box::pin(async move {
            eprintln!(
                "[native_iceberg_smoke_report] phase=table_format_scan_planning_start table_name={} staging_mode={}",
                self.table_name,
                self.mode.code(),
            );
            let scan_plan = self.inner.plan_table_format_scan(request).await?;
            let table_stage_dir = self.tempdir.path().join(sanitize_staged_path_component(&self.table_name));
            tokio::fs::create_dir_all(&table_stage_dir).await.map_err(|error| {
                Error::new(
                    ErrorCode::InvalidExecutionPlan,
                    "datafusion_cudf.table_format.iceberg.stage_object_store_data_files",
                    "failed to create Iceberg object-store staging directory",
                )
                .with_fact("table_name", &self.table_name)
                .with_fact("staging_mode", self.mode.code())
                .with_fact("source_error", error.to_string())
            })?;
            let (scan_plan, summary) =
                stage_object_store_data_files(scan_plan, &self.file_io, self.mode, &table_stage_dir).await.map_err(
                    |error| {
                        Error::new(
                            ErrorCode::InvalidExecutionPlan,
                            "datafusion_cudf.table_format.iceberg.stage_object_store_data_files",
                            "failed to stage Iceberg object-store data files for native smoke execution",
                        )
                        .with_fact("table_name", &self.table_name)
                        .with_fact("staging_mode", self.mode.code())
                        .with_fact("source_error", error.to_string())
                    },
                )?;
            self.records.lock().expect("staging records mutex poisoned").push(StagingRecord {
                table_name: self.table_name.clone(),
                scan_plan: scan_plan.clone(),
                summary: summary.clone(),
            });
            let diagnostics = scan_plan.diagnostics();
            eprintln!(
                "[native_iceberg_smoke_report] phase=table_format_scan_planning_finish table_name={} planned_data_file_count={} planned_bytes={} native_source_read_capability={} staging_status={}",
                self.table_name,
                diagnostics.planned_data_file_count,
                optional_u64(diagnostics.planned_bytes),
                empty_as_none(&diagnostics.native_source_read_capability),
                summary.status.code(),
            );
            Ok(scan_plan)
        })
    }
}

fn query_error_report_row(
    query_id: String,
    fixture_mode: String,
    catalog_type: String,
    table_ident: String,
    sql: String,
    staging_records: &Arc<Mutex<Vec<StagingRecord>>>,
    credential_source: String,
    error: &(dyn StdError + Send + Sync),
) -> IcebergNativeSmokeReportRow {
    let records = staging_records_snapshot(staging_records);
    let mut row = IcebergNativeSmokeReportRow::new(
        query_id,
        fixture_mode,
        catalog_type,
        table_ident,
        sql,
        aggregate_scan_diagnostics(&records),
        NativeDataFusionPlanningOutcome::RejectedBeforeExecution,
        Some("query_error".to_string()),
        NativeDataFusionPostNativeExecutionPath::Cpu,
        None,
        aggregate_staging_summary(&records),
        credential_source,
    );
    row.parity_status = ParityStatus::NativeRejected;
    row.parity_detail = Some(stable_tsv_field(&error.to_string()));
    row
}

async fn run_query_with_registered_native_tables(
    query_id: String,
    fixture_mode: String,
    catalog_type: String,
    table_ident: String,
    sql: String,
    ctx: &SessionContext,
    native_rule: &CudfNativePhysicalOptimizerRule,
    staging_records: Arc<Mutex<Vec<StagingRecord>>>,
    run_cpu_parity: bool,
    credential_source: String,
    cpu_baseline_timeout_secs: Option<u64>,
    cpu_baseline_session_profile: CpuBaselineSessionProfile,
    execution_mode: SmokeQueryExecutionMode,
) -> SmokeResult<IcebergNativeSmokeReportRow> {
    eprintln!(
        "[native_iceberg_smoke_report] phase=query_planning_start query_id={} run_cpu_parity={} credential_source={}",
        query_id, run_cpu_parity, credential_source,
    );
    let df = ctx.sql(&sql).await?;
    let task_ctx = df.task_ctx();
    let physical_plan = df.create_physical_plan().await?;
    let planning_row = datafusion_cudf::NativeDataFusionPlanningReport::from_retained_rule(
        query_id.clone(),
        Arc::clone(&physical_plan),
        native_rule,
    )
    .rows
    .into_iter()
    .next()
    .ok_or_else(|| smoke_error("native Iceberg smoke report produced no planning rows"))?;
    let records = staging_records_snapshot(&staging_records);
    let diagnostics = aggregate_scan_diagnostics(&records);
    let staging_summary = aggregate_staging_summary(&records);
    let mut row = IcebergNativeSmokeReportRow::new(
        query_id,
        fixture_mode,
        catalog_type,
        table_ident,
        sql.clone(),
        diagnostics,
        planning_row.outcome,
        planning_row.reason_code.as_deref().and_then(FallbackReason::from_code).map(|reason| reason.code().to_string()),
        planning_row.post_native_execution_path,
        None,
        staging_summary,
        credential_source,
    );

    let should_execute_native = planning_row.outcome == NativeDataFusionPlanningOutcome::AcceptedReplaced
        && planning_row.post_native_execution_path == NativeDataFusionPostNativeExecutionPath::Native;
    eprintln!(
        "[native_iceberg_smoke_report] phase=query_planning_finish query_id={} native_planning_outcome={} post_native_execution_path={} planned_data_file_count={} native_source_read_capability={}",
        row.query_id,
        row.native_planning_outcome.as_str(),
        row.post_native_execution_path.as_str(),
        row.diagnostics.planned_data_file_count,
        empty_as_none(&row.diagnostics.native_source_read_capability),
    );
    eprintln!(
        "[native_iceberg_smoke_report] phase=native_optimizer_diagnostics query_id={} shared_subplan_considered_count={} shared_subplan_candidate_count={} shared_subplan_applied_count={} local_source_reuse_considered_count={} local_source_reuse_applied_count={}",
        row.query_id,
        planning_row.native_optimizer.shared_subplan_materialization.considered_count,
        planning_row.native_optimizer.shared_subplan_materialization.candidate_count,
        planning_row.native_optimizer.shared_subplan_materialization.applied_count,
        planning_row.native_optimizer.local_parquet_source_reuse.considered_count,
        planning_row.native_optimizer.local_parquet_source_reuse.applied_count,
    );
    let mut native_batches = None;
    let native_schema = physical_plan.schema();
    if execution_mode.runs_native() && should_execute_native {
        row.native_runtime_started = true;
        eprintln!(
            "[native_iceberg_smoke_report] phase=native_runtime_start query_id={} native_source_read_capability={} direct_device_read_disposition={}",
            row.query_id,
            empty_as_none(&row.diagnostics.native_source_read_capability),
            row.diagnostics.direct_device_read_disposition.code(),
        );
        let started = Instant::now();
        match collect(Arc::clone(&physical_plan), task_ctx.into()).await {
            Ok(batches) => {
                row.native_runtime_elapsed_ms = Some(started.elapsed().as_millis());
                row.native_runtime_outcome = NativeDataFusionRuntimeOutcome::Succeeded;
                row.native_output_batches = Some(batches.len());
                row.native_output_rows = Some(batches.iter().map(RecordBatch::num_rows).sum());
                row.native_count_value = count_value_from_batches(&batches);
                let fingerprint = fingerprint_output_batches(native_schema.as_ref(), &batches, &sql);
                row.native_output_fingerprint = Some(fingerprint.native_fingerprint);
                row.comparison_mode = fingerprint.mode;
                native_batches = Some(batches);
                eprintln!(
                    "[native_iceberg_smoke_report] phase=native_runtime_finish query_id={} outcome={} elapsed_ms={} output_rows={} output_batches={}",
                    row.query_id,
                    row.native_runtime_outcome.as_str(),
                    row.native_runtime_elapsed_ms.unwrap_or(0),
                    row.native_output_rows.unwrap_or(0),
                    row.native_output_batches.unwrap_or(0),
                );
            },
            Err(error) => {
                row.native_runtime_elapsed_ms = Some(started.elapsed().as_millis());
                row.native_runtime_outcome = NativeDataFusionRuntimeOutcome::Failed;
                row.parity_detail = Some(stable_tsv_field(&error.to_string()));
                eprintln!(
                    "[native_iceberg_smoke_report] phase=native_runtime_finish query_id={} outcome={} elapsed_ms={} error={}",
                    row.query_id,
                    row.native_runtime_outcome.as_str(),
                    row.native_runtime_elapsed_ms.unwrap_or(0),
                    stable_tsv_field(&error.to_string()),
                );
            },
        }
        if let Some(metrics) = physical_plan.metrics() {
            let native_metrics = NativeDataFusionExecutionMetrics::from_metrics(&metrics);
            eprintln!(
                "[native_iceberg_smoke_report] phase=native_runtime_metrics query_id={} parquet_files={} row_groups_selected={} read_columns={} source_chunks_planned={} source_chunks_committed={} source_cache_hits={} source_cache_misses={} source_cache_inserts={} hash_join_build_count={} one_shot_join_count={} task_runtime_row_count={} task_operator_row_count={} aggregate_strategy_row_count={} join_strategy_row_count={} parquet_read_time_ms={} source_read_time_ms={} operator_time_ms={} chunk_operator_time_ms={} final_merge_time_ms={} final_sort_time_ms={} materialization_time_ms={}",
                row.query_id,
                native_metrics.parquet_files,
                native_metrics.row_groups_selected,
                native_metrics.read_columns,
                native_metrics.source_chunks_planned,
                native_metrics.source_chunks_committed,
                native_metrics.source_cache_hits,
                native_metrics.source_cache_misses,
                native_metrics.source_cache_inserts,
                native_metrics.hash_join_build_count,
                native_metrics.one_shot_join_count,
                native_metrics.task_runtime_rows.len(),
                native_metrics.task_operator_rows.len(),
                native_metrics.aggregate_strategy_rows.len(),
                native_metrics.join_strategy_rows.len(),
                metric_duration_ms_text(&metrics, "parquet_read_time"),
                metric_duration_ms_text(&metrics, "source_read_time"),
                metric_duration_ms_text(&metrics, "operator_time"),
                metric_duration_ms_text(&metrics, "chunk_operator_time"),
                metric_duration_ms_text(&metrics, "final_merge_time"),
                metric_duration_ms_text(&metrics, "final_sort_time"),
                metric_duration_ms_text(&metrics, "materialization_time"),
            );
            eprintln!(
                "[native_iceberg_smoke_report] phase=native_task_operator_rows query_id={} rows={}",
                row.query_id,
                format_task_operator_rows(&native_metrics),
            );
            eprintln!(
                "[native_iceberg_smoke_report] phase=native_task_runtime_rows query_id={} rows={}",
                row.query_id,
                format_task_runtime_rows(&native_metrics),
            );
            eprintln!(
                "[native_iceberg_smoke_report] phase=native_source_read_runtime_rows query_id={} rows={}",
                row.query_id,
                format_source_read_runtime_rows(&native_metrics),
            );
            eprintln!(
                "[native_iceberg_smoke_report] phase=native_aggregate_strategy_rows query_id={} rows={}",
                row.query_id,
                format_aggregate_strategy_rows(&native_metrics),
            );
            eprintln!(
                "[native_iceberg_smoke_report] phase=native_join_strategy_rows query_id={} rows={}",
                row.query_id,
                format_join_strategy_rows(&native_metrics),
            );
            row.record_native_metrics(native_metrics);
        }
    }

    if execution_mode.runs_cpu_baseline() && run_cpu_parity && should_execute_native {
        row.cpu_runtime_started = true;
        eprintln!(
            "[native_iceberg_smoke_report] phase=cpu_s3_baseline_start query_id={} table_count={} session_profile={}",
            row.query_id,
            records.iter().map(|record| record.table_name.as_str()).collect::<HashSet<_>>().len(),
            cpu_baseline_session_profile.code(),
        );
        let started = Instant::now();
        match collect_cpu_baseline_with_timeout(&sql, &records, cpu_baseline_timeout_secs, cpu_baseline_session_profile)
            .await
        {
            Ok((cpu_schema, cpu_batches)) => {
                row.cpu_runtime_elapsed_ms = Some(started.elapsed().as_millis());
                row.cpu_runtime_outcome = NativeDataFusionRuntimeOutcome::Succeeded;
                row.cpu_output_batches = Some(cpu_batches.len());
                row.cpu_output_rows = Some(cpu_batches.iter().map(RecordBatch::num_rows).sum());
                let cpu_fingerprint = fingerprint_output_batches(cpu_schema.as_ref(), &cpu_batches, &sql);
                row.cpu_output_fingerprint = Some(cpu_fingerprint.cpu_fingerprint);
                row.comparison_mode = cpu_fingerprint.mode;
                if let Some(native_batches) = native_batches.as_ref() {
                    eprintln!(
                        "[native_iceberg_smoke_report] phase=parity_collection_start query_id={} cpu_output_rows={} native_output_rows={}",
                        row.query_id,
                        row.cpu_output_rows.unwrap_or(0),
                        row.native_output_rows.unwrap_or(0),
                    );
                    let comparison = compare_record_batches(
                        cpu_schema.as_ref(),
                        &cpu_batches,
                        native_schema.as_ref(),
                        native_batches,
                        smoke_parity_uses_ordered_rows(&sql, cpu_schema.as_ref()),
                        ParityFloatTolerance::exact(),
                    );
                    row.parity_status = comparison.status;
                    row.parity_detail = comparison.detail;
                    row.cpu_output_fingerprint = Some(comparison.cpu_fingerprint);
                    row.native_output_fingerprint = Some(comparison.native_fingerprint);
                    row.comparison_mode = comparison.mode;
                    eprintln!(
                        "[native_iceberg_smoke_report] phase=parity_collection_finish query_id={} parity_status={} comparison_mode={}",
                        row.query_id,
                        row.parity_status.as_str(),
                        row.comparison_mode.as_str(),
                    );
                } else if row.native_runtime_outcome == NativeDataFusionRuntimeOutcome::Failed {
                    row.parity_status = ParityStatus::NativeFailed;
                }
                eprintln!(
                    "[native_iceberg_smoke_report] phase=cpu_s3_baseline_finish query_id={} outcome={} elapsed_ms={} output_rows={} output_batches={}",
                    row.query_id,
                    row.cpu_runtime_outcome.as_str(),
                    row.cpu_runtime_elapsed_ms.unwrap_or(0),
                    row.cpu_output_rows.unwrap_or(0),
                    row.cpu_output_batches.unwrap_or(0),
                );
            },
            Err(error) => {
                row.cpu_runtime_elapsed_ms = Some(started.elapsed().as_millis());
                row.cpu_runtime_outcome = NativeDataFusionRuntimeOutcome::Failed;
                let cpu_detail = stable_tsv_field(&error.to_string());
                if row.native_runtime_outcome == NativeDataFusionRuntimeOutcome::Failed {
                    row.parity_status = ParityStatus::NativeFailed;
                    if let Some(native_detail) = row.parity_detail.take() {
                        row.parity_detail =
                            Some(format!("native_failure={native_detail};cpu_baseline_failure={cpu_detail}"));
                    } else {
                        row.parity_detail = Some(cpu_detail);
                    }
                } else {
                    row.parity_status = ParityStatus::CpuFailed;
                    row.parity_detail = Some(cpu_detail);
                }
                eprintln!(
                    "[native_iceberg_smoke_report] phase=cpu_s3_baseline_finish query_id={} outcome={} elapsed_ms={} error={}",
                    row.query_id,
                    row.cpu_runtime_outcome.as_str(),
                    row.cpu_runtime_elapsed_ms.unwrap_or(0),
                    stable_tsv_field(&error.to_string()),
                );
            },
        }
    } else {
        row.finalize_count_parity_if_needed();
    }

    Ok(row)
}

fn native_session_context(
    source_storage_profile: Option<datafusion_cudf::SourceStorageProfile>,
    native_execution_profile: NativeExecutionProfile,
) -> SmokeResult<(SessionContext, Arc<CudfNativePhysicalOptimizerRule>)> {
    let mut execution_config =
        native_execution_config_for_profile_with_env(native_execution_profile, |name| std::env::var(name).ok())?;
    if let Some(profile) = source_storage_profile {
        execution_config = execution_config.with_source_storage_profile(profile);
    }
    let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(
        CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_native_execution_config(execution_config),
    ));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(smoke_session_config())
        .with_cudf_native_rule(Arc::clone(&native_rule))
        .build();
    Ok((SessionContext::new_with_state(state), native_rule))
}

pub(crate) fn native_execution_config_for_profile_with_env(
    profile: NativeExecutionProfile,
    mut get: impl FnMut(&str) -> Option<String>,
) -> SmokeResult<NativeCudfExecutionConfig> {
    let mut config = match profile {
        NativeExecutionProfile::Smoke => NativeCudfExecutionConfig::bounded_auto_sized(),
        NativeExecutionProfile::TpchBenchmark | NativeExecutionProfile::TpcdsBenchmark => {
            let device_budget_bytes = optional_env_u64_get(&mut get, NATIVE_DEVICE_BUDGET_BYTES_ENV)?
                .unwrap_or_else(NativeCudfExecutionConfig::auto_sized_device_budget_bytes);
            let max_source_chunk_bytes = optional_env_u64_get(&mut get, NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV)?
                .unwrap_or(TPCH_NATIVE_MAX_SOURCE_CHUNK_BYTES_DEFAULT);
            let max_row_groups_per_chunk = optional_env_usize_get(&mut get, NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV)?
                .unwrap_or(TPCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_DEFAULT);
            let max_retry_attempts = optional_env_usize_get(&mut get, NATIVE_MAX_RETRY_ATTEMPTS_ENV)?
                .unwrap_or(NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS);
            let min_retry_chunk_row_groups = optional_env_usize_get(&mut get, NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV)?
                .unwrap_or(NativeCudfExecutionConfig::DEFAULT_BOUNDED_MIN_RETRY_CHUNK_ROW_GROUPS);
            let max_source_reads_in_flight = optional_env_usize_get(&mut get, NATIVE_MAX_SOURCE_READS_IN_FLIGHT_ENV)?
                .unwrap_or(TPCH_NATIVE_MAX_SOURCE_READS_IN_FLIGHT_DEFAULT);
            let statistical_aggregate_execution_mode =
                native_statistical_aggregate_execution_mode_for_profile_with_env(profile, &mut get)?;
            NativeCudfExecutionConfig::bounded(
                device_budget_bytes,
                max_source_chunk_bytes,
                max_row_groups_per_chunk,
                max_retry_attempts,
                min_retry_chunk_row_groups,
            )
            .with_statistical_aggregate_execution_mode(statistical_aggregate_execution_mode)
            .with_source_scheduler_policy(SourceSchedulerPolicy::new(max_source_reads_in_flight))
        },
    };
    config = config.with_native_pipeline_execution_mode(NativePipelineExecutionMode::StrictBounded);
    Ok(config)
}

fn native_statistical_aggregate_execution_mode_for_profile_with_env(
    profile: NativeExecutionProfile,
    get: &mut impl FnMut(&str) -> Option<String>,
) -> SmokeResult<StatisticalAggregateExecutionMode> {
    let default = match profile {
        NativeExecutionProfile::TpcdsBenchmark => StatisticalAggregateExecutionMode::GpuNativeTolerant,
        NativeExecutionProfile::Smoke | NativeExecutionProfile::TpchBenchmark => {
            StatisticalAggregateExecutionMode::DataFusionExactHostWelford
        },
    };
    optional_env_statistical_aggregate_execution_mode_get(get)?.map_or(Ok(default), Ok)
}

fn optional_env_statistical_aggregate_execution_mode_get(
    get: &mut impl FnMut(&str) -> Option<String>,
) -> SmokeResult<Option<StatisticalAggregateExecutionMode>> {
    match get(NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV)
        .or_else(|| get(NATIVE_STDDEV_SAMP_EXECUTION_MODE_ENV))
        .filter(|value| !value.trim().is_empty())
    {
        Some(value) => value.trim().parse::<StatisticalAggregateExecutionMode>().map(Some).map_err(|error| {
            smoke_error(format!(
                "unsupported Iceberg native statistical aggregate execution mode={}; expected datafusion_exact_host_welford or gpu_native_tolerant: {error}",
                value.trim()
            ))
        }),
        None => Ok(None),
    }
}

async fn stage_object_store_data_files(
    scan_plan: TableFormatScanPlan,
    file_io: &FileIO,
    mode: ObjectStoreStagingMode,
    root: &Path,
) -> SmokeResult<(TableFormatScanPlan, ObjectStoreStagingSummary)> {
    if !scan_plan_has_stageable_object_store_data_files(&scan_plan) {
        return Ok((scan_plan, ObjectStoreStagingSummary::new(mode, ObjectStoreStagingStatus::NotNeeded)));
    }
    if scan_plan_has_direct_native_object_store_data_files(&scan_plan) && mode != ObjectStoreStagingMode::Required {
        return Ok((scan_plan, ObjectStoreStagingSummary::new(mode, ObjectStoreStagingStatus::DirectNative)));
    }
    if mode == ObjectStoreStagingMode::Disabled {
        return Ok((scan_plan, ObjectStoreStagingSummary::new(mode, ObjectStoreStagingStatus::Disabled)));
    }

    let mut staged_files = Vec::new();
    for (index, file) in scan_plan.data_files.iter().enumerate() {
        if file.format == TableFormatDataFileFormat::Parquet
            && matches!(file.read_capability, TableFormatReadCapability::ObjectStore { .. })
        {
            let staged_path = staged_file_path(root, index, &file.location);
            let bytes = file_io.new_input(&file.location)?.read().await?;
            tokio::fs::write(&staged_path, bytes.as_ref()).await?;
            staged_files.push(StagedObjectStoreDataFile {
                id: file.id.clone(),
                local_path: staged_path,
                staged_bytes: bytes.len() as u64,
            });
        }
    }
    let staged_data_file_count = staged_files.len();
    let staged_bytes = staged_files.iter().map(|file| file.staged_bytes).sum();
    Ok((
        rewrite_scan_plan_for_staged_object_store_files(scan_plan, &staged_files),
        ObjectStoreStagingSummary::new(mode, ObjectStoreStagingStatus::StagedLocal)
            .with_staged(staged_data_file_count, staged_bytes),
    ))
}

fn scan_plan_has_stageable_object_store_data_files(scan_plan: &TableFormatScanPlan) -> bool {
    scan_plan.data_files.iter().any(|file| {
        file.format == TableFormatDataFileFormat::Parquet
            && matches!(file.read_capability, TableFormatReadCapability::ObjectStore { .. })
    })
}

fn scan_plan_has_direct_native_object_store_data_files(scan_plan: &TableFormatScanPlan) -> bool {
    scan_plan.data_files.iter().any(|file| {
        file.format == TableFormatDataFileFormat::Parquet
            && matches!(file.read_capability, TableFormatReadCapability::ObjectStore { .. })
            && file.read_capability.is_native_parquet_capable()
            && !scan_plan.unsupported_features.iter().any(|feature| {
                feature.code == TableFormatUnsupportedFeatureCode::IcebergObjectStoreReadUnsupported
                    && feature.file_id.as_deref() == Some(file.id.as_str())
            })
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StagedObjectStoreDataFile {
    pub id: String,
    pub local_path: PathBuf,
    pub staged_bytes: u64,
}

pub(crate) fn rewrite_scan_plan_for_staged_object_store_files(
    mut scan_plan: TableFormatScanPlan,
    staged_files: &[StagedObjectStoreDataFile],
) -> TableFormatScanPlan {
    let staged_by_id = staged_files.iter().map(|file| (file.id.as_str(), file)).collect::<HashMap<_, _>>();
    for file in &mut scan_plan.data_files {
        if let Some(staged) = staged_by_id.get(file.id.as_str()) {
            file.location = staged.local_path.to_string_lossy().into_owned();
            file.read_capability = TableFormatReadCapability::local_file_system();
            file.resolved_parquet_source = resolve_table_format_parquet_source_with_metadata(
                &file.location,
                &file.read_capability,
                &TableFormatParquetReadStrategyConfig::default(),
                &TableFormatParquetSourceMetadata::from_size_and_row_count(Some(staged.staged_bytes), file.row_count),
            )
            .ok();
            file.size_bytes = Some(staged.staged_bytes);
        }
    }

    let staged_ids = staged_by_id.keys().copied().collect::<HashSet<_>>();
    scan_plan.unsupported_features.retain(|feature| {
        feature.code != TableFormatUnsupportedFeatureCode::IcebergObjectStoreReadUnsupported
            || !feature.file_id.as_deref().is_some_and(|file_id| staged_ids.contains(file_id))
    });
    scan_plan
}

fn staged_file_path(root: &Path, index: usize, location: &str) -> PathBuf {
    let basename = location
        .rsplit('/')
        .find(|part| !part.is_empty())
        .map(sanitize_staged_path_component)
        .filter(|part| !part.is_empty())
        .unwrap_or_else(|| "data.parquet".to_string());
    root.join(format!("staged-{index:05}-{basename}"))
}

fn sanitize_staged_path_component(value: &str) -> String {
    value.chars().map(|ch| if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_') { ch } else { '_' }).collect()
}

fn staging_records_snapshot(records: &Arc<Mutex<Vec<StagingRecord>>>) -> Vec<StagingRecord> {
    records.lock().expect("staging records mutex poisoned").clone()
}

fn clear_staging_records(records: &Arc<Mutex<Vec<StagingRecord>>>) {
    records.lock().expect("staging records mutex poisoned").clear();
}

fn aggregate_scan_diagnostics(records: &[StagingRecord]) -> TableFormatScanDiagnostics {
    if records.is_empty() {
        return TableFormatScanPlan::new(datafusion_cudf::TableFormat::Iceberg).diagnostics();
    }

    let diagnostics = records.iter().map(|record| record.scan_plan.diagnostics()).collect::<Vec<_>>();
    let planned_bytes = diagnostics
        .iter()
        .map(|diagnostics| diagnostics.planned_bytes)
        .try_fold(0u64, |total, bytes| bytes.map(|bytes| total.saturating_add(bytes)));
    let mut rejected_features = diagnostics
        .iter()
        .flat_map(|diagnostics| diagnostics.rejected_feature_list.iter().cloned())
        .collect::<Vec<_>>();
    rejected_features.sort();
    rejected_features.dedup();
    let mut capabilities = diagnostics
        .iter()
        .flat_map(|diagnostics| diagnostics.native_source_read_capability.split(','))
        .filter(|capability| !capability.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    capabilities.sort();
    capabilities.dedup();
    let source_read_strategy = joined_values(
        diagnostics
            .iter()
            .flat_map(|diagnostics| diagnostics.source_read_strategy.split(','))
            .filter(|value| !value.is_empty() && *value != "unknown")
            .map(str::to_string),
    );
    let source_decode_input = joined_values(
        diagnostics
            .iter()
            .flat_map(|diagnostics| diagnostics.source_decode_input.split(','))
            .filter(|value| !value.is_empty() && *value != "unknown")
            .map(str::to_string),
    );
    let credential_mode = joined_values(
        diagnostics
            .iter()
            .flat_map(|diagnostics| diagnostics.credential_mode.split(','))
            .filter(|value| !value.is_empty() && *value != "unknown")
            .map(str::to_string),
    );
    let lakehouse_cache_status = joined_values(
        diagnostics
            .iter()
            .flat_map(|diagnostics| diagnostics.lakehouse_cache_status.split(','))
            .filter(|value| !value.is_empty() && *value != "unknown")
            .map(str::to_string),
    );
    let snapshot_id =
        joined_optional_values(diagnostics.iter().filter_map(|diagnostics| diagnostics.snapshot_id.clone()));
    let catalog_type =
        joined_optional_values(diagnostics.iter().filter_map(|diagnostics| diagnostics.catalog_type.clone()));
    let table_format = joined_values(diagnostics.iter().map(|diagnostics| diagnostics.table_format.clone()));
    let direct_device_read_disposition = if diagnostics.iter().all(|diagnostics| {
        diagnostics.direct_device_read_disposition == TableFormatDirectDeviceReadDisposition::Eligible
    }) {
        TableFormatDirectDeviceReadDisposition::Eligible
    } else {
        TableFormatDirectDeviceReadDisposition::Rejected
    };

    TableFormatScanDiagnostics {
        table_format,
        snapshot_id,
        catalog_type,
        planned_data_file_count: diagnostics.iter().map(|diagnostics| diagnostics.planned_data_file_count).sum(),
        planned_delete_file_count: diagnostics.iter().map(|diagnostics| diagnostics.planned_delete_file_count).sum(),
        planned_bytes,
        admitted_native_source_file_count: diagnostics
            .iter()
            .map(|diagnostics| diagnostics.admitted_native_source_file_count)
            .sum(),
        rejected_feature_list: rejected_features,
        native_source_read_capability: capabilities.join(","),
        source_read_strategy,
        source_decode_input,
        credential_mode,
        lakehouse_cache_status,
        direct_device_read_disposition,
    }
}

fn aggregate_staging_summary(records: &[StagingRecord]) -> ObjectStoreStagingSummary {
    let Some(first) = records.first() else {
        return ObjectStoreStagingSummary::default();
    };
    let status = if records.iter().any(|record| record.summary.status == ObjectStoreStagingStatus::StagedLocal) {
        ObjectStoreStagingStatus::StagedLocal
    } else if records.iter().any(|record| record.summary.status == ObjectStoreStagingStatus::DirectNative) {
        ObjectStoreStagingStatus::DirectNative
    } else if records.iter().any(|record| record.summary.status == ObjectStoreStagingStatus::Disabled) {
        ObjectStoreStagingStatus::Disabled
    } else {
        ObjectStoreStagingStatus::NotNeeded
    };
    ObjectStoreStagingSummary::new(first.summary.mode, status).with_staged(
        records.iter().map(|record| record.summary.staged_data_file_count).sum(),
        records.iter().map(|record| record.summary.staged_bytes).sum(),
    )
}

fn joined_optional_values(values: impl Iterator<Item = String>) -> Option<String> {
    let joined = joined_values(values);
    if joined.is_empty() { None } else { Some(joined) }
}

fn joined_values(values: impl Iterator<Item = String>) -> String {
    let mut values = values.collect::<Vec<_>>();
    values.sort();
    values.dedup();
    values.join(",")
}

async fn collect_cpu_baseline(
    sql: &str,
    records: &[StagingRecord],
    session_profile: CpuBaselineSessionProfile,
) -> SmokeResult<(arrow::datatypes::SchemaRef, Vec<RecordBatch>)> {
    let ctx = SessionContext::new_with_config(cpu_baseline_session_config(session_profile));
    let mut seen_tables = HashSet::new();
    for record in records {
        if seen_tables.insert(record.table_name.clone()) {
            register_cpu_table_from_scan_plan(&ctx, &record.table_name, &record.scan_plan).await?;
        }
    }
    let df = ctx.sql(sql).await?;
    let task_ctx = df.task_ctx();
    let plan = df.create_physical_plan().await?;
    let schema = plan.schema();
    let batches = collect(plan, task_ctx.into()).await?;
    Ok((schema, batches))
}

async fn collect_cpu_baseline_with_timeout(
    sql: &str,
    records: &[StagingRecord],
    timeout_secs: Option<u64>,
    session_profile: CpuBaselineSessionProfile,
) -> SmokeResult<(arrow::datatypes::SchemaRef, Vec<RecordBatch>)> {
    let future = collect_cpu_baseline(sql, records, session_profile);
    match timeout_secs {
        Some(timeout_secs) => {
            tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), future).await.map_err(|_| {
                smoke_error(format!("timeout_category=cpu_s3_baseline_timeout;timeout_secs={timeout_secs}"))
            })?
        },
        None => future.await,
    }
}

pub(crate) fn cpu_baseline_session_config(profile: CpuBaselineSessionProfile) -> SessionConfig {
    match profile {
        CpuBaselineSessionProfile::Smoke => smoke_session_config(),
        CpuBaselineSessionProfile::TpchTuned => tpch_tuned_cpu_session_config(),
    }
}

pub(crate) fn smoke_parity_uses_ordered_rows(sql: &str, schema: &arrow::datatypes::Schema) -> bool {
    sql_has_top_level_order_by(sql) && sql_top_level_order_by_covers_output_columns(sql, schema)
}

pub(crate) fn fingerprint_output_batches(
    schema: &arrow::datatypes::Schema,
    batches: &[RecordBatch],
    sql: &str,
) -> datafusion_cudf::ParityComparisonResult {
    compare_record_batches(
        schema,
        batches,
        schema,
        batches,
        smoke_parity_uses_ordered_rows(sql, schema),
        ParityFloatTolerance::exact(),
    )
}

async fn register_cpu_table_from_scan_plan(
    ctx: &SessionContext,
    table_name: &str,
    scan_plan: &TableFormatScanPlan,
) -> SmokeResult<()> {
    let urls = cpu_baseline_listing_urls_from_scan_plan(scan_plan)?;
    if urls.is_empty() {
        return Err(smoke_error(format!(
            "no CPU-readable Parquet data files available for CPU baseline table {table_name}"
        )));
    }

    let url_count = urls.len();
    eprintln!(
        "[native_iceberg_smoke_report] phase=cpu_s3_baseline_registration_start table_name={} parquet_file_count={}",
        table_name, url_count,
    );
    register_cpu_object_stores(ctx, &urls).await?;
    let listing_options = ListingOptions::new(Arc::new(ParquetFormat::default()))
        .with_file_extension(cpu_baseline_listing_file_extension(&urls));
    let inferred_schema = listing_options.infer_schema(&ctx.state(), &urls[0]).await?;
    let config = ListingTableConfig::new_with_multi_paths(urls)
        .with_listing_options(listing_options)
        .with_schema(inferred_schema);
    ctx.register_table(table_name, Arc::new(ListingTable::try_new(config)?))?;
    eprintln!(
        "[native_iceberg_smoke_report] phase=cpu_s3_baseline_registration_finish table_name={} parquet_file_count={}",
        table_name, url_count,
    );
    Ok(())
}

pub(crate) fn cpu_baseline_listing_urls_from_scan_plan(
    scan_plan: &TableFormatScanPlan,
) -> SmokeResult<Vec<ListingTableUrl>> {
    scan_plan
        .data_files
        .iter()
        .filter(|file| file.format == TableFormatDataFileFormat::Parquet)
        .filter_map(cpu_baseline_listing_url_from_data_file)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| Box::new(error) as Box<dyn StdError + Send + Sync>)
}

fn cpu_baseline_listing_url_from_data_file(
    file: &datafusion_cudf::TableFormatDataFile,
) -> Option<datafusion::error::Result<ListingTableUrl>> {
    if let Some(resolved) = &file.resolved_parquet_source
        && resolved.read_strategy == ParquetReadStrategy::LocalCacheKvikio
        && Path::new(&resolved.reader_location).exists()
    {
        return Some(ListingTableUrl::parse(resolved.reader_location.as_str()));
    }

    match &file.read_capability {
        capability if capability == &TableFormatReadCapability::local_file_system() => {
            Some(ListingTableUrl::parse(file.location.as_str()))
        },
        TableFormatReadCapability::ObjectStore { scheme } if scheme.eq_ignore_ascii_case("s3") => {
            Some(ListingTableUrl::parse(file.location.as_str()))
        },
        _ => None,
    }
}

pub(crate) fn cpu_baseline_listing_file_extension(urls: &[ListingTableUrl]) -> &'static str {
    if urls.iter().any(|url| url.file_extension() != Some("parquet")) { "" } else { ".parquet" }
}

async fn register_cpu_object_stores(ctx: &SessionContext, urls: &[ListingTableUrl]) -> SmokeResult<()> {
    let mut registered_s3_stores = HashSet::new();
    for url in urls.iter().filter(|url| url.scheme() == "s3") {
        let store_url = url.object_store();
        if registered_s3_stores.insert(store_url.as_str().to_string()) {
            let store = build_cpu_s3_object_store(url.as_str()).await?;
            ctx.register_object_store(store_url.as_ref(), Arc::new(store));
            ctx.runtime_env().object_store(store_url.clone()).map_err(|error| {
                smoke_error(format!("failed to register direct S3 CPU baseline object store: {error}"))
            })?;
            ctx.runtime_env().object_store(url).map_err(|error| {
                smoke_error(format!("failed to resolve direct S3 CPU baseline table URL after registration: {error}"))
            })?;
        }
    }
    Ok(())
}

async fn build_cpu_s3_object_store(s3_url: &str) -> SmokeResult<object_store::aws::AmazonS3> {
    let mut builder = AmazonS3Builder::from_env().with_url(s3_url.to_string());
    if let Ok(endpoint) = std::env::var(REST_S3_ENDPOINT_ENV)
        && !endpoint.trim().is_empty()
    {
        let endpoint = endpoint.trim().to_string();
        if endpoint.starts_with("http://") {
            builder = builder.with_allow_http(true);
        }
        builder = builder.with_endpoint(endpoint);
    }
    if let Ok(path_style) = std::env::var(REST_S3_PATH_STYLE_ENV)
        && matches!(path_style.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")
    {
        builder = builder.with_virtual_hosted_style_request(false);
    }
    if let Ok(region) = std::env::var(REST_S3_REGION_ENV)
        && !region.trim().is_empty()
    {
        builder = builder.with_region(region.trim());
    }
    if let (Ok(access_key_id), Ok(secret_access_key)) =
        (std::env::var(REST_S3_ACCESS_KEY_ID_ENV), std::env::var(REST_S3_SECRET_ACCESS_KEY_ENV))
        && !access_key_id.trim().is_empty()
        && !secret_access_key.trim().is_empty()
    {
        builder = builder.with_access_key_id(access_key_id.trim()).with_secret_access_key(secret_access_key.trim());
        if let Ok(session_token) = std::env::var(REST_S3_SESSION_TOKEN_ENV)
            && !session_token.trim().is_empty()
        {
            builder = builder.with_token(session_token.trim());
        }
        return Ok(builder.build()?);
    }

    let mut loader = aws_config::defaults(BehaviorVersion::latest());
    if let Ok(profile) = std::env::var(AWS_PROFILE_ENV)
        && !profile.trim().is_empty()
    {
        loader = loader.profile_name(profile);
    }
    let sdk_config = loader.load().await;
    if let Some(region) = sdk_config.region() {
        builder = builder.with_region(region.as_ref());
    }
    if let Some(credentials_provider) = sdk_config.credentials_provider() {
        let credentials = credentials_provider.provide_credentials().await.map_err(|error| {
            smoke_error(format!("failed to load AWS credentials for direct S3 CPU baseline: {error}"))
        })?;
        builder = builder
            .with_access_key_id(credentials.access_key_id())
            .with_secret_access_key(credentials.secret_access_key());
        if let Some(session_token) = credentials.session_token() {
            builder = builder.with_token(session_token);
        }
    }

    Ok(builder.build()?)
}

async fn load_iceberg_table(fixture: &IcebergNativeSmokeFixture) -> SmokeResult<Table> {
    match fixture {
        IcebergNativeSmokeFixture::LocalMetadata { metadata_location, table_ident } => {
            let ident = table_ident_from_parts(table_ident)?;
            let table = StaticTable::from_metadata_file(metadata_location, ident, FileIO::new_with_fs()).await?;
            Ok(table.into_table())
        },
        IcebergNativeSmokeFixture::GlueS3 {
            catalog_name,
            database,
            table,
            warehouse,
            catalog_id,
            region,
            profile,
            ..
        } => {
            let config = GlueIcebergTableConfig {
                catalog_name: catalog_name.clone(),
                database: database.clone(),
                table: table.clone(),
                warehouse: warehouse.clone(),
                catalog_id: catalog_id.clone(),
                region: region.clone(),
                profile: profile.clone(),
            };
            Ok(load_glue_table(config).await?.table)
        },
        IcebergNativeSmokeFixture::RestS3 { namespace, table, .. } => {
            let backend =
                RestIcebergCatalogBackend::from_config_env(fixture.rest_config().expect("rest config")).await?;
            let ident = TableIdent::new(NamespaceIdent::new(namespace.clone()), table.clone());
            Ok(backend.load_table(&ident).await?)
        },
    }
}

fn iceberg_table_arrow_schema(table: &Table) -> SmokeResult<arrow::datatypes::SchemaRef> {
    Ok(iceberg_schema_to_datafusion_arrow_schema(table.current_schema_ref().as_ref())?)
}

fn smoke_session_config() -> SessionConfig {
    let mut config = SessionConfig::new()
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_sorts(false)
        .with_target_partitions(1);
    config.options_mut().optimizer.enable_topk_dynamic_filter_pushdown = false;
    config.options_mut().optimizer.enable_aggregate_dynamic_filter_pushdown = false;
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = false;
    config
}

fn expected_count_from_scan_plan(plan: &TableFormatScanPlan) -> Option<u64> {
    plan.data_files
        .iter()
        .map(|file| file.row_count)
        .try_fold(0u64, |total, row_count| row_count.map(|row_count| total.saturating_add(row_count)))
}

fn count_value_from_batches(batches: &[RecordBatch]) -> Option<u64> {
    let first_batch = batches.first()?;
    if first_batch.num_rows() != 1 || first_batch.num_columns() != 1 {
        return None;
    }
    let column = first_batch.column(0);
    if column.is_null(0) {
        return None;
    }
    if let Some(array) = column.as_any().downcast_ref::<UInt64Array>() {
        return Some(array.value(0));
    }
    if let Some(array) = column.as_any().downcast_ref::<UInt32Array>() {
        return Some(array.value(0) as u64);
    }
    if let Some(array) = column.as_any().downcast_ref::<Int64Array>() {
        return u64::try_from(array.value(0)).ok();
    }
    if let Some(array) = column.as_any().downcast_ref::<Int32Array>() {
        return u64::try_from(array.value(0)).ok();
    }
    None
}

fn parse_table_ident(value: &str) -> SmokeResult<Vec<String>> {
    let parts = value.split('.').map(str::trim).filter(|part| !part.is_empty()).map(str::to_string).collect::<Vec<_>>();
    if parts.len() < 2 {
        return Err(smoke_error(format!("table identifier {value:?} must include namespace and table name")));
    }
    Ok(parts)
}

fn parse_query_ids(value: &str) -> SmokeResult<Vec<String>> {
    let query_ids = value.split(',').map(str::trim).filter(|query_id| !query_id.is_empty()).collect::<Vec<_>>();
    let known = TPCH_QUERIES.iter().map(|query| query.id).collect::<BTreeSet<_>>();
    for query_id in &query_ids {
        if !known.contains(query_id) {
            return Err(smoke_error(format!("unknown TPC-H query id {query_id:?} in {QUERY_IDS_ENV}")));
        }
    }
    Ok(query_ids.into_iter().map(str::to_string).collect())
}

fn parse_tpcds_query_ids(value: &str) -> SmokeResult<Vec<String>> {
    let query_ids = value.split(',').map(str::trim).filter(|query_id| !query_id.is_empty()).collect::<Vec<_>>();
    let known = TPCDS_QUERIES.iter().map(|query| query.id).collect::<BTreeSet<_>>();
    for query_id in &query_ids {
        if !known.contains(query_id) {
            return Err(smoke_error(format!("unknown TPC-DS query id {query_id:?} in {TPCDS_QUERY_IDS_ENV}")));
        }
    }
    Ok(query_ids.into_iter().map(str::to_string).collect())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CustomQueryProbe {
    id: String,
    sql: String,
}

fn load_custom_query_manifest(path: &Path) -> SmokeResult<Vec<CustomQueryProbe>> {
    let contents = fs::read_to_string(path)?;
    let value: serde_json::Value = serde_json::from_str(&contents)?;
    let entries = value
        .as_array()
        .ok_or_else(|| smoke_error(format!("custom query manifest {} must be a JSON array", path.display())))?;
    let mut probes = Vec::new();
    let mut ids = BTreeSet::new();
    for (index, entry) in entries.iter().enumerate() {
        let object = entry.as_object().ok_or_else(|| {
            smoke_error(format!("custom query manifest entry {index} in {} must be an object", path.display()))
        })?;
        let id = object
            .get("id")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|id| !id.is_empty())
            .ok_or_else(|| smoke_error(format!("custom query manifest entry {index} is missing non-empty id")))?;
        if !id.chars().all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-') {
            return Err(smoke_error(format!(
                "custom query id {id:?} may only contain ASCII letters, digits, '_' or '-'"
            )));
        }
        if !ids.insert(id.to_string()) {
            return Err(smoke_error(format!("duplicate custom query id {id:?} in {}", path.display())));
        }
        let sql = object
            .get("sql")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|sql| !sql.is_empty())
            .ok_or_else(|| smoke_error(format!("custom query manifest entry {id:?} is missing non-empty sql")))?;
        probes.push(CustomQueryProbe { id: id.to_string(), sql: sql.to_string() });
    }
    if probes.is_empty() {
        return Err(smoke_error(format!("custom query manifest {} did not define any probes", path.display())));
    }
    Ok(probes)
}

fn normalize_tpch_scale_factor(value: &str) -> SmokeResult<String> {
    normalize_workload_scale_factor(value, TPCH_SCALE_FACTOR_ENV)
}

fn normalize_tpcds_scale_factor(value: &str) -> SmokeResult<String> {
    normalize_workload_scale_factor(value, TPCDS_SCALE_FACTOR_ENV)
}

fn normalize_workload_scale_factor(value: &str, env_name: &str) -> SmokeResult<String> {
    let scale = value.trim().trim_start_matches("SF").trim_start_matches("sf");
    if scale.is_empty() || !scale.chars().all(|ch| ch.is_ascii_digit()) {
        return Err(smoke_error(format!("unsupported {env_name}={value}; expected sf1, sf10, or sf100")));
    }
    Ok(format!("sf{scale}"))
}

fn table_ident_from_parts(parts: &[String]) -> SmokeResult<TableIdent> {
    TableIdent::from_strs(parts.iter().map(String::as_str))
        .map_err(|error| Box::new(error) as Box<dyn StdError + Send + Sync>)
}

fn required_env(get: &mut impl FnMut(&str) -> Option<String>, name: &'static str) -> SmokeResult<String> {
    get(name).filter(|value| !value.is_empty()).ok_or_else(|| smoke_error(format!("{name} is required")))
}

fn optional_u64(value: Option<u64>) -> String {
    value.map(|value| value.to_string()).unwrap_or_else(|| "none".to_string())
}

fn parse_optional_bool(value: Option<String>, name: &'static str) -> SmokeResult<bool> {
    match value.map(|value| value.trim().to_ascii_lowercase()).as_deref() {
        None | Some("") | Some("false" | "0" | "off" | "no") => Ok(false),
        Some("true" | "1" | "on" | "yes") => Ok(true),
        Some(other) => {
            Err(smoke_error(format!("unsupported {name}={other}; expected true/false, 1/0, on/off, or yes/no")))
        },
    }
}

fn parse_optional_timeout_secs(value: Option<String>, name: &'static str) -> SmokeResult<Option<u64>> {
    match value {
        Some(value) if !value.trim().is_empty() => {
            let parsed = value.trim().parse::<u64>().map_err(|error| {
                smoke_error(format!("invalid {name}={}: expected unsigned integer seconds: {error}", value.trim()))
            })?;
            if parsed == 0 {
                return Err(smoke_error(format!("{name} must be greater than zero when set")));
            }
            Ok(Some(parsed))
        },
        _ => Ok(None),
    }
}

fn optional_env_u64_get(get: &mut impl FnMut(&str) -> Option<String>, name: &'static str) -> SmokeResult<Option<u64>> {
    parse_optional_positive_u64(get(name), name)
}

fn optional_env_usize_get(
    get: &mut impl FnMut(&str) -> Option<String>,
    name: &'static str,
) -> SmokeResult<Option<usize>> {
    match parse_optional_positive_u64(get(name), name)? {
        Some(value) => usize::try_from(value)
            .map(Some)
            .map_err(|_| smoke_error(format!("{name} is too large for this platform's usize"))),
        None => Ok(None),
    }
}

fn parse_optional_positive_u64(value: Option<String>, name: &'static str) -> SmokeResult<Option<u64>> {
    match value {
        Some(value) if !value.trim().is_empty() => {
            let parsed = value.trim().parse::<u64>().map_err(|error| {
                smoke_error(format!("invalid {name}={}: expected unsigned integer: {error}", value.trim()))
            })?;
            if parsed == 0 {
                return Err(smoke_error(format!("{name} must be greater than zero when set")));
            }
            Ok(Some(parsed))
        },
        _ => Ok(None),
    }
}

fn optional_usize(value: Option<usize>) -> String {
    value.map(|value| value.to_string()).unwrap_or_else(|| "none".to_string())
}

fn optional_u128(value: Option<u128>) -> String {
    value.map(|value| value.to_string()).unwrap_or_else(|| "none".to_string())
}

fn optional_duration_ms_text(value: Option<usize>) -> String {
    value.map_or_else(|| "none".to_string(), |value| format!("{:.3}", value as f64 / 1_000_000.0))
}

fn metric_duration_ms_text(metrics: &MetricsSet, name: &str) -> String {
    optional_duration_ms_text(metrics.sum_by_name(name).map(|value| value.as_usize()))
}

fn format_task_operator_rows(metrics: &NativeDataFusionExecutionMetrics) -> String {
    if metrics.task_operator_rows.is_empty() {
        return "none".to_string();
    }
    let mut rows = metrics.task_operator_rows.iter().collect::<Vec<_>>();
    rows.sort_by_key(|row| row.to_stable_kv_row());
    rows.iter().map(|row| sanitize_nested(&row.to_stable_kv_row())).collect::<Vec<_>>().join("|")
}

fn format_task_runtime_rows(metrics: &NativeDataFusionExecutionMetrics) -> String {
    if metrics.task_runtime_rows.is_empty() {
        return "none".to_string();
    }
    let mut rows = metrics.task_runtime_rows.iter().collect::<Vec<_>>();
    rows.sort_by_key(|row| row.to_stable_kv_row());
    rows.iter().map(|row| sanitize_nested(&row.to_stable_kv_row())).collect::<Vec<_>>().join("|")
}

fn format_source_read_runtime_rows(metrics: &NativeDataFusionExecutionMetrics) -> String {
    if metrics.source_read_runtime_rows.is_empty() {
        return "none".to_string();
    }
    let mut rows = metrics.source_read_runtime_rows.iter().collect::<Vec<_>>();
    rows.sort_by_key(|row| row.to_stable_kv_row());
    rows.iter().map(|row| sanitize_nested(&row.to_stable_kv_row())).collect::<Vec<_>>().join("|")
}

fn format_aggregate_strategy_rows(metrics: &NativeDataFusionExecutionMetrics) -> String {
    if metrics.aggregate_strategy_rows.is_empty() {
        return "none".to_string();
    }
    let mut rows = metrics.aggregate_strategy_rows.iter().collect::<Vec<_>>();
    rows.sort_by_key(|row| row.to_stable_kv_row());
    rows.iter().map(|row| sanitize_nested(&row.to_stable_kv_row())).collect::<Vec<_>>().join("|")
}

fn format_join_strategy_rows(metrics: &NativeDataFusionExecutionMetrics) -> String {
    if metrics.join_strategy_rows.is_empty() {
        return "none".to_string();
    }
    let mut rows = metrics.join_strategy_rows.iter().collect::<Vec<_>>();
    rows.sort_by_key(|row| row.to_stable_kv_row());
    rows.iter().map(|row| sanitize_nested(&row.to_stable_kv_row())).collect::<Vec<_>>().join("|")
}

fn sanitize_nested(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ").replace('|', "/")
}

fn list_or_none(values: &[String]) -> String {
    if values.is_empty() { "none".to_string() } else { values.join(",") }
}

fn empty_as_none(value: &str) -> String {
    if value.is_empty() { "none".to_string() } else { value.to_string() }
}

fn stable_tsv_field(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}

fn smoke_error(message: impl Into<String>) -> Box<dyn StdError + Send + Sync> {
    Box::new(SmokeError(message.into()))
}

#[derive(Debug)]
struct SmokeError(String);

impl fmt::Display for SmokeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl StdError for SmokeError {}

trait FallbackReasonExt {
    fn from_code(code: &str) -> Option<FallbackReason>;
}

impl FallbackReasonExt for FallbackReason {
    fn from_code(code: &str) -> Option<FallbackReason> {
        [
            FallbackReason::TableFormatUnsupported,
            FallbackReason::TableFormatPlanningFailed,
            FallbackReason::IcebergNonParquetDataFile,
            FallbackReason::IcebergDeleteFilesUnsupported,
            FallbackReason::IcebergFieldIdMappingUnsupported,
            FallbackReason::IcebergNameMappingUnsupported,
            FallbackReason::IcebergNestedProjectionUnsupported,
            FallbackReason::IcebergPartitionTransformUnsupported,
            FallbackReason::IcebergResidualPredicateUnsupported,
            FallbackReason::IcebergRowSelectionUnsupported,
            FallbackReason::IcebergObjectStoreReadUnsupported,
            FallbackReason::IcebergSnapshotUnsupported,
            FallbackReason::TableFormatGpuOnlyLeafNotRewritten,
        ]
        .into_iter()
        .find(|reason| reason.code() == code)
    }
}
