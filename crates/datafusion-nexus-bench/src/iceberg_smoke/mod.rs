use std::{
    collections::{BTreeSet, HashMap, HashSet},
    error::Error as StdError,
    fmt, fs,
    io::Write,
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
use datafusion::{
    datasource::{
        file_format::parquet::ParquetFormat,
        listing::{ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl},
    },
    execution::SessionStateBuilder,
    physical_plan::{
        ExecutionPlan, collect, displayable,
        metrics::{MetricValue, MetricsSet},
    },
    prelude::{SessionConfig, SessionContext},
};
use datafusion_nexus::{
    CudfNativePhysicalOptimizerRule, CudfOptimizerConfig, CudfSessionStateBuilderExt, Error, ErrorCode, FallbackReason,
    GpuFallbackPolicy, NativeCudfExecutionConfig, NativeDataFusionExecutionMetrics, NativeDataFusionPlanningOutcome,
    NativeDataFusionPostNativeExecutionPath, NativeDataFusionRuntimeOutcome, NativePipelineExecutionMode,
    ParityComparisonMode, ParityFloatTolerance, ParityStatus, StatisticalAggregateExecutionMode,
    TableFormatDataFileFormat, TableFormatDirectDeviceReadDisposition, TableFormatParquetReadStrategyConfig,
    TableFormatParquetSourceMetadata, TableFormatPlanner, TableFormatReadCapability, TableFormatScanDiagnostics,
    TableFormatScanPlan, TableFormatScanRequest, TableFormatUnsupportedFeatureCode, compare_record_batches,
    iceberg::{
        AWS_DEFAULT_REGION_ENV, AWS_PROFILE_ENV, AWS_REGION_ENV, CudfIcebergTableProvider, GlueIcebergCredentialSource,
        GlueIcebergTableConfig, IcebergTableFormatPlanner, NEXUS_ICEBERG_GLUE_CATALOG_ID_ENV,
        NEXUS_ICEBERG_GLUE_CATALOG_NAME_ENV, NEXUS_ICEBERG_GLUE_DATABASE_ENV, NEXUS_ICEBERG_GLUE_TABLE_ENV,
        NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV,
        catalog::{IcebergCatalogBackend, RestIcebergCatalogBackend, RestIcebergCatalogConfig},
        iceberg_schema_to_datafusion_arrow_schema, load_glue_table,
    },
    register_source_unique_keys_for_path, resolve_table_format_parquet_source_with_metadata,
    schema_with_source_unique_keys, sql_has_top_level_order_by, sql_top_level_order_by_covers_output_columns,
};
use futures::future::BoxFuture;
use iceberg::{
    NamespaceIdent, TableIdent,
    io::FileIO,
    table::{StaticTable, Table},
};
use nexus_query_engine::{ParquetReadStrategy, SourceSchedulerPolicy};
use object_store::aws::AmazonS3Builder;

use crate::{
    TPCDS_QUERIES, TPCDS_TABLES, TPCH_QUERIES, TPCH_TABLES, tpcds_native_report_session_config,
    tpch_report_session_config, tpch_tuned_cpu_session_config,
};

type SmokeResult<T> = Result<T, Box<dyn StdError + Send + Sync>>;

const REPORT_SCHEMA_VERSION: u32 = 10;
const REPORT_COLUMNS: &[&str] = &[
    "query_id",
    "fixture_mode",
    "catalog_type",
    "table_ident",
    "sql",
    "table_format",
    "snapshot_id",
    "planned_data_file_count",
    "planned_delete_file_count",
    "planned_bytes",
    "admitted_native_source_file_count",
    "rejected_feature_list",
    "native_source_read_capability",
    "source_read_strategy",
    "source_decode_input",
    "credential_mode",
    "lakehouse_cache_status",
    "lakehouse_cache_read_mode",
    "lakehouse_cache_fill_mode",
    "source_fallback_reason",
    "direct_device_read_disposition",
    "object_store_staging_mode",
    "object_store_staging_status",
    "staged_data_file_count",
    "staged_bytes",
    "native_planning_outcome",
    "native_reason_code",
    "post_native_execution_path",
    "cpu_arrow_reader_fallback",
    "native_runtime_started",
    "native_runtime_outcome",
    "native_runtime_elapsed_ms",
    "native_output_rows",
    "native_output_batches",
    "source_fetch_bytes",
    "source_fetch_count",
    "host_fetch_budget_wait_time_ms",
    "source_fetch_concurrency_wait_time_ms",
    "source_fetch_time_ms",
    "source_decode_time_ms",
    "native_source_read_runtime_rows",
    "lakehouse_cache_fill_event_snapshot_rows",
    "expected_count",
    "native_count_value",
    "cpu_runtime_started",
    "cpu_runtime_outcome",
    "cpu_runtime_elapsed_ms",
    "cpu_output_rows",
    "cpu_output_batches",
    "cpu_output_fingerprint",
    "native_output_fingerprint",
    "comparison_mode",
    "parity_status",
    "parity_detail",
    "credential_source",
];

const FIXTURE_MODE_ENV: &str = "NEXUS_ICEBERG_SMOKE_FIXTURE_MODE";
const LOCAL_METADATA_LOCATION_ENV: &str = "NEXUS_ICEBERG_METADATA_LOCATION";
const LOCAL_TABLE_IDENT_ENV: &str = "NEXUS_ICEBERG_TABLE_IDENT";
const TABLE_ALIAS_ENV: &str = "NEXUS_ICEBERG_TABLE_ALIAS";
const QUERY_ENV: &str = "NEXUS_ICEBERG_SMOKE_QUERY";
const QUERY_IDS_ENV: &str = "NEXUS_ICEBERG_SMOKE_QUERY_IDS";
const TPCDS_QUERY_IDS_ENV: &str = "NEXUS_ICEBERG_TPCDS_SMOKE_QUERY_IDS";
const CUSTOM_WORKLOAD_ENV: &str = "NEXUS_ICEBERG_CUSTOM_WORKLOAD";
const CUSTOM_QUERY_FILE_ENV: &str = "NEXUS_ICEBERG_CUSTOM_QUERY_FILE";
const STREAM_TSV_ENV: &str = "NEXUS_ICEBERG_SMOKE_STREAM_TSV";
const CONTINUE_ON_QUERY_ERROR_ENV: &str = "NEXUS_ICEBERG_SMOKE_CONTINUE_ON_QUERY_ERROR";
const SMOKE_EXECUTION_MODE_ENV: &str = "NEXUS_ICEBERG_SMOKE_EXECUTION_MODE";
const CPU_BASELINE_TIMEOUT_SECS_ENV: &str = "NEXUS_ICEBERG_SMOKE_CPU_BASELINE_TIMEOUT_SECS";
const DEBUG_NATIVE_PLAN_ENV: &str = "NEXUS_ICEBERG_SMOKE_DEBUG_NATIVE_PLAN";
const NATIVE_DEVICE_BUDGET_BYTES_ENV: &str = "NEXUS_ICEBERG_NATIVE_DEVICE_BUDGET_BYTES";
const NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV: &str = "NEXUS_ICEBERG_NATIVE_MAX_SOURCE_CHUNK_BYTES";
const NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV: &str = "NEXUS_ICEBERG_NATIVE_MAX_ROW_GROUPS_PER_CHUNK";
const NATIVE_MAX_RETRY_ATTEMPTS_ENV: &str = "NEXUS_ICEBERG_NATIVE_MAX_RETRY_ATTEMPTS";
const NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV: &str = "NEXUS_ICEBERG_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS";
const NATIVE_MAX_SOURCE_READS_IN_FLIGHT_ENV: &str = "NEXUS_ICEBERG_NATIVE_MAX_SOURCE_READS_IN_FLIGHT";
const NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV: &str =
    "NEXUS_ICEBERG_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE";
const NATIVE_STDDEV_SAMP_EXECUTION_MODE_ENV: &str = "NEXUS_ICEBERG_NATIVE_STDDEV_SAMP_EXECUTION_MODE";
const TPCH_SCALE_FACTOR_ENV: &str = "NEXUS_ICEBERG_TPCH_SCALE_FACTOR";
const TPCDS_SCALE_FACTOR_ENV: &str = "NEXUS_ICEBERG_TPCDS_SCALE_FACTOR";
const OBJECT_STORE_STAGING_ENV: &str = "NEXUS_ICEBERG_SMOKE_OBJECT_STORE_STAGING";
const GLUE_URI_ENV: &str = "NEXUS_ICEBERG_GLUE_URI";
const REST_URI_ENV: &str = "NEXUS_ICEBERG_REST_URI";
const REST_WAREHOUSE_ENV: &str = "NEXUS_ICEBERG_WAREHOUSE";
const REST_NAMESPACE_ENV: &str = "NEXUS_ICEBERG_NAMESPACE";
const REST_TABLE_ENV: &str = "NEXUS_ICEBERG_REST_TABLE";
const REST_S3_ENDPOINT_ENV: &str = "NEXUS_ICEBERG_S3_ENDPOINT";
const REST_S3_REGION_ENV: &str = "NEXUS_ICEBERG_S3_REGION";
const REST_S3_PATH_STYLE_ENV: &str = "NEXUS_ICEBERG_S3_PATH_STYLE";
const REST_S3_ACCESS_KEY_ID_ENV: &str = "NEXUS_ICEBERG_S3_ACCESS_KEY_ID";
const REST_S3_SECRET_ACCESS_KEY_ENV: &str = "NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY";
const REST_S3_SESSION_TOKEN_ENV: &str = "NEXUS_ICEBERG_S3_SESSION_TOKEN";
const DEFAULT_TPCH_QUERY_IDS: &[&str] = &["q01", "q03", "q06", "q10"];
const DEFAULT_SMOKE_GLUE_CATALOG_NAME: &str = "glue";
const DEFAULT_SMOKE_GLUE_DATABASE: &str = "tpch_sf1";
const DEFAULT_SMOKE_GLUE_TABLE: &str = "lineitem";
const DEFAULT_SMOKE_GLUE_WAREHOUSE: &str = "s3://lakehouse/warehouse";
const DEFAULT_SMOKE_GLUE_CATALOG_ID: &str = "";
const DEFAULT_SMOKE_GLUE_REGION: &str = "us-east-1";
const DEFAULT_REST_URI: &str = "http://localhost:8181";
const DEFAULT_REST_WAREHOUSE: &str = "s3://lakehouse/warehouse";
const DEFAULT_REST_NAMESPACE: &str = "tpch_sf1";
const DEFAULT_REST_TABLE: &str = "lineitem";
const TPCH_NATIVE_MAX_SOURCE_CHUNK_BYTES_DEFAULT: u64 = 32 * 1024 * 1024 * 1024;
const TPCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_DEFAULT: usize = 1024;
const TPCH_NATIVE_MAX_SOURCE_READS_IN_FLIGHT_DEFAULT: usize = 2;

mod config;
mod cpu_baseline;
mod fixture;
mod metrics;
mod report;
mod runner;
mod staging;
mod tsv;

pub use config::{
    CpuBaselineSessionProfile, CustomWorkload, IcebergNativeSmokeConfig, IcebergNativeSmokeFixture,
    NativeExecutionProfile, ObjectStoreStagingMode, SmokeQueryExecutionMode,
    native_execution_config_for_profile_with_env, native_optimizer_options_for_profile,
    native_session_config_for_profile, tpcds_glue_table_name, tpch_glue_table_name,
};
pub(crate) use config::{load_custom_query_manifest, native_session_context, table_ident_from_parts};
pub(crate) use cpu_baseline::{
    collect_cpu_baseline_with_timeout, count_value_from_batches, expected_count_from_scan_plan, smoke_session_config,
};
pub use cpu_baseline::{
    cpu_baseline_listing_file_extension, cpu_baseline_listing_urls_from_scan_plan, cpu_baseline_session_config,
    fingerprint_output_batches, smoke_parity_uses_ordered_rows,
};
pub(crate) use fixture::{iceberg_table_arrow_schema, load_iceberg_table};
pub(crate) use metrics::*;
pub use report::{IcebergNativeSmokeReport, IcebergNativeSmokeReportRow, write_smoke_tsv_header, write_smoke_tsv_row};
pub use runner::{run_from_env, run_report, run_report_streaming, run_single_tpch_query_report_with_execution_mode};
pub use staging::{
    ObjectStoreStagingStatus, ObjectStoreStagingSummary, StagedObjectStoreDataFile,
    rewrite_scan_plan_for_staged_object_store_files,
};
pub(crate) use staging::{
    StagingRecord, StagingTableFormatPlanner, aggregate_scan_diagnostics, aggregate_staging_summary,
    clear_staging_records, staging_records_snapshot,
};
pub use tsv::stable_tsv_field;
pub(crate) use tsv::{
    FallbackReasonExt, empty_as_none, list_or_none, optional_duration_ms_text, optional_env_u64_get,
    optional_env_usize_get, optional_u64, optional_u128, optional_usize, parse_optional_bool,
    parse_optional_timeout_secs, smoke_error,
};
