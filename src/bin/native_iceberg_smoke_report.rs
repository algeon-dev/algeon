use std::{
    collections::{BTreeSet, HashMap, HashSet},
    error::Error as StdError,
    fmt,
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
    physical_plan::collect,
    prelude::{SessionConfig, SessionContext},
};
use datafusion_cudf::{
    CudfNativePhysicalOptimizerRule, CudfOptimizerConfig, CudfSessionStateBuilderExt, Error, ErrorCode, FallbackReason,
    GpuFallbackPolicy, NativeDataFusionPlanningOutcome, NativeDataFusionPostNativeExecutionPath,
    NativeDataFusionRuntimeOutcome, NativePipelineExecutionMode, ParityComparisonMode, ParityFloatTolerance,
    ParityStatus, TPCH_QUERIES, TPCH_TABLES, TableFormatDataFileFormat, TableFormatDirectDeviceReadDisposition,
    TableFormatPlanner, TableFormatReadCapability, TableFormatScanDiagnostics, TableFormatScanPlan,
    TableFormatScanRequest, TableFormatUnsupportedFeatureCode, compare_record_batches,
    iceberg::{
        AWS_DEFAULT_REGION_ENV, AWS_PROFILE_ENV, AWS_REGION_ENV, CUDF_ICEBERG_GLUE_CATALOG_ID_ENV,
        CUDF_ICEBERG_GLUE_CATALOG_NAME_ENV, CUDF_ICEBERG_GLUE_DATABASE_ENV, CUDF_ICEBERG_GLUE_TABLE_ENV,
        CUDF_ICEBERG_GLUE_WAREHOUSE_ENV, CudfIcebergTableProvider, GlueIcebergCredentialSource, GlueIcebergTableConfig,
        IcebergTableFormatPlanner, load_glue_table,
    },
    sql_has_top_level_order_by,
};
use futures::future::BoxFuture;
use iceberg::{
    TableIdent,
    io::FileIO,
    table::{StaticTable, Table},
};
use object_store::aws::AmazonS3Builder;

type SmokeResult<T> = Result<T, Box<dyn StdError + Send + Sync>>;

const REPORT_SCHEMA_VERSION: u32 = 4;
const REPORT_HEADER: &str = "query_id\tfixture_mode\tcatalog_type\ttable_ident\tsql\ttable_format\tsnapshot_id\tplanned_data_file_count\tplanned_delete_file_count\tplanned_bytes\tadmitted_native_source_file_count\trejected_feature_list\tnative_source_read_capability\tdirect_device_read_disposition\tobject_store_staging_mode\tobject_store_staging_status\tstaged_data_file_count\tstaged_bytes\tnative_planning_outcome\tnative_reason_code\tpost_native_execution_path\tcpu_arrow_reader_fallback\tnative_runtime_started\tnative_runtime_outcome\tnative_runtime_elapsed_ms\tnative_output_rows\tnative_output_batches\texpected_count\tnative_count_value\tcpu_runtime_started\tcpu_runtime_outcome\tcpu_runtime_elapsed_ms\tcpu_output_rows\tcpu_output_batches\tcpu_output_fingerprint\tnative_output_fingerprint\tcomparison_mode\tparity_status\tparity_detail\tcredential_source";

const FIXTURE_MODE_ENV: &str = "CUDF_ICEBERG_SMOKE_FIXTURE_MODE";
const LOCAL_METADATA_LOCATION_ENV: &str = "CUDF_ICEBERG_METADATA_LOCATION";
const LOCAL_TABLE_IDENT_ENV: &str = "CUDF_ICEBERG_TABLE_IDENT";
const TABLE_ALIAS_ENV: &str = "CUDF_ICEBERG_TABLE_ALIAS";
const QUERY_ENV: &str = "CUDF_ICEBERG_SMOKE_QUERY";
const QUERY_IDS_ENV: &str = "CUDF_ICEBERG_SMOKE_QUERY_IDS";
const TPCH_SCALE_FACTOR_ENV: &str = "CUDF_ICEBERG_TPCH_SCALE_FACTOR";
const OBJECT_STORE_STAGING_ENV: &str = "CUDF_ICEBERG_SMOKE_OBJECT_STORE_STAGING";
const DEFAULT_TPCH_QUERY_IDS: &[&str] = &["q01", "q03", "q06", "q10"];
const DEFAULT_SMOKE_GLUE_CATALOG_NAME: &str = "glue";
const DEFAULT_SMOKE_GLUE_DATABASE: &str = "demo_db";
const DEFAULT_SMOKE_GLUE_TABLE: &str = "tpch_sf1_lineitem";
const DEFAULT_SMOKE_GLUE_WAREHOUSE: &str = "s3://example-warehouse/demo_db/";
const DEFAULT_SMOKE_GLUE_CATALOG_ID: &str = "123456789012";
const DEFAULT_SMOKE_GLUE_REGION: &str = "us-east-1";

#[tokio::main]
async fn main() {
    match run_from_env().await {
        Ok(report) => print!("{}", report.to_tsv()),
        Err(error) => {
            eprintln!("[native_iceberg_smoke_report] error={}", stable_tsv_field(&error.to_string()));
            std::process::exit(1);
        },
    }
}

pub(crate) async fn run_from_env() -> SmokeResult<IcebergNativeSmokeReport> {
    run_report(IcebergNativeSmokeConfig::from_env()?).await
}

pub(crate) async fn run_report(config: IcebergNativeSmokeConfig) -> SmokeResult<IcebergNativeSmokeReport> {
    if !config.tpch_query_ids.is_empty() {
        return run_tpch_subset_report(config).await;
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
    let planner = IcebergTableFormatPlanner::new(table).with_catalog_type(catalog_type.clone());
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
    let (ctx, native_rule) = native_session_context();
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
    )
    .await?;
    row.expected_count = staging_records_snapshot(&staging_records)
        .first()
        .and_then(|record| expected_count_from_scan_plan(&record.scan_plan));
    row.finalize_count_parity_if_needed();
    Ok(IcebergNativeSmokeReport { rows: vec![row] })
}

async fn run_tpch_subset_report(config: IcebergNativeSmokeConfig) -> SmokeResult<IcebergNativeSmokeReport> {
    let IcebergNativeSmokeFixture::GlueS3 { .. } = &config.fixture else {
        return Err(smoke_error("TPC-H Iceberg smoke query ids require CUDF_ICEBERG_SMOKE_FIXTURE_MODE=glue_s3"));
    };

    let catalog_type = config.fixture.catalog_type().to_string();
    let staging_records = Arc::new(Mutex::new(Vec::new()));
    let staging_tempdir = Arc::new(tempfile::Builder::new().prefix("datafusion-cudf-iceberg-tpch-stage-").tempdir()?);
    let (ctx, native_rule) = native_session_context();

    for table_name in TPCH_TABLES {
        let table_fixture =
            config.fixture.with_glue_table(tpch_glue_table_name(table_name, &config.tpch_scale_factor))?;
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
        let planner = IcebergTableFormatPlanner::new(table).with_catalog_type(catalog_type.clone());
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
        let row = run_query_with_registered_native_tables(
            query.id.to_string(),
            config.fixture.mode_code().to_string(),
            catalog_type.clone(),
            config.fixture.tpch_all_tables_ident_display(&config.tpch_scale_factor),
            query.sql.to_string(),
            &ctx,
            native_rule.as_ref(),
            Arc::clone(&staging_records),
            true,
            config.fixture.credential_source().to_string(),
        )
        .await?;
        rows.push(row);
    }
    Ok(IcebergNativeSmokeReport { rows })
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
            other => {
                return Err(smoke_error(format!(
                    "unsupported {FIXTURE_MODE_ENV}={other}; expected local_metadata or glue_s3"
                )));
            },
        };
        let table_alias = get(TABLE_ALIAS_ENV).unwrap_or_else(|| fixture.default_table_alias());
        let query_env = get(QUERY_ENV);
        let sql = query_env.clone().unwrap_or_else(|| format!("select count(*) as row_count from {table_alias}"));
        let object_store_staging = ObjectStoreStagingMode::from_env_value(get(OBJECT_STORE_STAGING_ENV).as_deref())?;
        let tpch_scale_factor =
            normalize_tpch_scale_factor(get(TPCH_SCALE_FACTOR_ENV).unwrap_or_else(|| "sf1".to_string()).as_str())?;
        let tpch_query_ids = match get(QUERY_IDS_ENV) {
            Some(value) => parse_query_ids(&value)?,
            None if query_env.is_none()
                && !explicit_glue_table
                && matches!(fixture, IcebergNativeSmokeFixture::GlueS3 { .. }) =>
            {
                DEFAULT_TPCH_QUERY_IDS.iter().map(|query_id| (*query_id).to_string()).collect()
            },
            None => Vec::new(),
        };
        Ok(Self {
            query_id: "iceberg_smoke".to_string(),
            table_alias,
            sql,
            fixture,
            object_store_staging,
            tpch_query_ids,
            tpch_scale_factor,
        })
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
}

impl IcebergNativeSmokeFixture {
    pub(crate) fn mode_code(&self) -> &'static str {
        match self {
            Self::LocalMetadata { .. } => "local_metadata",
            Self::GlueS3 { .. } => "glue_s3",
        }
    }

    pub(crate) fn catalog_type(&self) -> &'static str {
        match self {
            Self::LocalMetadata { .. } => "local",
            Self::GlueS3 { .. } => "glue",
        }
    }

    fn default_table_alias(&self) -> String {
        match self {
            Self::LocalMetadata { table_ident, .. } => {
                table_ident.last().cloned().unwrap_or_else(|| "iceberg_t".to_string())
            },
            Self::GlueS3 { table, .. } => table.clone(),
        }
    }

    fn table_ident_display(&self) -> String {
        match self {
            Self::LocalMetadata { table_ident, .. } => table_ident.join("."),
            Self::GlueS3 { database, table, .. } => format!("{database}.{table}"),
        }
    }

    pub(crate) fn credential_source(&self) -> &'static str {
        match self {
            Self::LocalMetadata { .. } => "not_applicable",
            Self::GlueS3 { credential_source, .. } => credential_source.code(),
        }
    }

    fn with_glue_table(&self, table: String) -> SmokeResult<Self> {
        let Self::GlueS3 { catalog_name, database, warehouse, catalog_id, region, profile, credential_source, .. } =
            self
        else {
            return Err(smoke_error("only glue_s3 fixtures can be rewritten to another Glue table"));
        };
        Ok(Self::GlueS3 {
            catalog_name: catalog_name.clone(),
            database: database.clone(),
            table,
            warehouse: warehouse.clone(),
            catalog_id: catalog_id.clone(),
            region: region.clone(),
            profile: profile.clone(),
            credential_source: *credential_source,
        })
    }

    pub(crate) fn tpch_all_tables_ident_display(&self, scale_factor: &str) -> String {
        match self {
            Self::GlueS3 { database, .. } => format!("{database}_tpch_{scale_factor}_all_tables"),
            Self::LocalMetadata { table_ident, .. } => table_ident.join("."),
        }
    }
}

pub(crate) fn tpch_glue_table_name(table_name: &str, scale_factor: &str) -> String {
    format!("tpch_{scale_factor}_{table_name}")
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
    let mut native_batches = None;
    let native_schema = physical_plan.schema();
    if should_execute_native {
        row.native_runtime_started = true;
        eprintln!(
            "[native_iceberg_smoke_report] phase=native_direct_s3_runtime_start query_id={} native_source_read_capability={} direct_device_read_disposition={}",
            row.query_id,
            empty_as_none(&row.diagnostics.native_source_read_capability),
            row.diagnostics.direct_device_read_disposition.code(),
        );
        let started = Instant::now();
        match collect(physical_plan, task_ctx.into()).await {
            Ok(batches) => {
                row.native_runtime_elapsed_ms = Some(started.elapsed().as_millis());
                row.native_runtime_outcome = NativeDataFusionRuntimeOutcome::Succeeded;
                row.native_output_batches = Some(batches.len());
                row.native_output_rows = Some(batches.iter().map(RecordBatch::num_rows).sum());
                row.native_count_value = count_value_from_batches(&batches);
                native_batches = Some(batches);
                eprintln!(
                    "[native_iceberg_smoke_report] phase=native_direct_s3_runtime_finish query_id={} outcome={} elapsed_ms={} output_rows={} output_batches={}",
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
                    "[native_iceberg_smoke_report] phase=native_direct_s3_runtime_finish query_id={} outcome={} elapsed_ms={} error={}",
                    row.query_id,
                    row.native_runtime_outcome.as_str(),
                    row.native_runtime_elapsed_ms.unwrap_or(0),
                    stable_tsv_field(&error.to_string()),
                );
            },
        }
    }

    if run_cpu_parity && should_execute_native {
        row.cpu_runtime_started = true;
        eprintln!(
            "[native_iceberg_smoke_report] phase=cpu_s3_baseline_start query_id={} table_count={}",
            row.query_id,
            records.iter().map(|record| record.table_name.as_str()).collect::<HashSet<_>>().len(),
        );
        let started = Instant::now();
        match collect_cpu_baseline(&sql, &records).await {
            Ok((cpu_schema, cpu_batches)) => {
                row.cpu_runtime_elapsed_ms = Some(started.elapsed().as_millis());
                row.cpu_runtime_outcome = NativeDataFusionRuntimeOutcome::Succeeded;
                row.cpu_output_batches = Some(cpu_batches.len());
                row.cpu_output_rows = Some(cpu_batches.iter().map(RecordBatch::num_rows).sum());
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
                        sql_has_top_level_order_by(&sql),
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

fn native_session_context() -> (SessionContext, Arc<CudfNativePhysicalOptimizerRule>) {
    let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(
        CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred)
            .with_native_pipeline_execution_mode(NativePipelineExecutionMode::StrictBounded),
    ));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(smoke_session_config())
        .with_cudf_native_rule(Arc::clone(&native_rule))
        .build();
    (SessionContext::new_with_state(state), native_rule)
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
            && matches!(&file.read_capability, TableFormatReadCapability::ObjectStore { scheme } if scheme == "s3")
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
) -> SmokeResult<(arrow::datatypes::SchemaRef, Vec<RecordBatch>)> {
    let ctx = SessionContext::new_with_config(smoke_session_config());
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
    let listing_options = ListingOptions::new(Arc::new(ParquetFormat::default())).with_file_extension(".parquet");
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
        .filter_map(|file| match &file.read_capability {
            capability if capability == &TableFormatReadCapability::local_file_system() => {
                Some(ListingTableUrl::parse(file.location.as_str()))
            },
            TableFormatReadCapability::ObjectStore { scheme } if scheme.eq_ignore_ascii_case("s3") => {
                Some(ListingTableUrl::parse(file.location.as_str()))
            },
            _ => None,
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| Box::new(error) as Box<dyn StdError + Send + Sync>)
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
    }
}

fn iceberg_table_arrow_schema(table: &Table) -> SmokeResult<arrow::datatypes::SchemaRef> {
    let schema = iceberg::arrow::schema_to_arrow_schema(table.current_schema_ref().as_ref())?;
    Ok(Arc::new(schema))
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

fn normalize_tpch_scale_factor(value: &str) -> SmokeResult<String> {
    let scale = value.trim().trim_start_matches("SF").trim_start_matches("sf");
    if scale.is_empty() || !scale.chars().all(|ch| ch.is_ascii_digit()) {
        return Err(smoke_error(format!("unsupported {TPCH_SCALE_FACTOR_ENV}={value}; expected sf1, sf10, or sf100")));
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

fn optional_usize(value: Option<usize>) -> String {
    value.map(|value| value.to_string()).unwrap_or_else(|| "none".to_string())
}

fn optional_u128(value: Option<u128>) -> String {
    value.map(|value| value.to_string()).unwrap_or_else(|| "none".to_string())
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
