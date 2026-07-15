use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IcebergNativeSmokeConfig {
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
pub enum SmokeQueryExecutionMode {
    NativeAndCpu,
    NativeOnly,
    CpuOnly,
}

impl SmokeQueryExecutionMode {
    pub(crate) fn from_env_value(value: &str) -> SmokeResult<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "native_and_cpu" | "native+cpu" | "all" => Ok(Self::NativeAndCpu),
            "native_only" | "native" => Ok(Self::NativeOnly),
            "cpu_only" | "cpu" => Ok(Self::CpuOnly),
            other => Err(smoke_error(format!(
                "unsupported {SMOKE_EXECUTION_MODE_ENV}={other}; expected native_and_cpu, native_only, or cpu_only"
            ))),
        }
    }

    pub(crate) const fn runs_native(self) -> bool {
        matches!(self, Self::NativeAndCpu | Self::NativeOnly)
    }

    pub(crate) const fn runs_cpu_baseline(self) -> bool {
        matches!(self, Self::NativeAndCpu | Self::CpuOnly)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuBaselineSessionProfile {
    Smoke,
    TpchTuned,
}

impl CpuBaselineSessionProfile {
    pub(crate) const fn code(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::TpchTuned => "tpch_tuned",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeExecutionProfile {
    Smoke,
    TpchBenchmark,
    TpcdsBenchmark,
}

impl IcebergNativeSmokeConfig {
    pub fn from_env() -> SmokeResult<Self> {
        Self::from_env_get(|name| std::env::var(name).ok())
    }

    pub fn from_env_get(mut get: impl FnMut(&str) -> Option<String>) -> SmokeResult<Self> {
        let mode = get(FIXTURE_MODE_ENV).unwrap_or_else(|| {
            if get(LOCAL_METADATA_LOCATION_ENV).is_some() {
                "local_metadata".to_string()
            } else {
                "glue_s3".to_string()
            }
        });
        let mut explicit_glue_table = false;
        let mut explicit_catalog_namespace = false;
        let fixture = match mode.as_str() {
            "local_metadata" => {
                let metadata_location = required_env(&mut get, LOCAL_METADATA_LOCATION_ENV)?;
                let table_ident = parse_table_ident(
                    get(LOCAL_TABLE_IDENT_ENV).unwrap_or_else(|| "local.iceberg_t".to_string()).as_str(),
                )?;
                IcebergNativeSmokeFixture::LocalMetadata { metadata_location, table_ident }
            },
            "glue_s3" => {
                explicit_glue_table = get(NEXUS_ICEBERG_GLUE_TABLE_ENV).is_some();
                let database_env = get(NEXUS_ICEBERG_GLUE_DATABASE_ENV);
                explicit_catalog_namespace = database_env.is_some();
                let catalog_name = get(NEXUS_ICEBERG_GLUE_CATALOG_NAME_ENV)
                    .unwrap_or_else(|| DEFAULT_SMOKE_GLUE_CATALOG_NAME.to_string());
                let database = database_env.unwrap_or_else(|| DEFAULT_SMOKE_GLUE_DATABASE.to_string());
                let table = get(NEXUS_ICEBERG_GLUE_TABLE_ENV).unwrap_or_else(|| DEFAULT_SMOKE_GLUE_TABLE.to_string());
                let warehouse =
                    get(NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV).unwrap_or_else(|| DEFAULT_SMOKE_GLUE_WAREHOUSE.to_string());
                let region = get(AWS_REGION_ENV)
                    .or_else(|| get(AWS_DEFAULT_REGION_ENV))
                    .unwrap_or_else(|| DEFAULT_SMOKE_GLUE_REGION.to_string());
                let uri = get(GLUE_URI_ENV).filter(|value| !value.trim().is_empty());
                let s3_endpoint = get(REST_S3_ENDPOINT_ENV).filter(|value| !value.trim().is_empty());
                let s3_region = get(REST_S3_REGION_ENV).filter(|value| !value.trim().is_empty());
                let s3_path_style_access = get(REST_S3_PATH_STYLE_ENV)
                    .filter(|value| !value.trim().is_empty())
                    .map(|value| matches!(value.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"));
                let mut glue_config = GlueIcebergTableConfig::new(catalog_name, database, table, warehouse, region);
                if let Some(profile) = get(AWS_PROFILE_ENV).filter(|value| !value.trim().is_empty()) {
                    glue_config = glue_config.with_profile(profile);
                }
                glue_config.catalog_id = match get(NEXUS_ICEBERG_GLUE_CATALOG_ID_ENV) {
                    Some(value) if value.trim().is_empty() => None,
                    Some(value) => Some(value),
                    None if DEFAULT_SMOKE_GLUE_CATALOG_ID.trim().is_empty() => None,
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
                    uri,
                    s3_endpoint,
                    s3_region,
                    s3_path_style_access,
                    credential_source,
                }
            },
            "rest_s3" => {
                explicit_glue_table = get(REST_TABLE_ENV).is_some();
                let catalog_handle = "lake".to_string();
                let uri = get(REST_URI_ENV).unwrap_or_else(|| DEFAULT_REST_URI.to_string());
                let warehouse = get(REST_WAREHOUSE_ENV).unwrap_or_else(|| DEFAULT_REST_WAREHOUSE.to_string());
                let namespace_env = get(REST_NAMESPACE_ENV);
                explicit_catalog_namespace = namespace_env.is_some();
                let namespace = namespace_env.unwrap_or_else(|| DEFAULT_REST_NAMESPACE.to_string());
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
        let mut fixture = fixture;
        if !explicit_catalog_namespace {
            let default_namespace =
                if !tpch_query_ids.is_empty() || matches!(custom_workload, Some(CustomWorkload::Tpch)) {
                    Some(format!("tpch_{tpch_scale_factor}"))
                } else if !tpcds_query_ids.is_empty() || matches!(custom_workload, Some(CustomWorkload::Tpcds)) {
                    Some(format!("tpcds_{tpcds_scale_factor}"))
                } else {
                    None
                };
            if let Some(default_namespace) = default_namespace {
                match &mut fixture {
                    IcebergNativeSmokeFixture::GlueS3 { database, .. } => *database = default_namespace,
                    IcebergNativeSmokeFixture::RestS3 { namespace, .. } => *namespace = default_namespace,
                    IcebergNativeSmokeFixture::LocalMetadata { .. } => {},
                }
            }
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
pub enum CustomWorkload {
    Tpch,
    Tpcds,
}

impl CustomWorkload {
    pub(crate) fn from_env_value(value: &str) -> SmokeResult<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "tpch" | "tpc-h" => Ok(Self::Tpch),
            "tpcds" | "tpc-ds" => Ok(Self::Tpcds),
            other => Err(smoke_error(format!("unsupported {CUSTOM_WORKLOAD_ENV}={other}; expected tpch or tpcds"))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectStoreStagingMode {
    Auto,
    Disabled,
    Required,
}

impl ObjectStoreStagingMode {
    pub(crate) fn from_env_value(value: Option<&str>) -> SmokeResult<Self> {
        match value.map(str::trim).filter(|value| !value.is_empty()) {
            None | Some("auto") => Ok(Self::Auto),
            Some("disabled" | "false" | "0" | "off") => Ok(Self::Disabled),
            Some("required" | "true" | "1" | "on") => Ok(Self::Required),
            Some(other) => Err(smoke_error(format!(
                "unsupported {OBJECT_STORE_STAGING_ENV}={other}; expected auto, disabled, or required"
            ))),
        }
    }

    pub(crate) const fn code(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Disabled => "disabled",
            Self::Required => "required",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IcebergNativeSmokeFixture {
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
        uri: Option<String>,
        s3_endpoint: Option<String>,
        s3_region: Option<String>,
        s3_path_style_access: Option<bool>,
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
    pub fn mode_code(&self) -> &'static str {
        match self {
            Self::LocalMetadata { .. } => "local_metadata",
            Self::GlueS3 { .. } => "glue_s3",
            Self::RestS3 { .. } => "rest_s3",
        }
    }

    pub fn catalog_type(&self) -> &'static str {
        match self {
            Self::LocalMetadata { .. } => "local",
            Self::GlueS3 { .. } => "glue",
            Self::RestS3 { .. } => "rest",
        }
    }

    /// S3 warehouse URL for object-store-backed fixtures (used to register a
    /// DataFusion object store so footer-only pruning can read Parquet footers).
    pub(crate) fn warehouse(&self) -> Option<&str> {
        match self {
            Self::GlueS3 { warehouse, .. } | Self::RestS3 { warehouse, .. } => Some(warehouse.as_str()),
            Self::LocalMetadata { .. } => None,
        }
    }

    pub(crate) fn default_table_alias(&self) -> String {
        match self {
            Self::LocalMetadata { table_ident, .. } => {
                table_ident.last().cloned().unwrap_or_else(|| "iceberg_t".to_string())
            },
            Self::GlueS3 { table, .. } => table.clone(),
            Self::RestS3 { table, .. } => table.clone(),
        }
    }

    pub(crate) fn table_ident_display(&self) -> String {
        match self {
            Self::LocalMetadata { table_ident, .. } => table_ident.join("."),
            Self::GlueS3 { database, table, .. } => format!("{database}.{table}"),
            Self::RestS3 { namespace, table, .. } => format!("{namespace}.{table}"),
        }
    }

    pub fn credential_source(&self) -> &'static str {
        match self {
            Self::LocalMetadata { .. } => "not_applicable",
            Self::GlueS3 { credential_source, .. } => credential_source.code(),
            Self::RestS3 { credential_source, .. } => credential_source,
        }
    }

    pub(crate) fn with_tpch_table(&self, table_name: &str, scale_factor: &str) -> SmokeResult<Self> {
        match self {
            Self::GlueS3 {
                catalog_name,
                database,
                warehouse,
                catalog_id,
                region,
                profile,
                uri,
                s3_endpoint,
                s3_region,
                s3_path_style_access,
                credential_source,
                ..
            } => Ok(Self::GlueS3 {
                catalog_name: catalog_name.clone(),
                database: database.clone(),
                table: tpch_glue_table_name(table_name, scale_factor),
                warehouse: warehouse.clone(),
                catalog_id: catalog_id.clone(),
                region: region.clone(),
                profile: profile.clone(),
                uri: uri.clone(),
                s3_endpoint: s3_endpoint.clone(),
                s3_region: s3_region.clone(),
                s3_path_style_access: *s3_path_style_access,
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
                credential_source,
            }),
            Self::LocalMetadata { .. } => {
                Err(smoke_error("only glue_s3/rest_s3 fixtures can be rewritten to another TPC-H table"))
            },
        }
    }

    pub fn with_tpcds_table(&self, table_name: &str, scale_factor: &str) -> SmokeResult<Self> {
        match self {
            Self::GlueS3 {
                catalog_name,
                database,
                warehouse,
                catalog_id,
                region,
                profile,
                uri,
                s3_endpoint,
                s3_region,
                s3_path_style_access,
                credential_source,
                ..
            } => Ok(Self::GlueS3 {
                catalog_name: catalog_name.clone(),
                database: database.clone(),
                table: tpcds_glue_table_name(table_name, scale_factor),
                warehouse: warehouse.clone(),
                catalog_id: catalog_id.clone(),
                region: region.clone(),
                profile: profile.clone(),
                uri: uri.clone(),
                s3_endpoint: s3_endpoint.clone(),
                s3_region: s3_region.clone(),
                s3_path_style_access: *s3_path_style_access,
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
                credential_source,
            }),
            Self::LocalMetadata { .. } => {
                Err(smoke_error("only glue_s3/rest_s3 fixtures can be rewritten to another TPC-DS table"))
            },
        }
    }

    pub fn tpch_all_tables_ident_display(&self, scale_factor: &str) -> String {
        let _ = scale_factor;
        match self {
            Self::GlueS3 { database, .. } => format!("{database}_all_tables"),
            Self::RestS3 { namespace, .. } => format!("{namespace}_all_tables"),
            Self::LocalMetadata { table_ident, .. } => table_ident.join("."),
        }
    }

    pub fn tpcds_all_tables_ident_display(&self, scale_factor: &str) -> String {
        let _ = scale_factor;
        match self {
            Self::GlueS3 { database, .. } => format!("{database}_all_tables"),
            Self::RestS3 { namespace, .. } => format!("{namespace}_all_tables"),
            Self::LocalMetadata { table_ident, .. } => table_ident.join("."),
        }
    }

    pub(crate) fn rest_config(&self) -> Option<RestIcebergCatalogConfig> {
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

    pub(crate) fn glue_catalog_config(
        &self,
    ) -> Option<datafusion_nexus::table_format::iceberg::catalog::GlueIcebergCatalogConfig> {
        let Self::GlueS3 {
            catalog_name,
            warehouse,
            region,
            catalog_id,
            profile,
            uri,
            s3_endpoint,
            s3_region,
            s3_path_style_access,
            ..
        } = self
        else {
            return None;
        };
        let mut config = datafusion_nexus::table_format::iceberg::catalog::GlueIcebergCatalogConfig::new(
            catalog_name,
            warehouse,
            region,
        );
        if let Some(catalog_id) = catalog_id.as_ref() {
            config = config.with_catalog_id(catalog_id.clone());
        }
        if let Some(profile) = profile.as_ref() {
            config = config.with_profile(profile.clone());
        }
        if let Some(uri) = uri.as_ref() {
            config = config.with_uri(uri.clone());
        }
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

    pub(crate) fn parquet_read_strategy_config(&self) -> TableFormatParquetReadStrategyConfig {
        let config = match self {
            Self::LocalMetadata { .. } => TableFormatParquetReadStrategyConfig::from_env(),
            Self::GlueS3 { .. } => self.glue_catalog_config().expect("glue config").parquet_read_strategy_config(),
            Self::RestS3 { .. } => self.rest_config().expect("rest config").parquet_read_strategy_config(),
        };
        if let Some(profile) = self.source_storage_profile() {
            config.with_source_storage_profile_remote_direct_capability(&profile)
        } else {
            config
        }
    }

    pub(crate) fn source_storage_profile(&self) -> Option<nexus_query_engine::runtime::SourceStorageProfile> {
        match self {
            Self::LocalMetadata { .. } => None,
            Self::GlueS3 { .. } => {
                Some(self.glue_catalog_config().expect("glue config").source_storage_profile_from_env())
            },
            Self::RestS3 { .. } => Some(self.rest_config().expect("rest config").source_storage_profile_from_env()),
        }
    }
}

pub fn tpch_glue_table_name(table_name: &str, scale_factor: &str) -> String {
    let _ = scale_factor;
    table_name.to_string()
}

pub fn tpcds_glue_table_name(table_name: &str, scale_factor: &str) -> String {
    let _ = scale_factor;
    table_name.to_string()
}

/// Shared cuDF runtime for Iceberg native smoke/benchmark runs.
///
/// Avoids charging each query for CUDA/runtime setup in
/// `native_runtime_elapsed_ms`, matching the non-Iceberg benchmark path.
fn shared_native_runtime() -> SmokeResult<Arc<cudf::Runtime>> {
    static SHARED_RUNTIME: std::sync::OnceLock<Result<Arc<cudf::Runtime>, String>> = std::sync::OnceLock::new();
    SHARED_RUNTIME
        .get_or_init(|| {
            cudf::Runtime::new()
                .map(Arc::new)
                .map_err(|error| format!("failed to initialize shared native Iceberg benchmark runtime: {error}"))
        })
        .clone()
        .map_err(smoke_error)
}

pub(crate) async fn native_session_context(
    source_storage_profile: Option<nexus_query_engine::runtime::SourceStorageProfile>,
    native_execution_profile: NativeExecutionProfile,
) -> SmokeResult<(SessionContext, Arc<CudfNativePhysicalOptimizerRule>)> {
    let (mut execution_config, fallback_policy) = native_session_execution_config_from_env(native_execution_profile)?;
    execution_config = execution_config.with_shared_runtime(shared_native_runtime()?);
    if let Some(profile) = resolve_default_chain_source_storage_profile(source_storage_profile).await? {
        execution_config = execution_config.with_source_storage_profile(profile);
    }
    let native_optimizer_options = native_optimizer_options_for_profile(native_execution_profile, &execution_config);
    let native_rule = Arc::new(CudfNativePhysicalOptimizerRule::new(
        CudfOptimizerConfig::new(fallback_policy)
            .with_native_execution_config(execution_config)
            .with_native_optimizer_options(native_optimizer_options),
    ));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(native_session_config_for_profile(native_execution_profile))
        .with_cudf_native_rule(Arc::clone(&native_rule))
        .build();
    Ok((SessionContext::new_with_state(state), native_rule))
}

pub fn native_session_config_for_profile(profile: NativeExecutionProfile) -> SessionConfig {
    match profile {
        NativeExecutionProfile::Smoke => smoke_session_config(),
        NativeExecutionProfile::TpchBenchmark => tpch_report_session_config(),
        NativeExecutionProfile::TpcdsBenchmark => tpcds_native_report_session_config(),
    }
}

pub fn native_optimizer_options_for_profile(
    profile: NativeExecutionProfile,
    execution_config: &NativeCudfExecutionConfig,
) -> nexus_query_engine::optimizer::NativeOptimizerOptions {
    let mut options = nexus_query_engine::optimizer::NativeOptimizerOptions::default();
    if matches!(profile, NativeExecutionProfile::TpchBenchmark | NativeExecutionProfile::TpcdsBenchmark)
        && let nexus_query_engine::runtime::MemoryControlPolicy::Bounded { max_source_chunk_bytes, .. } =
            execution_config.memory_control_policy()
    {
        let max_source_reads_in_flight =
            u64::try_from(execution_config.source_scheduler_policy().max_source_reads_in_flight()).unwrap_or(u64::MAX);
        options.local_parquet_source_reuse.max_variant_selected_bytes =
            max_source_chunk_bytes.saturating_mul(max_source_reads_in_flight);
    }
    options
}

async fn resolve_default_chain_source_storage_profile(
    source_storage_profile: Option<nexus_query_engine::runtime::SourceStorageProfile>,
) -> SmokeResult<Option<nexus_query_engine::runtime::SourceStorageProfile>> {
    let Some(profile) = source_storage_profile else {
        return Ok(None);
    };
    let s3 = profile.s3_profile();
    if !matches!(s3.credential, nexus_query_engine::runtime::S3SourceStorageCredential::DefaultChain) {
        return Ok(Some(profile));
    }

    let mut loader = aws_config::defaults(BehaviorVersion::latest());
    if let Ok(profile_name) = std::env::var(AWS_PROFILE_ENV)
        && !profile_name.trim().is_empty()
    {
        loader = loader.profile_name(profile_name);
    }
    let sdk_config = loader.load().await;
    let Some(credentials_provider) = sdk_config.credentials_provider() else {
        return Ok(Some(profile));
    };
    let credentials = credentials_provider.provide_credentials().await.map_err(|error| {
        smoke_error(format!("failed to load AWS credentials for native remote-direct S3 reads: {error}"))
    })?;
    let mut resolved_s3 = nexus_query_engine::runtime::S3SourceStorageProfile::default_chain();
    if let Some(region) = s3
        .region
        .as_deref()
        .filter(|region| !region.trim().is_empty())
        .or_else(|| sdk_config.region().map(|region| region.as_ref()))
    {
        resolved_s3 = resolved_s3.with_region(region.to_string());
    }
    if let Some(endpoint) = s3.endpoint.as_ref() {
        resolved_s3 = resolved_s3.with_endpoint(endpoint.clone());
    }
    if let Some(virtual_hosted_style_request) = s3.virtual_hosted_style_request {
        resolved_s3 = resolved_s3.with_path_style_access(!virtual_hosted_style_request);
    }
    if let Some(allow_http) = s3.allow_http {
        resolved_s3 = resolved_s3.with_allow_http(allow_http);
    }
    resolved_s3 = resolved_s3.with_static_credentials(
        credentials.access_key_id().to_string(),
        credentials.secret_access_key().to_string(),
        credentials.session_token().map(str::to_string),
    );
    Ok(Some(nexus_query_engine::runtime::SourceStorageProfile::s3(profile.id().to_string(), resolved_s3)))
}

pub fn native_execution_config_for_profile_with_env(
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

pub(crate) fn native_statistical_aggregate_execution_mode_for_profile_with_env(
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

pub(crate) fn optional_env_statistical_aggregate_execution_mode_get(
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
pub(crate) fn parse_table_ident(value: &str) -> SmokeResult<Vec<String>> {
    let parts = value.split('.').map(str::trim).filter(|part| !part.is_empty()).map(str::to_string).collect::<Vec<_>>();
    if parts.len() < 2 {
        return Err(smoke_error(format!("table identifier {value:?} must include namespace and table name")));
    }
    Ok(parts)
}

pub(crate) fn parse_query_ids(value: &str) -> SmokeResult<Vec<String>> {
    let query_ids = value.split(',').map(str::trim).filter(|query_id| !query_id.is_empty()).collect::<Vec<_>>();
    let known = TPCH_QUERIES.iter().map(|query| query.id).collect::<BTreeSet<_>>();
    for query_id in &query_ids {
        if !known.contains(query_id) {
            return Err(smoke_error(format!("unknown TPC-H query id {query_id:?} in {QUERY_IDS_ENV}")));
        }
    }
    Ok(query_ids.into_iter().map(str::to_string).collect())
}

pub(crate) fn parse_tpcds_query_ids(value: &str) -> SmokeResult<Vec<String>> {
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
pub(crate) struct CustomQueryProbe {
    pub(crate) id: String,
    pub(crate) sql: String,
}

pub(crate) fn load_custom_query_manifest(path: &Path) -> SmokeResult<Vec<CustomQueryProbe>> {
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

pub(crate) fn normalize_tpch_scale_factor(value: &str) -> SmokeResult<String> {
    normalize_workload_scale_factor(value, TPCH_SCALE_FACTOR_ENV)
}

pub(crate) fn normalize_tpcds_scale_factor(value: &str) -> SmokeResult<String> {
    normalize_workload_scale_factor(value, TPCDS_SCALE_FACTOR_ENV)
}

pub(crate) fn normalize_workload_scale_factor(value: &str, env_name: &str) -> SmokeResult<String> {
    let scale = value.trim().trim_start_matches("SF").trim_start_matches("sf");
    if scale.is_empty() || !scale.chars().all(|ch| ch.is_ascii_digit()) {
        return Err(smoke_error(format!("unsupported {env_name}={value}; expected sf1, sf10, or sf100")));
    }
    Ok(format!("sf{scale}"))
}

pub(crate) fn table_ident_from_parts(parts: &[String]) -> SmokeResult<TableIdent> {
    TableIdent::from_strs(parts.iter().map(String::as_str))
        .map_err(|error| Box::new(error) as Box<dyn StdError + Send + Sync>)
}

pub(crate) fn required_env(get: &mut impl FnMut(&str) -> Option<String>, name: &'static str) -> SmokeResult<String> {
    get(name).filter(|value| !value.is_empty()).ok_or_else(|| smoke_error(format!("{name} is required")))
}
