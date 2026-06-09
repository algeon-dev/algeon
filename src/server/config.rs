#[cfg(feature = "iceberg")]
use std::fmt;
use std::{collections::BTreeMap, fs, net::SocketAddr, path::PathBuf, time::Duration};

use serde::Deserialize;

use super::{
    auth::AuthConfig,
    error::{Result, ServerError},
};
#[cfg(feature = "iceberg")]
use crate::table_format::scan::{DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID, credential_free_endpoint_identity};
#[cfg(feature = "iceberg")]
use crate::{
    AWS_ENDPOINT_URL_ENV, AWS_ENDPOINT_URL_S3_ENV, NEXUS_ICEBERG_PARQUET_READ_STRATEGY_ENV,
    NEXUS_ICEBERG_REMOTE_DIRECT_ENV, NEXUS_ICEBERG_REMOTE_DIRECT_READ_MODE_ENV,
    NEXUS_ICEBERG_REMOTE_HOST_BUFFER_RANGED_ENV, NEXUS_LAKEHOUSE_CACHE_MAX_BYTES_ENV, NEXUS_LAKEHOUSE_CACHE_POLICY_ENV,
    NEXUS_LAKEHOUSE_CACHE_READ_ENV, NEXUS_LAKEHOUSE_CACHE_ROOT_ENV, TableFormatLakehouseCachePolicy,
    TableFormatLakehouseCacheReadPolicy, TableFormatParquetReadStrategyConfig,
    TableFormatParquetReadStrategyPreference,
};
use crate::{CudfOptimizerConfig, GpuFallbackPolicy, NativeCudfExecutionConfig};
#[cfg(feature = "cugraph")]
use crate::{CugraphAlgorithm, CugraphGraphConstructionPolicy, CugraphGraphOptions};

/// Socket address the Flight SQL server listens on. Defaults to `0.0.0.0:50051`.
pub const NEXUS_SERVER_BIND_ENV: &str = "NEXUS_SERVER_BIND";
/// Tracing log filter for the server process. Defaults to `info`.
pub const NEXUS_SERVER_LOG_ENV: &str = "NEXUS_SERVER_LOG";
/// Maximum number of Flight SQL statements admitted at once. Defaults to `1`;
/// set a larger value only when the GPU has enough memory for concurrent work.
pub const NEXUS_SERVER_MAX_IN_FLIGHT_ENV: &str = "NEXUS_SERVER_MAX_IN_FLIGHT";
/// Seconds an unconsumed Flight SQL statement ticket may hold admission before
/// it expires and releases its slot. Defaults to `300`.
pub const NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS_ENV: &str = "NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS";
/// DataFusion physical-plan profile for Flight SQL sessions. Use `native` for
/// GPU-native plan shapes or `cpu_fallback` to keep DataFusion's CPU-oriented
/// execution settings; the default is `native` when Iceberg is configured and
/// `cpu_fallback` otherwise.
pub const NEXUS_SERVER_SESSION_PROFILE_ENV: &str = "NEXUS_SERVER_SESSION_PROFILE";
/// Native bounded-execution device-memory budget in bytes. When one bounded
/// budget value is set, the source chunk byte and row-group limits must also be set.
pub const NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV: &str = "NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES";
/// Native memory-control mode. Use `bounded` for memory-controlled execution
/// or `unbounded` to disable native bounded-memory controls; defaults to
/// `bounded`.
pub const NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV: &str = "NEXUS_SERVER_NATIVE_MEMORY_POLICY";
/// Maximum source row groups planned into one native bounded source chunk.
/// Required with the other bounded budget values when set explicitly.
pub const NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV: &str = "NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK";
/// Maximum source bytes planned into one native bounded source chunk. Required
/// with the other bounded budget values when set explicitly.
pub const NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV: &str = "NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES";
/// Path to a TOML server config file. The file currently supplies the `[auth]`
/// section, including nested users that cannot be represented as scalar env vars.
pub const NEXUS_SERVER_CONFIG_FILE_ENV: &str = "NEXUS_SERVER_CONFIG_FILE";

/// Enables registration of cuGraph SQL functions. Must be `true` before any
/// other cuGraph server setting is accepted.
pub const NEXUS_SERVER_CUGRAPH_ENABLED_ENV: &str = "NEXUS_SERVER_CUGRAPH_ENABLED";
/// Optional comma-separated allowlist of cuGraph SQL function names, such as
/// `cugraph_pagerank`. When unset, all registered cuGraph functions are enabled.
pub const NEXUS_SERVER_CUGRAPH_ALGORITHMS_ENV: &str = "NEXUS_SERVER_CUGRAPH_ALGORITHMS";
/// Default cuGraph edge direction setting for SQL calls. Accepts boolean
/// values; per-call `options_json` can override it.
pub const NEXUS_SERVER_CUGRAPH_DIRECTED_ENV: &str = "NEXUS_SERVER_CUGRAPH_DIRECTED";
/// Default cuGraph vertex renumbering setting for SQL calls. Accepts boolean
/// values; per-call `options_json` can override it.
pub const NEXUS_SERVER_CUGRAPH_RENUMBER_ENV: &str = "NEXUS_SERVER_CUGRAPH_RENUMBER";
/// Default cuGraph graph construction backend. Use `python_cugraph` or
/// `raw_libcugraph`; per-call `options_json` can override it.
pub const NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY_ENV: &str = "NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY";

/// DataFusion catalog name exposed to SQL for Iceberg tables. Preferred over
/// the Glue-specific catalog-name alias.
pub const NEXUS_ICEBERG_CATALOG_NAME_ENV: &str = "NEXUS_ICEBERG_CATALOG_NAME";
/// Iceberg namespace exposed to SQL. Preferred over the Glue database alias
/// and required when Iceberg server registration is configured.
pub const NEXUS_ICEBERG_NAMESPACE_ENV: &str = "NEXUS_ICEBERG_NAMESPACE";
/// Optional comma-separated Iceberg table registrations. Entries may be
/// `table` or `alias=table`; when unset, tables resolve lazily through the
/// dynamic catalog.
pub const NEXUS_ICEBERG_TABLES_ENV: &str = "NEXUS_ICEBERG_TABLES";

/// Selects which Iceberg catalog backend the server connects to. Accepts
/// `glue` (default when unset) or `rest`. Any other value is rejected at
/// config load time.
pub const NEXUS_ICEBERG_CATALOG_KIND_ENV: &str = "NEXUS_ICEBERG_CATALOG_KIND";

/// Glue catalog handle passed to iceberg-rust. Also acts as the legacy
/// DataFusion catalog-name fallback when `NEXUS_ICEBERG_CATALOG_NAME` is unset.
pub const NEXUS_ICEBERG_GLUE_CATALOG_NAME_ENV: &str = "NEXUS_ICEBERG_GLUE_CATALOG_NAME";
/// Legacy Glue database alias for `NEXUS_ICEBERG_NAMESPACE`.
pub const NEXUS_ICEBERG_GLUE_DATABASE_ENV: &str = "NEXUS_ICEBERG_GLUE_DATABASE";
/// Iceberg warehouse location, typically an `s3://...` URI. Required for both
/// Glue and REST Iceberg backends.
pub const NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV: &str = "NEXUS_ICEBERG_WAREHOUSE";
/// Optional AWS Glue catalog ID used to scope Glue requests.
pub const NEXUS_ICEBERG_GLUE_CATALOG_ID_ENV: &str = "NEXUS_ICEBERG_GLUE_CATALOG_ID";
/// Legacy Glue table registration alias for `NEXUS_ICEBERG_TABLES`.
pub const NEXUS_ICEBERG_GLUE_TABLES_ENV: &str = "NEXUS_ICEBERG_GLUE_TABLES";
/// Optional Glue-compatible endpoint URI, mainly for local or test catalogs.
pub const NEXUS_ICEBERG_GLUE_URI_ENV: &str = "NEXUS_ICEBERG_GLUE_URI";

/// REST Iceberg catalog base URI. Required when `NEXUS_ICEBERG_CATALOG_KIND=rest`.
pub const NEXUS_ICEBERG_REST_URI_ENV: &str = "NEXUS_ICEBERG_REST_URI";
/// Optional REST Iceberg catalog prefix, commonly used for tenant or warehouse routing.
pub const NEXUS_ICEBERG_REST_PREFIX_ENV: &str = "NEXUS_ICEBERG_REST_PREFIX";
/// Optional REST Iceberg bearer token. Treated as a secret and redacted from
/// diagnostics.
pub const NEXUS_ICEBERG_REST_TOKEN_ENV: &str = "NEXUS_ICEBERG_REST_TOKEN";
/// Internal REST catalog handle passed to iceberg-rust. Independent of the
/// DataFusion catalog name and defaults to `rest`.
pub const NEXUS_ICEBERG_REST_CATALOG_HANDLE_ENV: &str = "NEXUS_ICEBERG_REST_CATALOG_HANDLE";
/// S3-compatible object-store endpoint for Iceberg data, such as MinIO or
/// RustFS. Applies to both Glue and REST Iceberg backends.
pub const NEXUS_ICEBERG_S3_ENDPOINT_ENV: &str = "NEXUS_ICEBERG_S3_ENDPOINT";
/// S3 region for Iceberg object storage. Applies to both Glue and REST Iceberg backends.
pub const NEXUS_ICEBERG_S3_REGION_ENV: &str = "NEXUS_ICEBERG_S3_REGION";
/// Enables path-style S3 addressing for Iceberg object storage. Use `true` for
/// many local S3-compatible stores.
pub const NEXUS_ICEBERG_S3_PATH_STYLE_ENV: &str = "NEXUS_ICEBERG_S3_PATH_STYLE";

const DEFAULT_BIND_ADDR: &str = "0.0.0.0:50051";
const DEFAULT_LOG_FILTER: &str = "info";
const DEFAULT_MAX_IN_FLIGHT: usize = 1;
const DEFAULT_STATEMENT_TICKET_TTL_SECS: u64 = 300;
#[cfg(feature = "iceberg")]
const DEFAULT_GLUE_CATALOG_NAME: &str = "glue";
#[cfg(feature = "iceberg")]
const DEFAULT_REST_CATALOG_HANDLE: &str = "rest";

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub bind_addr: SocketAddr,
    pub log_filter: String,
    pub max_in_flight: usize,
    /// Wall-clock time a planned Flight SQL statement ticket may hold an
    /// admission permit while waiting for `DoGet`. Expired tickets are dropped
    /// and their permits are released; the default is 300 seconds.
    pub statement_ticket_ttl: Duration,
    pub native_optimizer_config: CudfOptimizerConfig,
    /// DataFusion physical-plan profile applied to the Flight SQL session.
    /// See [`NEXUS_SERVER_SESSION_PROFILE_ENV`] and
    /// [`ServerSessionProfile`].
    pub session_profile: ServerSessionProfile,
    /// Authentication mode applied to incoming Flight SQL RPCs. Loaded
    /// from the TOML overlay pointed to by
    /// [`NEXUS_SERVER_CONFIG_FILE_ENV`]; defaults to
    /// [`super::auth::AuthMode::Disabled`] when no file is provided.
    pub auth: AuthConfig,
    #[cfg(feature = "iceberg")]
    pub iceberg: Option<IcebergServerConfig>,
    #[cfg(feature = "cugraph")]
    pub cugraph: Option<CugraphServerConfig>,
}

/// Raw shape of the optional TOML overlay loaded from
/// [`NEXUS_SERVER_CONFIG_FILE_ENV`]. Currently only carries
/// the `[auth]` section; expand here when other sections need
/// TOML-driven config.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields, default)]
struct ServerFileOverlay {
    auth: AuthConfig,
}

/// DataFusion physical-plan profile applied to the Flight SQL session.
///
/// Decisions tracked here only affect DataFusion's logical/physical plan shape
/// for queries served through Flight SQL. They are orthogonal to the native
/// nexus-query-engine memory budget, which is controlled by the
/// `NEXUS_SERVER_NATIVE_*` environment variables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerSessionProfile {
    /// Native-friendly profile that matches the workload harness defaults:
    /// `target_partitions=1`, repartitioning disabled for file scans /
    /// aggregations / joins / sorts, and all dynamic-filter-pushdown variants
    /// disabled. Required for tables backed by GPU-only planning leaves such
    /// as `CudfTableFormatScanExec` because the CPU plan that DataFusion would
    /// otherwise produce is not executable end-to-end.
    Native,
    /// Preserve DataFusion's CPU-style profile derived from
    /// `DATAFUSION_EXECUTION_*` environment variables (single underscores,
    /// matching DataFusion's `SessionConfig::from_env()` mapping).
    ///
    /// Combine with Iceberg only when this binary is built with the
    /// `iceberg-cpu` feature. With that feature on, the dynamic catalog wraps
    /// loaded Iceberg tables in
    /// `iceberg_datafusion::IcebergStaticTableProvider` (real CPU Arrow
    /// reader) instead of the GPU-only `CudfIcebergTableProvider`, and
    /// `target_partitions` > 1 provides real CPU parallelism.
    ///
    /// **Without `iceberg-cpu` the Iceberg pairing is testing-only.** The
    /// catalog still hands out the GPU-only `CudfTableFormatScanExec` leaf
    /// which has no CPU `execute()`, so plain `ORDER BY` queries hit the
    /// Milestone 1 leaf guard and fail in planning. The server emits a startup
    /// WARN flagging this combination; the WARN goes away once `iceberg-cpu`
    /// is compiled in.
    ///
    /// For CPU-only sessions that do not register an Iceberg catalog
    /// (e.g., a server serving CSV / vanilla Parquet sources), the
    /// `iceberg-cpu` feature is irrelevant — this profile is the right choice
    /// regardless.
    CpuFallback,
}

impl ServerSessionProfile {
    pub fn parse(raw: &str) -> std::result::Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "native" => Ok(Self::Native),
            "cpu_fallback" | "cpu-fallback" => Ok(Self::CpuFallback),
            other => Err(format!("must be `native` or `cpu_fallback`, got `{other}`")),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::CpuFallback => "cpu_fallback",
        }
    }
}

impl std::fmt::Display for ServerSessionProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(feature = "iceberg")]
#[derive(Clone, PartialEq, Eq)]
pub struct IcebergServerConfig {
    /// DataFusion catalog name (e.g. "iceberg" or "glue").  Exposed as the
    /// first part of `catalog.namespace.table` three-part addressing.
    pub catalog_name: String,
    /// Iceberg namespace / Glue database name (e.g. "tpch_sf1"). This is the
    /// dynamic catalog visibility scope and the second part of three-part addressing.
    pub namespace: String,
    /// Pre-declared table alias registrations seeded before catalog
    /// registration. Empty means dynamic-catalog-only (all tables resolve
    /// lazily via three-part addressing).
    pub tables: Vec<IcebergTableRegistrationConfig>,
    /// Backend-specific connection payload. The enum makes invalid
    /// combinations unrepresentable (a Glue payload cannot carry a REST URI and
    /// vice versa). Selected by [`NEXUS_ICEBERG_CATALOG_KIND_ENV`].
    pub backend: IcebergCatalogBackendConfig,
    /// Server-validated table-format Parquet strategy passed into catalog
    /// planning paths. Server mode must not let those paths reread process env.
    pub parquet_read_strategy_config: TableFormatParquetReadStrategyConfig,
}

#[cfg(feature = "iceberg")]
impl fmt::Debug for IcebergServerConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IcebergServerConfig")
            .field("catalog_name", &self.catalog_name)
            .field("namespace", &self.namespace)
            .field("tables", &self.tables)
            .field("backend", &self.backend)
            .field("parquet_read_strategy_config", &"<redacted>")
            .finish()
    }
}

/// Backend-specific Iceberg catalog connection config.
///
/// Validation lives per-variant: the Glue branch requires a region; the REST
/// branch requires a URI and never requires a region.
#[cfg(feature = "iceberg")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IcebergCatalogBackendConfig {
    Glue(GlueServerBackendConfig),
    Rest(crate::iceberg::catalog::RestIcebergCatalogConfig),
}

/// Glue-backend connection config carried by [`IcebergCatalogBackendConfig::Glue`].
///
/// `catalog_handle` is the internal handle passed to iceberg-rust's
/// `GlueCatalogBuilder`; it is independent of the DataFusion `catalog_name` so
/// operators can choose a Trino-style connector name without changing the Glue
/// catalog identifier. `region` is required for Glue. `catalog_id` / `profile`
/// are optional AWS scoping knobs. `uri` and `s3_*` fields support local
/// Glue-compatible tests against Moto plus an S3-compatible object store.
#[cfg(feature = "iceberg")]
#[derive(Clone, PartialEq, Eq)]
pub struct GlueServerBackendConfig {
    pub catalog_handle: String,
    pub warehouse: String,
    pub region: String,
    pub catalog_id: Option<String>,
    pub profile: Option<String>,
    pub uri: Option<String>,
    pub s3_endpoint: Option<String>,
    pub s3_region: Option<String>,
    pub s3_path_style_access: Option<bool>,
}

#[cfg(feature = "iceberg")]
impl fmt::Debug for GlueServerBackendConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GlueServerBackendConfig")
            .field("catalog_handle", &self.catalog_handle)
            .field("warehouse", &self.warehouse)
            .field("region", &self.region)
            .field("catalog_id", &self.catalog_id)
            .field("profile", &self.profile)
            .field("uri", &self.uri.as_ref().map(|_| "<redacted>"))
            .field("s3_endpoint", &self.s3_endpoint.as_ref().map(|_| "<redacted>"))
            .field("s3_region", &self.s3_region)
            .field("s3_path_style_access", &self.s3_path_style_access)
            .finish()
    }
}

#[cfg(feature = "iceberg")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IcebergTableRegistrationConfig {
    pub table: String,
    pub alias: String,
}

#[cfg(feature = "cugraph")]
#[derive(Debug, Clone, PartialEq)]
pub struct CugraphServerConfig {
    /// Optional SQL-function allowlist. `None` means all registry algorithms
    /// are registered; `Some` registers only the named algorithms.
    pub allowed_algorithms: Option<std::collections::BTreeSet<CugraphAlgorithm>>,
    /// Operator-level graph construction defaults used by cuGraph SQL UDTFs.
    /// Per-call `options_json` still overrides these values.
    pub default_graph_options: CugraphGraphOptions,
}

impl ServerConfig {
    pub fn from_env() -> Result<Self> {
        Self::from_env_get(|name| std::env::var(name).ok())
    }

    pub fn from_env_get(mut get: impl FnMut(&str) -> Option<String>) -> Result<Self> {
        let mut values = EnvValues::new(&mut get);
        let bind_addr = values.socket_addr(NEXUS_SERVER_BIND_ENV, DEFAULT_BIND_ADDR)?;
        let log_filter = values.string(NEXUS_SERVER_LOG_ENV).unwrap_or_else(|| DEFAULT_LOG_FILTER.to_string());
        let max_in_flight = values.positive_usize(NEXUS_SERVER_MAX_IN_FLIGHT_ENV, DEFAULT_MAX_IN_FLIGHT)?;
        let statement_ticket_ttl = Duration::from_secs(
            values.positive_u64(NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS_ENV, DEFAULT_STATEMENT_TICKET_TTL_SECS)?,
        );
        let native_optimizer_config = native_optimizer_config(&mut values)?;
        #[cfg(feature = "iceberg")]
        let mut native_optimizer_config = native_optimizer_config;

        #[cfg(feature = "iceberg")]
        let mut iceberg = iceberg_config(&mut values)?;

        #[cfg(feature = "iceberg")]
        if let Some(config) = iceberg.as_mut() {
            let profile = iceberg_source_storage_profile(config, &mut values);
            config.parquet_read_strategy_config = config
                .parquet_read_strategy_config
                .clone()
                .with_source_storage_profile_remote_direct_capability(&profile);
            let execution_config =
                native_optimizer_config.native_execution_config().clone().with_source_storage_profile(profile);
            native_optimizer_config = native_optimizer_config.with_native_execution_config(execution_config);
        }

        #[cfg(not(feature = "iceberg"))]
        reject_iceberg_env_without_feature(&mut values)?;

        #[cfg(feature = "cugraph")]
        let cugraph = cugraph_config(&mut values)?;

        #[cfg(not(feature = "cugraph"))]
        reject_cugraph_env_without_feature(&mut values)?;

        let explicit_profile = values.optional_session_profile(NEXUS_SERVER_SESSION_PROFILE_ENV)?;
        #[cfg(feature = "iceberg")]
        let default_profile =
            if iceberg.is_some() { ServerSessionProfile::Native } else { ServerSessionProfile::CpuFallback };
        #[cfg(not(feature = "iceberg"))]
        let default_profile = ServerSessionProfile::CpuFallback;
        let session_profile = explicit_profile.unwrap_or(default_profile);

        let overlay_path = values.string(NEXUS_SERVER_CONFIG_FILE_ENV).map(PathBuf::from);
        let overlay = match overlay_path {
            Some(path) => load_overlay(&path)?,
            None => ServerFileOverlay::default(),
        };
        overlay.auth.mode.validate()?;

        Ok(Self {
            bind_addr,
            log_filter,
            max_in_flight,
            statement_ticket_ttl,
            native_optimizer_config,
            session_profile,
            auth: overlay.auth,
            #[cfg(feature = "iceberg")]
            iceberg,
            #[cfg(feature = "cugraph")]
            cugraph,
        })
    }
}

fn load_overlay(path: &std::path::Path) -> Result<ServerFileOverlay> {
    let raw = fs::read_to_string(path).map_err(|error| {
        ServerError::configuration_with_source(format!("failed to read server config file `{}`", path.display()), error)
    })?;
    toml::from_str::<ServerFileOverlay>(&raw).map_err(|error| {
        ServerError::configuration_with_source(
            format!("failed to parse server config file `{}`", path.display()),
            error,
        )
    })
}

fn native_optimizer_config(values: &mut EnvValues<'_>) -> Result<CudfOptimizerConfig> {
    let memory_policy = values
        .string(NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV)
        .map(|raw| parse_native_memory_policy(&raw))
        .transpose()?
        .unwrap_or(ServerNativeMemoryPolicy::Bounded);
    let device_budget = values.optional_u64(NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV)?;
    let max_source_chunk = values.optional_u64(NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV)?;
    let max_row_groups = values.optional_usize(NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV)?;

    let execution_config = match memory_policy {
        ServerNativeMemoryPolicy::Unbounded => {
            if device_budget.is_some() || max_source_chunk.is_some() || max_row_groups.is_some() {
                return Err(ServerError::invalid_config(
                    NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV,
                    "unbounded native execution cannot be combined with bounded native budget env vars",
                ));
            }
            NativeCudfExecutionConfig::unbounded()
        },
        ServerNativeMemoryPolicy::Bounded => match (device_budget, max_source_chunk, max_row_groups) {
            (None, None, None) => NativeCudfExecutionConfig::bounded_auto_sized(),
            (Some(device_budget), Some(max_source_chunk), Some(max_row_groups)) => {
                if device_budget == 0 {
                    return Err(ServerError::invalid_config(
                        NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV,
                        "must be greater than zero",
                    ));
                }
                if max_source_chunk == 0 {
                    return Err(ServerError::invalid_config(
                        NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV,
                        "must be greater than zero",
                    ));
                }
                if max_row_groups == 0 {
                    return Err(ServerError::invalid_config(
                        NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV,
                        "must be greater than zero",
                    ));
                }
                NativeCudfExecutionConfig::bounded(device_budget, max_source_chunk, max_row_groups, 1, 1)
            },
            _ => {
                return Err(ServerError::invalid_config(
                    NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV,
                    "bounded native execution requires device budget, max source chunk bytes, and max row groups per chunk",
                ));
            },
        },
    };

    Ok(CudfOptimizerConfig::new(GpuFallbackPolicy::default()).with_native_execution_config(execution_config))
}

#[cfg(feature = "iceberg")]
fn iceberg_source_storage_profile(
    config: &IcebergServerConfig,
    values: &mut EnvValues<'_>,
) -> nexus_query_engine::SourceStorageProfile {
    match &config.backend {
        IcebergCatalogBackendConfig::Glue(glue) => {
            let mut catalog_config = crate::iceberg::catalog::GlueIcebergCatalogConfig::new(
                glue.catalog_handle.clone(),
                glue.warehouse.clone(),
                glue.region.clone(),
            );
            if let Some(catalog_id) = glue.catalog_id.as_ref() {
                catalog_config = catalog_config.with_catalog_id(catalog_id.clone());
            }
            if let Some(profile) = glue.profile.as_ref() {
                catalog_config = catalog_config.with_profile(profile.clone());
            }
            if let Some(uri) = glue.uri.as_ref() {
                catalog_config = catalog_config.with_uri(uri.clone());
            }
            if let Some(endpoint) = glue.s3_endpoint.as_ref() {
                catalog_config = catalog_config.with_s3_endpoint(endpoint.clone());
            }
            if let Some(region) = glue.s3_region.as_ref() {
                catalog_config = catalog_config.with_s3_region(region.clone());
            }
            if let Some(path_style) = glue.s3_path_style_access {
                catalog_config = catalog_config.with_s3_path_style_access(path_style);
            }
            catalog_config.source_storage_profile_with_env(|name| values.string_uncached(name))
        },
        IcebergCatalogBackendConfig::Rest(rest) => {
            rest.source_storage_profile_with_env(|name| values.string_uncached(name))
        },
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ServerNativeMemoryPolicy {
    Bounded,
    Unbounded,
}

fn parse_native_memory_policy(raw: &str) -> Result<ServerNativeMemoryPolicy> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "bounded" => Ok(ServerNativeMemoryPolicy::Bounded),
        "unbounded" => Ok(ServerNativeMemoryPolicy::Unbounded),
        other => Err(ServerError::invalid_config(
            NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV,
            format!("must be `bounded` or `unbounded`, got `{other}`"),
        )),
    }
}

#[cfg(feature = "iceberg")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IcebergCatalogKind {
    Glue,
    Rest,
}

#[cfg(feature = "iceberg")]
fn iceberg_catalog_kind(values: &mut EnvValues<'_>) -> Result<IcebergCatalogKind> {
    match values.string(NEXUS_ICEBERG_CATALOG_KIND_ENV) {
        None => Ok(IcebergCatalogKind::Glue),
        Some(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "glue" => Ok(IcebergCatalogKind::Glue),
            "rest" => Ok(IcebergCatalogKind::Rest),
            other => Err(ServerError::invalid_config(
                NEXUS_ICEBERG_CATALOG_KIND_ENV,
                format!("must be `glue` or `rest`, got `{other}`"),
            )),
        },
    }
}

#[cfg(feature = "iceberg")]
fn iceberg_config(values: &mut EnvValues<'_>) -> Result<Option<IcebergServerConfig>> {
    let requested = values.any_present([
        NEXUS_ICEBERG_CATALOG_KIND_ENV,
        NEXUS_ICEBERG_CATALOG_NAME_ENV,
        NEXUS_ICEBERG_NAMESPACE_ENV,
        NEXUS_ICEBERG_TABLES_ENV,
        NEXUS_ICEBERG_GLUE_CATALOG_NAME_ENV,
        NEXUS_ICEBERG_GLUE_DATABASE_ENV,
        NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV,
        NEXUS_ICEBERG_GLUE_CATALOG_ID_ENV,
        NEXUS_ICEBERG_GLUE_TABLES_ENV,
        NEXUS_ICEBERG_GLUE_URI_ENV,
        NEXUS_ICEBERG_REST_URI_ENV,
        NEXUS_ICEBERG_REST_PREFIX_ENV,
        NEXUS_ICEBERG_REST_TOKEN_ENV,
        NEXUS_ICEBERG_REST_CATALOG_HANDLE_ENV,
        NEXUS_ICEBERG_S3_ENDPOINT_ENV,
        NEXUS_ICEBERG_S3_REGION_ENV,
        NEXUS_ICEBERG_S3_PATH_STYLE_ENV,
        NEXUS_ICEBERG_PARQUET_READ_STRATEGY_ENV,
        NEXUS_ICEBERG_REMOTE_DIRECT_ENV,
        NEXUS_ICEBERG_REMOTE_DIRECT_READ_MODE_ENV,
        NEXUS_ICEBERG_REMOTE_HOST_BUFFER_RANGED_ENV,
        NEXUS_LAKEHOUSE_CACHE_ROOT_ENV,
        NEXUS_LAKEHOUSE_CACHE_READ_ENV,
        NEXUS_LAKEHOUSE_CACHE_POLICY_ENV,
        NEXUS_LAKEHOUSE_CACHE_MAX_BYTES_ENV,
    ]);
    if !requested {
        return Ok(None);
    }

    let kind = iceberg_catalog_kind(values)?;

    // Backend-neutral NEXUS_ICEBERG_NAMESPACE takes precedence; fall
    // back to the Glue-specific NEXUS_ICEBERG_GLUE_DATABASE alias.
    let namespace = values
        .string(NEXUS_ICEBERG_NAMESPACE_ENV)
        .or_else(|| values.string(NEXUS_ICEBERG_GLUE_DATABASE_ENV))
        .ok_or_else(|| {
            ServerError::invalid_config(
                NEXUS_ICEBERG_NAMESPACE_ENV,
                "is required (also accepted as NEXUS_ICEBERG_GLUE_DATABASE)",
            )
        })?;

    let warehouse = values.required_string(NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV)?;

    // Backend-neutral NEXUS_ICEBERG_TABLES takes precedence; fall
    // back to the Glue-specific NEXUS_ICEBERG_GLUE_TABLES alias.
    // When neither is set the dynamic catalog serves all tables and aliases
    // are not required (empty vec is valid in Milestone 2).
    let raw_tables = values.string(NEXUS_ICEBERG_TABLES_ENV).or_else(|| values.string(NEXUS_ICEBERG_GLUE_TABLES_ENV));
    let tables = match raw_tables {
        Some(ref raw) => {
            let tables_env_name = if values.string(NEXUS_ICEBERG_TABLES_ENV).is_some() {
                NEXUS_ICEBERG_TABLES_ENV
            } else {
                NEXUS_ICEBERG_GLUE_TABLES_ENV
            };
            parse_iceberg_tables(tables_env_name, raw)?
        },
        None => vec![],
    };

    // Backend-neutral NEXUS_ICEBERG_CATALOG_NAME takes precedence;
    // fall back to the Glue-specific alias, then to the compiled-in default.
    let default_catalog_name = match kind {
        IcebergCatalogKind::Glue => DEFAULT_GLUE_CATALOG_NAME,
        IcebergCatalogKind::Rest => DEFAULT_REST_CATALOG_HANDLE,
    };
    let catalog_name = values
        .string(NEXUS_ICEBERG_CATALOG_NAME_ENV)
        .or_else(|| values.string(NEXUS_ICEBERG_GLUE_CATALOG_NAME_ENV))
        .unwrap_or_else(|| default_catalog_name.to_string());

    let backend = match kind {
        IcebergCatalogKind::Glue => IcebergCatalogBackendConfig::Glue(glue_backend_config(values, &warehouse)?),
        IcebergCatalogKind::Rest => IcebergCatalogBackendConfig::Rest(rest_backend_config(values, warehouse.clone())?),
    };
    let parquet_read_strategy_config = iceberg_parquet_read_strategy_config(values, &backend)?;

    Ok(Some(IcebergServerConfig { catalog_name, namespace, tables, backend, parquet_read_strategy_config }))
}

#[cfg(feature = "iceberg")]
fn iceberg_parquet_read_strategy_config(
    values: &mut EnvValues<'_>,
    backend: &IcebergCatalogBackendConfig,
) -> Result<TableFormatParquetReadStrategyConfig> {
    let mut config = table_format_parquet_read_strategy_config(values)?;
    config.storage_profile_id = Some(DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID.to_string());
    if let Some(endpoint) = iceberg_backend_s3_endpoint(backend).filter(|endpoint| !endpoint.trim().is_empty()) {
        config.storage_endpoint = Some(credential_free_endpoint_identity(endpoint));
    }
    if iceberg_backend_requires_scoped_remote_direct_profile(backend) {
        config.credential_mode = nexus_query_engine::CredentialMode::ScopedResolved;
    }
    Ok(config)
}

#[cfg(feature = "iceberg")]
fn iceberg_backend_s3_endpoint(backend: &IcebergCatalogBackendConfig) -> Option<&str> {
    match backend {
        IcebergCatalogBackendConfig::Glue(glue) => glue.s3_endpoint.as_deref(),
        IcebergCatalogBackendConfig::Rest(rest) => rest.s3_endpoint.as_deref(),
    }
}

#[cfg(feature = "iceberg")]
fn iceberg_backend_requires_scoped_remote_direct_profile(backend: &IcebergCatalogBackendConfig) -> bool {
    match backend {
        IcebergCatalogBackendConfig::Glue(glue) => {
            !glue.region.trim().is_empty()
                || glue.s3_region.as_ref().is_some_and(|region| !region.trim().is_empty())
                || glue.s3_endpoint.as_ref().is_some_and(|endpoint| !endpoint.trim().is_empty())
                || glue.s3_path_style_access.is_some()
        },
        IcebergCatalogBackendConfig::Rest(rest) => {
            rest.s3_region.as_ref().is_some_and(|region| !region.trim().is_empty())
                || rest.s3_endpoint.as_ref().is_some_and(|endpoint| !endpoint.trim().is_empty())
                || rest.s3_path_style_access.is_some()
        },
    }
}

#[cfg(feature = "iceberg")]
fn table_format_parquet_read_strategy_config(
    values: &mut EnvValues<'_>,
) -> Result<TableFormatParquetReadStrategyConfig> {
    let preference = values
        .string(NEXUS_ICEBERG_PARQUET_READ_STRATEGY_ENV)
        .map(|raw| parse_table_format_parquet_read_strategy(&raw))
        .transpose()?
        .unwrap_or_default();
    let remote_direct_enabled = values.optional_bool(NEXUS_ICEBERG_REMOTE_DIRECT_ENV)?.unwrap_or(true);
    let remote_direct_read_mode = values
        .string(NEXUS_ICEBERG_REMOTE_DIRECT_READ_MODE_ENV)
        .map(|raw| parse_remote_parquet_read_mode(&raw))
        .transpose()?
        .unwrap_or_default();
    if !remote_direct_enabled && remote_direct_read_mode == nexus_query_engine::RemoteParquetReadMode::RequireDirect {
        return Err(ServerError::invalid_config(
            NEXUS_ICEBERG_REMOTE_DIRECT_ENV,
            format!("cannot be disabled when {NEXUS_ICEBERG_REMOTE_DIRECT_READ_MODE_ENV} is `require_direct`"),
        ));
    }
    let remote_host_buffer_ranged_enabled =
        values.optional_bool(NEXUS_ICEBERG_REMOTE_HOST_BUFFER_RANGED_ENV)?.unwrap_or(false);
    let lakehouse_cache_root = values.string(NEXUS_LAKEHOUSE_CACHE_ROOT_ENV).map(PathBuf::from);
    let lakehouse_cache_read_policy = values
        .string(NEXUS_LAKEHOUSE_CACHE_READ_ENV)
        .map(|raw| parse_lakehouse_cache_read_policy(&raw))
        .transpose()?
        .unwrap_or_default();
    let lakehouse_cache_policy = values
        .string(NEXUS_LAKEHOUSE_CACHE_POLICY_ENV)
        .map(|raw| parse_lakehouse_cache_policy(&raw))
        .transpose()?
        .unwrap_or_default();
    let lakehouse_cache_max_bytes = match values.optional_u64(NEXUS_LAKEHOUSE_CACHE_MAX_BYTES_ENV)? {
        Some(0) => {
            return Err(ServerError::invalid_config(NEXUS_LAKEHOUSE_CACHE_MAX_BYTES_ENV, "must be greater than zero"));
        },
        Some(value) => Some(value),
        None => None,
    };
    let storage_endpoint = values
        .string(NEXUS_ICEBERG_S3_ENDPOINT_ENV)
        .or_else(|| values.string(AWS_ENDPOINT_URL_S3_ENV))
        .or_else(|| values.string(AWS_ENDPOINT_URL_ENV))
        .map(|endpoint| credential_free_endpoint_identity(&endpoint));
    let has_explicit_s3_credentials = values.string(crate::iceberg::NEXUS_ICEBERG_S3_ACCESS_KEY_ID_ENV).is_some()
        || values.string(crate::iceberg::NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY_ENV).is_some()
        || values.string(crate::iceberg::NEXUS_ICEBERG_S3_SESSION_TOKEN_ENV).is_some();
    let has_storage_region = values.string(NEXUS_ICEBERG_S3_REGION_ENV).is_some();
    let credential_mode = if storage_endpoint.is_some() || has_storage_region || has_explicit_s3_credentials {
        nexus_query_engine::CredentialMode::ScopedResolved
    } else {
        nexus_query_engine::CredentialMode::SharedProcessEnv
    };

    Ok(TableFormatParquetReadStrategyConfig {
        preference,
        remote_direct_enabled,
        remote_direct_read_mode,
        remote_host_buffer_ranged_enabled,
        lakehouse_cache_root,
        lakehouse_cache_read_policy,
        lakehouse_cache_policy,
        lakehouse_cache_max_bytes,
        storage_endpoint,
        credential_mode,
        ..Default::default()
    })
}

#[cfg(feature = "iceberg")]
fn parse_table_format_parquet_read_strategy(raw: &str) -> Result<TableFormatParquetReadStrategyPreference> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(TableFormatParquetReadStrategyPreference::Auto),
        "host_buffer" => Ok(TableFormatParquetReadStrategyPreference::HostBuffer),
        "local_cache" => Ok(TableFormatParquetReadStrategyPreference::LocalCache),
        "remote_kvikio" => Ok(TableFormatParquetReadStrategyPreference::RemoteKvikio),
        other => Err(ServerError::invalid_config(
            NEXUS_ICEBERG_PARQUET_READ_STRATEGY_ENV,
            format!("must be `auto`, `host_buffer`, `local_cache`, or `remote_kvikio`, got `{other}`"),
        )),
    }
}

#[cfg(feature = "iceberg")]
fn parse_remote_parquet_read_mode(raw: &str) -> Result<nexus_query_engine::RemoteParquetReadMode> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "prefer" | "prefer_direct" => Ok(nexus_query_engine::RemoteParquetReadMode::PreferDirect),
        "require" | "require_direct" => Ok(nexus_query_engine::RemoteParquetReadMode::RequireDirect),
        "allow_host_fallback" | "host_fallback" => Ok(nexus_query_engine::RemoteParquetReadMode::AllowHostFallback),
        other => Err(ServerError::invalid_config(
            NEXUS_ICEBERG_REMOTE_DIRECT_READ_MODE_ENV,
            format!("must be `prefer_direct`, `require_direct`, or `allow_host_fallback`, got `{other}`"),
        )),
    }
}

#[cfg(feature = "iceberg")]
fn parse_lakehouse_cache_read_policy(raw: &str) -> Result<TableFormatLakehouseCacheReadPolicy> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "enabled" => Ok(TableFormatLakehouseCacheReadPolicy::Enabled),
        "disabled" | "bypass" | "bypassed" => Ok(TableFormatLakehouseCacheReadPolicy::Disabled),
        other => Err(ServerError::invalid_config(
            NEXUS_LAKEHOUSE_CACHE_READ_ENV,
            format!("must be `enabled`, `disabled`, or `bypass`, got `{other}`"),
        )),
    }
}

#[cfg(feature = "iceberg")]
fn parse_lakehouse_cache_policy(raw: &str) -> Result<TableFormatLakehouseCachePolicy> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "disabled" => Ok(TableFormatLakehouseCachePolicy::Disabled),
        "read_through" => Ok(TableFormatLakehouseCachePolicy::ReadThrough),
        "fill_before_read" => Ok(TableFormatLakehouseCachePolicy::FillBeforeRead),
        other => Err(ServerError::invalid_config(
            NEXUS_LAKEHOUSE_CACHE_POLICY_ENV,
            format!("must be `disabled`, `read_through`, or `fill_before_read`, got `{other}`"),
        )),
    }
}

#[cfg(feature = "iceberg")]
fn glue_backend_config(values: &mut EnvValues<'_>, warehouse: &str) -> Result<GlueServerBackendConfig> {
    let region = values
        .string(crate::iceberg::AWS_REGION_ENV)
        .or_else(|| values.string(crate::iceberg::AWS_DEFAULT_REGION_ENV))
        .ok_or_else(|| ServerError::invalid_config(crate::iceberg::AWS_REGION_ENV, "is required for Glue Iceberg"))?;

    // catalog_handle is the internal handle passed to iceberg-rust's
    // GlueCatalogBuilder; it is independent of the DataFusion catalog_name so
    // users can choose a different connector name without changing the Glue
    // catalog identifier.  Controlled only by the Glue-specific env var.
    let catalog_handle =
        values.string(NEXUS_ICEBERG_GLUE_CATALOG_NAME_ENV).unwrap_or_else(|| DEFAULT_GLUE_CATALOG_NAME.to_string());

    Ok(GlueServerBackendConfig {
        catalog_handle,
        warehouse: warehouse.to_string(),
        region,
        catalog_id: values.string(NEXUS_ICEBERG_GLUE_CATALOG_ID_ENV),
        profile: values.string(crate::iceberg::AWS_PROFILE_ENV),
        uri: values.string(NEXUS_ICEBERG_GLUE_URI_ENV),
        s3_endpoint: values.string(NEXUS_ICEBERG_S3_ENDPOINT_ENV),
        s3_region: values.string(NEXUS_ICEBERG_S3_REGION_ENV),
        s3_path_style_access: values.optional_bool(NEXUS_ICEBERG_S3_PATH_STYLE_ENV)?,
    })
}

#[cfg(feature = "iceberg")]
fn rest_backend_config(
    values: &mut EnvValues<'_>,
    warehouse: String,
) -> Result<crate::iceberg::catalog::RestIcebergCatalogConfig> {
    let uri = values.required_string(NEXUS_ICEBERG_REST_URI_ENV)?;
    let catalog_handle =
        values.string(NEXUS_ICEBERG_REST_CATALOG_HANDLE_ENV).unwrap_or_else(|| DEFAULT_REST_CATALOG_HANDLE.to_string());

    let mut config = crate::iceberg::catalog::RestIcebergCatalogConfig::new(catalog_handle, uri, warehouse);
    if let Some(prefix) = values.string(NEXUS_ICEBERG_REST_PREFIX_ENV) {
        config = config.with_prefix(prefix);
    }
    if let Some(token) = values.string(NEXUS_ICEBERG_REST_TOKEN_ENV) {
        config = config.with_token(token);
    }
    if let Some(endpoint) = values.string(NEXUS_ICEBERG_S3_ENDPOINT_ENV) {
        config = config.with_s3_endpoint(endpoint);
    }
    if let Some(region) = values.string(NEXUS_ICEBERG_S3_REGION_ENV) {
        config = config.with_s3_region(region);
    }
    if let Some(path_style) = values.optional_bool(NEXUS_ICEBERG_S3_PATH_STYLE_ENV)? {
        config = config.with_s3_path_style_access(path_style);
    }
    Ok(config)
}

#[cfg(feature = "iceberg")]
fn parse_iceberg_tables(env_name: &'static str, raw: &str) -> Result<Vec<IcebergTableRegistrationConfig>> {
    let mut tables = Vec::new();
    for entry in raw.split(',').map(str::trim).filter(|entry| !entry.is_empty()) {
        let (alias, table) = match entry.split_once('=') {
            Some((alias, table)) => (alias.trim(), table.trim()),
            None => (entry, entry),
        };
        if alias.is_empty() || table.is_empty() {
            return Err(ServerError::invalid_config(env_name, "entries must be table or alias=table"));
        }
        tables.push(IcebergTableRegistrationConfig { table: table.to_string(), alias: alias.to_string() });
    }
    if tables.is_empty() {
        return Err(ServerError::invalid_config(env_name, "must contain at least one table"));
    }
    Ok(tables)
}

#[cfg(feature = "cugraph")]
fn cugraph_config(values: &mut EnvValues<'_>) -> Result<Option<CugraphServerConfig>> {
    let requested = values.any_present([
        NEXUS_SERVER_CUGRAPH_ALGORITHMS_ENV,
        NEXUS_SERVER_CUGRAPH_DIRECTED_ENV,
        NEXUS_SERVER_CUGRAPH_RENUMBER_ENV,
        NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY_ENV,
    ]);
    let enabled = values.optional_bool(NEXUS_SERVER_CUGRAPH_ENABLED_ENV)?;
    if enabled != Some(true) {
        if requested {
            return Err(ServerError::invalid_config(
                NEXUS_SERVER_CUGRAPH_ENABLED_ENV,
                "must be true when cuGraph server settings are provided",
            ));
        }
        return Ok(None);
    }

    let allowed_algorithms = values
        .string(NEXUS_SERVER_CUGRAPH_ALGORITHMS_ENV)
        .map(|raw| parse_cugraph_algorithms(NEXUS_SERVER_CUGRAPH_ALGORITHMS_ENV, &raw))
        .transpose()?;
    let mut default_graph_options = CugraphGraphOptions::default();
    if let Some(directed) = values.optional_bool(NEXUS_SERVER_CUGRAPH_DIRECTED_ENV)? {
        default_graph_options.directed = directed;
    }
    if let Some(renumber) = values.optional_bool(NEXUS_SERVER_CUGRAPH_RENUMBER_ENV)? {
        default_graph_options.renumber = renumber;
    }
    if let Some(raw) = values.string(NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY_ENV) {
        default_graph_options.construction_policy =
            parse_cugraph_construction_policy(NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY_ENV, &raw)?;
    }

    Ok(Some(CugraphServerConfig { allowed_algorithms, default_graph_options }))
}

#[cfg(feature = "cugraph")]
fn parse_cugraph_algorithms(env_name: &'static str, raw: &str) -> Result<std::collections::BTreeSet<CugraphAlgorithm>> {
    let mut algorithms = std::collections::BTreeSet::new();
    for entry in raw.split(',').map(str::trim).filter(|entry| !entry.is_empty()) {
        let algorithm =
            CugraphAlgorithm::all().find(|algorithm| algorithm.function_name() == entry).ok_or_else(|| {
                ServerError::invalid_config(
                    env_name,
                    format!("unknown cuGraph algorithm `{entry}`; use SQL function names such as `cugraph_pagerank`"),
                )
            })?;
        algorithms.insert(algorithm);
    }
    if algorithms.is_empty() {
        return Err(ServerError::invalid_config(env_name, "must contain at least one SQL function name"));
    }
    Ok(algorithms)
}

#[cfg(feature = "cugraph")]
fn parse_cugraph_construction_policy(env_name: &'static str, raw: &str) -> Result<CugraphGraphConstructionPolicy> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "python_cugraph" => Ok(CugraphGraphConstructionPolicy::PythonCugraph),
        "raw_libcugraph" => Ok(CugraphGraphConstructionPolicy::RawLibcugraph),
        other => Err(ServerError::invalid_config(
            env_name,
            format!("must be `python_cugraph` or `raw_libcugraph`, got `{other}`"),
        )),
    }
}

#[cfg(not(feature = "iceberg"))]
fn reject_iceberg_env_without_feature(values: &mut EnvValues<'_>) -> Result<()> {
    if values.any_present([
        NEXUS_ICEBERG_CATALOG_KIND_ENV,
        NEXUS_ICEBERG_CATALOG_NAME_ENV,
        NEXUS_ICEBERG_NAMESPACE_ENV,
        NEXUS_ICEBERG_TABLES_ENV,
        NEXUS_ICEBERG_GLUE_CATALOG_NAME_ENV,
        NEXUS_ICEBERG_GLUE_DATABASE_ENV,
        NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV,
        NEXUS_ICEBERG_GLUE_CATALOG_ID_ENV,
        NEXUS_ICEBERG_GLUE_TABLES_ENV,
        NEXUS_ICEBERG_GLUE_URI_ENV,
        NEXUS_ICEBERG_REST_URI_ENV,
        NEXUS_ICEBERG_REST_PREFIX_ENV,
        NEXUS_ICEBERG_REST_TOKEN_ENV,
        NEXUS_ICEBERG_REST_CATALOG_HANDLE_ENV,
        NEXUS_ICEBERG_S3_ENDPOINT_ENV,
        NEXUS_ICEBERG_S3_REGION_ENV,
        NEXUS_ICEBERG_S3_PATH_STYLE_ENV,
        crate::NEXUS_ICEBERG_PARQUET_READ_STRATEGY_ENV,
        crate::NEXUS_ICEBERG_REMOTE_DIRECT_ENV,
        crate::NEXUS_ICEBERG_REMOTE_DIRECT_READ_MODE_ENV,
        crate::NEXUS_ICEBERG_REMOTE_HOST_BUFFER_RANGED_ENV,
        crate::NEXUS_LAKEHOUSE_CACHE_ROOT_ENV,
        crate::NEXUS_LAKEHOUSE_CACHE_READ_ENV,
        crate::NEXUS_LAKEHOUSE_CACHE_POLICY_ENV,
        crate::NEXUS_LAKEHOUSE_CACHE_MAX_BYTES_ENV,
    ]) {
        return Err(ServerError::invalid_config(
            NEXUS_ICEBERG_NAMESPACE_ENV,
            "Iceberg server registration requires building with --features server,iceberg",
        ));
    }
    Ok(())
}

#[cfg(not(feature = "cugraph"))]
fn reject_cugraph_env_without_feature(values: &mut EnvValues<'_>) -> Result<()> {
    if values.any_present([
        NEXUS_SERVER_CUGRAPH_ENABLED_ENV,
        NEXUS_SERVER_CUGRAPH_ALGORITHMS_ENV,
        NEXUS_SERVER_CUGRAPH_DIRECTED_ENV,
        NEXUS_SERVER_CUGRAPH_RENUMBER_ENV,
        NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY_ENV,
    ]) {
        return Err(ServerError::invalid_config(
            NEXUS_SERVER_CUGRAPH_ENABLED_ENV,
            "cuGraph server registration requires building with --features server,cugraph or --features server-cugraph",
        ));
    }
    Ok(())
}

struct EnvValues<'a> {
    get: &'a mut dyn FnMut(&str) -> Option<String>,
    cache: BTreeMap<&'static str, Option<String>>,
}

impl<'a> EnvValues<'a> {
    fn new(get: &'a mut dyn FnMut(&str) -> Option<String>) -> Self {
        Self { get, cache: BTreeMap::new() }
    }

    fn any_present<const N: usize>(&mut self, names: [&'static str; N]) -> bool {
        names.into_iter().any(|name| self.string(name).is_some())
    }

    #[cfg(feature = "iceberg")]
    fn required_string(&mut self, name: &'static str) -> Result<String> {
        self.string(name).ok_or_else(|| ServerError::invalid_config(name, "is required"))
    }

    fn string(&mut self, name: &'static str) -> Option<String> {
        if let Some(cached) = self.cache.get(name) {
            return cached.clone();
        }
        let value = (self.get)(name).and_then(|value| {
            let trimmed = value.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        });
        self.cache.insert(name, value.clone());
        value
    }

    #[cfg(feature = "iceberg")]
    fn string_uncached(&mut self, name: &str) -> Option<String> {
        (self.get)(name).and_then(|value| {
            let trimmed = value.trim();
            if trimmed.is_empty() { None } else { Some(trimmed.to_string()) }
        })
    }

    fn socket_addr(&mut self, name: &'static str, default: &str) -> Result<SocketAddr> {
        let raw = self.string(name).unwrap_or_else(|| default.to_string());
        raw.parse().map_err(|error| ServerError::invalid_config(name, format!("must be a socket address: {error}")))
    }

    fn positive_usize(&mut self, name: &'static str, default: usize) -> Result<usize> {
        let value = self.optional_usize(name)?.unwrap_or(default);
        if value == 0 {
            return Err(ServerError::invalid_config(name, "must be greater than zero"));
        }
        Ok(value)
    }

    fn positive_u64(&mut self, name: &'static str, default: u64) -> Result<u64> {
        let value = self.optional_u64(name)?.unwrap_or(default);
        if value == 0 {
            return Err(ServerError::invalid_config(name, "must be greater than zero"));
        }
        Ok(value)
    }

    fn optional_usize(&mut self, name: &'static str) -> Result<Option<usize>> {
        self.string(name)
            .map(|raw| {
                raw.parse::<usize>()
                    .map_err(|error| ServerError::invalid_config(name, format!("must be an integer: {error}")))
            })
            .transpose()
    }

    fn optional_u64(&mut self, name: &'static str) -> Result<Option<u64>> {
        self.string(name)
            .map(|raw| {
                raw.parse::<u64>()
                    .map_err(|error| ServerError::invalid_config(name, format!("must be an integer: {error}")))
            })
            .transpose()
    }

    fn optional_session_profile(&mut self, name: &'static str) -> Result<Option<ServerSessionProfile>> {
        self.string(name)
            .map(|raw| ServerSessionProfile::parse(&raw).map_err(|reason| ServerError::invalid_config(name, reason)))
            .transpose()
    }

    #[cfg(any(feature = "cugraph", feature = "iceberg"))]
    fn optional_bool(&mut self, name: &'static str) -> Result<Option<bool>> {
        self.string(name)
            .map(|raw| match raw.trim().to_ascii_lowercase().as_str() {
                "1" | "true" | "yes" | "on" => Ok(true),
                "0" | "false" | "no" | "off" => Ok(false),
                other => {
                    Err(ServerError::invalid_config(name, format!("must be a boolean (`true`/`false`), got `{other}`")))
                },
            })
            .transpose()
    }
}
