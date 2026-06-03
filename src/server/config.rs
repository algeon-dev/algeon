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
    AWS_ENDPOINT_URL_ENV, AWS_ENDPOINT_URL_S3_ENV, CUDF_ICEBERG_PARQUET_READ_STRATEGY_ENV,
    CUDF_ICEBERG_REMOTE_KVIKIO_EXPERIMENTAL_ENV, CUDF_LAKEHOUSE_CACHE_MAX_BYTES_ENV, CUDF_LAKEHOUSE_CACHE_POLICY_ENV,
    CUDF_LAKEHOUSE_CACHE_ROOT_ENV, TableFormatLakehouseCachePolicy, TableFormatParquetReadStrategyConfig,
    TableFormatParquetReadStrategyPreference,
};
use crate::{CudfOptimizerConfig, GpuFallbackPolicy, NativeCudfExecutionConfig};
#[cfg(feature = "cugraph")]
use crate::{CugraphAlgorithm, CugraphGraphConstructionPolicy, CugraphGraphOptions};

pub const DATAFUSION_NEXUS_SERVER_BIND_ENV: &str = "DATAFUSION_NEXUS_SERVER_BIND";
pub const DATAFUSION_NEXUS_SERVER_LOG_ENV: &str = "DATAFUSION_NEXUS_SERVER_LOG";
pub const DATAFUSION_NEXUS_SERVER_MAX_IN_FLIGHT_ENV: &str = "DATAFUSION_NEXUS_SERVER_MAX_IN_FLIGHT";
pub const DATAFUSION_NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS_ENV: &str =
    "DATAFUSION_NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS";
/// Selects the DataFusion physical-plan profile applied to the Flight SQL
/// session. Valid values are `native` and `cpu_fallback`. When unset, the
/// profile defaults to `native` if an Iceberg catalog is configured and to
/// `cpu_fallback` otherwise. The native profile pins `target_partitions=1`,
/// disables all DataFusion repartitioning, and disables dynamic filter
/// pushdown so plans match the native-lowerable shapes the cudf optimizer rule
/// expects. The CPU-fallback profile leaves DataFusion's standard
/// `DATAFUSION_EXECUTION_*` settings in place (DataFusion's
/// `SessionConfig::from_env()` maps option keys to env vars by upper-casing
/// and replacing `.` with `_`, so single-underscore names like
/// `DATAFUSION_EXECUTION_TARGET_PARTITIONS` are what actually take effect).
pub const DATAFUSION_NEXUS_SERVER_SESSION_PROFILE_ENV: &str = "DATAFUSION_NEXUS_SERVER_SESSION_PROFILE";
pub const DATAFUSION_NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV: &str =
    "DATAFUSION_NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES";
pub const DATAFUSION_NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV: &str = "DATAFUSION_NEXUS_SERVER_NATIVE_MEMORY_POLICY";
pub const DATAFUSION_NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV: &str =
    "DATAFUSION_NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK";
pub const DATAFUSION_NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV: &str =
    "DATAFUSION_NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES";
/// Path to a TOML file that supplies the `[auth]` block (and may
/// supply other future sections). Env vars still override scalar
/// fields where they exist; TOML is the only way to configure the
/// `auth` section because it carries a nested user list.
pub const DATAFUSION_NEXUS_SERVER_CONFIG_FILE_ENV: &str = "DATAFUSION_NEXUS_SERVER_CONFIG_FILE";

pub const DATAFUSION_NEXUS_SERVER_CUGRAPH_ENABLED_ENV: &str = "DATAFUSION_NEXUS_SERVER_CUGRAPH_ENABLED";
pub const DATAFUSION_NEXUS_SERVER_CUGRAPH_ALGORITHMS_ENV: &str = "DATAFUSION_NEXUS_SERVER_CUGRAPH_ALGORITHMS";
pub const DATAFUSION_NEXUS_SERVER_CUGRAPH_DIRECTED_ENV: &str = "DATAFUSION_NEXUS_SERVER_CUGRAPH_DIRECTED";
pub const DATAFUSION_NEXUS_SERVER_CUGRAPH_RENUMBER_ENV: &str = "DATAFUSION_NEXUS_SERVER_CUGRAPH_RENUMBER";
pub const DATAFUSION_NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY_ENV: &str =
    "DATAFUSION_NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY";

// Backend-neutral Iceberg env vars (preferred for new deployments).
pub const DATAFUSION_NEXUS_ICEBERG_CATALOG_NAME_ENV: &str = "DATAFUSION_NEXUS_ICEBERG_CATALOG_NAME";
pub const DATAFUSION_NEXUS_ICEBERG_NAMESPACE_ENV: &str = "DATAFUSION_NEXUS_ICEBERG_NAMESPACE";
pub const DATAFUSION_NEXUS_ICEBERG_TABLES_ENV: &str = "DATAFUSION_NEXUS_ICEBERG_TABLES";

/// Selects which Iceberg catalog backend the server connects to. Accepts
/// `glue` (default when unset) or `rest`. Any other value is rejected at
/// config load time.
pub const DATAFUSION_NEXUS_ICEBERG_CATALOG_KIND_ENV: &str = "DATAFUSION_NEXUS_ICEBERG_CATALOG_KIND";

// Glue-specific env vars (retained for backward compatibility).
pub const DATAFUSION_NEXUS_ICEBERG_GLUE_CATALOG_NAME_ENV: &str = "DATAFUSION_NEXUS_ICEBERG_GLUE_CATALOG_NAME";
pub const DATAFUSION_NEXUS_ICEBERG_GLUE_DATABASE_ENV: &str = "DATAFUSION_NEXUS_ICEBERG_GLUE_DATABASE";
pub const DATAFUSION_NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV: &str = "DATAFUSION_NEXUS_ICEBERG_WAREHOUSE";
pub const DATAFUSION_NEXUS_ICEBERG_GLUE_CATALOG_ID_ENV: &str = "DATAFUSION_NEXUS_ICEBERG_GLUE_CATALOG_ID";
pub const DATAFUSION_NEXUS_ICEBERG_GLUE_TABLES_ENV: &str = "DATAFUSION_NEXUS_ICEBERG_GLUE_TABLES";
pub const DATAFUSION_NEXUS_ICEBERG_GLUE_URI_ENV: &str = "DATAFUSION_NEXUS_ICEBERG_GLUE_URI";

// REST-specific env vars. The warehouse reuses
// `DATAFUSION_NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV` (`DATAFUSION_NEXUS_ICEBERG_WAREHOUSE`).
pub const DATAFUSION_NEXUS_ICEBERG_REST_URI_ENV: &str = "DATAFUSION_NEXUS_ICEBERG_REST_URI";
pub const DATAFUSION_NEXUS_ICEBERG_REST_PREFIX_ENV: &str = "DATAFUSION_NEXUS_ICEBERG_REST_PREFIX";
pub const DATAFUSION_NEXUS_ICEBERG_REST_TOKEN_ENV: &str = "DATAFUSION_NEXUS_ICEBERG_REST_TOKEN";
pub const DATAFUSION_NEXUS_ICEBERG_REST_CATALOG_HANDLE_ENV: &str = "DATAFUSION_NEXUS_ICEBERG_REST_CATALOG_HANDLE";
pub const DATAFUSION_NEXUS_ICEBERG_S3_ENDPOINT_ENV: &str = "DATAFUSION_NEXUS_ICEBERG_S3_ENDPOINT";
pub const DATAFUSION_NEXUS_ICEBERG_S3_REGION_ENV: &str = "DATAFUSION_NEXUS_ICEBERG_S3_REGION";
pub const DATAFUSION_NEXUS_ICEBERG_S3_PATH_STYLE_ENV: &str = "DATAFUSION_NEXUS_ICEBERG_S3_PATH_STYLE";

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
    /// See [`DATAFUSION_NEXUS_SERVER_SESSION_PROFILE_ENV`] and
    /// [`ServerSessionProfile`].
    pub session_profile: ServerSessionProfile,
    /// Authentication mode applied to incoming Flight SQL RPCs. Loaded
    /// from the TOML overlay pointed to by
    /// [`DATAFUSION_NEXUS_SERVER_CONFIG_FILE_ENV`]; defaults to
    /// [`super::auth::AuthMode::Disabled`] when no file is provided.
    pub auth: AuthConfig,
    #[cfg(feature = "iceberg")]
    pub iceberg: Option<IcebergServerConfig>,
    #[cfg(feature = "cugraph")]
    pub cugraph: Option<CugraphServerConfig>,
}

/// Raw shape of the optional TOML overlay loaded from
/// [`DATAFUSION_NEXUS_SERVER_CONFIG_FILE_ENV`]. Currently only carries
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
/// `NATIVE_*` environment variables.
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
    /// vice versa). Selected by [`DATAFUSION_NEXUS_ICEBERG_CATALOG_KIND_ENV`].
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
        let bind_addr = values.socket_addr(DATAFUSION_NEXUS_SERVER_BIND_ENV, DEFAULT_BIND_ADDR)?;
        let log_filter =
            values.string(DATAFUSION_NEXUS_SERVER_LOG_ENV).unwrap_or_else(|| DEFAULT_LOG_FILTER.to_string());
        let max_in_flight = values.positive_usize(DATAFUSION_NEXUS_SERVER_MAX_IN_FLIGHT_ENV, DEFAULT_MAX_IN_FLIGHT)?;
        let statement_ticket_ttl =
            Duration::from_secs(values.positive_u64(
                DATAFUSION_NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS_ENV,
                DEFAULT_STATEMENT_TICKET_TTL_SECS,
            )?);
        let native_optimizer_config = native_optimizer_config(&mut values)?;
        #[cfg(feature = "iceberg")]
        let mut native_optimizer_config = native_optimizer_config;

        #[cfg(feature = "iceberg")]
        let iceberg = iceberg_config(&mut values)?;

        #[cfg(feature = "iceberg")]
        if let Some(profile) = iceberg.as_ref().map(|config| iceberg_source_storage_profile(config, &mut values)) {
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

        let explicit_profile = values.optional_session_profile(DATAFUSION_NEXUS_SERVER_SESSION_PROFILE_ENV)?;
        #[cfg(feature = "iceberg")]
        let default_profile =
            if iceberg.is_some() { ServerSessionProfile::Native } else { ServerSessionProfile::CpuFallback };
        #[cfg(not(feature = "iceberg"))]
        let default_profile = ServerSessionProfile::CpuFallback;
        let session_profile = explicit_profile.unwrap_or(default_profile);

        let overlay_path = values.string(DATAFUSION_NEXUS_SERVER_CONFIG_FILE_ENV).map(PathBuf::from);
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
        .string(DATAFUSION_NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV)
        .map(|raw| parse_native_memory_policy(&raw))
        .transpose()?
        .unwrap_or(ServerNativeMemoryPolicy::Bounded);
    let device_budget = values.optional_u64(DATAFUSION_NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV)?;
    let max_source_chunk = values.optional_u64(DATAFUSION_NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV)?;
    let max_row_groups = values.optional_usize(DATAFUSION_NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV)?;

    let execution_config = match memory_policy {
        ServerNativeMemoryPolicy::Unbounded => {
            if device_budget.is_some() || max_source_chunk.is_some() || max_row_groups.is_some() {
                return Err(ServerError::invalid_config(
                    DATAFUSION_NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV,
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
                        DATAFUSION_NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV,
                        "must be greater than zero",
                    ));
                }
                if max_source_chunk == 0 {
                    return Err(ServerError::invalid_config(
                        DATAFUSION_NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV,
                        "must be greater than zero",
                    ));
                }
                if max_row_groups == 0 {
                    return Err(ServerError::invalid_config(
                        DATAFUSION_NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV,
                        "must be greater than zero",
                    ));
                }
                NativeCudfExecutionConfig::bounded(device_budget, max_source_chunk, max_row_groups, 1, 1)
            },
            _ => {
                return Err(ServerError::invalid_config(
                    DATAFUSION_NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV,
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
            DATAFUSION_NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV,
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
    match values.string(DATAFUSION_NEXUS_ICEBERG_CATALOG_KIND_ENV) {
        None => Ok(IcebergCatalogKind::Glue),
        Some(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "glue" => Ok(IcebergCatalogKind::Glue),
            "rest" => Ok(IcebergCatalogKind::Rest),
            other => Err(ServerError::invalid_config(
                DATAFUSION_NEXUS_ICEBERG_CATALOG_KIND_ENV,
                format!("must be `glue` or `rest`, got `{other}`"),
            )),
        },
    }
}

#[cfg(feature = "iceberg")]
fn iceberg_config(values: &mut EnvValues<'_>) -> Result<Option<IcebergServerConfig>> {
    let requested = values.any_present([
        DATAFUSION_NEXUS_ICEBERG_CATALOG_KIND_ENV,
        DATAFUSION_NEXUS_ICEBERG_CATALOG_NAME_ENV,
        DATAFUSION_NEXUS_ICEBERG_NAMESPACE_ENV,
        DATAFUSION_NEXUS_ICEBERG_TABLES_ENV,
        DATAFUSION_NEXUS_ICEBERG_GLUE_CATALOG_NAME_ENV,
        DATAFUSION_NEXUS_ICEBERG_GLUE_DATABASE_ENV,
        DATAFUSION_NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV,
        DATAFUSION_NEXUS_ICEBERG_GLUE_CATALOG_ID_ENV,
        DATAFUSION_NEXUS_ICEBERG_GLUE_TABLES_ENV,
        DATAFUSION_NEXUS_ICEBERG_GLUE_URI_ENV,
        DATAFUSION_NEXUS_ICEBERG_REST_URI_ENV,
        DATAFUSION_NEXUS_ICEBERG_REST_PREFIX_ENV,
        DATAFUSION_NEXUS_ICEBERG_REST_TOKEN_ENV,
        DATAFUSION_NEXUS_ICEBERG_REST_CATALOG_HANDLE_ENV,
        DATAFUSION_NEXUS_ICEBERG_S3_ENDPOINT_ENV,
        DATAFUSION_NEXUS_ICEBERG_S3_REGION_ENV,
        DATAFUSION_NEXUS_ICEBERG_S3_PATH_STYLE_ENV,
        CUDF_ICEBERG_PARQUET_READ_STRATEGY_ENV,
        CUDF_ICEBERG_REMOTE_KVIKIO_EXPERIMENTAL_ENV,
        CUDF_LAKEHOUSE_CACHE_ROOT_ENV,
        CUDF_LAKEHOUSE_CACHE_POLICY_ENV,
        CUDF_LAKEHOUSE_CACHE_MAX_BYTES_ENV,
    ]);
    if !requested {
        return Ok(None);
    }

    let kind = iceberg_catalog_kind(values)?;

    // Backend-neutral DATAFUSION_NEXUS_ICEBERG_NAMESPACE takes precedence; fall
    // back to the Glue-specific DATAFUSION_NEXUS_ICEBERG_GLUE_DATABASE alias.
    let namespace = values
        .string(DATAFUSION_NEXUS_ICEBERG_NAMESPACE_ENV)
        .or_else(|| values.string(DATAFUSION_NEXUS_ICEBERG_GLUE_DATABASE_ENV))
        .ok_or_else(|| {
            ServerError::invalid_config(
                DATAFUSION_NEXUS_ICEBERG_NAMESPACE_ENV,
                "is required (also accepted as DATAFUSION_NEXUS_ICEBERG_GLUE_DATABASE)",
            )
        })?;

    let warehouse = values.required_string(DATAFUSION_NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV)?;

    // Backend-neutral DATAFUSION_NEXUS_ICEBERG_TABLES takes precedence; fall
    // back to the Glue-specific DATAFUSION_NEXUS_ICEBERG_GLUE_TABLES alias.
    // When neither is set the dynamic catalog serves all tables and aliases
    // are not required (empty vec is valid in Milestone 2).
    let raw_tables = values
        .string(DATAFUSION_NEXUS_ICEBERG_TABLES_ENV)
        .or_else(|| values.string(DATAFUSION_NEXUS_ICEBERG_GLUE_TABLES_ENV));
    let tables = match raw_tables {
        Some(ref raw) => {
            let tables_env_name = if values.string(DATAFUSION_NEXUS_ICEBERG_TABLES_ENV).is_some() {
                DATAFUSION_NEXUS_ICEBERG_TABLES_ENV
            } else {
                DATAFUSION_NEXUS_ICEBERG_GLUE_TABLES_ENV
            };
            parse_iceberg_tables(tables_env_name, raw)?
        },
        None => vec![],
    };

    // Backend-neutral DATAFUSION_NEXUS_ICEBERG_CATALOG_NAME takes precedence;
    // fall back to the Glue-specific alias, then to the compiled-in default.
    let default_catalog_name = match kind {
        IcebergCatalogKind::Glue => DEFAULT_GLUE_CATALOG_NAME,
        IcebergCatalogKind::Rest => DEFAULT_REST_CATALOG_HANDLE,
    };
    let catalog_name = values
        .string(DATAFUSION_NEXUS_ICEBERG_CATALOG_NAME_ENV)
        .or_else(|| values.string(DATAFUSION_NEXUS_ICEBERG_GLUE_CATALOG_NAME_ENV))
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
fn table_format_parquet_read_strategy_config(
    values: &mut EnvValues<'_>,
) -> Result<TableFormatParquetReadStrategyConfig> {
    let preference = values
        .string(CUDF_ICEBERG_PARQUET_READ_STRATEGY_ENV)
        .map(|raw| parse_table_format_parquet_read_strategy(&raw))
        .transpose()?
        .unwrap_or_default();
    let remote_kvikio_experimental =
        values.optional_bool(CUDF_ICEBERG_REMOTE_KVIKIO_EXPERIMENTAL_ENV)?.unwrap_or(false);
    let lakehouse_cache_root = values.string(CUDF_LAKEHOUSE_CACHE_ROOT_ENV).map(PathBuf::from);
    let lakehouse_cache_policy = values
        .string(CUDF_LAKEHOUSE_CACHE_POLICY_ENV)
        .map(|raw| parse_lakehouse_cache_policy(&raw))
        .transpose()?
        .unwrap_or_default();
    let lakehouse_cache_max_bytes = match values.optional_u64(CUDF_LAKEHOUSE_CACHE_MAX_BYTES_ENV)? {
        Some(0) => {
            return Err(ServerError::invalid_config(CUDF_LAKEHOUSE_CACHE_MAX_BYTES_ENV, "must be greater than zero"));
        },
        Some(value) => Some(value),
        None => None,
    };
    let storage_endpoint = values
        .string(DATAFUSION_NEXUS_ICEBERG_S3_ENDPOINT_ENV)
        .or_else(|| values.string(AWS_ENDPOINT_URL_S3_ENV))
        .or_else(|| values.string(AWS_ENDPOINT_URL_ENV))
        .map(|endpoint| credential_free_endpoint_identity(&endpoint));

    Ok(TableFormatParquetReadStrategyConfig {
        preference,
        remote_kvikio_experimental,
        lakehouse_cache_root,
        lakehouse_cache_policy,
        lakehouse_cache_max_bytes,
        storage_endpoint,
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
            CUDF_ICEBERG_PARQUET_READ_STRATEGY_ENV,
            format!("must be `auto`, `host_buffer`, `local_cache`, or `remote_kvikio`, got `{other}`"),
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
            CUDF_LAKEHOUSE_CACHE_POLICY_ENV,
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
    let catalog_handle = values
        .string(DATAFUSION_NEXUS_ICEBERG_GLUE_CATALOG_NAME_ENV)
        .unwrap_or_else(|| DEFAULT_GLUE_CATALOG_NAME.to_string());

    Ok(GlueServerBackendConfig {
        catalog_handle,
        warehouse: warehouse.to_string(),
        region,
        catalog_id: values.string(DATAFUSION_NEXUS_ICEBERG_GLUE_CATALOG_ID_ENV),
        profile: values.string(crate::iceberg::AWS_PROFILE_ENV),
        uri: values.string(DATAFUSION_NEXUS_ICEBERG_GLUE_URI_ENV),
        s3_endpoint: values.string(DATAFUSION_NEXUS_ICEBERG_S3_ENDPOINT_ENV),
        s3_region: values.string(DATAFUSION_NEXUS_ICEBERG_S3_REGION_ENV),
        s3_path_style_access: values.optional_bool(DATAFUSION_NEXUS_ICEBERG_S3_PATH_STYLE_ENV)?,
    })
}

#[cfg(feature = "iceberg")]
fn rest_backend_config(
    values: &mut EnvValues<'_>,
    warehouse: String,
) -> Result<crate::iceberg::catalog::RestIcebergCatalogConfig> {
    let uri = values.required_string(DATAFUSION_NEXUS_ICEBERG_REST_URI_ENV)?;
    let catalog_handle = values
        .string(DATAFUSION_NEXUS_ICEBERG_REST_CATALOG_HANDLE_ENV)
        .unwrap_or_else(|| DEFAULT_REST_CATALOG_HANDLE.to_string());

    let mut config = crate::iceberg::catalog::RestIcebergCatalogConfig::new(catalog_handle, uri, warehouse);
    if let Some(prefix) = values.string(DATAFUSION_NEXUS_ICEBERG_REST_PREFIX_ENV) {
        config = config.with_prefix(prefix);
    }
    if let Some(token) = values.string(DATAFUSION_NEXUS_ICEBERG_REST_TOKEN_ENV) {
        config = config.with_token(token);
    }
    if let Some(endpoint) = values.string(DATAFUSION_NEXUS_ICEBERG_S3_ENDPOINT_ENV) {
        config = config.with_s3_endpoint(endpoint);
    }
    if let Some(region) = values.string(DATAFUSION_NEXUS_ICEBERG_S3_REGION_ENV) {
        config = config.with_s3_region(region);
    }
    if let Some(path_style) = values.optional_bool(DATAFUSION_NEXUS_ICEBERG_S3_PATH_STYLE_ENV)? {
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
        DATAFUSION_NEXUS_SERVER_CUGRAPH_ALGORITHMS_ENV,
        DATAFUSION_NEXUS_SERVER_CUGRAPH_DIRECTED_ENV,
        DATAFUSION_NEXUS_SERVER_CUGRAPH_RENUMBER_ENV,
        DATAFUSION_NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY_ENV,
    ]);
    let enabled = values.optional_bool(DATAFUSION_NEXUS_SERVER_CUGRAPH_ENABLED_ENV)?;
    if enabled != Some(true) {
        if requested {
            return Err(ServerError::invalid_config(
                DATAFUSION_NEXUS_SERVER_CUGRAPH_ENABLED_ENV,
                "must be true when cuGraph server settings are provided",
            ));
        }
        return Ok(None);
    }

    let allowed_algorithms = values
        .string(DATAFUSION_NEXUS_SERVER_CUGRAPH_ALGORITHMS_ENV)
        .map(|raw| parse_cugraph_algorithms(DATAFUSION_NEXUS_SERVER_CUGRAPH_ALGORITHMS_ENV, &raw))
        .transpose()?;
    let mut default_graph_options = CugraphGraphOptions::default();
    if let Some(directed) = values.optional_bool(DATAFUSION_NEXUS_SERVER_CUGRAPH_DIRECTED_ENV)? {
        default_graph_options.directed = directed;
    }
    if let Some(renumber) = values.optional_bool(DATAFUSION_NEXUS_SERVER_CUGRAPH_RENUMBER_ENV)? {
        default_graph_options.renumber = renumber;
    }
    if let Some(raw) = values.string(DATAFUSION_NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY_ENV) {
        default_graph_options.construction_policy =
            parse_cugraph_construction_policy(DATAFUSION_NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY_ENV, &raw)?;
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
        DATAFUSION_NEXUS_ICEBERG_CATALOG_KIND_ENV,
        DATAFUSION_NEXUS_ICEBERG_CATALOG_NAME_ENV,
        DATAFUSION_NEXUS_ICEBERG_NAMESPACE_ENV,
        DATAFUSION_NEXUS_ICEBERG_TABLES_ENV,
        DATAFUSION_NEXUS_ICEBERG_GLUE_CATALOG_NAME_ENV,
        DATAFUSION_NEXUS_ICEBERG_GLUE_DATABASE_ENV,
        DATAFUSION_NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV,
        DATAFUSION_NEXUS_ICEBERG_GLUE_CATALOG_ID_ENV,
        DATAFUSION_NEXUS_ICEBERG_GLUE_TABLES_ENV,
        DATAFUSION_NEXUS_ICEBERG_GLUE_URI_ENV,
        DATAFUSION_NEXUS_ICEBERG_REST_URI_ENV,
        DATAFUSION_NEXUS_ICEBERG_REST_PREFIX_ENV,
        DATAFUSION_NEXUS_ICEBERG_REST_TOKEN_ENV,
        DATAFUSION_NEXUS_ICEBERG_REST_CATALOG_HANDLE_ENV,
        DATAFUSION_NEXUS_ICEBERG_S3_ENDPOINT_ENV,
        DATAFUSION_NEXUS_ICEBERG_S3_REGION_ENV,
        DATAFUSION_NEXUS_ICEBERG_S3_PATH_STYLE_ENV,
        crate::CUDF_ICEBERG_PARQUET_READ_STRATEGY_ENV,
        crate::CUDF_ICEBERG_REMOTE_KVIKIO_EXPERIMENTAL_ENV,
        crate::CUDF_LAKEHOUSE_CACHE_ROOT_ENV,
        crate::CUDF_LAKEHOUSE_CACHE_POLICY_ENV,
        crate::CUDF_LAKEHOUSE_CACHE_MAX_BYTES_ENV,
    ]) {
        return Err(ServerError::invalid_config(
            DATAFUSION_NEXUS_ICEBERG_NAMESPACE_ENV,
            "Iceberg server registration requires building with --features server,iceberg",
        ));
    }
    Ok(())
}

#[cfg(not(feature = "cugraph"))]
fn reject_cugraph_env_without_feature(values: &mut EnvValues<'_>) -> Result<()> {
    if values.any_present([
        DATAFUSION_NEXUS_SERVER_CUGRAPH_ENABLED_ENV,
        DATAFUSION_NEXUS_SERVER_CUGRAPH_ALGORITHMS_ENV,
        DATAFUSION_NEXUS_SERVER_CUGRAPH_DIRECTED_ENV,
        DATAFUSION_NEXUS_SERVER_CUGRAPH_RENUMBER_ENV,
        DATAFUSION_NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY_ENV,
    ]) {
        return Err(ServerError::invalid_config(
            DATAFUSION_NEXUS_SERVER_CUGRAPH_ENABLED_ENV,
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
        let value = match self.optional_usize(name)? {
            Some(value) => value,
            None => default,
        };
        if value == 0 {
            return Err(ServerError::invalid_config(name, "must be greater than zero"));
        }
        Ok(value)
    }

    fn positive_u64(&mut self, name: &'static str, default: u64) -> Result<u64> {
        let value = match self.optional_u64(name)? {
            Some(value) => value,
            None => default,
        };
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
