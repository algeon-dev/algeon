use std::{collections::BTreeMap, net::SocketAddr, time::Duration};

use super::error::{Result, ServerError};
use crate::{CudfOptimizerConfig, NativeCudfExecutionConfig};

pub const DATAFUSION_CUDF_SERVER_BIND_ENV: &str = "DATAFUSION_CUDF_SERVER_BIND";
pub const DATAFUSION_CUDF_SERVER_LOG_ENV: &str = "DATAFUSION_CUDF_SERVER_LOG";
pub const DATAFUSION_CUDF_SERVER_MAX_IN_FLIGHT_ENV: &str = "DATAFUSION_CUDF_SERVER_MAX_IN_FLIGHT";
pub const DATAFUSION_CUDF_SERVER_STATEMENT_TICKET_TTL_SECS_ENV: &str =
    "DATAFUSION_CUDF_SERVER_STATEMENT_TICKET_TTL_SECS";
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
pub const DATAFUSION_CUDF_SERVER_SESSION_PROFILE_ENV: &str = "DATAFUSION_CUDF_SERVER_SESSION_PROFILE";
pub const DATAFUSION_CUDF_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV: &str =
    "DATAFUSION_CUDF_SERVER_NATIVE_DEVICE_BUDGET_BYTES";
pub const DATAFUSION_CUDF_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV: &str =
    "DATAFUSION_CUDF_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK";
pub const DATAFUSION_CUDF_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV: &str =
    "DATAFUSION_CUDF_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES";

// Backend-neutral Iceberg env vars (preferred for new deployments).
pub const DATAFUSION_CUDF_ICEBERG_CATALOG_NAME_ENV: &str = "DATAFUSION_CUDF_ICEBERG_CATALOG_NAME";
pub const DATAFUSION_CUDF_ICEBERG_NAMESPACE_ENV: &str = "DATAFUSION_CUDF_ICEBERG_NAMESPACE";
pub const DATAFUSION_CUDF_ICEBERG_TABLES_ENV: &str = "DATAFUSION_CUDF_ICEBERG_TABLES";

// Glue-specific env vars (retained for backward compatibility).
pub const DATAFUSION_CUDF_ICEBERG_GLUE_CATALOG_NAME_ENV: &str = "DATAFUSION_CUDF_ICEBERG_GLUE_CATALOG_NAME";
pub const DATAFUSION_CUDF_ICEBERG_GLUE_DATABASE_ENV: &str = "DATAFUSION_CUDF_ICEBERG_GLUE_DATABASE";
pub const DATAFUSION_CUDF_ICEBERG_GLUE_WAREHOUSE_ENV: &str = "DATAFUSION_CUDF_ICEBERG_WAREHOUSE";
pub const DATAFUSION_CUDF_ICEBERG_GLUE_CATALOG_ID_ENV: &str = "DATAFUSION_CUDF_ICEBERG_GLUE_CATALOG_ID";
pub const DATAFUSION_CUDF_ICEBERG_GLUE_TABLES_ENV: &str = "DATAFUSION_CUDF_ICEBERG_GLUE_TABLES";

const DEFAULT_BIND_ADDR: &str = "0.0.0.0:50051";
const DEFAULT_LOG_FILTER: &str = "info";
const DEFAULT_MAX_IN_FLIGHT: usize = 1;
const DEFAULT_STATEMENT_TICKET_TTL_SECS: u64 = 300;
#[cfg(feature = "iceberg")]
const DEFAULT_GLUE_CATALOG_NAME: &str = "glue";

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
    /// See [`DATAFUSION_CUDF_SERVER_SESSION_PROFILE_ENV`] and
    /// [`ServerSessionProfile`].
    pub session_profile: ServerSessionProfile,
    #[cfg(feature = "iceberg")]
    pub iceberg: Option<IcebergServerConfig>,
}

/// DataFusion physical-plan profile applied to the Flight SQL session.
///
/// Decisions tracked here only affect DataFusion's logical/physical plan shape
/// for queries served through Flight SQL. They are orthogonal to the native
/// cudf-query-engine memory budget, which is controlled by the
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
    /// matching DataFusion's `SessionConfig::from_env()` mapping). Suitable
    /// when the session only serves CPU-readable sources and benefits from
    /// DataFusion's parallel CPU exchanges.
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IcebergServerConfig {
    /// DataFusion catalog name (e.g. "iceberg" or "glue").  Exposed as the
    /// first part of `catalog.namespace.table` three-part addressing.
    pub catalog_name: String,
    /// Internal catalog handle passed to the iceberg-rust Glue catalog builder.
    /// Independent of `catalog_name` so users can set `DATAFUSION_CUDF_ICEBERG_CATALOG_NAME=iceberg`
    /// for Trino-style addressing without changing the Glue catalog identifier.
    /// Defaults to "glue"; controlled only by `DATAFUSION_CUDF_ICEBERG_GLUE_CATALOG_NAME`.
    pub glue_catalog_handle: String,
    /// Iceberg namespace / Glue database name (e.g. "demo_db"). This is the
    /// dynamic catalog visibility scope and the second part of three-part addressing.
    pub namespace: String,
    pub warehouse: String,
    pub catalog_id: Option<String>,
    pub region: String,
    pub profile: Option<String>,
    pub tables: Vec<IcebergTableRegistrationConfig>,
}

#[cfg(feature = "iceberg")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IcebergTableRegistrationConfig {
    pub table: String,
    pub alias: String,
}

impl ServerConfig {
    pub fn from_env() -> Result<Self> {
        Self::from_env_get(|name| std::env::var(name).ok())
    }

    pub fn from_env_get(mut get: impl FnMut(&str) -> Option<String>) -> Result<Self> {
        let mut values = EnvValues::new(&mut get);
        let bind_addr = values.socket_addr(DATAFUSION_CUDF_SERVER_BIND_ENV, DEFAULT_BIND_ADDR)?;
        let log_filter =
            values.string(DATAFUSION_CUDF_SERVER_LOG_ENV).unwrap_or_else(|| DEFAULT_LOG_FILTER.to_string());
        let max_in_flight = values.positive_usize(DATAFUSION_CUDF_SERVER_MAX_IN_FLIGHT_ENV, DEFAULT_MAX_IN_FLIGHT)?;
        let statement_ticket_ttl =
            Duration::from_secs(values.positive_u64(
                DATAFUSION_CUDF_SERVER_STATEMENT_TICKET_TTL_SECS_ENV,
                DEFAULT_STATEMENT_TICKET_TTL_SECS,
            )?);
        let native_optimizer_config = native_optimizer_config(&mut values)?;

        #[cfg(feature = "iceberg")]
        let iceberg = iceberg_config(&mut values)?;

        #[cfg(not(feature = "iceberg"))]
        reject_iceberg_env_without_feature(&mut values)?;

        let explicit_profile = values.optional_session_profile(DATAFUSION_CUDF_SERVER_SESSION_PROFILE_ENV)?;
        #[cfg(feature = "iceberg")]
        let default_profile =
            if iceberg.is_some() { ServerSessionProfile::Native } else { ServerSessionProfile::CpuFallback };
        #[cfg(not(feature = "iceberg"))]
        let default_profile = ServerSessionProfile::CpuFallback;
        let session_profile = explicit_profile.unwrap_or(default_profile);

        Ok(Self {
            bind_addr,
            log_filter,
            max_in_flight,
            statement_ticket_ttl,
            native_optimizer_config,
            session_profile,
            #[cfg(feature = "iceberg")]
            iceberg,
        })
    }
}

fn native_optimizer_config(values: &mut EnvValues<'_>) -> Result<CudfOptimizerConfig> {
    let device_budget = values.optional_u64(DATAFUSION_CUDF_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV)?;
    let max_source_chunk = values.optional_u64(DATAFUSION_CUDF_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV)?;
    let max_row_groups = values.optional_usize(DATAFUSION_CUDF_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV)?;

    let execution_config = match (device_budget, max_source_chunk, max_row_groups) {
        (None, None, None) => NativeCudfExecutionConfig::unbounded(),
        (Some(device_budget), Some(max_source_chunk), Some(max_row_groups)) => {
            if device_budget == 0 {
                return Err(ServerError::invalid_config(
                    DATAFUSION_CUDF_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV,
                    "must be greater than zero",
                ));
            }
            if max_source_chunk == 0 {
                return Err(ServerError::invalid_config(
                    DATAFUSION_CUDF_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV,
                    "must be greater than zero",
                ));
            }
            if max_row_groups == 0 {
                return Err(ServerError::invalid_config(
                    DATAFUSION_CUDF_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV,
                    "must be greater than zero",
                ));
            }
            NativeCudfExecutionConfig::bounded(device_budget, max_source_chunk, max_row_groups, 1, 1)
        },
        _ => {
            return Err(ServerError::invalid_config(
                DATAFUSION_CUDF_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV,
                "bounded native execution requires device budget, max source chunk bytes, and max row groups per chunk",
            ));
        },
    };

    Ok(CudfOptimizerConfig::default().with_native_execution_config(execution_config))
}

#[cfg(feature = "iceberg")]
fn iceberg_config(values: &mut EnvValues<'_>) -> Result<Option<IcebergServerConfig>> {
    let requested = values.any_present([
        DATAFUSION_CUDF_ICEBERG_CATALOG_NAME_ENV,
        DATAFUSION_CUDF_ICEBERG_NAMESPACE_ENV,
        DATAFUSION_CUDF_ICEBERG_TABLES_ENV,
        DATAFUSION_CUDF_ICEBERG_GLUE_CATALOG_NAME_ENV,
        DATAFUSION_CUDF_ICEBERG_GLUE_DATABASE_ENV,
        DATAFUSION_CUDF_ICEBERG_GLUE_WAREHOUSE_ENV,
        DATAFUSION_CUDF_ICEBERG_GLUE_CATALOG_ID_ENV,
        DATAFUSION_CUDF_ICEBERG_GLUE_TABLES_ENV,
    ]);
    if !requested {
        return Ok(None);
    }

    // Backend-neutral DATAFUSION_CUDF_ICEBERG_NAMESPACE takes precedence; fall
    // back to the Glue-specific DATAFUSION_CUDF_ICEBERG_GLUE_DATABASE alias.
    let namespace = values
        .string(DATAFUSION_CUDF_ICEBERG_NAMESPACE_ENV)
        .or_else(|| values.string(DATAFUSION_CUDF_ICEBERG_GLUE_DATABASE_ENV))
        .ok_or_else(|| {
            ServerError::invalid_config(
                DATAFUSION_CUDF_ICEBERG_NAMESPACE_ENV,
                "is required (also accepted as DATAFUSION_CUDF_ICEBERG_GLUE_DATABASE)",
            )
        })?;

    let warehouse = values.required_string(DATAFUSION_CUDF_ICEBERG_GLUE_WAREHOUSE_ENV)?;
    let region = values
        .string(crate::iceberg::AWS_REGION_ENV)
        .or_else(|| values.string(crate::iceberg::AWS_DEFAULT_REGION_ENV))
        .ok_or_else(|| ServerError::invalid_config(crate::iceberg::AWS_REGION_ENV, "is required for Glue Iceberg"))?;

    // Backend-neutral DATAFUSION_CUDF_ICEBERG_TABLES takes precedence; fall
    // back to the Glue-specific DATAFUSION_CUDF_ICEBERG_GLUE_TABLES alias.
    // When neither is set the dynamic catalog serves all tables and aliases
    // are not required (empty vec is valid in Phase 2).
    let raw_tables = values
        .string(DATAFUSION_CUDF_ICEBERG_TABLES_ENV)
        .or_else(|| values.string(DATAFUSION_CUDF_ICEBERG_GLUE_TABLES_ENV));
    let tables = match raw_tables {
        Some(ref raw) => {
            let tables_env_name = if values.string(DATAFUSION_CUDF_ICEBERG_TABLES_ENV).is_some() {
                DATAFUSION_CUDF_ICEBERG_TABLES_ENV
            } else {
                DATAFUSION_CUDF_ICEBERG_GLUE_TABLES_ENV
            };
            parse_iceberg_tables(tables_env_name, raw)?
        },
        None => vec![],
    };

    // Backend-neutral DATAFUSION_CUDF_ICEBERG_CATALOG_NAME takes precedence;
    // fall back to the Glue-specific alias, then to the compiled-in default.
    let catalog_name = values
        .string(DATAFUSION_CUDF_ICEBERG_CATALOG_NAME_ENV)
        .or_else(|| values.string(DATAFUSION_CUDF_ICEBERG_GLUE_CATALOG_NAME_ENV))
        .unwrap_or_else(|| DEFAULT_GLUE_CATALOG_NAME.to_string());

    // glue_catalog_handle is the internal handle passed to iceberg-rust's
    // GlueCatalogBuilder; it is independent of the DataFusion catalog_name so
    // users can choose a different connector name without changing the Glue
    // catalog identifier.  Controlled only by the Glue-specific env var.
    let glue_catalog_handle = values
        .string(DATAFUSION_CUDF_ICEBERG_GLUE_CATALOG_NAME_ENV)
        .unwrap_or_else(|| DEFAULT_GLUE_CATALOG_NAME.to_string());

    Ok(Some(IcebergServerConfig {
        catalog_name,
        glue_catalog_handle,
        namespace,
        warehouse,
        catalog_id: values.string(DATAFUSION_CUDF_ICEBERG_GLUE_CATALOG_ID_ENV),
        region,
        profile: values.string(crate::iceberg::AWS_PROFILE_ENV),
        tables,
    }))
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

#[cfg(not(feature = "iceberg"))]
fn reject_iceberg_env_without_feature(values: &mut EnvValues<'_>) -> Result<()> {
    if values.any_present([
        DATAFUSION_CUDF_ICEBERG_CATALOG_NAME_ENV,
        DATAFUSION_CUDF_ICEBERG_NAMESPACE_ENV,
        DATAFUSION_CUDF_ICEBERG_TABLES_ENV,
        DATAFUSION_CUDF_ICEBERG_GLUE_CATALOG_NAME_ENV,
        DATAFUSION_CUDF_ICEBERG_GLUE_DATABASE_ENV,
        DATAFUSION_CUDF_ICEBERG_GLUE_WAREHOUSE_ENV,
        DATAFUSION_CUDF_ICEBERG_GLUE_CATALOG_ID_ENV,
        DATAFUSION_CUDF_ICEBERG_GLUE_TABLES_ENV,
    ]) {
        return Err(ServerError::invalid_config(
            DATAFUSION_CUDF_ICEBERG_NAMESPACE_ENV,
            "Iceberg server registration requires building with --features server,iceberg",
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
}
