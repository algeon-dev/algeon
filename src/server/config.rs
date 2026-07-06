#[cfg(feature = "iceberg")]
use std::fmt;
use std::{collections::BTreeMap, fs, net::SocketAddr, path::PathBuf, time::Duration};

use nexus_query_engine::{
    MemoryControlPolicy, NativePipelineExecutionMode, SourceChunkPlanningMode, SourceSchedulerPolicy,
    StatisticalAggregateExecutionMode,
};
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
/// Maximum number of Flight SQL statements admitted at once. Defaults to `2`;
/// set a larger value only when the GPU has enough memory for concurrent work.
pub const NEXUS_SERVER_MAX_IN_FLIGHT_ENV: &str = "NEXUS_SERVER_MAX_IN_FLIGHT";
/// Preferred maximum number of Flight SQL statements admitted at once. Defaults
/// to `2`; `NEXUS_SERVER_MAX_IN_FLIGHT` remains accepted as a compatibility
/// alias.
pub const NEXUS_SERVER_MAX_CONCURRENT_QUERIES_ENV: &str = "NEXUS_SERVER_MAX_CONCURRENT_QUERIES";
/// Fraction of a GPU's total memory eligible for server native-query budgets.
/// The fixed native headroom remains a floor, so `1.0` preserves the historical
/// auto-sized `total - headroom` behavior.
pub const NEXUS_SERVER_DEVICE_BUDGET_FRACTION_ENV: &str = "NEXUS_SERVER_DEVICE_BUDGET_FRACTION";
/// Comma-separated CUDA device ordinals available to the Flight SQL server.
/// Independent queries are placed across this set by the native query service.
pub const NEXUS_SERVER_GPU_DEVICES_ENV: &str = "NEXUS_SERVER_GPU_DEVICES";
/// Seconds an unconsumed Flight SQL statement ticket may hold admission before
/// it expires and releases its slot. Defaults to `300`.
pub const NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS_ENV: &str = "NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS";
/// DataFusion physical-plan profile for Flight SQL sessions. Use `native` for
/// GPU-native plan shapes or `cpu_fallback` to keep DataFusion's CPU-oriented
/// execution settings; the default is `native` when Iceberg is configured and
/// `cpu_fallback` otherwise.
pub const NEXUS_SERVER_SESSION_PROFILE_ENV: &str = "NEXUS_SERVER_SESSION_PROFILE";
/// DataFusion catalog name that hosts the mutable session workspace overlay.
/// Defaults to `datafusion` when a backing source schema is configured.
pub const NEXUS_SERVER_WORKSPACE_CATALOG_ENV: &str = "NEXUS_SERVER_WORKSPACE_CATALOG";
/// DataFusion schema name that hosts the mutable session workspace overlay.
/// Defaults to `public` when a backing source schema is configured.
pub const NEXUS_SERVER_WORKSPACE_SCHEMA_ENV: &str = "NEXUS_SERVER_WORKSPACE_SCHEMA";
/// Source catalog that the mutable workspace overlay can read through.
pub const NEXUS_SERVER_WORKSPACE_BACKING_CATALOG_ENV: &str = "NEXUS_SERVER_WORKSPACE_BACKING_CATALOG";
/// Source schema that the mutable workspace overlay can read through.
pub const NEXUS_SERVER_WORKSPACE_BACKING_SCHEMA_ENV: &str = "NEXUS_SERVER_WORKSPACE_BACKING_SCHEMA";
/// Optional comma-separated source names exposed through the workspace overlay.
/// When unset, all backing schema table names are visible through the overlay.
pub const NEXUS_SERVER_WORKSPACE_BACKING_ALIASES_ENV: &str = "NEXUS_SERVER_WORKSPACE_BACKING_ALIASES";
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
/// Maximum bounded retry attempts for source splitting, final merge, and final
/// sort retry paths. Defaults to the native library bounded default.
pub const NEXUS_SERVER_NATIVE_MAX_RETRY_ATTEMPTS_ENV: &str = "NEXUS_SERVER_NATIVE_MAX_RETRY_ATTEMPTS";
/// Row-group count at or below which bounded source retry splitting stops.
/// Defaults to the native library bounded default.
pub const NEXUS_SERVER_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV: &str = "NEXUS_SERVER_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS";
/// Maximum active native source reads allowed per bounded query. Values above
/// `1` allow source prefetch overlap when the plan shape supports it.
pub const NEXUS_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT_ENV: &str = "NEXUS_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT";
/// Bounded source chunk planning mode. Accepts `conservative` or
/// `local_parquet_throughput`.
pub const NEXUS_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV: &str = "NEXUS_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE";
/// Seconds a native object-store source read may wait for host-buffer budget
/// or source-fetch concurrency before returning a resource timeout. This
/// service-level source IO setting applies in both bounded and unbounded mode.
pub const NEXUS_SERVER_NATIVE_HOST_FETCH_ACQUIRE_TIMEOUT_SECS_ENV: &str =
    "NEXUS_SERVER_NATIVE_HOST_FETCH_ACQUIRE_TIMEOUT_SECS";
/// Native bounded-pipeline execution mode. Accepts `strict_bounded` or
/// `adaptive`.
pub const NEXUS_SERVER_NATIVE_PIPELINE_EXECUTION_MODE_ENV: &str = "NEXUS_SERVER_NATIVE_PIPELINE_EXECUTION_MODE";
/// Floating statistical aggregate execution contract. Accepts
/// `datafusion_exact_host_welford` or `gpu_native_tolerant`.
pub const NEXUS_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV: &str =
    "NEXUS_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE";
/// Path to a TOML server config file. The file supplies durable server config
/// such as `[auth]` and `[native]`; scalar env vars remain bootstrap overrides.
pub const NEXUS_SERVER_CONFIG_FILE_ENV: &str = "NEXUS_SERVER_CONFIG_FILE";

/// Scope for a server `[native]` configuration option.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ServerNativeConfigScope {
    /// Native execution setting that applies in both bounded and unbounded mode.
    NativeExecution,
    /// Bounded execution setting rejected when native memory policy is unbounded.
    BoundedExecution,
    /// Service-level source IO resource setting, not session-scoped.
    SourceIoResource,
}

impl ServerNativeConfigScope {
    /// Whether this option is active only under bounded native memory policy.
    pub const fn is_bounded_only(self) -> bool {
        matches!(self, Self::BoundedExecution)
    }
}

/// Discoverable metadata for one server `[native]` TOML/env option.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct NexusServerNativeConfigOption {
    /// TOML field including the `[native]` section prefix, for example
    /// `native.memory_policy`.
    pub toml_field: &'static str,
    /// Environment variable that overrides the TOML field.
    pub env_var: &'static str,
    /// Human-readable value type used by docs and operator surfaces.
    pub value_type: &'static str,
    /// Server-side scope for this option.
    pub scope: ServerNativeConfigScope,
    /// Matching DataFusion session key when this is also session-scoped.
    pub session_key: Option<&'static str>,
    /// Human-readable operator description.
    pub description: &'static str,
}

/// Server `[native]` options exposed through TOML and environment variables.
///
/// This table is the programmatic discovery surface for operator docs and
/// config introspection. Parser code remains explicit so validation and
/// bounded/unbounded interactions stay local to `native_optimizer_config`.
pub const NEXUS_SERVER_NATIVE_CONFIG_OPTIONS: &[NexusServerNativeConfigOption] = &[
    NexusServerNativeConfigOption {
        toml_field: "native.memory_policy",
        env_var: NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV,
        value_type: "enum: bounded|unbounded",
        scope: ServerNativeConfigScope::NativeExecution,
        session_key: Some("datafusion_nexus.native.memory_policy"),
        description: "Native memory-control mode: bounded or unbounded.",
    },
    NexusServerNativeConfigOption {
        toml_field: "native.device_budget_bytes",
        env_var: NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV,
        value_type: "positive integer bytes",
        scope: ServerNativeConfigScope::BoundedExecution,
        session_key: Some("datafusion_nexus.native.device_budget_bytes"),
        description: "Bounded native device-memory budget in bytes.",
    },
    NexusServerNativeConfigOption {
        toml_field: "native.max_source_chunk_bytes",
        env_var: NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV,
        value_type: "positive integer bytes",
        scope: ServerNativeConfigScope::BoundedExecution,
        session_key: Some("datafusion_nexus.native.max_source_chunk_bytes"),
        description: "Maximum source bytes planned into one native bounded source chunk.",
    },
    NexusServerNativeConfigOption {
        toml_field: "native.max_row_groups_per_chunk",
        env_var: NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV,
        value_type: "positive integer",
        scope: ServerNativeConfigScope::BoundedExecution,
        session_key: Some("datafusion_nexus.native.max_row_groups_per_chunk"),
        description: "Maximum source row groups planned into one native bounded source chunk.",
    },
    NexusServerNativeConfigOption {
        toml_field: "native.max_retry_attempts",
        env_var: NEXUS_SERVER_NATIVE_MAX_RETRY_ATTEMPTS_ENV,
        value_type: "positive integer",
        scope: ServerNativeConfigScope::BoundedExecution,
        session_key: Some("datafusion_nexus.native.max_retry_attempts"),
        description: "Maximum bounded retry attempts.",
    },
    NexusServerNativeConfigOption {
        toml_field: "native.min_retry_chunk_row_groups",
        env_var: NEXUS_SERVER_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV,
        value_type: "positive integer",
        scope: ServerNativeConfigScope::BoundedExecution,
        session_key: Some("datafusion_nexus.native.min_retry_chunk_row_groups"),
        description: "Row-group count at or below which bounded source retry splitting stops.",
    },
    NexusServerNativeConfigOption {
        toml_field: "native.max_source_reads_in_flight",
        env_var: NEXUS_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT_ENV,
        value_type: "positive integer",
        scope: ServerNativeConfigScope::BoundedExecution,
        session_key: Some("datafusion_nexus.native.max_source_reads_in_flight"),
        description: "Maximum active native source reads allowed per bounded query.",
    },
    NexusServerNativeConfigOption {
        toml_field: "native.source_chunk_planning_mode",
        env_var: NEXUS_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV,
        value_type: "enum: conservative|local_parquet_throughput",
        scope: ServerNativeConfigScope::BoundedExecution,
        session_key: Some("datafusion_nexus.native.source_chunk_planning_mode"),
        description: "Bounded source chunk planning mode.",
    },
    NexusServerNativeConfigOption {
        toml_field: "native.host_fetch_acquire_timeout_secs",
        env_var: NEXUS_SERVER_NATIVE_HOST_FETCH_ACQUIRE_TIMEOUT_SECS_ENV,
        value_type: "positive integer seconds",
        scope: ServerNativeConfigScope::SourceIoResource,
        session_key: None,
        description: "Service-level host-buffer budget and source-fetch concurrency wait timeout.",
    },
    NexusServerNativeConfigOption {
        toml_field: "native.pipeline_execution_mode",
        env_var: NEXUS_SERVER_NATIVE_PIPELINE_EXECUTION_MODE_ENV,
        value_type: "enum: strict_bounded|adaptive",
        scope: ServerNativeConfigScope::BoundedExecution,
        session_key: Some("datafusion_nexus.native.pipeline_execution_mode"),
        description: "Native bounded-pipeline execution mode.",
    },
    NexusServerNativeConfigOption {
        toml_field: "native.statistical_aggregate_execution_mode",
        env_var: NEXUS_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV,
        value_type: "enum: datafusion_exact_host_welford|gpu_native_tolerant",
        scope: ServerNativeConfigScope::NativeExecution,
        session_key: Some("datafusion_nexus.native.statistical_aggregate_execution_mode"),
        description: "Floating statistical aggregate execution contract.",
    },
];

/// Enables registration of cuGraph SQL functions. Must be `true` before any
/// other cuGraph server setting is accepted. On a build without the `cugraph`
/// feature an explicit `false` is accepted (it requests no capability), so one
/// env file can serve both artifact flavors; `true` and every other cuGraph
/// setting are still rejected there.
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
/// Explicit S3 credential source for Iceberg object storage. The only accepted
/// value is `default_chain`: an operator opt-in to the ambient AWS default
/// credential chain (env vars, profile, IMDS instance role), including with a
/// custom `NEXUS_ICEBERG_S3_ENDPOINT` where the default is otherwise unsigned
/// requests. Mutually exclusive with static `NEXUS_ICEBERG_S3_*` credentials.
pub const NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV: &str = "NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE";
/// Seconds allowed for Iceberg table-format scan planning before Flight SQL
/// returns a structured `planning_timeout` error. Defaults to `60`.
pub const NEXUS_ICEBERG_SCAN_PLANNING_TIMEOUT_SECS_ENV: &str = "NEXUS_ICEBERG_SCAN_PLANNING_TIMEOUT_SECS";

const DEFAULT_BIND_ADDR: &str = "0.0.0.0:50051";
const DEFAULT_LOG_FILTER: &str = "info";
const DEFAULT_MAX_IN_FLIGHT: usize = 2;
const DEFAULT_STATEMENT_TICKET_TTL_SECS: u64 = 300;
#[cfg(feature = "iceberg")]
const DEFAULT_GLUE_CATALOG_NAME: &str = "glue";
#[cfg(feature = "iceberg")]
const DEFAULT_REST_CATALOG_HANDLE: &str = "rest";
#[cfg(feature = "iceberg")]
const DEFAULT_ICEBERG_SCAN_PLANNING_TIMEOUT_SECS: u64 = 60;

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub bind_addr: SocketAddr,
    pub log_filter: String,
    pub max_in_flight: usize,
    pub gpu_devices: Vec<usize>,
    pub device_budget_fraction: Option<f64>,
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
    /// Optional mutable workspace overlay that can read through to a
    /// configured source schema while keeping all DDL local to the workspace.
    pub workspace: Option<WorkspaceOverlayConfig>,
    #[cfg(feature = "iceberg")]
    pub iceberg: Option<IcebergServerConfig>,
    #[cfg(feature = "cugraph")]
    pub cugraph: Option<CugraphServerConfig>,
}

/// Raw shape of the optional TOML overlay loaded from
/// [`NEXUS_SERVER_CONFIG_FILE_ENV`].
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields, default)]
struct ServerFileOverlay {
    auth: AuthConfig,
    native: ServerNativeFileConfig,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields, default)]
struct ServerNativeFileConfig {
    memory_policy: Option<String>,
    device_budget_bytes: Option<u64>,
    max_source_chunk_bytes: Option<u64>,
    max_row_groups_per_chunk: Option<usize>,
    max_retry_attempts: Option<usize>,
    min_retry_chunk_row_groups: Option<usize>,
    max_source_reads_in_flight: Option<usize>,
    source_chunk_planning_mode: Option<String>,
    host_fetch_acquire_timeout_secs: Option<u64>,
    pipeline_execution_mode: Option<String>,
    statistical_aggregate_execution_mode: Option<String>,
}

#[derive(Debug, Clone, Copy)]
struct NativeConfigValue<T> {
    value: T,
    field: &'static str,
}

impl<T> NativeConfigValue<T> {
    const fn new(value: T, field: &'static str) -> Self {
        Self { value, field }
    }
}

/// DataFusion physical-plan profile applied to the Flight SQL session.
///
/// Decisions tracked here only affect DataFusion's logical/physical plan shape
/// for queries served through Flight SQL. They are orthogonal to the native
/// nexus-query-engine memory budget, which is controlled by the server native
/// config (`[native]` TOML or `NEXUS_SERVER_NATIVE_*` environment variables).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerSessionProfile {
    /// Native-friendly profile that matches the workload harness defaults:
    /// `target_partitions=1`, repartitioning disabled for file scans /
    /// aggregations / joins / sorts, and all dynamic-filter-pushdown variants
    /// disabled. Required for GPU-only table-format leaves.
    Native,
    /// Preserve DataFusion's CPU-style profile derived from
    /// `DATAFUSION_EXECUTION_*` environment variables (single underscores,
    /// matching DataFusion's `SessionConfig::from_env()` mapping).
    ///
    /// For Iceberg, this requires the `iceberg-cpu` feature; otherwise the
    /// catalog still exposes GPU-only leaves and planning can fail. Non-Iceberg
    /// CPU sessions do not need that feature.
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceOverlayConfig {
    pub catalog: String,
    pub schema: String,
    pub backing_catalog: String,
    pub backing_schema: String,
    /// `None` means every table reported by the backing schema is visible.
    pub backing_aliases: Option<Vec<String>>,
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
    /// Wall-clock budget applied to Iceberg scan planning.
    pub scan_planning_timeout: Duration,
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
            .field("scan_planning_timeout", &self.scan_planning_timeout)
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
    Rest(crate::table_format::iceberg::catalog::RestIcebergCatalogConfig),
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
        let admission_limit_explicit =
            values.any_present([NEXUS_SERVER_MAX_CONCURRENT_QUERIES_ENV, NEXUS_SERVER_MAX_IN_FLIGHT_ENV]);
        let max_concurrent_queries = server_max_concurrent_queries(&mut values)?;
        let mut max_in_flight = max_concurrent_queries.value;
        let gpu_devices = server_gpu_devices(&mut values)?;
        let device_budget_fraction = values.optional_f64(NEXUS_SERVER_DEVICE_BUDGET_FRACTION_ENV)?;
        let statement_ticket_ttl = Duration::from_secs(
            values.positive_u64(NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS_ENV, DEFAULT_STATEMENT_TICKET_TTL_SECS)?,
        );
        let overlay_path = values.string(NEXUS_SERVER_CONFIG_FILE_ENV).map(PathBuf::from);
        let overlay = match overlay_path {
            Some(path) => load_overlay(&path)?,
            None => ServerFileOverlay::default(),
        };
        let native_optimizer_config =
            native_optimizer_config(&mut values, &overlay.native, &gpu_devices, device_budget_fraction)?;
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

        let workspace = workspace_overlay_config(&mut values)?;

        let explicit_profile = values.optional_session_profile(NEXUS_SERVER_SESSION_PROFILE_ENV)?;
        #[cfg(feature = "iceberg")]
        let default_profile =
            if iceberg.is_some() { ServerSessionProfile::Native } else { ServerSessionProfile::CpuFallback };
        #[cfg(not(feature = "iceberg"))]
        let default_profile = ServerSessionProfile::CpuFallback;
        let session_profile = explicit_profile.unwrap_or(default_profile);

        overlay.auth.mode.validate()?;
        if matches!(
            native_optimizer_config.native_execution_config().memory_control_policy(),
            MemoryControlPolicy::Unbounded
        ) && max_in_flight > 1
        {
            if admission_limit_explicit {
                return Err(ServerError::invalid_config(
                    max_concurrent_queries.field,
                    "requires bounded native memory policy when greater than one",
                ));
            }
            max_in_flight = 1;
        }

        Ok(Self {
            bind_addr,
            log_filter,
            max_in_flight,
            gpu_devices,
            device_budget_fraction,
            statement_ticket_ttl,
            native_optimizer_config,
            session_profile,
            auth: overlay.auth,
            workspace,
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

fn server_max_concurrent_queries(values: &mut EnvValues<'_>) -> Result<NativeConfigValue<usize>> {
    let preferred = values
        .optional_usize(NEXUS_SERVER_MAX_CONCURRENT_QUERIES_ENV)?
        .map(|value| NativeConfigValue::new(value, NEXUS_SERVER_MAX_CONCURRENT_QUERIES_ENV));
    let legacy = values
        .optional_usize(NEXUS_SERVER_MAX_IN_FLIGHT_ENV)?
        .map(|value| NativeConfigValue::new(value, NEXUS_SERVER_MAX_IN_FLIGHT_ENV));
    let selected = match (preferred, legacy) {
        (Some(preferred), Some(legacy)) if preferred.value != legacy.value => {
            return Err(ServerError::invalid_config(
                NEXUS_SERVER_MAX_CONCURRENT_QUERIES_ENV,
                format!(
                    "conflicts with {NEXUS_SERVER_MAX_IN_FLIGHT_ENV}; set only one admission limit or use the same value"
                ),
            ));
        },
        (Some(preferred), _) => preferred,
        (None, Some(legacy)) => legacy,
        (None, None) => NativeConfigValue::new(DEFAULT_MAX_IN_FLIGHT, NEXUS_SERVER_MAX_CONCURRENT_QUERIES_ENV),
    };
    if selected.value == 0 {
        return Err(ServerError::invalid_config(selected.field, "must be greater than zero"));
    }
    Ok(selected)
}

fn server_gpu_devices(values: &mut EnvValues<'_>) -> Result<Vec<usize>> {
    let Some(raw) = values.string(NEXUS_SERVER_GPU_DEVICES_ENV) else {
        return Ok(vec![0]);
    };
    let mut devices = Vec::new();
    for part in raw.split(',') {
        let ordinal = part.trim().parse::<usize>().map_err(|error| {
            ServerError::invalid_config(NEXUS_SERVER_GPU_DEVICES_ENV, format!("must contain integer ordinals: {error}"))
        })?;
        if devices.contains(&ordinal) {
            return Err(ServerError::invalid_config(NEXUS_SERVER_GPU_DEVICES_ENV, "must not contain duplicates"));
        }
        devices.push(ordinal);
    }
    Ok(devices)
}

fn workspace_overlay_config(values: &mut EnvValues<'_>) -> Result<Option<WorkspaceOverlayConfig>> {
    let requested = values.any_present([
        NEXUS_SERVER_WORKSPACE_CATALOG_ENV,
        NEXUS_SERVER_WORKSPACE_SCHEMA_ENV,
        NEXUS_SERVER_WORKSPACE_BACKING_CATALOG_ENV,
        NEXUS_SERVER_WORKSPACE_BACKING_SCHEMA_ENV,
        NEXUS_SERVER_WORKSPACE_BACKING_ALIASES_ENV,
    ]);
    if !requested {
        return Ok(None);
    }

    let backing_catalog = values.string(NEXUS_SERVER_WORKSPACE_BACKING_CATALOG_ENV);
    let backing_schema = values.string(NEXUS_SERVER_WORKSPACE_BACKING_SCHEMA_ENV);
    let (backing_catalog, backing_schema) = match (backing_catalog, backing_schema) {
        (Some(catalog), Some(schema)) => (catalog, schema),
        (None, None) => {
            return Err(ServerError::invalid_config(
                NEXUS_SERVER_WORKSPACE_BACKING_CATALOG_ENV,
                format!("requires {NEXUS_SERVER_WORKSPACE_BACKING_SCHEMA_ENV} when workspace overlay is configured"),
            ));
        },
        (None, Some(_)) => {
            return Err(ServerError::invalid_config(
                NEXUS_SERVER_WORKSPACE_BACKING_CATALOG_ENV,
                format!("is required when {NEXUS_SERVER_WORKSPACE_BACKING_SCHEMA_ENV} is set"),
            ));
        },
        (Some(_), None) => {
            return Err(ServerError::invalid_config(
                NEXUS_SERVER_WORKSPACE_BACKING_SCHEMA_ENV,
                format!("is required when {NEXUS_SERVER_WORKSPACE_BACKING_CATALOG_ENV} is set"),
            ));
        },
    };

    let catalog = values.string(NEXUS_SERVER_WORKSPACE_CATALOG_ENV).unwrap_or_else(|| "datafusion".to_string());
    let schema = values.string(NEXUS_SERVER_WORKSPACE_SCHEMA_ENV).unwrap_or_else(|| "public".to_string());
    if catalog == backing_catalog && schema == backing_schema {
        return Err(ServerError::invalid_config(
            NEXUS_SERVER_WORKSPACE_CATALOG_ENV,
            "workspace catalog/schema must differ from backing catalog/schema",
        ));
    }

    let backing_aliases = values
        .string(NEXUS_SERVER_WORKSPACE_BACKING_ALIASES_ENV)
        .map(|raw| parse_workspace_aliases(NEXUS_SERVER_WORKSPACE_BACKING_ALIASES_ENV, &raw))
        .transpose()?;

    Ok(Some(WorkspaceOverlayConfig { catalog, schema, backing_catalog, backing_schema, backing_aliases }))
}

fn parse_workspace_aliases(env_name: &'static str, raw: &str) -> Result<Vec<String>> {
    let mut aliases = Vec::new();
    for alias in raw.split(',').map(str::trim).filter(|alias| !alias.is_empty()) {
        if aliases.iter().any(|existing| existing == alias) {
            return Err(ServerError::invalid_config(env_name, format!("contains duplicate alias `{alias}`")));
        }
        aliases.push(alias.to_string());
    }
    if aliases.is_empty() {
        return Err(ServerError::invalid_config(env_name, "must contain at least one alias"));
    }
    Ok(aliases)
}

fn native_optimizer_config(
    values: &mut EnvValues<'_>,
    file: &ServerNativeFileConfig,
    gpu_devices: &[usize],
    device_budget_fraction: Option<f64>,
) -> Result<CudfOptimizerConfig> {
    let memory_policy = optional_native_string(
        values,
        NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV,
        "native.memory_policy",
        file.memory_policy.as_deref(),
    )
    .map(|raw| {
        parse_native_memory_policy(raw.field, &raw.value)
            .map(|memory_policy| NativeConfigValue::new(memory_policy, raw.field))
    })
    .transpose()?
    .unwrap_or(NativeConfigValue::new(ServerNativeMemoryPolicy::Bounded, NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV));
    let env_forces_unbounded = matches!(memory_policy.value, ServerNativeMemoryPolicy::Unbounded)
        && memory_policy.field == NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV;
    let device_budget = optional_native_u64(
        values,
        NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV,
        "native.device_budget_bytes",
        if env_forces_unbounded { None } else { file.device_budget_bytes },
    )?;
    if device_budget_fraction.is_some() && device_budget.is_some() {
        return Err(ServerError::invalid_config(
            NEXUS_SERVER_DEVICE_BUDGET_FRACTION_ENV,
            format!("cannot be combined with {NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV}"),
        ));
    }
    let max_source_chunk = optional_native_u64(
        values,
        NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV,
        "native.max_source_chunk_bytes",
        if env_forces_unbounded { None } else { file.max_source_chunk_bytes },
    )?;
    let max_row_groups = optional_native_usize(
        values,
        NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV,
        "native.max_row_groups_per_chunk",
        if env_forces_unbounded { None } else { file.max_row_groups_per_chunk },
    )?;
    let max_retry_attempts = optional_native_usize(
        values,
        NEXUS_SERVER_NATIVE_MAX_RETRY_ATTEMPTS_ENV,
        "native.max_retry_attempts",
        if env_forces_unbounded { None } else { file.max_retry_attempts },
    )?;
    let min_retry_chunk_row_groups = optional_native_usize(
        values,
        NEXUS_SERVER_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV,
        "native.min_retry_chunk_row_groups",
        if env_forces_unbounded { None } else { file.min_retry_chunk_row_groups },
    )?;
    let max_source_reads_in_flight = optional_native_usize(
        values,
        NEXUS_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT_ENV,
        "native.max_source_reads_in_flight",
        if env_forces_unbounded { None } else { file.max_source_reads_in_flight },
    )?;
    let source_chunk_planning_mode = optional_native_string(
        values,
        NEXUS_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV,
        "native.source_chunk_planning_mode",
        if env_forces_unbounded { None } else { file.source_chunk_planning_mode.as_deref() },
    )
    .map(|raw| {
        parse_source_chunk_planning_mode(raw.field, &raw.value).map(|mode| NativeConfigValue::new(mode, raw.field))
    })
    .transpose()?;
    let host_fetch_acquire_timeout_secs = optional_native_u64(
        values,
        NEXUS_SERVER_NATIVE_HOST_FETCH_ACQUIRE_TIMEOUT_SECS_ENV,
        "native.host_fetch_acquire_timeout_secs",
        file.host_fetch_acquire_timeout_secs,
    )?;
    let native_pipeline_execution_mode = optional_native_string(
        values,
        NEXUS_SERVER_NATIVE_PIPELINE_EXECUTION_MODE_ENV,
        "native.pipeline_execution_mode",
        if env_forces_unbounded { None } else { file.pipeline_execution_mode.as_deref() },
    )
    .map(|raw| {
        parse_native_pipeline_execution_mode(raw.field, &raw.value).map(|mode| NativeConfigValue::new(mode, raw.field))
    })
    .transpose()?;
    let statistical_aggregate_execution_mode = optional_native_string(
        values,
        NEXUS_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV,
        "native.statistical_aggregate_execution_mode",
        file.statistical_aggregate_execution_mode.as_deref(),
    )
    .map(|raw| {
        parse_statistical_aggregate_execution_mode(raw.field, &raw.value)
            .map(|mode| NativeConfigValue::new(mode, raw.field))
    })
    .transpose()?;
    if matches!(memory_policy.value, ServerNativeMemoryPolicy::Unbounded)
        && (device_budget.is_some()
            || max_source_chunk.is_some()
            || max_row_groups.is_some()
            || max_retry_attempts.is_some()
            || min_retry_chunk_row_groups.is_some()
            || max_source_reads_in_flight.is_some()
            || source_chunk_planning_mode.is_some()
            || native_pipeline_execution_mode.is_some()
            || device_budget_fraction.is_some())
    {
        return Err(ServerError::invalid_config(
            memory_policy.field,
            "unbounded native execution cannot be combined with bounded native execution settings",
        ));
    }

    let device_ordinal = *gpu_devices.first().unwrap_or(&0);
    let mut execution_config = match memory_policy.value {
        ServerNativeMemoryPolicy::Unbounded => NativeCudfExecutionConfig::unbounded(),
        ServerNativeMemoryPolicy::Bounded => match (&device_budget, &max_source_chunk, &max_row_groups) {
            (None, None, None) => {
                let device_budget_bytes = match device_budget_fraction {
                    Some(fraction) => NativeCudfExecutionConfig::auto_sized_device_budget_bytes_for_device_with_fraction(
                        device_ordinal,
                        fraction,
                    ),
                    None => NativeCudfExecutionConfig::auto_sized_device_budget_bytes_for_device(device_ordinal),
                };
                NativeCudfExecutionConfig::bounded(
                    device_budget_bytes,
                    NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_SOURCE_CHUNK_BYTES,
                    NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_ROW_GROUPS_PER_CHUNK,
                    max_retry_attempts
                        .as_ref()
                        .map_or(NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS, |value| value.value),
                    min_retry_chunk_row_groups
                        .as_ref()
                        .map_or(NativeCudfExecutionConfig::DEFAULT_BOUNDED_MIN_RETRY_CHUNK_ROW_GROUPS, |value| {
                            value.value
                        }),
                )
            },
            (None, Some(max_source_chunk), Some(max_row_groups)) if device_budget_fraction.is_some() => {
                if max_source_chunk.value == 0 {
                    return Err(ServerError::invalid_config(max_source_chunk.field, "must be greater than zero"));
                }
                if max_row_groups.value == 0 {
                    return Err(ServerError::invalid_config(max_row_groups.field, "must be greater than zero"));
                }
                NativeCudfExecutionConfig::bounded(
                    NativeCudfExecutionConfig::auto_sized_device_budget_bytes_for_device_with_fraction(
                        device_ordinal,
                        device_budget_fraction.expect("checked above"),
                    ),
                    max_source_chunk.value,
                    max_row_groups.value,
                    max_retry_attempts
                        .as_ref()
                        .map_or(NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS, |value| value.value),
                    min_retry_chunk_row_groups
                        .as_ref()
                        .map_or(NativeCudfExecutionConfig::DEFAULT_BOUNDED_MIN_RETRY_CHUNK_ROW_GROUPS, |value| {
                            value.value
                        }),
                )
            },
            (Some(device_budget), Some(max_source_chunk), Some(max_row_groups)) => {
                if device_budget.value == 0 {
                    return Err(ServerError::invalid_config(device_budget.field, "must be greater than zero"));
                }
                if max_source_chunk.value == 0 {
                    return Err(ServerError::invalid_config(max_source_chunk.field, "must be greater than zero"));
                }
                if max_row_groups.value == 0 {
                    return Err(ServerError::invalid_config(max_row_groups.field, "must be greater than zero"));
                }
                NativeCudfExecutionConfig::bounded(
                    device_budget.value,
                    max_source_chunk.value,
                    max_row_groups.value,
                    max_retry_attempts
                        .as_ref()
                        .map_or(NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS, |value| value.value),
                    min_retry_chunk_row_groups
                        .as_ref()
                        .map_or(NativeCudfExecutionConfig::DEFAULT_BOUNDED_MIN_RETRY_CHUNK_ROW_GROUPS, |value| {
                            value.value
                        }),
                )
            },
            _ => {
                return Err(ServerError::invalid_config(
                    first_present_bounded_budget_field(&device_budget, &max_source_chunk, &max_row_groups),
                    "bounded native execution requires device budget, max source chunk bytes, and max row groups per chunk",
                ));
            },
        },
    }
    .with_device_ordinal(device_ordinal);
    if let Some(max_retry_attempts) = max_retry_attempts {
        if max_retry_attempts.value == 0 {
            return Err(ServerError::invalid_config(max_retry_attempts.field, "must be greater than zero"));
        }
        if let MemoryControlPolicy::Bounded {
            device_budget_bytes,
            max_source_chunk_bytes,
            max_row_groups_per_chunk,
            min_retry_chunk_row_groups,
            ..
        } = execution_config.memory_control_policy()
        {
            execution_config = execution_config.with_memory_control_policy(MemoryControlPolicy::Bounded {
                device_budget_bytes,
                max_source_chunk_bytes,
                max_row_groups_per_chunk,
                max_retry_attempts: max_retry_attempts.value,
                min_retry_chunk_row_groups,
            });
        }
    }
    if let Some(min_retry_chunk_row_groups) = min_retry_chunk_row_groups {
        if min_retry_chunk_row_groups.value == 0 {
            return Err(ServerError::invalid_config(min_retry_chunk_row_groups.field, "must be greater than zero"));
        }
        if let MemoryControlPolicy::Bounded {
            device_budget_bytes,
            max_source_chunk_bytes,
            max_row_groups_per_chunk,
            max_retry_attempts,
            ..
        } = execution_config.memory_control_policy()
        {
            execution_config = execution_config.with_memory_control_policy(MemoryControlPolicy::Bounded {
                device_budget_bytes,
                max_source_chunk_bytes,
                max_row_groups_per_chunk,
                max_retry_attempts,
                min_retry_chunk_row_groups: min_retry_chunk_row_groups.value,
            });
        }
    }
    if let Some(max_source_reads_in_flight) = max_source_reads_in_flight {
        if max_source_reads_in_flight.value == 0 {
            return Err(ServerError::invalid_config(max_source_reads_in_flight.field, "must be greater than zero"));
        }
        execution_config =
            execution_config.with_source_scheduler_policy(SourceSchedulerPolicy::new(max_source_reads_in_flight.value));
    }
    if let Some(mode) = source_chunk_planning_mode {
        execution_config = execution_config.with_source_chunk_planning_mode(mode.value);
    }
    if let Some(timeout_secs) = host_fetch_acquire_timeout_secs {
        if timeout_secs.value == 0 {
            return Err(ServerError::invalid_config(timeout_secs.field, "must be greater than zero"));
        }
        execution_config = execution_config.with_host_fetch_acquire_timeout(Duration::from_secs(timeout_secs.value));
    }
    if let Some(mode) = native_pipeline_execution_mode {
        execution_config = execution_config.with_native_pipeline_execution_mode(mode.value);
    }
    if let Some(mode) = statistical_aggregate_execution_mode {
        execution_config = execution_config.with_statistical_aggregate_execution_mode(mode.value);
    }

    Ok(CudfOptimizerConfig::new(GpuFallbackPolicy::default()).with_native_execution_config(execution_config))
}

fn optional_native_string(
    values: &mut EnvValues<'_>,
    env_name: &'static str,
    file_field: &'static str,
    file_value: Option<&str>,
) -> Option<NativeConfigValue<String>> {
    values
        .string(env_name)
        .map(|value| NativeConfigValue::new(value, env_name))
        .or_else(|| file_value.map(|value| NativeConfigValue::new(value.to_string(), file_field)))
}

fn optional_native_u64(
    values: &mut EnvValues<'_>,
    env_name: &'static str,
    file_field: &'static str,
    file_value: Option<u64>,
) -> Result<Option<NativeConfigValue<u64>>> {
    Ok(match values.optional_u64(env_name)? {
        Some(value) => Some(NativeConfigValue::new(value, env_name)),
        None => file_value.map(|value| NativeConfigValue::new(value, file_field)),
    })
}

fn optional_native_usize(
    values: &mut EnvValues<'_>,
    env_name: &'static str,
    file_field: &'static str,
    file_value: Option<usize>,
) -> Result<Option<NativeConfigValue<usize>>> {
    Ok(match values.optional_usize(env_name)? {
        Some(value) => Some(NativeConfigValue::new(value, env_name)),
        None => file_value.map(|value| NativeConfigValue::new(value, file_field)),
    })
}

fn first_present_bounded_budget_field(
    device_budget: &Option<NativeConfigValue<u64>>,
    max_source_chunk: &Option<NativeConfigValue<u64>>,
    max_row_groups: &Option<NativeConfigValue<usize>>,
) -> &'static str {
    device_budget
        .as_ref()
        .map(|value| value.field)
        .or_else(|| max_source_chunk.as_ref().map(|value| value.field))
        .or_else(|| max_row_groups.as_ref().map(|value| value.field))
        .unwrap_or(NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV)
}

#[cfg(feature = "iceberg")]
fn iceberg_source_storage_profile(
    config: &IcebergServerConfig,
    values: &mut EnvValues<'_>,
) -> nexus_query_engine::SourceStorageProfile {
    match &config.backend {
        IcebergCatalogBackendConfig::Glue(glue) => {
            let mut catalog_config = crate::table_format::iceberg::catalog::GlueIcebergCatalogConfig::new(
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

fn parse_native_memory_policy(field: &'static str, raw: &str) -> Result<ServerNativeMemoryPolicy> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "bounded" => Ok(ServerNativeMemoryPolicy::Bounded),
        "unbounded" => Ok(ServerNativeMemoryPolicy::Unbounded),
        other => Err(ServerError::invalid_config(field, format!("must be `bounded` or `unbounded`, got `{other}`"))),
    }
}

fn parse_source_chunk_planning_mode(field: &'static str, raw: &str) -> Result<SourceChunkPlanningMode> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "conservative" => Ok(SourceChunkPlanningMode::Conservative),
        "local_parquet_throughput" => Ok(SourceChunkPlanningMode::LocalParquetThroughput),
        other => Err(ServerError::invalid_config(
            field,
            format!("must be `conservative` or `local_parquet_throughput`, got `{other}`"),
        )),
    }
}

fn parse_native_pipeline_execution_mode(field: &'static str, raw: &str) -> Result<NativePipelineExecutionMode> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "strict_bounded" => Ok(NativePipelineExecutionMode::StrictBounded),
        "adaptive" => Ok(NativePipelineExecutionMode::Adaptive),
        other => {
            Err(ServerError::invalid_config(field, format!("must be `strict_bounded` or `adaptive`, got `{other}`")))
        },
    }
}

fn parse_statistical_aggregate_execution_mode(
    field: &'static str,
    raw: &str,
) -> Result<StatisticalAggregateExecutionMode> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "datafusion_exact_host_welford" => Ok(StatisticalAggregateExecutionMode::DataFusionExactHostWelford),
        "gpu_native_tolerant" => Ok(StatisticalAggregateExecutionMode::GpuNativeTolerant),
        other => Err(ServerError::invalid_config(
            field,
            format!("must be `datafusion_exact_host_welford` or `gpu_native_tolerant`, got `{other}`"),
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
        NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV,
        NEXUS_ICEBERG_PARQUET_READ_STRATEGY_ENV,
        NEXUS_ICEBERG_REMOTE_DIRECT_ENV,
        NEXUS_ICEBERG_REMOTE_DIRECT_READ_MODE_ENV,
        NEXUS_ICEBERG_REMOTE_HOST_BUFFER_RANGED_ENV,
        NEXUS_LAKEHOUSE_CACHE_ROOT_ENV,
        NEXUS_LAKEHOUSE_CACHE_READ_ENV,
        NEXUS_LAKEHOUSE_CACHE_POLICY_ENV,
        NEXUS_LAKEHOUSE_CACHE_MAX_BYTES_ENV,
        NEXUS_ICEBERG_SCAN_PLANNING_TIMEOUT_SECS_ENV,
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
    let scan_planning_timeout = Duration::from_secs(
        values
            .positive_u64(NEXUS_ICEBERG_SCAN_PLANNING_TIMEOUT_SECS_ENV, DEFAULT_ICEBERG_SCAN_PLANNING_TIMEOUT_SECS)?,
    );

    Ok(Some(IcebergServerConfig {
        catalog_name,
        namespace,
        tables,
        backend,
        parquet_read_strategy_config,
        scan_planning_timeout,
    }))
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
    let has_explicit_s3_credentials =
        values.string(crate::table_format::iceberg::NEXUS_ICEBERG_S3_ACCESS_KEY_ID_ENV).is_some()
            || values.string(crate::table_format::iceberg::NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY_ENV).is_some()
            || values.string(crate::table_format::iceberg::NEXUS_ICEBERG_S3_SESSION_TOKEN_ENV).is_some();
    let credential_source = values.string(NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV);
    if let Some(raw) = credential_source.as_deref() {
        if !raw.eq_ignore_ascii_case(crate::table_format::iceberg::NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_DEFAULT_CHAIN) {
            return Err(ServerError::invalid_config(
                NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV,
                format!("must be `default_chain` when set, got `{raw}`"),
            ));
        }
        if has_explicit_s3_credentials {
            return Err(ServerError::invalid_config(
                NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV,
                "cannot be combined with static NEXUS_ICEBERG_S3_ACCESS_KEY_ID / \
                 NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY / NEXUS_ICEBERG_S3_SESSION_TOKEN credentials",
            ));
        }
    }
    let has_storage_region = values.string(NEXUS_ICEBERG_S3_REGION_ENV).is_some();
    let credential_mode = if storage_endpoint.is_some()
        || has_storage_region
        || has_explicit_s3_credentials
        || credential_source.is_some()
    {
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
        .string(crate::table_format::iceberg::AWS_REGION_ENV)
        .or_else(|| values.string(crate::table_format::iceberg::AWS_DEFAULT_REGION_ENV))
        .ok_or_else(|| {
            ServerError::invalid_config(crate::table_format::iceberg::AWS_REGION_ENV, "is required for Glue Iceberg")
        })?;

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
        profile: values.string(crate::table_format::iceberg::AWS_PROFILE_ENV),
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
) -> Result<crate::table_format::iceberg::catalog::RestIcebergCatalogConfig> {
    let uri = values.required_string(NEXUS_ICEBERG_REST_URI_ENV)?;
    let catalog_handle =
        values.string(NEXUS_ICEBERG_REST_CATALOG_HANDLE_ENV).unwrap_or_else(|| DEFAULT_REST_CATALOG_HANDLE.to_string());

    let mut config =
        crate::table_format::iceberg::catalog::RestIcebergCatalogConfig::new(catalog_handle, uri, warehouse);
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
        NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV,
        crate::NEXUS_ICEBERG_PARQUET_READ_STRATEGY_ENV,
        crate::NEXUS_ICEBERG_REMOTE_DIRECT_ENV,
        crate::NEXUS_ICEBERG_REMOTE_DIRECT_READ_MODE_ENV,
        crate::NEXUS_ICEBERG_REMOTE_HOST_BUFFER_RANGED_ENV,
        crate::NEXUS_LAKEHOUSE_CACHE_ROOT_ENV,
        crate::NEXUS_LAKEHOUSE_CACHE_READ_ENV,
        crate::NEXUS_LAKEHOUSE_CACHE_POLICY_ENV,
        crate::NEXUS_LAKEHOUSE_CACHE_MAX_BYTES_ENV,
        NEXUS_ICEBERG_SCAN_PLANNING_TIMEOUT_SECS_ENV,
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
    // An explicit `NEXUS_SERVER_CUGRAPH_ENABLED=false` requests no capability
    // and is accepted, so one env file can serve both artifact flavors.
    if values.any_present([
        NEXUS_SERVER_CUGRAPH_ALGORITHMS_ENV,
        NEXUS_SERVER_CUGRAPH_DIRECTED_ENV,
        NEXUS_SERVER_CUGRAPH_RENUMBER_ENV,
        NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY_ENV,
    ]) || values.optional_bool(NEXUS_SERVER_CUGRAPH_ENABLED_ENV)? == Some(true)
    {
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
        raw.parse().map_err(|error| ServerError::invalid_config_with_source(name, "must be a socket address", error))
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
                    .map_err(|error| ServerError::invalid_config_with_source(name, "must be an integer", error))
            })
            .transpose()
    }

    fn optional_u64(&mut self, name: &'static str) -> Result<Option<u64>> {
        self.string(name)
            .map(|raw| {
                raw.parse::<u64>()
                    .map_err(|error| ServerError::invalid_config_with_source(name, "must be an integer", error))
            })
            .transpose()
    }

    fn optional_f64(&mut self, name: &'static str) -> Result<Option<f64>> {
        self.string(name)
            .map(|raw| {
                let value = raw.parse::<f64>().map_err(|error| {
                    ServerError::invalid_config_with_source(name, "must be a decimal number", error)
                })?;
                if !value.is_finite() || !(0.0..=1.0).contains(&value) || value == 0.0 {
                    return Err(ServerError::invalid_config(name, "must be greater than 0 and at most 1"));
                }
                Ok(value)
            })
            .transpose()
    }

    fn optional_session_profile(&mut self, name: &'static str) -> Result<Option<ServerSessionProfile>> {
        self.string(name)
            .map(|raw| ServerSessionProfile::parse(&raw).map_err(|reason| ServerError::invalid_config(name, reason)))
            .transpose()
    }

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
