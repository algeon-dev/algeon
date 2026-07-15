#[cfg(feature = "iceberg")]
use std::fmt;
use std::{collections::BTreeMap, fs, net::SocketAddr, num::NonZeroUsize, path::PathBuf, time::Duration};

use nexus_query_engine::{
    mode::{NativePipelineExecutionMode, StatisticalAggregateExecutionMode},
    runtime::{
        MemoryControlPolicy, QueryMemoryDomainPolicy, QueryPlacementPolicy, SlotGuarantee, SourceChunkPlanningMode,
        SourceSchedulerPolicy,
    },
};
use serde::Deserialize;

use super::{
    auth::AuthConfig,
    error::{Result, ServerError},
};
#[cfg(feature = "iceberg")]
use crate::table_format::scan::{DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID, credential_free_endpoint_identity};
use crate::{
    backend::{MemoryShare, NexusAdmissionPolicy},
    native_exec::NativeCudfExecutionConfig,
    planner::CudfOptimizerConfig,
    policy::GpuFallbackPolicy,
};
#[cfg(feature = "cugraph")]
use crate::{graph_sql::CugraphGraphConstructionPolicy, graph_sql::CugraphGraphOptions};
#[cfg(feature = "iceberg")]
use crate::{
    table_format::AWS_ENDPOINT_URL_ENV, table_format::AWS_ENDPOINT_URL_S3_ENV,
    table_format::NEXUS_ICEBERG_PARQUET_READ_STRATEGY_ENV, table_format::NEXUS_ICEBERG_REMOTE_DIRECT_ENV,
    table_format::NEXUS_ICEBERG_REMOTE_DIRECT_READ_MODE_ENV, table_format::NEXUS_ICEBERG_REMOTE_HOST_BUFFER_RANGED_ENV,
    table_format::NEXUS_LAKEHOUSE_CACHE_MAX_BYTES_ENV, table_format::NEXUS_LAKEHOUSE_CACHE_POLICY_ENV,
    table_format::NEXUS_LAKEHOUSE_CACHE_READ_ENV, table_format::NEXUS_LAKEHOUSE_CACHE_ROOT_ENV,
    table_format::TableFormatLakehouseCachePolicy, table_format::TableFormatLakehouseCacheReadPolicy,
    table_format::TableFormatParquetReadStrategyConfig, table_format::TableFormatParquetReadStrategyPreference,
};

mod admission;
mod cugraph;
mod env;
mod file;
mod iceberg;
mod native;
mod workspace;

pub use admission::{
    NEXUS_SERVER_ADMISSION_GRANT_MIN_ENV, NEXUS_SERVER_ADMISSION_GRANT_WANT_ENV,
    NEXUS_SERVER_ADMISSION_MAX_QUERIES_PER_DEVICE_ENV, NEXUS_SERVER_ADMISSION_PINNED_DEVICE_ENV,
    NEXUS_SERVER_ADMISSION_PLACEMENT_POLICY_ENV, NEXUS_SERVER_ADMISSION_SLOT_GUARANTEE_ENV,
    NEXUS_SERVER_ADMISSION_WAIT_TIMEOUT_MS_ENV, NEXUS_SERVER_ADMISSION_WORKLOAD_PROFILE_ENV,
    NEXUS_SERVER_CPU_FALLBACK_MAX_CONCURRENT_STATEMENTS_ENV, NEXUS_SERVER_CPU_FALLBACK_WAIT_TIMEOUT_MS_ENV,
    NEXUS_SERVER_DEVICE_BUDGET_FRACTION_ENV, NEXUS_SERVER_GPU_DEVICES_ENV, NEXUS_SERVER_MAX_CONCURRENT_QUERIES_ENV,
    NEXUS_SERVER_MAX_IN_FLIGHT_ENV,
};
use admission::{
    first_non_neutral_unbounded_admission_field, server_admission_placement_config, server_admission_policy,
    server_admission_workload_profile, server_cpu_fallback_max_concurrent_statements, server_cpu_fallback_wait_timeout,
    server_gpu_devices, server_max_concurrent_queries, server_placement_eligible_device_count,
    validate_admission_grant_min_capacity,
};
#[cfg(feature = "cugraph")]
pub use cugraph::CugraphServerConfig;
#[cfg(feature = "cugraph")]
use cugraph::cugraph_config;
#[cfg(not(feature = "cugraph"))]
use cugraph::reject_cugraph_env_without_feature;
pub use cugraph::{
    NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY_ENV, NEXUS_SERVER_CUGRAPH_DIRECTED_ENV, NEXUS_SERVER_CUGRAPH_ENABLED_ENV,
    NEXUS_SERVER_CUGRAPH_RENUMBER_ENV,
};
use env::{DEFAULT_BIND_ADDR, DEFAULT_LOG_FILTER, DEFAULT_STATEMENT_TICKET_TTL_SECS, EnvValues};
pub use env::{
    NEXUS_SERVER_BIND_ENV, NEXUS_SERVER_CONFIG_FILE_ENV, NEXUS_SERVER_LOG_ENV, NEXUS_SERVER_SESSION_PROFILE_ENV,
    NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS_ENV, ServerSessionProfile,
};
use file::{ServerFileOverlay, load_overlay};
#[cfg(not(feature = "iceberg"))]
use iceberg::reject_iceberg_env_without_feature;
#[cfg(feature = "iceberg")]
pub use iceberg::{
    GlueServerBackendConfig, IcebergCatalogBackendConfig, IcebergServerConfig, IcebergTableRegistrationConfig,
};
pub use iceberg::{
    NEXUS_ICEBERG_CATALOG_KIND_ENV, NEXUS_ICEBERG_CATALOG_NAME_ENV, NEXUS_ICEBERG_GLUE_CATALOG_ID_ENV,
    NEXUS_ICEBERG_GLUE_CATALOG_NAME_ENV, NEXUS_ICEBERG_GLUE_DATABASE_ENV, NEXUS_ICEBERG_GLUE_TABLES_ENV,
    NEXUS_ICEBERG_GLUE_URI_ENV, NEXUS_ICEBERG_GLUE_WAREHOUSE_ENV, NEXUS_ICEBERG_NAMESPACE_ENV,
    NEXUS_ICEBERG_REST_CATALOG_HANDLE_ENV, NEXUS_ICEBERG_REST_PREFIX_ENV, NEXUS_ICEBERG_REST_TOKEN_ENV,
    NEXUS_ICEBERG_REST_URI_ENV, NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV, NEXUS_ICEBERG_S3_ENDPOINT_ENV,
    NEXUS_ICEBERG_S3_PATH_STYLE_ENV, NEXUS_ICEBERG_S3_REGION_ENV, NEXUS_ICEBERG_SCAN_PLANNING_TIMEOUT_SECS_ENV,
    NEXUS_ICEBERG_TABLES_ENV,
};
#[cfg(feature = "iceberg")]
use iceberg::{iceberg_config, iceberg_source_storage_profile};
use native::native_optimizer_config;
pub use native::{
    NEXUS_SERVER_NATIVE_CONFIG_OPTIONS, NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV,
    NEXUS_SERVER_NATIVE_GRAPH_CACHE_MAX_RETAINED_BYTES_ENV, NEXUS_SERVER_NATIVE_HOST_FETCH_ACQUIRE_TIMEOUT_SECS_ENV,
    NEXUS_SERVER_NATIVE_MAX_RETRY_ATTEMPTS_ENV, NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV,
    NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, NEXUS_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT_ENV,
    NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV, NEXUS_SERVER_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV,
    NEXUS_SERVER_NATIVE_PIPELINE_EXECUTION_MODE_ENV, NEXUS_SERVER_NATIVE_QUERY_MEMORY_DOMAIN_CEILING_MULTIPLIER_ENV,
    NEXUS_SERVER_NATIVE_SOURCE_CACHE_MAX_RETAINED_BYTES_ENV, NEXUS_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV,
    NEXUS_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV, NexusServerNativeConfigOption,
    ServerNativeConfigScope,
};
use workspace::workspace_overlay_config;
pub use workspace::{
    NEXUS_SERVER_WORKSPACE_BACKING_ALIASES_ENV, NEXUS_SERVER_WORKSPACE_BACKING_CATALOG_ENV,
    NEXUS_SERVER_WORKSPACE_BACKING_SCHEMA_ENV, NEXUS_SERVER_WORKSPACE_CATALOG_ENV, NEXUS_SERVER_WORKSPACE_SCHEMA_ENV,
    WorkspaceOverlayConfig,
};

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub bind_addr: SocketAddr,
    pub log_filter: String,
    pub max_in_flight: usize,
    pub admission_policy: NexusAdmissionPolicy,
    pub cpu_fallback_max_concurrent_statements: Option<NonZeroUsize>,
    pub cpu_fallback_wait_timeout: Option<Duration>,
    pub gpu_devices: Vec<usize>,
    pub device_budget_fraction: Option<f64>,
    /// Wall-clock time a planned Flight SQL statement ticket may wait for
    /// `DoGet`. Expired tickets are dropped; GPU admission is acquired later
    /// by native plan execution.
    pub statement_ticket_ttl: Duration,
    pub native_optimizer_config: CudfOptimizerConfig,
    /// DataFusion physical-plan profile applied to the Flight SQL session.
    /// See [`NEXUS_SERVER_SESSION_PROFILE_ENV`] and
    /// [`ServerSessionProfile`].
    pub session_profile: ServerSessionProfile,
    /// Authentication mode applied to incoming Flight SQL RPCs. Loaded
    /// from the TOML overlay pointed to by
    /// [`NEXUS_SERVER_CONFIG_FILE_ENV`]; defaults to
    /// [`crate::server::AuthMode::Disabled`] when no file is provided.
    pub auth: AuthConfig,
    /// Optional mutable workspace overlay that can read through to a
    /// configured source schema while keeping all DDL local to the workspace.
    pub workspace: Option<WorkspaceOverlayConfig>,
    #[cfg(feature = "iceberg")]
    pub iceberg: Option<IcebergServerConfig>,
    #[cfg(feature = "cugraph")]
    pub cugraph: Option<CugraphServerConfig>,
}
impl ServerConfig {
    pub fn from_env() -> Result<Self> {
        Self::from_env_get(|name| std::env::var(name).ok())
    }

    pub fn from_env_get(mut get: impl FnMut(&str) -> Option<String>) -> Result<Self> {
        let mut values = EnvValues::new(&mut get);
        let bind_addr = values.socket_addr(NEXUS_SERVER_BIND_ENV, DEFAULT_BIND_ADDR)?;
        let log_filter = values.string(NEXUS_SERVER_LOG_ENV).unwrap_or_else(|| DEFAULT_LOG_FILTER.to_string());
        let overlay_path = values.string(NEXUS_SERVER_CONFIG_FILE_ENV).map(PathBuf::from);
        let overlay = match overlay_path {
            Some(path) => load_overlay(&path)?,
            None => ServerFileOverlay::default(),
        };
        let admission_limit_explicit = values
            .any_present([NEXUS_SERVER_MAX_CONCURRENT_QUERIES_ENV, NEXUS_SERVER_MAX_IN_FLIGHT_ENV])
            || overlay.admission.max_concurrent_queries.is_some();
        let gpu_devices = server_gpu_devices(&mut values)?;
        let admission_workload_profile = server_admission_workload_profile(&mut values, &overlay.admission)?;
        let admission_placement = server_admission_placement_config(&mut values, &overlay.admission, &gpu_devices)?;
        let max_concurrent_queries = server_max_concurrent_queries(
            &mut values,
            &overlay.admission,
            admission_workload_profile.value,
            server_placement_eligible_device_count(gpu_devices.len(), admission_placement.placement_policy),
        )?;
        let mut max_in_flight = max_concurrent_queries.value;
        let admission = server_admission_policy(
            &mut values,
            &overlay.admission,
            max_in_flight,
            admission_workload_profile,
            admission_placement,
        )?;
        let mut admission_policy = admission.policy.clone();
        let cpu_fallback_max_concurrent_statements =
            server_cpu_fallback_max_concurrent_statements(&mut values, &overlay.fallback)?;
        let cpu_fallback_wait_timeout = server_cpu_fallback_wait_timeout(&mut values, &overlay.fallback)?;
        let device_budget_fraction = values.optional_f64(NEXUS_SERVER_DEVICE_BUDGET_FRACTION_ENV)?;
        let statement_ticket_ttl = Duration::from_secs(
            values.positive_u64(NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS_ENV, DEFAULT_STATEMENT_TICKET_TTL_SECS)?,
        );
        let mut native_optimizer_config =
            native_optimizer_config(&mut values, &overlay.native, &gpu_devices, device_budget_fraction)?;
        validate_admission_grant_min_capacity(&admission, &native_optimizer_config)?;
        if let Some(limit) = cpu_fallback_max_concurrent_statements {
            native_optimizer_config = native_optimizer_config
                .with_cpu_fallback_wait_timeout(cpu_fallback_wait_timeout.unwrap_or(Duration::ZERO))
                .with_cpu_fallback_max_concurrent_statements(limit);
        }
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
            admission_policy = admission_policy.with_max_concurrent_queries(max_in_flight);
        }
        if matches!(
            native_optimizer_config.native_execution_config().memory_control_policy(),
            MemoryControlPolicy::Unbounded
        ) && let Some(field) =
            first_non_neutral_unbounded_admission_field(admission_workload_profile, &admission, admission_placement)
        {
            return Err(ServerError::invalid_config(field, "requires bounded native memory policy"));
        }

        Ok(Self {
            bind_addr,
            log_filter,
            max_in_flight,
            admission_policy,
            cpu_fallback_max_concurrent_statements,
            cpu_fallback_wait_timeout,
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
