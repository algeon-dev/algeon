use std::{
    net::SocketAddr,
    path::PathBuf,
    time::{Duration, Instant},
};

use nexus_query_engine::{
    mode::StatisticalAggregateExecutionMode,
    runtime::{SourceChunkPlanningMode, SourceSchedulerPolicy},
};
use serde::Deserialize;

use super::{
    auth::AuthConfig,
    error::{Result, ServerError},
};
#[cfg(feature = "iceberg")]
use crate::table_format::scan::{DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID, credential_free_endpoint_identity};
use crate::{
    backend::{GpuMemoryOwnership, NexusGpuDeviceConfig, NexusGpuDeviceProfile, NexusGpuQueuePolicy},
    cugraph_sql::{CugraphGraphConstructionPolicy, CugraphGraphOptions},
    native_exec::NexusNativeExecutionConfig,
    planner::NexusNativeOptimizerConfig,
};
#[cfg(feature = "iceberg")]
use crate::{
    table_format::AWS_ENDPOINT_URL_ENV, table_format::AWS_ENDPOINT_URL_S3_ENV,
    table_format::NEXUS_LAKEHOUSE_CACHE_MAX_BYTES_ENV, table_format::NEXUS_LAKEHOUSE_CACHE_POLICY_ENV,
    table_format::NEXUS_LAKEHOUSE_CACHE_READ_ENV, table_format::NEXUS_LAKEHOUSE_CACHE_ROOT_ENV,
    table_format::TableFormatLakehouseCachePolicy, table_format::TableFormatLakehouseCacheReadPolicy,
    table_format::TableFormatParquetReadStrategyConfig,
};

mod admission;
mod cugraph;
mod env;
mod file;
mod iceberg;
mod native;
mod observability;
mod workspace;

pub use admission::{
    NEXUS_SERVER_ADMISSION_MAX_OVERTAKES_PER_WAITER_ENV, NEXUS_SERVER_ADMISSION_MAX_QUEUE_WAIT_MS_ENV,
    NEXUS_SERVER_ADMISSION_MAX_QUEUED_ATTEMPTS_ENV, NEXUS_SERVER_GPU_DEVICES_ENV,
};
use admission::{server_device_profiles, server_gpu_devices, server_queue_policy};
use cugraph::cugraph_config;
pub use cugraph::{
    CugraphServerConfig, NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY_ENV, NEXUS_SERVER_CUGRAPH_ENABLED_ENV,
};
use env::{DEFAULT_BIND_ADDR, DEFAULT_STATEMENT_TICKET_TTL_SECS, EnvValues};
pub use env::{
    NEXUS_SERVER_BIND_ENV, NEXUS_SERVER_CONFIG_FILE_ENV, NEXUS_SERVER_LOG_ENV,
    NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS_ENV,
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
    NEXUS_ICEBERG_GLUE_CATALOG_NAME_ENV, NEXUS_ICEBERG_GLUE_URI_ENV, NEXUS_ICEBERG_NAMESPACE_ENV,
    NEXUS_ICEBERG_REST_CATALOG_HANDLE_ENV, NEXUS_ICEBERG_REST_PREFIX_ENV, NEXUS_ICEBERG_REST_TOKEN_ENV,
    NEXUS_ICEBERG_REST_URI_ENV, NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV, NEXUS_ICEBERG_S3_ENDPOINT_ENV,
    NEXUS_ICEBERG_S3_PATH_STYLE_ENV, NEXUS_ICEBERG_S3_REGION_ENV, NEXUS_ICEBERG_SCAN_PLANNING_TIMEOUT_SECS_ENV,
    NEXUS_ICEBERG_TABLES_ENV, NEXUS_ICEBERG_WAREHOUSE_ENV,
};
#[cfg(feature = "iceberg")]
use iceberg::{iceberg_config, iceberg_source_storage_profile};
use native::native_optimizer_config;
pub use native::{
    NEXUS_SERVER_FINAL_PLAN_REQUIREMENT_ENV, NEXUS_SERVER_NATIVE_CONFIG_OPTIONS,
    NEXUS_SERVER_NATIVE_MAX_RETRY_ATTEMPTS_ENV, NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV,
    NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, NEXUS_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT_ENV,
    NEXUS_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV, NEXUS_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV,
    NexusServerNativeConfigOption, ServerNativeConfigScope,
};
use observability::server_observability_config;
pub use observability::{
    NEXUS_SERVER_OBSERVABILITY_ACTIVE_LOGICAL_CAPTURE_MAX_BYTES_ENV,
    NEXUS_SERVER_OBSERVABILITY_GLOBAL_SEALED_MAX_BYTES_ENV, NEXUS_SERVER_OBSERVABILITY_GLOBAL_SEALED_MAX_COUNT_ENV,
    NEXUS_SERVER_OBSERVABILITY_PER_QUERY_MAX_BYTES_ENV, NEXUS_SERVER_OBSERVABILITY_PER_QUERY_MAX_RECORDS_ENV,
    NEXUS_SERVER_OBSERVABILITY_SEALED_TTL_SECS_ENV, NEXUS_SERVER_OBSERVABILITY_SHUTDOWN_GRACE_SECS_ENV,
    ServerObservabilityConfig,
};
use workspace::workspace_overlay_config;
pub use workspace::{
    NEXUS_SERVER_WORKSPACE_BACKING_ALIASES_ENV, NEXUS_SERVER_WORKSPACE_BACKING_CATALOG_ENV,
    NEXUS_SERVER_WORKSPACE_BACKING_SCHEMA_ENV, NEXUS_SERVER_WORKSPACE_CATALOG_ENV, NEXUS_SERVER_WORKSPACE_SCHEMA_ENV,
    WorkspaceOverlayConfig,
};

// The resource-only admission contract has one configuration surface. These
// historical environment switches must fail closed instead of silently starting a deployment with
// different defaults. Keep this list centrally adjacent to the first env
// read so every server configuration path applies the same migration gate.
const REMOVED_LEGACY_SERVER_ENVIRONMENT_VARIABLES: &[&str] = &[
    "NEXUS_SERVER_ADMISSION_MAX_ACTIVE_ATTEMPTS_PER_DEVICE",
    "NEXUS_SERVER_ADMISSION_ATTEMPT_FLOOR_BYTES",
    "NEXUS_SERVER_ADMISSION_SOURCE_WORK_PER_ATTEMPT",
    "NEXUS_SERVER_ADMISSION_WORKLOAD_PROFILE",
    "NEXUS_SERVER_ADMISSION_PINNED_DEVICE",
    "NEXUS_SERVER_ADMISSION_MAX_QUERIES_PER_DEVICE",
    "NEXUS_SERVER_ADMISSION_WAIT_TIMEOUT_MS",
    "NEXUS_SERVER_MAX_CONCURRENT_QUERIES",
    "NEXUS_SERVER_MAX_IN_FLIGHT",
    "NEXUS_SERVER_NATIVE_MEMORY_POLICY",
    "NEXUS_SERVER_NATIVE_PIPELINE_EXECUTION_MODE",
    "NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES",
    "NEXUS_SERVER_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS",
    "NEXUS_SERVER_CUGRAPH_DIRECTED",
    "NEXUS_SERVER_CUGRAPH_RENUMBER",
    "NEXUS_SERVER_NATIVE_SOURCE_CACHE_MAX_RETAINED_BYTES",
    "NEXUS_SERVER_NATIVE_GRAPH_CACHE_MAX_RETAINED_BYTES",
    "NEXUS_SERVER_NATIVE_QUERY_MEMORY_DOMAIN_CEILING_MULTIPLIER",
    "NEXUS_SERVER_DEVICE_BUDGET_FRACTION",
];

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub bind_addr: SocketAddr,
    /// Queue-only admission policy. Device-local bounds live in
    /// `gpu_device_profiles` and are never duplicated here.
    pub queue_policy: NexusGpuQueuePolicy,
    /// Complete ordinal-keyed resource envelopes for the selected GPUs.
    /// The server never broadcasts a scalar admission setting across this
    /// collection.
    pub gpu_device_profiles: Vec<NexusGpuDeviceProfile>,
    pub gpu_devices: Vec<usize>,
    /// Exact deployment ownership declaration consumed when a bounded GPU
    /// backend is constructed. Configuration loading intentionally leaves this
    /// empty; the embedding/launch profile must make the ownership promise.
    pub gpu_memory_ownership: Vec<(usize, GpuMemoryOwnership)>,
    /// Wall-clock time a planned Flight SQL statement ticket may wait for
    /// `DoGet`. Expired tickets are dropped; GPU admission is acquired later
    /// by native plan execution.
    pub statement_ticket_ttl: Duration,
    /// Bounded native runtime-artifact capture and sealed retention policy.
    pub observability: ServerObservabilityConfig,
    pub native_optimizer_config: NexusNativeOptimizerConfig,
    /// Authentication mode applied to incoming Flight SQL RPCs. Loaded from
    /// the optional TOML overlay and disabled by default.
    pub auth: AuthConfig,
    /// Optional mutable workspace overlay that can read through to a
    /// configured source schema while keeping all DDL local to the workspace.
    pub workspace: Option<WorkspaceOverlayConfig>,
    #[cfg(feature = "iceberg")]
    pub iceberg: Option<IcebergServerConfig>,
    pub cugraph: Option<CugraphServerConfig>,
}
impl ServerConfig {
    /// Attach the deployment ownership promise required by bounded backend
    /// construction. The backend validates this declaration against
    /// [`Self::gpu_devices`] before acquiring any device lease.
    pub fn with_gpu_memory_ownership(
        mut self,
        ownership: impl IntoIterator<Item = (usize, GpuMemoryOwnership)>,
    ) -> Self {
        self.gpu_memory_ownership = ownership.into_iter().collect();
        self
    }

    pub fn from_env() -> Result<Self> {
        Self::from_env_get(|name| std::env::var(name).ok())
    }

    pub fn from_env_get(mut get: impl FnMut(&str) -> Option<String>) -> Result<Self> {
        let mut values = EnvValues::new(&mut get);
        reject_removed_legacy_server_environment(&mut values)?;
        let bind_addr = values.socket_addr(NEXUS_SERVER_BIND_ENV, DEFAULT_BIND_ADDR)?;
        let overlay_path = values.string(NEXUS_SERVER_CONFIG_FILE_ENV).map(PathBuf::from);
        let overlay = match overlay_path {
            Some(path) => load_overlay(&path)?,
            None => ServerFileOverlay::default(),
        };
        let gpu_devices = server_gpu_devices(&mut values)?;
        let queue_policy = server_queue_policy(&mut values, &overlay.admission)?;
        let statement_ticket_ttl = checked_config_duration(
            NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS_ENV,
            values.positive_u64(NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS_ENV, DEFAULT_STATEMENT_TICKET_TTL_SECS)?,
        )?;
        let observability = server_observability_config(&mut values, &overlay.observability)?;
        let native_optimizer_config = native_optimizer_config(&mut values, &overlay.native)?;
        let gpu_device_profiles = server_device_profiles(&overlay.admission, &gpu_devices)?;
        #[cfg(feature = "iceberg")]
        let mut native_optimizer_config = native_optimizer_config;

        #[cfg(feature = "iceberg")]
        let mut iceberg = iceberg_config(&mut values)?;

        #[cfg(feature = "iceberg")]
        if let Some(config) = iceberg.as_mut() {
            let profile = iceberg_source_storage_profile(config);
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

        let cugraph = cugraph_config(&mut values)?;

        let workspace = workspace_overlay_config(&mut values)?;

        overlay.auth.mode.validate()?;

        let config = Self {
            bind_addr,
            queue_policy,
            gpu_device_profiles,
            gpu_devices,
            gpu_memory_ownership: Vec::new(),
            statement_ticket_ttl,
            observability,
            native_optimizer_config,
            auth: overlay.auth,
            workspace,
            #[cfg(feature = "iceberg")]
            iceberg,
            cugraph,
        };
        config.validate_gpu_device_profiles()?;
        Ok(config)
    }

    /// Prove that the public config struct still carries one profile for each
    /// selected ordinal.  `from_env_get` already applies this check while
    /// parsing TOML, but callers can construct `ServerConfig` directly before
    /// injecting or constructing a Flight backend.
    pub(crate) fn validate_gpu_device_profiles(&self) -> Result<()> {
        let selected = self.gpu_devices.iter().copied().collect::<std::collections::BTreeSet<_>>();
        if selected.len() != self.gpu_devices.len() {
            return Err(ServerError::invalid_config(
                NEXUS_SERVER_GPU_DEVICES_ENV,
                "must not contain duplicate device ordinals",
            ));
        }
        let mut configured = std::collections::BTreeSet::new();
        for profile in &self.gpu_device_profiles {
            if !configured.insert(profile.device_ordinal()) {
                return Err(ServerError::invalid_config(
                    "admission.device_profiles",
                    format!("contains duplicate profile for device ordinal {}", profile.device_ordinal()),
                )
                .with_fact("reason", "duplicate_device")
                .with_fact("device_ordinal", profile.device_ordinal()));
            }
        }
        let missing = selected.difference(&configured).copied().collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(ServerError::invalid_config(
                "admission.device_profiles",
                "must configure every device selected by NEXUS_SERVER_GPU_DEVICES",
            )
            .with_fact("reason", "missing_devices")
            .with_fact("device_ordinals", format_device_ordinals(&missing)));
        }
        let extra = configured.difference(&selected).copied().collect::<Vec<_>>();
        if !extra.is_empty() {
            return Err(ServerError::invalid_config(
                "admission.device_profiles",
                "must not configure a device outside NEXUS_SERVER_GPU_DEVICES",
            )
            .with_fact("reason", "unconfigured_devices")
            .with_fact("device_ordinals", format_device_ordinals(&extra)));
        }
        Ok(())
    }

    pub(crate) fn resolve_gpu_devices(&self) -> Result<Vec<NexusGpuDeviceConfig>> {
        self.validate_gpu_device_profiles()?;

        let selected = self.gpu_devices.iter().copied().collect::<std::collections::BTreeSet<_>>();
        let mut declared = std::collections::BTreeMap::new();
        for (ordinal, ownership) in self.gpu_memory_ownership.iter().copied() {
            if declared.insert(ordinal, ownership).is_some() {
                return Err(ServerError::invalid_config(
                    "gpu_memory_ownership",
                    format!("contains duplicate GPU memory ownership declaration for device ordinal {ordinal}"),
                )
                .with_fact("reason", "duplicate_device")
                .with_fact("device_ordinal", ordinal));
            }
        }
        let declared_ordinals = declared.keys().copied().collect::<std::collections::BTreeSet<_>>();
        let missing = selected.difference(&declared_ordinals).copied().collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(ServerError::invalid_config(
                "gpu_memory_ownership",
                "must declare GPU memory ownership for every selected device",
            )
            .with_fact("reason", "missing_devices")
            .with_fact("device_ordinals", format_device_ordinals(&missing)));
        }
        let extra = declared_ordinals.difference(&selected).copied().collect::<Vec<_>>();
        if !extra.is_empty() {
            return Err(ServerError::invalid_config(
                "gpu_memory_ownership",
                "must not declare GPU memory ownership for an unselected device",
            )
            .with_fact("reason", "extra_devices")
            .with_fact("device_ordinals", format_device_ordinals(&extra)));
        }

        let devices = self
            .gpu_device_profiles
            .iter()
            .map(|profile| {
                let ownership = declared[&profile.device_ordinal()];
                NexusGpuDeviceConfig::new(profile.clone(), ownership)
            })
            .collect();
        Ok(devices)
    }
}

fn format_device_ordinals(device_ordinals: &[usize]) -> String {
    device_ordinals.iter().map(usize::to_string).collect::<Vec<_>>().join(",")
}

fn reject_removed_legacy_server_environment(values: &mut EnvValues<'_>) -> Result<()> {
    for name in REMOVED_LEGACY_SERVER_ENVIRONMENT_VARIABLES {
        if values.string(name).is_some() {
            return Err(ServerError::invalid_config(
                name,
                "this server setting was removed by the resource-only admission contract; remove it",
            ));
        }
    }
    Ok(())
}

fn checked_config_duration(field: &'static str, seconds: u64) -> Result<Duration> {
    let duration = Duration::from_secs(seconds);
    if Instant::now().checked_add(duration).is_none() {
        return Err(ServerError::invalid_config(field, "duration exceeds the monotonic clock range"));
    }
    Ok(duration)
}
