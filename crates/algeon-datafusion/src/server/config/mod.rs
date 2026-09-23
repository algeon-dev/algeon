use std::{
    net::SocketAddr,
    path::PathBuf,
    time::{Duration, Instant},
};

use query_engine::{
    mode::StatisticalAggregateExecutionMode,
    runtime::{SourceChunkPlanningMode, SourceSchedulerPolicy},
};
use serde::Deserialize;

use super::{
    auth::AuthConfig,
    error::{Result, ServerError},
};
use crate::server::{
    backend::{AlgeonGpuDeviceConfig, AlgeonGpuDeviceProfile, AlgeonGpuQueuePolicy, GpuMemoryOwnership},
    cugraph_sql::{CugraphGraphConstructionPolicy, CugraphGraphOptions},
    cuml_sql::CumlModelStoreConfig,
    native_exec::{AlgeonNativeExecutionConfig, SourceChunking},
    planner::AlgeonNativeOptimizerConfig,
};

mod admission;
mod cugraph;
mod env;
mod file;
mod models;
mod native;
mod observability;
mod workspace;

pub use admission::{
    ALGEON_SERVER_ADMISSION_MAX_QUEUE_WAIT_MS_ENV, ALGEON_SERVER_ADMISSION_MAX_QUEUED_ATTEMPTS_ENV,
    ALGEON_SERVER_GPU_DEVICES_ENV,
};
use admission::{server_device_profiles, server_gpu_devices, server_queue_policy};
use cugraph::cugraph_config;
pub use cugraph::{
    ALGEON_SERVER_CUGRAPH_CONSTRUCTION_POLICY_ENV, ALGEON_SERVER_CUGRAPH_ENABLED_ENV, CugraphServerConfig,
};
pub use env::{
    ALGEON_SERVER_BIND_ENV, ALGEON_SERVER_CONFIG_FILE_ENV, ALGEON_SERVER_LOG_ENV,
    ALGEON_SERVER_STATEMENT_TICKET_TTL_SECS_ENV,
};
use env::{DEFAULT_BIND_ADDR, DEFAULT_STATEMENT_TICKET_TTL_SECS, EnvValues};
use file::{ServerFileOverlay, load_overlay};
use models::models_config;
use native::native_optimizer_config;
pub use native::{
    ALGEON_SERVER_EXECUTION_MODE_ENV, ALGEON_SERVER_FINAL_PLAN_REQUIREMENT_ENV, ALGEON_SERVER_NATIVE_CONFIG_OPTIONS,
    ALGEON_SERVER_NATIVE_MAX_RETRY_ATTEMPTS_ENV, ALGEON_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV,
    ALGEON_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, ALGEON_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT_ENV,
    ALGEON_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV, ALGEON_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV,
    AlgeonServerNativeConfigOption, ServerNativeConfigScope,
};
use observability::server_observability_config;
pub use observability::{
    ALGEON_SERVER_OBSERVABILITY_ACTIVE_LOGICAL_CAPTURE_MAX_BYTES_ENV,
    ALGEON_SERVER_OBSERVABILITY_GLOBAL_SEALED_MAX_BYTES_ENV, ALGEON_SERVER_OBSERVABILITY_GLOBAL_SEALED_MAX_COUNT_ENV,
    ALGEON_SERVER_OBSERVABILITY_PER_QUERY_MAX_BYTES_ENV, ALGEON_SERVER_OBSERVABILITY_PER_QUERY_MAX_RECORDS_ENV,
    ALGEON_SERVER_OBSERVABILITY_SEALED_TTL_SECS_ENV, ALGEON_SERVER_OBSERVABILITY_SHUTDOWN_GRACE_SECS_ENV,
    ServerObservabilityConfig,
};
use workspace::workspace_overlay_config;
pub use workspace::{
    ALGEON_SERVER_WORKSPACE_BACKING_ALIASES_ENV, ALGEON_SERVER_WORKSPACE_BACKING_CATALOG_ENV,
    ALGEON_SERVER_WORKSPACE_BACKING_SCHEMA_ENV, ALGEON_SERVER_WORKSPACE_CATALOG_ENV,
    ALGEON_SERVER_WORKSPACE_SCHEMA_ENV, WorkspaceOverlayConfig,
};

// The resource-only admission contract has one configuration surface. These
// historical environment switches must fail closed instead of silently starting a deployment with
// different defaults. Keep this list centrally adjacent to the first env
// read so every server configuration path applies the same migration gate.
const REMOVED_LEGACY_SERVER_ENVIRONMENT_VARIABLES: &[&str] = &[
    "ALGEON_SERVER_ADMISSION_MAX_ACTIVE_ATTEMPTS_PER_DEVICE",
    "ALGEON_SERVER_ADMISSION_ATTEMPT_FLOOR_BYTES",
    "ALGEON_SERVER_ADMISSION_SOURCE_WORK_PER_ATTEMPT",
    "ALGEON_SERVER_ADMISSION_MAX_OVERTAKES_PER_WAITER",
    "ALGEON_SERVER_ADMISSION_WORKLOAD_PROFILE",
    "ALGEON_SERVER_ADMISSION_PINNED_DEVICE",
    "ALGEON_SERVER_ADMISSION_MAX_QUERIES_PER_DEVICE",
    "ALGEON_SERVER_ADMISSION_WAIT_TIMEOUT_MS",
    "ALGEON_SERVER_MAX_CONCURRENT_QUERIES",
    "ALGEON_SERVER_MAX_IN_FLIGHT",
    "ALGEON_SERVER_NATIVE_MEMORY_POLICY",
    "ALGEON_SERVER_NATIVE_PIPELINE_EXECUTION_MODE",
    "ALGEON_SERVER_NATIVE_DEVICE_BUDGET_BYTES",
    "ALGEON_SERVER_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS",
    "ALGEON_SERVER_CUGRAPH_DIRECTED",
    "ALGEON_SERVER_CUGRAPH_RENUMBER",
    "ALGEON_SERVER_NATIVE_SOURCE_CACHE_MAX_RETAINED_BYTES",
    "ALGEON_SERVER_NATIVE_GRAPH_CACHE_MAX_RETAINED_BYTES",
    "ALGEON_SERVER_NATIVE_QUERY_MEMORY_DOMAIN_CEILING_MULTIPLIER",
    "ALGEON_SERVER_DEVICE_BUDGET_FRACTION",
];

// The Iceberg catalog integration was deleted from the server. Its settings
// fail closed for the same reason as the legacy switches: a deployment that
// still sets them would otherwise start without the tables it expects.
const REMOVED_ICEBERG_SERVER_ENVIRONMENT_VARIABLES: &[&str] = &[
    "ALGEON_ICEBERG_CATALOG_KIND",
    "ALGEON_ICEBERG_CATALOG_NAME",
    "ALGEON_ICEBERG_NAMESPACE",
    "ALGEON_ICEBERG_TABLES",
    "ALGEON_ICEBERG_GLUE_CATALOG_NAME",
    "ALGEON_ICEBERG_WAREHOUSE",
    "ALGEON_ICEBERG_GLUE_CATALOG_ID",
    "ALGEON_ICEBERG_GLUE_URI",
    "ALGEON_ICEBERG_REST_URI",
    "ALGEON_ICEBERG_REST_PREFIX",
    "ALGEON_ICEBERG_REST_TOKEN",
    "ALGEON_ICEBERG_REST_CATALOG_HANDLE",
    "ALGEON_ICEBERG_S3_ENDPOINT",
    "ALGEON_ICEBERG_S3_REGION",
    "ALGEON_ICEBERG_S3_PATH_STYLE",
    "ALGEON_ICEBERG_S3_CREDENTIAL_SOURCE",
    "ALGEON_ICEBERG_SCAN_PLANNING_TIMEOUT_SECS",
    "ALGEON_LAKEHOUSE_CACHE_ROOT",
    "ALGEON_LAKEHOUSE_CACHE_READ",
    "ALGEON_LAKEHOUSE_CACHE_POLICY",
    "ALGEON_LAKEHOUSE_CACHE_MAX_BYTES",
];

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub bind_addr: SocketAddr,
    /// Queue-only admission policy. Device-local bounds live in
    /// `gpu_device_profiles` and are never duplicated here.
    pub queue_policy: AlgeonGpuQueuePolicy,
    /// Complete ordinal-keyed resource envelopes for the selected GPUs.
    /// The server never broadcasts a scalar admission setting across this
    /// collection.
    pub gpu_device_profiles: Vec<AlgeonGpuDeviceProfile>,
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
    pub native_optimizer_config: AlgeonNativeOptimizerConfig,
    /// Authentication mode applied to incoming Flight SQL RPCs. Loaded from
    /// the optional TOML overlay and disabled by default.
    pub auth: AuthConfig,
    /// Optional mutable workspace overlay that can read through to a
    /// configured source schema while keeping all DDL local to the workspace.
    pub workspace: Option<WorkspaceOverlayConfig>,
    pub cugraph: Option<CugraphServerConfig>,
    /// Models root for `cuml_forest_predict`, from the `[models]` table.
    /// `None` leaves the function unavailable with `model_root_unconfigured`.
    pub models: Option<CumlModelStoreConfig>,
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
        reject_removed_server_environment(
            &mut values,
            REMOVED_LEGACY_SERVER_ENVIRONMENT_VARIABLES,
            "this server setting was removed by the resource-only admission contract; remove it",
        )?;
        let bind_addr = values.socket_addr(ALGEON_SERVER_BIND_ENV, DEFAULT_BIND_ADDR)?;
        let overlay_path = values.string(ALGEON_SERVER_CONFIG_FILE_ENV).map(PathBuf::from);
        let overlay = match overlay_path {
            Some(path) => load_overlay(&path)?,
            None => ServerFileOverlay::default(),
        };
        let gpu_devices = server_gpu_devices(&mut values)?;
        let queue_policy = server_queue_policy(&mut values, &overlay.admission)?;
        let statement_ticket_ttl = checked_config_duration(
            ALGEON_SERVER_STATEMENT_TICKET_TTL_SECS_ENV,
            values.positive_u64(ALGEON_SERVER_STATEMENT_TICKET_TTL_SECS_ENV, DEFAULT_STATEMENT_TICKET_TTL_SECS)?,
        )?;
        let observability = server_observability_config(&mut values, &overlay.observability)?;
        let native_optimizer_config = native_optimizer_config(&mut values, &overlay.native)?;
        let gpu_device_profiles = server_device_profiles(&overlay.admission, &gpu_devices)?;
        reject_removed_server_environment(
            &mut values,
            REMOVED_ICEBERG_SERVER_ENVIRONMENT_VARIABLES,
            "this server setting belonged to the removed Iceberg integration; remove it",
        )?;

        let cugraph = cugraph_config(&mut values)?;

        let workspace = workspace_overlay_config(&mut values)?;
        let models = models_config(&overlay.models)?;

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
            cugraph,
            models,
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
                ALGEON_SERVER_GPU_DEVICES_ENV,
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
                "must configure every device selected by ALGEON_SERVER_GPU_DEVICES",
            )
            .with_fact("reason", "missing_devices")
            .with_fact("device_ordinals", format_device_ordinals(&missing)));
        }
        let extra = configured.difference(&selected).copied().collect::<Vec<_>>();
        if !extra.is_empty() {
            return Err(ServerError::invalid_config(
                "admission.device_profiles",
                "must not configure a device outside ALGEON_SERVER_GPU_DEVICES",
            )
            .with_fact("reason", "unconfigured_devices")
            .with_fact("device_ordinals", format_device_ordinals(&extra)));
        }
        Ok(())
    }

    pub(crate) fn resolve_gpu_devices(&self) -> Result<Vec<AlgeonGpuDeviceConfig>> {
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
                AlgeonGpuDeviceConfig::new(profile.clone(), ownership)
            })
            .collect();
        Ok(devices)
    }
}

fn format_device_ordinals(device_ordinals: &[usize]) -> String {
    device_ordinals.iter().map(usize::to_string).collect::<Vec<_>>().join(",")
}

fn reject_removed_server_environment(
    values: &mut EnvValues<'_>,
    names: &[&'static str],
    reason: &'static str,
) -> Result<()> {
    for name in names.iter().copied() {
        if values.string(name).is_some() {
            return Err(ServerError::invalid_config(name, reason));
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
