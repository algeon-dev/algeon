use std::{
    net::SocketAddr,
    path::PathBuf,
    time::{Duration, Instant},
};

use query_planning::{execution_contracts::SourceChunkPlanningMode, mode::StatisticalAggregateExecutionMode};
use query_runtime::runtime::SourceSchedulerPolicy;
use serde::Deserialize;

use super::{
    auth::AuthConfig,
    error::{Result, ServerError},
};
use crate::server::{
    backend::{AlgeonGpuDeviceConfig, AlgeonGpuDeviceProfile, AlgeonGpuQueuePolicy, GpuMemoryOwnership},
    coordinator::QueryCoordinatorConfig,
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
mod s3;
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
    ALGEON_SERVER_BIND_ENV, ALGEON_SERVER_CONFIG_FILE_ENV, ALGEON_SERVER_HTTP2_KEEPALIVE_INTERVAL_SECS_ENV,
    ALGEON_SERVER_HTTP2_KEEPALIVE_TIMEOUT_SECS_ENV, ALGEON_SERVER_LOG_ENV, ALGEON_SERVER_STATEMENT_TICKET_TTL_SECS_ENV,
};
use env::{DEFAULT_BIND_ADDR, DEFAULT_STATEMENT_TICKET_TTL_SECS, EnvValues};
use file::{ServerCugraphFileConfig, ServerFileOverlay, load_overlay};
use models::models_config;
use native::native_optimizer_config;
pub use native::{
    ALGEON_SERVER_EXECUTION_MODE_ENV, ALGEON_SERVER_NATIVE_CONFIG_OPTIONS,
    ALGEON_SERVER_NATIVE_LAKEHOUSE_FOOTER_PRUNING_ENV, ALGEON_SERVER_NATIVE_MAX_RETRY_ATTEMPTS_ENV,
    ALGEON_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, ALGEON_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV,
    ALGEON_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT_ENV, ALGEON_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV,
    ALGEON_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV, AlgeonServerNativeConfigOption,
    ServerNativeConfigScope,
};
use observability::server_observability_config;
pub use observability::{
    ALGEON_SERVER_OBSERVABILITY_ACTIVE_LOGICAL_CAPTURE_MAX_BYTES_ENV,
    ALGEON_SERVER_OBSERVABILITY_GLOBAL_SEALED_MAX_BYTES_ENV, ALGEON_SERVER_OBSERVABILITY_GLOBAL_SEALED_MAX_COUNT_ENV,
    ALGEON_SERVER_OBSERVABILITY_PER_QUERY_MAX_BYTES_ENV, ALGEON_SERVER_OBSERVABILITY_PER_QUERY_MAX_RECORDS_ENV,
    ALGEON_SERVER_OBSERVABILITY_SEALED_TTL_SECS_ENV, ALGEON_SERVER_OBSERVABILITY_SHUTDOWN_GRACE_SECS_ENV,
    ServerObservabilityConfig,
};
use s3::s3_storage;
pub use s3::{
    ALGEON_SERVER_S3_ENDPOINT_ENV, ALGEON_SERVER_S3_PATH_STYLE_ENV, ALGEON_SERVER_S3_REGION_ENV,
    ALGEON_SERVER_S3_UNSIGNED_ENV,
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

// `native_required` replaced the separate completed-plan requirement; a
// deployment that still sets it would otherwise allow DataFusion CPU plans.
const REMOVED_EXECUTION_POLICY_SERVER_ENVIRONMENT_VARIABLES: &[&str] = &["ALGEON_SERVER_FINAL_PLAN_REQUIREMENT"];

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
    "ALGEON_ICEBERG_FOOTER_PRUNING",
    "ALGEON_LAKEHOUSE_CACHE_ROOT",
    "ALGEON_LAKEHOUSE_CACHE_READ",
    "ALGEON_LAKEHOUSE_CACHE_POLICY",
    "ALGEON_LAKEHOUSE_CACHE_MAX_BYTES",
];

/// HTTP/2 transport loss detection. Acknowledging clients may retain unread results.
#[derive(Debug, Clone, Copy)]
pub struct Http2Keepalive {
    /// Time without inbound traffic before a ping is sent.
    pub interval: Duration,
    /// Time to acknowledge a ping before the connection closes.
    pub timeout: Duration,
}

impl Default for Http2Keepalive {
    fn default() -> Self {
        Self { interval: Duration::from_secs(30), timeout: Duration::from_secs(10) }
    }
}

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
    /// Keepalive intervals for accepted HTTP/2 connections.
    pub http2_keepalive: Http2Keepalive,
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
        let keepalive_defaults = Http2Keepalive::default();
        let http2_keepalive = Http2Keepalive {
            interval: checked_config_duration(
                ALGEON_SERVER_HTTP2_KEEPALIVE_INTERVAL_SECS_ENV,
                values.positive_u64(
                    ALGEON_SERVER_HTTP2_KEEPALIVE_INTERVAL_SECS_ENV,
                    keepalive_defaults.interval.as_secs(),
                )?,
            )?,
            timeout: checked_config_duration(
                ALGEON_SERVER_HTTP2_KEEPALIVE_TIMEOUT_SECS_ENV,
                values.positive_u64(
                    ALGEON_SERVER_HTTP2_KEEPALIVE_TIMEOUT_SECS_ENV,
                    keepalive_defaults.timeout.as_secs(),
                )?,
            )?,
        };
        let observability = server_observability_config(&mut values, &overlay.observability)?;
        reject_removed_server_environment(
            &mut values,
            REMOVED_EXECUTION_POLICY_SERVER_ENVIRONMENT_VARIABLES,
            "replaced by ALGEON_SERVER_EXECUTION_MODE=native_required; remove it",
        )?;
        let native_optimizer_config = native_optimizer_config(&mut values, &overlay.native)?;
        let native_optimizer_config = match s3_storage(&mut values, overlay.s3.as_ref())? {
            Some(storage) => {
                let execution = native_optimizer_config.native_execution_config().clone().with_s3_storage(storage);
                native_optimizer_config.with_native_execution_config(execution)
            },
            None => native_optimizer_config,
        };
        let gpu_device_profiles = server_device_profiles(&overlay.admission, &gpu_devices)?;
        reject_removed_server_environment(
            &mut values,
            REMOVED_ICEBERG_SERVER_ENVIRONMENT_VARIABLES,
            "this server setting belonged to the removed Iceberg integration; remove it",
        )?;

        let cugraph = cugraph_config(&mut values, &overlay.cugraph)?;

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
            http2_keepalive,
            observability,
            native_optimizer_config,
            auth: overlay.auth,
            workspace,
            cugraph,
            models,
        };
        config.validate_gpu_device_profiles()?;
        config.validate_admission_within_live_query_capacity()?;
        // Reject a default capture budget that overflows at load time.
        config.active_logical_capture_budget_bytes()?;
        Ok(config)
    }

    /// Logical capture budget of the query coordinator under this config's
    /// admission bounds; see [`ServerObservabilityConfig::active_logical_capture_max_bytes`].
    pub(crate) fn active_logical_capture_budget_bytes(&self) -> Result<u64> {
        self.observability
            .active_logical_capture_budget_bytes(admission_live_attempts(&self.queue_policy, &self.gpu_device_profiles))
    }

    /// Logical capture bytes of every attempt this config's admission bounds
    /// hold at once.
    pub(crate) fn admission_logical_capture_bytes(&self) -> u64 {
        self.observability
            .admission_logical_capture_bytes(admission_live_attempts(&self.queue_policy, &self.gpu_device_profiles))
    }

    /// The coordinator registers at most `MAX_LIVE_QUERIES` live statements of
    /// every kind; admission bounds must stay below that cap. Statements that
    /// hold no admission slot (planning, ticketed, CPU-only, metadata,
    /// finalizing) count against the same cap, so registration can still
    /// refuse a statement while admission has room. Config loading and
    /// `FlightSqlServerBuilder::build` both run this one check.
    pub(crate) fn validate_admission_within_live_query_capacity(&self) -> Result<()> {
        let admission_live_attempts = admission_live_attempts(&self.queue_policy, &self.gpu_device_profiles);
        let max_live_queries = QueryCoordinatorConfig::MAX_LIVE_QUERIES;
        let capacity = u64::try_from(max_live_queries).unwrap_or(u64::MAX);
        if admission_live_attempts < capacity {
            return Ok(());
        }
        // The active attempts the device profiles allow fill registration on
        // their own, or the queue added to them does.
        let active_attempts = admission_live_attempts
            .saturating_sub(u64::try_from(self.queue_policy.max_queued_attempts()).unwrap_or(u64::MAX));
        let field = if active_attempts >= capacity {
            "admission.device_profiles.max_active_attempts"
        } else {
            ALGEON_SERVER_ADMISSION_MAX_QUEUED_ATTEMPTS_ENV
        };
        Err(ServerError::invalid_config(
            field,
            "admission bounds must hold fewer attempts than the server registers live queries",
        )
        .with_fact("reason", "admission_exceeds_live_query_capacity")
        .with_fact("admission_live_attempts", admission_live_attempts)
        .with_fact("max_live_queries", max_live_queries))
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

/// Attempts admission can hold at once: every device's active attempts plus
/// the queue shared by all devices.
fn admission_live_attempts(queue_policy: &AlgeonGpuQueuePolicy, profiles: &[AlgeonGpuDeviceProfile]) -> u64 {
    let attempts = profiles
        .iter()
        .map(AlgeonGpuDeviceProfile::max_active_attempts)
        .fold(queue_policy.max_queued_attempts(), usize::saturating_add);
    u64::try_from(attempts).unwrap_or(u64::MAX)
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
