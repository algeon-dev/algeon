//! Process-scoped GPU backend facade for embedded DataFusion sessions.
//!
//! Build one backend with [`NexusGpuBackend::builder`], install it on
//! caller-owned session builders, and close it explicitly with
//! [`NexusGpuBackend::close_and_wait`] or
//! [`NexusGpuBackend::close_and_wait_async`].

#[cfg(test)]
use std::sync::{Mutex, MutexGuard};
use std::{collections::BTreeMap, sync::Arc, time::Duration};

use datafusion::execution::{SessionState, SessionStateBuilder};
#[cfg(feature = "qualification")]
pub use nexus_query_engine::runtime::DeviceQuarantineInjection;
use nexus_query_engine::runtime::{
    DeploymentIoCapability, GdsDriverProbeStatus, GpuBackendControllerDraft, GpuBackendCore, GpuBackendCoreDraft,
    GpuBackendSession, KvikioCompatModeStatus, MemoryControlPolicy, NvidiaFsProbeStatus, QueryExecutionConfig,
    QueueBounds,
};
pub use nexus_query_engine::runtime::{DeviceCapabilities, GpuDeviceMemoryLimit, GpuMemoryOwnership};
/// Canonical device profile bound to an explicit process-scoped GPU memory
/// ownership promise.
///
/// These are the runtime-owned device construction types under application
/// facing Nexus names: one shared type identity, no second defaults or
/// validation source in this adapter.
pub use nexus_query_engine::runtime::{
    GpuBackendDeviceConfig as NexusGpuDeviceConfig,
    GpuBackendDeviceInstallationBounds as NexusGpuDeviceInstallationBounds,
    GpuBackendDeviceProfile as NexusGpuDeviceProfile,
};

use crate::{
    error::{Error, ErrorCode, Result},
    native_exec::{NexusNativeExecutionConfig, QueryAdmissionSource},
    observability::{
        BackendMetricsSnapshot, CacheSourceEventSnapshot, DeviceCacheReport, DeviceLedgerReport,
        DeviceMemoryFailureReport, DeviceTelemetrySnapshot, GpuBackendCloseSnapshot, GpuBackendDeviceStartupRecord,
        GpuExecutionPoolSnapshot, GpuExecutionQuerySnapshot, QueryServiceAdmissionSnapshot,
        QueryServiceDeviceAdmissionSnapshot,
    },
    planner::{DataFusionCpuExecutionPolicy, NexusNativeOptimizerConfig, NexusNativePhysicalOptimizerRule},
    session::NexusSessionStateBuilderExt,
};

/// Process-scoped native GPU backend shared by one or more DataFusion sessions.
///
/// Cloning a backend yields another shared handle to the same backend generation,
/// not an independent backend instance. When any clone signals close, all clones
/// and installed sessions observe the same closing gate immediately.
///
/// Installed [`SessionState`] instances retain only a session capability that
/// cannot extend GPU resource ownership. An idle [`datafusion::prelude::SessionContext`]
/// does not hold an active lease and does not need to be dropped before closing the backend.
/// Once closing begins, new attempts and admissions are rejected fail-closed, while
/// in-flight work drains according to contract. Published attempt reports and observation
/// snapshots remain readable after GPU resources have achieved quiescence.
///
/// Dropping the last owning backend handle signals close without waiting for
/// quiescence and cannot report a timeout or cleanup failure. The process
/// owner should explicitly wait through [`Self::close_and_wait`] or
/// [`Self::close_and_wait_async`].
#[derive(Debug, Clone)]
pub struct NexusGpuBackend {
    core: GpuBackendCore,
    session: GpuBackendSession,
    optimizer_config: NexusNativeOptimizerConfig,
    devices: Vec<NexusGpuDeviceConfig>,
    queue_policy: NexusGpuQueuePolicy,
}

/// Queue-only policy for the shared admission queue.
///
/// Device-local capacity, query-limit, concurrency, and source-work settings belong
/// exclusively to [`NexusGpuDeviceProfile`].  Keeping this policy queue-only
/// prevents a scalar from being implicitly broadcast to heterogeneous devices.
///
/// Values are constructed from [`Default`] and adjusted through the `with_*`
/// builders; zero bounds are rejected when the backend is built.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct NexusGpuQueuePolicy {
    max_queued_attempts: usize,
    max_queue_wait: Duration,
}

impl NexusGpuQueuePolicy {
    /// Maximum queued admission waiters. Default 64.
    pub fn max_queued_attempts(&self) -> usize {
        self.max_queued_attempts
    }

    /// Maximum time a waiter may queue before a structured timeout. Default 30s.
    pub fn max_queue_wait(&self) -> Duration {
        self.max_queue_wait
    }

    pub fn with_max_queued_attempts(mut self, max_queued_attempts: usize) -> Self {
        self.max_queued_attempts = max_queued_attempts;
        self
    }

    pub fn with_max_queue_wait(mut self, max_queue_wait: Duration) -> Self {
        self.max_queue_wait = max_queue_wait;
        self
    }
}

impl Default for NexusGpuQueuePolicy {
    fn default() -> Self {
        let bounds = QueueBounds::default();
        Self { max_queued_attempts: bounds.max_waiters, max_queue_wait: bounds.max_wait }
    }
}

/// Builder for [`NexusGpuBackend`].
#[derive(Debug, Clone, Default)]
pub struct NexusGpuBackendBuilder {
    optimizer_config: NexusNativeOptimizerConfig,
    devices: Vec<NexusGpuDeviceConfig>,
    queue_policy: NexusGpuQueuePolicy,
    attempt_completion_queue_capacity: Option<usize>,
}

impl NexusGpuBackend {
    pub fn builder() -> NexusGpuBackendBuilder {
        NexusGpuBackendBuilder::default()
    }

    /// Installs the backend-managed DataFusion integration on `builder`.
    ///
    /// The installation binds the native optimizer to this backend's admission
    /// authority, and installs the GPU coverage UDTF. The resulting session
    /// plans and admits native execution through this backend's ledger.
    ///
    /// This method does not install optional SQL families such as cuVS or
    /// cuGraph. Call [`NexusSessionStateBuilderExt::try_with_cuvs_sql`] or
    /// [`NexusSessionStateBuilderExt::try_with_cugraph_sql`] on the returned
    /// builder when those families are required.
    pub fn install_on(&self, builder: SessionStateBuilder) -> Result<SessionStateBuilder> {
        install_backend_optimizer_config(builder, &self.optimizer_config)
    }

    /// Performs the one controlled device-quarantine transition the operator
    /// qualification tool is allowed to drive. It closes admission on exactly
    /// one live device ledger through the same fleet admission gate placement
    /// uses; it neither fabricates a CUDA fault nor mutates cuDF global
    /// health, and it is not a production backend mode.
    #[cfg(feature = "qualification")]
    pub fn quarantine_device_for_qualification(&self, device_ordinal: usize) -> Result<DeviceQuarantineInjection> {
        self.core.fault_injector().quarantine_device(device_ordinal).map_err(|error| {
            Error::from_native(
                "datafusion_nexus.backend.quarantine_device_for_qualification",
                "controlled device quarantine failed",
                error,
            )
        })
    }

    /// Immutable per-device construction records shared by every clone and
    /// session installed from this backend.
    pub fn device_startup_records(&self) -> &[GpuBackendDeviceStartupRecord] {
        self.core.device_startup_records()
    }

    /// Non-blockingly signals backend shutdown.
    ///
    /// New attempt admissions are rejected immediately, while active attempts drain
    /// cooperatively. Returns `true` if this call initiated the close, or `false`
    /// if the backend was already closing or closed.
    pub fn close(&self) -> bool {
        self.core.close()
    }

    /// Returns the current two-scope backend close state without altering state or
    /// initiating a close.
    ///
    /// Callers that track shutdown progression can inspect whether GPU resources
    /// and observation records have quiesced without mutating the closing gate.
    pub fn close_snapshot(&self) -> GpuBackendCloseSnapshot {
        self.core.close_snapshot()
    }

    /// Canonical shutdown path for embedded and process owners.
    ///
    /// Initiates backend shutdown and blocks until both GPU resources and observation
    /// records achieve quiescence, or until `timeout` elapses.
    pub fn close_and_wait(&self, timeout: Duration) -> Result<GpuBackendCloseSnapshot> {
        self.core.close_and_wait(timeout).map_err(|error| {
            Error::from_native("datafusion_nexus.backend.close", "failed to close GPU backend before timeout", error)
        })
    }

    /// Asynchronous canonical shutdown path for embedded owners.
    ///
    /// Initiates backend shutdown and waits for GPU-resource and observation
    /// quiescence without blocking the calling async executor worker, or
    /// until `timeout` elapses. It shares the close coordinator and returns
    /// the same snapshot contract as [`Self::close_and_wait`].
    pub async fn close_and_wait_async(&self, timeout: Duration) -> Result<GpuBackendCloseSnapshot> {
        self.core.close_and_wait_async(timeout).await.map_err(|error| {
            Error::from_native("datafusion_nexus.backend.close", "failed to close GPU backend before timeout", error)
        })
    }

    /// Shutdown path for callers that already hold a shared absolute deadline.
    ///
    /// Initiates backend shutdown and blocks until quiescence or until `deadline`
    /// is reached in the monotonic clock domain.
    pub fn close_and_wait_until(&self, deadline: std::time::Instant) -> Result<GpuBackendCloseSnapshot> {
        self.core.close_and_wait_until(deadline).map_err(|error| {
            Error::from_native("datafusion_nexus.backend.close", "failed to close GPU backend before deadline", error)
        })
    }

    /// Returns one coherent, fixed-cardinality backend metrics snapshot.
    ///
    /// Applications own any exporter or retention policy.  This facade only
    /// exposes the backend generation snapshot produced by the engine; it
    /// does not install an exporter or create an HTTP metrics endpoint.
    pub fn backend_metrics_snapshot(&self) -> BackendMetricsSnapshot {
        self.session.backend_metrics_snapshot()
    }

    pub fn admission_snapshot(&self) -> Result<QueryServiceAdmissionSnapshot> {
        self.session.admission_snapshot().map_err(|error| {
            Error::from_native(
                "datafusion_nexus.backend.admission_snapshot",
                "failed to read backend admission snapshot",
                error,
            )
        })
    }

    pub fn queue_policy(&self) -> &NexusGpuQueuePolicy {
        &self.queue_policy
    }

    pub fn device_admission_snapshots(&self) -> Result<Vec<QueryServiceDeviceAdmissionSnapshot>> {
        self.session.device_admission_snapshots().map_err(|error| {
            Error::from_native(
                "datafusion_nexus.backend.device_admission_snapshots",
                "failed to read per-device admission snapshots",
                error,
            )
        })
    }

    /// Returns the admission source of truth; telemetry is diagnostic only.
    pub fn device_ledger_reports(&self) -> Result<Vec<DeviceLedgerReport>> {
        self.session.device_ledger_reports().map_err(|error| {
            Error::from_native(
                "datafusion_nexus.backend.device_ledger_reports",
                "failed to read device ledger reports",
                error,
            )
        })
    }

    /// Returns the physical resident-cache allocation-domain observations.
    /// They are diagnostic cross-checks; admission stays owned by the ledger.
    pub fn device_cache_reports(&self) -> Result<Vec<DeviceCacheReport>> {
        self.session.device_cache_reports().map_err(|error| {
            Error::from_native(
                "datafusion_nexus.backend.device_cache_reports",
                "failed to read device cache reports",
                error,
            )
        })
    }

    /// Returns bounded source-cache events from the cache owners themselves.
    /// These observations support correlation diagnostics; the ledger remains
    /// the sole admission and residency authority.
    pub fn device_cache_source_event_snapshots(&self) -> Result<Vec<CacheSourceEventSnapshot>> {
        self.session.device_cache_source_event_snapshots().map_err(|error| {
            Error::from_native(
                "datafusion_nexus.backend.device_cache_source_event_snapshots",
                "failed to read device cache source events",
                error,
            )
        })
    }

    pub fn device_memory_failure_reports(&self) -> Result<Vec<DeviceMemoryFailureReport>> {
        self.session.device_memory_failure_reports().map_err(|error| {
            Error::from_native(
                "datafusion_nexus.backend.device_memory_failure_reports",
                "failed to read device memory failure reports",
                error,
            )
        })
    }

    /// Returns diagnostic CUDA/NVML samples, never admission state.
    pub fn device_telemetry_snapshots(&self) -> Result<Vec<DeviceTelemetrySnapshot>> {
        self.session.device_telemetry_snapshots().map_err(|error| {
            Error::from_native(
                "datafusion_nexus.backend.device_telemetry_snapshots",
                "failed to read device telemetry snapshots",
                error,
            )
        })
    }

    pub fn gpu_execution_pool_reports(&self) -> Result<Vec<GpuExecutionPoolSnapshot>> {
        self.session.gpu_execution_pool_reports().map_err(|error| {
            Error::from_native(
                "datafusion_nexus.backend.gpu_execution_pool_reports",
                "failed to read GPU execution pool reports",
                error,
            )
        })
    }

    pub fn gpu_execution_query_report(&self) -> Result<GpuExecutionQuerySnapshot> {
        self.session.gpu_execution_query_report().map_err(|error| {
            Error::from_native(
                "datafusion_nexus.backend.gpu_execution_query_report",
                "failed to read GPU execution query report",
                error,
            )
        })
    }

    pub fn same_backend(&self, other: &Self) -> bool {
        self.core.identity() == other.core.identity()
    }

    pub(crate) fn session_capability(&self) -> &GpuBackendSession {
        &self.session
    }

    /// Whether the session's installed native optimizer is bound to this
    /// backend, so execution planned through it is admitted by this backend.
    ///
    /// Returns false when the session has no native optimizer installed, the
    /// installed optimizer is bound to another backend, or its execution
    /// config carries no query admission authority.
    pub fn is_session_state_bound(&self, state: &SessionState) -> bool {
        crate::session::installed_nexus_native_optimizer_config(state)
            .is_some_and(|optimizer_config| self.is_execution_config_bound(optimizer_config.native_execution_config()))
    }

    /// Whether the config's query admission source is bound to this backend's
    /// session, so execution through it is admitted by this backend.
    pub(crate) fn is_execution_config_bound(&self, config: &NexusNativeExecutionConfig) -> bool {
        config.query_admission_source_matches(&self.session)
    }

    pub fn optimizer_config(&self) -> &NexusNativeOptimizerConfig {
        &self.optimizer_config
    }

    /// Exact-request BFS coalescing counters shared by sessions from this backend.
    #[cfg(feature = "cugraph")]
    pub fn cugraph_bfs_singleflight_snapshot(&self) -> crate::native_exec::GraphBfsSingleflightSnapshot {
        self.optimizer_config.native_execution_config().graph_bfs_singleflight_snapshot()
    }

    pub fn devices(&self) -> &[NexusGpuDeviceConfig] {
        &self.devices
    }

    fn build_bounded(
        optimizer_config: NexusNativeOptimizerConfig,
        configured_devices: Vec<NexusGpuDeviceConfig>,
        queue_policy: NexusGpuQueuePolicy,
        attempt_completion_queue_capacity: Option<usize>,
    ) -> Result<Self> {
        let validated = validated_backend_construction(optimizer_config, configured_devices, queue_policy)?;
        let draft = match attempt_completion_queue_capacity {
            Some(capacity) => validated.core_draft.with_attempt_completion_queue_capacity(capacity),
            None => validated.core_draft,
        };
        let core = GpuBackendCore::builder(draft).build().map_err(|error| {
            Error::from_native(
                "datafusion_nexus.backend.gpu_core",
                "failed to initialize ownership-enforcing GPU backend core",
                error,
            )
        })?;
        let session = core.session_capability();
        let max_device_total_memory_bytes = session.max_device_total_memory_bytes().map_err(|error| {
            Error::from_native(
                "datafusion_nexus.backend.device_projection",
                "failed to read backend device facts",
                error,
            )
        })?;
        // Keep this as the unadmitted planning template. Every execution
        // context is instead materialized from the device selected by the
        // per-device ledger and its immutable grant.
        let execution_config = validated
            .optimizer_config
            .native_execution_config()
            .clone()
            .with_query_admission_source(QueryAdmissionSource::new(session.clone(), max_device_total_memory_bytes));
        let optimizer_config = validated.optimizer_config.with_native_execution_config(execution_config);
        Ok(NexusGpuBackend {
            core,
            session,
            optimizer_config,
            devices: validated.devices,
            queue_policy: validated.queue_policy,
        })
    }
}

fn install_backend_optimizer_config(
    builder: SessionStateBuilder,
    optimizer_config: &NexusNativeOptimizerConfig,
) -> Result<SessionStateBuilder> {
    let rule = Arc::new(NexusNativePhysicalOptimizerRule::new(optimizer_config.clone()));
    Ok(builder.with_nexus_native_rule(rule))
}

/// Validated backend-builder input shared by production construction and root
/// unit tests.
///
/// This is the first construction segment: normalize and validate profiles,
/// ownership, queue, and execution config without acquiring the owner lease or
/// creating CUDA resources. Only production `build()` enters the second
/// segment (lease acquisition and `GpuBackendCore` construction).
#[derive(Debug)]
struct ValidatedBackendConstruction {
    optimizer_config: NexusNativeOptimizerConfig,
    devices: Vec<NexusGpuDeviceConfig>,
    queue_policy: NexusGpuQueuePolicy,
    core_draft: GpuBackendCoreDraft,
}

fn validated_backend_construction(
    optimizer_config: NexusNativeOptimizerConfig,
    configured_devices: Vec<NexusGpuDeviceConfig>,
    queue_policy: NexusGpuQueuePolicy,
) -> Result<ValidatedBackendConstruction> {
    let devices = resolve_device_configs(configured_devices)?;
    validate_queue_policy(&queue_policy)?;
    let execution_config = optimizer_config.native_execution_config().clone();
    let service_config = backend_query_execution_config(&execution_config)?;
    let queue_bounds = backend_queue_bounds(&queue_policy);
    let controller = GpuBackendControllerDraft::new(service_config, queue_bounds);
    let core_draft = GpuBackendCoreDraft::new(devices.clone(), controller, nongds_deployment());
    core_draft.validate().map_err(|error| {
        Error::from_native(
            "datafusion_nexus.backend.gpu_core",
            "invalid ownership-enforcing GPU backend core configuration",
            error,
        )
    })?;
    Ok(ValidatedBackendConstruction { optimizer_config, devices, queue_policy, core_draft })
}

/// Test fixture for paths that genuinely execute an installed GPU backend. The
/// process lease remains production-exclusive; tests serialize construction
/// and deterministically close the backend before releasing the fixture lock.
#[cfg(test)]
pub(crate) struct InstalledTestGpuBackend {
    backend: Option<NexusGpuBackend>,
    _serial: MutexGuard<'static, ()>,
}

#[cfg(test)]
impl InstalledTestGpuBackend {
    pub(crate) fn new() -> Self {
        Self::with_config(
            NexusNativeExecutionConfig::bounded(256 * 1024 * 1024, 64 * 1024 * 1024, 2, 3, 1),
            NexusGpuDeviceProfile::new(0)
                .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 256 * 1024 * 1024 }),
            NexusGpuQueuePolicy::default(),
        )
    }

    /// Builds a serialized real backend with test-selected immutable resource
    /// and queue envelopes. Tests retain this fixture for the full lifetime of
    /// every config or ticket cloned from its backend.
    pub(crate) fn with_config(
        native_execution_config: NexusNativeExecutionConfig,
        device_profile: NexusGpuDeviceProfile,
        queue_policy: NexusGpuQueuePolicy,
    ) -> Self {
        Self::with_config_and_datafusion_cpu_policy(
            native_execution_config,
            device_profile,
            queue_policy,
            DataFusionCpuExecutionPolicy::Allow,
        )
    }

    /// Builds a serialized real backend with an explicit completed-plan
    /// DataFusion CPU policy. This keeps integration tests on the same frozen
    /// backend-installation path as production cells that add a final-plan gate.
    pub(crate) fn with_config_and_datafusion_cpu_policy(
        native_execution_config: NexusNativeExecutionConfig,
        device_profile: NexusGpuDeviceProfile,
        queue_policy: NexusGpuQueuePolicy,
        datafusion_cpu_execution_policy: DataFusionCpuExecutionPolicy,
    ) -> Self {
        static SERIAL: Mutex<()> = Mutex::new(());

        let serial = SERIAL.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let device = NexusGpuDeviceConfig::new(device_profile, GpuMemoryOwnership::WholeDeviceExclusive);
        let backend = NexusGpuBackend::builder()
            .with_native_execution_config(native_execution_config)
            .with_datafusion_cpu_execution_policy(datafusion_cpu_execution_policy)
            .devices([device])
            .queue_policy(queue_policy)
            .build()
            .expect("test backend configuration must be valid");
        Self { backend: Some(backend), _serial: serial }
    }

    pub(crate) fn backend(&self) -> &NexusGpuBackend {
        self.backend.as_ref().expect("test backend remains live until fixture drop")
    }

    pub(crate) fn optimizer_config(&self) -> &NexusNativeOptimizerConfig {
        self.backend().optimizer_config()
    }

    pub(crate) fn admit_query(&self) -> nexus_query_engine::error::Result<nexus_query_engine::runtime::QueryHandle> {
        self.backend().session_capability().admit_query()
    }

    pub(crate) fn install_on(&self, builder: SessionStateBuilder) -> Result<SessionStateBuilder> {
        self.backend().install_on(builder)
    }
}

#[cfg(test)]
impl Default for InstalledTestGpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
impl InstalledTestGpuBackend {
    pub(crate) fn close_and_wait(mut self, timeout: Duration) -> Result<GpuBackendCloseSnapshot> {
        let backend = self.backend.take().expect("test backend remains live until fixture drop");
        backend.close_and_wait(timeout)
    }
}

#[cfg(test)]
impl Drop for InstalledTestGpuBackend {
    fn drop(&mut self) {
        let Some(backend) = self.backend.take() else {
            return;
        };
        let close = backend.close_and_wait(Duration::from_secs(10));
        drop(backend);
        if let Err(error) = close
            && !std::thread::panicking()
        {
            panic!("root GPU unit test backend failed to close: {error}");
        }
    }
}

impl NexusGpuBackendBuilder {
    pub fn with_optimizer_config(mut self, optimizer_config: NexusNativeOptimizerConfig) -> Self {
        self.optimizer_config = optimizer_config;
        self
    }

    /// Retain bounded reports whose completion receivers were abandoned.
    pub fn with_attempt_completion_queue_capacity(mut self, capacity: usize) -> Self {
        self.attempt_completion_queue_capacity = Some(capacity);
        self
    }

    /// Set the completed-plan DataFusion CPU policy enforced after native
    /// rewriting for every backend-installed physical plan.
    pub fn with_datafusion_cpu_execution_policy(mut self, policy: DataFusionCpuExecutionPolicy) -> Self {
        self.optimizer_config = self.optimizer_config.with_datafusion_cpu_execution_policy(policy);
        self
    }

    /// Replaces the canonical execution settings copied into admitted plans.
    ///
    /// The execution config is an unadmitted planning template. It never
    /// selects a device or contributes device capacity; those facts are owned
    /// exclusively by [`NexusGpuDeviceProfile`].
    pub fn with_native_execution_config(mut self, native_execution_config: NexusNativeExecutionConfig) -> Self {
        self.optimizer_config = self.optimizer_config.with_native_execution_config(native_execution_config);
        self
    }

    /// Supplies the complete ordinal-keyed device declaration set.
    pub fn devices(mut self, devices: impl IntoIterator<Item = NexusGpuDeviceConfig>) -> Self {
        self.devices = devices.into_iter().collect();
        self
    }

    /// Sets the shared queue-only GPU admission policy.
    pub fn queue_policy(mut self, queue_policy: NexusGpuQueuePolicy) -> Self {
        self.queue_policy = queue_policy;
        self
    }

    pub fn build(self) -> Result<NexusGpuBackend> {
        NexusGpuBackend::build_bounded(
            self.optimizer_config,
            self.devices,
            self.queue_policy,
            self.attempt_completion_queue_capacity,
        )
    }
}

fn resolve_device_configs(configured_devices: Vec<NexusGpuDeviceConfig>) -> Result<Vec<NexusGpuDeviceConfig>> {
    if configured_devices.is_empty() {
        return Err(invalid_backend_config(
            "devices",
            "backend construction requires at least one explicit ordinal-keyed device declaration",
        )
        .with_fact("reason", "missing_devices"));
    }

    let mut by_ordinal = BTreeMap::new();
    for device in configured_devices {
        let device_ordinal = device.device_ordinal();
        if by_ordinal.insert(device_ordinal, device).is_some() {
            return Err(invalid_backend_config(
                "devices",
                "must contain exactly one configured declaration for each device ordinal",
            )
            .with_fact("reason", "duplicate_device")
            .with_fact("device_ordinal", device_ordinal));
        }
    }
    Ok(by_ordinal.into_values().collect())
}

fn backend_query_execution_config(execution_config: &NexusNativeExecutionConfig) -> Result<QueryExecutionConfig> {
    let MemoryControlPolicy::Bounded {
        device_budget_bytes: _,
        max_source_chunk_bytes,
        max_row_groups_per_chunk,
        max_retry_attempts,
        min_retry_chunk_row_groups,
    } = execution_config.memory_control_policy()
    else {
        return Err(invalid_backend_config(
            "native_execution_config",
            "NexusGpuBackend admission requires bounded native execution",
        ));
    };
    validate_positive_u64("native_execution_config.max_source_chunk_bytes", max_source_chunk_bytes)?;
    validate_positive_usize("native_execution_config.max_row_groups_per_chunk", max_row_groups_per_chunk)?;
    validate_positive_usize("native_execution_config.max_retry_attempts", max_retry_attempts)?;
    validate_positive_usize("native_execution_config.min_retry_chunk_row_groups", min_retry_chunk_row_groups)?;
    Ok(QueryExecutionConfig {
        lakehouse_cache_fill: execution_config.lakehouse_cache_fill_config().clone(),
        source_storage_profiles: execution_config.source_storage_profiles().to_vec(),
        boundary_policy: execution_config.boundary_policy(),
        source_scheduler_policy: execution_config.source_scheduler_policy(),
        source_chunk_planning_mode: execution_config.source_chunk_planning_mode(),
        statistical_aggregate_execution_mode: execution_config.statistical_aggregate_execution_mode(),
        dynamic_filter_execution_options: execution_config.dynamic_filter_execution_options(),
        keyed_hash_join_strategy_options: execution_config.keyed_hash_join_strategy_options(),
        aggregate_strategy_execution_options: execution_config.aggregate_strategy_execution_options(),
        source_chunk_planning_options: execution_config.source_chunk_planning_options(),
        max_source_chunk_bytes,
        max_row_groups_per_chunk,
        max_retry_attempts,
        min_retry_chunk_row_groups,
    })
}

fn backend_queue_bounds(queue_policy: &NexusGpuQueuePolicy) -> QueueBounds {
    QueueBounds { max_waiters: queue_policy.max_queued_attempts, max_wait: queue_policy.max_queue_wait }
}

fn validate_queue_policy(policy: &NexusGpuQueuePolicy) -> Result<()> {
    validate_positive_usize("max_queued_attempts", policy.max_queued_attempts)?;
    if policy.max_queue_wait.is_zero() {
        return Err(invalid_backend_config("max_queue_wait", "must be greater than zero"));
    }
    Ok(())
}

fn validate_positive_u64(field: &'static str, value: u64) -> Result<()> {
    if value == 0 {
        return Err(invalid_backend_config(field, "must be greater than zero"));
    }
    Ok(())
}

fn validate_positive_usize(field: &'static str, value: usize) -> Result<()> {
    if value == 0 {
        return Err(invalid_backend_config(field, "must be greater than zero"));
    }
    Ok(())
}

fn invalid_backend_config(field: &'static str, message: impl Into<String>) -> Error {
    Error::new(ErrorCode::InvalidRequest, "datafusion_nexus.backend.config", message).with_fact("field", field)
}

fn nongds_deployment() -> DeploymentIoCapability {
    DeploymentIoCapability::from_probe_inputs(
        NvidiaFsProbeStatus::DriverMissing,
        KvikioCompatModeStatus::Unset,
        GdsDriverProbeStatus::ProbeUnimplemented,
    )
}

#[cfg(test)]
#[path = "backend_tests.rs"]
mod tests;
