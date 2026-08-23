//! Process-scoped GPU backend facade for embedded DataFusion sessions.

#[cfg(any(test, feature = "test-support"))]
use std::sync::{Mutex, MutexGuard};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::Duration,
};

use datafusion::{
    common::DataFusionError,
    execution::{SessionState, SessionStateBuilder},
};
#[cfg(any(test, feature = "test-support"))]
use nexus_query_engine::runtime::QueryAdmissionRequest;
use nexus_query_engine::runtime::{
    DeploymentIoCapability, GdsDriverProbeStatus, GpuBackendControllerDraft, GpuBackendCore, GpuBackendCoreDraft,
    GpuBackendDeviceConfig, GpuBackendSession, KvikioCompatModeStatus, MemoryControlPolicy, NvidiaFsProbeStatus,
    QueryExecutionConfig, QueueBounds,
};
pub use nexus_query_engine::runtime::{DeviceCapabilities, GpuDeviceMemoryLimit, GpuMemoryOwnership};
#[cfg(any(test, feature = "test-support", feature = "bench-support"))]
#[doc(hidden)]
pub use nexus_query_engine::runtime::{DeviceQuarantineInjection, GpuBackendFaultInjector};

use crate::{
    error::{DataFusionErrorExt, Error, ErrorCode, Result},
    native_exec::{NexusNativeExecutionConfig, QueryAdmissionSource},
    observability::{
        BackendMetricsSnapshot, CacheSourceEventSnapshot, DeviceCacheReport, DeviceLedgerReport,
        DeviceMemoryFailureReport, DeviceTelemetrySnapshot, GpuBackendCloseSnapshot, GpuBackendDeviceStartupRecord,
        GpuExecutionPoolSnapshot, GpuExecutionQuerySnapshot, QueryServiceAdmissionSnapshot,
        QueryServiceDeviceAdmissionSnapshot,
    },
    planner::{FinalPlanRequirement, NexusNativeOptimizerConfig, NexusNativePhysicalOptimizerRule},
    session::{NexusSessionStateBuilderExt, install_nexus_native_rule_with_session_config},
    session_config::DataFusionNexusSessionConfig,
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
#[derive(Debug, Clone)]
pub struct NexusGpuBackend {
    core: GpuBackendCore,
    session: GpuBackendSession,
    optimizer_config: NexusNativeOptimizerConfig,
    device_ordinals: Vec<usize>,
    device_profiles: Vec<NexusGpuDeviceProfile>,
    device_memory_ownership: Vec<(usize, GpuMemoryOwnership)>,
    queue_policy: NexusGpuQueuePolicy,
}

/// Explicit immutable resource envelope for one CUDA device.
///
/// A profile is keyed by its CUDA ordinal and is projected one-for-one into
/// the engine's [`GpuBackendDeviceConfig`].  It deliberately owns every
/// device-local bound: capacity, reserve, floor, active-attempt count,
/// source work, installation bounds, capabilities, and telemetry.
/// Queue bounds live in [`NexusGpuQueuePolicy`], because they govern the
/// one shared admission queue rather than a device envelope.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct NexusGpuDeviceProfile {
    device_ordinal: usize,
    memory_limit: GpuDeviceMemoryLimit,
    backend_reserve_bytes: u64,
    cache_cap_bytes: u64,
    attempt_floor_bytes: Option<u64>,
    source_work_per_attempt: usize,
    max_active_attempts: usize,
    installation_bounds: Option<NexusGpuDeviceInstallationBounds>,
    device_capabilities: DeviceCapabilities,
    install_telemetry: bool,
}

/// Explicit installation limits for one device profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct NexusGpuDeviceInstallationBounds {
    pub stream_pool_size: usize,
    pub worker_count: usize,
    pub lane_count: usize,
}

impl NexusGpuDeviceInstallationBounds {
    /// Creates explicit stream, worker, and execution-lane bounds.
    pub const fn new(stream_pool_size: usize, worker_count: usize, lane_count: usize) -> Self {
        Self { stream_pool_size, worker_count, lane_count }
    }
}

impl NexusGpuDeviceProfile {
    /// Starts a profile with automatic capacity discovery and one active
    /// attempt.  Callers may override any device-local bound below.
    pub fn new(device_ordinal: usize) -> Self {
        // Project engine-owned defaults instead of carrying a second root
        // definition. Memory ownership is supplied when this profile is
        // lowered to the engine config and does not affect these defaults.
        let defaults = GpuBackendDeviceConfig::new(device_ordinal, GpuMemoryOwnership::WholeDeviceExclusive);
        Self {
            device_ordinal: defaults.device_ordinal(),
            memory_limit: defaults.memory_limit(),
            backend_reserve_bytes: defaults.backend_reserve_bytes(),
            cache_cap_bytes: 0,
            attempt_floor_bytes: defaults.attempt_floor_bytes(),
            // Source work is a device-local controller capacity, not an
            // execution/session tuning default. Keep the profile concrete
            // from construction so changing execution prefetch never changes
            // its stream or worker envelope.
            source_work_per_attempt: defaults.source_work_per_attempt(),
            max_active_attempts: defaults.max_active_attempts(),
            installation_bounds: None,
            device_capabilities: defaults.device_capabilities().clone(),
            install_telemetry: defaults.install_telemetry(),
        }
    }

    pub fn with_memory_limit(mut self, memory_limit: GpuDeviceMemoryLimit) -> Self {
        self.memory_limit = memory_limit;
        self
    }

    pub fn with_backend_reserve_bytes(mut self, backend_reserve_bytes: u64) -> Self {
        self.backend_reserve_bytes = backend_reserve_bytes;
        self
    }

    /// A zero cap disables resident caching.
    pub fn with_cache_cap_bytes(mut self, cache_cap_bytes: u64) -> Self {
        self.cache_cap_bytes = cache_cap_bytes;
        self
    }

    pub fn with_attempt_floor_bytes(mut self, attempt_floor_bytes: Option<u64>) -> Self {
        self.attempt_floor_bytes = attempt_floor_bytes;
        self
    }

    pub fn with_source_work_per_attempt(mut self, source_work_per_attempt: usize) -> Self {
        self.source_work_per_attempt = source_work_per_attempt;
        self
    }

    pub fn with_max_active_attempts(mut self, max_active_attempts: usize) -> Self {
        self.max_active_attempts = max_active_attempts;
        self
    }

    pub fn with_installation_bounds(mut self, stream_pool_size: usize, worker_count: usize, lane_count: usize) -> Self {
        self.installation_bounds =
            Some(NexusGpuDeviceInstallationBounds::new(stream_pool_size, worker_count, lane_count));
        self
    }

    pub fn with_device_capabilities(mut self, device_capabilities: DeviceCapabilities) -> Self {
        self.device_capabilities = device_capabilities;
        self
    }

    pub fn with_telemetry(mut self, install_telemetry: bool) -> Self {
        self.install_telemetry = install_telemetry;
        self
    }

    pub fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    pub fn memory_limit(&self) -> GpuDeviceMemoryLimit {
        self.memory_limit
    }

    pub fn backend_reserve_bytes(&self) -> u64 {
        self.backend_reserve_bytes
    }

    pub fn attempt_floor_bytes(&self) -> Option<u64> {
        self.attempt_floor_bytes
    }

    pub fn source_work_per_attempt(&self) -> usize {
        self.source_work_per_attempt
    }

    pub fn max_active_attempts(&self) -> usize {
        self.max_active_attempts
    }

    pub fn installation_bounds(&self) -> Option<NexusGpuDeviceInstallationBounds> {
        self.installation_bounds
    }

    pub fn device_capabilities(&self) -> &DeviceCapabilities {
        &self.device_capabilities
    }

    pub fn install_telemetry(&self) -> bool {
        self.install_telemetry
    }

    pub(crate) fn engine_config(&self, memory_ownership: GpuMemoryOwnership) -> GpuBackendDeviceConfig {
        let mut device = GpuBackendDeviceConfig::new(self.device_ordinal, memory_ownership)
            .with_memory_limit(self.memory_limit)
            .with_device_capabilities(self.device_capabilities.clone())
            .with_backend_reserve_bytes(self.backend_reserve_bytes)
            .with_cache_cap_bytes(self.cache_cap_bytes)
            .with_source_work_per_attempt(self.source_work_per_attempt)
            .with_max_active_attempts(self.max_active_attempts)
            .with_telemetry(self.install_telemetry);
        if let Some(floor) = self.attempt_floor_bytes {
            device = device.with_attempt_floor_bytes(floor);
        }
        if let Some(bounds) = self.installation_bounds {
            device = device.with_installation_bounds(bounds.stream_pool_size, bounds.worker_count, bounds.lane_count);
        }
        device
    }

    #[cfg(feature = "test-support")]
    #[doc(hidden)]
    pub fn engine_config_for_test(&self, memory_ownership: GpuMemoryOwnership) -> GpuBackendDeviceConfig {
        self.engine_config(memory_ownership)
    }

    pub fn cache_cap_bytes(&self) -> u64 {
        self.cache_cap_bytes
    }
}

/// Queue-only policy for the shared admission queue.
///
/// Device-local capacity, floor, concurrency, and source-work settings belong
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
    max_overtakes_per_waiter: u32,
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

    /// Finite overtake budget per queued waiter before slot protection. Default 4.
    pub fn max_overtakes_per_waiter(&self) -> u32 {
        self.max_overtakes_per_waiter
    }

    pub fn with_max_queued_attempts(mut self, max_queued_attempts: usize) -> Self {
        self.max_queued_attempts = max_queued_attempts;
        self
    }

    pub fn with_max_queue_wait(mut self, max_queue_wait: Duration) -> Self {
        self.max_queue_wait = max_queue_wait;
        self
    }

    pub fn with_max_overtakes_per_waiter(mut self, max_overtakes_per_waiter: u32) -> Self {
        self.max_overtakes_per_waiter = max_overtakes_per_waiter;
        self
    }
}

impl Default for NexusGpuQueuePolicy {
    fn default() -> Self {
        let bounds = QueueBounds::default();
        Self {
            max_queued_attempts: bounds.max_waiters,
            max_queue_wait: bounds.max_wait,
            max_overtakes_per_waiter: bounds.max_overtakes_per_waiter,
        }
    }
}

/// Builder for [`NexusGpuBackend`].
#[derive(Debug, Clone)]
pub struct NexusGpuBackendBuilder {
    native_execution_config: NexusNativeExecutionConfig,
    device_profiles: Vec<NexusGpuDeviceProfile>,
    device_memory_ownership: Vec<(usize, GpuMemoryOwnership)>,
    queue_policy: NexusGpuQueuePolicy,
    final_plan_requirements: Vec<FinalPlanRequirement>,
    attempt_completion_queue_capacity: Option<usize>,
}

impl Default for NexusGpuBackendBuilder {
    fn default() -> Self {
        Self {
            native_execution_config: NexusNativeExecutionConfig::bounded_default(),
            device_profiles: Vec::new(),
            device_memory_ownership: Vec::new(),
            queue_policy: NexusGpuQueuePolicy::default(),
            final_plan_requirements: Vec::new(),
            attempt_completion_queue_capacity: None,
        }
    }
}

impl NexusGpuBackend {
    pub fn builder() -> NexusGpuBackendBuilder {
        NexusGpuBackendBuilder::default()
    }

    /// Installs the backend-managed DataFusion integration on `builder`.
    ///
    /// The installation binds the native optimizer and
    /// [`DataFusionNexusSessionConfig`]
    /// to this backend's admission authority, and installs the GPU coverage
    /// UDTF, common GPU function catalog, and relation-aware cuVS planner and
    /// metadata family. The resulting session plans and admits native
    /// execution through this backend's ledger.
    ///
    /// This method does not install cuGraph. Use
    /// [`Self::install_on_with_cugraph_sql`], or call
    /// [`NexusSessionStateBuilderExt::try_with_cugraph_sql`] on the returned
    /// builder. Installing cuGraph first is rejected because its captured
    /// configuration is not yet bound to this backend's admission authority.
    pub fn install_on(&self, builder: SessionStateBuilder) -> Result<SessionStateBuilder> {
        install_backend_optimizer_config(builder, &self.optimizer_config)
    }

    /// Installs [`Self::install_on`] followed by the cuGraph SQL family.
    ///
    /// This is equivalent to calling [`Self::install_on`] and then
    /// [`NexusSessionStateBuilderExt::try_with_cugraph_sql`] with `config`.
    /// Without the `cugraph` feature, execution returns a structured
    /// feature-disabled error. Reinstalling the family with an equal
    /// configuration is idempotent; a different descriptor projection is a
    /// structured installation error.
    pub fn install_on_with_cugraph_sql(
        &self,
        builder: SessionStateBuilder,
        cugraph_config: crate::cugraph_sql::CugraphSqlConfig,
    ) -> Result<SessionStateBuilder> {
        install_cugraph_sql_on_installed_backend(self.install_on(builder)?, &self.optimizer_config, cugraph_config)
    }

    /// Opaque identity shared by every clone and installed session.
    #[cfg(any(test, feature = "test-support", feature = "bench-support"))]
    #[doc(hidden)]
    pub fn identity(&self) -> nexus_query_engine::runtime::GpuBackendCoreIdentity {
        self.core.identity()
    }

    /// Returns an explicit capability for controlled device-quarantine
    /// validation. It is not a production backend mode and does not fabricate
    /// a CUDA fault.
    #[cfg(any(test, feature = "test-support", feature = "bench-support"))]
    #[doc(hidden)]
    pub fn fault_injector(&self) -> GpuBackendFaultInjector {
        self.core.fault_injector()
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

    /// Blocks the caller until admission succeeds or fails.
    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn admit_query(&self) -> nexus_query_engine::error::Result<nexus_query_engine::runtime::QueryHandle> {
        self.session.admit_query()
    }

    /// Blocks until the backend admits a query with the exact engine-issued plan proof.
    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn admit_query_with_plan_capability_proof(
        &self,
        proof: nexus_query_engine::admission::QueryPlanCapabilityProof,
    ) -> nexus_query_engine::error::Result<nexus_query_engine::runtime::QueryHandle> {
        self.session.admit_query_with_request(QueryAdmissionRequest::new().with_query_plan_capability_proof(proof))
    }

    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn try_admit_query(
        &self,
    ) -> nexus_query_engine::error::Result<Option<nexus_query_engine::runtime::QueryHandle>> {
        self.session.try_admit_query_with_request(QueryAdmissionRequest::new())
    }

    /// Awaits admission without blocking a thread; dropping the future removes its request.
    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub async fn admit_async(&self) -> nexus_query_engine::error::Result<nexus_query_engine::runtime::QueryHandle> {
        self.session.admit_async_with_request(QueryAdmissionRequest::new()).await
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

    /// Enables or disables exact-request BFS coalescing for subsequent executions.
    #[cfg(feature = "cugraph")]
    pub fn set_cugraph_bfs_singleflight_enabled(&self, enabled: bool) {
        self.optimizer_config.native_execution_config().set_graph_bfs_singleflight_enabled(enabled);
    }

    pub fn device_ordinals(&self) -> &[usize] {
        &self.device_ordinals
    }

    /// Immutable root profiles requested for every selected device.  Runtime
    /// discovery resolves automatic capacity values separately for each
    /// ordinal; use [`Self::device_startup_records`] for those frozen facts.
    pub fn device_profiles(&self) -> &[NexusGpuDeviceProfile] {
        &self.device_profiles
    }

    /// Exact deployment ownership declaration for every selected device.
    pub fn device_memory_ownership(&self) -> &[(usize, GpuMemoryOwnership)] {
        &self.device_memory_ownership
    }

    pub fn from_optimizer_config_for_admission(
        optimizer_config: NexusNativeOptimizerConfig,
        device_profiles: Vec<NexusGpuDeviceProfile>,
        device_memory_ownership: Vec<(usize, GpuMemoryOwnership)>,
        queue_policy: NexusGpuQueuePolicy,
    ) -> Result<Self> {
        Self::build_bounded(optimizer_config, device_profiles, device_memory_ownership, queue_policy, None)
    }

    fn build_bounded(
        optimizer_config: NexusNativeOptimizerConfig,
        configured_profiles: Vec<NexusGpuDeviceProfile>,
        device_memory_ownership: Vec<(usize, GpuMemoryOwnership)>,
        queue_policy: NexusGpuQueuePolicy,
        attempt_completion_queue_capacity: Option<usize>,
    ) -> Result<Self> {
        let device_profiles = resolve_device_profiles(configured_profiles)?;
        let device_ordinals = device_profiles.iter().map(NexusGpuDeviceProfile::device_ordinal).collect::<Vec<_>>();
        let ownership_by_ordinal = validate_device_memory_ownership(&device_ordinals, &device_memory_ownership)?;
        let device_memory_ownership =
            ownership_by_ordinal.iter().map(|(&ordinal, &ownership)| (ordinal, ownership)).collect();
        validate_queue_policy(&queue_policy)?;
        let execution_config = optimizer_config.native_execution_config().clone();
        let service_config = backend_query_execution_config(&execution_config)?;
        let queue_bounds = backend_queue_bounds(&queue_policy);
        let controller = GpuBackendControllerDraft::new(service_config, queue_bounds);
        let devices = device_profiles
            .iter()
            .map(|profile| profile.engine_config(ownership_by_ordinal[&profile.device_ordinal()]))
            .collect::<Vec<_>>();
        let draft = GpuBackendCoreDraft::new(devices, controller, nongds_deployment());
        let draft = match attempt_completion_queue_capacity {
            Some(capacity) => draft.with_attempt_completion_queue_capacity(capacity),
            None => draft,
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
        let execution_config = execution_config
            .with_query_admission_source(QueryAdmissionSource::new(session.clone(), max_device_total_memory_bytes));
        let optimizer_config = optimizer_config.with_native_execution_config(execution_config);
        Ok(NexusGpuBackend {
            core,
            session,
            optimizer_config,
            device_ordinals,
            device_profiles,
            device_memory_ownership,
            queue_policy,
        })
    }
}

fn install_backend_optimizer_config(
    builder: SessionStateBuilder,
    optimizer_config: &NexusNativeOptimizerConfig,
) -> Result<SessionStateBuilder> {
    let builder = builder.try_with_cuvs_sql().map_err(|error| {
        Error::new(
            ErrorCode::InvalidRequest,
            "datafusion_nexus.backend.install_on",
            "failed to install the Nexus cuVS query planner",
        )
        .with_source(error)
    })?;
    let builder = {
        let mut builder = builder;
        reject_preexisting_cugraph_sql(&mut builder)?;
        builder
    };
    let rule = Arc::new(NexusNativePhysicalOptimizerRule::new(optimizer_config.clone()));
    let session_config =
        DataFusionNexusSessionConfig::from_optimizer_config(optimizer_config).frozen_for_backend_admission();
    install_nexus_native_rule_with_session_config(builder, rule, session_config).map_err(|error| {
        Error::new(
            ErrorCode::InvalidRequest,
            "datafusion_nexus.backend.install_on",
            "pre-existing datafusion-nexus session config conflicts with backend admission",
        )
        .with_fact("reason", "session_config_conflicts_with_backend_admission")
        .with_source(error)
    })
}

/// Pure CUDA-free backend-builder validation projection used by adapter and
/// extracted-server validation tests.
///
/// This is intentionally not a `NexusGpuBackend`: it owns no engine resources
/// and cannot admit or execute a query. It exercises the same profile,
/// ownership, queue, and core-draft validation as backend construction without
/// creating CUDA resources.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct TestBackendValidationProjection {
    query_execution_config: QueryExecutionConfig,
    queue_policy: NexusGpuQueuePolicy,
    optimizer_config: NexusNativeOptimizerConfig,
    device_ordinals: Vec<usize>,
    device_profiles: Vec<NexusGpuDeviceProfile>,
    device_memory_ownership: Vec<(usize, GpuMemoryOwnership)>,
}

#[cfg(any(test, feature = "test-support"))]
impl TestBackendValidationProjection {
    pub fn query_execution_config(&self) -> nexus_query_engine::error::Result<QueryExecutionConfig> {
        Ok(self.query_execution_config.clone())
    }

    pub fn queue_policy(&self) -> &NexusGpuQueuePolicy {
        &self.queue_policy
    }

    pub fn optimizer_config(&self) -> &NexusNativeOptimizerConfig {
        &self.optimizer_config
    }

    pub fn device_ordinals(&self) -> &[usize] {
        &self.device_ordinals
    }

    pub fn device_profiles(&self) -> &[NexusGpuDeviceProfile] {
        &self.device_profiles
    }

    pub fn device_memory_ownership(&self) -> &[(usize, GpuMemoryOwnership)] {
        &self.device_memory_ownership
    }

    pub fn install_on(&self, builder: SessionStateBuilder) -> Result<SessionStateBuilder> {
        install_backend_optimizer_config(builder, &self.optimizer_config)
    }

    pub fn install_on_with_cugraph_sql(
        &self,
        builder: SessionStateBuilder,
        cugraph_config: crate::cugraph_sql::CugraphSqlConfig,
    ) -> Result<SessionStateBuilder> {
        install_cugraph_sql_on_installed_backend(self.install_on(builder)?, &self.optimizer_config, cugraph_config)
    }
}

/// Test fixture for paths that genuinely execute an installed GPU backend. The
/// process lease remains production-exclusive; tests serialize construction
/// and deterministically close the backend before releasing the fixture lock.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub struct InstalledTestGpuBackend {
    backend: Option<NexusGpuBackend>,
    _serial: MutexGuard<'static, ()>,
}

#[cfg(any(test, feature = "test-support"))]
impl InstalledTestGpuBackend {
    pub fn new() -> Self {
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
    pub fn with_config(
        native_execution_config: NexusNativeExecutionConfig,
        device_profile: NexusGpuDeviceProfile,
        queue_policy: NexusGpuQueuePolicy,
    ) -> Self {
        Self::with_config_and_final_plan_requirements(native_execution_config, device_profile, queue_policy, [])
    }

    /// Builds a serialized real backend with explicit final-plan requirements.
    /// This keeps integration tests on the same frozen backend-installation
    /// path as production cells that add a final-plan gate.
    pub fn with_config_and_final_plan_requirements(
        native_execution_config: NexusNativeExecutionConfig,
        device_profile: NexusGpuDeviceProfile,
        queue_policy: NexusGpuQueuePolicy,
        final_plan_requirements: impl IntoIterator<Item = FinalPlanRequirement>,
    ) -> Self {
        static SERIAL: Mutex<()> = Mutex::new(());

        let serial = SERIAL.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let device_ordinal = device_profile.device_ordinal();
        let mut builder = NexusGpuBackend::builder()
            .with_native_execution_config(native_execution_config)
            .device_profiles([device_profile])
            .device_memory_ownership([(device_ordinal, GpuMemoryOwnership::WholeDeviceExclusive)])
            .queue_policy(queue_policy);
        for requirement in final_plan_requirements {
            builder = builder.with_final_plan_requirement(requirement);
        }
        let backend = builder.build().expect("test backend configuration must be valid");
        Self { backend: Some(backend), _serial: serial }
    }

    pub fn backend(&self) -> &NexusGpuBackend {
        self.backend.as_ref().expect("test backend remains live until fixture drop")
    }

    pub fn optimizer_config(&self) -> &NexusNativeOptimizerConfig {
        self.backend().optimizer_config()
    }

    pub fn admit_query(&self) -> nexus_query_engine::error::Result<nexus_query_engine::runtime::QueryHandle> {
        self.backend().admit_query()
    }

    pub fn install_on(&self, builder: SessionStateBuilder) -> Result<SessionStateBuilder> {
        self.backend().install_on(builder)
    }
}

#[cfg(any(test, feature = "test-support"))]
impl Default for InstalledTestGpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(any(test, feature = "test-support"))]
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
    /// Supplies the complete ordinal-keyed resource profile set.
    ///
    /// Every backend requires this input, including a single-device backend:
    /// profiles are the only selector and source of device-local capacity.
    pub fn device_profiles(mut self, device_profiles: impl IntoIterator<Item = NexusGpuDeviceProfile>) -> Self {
        self.device_profiles = device_profiles.into_iter().collect();
        self
    }

    /// Retain bounded reports whose completion receivers were abandoned.
    pub fn with_attempt_completion_queue_capacity(mut self, capacity: usize) -> Self {
        self.attempt_completion_queue_capacity = Some(capacity);
        self
    }

    /// Require each completed backend-installed physical plan to satisfy this
    /// native execution constraint.
    pub fn with_final_plan_requirement(mut self, requirement: FinalPlanRequirement) -> Self {
        if !self.final_plan_requirements.contains(&requirement) {
            self.final_plan_requirements.push(requirement);
        }
        self
    }

    /// Reject backend-installed physical plans that retain DataFusion CPU execution.
    pub fn with_no_datafusion_cpu_requirement(self) -> Self {
        self.with_final_plan_requirement(FinalPlanRequirement::NoDataFusionCpu)
    }

    /// Replaces the canonical execution settings copied into admitted plans.
    ///
    /// The execution config is an unadmitted planning template. It never
    /// selects a device or contributes device capacity; those facts are owned
    /// exclusively by [`NexusGpuDeviceProfile`].
    pub fn with_native_execution_config(mut self, native_execution_config: NexusNativeExecutionConfig) -> Self {
        self.native_execution_config = native_execution_config;
        self
    }

    /// Declare deployment ownership for exactly the profiled devices. Missing,
    /// extra, and duplicate ordinals are rejected before the engine begins
    /// backend construction.
    pub fn device_memory_ownership(
        mut self,
        device_memory_ownership: impl IntoIterator<Item = (usize, GpuMemoryOwnership)>,
    ) -> Self {
        self.device_memory_ownership = device_memory_ownership.into_iter().collect();
        self
    }

    /// Sets the shared queue-only GPU admission policy.
    pub fn queue_policy(mut self, queue_policy: NexusGpuQueuePolicy) -> Self {
        self.queue_policy = queue_policy;
        self
    }

    pub fn build(self) -> Result<NexusGpuBackend> {
        self.build_inner()
    }

    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn build_test_validation_projection(self) -> Result<TestBackendValidationProjection> {
        let optimizer_config = self.final_plan_requirements.into_iter().fold(
            NexusNativeOptimizerConfig::default().with_native_execution_config(self.native_execution_config),
            |optimizer_config, requirement| optimizer_config.with_final_plan_requirement(requirement),
        );
        build_test_validation_projection(
            optimizer_config,
            self.device_profiles,
            self.device_memory_ownership,
            self.queue_policy,
        )
    }

    fn build_inner(self) -> Result<NexusGpuBackend> {
        let optimizer_config = self.final_plan_requirements.into_iter().fold(
            NexusNativeOptimizerConfig::default().with_native_execution_config(self.native_execution_config),
            |optimizer_config, requirement| optimizer_config.with_final_plan_requirement(requirement),
        );
        NexusGpuBackend::build_bounded(
            optimizer_config,
            self.device_profiles,
            self.device_memory_ownership,
            self.queue_policy,
            self.attempt_completion_queue_capacity,
        )
    }
}

fn resolve_device_profiles(configured_profiles: Vec<NexusGpuDeviceProfile>) -> Result<Vec<NexusGpuDeviceProfile>> {
    if configured_profiles.is_empty() {
        return Err(invalid_backend_config(
            "device_profiles",
            "backend construction requires at least one explicit ordinal-keyed device profile",
        )
        .with_fact("reason", "missing_device_profiles"));
    }

    let mut by_ordinal = BTreeMap::new();
    for profile in configured_profiles {
        validate_device_profile(&profile)?;
        let device_ordinal = profile.device_ordinal();
        if by_ordinal.insert(device_ordinal, profile).is_some() {
            return Err(invalid_backend_config(
                "device_profiles",
                "must contain exactly one configured profile for each device ordinal",
            )
            .with_fact("reason", "duplicate_device")
            .with_fact("device_ordinal", device_ordinal));
        }
    }
    Ok(by_ordinal.into_values().collect())
}

fn validate_device_profile(profile: &NexusGpuDeviceProfile) -> Result<()> {
    validate_positive_usize("device_profiles.max_active_attempts", profile.max_active_attempts())?;
    if let Some(floor) = profile.attempt_floor_bytes()
        && floor == 0
    {
        return Err(invalid_backend_config(
            "device_profiles.attempt_floor_bytes",
            "must be greater than zero when configured",
        )
        .with_fact("device_ordinal", profile.device_ordinal()));
    }
    if profile.source_work_per_attempt() == 0 {
        return Err(invalid_backend_config("device_profiles.source_work_per_attempt", "must be greater than zero")
            .with_fact("device_ordinal", profile.device_ordinal()));
    }
    if let Some(bounds) = profile.installation_bounds() {
        for (field, value) in [
            ("device_profiles.stream_pool_size", bounds.stream_pool_size),
            ("device_profiles.worker_count", bounds.worker_count),
            ("device_profiles.lane_count", bounds.lane_count),
        ] {
            if value == 0 {
                return Err(invalid_backend_config(field, "must be greater than zero")
                    .with_fact("device_ordinal", profile.device_ordinal()));
            }
        }
    }
    Ok(())
}

#[cfg(any(test, feature = "test-support"))]
fn build_test_validation_projection(
    optimizer_config: NexusNativeOptimizerConfig,
    configured_profiles: Vec<NexusGpuDeviceProfile>,
    device_memory_ownership: Vec<(usize, GpuMemoryOwnership)>,
    queue_policy: NexusGpuQueuePolicy,
) -> Result<TestBackendValidationProjection> {
    let device_profiles = resolve_device_profiles(configured_profiles)?;
    let device_ordinals = device_profiles.iter().map(NexusGpuDeviceProfile::device_ordinal).collect::<Vec<_>>();
    let ownership_by_ordinal = validate_device_memory_ownership(&device_ordinals, &device_memory_ownership)?;
    let device_memory_ownership =
        ownership_by_ordinal.iter().map(|(&ordinal, &ownership)| (ordinal, ownership)).collect();
    validate_queue_policy(&queue_policy)?;
    let execution_config = optimizer_config.native_execution_config().clone();
    let query_execution_config = backend_query_execution_config(&execution_config)?;
    let queue_bounds = backend_queue_bounds(&queue_policy);
    let controller_draft = GpuBackendControllerDraft::new(query_execution_config.clone(), queue_bounds.clone());
    let devices = device_profiles
        .iter()
        .map(|profile| profile.engine_config(ownership_by_ordinal[&profile.device_ordinal()]))
        .collect::<Vec<_>>();
    GpuBackendCoreDraft::new(devices, controller_draft, nongds_deployment()).validate().map_err(|error| {
        Error::from_native(
            "datafusion_nexus.backend.gpu_core",
            "invalid ownership-enforcing GPU backend core configuration",
            error,
        )
    })?;

    Ok(TestBackendValidationProjection {
        query_execution_config,
        queue_policy,
        optimizer_config,
        device_ordinals,
        device_profiles,
        device_memory_ownership,
    })
}

#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn test_validation_projection_from_optimizer_config(
    optimizer_config: NexusNativeOptimizerConfig,
    device_profiles: Vec<NexusGpuDeviceProfile>,
    device_memory_ownership: Vec<(usize, GpuMemoryOwnership)>,
    queue_policy: NexusGpuQueuePolicy,
) -> Result<TestBackendValidationProjection> {
    build_test_validation_projection(optimizer_config, device_profiles, device_memory_ownership, queue_policy)
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
        graph_memory_calibration_override: execution_config.graph_memory_calibration_override(),
        max_source_chunk_bytes,
        max_row_groups_per_chunk,
        max_retry_attempts,
        min_retry_chunk_row_groups,
    })
}

fn backend_queue_bounds(queue_policy: &NexusGpuQueuePolicy) -> QueueBounds {
    QueueBounds {
        max_waiters: queue_policy.max_queued_attempts,
        max_wait: queue_policy.max_queue_wait,
        max_overtakes_per_waiter: queue_policy.max_overtakes_per_waiter,
    }
}

fn validate_queue_policy(policy: &NexusGpuQueuePolicy) -> Result<()> {
    validate_positive_usize("max_queued_attempts", policy.max_queued_attempts)?;
    if policy.max_queue_wait.is_zero() {
        return Err(invalid_backend_config("max_queue_wait", "must be greater than zero"));
    }
    if policy.max_overtakes_per_waiter == 0 {
        return Err(invalid_backend_config("max_overtakes_per_waiter", "must be greater than zero"));
    }
    Ok(())
}

fn validate_device_memory_ownership(
    device_ordinals: &[usize],
    declarations: &[(usize, GpuMemoryOwnership)],
) -> Result<BTreeMap<usize, GpuMemoryOwnership>> {
    let selected = device_ordinals.iter().copied().collect::<BTreeSet<_>>();
    let mut ownership_by_ordinal = BTreeMap::new();
    for (device_ordinal, ownership) in declarations.iter().copied() {
        if ownership_by_ordinal.insert(device_ordinal, ownership).is_some() {
            return Err(invalid_backend_config(
                "device_memory_ownership",
                "must contain exactly one declaration for each selected device",
            )
            .with_fact("reason", "duplicate_device")
            .with_fact("device_ordinal", device_ordinal));
        }
    }

    let declared = ownership_by_ordinal.keys().copied().collect::<BTreeSet<_>>();
    let missing = selected.difference(&declared).copied().collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(invalid_backend_config(
            "device_memory_ownership",
            "must declare ownership for every selected device",
        )
        .with_fact("reason", "missing_devices")
        .with_fact("device_ordinals", format_device_ordinals(&missing)));
    }
    let extra = declared.difference(&selected).copied().collect::<Vec<_>>();
    if !extra.is_empty() {
        return Err(invalid_backend_config(
            "device_memory_ownership",
            "must not declare ownership for an unselected device",
        )
        .with_fact("reason", "extra_devices")
        .with_fact("device_ordinals", format_device_ordinals(&extra)));
    }
    Ok(ownership_by_ordinal)
}

fn format_device_ordinals(device_ordinals: &[usize]) -> String {
    device_ordinals.iter().map(usize::to_string).collect::<Vec<_>>().join(",")
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

fn reject_preexisting_cugraph_sql(builder: &mut SessionStateBuilder) -> Result<()> {
    let Some(functions) = builder.table_functions().as_ref() else {
        return Ok(());
    };
    let Some(function_name) =
        functions.keys().filter(|name| is_cugraph_sql_table_function_name(name.as_str())).min().cloned()
    else {
        return Ok(());
    };
    Err(Error::new(
        ErrorCode::InvalidRequest,
        "datafusion_nexus.backend.install_on",
        "cuGraph SQL must be registered through backend admission",
    )
    .with_fact("reason", "preexisting_cugraph_sql_without_backend_admission")
    .with_fact("table_function", function_name)
    .with_fact("remediation", "use install_on_with_cugraph_sql or call try_with_cugraph_sql after install_on"))
}

fn install_cugraph_sql_on_installed_backend(
    builder: SessionStateBuilder,
    optimizer_config: &NexusNativeOptimizerConfig,
    cugraph_config: crate::cugraph_sql::CugraphSqlConfig,
) -> Result<SessionStateBuilder> {
    use crate::session::NexusSessionStateBuilderExt;

    builder
        .try_with_cugraph_sql(
            cugraph_config.with_native_execution_config(optimizer_config.native_execution_config().clone()),
        )
        .map_err(preserve_gpu_function_install_error)
}

/// Propagate a cuGraph installation failure without losing the structured
/// Nexus classification minted by the shared catalog installer.
fn preserve_gpu_function_install_error(error: DataFusionError) -> Error {
    if let Some(nexus) = error.find_nexus_error() {
        return Error::new(nexus.code(), nexus.operation(), nexus.message().to_string())
            .with_preserved_facts(nexus.facts().clone())
            .with_source(error);
    }
    Error::new(
        ErrorCode::InvalidRequest,
        "datafusion_nexus.backend.install_on_with_cugraph_sql",
        "failed to install cuGraph SQL functions",
    )
    .with_source(error)
}

fn is_cugraph_sql_table_function_name(name: &str) -> bool {
    use crate::cugraph_sql::CugraphAlgorithm;

    CugraphAlgorithm::all().any(|algorithm| algorithm.function_name() == name)
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
