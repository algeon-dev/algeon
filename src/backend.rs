//! Process-scoped GPU backend facade for embedded DataFusion sessions.

#[cfg(test)]
use std::sync::{Mutex, MutexGuard};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::Duration,
};

use datafusion::execution::SessionStateBuilder;
use nexus_query_engine::runtime::{
    DeploymentIoCapability, GdsDriverProbeStatus, GpuBackendControllerDraft, GpuBackendCore, GpuBackendCoreDraft,
    GpuBackendDeviceConfig, GpuBackendSession, KvikioCompatModeStatus, MemoryControlPolicy, NvidiaFsProbeStatus,
    QueryAdmissionRequest, QueryExecutionConfig, QueueBounds,
};
pub use nexus_query_engine::runtime::{DeviceCapabilities, GpuDeviceMemoryLimit, GpuMemoryOwnership};

use crate::{
    error::{Error, ErrorCode, Result},
    native_exec::{NexusNativeExecutionConfig, QueryAdmissionSource},
    planner::{FinalPlanRequirement, NexusNativeOptimizerConfig, NexusNativePhysicalOptimizerRule},
    session::{NexusSessionStateBuilderExt, install_nexus_native_rule_with_session_config},
    session_config::DataFusionNexusSessionConfig,
};

/// Process-scoped native GPU backend shared by one or more DataFusion sessions.
#[derive(Debug, Clone)]
pub struct NexusGpuBackend {
    core: GpuBackendCore,
    session: GpuBackendSession,
    optimizer_config: NexusNativeOptimizerConfig,
    device_ordinals: Vec<usize>,
    device_profiles: Vec<NexusGpuDeviceProfile>,
    device_memory_ownership: Vec<(usize, GpuMemoryOwnership)>,
}

/// Explicit immutable resource envelope for one CUDA device.
///
/// A profile is keyed by its CUDA ordinal and is projected one-for-one into
/// the engine's [`GpuBackendDeviceConfig`].  It deliberately owns every
/// device-local bound: capacity, reserve, floor, active-attempt count,
/// source work, installation bounds, capabilities, and optional diagnostics.
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
    install_diagnostics: bool,
}

/// Explicit installation limits for one device profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct NexusGpuDeviceInstallationBounds {
    pub stream_pool_size: usize,
    pub worker_count: usize,
    pub lane_count: usize,
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
            max_active_attempts: defaults.max_active_queries(),
            installation_bounds: None,
            device_capabilities: defaults.device_capabilities().clone(),
            install_telemetry: defaults.install_telemetry(),
            install_diagnostics: defaults.install_diagnostics(),
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
            Some(NexusGpuDeviceInstallationBounds { stream_pool_size, worker_count, lane_count });
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

    pub fn with_diagnostics(mut self, install_diagnostics: bool) -> Self {
        self.install_diagnostics = install_diagnostics;
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

    pub fn install_diagnostics(&self) -> bool {
        self.install_diagnostics
    }

    pub(crate) fn engine_config(&self, memory_ownership: GpuMemoryOwnership) -> GpuBackendDeviceConfig {
        let mut device = GpuBackendDeviceConfig::new(self.device_ordinal, memory_ownership)
            .with_memory_limit(self.memory_limit)
            .with_device_capabilities(self.device_capabilities.clone())
            .with_backend_reserve_bytes(self.backend_reserve_bytes)
            .with_cache_cap_bytes(self.cache_cap_bytes)
            .with_source_work_per_attempt(self.source_work_per_attempt)
            .with_max_active_queries(self.max_active_attempts)
            .with_telemetry(self.install_telemetry)
            .with_diagnostics(self.install_diagnostics);
        if let Some(floor) = self.attempt_floor_bytes {
            device = device.with_attempt_floor_bytes(floor);
        }
        if let Some(bounds) = self.installation_bounds {
            device = device.with_installation_bounds(bounds.stream_pool_size, bounds.worker_count, bounds.lane_count);
        }
        device
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
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct NexusGpuQueuePolicy {
    /// Maximum queued admission waiters. Default 64.
    pub max_queued_attempts: usize,
    /// Maximum time a waiter may queue before a structured timeout. Default 30s.
    pub max_queue_wait: Duration,
    /// Finite overtake budget per queued waiter before slot protection. Default 4.
    pub max_overtakes_per_waiter: u32,
}

impl NexusGpuQueuePolicy {
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

impl From<QueueBounds> for NexusGpuQueuePolicy {
    fn from(bounds: QueueBounds) -> Self {
        Self {
            max_queued_attempts: bounds.max_waiters,
            max_queue_wait: bounds.max_wait,
            max_overtakes_per_waiter: bounds.max_overtakes_per_waiter,
        }
    }
}

impl From<NexusGpuQueuePolicy> for QueueBounds {
    fn from(policy: NexusGpuQueuePolicy) -> Self {
        Self {
            max_waiters: policy.max_queued_attempts,
            max_wait: policy.max_queue_wait,
            max_overtakes_per_waiter: policy.max_overtakes_per_waiter,
        }
    }
}

impl Default for NexusGpuQueuePolicy {
    fn default() -> Self {
        QueueBounds::default().into()
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

    /// Install this backend's native optimizer into a DataFusion session builder.
    pub fn install_on(&self, builder: SessionStateBuilder) -> Result<SessionStateBuilder> {
        install_backend_optimizer_config(builder, &self.optimizer_config)
    }

    pub fn install_on_with_cugraph_sql(
        &self,
        builder: SessionStateBuilder,
        cugraph_config: crate::cugraph_sql::CugraphSqlConfig,
    ) -> Result<SessionStateBuilder> {
        use crate::session::NexusSessionStateBuilderExt;

        let builder = self.install_on(builder)?;
        Ok(builder.with_cugraph_sql(
            cugraph_config.with_native_execution_config(self.optimizer_config.native_execution_config().clone()),
        ))
    }

    /// Opaque identity shared by every clone and installed session.
    pub fn identity(&self) -> nexus_query_engine::runtime::GpuBackendCoreIdentity {
        self.core.identity()
    }

    /// Immutable per-device construction records shared by every clone and
    /// session installed from this backend.
    pub fn device_startup_records(&self) -> &[nexus_query_engine::runtime::GpuBackendDeviceStartupRecord] {
        self.core.device_startup_records()
    }

    pub fn close(&self) -> bool {
        self.core.close()
    }

    pub fn close_and_wait(
        &self,
        timeout: Duration,
    ) -> nexus_query_engine::error::Result<nexus_query_engine::runtime::GpuBackendCloseSnapshot> {
        self.core.close_and_wait(timeout)
    }

    pub fn close_and_wait_until(
        &self,
        deadline: std::time::Instant,
    ) -> nexus_query_engine::error::Result<nexus_query_engine::runtime::GpuBackendCloseSnapshot> {
        self.core.close_and_wait_until(deadline)
    }

    /// Returns one coherent, fixed-cardinality backend metrics snapshot.
    ///
    /// Applications own any exporter or retention policy.  This facade only
    /// exposes the backend generation snapshot produced by the engine; it
    /// does not install an exporter or create an HTTP metrics endpoint.
    pub fn backend_metrics_snapshot(&self) -> nexus_query_engine::runtime::BackendMetricsSnapshot {
        self.session.backend_metrics_snapshot()
    }

    pub fn admission_snapshot(
        &self,
    ) -> nexus_query_engine::error::Result<nexus_query_engine::runtime::QueryServiceAdmissionSnapshot> {
        self.session.admission_snapshot()
    }

    pub fn controller_config(&self) -> nexus_query_engine::error::Result<QueryExecutionConfig> {
        self.session.config()
    }

    pub fn controller_queue_bounds(&self) -> nexus_query_engine::error::Result<QueueBounds> {
        self.session.queue_bounds()
    }

    pub fn device_admission_snapshots(
        &self,
    ) -> nexus_query_engine::error::Result<Vec<nexus_query_engine::runtime::QueryServiceDeviceAdmissionSnapshot>> {
        self.session.device_admission_snapshots()
    }

    pub fn device_ledger_reports(
        &self,
    ) -> nexus_query_engine::error::Result<Vec<nexus_query_engine::runtime::DeviceLedgerReport>> {
        self.session.device_ledger_reports()
    }

    pub fn device_memory_failure_reports(
        &self,
    ) -> nexus_query_engine::error::Result<Vec<nexus_query_engine::runtime::DeviceMemoryFailureReport>> {
        self.session.device_memory_failure_reports()
    }

    pub fn device_telemetry_snapshots(
        &self,
    ) -> nexus_query_engine::error::Result<Vec<nexus_query_engine::runtime::DeviceTelemetrySnapshot>> {
        self.session.device_telemetry_snapshots()
    }

    pub fn gpu_execution_pool_reports(
        &self,
    ) -> nexus_query_engine::error::Result<Vec<nexus_query_engine::runtime::GpuExecutionPoolSnapshot>> {
        self.session.gpu_execution_pool_reports()
    }

    pub fn gpu_execution_query_report(
        &self,
    ) -> nexus_query_engine::error::Result<nexus_query_engine::runtime::GpuExecutionQuerySnapshot> {
        self.session.gpu_execution_query_report()
    }

    pub fn admit_query(&self) -> nexus_query_engine::error::Result<nexus_query_engine::runtime::QueryHandle> {
        self.session.admit_query()
    }

    pub fn try_admit_query(
        &self,
    ) -> nexus_query_engine::error::Result<Option<nexus_query_engine::runtime::QueryHandle>> {
        self.session.try_admit_query_with_request(QueryAdmissionRequest::new())
    }

    pub async fn admit_async(&self) -> nexus_query_engine::error::Result<nexus_query_engine::runtime::QueryHandle> {
        self.session.admit_async_with_request(QueryAdmissionRequest::new()).await
    }

    pub fn same_backend(&self, other: &Self) -> bool {
        self.identity() == other.identity()
    }

    pub(crate) fn session_capability(&self) -> &GpuBackendSession {
        &self.session
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

    #[cfg(feature = "server")]
    pub(crate) fn from_optimizer_config_for_admission(
        optimizer_config: NexusNativeOptimizerConfig,
        device_profiles: Vec<NexusGpuDeviceProfile>,
        device_memory_ownership: Vec<(usize, GpuMemoryOwnership)>,
        queue_policy: NexusGpuQueuePolicy,
    ) -> Result<Option<Self>> {
        Self::from_optimizer_config_inner(optimizer_config, device_profiles, device_memory_ownership, queue_policy)
    }

    #[cfg(feature = "server")]
    fn from_optimizer_config_inner(
        optimizer_config: NexusNativeOptimizerConfig,
        device_profiles: Vec<NexusGpuDeviceProfile>,
        device_memory_ownership: Vec<(usize, GpuMemoryOwnership)>,
        queue_policy: NexusGpuQueuePolicy,
    ) -> Result<Option<Self>> {
        Self::build_bounded(optimizer_config, device_profiles, device_memory_ownership, queue_policy, None).map(Some)
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

/// Pure CUDA-free backend-builder validation projection used only by root unit tests.
///
/// This is intentionally not a `NexusGpuBackend`: it owns no engine resources
/// and cannot admit or execute a query. It exercises the same profile,
/// ownership, queue, and core-draft validation as backend construction without
/// creating CUDA resources.
#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) struct TestBackendValidationProjection {
    query_execution_config: QueryExecutionConfig,
    queue_bounds: QueueBounds,
    optimizer_config: NexusNativeOptimizerConfig,
    device_ordinals: Vec<usize>,
    device_profiles: Vec<NexusGpuDeviceProfile>,
    device_memory_ownership: Vec<(usize, GpuMemoryOwnership)>,
}

#[cfg(test)]
impl TestBackendValidationProjection {
    pub(crate) fn query_execution_config(&self) -> nexus_query_engine::error::Result<QueryExecutionConfig> {
        Ok(self.query_execution_config.clone())
    }

    pub(crate) fn queue_bounds(&self) -> nexus_query_engine::error::Result<QueueBounds> {
        Ok(self.queue_bounds.clone())
    }

    pub(crate) fn optimizer_config(&self) -> &NexusNativeOptimizerConfig {
        &self.optimizer_config
    }

    pub(crate) fn device_ordinals(&self) -> &[usize] {
        &self.device_ordinals
    }

    pub(crate) fn device_profiles(&self) -> &[NexusGpuDeviceProfile] {
        &self.device_profiles
    }

    pub(crate) fn device_memory_ownership(&self) -> &[(usize, GpuMemoryOwnership)] {
        &self.device_memory_ownership
    }

    pub(crate) fn install_on(&self, builder: SessionStateBuilder) -> Result<SessionStateBuilder> {
        install_backend_optimizer_config(builder, &self.optimizer_config)
    }
}

/// Root-unit-test fixture for paths that genuinely execute an installed GPU
/// backend. The process lease remains production-exclusive; tests serialize
/// construction and deterministically close the backend before releasing the
/// fixture lock.
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
        static SERIAL: Mutex<()> = Mutex::new(());

        let serial = SERIAL.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let device_ordinal = device_profile.device_ordinal();
        let backend = NexusGpuBackend::builder()
            .with_native_execution_config(native_execution_config)
            .device_profiles([device_profile])
            .device_memory_ownership([(device_ordinal, GpuMemoryOwnership::WholeDeviceExclusive)])
            .queue_policy(queue_policy)
            .build()
            .expect("root GPU unit test backend must build");
        Self { backend: Some(backend), _serial: serial }
    }

    pub(crate) fn backend(&self) -> &NexusGpuBackend {
        self.backend.as_ref().expect("test backend remains live until fixture drop")
    }

    pub(crate) fn optimizer_config(&self) -> &NexusNativeOptimizerConfig {
        self.backend().optimizer_config()
    }

    pub(crate) fn admit_query(&self) -> nexus_query_engine::error::Result<nexus_query_engine::runtime::QueryHandle> {
        self.backend().admit_query()
    }

    pub(crate) fn install_on(&self, builder: SessionStateBuilder) -> Result<SessionStateBuilder> {
        self.backend().install_on(builder)
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

    #[cfg(test)]
    pub(crate) fn build_test_validation_projection(self) -> Result<TestBackendValidationProjection> {
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

#[cfg(test)]
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
        queue_bounds,
        optimizer_config,
        device_ordinals,
        device_profiles,
        device_memory_ownership,
    })
}

#[cfg(all(test, feature = "server"))]
pub(crate) fn test_validation_projection_from_optimizer_config(
    optimizer_config: NexusNativeOptimizerConfig,
    device_profiles: Vec<NexusGpuDeviceProfile>,
    device_memory_ownership: Vec<(usize, GpuMemoryOwnership)>,
    queue_policy: NexusGpuQueuePolicy,
) -> Result<Option<TestBackendValidationProjection>> {
    build_test_validation_projection(optimizer_config, device_profiles, device_memory_ownership, queue_policy).map(Some)
}

fn backend_query_execution_config(execution_config: &NexusNativeExecutionConfig) -> Result<QueryExecutionConfig> {
    let MemoryControlPolicy::Bounded {
        device_budget_bytes,
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
    // This only validates the unadmitted execution template. Per-device
    // capacity is never derived from this byte value: profiles own it.
    validate_positive_u64("native_execution_config.device_budget_bytes", device_budget_bytes)?;
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
        stddev_samp_execution_mode: execution_config.stddev_samp_execution_mode(),
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
    queue_policy.clone().into()
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
    .with_fact("remediation", "use install_on_with_cugraph_sql or call with_cugraph_sql after install_on"))
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
