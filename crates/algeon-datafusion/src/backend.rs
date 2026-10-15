//! Process-scoped GPU backend facade for embedded DataFusion sessions.
//!
//! Build one backend with [`AlgeonGpuBackend::builder`], install it on
//! caller-owned session builders, and close it explicitly with
//! [`AlgeonGpuBackend::close_and_wait`] or
//! [`AlgeonGpuBackend::close_and_wait_async`].

#[cfg(test)]
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use datafusion::execution::{SessionState, SessionStateBuilder};
use query_planning::error::QueryError;
#[cfg(all(test, feature = "cugraph"))]
use query_planning::error::Result as QueryEngineResult;
#[cfg(feature = "qualification")]
pub use query_runtime::runtime::DeviceQuarantineInjection;
#[cfg(all(test, feature = "cugraph"))]
use query_runtime::runtime::QueryHandle;
pub use query_runtime::runtime::{
    DeviceCapabilities, GpuDeviceCapacityProbe, GpuDeviceMemoryLimit, GpuMemoryOwnership,
};
use query_runtime::runtime::{
    GpuBackendConfig, GpuBackendCore, GpuBackendSession, QueryExecutionConfig, QueueBounds, ValidatedGpuBackendConfig,
    probe_device_capacity,
};
/// Canonical device profile bound to an explicit process-scoped GPU memory
/// ownership promise.
///
/// Re-exports runtime-owned device construction types under application-facing
/// names. They share type identity with runtime definitions without introducing
/// adapter-specific defaults or separate validation.
pub use query_runtime::{
    runtime::GpuBackendDeviceConfig as AlgeonGpuDeviceConfig,
    runtime::GpuBackendDeviceInstallationBounds as AlgeonGpuDeviceInstallationBounds,
    runtime::GpuBackendDeviceProfile as AlgeonGpuDeviceProfile,
};

#[cfg(feature = "cugraph")]
use crate::cugraph_sql::GraphBfsSingleflightSnapshot;
use crate::{
    build_info::algeon_version_table_function,
    error::{Error, Result},
    native_exec::{AlgeonNativeExecutionConfig, QueryAdmissionSource, SourceChunking},
    observability::{
        BackendDiagnostics, BackendMetricsSnapshot, DeviceTelemetrySnapshot, GpuBackendCloseSnapshot,
        GpuBackendDeviceStartupRecord, KvikioProcessConfig,
    },
    planner::{AlgeonGpuExecutionMode, AlgeonNativeOptimizerConfig},
    s3_storage::install_s3_storage,
    session::{AlgeonSessionStateBuilderExt, installed_algeon_native_optimizer_config},
};

/// Process-scoped native GPU backend shared by one or more DataFusion sessions.
///
/// Cloning a backend yields another shared handle to the same backend generation.
/// When any clone signals close, all clones and installed sessions observe backend
/// closure immediately.
///
/// Installed [`SessionState`] instances retain only a session capability that
/// cannot extend GPU resource ownership. An idle [`datafusion::prelude::SessionContext`]
/// does not hold an active lease and does not need to be dropped before closing the backend.
/// Once closing begins, new attempts and admissions are rejected fail-closed, while
/// active attempts complete or terminate according to contract. Published attempt reports
/// and observation snapshots remain readable after GPU resources have achieved quiescence.
///
/// Dropping the last owning backend handle signals close without waiting for
/// quiescence and cannot report a timeout or cleanup failure. The process
/// owner should explicitly wait through [`Self::close_and_wait`] or
/// [`Self::close_and_wait_async`].
#[derive(Debug, Clone)]
pub struct AlgeonGpuBackend {
    /// The owning handle, through which close and shutdown coordination run.
    core: GpuBackendCore,
    /// Capability copied into installed sessions and admission sources. It
    /// admits and observes attempts; it cannot extend the backend's lifetime
    /// or its GPU ownership.
    session: GpuBackendSession,
    /// Planning template bound to `session`, copied into every installed
    /// session's native rule. It never names a device.
    optimizer_config: AlgeonNativeOptimizerConfig,
    devices: Vec<AlgeonGpuDeviceConfig>,
    queue_policy: AlgeonGpuQueuePolicy,
}

/// Queue-only policy for the shared admission queue.
///
/// Device capacity, query limits, concurrency, and source settings are configured
/// per device in [`AlgeonGpuDeviceProfile`]. Keeping this policy queue-only
/// prevents uniform scalar bounds from being applied across heterogeneous devices.
///
/// Values are constructed from [`Default`] and adjusted through the `with_*`
/// builders; zero bounds are rejected when the backend is built.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AlgeonGpuQueuePolicy {
    max_queued_attempts: usize,
    max_queue_wait: Duration,
}

impl AlgeonGpuQueuePolicy {
    /// Maximum queued admission waiters. Default 64.
    pub fn max_queued_attempts(&self) -> usize {
        self.max_queued_attempts
    }

    /// Maximum time a waiter may queue before a structured timeout. Default 30s.
    pub fn max_queue_wait(&self) -> Duration {
        self.max_queue_wait
    }

    /// Sets the number of admission waiters permitted across the backend.
    ///
    /// Validation at backend construction rejects zero.
    pub fn with_max_queued_attempts(mut self, max_queued_attempts: usize) -> Self {
        self.max_queued_attempts = max_queued_attempts;
        self
    }

    /// Sets the maximum admission queue wait before a timeout.
    ///
    /// Validation at backend construction rejects a zero duration.
    pub fn with_max_queue_wait(mut self, max_queue_wait: Duration) -> Self {
        self.max_queue_wait = max_queue_wait;
        self
    }
}

impl Default for AlgeonGpuQueuePolicy {
    fn default() -> Self {
        let bounds = QueueBounds::default();
        Self { max_queued_attempts: bounds.max_waiters, max_queue_wait: bounds.max_wait }
    }
}

/// Builder for [`AlgeonGpuBackend`].
#[derive(Debug, Clone, Default)]
pub struct AlgeonGpuBackendBuilder {
    optimizer_config: AlgeonNativeOptimizerConfig,
    devices: Vec<AlgeonGpuDeviceConfig>,
    queue_policy: AlgeonGpuQueuePolicy,
    attempt_completion_queue_capacity: Option<usize>,
}

impl AlgeonGpuBackend {
    /// Starts a backend declaration with default optimizer and queue policies.
    ///
    /// Supply device declarations before calling [`AlgeonGpuBackendBuilder::build`].
    pub fn builder() -> AlgeonGpuBackendBuilder {
        AlgeonGpuBackendBuilder::default()
    }

    /// Installs the backend-managed DataFusion integration on `builder`.
    ///
    /// Binds the native optimizer to this backend's admission authority,
    /// installs the GPU coverage UDTF, and registers `algeon_version()`, which
    /// returns [`crate::build_info::build_info`] and this backend's execution mode
    /// as one row. The resulting session plans and admits native execution through
    /// this backend's ledger. With [`AlgeonNativeExecutionConfig::with_s3_storage`],
    /// the session also resolves unregistered `s3://` URLs from that storage,
    /// configuring DataFusion scans with the native reader endpoint, region,
    /// and credentials.
    ///
    /// This method does not install optional SQL families such as cuVS or
    /// cuGraph. Call [`AlgeonSessionStateBuilderExt::try_with_cuvs_sql`] or
    /// [`AlgeonSessionStateBuilderExt::try_with_cugraph_sql`] on the returned
    /// builder when those families are required.
    pub fn install_on(&self, builder: SessionStateBuilder) -> Result<SessionStateBuilder> {
        install_backend_optimizer_config(builder, &self.optimizer_config)
    }

    /// Performs controlled device quarantine for qualification tooling by
    /// closing admission on the specified device ledger. This operation does not
    /// simulate CUDA faults or alter cuDF global health, and is not a production
    /// backend mode.
    #[cfg(feature = "qualification")]
    pub fn quarantine_device_for_qualification(&self, device_ordinal: usize) -> Result<DeviceQuarantineInjection> {
        self.core.fault_injector().quarantine_device(device_ordinal).map_err(|error| {
            Error::from_native(
                "algeon_datafusion.backend.quarantine_device_for_qualification",
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

    /// Returns KvikIO process settings applied when this backend was built,
    /// including compatibility mode, thread count, and task size shared by
    /// every source read in the process. Returns `None` for builds without
    /// KvikIO support.
    pub fn kvikio_process_config(&self) -> Option<KvikioProcessConfig> {
        self.core.kvikio_process_config()
    }

    /// Probes declarable capacity for one CUDA device without building a backend.
    ///
    /// Probes declarable memory capacity without acquiring the process owner lease
    /// or initializing a memory manager. Callers can use the probe result to select
    /// an explicit limit before calling [`AlgeonGpuBackendBuilder::build`].
    /// Declarations up to `max_declarable_limit_bytes` satisfy backend capacity checks.
    /// The probe reports both the CUDA-visible total used for ownership limits and,
    /// when NVML is available, the physical device total.
    pub fn probe_device_capacity(
        device_ordinal: usize,
        memory_ownership: GpuMemoryOwnership,
        backend_reserve_bytes: u64,
    ) -> Result<GpuDeviceCapacityProbe> {
        probe_device_capacity(device_ordinal, memory_ownership, backend_reserve_bytes).map_err(|error| {
            Error::from_native_with_blocking_diagnostics(
                "algeon_datafusion.backend.probe_device_capacity",
                "failed to probe declarable GPU device capacity",
                error,
            )
        })
    }

    /// Non-blockingly signals backend shutdown.
    ///
    /// New attempt admissions are rejected immediately, while active attempts complete
    /// cooperatively. Returns `true` if this call initiated the close, or `false`
    /// if the backend was already closing or closed.
    pub fn close(&self) -> bool {
        self.core.close()
    }

    /// Returns the current two-scope backend close state without altering state or
    /// initiating a close.
    ///
    /// Callers tracking shutdown progression can inspect whether GPU resources
    /// and observation records have quiesced without altering closing state.
    pub fn close_snapshot(&self) -> GpuBackendCloseSnapshot {
        self.core.close_snapshot()
    }

    /// Canonical shutdown path for embedded and process owners.
    ///
    /// Initiates backend shutdown and blocks until both GPU resources and observation
    /// records achieve quiescence, or until `timeout` elapses. It blocks the
    /// calling thread; from inside an async runtime use
    /// [`Self::close_and_wait_async`].
    pub fn close_and_wait(&self, timeout: Duration) -> Result<GpuBackendCloseSnapshot> {
        self.core.close_and_wait(timeout).map_err(|error| {
            Error::from_native("algeon_datafusion.backend.close", "failed to close GPU backend before timeout", error)
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
            Error::from_native("algeon_datafusion.backend.close", "failed to close GPU backend before timeout", error)
        })
    }

    /// Shutdown path for callers that already hold a shared absolute deadline.
    ///
    /// Initiates backend shutdown and blocks until quiescence or until `deadline`
    /// is reached in the monotonic clock domain.
    pub fn close_and_wait_until(&self, deadline: std::time::Instant) -> Result<GpuBackendCloseSnapshot> {
        self.core.close_and_wait_until(deadline).map_err(|error| {
            Error::from_native("algeon_datafusion.backend.close", "failed to close GPU backend before deadline", error)
        })
    }

    /// Returns one coherent, fixed-cardinality backend metrics snapshot.
    ///
    /// Applications own any exporter or retention policy. This facade only
    /// exposes the backend generation snapshot produced by the engine; it
    /// does not install an exporter or create an HTTP metrics endpoint.
    pub fn backend_metrics_snapshot(&self) -> BackendMetricsSnapshot {
        self.session.backend_metrics_snapshot()
    }

    /// Reads the backend's admission, ledger, cache, memory-failure, and
    /// execution-pool reports once. The ledgers are the admission authority;
    /// the other reports are diagnostic cross-checks.
    pub fn diagnostics(&self) -> Result<BackendDiagnostics> {
        self.session.diagnostics().map_err(|error| {
            Error::from_native("algeon_datafusion.backend.diagnostics", "failed to read backend diagnostics", error)
        })
    }

    /// Returns the immutable queue bounds shared by sessions installed from this backend.
    pub fn queue_policy(&self) -> &AlgeonGpuQueuePolicy {
        &self.queue_policy
    }

    /// Returns diagnostic CUDA/NVML samples. The samples carry no admission state.
    pub fn device_telemetry_snapshots(&self) -> Result<Vec<DeviceTelemetrySnapshot>> {
        self.session.device_telemetry_snapshots().map_err(|error| {
            Error::from_native(
                "algeon_datafusion.backend.device_telemetry_snapshots",
                "failed to read device telemetry snapshots",
                error,
            )
        })
    }

    /// Tests whether both handles refer to the same backend generation.
    ///
    /// Cloned handles compare equal; matching device declarations alone do not.
    pub fn same_backend(&self, other: &Self) -> bool {
        self.core.identity() == other.core.identity()
    }

    pub(crate) fn session_capability(&self) -> &GpuBackendSession {
        &self.session
    }

    /// Whether the session's installed native optimizer is bound to this
    /// backend for query admission.
    ///
    /// Returns false when the session has no native optimizer installed, the
    /// installed optimizer is bound to another backend, or its execution
    /// config carries no query admission authority.
    pub fn is_session_state_bound(&self, state: &SessionState) -> bool {
        installed_algeon_native_optimizer_config(state)
            .is_some_and(|optimizer_config| self.is_execution_config_bound(optimizer_config.native_execution_config()))
    }

    /// Whether the config's query admission source is bound to this backend's
    /// session for query admission.
    pub(crate) fn is_execution_config_bound(&self, config: &AlgeonNativeExecutionConfig) -> bool {
        config.query_admission_source_matches(&self.session)
    }

    /// Returns the planning template bound to this backend's admission authority.
    pub fn optimizer_config(&self) -> &AlgeonNativeOptimizerConfig {
        &self.optimizer_config
    }

    /// Exact-request BFS coalescing counters shared by sessions from this backend.
    #[cfg(feature = "cugraph")]
    pub fn cugraph_bfs_singleflight_snapshot(&self) -> GraphBfsSingleflightSnapshot {
        self.optimizer_config.native_execution_config().bfs_singleflight().snapshot()
    }

    /// Returns the validated device declarations retained at backend construction.
    pub fn devices(&self) -> &[AlgeonGpuDeviceConfig] {
        &self.devices
    }

    fn build_bounded(
        optimizer_config: AlgeonNativeOptimizerConfig,
        configured_devices: Vec<AlgeonGpuDeviceConfig>,
        queue_policy: AlgeonGpuQueuePolicy,
        attempt_completion_queue_capacity: Option<usize>,
    ) -> Result<Self> {
        let validated = validated_backend_construction(
            optimizer_config,
            configured_devices,
            queue_policy,
            attempt_completion_queue_capacity,
        )?;
        let devices = validated.core_config.devices().cloned().collect();
        let core = GpuBackendCore::new(validated.core_config).map_err(|error| {
            backend_core_build_error("failed to initialize ownership-enforcing GPU backend core", error)
        })?;
        let session = core.session_capability();
        let max_device_total_memory_bytes = core
            .device_startup_records()
            .iter()
            .map(|record| record.facts.total_bytes)
            .max()
            .expect("a published backend has a startup record for each validated device");
        // Keep this as the unadmitted planning template. Every execution
        // context is materialized from the device selected by the per-device
        // ledger and its immutable grant.
        let execution_config = validated
            .optimizer_config
            .native_execution_config()
            .clone()
            .with_query_admission_source(QueryAdmissionSource::new(session.clone(), max_device_total_memory_bytes));
        let optimizer_config = validated.optimizer_config.with_native_execution_config(execution_config);
        Ok(AlgeonGpuBackend { core, session, optimizer_config, devices, queue_policy: validated.queue_policy })
    }
}

/// Wraps a native backend-core construction failure for the `build()` boundary.
///
/// Construction refuses with its structured cause (`reason` plus the byte
/// counts the caller needs to pick a declarable capacity) attached as native
/// facts. This wrapper folds that cause into the adapter message so consumers
/// that only print the outermost error still see the actionable numbers.
/// Without the fold, such a consumer sees only the adapter message.
fn backend_core_build_error(message: &'static str, source: QueryError) -> Error {
    Error::from_native_with_blocking_diagnostics("algeon_datafusion.backend.gpu_core", message, source)
}

pub(crate) fn install_backend_optimizer_config(
    builder: SessionStateBuilder,
    optimizer_config: &AlgeonNativeOptimizerConfig,
) -> Result<SessionStateBuilder> {
    let builder = match optimizer_config.native_execution_config().s3_storage() {
        Some(storage) => install_s3_storage(builder, storage)?,
        None => builder,
    };
    let mut builder = builder.with_algeon_native(optimizer_config.clone());
    let version = algeon_version_table_function(optimizer_config.execution_mode());
    builder.table_functions().get_or_insert_with(Default::default).insert(version.name().to_string(), version);
    Ok(builder)
}

/// Validated backend-builder input shared by production construction and root
/// unit tests.
///
/// Normalizes and validates profiles, ownership, queue parameters, and
/// execution configuration prior to lease acquisition or CUDA resource creation.
/// Production `build()` performs the subsequent stage of lease acquisition and
/// `GpuBackendCore` construction.
#[derive(Debug)]
struct ValidatedBackendConstruction {
    optimizer_config: AlgeonNativeOptimizerConfig,
    queue_policy: AlgeonGpuQueuePolicy,
    core_config: ValidatedGpuBackendConfig,
}

fn validated_backend_construction(
    optimizer_config: AlgeonNativeOptimizerConfig,
    configured_devices: Vec<AlgeonGpuDeviceConfig>,
    queue_policy: AlgeonGpuQueuePolicy,
    attempt_completion_queue_capacity: Option<usize>,
) -> Result<ValidatedBackendConstruction> {
    let service_config = backend_query_execution_config(optimizer_config.native_execution_config());
    let mut core_config =
        GpuBackendConfig::new(configured_devices, service_config, backend_queue_bounds(&queue_policy));
    if let Some(capacity) = attempt_completion_queue_capacity {
        core_config = core_config.with_attempt_completion_queue_capacity(capacity);
    }
    let core_config = core_config.validate().map_err(|error| {
        backend_core_build_error("invalid ownership-enforcing GPU backend core configuration", error)
    })?;
    Ok(ValidatedBackendConstruction { optimizer_config, queue_policy, core_config })
}

/// Test fixture for paths that execute an installed GPU backend. The
/// process lease remains production-exclusive; tests serialize construction
/// and deterministically close the backend before releasing the fixture lock.
#[cfg(test)]
pub(crate) struct InstalledTestGpuBackend {
    backend: Option<AlgeonGpuBackend>,
    _serial: MutexGuard<'static, ()>,
}

#[cfg(test)]
impl InstalledTestGpuBackend {
    pub(crate) fn new() -> Self {
        Self::with_config(
            AlgeonNativeExecutionConfig::new(
                SourceChunking::try_new(64 * 1024 * 1024, 2, 3, 1).expect("valid source chunking"),
            ),
            AlgeonGpuDeviceProfile::new(0)
                .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 256 * 1024 * 1024 }),
            AlgeonGpuQueuePolicy::default(),
        )
    }

    /// Builds a serialized backend instance with test-selected immutable resource
    /// and queue envelopes. Tests retain this fixture for the full lifetime of
    /// every config or ticket cloned from its backend.
    pub(crate) fn with_config(
        native_execution_config: AlgeonNativeExecutionConfig,
        device_profile: AlgeonGpuDeviceProfile,
        queue_policy: AlgeonGpuQueuePolicy,
    ) -> Self {
        Self::with_config_and_execution_mode(
            native_execution_config,
            device_profile,
            queue_policy,
            AlgeonGpuExecutionMode::NativePreferred,
        )
    }

    /// Builds a serialized backend instance with an explicit execution mode. This
    /// keeps integration tests on the same frozen backend-installation path as
    /// production cells that require native execution.
    pub(crate) fn with_config_and_execution_mode(
        native_execution_config: AlgeonNativeExecutionConfig,
        device_profile: AlgeonGpuDeviceProfile,
        queue_policy: AlgeonGpuQueuePolicy,
        execution_mode: AlgeonGpuExecutionMode,
    ) -> Self {
        static SERIAL: Mutex<()> = Mutex::new(());

        let serial = SERIAL.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let device = AlgeonGpuDeviceConfig::new(device_profile, GpuMemoryOwnership::WholeDeviceExclusive);
        // Reports of attempts no caller awaits, such as the internal attempt of
        // a plain `collect`, queue here for tests to inspect.
        let backend = AlgeonGpuBackend::builder()
            .with_native_execution_config(native_execution_config)
            .with_execution_mode(execution_mode)
            .devices([device])
            .queue_policy(queue_policy)
            .with_attempt_completion_queue_capacity(64)
            .build()
            .expect("test backend configuration must be valid");
        Self { backend: Some(backend), _serial: serial }
    }

    pub(crate) fn backend(&self) -> &AlgeonGpuBackend {
        self.backend.as_ref().expect("test backend remains live until fixture drop")
    }

    pub(crate) fn optimizer_config(&self) -> &AlgeonNativeOptimizerConfig {
        self.backend().optimizer_config()
    }

    #[cfg(feature = "cugraph")]
    pub(crate) fn admit_query(&self) -> QueryEngineResult<QueryHandle> {
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

impl AlgeonGpuBackendBuilder {
    /// Replaces the optimizer template used by installed sessions.
    ///
    /// Backend construction binds its execution settings to this backend's admission authority.
    pub fn with_optimizer_config(mut self, optimizer_config: AlgeonNativeOptimizerConfig) -> Self {
        self.optimizer_config = optimizer_config;
        self
    }

    /// Retain bounded reports whose completion receivers were abandoned.
    pub fn with_attempt_completion_queue_capacity(mut self, capacity: usize) -> Self {
        self.attempt_completion_queue_capacity = Some(capacity);
        self
    }

    /// Selects relational rewriting and final-plan CPU coverage policy for installed sessions.
    pub fn with_execution_mode(mut self, mode: AlgeonGpuExecutionMode) -> Self {
        self.optimizer_config = self.optimizer_config.with_execution_mode(mode);
        self
    }

    /// Replaces the canonical execution settings copied into admitted plans.
    ///
    /// The execution config is an unadmitted planning template. It never
    /// selects a device or contributes device capacity; those facts are owned
    /// exclusively by [`AlgeonGpuDeviceProfile`].
    pub fn with_native_execution_config(mut self, native_execution_config: AlgeonNativeExecutionConfig) -> Self {
        self.optimizer_config = self.optimizer_config.with_native_execution_config(native_execution_config);
        self
    }

    /// Supplies the complete ordinal-keyed device declaration set.
    pub fn devices(mut self, devices: impl IntoIterator<Item = AlgeonGpuDeviceConfig>) -> Self {
        self.devices = devices.into_iter().collect();
        self
    }

    /// Sets the shared queue-only GPU admission policy.
    pub fn queue_policy(mut self, queue_policy: AlgeonGpuQueuePolicy) -> Self {
        self.queue_policy = queue_policy;
        self
    }

    /// Validates the declaration, claims process GPU ownership, and initializes the backend.
    ///
    /// Returns a typed configuration, discovery, ownership, or resource-initialization
    /// error if construction cannot publish a usable backend. Use [`Self::validate`]
    /// for checks that do not acquire GPU resources.
    pub fn build(self) -> Result<AlgeonGpuBackend> {
        AlgeonGpuBackend::build_bounded(
            self.optimizer_config,
            self.devices,
            self.queue_policy,
            self.attempt_completion_queue_capacity,
        )
    }

    /// Dry-runs builder validation without acquiring GPU resources.
    ///
    /// Runs the pre-discovery normalization and validation performed by
    /// `build()` prior to claiming the process owner lease or touching CUDA.
    /// Validated configuration includes device declarations, installation
    /// bounds, memory-manager options, and queue policy. Capacity-vs-facts
    /// checks still require discovery at `build()` time; combine with
    /// [`AlgeonGpuBackend::probe_device_capacity`] to cover those before
    /// building.
    pub fn validate(self) -> Result<()> {
        validated_backend_construction(
            self.optimizer_config,
            self.devices,
            self.queue_policy,
            self.attempt_completion_queue_capacity,
        )?;
        Ok(())
    }
}

fn backend_query_execution_config(execution_config: &AlgeonNativeExecutionConfig) -> QueryExecutionConfig {
    let SourceChunking {
        max_source_chunk_bytes,
        max_row_groups_per_chunk,
        max_retry_attempts,
        min_retry_chunk_row_groups,
    } = execution_config.source_chunking();
    QueryExecutionConfig {
        lakehouse_cache_fill: execution_config.lakehouse_cache_fill_config().clone(),
        source_storage_profiles: execution_config.source_storage_profiles().to_vec(),
        source_scheduler_policy: execution_config.source_scheduler_policy(),
        source_chunk_planning_mode: execution_config.source_chunk_planning_mode(),
        statistical_aggregate_execution_mode: execution_config.statistical_aggregate_execution_mode(),
        planning_policy: execution_config.planning_policy(),
        aggregate_strategy_execution_options: execution_config.aggregate_strategy_execution_options(),
        source_chunk_planning_options: execution_config.source_chunk_planning_options(),
        max_source_chunk_bytes: max_source_chunk_bytes.get(),
        max_row_groups_per_chunk: max_row_groups_per_chunk.get(),
        max_retry_attempts: max_retry_attempts.get(),
        min_retry_chunk_row_groups: min_retry_chunk_row_groups.get(),
    }
}

fn backend_queue_bounds(queue_policy: &AlgeonGpuQueuePolicy) -> QueueBounds {
    QueueBounds { max_waiters: queue_policy.max_queued_attempts, max_wait: queue_policy.max_queue_wait }
}

#[cfg(test)]
mod tests;
