//! Ownership-enforcing GPU backend construction and lifecycle facade.

use std::{
    any::Any,
    collections::BTreeMap,
    fmt,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Condvar, Mutex, OnceLock, Weak,
        atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use super::{
    DeploymentIoCapability, DeviceCapabilities, DeviceFacts, DeviceLedgerReport, DeviceMemoryFailureReport,
    DeviceTelemetrySnapshot, GpuExecutionPoolSnapshot, GpuExecutionQuerySnapshot, QueryAdmissionRequest,
    QueryExecutionConfig, QueryHandle, QueryService, QueryServiceAdmissionSnapshot,
    QueryServiceDeviceAdmissionSnapshot, QueueBounds,
    backend_lifecycle::{
        BackendLifecycle, GpuLifecyclePermit, ObservationPermit, PublicHandleLease, PublicHandleTracker,
    },
    gpu::{LiveBackendLease, claim_live_backend, normalize_device_ordinals},
    process_exit::{
        ProcessExitPhase, ProcessExitRegistration, ProcessExitShutdown, register_cuda_process_exit_shutdown,
    },
    query_service::{QueryServiceDrainOutcome, QueryServiceDrainPhaseOutcome},
};
use crate::{
    attempt::{
        AttemptBackendCounters, AttemptBackendCountersSnapshot, AttemptBegin, AttemptCompletionQueue, AttemptOwner,
        BeginAttemptRequest,
    },
    error::{QueryError, QueryErrorCode, Result},
    observability::in_runtime_observation_delivery,
};

const BUILD_OPERATION: &str = "nexus_query_engine.gpu_backend_core.build";
const CLOSE_OPERATION: &str = "nexus_query_engine.gpu_backend_core.close";
const RESOURCE_OPERATION: &str = "nexus_query_engine.gpu_backend_core.resource";
const CLEANUP_SLICE: Duration = Duration::from_millis(100);

/// Deployment ownership promised for one CUDA device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GpuMemoryOwnership {
    /// Nexus is the exclusive allocator owner for the complete visible device.
    WholeDeviceExclusive,
    /// Nexus is constrained by an immutable process-manager allocation limit.
    StaticPartition { allocation_limit_bytes: u64 },
}

impl GpuMemoryOwnership {
    pub fn allocation_limit_bytes(self) -> Option<u64> {
        match self {
            Self::WholeDeviceExclusive => None,
            Self::StaticPartition { allocation_limit_bytes } => Some(allocation_limit_bytes),
        }
    }
}

/// Soft admission capacity calculation for one device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GpuDeviceMemoryLimit {
    Automatic { headroom_bytes: u64, fallback_bytes: u64, fraction: Option<GpuMemoryFraction> },
    Explicit { limit_bytes: u64 },
}

impl Default for GpuDeviceMemoryLimit {
    fn default() -> Self {
        Self::Automatic { headroom_bytes: 500 * 1024 * 1024, fallback_bytes: 256 * 1024 * 1024, fraction: None }
    }
}

/// Validated automatic-capacity fraction retained unresolved until post-claim
/// device discovery. The private bit representation makes `Eq`/`Hash` sound:
/// NaN, infinities, zero, and out-of-range values cannot be constructed.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuMemoryFraction(u64);

impl GpuMemoryFraction {
    pub fn new(value: f64) -> Result<Self> {
        if !value.is_finite() || value <= 0.0 || value > 1.0 {
            return Err(invalid_config("memory_limit.fraction", "must be greater than zero and at most one")
                .with_fact("reason", "invalid_fraction"));
        }
        Ok(Self(value.to_bits()))
    }

    pub fn value(self) -> f64 {
        f64::from_bits(self.0)
    }

    pub(crate) fn capacity_bytes(self, owned_capacity_bytes: u64) -> u64 {
        ((owned_capacity_bytes as f64 * self.value()).floor() as u64).min(owned_capacity_bytes)
    }
}

impl fmt::Debug for GpuMemoryFraction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_tuple("GpuMemoryFraction").field(&self.value()).finish()
    }
}

/// Pure per-device construction profile validated before owner acquisition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuBackendDeviceConfig {
    device_ordinal: usize,
    device_capabilities: DeviceCapabilities,
    memory_ownership: GpuMemoryOwnership,
    memory_limit: GpuDeviceMemoryLimit,
    backend_reserve_bytes: u64,
    attempt_floor_bytes: Option<u64>,
    source_work_per_attempt: usize,
    max_active_queries: usize,
    stream_pool_size: usize,
    worker_count: usize,
    lane_count: usize,
    installation_bounds_set: bool,
    install_telemetry: bool,
    install_diagnostics: bool,
}

impl GpuBackendDeviceConfig {
    pub fn new(device_ordinal: usize, memory_ownership: GpuMemoryOwnership) -> Self {
        Self {
            device_ordinal,
            device_capabilities: DeviceCapabilities::default(),
            memory_ownership,
            memory_limit: GpuDeviceMemoryLimit::default(),
            backend_reserve_bytes: 0,
            attempt_floor_bytes: None,
            source_work_per_attempt: 1,
            max_active_queries: 1,
            stream_pool_size: 1,
            worker_count: 1,
            lane_count: 1,
            installation_bounds_set: false,
            install_telemetry: true,
            install_diagnostics: true,
        }
    }

    pub fn with_memory_limit(mut self, memory_limit: GpuDeviceMemoryLimit) -> Self {
        self.memory_limit = memory_limit;
        self
    }

    /// Freezes the native execution capabilities available on this device.
    ///
    /// The default accepts every capability with a modeled native contract.
    /// Supplying an exact profile is useful for heterogeneous deployments and
    /// is consumed by admission before any waiter is queued.
    pub fn with_device_capabilities(mut self, device_capabilities: DeviceCapabilities) -> Self {
        self.device_capabilities = device_capabilities;
        self
    }

    pub fn with_backend_reserve_bytes(mut self, backend_reserve_bytes: u64) -> Self {
        self.backend_reserve_bytes = backend_reserve_bytes;
        self
    }

    /// Sets the immutable minimum managed-memory floor for each active attempt.
    /// When unset, it resolves per device to
    /// `managed_capacity_bytes / max(8, max_active_queries)` so the frozen
    /// floor-times-active-limit invariant always holds by default.
    pub fn with_attempt_floor_bytes(mut self, attempt_floor_bytes: u64) -> Self {
        self.attempt_floor_bytes = Some(attempt_floor_bytes);
        self
    }

    /// Sets source work reserved per active attempt.
    ///
    /// This is part of the immutable device envelope; execution/session
    /// source scheduling controls must not change controller capacity.
    pub fn with_source_work_per_attempt(mut self, source_work_per_attempt: usize) -> Self {
        self.source_work_per_attempt = source_work_per_attempt;
        self
    }

    pub fn with_max_active_queries(mut self, max_active_queries: usize) -> Self {
        self.max_active_queries = max_active_queries;
        self
    }

    pub fn with_installation_bounds(mut self, stream_pool_size: usize, worker_count: usize, lane_count: usize) -> Self {
        self.stream_pool_size = stream_pool_size;
        self.worker_count = worker_count;
        self.lane_count = lane_count;
        self.installation_bounds_set = true;
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

    /// Immutable native execution capabilities selected for this device.
    pub fn device_capabilities(&self) -> &DeviceCapabilities {
        &self.device_capabilities
    }

    pub fn memory_ownership(&self) -> GpuMemoryOwnership {
        self.memory_ownership
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

    pub fn max_active_queries(&self) -> usize {
        self.max_active_queries
    }

    pub fn stream_pool_size(&self) -> usize {
        self.stream_pool_size
    }

    pub fn worker_count(&self) -> usize {
        self.worker_count
    }

    pub fn lane_count(&self) -> usize {
        self.lane_count
    }
}

/// Pure, resource-independent policy for the bounded controller.
///
/// Resource-derived fields in the supplied policy are deliberately cleared.
/// The engine resolves them from immutable post-claim device plans; root
/// code cannot feed an ordinal-zero or common-minimum capacity back into
/// construction.
#[derive(Debug, Clone)]
pub struct GpuBackendControllerDraft {
    execution_config: QueryExecutionConfig,
    queue_bounds: QueueBounds,
}

impl GpuBackendControllerDraft {
    /// Creates a controller draft from independent execution and queue policy.
    pub fn new(execution_config: QueryExecutionConfig, queue_bounds: QueueBounds) -> Self {
        Self { execution_config, queue_bounds }
    }

    pub(crate) fn execution_config(&self) -> &QueryExecutionConfig {
        &self.execution_config
    }

    pub(crate) fn queue_bounds(&self) -> &QueueBounds {
        &self.queue_bounds
    }

    pub(crate) fn into_parts(self) -> (QueryExecutionConfig, QueueBounds) {
        (self.execution_config, self.queue_bounds)
    }
}

/// Complete pure P0 construction draft.
///
/// The fields stay private so validation cannot be bypassed with a struct
/// literal. Automatic values and device-local resource profiles remain unresolved until the
/// engine owns the complete device set and has frozen every device fact.
#[derive(Debug, Clone)]
pub struct GpuBackendCoreDraft {
    devices: Vec<GpuBackendDeviceConfig>,
    controller: GpuBackendControllerDraft,
    deployment_io_capability: DeploymentIoCapability,
    attempt_completion_queue_capacity: Option<usize>,
}

impl GpuBackendCoreDraft {
    pub fn new(
        devices: Vec<GpuBackendDeviceConfig>,
        controller: GpuBackendControllerDraft,
        deployment_io_capability: DeploymentIoCapability,
    ) -> Self {
        Self { devices, controller, deployment_io_capability, attempt_completion_queue_capacity: None }
    }

    pub fn with_attempt_completion_queue_capacity(mut self, capacity: usize) -> Self {
        self.attempt_completion_queue_capacity = Some(capacity);
        self
    }

    pub fn devices(&self) -> &[GpuBackendDeviceConfig] {
        &self.devices
    }

    pub fn derived_global_max_active_queries(&self) -> Result<usize> {
        self.devices.iter().try_fold(0usize, |sum, device| {
            sum.checked_add(device.max_active_queries).ok_or_else(|| {
                invalid_config("max_active_queries", "derived global maximum overflowed")
                    .with_fact("reason", "derived_global_max_overflow")
            })
        })
    }

    /// Resolves derived installation bounds before validation.
    ///
    /// Devices without explicit installation bounds inherit the frozen
    /// `A_d * (1 + S_d)` stream/job-slot formula with `A_d` root lanes;
    /// explicit bounds are validated against that formula instead.
    pub(crate) fn normalize_installation_bounds(&mut self) -> Result<()> {
        for device in &mut self.devices {
            let source_work = device.source_work_per_attempt;
            let aux = device.max_active_queries.checked_mul(source_work).ok_or_else(|| {
                invalid_device_config(device.device_ordinal, "source_work_per_attempt", "overflowed")
                    .with_fact("reason", "aux_source_work_overflow")
            })?;
            let required = device.max_active_queries.checked_add(aux).ok_or_else(|| {
                invalid_device_config(device.device_ordinal, "stream_pool_size", "overflowed")
                    .with_fact("reason", "required_slot_overflow")
            })?;
            if device.installation_bounds_set {
                if device.stream_pool_size < required {
                    return Err(invalid_device_config(
                        device.device_ordinal,
                        "stream_pool_size",
                        "must cover root and auxiliary source work for every active attempt",
                    )
                    .with_fact("reason", "stream_pool_below_required_formula")
                    .with_fact("stream_pool_size", device.stream_pool_size)
                    .with_fact("required_slot_capacity", required)
                    .with_fact("max_active_queries", device.max_active_queries)
                    .with_fact("source_work_per_attempt", source_work));
                }
            } else {
                device.stream_pool_size = required;
                device.lane_count = device.max_active_queries;
                // One root worker scope exists for every concurrently
                // admitted attempt. Auxiliary work remains bounded by its
                // class permits and stream slots; leaving the constructor's
                // single-worker default here would make a public `A_d > 1`
                // profile fail validation after automatic derivation.
                device.worker_count = device.max_active_queries;
            }
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        let ordinals = self.devices.iter().map(|device| device.device_ordinal).collect::<Vec<_>>();
        normalize_device_ordinals(&ordinals)?;
        for device in &self.devices {
            validate_device_config(device)?;
        }
        let derived = self.derived_global_max_active_queries()?;
        if derived == 0 {
            return Err(invalid_config("max_active_queries", "derived global maximum must be positive")
                .with_fact("reason", "zero"));
        }
        validate_controller_draft(&self.controller, &ordinals, &self.deployment_io_capability)
    }
}

fn validate_controller_draft(
    controller: &GpuBackendControllerDraft,
    ordinals: &[usize],
    deployment: &DeploymentIoCapability,
) -> Result<()> {
    QueryService::validate_controller_policy_for_backend(
        controller.execution_config(),
        controller.queue_bounds(),
        ordinals,
        deployment,
    )
}

fn validate_device_config(device: &GpuBackendDeviceConfig) -> Result<()> {
    for (field, value) in [
        ("max_active_queries", device.max_active_queries),
        ("source_work_per_attempt", device.source_work_per_attempt),
        ("stream_pool_size", device.stream_pool_size),
        ("worker_count", device.worker_count),
        ("lane_count", device.lane_count),
    ] {
        if value == 0 {
            return Err(invalid_device_config(device.device_ordinal, field, "must be greater than zero")
                .with_fact("reason", "zero"));
        }
    }
    if device.attempt_floor_bytes == Some(0) {
        return Err(invalid_device_config(device.device_ordinal, "attempt_floor_bytes", "must be greater than zero")
            .with_fact("reason", "zero"));
    }
    for (field, value) in [
        ("stream_pool_size", device.stream_pool_size),
        ("worker_count", device.worker_count),
        ("lane_count", device.lane_count),
    ] {
        if value < device.max_active_queries {
            return Err(invalid_device_config(
                device.device_ordinal,
                field,
                "must cover every configured active query slot",
            )
            .with_fact("reason", "installation_bound_below_active_limit")
            .with_fact("installation_bound", value)
            .with_fact("max_active_queries", device.max_active_queries));
        }
    }
    match device.memory_limit {
        GpuDeviceMemoryLimit::Automatic { fallback_bytes, .. } if fallback_bytes == 0 => {
            return Err(invalid_device_config(
                device.device_ordinal,
                "memory_limit.fallback_bytes",
                "must be greater than zero",
            )
            .with_fact("reason", "zero"));
        },
        GpuDeviceMemoryLimit::Automatic { fallback_bytes, .. } if device.backend_reserve_bytes >= fallback_bytes => {
            return Err(invalid_device_config(
                device.device_ordinal,
                "backend_reserve_bytes",
                "must be below the automatic fallback capacity",
            )
            .with_fact("reason", "reserve_exhausts_fallback")
            .with_fact("fallback_bytes", fallback_bytes));
        },
        GpuDeviceMemoryLimit::Explicit { limit_bytes } if limit_bytes == 0 => {
            return Err(invalid_device_config(
                device.device_ordinal,
                "memory_limit.limit_bytes",
                "must be greater than zero",
            )
            .with_fact("reason", "zero"));
        },
        GpuDeviceMemoryLimit::Explicit { limit_bytes } if device.backend_reserve_bytes >= limit_bytes => {
            return Err(invalid_device_config(
                device.device_ordinal,
                "backend_reserve_bytes",
                "must be below the explicit device memory limit",
            )
            .with_fact("reason", "reserve_exhausts_explicit_limit")
            .with_fact("limit_bytes", limit_bytes));
        },
        _ => {},
    }
    if let GpuMemoryOwnership::StaticPartition { allocation_limit_bytes } = device.memory_ownership {
        if allocation_limit_bytes == 0 {
            return Err(invalid_device_config(
                device.device_ordinal,
                "memory_ownership.allocation_limit_bytes",
                "must be greater than zero",
            )
            .with_fact("reason", "zero"));
        }
        native_allocation_limit_bytes(device)?;
        if device.backend_reserve_bytes >= allocation_limit_bytes {
            return Err(invalid_device_config(
                device.device_ordinal,
                "backend_reserve_bytes",
                "must be below the enforced static partition",
            )
            .with_fact("reason", "reserve_exhausts_static_partition")
            .with_fact("allocation_limit_bytes", allocation_limit_bytes));
        }
        if let GpuDeviceMemoryLimit::Explicit { limit_bytes } = device.memory_limit
            && limit_bytes > allocation_limit_bytes
        {
            return Err(invalid_device_config(
                device.device_ordinal,
                "memory_limit.limit_bytes",
                "cannot exceed the enforced static partition",
            )
            .with_fact("reason", "limit_exceeds_static_partition")
            .with_fact("allocation_limit_bytes", allocation_limit_bytes));
        }
    }
    let allocation_limit_bytes = native_allocation_limit_bytes(device)?;
    cudf::DeviceMemoryManagerOptions { allocation_limit_bytes, ..Default::default() }.validate().map_err(
        |error| {
            QueryError::from_cudf(BUILD_OPERATION, "invalid per-device memory manager installation options", error)
                .with_fact("field", "device_memory_manager_options")
                .with_fact("device_ordinal", device.device_ordinal)
                .with_fact("reason", "invalid_device_memory_manager_options")
        },
    )?;
    Ok(())
}

fn native_allocation_limit_bytes(device: &GpuBackendDeviceConfig) -> Result<Option<usize>> {
    device
        .memory_ownership
        .allocation_limit_bytes()
        .map(|limit| {
            usize::try_from(limit).map_err(|_error| {
                invalid_device_config(
                    device.device_ordinal,
                    "memory_ownership.allocation_limit_bytes",
                    "does not fit the native allocation-size type",
                )
                .with_fact("reason", "out_of_range")
                .with_fact("allocation_limit_bytes", limit)
            })
        })
        .transpose()
}

fn invalid_device_config(device_ordinal: usize, field: &'static str, message: &'static str) -> QueryError {
    invalid_config(field, message).with_fact("device_ordinal", device_ordinal)
}

fn invalid_config(field: &'static str, message: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::InvalidPlan, BUILD_OPERATION, message).with_fact("field", field)
}

/// Construction boundary exposed to deterministic probes and failure injectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub(crate) enum GpuBackendConstructionCheckpoint {
    BeforeDeviceDiscovery { device_ordinal: usize },
    AfterDeviceDiscovery { device_ordinal: usize },
    BeforeMemoryManager { device_ordinal: usize },
    AfterMemoryManager { device_ordinal: usize },
    BeforeStreamPool { device_ordinal: usize },
    AfterStreamPool { device_ordinal: usize },
    BeforeDeviceContext { device_ordinal: usize },
    AfterDeviceContext { device_ordinal: usize },
    BeforeWorkerPool { device_ordinal: usize },
    AfterWorkerPool { device_ordinal: usize },
    BeforeStartupRecords,
    AfterStartupRecords,
    BeforeCleanupWorker,
    AfterCleanupWorker,
    Published,
}

/// Injectable construction observer. Returning an error injects failure at the
/// named transaction boundary.
pub(crate) trait GpuBackendConstructionProbe: Send + Sync + fmt::Debug {
    fn checkpoint(&self, checkpoint: GpuBackendConstructionCheckpoint) -> Result<()>;
}

#[derive(Debug)]
struct NoopConstructionProbe;

impl GpuBackendConstructionProbe for NoopConstructionProbe {
    fn checkpoint(&self, _checkpoint: GpuBackendConstructionCheckpoint) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
trait ObservationCompletionExecutorFactory: Send + Sync + fmt::Debug {
    fn spawn(&self) -> Result<ObservationCompletionExecutor>;
}

#[cfg(test)]
#[derive(Debug)]
struct DefaultObservationCompletionExecutorFactory;

#[cfg(test)]
impl ObservationCompletionExecutorFactory for DefaultObservationCompletionExecutorFactory {
    fn spawn(&self) -> Result<ObservationCompletionExecutor> {
        ObservationCompletionExecutor::spawn()
    }
}

/// Builder for the one public GPU backend construction seam.
#[derive(Debug)]
pub struct GpuBackendCoreBuilder {
    draft: GpuBackendCoreDraft,
    probe: Arc<dyn GpuBackendConstructionProbe>,
    #[cfg(test)]
    observation_completion_executor_factory: Arc<dyn ObservationCompletionExecutorFactory>,
}

impl GpuBackendCoreBuilder {
    #[cfg(test)]
    pub(crate) fn with_construction_probe(mut self, probe: Arc<dyn GpuBackendConstructionProbe>) -> Self {
        self.probe = probe;
        self
    }

    #[cfg(test)]
    fn with_observation_completion_executor_factory(
        mut self,
        factory: Arc<dyn ObservationCompletionExecutorFactory>,
    ) -> Self {
        self.observation_completion_executor_factory = factory;
        self
    }

    pub fn build(self) -> Result<GpuBackendCore> {
        self.build_with_construction_journal(Arc::new(ConstructionResourceJournal::default()))
    }

    fn build_with_construction_journal(mut self, journal: Arc<ConstructionResourceJournal>) -> Result<GpuBackendCore> {
        self.draft.normalize_installation_bounds()?;
        self.draft.validate()?;
        #[cfg(test)]
        let observation_completion_executor_factory = Arc::clone(&self.observation_completion_executor_factory);
        let GpuBackendCoreDraft { devices, controller, deployment_io_capability, attempt_completion_queue_capacity } =
            self.draft;
        let ordinals = devices.iter().map(|device| device.device_ordinal).collect::<Vec<_>>();
        let owner_lease = claim_live_backend(&ordinals)?;

        // The generation shell and cleanup worker are the rollback journal for
        // every post-claim construction step. In particular, the live owner
        // lease is never held only by this builder's stack while GPU resources
        // are being installed.
        let generation_id = next_backend_identity();
        let generation = Arc::new(ResourceGeneration::new(generation_id, owner_lease, Arc::clone(&journal)));
        let resource_core = Arc::new(ResourceCore {
            identity: next_backend_identity(),
            generation: Mutex::new(Some(Arc::clone(&generation))),
        });
        let observation_core = Arc::new(ObservationCore {
            identity: next_backend_identity(),
            lifecycle: BackendLifecycle::new(),
            attempt_counters: Arc::new(AttemptBackendCounters::default()),
            attempt_completion_queue: attempt_completion_queue_capacity
                .and_then(AttemptCompletionQueue::new)
                .map(Arc::new),
            completion_executor: {
                #[cfg(test)]
                {
                    observation_completion_executor_factory.spawn()?
                }
                #[cfg(not(test))]
                {
                    ObservationCompletionExecutor::spawn()?
                }
            },
        });

        self.probe.checkpoint(GpuBackendConstructionCheckpoint::BeforeCleanupWorker)?;
        let coordinator = CloseCoordinator::spawn(Arc::clone(&resource_core), Arc::clone(&observation_core))?;
        let mut rollback = ConstructionRollback::new(Arc::clone(&coordinator));

        let build_result = (|| {
            self.probe.checkpoint(GpuBackendConstructionCheckpoint::AfterCleanupWorker)?;
            let token = generation.owner_lease.token();
            let service = QueryService::install_for_backend(
                ordinals,
                controller,
                deployment_io_capability,
                &token,
                &devices,
                self.probe.as_ref(),
                &journal,
            )?;
            drop(token);

            generation.service.set(service).map_err(|_service| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    BUILD_OPERATION,
                    "backend resource generation was initialized more than once",
                )
                .with_fact("reason", "duplicate_resource_generation_install")
            })?;
            // CUDA has been initialized while installing the service-owned
            // device resources. Register the coordinator only now so its
            // `atexit` handler runs before libcudart's finalizer.
            observation_core.completion_executor.activate_process_exit();
            coordinator.activate_process_exit();

            let service =
                generation.service().expect("resource service is available immediately after successful installation");

            let observed = service.backend_device_startup_facts();
            let worker_pools = service.gpu_execution_pool_reports();
            if observed.len() != devices.len() || worker_pools.len() != devices.len() {
                return Err(QueryError::new(
                    QueryErrorCode::Internal,
                    BUILD_OPERATION,
                    "constructed GPU resources do not cover every validated device profile",
                )
                .with_fact("reason", "incomplete_device_startup_records")
                .with_fact("profile_count", devices.len())
                .with_fact("device_record_count", observed.len())
                .with_fact("worker_pool_count", worker_pools.len()));
            }
            let mut startup_records = Vec::with_capacity(devices.len());
            for profile in &devices {
                let (facts, capacity_bytes, installation, health_identity, resource_profile) = observed
                    .iter()
                    .find(|(facts, _, _, _, _)| facts.device == profile.device_ordinal)
                    .cloned()
                    .ok_or_else(|| {
                        QueryError::new(
                            QueryErrorCode::Internal,
                            BUILD_OPERATION,
                            "validated device has no immutable startup record",
                        )
                        .with_fact("reason", "device_startup_record_missing")
                        .with_fact("device_ordinal", profile.device_ordinal)
                    })?;
                let worker_pool = worker_pools
                    .iter()
                    .find(|pool| pool.device_ordinal == profile.device_ordinal)
                    .ok_or_else(|| {
                        QueryError::new(
                            QueryErrorCode::Internal,
                            BUILD_OPERATION,
                            "validated device has no GPU worker pool",
                        )
                        .with_fact("reason", "device_worker_pool_missing")
                        .with_fact("device_ordinal", profile.device_ordinal)
                    })?;
                if worker_pool.worker_count != profile.worker_count {
                    return Err(QueryError::new(
                        QueryErrorCode::Internal,
                        BUILD_OPERATION,
                        "GPU worker pool does not match its validated device profile",
                    )
                    .with_fact("reason", "device_worker_pool_mismatch")
                    .with_fact("device_ordinal", profile.device_ordinal)
                    .with_fact("profile_worker_count", profile.worker_count)
                    .with_fact("installed_worker_count", worker_pool.worker_count));
                }
                let allocation_limit_bytes = native_allocation_limit_bytes(profile)?;
                let installed_allocation_limit =
                    installation.effective_options.map(|options| options.allocation_limit_bytes);
                if installed_allocation_limit != Some(allocation_limit_bytes) {
                    return Err(QueryError::new(
                        QueryErrorCode::Internal,
                        BUILD_OPERATION,
                        "device memory manager record does not match its validated installation profile",
                    )
                    .with_fact("reason", "device_memory_manager_record_mismatch")
                    .with_fact("device_ordinal", profile.device_ordinal)
                    .with_fact("expected_allocation_limit_bytes", format!("{allocation_limit_bytes:?}"))
                    .with_fact("installed_options", format!("{:?}", installation.effective_options)));
                }
                startup_records.push(GpuBackendDeviceStartupRecord {
                    device_ordinal: facts.device,
                    facts,
                    capacity_bytes,
                    resource_profile,
                    memory_ownership: profile.memory_ownership,
                    memory_limit: profile.memory_limit,
                    backend_reserve_bytes: profile.backend_reserve_bytes,
                    max_active_queries: profile.max_active_queries,
                    stream_pool_size: profile.stream_pool_size,
                    worker_count: worker_pool.worker_count,
                    lane_count: profile.lane_count,
                    install_telemetry: profile.install_telemetry,
                    install_diagnostics: profile.install_diagnostics,
                    memory_manager_installation: installation,
                    device_health_generation: health_identity.device_health_generation,
                    backend_health_generation: health_identity.backend_health_generation,
                });
            }

            self.probe.checkpoint(GpuBackendConstructionCheckpoint::BeforeStartupRecords)?;
            let startup_records: Arc<[GpuBackendDeviceStartupRecord]> = startup_records.into();
            journal.record_startup_records(Arc::clone(&startup_records))?;
            generation.startup_records.set(Arc::clone(&startup_records)).map_err(|_records| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    BUILD_OPERATION,
                    "backend startup records were initialized more than once",
                )
                .with_fact("reason", "duplicate_startup_record_install")
            })?;
            self.probe.checkpoint(GpuBackendConstructionCheckpoint::AfterStartupRecords)?;
            let close_on_last = Arc::clone(&coordinator);
            let public_handles = PublicHandleTracker::new(move || {
                close_on_last.signal();
            });
            let public_handle = public_handles.acquire().expect("new public-handle tracker accepts its first lease");
            self.probe.checkpoint(GpuBackendConstructionCheckpoint::Published)?;
            Ok(GpuBackendCore {
                resource_core,
                observation_core,
                coordinator,
                public_handle,
                startup_records,
                generation_id,
            })
        })();
        match build_result {
            Ok(core) => {
                rollback.disarm();
                Ok(core)
            },
            Err(error) => Err(rollback.attach_cleanup_facts(error)),
        }
    }
}

/// Immutable per-device record frozen before backend publication.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuBackendDeviceStartupRecord {
    pub device_ordinal: usize,
    pub facts: DeviceFacts,
    pub capacity_bytes: u64,
    /// Full ordinal-local resource-controller profile frozen after discovery.
    pub resource_profile: crate::runtime::DeviceResourceProfile,
    pub memory_ownership: GpuMemoryOwnership,
    pub memory_limit: GpuDeviceMemoryLimit,
    pub backend_reserve_bytes: u64,
    pub max_active_queries: usize,
    pub stream_pool_size: usize,
    pub worker_count: usize,
    pub lane_count: usize,
    pub install_telemetry: bool,
    pub install_diagnostics: bool,
    pub memory_manager_installation: cudf::DeviceMemoryManagerInstallationState,
    /// Persistent engine health epoch bound to this device installation.
    pub device_health_generation: u64,
    /// Backend-owner health epoch for this published core generation.
    pub backend_health_generation: u64,
}

/// Shared identity facts used to prove facade/session clones reuse one domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuBackendCoreIdentity {
    pub resource_core: u64,
    pub observation_core: u64,
    pub resource_generation: u64,
}

/// One process-owned GPU backend core.
#[derive(Debug)]
pub struct GpuBackendCore {
    resource_core: Arc<ResourceCore>,
    observation_core: Arc<ObservationCore>,
    coordinator: Arc<CloseCoordinator>,
    public_handle: PublicHandleLease,
    startup_records: Arc<[GpuBackendDeviceStartupRecord]>,
    generation_id: u64,
}

impl Clone for GpuBackendCore {
    fn clone(&self) -> Self {
        Self {
            resource_core: Arc::clone(&self.resource_core),
            observation_core: Arc::clone(&self.observation_core),
            coordinator: Arc::clone(&self.coordinator),
            public_handle: self.public_handle.clone(),
            startup_records: Arc::clone(&self.startup_records),
            generation_id: self.generation_id,
        }
    }
}

impl GpuBackendCore {
    pub fn builder(draft: GpuBackendCoreDraft) -> GpuBackendCoreBuilder {
        GpuBackendCoreBuilder {
            draft,
            probe: Arc::new(NoopConstructionProbe),
            #[cfg(test)]
            observation_completion_executor_factory: Arc::new(DefaultObservationCompletionExecutorFactory),
        }
    }

    pub fn session_capability(&self) -> GpuBackendSession {
        GpuBackendSession {
            resource_core: Arc::downgrade(&self.resource_core),
            observation_core: Arc::clone(&self.observation_core),
            identities: self.identity(),
        }
    }

    pub fn device_startup_records(&self) -> &[GpuBackendDeviceStartupRecord] {
        &self.startup_records
    }

    pub fn identity(&self) -> GpuBackendCoreIdentity {
        GpuBackendCoreIdentity {
            resource_core: self.resource_core.identity,
            observation_core: self.observation_core.identity,
            resource_generation: self.generation_id,
        }
    }

    pub fn begin_attempt(&self, request: BeginAttemptRequest) -> Result<AttemptBegin> {
        self.session_capability().begin_attempt(request)
    }

    pub fn close(&self) -> bool {
        self.coordinator.signal()
    }

    pub fn close_snapshot(&self) -> GpuBackendCloseSnapshot {
        self.coordinator.state.snapshot(&self.observation_core.lifecycle)
    }

    pub fn close_and_wait(&self, timeout: Duration) -> Result<GpuBackendCloseSnapshot> {
        reject_reentrant_observation_close()?;
        let deadline = Instant::now().checked_add(timeout).unwrap_or_else(Instant::now);
        self.close_and_wait_until(deadline)
    }

    pub fn close_and_wait_until(&self, deadline: Instant) -> Result<GpuBackendCloseSnapshot> {
        reject_reentrant_observation_close()?;
        self.coordinator.close_and_wait_until(deadline)
    }

    pub async fn close_and_wait_async(&self, timeout: Duration) -> Result<GpuBackendCloseSnapshot> {
        reject_reentrant_observation_close()?;
        let coordinator = Arc::clone(&self.coordinator);
        tokio::task::spawn_blocking(move || {
            let deadline = Instant::now().checked_add(timeout).unwrap_or_else(Instant::now);
            coordinator.close_and_wait_until(deadline)
        })
        .await
        .map_err(|error| {
            QueryError::new(QueryErrorCode::Internal, CLOSE_OPERATION, "backend close wait task failed")
                .with_fact("reason", "close_wait_task_failed")
                .with_fact("join_error", error.to_string())
        })?
    }
}

/// Opaque capability installed into session/optimizer state.
///
/// It cannot construct or expose raw runtime resources and its weak resource
/// reference cannot keep device ownership alive after public facade close.
#[derive(Clone, Debug)]
pub struct GpuBackendSession {
    resource_core: Weak<ResourceCore>,
    observation_core: Arc<ObservationCore>,
    identities: GpuBackendCoreIdentity,
}

impl GpuBackendSession {
    pub fn identity(&self) -> GpuBackendCoreIdentity {
        self.identities
    }

    /// Returns whether the backend close gate has been signalled.
    ///
    /// An attempt that won the begin gate observes this to terminalize
    /// `unplanned` with `backend_closing` evidence instead of starting
    /// planning; the begin gate itself remains the only linearization point.
    pub fn is_closing(&self) -> bool {
        self.observation_core.lifecycle.is_closing()
    }

    pub fn begin_attempt(&self, request: BeginAttemptRequest) -> Result<AttemptBegin> {
        let permit = self.observation_core.lifecycle.try_begin_observation()?;
        self.observation_core.attempt_counters.attempts_begun.fetch_add(1, Ordering::AcqRel);
        Ok(AttemptOwner::begin(
            request,
            permit,
            Arc::clone(&self.observation_core.attempt_counters),
            self.observation_core.attempt_completion_queue.clone(),
        ))
    }

    pub fn take_abandoned_attempt_reports(&self, max: usize) -> Vec<Arc<crate::observability::QueryAttemptReport>> {
        self.observation_core.attempt_completion_queue.as_ref().map_or_else(Vec::new, |queue| queue.drain(max))
    }

    pub fn attempt_counters_snapshot(&self) -> AttemptBackendCountersSnapshot {
        self.observation_core.attempt_counters.snapshot()
    }

    fn acquire_resource(&self) -> Result<(ObservationPermit, GpuResourcePermit, ObservationCompletionDispatcher)> {
        let observation = self.observation_core.lifecycle.try_begin_observation()?;
        let resource_core = self.resource_core.upgrade().ok_or_else(backend_closing_resource_error)?;
        let generation = resource_core.generation.lock().map_err(|_error| resource_core_poisoned_error())?;
        let lifecycle = self.observation_core.lifecycle.try_begin_gpu_resource()?;
        let generation = generation.as_ref().cloned().ok_or_else(backend_closing_resource_error)?;
        Ok((
            observation,
            GpuResourcePermit { generation, lifecycle },
            self.observation_core.completion_executor.dispatcher(),
        ))
    }

    pub fn admit_query_with_request(&self, request: QueryAdmissionRequest) -> Result<QueryHandle> {
        let (observation, resource, completion_dispatcher) = self.acquire_resource()?;
        let service = resource.generation.service().cloned().ok_or_else(resource_generation_unpublished_error)?;
        service.admit_query_with_backend_permits(request, observation, resource, completion_dispatcher)
    }

    pub fn admit_query(&self) -> Result<QueryHandle> {
        self.admit_query_with_request(QueryAdmissionRequest::new())
    }

    pub fn try_admit_query_with_request(&self, request: QueryAdmissionRequest) -> Result<Option<QueryHandle>> {
        let (observation, resource, completion_dispatcher) = self.acquire_resource()?;
        let service = resource.generation.service().cloned().ok_or_else(resource_generation_unpublished_error)?;
        service.try_admit_query_with_backend_permits(request, observation, resource, completion_dispatcher)
    }

    pub async fn admit_async_with_request(&self, request: QueryAdmissionRequest) -> Result<QueryHandle> {
        let (observation, resource, completion_dispatcher) = self.acquire_resource()?;
        let service = resource.generation.service().cloned().ok_or_else(resource_generation_unpublished_error)?;
        service.admit_async_with_backend_permits(request, observation, resource, completion_dispatcher).await
    }

    pub fn max_device_total_memory_bytes(&self) -> Result<u64> {
        self.with_service(|service| Ok(service.max_device_total_memory_bytes()))
    }

    pub fn same_backend(&self, other: &Self) -> bool {
        self.identities == other.identities
    }

    pub fn config(&self) -> Result<QueryExecutionConfig> {
        self.with_service(|service| Ok(service.config().clone()))
    }

    pub fn queue_bounds(&self) -> Result<QueueBounds> {
        self.with_service(|service| Ok(service.queue_bounds().clone()))
    }

    pub fn admission_snapshot(&self) -> Result<QueryServiceAdmissionSnapshot> {
        self.with_service(QueryService::admission_snapshot)
    }

    pub fn device_admission_snapshots(&self) -> Result<Vec<QueryServiceDeviceAdmissionSnapshot>> {
        self.with_service(|service| Ok(service.device_admission_snapshots()))
    }

    pub fn device_ledger_reports(&self) -> Result<Vec<DeviceLedgerReport>> {
        self.with_service(QueryService::device_ledger_reports)
    }

    pub fn device_memory_failure_reports(&self) -> Result<Vec<DeviceMemoryFailureReport>> {
        self.with_service(|service| Ok(service.device_memory_failure_reports()))
    }

    pub fn device_telemetry_snapshots(&self) -> Result<Vec<DeviceTelemetrySnapshot>> {
        self.with_service(|service| Ok(service.device_telemetry_snapshots()))
    }

    pub fn gpu_execution_pool_reports(&self) -> Result<Vec<GpuExecutionPoolSnapshot>> {
        self.with_service(|service| Ok(service.gpu_execution_pool_reports()))
    }

    pub fn gpu_execution_query_report(&self) -> Result<GpuExecutionQuerySnapshot> {
        self.with_service(|service| Ok(service.gpu_execution_query_report()))
    }

    fn with_service<T>(&self, operation: impl FnOnce(&QueryService) -> Result<T>) -> Result<T> {
        let (_observation, resource, _completion_dispatcher) = self.acquire_resource()?;
        let service = resource.generation.service().ok_or_else(resource_generation_unpublished_error)?;
        operation(service)
    }
}

/// Permit required by private service admission.
///
/// The resource generation field is declared before the lifecycle count so it
/// is released first; a close waiter that observes the count reach zero then
/// knows no permit can still retain the generation.
pub(crate) struct GpuResourcePermit {
    generation: Arc<ResourceGeneration>,
    lifecycle: GpuLifecyclePermit,
}

impl fmt::Debug for GpuResourcePermit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GpuResourcePermit")
            .field("resource_generation", &self.generation.generation_id)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
struct ResourceCore {
    identity: u64,
    generation: Mutex<Option<Arc<ResourceGeneration>>>,
}

#[derive(Debug)]
struct ObservationCore {
    identity: u64,
    lifecycle: BackendLifecycle,
    attempt_counters: Arc<AttemptBackendCounters>,
    attempt_completion_queue: Option<Arc<AttemptCompletionQueue>>,
    completion_executor: ObservationCompletionExecutor,
}

type ObservationCompletionCallback = Box<dyn FnOnce() + Send + 'static>;

/// A completion callback that recovers itself if the worker drops it before
/// explicitly running it.
///
/// A receiver drops its queued messages when the worker exits unexpectedly.
/// Permit release and observer sealing are terminal work, so silently dropping
/// either would strand backend observation quiescence.
struct ObservationCompletionTask {
    callback: Option<ObservationCompletionCallback>,
}

impl ObservationCompletionTask {
    fn new<F>(callback: F) -> Self
    where F: FnOnce() + Send + 'static {
        Self { callback: Some(Box::new(callback)) }
    }

    fn run(mut self) {
        if let Some(callback) = self.callback.take() {
            execute_observation_completion(callback);
        }
    }
}

impl Drop for ObservationCompletionTask {
    fn drop(&mut self) {
        if let Some(callback) = self.callback.take() {
            execute_observation_completion(callback);
        }
    }
}

enum ObservationCompletionMessage {
    Completion(ObservationCompletionTask),
    #[cfg(test)]
    InjectWorkerFaultForTest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ObservationCompletionWorkerLane {
    Observation,
    ActivityTail,
}

impl ObservationCompletionWorkerLane {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Observation => "observation",
            Self::ActivityTail => "activity_tail",
        }
    }

    const fn thread_name(self) -> &'static str {
        match self {
            Self::Observation => "nexus-observation-completion-observation",
            Self::ActivityTail => "nexus-observation-completion-activity-tail",
        }
    }

    const fn slot(self) -> usize {
        match self {
            Self::Observation => 0,
            Self::ActivityTail => 1,
        }
    }
}

const OBSERVATION_COMPLETION_WORKER_RUNNING: u8 = 0;
const OBSERVATION_COMPLETION_WORKER_STOPPED: u8 = 1;
const OBSERVATION_COMPLETION_POLL_INTERVAL: Duration = Duration::from_millis(10);

/// Supervisor-visible state for the detached completion worker.
///
/// A caught supervisor fault is retained as a diagnostic without destroying
/// the only pre-created delivery thread. The worker continues consuming its
/// receiver, which keeps completion off callers such as the stream-cleanup
/// supervisor. Receiver teardown still recovers queued RAII tasks.
#[derive(Debug)]
struct ObservationCompletionExecutorState {
    worker_terminal: AtomicU8,
    worker_fault_count: AtomicU64,
    shutdown_requested: AtomicBool,
}

impl ObservationCompletionExecutorState {
    fn new() -> Self {
        Self {
            worker_terminal: AtomicU8::new(OBSERVATION_COMPLETION_WORKER_RUNNING),
            worker_fault_count: AtomicU64::new(0),
            shutdown_requested: AtomicBool::new(false),
        }
    }

    fn accepts_dispatch(&self) -> bool {
        !self.shutdown_requested.load(Ordering::Acquire)
            && self.worker_terminal.load(Ordering::Acquire) == OBSERVATION_COMPLETION_WORKER_RUNNING
    }

    fn request_shutdown(&self) {
        self.shutdown_requested.store(true, Ordering::Release);
    }

    fn mark_stopped(&self) {
        self.worker_terminal.store(OBSERVATION_COMPLETION_WORKER_STOPPED, Ordering::Release);
    }

    fn record_fault(&self) {
        self.worker_fault_count.fetch_add(1, Ordering::Relaxed);
    }

    #[cfg(test)]
    fn worker_fault_count(&self) -> u64 {
        self.worker_fault_count.load(Ordering::Acquire)
    }
}

/// Retains the completion-worker join handles until their final owner either
/// drains naturally or the process-exit registry stops them. Completion work
/// is host-side, but activity-tail destructors can reenter backend lifecycle
/// code, so their workers must not outlive CUDA's process finalizer either.
#[derive(Debug)]
struct ObservationCompletionSupervisor {
    observation_state: Arc<ObservationCompletionExecutorState>,
    activity_tail_state: Arc<ObservationCompletionExecutorState>,
    join: Mutex<[Option<thread::JoinHandle<()>>; 2]>,
    active_workers: AtomicUsize,
    process_exit_registration: Mutex<Option<ProcessExitRegistration>>,
}

impl ObservationCompletionSupervisor {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            observation_state: Arc::new(ObservationCompletionExecutorState::new()),
            activity_tail_state: Arc::new(ObservationCompletionExecutorState::new()),
            join: Mutex::new([None, None]),
            active_workers: AtomicUsize::new(0),
            process_exit_registration: Mutex::new(None),
        })
    }

    fn state(&self, lane: ObservationCompletionWorkerLane) -> Arc<ObservationCompletionExecutorState> {
        match lane {
            ObservationCompletionWorkerLane::Observation => Arc::clone(&self.observation_state),
            ObservationCompletionWorkerLane::ActivityTail => Arc::clone(&self.activity_tail_state),
        }
    }

    fn spawn_worker(
        self: &Arc<Self>,
        lane: ObservationCompletionWorkerLane,
        state: Arc<ObservationCompletionExecutorState>,
    ) -> Result<mpsc::Sender<ObservationCompletionMessage>> {
        let (sender, receiver) = mpsc::channel::<ObservationCompletionMessage>();
        self.active_workers.fetch_add(1, Ordering::AcqRel);
        let worker_supervisor = Arc::clone(self);
        let failed_supervisor = Arc::clone(self);
        let name = lane.thread_name();
        let worker = thread::Builder::new()
            .name(name.to_string())
            .spawn(move || {
                run_observation_completion_worker(receiver, state, name);
                worker_supervisor.worker_stopped();
            })
            .map_err(|error| {
                failed_supervisor.active_workers.fetch_sub(1, Ordering::AcqRel);
                QueryError::new(
                    QueryErrorCode::Internal,
                    BUILD_OPERATION,
                    "failed to create backend observation completion worker",
                )
                .with_fact("reason", "observation_completion_worker_spawn_failed")
                .with_fact("worker_lane", lane.as_str())
                .with_fact("io_error", error.to_string())
            })?;
        self.join.lock().unwrap_or_else(std::sync::PoisonError::into_inner)[lane.slot()] = Some(worker);
        Ok(sender)
    }

    fn activate_process_exit(self: &Arc<Self>) {
        self.activate_process_exit_with(register_cuda_process_exit_shutdown);
    }

    /// Registers outside the slot mutex. Process-exit registration may invoke
    /// the supplied shutdown owner synchronously when global exit has already
    /// begun; that owner joins workers and re-enters `unregister_process_exit`.
    /// Keeping the mutex out of that callback path prevents a self-deadlock.
    fn activate_process_exit_with(
        self: &Arc<Self>,
        register: impl FnOnce(Arc<dyn ProcessExitShutdown>) -> Option<ProcessExitRegistration>,
    ) {
        {
            let registration = self.process_exit_registration.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if registration.is_some() || self.active_workers.load(Ordering::Acquire) == 0 {
                return;
            }
        }

        let shutdown: Arc<dyn ProcessExitShutdown> = self.clone();
        let registration = register(shutdown);
        let registration_to_unregister = {
            let mut slot = self.process_exit_registration.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if self.active_workers.load(Ordering::Acquire) == 0 || slot.is_some() {
                registration
            } else {
                *slot = registration;
                None
            }
        };
        if let Some(registration) = registration_to_unregister {
            registration.unregister();
        }
    }

    fn unregister_process_exit(&self) {
        let registration =
            self.process_exit_registration.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take();
        if let Some(registration) = registration {
            registration.unregister();
        }
    }

    fn worker_stopped(&self) {
        if self.active_workers.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.unregister_process_exit();
        }
    }

    fn request_shutdown(&self) {
        self.observation_state.request_shutdown();
        self.activity_tail_state.request_shutdown();
    }

    fn join_for_process_exit(&self) {
        self.request_shutdown();
        let mut join = self.join.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        for slot in join.iter_mut() {
            let Some(handle) = slot.take() else {
                continue;
            };
            if handle.thread().id() == thread::current().id() {
                *slot = Some(handle);
                continue;
            }
            let _ = handle.join();
        }
        drop(join);
        self.unregister_process_exit();
    }
}

impl ProcessExitShutdown for ObservationCompletionSupervisor {
    fn phase(&self) -> ProcessExitPhase {
        ProcessExitPhase::ObservationCompletion
    }

    fn shutdown_for_process_exit(&self) {
        self.join_for_process_exit();
    }
}

/// Cloneable post-GPU cleanup dispatch capability.
///
/// The observation lane seals and delivers snapshots before releasing an
/// observation permit. The activity-tail lane drops opaque query activity only
/// after that fence is gone. Each lane has a pre-created worker and neither
/// holds a [`ResourceCore`] or GPU resource.
#[derive(Clone)]
pub(crate) struct ObservationCompletionDispatcher {
    observation_sender: mpsc::Sender<ObservationCompletionMessage>,
    observation_state: Arc<ObservationCompletionExecutorState>,
    activity_tail_sender: mpsc::Sender<ObservationCompletionMessage>,
    activity_tail_state: Arc<ObservationCompletionExecutorState>,
}

impl fmt::Debug for ObservationCompletionDispatcher {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("ObservationCompletionDispatcher").finish_non_exhaustive()
    }
}

impl ObservationCompletionDispatcher {
    /// Runs one finalization payload after its caller has released GPU state.
    ///
    /// A stopped or disconnected worker cannot consume the payload, so the
    /// dispatcher synchronously executes it. A recoverable worker fault does
    /// not take this path: the pre-created worker records it and continues.
    pub(crate) fn dispatch<F>(&self, completion: F)
    where F: FnOnce() + Send + 'static {
        self.dispatch_to(&self.observation_sender, &self.observation_state, completion);
    }

    /// Runs opaque activity destruction after the observation lane released
    /// its permit. This distinct lane prevents a blocking activity callback
    /// from starving later observation completion required by backend close.
    pub(crate) fn dispatch_activity_tail<F>(&self, completion: F)
    where F: FnOnce() + Send + 'static {
        self.dispatch_to(&self.activity_tail_sender, &self.activity_tail_state, completion);
    }

    fn dispatch_to<F>(
        &self,
        sender: &mpsc::Sender<ObservationCompletionMessage>,
        state: &ObservationCompletionExecutorState,
        completion: F,
    ) where
        F: FnOnce() + Send + 'static,
    {
        let completion = ObservationCompletionTask::new(completion);
        if !state.accepts_dispatch() {
            completion.run();
            return;
        }
        if let Err(mpsc::SendError(ObservationCompletionMessage::Completion(completion))) =
            sender.send(ObservationCompletionMessage::Completion(completion))
        {
            completion.run();
        }
    }

    #[cfg(test)]
    fn disconnected_for_test() -> Self {
        let (observation_sender, observation_receiver) = mpsc::channel();
        let (activity_tail_sender, activity_tail_receiver) = mpsc::channel();
        drop(observation_receiver);
        drop(activity_tail_receiver);
        Self {
            observation_sender,
            observation_state: Arc::new(ObservationCompletionExecutorState::new()),
            activity_tail_sender,
            activity_tail_state: Arc::new(ObservationCompletionExecutorState::new()),
        }
    }
}

/// Pre-created, backend-owned worker for detached observation completion.
///
/// The executor owns no backend resources. Its dispatcher may be retained by a
/// query after the resource generation is gone, so an enqueue failure must
/// execute the payload inline rather than drop its permit. Both receivers drain
/// accepted payloads after the executor owner drops; dispatcher clones retain
/// senders until every future completion has either been enqueued or
/// synchronously finalized.
#[derive(Debug)]
struct ObservationCompletionExecutor {
    dispatcher: ObservationCompletionDispatcher,
    supervisor: Arc<ObservationCompletionSupervisor>,
}

impl ObservationCompletionExecutor {
    fn spawn() -> Result<Self> {
        Self::spawn_with_lane_spawner(|lane, state, supervisor| supervisor.spawn_worker(lane, state))
    }

    fn spawn_with_lane_spawner(
        spawn_lane: impl Fn(
            ObservationCompletionWorkerLane,
            Arc<ObservationCompletionExecutorState>,
            Arc<ObservationCompletionSupervisor>,
        ) -> Result<mpsc::Sender<ObservationCompletionMessage>>,
    ) -> Result<Self> {
        let supervisor = ObservationCompletionSupervisor::new();
        let observation_state = supervisor.state(ObservationCompletionWorkerLane::Observation);
        let observation_sender = spawn_lane(
            ObservationCompletionWorkerLane::Observation,
            Arc::clone(&observation_state),
            Arc::clone(&supervisor),
        )?;
        let activity_tail_state = supervisor.state(ObservationCompletionWorkerLane::ActivityTail);
        let activity_tail_sender = match spawn_lane(
            ObservationCompletionWorkerLane::ActivityTail,
            Arc::clone(&activity_tail_state),
            Arc::clone(&supervisor),
        ) {
            Ok(sender) => sender,
            Err(error) => {
                // Explicitly disconnect the already-started observation
                // lane before returning an unpublished construction error.
                drop(observation_sender);
                supervisor.join_for_process_exit();
                return Err(error);
            },
        };
        Ok(Self {
            dispatcher: ObservationCompletionDispatcher {
                observation_sender,
                observation_state,
                activity_tail_sender,
                activity_tail_state,
            },
            supervisor,
        })
    }

    /// Registers both lanes only after backend service construction has
    /// entered CUDA. This keeps the engine's `atexit` callback ahead of
    /// libcudart while preserving the executor's detached delivery contract.
    fn activate_process_exit(&self) {
        self.supervisor.activate_process_exit();
    }

    #[cfg(test)]
    fn spawn_with_lane_failure_for_test(failing_lane: ObservationCompletionWorkerLane) -> Result<Self> {
        Self::spawn_with_lane_spawner(|lane, state, supervisor| {
            if lane == failing_lane {
                return Err(QueryError::new(
                    QueryErrorCode::Internal,
                    BUILD_OPERATION,
                    "injected backend observation completion worker spawn failure",
                )
                .with_fact("reason", "observation_completion_worker_spawn_failed")
                .with_fact("worker_lane", lane.as_str()));
            }
            supervisor.spawn_worker(lane, state)
        })
    }

    fn dispatcher(&self) -> ObservationCompletionDispatcher {
        self.dispatcher.clone()
    }

    #[cfg(test)]
    fn inject_observation_worker_fault_for_test(&self) {
        self.dispatcher
            .observation_sender
            .send(ObservationCompletionMessage::InjectWorkerFaultForTest)
            .expect("test completion worker is connected");
    }

    #[cfg(test)]
    fn observation_worker_fault_count(&self) -> u64 {
        self.dispatcher.observation_state.worker_fault_count()
    }
}

fn run_observation_completion_worker(
    receiver: mpsc::Receiver<ObservationCompletionMessage>,
    state: Arc<ObservationCompletionExecutorState>,
    name: &'static str,
) {
    loop {
        if state.shutdown_requested.load(Ordering::Acquire) {
            state.mark_stopped();
            return;
        }
        let received =
            catch_unwind(AssertUnwindSafe(|| match receiver.recv_timeout(OBSERVATION_COMPLETION_POLL_INTERVAL) {
                Ok(message) => {
                    match message {
                        ObservationCompletionMessage::Completion(completion) => completion.run(),
                        #[cfg(test)]
                        ObservationCompletionMessage::InjectWorkerFaultForTest => {
                            panic!("injected observation completion worker fault");
                        },
                    }
                    true
                },
                Err(mpsc::RecvTimeoutError::Timeout) => true,
                Err(mpsc::RecvTimeoutError::Disconnected) => false,
            }));
        match received {
            Ok(true) => {},
            Ok(false) => {
                state.mark_stopped();
                return;
            },
            Err(_) => {
                state.record_fault();
                eprintln!(
                    "[nexus-query-engine] warning: backend observation completion worker {name} recovered from a fault"
                );
            },
        }
    }
}

fn execute_observation_completion(completion: ObservationCompletionCallback) {
    if catch_unwind(AssertUnwindSafe(completion)).is_err() {
        eprintln!("[nexus-query-engine] warning: backend observation completion callback panicked");
    }
}

#[derive(Debug)]
struct ResourceGeneration {
    service: OnceLock<QueryService>,
    startup_records: OnceLock<Arc<[GpuBackendDeviceStartupRecord]>>,
    generation_id: u64,
    construction_journal: Arc<ConstructionResourceJournal>,
    // Declared last and explicitly dropped last by the cleanup worker.
    owner_lease: LiveBackendLease,
}

impl ResourceGeneration {
    fn new(
        generation_id: u64,
        owner_lease: LiveBackendLease,
        construction_journal: Arc<ConstructionResourceJournal>,
    ) -> Self {
        Self {
            service: OnceLock::new(),
            startup_records: OnceLock::new(),
            generation_id,
            construction_journal,
            owner_lease,
        }
    }

    fn service(&self) -> Option<&QueryService> {
        self.service.get()
    }
}

#[derive(Debug)]
enum ConstructionResource {
    MemoryManager(cudf::DeviceMemoryManager),
    StreamPool(super::ServiceStreamPool),
    DeviceContext(super::DeviceContext),
    ExecutionPool(super::GpuExecutionPool),
    StartupRecords(Arc<[GpuBackendDeviceStartupRecord]>),
}

#[derive(Debug, Default)]
pub(crate) struct ConstructionResourceJournal {
    entries: Mutex<Vec<ConstructionResource>>,
}

impl ConstructionResourceJournal {
    pub(crate) fn record_memory_manager(&self, manager: cudf::DeviceMemoryManager) -> Result<()> {
        self.record(ConstructionResource::MemoryManager(manager))
    }

    pub(crate) fn record_stream_pool(&self, pool: super::ServiceStreamPool) -> Result<()> {
        self.record(ConstructionResource::StreamPool(pool))
    }

    pub(crate) fn record_device_context(&self, context: super::DeviceContext) -> Result<()> {
        self.record(ConstructionResource::DeviceContext(context))
    }

    pub(crate) fn record_execution_pool(&self, pool: super::GpuExecutionPool) -> Result<()> {
        self.record(ConstructionResource::ExecutionPool(pool))
    }

    fn record_startup_records(&self, records: Arc<[GpuBackendDeviceStartupRecord]>) -> Result<()> {
        self.record(ConstructionResource::StartupRecords(records))
    }

    fn record(&self, resource: ConstructionResource) -> Result<()> {
        self.entries.lock().map_err(|_| construction_journal_poisoned())?.push(resource);
        Ok(())
    }

    fn close_and_drain_until(&self, deadline: Instant) -> Result<()> {
        let entries = self.entries.lock().map_err(|_| construction_journal_poisoned())?;
        let mut first_deadline = None;
        let mut first_permanent = None;
        for entry in entries.iter().rev() {
            let result = match entry {
                ConstructionResource::ExecutionPool(pool) => pool.close_and_wait_until(deadline),
                ConstructionResource::StreamPool(pool) => {
                    pool.close_and_wait_until(deadline, "nexus_query_engine.gpu_backend_core.rollback")
                },
                ConstructionResource::MemoryManager(manager) => {
                    let _ = manager.device();
                    Ok(())
                },
                ConstructionResource::DeviceContext(context) => {
                    let _ = context.device();
                    Ok(())
                },
                ConstructionResource::StartupRecords(records) => {
                    let _ = records.len();
                    Ok(())
                },
            };
            if let Err(error) = result {
                if error.is_cleanup_deadline_elapsed() {
                    if first_deadline.is_none() {
                        first_deadline = Some(error);
                    }
                } else if first_permanent.is_none() {
                    first_permanent = Some(error);
                }
            }
        }
        if let Some(error) = first_permanent.or(first_deadline) {
            return Err(error);
        }
        Ok(())
    }
}

fn construction_journal_poisoned() -> QueryError {
    QueryError::new(QueryErrorCode::Internal, BUILD_OPERATION, "construction resource journal mutex was poisoned")
        .with_fact("reason", "construction_journal_poisoned")
}

/// Signals the pre-created cleanup coordinator unless construction reaches
/// publication. This is intentionally armed before the first GPU mutation.
struct ConstructionRollback {
    coordinator: Option<Arc<CloseCoordinator>>,
}

impl ConstructionRollback {
    fn new(coordinator: Arc<CloseCoordinator>) -> Self {
        Self { coordinator: Some(coordinator) }
    }

    fn disarm(&mut self) {
        self.coordinator.take();
    }

    fn attach_cleanup_facts(&mut self, error: QueryError) -> QueryError {
        let Some(coordinator) = self.coordinator.take() else {
            return error.with_fact("rollback_cleanup", "not_armed");
        };
        let deadline = Instant::now().checked_add(CLEANUP_SLICE).unwrap_or_else(Instant::now);
        match coordinator.close_and_wait_until(deadline) {
            Ok(snapshot) => error
                .with_fact("rollback_cleanup", "quiescent")
                .with_fact("rollback_gpu_resource_quiescent", snapshot.gpu_resource_quiescent)
                .with_fact("rollback_observation_quiescent", snapshot.observation_quiescent),
            Err(cleanup_error) => error
                .with_fact("rollback_cleanup", "transferred")
                .with_fact("rollback_cleanup_code", cleanup_error.code().as_str())
                .with_fact("rollback_cleanup_reason", cleanup_error.fact_value("reason").unwrap_or("unknown")),
        }
    }
}

impl Drop for ConstructionRollback {
    fn drop(&mut self) {
        if let Some(coordinator) = self.coordinator.take() {
            coordinator.signal();
        }
    }
}

#[derive(Debug)]
struct CloseCoordinator {
    sender: Mutex<Option<mpsc::Sender<()>>>,
    state: Arc<CloseCompletion>,
    lifecycle: BackendLifecycle,
    join: Mutex<Option<thread::JoinHandle<()>>>,
    process_exit_requested: Arc<AtomicBool>,
    process_exit_registration: Mutex<Option<ProcessExitRegistration>>,
}

impl CloseCoordinator {
    fn spawn(resource: Arc<ResourceCore>, observation: Arc<ObservationCore>) -> Result<Arc<Self>> {
        let (sender, receiver) = mpsc::channel();
        let state = Arc::new(CloseCompletion::new());
        let process_exit_requested = Arc::new(AtomicBool::new(false));
        let coordinator = Arc::new(Self {
            sender: Mutex::new(Some(sender)),
            state: Arc::clone(&state),
            lifecycle: observation.lifecycle.clone(),
            join: Mutex::new(None),
            process_exit_requested: Arc::clone(&process_exit_requested),
            process_exit_registration: Mutex::new(None),
        });
        let worker_state = Arc::clone(&state);
        let worker_lifecycle = observation.lifecycle.clone();
        let resource_for_panic = Arc::clone(&resource);
        let worker_coordinator = Arc::clone(&coordinator);
        let join = thread::Builder::new()
            .name("nexus-gpu-close".to_string())
            .spawn(move || {
                let result = catch_unwind(AssertUnwindSafe(|| {
                    let _ = receiver.recv();
                    run_cleanup(resource, worker_lifecycle.clone(), Arc::clone(&worker_state), &process_exit_requested);
                }));
                if result.is_err() {
                    worker_state.fail("cleanup_worker", "cleanup worker panicked");
                    // A panic must not drop a still-installed resource
                    // generation and release its owner lease.
                    std::mem::forget(resource_for_panic);
                }
                worker_coordinator.unregister_process_exit();
            })
            .map_err(|error| {
                QueryError::new(QueryErrorCode::Internal, BUILD_OPERATION, "failed to create backend cleanup worker")
                    .with_fact("reason", "cleanup_worker_spawn_failed")
                    .with_fact("io_error", error.to_string())
            })?;
        *coordinator.join.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(join);
        Ok(coordinator)
    }

    fn activate_process_exit(self: &Arc<Self>) {
        let shutdown: Arc<dyn ProcessExitShutdown> = self.clone();
        let registration = register_cuda_process_exit_shutdown(shutdown);
        *self.process_exit_registration.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = registration;
    }

    fn unregister_process_exit(&self) {
        let registration =
            self.process_exit_registration.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take();
        if let Some(registration) = registration {
            registration.unregister();
        }
    }

    fn signal(&self) -> bool {
        let gate_winner = self.lifecycle.close();
        if !gate_winner {
            return false;
        }
        self.state.mark_closing();
        let sender = self.sender.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take();
        if sender.is_none_or(|sender| sender.send(()).is_err()) {
            self.state.fail("cleanup_signal", "cleanup worker disconnected before close signal");
        }
        true
    }

    fn close_and_wait_until(&self, deadline: Instant) -> Result<GpuBackendCloseSnapshot> {
        self.signal();
        let snapshot = self.state.wait_until(deadline, &self.lifecycle)?;
        self.join_after_quiescence()?;
        Ok(snapshot)
    }

    fn join_after_quiescence(&self) -> Result<()> {
        let mut join = self.join.lock().map_err(|_| {
            QueryError::new(QueryErrorCode::Internal, CLOSE_OPERATION, "backend cleanup worker join mutex was poisoned")
                .with_fact("reason", "cleanup_worker_join_poisoned")
        })?;
        let Some(handle) = join.take() else {
            return Ok(());
        };
        if handle.thread().id() == thread::current().id() {
            // A reentrant close invoked by the cleanup worker itself cannot
            // join its own entry point. The worker will unregister only after
            // it has completed all GPU and observation teardown.
            *join = Some(handle);
            return Ok(());
        }
        let joined = handle.join();
        drop(join);
        self.unregister_process_exit();
        joined.map_err(|panic| {
            QueryError::new(QueryErrorCode::Internal, CLOSE_OPERATION, "backend cleanup worker panicked during join")
                .with_fact("reason", "cleanup_worker_join_panicked")
                .with_fact("panic_message", panic_message(panic.as_ref()))
        })
    }

    fn join_for_process_exit(&self) {
        // `std::process::exit` does not unwind the caller's stack. An admitted
        // handle owned by that stack can therefore never participate in the
        // ordinary quiescence protocol. Tell the cleanup worker to stop at its
        // next bounded phase boundary and leak the resource generation; the
        // separately registered execution/stream supervisors then terminate
        // their workers without allowing CUDA destructors to run after this
        // process-exit hook returns.
        self.process_exit_requested.store(true, Ordering::Release);
        self.signal();
        let mut join = self.join.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(handle) = join.take() else {
            return;
        };
        if handle.thread().id() == thread::current().id() {
            *join = Some(handle);
            return;
        }
        let _ = handle.join();
        drop(join);
        self.unregister_process_exit();
    }
}

impl ProcessExitShutdown for CloseCoordinator {
    fn phase(&self) -> ProcessExitPhase {
        ProcessExitPhase::BackendCoordinator
    }

    fn shutdown_for_process_exit(&self) {
        self.join_for_process_exit();
    }
}

fn run_cleanup(
    resource: Arc<ResourceCore>,
    lifecycle: BackendLifecycle,
    state: Arc<CloseCompletion>,
    process_exit_requested: &AtomicBool,
) {
    lifecycle.close();
    let generation = match resource.generation.lock() {
        Ok(mut generation) => generation.take(),
        Err(_error) => {
            state.fail("resource_generation", "resource-core generation mutex was poisoned");
            std::mem::forget(Arc::clone(&resource));
            return;
        },
    };
    let Some(generation) = generation else {
        state.mark_gpu_quiescent();
        wait_for_observation_forever(&lifecycle, &state, process_exit_requested);
        return;
    };
    let generation = FailClosedGeneration::new(generation);
    if let Some(service) = generation.service() {
        service.close().unwrap_or_else(|error| {
            eprintln!("[nexus-query-engine] backend close signal returned an error: {error}");
        });
        if process_exit_requested.load(Ordering::Acquire) {
            return;
        }
        loop {
            let deadline = Instant::now().checked_add(CLEANUP_SLICE).unwrap_or_else(Instant::now);
            let outcome: QueryServiceDrainOutcome = service.close_drain_step_until(deadline);
            let mut all_quiescent = true;
            let mut permanent_failure = None;
            for phase_outcome in outcome.phases() {
                match phase_outcome {
                    QueryServiceDrainPhaseOutcome::Quiescent { phase } => {
                        state.record_quiescent(phase.as_str());
                    },
                    QueryServiceDrainPhaseOutcome::DeadlineElapsed { phase, .. } => {
                        all_quiescent = false;
                        state.record_deadline(phase.as_str());
                    },
                    QueryServiceDrainPhaseOutcome::PermanentFailure { phase, error } => {
                        all_quiescent = false;
                        if permanent_failure.is_none() {
                            permanent_failure = Some((phase.as_str(), error));
                        }
                    },
                }
            }
            if let Some((phase, error)) = permanent_failure {
                state.fail(phase, error);
                return;
            }
            if all_quiescent {
                break;
            }
            if process_exit_requested.load(Ordering::Acquire) {
                return;
            }
        }
    }
    // The service first closes admission and drains work using the resources
    // retained by the permanent journal. Only then may the journal close and
    // verify every runtime/resource handle. Construction rollback has no
    // published service and therefore reaches this same journal path directly.
    loop {
        let deadline = Instant::now().checked_add(CLEANUP_SLICE).unwrap_or_else(Instant::now);
        match generation.construction_journal().close_and_drain_until(deadline) {
            Ok(()) => {
                state.record_quiescent("construction_journal");
                break;
            },
            Err(error) if error.is_cleanup_deadline_elapsed() => {
                state.record_deadline("construction_journal");
            },
            Err(error) => {
                state.fail("construction_journal", error);
                return;
            },
        }
        if process_exit_requested.load(Ordering::Acquire) {
            return;
        }
    }
    loop {
        let deadline = Instant::now().checked_add(CLEANUP_SLICE).unwrap_or_else(Instant::now);
        match lifecycle.wait_for_gpu_quiescence_until(deadline) {
            Ok(()) => {
                state.record_quiescent("gpu_permit_quiescence");
                break;
            },
            Err(error) if error.is_cleanup_deadline_elapsed() => {
                state.record_deadline("gpu_permit_quiescence");
            },
            Err(error) => {
                state.fail("gpu_permit_quiescence", error);
                return;
            },
        }
        if process_exit_requested.load(Ordering::Acquire) {
            return;
        }
    }
    generation.release();
    drop(resource);
    state.mark_gpu_quiescent();
    wait_for_observation_forever(&lifecycle, &state, process_exit_requested);
}

fn wait_for_observation_forever(
    lifecycle: &BackendLifecycle,
    state: &CloseCompletion,
    process_exit_requested: &AtomicBool,
) {
    loop {
        let deadline = Instant::now().checked_add(CLEANUP_SLICE).unwrap_or_else(Instant::now);
        match lifecycle.wait_for_observation_quiescence_until(deadline) {
            Ok(()) => {
                state.record_quiescent("observation_quiescence");
                state.mark_observation_quiescent();
                return;
            },
            Err(error) if error.is_cleanup_deadline_elapsed() => {
                state.record_deadline("observation_quiescence");
            },
            Err(error) => {
                state.fail("observation_quiescence", error);
                return;
            },
        }
        if process_exit_requested.load(Ordering::Acquire) {
            return;
        }
    }
}

struct FailClosedGeneration {
    generation: Option<Arc<ResourceGeneration>>,
}

impl FailClosedGeneration {
    fn new(generation: Arc<ResourceGeneration>) -> Self {
        Self { generation: Some(generation) }
    }

    fn service(&self) -> Option<&QueryService> {
        self.generation.as_ref().expect("generation retained until release").service()
    }

    fn construction_journal(&self) -> &ConstructionResourceJournal {
        &self.generation.as_ref().expect("generation retained until release").construction_journal
    }

    fn release(mut self) {
        drop(self.generation.take());
    }
}

impl Drop for FailClosedGeneration {
    fn drop(&mut self) {
        if let Some(generation) = self.generation.take() {
            std::mem::forget(generation);
        }
    }
}

const PHASE_OPEN: u8 = 0;
const PHASE_CLOSING: u8 = 1;
const PHASE_GPU_QUIESCENT: u8 = 2;
const PHASE_OBSERVATION_QUIESCENT: u8 = 3;
const PHASE_FAILED: u8 = 4;

#[derive(Debug)]
struct CloseCompletion {
    phase: AtomicU8,
    error: Mutex<Option<CloseFailure>>,
    deadline_elapsed: AtomicU64,
    late_successes: AtomicU64,
    cleanup_phases: Mutex<BTreeMap<&'static str, CleanupPhaseProgress>>,
    wait: Mutex<()>,
    changed: Condvar,
}

#[derive(Debug, Clone)]
struct CloseFailure {
    phase: &'static str,
    error: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CleanupPhaseProgress {
    DeadlineElapsed,
    Quiescent,
}

impl CloseCompletion {
    fn new() -> Self {
        Self {
            phase: AtomicU8::new(PHASE_OPEN),
            error: Mutex::new(None),
            deadline_elapsed: AtomicU64::new(0),
            late_successes: AtomicU64::new(0),
            cleanup_phases: Mutex::new(BTreeMap::new()),
            wait: Mutex::new(()),
            changed: Condvar::new(),
        }
    }

    fn mark_closing(&self) {
        self.advance(PHASE_CLOSING);
    }

    fn mark_gpu_quiescent(&self) {
        self.advance(PHASE_GPU_QUIESCENT);
    }

    fn mark_observation_quiescent(&self) {
        self.advance(PHASE_OBSERVATION_QUIESCENT);
    }

    fn advance(&self, phase: u8) {
        let mut current = self.phase.load(Ordering::Acquire);
        while current < phase && current != PHASE_FAILED {
            match self.phase.compare_exchange_weak(current, phase, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => {
                    self.changed.notify_all();
                    return;
                },
                Err(observed) => current = observed,
            }
        }
    }

    fn record_deadline(&self, phase: &'static str) {
        let mut phases = self.cleanup_phases.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if !phases.contains_key(phase) {
            phases.insert(phase, CleanupPhaseProgress::DeadlineElapsed);
            self.deadline_elapsed.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn record_quiescent(&self, phase: &'static str) {
        let mut phases = self.cleanup_phases.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if phases.insert(phase, CleanupPhaseProgress::Quiescent) == Some(CleanupPhaseProgress::DeadlineElapsed) {
            self.late_successes.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn fail(&self, phase: &'static str, error: impl fmt::Display) {
        let mut failure = self.error.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if failure.is_none() {
            *failure = Some(CloseFailure { phase, error: error.to_string() });
        }
        drop(failure);
        // A later observation-cleanup failure must not erase the already
        // established GPU-resource-quiescent fact. Before GPU quiescence the
        // failed sentinel also prevents any later success transition.
        if self.phase.load(Ordering::Acquire) < PHASE_GPU_QUIESCENT {
            self.phase.store(PHASE_FAILED, Ordering::Release);
        }
        self.changed.notify_all();
    }

    fn snapshot(&self, lifecycle: &BackendLifecycle) -> GpuBackendCloseSnapshot {
        let phase = self.phase.load(Ordering::Acquire);
        let lifecycle = lifecycle.snapshot();
        let cleanup_failed = self.error.lock().unwrap_or_else(std::sync::PoisonError::into_inner).is_some();
        GpuBackendCloseSnapshot {
            closing: lifecycle.closing,
            gpu_resource_quiescent: phase == PHASE_GPU_QUIESCENT || phase == PHASE_OBSERVATION_QUIESCENT,
            observation_quiescent: phase == PHASE_OBSERVATION_QUIESCENT,
            active_gpu_permits: lifecycle.active_gpu_permits,
            active_observation_permits: lifecycle.active_observation_permits,
            cleanup_failed,
            cleanup_deadline_elapsed_count: self.deadline_elapsed.load(Ordering::Relaxed),
            cleanup_late_success_count: self.late_successes.load(Ordering::Relaxed),
        }
    }

    fn wait_until(&self, deadline: Instant, lifecycle: &BackendLifecycle) -> Result<GpuBackendCloseSnapshot> {
        let mut wait = super::deadline::mutex_lock_until(&self.wait, deadline).map_err(|error| match error {
            super::deadline::DeadlineMutexLockError::Poisoned => close_wait_poisoned_error(),
            super::deadline::DeadlineMutexLockError::TimedOut => close_timeout_error(self.snapshot(lifecycle)),
        })?;
        loop {
            let snapshot = self.snapshot(lifecycle);
            if snapshot.cleanup_failed {
                let cleanup_failure =
                    self.error.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone().unwrap_or_else(|| {
                        CloseFailure { phase: "unknown", error: "unknown cleanup failure".to_string() }
                    });
                return Err(QueryError::new(
                    QueryErrorCode::Internal,
                    CLOSE_OPERATION,
                    "backend cleanup failed closed",
                )
                .with_fact("reason", "cleanup_failed_closed")
                .with_fact("cleanup_phase", cleanup_failure.phase)
                .with_fact("cleanup_error", cleanup_failure.error));
            }
            if snapshot.observation_quiescent {
                return Ok(snapshot);
            }
            if super::deadline::deadline_expired(deadline) {
                return Err(close_timeout_error(snapshot));
            }
            let Some(remaining) = super::deadline::deadline_remaining(deadline) else {
                return Err(close_timeout_error(snapshot));
            };
            let (next, timeout) =
                self.changed.wait_timeout(wait, remaining).map_err(|_error| close_wait_poisoned_error())?;
            wait = next;
            if timeout.timed_out() {
                // A notify that lands between the deadline expiring and the
                // waiter rescheduling must not be reported as a timeout: take
                // one final snapshot before failing.
                let snapshot = self.snapshot(lifecycle);
                if snapshot.observation_quiescent {
                    return Ok(snapshot);
                }
                return Err(close_timeout_error(snapshot));
            }
        }
    }
}

/// Two-scope close state. GPU quiescence can become true while observation
/// permits still exist; at that point the device-owner lease has been released.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuBackendCloseSnapshot {
    pub closing: bool,
    pub gpu_resource_quiescent: bool,
    pub observation_quiescent: bool,
    pub active_gpu_permits: usize,
    pub active_observation_permits: usize,
    pub cleanup_failed: bool,
    pub cleanup_deadline_elapsed_count: u64,
    pub cleanup_late_success_count: u64,
}

fn close_timeout_error(snapshot: GpuBackendCloseSnapshot) -> QueryError {
    QueryError::new(QueryErrorCode::ServiceOverloaded, CLOSE_OPERATION, "timed out waiting for backend close")
        .with_fact("reason", "backend_close_timeout")
        .with_fact("gpu_resource_quiescent", snapshot.gpu_resource_quiescent)
        .with_fact("observation_quiescent", snapshot.observation_quiescent)
        .with_fact("active_gpu_permits", snapshot.active_gpu_permits)
        .with_fact("active_observation_permits", snapshot.active_observation_permits)
        .with_fact("cleanup_deadline_elapsed_count", snapshot.cleanup_deadline_elapsed_count)
        .with_fact("cleanup_late_success_count", snapshot.cleanup_late_success_count)
}

fn close_wait_poisoned_error() -> QueryError {
    QueryError::new(QueryErrorCode::Internal, CLOSE_OPERATION, "backend close completion mutex was poisoned")
        .with_fact("reason", "close_wait_poisoned")
}

fn panic_message(panic: &(dyn Any + Send)) -> String {
    panic
        .downcast_ref::<&'static str>()
        .map(|message| (*message).to_string())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non-string panic payload".to_string())
}

fn reject_reentrant_observation_close() -> Result<()> {
    if in_runtime_observation_delivery() {
        return Err(QueryError::new(
            QueryErrorCode::Internal,
            CLOSE_OPERATION,
            "backend close-and-wait cannot run from runtime observation delivery",
        )
        .with_fact("reason", "reentrant_observation_close"));
    }
    Ok(())
}

fn backend_closing_resource_error() -> QueryError {
    QueryError::new(
        QueryErrorCode::Cancelled,
        RESOURCE_OPERATION,
        "backend is closing and has no live resource generation",
    )
    .with_fact("reason", "backend_closing")
}

fn resource_core_poisoned_error() -> QueryError {
    QueryError::new(QueryErrorCode::Internal, RESOURCE_OPERATION, "backend resource-core mutex was poisoned")
        .with_fact("reason", "resource_core_poisoned")
}

fn resource_generation_unpublished_error() -> QueryError {
    QueryError::new(QueryErrorCode::Internal, RESOURCE_OPERATION, "backend resource generation is not published")
        .with_fact("reason", "resource_generation_unpublished")
}

fn next_backend_identity() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::{super::query_service::QueryServiceDrainPhase, *};

    #[derive(Debug)]
    struct ActivityPermitOrderingProbe {
        lifecycle: BackendLifecycle,
        backend: GpuBackendCore,
        observed_active_gpu_permits: Arc<std::sync::atomic::AtomicUsize>,
        completion: mpsc::SyncSender<std::result::Result<(), String>>,
    }

    impl Drop for ActivityPermitOrderingProbe {
        fn drop(&mut self) {
            self.observed_active_gpu_permits.store(self.lifecycle.snapshot().active_gpu_permits, Ordering::Release);
            let result = (|| {
                let close = self.backend.close_and_wait(Duration::from_secs(10)).map_err(|error| error.to_string())?;
                if !close.gpu_resource_quiescent || !close.observation_quiescent {
                    return Err(format!("backend close was not quiescent: {close:?}"));
                }
                let replacement =
                    GpuBackendCore::builder(one_device_backend_draft(0)).build().map_err(|error| error.to_string())?;
                let replacement_close =
                    replacement.close_and_wait(Duration::from_secs(10)).map_err(|error| error.to_string())?;
                if !replacement_close.gpu_resource_quiescent || !replacement_close.observation_quiescent {
                    return Err(format!("replacement backend close was not quiescent: {replacement_close:?}"));
                }
                Ok(())
            })();
            let _ = self.completion.send(result);
        }
    }

    #[derive(Debug)]
    struct ReentrantCloseActivityProbe {
        backend: GpuBackendCore,
        entered: mpsc::SyncSender<()>,
        completion: mpsc::SyncSender<std::result::Result<GpuBackendCloseSnapshot, String>>,
    }

    impl Drop for ReentrantCloseActivityProbe {
        fn drop(&mut self) {
            let _ = self.entered.send(());
            let result = self.backend.close_and_wait(Duration::from_secs(10)).map_err(|error| error.to_string());
            let _ = self.completion.send(result);
        }
    }

    fn live_backend_test_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn inert_backend_for_observation_delivery_test() -> GpuBackendCore {
        let resource_core = Arc::new(ResourceCore { identity: next_backend_identity(), generation: Mutex::new(None) });
        let observation_core = Arc::new(ObservationCore {
            identity: next_backend_identity(),
            lifecycle: BackendLifecycle::new(),
            attempt_counters: Arc::new(AttemptBackendCounters::default()),
            attempt_completion_queue: None,
            completion_executor: ObservationCompletionExecutor::spawn().expect("test creates observation executor"),
        });
        let coordinator = CloseCoordinator::spawn(Arc::clone(&resource_core), Arc::clone(&observation_core))
            .expect("test creates close coordinator");
        let close_on_last = Arc::clone(&coordinator);
        let public_handles = PublicHandleTracker::new(move || {
            close_on_last.signal();
        });
        let public_handle = public_handles.acquire().expect("test creates public handle");
        GpuBackendCore {
            resource_core,
            observation_core,
            coordinator,
            public_handle,
            startup_records: Arc::from([]),
            generation_id: next_backend_identity(),
        }
    }

    #[test]
    fn close_and_wait_reclaims_the_backend_cleanup_worker() {
        let backend = inert_backend_for_observation_delivery_test();
        let coordinator = Arc::clone(&backend.coordinator);

        let close = backend.close_and_wait(Duration::from_secs(2)).expect("inert backend closes");

        assert!(close.gpu_resource_quiescent);
        assert!(close.observation_quiescent);
        assert!(coordinator.join.lock().unwrap().is_none(), "close must reclaim its cleanup worker handle");
    }

    #[test]
    fn observation_completion_executor_runs_detached_callbacks_and_isolates_panics() {
        let executor = ObservationCompletionExecutor::spawn().expect("test creates observation executor");
        let dispatcher = executor.dispatcher();
        let caller = thread::current().id();
        let (sender, receiver) = mpsc::sync_channel(2);

        dispatcher.dispatch({
            let sender = sender.clone();
            move || sender.send(thread::current().id()).expect("test receives completion worker id")
        });
        assert_ne!(
            receiver.recv_timeout(Duration::from_secs(2)).expect("completion worker ran callback"),
            caller,
            "pre-created completion worker must detach finalization from the caller"
        );

        dispatcher.dispatch(|| panic!("injected observation completion panic"));
        dispatcher.dispatch(move || sender.send(thread::current().id()).expect("worker survives callback panic"));
        assert_ne!(
            receiver.recv_timeout(Duration::from_secs(2)).expect("completion worker survived callback panic"),
            caller
        );
    }

    #[test]
    fn observation_and_activity_tail_callbacks_use_distinct_precreated_workers() {
        let executor = ObservationCompletionExecutor::spawn().expect("test creates observation executor");
        let dispatcher = executor.dispatcher();
        let (sender, receiver) = mpsc::sync_channel(2);

        dispatcher.dispatch({
            let sender = sender.clone();
            move || sender.send(thread::current().id()).expect("test observes observation worker")
        });
        dispatcher.dispatch_activity_tail(move || {
            sender.send(thread::current().id()).expect("test observes activity-tail worker")
        });

        let observation_worker = receiver.recv_timeout(Duration::from_secs(2)).expect("observation callback ran");
        let activity_tail_worker = receiver.recv_timeout(Duration::from_secs(2)).expect("activity-tail callback ran");
        assert_ne!(
            observation_worker, activity_tail_worker,
            "activity drops must not occupy observation delivery capacity"
        );
    }

    #[test]
    fn process_exit_joins_observation_completion_workers_and_closes_dispatch() {
        let executor = ObservationCompletionExecutor::spawn().expect("test creates observation executor");
        let dispatcher = executor.dispatcher();

        executor.supervisor.join_for_process_exit();

        assert_eq!(executor.supervisor.active_workers.load(Ordering::Acquire), 0);
        assert!(executor.supervisor.join.lock().unwrap().iter().all(Option::is_none));
        let caller = thread::current().id();
        let (sender, receiver) = mpsc::sync_channel(1);
        dispatcher.dispatch(move || sender.send(thread::current().id()).expect("fallback dispatch runs"));
        assert_eq!(receiver.recv_timeout(Duration::from_secs(2)).expect("fallback completion runs"), caller);
    }

    #[test]
    fn process_exit_with_live_admitted_handle_does_not_hang() {
        const CHILD_ENV: &str = "NEXUS_TEST_PROCESS_EXIT_WITH_LIVE_ADMISSION";
        const TEST_NAME: &str = "runtime::backend_core::tests::process_exit_with_live_admitted_handle_does_not_hang";

        if std::env::var_os(CHILD_ENV).is_some() {
            let backend =
                GpuBackendCore::builder(one_device_backend_draft(0)).build().expect("child process builds a backend");
            let session = backend.session_capability();
            let _live_handle = session.admit_query().expect("child process retains one admitted handle");
            // This intentionally skips every stack destructor. The registered
            // process-exit owners must still terminate without waiting for the
            // live handle above to release its admission slot.
            std::process::exit(0);
        }

        let executable = std::env::current_exe().expect("current test executable is available");
        let mut child = std::process::Command::new(executable)
            .arg(TEST_NAME)
            .arg("--exact")
            .arg("--nocapture")
            .env(CHILD_ENV, "1")
            .spawn()
            .expect("spawn process-exit regression child");
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            if let Some(status) = child.try_wait().expect("poll process-exit regression child") {
                assert!(status.success(), "process-exit regression child failed with {status}");
                break;
            }
            if Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                panic!("process exit hung while an admitted handle remained live");
            }
            thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn late_process_exit_registration_runs_callback_without_holding_registration_slot_mutex() {
        let supervisor = ObservationCompletionSupervisor::new();
        // The injected registration path models a global process-exit race:
        // registry registration synchronously invokes the owner, which in
        // turn calls `unregister_process_exit`.
        supervisor.active_workers.store(1, Ordering::Release);
        let callback_supervisor = Arc::clone(&supervisor);
        let callback_ran = Arc::new(AtomicBool::new(false));
        let callback_ran_slot = Arc::clone(&callback_ran);

        supervisor.activate_process_exit_with(move |shutdown| {
            assert!(
                callback_supervisor.process_exit_registration.try_lock().is_ok(),
                "registration slot mutex must be unlocked while registry registration calls back"
            );
            shutdown.shutdown_for_process_exit();
            callback_ran_slot.store(true, Ordering::Release);
            None
        });

        assert!(callback_ran.load(Ordering::Acquire));
        supervisor.active_workers.store(0, Ordering::Release);
    }

    #[test]
    fn observation_completion_dispatcher_synchronously_finalizes_when_enqueue_fails() {
        let dispatcher = ObservationCompletionDispatcher::disconnected_for_test();
        let caller = thread::current().id();
        let (sender, receiver) = mpsc::sync_channel(1);

        dispatcher.dispatch(move || sender.send(thread::current().id()).expect("fallback runs completion"));

        assert_eq!(receiver.recv_timeout(Duration::from_secs(2)).expect("fallback completed"), caller);
    }

    #[test]
    fn recoverable_observation_completion_worker_fault_keeps_later_delivery_off_caller() {
        let executor = ObservationCompletionExecutor::spawn().expect("test creates observation executor");
        let dispatcher = executor.dispatcher();
        let caller = thread::current().id();
        let (sender, receiver) = mpsc::sync_channel(2);

        dispatcher.dispatch({
            let sender = sender.clone();
            move || sender.send(thread::current().id()).expect("test observes executor worker thread")
        });
        let worker_thread =
            receiver.recv_timeout(Duration::from_secs(2)).expect("executor worker ran initial callback");

        executor.inject_observation_worker_fault_for_test();
        dispatcher.dispatch(move || sender.send(thread::current().id()).expect("recovered queued payload completed"));

        let delivery_thread =
            receiver.recv_timeout(Duration::from_secs(2)).expect("worker resumes queued completion after a fault");
        assert_eq!(delivery_thread, worker_thread, "recovered delivery must stay on the pre-created executor thread");
        assert_ne!(delivery_thread, caller, "recovered delivery must not run on the caller or stream supervisor");
        assert_eq!(executor.observation_worker_fault_count(), 1, "worker supervisor records the recovered fault");
    }

    #[test]
    fn observation_completion_worker_drains_accepted_payloads_after_executor_and_dispatchers_drop() {
        let executor = ObservationCompletionExecutor::spawn().expect("test creates observation executor");
        let dispatcher = executor.dispatcher();
        let gate = Arc::new(BlockingGate::default());
        let worker_gate = Arc::clone(&gate);
        let (entered_sender, entered_receiver) = mpsc::sync_channel(1);
        let (completed_sender, completed_receiver) = mpsc::sync_channel(1);

        dispatcher.dispatch(move || {
            entered_sender.send(()).expect("test observes first completion callback");
            worker_gate.wait();
        });
        entered_receiver.recv_timeout(Duration::from_secs(2)).expect("completion worker began first callback");
        dispatcher.dispatch(move || completed_sender.send(()).expect("test observes drained completion callback"));

        drop(dispatcher);
        drop(executor);
        gate.release();

        completed_receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("worker drained accepted completion after owner drop");
    }

    #[test]
    fn observation_delivery_rejects_all_blocking_close_apis_without_signaling_close() {
        let backend = inert_backend_for_observation_delivery_test();
        let (sender, receiver) = mpsc::sync_channel(1);
        let observer_backend = backend.clone();
        let owner = crate::observability::RuntimeObservationOwner::new(
            "reentrant-observation-close",
            None,
            crate::observability::RuntimeObservationConfig::bounded(8, 4 * 1024),
        );
        owner.register_seal_observer(move |_| {
            let sync = observer_backend.close_and_wait(Duration::from_secs(1));
            let until = observer_backend.close_and_wait_until(Instant::now() + Duration::from_secs(1));
            let asynchronous = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("test builds current-thread runtime")
                .block_on(observer_backend.close_and_wait_async(Duration::from_secs(1)));
            sender.send((sync, until, asynchronous)).expect("test receives close rejections");
        });

        owner
            .finish_query(crate::observability::RuntimeTerminalOutcome::Succeeded)
            .expect("test seals observation owner");
        let (sync, until, asynchronous) = receiver.recv_timeout(Duration::from_secs(2)).expect("observer delivered");
        for error in [sync.unwrap_err(), until.unwrap_err(), asynchronous.unwrap_err()] {
            assert_eq!(error.code(), QueryErrorCode::Internal);
            assert_eq!(error.fact_value("reason"), Some("reentrant_observation_close"));
        }
        assert!(!backend.close_snapshot().closing, "reentrant blocking close must not signal close");
        let close = backend.close_and_wait(Duration::from_secs(2)).expect("non-reentrant close succeeds");
        assert!(close.gpu_resource_quiescent);
        assert!(close.observation_quiescent);
    }

    fn journal_resource_labels(journal: &ConstructionResourceJournal) -> Result<Vec<&'static str>> {
        let entries = journal.entries.lock().map_err(|_| construction_journal_poisoned())?;
        Ok(entries
            .iter()
            .map(|entry| match entry {
                ConstructionResource::MemoryManager(_) => "memory_manager",
                ConstructionResource::StreamPool(_) => "stream_pool",
                ConstructionResource::DeviceContext(_) => "device_context",
                ConstructionResource::ExecutionPool(_) => "execution_pool",
                ConstructionResource::StartupRecords(_) => "startup_records",
            })
            .collect())
    }

    #[derive(Debug, Default)]
    struct CountingProbe(std::sync::atomic::AtomicUsize);

    impl GpuBackendConstructionProbe for CountingProbe {
        fn checkpoint(&self, _checkpoint: GpuBackendConstructionCheckpoint) -> Result<()> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    #[derive(Debug)]
    struct FailingProbe(GpuBackendConstructionCheckpoint);

    impl GpuBackendConstructionProbe for FailingProbe {
        fn checkpoint(&self, checkpoint: GpuBackendConstructionCheckpoint) -> Result<()> {
            if checkpoint == self.0 {
                return Err(QueryError::new(
                    QueryErrorCode::Internal,
                    BUILD_OPERATION,
                    "injected construction failure",
                )
                .with_fact("reason", "injected_construction_failure"));
            }
            Ok(())
        }
    }

    #[derive(Debug)]
    struct FailingObservationCompletionExecutorFactory {
        failing_lane: ObservationCompletionWorkerLane,
    }

    impl ObservationCompletionExecutorFactory for FailingObservationCompletionExecutorFactory {
        fn spawn(&self) -> Result<ObservationCompletionExecutor> {
            ObservationCompletionExecutor::spawn_with_lane_failure_for_test(self.failing_lane)
        }
    }

    #[derive(Debug, Default)]
    struct PublishedCheckpointProbe(AtomicU8);

    impl GpuBackendConstructionProbe for PublishedCheckpointProbe {
        fn checkpoint(&self, checkpoint: GpuBackendConstructionCheckpoint) -> Result<()> {
            if matches!(checkpoint, GpuBackendConstructionCheckpoint::Published) {
                self.0.store(1, Ordering::Release);
            }
            Ok(())
        }
    }

    #[derive(Debug)]
    struct JournalBoundaryProbe {
        checkpoint: GpuBackendConstructionCheckpoint,
        journal: Weak<ConstructionResourceJournal>,
        expected_resources: &'static [&'static str],
        inspected: AtomicU8,
    }

    impl GpuBackendConstructionProbe for JournalBoundaryProbe {
        fn checkpoint(&self, checkpoint: GpuBackendConstructionCheckpoint) -> Result<()> {
            if checkpoint != self.checkpoint {
                return Ok(());
            }
            let journal = self.journal.upgrade().ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    BUILD_OPERATION,
                    "construction journal was dropped before probe",
                )
                .with_fact("reason", "construction_journal_dropped_before_probe")
            })?;
            let observed = journal_resource_labels(&journal)?;
            if observed != self.expected_resources {
                return Err(QueryError::new(
                    QueryErrorCode::Internal,
                    BUILD_OPERATION,
                    "construction resource was not journaled before its injected boundary",
                )
                .with_fact("reason", "construction_journal_boundary_mismatch")
                .with_fact("checkpoint", format!("{checkpoint:?}"))
                .with_fact("expected_resources", self.expected_resources.join(","))
                .with_fact("observed_resources", observed.join(",")));
            }
            self.inspected.store(1, Ordering::Release);
            Err(QueryError::new(QueryErrorCode::Internal, BUILD_OPERATION, "injected construction failure")
                .with_fact("reason", "injected_construction_failure"))
        }
    }

    #[derive(Debug, Default)]
    struct BlockingGate {
        released: Mutex<bool>,
        changed: Condvar,
    }

    impl BlockingGate {
        fn wait(&self) {
            let mut released = self.released.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            while !*released {
                released = self.changed.wait(released).unwrap_or_else(std::sync::PoisonError::into_inner);
            }
        }

        fn release(&self) {
            *self.released.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = true;
            self.changed.notify_all();
        }
    }

    fn deployment() -> DeploymentIoCapability {
        use super::super::{GdsDriverProbeStatus, KvikioCompatModeStatus, NvidiaFsProbeStatus};
        DeploymentIoCapability::from_probe_inputs(
            NvidiaFsProbeStatus::DriverMissing,
            KvikioCompatModeStatus::Unset,
            GdsDriverProbeStatus::ProbeUnimplemented,
        )
    }

    fn controller(policy: QueryExecutionConfig) -> GpuBackendControllerDraft {
        GpuBackendControllerDraft::new(policy, QueueBounds::default())
    }

    fn one_device_backend_draft(device_ordinal: usize) -> GpuBackendCoreDraft {
        GpuBackendCoreDraft::new(
            vec![GpuBackendDeviceConfig::new(device_ordinal, GpuMemoryOwnership::WholeDeviceExclusive)],
            controller(QueryExecutionConfig::default()),
            deployment(),
        )
    }

    fn wait_for_live_owner_release(device_ordinal: usize) {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            match claim_live_backend(&[device_ordinal]) {
                Ok(lease) => {
                    drop(lease);
                    return;
                },
                Err(error)
                    if error.fact_value("reason") == Some("backend_already_live") && Instant::now() < deadline =>
                {
                    thread::sleep(Duration::from_millis(5));
                },
                Err(error) => panic!("owner claim was not released by construction cleanup: {error}"),
            }
        }
    }

    #[test]
    fn pure_config_rejects_invalid_device_profiles_before_construction() {
        let config = GpuBackendCoreDraft::new(
            vec![GpuBackendDeviceConfig::new(0, GpuMemoryOwnership::WholeDeviceExclusive).with_max_active_queries(0)],
            controller(QueryExecutionConfig::default()),
            deployment(),
        );
        let error = config.validate().unwrap_err();
        assert_eq!(error.fact_value("field"), Some("max_active_queries"));
        assert_eq!(error.fact_value("reason"), Some("zero"));
    }

    #[test]
    fn pure_config_rejects_explicit_zero_attempt_floor_before_construction() {
        let config = GpuBackendCoreDraft::new(
            vec![GpuBackendDeviceConfig::new(0, GpuMemoryOwnership::WholeDeviceExclusive).with_attempt_floor_bytes(0)],
            controller(QueryExecutionConfig::default()),
            deployment(),
        );

        let error = config.validate().unwrap_err();

        assert_eq!(error.fact_value("field"), Some("attempt_floor_bytes"));
        assert_eq!(error.fact_value("reason"), Some("zero"));
    }

    #[test]
    fn observation_completion_executor_spawn_failure_never_publishes_or_retains_live_owner() {
        let _owner_test = live_backend_test_lock();
        let device_ordinal = usize::MAX - 3;
        for failing_lane in
            [ObservationCompletionWorkerLane::Observation, ObservationCompletionWorkerLane::ActivityTail]
        {
            let published = Arc::new(PublishedCheckpointProbe::default());

            let error = GpuBackendCore::builder(one_device_backend_draft(device_ordinal))
                .with_construction_probe(published.clone())
                .with_observation_completion_executor_factory(Arc::new(FailingObservationCompletionExecutorFactory {
                    failing_lane,
                }))
                .build()
                .unwrap_err();

            assert_eq!(error.fact_value("reason"), Some("observation_completion_worker_spawn_failed"));
            assert_eq!(error.fact_value("worker_lane"), Some(failing_lane.as_str()));
            assert_eq!(published.0.load(Ordering::Acquire), 0, "executor spawn failure must precede publication");
            wait_for_live_owner_release(device_ordinal);
        }
    }

    #[test]
    fn invalid_config_records_no_construction_probe_activity() {
        let config = GpuBackendCoreDraft::new(
            vec![
                GpuBackendDeviceConfig::new(0, GpuMemoryOwnership::WholeDeviceExclusive)
                    .with_backend_reserve_bytes(256 * 1024 * 1024),
            ],
            controller(QueryExecutionConfig::default()),
            deployment(),
        );
        let probe = Arc::new(CountingProbe::default());

        let error = GpuBackendCore::builder(config).with_construction_probe(probe.clone()).build().unwrap_err();

        assert_eq!(error.fact_value("reason"), Some("reserve_exhausts_fallback"));
        assert_eq!(probe.0.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn derived_global_limit_is_checked_and_defaults_to_one_per_device() {
        let config = GpuBackendCoreDraft::new(
            vec![
                GpuBackendDeviceConfig::new(0, GpuMemoryOwnership::WholeDeviceExclusive),
                GpuBackendDeviceConfig::new(1, GpuMemoryOwnership::WholeDeviceExclusive),
            ],
            controller(QueryExecutionConfig::default()),
            deployment(),
        );
        assert_eq!(config.derived_global_max_active_queries().unwrap(), 2);
    }

    #[test]
    fn automatic_installation_bounds_cover_every_active_attempt() {
        let mut draft = GpuBackendCoreDraft::new(
            vec![GpuBackendDeviceConfig::new(0, GpuMemoryOwnership::WholeDeviceExclusive).with_max_active_queries(2)],
            controller(QueryExecutionConfig::default()),
            deployment(),
        );

        draft.normalize_installation_bounds().expect("automatic bounds derive");
        let device = draft.devices().first().expect("one device profile");
        assert_eq!(device.lane_count(), 2);
        assert_eq!(device.worker_count(), 2);
        assert!(device.stream_pool_size() >= 2);
        draft.validate().expect("derived multi-attempt installation profile validates");
    }

    #[test]
    fn automatic_bounds_keep_device_source_work_independent_of_execution_prefetch() {
        let execution = QueryExecutionConfig {
            source_scheduler_policy: crate::runtime::SourceSchedulerPolicy::new(7),
            ..QueryExecutionConfig::default()
        };
        let mut draft = GpuBackendCoreDraft::new(
            vec![GpuBackendDeviceConfig::new(0, GpuMemoryOwnership::WholeDeviceExclusive).with_max_active_queries(2)],
            controller(execution),
            deployment(),
        );

        draft.normalize_installation_bounds().expect("automatic bounds derive");

        let device = draft.devices().first().expect("one device profile");
        assert_eq!(device.source_work_per_attempt(), 1);
        assert_eq!(device.stream_pool_size(), 4);
        draft.validate().expect("device-local default remains valid");
    }

    fn assert_query_activity_allows_close_and_replacement_after_release(acquire_context: bool) {
        let _owner_test = live_backend_test_lock();
        if cudf::DeviceId::count().unwrap_or(0) == 0 {
            return;
        }
        let backend = GpuBackendCore::builder(one_device_backend_draft(0)).build().unwrap();
        let lifecycle = backend.observation_core.lifecycle.clone();
        let observed_active_gpu_permits = Arc::new(std::sync::atomic::AtomicUsize::new(usize::MAX));
        let (completion, completed) = mpsc::sync_channel(1);
        let activity = crate::runtime::QueryActivityLease::new(ActivityPermitOrderingProbe {
            lifecycle: lifecycle.clone(),
            backend: backend.clone(),
            observed_active_gpu_permits: Arc::clone(&observed_active_gpu_permits),
            completion,
        });
        let request = QueryAdmissionRequest::new().with_runtime_observation_activity(activity);
        let handle = backend.session_capability().admit_query_with_request(request).unwrap();
        assert_eq!(lifecycle.snapshot().active_gpu_permits, 1);
        if acquire_context {
            let context = handle.acquire_context().expect("query acquires a service-stream context");
            drop(context);
        }

        handle.finish();
        drop(handle);

        completed
            .recv_timeout(Duration::from_secs(30))
            .expect("activity probe completed")
            .expect("activity observed close quiescence and replacement ownership");
        assert_eq!(observed_active_gpu_permits.load(Ordering::Acquire), 0);
        let close = backend.close_and_wait(Duration::from_secs(10)).unwrap();
        assert!(close.gpu_resource_quiescent);
        assert!(close.observation_quiescent);
    }

    #[test]
    fn query_activity_returns_only_after_service_resources_and_backend_gpu_permit_release() {
        assert_query_activity_allows_close_and_replacement_after_release(false);
    }

    #[test]
    fn context_cleanup_completes_before_query_activity_can_close_and_replace_backend() {
        assert_query_activity_allows_close_and_replacement_after_release(true);
    }

    #[test]
    fn backend_observation_quiescence_waits_for_detached_seal_observers() {
        let _owner_test = live_backend_test_lock();
        if cudf::DeviceId::count().unwrap_or(0) == 0 {
            return;
        }
        let backend = GpuBackendCore::builder(one_device_backend_draft(0)).build().unwrap();
        let owner = crate::observability::RuntimeObservationOwner::new(
            "detached-seal-observer",
            None,
            crate::observability::RuntimeObservationConfig::bounded(32, 32 * 1024),
        );
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let observer_gate = Arc::clone(&gate);
        let (entered_sender, entered_receiver) = mpsc::sync_channel(1);
        owner.register_seal_observer(move |_snapshot| {
            let _ = entered_sender.send(());
            let (lock, changed) = &*observer_gate;
            let mut released = lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            while !*released {
                released = changed.wait(released).unwrap_or_else(std::sync::PoisonError::into_inner);
            }
        });
        let handle = backend
            .session_capability()
            .admit_query_with_request(QueryAdmissionRequest::new().with_runtime_observation_owner(owner))
            .unwrap();
        drop(handle.acquire_context().expect("query acquires a service-stream context"));
        handle.finish();
        drop(handle);

        entered_receiver
            .recv_timeout(Duration::from_secs(10))
            .expect("detached observation completion reaches its seal observer");
        let error = backend.close_and_wait(Duration::from_millis(250)).unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("backend_close_timeout"));
        assert_eq!(error.fact_value("gpu_resource_quiescent"), Some("true"));
        assert_eq!(error.fact_value("observation_quiescent"), Some("false"));
        assert_eq!(error.fact_value("active_gpu_permits"), Some("0"));
        assert_eq!(error.fact_value("active_observation_permits"), Some("1"));

        let (lock, changed) = &*gate;
        *lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = true;
        changed.notify_all();
        let close = backend.close_and_wait(Duration::from_secs(10)).unwrap();
        assert!(close.gpu_resource_quiescent);
        assert!(close.observation_quiescent);
    }

    #[test]
    fn reentrant_activity_close_does_not_block_later_query_stream_cleanup() {
        let _owner_test = live_backend_test_lock();
        if cudf::DeviceId::count().unwrap_or(0) == 0 {
            return;
        }
        let backend = GpuBackendCore::builder(GpuBackendCoreDraft::new(
            vec![
                GpuBackendDeviceConfig::new(0, GpuMemoryOwnership::WholeDeviceExclusive)
                    .with_max_active_queries(2)
                    .with_installation_bounds(6, 2, 2),
            ],
            controller(QueryExecutionConfig::default()),
            deployment(),
        ))
        .build()
        .unwrap();
        let (entered_sender, entered_receiver) = mpsc::sync_channel(1);
        let (completion_sender, completion_receiver) = mpsc::sync_channel(1);
        let first = backend
            .session_capability()
            .admit_query_with_request(QueryAdmissionRequest::new().with_runtime_observation_activity(
                crate::runtime::QueryActivityLease::new(ReentrantCloseActivityProbe {
                    backend: backend.clone(),
                    entered: entered_sender,
                    completion: completion_sender,
                }),
            ))
            .unwrap();
        let second = backend.session_capability().admit_query_with_request(QueryAdmissionRequest::new()).unwrap();
        drop(first.acquire_context().expect("first query acquires a service-stream context"));
        drop(second.acquire_context().expect("second query acquires a service-stream context"));

        first.finish();
        drop(first);
        entered_receiver
            .recv_timeout(Duration::from_secs(10))
            .expect("first activity begins its reentrant backend close");

        second.finish();
        drop(second);

        let close = completion_receiver
            .recv_timeout(Duration::from_secs(20))
            .expect("later stream cleanup unblocks the reentrant close")
            .expect("reentrant close reaches full quiescence");
        assert!(close.gpu_resource_quiescent);
        assert!(close.observation_quiescent);
    }

    #[test]
    fn post_worker_failure_transfers_owner_release_to_cleanup_coordinator() {
        let _owner_test = live_backend_test_lock();
        let ordinal = usize::MAX - 1;
        let config = one_device_backend_draft(ordinal);

        let error = GpuBackendCore::builder(config)
            .with_construction_probe(Arc::new(FailingProbe(GpuBackendConstructionCheckpoint::AfterCleanupWorker)))
            .build()
            .unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("injected_construction_failure"));
        assert!(matches!(error.fact_value("rollback_cleanup"), Some("quiescent" | "transferred")));

        wait_for_live_owner_release(ordinal);
    }

    #[test]
    fn every_constructed_resource_is_journaled_before_its_after_boundary() {
        let _owner_test = live_backend_test_lock();
        if cudf::DeviceId::count().unwrap_or(0) == 0 {
            return;
        }

        const MANAGER: &[&str] = &["memory_manager"];
        const STREAM: &[&str] = &["memory_manager", "stream_pool"];
        const CONTEXT: &[&str] = &["memory_manager", "stream_pool", "device_context"];
        const WORKER: &[&str] = &["memory_manager", "stream_pool", "device_context", "execution_pool"];
        const STARTUP: &[&str] =
            &["memory_manager", "stream_pool", "device_context", "execution_pool", "startup_records"];
        let cases = [
            (GpuBackendConstructionCheckpoint::AfterMemoryManager { device_ordinal: 0 }, MANAGER),
            (GpuBackendConstructionCheckpoint::AfterStreamPool { device_ordinal: 0 }, STREAM),
            (GpuBackendConstructionCheckpoint::AfterDeviceContext { device_ordinal: 0 }, CONTEXT),
            (GpuBackendConstructionCheckpoint::AfterWorkerPool { device_ordinal: 0 }, WORKER),
            (GpuBackendConstructionCheckpoint::AfterStartupRecords, STARTUP),
            (GpuBackendConstructionCheckpoint::Published, STARTUP),
        ];

        for (checkpoint, expected_resources) in cases {
            let journal = Arc::new(ConstructionResourceJournal::default());
            let probe = Arc::new(JournalBoundaryProbe {
                checkpoint,
                journal: Arc::downgrade(&journal),
                expected_resources,
                inspected: AtomicU8::new(0),
            });
            let error = GpuBackendCore::builder(one_device_backend_draft(0))
                .with_construction_probe(probe.clone())
                .build_with_construction_journal(journal)
                .unwrap_err();

            assert_eq!(error.fact_value("reason"), Some("injected_construction_failure"), "{checkpoint:?}");
            assert_eq!(probe.inspected.load(Ordering::Acquire), 1, "{checkpoint:?}");
            assert!(matches!(error.fact_value("rollback_cleanup"), Some("quiescent" | "transferred")));
            wait_for_live_owner_release(0);
        }
    }

    #[test]
    fn owner_is_retained_until_blocked_execution_runtime_drains_and_joins() {
        let _owner_test = live_backend_test_lock();
        let ordinal = usize::MAX - 2;
        let owner_lease = claim_live_backend(&[ordinal]).unwrap();
        let journal = Arc::new(ConstructionResourceJournal::default());
        let pool = super::super::GpuExecutionPool::new(ordinal, 1).unwrap();
        journal.record_execution_pool(pool.clone()).unwrap();
        let gate = Arc::new(BlockingGate::default());
        let worker_gate = Arc::clone(&gate);
        let (started_sender, started_receiver) = mpsc::sync_channel(0);
        pool.spawn(super::super::GpuExecutionJobKind::Cleanup, async move {
            started_sender.send(()).unwrap();
            worker_gate.wait();
        })
        .unwrap();
        started_receiver.recv().unwrap();

        let generation = Arc::new(ResourceGeneration::new(next_backend_identity(), owner_lease, Arc::clone(&journal)));
        let resource_core =
            Arc::new(ResourceCore { identity: next_backend_identity(), generation: Mutex::new(Some(generation)) });
        let observation_core = Arc::new(ObservationCore {
            identity: next_backend_identity(),
            lifecycle: BackendLifecycle::new(),
            attempt_counters: Arc::new(AttemptBackendCounters::default()),
            attempt_completion_queue: None,
            completion_executor: ObservationCompletionExecutor::spawn().unwrap(),
        });
        let coordinator = CloseCoordinator::spawn(Arc::clone(&resource_core), Arc::clone(&observation_core)).unwrap();

        assert!(coordinator.signal());
        let timeout = coordinator
            .state
            .wait_until(Instant::now() + Duration::from_millis(250), &observation_core.lifecycle)
            .unwrap_err();
        assert_eq!(timeout.fact_value("reason"), Some("backend_close_timeout"));
        let replacement = claim_live_backend(&[ordinal]).unwrap_err();
        assert_eq!(replacement.fact_value("reason"), Some("backend_already_live"));

        gate.release();
        let snapshot =
            coordinator.state.wait_until(Instant::now() + Duration::from_secs(5), &observation_core.lifecycle).unwrap();
        assert!(snapshot.gpu_resource_quiescent);
        assert!(snapshot.observation_quiescent);
        pool.join_supervisor().expect("coordinator joined the runtime supervisor before releasing ownership");
        let replacement = claim_live_backend(&[ordinal]).expect("joined runtime permits a replacement owner");
        drop(replacement);
    }

    #[test]
    fn cleanup_deadlines_become_late_success_only_for_the_same_phase() {
        let state = CloseCompletion::new();
        let lifecycle = BackendLifecycle::new();
        let phases = [
            QueryServiceDrainPhase::StreamPoolClose.as_str(),
            QueryServiceDrainPhase::AdmissionQuiescence.as_str(),
            QueryServiceDrainPhase::ExecutionPoolQuiescence.as_str(),
            QueryServiceDrainPhase::StreamPoolQuiescence.as_str(),
            QueryServiceDrainPhase::ResourceEpoch.as_str(),
            QueryServiceDrainPhase::FinalCompletion.as_str(),
            "construction_journal",
            "gpu_permit_quiescence",
            "observation_quiescence",
        ];
        for phase in phases {
            state.record_deadline(phase);
            state.record_deadline(phase);
        }
        assert_eq!(state.snapshot(&lifecycle).cleanup_deadline_elapsed_count, phases.len() as u64);

        state.record_quiescent(phases[0]);
        let snapshot = state.snapshot(&lifecycle);
        assert_eq!(snapshot.cleanup_late_success_count, 1);
        let progress = state.cleanup_phases.lock().unwrap();
        assert_eq!(progress.get(phases[0]), Some(&CleanupPhaseProgress::Quiescent));
        assert!(phases[1..].iter().all(|phase| progress.get(phase) == Some(&CleanupPhaseProgress::DeadlineElapsed)));
        drop(progress);

        for phase in &phases[1..] {
            state.record_quiescent(phase);
            state.record_quiescent(phase);
        }
        state.mark_closing();
        state.mark_gpu_quiescent();
        state.mark_observation_quiescent();
        let snapshot = state.snapshot(&lifecycle);

        assert_eq!(snapshot.cleanup_deadline_elapsed_count, phases.len() as u64);
        assert_eq!(snapshot.cleanup_late_success_count, phases.len() as u64);
        assert!(!snapshot.cleanup_failed);
    }

    #[test]
    fn observation_failure_does_not_erase_gpu_quiescence() {
        let state = CloseCompletion::new();
        let lifecycle = BackendLifecycle::new();
        state.mark_closing();
        state.mark_gpu_quiescent();
        state.fail("observation_quiescence", "injected observation cleanup failure");

        let snapshot = state.snapshot(&lifecycle);
        assert!(snapshot.gpu_resource_quiescent);
        assert!(!snapshot.observation_quiescent);
        assert!(snapshot.cleanup_failed);
    }

    #[test]
    fn begin_attempt_after_close_is_rejected_without_creating_an_attempt() {
        let _owner_test = live_backend_test_lock();
        if cudf::DeviceId::count().unwrap_or(0) == 0 {
            return;
        }
        let backend = GpuBackendCore::builder(one_device_backend_draft(0)).build().unwrap();
        let session = backend.session_capability();
        assert!(backend.close());
        let before = session.attempt_counters_snapshot();
        let error = session.begin_attempt(BeginAttemptRequest::default()).unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::Cancelled);
        assert_eq!(error.fact_value("reason"), Some("backend_closing"));
        let after = session.attempt_counters_snapshot();
        assert_eq!(before.attempts_begun, after.attempts_begun, "a losing begin creates no counted attempt");
        backend.close_and_wait(Duration::from_secs(10)).unwrap();
    }

    #[test]
    fn begin_attempt_winning_the_close_gate_is_counted_until_report_ready() {
        let _owner_test = live_backend_test_lock();
        if cudf::DeviceId::count().unwrap_or(0) == 0 {
            return;
        }
        let backend = GpuBackendCore::builder(one_device_backend_draft(0)).build().unwrap();
        let session = backend.session_capability();
        let begin = session.begin_attempt(BeginAttemptRequest::default()).unwrap();
        assert_eq!(session.attempt_counters_snapshot().attempts_begun, 1);

        assert!(backend.close());
        let snapshot = backend.close_snapshot();
        assert!(!snapshot.observation_quiescent, "an open attempt holds observation scope open");

        // A CPU-only attempt terminalizes without GPU cleanup and publishes
        // its report; only then may observation quiescence complete.
        begin.owner.commit_terminal(crate::observability::RuntimeTerminalOutcome::Succeeded).unwrap();
        let report = begin
            .completion
            .wait_timeout(Duration::from_secs(5))
            .expect("the attempt publishes its report during close");
        assert!(matches!(report.outcome, crate::observability::RuntimeTerminalOutcome::Succeeded));
        drop(begin);
        assert_eq!(session.attempt_counters_snapshot().reports_ready, 1);
        let close = backend.close_and_wait(Duration::from_secs(10)).unwrap();
        assert!(close.observation_quiescent);
    }

    #[test]
    fn abandoned_attempt_reports_land_in_the_bounded_completion_queue() {
        let _owner_test = live_backend_test_lock();
        if cudf::DeviceId::count().unwrap_or(0) == 0 {
            return;
        }
        let draft = one_device_backend_draft(0).with_attempt_completion_queue_capacity(2);
        let backend = GpuBackendCore::builder(draft).build().unwrap();
        let session = backend.session_capability();

        let begin = session.begin_attempt(BeginAttemptRequest::default()).unwrap();
        drop(begin.completion);
        begin.owner.commit_terminal(crate::observability::RuntimeTerminalOutcome::Dropped).unwrap();
        drop(begin.owner);

        let drained = session.take_abandoned_attempt_reports(8);
        assert_eq!(drained.len(), 1);
        assert!(matches!(drained[0].outcome, crate::observability::RuntimeTerminalOutcome::Dropped));
        let counters = session.attempt_counters_snapshot();
        assert_eq!(counters.completion_queue_pushed, 1);
        assert_eq!(counters.completion_queue_overflowed, 0);
        backend.close_and_wait(Duration::from_secs(10)).unwrap();
    }
}
