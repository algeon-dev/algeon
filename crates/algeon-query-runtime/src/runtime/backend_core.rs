//! Ownership-enforcing GPU backend construction and lifecycle facade.
//!
//! Configuration specifications (`GpuBackendDeviceProfile`,
//! `GpuBackendDeviceConfig`, `GpuBackendConfig`) are validated into a
//! `ValidatedGpuBackendConfig` before process-wide resources are acquired.
//! `GpuBackendCore::construct` claims the exclusive live-backend lease for the
//! configured devices, installs the query service and associated GPU resources within
//! a resource generation, and publishes the core once each device startup record
//! matches its validated profile. Downstream ownership is partitioned across
//! distinct handle types: `GpuBackendCore` acts as a reference-counted owning handle
//! that sustains the resource generation until the final instance drops, whereas
//! `GpuBackendSession` maintains a `Weak` reference to prevent extending device
//! ownership beyond backend close.
//!
//! The lifetime invariant ensures that admission, service reads, and attempt
//! initiations can hold the resource generation only when granted by the
//! `BackendLifecycle` gate prior to close. Backend closure releases the live
//! owner lease after the query service achieves quiescence and all active holders
//! terminate. The lifecycle gate resides in `backend_lifecycle`, while generation
//! cleanup executes in `backend_close`. The governing admission contract is defined in
//! `website/docs/design/admission.mdx#from-capability-validation-to-admission`, and thread
//! and runtime ownership rules are described in `website/docs/design/index.mdx#runtimes-and-threads`.

use std::{
    fmt,
    sync::{
        Arc, Mutex, OnceLock, PoisonError, Weak,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use cudf::{
    DeviceMemoryManager, DeviceMemoryManagerInstallationState, DeviceMemoryManagerOptions,
    io::{KvikioProcessConfig, configure_kvikio_process},
};
use query_planning::error::{QueryError, QueryErrorCode, Result, bytes_with_gib};
use tokio::task::spawn_blocking;

#[cfg(feature = "qualification")]
use super::DeviceQuarantineInjection;
pub use super::backend_close::GpuBackendCloseSnapshot;
pub(crate) use super::observation_completion::ObservationCompletionDispatcher;
use super::{
    BackendDiagnostics, DeviceCapabilities, DeviceFacts, DeviceResourceProfile, DeviceTelemetrySnapshot,
    QueryAdmissionRequest, QueryExecutionConfig, QueryHandle, QueueBounds,
    backend_close::{CloseCoordinator, ConstructionRollback, reject_reentrant_observation_close},
    backend_lifecycle::{
        BackendLifecycle, GpuLifecyclePermit, ObservationPermit, PublicHandleLease, PublicHandleTracker,
    },
    backend_metrics::{
        BACKEND_METRIC_DEVICE_CAPACITY, BackendAttemptDurations, BackendMetricsSnapshot, BackendMetricsStore,
    },
    gpu::{AlgeonGpuInventory, GpuInventory, LiveBackendLease, claim_live_backend, normalize_device_ordinals},
    gpu_execution_pool::RESERVED_LIFECYCLE_WORKERS,
    observation_completion::ObservationCompletionExecutor,
    query_service::QueryService,
};
use crate::{
    attempt::{
        AttemptBackendCounters, AttemptBackendCountersSnapshot, AttemptBegin, AttemptCompletionQueue, AttemptOwner,
        BeginAttemptRequest,
    },
    native_error::cudf_query_error,
    observability::QueryAttemptReport,
};

const BUILD_OPERATION: &str = "query_engine.gpu_backend_core.build";
const CLOSE_OPERATION: &str = "query_engine.gpu_backend_core.close";
const RESOURCE_OPERATION: &str = "query_engine.gpu_backend_core.resource";

/// Deployment ownership promised for one CUDA device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GpuMemoryOwnership {
    /// Algeon is the exclusive allocator owner for the complete visible device.
    WholeDeviceExclusive,
    /// Algeon is constrained by an immutable process-manager allocation limit.
    StaticPartition {
        /// Immutable process-manager allocator limit in bytes.
        allocation_limit_bytes: u64,
    },
}

impl GpuMemoryOwnership {
    pub(crate) fn allocation_limit_bytes(self) -> Option<u64> {
        match self {
            Self::WholeDeviceExclusive => None,
            Self::StaticPartition { allocation_limit_bytes } => Some(allocation_limit_bytes),
        }
    }
}

/// Pre-build capacity probe for one CUDA device.
///
/// Reports the declarable capacity of one device without acquiring the process
/// owner lease or installing any allocator: the CUDA-visible totals, the NVML
/// physical total when reported, the owned capacity for the requested
/// ownership, and the managed capacity left after the backend reserve.
/// Build-time resolution stays the enforcement authority; this probe
/// predicts ownership-vs-facts inputs without requiring callers to resolve
/// limits through repeated `build()` failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct GpuDeviceCapacityProbe {
    pub(crate) device_ordinal: usize,
    /// NVML-reported physical framebuffer capacity, when the optional NVML
    /// substrate reported one. CUDA-visible capacity may be smaller by the
    /// driver-reserved framebuffer.
    pub(crate) physical_total_bytes: Option<u64>,
    /// CUDA-visible allocation capacity. Ownership and explicit limits are
    /// compared against this value. The NVML total plays no part in that comparison.
    pub cuda_total_bytes: u64,
    /// CUDA-visible free allocation bytes sampled by the probe; not an admission balance.
    pub cuda_free_bytes: u64,
    /// CUDA-visible total or the promised static-partition limit, in bytes.
    pub owned_capacity_bytes: u64,
    /// Largest explicit managed-capacity limit this ownership accepts,
    /// before the backend reserve is subtracted.
    pub max_declarable_limit_bytes: u64,
    pub(crate) backend_reserve_bytes: u64,
    /// Owned capacity minus backend reserve, in bytes.
    pub max_managed_capacity_bytes: u64,
}

/// Queries declarable capacity for one device without building a backend.
///
/// This probe acquires no live-backend lease and installs no memory
/// manager. Setting the explicit limit to at most `max_declarable_limit_bytes`
/// ensures the build will not fail on capacity limits. Refusals carry the same
/// `reason` facts reported by backend initialization.
pub fn probe_device_capacity(
    device_ordinal: usize,
    memory_ownership: GpuMemoryOwnership,
    backend_reserve_bytes: u64,
) -> Result<GpuDeviceCapacityProbe> {
    let facts = AlgeonGpuInventory.facts(device_ordinal)?;
    let (owned_capacity_bytes, max_managed_capacity_bytes) = resolve_probe_capacities(
        device_ordinal,
        &facts,
        memory_ownership.allocation_limit_bytes(),
        backend_reserve_bytes,
    )?;
    Ok(GpuDeviceCapacityProbe {
        device_ordinal,
        physical_total_bytes: facts.physical_total_bytes,
        cuda_total_bytes: facts.total_bytes,
        cuda_free_bytes: facts.free_bytes,
        owned_capacity_bytes,
        max_declarable_limit_bytes: owned_capacity_bytes,
        backend_reserve_bytes,
        max_managed_capacity_bytes,
    })
}

fn resolve_probe_capacities(
    device_ordinal: usize,
    facts: &DeviceFacts,
    allocation_limit_bytes: Option<u64>,
    backend_reserve_bytes: u64,
) -> Result<(u64, u64)> {
    const OPERATION: &str = "query_engine.gpu_backend_core.probe_capacity";
    if let Some(allocation_limit) = allocation_limit_bytes.filter(|limit| *limit > facts.total_bytes) {
        return Err(QueryError::new(
            QueryErrorCode::InvalidConfiguration,
            OPERATION,
            format!(
                "device {device_ordinal} static partition allocation_limit_bytes is {}, above the CUDA-visible device memory of {}; the largest accepted allocation_limit_bytes is {}",
                bytes_with_gib(allocation_limit),
                bytes_with_gib(facts.total_bytes),
                bytes_with_gib(facts.total_bytes),
            ),
        )
        .with_fact("device_ordinal", device_ordinal)
        .with_fact("allocation_limit_bytes", allocation_limit)
        .with_fact("cuda_total_bytes", facts.total_bytes)
        .with_fact(
            "physical_total_bytes",
            facts.physical_total_bytes.map_or("unavailable".to_owned(), |total| total.to_string()),
        )
        .with_fact("reason", "static_partition_exceeds_physical_capacity"));
    }
    let owned_capacity_bytes = allocation_limit_bytes.unwrap_or(facts.total_bytes);
    let max_managed_capacity_bytes =
        owned_capacity_bytes.checked_sub(backend_reserve_bytes).filter(|capacity| *capacity > 0).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InvalidConfiguration,
                OPERATION,
                format!(
                    "device {device_ordinal} backend_reserve_bytes is {} and leaves no managed capacity out of the owned {}; the largest accepted backend_reserve_bytes is {}",
                    bytes_with_gib(backend_reserve_bytes),
                    bytes_with_gib(owned_capacity_bytes),
                    bytes_with_gib(owned_capacity_bytes.saturating_sub(1)),
                ),
            )
            .with_fact("device_ordinal", device_ordinal)
            .with_fact("owned_capacity_bytes", owned_capacity_bytes)
            .with_fact("backend_reserve_bytes", backend_reserve_bytes)
            .with_fact("reason", "backend_reserve_exhausts_capacity")
        })?;
    Ok((owned_capacity_bytes, max_managed_capacity_bytes))
}

/// Soft admission capacity calculation for one device.
///
/// `Automatic` resolves at build time, after the device is discovered, to the
/// owned capacity minus `headroom_bytes`, or to `fallback_bytes` when the
/// subtraction leaves nothing; `fraction` further caps the result at that
/// share of the owned capacity. `Explicit` is accepted as given and is
/// rejected when it exceeds the owned capacity: at validation for a static
/// partition, at build for a whole device. Both describe the ledger's
/// configured limit; admission commits grants against it, and neither
/// constrains the allocator installed on the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GpuDeviceMemoryLimit {
    /// Discovers admission capacity from the promised owned device capacity.
    /// The default reserves 500 MiB, falls back to 256 MiB and applies no fraction.
    Automatic {
        /// Bytes subtracted from owned capacity before resolving the limit.
        headroom_bytes: u64,
        /// Positive byte limit used when subtraction leaves no capacity.
        fallback_bytes: u64,
        /// Optional further cap on the share of owned capacity.
        fraction: Option<GpuMemoryFraction>,
    },
    /// Declares an immutable byte limit checked against owned device capacity.
    Explicit {
        /// Admission capacity in bytes before backend reserve is subtracted.
        limit_bytes: u64,
    },
}

impl Default for GpuDeviceMemoryLimit {
    fn default() -> Self {
        Self::Automatic { headroom_bytes: 500 * 1024 * 1024, fallback_bytes: 256 * 1024 * 1024, fraction: None }
    }
}

/// Validated automatic-capacity fraction retained unresolved until post-claim
/// device discovery. The private bit representation makes `Eq`/`Hash` sound.
/// NaN, infinities, zero, and out-of-range values cannot be constructed.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuMemoryFraction(u64);

impl GpuMemoryFraction {
    /// Validates a capacity fraction in `(0, 1]`.
    /// Returns `InvalidConfiguration` for zero, negative values, values above
    /// one, NaN or infinity.
    pub fn new(value: f64) -> Result<Self> {
        if !value.is_finite() || value <= 0.0 || value > 1.0 {
            return Err(invalid_config("memory_limit.fraction", "must be greater than zero and at most one")
                .with_fact("reason", "invalid_fraction"));
        }
        Ok(Self(value.to_bits()))
    }

    /// Returns the validated fraction without applying it to a device capacity.
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

/// Explicit installation limits for one device profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct GpuBackendDeviceInstallationBounds {
    /// Installed streams, covering root and auxiliary slots for active attempts.
    pub stream_pool_size: usize,
    /// Worker threads, covering active attempts plus the reserved lifecycle worker.
    pub worker_count: usize,
    /// Execution lanes, at least the maximum active-attempt count.
    pub lane_count: usize,
}

impl GpuBackendDeviceInstallationBounds {
    /// Creates explicit stream, worker, and execution-lane bounds.
    pub const fn new(stream_pool_size: usize, worker_count: usize, lane_count: usize) -> Self {
        Self { stream_pool_size, worker_count, lane_count }
    }
}

/// Pure per-device construction profile validated before owner acquisition.
///
/// Encapsulates device-local resource policies and configuration defaults,
/// including capacity, reserve, query limit, active-attempt count, source work,
/// installation bounds, capabilities, and telemetry. Retains no memory ownership;
/// the deployment owner binds a profile with explicit [`GpuMemoryOwnership`]
/// via [`GpuBackendDeviceConfig::new`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuBackendDeviceProfile {
    device_ordinal: usize,
    device_capabilities: DeviceCapabilities,
    memory_limit: GpuDeviceMemoryLimit,
    backend_reserve_bytes: u64,
    cache_cap_bytes: u64,
    default_query_limit_bytes: Option<u64>,
    source_work_per_attempt: usize,
    max_active_attempts: usize,
    installation_bounds: Option<GpuBackendDeviceInstallationBounds>,
    install_telemetry: bool,
}

impl GpuBackendDeviceProfile {
    /// Absence derives the default grant from the resolved query share
    /// divided by the maximum active-attempt count.
    pub(crate) const DEFAULT_QUERY_LIMIT_BYTES: Option<u64> = None;
    /// One auxiliary source-work slot is reserved for each active attempt by
    /// default.
    pub(crate) const DEFAULT_SOURCE_WORK_PER_ATTEMPT: usize = 1;
    /// Device admission begins with two active attempts unless the installer
    /// explicitly freezes a different per-device limit.
    pub(crate) const DEFAULT_MAX_ACTIVE_ATTEMPTS: usize = 2;

    /// Starts a profile with automatic capacity discovery and two active
    /// attempts. Callers may override any device-local bound below.
    pub fn new(device_ordinal: usize) -> Self {
        Self {
            device_ordinal,
            device_capabilities: DeviceCapabilities::default(),
            memory_limit: GpuDeviceMemoryLimit::default(),
            backend_reserve_bytes: 0,
            cache_cap_bytes: 0,
            default_query_limit_bytes: Self::DEFAULT_QUERY_LIMIT_BYTES,
            source_work_per_attempt: Self::DEFAULT_SOURCE_WORK_PER_ATTEMPT,
            max_active_attempts: Self::DEFAULT_MAX_ACTIVE_ATTEMPTS,
            installation_bounds: None,
            install_telemetry: true,
        }
    }

    /// Sets the admission capacity policy. Validation and device discovery
    /// resolve the policy before managed capacity is installed.
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

    /// Sets bytes reserved outside managed query and cache capacity.
    /// Backend construction rejects a reserve that leaves no managed capacity.
    pub fn with_backend_reserve_bytes(mut self, backend_reserve_bytes: u64) -> Self {
        self.backend_reserve_bytes = backend_reserve_bytes;
        self
    }

    /// Sets the aggregate accounted resident-cache ceiling for this device.
    pub fn with_cache_cap_bytes(mut self, cache_cap_bytes: u64) -> Self {
        self.cache_cap_bytes = cache_cap_bytes;
        self
    }

    /// Returns the accounted resident-cache ceiling in bytes. The default is zero.
    pub fn cache_cap_bytes(&self) -> u64 {
        self.cache_cap_bytes
    }

    /// Sets the immutable default memory limit for each query on this device.
    /// When unset, it resolves to the query share divided by the maximum
    /// active-attempt count, rounded down to whole bytes.
    pub fn with_default_query_limit_bytes(mut self, default_query_limit_bytes: u64) -> Self {
        self.default_query_limit_bytes = Some(default_query_limit_bytes);
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

    /// Sets the device admission concurrency bound. Zero is rejected by
    /// backend validation; existing explicit installation bounds remain in force.
    pub fn with_max_active_attempts(mut self, max_active_attempts: usize) -> Self {
        self.max_active_attempts = max_active_attempts;
        self
    }

    /// Sets the active-attempt limit and clears explicit installation bounds
    /// so construction derives them from this profile.
    ///
    /// Prefer this over spelling `with_max_active_attempts` plus
    /// `with_installation_bounds` separately: the pair carries the same
    /// number twice with no type-level guarantee they agree. Explicit bounds
    /// remain available through `with_installation_bounds` for deployments
    /// that deliberately overprovision stream or worker capacity.
    pub fn with_derived_installation_bounds(mut self, max_active_attempts: usize) -> Self {
        self.max_active_attempts = max_active_attempts;
        self.installation_bounds = None;
        self
    }

    /// Sets explicit stream, worker, and execution-lane bounds.
    ///
    /// Most callers should prefer [`Self::with_derived_installation_bounds`];
    /// use this only to deliberately overprovision beyond the derived
    /// `A_d * (1 + S_d)` stream formula.
    pub fn with_installation_bounds(mut self, installation_bounds: GpuBackendDeviceInstallationBounds) -> Self {
        self.installation_bounds = Some(installation_bounds);
        self
    }

    /// Enables or disables installation of diagnostic device telemetry.
    /// Telemetry does not participate in query admission accounting.
    pub fn with_telemetry(mut self, install_telemetry: bool) -> Self {
        self.install_telemetry = install_telemetry;
        self
    }

    /// Returns the CUDA device ordinal selected by this declaration.
    pub fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    /// Immutable native execution capabilities selected for this device.
    pub fn device_capabilities(&self) -> &DeviceCapabilities {
        &self.device_capabilities
    }

    /// Returns the unresolved admission capacity policy, resolved at device discovery.
    pub fn memory_limit(&self) -> GpuDeviceMemoryLimit {
        self.memory_limit
    }

    /// Returns bytes reserved outside managed query and cache capacity. The default is zero.
    pub fn backend_reserve_bytes(&self) -> u64 {
        self.backend_reserve_bytes
    }

    /// Returns the explicit default query grant in bytes. `None` requests
    /// `floor((managed_capacity_bytes - cache_cap_bytes) / max_active_attempts)`
    /// after device capacity resolution.
    pub fn default_query_limit_bytes(&self) -> Option<u64> {
        self.default_query_limit_bytes
    }

    /// Returns auxiliary source-work slots reserved for each active attempt. The default is one.
    pub fn source_work_per_attempt(&self) -> usize {
        self.source_work_per_attempt
    }

    /// Returns the maximum simultaneously admitted attempts on this device. The default is two.
    pub fn max_active_attempts(&self) -> usize {
        self.max_active_attempts
    }

    /// Returns the explicit installation bounds. `None` requests the values
    /// returned by [`Self::derived_installation_bounds`].
    pub fn installation_bounds(&self) -> Option<GpuBackendDeviceInstallationBounds> {
        self.installation_bounds
    }

    /// The installation bounds a profile without explicit ones receives:
    /// `A_d * (1 + S_d)` stream slots, `A_d` execution lanes, and one worker
    /// per active attempt plus the worker reserved for context setup and
    /// release cleanup. Explicit bounds must cover the stream slots and
    /// workers of this derivation.
    pub fn derived_installation_bounds(&self) -> Result<GpuBackendDeviceInstallationBounds> {
        let auxiliary = self.max_active_attempts.checked_mul(self.source_work_per_attempt).ok_or_else(|| {
            invalid_device_config(self.device_ordinal, "source_work_per_attempt", "overflowed")
                .with_fact("reason", "aux_source_work_overflow")
        })?;
        let stream_pool_size = self.max_active_attempts.checked_add(auxiliary).ok_or_else(|| {
            invalid_device_config(self.device_ordinal, "stream_pool_size", "overflowed")
                .with_fact("reason", "required_slot_overflow")
        })?;
        let worker_count = self.max_active_attempts.checked_add(RESERVED_LIFECYCLE_WORKERS).ok_or_else(|| {
            invalid_device_config(self.device_ordinal, "worker_count", "overflowed")
                .with_fact("reason", "required_worker_overflow")
        })?;
        Ok(GpuBackendDeviceInstallationBounds::new(stream_pool_size, worker_count, self.max_active_attempts))
    }

    /// Returns whether device telemetry is installed. New profiles enable it.
    pub fn install_telemetry(&self) -> bool {
        self.install_telemetry
    }
}

/// Pure per-device construction value binding a profile to deployment ownership.
///
/// The profile owns every device-local default; this value only completes the
/// binding with the process owner's explicit [`GpuMemoryOwnership`]. Backend
/// construction resolves installation bounds and validates the bound value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuBackendDeviceConfig {
    profile: GpuBackendDeviceProfile,
    memory_ownership: GpuMemoryOwnership,
}

impl GpuBackendDeviceConfig {
    /// Builds one device config per ordinal from a shared template profile.
    ///
    /// Every returned config clones the template policy (capacity, reserve,
    /// query limit, attempts, bounds, capabilities, telemetry) and rebinds
    /// only the device ordinal; the template's own ordinal is ignored. The
    /// deployment ownership is shared by every device. This is the shorthand
    /// for the common homogeneous multi-device deployment; heterogeneous
    /// devices still declare one [`Self::new`] each.
    pub fn homogeneous(
        device_ordinals: impl IntoIterator<Item = usize>,
        template: &GpuBackendDeviceProfile,
        memory_ownership: GpuMemoryOwnership,
    ) -> Vec<Self> {
        device_ordinals
            .into_iter()
            .map(|device_ordinal| {
                let mut profile = template.clone();
                profile.device_ordinal = device_ordinal;
                Self::new(profile, memory_ownership)
            })
            .collect()
    }

    /// Binds a device profile to the deployment ownership promise.
    /// Validation occurs when the backend builder validates or builds this config.
    pub fn new(profile: GpuBackendDeviceProfile, memory_ownership: GpuMemoryOwnership) -> Self {
        Self { profile, memory_ownership }
    }

    /// Returns the device-local construction policy bound to this ownership promise.
    pub fn profile(&self) -> &GpuBackendDeviceProfile {
        &self.profile
    }

    /// Returns the CUDA device ordinal selected by this declaration.
    pub fn device_ordinal(&self) -> usize {
        self.profile.device_ordinal
    }

    /// Immutable native execution capabilities selected for this device.
    pub fn device_capabilities(&self) -> &DeviceCapabilities {
        &self.profile.device_capabilities
    }

    /// Returns the deployment ownership promised by the process owner.
    pub fn memory_ownership(&self) -> GpuMemoryOwnership {
        self.memory_ownership
    }

    /// Returns the unresolved admission capacity policy, resolved at device discovery.
    pub fn memory_limit(&self) -> GpuDeviceMemoryLimit {
        self.profile.memory_limit
    }

    /// Returns bytes reserved outside managed query and cache capacity. The default is zero.
    pub fn backend_reserve_bytes(&self) -> u64 {
        self.profile.backend_reserve_bytes
    }

    /// Returns the accounted resident-cache ceiling in bytes. The default is zero.
    pub fn cache_cap_bytes(&self) -> u64 {
        self.profile.cache_cap_bytes
    }

    /// Returns the explicit default query grant in bytes. `None` requests
    /// `floor((managed_capacity_bytes - cache_cap_bytes) / max_active_attempts)`
    /// after device capacity resolution.
    pub fn default_query_limit_bytes(&self) -> Option<u64> {
        self.profile.default_query_limit_bytes
    }

    /// Returns auxiliary source-work slots reserved for each active attempt. The default is one.
    pub fn source_work_per_attempt(&self) -> usize {
        self.profile.source_work_per_attempt
    }

    /// Returns the maximum simultaneously admitted attempts on this device. The default is two.
    pub fn max_active_attempts(&self) -> usize {
        self.profile.max_active_attempts
    }

    /// Returns whether device telemetry is installed. New profiles enable it.
    pub fn install_telemetry(&self) -> bool {
        self.profile.install_telemetry
    }
}

/// Backend construction input for the device set, the execution and queue
/// policy, and the optional attempt completion queue.
///
/// [`Self::validate`] produces the [`ValidatedGpuBackendConfig`] required by
/// [`GpuBackendCore::new`], ensuring declaration rules execute prior to claiming
/// the process owner.
#[derive(Debug, Clone)]
pub struct GpuBackendConfig {
    devices: Vec<GpuBackendDeviceConfig>,
    execution_config: QueryExecutionConfig,
    queue_bounds: QueueBounds,
    attempt_completion_queue_capacity: Option<usize>,
}

impl GpuBackendConfig {
    pub fn new(
        devices: Vec<GpuBackendDeviceConfig>,
        execution_config: QueryExecutionConfig,
        queue_bounds: QueueBounds,
    ) -> Self {
        Self { devices, execution_config, queue_bounds, attempt_completion_queue_capacity: None }
    }

    pub fn with_attempt_completion_queue_capacity(mut self, capacity: usize) -> Self {
        self.attempt_completion_queue_capacity = Some(capacity);
        self
    }

    /// Resolves each device's installation bounds and checks every device,
    /// execution, and queue rule. Devices are ordered by ordinal.
    pub fn validate(self) -> Result<ValidatedGpuBackendConfig> {
        let Self { mut devices, execution_config, queue_bounds, attempt_completion_queue_capacity } = self;
        let ordinals = devices.iter().map(GpuBackendDeviceConfig::device_ordinal).collect::<Vec<_>>();
        normalize_device_ordinals(&ordinals)?;
        devices.sort_by_key(GpuBackendDeviceConfig::device_ordinal);
        if let Some(device) = devices.iter().find(|device| device.device_ordinal() >= BACKEND_METRIC_DEVICE_CAPACITY) {
            return Err(invalid_device_config(
                device.device_ordinal(),
                "device_ordinal",
                "exceeds the fixed backend metrics device capacity",
            )
            .with_fact("device_capacity", BACKEND_METRIC_DEVICE_CAPACITY)
            .with_fact("reason", "backend_metrics_device_ordinal_out_of_range"));
        }
        let devices = devices.into_iter().map(validate_device).collect::<Result<Vec<_>>>()?;
        let derived = devices.iter().try_fold(0usize, |sum, device| {
            sum.checked_add(device.config.profile.max_active_attempts).ok_or_else(|| {
                invalid_config("max_active_attempts", "derived global maximum overflowed")
                    .with_fact("reason", "derived_global_max_overflow")
            })
        })?;
        if derived == 0 {
            return Err(invalid_config("max_active_attempts", "derived global maximum must be positive")
                .with_fact("reason", "zero"));
        }
        execution_config.validate()?;
        queue_bounds.validate()?;
        Ok(ValidatedGpuBackendConfig { devices, execution_config, queue_bounds, attempt_completion_queue_capacity })
    }
}

/// Configuration for [`GpuBackendCore::new`] with verified rules, resolved
/// installation bounds, and devices ordered by ordinal.
#[derive(Debug, Clone)]
pub struct ValidatedGpuBackendConfig {
    devices: Vec<ValidatedGpuBackendDevice>,
    execution_config: QueryExecutionConfig,
    queue_bounds: QueueBounds,
    attempt_completion_queue_capacity: Option<usize>,
}

impl ValidatedGpuBackendConfig {
    /// The validated device configs, ordered by ordinal.
    pub fn devices(&self) -> impl ExactSizeIterator<Item = &GpuBackendDeviceConfig> + '_ {
        self.devices.iter().map(|device| &device.config)
    }

    pub(crate) fn validated_devices(&self) -> &[ValidatedGpuBackendDevice] {
        &self.devices
    }

    pub(crate) fn device_ordinals(&self) -> Vec<usize> {
        self.devices().map(GpuBackendDeviceConfig::device_ordinal).collect()
    }

    pub(crate) fn execution_config(&self) -> &QueryExecutionConfig {
        &self.execution_config
    }

    pub(crate) fn queue_bounds(&self) -> &QueueBounds {
        &self.queue_bounds
    }
}

/// One validated device: its config, the installation bounds it is built
/// with, and its static partition in the native allocation-size type.
#[derive(Debug, Clone)]
pub(crate) struct ValidatedGpuBackendDevice {
    pub(crate) config: GpuBackendDeviceConfig,
    pub(crate) installation_bounds: GpuBackendDeviceInstallationBounds,
    pub(crate) allocation_limit_bytes: Option<usize>,
}

fn validate_device(config: GpuBackendDeviceConfig) -> Result<ValidatedGpuBackendDevice> {
    let installation_bounds = resolve_installation_bounds(&config.profile)?;
    let allocation_limit_bytes = validate_device_config(&config, installation_bounds)?;
    Ok(ValidatedGpuBackendDevice { config, installation_bounds, allocation_limit_bytes })
}

/// A profile without explicit installation bounds takes
/// [`GpuBackendDeviceProfile::derived_installation_bounds`]; explicit bounds
/// must cover its stream slots.
fn resolve_installation_bounds(profile: &GpuBackendDeviceProfile) -> Result<GpuBackendDeviceInstallationBounds> {
    let derived = profile.derived_installation_bounds()?;
    let Some(explicit) = profile.installation_bounds else {
        return Ok(derived);
    };
    if explicit.stream_pool_size < derived.stream_pool_size {
        return Err(invalid_device_config(
            profile.device_ordinal,
            "stream_pool_size",
            "must cover root and auxiliary source work for every active attempt",
        )
        .with_fact("reason", "stream_pool_below_required_formula")
        .with_fact("stream_pool_size", explicit.stream_pool_size)
        .with_fact("required_slot_capacity", derived.stream_pool_size)
        .with_fact("max_active_attempts", profile.max_active_attempts)
        .with_fact("source_work_per_attempt", profile.source_work_per_attempt));
    }
    Ok(explicit)
}

/// Checks one device's rules and returns its static partition in the native
/// allocation-size type.
fn validate_device_config(
    device: &GpuBackendDeviceConfig,
    bounds: GpuBackendDeviceInstallationBounds,
) -> Result<Option<usize>> {
    for (field, value) in [
        ("max_active_attempts", device.profile.max_active_attempts),
        ("source_work_per_attempt", device.profile.source_work_per_attempt),
        ("stream_pool_size", bounds.stream_pool_size),
        ("worker_count", bounds.worker_count),
        ("lane_count", bounds.lane_count),
    ] {
        if value == 0 {
            return Err(invalid_device_config(device.profile.device_ordinal, field, "must be greater than zero")
                .with_fact("reason", "zero"));
        }
    }
    if device.profile.default_query_limit_bytes == Some(0) {
        return Err(invalid_device_config(
            device.profile.device_ordinal,
            "default_query_limit_bytes",
            "must be greater than zero",
        )
        .with_fact("reason", "zero"));
    }
    // Every active attempt's query execution runs outside the reserved
    // lifecycle workers.
    let required_workers = device.profile.max_active_attempts.saturating_add(RESERVED_LIFECYCLE_WORKERS);
    if bounds.worker_count < required_workers {
        return Err(invalid_device_config(
            device.profile.device_ordinal,
            "worker_count",
            "must cover every configured active query slot plus the reserved lifecycle worker",
        )
        .with_fact("reason", "worker_count_below_active_limit_and_lifecycle_reserve")
        .with_fact("installation_bound", bounds.worker_count)
        .with_fact("max_active_attempts", device.profile.max_active_attempts)
        .with_fact("reserved_lifecycle_workers", RESERVED_LIFECYCLE_WORKERS));
    }
    for (field, value) in [("stream_pool_size", bounds.stream_pool_size), ("lane_count", bounds.lane_count)] {
        if value < device.profile.max_active_attempts {
            return Err(invalid_device_config(
                device.profile.device_ordinal,
                field,
                "must cover every configured active query slot",
            )
            .with_fact("reason", "installation_bound_below_active_limit")
            .with_fact("installation_bound", value)
            .with_fact("max_active_attempts", device.profile.max_active_attempts));
        }
    }
    match device.profile.memory_limit {
        GpuDeviceMemoryLimit::Automatic { fallback_bytes: 0, .. } => {
            return Err(invalid_device_config(
                device.profile.device_ordinal,
                "memory_limit.fallback_bytes",
                "must be greater than zero",
            )
            .with_fact("reason", "zero"));
        },
        GpuDeviceMemoryLimit::Automatic { fallback_bytes, .. }
            if device.profile.backend_reserve_bytes >= fallback_bytes =>
        {
            return Err(invalid_device_config(
                device.profile.device_ordinal,
                "backend_reserve_bytes",
                format!(
                    "is {}, at or above the automatic fallback capacity of {}; the largest accepted value is {}",
                    bytes_with_gib(device.profile.backend_reserve_bytes),
                    bytes_with_gib(fallback_bytes),
                    bytes_with_gib(fallback_bytes - 1),
                ),
            )
            .with_fact("reason", "reserve_exhausts_fallback")
            .with_fact("fallback_bytes", fallback_bytes));
        },
        GpuDeviceMemoryLimit::Explicit { limit_bytes: 0 } => {
            return Err(invalid_device_config(
                device.profile.device_ordinal,
                "memory_limit_bytes",
                "must be greater than zero",
            )
            .with_fact("reason", "zero"));
        },
        GpuDeviceMemoryLimit::Explicit { limit_bytes } if device.profile.backend_reserve_bytes >= limit_bytes => {
            return Err(invalid_device_config(
                device.profile.device_ordinal,
                "backend_reserve_bytes",
                format!(
                    "is {}, at or above memory_limit_bytes of {}; the largest accepted value is {}",
                    bytes_with_gib(device.profile.backend_reserve_bytes),
                    bytes_with_gib(limit_bytes),
                    bytes_with_gib(limit_bytes - 1),
                ),
            )
            .with_fact("reason", "reserve_exhausts_explicit_limit")
            .with_fact("limit_bytes", limit_bytes));
        },
        _ => {},
    }
    if let GpuMemoryOwnership::StaticPartition { allocation_limit_bytes } = device.memory_ownership {
        if allocation_limit_bytes == 0 {
            return Err(invalid_device_config(
                device.profile.device_ordinal,
                "memory_ownership.allocation_limit_bytes",
                "must be greater than zero",
            )
            .with_fact("reason", "zero"));
        }
        native_allocation_limit_bytes(device)?;
        if device.profile.backend_reserve_bytes >= allocation_limit_bytes {
            return Err(invalid_device_config(
                device.profile.device_ordinal,
                "backend_reserve_bytes",
                format!(
                    "is {}, at or above the static partition of {}; the largest accepted value is {}",
                    bytes_with_gib(device.profile.backend_reserve_bytes),
                    bytes_with_gib(allocation_limit_bytes),
                    bytes_with_gib(allocation_limit_bytes - 1),
                ),
            )
            .with_fact("reason", "reserve_exhausts_static_partition")
            .with_fact("allocation_limit_bytes", allocation_limit_bytes));
        }
        if let GpuDeviceMemoryLimit::Explicit { limit_bytes } = device.profile.memory_limit
            && limit_bytes > allocation_limit_bytes
        {
            return Err(invalid_device_config(
                device.profile.device_ordinal,
                "memory_limit_bytes",
                format!(
                    "is {}, above the static partition of {}; the largest accepted value is {}",
                    bytes_with_gib(limit_bytes),
                    bytes_with_gib(allocation_limit_bytes),
                    bytes_with_gib(allocation_limit_bytes),
                ),
            )
            .with_fact("reason", "limit_exceeds_static_partition")
            .with_fact("allocation_limit_bytes", allocation_limit_bytes));
        }
    }
    let allocation_limit_bytes = native_allocation_limit_bytes(device)?;
    DeviceMemoryManagerOptions { allocation_limit_bytes, ..Default::default() }.validate().map_err(|error| {
        cudf_query_error(BUILD_OPERATION, "invalid per-device memory manager installation options", error)
            .with_fact("field", "device_memory_manager_options")
            .with_fact("device_ordinal", device.profile.device_ordinal)
            .with_fact("reason", "invalid_device_memory_manager_options")
    })?;
    Ok(allocation_limit_bytes)
}

fn native_allocation_limit_bytes(device: &GpuBackendDeviceConfig) -> Result<Option<usize>> {
    device
        .memory_ownership
        .allocation_limit_bytes()
        .map(|limit| {
            usize::try_from(limit).map_err(|_error| {
                invalid_device_config(
                    device.profile.device_ordinal,
                    "memory_ownership.allocation_limit_bytes",
                    "does not fit the native allocation-size type",
                )
                .with_fact("reason", "out_of_range")
                .with_fact("allocation_limit_bytes", limit)
            })
        })
        .transpose()
}

fn invalid_device_config(device_ordinal: usize, field: &'static str, message: impl fmt::Display) -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidConfiguration,
        BUILD_OPERATION,
        format!("device {device_ordinal} {field} {message}"),
    )
    .with_fact("field", field)
    .with_fact("device_ordinal", device_ordinal)
}

fn invalid_config(field: &'static str, message: impl fmt::Display) -> QueryError {
    QueryError::new(QueryErrorCode::InvalidConfiguration, BUILD_OPERATION, format!("{field} {message}"))
        .with_fact("field", field)
}

/// Construction boundary exposed to deterministic probes and failure injectors.
///
/// Execution order differs from declaration order. `BeforeCleanupWorker` and
/// `AfterCleanupWorker` bracket the close coordinator spawn and come first,
/// because the coordinator is the rollback owner for everything after it. The
/// per-device checkpoints then run inside the service install in ordinal
/// order, followed by the startup-record pair and `Published`. A probe that
/// fails at any checkpoint after `AfterCleanupWorker` observes the resources
/// recorded so far in the construction journal, and the coordinator drains
/// them.
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
pub(crate) struct NoopConstructionProbe;

impl GpuBackendConstructionProbe for NoopConstructionProbe {
    fn checkpoint(&self, _checkpoint: GpuBackendConstructionCheckpoint) -> Result<()> {
        Ok(())
    }
}

impl GpuBackendCore {
    /// Claims the process owner of the configured devices, installs their
    /// GPU resources, and publishes the backend.
    ///
    /// Only one backend may be live per process; a second call while one is
    /// live, or while a closed one is still draining, fails with the
    /// `backend_already_live` reason. Construction blocks the calling thread
    /// for the CUDA context, memory manager, and stream pool installation of
    /// every device. On failure the close coordinator takes over every
    /// installed resource, and the error carries `rollback_cleanup` facts that
    /// say whether that cleanup reached quiescence within its bounded slice or
    /// was transferred to the coordinator's thread. In the latter case the
    /// owner lease is released asynchronously, so a retry may briefly observe
    /// `backend_already_live`.
    pub fn new(config: ValidatedGpuBackendConfig) -> Result<Self> {
        Self::construct(
            config,
            &NoopConstructionProbe,
            Arc::new(ConstructionResourceJournal::default()),
            ObservationCompletionExecutor::spawn,
        )
    }

    /// The construction transaction behind [`Self::new`].
    ///
    /// Ordering matters at three points. The owner lease is claimed before any
    /// GPU resource exists and moves into the resource generation immediately,
    /// so a panic or early return cannot drop it from this stack while
    /// installed resources are still live. The close coordinator is spawned
    /// before the first device install because from that point the
    /// coordinator is the rollback owner and this function is not.
    /// Process-exit registration happens after the service install because
    /// that install is what initializes CUDA, and the registration must
    /// follow libcudart's own `atexit` handler to run before it.
    fn construct(
        config: ValidatedGpuBackendConfig,
        probe: &dyn GpuBackendConstructionProbe,
        journal: Arc<ConstructionResourceJournal>,
        spawn_completion_executor: impl FnOnce() -> Result<ObservationCompletionExecutor>,
    ) -> Result<Self> {
        let ordinals = config.device_ordinals();
        let generation_id = next_backend_identity();
        let backend_metrics = BackendMetricsStore::new(generation_id, &ordinals)?;
        let owner_lease = claim_live_backend(&ordinals)?;

        // The generation shell and cleanup worker are the rollback journal for
        // every post-claim construction step. In particular, the live owner
        // lease is never held only by this builder's stack while GPU resources
        // are being installed.
        let generation = Arc::new(ResourceGeneration::new(generation_id, owner_lease, Arc::clone(&journal)));
        let resource_core = Arc::new(ResourceCore {
            identity: next_backend_identity(),
            generation: Mutex::new(Some(Arc::clone(&generation))),
        });
        let observation_core = Arc::new(ObservationCore {
            identity: next_backend_identity(),
            lifecycle: BackendLifecycle::new(),
            attempt_counters: Arc::new(AttemptBackendCounters::default()),
            attempt_completion_queue: config
                .attempt_completion_queue_capacity
                .and_then(AttemptCompletionQueue::new)
                .map(Arc::new),
            backend_metrics: backend_metrics.clone(),
            completion_executor: spawn_completion_executor()?,
        });

        probe.checkpoint(GpuBackendConstructionCheckpoint::BeforeCleanupWorker)?;
        let coordinator = CloseCoordinator::spawn(Arc::clone(&resource_core), Arc::clone(&observation_core))?;
        let mut rollback = ConstructionRollback::new(Arc::clone(&coordinator));

        let build_result = (|| {
            probe.checkpoint(GpuBackendConstructionCheckpoint::AfterCleanupWorker)?;
            let token = generation.owner_lease.token();
            let service = QueryService::install_for_backend(&config, &token, probe, &journal, &backend_metrics)?;
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
            // KvikIO's process-global pool is configured now, with CUDA
            // initialized and no source read in existence yet.
            let kvikio_process_config = configure_kvikio_process_defaults(&ordinals)?;

            let service =
                generation.service().expect("resource service is available immediately after successful installation");

            // The startup records are cross-checked against the validated profiles. They
            // are not copied from them. A device whose installed worker count or allocation
            // limit differs from its profile means the service installed something other
            // than what validation approved, and the backend fails closed without publishing.
            let devices = config.validated_devices();
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
            for device in devices {
                let profile = &device.config;
                let (facts, capacity_bytes, installation, health_identity, resource_profile) = observed
                    .iter()
                    .find(|(facts, _, _, _, _)| facts.device == profile.device_ordinal())
                    .cloned()
                    .ok_or_else(|| {
                        QueryError::new(
                            QueryErrorCode::Internal,
                            BUILD_OPERATION,
                            "validated device has no immutable startup record",
                        )
                        .with_fact("reason", "device_startup_record_missing")
                        .with_fact("device_ordinal", profile.device_ordinal())
                    })?;
                let worker_pool = worker_pools
                    .iter()
                    .find(|pool| pool.device_ordinal == profile.device_ordinal())
                    .ok_or_else(|| {
                        QueryError::new(
                            QueryErrorCode::Internal,
                            BUILD_OPERATION,
                            "validated device has no GPU worker pool",
                        )
                        .with_fact("reason", "device_worker_pool_missing")
                        .with_fact("device_ordinal", profile.device_ordinal())
                    })?;
                if worker_pool.worker_count != device.installation_bounds.worker_count {
                    return Err(QueryError::new(
                        QueryErrorCode::Internal,
                        BUILD_OPERATION,
                        "GPU worker pool does not match its validated device profile",
                    )
                    .with_fact("reason", "device_worker_pool_mismatch")
                    .with_fact("device_ordinal", profile.device_ordinal())
                    .with_fact("profile_worker_count", device.installation_bounds.worker_count)
                    .with_fact("installed_worker_count", worker_pool.worker_count));
                }
                let allocation_limit_bytes = device.allocation_limit_bytes;
                let installed_allocation_limit =
                    installation.effective_options.map(|options| options.allocation_limit_bytes);
                if installed_allocation_limit != Some(allocation_limit_bytes) {
                    return Err(QueryError::new(
                        QueryErrorCode::Internal,
                        BUILD_OPERATION,
                        "device memory manager record does not match its validated installation profile",
                    )
                    .with_fact("reason", "device_memory_manager_record_mismatch")
                    .with_fact("device_ordinal", profile.device_ordinal())
                    .with_fact("expected_allocation_limit_bytes", format!("{allocation_limit_bytes:?}"))
                    .with_fact("installed_options", format!("{:?}", installation.effective_options)));
                }
                startup_records.push(GpuBackendDeviceStartupRecord {
                    device_ordinal: facts.device,
                    model_name: facts.model_name.clone(),
                    uuid: facts.uuid.clone(),
                    physical_total_bytes: facts.physical_total_bytes,
                    facts,
                    capacity_bytes,
                    resource_profile,
                    memory_ownership: profile.memory_ownership,
                    memory_limit: profile.memory_limit(),
                    backend_reserve_bytes: profile.backend_reserve_bytes(),
                    max_active_attempts: profile.max_active_attempts(),
                    stream_pool_size: device.installation_bounds.stream_pool_size,
                    worker_count: worker_pool.worker_count,
                    lane_count: device.installation_bounds.lane_count,
                    install_telemetry: profile.install_telemetry(),
                    memory_manager_installation: installation,
                    device_health_generation: health_identity.device_health_generation,
                    backend_health_generation: health_identity.backend_health_generation,
                });
            }

            probe.checkpoint(GpuBackendConstructionCheckpoint::BeforeStartupRecords)?;
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
            probe.checkpoint(GpuBackendConstructionCheckpoint::AfterStartupRecords)?;
            // Public handles are counted by the tracker because `Arc` strong
            // counts also include sessions and the coordinator, whose
            // references must not keep the backend open.
            let close_on_last = Arc::clone(&coordinator);
            let public_handles = PublicHandleTracker::new(move || {
                close_on_last.signal();
            });
            let public_handle = public_handles.acquire().expect("new public-handle tracker accepts its first lease");
            probe.checkpoint(GpuBackendConstructionCheckpoint::Published)?;
            Ok(GpuBackendCore {
                resource_core,
                observation_core,
                coordinator,
                public_handle,
                startup_records,
                kvikio_process_config,
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

/// Applies [`KvikioProcessConfig::DEFAULT`] on the first owned device and
/// returns the effective settings. A cuDF build without KvikIO support leaves
/// the pool to libcudf and reports `None`.
fn configure_kvikio_process_defaults(ordinals: &[usize]) -> Result<Option<KvikioProcessConfig>> {
    let device_ordinal = *ordinals.first().expect("a validated backend config names at least one device");
    match configure_kvikio_process(device_ordinal, &KvikioProcessConfig::DEFAULT) {
        Ok(config) => Ok(Some(config)),
        Err(error) if error.code() == cudf::ErrorCode::UnsupportedOperation => Ok(None),
        Err(error) => Err(cudf_query_error(BUILD_OPERATION, "failed to configure KvikIO process defaults", error)
            .with_fact("device_ordinal", device_ordinal)),
    }
}

/// Immutable per-device record frozen before backend publication.
///
/// Each record combines the discovered device facts with the configuration
/// the device was installed under. It is read without any permit, so it stays
/// available after close for diagnostics about a generation that no longer
/// owns the device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuBackendDeviceStartupRecord {
    /// CUDA ordinal associated with this immutable installation record.
    pub device_ordinal: usize,
    /// NVML product name captured with the immutable device inventory facts.
    pub model_name: Option<String>,
    /// NVML UUID captured with the immutable device inventory facts.
    pub uuid: Option<String>,
    /// NVML-reported physical framebuffer capacity captured with the immutable
    /// device inventory facts.
    pub physical_total_bytes: Option<u64>,
    /// Device facts sampled during backend construction.
    pub facts: DeviceFacts,
    /// Managed capacity the ledger commits grants against: the resolved
    /// [`GpuDeviceMemoryLimit`] minus `backend_reserve_bytes`.
    pub capacity_bytes: u64,
    /// Full ordinal-local resource-controller profile frozen after discovery.
    pub resource_profile: DeviceResourceProfile,
    /// Deployment ownership promise used to resolve installed capacity.
    pub memory_ownership: GpuMemoryOwnership,
    /// Admission capacity policy used for this device installation.
    pub memory_limit: GpuDeviceMemoryLimit,
    /// Bytes excluded from managed query and cache capacity.
    pub backend_reserve_bytes: u64,
    /// Maximum simultaneous device-backed admitted attempts.
    pub max_active_attempts: usize,
    /// Installed CUDA stream-pool capacity.
    pub stream_pool_size: usize,
    /// Installed execution-worker thread count, including lifecycle workers.
    pub worker_count: usize,
    /// Installed device execution-lane count.
    pub lane_count: usize,
    /// Whether diagnostic telemetry was requested for this installation.
    pub install_telemetry: bool,
    /// Native memory-manager installation status recorded during construction.
    pub memory_manager_installation: DeviceMemoryManagerInstallationState,
    /// Persistent engine health epoch bound to this device installation.
    pub device_health_generation: u64,
    /// Backend-owner health epoch for this published core generation.
    pub backend_health_generation: u64,
}

/// Shared identity facts used to prove facade/session clones reuse one domain.
///
/// All three ids come from one process-wide counter, so two backends built in
/// the same process never share an id even after the first has closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuBackendCoreIdentity {
    pub resource_core: u64,
    pub observation_core: u64,
    pub resource_generation: u64,
}

/// One process-owned GPU backend core.
///
/// Every value of this type is a public handle. Clones mint a new
/// `PublicHandleLease`, and dropping the last lease signals close the same
/// way [`Self::close`] does. Dropping all handles therefore never leaks the device
/// owner, but neither does it wait for cleanup; callers that need to know when
/// the devices are reusable call one of the `close_and_wait*` methods first.
#[derive(Debug)]
pub struct GpuBackendCore {
    resource_core: Arc<ResourceCore>,
    observation_core: Arc<ObservationCore>,
    coordinator: Arc<CloseCoordinator>,
    /// The counted handle. Its tracker fires the close signal on the last
    /// drop; `Arc` counts on the cores are not consulted because sessions and
    /// the coordinator hold them too.
    public_handle: PublicHandleLease,
    startup_records: Arc<[GpuBackendDeviceStartupRecord]>,
    kvikio_process_config: Option<KvikioProcessConfig>,
    generation_id: u64,
}

/// Explicit capability for controlled fault validation on one live backend.
///
/// Constructing or holding this value does not alter backend behavior. Only
/// [`Self::quarantine_device`] changes state. It transitions the existing
/// per-device admission ledger and introduces no separate validation or
/// production scheduler mode.
#[cfg(feature = "qualification")]
#[derive(Clone, Debug)]
pub struct GpuBackendFaultInjector {
    session: GpuBackendSession,
}

#[cfg(feature = "qualification")]
impl GpuBackendFaultInjector {
    /// Closes admission on exactly one installed device ledger.
    ///
    /// This emulates the controller-visible quarantine result of a proven
    /// device-local native fault without fabricating an unsafe CUDA fault. The
    /// remaining healthy compatible ledgers remain eligible for placement.
    pub fn quarantine_device(&self, device_ordinal: usize) -> Result<DeviceQuarantineInjection> {
        self.session.with_service(|service| service.inject_device_quarantine(device_ordinal))
    }
}

impl Clone for GpuBackendCore {
    fn clone(&self) -> Self {
        Self {
            resource_core: Arc::clone(&self.resource_core),
            observation_core: Arc::clone(&self.observation_core),
            coordinator: Arc::clone(&self.coordinator),
            public_handle: self.public_handle.clone(),
            startup_records: Arc::clone(&self.startup_records),
            kvikio_process_config: self.kvikio_process_config,
            generation_id: self.generation_id,
        }
    }
}

impl GpuBackendCore {
    /// Mints the capability that session and optimizer state hold.
    ///
    /// A session is not a public handle: it holds the resource core weakly, so
    /// any number of live sessions neither delay the last-handle close signal
    /// nor keep the resource generation reachable after close has taken it.
    /// The observation core is held strongly because attempt reports and
    /// metrics must stay readable after the GPU resources are gone.
    pub fn session_capability(&self) -> GpuBackendSession {
        GpuBackendSession {
            resource_core: Arc::downgrade(&self.resource_core),
            observation_core: Arc::clone(&self.observation_core),
            identities: self.identity(),
        }
    }

    /// Returns an explicit capability for controlled device-quarantine
    /// validation. This is not a backend construction or production mode.
    #[cfg(feature = "qualification")]
    pub fn fault_injector(&self) -> GpuBackendFaultInjector {
        GpuBackendFaultInjector { session: self.session_capability() }
    }

    pub fn device_startup_records(&self) -> &[GpuBackendDeviceStartupRecord] {
        &self.startup_records
    }

    /// KvikIO's process settings as applied when this backend was built, or
    /// `None` for a cuDF build without KvikIO support.
    pub fn kvikio_process_config(&self) -> Option<KvikioProcessConfig> {
        self.kvikio_process_config
    }

    pub fn identity(&self) -> GpuBackendCoreIdentity {
        GpuBackendCoreIdentity {
            resource_core: self.resource_core.identity,
            observation_core: self.observation_core.identity,
            resource_generation: self.generation_id,
        }
    }

    /// See [`GpuBackendSession::begin_attempt`].
    pub fn begin_attempt(&self, request: BeginAttemptRequest) -> Result<AttemptBegin> {
        self.session_capability().begin_attempt(request)
    }

    /// Returns one coherent pull snapshot for this backend resource generation.
    ///
    /// The snapshot has a monotonic store generation so a consumer can detect
    /// whether it combined observations from different points in time.
    pub fn backend_metrics_snapshot(&self) -> BackendMetricsSnapshot {
        self.observation_core.backend_metrics.snapshot()
    }

    /// Signals close without waiting. Returns `true` only for the caller that
    /// won the lifecycle gate; later calls, and a last-handle drop after an
    /// explicit close, return `false` and change nothing.
    ///
    /// From the winning call on, every new admission, service read, and
    /// attempt begin fails with the `backend_closing` reason, while work that
    /// already holds a permit runs to completion. Cleanup proceeds on the
    /// coordinator's thread.
    pub fn close(&self) -> bool {
        self.coordinator.signal()
    }

    /// Point-in-time close progress. Readable at any time, including before
    /// close and from inside observation delivery.
    pub fn close_snapshot(&self) -> GpuBackendCloseSnapshot {
        self.coordinator.state.snapshot(&self.observation_core.lifecycle)
    }

    /// Signals close and blocks the calling thread until both quiescence
    /// scopes are reached or `timeout` elapses.
    ///
    /// A timeout returns `ServiceOverloaded` with reason `backend_close_timeout`
    /// and the quiescence facts at that moment; cleanup continues on the
    /// coordinator's thread and a later call can observe its completion. A
    /// cleanup failure returns `Internal` with reason `cleanup_failed_closed`;
    /// the owner lease is then retained, and the devices cannot be claimed
    /// again in this process. Calling from a runtime observation observer is
    /// rejected with `reentrant_observation_close` before close is signalled,
    /// because that observer is itself observation work the wait would need to
    /// finish.
    pub fn close_and_wait(&self, timeout: Duration) -> Result<GpuBackendCloseSnapshot> {
        reject_reentrant_observation_close()?;
        let deadline = Instant::now().checked_add(timeout).unwrap_or_else(Instant::now);
        self.close_and_wait_until(deadline)
    }

    /// [`Self::close_and_wait`] against an absolute deadline. Lock contention
    /// inside the wait consumes this same deadline.
    pub fn close_and_wait_until(&self, deadline: Instant) -> Result<GpuBackendCloseSnapshot> {
        reject_reentrant_observation_close()?;
        self.coordinator.close_and_wait_until(deadline)
    }

    /// [`Self::close_and_wait`] for callers on a Tokio runtime. The wait parks
    /// a thread, so it runs on the host runtime's blocking pool; the reentrancy
    /// check runs first, on the caller's task, so a rejected call never spawns.
    pub async fn close_and_wait_async(&self, timeout: Duration) -> Result<GpuBackendCloseSnapshot> {
        reject_reentrant_observation_close()?;
        let coordinator = Arc::clone(&self.coordinator);
        spawn_blocking(move || {
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
    /// An attempt that won the begin gate reads this flag and terminalizes
    /// `unplanned` with `backend_closing` evidence. Planning does not start in
    /// that case. The begin gate itself remains the only linearization point.
    pub fn is_closing(&self) -> bool {
        self.observation_core.lifecycle.is_closing()
    }

    /// Begins one attempt lifecycle under this backend's observation scope.
    ///
    /// The observation permit is the linearization point: an attempt that
    /// obtains it is counted in `attempts_begun` and holds the observation
    /// scope open until its report is ready, so close cannot complete without
    /// that report. A begin that loses to close fails with `backend_closing`
    /// and leaves no counted attempt. No GPU permit is taken here; the attempt
    /// claims a device only when its first native fragment is admitted.
    pub fn begin_attempt(&self, request: BeginAttemptRequest) -> Result<AttemptBegin> {
        let permit = self.observation_core.lifecycle.try_begin_observation()?;
        self.observation_core.attempt_counters.attempts_begun.fetch_add(1, Ordering::AcqRel);
        let begin = AttemptOwner::begin(
            request,
            permit,
            Arc::clone(&self.observation_core.attempt_counters),
            self.observation_core.attempt_completion_queue.clone(),
        );
        register_backend_attempt_metrics_sink(&begin.owner, self.observation_core.backend_metrics.clone());
        Ok(begin)
    }

    /// Returns the same coherent generation snapshot exposed by the owning
    /// backend core without requiring access to live GPU resources.
    pub fn backend_metrics_snapshot(&self) -> BackendMetricsSnapshot {
        self.observation_core.backend_metrics.snapshot()
    }

    /// Drains up to `max` reports of attempts whose completion receiver was
    /// dropped before the report was ready. Empty when the backend was built
    /// without an attempt completion queue.
    pub fn take_abandoned_attempt_reports(&self, max: usize) -> Vec<Arc<QueryAttemptReport>> {
        self.observation_core.attempt_completion_queue.as_ref().map_or_else(Vec::new, |queue| queue.drain(max))
    }

    pub fn attempt_counters_snapshot(&self) -> AttemptBackendCountersSnapshot {
        self.observation_core.attempt_counters.snapshot()
    }

    /// Takes the permits an admission or service read needs to enter the
    /// current resource generation.
    ///
    /// The order is load-bearing. The GPU permit is taken while the generation
    /// slot's mutex is held, and the close worker takes the slot under that
    /// same mutex only after setting the close bit. A caller that wins the GPU
    /// permit therefore always finds the generation still in the slot, and
    /// the close worker, which waits for the GPU permit count to reach zero
    /// before releasing the generation, can never observe a count of zero
    /// while a caller still holds a generation clone. The observation permit
    /// is taken first so a backend that is closing is rejected before the
    /// mutex is touched.
    fn acquire_resource(&self) -> Result<(ObservationPermit, GpuResourcePermit, ObservationCompletionDispatcher)> {
        let observation = self.observation_core.lifecycle.try_begin_observation()?;
        let resource_core = self.resource_core.upgrade().ok_or_else(backend_closing_resource_error)?;
        let generation = resource_core.generation.lock().map_err(|_error| resource_core_poisoned_error())?;
        let lifecycle = self.observation_core.lifecycle.try_begin_gpu_resource()?;
        let generation = generation.as_ref().cloned().ok_or_else(backend_closing_resource_error)?;
        Ok((
            observation,
            GpuResourcePermit { generation, _lifecycle: lifecycle },
            self.observation_core.completion_executor.dispatcher(),
        ))
    }

    /// Admits a query on the calling thread, which parks until the request is
    /// admitted or rejected. Production frontends use
    /// [`Self::admit_async_with_request`].
    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    pub fn admit_query_with_request(&self, request: QueryAdmissionRequest) -> Result<QueryHandle> {
        let (observation, resource, completion_dispatcher) = self.acquire_resource()?;
        let service = resource.generation.service().cloned().ok_or_else(resource_generation_unpublished_error)?;
        service.admit_query_with_backend_permits(request, observation, resource, completion_dispatcher)
    }

    /// Admits a fresh native attempt; see [`QueryAdmissionRequest::test_attempt`].
    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    pub fn admit_query(&self) -> Result<QueryHandle> {
        self.admit_query_with_request(QueryAdmissionRequest::test_attempt())
    }

    /// Admits a query only if it can be placed now; otherwise the request
    /// leaves the queue and this returns `None`.
    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    pub fn try_admit_query_with_request(&self, request: QueryAdmissionRequest) -> Result<Option<QueryHandle>> {
        let (observation, resource, completion_dispatcher) = self.acquire_resource()?;
        let service = resource.generation.service().cloned().ok_or_else(resource_generation_unpublished_error)?;
        service
            .enqueue_admission_with_backend_permits(request, observation, resource, completion_dispatcher)?
            .try_admit()
    }

    /// Admits one native attempt, waiting in the bounded admission queue
    /// until a grant is placed, the request is cancelled, or its queue
    /// deadline passes.
    ///
    /// The observation and GPU permits are taken before the request enters
    /// the queue and travel with it inside its `QueryCompletion`, so an
    /// attempt that is queued, admitted, or cleaning up holds the resource
    /// generation for its whole lifetime, and close waits for it. On a
    /// rejected or cancelled request the GPU permit is released before the
    /// observation is sealed and delivered, the observation permit covers that
    /// delivery and is released afterwards, and the opaque activity lease runs
    /// last because its destructor may reenter close. Failures before the
    /// queue carry `backend_closing`
    /// (`Cancelled`); failures inside the queue are the admission controller's
    /// (`ServiceOverloaded` for a full queue or an elapsed wait, the typed
    /// capability or health rejections otherwise).
    pub async fn admit_async_with_request(&self, request: QueryAdmissionRequest) -> Result<QueryHandle> {
        let (observation, resource, completion_dispatcher) = self.acquire_resource()?;
        let service = resource.generation.service().cloned().ok_or_else(resource_generation_unpublished_error)?;
        service.admit_async_with_backend_permits(request, observation, resource, completion_dispatcher).await
    }

    /// Whether both sessions were minted from the same published core.
    pub fn same_backend(&self, other: &Self) -> bool {
        self.identities == other.identities
    }

    /// Reads every service report once. Like every service read, it fails
    /// with `backend_closing` once close begins.
    pub fn diagnostics(&self) -> Result<BackendDiagnostics> {
        self.with_service(QueryService::diagnostics)
    }

    pub fn device_telemetry_snapshots(&self) -> Result<Vec<DeviceTelemetrySnapshot>> {
        self.with_service(|service| Ok(service.device_telemetry_snapshots()))
    }

    /// Runs a read against the live service while holding both permits, so
    /// the service and its device resources cannot be released mid-read.
    fn with_service<T>(&self, operation: impl FnOnce(&QueryService) -> Result<T>) -> Result<T> {
        let (_observation, resource, _completion_dispatcher) = self.acquire_resource()?;
        let service = resource.generation.service().ok_or_else(resource_generation_unpublished_error)?;
        operation(service)
    }
}

/// Projects each sealed attempt report into the backend metrics store. The
/// sink runs once per attempt at seal time; the store therefore stays a
/// projection of the observation snapshot and holds no events of its own.
fn register_backend_attempt_metrics_sink(owner: &AttemptOwner, backend_metrics: BackendMetricsStore) {
    owner.register_sealed_report_sink(move |report| {
        let projection = report.sealed_projection();
        backend_metrics.record_sealed_attempt(
            &projection,
            BackendAttemptDurations {
                total_ns: report.total_ns(),
                planning_ns: report.planning_ns(),
                queue_wait_ns: report.queue_wait_ns(),
                physical_execution_ns: report.physical_execution_ns(),
            },
        );
    });
}

/// Permit required by private service admission. Field order carries the
/// release order; see the field comment.
pub(crate) struct GpuResourcePermit {
    generation: Arc<ResourceGeneration>,
    // Declare this last. This keep-alive guard is never read, but its `Drop`
    // releases the GPU lifecycle count, and that release must happen only
    // after `generation` (declared above) so close waiters never observe a
    // zero count while a generation is still retained. Reordering the fields
    // breaks the invariant.
    _lifecycle: GpuLifecyclePermit,
}

impl fmt::Debug for GpuResourcePermit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GpuResourcePermit")
            .field("resource_generation", &self.generation.generation_id)
            .finish_non_exhaustive()
    }
}

/// The detachable slot that holds the current resource generation.
///
/// Held strongly by public handles and by the close worker, weakly by
/// sessions. The slot is written twice in a backend's life: filled at
/// construction and taken by the close worker after it has set the lifecycle
/// close bit. Because `acquire_resource` reads it under the same mutex while
/// taking its GPU permit, "closed" and "holds the generation" are mutually
/// exclusive for every caller.
#[derive(Debug)]
pub(crate) struct ResourceCore {
    identity: u64,
    pub(crate) generation: Mutex<Option<Arc<ResourceGeneration>>>,
}

/// State that outlives the GPU resources: the lifecycle gate, attempt
/// counters, the abandoned-report queue, metrics, and the observation
/// delivery executor.
///
/// Sessions hold this strongly so attempt begin and report delivery keep
/// working after GPU quiescence. The `lifecycle` clone is shared with the
/// close coordinator; both see the same close bit and permit counts.
#[derive(Debug)]
pub(crate) struct ObservationCore {
    identity: u64,
    pub(crate) lifecycle: BackendLifecycle,
    attempt_counters: Arc<AttemptBackendCounters>,
    attempt_completion_queue: Option<Arc<AttemptCompletionQueue>>,
    backend_metrics: BackendMetricsStore,
    completion_executor: ObservationCompletionExecutor,
}

/// Everything one backend generation owns on the devices, plus the lease
/// that makes it the process's live backend.
///
/// Created before any GPU resource exists and retained by every
/// `GpuResourcePermit`, so the service and journal cannot be dropped while an
/// admitted attempt or a service read still references them. The close worker
/// drops the last reference after the service has drained and the GPU permit
/// count has reached zero. A cleanup panic passes it to `std::mem::forget`.
/// Forgetting keeps the owner lease claimed, which prevents a replacement
/// backend from claiming devices with unknown outstanding GPU work.
#[derive(Debug)]
pub(crate) struct ResourceGeneration {
    /// Set once, after the service install succeeds. `None` only during the
    /// construction window, when no session can exist yet.
    service: OnceLock<QueryService>,
    startup_records: OnceLock<Arc<[GpuBackendDeviceStartupRecord]>>,
    generation_id: u64,
    /// Keeps every installed GPU resource alive through close so the drain
    /// can run against them after the service itself has been dropped.
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

    pub(crate) fn service(&self) -> Option<&QueryService> {
        self.service.get()
    }

    pub(crate) fn construction_journal(&self) -> &ConstructionResourceJournal {
        &self.construction_journal
    }
}

#[derive(Debug)]
enum ConstructionResource {
    MemoryManager(DeviceMemoryManager),
    StreamPool(super::ServiceStreamPool),
    DeviceContext(super::DeviceContext),
    ExecutionPool(super::GpuExecutionPool),
    StartupRecords(Arc<[GpuBackendDeviceStartupRecord]>),
}

/// Append-only record of every GPU resource created during construction, in
/// creation order.
///
/// The service install records each resource immediately after creating it
/// and before the next probe checkpoint, so a failure at any checkpoint leaves
/// a journal that names everything built so far. The entries are clones of
/// the handles the service also holds; the journal keeps the underlying
/// pools and contexts alive until the generation drops, so close can drain
/// them after the service is gone and construction rollback can drain them
/// when no service was ever published.
#[derive(Debug, Default)]
pub(crate) struct ConstructionResourceJournal {
    entries: Mutex<Vec<ConstructionResource>>,
}

impl ConstructionResourceJournal {
    pub(crate) fn record_memory_manager(&self, manager: DeviceMemoryManager) -> Result<()> {
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

    /// Closes and drains every journaled runtime in reverse creation order
    /// under one deadline.
    ///
    /// The call is idempotent. Pools that are already closed report
    /// quiescence at once, so the close worker calls this repeatedly in
    /// bounded slices. Every entry
    /// is visited even after a failure; a permanent error (anything other than
    /// an elapsed cleanup deadline) is returned in preference to a deadline
    /// error, so a slow drain cannot mask a broken one. Nothing is removed from
    /// the journal here; the resources are released when the generation drops.
    pub(crate) fn close_and_drain_until(&self, deadline: Instant) -> Result<()> {
        // Recording is one append, so a recovered guard holds every resource
        // recorded before a panic; each still needs its rollback.
        let entries = self.entries.lock().unwrap_or_else(PoisonError::into_inner);
        let mut first_deadline = None;
        let mut first_permanent = None;
        for entry in entries.iter().rev() {
            let result = match entry {
                ConstructionResource::ExecutionPool(pool) => pool.close_and_wait_until(deadline),
                ConstructionResource::StreamPool(pool) => {
                    pool.close_and_wait_until(deadline, "query_engine.gpu_backend_core.rollback")
                },
                // Memory managers, device contexts, and startup records have
                // no drain step. They are journaled for retention order only:
                // the runtimes above must stop before these are dropped with
                // the generation.
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

/// Process-wide identity counter shared by generation ids and both core ids,
/// so no two backends built in one process can present the same identity.
fn next_backend_identity() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Condvar, atomic::AtomicU8, mpsc},
        thread,
    };

    use tokio::runtime::Builder as TokioRuntimeBuilder;

    use super::*;
    use crate::{
        observability::{RuntimeObservationConfig, RuntimeObservationOwner, RuntimeTerminalOutcome},
        runtime::{
            BACKEND_METRIC_DEVICE_CAPACITY, GpuExecutionJobKind, GpuExecutionPool, QueryActivityLease,
            SourceSchedulerPolicy, observation_completion::ObservationCompletionWorkerLane,
        },
    };

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
                    GpuBackendCore::new(one_device_backend_config(0)).map_err(|error| error.to_string())?;
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
        let generation_id = next_backend_identity();
        let observation_core = Arc::new(ObservationCore {
            identity: next_backend_identity(),
            lifecycle: BackendLifecycle::new(),
            attempt_counters: Arc::new(AttemptBackendCounters::default()),
            attempt_completion_queue: None,
            backend_metrics: BackendMetricsStore::new(generation_id, &[])
                .expect("test backend metrics store accepts the inert generation"),
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
            kvikio_process_config: None,
            generation_id,
        }
    }

    #[test]
    fn close_recovers_a_poisoned_generation_slot() {
        let backend = inert_backend_for_observation_delivery_test();
        let resource_core = Arc::clone(&backend.resource_core);
        thread::spawn(move || {
            let _generation = resource_core.generation.lock().unwrap();
            panic!("poison the resource generation slot");
        })
        .join()
        .expect_err("the poisoning thread panics while holding the generation slot");

        let close = backend.close_and_wait(Duration::from_secs(2)).expect("close recovers the generation slot");

        assert!(close.gpu_resource_quiescent);
        assert!(close.observation_quiescent);
    }

    #[test]
    fn close_and_wait_reclaims_the_backend_cleanup_worker() {
        let backend = inert_backend_for_observation_delivery_test();
        let coordinator = Arc::clone(&backend.coordinator);

        let close = backend.close_and_wait(Duration::from_secs(2)).expect("inert backend closes");

        assert!(close.gpu_resource_quiescent);
        assert!(close.observation_quiescent);
        assert!(coordinator.worker_join_is_reclaimed_for_test(), "close must reclaim its cleanup worker handle");
    }

    #[test]
    fn engine_begin_records_one_sealed_projection_despite_a_panicking_peer_sink() {
        use crate::runtime::backend_metrics::{BackendAttemptDisposition, BackendAttemptOutcome};

        let backend = inert_backend_for_observation_delivery_test();
        let begin = backend.begin_attempt(BeginAttemptRequest::default()).expect("inert backend begins an attempt");
        begin.owner.register_sealed_report_sink(|_| panic!("injected peer sink panic"));

        begin.owner.commit_terminal(RuntimeTerminalOutcome::Succeeded).unwrap();
        let first = backend.backend_metrics_snapshot();
        assert_eq!(first.attempts.completed(BackendAttemptDisposition::Unplanned, BackendAttemptOutcome::Succeeded), 1,);

        // The owner drop is idempotent after report-ready. A peer sink panic
        // cannot duplicate or lose the engine-owned exact terminal update.
        drop(begin);
        let second = backend.backend_metrics_snapshot();
        assert_eq!(second.generation, first.generation);
        assert_eq!(
            second.attempts.completed(BackendAttemptDisposition::Unplanned, BackendAttemptOutcome::Succeeded),
            1,
        );

        let close = backend.close_and_wait(Duration::from_secs(2)).expect("inert backend closes");
        assert!(close.observation_quiescent);
    }

    #[test]
    fn process_exit_with_live_admitted_handle_does_not_hang() {
        const CHILD_ENV: &str = "ALGEON_TEST_PROCESS_EXIT_WITH_LIVE_ADMISSION";
        const TEST_NAME: &str = "runtime::backend_core::tests::process_exit_with_live_admitted_handle_does_not_hang";

        if std::env::var_os(CHILD_ENV).is_some() {
            let backend = GpuBackendCore::new(one_device_backend_config(0)).expect("child process builds a backend");
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
    fn observation_delivery_rejects_all_blocking_close_apis_without_signaling_close() {
        let backend = inert_backend_for_observation_delivery_test();
        let (sender, receiver) = mpsc::sync_channel(1);
        let observer_backend = backend.clone();
        let owner = RuntimeObservationOwner::new(
            "reentrant-observation-close",
            None,
            RuntimeObservationConfig::bounded(8, 4 * 1024),
        );
        owner.register_seal_observer(move |_| {
            let sync = observer_backend.close_and_wait(Duration::from_secs(1));
            let until = observer_backend.close_and_wait_until(Instant::now() + Duration::from_secs(1));
            let asynchronous = TokioRuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("test builds current-thread runtime")
                .block_on(observer_backend.close_and_wait_async(Duration::from_secs(1)));
            sender.send((sync, until, asynchronous)).expect("test receives close rejections");
        });

        owner.finish_query(Some(RuntimeTerminalOutcome::Succeeded)).expect("test seals observation owner");
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

    fn backend_config(devices: Vec<GpuBackendDeviceConfig>) -> GpuBackendConfig {
        GpuBackendConfig::new(devices, QueryExecutionConfig::default(), QueueBounds::default())
    }

    fn one_device_backend(device_ordinal: usize) -> GpuBackendConfig {
        backend_config(vec![GpuBackendDeviceConfig::new(
            GpuBackendDeviceProfile::new(device_ordinal),
            GpuMemoryOwnership::WholeDeviceExclusive,
        )])
    }

    fn one_device_backend_config(device_ordinal: usize) -> ValidatedGpuBackendConfig {
        one_device_backend(device_ordinal).validate().expect("one-device backend config validates")
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
        let config = backend_config(vec![GpuBackendDeviceConfig::new(
            GpuBackendDeviceProfile::new(0).with_max_active_attempts(0),
            GpuMemoryOwnership::WholeDeviceExclusive,
        )]);
        let error = config.validate().unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::InvalidConfiguration);
        assert_eq!(error.message(), "device 0 max_active_attempts must be greater than zero");
        assert_eq!(error.fact_value("field"), Some("max_active_attempts"));
        assert_eq!(error.fact_value("reason"), Some("zero"));
    }

    #[test]
    fn pure_config_rejects_explicit_zero_default_query_limit_before_construction() {
        let config = backend_config(vec![GpuBackendDeviceConfig::new(
            GpuBackendDeviceProfile::new(0).with_default_query_limit_bytes(0),
            GpuMemoryOwnership::WholeDeviceExclusive,
        )]);

        let error = config.validate().unwrap_err();

        assert_eq!(error.fact_value("field"), Some("default_query_limit_bytes"));
        assert_eq!(error.fact_value("reason"), Some("zero"));
    }

    #[test]
    fn observation_completion_executor_spawn_failure_never_publishes_or_retains_live_owner() {
        let _owner_test = live_backend_test_lock();
        // This fails before device installation, so use a bounded synthetic
        // ordinal that the fixed-cardinality backend metrics store accepts.
        let device_ordinal = BACKEND_METRIC_DEVICE_CAPACITY - 3;
        for failing_lane in
            [ObservationCompletionWorkerLane::Observation, ObservationCompletionWorkerLane::ActivityTail]
        {
            let published = Arc::new(PublishedCheckpointProbe::default());

            let error = GpuBackendCore::construct(
                one_device_backend_config(device_ordinal),
                published.as_ref(),
                Arc::default(),
                || ObservationCompletionExecutor::spawn_with_lane_failure_for_test(failing_lane),
            )
            .unwrap_err();

            assert_eq!(error.fact_value("reason"), Some("observation_completion_worker_spawn_failed"));
            assert_eq!(error.fact_value("worker_lane"), Some(failing_lane.as_str()));
            assert_eq!(published.0.load(Ordering::Acquire), 0, "executor spawn failure must precede publication");
            wait_for_live_owner_release(device_ordinal);
        }
    }

    #[test]
    fn pure_config_rejects_a_reserve_that_exhausts_the_fallback_capacity() {
        let config = backend_config(vec![GpuBackendDeviceConfig::new(
            GpuBackendDeviceProfile::new(0).with_backend_reserve_bytes(256 * 1024 * 1024),
            GpuMemoryOwnership::WholeDeviceExclusive,
        )]);

        let error = config.validate().unwrap_err();

        assert_eq!(error.code(), QueryErrorCode::InvalidConfiguration);
        assert_eq!(error.fact_value("reason"), Some("reserve_exhausts_fallback"));
        assert_eq!(
            error.message(),
            "device 0 backend_reserve_bytes is 268435456 bytes (0.25 GiB), at or above the automatic fallback capacity of 268435456 bytes (0.25 GiB); the largest accepted value is 268435455 bytes (0.24 GiB)"
        );
    }

    #[test]
    fn pure_config_rejects_an_explicit_limit_above_the_static_partition_with_the_largest_accepted_value() {
        const GIB: u64 = 1 << 30;
        let config = backend_config(vec![GpuBackendDeviceConfig::new(
            GpuBackendDeviceProfile::new(0)
                .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 100 * GIB }),
            GpuMemoryOwnership::StaticPartition { allocation_limit_bytes: 95 * GIB },
        )]);

        let error = config.validate().unwrap_err();

        assert_eq!(error.code(), QueryErrorCode::InvalidConfiguration);
        assert_eq!(error.fact_value("reason"), Some("limit_exceeds_static_partition"));
        assert_eq!(
            error.message(),
            "device 0 memory_limit_bytes is 107374182400 bytes (100.00 GiB), above the static partition of 102005473280 bytes (95.00 GiB); the largest accepted value is 102005473280 bytes (95.00 GiB)"
        );
    }

    #[test]
    fn automatic_installation_bounds_cover_every_active_attempt() {
        let config = backend_config(vec![GpuBackendDeviceConfig::new(
            GpuBackendDeviceProfile::new(0).with_max_active_attempts(2),
            GpuMemoryOwnership::WholeDeviceExclusive,
        )])
        .validate()
        .expect("derived multi-attempt installation profile validates");

        let bounds = config.validated_devices()[0].installation_bounds;
        assert_eq!(bounds.lane_count, 2);
        assert_eq!(bounds.worker_count, 3);
        assert!(bounds.stream_pool_size >= 2);
    }

    #[test]
    fn automatic_bounds_keep_device_source_work_independent_of_execution_prefetch() {
        let execution = QueryExecutionConfig {
            source_scheduler_policy: SourceSchedulerPolicy::new(7),
            ..QueryExecutionConfig::default()
        };
        let config = GpuBackendConfig::new(
            vec![GpuBackendDeviceConfig::new(
                GpuBackendDeviceProfile::new(0).with_max_active_attempts(2),
                GpuMemoryOwnership::WholeDeviceExclusive,
            )],
            execution,
            QueueBounds::default(),
        )
        .validate()
        .expect("device-local default remains valid");

        let device = &config.validated_devices()[0];
        assert_eq!(device.config.source_work_per_attempt(), 1);
        assert_eq!(device.installation_bounds.stream_pool_size, 4);
    }

    fn assert_query_activity_allows_close_and_replacement_after_release(acquire_context: bool) {
        let _owner_test = live_backend_test_lock();
        let backend = GpuBackendCore::new(one_device_backend_config(0)).unwrap();
        let lifecycle = backend.observation_core.lifecycle.clone();
        let observed_active_gpu_permits = Arc::new(std::sync::atomic::AtomicUsize::new(usize::MAX));
        let (completion, completed) = mpsc::sync_channel(1);
        let activity = QueryActivityLease::new(ActivityPermitOrderingProbe {
            lifecycle: lifecycle.clone(),
            backend: backend.clone(),
            observed_active_gpu_permits: Arc::clone(&observed_active_gpu_permits),
            completion,
        });
        let attempt = AttemptOwner::native_for_test();
        let request =
            QueryAdmissionRequest::new(attempt.admission_binding()).with_runtime_observation_activity(activity);
        let handle = backend.session_capability().admit_query_with_request(request).unwrap();
        assert_eq!(lifecycle.snapshot().active_gpu_permits, 1);
        if acquire_context {
            let context = handle.acquire_context().expect("query acquires a service-stream context");
            drop(context);
        }

        // Ending the attempt releases its grant; the handle releases nothing.
        drop(attempt);
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

    fn query_activity_returns_only_after_service_resources_and_backend_gpu_permit_release_case() {
        assert_query_activity_allows_close_and_replacement_after_release(false);
    }

    #[test]
    fn query_activity_returns_only_after_service_resources_and_backend_gpu_permit_release() {
        query_activity_returns_only_after_service_resources_and_backend_gpu_permit_release_case();
        context_cleanup_completes_before_query_activity_can_close_and_replace_backend();
    }

    fn context_cleanup_completes_before_query_activity_can_close_and_replace_backend() {
        assert_query_activity_allows_close_and_replacement_after_release(true);
    }

    #[test]
    fn backend_observation_quiescence_waits_for_detached_seal_observers() {
        let _owner_test = live_backend_test_lock();
        let backend = GpuBackendCore::new(one_device_backend_config(0)).unwrap();
        let attempt = AttemptOwner::native_for_test();
        let binding = attempt.admission_binding();
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let observer_gate = Arc::clone(&gate);
        let (entered_sender, entered_receiver) = mpsc::sync_channel(1);
        binding.register_seal_observer(move |_snapshot| {
            let _ = entered_sender.send(());
            let (lock, changed) = &*observer_gate;
            let mut released = lock.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            while !*released {
                released = changed.wait(released).unwrap_or_else(std::sync::PoisonError::into_inner);
            }
        });
        let handle =
            backend.session_capability().admit_query_with_request(QueryAdmissionRequest::new(binding)).unwrap();
        drop(handle.acquire_context().expect("query acquires a service-stream context"));
        drop(attempt);
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
        let backend = GpuBackendCore::new(
            backend_config(vec![GpuBackendDeviceConfig::new(
                GpuBackendDeviceProfile::new(0)
                    .with_default_query_limit_bytes(256 * 1024 * 1024)
                    .with_max_active_attempts(2)
                    .with_installation_bounds(GpuBackendDeviceInstallationBounds::new(6, 3, 2)),
                GpuMemoryOwnership::WholeDeviceExclusive,
            )])
            .validate()
            .expect("explicit-bounds backend config validates"),
        )
        .unwrap();
        let (entered_sender, entered_receiver) = mpsc::sync_channel(1);
        let (completion_sender, completion_receiver) = mpsc::sync_channel(1);
        let first_attempt = AttemptOwner::native_for_test();
        let first = backend
            .session_capability()
            .admit_query_with_request(
                QueryAdmissionRequest::new(first_attempt.admission_binding()).with_runtime_observation_activity(
                    QueryActivityLease::new(ReentrantCloseActivityProbe {
                        backend: backend.clone(),
                        entered: entered_sender,
                        completion: completion_sender,
                    }),
                ),
            )
            .unwrap();
        let second =
            backend.session_capability().admit_query_with_request(QueryAdmissionRequest::test_attempt()).unwrap();
        drop(first.acquire_context().expect("first query acquires a service-stream context"));
        drop(second.acquire_context().expect("second query acquires a service-stream context"));

        drop(first_attempt);
        drop(first);
        entered_receiver
            .recv_timeout(Duration::from_secs(10))
            .expect("first activity begins its reentrant backend close");

        second.end_test_attempt();
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
        // The probe stops before device installation; keep the ordinal inside
        // the backend metrics store's frozen device-label domain.
        let ordinal = BACKEND_METRIC_DEVICE_CAPACITY - 1;
        let error = GpuBackendCore::construct(
            one_device_backend_config(ordinal),
            &FailingProbe(GpuBackendConstructionCheckpoint::AfterCleanupWorker),
            Arc::default(),
            ObservationCompletionExecutor::spawn,
        )
        .unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("injected_construction_failure"));
        assert!(matches!(error.fact_value("rollback_cleanup"), Some("quiescent" | "transferred")));

        wait_for_live_owner_release(ordinal);
    }

    #[test]
    fn every_constructed_resource_is_journaled_before_its_after_boundary() {
        let _owner_test = live_backend_test_lock();

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
            let error = GpuBackendCore::construct(
                one_device_backend_config(0),
                probe.as_ref(),
                journal,
                ObservationCompletionExecutor::spawn,
            )
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
        let pool = GpuExecutionPool::new(ordinal, 2).unwrap();
        journal.record_execution_pool(pool.clone()).unwrap();
        let gate = Arc::new(BlockingGate::default());
        let worker_gate = Arc::clone(&gate);
        let (started_sender, started_receiver) = mpsc::sync_channel(0);
        pool.spawn(GpuExecutionJobKind::Cleanup, async move {
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
            backend_metrics: BackendMetricsStore::new(next_backend_identity(), &[])
                .expect("test backend metrics store accepts an empty device set"),
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
    fn dropping_last_public_handle_retains_owner_through_timeout_then_recovers_after_late_cleanup() {
        let _owner_test = live_backend_test_lock();
        let ordinal = usize::MAX - 3;
        let owner_lease = claim_live_backend(&[ordinal]).expect("test claims the live owner");
        let journal = Arc::new(ConstructionResourceJournal::default());
        let pool = GpuExecutionPool::new(ordinal, 2).expect("test execution pool");
        journal.record_execution_pool(pool.clone()).expect("journal records the execution pool");
        let gate = Arc::new(BlockingGate::default());
        let worker_gate = Arc::clone(&gate);
        let (started_sender, started_receiver) = mpsc::sync_channel(0);
        pool.spawn(GpuExecutionJobKind::Cleanup, async move {
            started_sender.send(()).expect("cleanup job start receiver remains live");
            worker_gate.wait();
        })
        .expect("cleanup job submits");
        started_receiver.recv().expect("cleanup job starts");

        let generation = Arc::new(ResourceGeneration::new(next_backend_identity(), owner_lease, Arc::clone(&journal)));
        let resource_core =
            Arc::new(ResourceCore { identity: next_backend_identity(), generation: Mutex::new(Some(generation)) });
        let observation_core = Arc::new(ObservationCore {
            identity: next_backend_identity(),
            lifecycle: BackendLifecycle::new(),
            attempt_counters: Arc::new(AttemptBackendCounters::default()),
            attempt_completion_queue: None,
            backend_metrics: BackendMetricsStore::new(next_backend_identity(), &[])
                .expect("test backend metrics store accepts an empty device set"),
            completion_executor: ObservationCompletionExecutor::spawn().expect("test observation executor"),
        });
        let coordinator = CloseCoordinator::spawn(Arc::clone(&resource_core), Arc::clone(&observation_core))
            .expect("cleanup coordinator");
        let close_on_last = Arc::clone(&coordinator);
        let public_handles = PublicHandleTracker::new(move || {
            close_on_last.signal();
        });
        let backend = GpuBackendCore {
            resource_core,
            observation_core: Arc::clone(&observation_core),
            coordinator: Arc::clone(&coordinator),
            public_handle: public_handles.acquire().expect("first public backend handle"),
            startup_records: Arc::from([]),
            kvikio_process_config: None,
            generation_id: next_backend_identity(),
        };

        drop(backend);
        let timeout = coordinator
            .state
            .wait_until(Instant::now() + Duration::from_millis(250), &observation_core.lifecycle)
            .expect_err("last-handle close cannot release a live execution runtime");
        assert_eq!(timeout.fact_value("reason"), Some("backend_close_timeout"));
        assert_eq!(timeout.fact_value("gpu_resource_quiescent"), Some("false"));
        assert!(coordinator.state.snapshot(&observation_core.lifecycle).closing, "last handle signals the close gate");
        let replacement =
            claim_live_backend(&[ordinal]).expect_err("late cleanup retains the device owner after last-handle drop");
        assert_eq!(replacement.fact_value("reason"), Some("backend_already_live"));

        gate.release();
        let snapshot = coordinator
            .state
            .wait_until(Instant::now() + Duration::from_secs(5), &observation_core.lifecycle)
            .expect("late cleanup eventually quiesces both scopes");
        assert!(snapshot.gpu_resource_quiescent);
        assert!(snapshot.observation_quiescent);
        assert!(snapshot.cleanup_late_success_count > 0, "the timed-out cleanup phase is recorded as a late success");
        pool.join_supervisor().expect("cleanup joins the execution supervisor before owner release");
        let replacement =
            claim_live_backend(&[ordinal]).expect("owner lease becomes reusable only after late cleanup quiesces");
        drop(replacement);
    }

    #[test]
    fn begin_attempt_after_close_is_rejected_without_creating_an_attempt() {
        let backend = inert_backend_for_observation_delivery_test();
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
        let backend = inert_backend_for_observation_delivery_test();
        let session = backend.session_capability();
        let begin = session.begin_attempt(BeginAttemptRequest::default()).unwrap();
        assert_eq!(session.attempt_counters_snapshot().attempts_begun, 1);

        assert!(backend.close());
        let snapshot = backend.close_snapshot();
        assert!(!snapshot.observation_quiescent, "an open attempt holds observation scope open");

        // A CPU-only attempt terminalizes without GPU cleanup and publishes
        // its report; only then may observation quiescence complete.
        begin.owner.commit_terminal(RuntimeTerminalOutcome::Succeeded).unwrap();
        let report = begin
            .completion
            .completion()
            .wait_timeout(Duration::from_secs(5))
            .expect("the attempt publishes its report during close");
        assert!(matches!(report.outcome, RuntimeTerminalOutcome::Succeeded));
        drop(begin);
        assert_eq!(session.attempt_counters_snapshot().reports_ready, 1);
        let close = backend.close_and_wait(Duration::from_secs(10)).unwrap();
        assert!(close.observation_quiescent);
    }

    #[test]
    fn close_after_report_ready_does_not_wait_for_a_held_completion_receiver() {
        let backend = inert_backend_for_observation_delivery_test();
        let begin = backend.begin_attempt(BeginAttemptRequest::default()).expect("attempt begins before close");

        begin.owner.commit_terminal(RuntimeTerminalOutcome::Succeeded).unwrap();
        let completion = begin.completion.completion();
        let report = completion.wait_timeout(Duration::from_secs(2)).expect("report is ready before close");
        assert!(matches!(report.outcome, RuntimeTerminalOutcome::Succeeded));

        assert!(backend.close());
        let close = backend
            .close_and_wait(Duration::from_secs(2))
            .expect("reported attempt no longer holds close observation scope");
        assert!(close.gpu_resource_quiescent);
        assert!(close.observation_quiescent);
        assert!(completion.is_ready(), "the caller may retain its completed result after backend close");
    }

    #[test]
    fn concurrent_begin_and_close_linearize_without_an_unaccounted_attempt() {
        const RACE_ITERATIONS: usize = 16;

        for _ in 0..RACE_ITERATIONS {
            let backend = inert_backend_for_observation_delivery_test();
            let session = backend.session_capability();
            let start = Arc::new(std::sync::Barrier::new(3));

            let begin_start = Arc::clone(&start);
            let begin_session = session.clone();
            let begin_worker = thread::spawn(move || {
                begin_start.wait();
                begin_session.begin_attempt(BeginAttemptRequest::default())
            });

            let close_start = Arc::clone(&start);
            let close_backend = backend.clone();
            let close_worker = thread::spawn(move || {
                close_start.wait();
                close_backend.close()
            });

            start.wait();
            let begin = begin_worker.join().expect("begin worker does not panic");
            assert!(close_worker.join().expect("close worker does not panic"), "the sole close signal wins");

            let attempts_begun = match begin {
                Ok(begin) => {
                    begin.owner.commit_terminal(RuntimeTerminalOutcome::Succeeded).unwrap();
                    let report = begin
                        .completion
                        .completion()
                        .wait_timeout(Duration::from_secs(2))
                        .expect("winning begin publishes");
                    assert!(matches!(report.outcome, RuntimeTerminalOutcome::Succeeded));
                    1
                },
                Err(error) => {
                    assert_eq!(error.code(), QueryErrorCode::Cancelled);
                    assert_eq!(error.fact_value("reason"), Some("backend_closing"));
                    0
                },
            };

            let counters = session.attempt_counters_snapshot();
            assert_eq!(counters.attempts_begun, attempts_begun, "only a begin-gate winner is counted");
            assert_eq!(counters.reports_ready, attempts_begun, "every counted begin publishes before close completes");
            let error =
                session.begin_attempt(BeginAttemptRequest::default()).expect_err("close excludes every later begin");
            assert_eq!(error.fact_value("reason"), Some("backend_closing"));

            let close = backend.close_and_wait(Duration::from_secs(2)).expect("linearized attempt state drains");
            assert!(close.gpu_resource_quiescent);
            assert!(close.observation_quiescent);
        }
    }

    #[test]
    fn abandoned_attempt_reports_land_in_the_bounded_completion_queue() {
        let _owner_test = live_backend_test_lock();
        let config = one_device_backend(0).with_attempt_completion_queue_capacity(2).validate().unwrap();
        let backend = GpuBackendCore::new(config).unwrap();
        let session = backend.session_capability();

        let begin = session.begin_attempt(BeginAttemptRequest::default()).unwrap();
        drop(begin.completion);
        begin.owner.commit_terminal(RuntimeTerminalOutcome::Dropped).unwrap();
        drop(begin.owner);

        let drained = session.take_abandoned_attempt_reports(8);
        assert_eq!(drained.len(), 1);
        assert!(matches!(drained[0].outcome, RuntimeTerminalOutcome::Dropped));
        let counters = session.attempt_counters_snapshot();
        assert_eq!(counters.completion_queue_pushed, 1);
        assert_eq!(counters.completion_queue_overflowed, 0);
        backend.close_and_wait(Duration::from_secs(10)).unwrap();
    }

    fn probe_facts(total_bytes: u64) -> DeviceFacts {
        DeviceFacts {
            device: 3,
            model_name: None,
            uuid: None,
            physical_total_bytes: Some(1_100),
            total_bytes,
            free_bytes: 500,
            sm_count: None,
            utilization_pct: None,
        }
    }

    #[test]
    fn probe_capacities_resolve_owned_and_managed_bounds() {
        let facts = probe_facts(1_000);
        for (limit, reserve, expected) in [
            (None, 0, (1_000, 1_000)),
            (None, 100, (1_000, 900)),
            (Some(600), 100, (600, 500)),
            (Some(1_000), 0, (1_000, 1_000)),
        ] {
            assert_eq!(resolve_probe_capacities(3, &facts, limit, reserve).expect("probe math resolves"), expected);
        }
    }

    #[test]
    fn probe_capacities_reject_partition_above_cuda_total() {
        let facts = probe_facts(1_000);
        let error = resolve_probe_capacities(3, &facts, Some(1_001), 0).unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("static_partition_exceeds_physical_capacity"));
        assert_eq!(error.fact_value("cuda_total_bytes"), Some("1000"));
        assert_eq!(error.fact_value("physical_total_bytes"), Some("1100"));
    }

    #[test]
    fn probe_capacities_reject_exhausting_reserve() {
        let facts = probe_facts(1_000);
        for reserve in [1_000, 1_001] {
            let error = resolve_probe_capacities(3, &facts, None, reserve).unwrap_err();
            assert_eq!(error.fact_value("reason"), Some("backend_reserve_exhausts_capacity"));
        }
    }

    #[test]
    fn derived_installation_bounds_clear_explicit_bounds() {
        let profile = GpuBackendDeviceProfile::new(0)
            .with_max_active_attempts(2)
            .with_installation_bounds(GpuBackendDeviceInstallationBounds::new(99, 99, 99))
            .with_derived_installation_bounds(4);
        assert_eq!(profile.max_active_attempts(), 4);
        assert_eq!(profile.installation_bounds(), None);
    }

    #[test]
    fn homogeneous_device_configs_rebind_template_ordinals() {
        let template = GpuBackendDeviceProfile::new(7)
            .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 512 })
            .with_max_active_attempts(3);
        let devices = GpuBackendDeviceConfig::homogeneous([0, 2], &template, GpuMemoryOwnership::WholeDeviceExclusive);
        assert_eq!(devices.len(), 2);
        assert_eq!(devices[0].device_ordinal(), 0);
        assert_eq!(devices[1].device_ordinal(), 2);
        for device in &devices {
            assert_eq!(device.memory_limit(), GpuDeviceMemoryLimit::Explicit { limit_bytes: 512 });
            assert_eq!(device.max_active_attempts(), 3);
            assert_eq!(device.memory_ownership(), GpuMemoryOwnership::WholeDeviceExclusive);
        }
    }

    #[test]
    fn probe_device_capacity_reports_live_cuda_totals() {
        let probe = probe_device_capacity(0, GpuMemoryOwnership::WholeDeviceExclusive, 0)
            .expect("GPU lane provides device 0 for capacity probing");
        assert_eq!(probe.device_ordinal, 0);
        assert!(probe.cuda_total_bytes > 0);
        assert!(probe.cuda_free_bytes <= probe.cuda_total_bytes);
        assert_eq!(probe.owned_capacity_bytes, probe.cuda_total_bytes);
        assert_eq!(probe.max_declarable_limit_bytes, probe.cuda_total_bytes);
        assert_eq!(probe.max_managed_capacity_bytes, probe.cuda_total_bytes);
    }
}
