//! Process-level GPU device registry for service execution.

use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

use cudf::{
    DeviceId, DeviceMemoryManager, DeviceMemoryManagerBase, DeviceMemoryManagerInstallationCause,
    DeviceMemoryManagerInstallationIdentity, DeviceMemoryManagerInstallationRetry,
    DeviceMemoryManagerInstallationStatus, InstallSpec,
    runtime::{DeviceHealthStatus, device_health_snapshot},
};
use query_planning::error::{QueryError, QueryErrorCode, Result, bytes_with_gib};

use super::{
    DeviceCapabilities, DeviceContext, DeviceLedger, DeviceMemoryFailureReport, DeviceMemoryFailureTracker,
    DeviceOwnerToken, DeviceResourceEnvelope, DeviceResourceProfile, GpuInventory,
};
use crate::{
    native_error::cudf_query_error,
    runtime::{
        GpuBackendConstructionCheckpoint, GpuBackendConstructionProbe, GpuMemoryFraction, ServiceStreamPool,
        backend_core::ConstructionResourceJournal, backend_metrics::BackendMetricsStore,
    },
};

/// Soft managed-capacity calculation for one installed device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeviceRegistryCapacityPolicy {
    Automatic { headroom_bytes: u64, fallback_bytes: u64, fraction: Option<GpuMemoryFraction> },
    Explicit { limit_bytes: u64 },
}

/// Physical allocation ownership represented by the process manager.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeviceRegistryMemoryPartition {
    WholeDevice,
    StaticPartition { allocation_limit_bytes: usize },
}

impl DeviceRegistryMemoryPartition {
    fn allocation_limit_bytes(self) -> Option<usize> {
        match self {
            Self::WholeDevice => None,
            Self::StaticPartition { allocation_limit_bytes } => Some(allocation_limit_bytes),
        }
    }
}

/// Fully resolved, independent installation profile for one device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DeviceRegistryInstallProfile {
    pub(crate) stream_pool_size: usize,
    /// Immutable native capabilities consumed before placement and queueing.
    pub(crate) device_capabilities: DeviceCapabilities,
    pub(crate) max_active_attempts: usize,
    pub(crate) capacity_policy: DeviceRegistryCapacityPolicy,
    pub(crate) backend_reserve_bytes: u64,
    pub(crate) cache_cap_bytes: u64,
    pub(crate) default_query_limit_bytes: Option<u64>,
    pub(crate) source_work_per_attempt: usize,
    pub(crate) worker_job_capacity: usize,
    pub(crate) lane_count: usize,
    /// Whether public diagnostics may actively sample CUDA/NVML telemetry for
    /// this ordinal. This never participates in admission or placement.
    pub(crate) install_telemetry: bool,
    pub(crate) memory_partition: DeviceRegistryMemoryPartition,
}

impl DeviceRegistryInstallProfile {
    fn cache_cap_bytes(&self) -> u64 {
        self.cache_cap_bytes
    }
}

/// Configuration used when installing a device registry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DeviceRegistryConfig {
    /// CUDA ordinals enabled for the service: non-empty, unique, and
    /// ascending, as `GpuBackendConfig::validate` leaves them.
    pub(crate) device_ordinals: Vec<usize>,
    device_profiles: HashMap<usize, DeviceRegistryInstallProfile>,
}

impl DeviceRegistryConfig {
    pub fn new(device_ordinals: Vec<usize>) -> Self {
        Self { device_ordinals, device_profiles: HashMap::new() }
    }

    pub(crate) fn with_device_profile(mut self, device_ordinal: usize, profile: DeviceRegistryInstallProfile) -> Self {
        self.device_profiles.insert(device_ordinal, profile);
        self
    }

    pub(crate) fn profile(&self, device_ordinal: usize) -> Option<&DeviceRegistryInstallProfile> {
        self.device_profiles.get(&device_ordinal)
    }
}

/// Registry of service contexts by CUDA device ordinal.
#[derive(Debug, Clone)]
pub(crate) struct DeviceRegistry {
    contexts: Arc<[Option<DeviceContext>]>,
    device_ordinals: Arc<[usize]>,
    telemetry_device_ordinals: Arc<[usize]>,
    resource_profiles: Arc<[DeviceResourceProfile]>,
}

#[derive(Debug)]
pub(crate) struct ResolvedDeviceInstallPlan {
    pub(crate) device_ordinal: usize,
    pub(crate) facts: super::DeviceFacts,
    pub(crate) profile: DeviceRegistryInstallProfile,
    pub(crate) resource_profile: DeviceResourceProfile,
}

/// Complete immutable device plan set frozen before the first process-memory
/// manager is installed.
#[derive(Debug)]
pub(crate) struct ResolvedDeviceInstallPlans {
    plans: Vec<ResolvedDeviceInstallPlan>,
    device_ordinals: Vec<usize>,
    max_ordinal: usize,
}

impl ResolvedDeviceInstallPlans {
    #[cfg(test)]
    pub(crate) fn plans(&self) -> &[ResolvedDeviceInstallPlan] {
        &self.plans
    }

    #[cfg(test)]
    pub(crate) fn device_ordinals(&self) -> &[usize] {
        &self.device_ordinals
    }

    fn telemetry_device_ordinals(&self) -> Vec<usize> {
        self.plans.iter().filter(|plan| plan.profile.install_telemetry).map(|plan| plan.device_ordinal).collect()
    }
}

struct ManagedDevice {
    device: cudf::DeviceId,
    discovered: ResolvedDeviceInstallPlan,
    memory_failure_tracker: DeviceMemoryFailureTracker,
    mr_manager: cudf::DeviceMemoryManager,
    memory_manager_installation: cudf::DeviceMemoryManagerInstallationState,
    engine_health_identity: EngineDeviceHealthIdentity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EngineDeviceHealthIdentity {
    pub(crate) device_health_generation: u64,
    pub(crate) backend_health_generation: u64,
    pub(crate) memory_manager_installation_generation: u64,
}

#[derive(Debug, Clone, Copy)]
struct PersistentDeviceHealthRecord {
    device_health_generation: u64,
    memory_manager_installation_generation: u64,
}

impl DeviceRegistry {
    /// Resolves and freezes every requested device plan without installing a
    /// process memory manager, stream pool, governor, or execution context.
    pub(crate) fn resolve_install_plans(
        inventory: &dyn GpuInventory,
        config: &DeviceRegistryConfig,
        probe: &dyn GpuBackendConstructionProbe,
    ) -> Result<ResolvedDeviceInstallPlans> {
        let ordinals = config.device_ordinals.clone();
        let max_ordinal = ordinals[ordinals.len() - 1];
        let mut discovered = Vec::with_capacity(ordinals.len());
        for ordinal in &ordinals {
            probe.checkpoint(GpuBackendConstructionCheckpoint::BeforeDeviceDiscovery { device_ordinal: *ordinal })?;
            let facts = inventory.facts(*ordinal)?;
            if facts.device != *ordinal {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "query_engine.gpu.device_registry.device",
                    "GPU inventory facts must describe the requested device",
                )
                .with_fact("device_ordinal", *ordinal)
                .with_fact("facts_device_ordinal", facts.device)
                .with_fact("reason", "inventory_device_mismatch"));
            }
            let profile = config.profile(*ordinal).ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::InvalidConfiguration,
                    "query_engine.gpu.device_registry.config",
                    "device registry requires an explicit installation profile for every configured ordinal",
                )
                .with_fact("field", "device_profiles")
                .with_fact("device_ordinal", *ordinal)
                .with_fact("reason", "missing_device_profile")
            })?;
            let resource_profile =
                resolve_resource_profile(*ordinal, profile, managed_capacity_bytes(*ordinal, &facts, profile)?)?;
            discovered.push(ResolvedDeviceInstallPlan {
                device_ordinal: *ordinal,
                facts,
                profile: profile.clone(),
                resource_profile,
            });
            probe.checkpoint(GpuBackendConstructionCheckpoint::AfterDeviceDiscovery { device_ordinal: *ordinal })?;
        }

        Ok(ResolvedDeviceInstallPlans { plans: discovered, device_ordinals: ordinals, max_ordinal })
    }

    /// Installs a previously frozen complete plan set. Device discovery stays
    /// frozen: this phase performs no new device selection and no
    /// automatic-capacity policy resolution. It does take one live free-bytes
    /// snapshot per device to freeze the managed capacity (`M`) against
    /// post-install reality; the manager installation itself consumes device
    /// memory that pre-install facts cannot see.
    pub(crate) fn install_resolved(
        resolved: ResolvedDeviceInstallPlans,
        owner: &DeviceOwnerToken<'_>,
        probe: &dyn GpuBackendConstructionProbe,
        construction_journal: &ConstructionResourceJournal,
        backend_metrics: &BackendMetricsStore,
        inventory: &dyn GpuInventory,
    ) -> Result<Self> {
        let _owner_generation = owner.generation();
        let telemetry_device_ordinals = resolved.telemetry_device_ordinals();
        let ResolvedDeviceInstallPlans { plans: discovered, device_ordinals: ordinals, max_ordinal } = resolved;

        let discovered = discovered
            .into_iter()
            .map(|plan| {
                let device = DeviceId::new(plan.device_ordinal).map_err(|error| {
                    cudf_query_error(
                        "query_engine.gpu.device_registry.device",
                        "failed to resolve CUDA device while installing registry",
                        error,
                    )
                    .with_fact("device_ordinal", plan.device_ordinal)
                })?;
                Ok((device, plan))
            })
            .collect::<Result<Vec<_>>>()?;

        // Freeze discovery for the complete set before installing any
        // process-lifetime memory manager. A discovery failure therefore
        // cannot leave a prefix of devices installed.
        let mut managed = Vec::with_capacity(discovered.len());
        for (device, discovered) in discovered {
            let ordinal = discovered.device_ordinal;
            probe.checkpoint(GpuBackendConstructionCheckpoint::BeforeMemoryManager { device_ordinal: ordinal })?;
            let installation_before = DeviceMemoryManager::installation_state_for_device(device).map_err(|error| {
                cudf_query_error(
                    "query_engine.gpu.device_registry.memory_manager",
                    "failed to inspect the process device-manager installation before reuse",
                    error,
                )
                .with_fact("device_ordinal", ordinal)
                .with_fact("reason", "device_memory_manager_inspection_failed")
            })?;
            let allocation_limit_bytes = discovered.profile.memory_partition.allocation_limit_bytes();
            let release_threshold_bytes =
                usize::try_from(discovered.resource_profile.managed_capacity_bytes).map_err(|_| {
                    QueryError::new(
                        QueryErrorCode::InvalidConfiguration,
                        "query_engine.gpu.device_registry.memory_manager",
                        "managed device capacity does not fit the native allocator size type",
                    )
                    .with_fact("device_ordinal", ordinal)
                    .with_fact("managed_capacity_bytes", discovered.resource_profile.managed_capacity_bytes)
                    .with_fact("reason", "managed_capacity_size_overflow")
                })?;
            let requested_options =
                canonical_device_memory_manager_options(allocation_limit_bytes, Some(release_threshold_bytes));
            if installation_before.effective_options.is_some_and(|existing| existing != requested_options) {
                return Err(device_memory_manager_options_conflict(device, installation_before, requested_options));
            }
            let memory_failure_tracker =
                DeviceMemoryFailureTracker::for_installation_state(ordinal, installation_before)?;
            let (mr_manager, memory_manager_installation) =
                install_device_memory_manager(device, &memory_failure_tracker, requested_options)?;
            construction_journal.record_memory_manager(mr_manager.clone())?;
            let installation_generation = memory_manager_installation.installation_generation.ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "query_engine.gpu.device_registry.memory_manager",
                    "owned device-manager installation has no callback generation",
                )
                .with_fact("device_ordinal", ordinal)
                .with_fact("reason", "memory_manager_installation_generation_missing")
            })?;
            memory_failure_tracker.commit_installation_generation(installation_generation)?;
            let engine_health_identity =
                bind_engine_device_health(device, memory_manager_installation, owner.generation())?;
            managed.push(ManagedDevice {
                device,
                discovered,
                memory_failure_tracker,
                mr_manager,
                memory_manager_installation,
                engine_health_identity,
            });
            probe.checkpoint(GpuBackendConstructionCheckpoint::AfterMemoryManager { device_ordinal: ordinal })?;
        }

        // Freeze managed capacity against post-install live free bytes. No
        // query object exists yet, so this snapshot sees only the backend's
        // own installation footprint plus external pressure. Own idle pool
        // backing is added back so reinstalling over a warm pool cannot
        // shrink the budget every generation. The frozen `M` only ever moves
        // down; re-resolution revalidates the cache share, the default grant,
        // and the parallelism target, failing closed when the static config
        // no longer fits post-install reality.
        let mut managed = managed;
        for target in managed.iter_mut() {
            let ordinal = target.device.ordinal();
            let live = inventory.facts(ordinal)?;
            if live.device != ordinal {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "query_engine.gpu.device_registry.device",
                    "GPU inventory facts must describe the requested device",
                )
                .with_fact("device_ordinal", ordinal)
                .with_fact("facts_device_ordinal", live.device)
                .with_fact("reason", "inventory_device_mismatch"));
            }
            let free_now = live.free_bytes;
            let pool_reserved =
                target.mr_manager.pool_usage().map(|usage| usage.reserved_current_bytes).map_err(|error| {
                    cudf_query_error(
                        "query_engine.gpu.device_registry.memory_manager",
                        "failed to read pool backing while freezing managed capacity",
                        error,
                    )
                    .with_fact("device_ordinal", ordinal)
                })?;
            let effective_free = free_now.saturating_add(pool_reserved);
            let planned = target.discovered.resource_profile.managed_capacity_bytes;
            if effective_free < planned {
                target.discovered.resource_profile =
                    resolve_resource_profile(ordinal, &target.discovered.profile, effective_free)?;
            }
        }

        let resource_profiles =
            managed.iter().map(|managed| managed.discovered.resource_profile.clone()).collect::<Vec<_>>();
        let mut contexts = vec![None; max_ordinal.saturating_add(1)];
        for managed in managed {
            let ordinal = managed.device.ordinal();
            probe.checkpoint(GpuBackendConstructionCheckpoint::BeforeStreamPool { device_ordinal: ordinal })?;
            let resource_profile = &managed.discovered.resource_profile;
            let stream_pool = ServiceStreamPool::new_with_classes(
                managed.device,
                managed.discovered.profile.stream_pool_size,
                resource_profile.max_active_attempts,
                resource_profile.aux_source_work_capacity,
            )?;
            construction_journal.record_stream_pool(stream_pool.clone())?;
            probe.checkpoint(GpuBackendConstructionCheckpoint::AfterStreamPool { device_ordinal: ordinal })?;
            probe.checkpoint(GpuBackendConstructionCheckpoint::BeforeDeviceContext { device_ordinal: ordinal })?;
            let ledger = Arc::new(DeviceLedger::new(managed.discovered.resource_profile.clone()));
            let ledger_metrics = backend_metrics.clone();
            ledger.register_transition_observer(Arc::new(move |transition| {
                ledger_metrics.record_ledger_transition(transition);
            }))?;
            let context = DeviceContext::new(
                managed.device,
                stream_pool,
                managed.mr_manager,
                managed.memory_manager_installation,
                managed.engine_health_identity,
                managed.memory_failure_tracker,
                ledger,
                managed.discovered.facts,
                managed.discovered.profile.max_active_attempts,
            )?;
            construction_journal.record_device_context(context.clone())?;
            contexts[ordinal] = Some(context);
            probe.checkpoint(GpuBackendConstructionCheckpoint::AfterDeviceContext { device_ordinal: ordinal })?;
        }
        Ok(Self {
            contexts: contexts.into(),
            device_ordinals: ordinals.into(),
            telemetry_device_ordinals: telemetry_device_ordinals.into(),
            resource_profiles: resource_profiles.into(),
        })
    }

    /// Looks up a context by CUDA device ordinal.
    pub(crate) fn context_by_ordinal(&self, ordinal: usize) -> Result<&DeviceContext> {
        self.contexts.get(ordinal).and_then(Option::as_ref).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.gpu.device_registry.context",
                "CUDA device is not installed in the device registry",
            )
            .with_fact("device_ordinal", ordinal)
        })
    }

    /// Iterates installed contexts in ordinal order.
    pub(crate) fn contexts(&self) -> impl Iterator<Item = &DeviceContext> {
        self.contexts.iter().filter_map(Option::as_ref)
    }

    /// Allocation-failure reports for all installed devices.
    pub(crate) fn memory_failure_reports(&self) -> Vec<DeviceMemoryFailureReport> {
        self.contexts().map(DeviceContext::memory_failure_report).collect()
    }

    /// Enabled CUDA ordinals in placement order.
    pub(crate) fn device_ordinals(&self) -> &[usize] {
        &self.device_ordinals
    }

    /// CUDA ordinals whose immutable startup profile enables live telemetry.
    /// This diagnostic selection is independent of the admission device set.
    pub(crate) fn telemetry_device_ordinals(&self) -> &[usize] {
        &self.telemetry_device_ordinals
    }

    /// The resource profile frozen when `device_ordinal` was installed.
    pub(crate) fn resource_profile(&self, device_ordinal: usize) -> Option<&DeviceResourceProfile> {
        self.resource_profiles.iter().find(|profile| profile.device_ordinal == device_ordinal)
    }
}

fn bind_engine_device_health(
    device: cudf::DeviceId,
    installation: cudf::DeviceMemoryManagerInstallationState,
    backend_health_generation: u64,
) -> Result<EngineDeviceHealthIdentity> {
    let health = device_health_snapshot(device);
    if health.status() == DeviceHealthStatus::Unhealthy {
        return Err(QueryError::new(
            QueryErrorCode::CudfPersistentFailure,
            "query_engine.gpu.device_registry.health",
            "unrecoverable CUDA device health prevents backend publication",
        )
        .with_fact("device_ordinal", device.ordinal())
        .with_fact("reason", "persistent_device_unhealthy")
        .with_fact("fault_count", health.fault_count()));
    }
    let installation_generation = installation.installation_generation.ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::Internal,
            "query_engine.gpu.device_registry.health",
            "owned reusable memory manager has no installation generation",
        )
        .with_fact("device_ordinal", device.ordinal())
        .with_fact("reason", "memory_manager_installation_generation_missing")
    })?;

    static NEXT_DEVICE_HEALTH_GENERATION: AtomicU64 = AtomicU64::new(1);
    static DEVICE_HEALTH: OnceLock<Mutex<HashMap<usize, PersistentDeviceHealthRecord>>> = OnceLock::new();
    let registry = DEVICE_HEALTH.get_or_init(|| Mutex::new(HashMap::new()));
    let mut registry = registry.lock().map_err(|_error| {
        QueryError::new(
            QueryErrorCode::Internal,
            "query_engine.gpu.device_registry.health",
            "engine device-health registry was poisoned",
        )
        .with_fact("device_ordinal", device.ordinal())
        .with_fact("reason", "device_health_registry_poisoned")
    })?;
    let persistent = match registry.get(&device.ordinal()).copied() {
        Some(existing) if existing.memory_manager_installation_generation == installation_generation => existing,
        Some(existing) => {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.gpu.device_registry.health",
                "memory-manager installation generation changed without an engine recovery transition",
            )
            .with_fact("device_ordinal", device.ordinal())
            .with_fact("reason", "incompatible_installation_generation_reuse")
            .with_fact(
                "expected_memory_manager_installation_generation",
                existing.memory_manager_installation_generation,
            )
            .with_fact("observed_memory_manager_installation_generation", installation_generation));
        },
        None => {
            let device_health_generation = NEXT_DEVICE_HEALTH_GENERATION.fetch_add(1, Ordering::Relaxed);
            let created = PersistentDeviceHealthRecord {
                device_health_generation,
                memory_manager_installation_generation: installation_generation,
            };
            registry.insert(device.ordinal(), created);
            created
        },
    };
    Ok(EngineDeviceHealthIdentity {
        device_health_generation: persistent.device_health_generation,
        backend_health_generation,
        memory_manager_installation_generation: installation_generation,
    })
}

/// Resolves the immutable resource profile for one device from a managed
/// capacity (`M`). The pre-install call validates the configured capacity;
/// the post-install capacity freeze reuses it with a smaller `M` when live
/// free bytes prove the static budget no longer fits.
fn resolve_resource_profile(
    device_ordinal: usize,
    profile: &DeviceRegistryInstallProfile,
    capacity_bytes: u64,
) -> Result<DeviceResourceProfile> {
    let configured_limit_bytes = capacity_bytes.checked_add(profile.backend_reserve_bytes).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidConfiguration,
            "query_engine.gpu.device_registry.capacity",
            format!(
                "device {device_ordinal} managed capacity of {} plus backend_reserve_bytes of {} overflows the 64-bit byte range; lower backend_reserve_bytes",
                bytes_with_gib(capacity_bytes),
                bytes_with_gib(profile.backend_reserve_bytes),
            ),
        )
        .with_fact("device_ordinal", device_ordinal)
        .with_fact("field", "backend_reserve_bytes")
        .with_fact("reason", "configured_capacity_overflow")
    })?;
    // An unresolved query limit derives from the service parallelism
    // target: `floor((M - K) / N)`, so configuring `N` alone yields
    // a parallel-safe default. Explicit limits are validated by the
    // immutable resource-profile resolver below.
    let default_query_limit_bytes = profile.default_query_limit_bytes.map_or_else(
        || {
            DeviceResourceProfile::derive_default_query_limit_bytes(
                device_ordinal,
                capacity_bytes,
                profile.cache_cap_bytes(),
                profile.max_active_attempts,
            )
        },
        Ok,
    )?;
    DeviceResourceProfile::resolve(DeviceResourceEnvelope {
        device_ordinal,
        device_capabilities: profile.device_capabilities.clone(),
        configured_limit_bytes,
        headroom_bytes: 0,
        backend_reserve_bytes: profile.backend_reserve_bytes,
        default_query_limit_bytes,
        max_active_attempts: profile.max_active_attempts,
        source_work_per_attempt: profile.source_work_per_attempt,
        stream_pool_size: profile.stream_pool_size,
        worker_job_capacity: profile.worker_job_capacity,
        cache_cap_bytes: profile.cache_cap_bytes(),
    })
}

fn managed_capacity_bytes(
    device_ordinal: usize,
    facts: &super::DeviceFacts,
    profile: &DeviceRegistryInstallProfile,
) -> Result<u64> {
    let allocation_limit =
        profile.memory_partition.allocation_limit_bytes().and_then(|limit| u64::try_from(limit).ok());
    if let Some(allocation_limit) = allocation_limit.filter(|limit| *limit > facts.total_bytes) {
        return Err(QueryError::new(
            QueryErrorCode::InvalidConfiguration,
            "query_engine.gpu.device_registry.capacity",
            format!(
                "device {device_ordinal} static partition allocation_limit_bytes is {}, above the CUDA-visible device memory of {}; the largest accepted allocation_limit_bytes is {}",
                bytes_with_gib(allocation_limit),
                bytes_with_gib(facts.total_bytes),
                bytes_with_gib(facts.total_bytes),
            ),
        )
        .with_fact("device_ordinal", device_ordinal)
        .with_fact("allocation_limit_bytes", allocation_limit)
        .with_fact("physical_total_bytes", facts.total_bytes)
        .with_fact("reason", "static_partition_exceeds_physical_capacity"));
    }
    let owned_capacity_bytes = allocation_limit.unwrap_or(facts.total_bytes);
    let configured_limit = match profile.capacity_policy {
        DeviceRegistryCapacityPolicy::Automatic { headroom_bytes, fallback_bytes, fraction } => {
            // For a static partition, automatic sizing starts from the owned
            // partition rather than physical VRAM. Headroom must reduce the
            // budget Algeon actually owns, not a larger whole-device number.
            let headroom_capacity = owned_capacity_bytes
                .checked_sub(headroom_bytes)
                .filter(|value| *value > 0)
                .unwrap_or(fallback_bytes.min(owned_capacity_bytes));
            fraction.map_or(headroom_capacity, |fraction| {
                headroom_capacity.min(fraction.capacity_bytes(owned_capacity_bytes))
            })
        },
        DeviceRegistryCapacityPolicy::Explicit { limit_bytes } => {
            if limit_bytes > owned_capacity_bytes {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidConfiguration,
                    "query_engine.gpu.device_registry.capacity",
                    format!(
                        "device {device_ordinal} memory_limit_bytes is {} but Algeon owns {} of device memory; the largest accepted memory_limit_bytes is {}",
                        bytes_with_gib(limit_bytes),
                        bytes_with_gib(owned_capacity_bytes),
                        bytes_with_gib(owned_capacity_bytes),
                    ),
                )
                .with_fact("device_ordinal", device_ordinal)
                .with_fact("configured_limit_bytes", limit_bytes)
                .with_fact("owned_capacity_bytes", owned_capacity_bytes)
                .with_fact("reason", "explicit_capacity_exceeds_owned_capacity"));
            }
            limit_bytes
        },
    };
    let usable_capacity = allocation_limit.map_or(configured_limit, |limit| configured_limit.min(limit));
    let managed_capacity = usable_capacity.checked_sub(profile.backend_reserve_bytes).filter(|capacity| *capacity > 0);
    managed_capacity.ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidConfiguration,
            "query_engine.gpu.device_registry.capacity",
            format!(
                "device {device_ordinal} backend_reserve_bytes is {} and leaves no managed capacity out of the usable {}; the largest accepted backend_reserve_bytes is {}",
                bytes_with_gib(profile.backend_reserve_bytes),
                bytes_with_gib(usable_capacity),
                bytes_with_gib(usable_capacity.saturating_sub(1)),
            ),
        )
        .with_fact("device_ordinal", device_ordinal)
        .with_fact("configured_limit_bytes", configured_limit)
        .with_fact("usable_capacity_bytes", usable_capacity)
        .with_fact("backend_reserve_bytes", profile.backend_reserve_bytes)
        .with_fact("reason", "backend_reserve_exhausts_capacity")
    })
}

fn install_device_memory_manager(
    device: cudf::DeviceId,
    memory_failure_tracker: &DeviceMemoryFailureTracker,
    options: cudf::DeviceMemoryManagerOptions,
) -> Result<(cudf::DeviceMemoryManager, cudf::DeviceMemoryManagerInstallationState)> {
    let callback_was_installed = memory_failure_tracker.report().callback_installed;
    let identity = DeviceMemoryManagerInstallationIdentity::new(
        *b"algeon-q-engine\0",
        1,
        partition_fingerprint(device.ordinal(), options.allocation_limit_bytes),
    );
    let (manager, outcome) = match DeviceMemoryManager::install(
        device,
        InstallSpec { options, owner: Some(identity), on_failure: Some(memory_failure_tracker.failure_callback()) },
    ) {
        Ok(outcome) => outcome,
        Err(failure) => {
            let installation_state = failure.installation_state();
            if let Some(state) = installation_state {
                if state.failure_callback_installed {
                    if let Some(generation) = state.installation_generation {
                        memory_failure_tracker.commit_installation_generation(generation)?;
                    }
                    memory_failure_tracker.mark_callback_installed();
                } else if !callback_was_installed {
                    memory_failure_tracker.mark_callback_install_unavailable();
                }
            } else if !callback_was_installed {
                memory_failure_tracker.mark_callback_install_unavailable();
            }

            if let Some(state) = installation_state
                && state.cause == DeviceMemoryManagerInstallationCause::OptionsMismatch
            {
                return Err(
                    device_memory_manager_options_conflict(device, state, options).with_source(failure.into_error())
                );
            }

            let error = cudf_query_error(
                "query_engine.gpu.device_registry.memory_manager",
                "failed to install or reuse the Algeon-owned device memory manager",
                failure.into_error(),
            )
            .with_fact("reason", "device_memory_manager_installation_rejected")
            .with_fact("device_ordinal", device.ordinal());
            return Err(match installation_state {
                Some(state) => error
                    .with_fact("installation_status", format!("{:?}", state.status))
                    .with_fact("installation_retry", format!("{:?}", state.retry))
                    .with_fact("installation_cause", format!("{:?}", state.cause)),
                None => error,
            });
        },
    };
    if outcome.state.status != DeviceMemoryManagerInstallationStatus::Owned
        || outcome.state.retry != DeviceMemoryManagerInstallationRetry::Reusable
    {
        if !callback_was_installed {
            memory_failure_tracker.mark_callback_install_unavailable();
        }
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.gpu.device_registry.memory_manager",
            "device memory manager installation is not owned and reusable by Algeon",
        )
        .with_fact("reason", "device_memory_manager_installation_unusable")
        .with_fact("device_ordinal", device.ordinal())
        .with_fact("installation_status", format!("{:?}", outcome.state.status))
        .with_fact("installation_retry", format!("{:?}", outcome.state.retry)));
    }
    memory_failure_tracker.mark_callback_installed();
    Ok((manager, outcome.state))
}

/// Immutable allocator options are the first process-level compatibility
/// boundary. Check them before recovering the callback-owned tracker: a
/// foreign allocator can be both callback-unrecoverable and incompatible, but
/// only the latter tells the caller why its requested backend cannot start.
fn device_memory_manager_options_conflict(
    device: cudf::DeviceId,
    installation: cudf::DeviceMemoryManagerInstallationState,
    requested_options: cudf::DeviceMemoryManagerOptions,
) -> QueryError {
    let existing_options = installation
        .effective_options
        .map(|existing| format!("{existing:?}"))
        .unwrap_or_else(|| "unavailable".to_owned());
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        "query_engine.gpu.device_registry.memory_manager",
        "device memory manager already exists with different immutable options",
    )
    .with_fact("reason", "device_memory_manager_options_conflict")
    .with_fact("device_ordinal", device.ordinal())
    .with_fact("installation_status", format!("{:?}", installation.status))
    .with_fact("installation_retry", format!("{:?}", installation.retry))
    .with_fact("installation_cause", "OptionsMismatch")
    .with_fact("existing_options", existing_options)
    .with_fact("requested_options", format!("{requested_options:?}"))
}

/// Hard query domains require cuDF's canonical stream-ordered async base.
/// This stays in the registry because allocation-manager installation is a
/// process-owned backend concern, not a caller-selectable execution policy.
fn canonical_device_memory_manager_options(
    allocation_limit_bytes: Option<usize>,
    release_threshold_bytes: Option<usize>,
) -> cudf::DeviceMemoryManagerOptions {
    cudf::DeviceMemoryManagerOptions {
        base: DeviceMemoryManagerBase::CudaAsync {
            initial_pool_size: None,
            release_threshold: release_threshold_bytes,
        },
        allocation_limit_bytes,
    }
}

fn partition_fingerprint(device_ordinal: usize, allocation_limit_bytes: Option<usize>) -> [u8; 16] {
    // Installation identity includes the exact immutable partition, rather
    // than relying only on the separately compared manager options. Use two
    // independent stable 64-bit mixes so a different device, ownership mode,
    // or byte limit cannot preserve the old mode-only fingerprint.
    let mode = u64::from(allocation_limit_bytes.is_some());
    let limit = allocation_limit_bytes.map_or(u64::MAX, |value| value as u64);
    let device = device_ordinal as u64;
    let low = mix_fingerprint(device ^ limit.rotate_left(17) ^ mode);
    let high = mix_fingerprint(limit ^ device.rotate_left(29) ^ mode.rotate_left(61));
    let mut fingerprint = [0; 16];
    fingerprint[..8].copy_from_slice(&low.to_le_bytes());
    fingerprint[8..].copy_from_slice(&high.to_le_bytes());
    fingerprint
}

fn mix_fingerprint(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{DeviceFacts, backend_core::NoopConstructionProbe};

    #[derive(Debug, Default)]
    struct RecordingProbe(Mutex<Vec<GpuBackendConstructionCheckpoint>>);

    impl GpuBackendConstructionProbe for RecordingProbe {
        fn checkpoint(&self, checkpoint: GpuBackendConstructionCheckpoint) -> Result<()> {
            self.0.lock().unwrap().push(checkpoint);
            Ok(())
        }
    }

    #[derive(Debug)]
    struct FakeInventory {
        facts: Vec<DeviceFacts>,
    }

    impl GpuInventory for FakeInventory {
        fn facts(&self, device_ordinal: usize) -> Result<DeviceFacts> {
            self.facts.get(device_ordinal).cloned().ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "query_engine.gpu.device_registry.test_inventory",
                    "missing fake device facts",
                )
            })
        }
    }

    #[test]
    fn resolving_complete_plan_set_has_no_installation_checkpoint() {
        let inventory = FakeInventory {
            facts: vec![
                DeviceFacts {
                    device: 0,
                    model_name: None,
                    uuid: None,
                    physical_total_bytes: None,
                    total_bytes: 1_000,
                    free_bytes: 500,
                    sm_count: Some(1),
                    utilization_pct: None,
                },
                DeviceFacts {
                    device: 1,
                    model_name: None,
                    uuid: None,
                    physical_total_bytes: None,
                    total_bytes: 2_000,
                    free_bytes: 500,
                    sm_count: Some(1),
                    utilization_pct: None,
                },
            ],
        };
        let profile = DeviceRegistryInstallProfile {
            stream_pool_size: 2,
            device_capabilities: DeviceCapabilities::default(),
            max_active_attempts: 1,
            capacity_policy: DeviceRegistryCapacityPolicy::Automatic {
                headroom_bytes: 100,
                fallback_bytes: 200,
                fraction: None,
            },
            backend_reserve_bytes: 0,
            cache_cap_bytes: 0,
            default_query_limit_bytes: None,
            source_work_per_attempt: 1,
            worker_job_capacity: 2,
            lane_count: 1,
            install_telemetry: true,
            memory_partition: DeviceRegistryMemoryPartition::WholeDevice,
        };
        let config = DeviceRegistryConfig::new(vec![0, 1])
            .with_device_profile(0, profile.clone())
            .with_device_profile(1, DeviceRegistryInstallProfile { install_telemetry: false, ..profile });
        let probe = RecordingProbe::default();

        let resolved = DeviceRegistry::resolve_install_plans(&inventory, &config, &probe).unwrap();

        assert_eq!(resolved.device_ordinals(), &[0, 1]);
        assert_eq!(resolved.telemetry_device_ordinals(), vec![0]);
        assert_eq!(
            resolved.plans().iter().map(|plan| plan.resource_profile.managed_capacity_bytes).collect::<Vec<_>>(),
            vec![900, 1_900]
        );
        assert_eq!(
            *probe.0.lock().unwrap(),
            vec![
                GpuBackendConstructionCheckpoint::BeforeDeviceDiscovery { device_ordinal: 0 },
                GpuBackendConstructionCheckpoint::AfterDeviceDiscovery { device_ordinal: 0 },
                GpuBackendConstructionCheckpoint::BeforeDeviceDiscovery { device_ordinal: 1 },
                GpuBackendConstructionCheckpoint::AfterDeviceDiscovery { device_ordinal: 1 },
            ]
        );
    }

    #[test]
    fn resolves_ordinal_local_resource_profiles_and_rejects_incompatible_device() {
        let inventory = FakeInventory {
            facts: vec![
                DeviceFacts {
                    device: 0,
                    model_name: None,
                    uuid: None,
                    physical_total_bytes: None,
                    total_bytes: 1_000,
                    free_bytes: 1_000,
                    sm_count: Some(1),
                    utilization_pct: None,
                },
                DeviceFacts {
                    device: 1,
                    model_name: None,
                    uuid: None,
                    physical_total_bytes: None,
                    total_bytes: 2_000,
                    free_bytes: 2_000,
                    sm_count: Some(1),
                    utilization_pct: None,
                },
            ],
        };
        let profile = |default_query_limit_bytes| DeviceRegistryInstallProfile {
            stream_pool_size: 4,
            device_capabilities: DeviceCapabilities::default(),
            max_active_attempts: 2,
            capacity_policy: DeviceRegistryCapacityPolicy::Automatic {
                headroom_bytes: 100,
                fallback_bytes: 100,
                fraction: None,
            },
            backend_reserve_bytes: 100,
            cache_cap_bytes: 0,
            default_query_limit_bytes,
            source_work_per_attempt: 1,
            worker_job_capacity: 4,
            lane_count: 2,
            install_telemetry: true,
            memory_partition: DeviceRegistryMemoryPartition::WholeDevice,
        };
        let plans = DeviceRegistry::resolve_install_plans(
            &inventory,
            &DeviceRegistryConfig::new(vec![0, 1])
                .with_device_profile(0, profile(None))
                .with_device_profile(1, profile(None)),
            &NoopConstructionProbe,
        )
        .unwrap();
        assert_eq!(plans.plans()[0].resource_profile.managed_capacity_bytes, 800);
        assert_eq!(plans.plans()[1].resource_profile.managed_capacity_bytes, 1_800);
        // An unspecified default derives from the parallelism target as
        // `floor((M - K) / N)`, never the whole managed capacity: with no
        // frozen residents the profile admits `N` default grants.
        assert_eq!(plans.plans()[0].resource_profile.query_share_bytes(), 800);
        assert_eq!(plans.plans()[1].resource_profile.query_share_bytes(), 1_800);
        assert_eq!(plans.plans()[0].resource_profile.default_query_limit_bytes, 400);
        assert_eq!(plans.plans()[1].resource_profile.default_query_limit_bytes, 900);

        let explicit = DeviceRegistry::resolve_install_plans(
            &inventory,
            &DeviceRegistryConfig::new(vec![0, 1])
                .with_device_profile(0, profile(None))
                .with_device_profile(1, profile(Some(1_000))),
            &NoopConstructionProbe,
        )
        .unwrap();
        assert_eq!(explicit.plans()[1].resource_profile.default_query_limit_bytes, 1_000);

        let error = DeviceRegistry::resolve_install_plans(
            &inventory,
            &DeviceRegistryConfig::new(vec![0, 1])
                .with_device_profile(0, profile(None))
                .with_device_profile(1, profile(Some(2_000))),
            &NoopConstructionProbe,
        )
        .unwrap_err();
        assert_eq!(error.fact_value("device_ordinal"), Some("1"));
        assert_eq!(error.fact_value("reason"), Some("default_query_limit_exceeds_managed_capacity"));
    }

    #[test]
    fn capacity_freeze_reresolves_the_contract_instead_of_silently_clamping() {
        let profile = |default_query_limit_bytes| DeviceRegistryInstallProfile {
            stream_pool_size: 4,
            device_capabilities: DeviceCapabilities::default(),
            max_active_attempts: 2,
            capacity_policy: DeviceRegistryCapacityPolicy::Automatic {
                headroom_bytes: 0,
                fallback_bytes: 1,
                fraction: None,
            },
            backend_reserve_bytes: 0,
            cache_cap_bytes: 8,
            default_query_limit_bytes,
            source_work_per_attempt: 1,
            worker_job_capacity: 4,
            lane_count: 2,
            install_telemetry: true,
            memory_partition: DeviceRegistryMemoryPartition::WholeDevice,
        };

        // M = 24, K = 8, N = 2, explicit default 8.
        let resolved = resolve_resource_profile(0, &profile(Some(8)), 24).unwrap();
        assert_eq!(resolved.managed_capacity_bytes, 24);
        assert_eq!(resolved.query_share_bytes(), 16);

        // Freezing at 16 keeps the explicit default: it still fits Q = 8.
        let frozen = resolve_resource_profile(0, &profile(Some(8)), 16).unwrap();
        assert_eq!(frozen.managed_capacity_bytes, 16);
        assert_eq!(frozen.query_share_bytes(), 8);
        assert_eq!(frozen.default_query_limit_bytes, 8);

        // Freezing at 15 fails closed: the explicit default no longer fits
        // Q = 7, and the freeze must not clamp it silently.
        let too_small = resolve_resource_profile(0, &profile(Some(8)), 15).unwrap_err();
        assert_eq!(too_small.fact_value("reason"), Some("default_query_limit_exceeds_managed_capacity"));

        // An unspecified default follows the frozen M down: floor((20-8)/2).
        let rederived = resolve_resource_profile(0, &profile(None), 20).unwrap();
        assert_eq!(rederived.managed_capacity_bytes, 20);
        assert_eq!(rederived.default_query_limit_bytes, 6);
    }

    #[test]
    fn managed_capacities_are_derived_independently_per_device() {
        let whole_facts = DeviceFacts {
            device: 0,
            model_name: None,
            uuid: None,
            physical_total_bytes: None,
            total_bytes: 1_000,
            free_bytes: 100,
            sm_count: Some(1),
            utilization_pct: None,
        };
        let whole = DeviceRegistryInstallProfile {
            stream_pool_size: 2,
            device_capabilities: DeviceCapabilities::default(),
            max_active_attempts: 2,
            capacity_policy: DeviceRegistryCapacityPolicy::Automatic {
                headroom_bytes: 100,
                fallback_bytes: 200,
                fraction: None,
            },
            backend_reserve_bytes: 50,
            cache_cap_bytes: 0,
            default_query_limit_bytes: None,
            source_work_per_attempt: 1,
            worker_job_capacity: 2,
            lane_count: 2,
            install_telemetry: true,
            memory_partition: DeviceRegistryMemoryPartition::WholeDevice,
        };
        let partition_facts = DeviceFacts { device: 1, total_bytes: 4_000, ..whole_facts.clone() };
        let partition = DeviceRegistryInstallProfile {
            stream_pool_size: 7,
            device_capabilities: DeviceCapabilities::default(),
            max_active_attempts: 7,
            capacity_policy: DeviceRegistryCapacityPolicy::Automatic {
                headroom_bytes: 100,
                fallback_bytes: 200,
                fraction: None,
            },
            backend_reserve_bytes: 25,
            cache_cap_bytes: 0,
            default_query_limit_bytes: None,
            source_work_per_attempt: 1,
            worker_job_capacity: 7,
            lane_count: 7,
            install_telemetry: true,
            memory_partition: DeviceRegistryMemoryPartition::StaticPartition { allocation_limit_bytes: 600 },
        };
        let explicit_facts = DeviceFacts { device: 2, total_bytes: 10_000, ..whole_facts.clone() };
        let explicit = DeviceRegistryInstallProfile {
            stream_pool_size: 3,
            device_capabilities: DeviceCapabilities::default(),
            max_active_attempts: 3,
            capacity_policy: DeviceRegistryCapacityPolicy::Explicit { limit_bytes: 700 },
            backend_reserve_bytes: 10,
            cache_cap_bytes: 0,
            default_query_limit_bytes: None,
            source_work_per_attempt: 1,
            worker_job_capacity: 3,
            lane_count: 3,
            install_telemetry: true,
            memory_partition: DeviceRegistryMemoryPartition::WholeDevice,
        };

        assert_eq!(managed_capacity_bytes(0, &whole_facts, &whole).unwrap(), 850);
        assert_eq!(managed_capacity_bytes(1, &partition_facts, &partition).unwrap(), 475);
        assert_eq!(managed_capacity_bytes(2, &explicit_facts, &explicit).unwrap(), 690);
    }

    #[test]
    fn partition_fingerprint_separates_ownership_mode_device_and_exact_limit() {
        let whole = partition_fingerprint(3, None);
        let partitioned = partition_fingerprint(3, Some(1_024));

        assert_ne!(whole, partitioned);
        assert_ne!(whole, partition_fingerprint(4, None));
        assert_ne!(partitioned, partition_fingerprint(3, Some(2_048)));
    }

    #[test]
    fn canonical_manager_uses_managed_capacity_as_shared_pool_release_threshold() {
        let options = canonical_device_memory_manager_options(Some(4_096), Some(8_192));

        assert_eq!(options.allocation_limit_bytes, Some(4_096));
        assert_eq!(
            options.base,
            cudf::DeviceMemoryManagerBase::CudaAsync { initial_pool_size: None, release_threshold: Some(8_192) }
        );
    }

    #[test]
    fn managed_capacity_never_exceeds_physical_or_owned_memory() {
        let facts = DeviceFacts {
            device: 0,
            model_name: None,
            uuid: None,
            physical_total_bytes: None,
            total_bytes: 100,
            free_bytes: 50,
            sm_count: Some(1),
            utilization_pct: None,
        };
        let automatic = DeviceRegistryInstallProfile {
            stream_pool_size: 1,
            device_capabilities: DeviceCapabilities::default(),
            max_active_attempts: 1,
            capacity_policy: DeviceRegistryCapacityPolicy::Automatic {
                headroom_bytes: 500,
                fallback_bytes: 256,
                fraction: None,
            },
            backend_reserve_bytes: 0,
            cache_cap_bytes: 0,
            default_query_limit_bytes: None,
            source_work_per_attempt: 1,
            worker_job_capacity: 1,
            lane_count: 1,
            install_telemetry: true,
            memory_partition: DeviceRegistryMemoryPartition::WholeDevice,
        };
        assert_eq!(managed_capacity_bytes(0, &facts, &automatic).unwrap(), 100);

        let explicit = DeviceRegistryInstallProfile {
            capacity_policy: DeviceRegistryCapacityPolicy::Explicit { limit_bytes: 101 },
            ..automatic.clone()
        };
        let error = managed_capacity_bytes(0, &facts, &explicit).unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("explicit_capacity_exceeds_owned_capacity"));

        let partition = DeviceRegistryInstallProfile {
            capacity_policy: DeviceRegistryCapacityPolicy::Explicit { limit_bytes: 100 },
            memory_partition: DeviceRegistryMemoryPartition::StaticPartition { allocation_limit_bytes: 101 },
            ..automatic
        };
        let error = managed_capacity_bytes(0, &facts, &partition).unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("static_partition_exceeds_physical_capacity"));
    }

    #[test]
    fn capacity_rejections_are_configuration_errors_that_state_value_limit_and_largest_accepted_value() {
        const GIB: u64 = 1 << 30;
        // A 96 GB card reports 101973819392 CUDA-visible bytes, below 95 GiB.
        let facts = DeviceFacts {
            device: 0,
            model_name: None,
            uuid: None,
            physical_total_bytes: None,
            total_bytes: 101_973_819_392,
            free_bytes: 101_000_000_000,
            sm_count: Some(1),
            utilization_pct: None,
        };
        let profile = DeviceRegistryInstallProfile {
            stream_pool_size: 1,
            device_capabilities: DeviceCapabilities::default(),
            max_active_attempts: 1,
            capacity_policy: DeviceRegistryCapacityPolicy::Explicit { limit_bytes: 95 * GIB },
            backend_reserve_bytes: 0,
            cache_cap_bytes: 0,
            default_query_limit_bytes: None,
            source_work_per_attempt: 1,
            worker_job_capacity: 1,
            lane_count: 1,
            install_telemetry: true,
            memory_partition: DeviceRegistryMemoryPartition::WholeDevice,
        };

        let error = managed_capacity_bytes(0, &facts, &profile).unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::InvalidConfiguration);
        assert_eq!(
            error.message(),
            "device 0 memory_limit_bytes is 102005473280 bytes (95.00 GiB) but Algeon owns 101973819392 bytes (94.97 GiB) of device memory; the largest accepted memory_limit_bytes is 101973819392 bytes (94.97 GiB)"
        );

        let reserve = DeviceRegistryInstallProfile {
            capacity_policy: DeviceRegistryCapacityPolicy::Explicit { limit_bytes: 90 * GIB },
            backend_reserve_bytes: 90 * GIB,
            ..profile.clone()
        };
        let error = managed_capacity_bytes(0, &facts, &reserve).unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::InvalidConfiguration);
        assert_eq!(
            error.message(),
            "device 0 backend_reserve_bytes is 96636764160 bytes (90.00 GiB) and leaves no managed capacity out of the usable 96636764160 bytes (90.00 GiB); the largest accepted backend_reserve_bytes is 96636764159 bytes (89.99 GiB)"
        );

        let partition = DeviceRegistryInstallProfile {
            memory_partition: DeviceRegistryMemoryPartition::StaticPartition {
                allocation_limit_bytes: usize::try_from(96 * GIB).unwrap(),
            },
            ..profile
        };
        let error = managed_capacity_bytes(0, &facts, &partition).unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::InvalidConfiguration);
        assert_eq!(
            error.message(),
            "device 0 static partition allocation_limit_bytes is 103079215104 bytes (96.00 GiB), above the CUDA-visible device memory of 101973819392 bytes (94.97 GiB); the largest accepted allocation_limit_bytes is 101973819392 bytes (94.97 GiB)"
        );
    }
}
