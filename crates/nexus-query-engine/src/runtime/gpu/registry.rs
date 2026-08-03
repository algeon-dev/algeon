//! Process-level GPU device registry for service execution.

use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

use super::{
    DeviceCapabilities, DeviceContext, DeviceLedger, DeviceMemoryFailureReport, DeviceMemoryFailureTracker,
    DeviceOwnerToken, DeviceResourceProfile, GpuInventory, PlacementPolicy, PlacementRequest,
};
use crate::{
    error::{QueryError, QueryErrorCode, Result},
    runtime::{
        GpuBackendConstructionCheckpoint, GpuBackendConstructionProbe, GpuMemoryFraction, ServiceStreamPool,
        backend_core::ConstructionResourceJournal,
    },
};

/// Soft managed-capacity calculation for one installed device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeviceRegistryCapacityPolicy {
    Automatic {
        headroom_bytes: u64,
        fallback_bytes: u64,
        fraction: Option<GpuMemoryFraction>,
        /// Legacy uniform construction may additionally cap the automatic
        /// result. P0 backend profiles leave this unset.
        maximum_bytes: Option<u64>,
    },
    Explicit {
        limit_bytes: u64,
    },
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
    pub(crate) max_active_queries: Option<usize>,
    pub(crate) capacity_policy: DeviceRegistryCapacityPolicy,
    pub(crate) backend_reserve_bytes: u64,
    pub(crate) attempt_floor_bytes: Option<u64>,
    pub(crate) source_work_per_attempt: usize,
    pub(crate) worker_job_capacity: usize,
    pub(crate) lane_count: usize,
    pub(crate) memory_partition: DeviceRegistryMemoryPartition,
}

/// Configuration used when installing a device registry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DeviceRegistryConfig {
    /// CUDA ordinals enabled for the service. Empty means all visible devices.
    pub device_ordinals: Vec<usize>,
    /// Stream-pool size installed for each enabled device.
    pub stream_pool_size: usize,
    /// Fixed capacity headroom subtracted from each device's total memory.
    pub capacity_headroom_bytes: u64,
    /// Fallback capacity used if the headroom calculation underflows.
    pub fallback_capacity_bytes: u64,
    /// Optional per-device governor capacity cap.
    ///
    /// The effective capacity is `min(inventory_capacity, device_capacity_bytes)`
    /// so a single config value cannot over-grant smaller GPUs.
    pub device_capacity_bytes: Option<u64>,
    device_profiles: HashMap<usize, DeviceRegistryInstallProfile>,
}

impl DeviceRegistryConfig {
    /// Creates a registry config for explicit device ordinals.
    pub fn new(device_ordinals: Vec<usize>, stream_pool_size: usize) -> Self {
        Self {
            device_ordinals,
            stream_pool_size,
            capacity_headroom_bytes: 500 * 1024 * 1024,
            fallback_capacity_bytes: 256 * 1024 * 1024,
            device_capacity_bytes: None,
            device_profiles: HashMap::new(),
        }
    }

    /// Sets the per-device capacity calculation parameters.
    pub fn with_capacity_policy(mut self, headroom_bytes: u64, fallback_bytes: u64) -> Self {
        self.capacity_headroom_bytes = headroom_bytes;
        self.fallback_capacity_bytes = fallback_bytes;
        self
    }

    /// Sets a maximum governor capacity for every installed device.
    pub fn with_device_capacity_bytes(mut self, device_capacity_bytes: u64) -> Self {
        self.device_capacity_bytes = Some(device_capacity_bytes);
        self
    }

    pub(crate) fn with_device_profile(mut self, device_ordinal: usize, profile: DeviceRegistryInstallProfile) -> Self {
        self.device_profiles.insert(device_ordinal, profile);
        self
    }

    pub(crate) fn profile(&self, device_ordinal: usize) -> DeviceRegistryInstallProfile {
        self.device_profiles.get(&device_ordinal).cloned().unwrap_or(DeviceRegistryInstallProfile {
            stream_pool_size: self.stream_pool_size,
            device_capabilities: DeviceCapabilities::default(),
            max_active_queries: None,
            capacity_policy: DeviceRegistryCapacityPolicy::Automatic {
                headroom_bytes: self.capacity_headroom_bytes,
                fallback_bytes: self.fallback_capacity_bytes,
                fraction: None,
                maximum_bytes: self.device_capacity_bytes,
            },
            backend_reserve_bytes: 0,
            attempt_floor_bytes: None,
            source_work_per_attempt: 1,
            worker_job_capacity: self.stream_pool_size,
            lane_count: self.stream_pool_size,
            memory_partition: DeviceRegistryMemoryPartition::WholeDevice,
        })
    }
}

/// Registry of service contexts by CUDA device ordinal.
#[derive(Debug, Clone)]
pub(crate) struct DeviceRegistry {
    contexts: Arc<[Option<DeviceContext>]>,
    device_ordinals: Arc<[usize]>,
    resource_profiles: Arc<[DeviceResourceProfile]>,
}

#[derive(Debug)]
pub(crate) struct ResolvedDeviceInstallPlan {
    pub(crate) device: cudf::DeviceId,
    pub(crate) facts: super::DeviceFacts,
    pub(crate) profile: DeviceRegistryInstallProfile,
    pub(crate) capacity_bytes: u64,
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
    pub(crate) fn plans(&self) -> &[ResolvedDeviceInstallPlan] {
        &self.plans
    }

    pub(crate) fn device_ordinals(&self) -> &[usize] {
        &self.device_ordinals
    }
}

struct ManagedDevice {
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
        probe: Option<&dyn GpuBackendConstructionProbe>,
    ) -> Result<ResolvedDeviceInstallPlans> {
        validate_registry_config(config)?;
        let ordinals = configured_ordinals(inventory, config)?;
        let max_ordinal = ordinals.iter().copied().max().unwrap_or(0);
        let mut discovered = Vec::with_capacity(ordinals.len());
        for ordinal in &ordinals {
            checkpoint(probe, GpuBackendConstructionCheckpoint::BeforeDeviceDiscovery { device_ordinal: *ordinal })?;
            let device = cudf::DeviceId::new(*ordinal).map_err(|error| {
                QueryError::from_cudf(
                    "nexus_query_engine.gpu.device_registry.device",
                    "failed to resolve CUDA device while installing registry",
                    error,
                )
                .with_fact("device_ordinal", *ordinal)
            })?;
            let facts = inventory.facts(*ordinal)?;
            if facts.device != *ordinal {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.gpu.device_registry.device",
                    "GPU inventory facts must describe the requested device",
                )
                .with_fact("device_ordinal", *ordinal)
                .with_fact("facts_device_ordinal", facts.device)
                .with_fact("reason", "inventory_device_mismatch"));
            }
            let profile = config.profile(*ordinal);
            let capacity_bytes = managed_capacity_bytes(*ordinal, &facts, &profile)?;
            let configured_limit_bytes =
                capacity_bytes.checked_add(profile.backend_reserve_bytes).ok_or_else(|| {
                    QueryError::new(
                        QueryErrorCode::InvalidPlan,
                        "nexus_query_engine.gpu.device_registry.capacity",
                        "configured capacity overflowed",
                    )
                    .with_fact("device_ordinal", *ordinal)
                    .with_fact("field", "backend_reserve_bytes")
                    .with_fact("reason", "configured_capacity_overflow")
                })?;
            // The default floor divides the managed capacity into eight
            // baseline shares, or into one share per active-attempt slot when
            // more than eight slots are configured, so the frozen
            // `floor * A_d <= managed_capacity` invariant always holds.
            // Explicit floors are validated against the same equation.
            let max_active = profile.max_active_queries.unwrap_or(1);
            let default_floor_divisor = u64::try_from(max_active).unwrap_or(u64::MAX).max(8);
            let attempt_floor_bytes = match profile.attempt_floor_bytes {
                Some(floor) => floor,
                None => {
                    let floor = capacity_bytes / default_floor_divisor;
                    if floor == 0 {
                        return Err(QueryError::new(
                            QueryErrorCode::InvalidPlan,
                            "nexus_query_engine.gpu.device_registry.capacity",
                            "managed capacity is too small for the default attempt floor",
                        )
                        .with_fact("device_ordinal", *ordinal)
                        .with_fact("field", "attempt_floor_bytes")
                        .with_fact("reason", "managed_capacity_too_small_for_default_floor")
                        .with_fact("managed_capacity_bytes", capacity_bytes)
                        .with_fact("max_active_attempts", max_active));
                    }
                    floor
                },
            };
            let resource_profile = DeviceResourceProfile::resolve_with_capabilities(
                *ordinal,
                configured_limit_bytes,
                0,
                profile.backend_reserve_bytes,
                attempt_floor_bytes,
                profile.max_active_queries.unwrap_or(1),
                profile.source_work_per_attempt,
                profile.stream_pool_size,
                profile.worker_job_capacity,
                profile.device_capabilities.clone(),
            )?;
            if profile.lane_count < resource_profile.max_active_attempts {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.gpu.device_registry.capacity",
                    "lane count must cover every active attempt",
                )
                .with_fact("device_ordinal", *ordinal)
                .with_fact("field", "lane_count")
                .with_fact("reason", "lane_count_below_active_limit")
                .with_fact("lane_count", profile.lane_count)
                .with_fact("max_active_attempts", resource_profile.max_active_attempts));
            }
            discovered.push(ResolvedDeviceInstallPlan { device, facts, profile, capacity_bytes, resource_profile });
            checkpoint(probe, GpuBackendConstructionCheckpoint::AfterDeviceDiscovery { device_ordinal: *ordinal })?;
        }

        Ok(ResolvedDeviceInstallPlans { plans: discovered, device_ordinals: ordinals, max_ordinal })
    }

    /// Installs a previously frozen complete plan set. No hardware discovery
    /// or automatic-capacity resolution is permitted in this phase.
    pub(crate) fn install_resolved(
        resolved: ResolvedDeviceInstallPlans,
        owner: &DeviceOwnerToken<'_>,
        probe: Option<&dyn GpuBackendConstructionProbe>,
        construction_journal: Option<&Arc<ConstructionResourceJournal>>,
    ) -> Result<Self> {
        let _owner_generation = owner.generation();
        let ResolvedDeviceInstallPlans { plans: discovered, device_ordinals: ordinals, max_ordinal } = resolved;

        // Freeze discovery for the complete set before installing any
        // process-lifetime memory manager. A discovery failure therefore
        // cannot leave a prefix of devices installed.
        let mut managed = Vec::with_capacity(discovered.len());
        for discovered in discovered {
            let ordinal = discovered.device.ordinal();
            checkpoint(probe, GpuBackendConstructionCheckpoint::BeforeMemoryManager { device_ordinal: ordinal })?;
            let installation_before = cudf::DeviceMemoryManager::installation_state_for_device(discovered.device)
                .map_err(|error| {
                    QueryError::from_cudf(
                        "nexus_query_engine.gpu.device_registry.memory_manager",
                        "failed to inspect the process device-manager installation before reuse",
                        error,
                    )
                    .with_fact("device_ordinal", ordinal)
                    .with_fact("reason", "device_memory_manager_inspection_failed")
                })?;
            let allocation_limit_bytes = discovered.profile.memory_partition.allocation_limit_bytes();
            let requested_options = canonical_device_memory_manager_options(allocation_limit_bytes);
            if installation_before.effective_options.is_some_and(|existing| existing != requested_options) {
                return Err(device_memory_manager_options_conflict(
                    discovered.device,
                    installation_before,
                    requested_options,
                ));
            }
            let memory_failure_tracker =
                DeviceMemoryFailureTracker::for_installation_state(ordinal, installation_before)?;
            let (mr_manager, memory_manager_installation) =
                install_device_memory_manager(discovered.device, &memory_failure_tracker, requested_options)?;
            if let Some(journal) = construction_journal {
                journal.record_memory_manager(mr_manager.clone())?;
            }
            let installation_generation = memory_manager_installation.installation_generation.ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "nexus_query_engine.gpu.device_registry.memory_manager",
                    "owned device-manager installation has no callback generation",
                )
                .with_fact("device_ordinal", ordinal)
                .with_fact("reason", "memory_manager_installation_generation_missing")
            })?;
            memory_failure_tracker.commit_installation_generation(installation_generation)?;
            let engine_health_identity =
                bind_engine_device_health(discovered.device, memory_manager_installation, owner.generation())?;
            managed.push(ManagedDevice {
                discovered,
                memory_failure_tracker,
                mr_manager,
                memory_manager_installation,
                engine_health_identity,
            });
            checkpoint(probe, GpuBackendConstructionCheckpoint::AfterMemoryManager { device_ordinal: ordinal })?;
        }

        let resource_profiles =
            managed.iter().map(|managed| managed.discovered.resource_profile.clone()).collect::<Vec<_>>();
        let mut contexts = vec![None; max_ordinal.saturating_add(1)];
        for managed in managed {
            let ordinal = managed.discovered.device.ordinal();
            checkpoint(probe, GpuBackendConstructionCheckpoint::BeforeStreamPool { device_ordinal: ordinal })?;
            let resource_profile = &managed.discovered.resource_profile;
            let stream_pool = ServiceStreamPool::new_with_classes(
                managed.discovered.device,
                managed.discovered.profile.stream_pool_size,
                resource_profile.max_active_attempts,
                resource_profile.aux_source_work_capacity,
            )?;
            if let Some(journal) = construction_journal {
                journal.record_stream_pool(stream_pool.clone())?;
            }
            checkpoint(probe, GpuBackendConstructionCheckpoint::AfterStreamPool { device_ordinal: ordinal })?;
            checkpoint(probe, GpuBackendConstructionCheckpoint::BeforeDeviceContext { device_ordinal: ordinal })?;
            let ledger = Arc::new(DeviceLedger::new(managed.discovered.resource_profile.clone()));
            let context = DeviceContext::new(
                managed.discovered.device,
                stream_pool,
                managed.mr_manager,
                managed.memory_manager_installation,
                managed.engine_health_identity,
                managed.memory_failure_tracker,
                ledger,
                managed.discovered.facts,
                managed.discovered.profile.max_active_queries,
            )?;
            if let Some(journal) = construction_journal {
                journal.record_device_context(context.clone())?;
            }
            contexts[ordinal] = Some(context);
            checkpoint(probe, GpuBackendConstructionCheckpoint::AfterDeviceContext { device_ordinal: ordinal })?;
        }
        Ok(Self {
            contexts: contexts.into(),
            device_ordinals: ordinals.into(),
            resource_profiles: resource_profiles.into(),
        })
    }

    /// Legacy internal construction wrapper. The ownership-enforcing backend
    /// uses the two explicit phases so it can derive its private controller
    /// from the frozen plans before this method's first mutation.
    pub(crate) fn install(
        inventory: &dyn GpuInventory,
        config: &DeviceRegistryConfig,
        owner: &DeviceOwnerToken<'_>,
        probe: Option<&dyn GpuBackendConstructionProbe>,
        construction_journal: Option<&Arc<ConstructionResourceJournal>>,
    ) -> Result<Self> {
        let resolved = Self::resolve_install_plans(inventory, config, probe)?;
        Self::install_resolved(resolved, owner, probe, construction_journal)
    }

    /// Looks up a context by CUDA device ordinal.
    pub fn context(&self, device: cudf::DeviceId) -> Result<&DeviceContext> {
        self.context_by_ordinal(device.ordinal())
    }

    /// Looks up a context by CUDA device ordinal.
    pub fn context_by_ordinal(&self, ordinal: usize) -> Result<&DeviceContext> {
        self.contexts.get(ordinal).and_then(Option::as_ref).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu.device_registry.context",
                "CUDA device is not installed in the device registry",
            )
            .with_fact("device_ordinal", ordinal)
        })
    }

    /// Iterates installed contexts in ordinal order.
    pub fn contexts(&self) -> impl Iterator<Item = &DeviceContext> {
        self.contexts.iter().filter_map(Option::as_ref)
    }

    /// Allocation-failure reports for all installed devices.
    pub fn memory_failure_reports(&self) -> Vec<DeviceMemoryFailureReport> {
        self.contexts().map(DeviceContext::memory_failure_report).collect()
    }

    /// Enabled CUDA ordinals in placement order.
    pub fn device_ordinals(&self) -> &[usize] {
        &self.device_ordinals
    }

    /// Number of installed device contexts.
    pub(crate) fn resource_profile(&self, device_ordinal: usize) -> Option<&DeviceResourceProfile> {
        self.resource_profiles.iter().find(|profile| profile.device_ordinal == device_ordinal)
    }

    /// Whether one installed device can execute every immutable capability
    /// requirement of an attempt. This reads only the profile frozen during
    /// installation; it never consults live CUDA state.
    pub(crate) fn supports_capabilities(
        &self,
        device_ordinal: usize,
        requirements: &[crate::capability::ReservationRequirement],
    ) -> bool {
        self.resource_profile(device_ordinal)
            .is_some_and(|profile| profile.device_capabilities.supports_all(requirements))
    }

    pub fn len(&self) -> usize {
        self.device_ordinals.len()
    }

    /// Returns whether the registry has no installed devices.
    pub fn is_empty(&self) -> bool {
        self.device_ordinals.is_empty()
    }

    /// Returns whether any candidate device has enough configured capacity for
    /// this query budget, ignoring current active grants.
    pub(crate) fn can_fit_query_budget(
        &self,
        policy: &dyn PlacementPolicy,
        request: &PlacementRequest,
    ) -> Result<bool> {
        for device in policy.candidates(self, request)? {
            let context = self.context(device)?;
            let required_floor_bytes = request.required_floor_bytes_for(context.ledger())?;
            if context.ledger().can_ever_fit_required_floor_bytes(required_floor_bytes)? {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

fn bind_engine_device_health(
    device: cudf::DeviceId,
    installation: cudf::DeviceMemoryManagerInstallationState,
    backend_health_generation: u64,
) -> Result<EngineDeviceHealthIdentity> {
    let health = cudf::runtime::device_health_snapshot(device);
    if health.status() == cudf::runtime::DeviceHealthStatus::Unhealthy {
        return Err(QueryError::new(
            QueryErrorCode::CudfPersistentFailure,
            "nexus_query_engine.gpu.device_registry.health",
            "unrecoverable CUDA device health prevents backend publication",
        )
        .with_fact("device_ordinal", device.ordinal())
        .with_fact("reason", "persistent_device_unhealthy")
        .with_fact("fault_count", health.fault_count()));
    }
    let installation_generation = installation.installation_generation.ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::Internal,
            "nexus_query_engine.gpu.device_registry.health",
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
            "nexus_query_engine.gpu.device_registry.health",
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
                "nexus_query_engine.gpu.device_registry.health",
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

fn managed_capacity_bytes(
    device_ordinal: usize,
    facts: &super::DeviceFacts,
    profile: &DeviceRegistryInstallProfile,
) -> Result<u64> {
    let allocation_limit =
        profile.memory_partition.allocation_limit_bytes().and_then(|limit| u64::try_from(limit).ok());
    if allocation_limit.is_some_and(|limit| limit > facts.total_bytes) {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.gpu.device_registry.capacity",
            "static partition cannot exceed discovered physical device memory",
        )
        .with_fact("device_ordinal", device_ordinal)
        .with_fact("allocation_limit_bytes", allocation_limit.unwrap_or_default())
        .with_fact("physical_total_bytes", facts.total_bytes)
        .with_fact("reason", "static_partition_exceeds_physical_capacity"));
    }
    let owned_capacity_bytes = allocation_limit.unwrap_or(facts.total_bytes);
    let configured_limit = match profile.capacity_policy {
        DeviceRegistryCapacityPolicy::Automatic { headroom_bytes, fallback_bytes, fraction, maximum_bytes } => {
            // For a static partition, automatic sizing starts from the owned
            // partition rather than physical VRAM. Headroom must reduce the
            // budget Nexus actually owns, not a larger whole-device number.
            let headroom_capacity = owned_capacity_bytes
                .checked_sub(headroom_bytes)
                .filter(|value| *value > 0)
                .unwrap_or(fallback_bytes.min(owned_capacity_bytes));
            let automatic = fraction.map_or(headroom_capacity, |fraction| {
                headroom_capacity.min(fraction.capacity_bytes(owned_capacity_bytes))
            });
            maximum_bytes.map_or(automatic, |maximum| automatic.min(maximum))
        },
        DeviceRegistryCapacityPolicy::Explicit { limit_bytes } => {
            if limit_bytes > owned_capacity_bytes {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.gpu.device_registry.capacity",
                    "explicit managed capacity cannot exceed owned device memory",
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
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.gpu.device_registry.capacity",
            "backend reserve exhausts the device's usable managed capacity",
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
    let identity = cudf::DeviceMemoryManagerInstallationIdentity::new(
        *b"nexus-query-eng\0",
        1,
        partition_fingerprint(device.ordinal(), options.allocation_limit_bytes),
    );
    let (manager, outcome) =
        match cudf::DeviceMemoryManager::try_install_for_device_with_owned_failure_callback_details_and_options(
            device,
            identity,
            memory_failure_tracker.failure_callback(),
            options,
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
                    && state.cause == cudf::DeviceMemoryManagerInstallationCause::OptionsMismatch
                {
                    return Err(device_memory_manager_options_conflict(device, state, options)
                        .with_source(failure.into_error()));
                }

                let error = QueryError::from_cudf(
                    "nexus_query_engine.gpu.device_registry.memory_manager",
                    "failed to install or reuse the Nexus-owned device memory manager",
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
    if outcome.state.status != cudf::DeviceMemoryManagerInstallationStatus::Owned
        || outcome.state.retry != cudf::DeviceMemoryManagerInstallationRetry::Reusable
    {
        if !callback_was_installed {
            memory_failure_tracker.mark_callback_install_unavailable();
        }
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.gpu.device_registry.memory_manager",
            "device memory manager installation is not owned and reusable by Nexus",
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
        "nexus_query_engine.gpu.device_registry.memory_manager",
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

/// P2 hard query domains require cuDF's canonical stream-ordered async base.
/// This stays in the registry because allocation-manager installation is a
/// process-owned backend concern, not a caller-selectable execution policy.
fn canonical_device_memory_manager_options(allocation_limit_bytes: Option<usize>) -> cudf::DeviceMemoryManagerOptions {
    cudf::DeviceMemoryManagerOptions { allocation_limit_bytes, ..Default::default() }
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

fn checkpoint(
    probe: Option<&dyn GpuBackendConstructionProbe>,
    construction_checkpoint: GpuBackendConstructionCheckpoint,
) -> Result<()> {
    match probe {
        Some(probe) => probe.checkpoint(construction_checkpoint),
        None => Ok(()),
    }
}

fn validate_registry_config(config: &DeviceRegistryConfig) -> Result<()> {
    if config.stream_pool_size == 0 {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.gpu.device_registry.config",
            "device registry stream pool size must be greater than zero",
        )
        .with_fact("field", "stream_pool_size")
        .with_fact("reason", "zero"));
    }
    if config.fallback_capacity_bytes == 0 {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.gpu.device_registry.config",
            "device registry fallback capacity must be greater than zero",
        )
        .with_fact("field", "fallback_capacity_bytes")
        .with_fact("reason", "zero"));
    }
    if let Some(device_capacity_bytes) = config.device_capacity_bytes
        && device_capacity_bytes == 0
    {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.gpu.device_registry.config",
            "device registry capacity cap must be greater than zero",
        )
        .with_fact("field", "device_capacity_bytes")
        .with_fact("reason", "zero"));
    }
    for (ordinal, profile) in &config.device_profiles {
        if profile.stream_pool_size == 0 {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu.device_registry.config",
                "per-device stream pool size must be greater than zero",
            )
            .with_fact("field", "device_profiles.stream_pool_size")
            .with_fact("device_ordinal", *ordinal)
            .with_fact("reason", "zero"));
        }
        if profile.max_active_queries == Some(0) {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu.device_registry.config",
                "per-device active-query limit must be greater than zero",
            )
            .with_fact("field", "device_profiles.max_active_queries")
            .with_fact("device_ordinal", *ordinal)
            .with_fact("reason", "zero"));
        }
        match profile.capacity_policy {
            DeviceRegistryCapacityPolicy::Automatic { fallback_bytes: 0, .. } => {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.gpu.device_registry.config",
                    "per-device fallback capacity must be greater than zero",
                )
                .with_fact("field", "device_profiles.fallback_capacity_bytes")
                .with_fact("device_ordinal", *ordinal)
                .with_fact("reason", "zero"));
            },
            DeviceRegistryCapacityPolicy::Automatic { maximum_bytes: Some(0), .. }
            | DeviceRegistryCapacityPolicy::Explicit { limit_bytes: 0 } => {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.gpu.device_registry.config",
                    "per-device configured memory limit must be greater than zero",
                )
                .with_fact("field", "device_profiles.memory_limit")
                .with_fact("device_ordinal", *ordinal)
                .with_fact("reason", "zero"));
            },
            _ => {},
        }
        canonical_device_memory_manager_options(profile.memory_partition.allocation_limit_bytes()).validate().map_err(
            |error| {
                QueryError::from_cudf(
                    "nexus_query_engine.gpu.device_registry.config",
                    "invalid per-device memory manager options",
                    error,
                )
                .with_fact("field", "device_profiles.device_memory_manager_options")
                .with_fact("device_ordinal", *ordinal)
                .with_fact("reason", "invalid_device_memory_manager_options")
            },
        )?;
    }
    Ok(())
}

fn configured_ordinals(inventory: &dyn GpuInventory, config: &DeviceRegistryConfig) -> Result<Vec<usize>> {
    let mut ordinals = if config.device_ordinals.is_empty() {
        let device_count = inventory.device_count()?;
        (0..device_count).collect::<Vec<_>>()
    } else {
        config.device_ordinals.clone()
    };
    if ordinals.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.gpu.device_registry.config",
            "device registry requires at least one CUDA device",
        )
        .with_fact("field", "device_ordinals")
        .with_fact("reason", "empty"));
    }
    let original_len = ordinals.len();
    ordinals.sort_unstable();
    ordinals.dedup();
    if !config.device_ordinals.is_empty() && ordinals.len() != original_len {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.gpu.device_registry.config",
            "device registry ordinals must not contain duplicates",
        )
        .with_fact("field", "device_ordinals")
        .with_fact("reason", "duplicate"));
    }
    Ok(ordinals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::DeviceFacts;

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
        fn device_count(&self) -> Result<usize> {
            Ok(self.facts.len())
        }

        fn facts(&self, device_ordinal: usize) -> Result<DeviceFacts> {
            self.facts.get(device_ordinal).cloned().ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.gpu.device_registry.test_inventory",
                    "missing fake device facts",
                )
            })
        }
    }

    #[test]
    fn configured_ordinals_default_to_all_inventory_devices() {
        let inventory = FakeInventory {
            facts: vec![
                DeviceFacts { device: 0, total_bytes: 100, free_bytes: 50, sm_count: Some(1), utilization_pct: None },
                DeviceFacts { device: 1, total_bytes: 100, free_bytes: 50, sm_count: Some(1), utilization_pct: None },
            ],
        };
        let config = DeviceRegistryConfig::new(Vec::new(), 1);

        assert_eq!(configured_ordinals(&inventory, &config).unwrap(), vec![0, 1]);
    }

    #[test]
    fn configured_ordinals_reject_duplicates() {
        let inventory = FakeInventory {
            facts: vec![DeviceFacts {
                device: 0,
                total_bytes: 100,
                free_bytes: 50,
                sm_count: Some(1),
                utilization_pct: None,
            }],
        };
        let config = DeviceRegistryConfig::new(vec![0, 0], 1);

        let error = configured_ordinals(&inventory, &config).unwrap_err();

        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
        assert_eq!(error.fact_value("field"), Some("device_ordinals"));
        assert_eq!(error.fact_value("reason"), Some("duplicate"));
    }

    #[test]
    fn configured_ordinals_preserve_explicit_values_for_checkpointed_discovery() {
        let inventory = FakeInventory {
            facts: vec![DeviceFacts {
                device: 0,
                total_bytes: 100,
                free_bytes: 50,
                sm_count: Some(1),
                utilization_pct: None,
            }],
        };
        let config = DeviceRegistryConfig::new(vec![usize::MAX], 1);

        assert_eq!(configured_ordinals(&inventory, &config).unwrap(), vec![usize::MAX]);
    }

    #[test]
    fn resolving_complete_plan_set_has_no_installation_checkpoint() {
        let inventory = FakeInventory {
            facts: vec![
                DeviceFacts {
                    device: 0,
                    total_bytes: 1_000,
                    free_bytes: 500,
                    sm_count: Some(1),
                    utilization_pct: None,
                },
                DeviceFacts {
                    device: 1,
                    total_bytes: 2_000,
                    free_bytes: 500,
                    sm_count: Some(1),
                    utilization_pct: None,
                },
            ],
        };
        let config = DeviceRegistryConfig::new(vec![0, 1], 2).with_capacity_policy(100, 200);
        let probe = RecordingProbe::default();

        let resolved = DeviceRegistry::resolve_install_plans(&inventory, &config, Some(&probe)).unwrap();

        assert_eq!(resolved.device_ordinals(), &[0, 1]);
        assert_eq!(resolved.plans().iter().map(|plan| plan.capacity_bytes).collect::<Vec<_>>(), vec![900, 1_900]);
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
                    total_bytes: 1_000,
                    free_bytes: 1_000,
                    sm_count: Some(1),
                    utilization_pct: None,
                },
                DeviceFacts {
                    device: 1,
                    total_bytes: 2_000,
                    free_bytes: 2_000,
                    sm_count: Some(1),
                    utilization_pct: None,
                },
            ],
        };
        let profile = |floor| DeviceRegistryInstallProfile {
            stream_pool_size: 4,
            device_capabilities: DeviceCapabilities::default(),
            max_active_queries: Some(2),
            capacity_policy: DeviceRegistryCapacityPolicy::Automatic {
                headroom_bytes: 100,
                fallback_bytes: 100,
                fraction: None,
                maximum_bytes: None,
            },
            backend_reserve_bytes: 100,
            attempt_floor_bytes: floor,
            source_work_per_attempt: 1,
            worker_job_capacity: 4,
            lane_count: 2,
            memory_partition: DeviceRegistryMemoryPartition::WholeDevice,
        };
        let plans = DeviceRegistry::resolve_install_plans(
            &inventory,
            &DeviceRegistryConfig::new(vec![0, 1], 4)
                .with_device_profile(0, profile(None))
                .with_device_profile(1, profile(None)),
            None,
        )
        .unwrap();
        assert_eq!(plans.plans()[0].resource_profile.managed_capacity_bytes, 800);
        assert_eq!(plans.plans()[1].resource_profile.managed_capacity_bytes, 1_800);
        assert_eq!(plans.plans()[0].resource_profile.attempt_floor_bytes, 100);
        assert_eq!(plans.plans()[1].resource_profile.attempt_floor_bytes, 225);

        let error = DeviceRegistry::resolve_install_plans(
            &inventory,
            &DeviceRegistryConfig::new(vec![0, 1], 4)
                .with_device_profile(0, profile(None))
                .with_device_profile(1, profile(Some(1_000))),
            None,
        )
        .unwrap_err();
        assert_eq!(error.fact_value("device_ordinal"), Some("1"));
        assert_eq!(error.fact_value("reason"), Some("attempt_floor_exceeds_managed_capacity"));
    }

    #[test]
    fn managed_capacities_are_derived_independently_per_device() {
        let whole_facts =
            DeviceFacts { device: 0, total_bytes: 1_000, free_bytes: 100, sm_count: Some(1), utilization_pct: None };
        let whole = DeviceRegistryInstallProfile {
            stream_pool_size: 2,
            device_capabilities: DeviceCapabilities::default(),
            max_active_queries: None,
            capacity_policy: DeviceRegistryCapacityPolicy::Automatic {
                headroom_bytes: 100,
                fallback_bytes: 200,
                fraction: None,
                maximum_bytes: None,
            },
            backend_reserve_bytes: 50,
            attempt_floor_bytes: None,
            source_work_per_attempt: 1,
            worker_job_capacity: 2,
            lane_count: 2,
            memory_partition: DeviceRegistryMemoryPartition::WholeDevice,
        };
        let partition_facts = DeviceFacts { device: 1, total_bytes: 4_000, ..whole_facts.clone() };
        let partition = DeviceRegistryInstallProfile {
            stream_pool_size: 7,
            device_capabilities: DeviceCapabilities::default(),
            max_active_queries: None,
            capacity_policy: DeviceRegistryCapacityPolicy::Automatic {
                headroom_bytes: 100,
                fallback_bytes: 200,
                fraction: None,
                maximum_bytes: None,
            },
            backend_reserve_bytes: 25,
            attempt_floor_bytes: None,
            source_work_per_attempt: 1,
            worker_job_capacity: 7,
            lane_count: 7,
            memory_partition: DeviceRegistryMemoryPartition::StaticPartition { allocation_limit_bytes: 600 },
        };
        let explicit_facts = DeviceFacts { device: 2, total_bytes: 10_000, ..whole_facts.clone() };
        let explicit = DeviceRegistryInstallProfile {
            stream_pool_size: 3,
            device_capabilities: DeviceCapabilities::default(),
            max_active_queries: None,
            capacity_policy: DeviceRegistryCapacityPolicy::Explicit { limit_bytes: 700 },
            backend_reserve_bytes: 10,
            attempt_floor_bytes: None,
            source_work_per_attempt: 1,
            worker_job_capacity: 3,
            lane_count: 3,
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
    fn managed_capacity_never_exceeds_physical_or_owned_memory() {
        let facts =
            DeviceFacts { device: 0, total_bytes: 100, free_bytes: 50, sm_count: Some(1), utilization_pct: None };
        let automatic = DeviceRegistryInstallProfile {
            stream_pool_size: 1,
            device_capabilities: DeviceCapabilities::default(),
            max_active_queries: None,
            capacity_policy: DeviceRegistryCapacityPolicy::Automatic {
                headroom_bytes: 500,
                fallback_bytes: 256,
                fraction: None,
                maximum_bytes: None,
            },
            backend_reserve_bytes: 0,
            attempt_floor_bytes: None,
            source_work_per_attempt: 1,
            worker_job_capacity: 1,
            lane_count: 1,
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
}
