//! Multi-tenant service admission harness for native execution.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, VecDeque},
    fmt,
    sync::{
        Arc, Condvar, Mutex, OnceLock, Weak,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use tokio::{sync::Notify, time};

use super::{
    CudfInventory, DeploymentIoCapability, DeviceMemoryFailureReport, ExecutionBoundaryPolicy, ExecutionCancellation,
    ExecutionContext, ExecutionContextParts, GdsConcurrencyGate, GpuInventory, GrantRequest, HostFetchBudget,
    LakehouseCacheFillEventSink, MemoryControlPolicy, MemoryGovernor, MemoryGovernorReport, MemoryGrant,
    MemoryResourcePolicy, PlacementPolicy, PlacementRequest, Priority, QueryPlacementPolicy, QueryQos,
    SourceChunkPlanningMode, SourceFetchConcurrencyGate, SourceFetchConfig, SourceObjectStoreRegistry,
    SourceSchedulerPolicy, SourceStorageProfile, default_host_fetch_budget_bytes, default_lakehouse_cache_fill_runtime,
    default_service_stream_pool_size_for_device, default_source_fetch_runtime, gpu::DeviceActiveQuerySlot,
    validate_service_stream_pool_device,
};
use crate::{
    adaptive_runtime::AdaptiveMemoryHistory,
    capability::{AggregateStrategyExecutionOptions, KeyedHashJoinStrategyOptions},
    chunk::SourceChunkPlanningOptions,
    dynamic_filter::DynamicFilterExecutionOptions,
    error::{QueryError, QueryErrorCode, Result},
    exec::GpuLocalParquetSourceCache,
    metrics::QueryId,
    mode::{NativePipelineExecutionMode, StddevSampExecutionMode},
    util::comma_separated_usize,
};

mod admission;
mod config;
mod handle;
#[cfg(test)]
mod tests;

pub use admission::QueryAdmissionWaiter;
use admission::{
    ActiveAdaptiveGrant, AdaptiveGrowthCandidate, AdmissionRequest, AdmissionState, PlacedQuery, remove_queued_request,
    service_closed_error, service_overloaded_error, weighted_adaptive_growth_allocations,
};
#[cfg(test)]
use config::guaranteed_min_slot_capacity_for_capacities;
pub use config::{QueryServiceConfig, QueryServiceTenantConfig, SlotGuarantee, TenantId};
use config::{
    admission_capacity_bounds, default_stream_pool_size_for_ordinals, guaranteed_min_slot_capacity,
    minimum_stream_pool_size, placement_policy_admits_ordinal, validate_stream_pool_admission_capacity,
};
use handle::QueryHandleLifetime;
pub use handle::{QueryHandle, TenantQueryContext};

/// Service-level entry point for tenant-aware native query admission.
#[derive(Clone, Debug)]
pub struct QueryService {
    inner: Arc<QueryServiceInner>,
}

/// Point-in-time service admission state for fairness and overload reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryServiceAdmissionSnapshot {
    /// Number of currently admitted query handles.
    pub active_queries: usize,
    /// Number of admissions currently waiting in the service queue.
    pub queued_queries: usize,
    /// Queue depth by configured tenant, in admission rotation order.
    pub queued_by_tenant: Vec<TenantQueueDepth>,
    /// Tenant that will be considered first on the next admission decision.
    pub next_tenant_id: Option<TenantId>,
}

/// Per-tenant queue depth in a [`QueryServiceAdmissionSnapshot`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TenantQueueDepth {
    /// Tenant identifier.
    pub tenant_id: TenantId,
    /// Number of queued admission requests for this tenant.
    pub queued_queries: usize,
}

/// Request-scoped query-service grant overrides.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct QueryGrantRequest {
    /// Optional per-request minimum grant floor in bytes.
    pub requested_grant_min_bytes: Option<u64>,
    /// Optional per-request preferred grant in bytes.
    pub requested_grant_bytes: Option<u64>,
}

impl QueryGrantRequest {
    /// Creates an empty request that uses the tenant grant band unchanged.
    pub const fn new() -> Self {
        Self { requested_grant_min_bytes: None, requested_grant_bytes: None }
    }

    /// Creates a request with only a preferred grant override.
    pub const fn with_requested_grant_bytes(requested_grant_bytes: u64) -> Self {
        Self { requested_grant_min_bytes: None, requested_grant_bytes: Some(requested_grant_bytes) }
    }

    /// Creates a request with only a minimum grant floor override.
    pub const fn with_requested_grant_min_bytes(requested_grant_min_bytes: u64) -> Self {
        Self { requested_grant_min_bytes: Some(requested_grant_min_bytes), requested_grant_bytes: None }
    }

    /// Sets the per-request preferred grant override.
    #[must_use]
    pub const fn prefer(mut self, requested_grant_bytes: u64) -> Self {
        self.requested_grant_bytes = Some(requested_grant_bytes);
        self
    }

    /// Sets the per-request minimum grant floor override.
    #[must_use]
    pub const fn floor(mut self, requested_grant_min_bytes: u64) -> Self {
        self.requested_grant_min_bytes = Some(requested_grant_min_bytes);
        self
    }
}

/// Effective grant band for one query-service tenant request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryGrantBand {
    /// Minimum grant bytes admission must provide before the query can run.
    pub min_bytes: u64,
    /// Preferred grant bytes admission tries to provide when capacity allows.
    pub want_bytes: u64,
}

#[derive(Debug)]
struct QueryServiceInner {
    config: QueryServiceConfig,
    tenant_grants: HashMap<TenantId, TenantGrantPolicy>,
    tenant_order: Vec<TenantId>,
    max_tenant_min_bytes: u64,
    fallback_memory_governor: MemoryGovernor,
    resources: Option<QueryServiceResources>,
    admission: Mutex<AdmissionState>,
    admission_available: Condvar,
    admission_notify: Notify,
    next_query_id: AtomicU64,
    next_admission_sequence: AtomicU64,
}

impl QueryServiceInner {
    fn notify_admission_available(&self) {
        self.admission_available.notify_all();
        self.admission_notify.notify_waiters();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TenantGrantPolicy {
    min_bytes: u64,
    want_bytes: u64,
}

impl TenantGrantPolicy {
    fn with_requested_grant_bytes(self, requested_grant_bytes: u64) -> Self {
        Self { want_bytes: requested_grant_bytes, ..self }
    }

    fn with_requested_grant_min_bytes(self, requested_grant_min_bytes: u64) -> Self {
        Self { min_bytes: requested_grant_min_bytes, ..self }
    }
}

impl From<TenantGrantPolicy> for QueryGrantBand {
    fn from(policy: TenantGrantPolicy) -> Self {
        Self { min_bytes: policy.min_bytes, want_bytes: policy.want_bytes }
    }
}

#[derive(Debug, Clone)]
struct QueryServiceResources {
    devices: super::DeviceRegistry,
    _governor_claim: Option<GovernorClaimLease>,
    host_fetch_budget: HostFetchBudget,
    source_fetch_concurrency: SourceFetchConcurrencyGate,
    lakehouse_cache_fill_budget: HostFetchBudget,
    lakehouse_cache_fill_concurrency: SourceFetchConcurrencyGate,
    gds_concurrency: GdsConcurrencyGate,
    source_fetch_runtime: Arc<tokio::runtime::Runtime>,
    lakehouse_cache_fill_runtime: Arc<tokio::runtime::Runtime>,
    source_object_stores: Arc<SourceObjectStoreRegistry>,
    deployment_io_capability: DeploymentIoCapability,
    local_parquet_source_caches: HashMap<usize, Arc<GpuLocalParquetSourceCache>>,
}

impl QueryServiceResources {
    fn local_parquet_source_cache_for_device(&self, device_ordinal: usize) -> Option<Arc<GpuLocalParquetSourceCache>> {
        self.local_parquet_source_caches.get(&device_ordinal).map(Arc::clone)
    }
}

#[derive(Debug, Clone)]
struct GovernorClaimLease {
    _inner: Arc<GovernorClaimLeaseInner>,
}

#[derive(Debug)]
struct GovernorClaimLeaseInner {
    ordinals: Vec<usize>,
    slot_guarantee: SlotGuarantee,
}

#[derive(Debug, Default)]
struct GovernorClaimState {
    optimistic_claims: usize,
    guaranteed_minimum: bool,
}

impl Drop for GovernorClaimLeaseInner {
    fn drop(&mut self) {
        let claims = governor_claims();
        let Ok(mut claims) = claims.lock() else {
            eprintln!("[nexus-query-engine] warning: failed to release query-service governor claims");
            return;
        };
        for ordinal in &self.ordinals {
            let Some(state) = claims.get_mut(ordinal) else {
                continue;
            };
            match self.slot_guarantee {
                SlotGuarantee::GuaranteedMinimum => state.guaranteed_minimum = false,
                SlotGuarantee::Optimistic => state.optimistic_claims = state.optimistic_claims.saturating_sub(1),
            }
            if state.optimistic_claims == 0 && !state.guaranteed_minimum {
                claims.remove(ordinal);
            }
        }
    }
}

fn governor_claims() -> &'static Mutex<BTreeMap<usize, GovernorClaimState>> {
    static CLAIMS: OnceLock<Mutex<BTreeMap<usize, GovernorClaimState>>> = OnceLock::new();
    CLAIMS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn claim_governor_ordinals(ordinals: &[usize], slot_guarantee: SlotGuarantee) -> Result<GovernorClaimLease> {
    let claims = governor_claims();
    let mut claims = claims.lock().map_err(|_error| {
        QueryError::new(
            QueryErrorCode::Internal,
            "nexus_query_engine.query_service.governor_claim",
            "query-service governor claim registry mutex was poisoned",
        )
    })?;
    if let Some(conflict) = ordinals
        .iter()
        .find(|ordinal| claims.get(ordinal).is_some_and(|state| governor_claim_conflicts(state, slot_guarantee)))
    {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.query_service.tenant.validate",
            "query-service slot guarantee cannot share an active device memory governor claim",
        )
        .with_fact("field", "slot_guarantee")
        .with_fact("reason", "shared_governor_guaranteed_minimum_conflict")
        .with_fact("device_ordinal", *conflict)
        .with_fact("device_ordinals", comma_separated_usize(ordinals))
        .with_fact("slot_guarantee", slot_guarantee.as_str()));
    }
    for ordinal in ordinals {
        let state = claims.entry(*ordinal).or_default();
        match slot_guarantee {
            SlotGuarantee::GuaranteedMinimum => state.guaranteed_minimum = true,
            SlotGuarantee::Optimistic => state.optimistic_claims = state.optimistic_claims.saturating_add(1),
        }
    }
    Ok(GovernorClaimLease { _inner: Arc::new(GovernorClaimLeaseInner { ordinals: ordinals.to_vec(), slot_guarantee }) })
}

fn governor_claim_conflicts(state: &GovernorClaimState, requested: SlotGuarantee) -> bool {
    match requested {
        SlotGuarantee::GuaranteedMinimum => state.guaranteed_minimum || state.optimistic_claims > 0,
        SlotGuarantee::Optimistic => state.guaranteed_minimum,
    }
}

fn validate_placement_policy_for_resources(
    config: &QueryServiceConfig,
    resources: Option<&QueryServiceResources>,
) -> Result<()> {
    let QueryPlacementPolicy::Pinned { device_ordinal } = config.placement_policy else {
        return Ok(());
    };
    let valid = resources.map_or(config.device_ordinal == device_ordinal, |resources| {
        resources.devices.context_by_ordinal(device_ordinal).is_ok()
    });
    if valid {
        return Ok(());
    }
    Err(QueryError::new(
        QueryErrorCode::InvalidPlan,
        "nexus_query_engine.query_service.config.validate",
        "pinned query placement device must be one of the query service devices",
    )
    .with_fact("field", "placement_policy")
    .with_fact("reason", "pinned_device_not_configured")
    .with_fact("pinned_device_ordinal", device_ordinal))
}

fn register_shared_device_admission_waiters(inner: &Arc<QueryServiceInner>) -> Result<()> {
    let Some(resources) = &inner.resources else {
        return Ok(());
    };
    let mut waiters = shared_device_admission_waiters().lock().map_err(|_error| {
        QueryError::new(
            QueryErrorCode::Internal,
            "nexus_query_engine.query_service.admission_waiters",
            "shared device admission waiter registry mutex was poisoned",
        )
    })?;
    for ordinal in resources.devices.device_ordinals() {
        let waiters_for_device = waiters.entry(*ordinal).or_default();
        prune_dead_shared_device_admission_waiters(waiters_for_device);
        waiters_for_device.push(Arc::downgrade(inner));
    }
    Ok(())
}

fn shared_device_admission_waiters() -> &'static Mutex<HashMap<usize, Vec<Weak<QueryServiceInner>>>> {
    static WAITERS: OnceLock<Mutex<HashMap<usize, Vec<Weak<QueryServiceInner>>>>> = OnceLock::new();
    WAITERS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn prune_dead_shared_device_admission_waiters(waiters_for_device: &mut Vec<Weak<QueryServiceInner>>) {
    waiters_for_device.retain(|weak| weak.strong_count() > 0);
}

fn notify_shared_device_admission_available(device_ordinal: usize) {
    let services = match shared_device_admission_waiters().lock() {
        Ok(mut waiters) => {
            let Some(waiters_for_device) = waiters.get_mut(&device_ordinal) else {
                return;
            };
            let mut services = Vec::with_capacity(waiters_for_device.len());
            waiters_for_device.retain(|weak| {
                if let Some(service) = weak.upgrade() {
                    services.push(service);
                    true
                } else {
                    false
                }
            });
            services
        },
        Err(_error) => {
            eprintln!(
                "[nexus-query-engine] warning: failed to lock shared device admission waiter registry for wakeup"
            );
            return;
        },
    };
    for service in services {
        let Ok(_guard) = service.admission.lock() else {
            eprintln!("[nexus-query-engine] warning: failed to lock query service admission for shared device wakeup");
            continue;
        };
        service.notify_admission_available();
    }
}

#[derive(Debug, Default)]
struct PendingDeviceSlotNotifications {
    device_ordinals: Vec<usize>,
}

impl PendingDeviceSlotNotifications {
    fn release_slot(&mut self, slot: DeviceActiveQuerySlot) {
        let device_ordinal = slot.device_ordinal();
        drop(slot);
        self.device_ordinals.push(device_ordinal);
    }

    fn has_pending(&self) -> bool {
        !self.device_ordinals.is_empty()
    }

    fn notify(&mut self) {
        self.device_ordinals.sort_unstable();
        self.device_ordinals.dedup();
        for device_ordinal in self.device_ordinals.drain(..) {
            notify_shared_device_admission_available(device_ordinal);
        }
    }
}

impl Drop for PendingDeviceSlotNotifications {
    fn drop(&mut self) {
        self.notify();
    }
}

impl QueryService {
    /// Returns true when both handles point at the same service instance.
    pub fn same_service(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Creates a service entry point and initializes service-owned runtime gates.
    pub fn new(config: QueryServiceConfig, tenants: Vec<QueryServiceTenantConfig>) -> Result<Self> {
        let device_ordinals = vec![config.device_ordinal];
        Self::new_with_device_ordinals(config, device_ordinals, tenants)
    }

    /// Creates a service entry point over an explicit set of CUDA device ordinals.
    pub fn new_with_device_ordinals(
        config: QueryServiceConfig,
        device_ordinals: Vec<usize>,
        tenants: Vec<QueryServiceTenantConfig>,
    ) -> Result<Self> {
        let deployment = DeploymentIoCapability::cached_current_host();
        Self::new_with_device_ordinals_and_deployment(config, device_ordinals, tenants, deployment)
    }

    /// Creates a service entry point with caller-supplied deployment diagnostics.
    pub fn new_with_deployment(
        config: QueryServiceConfig,
        tenants: Vec<QueryServiceTenantConfig>,
        deployment_io_capability: DeploymentIoCapability,
    ) -> Result<Self> {
        let device_ordinals = vec![config.device_ordinal];
        Self::new_with_device_ordinals_and_deployment(config, device_ordinals, tenants, deployment_io_capability)
    }

    /// Creates a service entry point over explicit devices with caller-supplied deployment diagnostics.
    pub fn new_with_device_ordinals_and_deployment(
        config: QueryServiceConfig,
        device_ordinals: Vec<usize>,
        tenants: Vec<QueryServiceTenantConfig>,
        deployment_io_capability: DeploymentIoCapability,
    ) -> Result<Self> {
        let default_stream_pool_size = default_stream_pool_size_for_ordinals(&device_ordinals)?;
        let minimum_stream_pool_size = minimum_stream_pool_size(
            config.max_concurrent_queries,
            config.source_scheduler_policy.max_source_reads_in_flight(),
        );
        let stream_pool_size =
            config.stream_pool_size.unwrap_or_else(|| default_stream_pool_size.max(minimum_stream_pool_size));
        validate_stream_pool_admission_capacity(
            stream_pool_size,
            config.max_concurrent_queries,
            config.source_scheduler_policy.max_source_reads_in_flight(),
        )?;
        let mut registry_config = super::DeviceRegistryConfig::new(device_ordinals, stream_pool_size);
        if let Some(device_capacity_bytes) = config.device_capacity_bytes {
            registry_config = registry_config.with_device_capacity_bytes(device_capacity_bytes);
        }
        let devices = super::DeviceRegistry::install(&super::CudfInventory, &registry_config)?;
        let local_parquet_source_caches = local_parquet_source_caches_for_ordinals(
            devices.device_ordinals(),
            config.local_parquet_source_cache_max_retained_bytes,
        );
        let resources = QueryServiceResources {
            devices,
            _governor_claim: None,
            host_fetch_budget: HostFetchBudget::new(
                config.source_fetch.host_fetch_budget_bytes.unwrap_or_else(default_host_fetch_budget_bytes),
            )?,
            source_fetch_concurrency: SourceFetchConcurrencyGate::new(
                config.source_fetch.host_fetch_max_concurrent_ops,
            )?,
            lakehouse_cache_fill_budget: HostFetchBudget::new(
                config.source_fetch.lakehouse_cache_fill_budget_bytes.unwrap_or_else(default_host_fetch_budget_bytes),
            )?,
            lakehouse_cache_fill_concurrency: SourceFetchConcurrencyGate::new(
                config.source_fetch.lakehouse_cache_fill_max_concurrent_ops,
            )?,
            gds_concurrency: GdsConcurrencyGate::new(config.source_fetch.gds_max_concurrent_ops)?,
            source_fetch_runtime: default_source_fetch_runtime()?,
            lakehouse_cache_fill_runtime: default_lakehouse_cache_fill_runtime()?,
            source_object_stores: Arc::new(SourceObjectStoreRegistry::with_profiles(
                config.source_storage_profiles.clone(),
            )?),
            deployment_io_capability,
            local_parquet_source_caches,
        };
        Self::build(config, tenants, Some(resources))
    }

    /// Creates a service entry point without CUDA/runtime resources.
    ///
    /// This constructor is intended for admission-only embedders and tests that
    /// need the service queue, tenant fairness, and memory-governor ledger
    /// without touching a GPU. Admitted handles from such a service can report
    /// their grants and release admission normally, but
    /// [`QueryHandle::acquire_context`] returns an error because no runtime
    /// resources were installed.
    pub fn new_admission_only(
        config: QueryServiceConfig,
        tenants: Vec<QueryServiceTenantConfig>,
        deployment_io_capability: DeploymentIoCapability,
    ) -> Result<Self> {
        Self::build_with_deployment(config, tenants, None, deployment_io_capability)
    }

    #[cfg(test)]
    fn new_for_admission_tests(
        config: QueryServiceConfig,
        tenants: Vec<QueryServiceTenantConfig>,
        deployment_io_capability: DeploymentIoCapability,
    ) -> Result<Self> {
        Self::new_admission_only(config, tenants, deployment_io_capability)
    }

    fn build(
        config: QueryServiceConfig,
        tenants: Vec<QueryServiceTenantConfig>,
        resources: Option<QueryServiceResources>,
    ) -> Result<Self> {
        let deployment =
            resources.as_ref().map(|resources| &resources.deployment_io_capability).cloned().unwrap_or_else(|| {
                DeploymentIoCapability::from_probe_inputs(
                    super::NvidiaFsProbeStatus::DriverMissing,
                    super::KvikioCompatModeStatus::Unset,
                    super::GdsDriverProbeStatus::ProbeUnimplemented,
                )
            });
        Self::build_with_deployment(config, tenants, resources, deployment)
    }

    fn build_with_deployment(
        config: QueryServiceConfig,
        tenants: Vec<QueryServiceTenantConfig>,
        resources: Option<QueryServiceResources>,
        deployment: DeploymentIoCapability,
    ) -> Result<Self> {
        let mut resources = resources;
        config.validate(&deployment)?;
        validate_placement_policy_for_resources(&config, resources.as_ref())?;
        if tenants.is_empty() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.query_service.tenant.validate",
                "query service requires at least one tenant",
            )
            .with_fact("field", "tenants")
            .with_fact("reason", "empty"));
        }

        let mut tenant_grants = HashMap::with_capacity(tenants.len());
        let mut tenant_order = Vec::with_capacity(tenants.len());
        let mut max_tenant_want_bytes = 0u64;
        let mut max_tenant_min_bytes = 0u64;
        for tenant in tenants {
            tenant.validate()?;
            max_tenant_want_bytes = max_tenant_want_bytes.max(tenant.device_budget_bytes);
            max_tenant_min_bytes = max_tenant_min_bytes.max(tenant.min_device_budget_bytes);
            if tenant_grants
                .insert(
                    tenant.tenant_id.clone(),
                    TenantGrantPolicy {
                        min_bytes: tenant.min_device_budget_bytes,
                        want_bytes: tenant.device_budget_bytes,
                    },
                )
                .is_some()
            {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.query_service.tenant.validate",
                    "query service tenant ids must be unique",
                )
                .with_fact("tenant_id", tenant.tenant_id.as_str())
                .with_fact("reason", "duplicate"));
            }
            tenant_order.push(tenant.tenant_id);
        }
        let max_concurrent_queries_u64 = u64::try_from(config.max_concurrent_queries).unwrap_or(u64::MAX);
        debug_assert!(
            config.device_capacity_bytes.is_some()
                || tenant_grants.values().all(|grant| grant.min_bytes == grant.want_bytes),
            "query-service grant ranges require explicit device_capacity_bytes; the compatibility capacity fallback is only for fixed-slice callers"
        );
        let compatibility_capacity = max_tenant_want_bytes.saturating_mul(max_concurrent_queries_u64).max(1);
        let device_capacity_bytes = config.device_capacity_bytes.unwrap_or(compatibility_capacity);
        if device_capacity_bytes == 0 {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.query_service.config.validate",
                "query service device capacity bytes must be greater than zero",
            )
            .with_fact("field", "device_capacity_bytes")
            .with_fact("reason", "zero"));
        }
        let (max_admissible_device_capacity_bytes, total_admissible_capacity_bytes) =
            admission_capacity_bounds(resources.as_ref(), device_capacity_bytes, config.placement_policy);
        for (tenant_id, grant) in &tenant_grants {
            if grant.min_bytes > max_admissible_device_capacity_bytes {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.query_service.tenant.validate",
                    "tenant minimum device budget exceeds memory governor capacity",
                )
                .with_fact("tenant_id", tenant_id.as_str())
                .with_fact("field", "min_device_budget_bytes")
                .with_fact("reason", "min_device_budget_above_capacity")
                .with_fact("device_capacity_bytes", max_admissible_device_capacity_bytes));
            }
        }
        if config.slot_guarantee == SlotGuarantee::GuaranteedMinimum {
            let guaranteed_min_slots = guaranteed_min_slot_capacity(
                resources.as_ref(),
                device_capacity_bytes,
                max_tenant_min_bytes,
                config.max_queries_per_device,
                config.placement_policy,
            );
            if guaranteed_min_slots < config.max_concurrent_queries {
                let guaranteed_min_capacity = max_tenant_min_bytes.saturating_mul(max_concurrent_queries_u64);
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.query_service.tenant.validate",
                    "tenant minimum device budgets exceed guaranteed slot capacity",
                )
                .with_fact("field", "min_device_budget_bytes")
                .with_fact("reason", "tenant_minima_exceed_guaranteed_slot_capacity")
                .with_fact("max_tenant_min_device_budget_bytes", max_tenant_min_bytes)
                .with_fact("max_concurrent_queries", config.max_concurrent_queries)
                .with_fact("required_capacity_bytes", guaranteed_min_capacity)
                .with_fact("device_capacity_bytes", total_admissible_capacity_bytes)
                .with_fact("guaranteed_min_slot_capacity", guaranteed_min_slots));
            }
        }
        if let Some(resources) = resources.as_mut() {
            let claim_ordinals = resources
                .devices
                .device_ordinals()
                .iter()
                .copied()
                .filter(|ordinal| placement_policy_admits_ordinal(config.placement_policy, *ordinal))
                .collect::<Vec<_>>();
            resources._governor_claim = Some(claim_governor_ordinals(&claim_ordinals, config.slot_guarantee)?);
        }
        let fallback_memory_governor = MemoryGovernor::new(config.device_ordinal, device_capacity_bytes)?;

        let inner = Arc::new(QueryServiceInner {
            config,
            tenant_grants,
            tenant_order,
            max_tenant_min_bytes,
            fallback_memory_governor,
            resources,
            admission: Mutex::new(AdmissionState::default()),
            admission_available: Condvar::new(),
            admission_notify: Notify::new(),
            next_query_id: AtomicU64::new(1),
            next_admission_sequence: AtomicU64::new(1),
        });
        register_shared_device_admission_waiters(&inner)?;
        Ok(Self { inner })
    }

    /// Admits a query for `tenant_id` under round-robin tenant fairness.
    ///
    /// The call waits only up to [`QueryServiceConfig::admission_wait_timeout`].
    /// Queue overflow and wait timeout return `service_overloaded` instead of
    /// parking the caller indefinitely.
    pub fn admit_query(&self, tenant_id: impl Into<TenantId>) -> Result<QueryHandle> {
        self.admit_query_with_qos(tenant_id, QueryQos::default())
    }

    /// Admits a query with caller-supplied QoS metadata.
    ///
    /// Priority affects queued admission order only; it does not preempt active
    /// CUDA work or change memory-governor accounting.
    pub fn admit_query_with_qos(&self, tenant_id: impl Into<TenantId>, qos: QueryQos) -> Result<QueryHandle> {
        self.admit_query_with_qos_and_requested_grant_bytes(tenant_id, qos, None)
    }

    /// Admits a query with caller-supplied QoS metadata and an optional explicit
    /// preferred grant.
    ///
    /// `requested_grant_bytes` overrides only the preferred grant for this
    /// request. It must be inside the tenant's configured `[min, want]` band and
    /// no larger than the service device capacity; invalid asks are rejected
    /// instead of silently clamped.
    pub fn admit_query_with_qos_and_requested_grant_bytes(
        &self,
        tenant_id: impl Into<TenantId>,
        qos: QueryQos,
        requested_grant_bytes: Option<u64>,
    ) -> Result<QueryHandle> {
        let grant_request = QueryGrantRequest { requested_grant_bytes, ..QueryGrantRequest::new() };
        self.admit_query_with_qos_and_grant_request(tenant_id, qos, grant_request)
    }

    /// Admits a query with caller-supplied QoS metadata and optional explicit
    /// minimum/preferred grant overrides.
    ///
    /// `requested_grant_min_bytes` raises the per-request floor inside the
    /// tenant's configured `[min, want]` band. Admission either grants at least
    /// that floor or queues the request; it never admits the request below the
    /// floor. `requested_grant_bytes` overrides only the preferred grant. Invalid
    /// request bands are rejected instead of silently clamped.
    pub fn admit_query_with_qos_and_grant_request(
        &self,
        tenant_id: impl Into<TenantId>,
        qos: QueryQos,
        grant_request: QueryGrantRequest,
    ) -> Result<QueryHandle> {
        let tenant_id = tenant_id.into();
        let grant_policy = self.tenant_grant_policy_for_request(&tenant_id, grant_request)?;
        let request_id = self.inner.next_query_id.fetch_add(1, Ordering::Relaxed);
        let requested_at = Instant::now();
        let deadline = requested_at.checked_add(self.inner.config.admission_wait_timeout).unwrap_or(requested_at);
        let request = AdmissionRequest { request_id, tenant_id: tenant_id.clone(), grant_policy, qos, deadline };

        let mut slot_notifications = PendingDeviceSlotNotifications::default();
        let mut state = self.lock_admission()?;
        if state.closed {
            return Err(service_closed_error(&tenant_id));
        }
        self.sweep_expired_queued_requests_locked(&mut state);
        if state.queued.len() >= self.inner.config.max_admission_queue_len {
            return Err(service_overloaded_error(
                "admission_queue_full",
                &tenant_id,
                state.queued.len(),
                self.inner.config.max_admission_queue_len,
            ));
        }
        state.queued.push_back(request);

        loop {
            if state.closed {
                self.remove_queued_request_without_admission_locked(&mut state, request_id);
                self.inner.notify_admission_available();
                return Err(service_closed_error(&tenant_id));
            }
            let request_is_admissible = match self.request_is_admissible(&mut state, request_id) {
                Ok(admissible) => admissible,
                Err(error) => {
                    self.abandon_queued_request_locked(&mut state, request_id);
                    return Err(error);
                },
            };
            if request_is_admissible
                && let Some(position) = state.queued.iter().position(|queued| queued.request_id == request_id)
            {
                let placed_query = match self.try_place_query(
                    &tenant_id,
                    request_id,
                    grant_policy,
                    qos,
                    Some(deadline),
                    state.active_queries,
                    &mut slot_notifications,
                ) {
                    Ok(placed_query) => placed_query,
                    Err(error) => {
                        self.abandon_queued_request_locked(&mut state, request_id);
                        return Err(error);
                    },
                };
                let Some(placed_query) = placed_query else {
                    if let Err(error) =
                        self.request_adaptive_grant_shrink_for_blocked_request_locked(&state, &state.queued[position])
                    {
                        self.abandon_queued_request_locked(&mut state, request_id);
                        return Err(error);
                    }
                    if slot_notifications.has_pending() {
                        drop(state);
                        slot_notifications.notify();
                        state = self.lock_admission()?;
                        if state.closed {
                            continue;
                        }
                    }
                    let now = Instant::now();
                    if now >= deadline {
                        self.abandon_queued_request_locked(&mut state, request_id);
                        return Err(self.memory_grant_wait_timeout_error(&tenant_id, grant_policy, state.queued.len()));
                    }
                    let wait = deadline.saturating_duration_since(now);
                    let (next_state, timeout) =
                        self.inner.admission_available.wait_timeout(state, wait).map_err(|_error| {
                            QueryError::new(
                                QueryErrorCode::Internal,
                                "nexus_query_engine.query_service.admission",
                                "query service admission mutex was poisoned while waiting",
                            )
                        })?;
                    state = next_state;
                    if timeout.timed_out() {
                        self.abandon_queued_request_locked(&mut state, request_id);
                        return Err(self.memory_grant_wait_timeout_error(&tenant_id, grant_policy, state.queued.len()));
                    }
                    continue;
                };
                let admitted = state.queued.remove(position).expect("admissible request exists");
                self.attach_admission_release_wakeup(&placed_query.memory_grant)?;
                self.attach_admission_budget_decrease_wakeup(&placed_query.memory_grant)?;
                state.active_queries = state.active_queries.saturating_add(1);
                self.register_active_adaptive_grant(
                    &mut state,
                    request_id,
                    placed_query.device_ordinal,
                    grant_policy,
                    &placed_query.memory_grant,
                );
                self.rotate_after_admission(&mut state, &admitted.tenant_id);
                self.refresh_active_adaptive_shrink_requests_locked(&state);
                if state.queued.is_empty() {
                    self.grow_active_adaptive_grants_locked(&mut state);
                }
                self.inner.notify_admission_available();
                let admission_sequence = self.inner.next_admission_sequence.fetch_add(1, Ordering::Relaxed);
                let service = Arc::clone(&self.inner);
                let device_budget_bytes = placed_query.memory_grant.budget_bytes();
                let device_ordinal = placed_query.device_ordinal;
                return Ok(QueryHandle {
                    lifetime: QueryHandleLifetime::new(
                        Arc::clone(&service),
                        tenant_id.clone(),
                        request_id,
                        placed_query.memory_grant,
                        placed_query.device_slot,
                    ),
                    service,
                    tenant_id,
                    query_id: request_id,
                    admission_sequence,
                    admission_wait: requested_at.elapsed(),
                    device_budget_bytes,
                    device_ordinal,
                });
            }

            let now = Instant::now();
            if now >= deadline {
                self.abandon_queued_request_locked(&mut state, request_id);
                return Err(service_overloaded_error(
                    "admission_wait_timeout",
                    &tenant_id,
                    state.queued.len(),
                    self.inner.config.max_admission_queue_len,
                ));
            }
            let wait = deadline.saturating_duration_since(now);
            let (next_state, timeout) = self.inner.admission_available.wait_timeout(state, wait).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "nexus_query_engine.query_service.admission",
                    "query service admission mutex was poisoned while waiting",
                )
            })?;
            state = next_state;
            if timeout.timed_out() {
                self.abandon_queued_request_locked(&mut state, request_id);
                return Err(service_overloaded_error(
                    "admission_wait_timeout",
                    &tenant_id,
                    state.queued.len(),
                    self.inner.config.max_admission_queue_len,
                ));
            }
        }
    }

    /// Enqueues a query admission request without blocking the caller.
    ///
    /// Call [`QueryAdmissionWaiter::try_admit`] until it returns a handle. If
    /// the waiter is dropped first, the queued request is removed.
    pub fn enqueue_admission(&self, tenant_id: impl Into<TenantId>) -> Result<QueryAdmissionWaiter> {
        self.enqueue_admission_with_qos(tenant_id, QueryQos::default())
    }

    /// Admits a query asynchronously without parking a blocking thread.
    ///
    /// This is the async counterpart to [`Self::admit_query`]. It uses the
    /// same queue, tenant fairness, timeout, and overload error taxonomy, but
    /// waits on an event-driven wakeup instead of polling
    /// [`QueryAdmissionWaiter::try_admit`].
    pub async fn admit_async(&self, tenant_id: impl Into<TenantId>) -> Result<QueryHandle> {
        self.admit_async_with_qos(tenant_id, QueryQos::default()).await
    }

    /// Admits a query asynchronously with caller-supplied QoS metadata.
    pub async fn admit_async_with_qos(&self, tenant_id: impl Into<TenantId>, qos: QueryQos) -> Result<QueryHandle> {
        self.admit_async_with_qos_and_requested_grant_bytes(tenant_id, qos, None).await
    }

    /// Admits a query asynchronously with caller-supplied QoS metadata and an
    /// optional explicit preferred grant.
    pub async fn admit_async_with_qos_and_requested_grant_bytes(
        &self,
        tenant_id: impl Into<TenantId>,
        qos: QueryQos,
        requested_grant_bytes: Option<u64>,
    ) -> Result<QueryHandle> {
        let grant_request = QueryGrantRequest { requested_grant_bytes, ..QueryGrantRequest::new() };
        self.admit_async_with_qos_and_grant_request(tenant_id, qos, grant_request).await
    }

    /// Admits a query asynchronously with caller-supplied QoS metadata and
    /// optional explicit minimum/preferred grant overrides.
    pub async fn admit_async_with_qos_and_grant_request(
        &self,
        tenant_id: impl Into<TenantId>,
        qos: QueryQos,
        grant_request: QueryGrantRequest,
    ) -> Result<QueryHandle> {
        let tenant_id = tenant_id.into();
        if let Some(handle) = self.try_admit_query_with_qos_and_grant_request(tenant_id.clone(), qos, grant_request)? {
            return Ok(handle);
        }
        let mut waiter = self.enqueue_admission_with_qos_and_grant_request(tenant_id, qos, grant_request)?;
        loop {
            let notified = self.inner.admission_notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if let Some(handle) = waiter.try_admit()? {
                return Ok(handle);
            }
            let remaining_wait = waiter.remaining_wait();
            if remaining_wait.is_zero() {
                continue;
            }
            let _ = time::timeout(remaining_wait, &mut notified).await;
        }
    }

    /// Enqueues a query admission request with caller-supplied QoS metadata.
    pub fn enqueue_admission_with_qos(
        &self,
        tenant_id: impl Into<TenantId>,
        qos: QueryQos,
    ) -> Result<QueryAdmissionWaiter> {
        self.enqueue_admission_with_qos_and_requested_grant_bytes(tenant_id, qos, None)
    }

    /// Enqueues a query admission request with caller-supplied QoS metadata and
    /// an optional explicit preferred grant.
    pub fn enqueue_admission_with_qos_and_requested_grant_bytes(
        &self,
        tenant_id: impl Into<TenantId>,
        qos: QueryQos,
        requested_grant_bytes: Option<u64>,
    ) -> Result<QueryAdmissionWaiter> {
        let grant_request = QueryGrantRequest { requested_grant_bytes, ..QueryGrantRequest::new() };
        self.enqueue_admission_with_qos_and_grant_request(tenant_id, qos, grant_request)
    }

    /// Enqueues a query admission request with caller-supplied QoS metadata and
    /// optional explicit minimum/preferred grant overrides.
    pub fn enqueue_admission_with_qos_and_grant_request(
        &self,
        tenant_id: impl Into<TenantId>,
        qos: QueryQos,
        grant_request: QueryGrantRequest,
    ) -> Result<QueryAdmissionWaiter> {
        let tenant_id = tenant_id.into();
        let grant_policy = self.tenant_grant_policy_for_request(&tenant_id, grant_request)?;
        let request_id = self.inner.next_query_id.fetch_add(1, Ordering::Relaxed);
        let requested_at = Instant::now();
        let deadline = requested_at.checked_add(self.inner.config.admission_wait_timeout).unwrap_or(requested_at);
        let mut state = self.lock_admission()?;
        if state.closed {
            return Err(service_closed_error(&tenant_id));
        }
        self.sweep_expired_queued_requests_locked(&mut state);
        if state.queued.len() >= self.inner.config.max_admission_queue_len {
            return Err(service_overloaded_error(
                "admission_queue_full",
                &tenant_id,
                state.queued.len(),
                self.inner.config.max_admission_queue_len,
            ));
        }
        state.queued.push_back(AdmissionRequest {
            request_id,
            tenant_id: tenant_id.clone(),
            grant_policy,
            qos,
            deadline,
        });
        self.inner.notify_admission_available();
        Ok(QueryAdmissionWaiter {
            service: self.clone(),
            tenant_id,
            request_id,
            requested_at,
            deadline,
            grant_policy,
            queued: true,
        })
    }

    /// Attempts to admit a query without queueing or waiting.
    ///
    /// Returns `Ok(None)` when all admission slots are active or when queued
    /// requests already exist and this caller must not bypass them.
    pub fn try_admit_query(&self, tenant_id: impl Into<TenantId>) -> Result<Option<QueryHandle>> {
        self.try_admit_query_with_qos(tenant_id, QueryQos::default())
    }

    /// Attempts to admit a query with caller-supplied QoS metadata without
    /// queueing or waiting.
    pub fn try_admit_query_with_qos(
        &self,
        tenant_id: impl Into<TenantId>,
        qos: QueryQos,
    ) -> Result<Option<QueryHandle>> {
        self.try_admit_query_with_qos_and_requested_grant_bytes(tenant_id, qos, None)
    }

    /// Attempts to admit a query with caller-supplied QoS metadata and an
    /// optional explicit preferred grant without queueing or waiting.
    pub fn try_admit_query_with_qos_and_requested_grant_bytes(
        &self,
        tenant_id: impl Into<TenantId>,
        qos: QueryQos,
        requested_grant_bytes: Option<u64>,
    ) -> Result<Option<QueryHandle>> {
        let grant_request = QueryGrantRequest { requested_grant_bytes, ..QueryGrantRequest::new() };
        self.try_admit_query_with_qos_and_grant_request(tenant_id, qos, grant_request)
    }

    /// Attempts to admit a query with caller-supplied QoS metadata and optional
    /// explicit minimum/preferred grant overrides.
    pub fn try_admit_query_with_qos_and_grant_request(
        &self,
        tenant_id: impl Into<TenantId>,
        qos: QueryQos,
        grant_request: QueryGrantRequest,
    ) -> Result<Option<QueryHandle>> {
        let tenant_id = tenant_id.into();
        let grant_policy = self.tenant_grant_policy_for_request(&tenant_id, grant_request)?;
        let mut slot_notifications = PendingDeviceSlotNotifications::default();
        let mut state = self.lock_admission()?;
        if state.closed {
            return Err(service_closed_error(&tenant_id));
        }
        self.sweep_expired_queued_requests_locked(&mut state);
        if state.active_queries >= self.inner.config.max_concurrent_queries || !state.queued.is_empty() {
            return Ok(None);
        }
        let query_id = self.inner.next_query_id.fetch_add(1, Ordering::Relaxed);
        let Some(placed_query) = self.try_place_query(
            &tenant_id,
            query_id,
            grant_policy,
            qos,
            None,
            state.active_queries,
            &mut slot_notifications,
        )?
        else {
            return Ok(None);
        };
        self.attach_admission_release_wakeup(&placed_query.memory_grant)?;
        self.attach_admission_budget_decrease_wakeup(&placed_query.memory_grant)?;
        state.active_queries = state.active_queries.saturating_add(1);
        self.register_active_adaptive_grant(
            &mut state,
            query_id,
            placed_query.device_ordinal,
            grant_policy,
            &placed_query.memory_grant,
        );
        self.rotate_after_admission(&mut state, &tenant_id);
        let admission_sequence = self.inner.next_admission_sequence.fetch_add(1, Ordering::Relaxed);
        let service = Arc::clone(&self.inner);
        let device_budget_bytes = placed_query.memory_grant.budget_bytes();
        let device_ordinal = placed_query.device_ordinal;
        Ok(Some(QueryHandle {
            lifetime: QueryHandleLifetime::new(
                Arc::clone(&service),
                tenant_id.clone(),
                query_id,
                placed_query.memory_grant,
                placed_query.device_slot,
            ),
            service,
            tenant_id,
            query_id,
            admission_sequence,
            admission_wait: Duration::ZERO,
            device_budget_bytes,
            device_ordinal,
        }))
    }

    /// Closes admission and wakes every queued waiter.
    ///
    /// Active query handles keep their slots until dropped or finished.
    pub fn close(&self) -> Result<()> {
        let mut state = self.lock_admission()?;
        state.closed = true;
        self.inner.notify_admission_available();
        Ok(())
    }

    /// Returns the configured preferred device budget for a tenant.
    pub fn tenant_budget(&self, tenant_id: &TenantId) -> Result<u64> {
        self.tenant_grant_policy(tenant_id).map(|grant| grant.want_bytes)
    }

    /// Returns the configured grant band for a tenant.
    pub fn tenant_grant_band(&self, tenant_id: &TenantId) -> Result<QueryGrantBand> {
        self.tenant_grant_policy(tenant_id).map(Into::into)
    }

    /// Resolve the effective preferred grant for a tenant request.
    pub fn tenant_requested_budget(&self, tenant_id: &TenantId, requested_grant_bytes: Option<u64>) -> Result<u64> {
        let grant_request = QueryGrantRequest { requested_grant_bytes, ..QueryGrantRequest::new() };
        self.tenant_grant_policy_for_request(tenant_id, grant_request).map(|grant| grant.want_bytes)
    }

    /// Resolve the effective preferred grant for a tenant request with an
    /// optional request-specific grant floor.
    pub fn tenant_requested_budget_with_grant_request(
        &self,
        tenant_id: &TenantId,
        grant_request: QueryGrantRequest,
    ) -> Result<u64> {
        self.tenant_grant_policy_for_request(tenant_id, grant_request).map(|grant| grant.want_bytes)
    }

    /// Resolve the effective grant band for a tenant request.
    pub fn tenant_requested_grant_band_with_grant_request(
        &self,
        tenant_id: &TenantId,
        grant_request: QueryGrantRequest,
    ) -> Result<QueryGrantBand> {
        self.tenant_grant_policy_for_request(tenant_id, grant_request).map(Into::into)
    }

    /// Largest device capacity represented by this query service.
    pub fn max_device_capacity_bytes(&self) -> u64 {
        self.inner.resources.as_ref().map_or_else(
            || self.inner.fallback_memory_governor.capacity_bytes(),
            |resources| resources.devices.contexts().map(|context| context.capacity_bytes()).max().unwrap_or(0),
        )
    }

    /// Largest physical device-total byte count represented by this query service.
    ///
    /// Admission-only services have no CUDA inventory, so they fall back to the
    /// configured governor capacity used by deterministic CPU-only tests.
    pub fn max_device_total_memory_bytes(&self) -> u64 {
        self.inner.resources.as_ref().map_or_else(
            || self.max_device_capacity_bytes(),
            |resources| resources.devices.contexts().map(|context| context.facts().total_bytes).max().unwrap_or(0),
        )
    }

    fn tenant_grant_policy(&self, tenant_id: &TenantId) -> Result<TenantGrantPolicy> {
        self.inner.tenant_grants.get(tenant_id).copied().ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.query_service.admission",
                "unknown tenant id for query service admission",
            )
            .with_fact("tenant_id", tenant_id.as_str())
        })
    }

    fn tenant_grant_policy_for_request(
        &self,
        tenant_id: &TenantId,
        grant_request: QueryGrantRequest,
    ) -> Result<TenantGrantPolicy> {
        let grant_policy = self.tenant_grant_policy(tenant_id)?;
        if let Some(requested_grant_bytes) = grant_request.requested_grant_bytes {
            self.validate_requested_grant_bytes(tenant_id, grant_policy, requested_grant_bytes)?;
        }
        if let Some(requested_grant_min_bytes) = grant_request.requested_grant_min_bytes {
            self.validate_requested_grant_min_bytes(tenant_id, grant_policy, requested_grant_min_bytes)?;
        }
        let mut effective_policy = grant_policy;
        if let Some(requested_grant_min_bytes) = grant_request.requested_grant_min_bytes {
            effective_policy = effective_policy.with_requested_grant_min_bytes(requested_grant_min_bytes);
        }
        if let Some(requested_grant_bytes) = grant_request.requested_grant_bytes {
            effective_policy = effective_policy.with_requested_grant_bytes(requested_grant_bytes);
        }
        if effective_policy.want_bytes < effective_policy.min_bytes {
            return Err(invalid_requested_grant_range_error(
                tenant_id,
                grant_policy,
                grant_request,
                "requested_grant_below_requested_minimum",
            ));
        }
        Ok(effective_policy)
    }

    fn validate_requested_grant_bytes(
        &self,
        tenant_id: &TenantId,
        grant_policy: TenantGrantPolicy,
        requested_grant_bytes: u64,
    ) -> Result<()> {
        if requested_grant_bytes < grant_policy.min_bytes {
            return Err(invalid_requested_grant_error(
                tenant_id,
                grant_policy,
                requested_grant_bytes,
                self.max_device_capacity_bytes(),
                "requested_grant_below_minimum",
            ));
        }
        if requested_grant_bytes > grant_policy.want_bytes {
            return Err(invalid_requested_grant_error(
                tenant_id,
                grant_policy,
                requested_grant_bytes,
                self.max_device_capacity_bytes(),
                "requested_grant_above_tenant_want",
            ));
        }
        let max_device_capacity_bytes = self.max_device_capacity_bytes();
        if requested_grant_bytes > max_device_capacity_bytes {
            return Err(invalid_requested_grant_error(
                tenant_id,
                grant_policy,
                requested_grant_bytes,
                max_device_capacity_bytes,
                "requested_grant_above_device_capacity",
            ));
        }
        Ok(())
    }

    fn validate_requested_grant_min_bytes(
        &self,
        tenant_id: &TenantId,
        grant_policy: TenantGrantPolicy,
        requested_grant_min_bytes: u64,
    ) -> Result<()> {
        if requested_grant_min_bytes < grant_policy.min_bytes {
            return Err(invalid_requested_grant_min_error(
                tenant_id,
                grant_policy,
                requested_grant_min_bytes,
                self.max_device_capacity_bytes(),
                "requested_grant_min_below_tenant_minimum",
            ));
        }
        if requested_grant_min_bytes > grant_policy.want_bytes {
            return Err(invalid_requested_grant_min_error(
                tenant_id,
                grant_policy,
                requested_grant_min_bytes,
                self.max_device_capacity_bytes(),
                "requested_grant_min_above_tenant_want",
            ));
        }
        let max_device_capacity_bytes = self.max_device_capacity_bytes();
        if requested_grant_min_bytes > max_device_capacity_bytes {
            return Err(invalid_requested_grant_min_error(
                tenant_id,
                grant_policy,
                requested_grant_min_bytes,
                max_device_capacity_bytes,
                "requested_grant_min_above_device_capacity",
            ));
        }
        if self.inner.config.slot_guarantee == SlotGuarantee::GuaranteedMinimum {
            let idle_grant_ceiling = self.guaranteed_min_idle_grant_ceiling_bytes()?;
            if requested_grant_min_bytes > idle_grant_ceiling {
                return Err(invalid_requested_grant_min_error(
                    tenant_id,
                    grant_policy,
                    requested_grant_min_bytes,
                    max_device_capacity_bytes,
                    "requested_grant_min_above_guaranteed_min_slot_ceiling",
                )
                .with_fact("guaranteed_min_idle_grant_ceiling_bytes", idle_grant_ceiling)
                .with_fact("max_concurrent_queries", self.inner.config.max_concurrent_queries)
                .with_fact("max_tenant_min_device_budget_bytes", self.inner.max_tenant_min_bytes)
                .with_fact("slot_guarantee", self.inner.config.slot_guarantee.as_str()));
            }
        }
        Ok(())
    }

    fn guaranteed_min_idle_grant_ceiling_bytes(&self) -> Result<u64> {
        if self.inner.config.slot_guarantee == SlotGuarantee::Optimistic {
            return Ok(self.max_device_capacity_bytes());
        }
        let Some(resources) = &self.inner.resources else {
            return Ok(self
                .inner
                .fallback_memory_governor
                .capacity_bytes()
                .saturating_sub(self.reserved_min_bytes_for_remaining_slots(0)));
        };
        let mut ceiling = 0u64;
        for context in resources.devices.contexts() {
            if !self.placement_policy_admits_ordinal(context.device().ordinal()) {
                continue;
            }
            let reserved_for_remaining_slots =
                self.reserved_min_bytes_for_selected_device_at_idle(resources, context.device())?;
            ceiling = ceiling.max(context.capacity_bytes().saturating_sub(reserved_for_remaining_slots));
        }
        Ok(ceiling)
    }

    fn remaining_slots_after_current(&self, active_queries: usize) -> usize {
        self.inner.config.max_concurrent_queries.saturating_sub(active_queries.saturating_add(1))
    }

    fn reserved_min_bytes_for_remaining_slots(&self, active_queries: usize) -> u64 {
        if self.inner.config.slot_guarantee == SlotGuarantee::Optimistic {
            return 0;
        }
        let remaining_slots = self
            .remaining_slots_after_current(active_queries)
            .min(self.future_slots_after_current_on_fallback_device(active_queries));
        self.inner.max_tenant_min_bytes.saturating_mul(u64::try_from(remaining_slots).unwrap_or(u64::MAX))
    }

    fn reserved_min_bytes_for_empty_slots(&self, active_queries: usize) -> u64 {
        if self.inner.config.slot_guarantee == SlotGuarantee::Optimistic {
            return 0;
        }
        let empty_slots = self
            .inner
            .config
            .max_concurrent_queries
            .saturating_sub(active_queries)
            .min(self.future_slots_on_fallback_device(active_queries));
        self.inner.max_tenant_min_bytes.saturating_mul(u64::try_from(empty_slots).unwrap_or(u64::MAX))
    }

    fn future_slots_after_current_on_fallback_device(&self, active_queries: usize) -> usize {
        self.inner.config.max_queries_per_device.map_or(usize::MAX, |max_queries_per_device| {
            max_queries_per_device.saturating_sub(active_queries.saturating_add(1))
        })
    }

    fn future_slots_on_fallback_device(&self, active_queries: usize) -> usize {
        self.inner
            .config
            .max_queries_per_device
            .map_or(usize::MAX, |max_queries_per_device| max_queries_per_device.saturating_sub(active_queries))
    }

    fn future_min_slots_on_device(
        &self,
        context: &super::DeviceContext,
        min_bytes: u64,
        active_after_selected: usize,
    ) -> Result<usize> {
        if min_bytes == 0 {
            return Ok(usize::MAX);
        }
        let memory_slots = usize::try_from(context.available_bytes()? / min_bytes).unwrap_or(usize::MAX);
        let capped_slots = self.inner.config.max_queries_per_device.map_or(memory_slots, |max_queries_per_device| {
            memory_slots.min(max_queries_per_device.saturating_sub(active_after_selected))
        });
        Ok(capped_slots)
    }

    fn idle_future_min_slots_on_device(
        &self,
        capacity_bytes: u64,
        min_bytes: u64,
        active_after_selected: usize,
    ) -> usize {
        if min_bytes == 0 {
            return usize::MAX;
        }
        let memory_slots = usize::try_from(capacity_bytes / min_bytes).unwrap_or(usize::MAX);
        self.inner.config.max_queries_per_device.map_or(memory_slots, |max_queries_per_device| {
            memory_slots.min(max_queries_per_device.saturating_sub(active_after_selected))
        })
    }

    fn reserved_min_bytes_for_selected_device_at_idle(
        &self,
        resources: &QueryServiceResources,
        selected_device: cudf::DeviceId,
    ) -> Result<u64> {
        if self.inner.config.slot_guarantee == SlotGuarantee::Optimistic {
            return Ok(0);
        }
        let remaining_slots = self.remaining_slots_after_current(0);
        if remaining_slots == 0 || self.inner.max_tenant_min_bytes == 0 {
            return Ok(0);
        }
        let selected_context = resources.devices.context(selected_device)?;
        let mut other_device_min_slots = 0usize;
        for context in resources.devices.contexts() {
            if context.device() == selected_device {
                continue;
            }
            if !self.placement_policy_admits_ordinal(context.device().ordinal()) {
                continue;
            }
            let slots =
                self.idle_future_min_slots_on_device(context.capacity_bytes(), self.inner.max_tenant_min_bytes, 0);
            other_device_min_slots = other_device_min_slots.saturating_add(slots);
            if other_device_min_slots >= remaining_slots {
                return Ok(0);
            }
        }
        let slots_to_reserve_on_selected = remaining_slots.saturating_sub(other_device_min_slots);
        let selected_future_slots =
            self.idle_future_min_slots_on_device(selected_context.capacity_bytes(), self.inner.max_tenant_min_bytes, 1);
        if slots_to_reserve_on_selected > selected_future_slots {
            return Ok(u64::MAX);
        }
        Ok(self
            .inner
            .max_tenant_min_bytes
            .saturating_mul(u64::try_from(slots_to_reserve_on_selected).unwrap_or(u64::MAX)))
    }

    fn reserved_min_bytes_for_selected_device(
        &self,
        resources: &QueryServiceResources,
        selected_device: cudf::DeviceId,
        active_queries: usize,
    ) -> Result<u64> {
        if self.inner.config.slot_guarantee == SlotGuarantee::Optimistic {
            return Ok(0);
        }
        let remaining_slots = self.remaining_slots_after_current(active_queries);
        if remaining_slots == 0 || self.inner.max_tenant_min_bytes == 0 {
            return Ok(0);
        }
        let selected_context = resources.devices.context(selected_device)?;
        let mut other_device_min_slots = 0usize;
        for context in resources.devices.contexts() {
            if context.device() == selected_device {
                continue;
            }
            if !self.placement_policy_admits_ordinal(context.device().ordinal()) {
                continue;
            }
            let slots =
                self.future_min_slots_on_device(context, self.inner.max_tenant_min_bytes, context.active_queries())?;
            other_device_min_slots = other_device_min_slots.saturating_add(slots);
            if other_device_min_slots >= remaining_slots {
                return Ok(0);
            }
        }
        let slots_to_reserve_on_selected = remaining_slots.saturating_sub(other_device_min_slots);
        let selected_future_slots = self.future_min_slots_on_device(
            selected_context,
            self.inner.max_tenant_min_bytes,
            selected_context.active_queries().saturating_add(1),
        )?;
        if slots_to_reserve_on_selected > selected_future_slots {
            return Ok(u64::MAX);
        }
        Ok(self
            .inner
            .max_tenant_min_bytes
            .saturating_mul(u64::try_from(slots_to_reserve_on_selected).unwrap_or(u64::MAX)))
    }

    fn reserved_min_bytes_for_selected_device_empty_slots(
        &self,
        resources: &QueryServiceResources,
        selected_device: cudf::DeviceId,
        active_queries: usize,
    ) -> Result<u64> {
        if self.inner.config.slot_guarantee == SlotGuarantee::Optimistic {
            return Ok(0);
        }
        let empty_slots = self.inner.config.max_concurrent_queries.saturating_sub(active_queries);
        if empty_slots == 0 || self.inner.max_tenant_min_bytes == 0 {
            return Ok(0);
        }
        let selected_context = resources.devices.context(selected_device)?;
        let mut other_device_min_slots = 0usize;
        for context in resources.devices.contexts() {
            if context.device() == selected_device {
                continue;
            }
            if !self.placement_policy_admits_ordinal(context.device().ordinal()) {
                continue;
            }
            let slots =
                self.future_min_slots_on_device(context, self.inner.max_tenant_min_bytes, context.active_queries())?;
            other_device_min_slots = other_device_min_slots.saturating_add(slots);
            if other_device_min_slots >= empty_slots {
                return Ok(0);
            }
        }
        let slots_to_reserve_on_selected = empty_slots.saturating_sub(other_device_min_slots);
        let selected_future_slots = self.future_min_slots_on_device(
            selected_context,
            self.inner.max_tenant_min_bytes,
            selected_context.active_queries(),
        )?;
        if slots_to_reserve_on_selected > selected_future_slots {
            return Ok(u64::MAX);
        }
        Ok(self
            .inner
            .max_tenant_min_bytes
            .saturating_mul(u64::try_from(slots_to_reserve_on_selected).unwrap_or(u64::MAX)))
    }

    fn grant_budget_bytes(
        &self,
        available_bytes: u64,
        grant_policy: TenantGrantPolicy,
        reserved_for_remaining_slots: u64,
    ) -> Option<u64> {
        let required_floor = grant_policy.min_bytes.saturating_add(reserved_for_remaining_slots);
        if available_bytes < required_floor {
            return None;
        }
        Some(
            grant_policy
                .want_bytes
                .min(available_bytes.saturating_sub(reserved_for_remaining_slots))
                .max(grant_policy.min_bytes),
        )
    }

    fn adaptive_shrink_deficit_bytes(
        available_bytes: u64,
        min_bytes: u64,
        reserved_for_remaining_slots: u64,
        capacity_bytes: u64,
    ) -> Option<u64> {
        if reserved_for_remaining_slots == u64::MAX {
            return None;
        }
        let required_floor = min_bytes.checked_add(reserved_for_remaining_slots)?;
        if required_floor > capacity_bytes || available_bytes >= required_floor {
            return None;
        }
        Some(required_floor - available_bytes)
    }

    fn try_grant_from_governor(
        &self,
        governor: &MemoryGovernor,
        query_id: u64,
        grant_policy: TenantGrantPolicy,
        qos: QueryQos,
        deadline: Option<Instant>,
        active_queries: usize,
    ) -> Result<Option<MemoryGrant>> {
        let available_bytes = governor.available_bytes()?;
        if available_bytes < grant_policy.min_bytes {
            return Ok(None);
        }
        let reserved_for_remaining_slots = self.reserved_min_bytes_for_remaining_slots(active_queries);
        let Some(want_bytes) = self.grant_budget_bytes(available_bytes, grant_policy, reserved_for_remaining_slots)
        else {
            return Ok(None);
        };
        let mut request = GrantRequest::new(query_id, grant_policy.min_bytes, want_bytes);
        request = request.with_qos(qos);
        if let Some(deadline) = deadline {
            request = request.with_deadline(deadline);
        }
        governor.try_request(request)
    }

    fn try_place_query(
        &self,
        tenant_id: &TenantId,
        query_id: u64,
        grant_policy: TenantGrantPolicy,
        qos: QueryQos,
        deadline: Option<Instant>,
        active_queries: usize,
        slot_notifications: &mut PendingDeviceSlotNotifications,
    ) -> Result<Option<PlacedQuery>> {
        let Some(resources) = &self.inner.resources else {
            if !self.fallback_active_query_cap_allows(active_queries) {
                return Ok(None);
            }
            let Some(memory_grant) = self.try_grant_from_governor(
                &self.inner.fallback_memory_governor,
                query_id,
                grant_policy,
                qos,
                deadline,
                active_queries,
            )?
            else {
                return Ok(None);
            };
            return Ok(Some(PlacedQuery {
                device_ordinal: self.inner.config.device_ordinal,
                memory_grant,
                device_slot: None,
            }));
        };

        let placement_request = PlacementRequest::with_grant_range(grant_policy.min_bytes, grant_policy.want_bytes)
            .with_tenant_id(tenant_id.as_str());
        let mut any_device_can_fit_minimum = false;
        for device in self.placement_candidates(resources, &placement_request)? {
            let context = resources.devices.context(device)?;
            if context.capacity_bytes() >= grant_policy.min_bytes {
                any_device_can_fit_minimum = true;
            }
            let available_bytes = context.available_bytes()?;
            if available_bytes < grant_policy.min_bytes {
                continue;
            }
            let reserved_for_remaining_slots =
                self.reserved_min_bytes_for_selected_device(resources, device, active_queries)?;
            let Some(want_bytes) = self.grant_budget_bytes(available_bytes, grant_policy, reserved_for_remaining_slots)
            else {
                continue;
            };
            let Some(device_slot) = context.try_acquire_query_slot(self.inner.config.max_queries_per_device) else {
                continue;
            };
            let mut request = GrantRequest::new(query_id, grant_policy.min_bytes, want_bytes);
            request = request.with_qos(qos);
            if let Some(deadline) = deadline {
                request = request.with_deadline(deadline);
            }
            match context.try_grant_query(request) {
                Ok(Some(memory_grant)) => {
                    return Ok(Some(PlacedQuery {
                        device_ordinal: device.ordinal(),
                        memory_grant,
                        device_slot: Some(device_slot),
                    }));
                },
                Ok(None) => slot_notifications.release_slot(device_slot),
                Err(error) => {
                    slot_notifications.release_slot(device_slot);
                    return Err(error);
                },
            }
        }
        if any_device_can_fit_minimum {
            return Ok(None);
        }
        Err(QueryError::new(
            QueryErrorCode::ServiceOverloaded,
            "nexus_query_engine.query_service.placement",
            "no configured CUDA device has enough capacity for the query minimum budget",
        )
        .with_fact("tenant_id", tenant_id.as_str())
        .with_fact("min_device_budget_bytes", grant_policy.min_bytes)
        .with_fact("reason", "query_min_budget_exceeds_device_capacity"))
    }

    fn attach_admission_release_wakeup(&self, grant: &MemoryGrant) -> Result<()> {
        let service = self.clone();
        grant.notify_on_release(Arc::new(move || {
            let mut guard = match service.inner.admission.lock() {
                Ok(guard) => guard,
                Err(_error) => {
                    eprintln!(
                        "[nexus-query-engine] warning: failed to lock query service admission for grant release wakeup"
                    );
                    return;
                },
            };
            if guard.queued.is_empty() {
                service.grow_active_adaptive_grants_locked(&mut guard);
            }
            service.inner.notify_admission_available();
        }))
    }

    fn attach_admission_budget_decrease_wakeup(&self, grant: &MemoryGrant) -> Result<()> {
        let service = self.clone();
        grant.notify_on_budget_decrease(Arc::new(move || {
            let mut guard = match service.inner.admission.lock() {
                Ok(guard) => guard,
                Err(_error) => {
                    eprintln!(
                        "[nexus-query-engine] warning: failed to lock query service admission for grant budget-decrease wakeup"
                    );
                    return;
                },
            };
            if guard.queued.is_empty() {
                service.grow_active_adaptive_grants_locked(&mut guard);
            }
            service.inner.notify_admission_available();
        }))
    }

    fn register_active_adaptive_grant(
        &self,
        state: &mut AdmissionState,
        query_id: u64,
        device_ordinal: usize,
        grant_policy: TenantGrantPolicy,
        memory_grant: &MemoryGrant,
    ) {
        if self.inner.config.native_pipeline_execution_mode != NativePipelineExecutionMode::Adaptive {
            return;
        }
        if grant_policy.want_bytes <= grant_policy.min_bytes {
            return;
        }
        state.active_adaptive_grants.insert(
            query_id,
            ActiveAdaptiveGrant {
                device_ordinal,
                min_bytes: memory_grant.min_bytes(),
                target_bytes: grant_policy.want_bytes.max(memory_grant.min_bytes()),
                memory_grant: memory_grant.clone(),
            },
        );
    }

    fn adaptive_growth_available_bytes(&self, state: &AdmissionState, device_ordinal: usize) -> Result<u64> {
        let Some(resources) = &self.inner.resources else {
            let available = self.inner.fallback_memory_governor.available_bytes()?;
            return Ok(available.saturating_sub(self.reserved_min_bytes_for_empty_slots(state.active_queries)));
        };
        let context = resources.devices.context_by_ordinal(device_ordinal)?;
        let available = context.available_bytes()?;
        let reserved =
            self.reserved_min_bytes_for_selected_device_empty_slots(resources, context.device(), state.active_queries)?;
        Ok(available.saturating_sub(reserved))
    }

    fn grow_active_adaptive_grants_locked(&self, state: &mut AdmissionState) {
        if self.inner.config.native_pipeline_execution_mode != NativePipelineExecutionMode::Adaptive {
            return;
        }
        let device_ordinals =
            state.active_adaptive_grants.values().map(|active| active.device_ordinal).collect::<BTreeSet<_>>();
        for device_ordinal in device_ordinals {
            let available_bytes = match self.adaptive_growth_available_bytes(state, device_ordinal) {
                Ok(bytes) => bytes,
                Err(error) => {
                    eprintln!("[nexus-query-engine] warning: failed to compute adaptive grant growth: {error}");
                    continue;
                },
            };
            if available_bytes == 0 {
                continue;
            }
            let mut candidates = Vec::new();
            let query_ids = state
                .active_adaptive_grants
                .iter()
                .filter_map(|(query_id, active)| (active.device_ordinal == device_ordinal).then_some(*query_id))
                .collect::<Vec<_>>();
            for query_id in query_ids {
                let Some(active) = state.active_adaptive_grants.get(&query_id) else {
                    continue;
                };
                let current_bytes = active.memory_grant.budget_bytes();
                let target_bytes = active.target_bytes.max(active.min_bytes);
                if current_bytes >= target_bytes {
                    continue;
                }
                candidates.push(AdaptiveGrowthCandidate {
                    query_id,
                    missing_bytes: target_bytes.saturating_sub(current_bytes),
                    weight: active.memory_grant.qos().budget_weight(),
                });
            }
            for (query_id, allocated_bytes) in weighted_adaptive_growth_allocations(&candidates, available_bytes) {
                let Some(active) = state.active_adaptive_grants.get(&query_id).cloned() else {
                    continue;
                };
                let current_bytes = active.memory_grant.budget_bytes();
                let target_bytes = active.target_bytes.max(active.min_bytes);
                let grow_bytes = target_bytes.saturating_sub(current_bytes).min(allocated_bytes);
                if grow_bytes == 0 {
                    continue;
                }
                match active.memory_grant.try_grow(grow_bytes) {
                    Ok(true) => continue,
                    Ok(false) => continue,
                    Err(error) => {
                        eprintln!("[nexus-query-engine] warning: failed to grow adaptive grant: {error}");
                        continue;
                    },
                }
            }
        }
    }

    fn request_adaptive_grant_shrink_for_blocked_request_locked(
        &self,
        state: &AdmissionState,
        request: &AdmissionRequest,
    ) -> Result<()> {
        if self.inner.config.native_pipeline_execution_mode != NativePipelineExecutionMode::Adaptive {
            return Ok(());
        }
        if state.active_queries >= self.inner.config.max_concurrent_queries {
            return Ok(());
        }

        let min_bytes = request.grant_policy.min_bytes;
        let Some(resources) = &self.inner.resources else {
            if !self.fallback_active_query_cap_allows(state.active_queries) {
                return Ok(());
            }
            let available_bytes = self.inner.fallback_memory_governor.available_bytes()?;
            if available_bytes < min_bytes {
                self.request_active_adaptive_grant_shrink_on_device_locked(
                    state,
                    self.inner.config.device_ordinal,
                    min_bytes - available_bytes,
                )?;
            }
            return Ok(());
        };

        let placement_request = PlacementRequest::with_grant_range(min_bytes, request.grant_policy.want_bytes)
            .with_tenant_id(request.tenant_id.as_str());
        for device in self.placement_candidates(resources, &placement_request)? {
            let context = resources.devices.context(device)?;
            if context.capacity_bytes() < min_bytes {
                continue;
            }
            if !self.device_active_query_cap_allows(context) {
                continue;
            }
            let available_bytes = context.available_bytes()?;
            let reserved_for_remaining_slots =
                self.reserved_min_bytes_for_selected_device(resources, device, state.active_queries)?;
            if self.grant_budget_bytes(available_bytes, request.grant_policy, reserved_for_remaining_slots).is_some() {
                return Ok(());
            }
            if let Some(deficit_bytes) = Self::adaptive_shrink_deficit_bytes(
                available_bytes,
                min_bytes,
                reserved_for_remaining_slots,
                context.capacity_bytes(),
            ) {
                self.request_active_adaptive_grant_shrink_on_device_locked(state, device.ordinal(), deficit_bytes)?;
            }
        }
        Ok(())
    }

    fn placement_candidates(
        &self,
        resources: &QueryServiceResources,
        request: &PlacementRequest,
    ) -> Result<Vec<cudf::DeviceId>> {
        self.inner.config.placement_policy.candidates(&resources.devices, request)
    }

    fn device_active_query_cap_allows(&self, context: &super::DeviceContext) -> bool {
        self.inner
            .config
            .max_queries_per_device
            .is_none_or(|max_queries_per_device| context.active_queries() < max_queries_per_device)
    }

    fn placement_policy_admits_ordinal(&self, ordinal: usize) -> bool {
        placement_policy_admits_ordinal(self.inner.config.placement_policy, ordinal)
    }

    fn fallback_active_query_cap_allows(&self, active_queries: usize) -> bool {
        self.inner
            .config
            .max_queries_per_device
            .is_none_or(|max_queries_per_device| active_queries < max_queries_per_device)
    }

    fn request_active_adaptive_grant_shrink_on_device_locked(
        &self,
        state: &AdmissionState,
        device_ordinal: usize,
        deficit_bytes: u64,
    ) -> Result<()> {
        if deficit_bytes == 0 {
            return Ok(());
        }
        let mut query_ids = state
            .active_adaptive_grants
            .iter()
            .filter_map(|(query_id, active)| (active.device_ordinal == device_ordinal).then_some(*query_id))
            .collect::<Vec<_>>();
        query_ids.sort_unstable();

        let mut remaining_deficit = deficit_bytes;
        for query_id in query_ids {
            let Some(active) = state.active_adaptive_grants.get(&query_id) else {
                continue;
            };
            let current_bytes = active.memory_grant.budget_bytes();
            if current_bytes <= active.min_bytes {
                continue;
            }
            let shrinkable_bytes = current_bytes - active.min_bytes;
            let shrink_bytes = shrinkable_bytes.min(remaining_deficit);
            active.memory_grant.request_shrink_to(current_bytes - shrink_bytes)?;
            remaining_deficit = remaining_deficit.saturating_sub(shrink_bytes);
            if remaining_deficit == 0 {
                break;
            }
        }
        Ok(())
    }

    fn refresh_active_adaptive_shrink_requests_locked(&self, state: &AdmissionState) {
        if self.inner.config.native_pipeline_execution_mode != NativePipelineExecutionMode::Adaptive {
            return;
        }
        for active in state.active_adaptive_grants.values() {
            if let Err(error) = active.memory_grant.clear_requested_shrink() {
                eprintln!("[nexus-query-engine] warning: failed to rescind adaptive grant shrink request: {error}");
            }
        }
        for request in &state.queued {
            if let Err(error) = self.request_adaptive_grant_shrink_for_blocked_request_locked(state, request) {
                eprintln!("[nexus-query-engine] warning: failed to refresh adaptive grant shrink request: {error}");
            }
        }
    }

    fn remove_queued_request_without_admission_locked(
        &self,
        state: &mut AdmissionState,
        request_id: u64,
    ) -> Option<AdmissionRequest> {
        let removed = remove_queued_request(state, request_id);
        if removed.is_some() {
            self.refresh_active_adaptive_shrink_requests_locked(state);
            if state.queued.is_empty() {
                self.grow_active_adaptive_grants_locked(state);
            }
        }
        removed
    }

    fn abandon_queued_request_locked(&self, state: &mut AdmissionState, request_id: u64) -> Option<AdmissionRequest> {
        let removed = self.remove_queued_request_without_admission_locked(state, request_id);
        if removed.is_some() {
            self.inner.notify_admission_available();
        }
        removed
    }

    /// Returns the service memory-governor report.
    #[deprecated(note = "use memory_governor_reports() to inspect every service device")]
    pub fn memory_governor_report(&self) -> Result<MemoryGovernorReport> {
        if let Some(resources) = &self.inner.resources {
            return resources.devices.context_by_ordinal(self.inner.config.device_ordinal)?.memory_governor().report();
        }
        self.inner.fallback_memory_governor.report()
    }

    /// Returns one memory-governor report for each installed service device.
    pub fn memory_governor_reports(&self) -> Result<Vec<MemoryGovernorReport>> {
        let Some(resources) = &self.inner.resources else {
            return Ok(vec![self.inner.fallback_memory_governor.report()?]);
        };
        resources.devices.contexts().map(|context| context.memory_governor().report()).collect()
    }

    /// Returns one device memory allocation-failure report for each installed service device.
    pub fn device_memory_failure_reports(&self) -> Vec<DeviceMemoryFailureReport> {
        let Some(resources) = &self.inner.resources else {
            return Vec::new();
        };
        resources.devices.memory_failure_reports()
    }

    /// Returns the service configuration.
    pub fn config(&self) -> &QueryServiceConfig {
        &self.inner.config
    }

    /// Returns tenant ids in their admission rotation order.
    pub fn tenant_order(&self) -> &[TenantId] {
        &self.inner.tenant_order
    }

    /// Returns a point-in-time admission snapshot for service reports.
    pub fn admission_snapshot(&self) -> Result<QueryServiceAdmissionSnapshot> {
        let mut state = self.lock_admission()?;
        self.sweep_expired_queued_requests_locked(&mut state);
        let queued_by_tenant = self
            .inner
            .tenant_order
            .iter()
            .map(|tenant_id| TenantQueueDepth {
                tenant_id: tenant_id.clone(),
                queued_queries: state.queued.iter().filter(|request| &request.tenant_id == tenant_id).count(),
            })
            .collect();
        Ok(QueryServiceAdmissionSnapshot {
            active_queries: state.active_queries,
            queued_queries: state.queued.len(),
            queued_by_tenant,
            next_tenant_id: self.inner.tenant_order.get(state.next_tenant_index).cloned(),
        })
    }

    fn lock_admission(&self) -> Result<std::sync::MutexGuard<'_, AdmissionState>> {
        self.inner.admission.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.query_service.admission",
                "query service admission mutex was poisoned",
            )
        })
    }

    fn request_is_admissible(&self, state: &mut AdmissionState, request_id: u64) -> Result<bool> {
        self.sweep_expired_queued_requests_locked(state);
        if state.active_queries >= self.inner.config.max_concurrent_queries {
            return Ok(false);
        }
        let Some(position) = self.select_next_request_position(state)? else {
            return Ok(false);
        };
        Ok(state.queued[position].request_id == request_id)
    }

    fn sweep_expired_queued_requests_locked(&self, state: &mut AdmissionState) {
        let now = Instant::now();
        let before = state.queued.len();
        state.queued.retain(|request| request.deadline > now);
        if state.queued.len() != before {
            self.refresh_active_adaptive_shrink_requests_locked(state);
            if state.queued.is_empty() {
                self.grow_active_adaptive_grants_locked(state);
            }
            self.inner.notify_admission_available();
        }
    }

    fn select_next_request_position(&self, state: &AdmissionState) -> Result<Option<usize>> {
        if state.queued.is_empty() {
            return Ok(None);
        }

        let mut first_candidate = None;
        for priority in Priority::DESCENDING {
            for offset in 0..self.inner.tenant_order.len() {
                let tenant_index = (state.next_tenant_index + offset) % self.inner.tenant_order.len();
                let tenant_id = &self.inner.tenant_order[tenant_index];
                for position in state.queued.iter().enumerate().filter_map(|(position, request)| {
                    (&request.tenant_id == tenant_id && request.qos.priority() == priority).then_some(position)
                }) {
                    first_candidate.get_or_insert(position);
                    if self.request_can_receive_minimum_grant(state, &state.queued[position])? {
                        return Ok(Some(position));
                    }
                }
            }
        }

        Ok(first_candidate.or(Some(0)))
    }

    fn request_can_receive_minimum_grant(&self, state: &AdmissionState, request: &AdmissionRequest) -> Result<bool> {
        let Some(resources) = &self.inner.resources else {
            if !self.fallback_active_query_cap_allows(state.active_queries) {
                return Ok(false);
            }
            let available_bytes = self.inner.fallback_memory_governor.available_bytes()?;
            return Ok(self
                .grant_budget_bytes(
                    available_bytes,
                    request.grant_policy,
                    self.reserved_min_bytes_for_remaining_slots(state.active_queries),
                )
                .is_some());
        };
        let placement_request =
            PlacementRequest::with_grant_range(request.grant_policy.min_bytes, request.grant_policy.want_bytes)
                .with_tenant_id(request.tenant_id.as_str());
        for device in self.placement_candidates(resources, &placement_request)? {
            let context = resources.devices.context(device)?;
            if !self.device_active_query_cap_allows(context) {
                continue;
            }
            let reserved_for_remaining_slots =
                self.reserved_min_bytes_for_selected_device(resources, context.device(), state.active_queries)?;
            if context.capacity_bytes() >= request.grant_policy.min_bytes
                && self
                    .grant_budget_bytes(context.available_bytes()?, request.grant_policy, reserved_for_remaining_slots)
                    .is_some()
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn memory_grant_wait_timeout_error(
        &self,
        tenant_id: &TenantId,
        grant_policy: TenantGrantPolicy,
        queued_queries: usize,
    ) -> QueryError {
        let mut error = service_overloaded_error(
            "memory_grant_wait_timeout",
            tenant_id,
            queued_queries,
            self.inner.config.max_admission_queue_len,
        )
        .with_fact("requested_grant_min_bytes", grant_policy.min_bytes)
        .with_fact("requested_grant_bytes", grant_policy.want_bytes)
        .with_fact("effective_min_device_budget_bytes", grant_policy.min_bytes)
        .with_fact("effective_want_device_budget_bytes", grant_policy.want_bytes)
        .with_fact("slot_guarantee", self.inner.config.slot_guarantee.as_str())
        .with_fact("max_concurrent_queries", self.inner.config.max_concurrent_queries)
        .with_fact("max_tenant_min_device_budget_bytes", self.inner.max_tenant_min_bytes);
        if let Ok(ceiling) = self.guaranteed_min_idle_grant_ceiling_bytes() {
            error = error.with_fact("guaranteed_min_idle_grant_ceiling_bytes", ceiling);
        }
        error
    }

    fn rotate_after_admission(&self, state: &mut AdmissionState, tenant_id: &TenantId) {
        if let Some(index) = self.inner.tenant_order.iter().position(|candidate| candidate == tenant_id) {
            state.next_tenant_index = (index + 1) % self.inner.tenant_order.len();
        }
    }
}

fn local_parquet_source_caches_for_ordinals(
    device_ordinals: &[usize],
    max_retained_bytes: Option<u64>,
) -> HashMap<usize, Arc<GpuLocalParquetSourceCache>> {
    let Some(max_retained_bytes) = max_retained_bytes else {
        return HashMap::new();
    };
    device_ordinals
        .iter()
        .copied()
        .map(|ordinal| (ordinal, Arc::new(GpuLocalParquetSourceCache::new(max_retained_bytes))))
        .collect()
}

fn invalid_requested_grant_error(
    tenant_id: &TenantId,
    grant_policy: TenantGrantPolicy,
    requested_grant_bytes: u64,
    max_device_capacity_bytes: u64,
    reason: &'static str,
) -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        "nexus_query_engine.query_service.requested_grant",
        "requested query grant bytes must be within the tenant grant band and device capacity",
    )
    .with_fact("tenant_id", tenant_id.as_str())
    .with_fact("requested_grant_bytes", requested_grant_bytes)
    .with_fact("min_device_budget_bytes", grant_policy.min_bytes)
    .with_fact("tenant_want_device_budget_bytes", grant_policy.want_bytes)
    .with_fact("max_device_capacity_bytes", max_device_capacity_bytes)
    .with_fact("reason", reason)
}

fn invalid_requested_grant_min_error(
    tenant_id: &TenantId,
    grant_policy: TenantGrantPolicy,
    requested_grant_min_bytes: u64,
    max_device_capacity_bytes: u64,
    reason: &'static str,
) -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        "nexus_query_engine.query_service.requested_grant_min",
        "requested query grant minimum bytes must be within the tenant grant band and device capacity",
    )
    .with_fact("tenant_id", tenant_id.as_str())
    .with_fact("requested_grant_min_bytes", requested_grant_min_bytes)
    .with_fact("min_device_budget_bytes", grant_policy.min_bytes)
    .with_fact("tenant_want_device_budget_bytes", grant_policy.want_bytes)
    .with_fact("max_device_capacity_bytes", max_device_capacity_bytes)
    .with_fact("reason", reason)
}

fn invalid_requested_grant_range_error(
    tenant_id: &TenantId,
    grant_policy: TenantGrantPolicy,
    grant_request: QueryGrantRequest,
    reason: &'static str,
) -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        "nexus_query_engine.query_service.requested_grant_range",
        "requested query grant preferred bytes must be at least the requested minimum bytes",
    )
    .with_fact("tenant_id", tenant_id.as_str())
    .with_fact(
        "requested_grant_min_bytes",
        grant_request.requested_grant_min_bytes.map_or_else(|| "none".to_string(), |value| value.to_string()),
    )
    .with_fact(
        "requested_grant_bytes",
        grant_request.requested_grant_bytes.map_or_else(|| "none".to_string(), |value| value.to_string()),
    )
    .with_fact("min_device_budget_bytes", grant_policy.min_bytes)
    .with_fact("tenant_want_device_budget_bytes", grant_policy.want_bytes)
    .with_fact("reason", reason)
}
