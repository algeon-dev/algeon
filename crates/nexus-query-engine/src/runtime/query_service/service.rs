//! Tenant admission, placement, and query-service orchestration.

use super::*;

/// Service-level entry point for tenant-aware native query admission.
#[derive(Clone, Debug)]
pub struct QueryService {
    pub(super) inner: Arc<QueryServiceInner>,
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

/// Process-wide active-query slots currently held on one service device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryServiceDeviceAdmissionSnapshot {
    /// CUDA device ordinal.
    pub device_ordinal: usize,
    /// Slots held across every `QueryService` in this process.
    pub process_active_query_slots: usize,
}

/// Per-device resident graph cache state for service diagnostics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QueryServiceGraphCacheSnapshot {
    /// CUDA device ordinal.
    pub device_ordinal: usize,
    /// Whether a positive resident graph cache cap configured this device.
    pub enabled: bool,
    /// Available plus checked-out entries.
    pub entries: usize,
    /// Device bytes retained by all entries.
    pub resident_bytes: u64,
    /// Device bytes held by checked-out entries.
    pub in_use_bytes: u64,
    /// Successful exact-runtime checkouts.
    pub hits: u64,
    /// Cold graph executions elected by the build coordinator.
    pub misses: u64,
    /// Entries evicted under capacity or governor pressure.
    pub evictions: u64,
    /// Graph-build generations currently in flight.
    pub builds_in_flight: usize,
    /// Follower waits joined since service creation.
    pub build_waits: u64,
}

fn unhealthy_placement_device_error(device: cudf::DeviceId, health: cudf::runtime::DeviceHealthSnapshot) -> QueryError {
    let mut error = QueryError::new(
        QueryErrorCode::CudfPersistentFailure,
        "nexus_query_engine.query_service.placement",
        "CUDA device is unhealthy and cannot admit new queries",
    )
    .with_fact("device_ordinal", device.ordinal())
    .with_fact("device_health", "unhealthy")
    .with_fact("device_fault_count", health.fault_count())
    .with_fact("reason", "device_health_rejected_admission");
    if let Some(first_fault) = health.first_fault() {
        error = error
            .with_fact("first_device_fault_kind", format!("{:?}", first_fault.kind()))
            .with_fact("first_device_fault_operation", first_fault.operation());
    }
    error
}

fn resolve_fallback_placement_device(device_ordinal: usize) -> Result<cudf::DeviceId> {
    #[cfg(test)]
    if let Ok(raw_ordinal) = i32::try_from(device_ordinal)
        && let Some(device) = cudf::testing::device_safety::injected_device_id(raw_ordinal)
    {
        return Ok(device);
    }

    cudf::DeviceId::new(device_ordinal).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.query_service.placement",
            "failed to resolve fallback CUDA device during query placement",
            error,
        )
    })
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

/// Complete request-scoped metadata for one query-service admission attempt.
#[derive(Debug, Default)]
#[non_exhaustive]
pub struct QueryAdmissionRequest {
    /// Priority and adaptive budget-growth metadata.
    pub qos: QueryQos,
    /// Optional request-specific memory grant overrides.
    pub grant_request: QueryGrantRequest,
    /// Optional placement hints used only to order candidate devices.
    pub placement_hint: QueryPlacementHint,
    /// Optional exact-runtime graph-cache hit price.
    ///
    /// QueryService uses this grant request only after it has reserved both an
    /// idle stream-pool runtime and the matching resident graph cache entry.
    /// Device-level cache presence alone never selects this price.
    #[cfg(feature = "cugraph")]
    pub warm_graph_grant_request: Option<QueryGrantRequest>,
    runtime_observation: RuntimeObservationHandle,
    runtime_observation_owner: Option<RuntimeObservationOwner>,
    runtime_observation_activity: Option<Result<QueryActivityLease>>,
}

struct AdmissionObservation {
    handle: RuntimeObservationHandle,
    owner: Option<RuntimeObservationOwner>,
    activity: Option<QueryActivityLease>,
    activity_error: Option<QueryError>,
}

impl AdmissionObservation {
    fn as_ref(&self) -> &RuntimeObservationHandle {
        &self.handle
    }

    fn reject_activity_error(&mut self, admission_request_id: AdmissionRequestId, tenant_id: &TenantId) -> Result<()> {
        let Some(error) = self.activity_error.take() else {
            return Ok(());
        };
        record_admission_lifecycle(
            &self.handle,
            admission_request_id,
            tenant_id,
            admission_rejection(&error),
            0,
            None,
            None,
        );
        Err(error)
    }
}

/// Opaque RAII token retained until query-service cleanup is complete.
///
/// The native engine never invokes the value. Dropping it returns activity
/// ownership to the adapter coordinator after queued admission, contexts,
/// grants, device slots, and cleanup work have all been released.
pub struct QueryActivityLease {
    _lease: Box<dyn Send + Sync>,
}

impl QueryActivityLease {
    pub fn new(lease: impl Send + Sync + 'static) -> Self {
        Self { _lease: Box::new(lease) }
    }
}

impl std::fmt::Debug for QueryActivityLease {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("QueryActivityLease").finish_non_exhaustive()
    }
}

impl QueryAdmissionRequest {
    /// Creates a request with default QoS, tenant-default grants, and no placement hint.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets caller-supplied QoS metadata.
    pub fn with_qos(mut self, qos: QueryQos) -> Self {
        self.qos = qos;
        self
    }

    /// Sets request-scoped minimum/preferred grant overrides.
    pub fn with_grant_request(mut self, grant_request: QueryGrantRequest) -> Self {
        self.grant_request = grant_request;
        self
    }

    /// Sets optional placement metadata.
    pub fn with_placement_hint(mut self, placement_hint: QueryPlacementHint) -> Self {
        self.placement_hint = placement_hint;
        self
    }

    /// Attaches the query observation owner created by the adapter before enqueue.
    pub fn with_runtime_observation_owner(mut self, owner: RuntimeObservationOwner) -> Self {
        self.runtime_observation = owner.handle();
        self.runtime_observation_owner = Some(owner);
        self
    }

    /// Attaches coordinator-owned recording access without query-seal authority.
    pub fn with_runtime_observation_handle(mut self, observation: RuntimeObservationHandle) -> Self {
        self.runtime_observation = observation;
        self.runtime_observation_owner = None;
        self
    }

    /// Retains an adapter-owned activity token through delayed native cleanup.
    pub fn with_runtime_observation_activity(mut self, activity: QueryActivityLease) -> Self {
        self.runtime_observation_activity = Some(Ok(activity));
        self
    }

    /// Carries fallible adapter activity acquisition into native admission.
    ///
    /// Coordinated callers must pass acquisition failure through this field so
    /// admission rejects structurally instead of silently omitting the lease.
    pub fn with_runtime_observation_activity_result(mut self, activity: Result<QueryActivityLease>) -> Self {
        self.runtime_observation_activity = Some(activity);
        self
    }

    fn into_runtime_observation(self) -> AdmissionObservation {
        let (activity, activity_error) = match self.runtime_observation_activity {
            Some(Ok(activity)) => (Some(activity), None),
            Some(Err(error)) => (None, Some(error)),
            None => (None, None),
        };
        AdmissionObservation {
            handle: self.runtime_observation,
            owner: self.runtime_observation_owner,
            activity,
            activity_error,
        }
    }

    /// Sets the grant request that is valid only for a reserved exact-runtime
    /// resident graph cache hit.
    #[cfg(feature = "cugraph")]
    pub fn with_warm_graph_grant_request(mut self, grant_request: QueryGrantRequest) -> Self {
        self.warm_graph_grant_request = Some(grant_request);
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
pub(super) struct QueryServiceInner {
    pub(super) config: QueryServiceConfig,
    tenant_grants: HashMap<TenantId, TenantGrantPolicy>,
    tenant_order: Vec<TenantId>,
    max_tenant_min_bytes: u64,
    fallback_memory_governor: MemoryGovernor,
    pub(super) resources: Option<QueryServiceResources>,
    pub(super) service_epoch: QueryServiceEpoch,
    admission_open: AtomicBool,
    pub(super) admission: Mutex<AdmissionState>,
    admission_available: Condvar,
    admission_notify: Notify,
    #[cfg(test)]
    pub(super) shutdown_waiters: AtomicUsize,
    next_admission_request_id: AtomicU64,
    pub(super) next_admission_sequence: AtomicU64,
}

impl QueryServiceInner {
    pub(super) fn notify_admission_available(&self) {
        self.admission_available.notify_all();
        self.admission_notify.notify_waiters();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct TenantGrantPolicy {
    pub(super) min_bytes: u64,
    pub(super) want_bytes: u64,
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

fn validate_placement_policy_for_device_ordinals(config: &QueryServiceConfig, device_ordinals: &[usize]) -> Result<()> {
    let QueryPlacementPolicy::Pinned { device_ordinal } = config.placement_policy else {
        return Ok(());
    };
    // An empty list means all visible devices; the resolved registry performs
    // the definitive check after inventory discovery.
    if device_ordinals.is_empty() || device_ordinals.contains(&device_ordinal) {
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

fn validate_query_service_tenants(tenants: &[QueryServiceTenantConfig]) -> Result<()> {
    if tenants.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.query_service.tenant.validate",
            "query service requires at least one tenant",
        )
        .with_fact("field", "tenants")
        .with_fact("reason", "empty"));
    }

    let mut tenant_ids = BTreeSet::new();
    for tenant in tenants {
        tenant.validate()?;
        if !tenant_ids.insert(tenant.tenant_id.as_str()) {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.query_service.tenant.validate",
                "query service tenant ids must be unique",
            )
            .with_fact("tenant_id", tenant.tenant_id.as_str())
            .with_fact("reason", "duplicate"));
        }
    }
    Ok(())
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
        // Resource-independent admission must precede process-global allocator
        // installation so those rejections cannot mutate CUDA/RMM state.
        config.validate(&deployment_io_capability)?;
        SourceObjectStoreRegistry::validate_profiles(&config.source_storage_profiles)?;
        validate_query_service_tenants(&tenants)?;
        validate_placement_policy_for_device_ordinals(&config, &device_ordinals)?;
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
        registry_config = registry_config.with_device_memory_manager_base(config.device_memory_manager_base);
        if let Some(device_capacity_bytes) = config.device_capacity_bytes {
            registry_config = registry_config.with_device_capacity_bytes(device_capacity_bytes);
        }
        let devices = super::DeviceRegistry::install(&super::CudfInventory, &registry_config)?;
        let gpu_execution_query_tracker = GpuExecutionQueryTracker::new();
        let gpu_execution_worker_count = gpu_execution_worker_count_per_device(&config);
        let gpu_execution_pools = devices
            .contexts()
            .map(|device| {
                GpuExecutionPool::new_with_tracker(
                    device.device().ordinal(),
                    gpu_execution_worker_count,
                    gpu_execution_query_tracker.clone(),
                )
                .map(|pool| (device.device().ordinal(), pool))
            })
            .collect::<Result<HashMap<_, _>>>()?;
        let local_parquet_source_caches =
            local_parquet_source_caches_for_devices(&devices, config.local_parquet_source_cache_max_retained_bytes)?;
        #[cfg(feature = "cugraph")]
        let resident_graph_caches = resident_graph_caches_for_devices(
            &devices,
            config.resident_graph_cache_max_retained_bytes,
            resident_graph_cache_max_copies_per_key(&config),
        )?;
        let resources = QueryServiceResources {
            devices,
            gpu_execution_pools,
            gpu_execution_query_tracker,
            _governor_claim: None,
            lakehouse_cache_fill_budget: CacheFillByteBudget::new(
                config.lakehouse_cache_fill.budget_bytes.unwrap_or_else(default_lakehouse_cache_fill_budget_bytes),
            )?,
            lakehouse_cache_fill_concurrency: CacheFillConcurrencyGate::new(
                config.lakehouse_cache_fill.max_concurrent_ops,
            )?,
            object_store_control_runtime: default_object_store_control_runtime()?,
            source_object_stores: Arc::new(SourceObjectStoreRegistry::with_profiles(
                config.source_storage_profiles.clone(),
            )?),
            deployment_io_capability,
            local_parquet_source_caches,
            #[cfg(feature = "cugraph")]
            resident_graph_caches,
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
    pub(super) fn new_for_admission_tests(
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
        SourceObjectStoreRegistry::validate_profiles(&config.source_storage_profiles)?;
        validate_query_service_tenants(&tenants)?;
        validate_placement_policy_for_resources(&config, resources.as_ref())?;

        let mut tenant_grants = HashMap::with_capacity(tenants.len());
        let mut tenant_order = Vec::with_capacity(tenants.len());
        let mut max_tenant_want_bytes = 0u64;
        let mut max_tenant_min_bytes = 0u64;
        for tenant in tenants {
            max_tenant_want_bytes = max_tenant_want_bytes.max(tenant.device_budget_bytes);
            max_tenant_min_bytes = max_tenant_min_bytes.max(tenant.min_device_budget_bytes);
            let previous = tenant_grants.insert(
                tenant.tenant_id.clone(),
                TenantGrantPolicy { min_bytes: tenant.min_device_budget_bytes, want_bytes: tenant.device_budget_bytes },
            );
            debug_assert!(previous.is_none(), "tenant preflight guarantees unique ids");
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

        let governor_claim = resources.as_ref().and_then(|resources| resources._governor_claim.as_ref()).cloned();
        let service_epoch = QueryServiceEpoch::new(
            governor_claim.map(|claim| Arc::new(move || claim.release()) as Arc<dyn Fn() + Send + Sync>),
        );
        let inner = Arc::new(QueryServiceInner {
            config,
            tenant_grants,
            tenant_order,
            max_tenant_min_bytes,
            fallback_memory_governor,
            resources,
            service_epoch,
            admission_open: AtomicBool::new(true),
            admission: Mutex::new(AdmissionState::default()),
            admission_available: Condvar::new(),
            admission_notify: Notify::new(),
            #[cfg(test)]
            shutdown_waiters: AtomicUsize::new(0),
            next_admission_request_id: AtomicU64::new(1),
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
        self.admit_query_with_request(
            tenant_id,
            QueryAdmissionRequest::new().with_qos(qos).with_grant_request(grant_request),
        )
    }

    /// Admits a query with consolidated request-scoped admission metadata.
    pub fn admit_query_with_request(
        &self,
        tenant_id: impl Into<TenantId>,
        request: QueryAdmissionRequest,
    ) -> Result<QueryHandle> {
        let tenant_id = tenant_id.into();
        let grant_request = request.grant_request;
        #[cfg(feature = "cugraph")]
        let warm_graph_grant_request = request.warm_graph_grant_request;
        let qos = request.qos;
        let placement_hint = request.placement_hint.clone();
        let mut observation_owner = request.into_runtime_observation();
        let admission_request_id =
            AdmissionRequestId::new(self.inner.next_admission_request_id.fetch_add(1, Ordering::Relaxed));
        observation_owner.reject_activity_error(admission_request_id, &tenant_id)?;
        let grant_policy = match self.tenant_grant_policy_for_request(&tenant_id, grant_request) {
            Ok(grant_policy) => grant_policy,
            Err(error) => {
                record_admission_lifecycle(
                    observation_owner.as_ref(),
                    admission_request_id,
                    &tenant_id,
                    admission_rejection(&error),
                    0,
                    None,
                    None,
                );
                return Err(error);
            },
        };
        #[cfg(feature = "cugraph")]
        let warm_graph_grant_policy = match warm_graph_grant_request
            .map(|request| self.tenant_grant_policy_for_request(&tenant_id, request))
            .transpose()
        {
            Ok(grant_policy) => grant_policy,
            Err(error) => {
                record_admission_lifecycle(
                    observation_owner.as_ref(),
                    admission_request_id,
                    &tenant_id,
                    admission_rejection(&error),
                    0,
                    None,
                    None,
                );
                return Err(error);
            },
        };
        let requested_at = Instant::now();
        let deadline = requested_at.checked_add(self.inner.config.admission_wait_timeout).unwrap_or(requested_at);
        let request = AdmissionRequest {
            admission_request_id,
            observation: observation_owner.handle.clone(),
            observation_owner: observation_owner.owner.clone(),
            observation_activity: observation_owner.activity.take(),
            tenant_id: tenant_id.clone(),
            grant_policy,
            #[cfg(feature = "cugraph")]
            warm_graph_grant_policy,
            qos,
            placement_hint,
            deadline,
        };

        let mut slot_notifications = PendingDeviceSlotNotifications::default();
        let mut state = self.lock_admission()?;
        if self.admission_is_closed(&state) {
            let error = service_closed_error(&tenant_id);
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                &tenant_id,
                admission_rejection(&error),
                state.queued.len(),
                None,
                None,
            );
            return Err(error);
        }
        self.sweep_expired_queued_requests_locked(&mut state);
        if state.queued.len() >= self.inner.config.max_admission_queue_len {
            let error = service_overloaded_error(
                "admission_queue_full",
                &tenant_id,
                state.queued.len(),
                self.inner.config.max_admission_queue_len,
            );
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                &tenant_id,
                admission_rejection(&error),
                state.queued.len(),
                None,
                None,
            );
            return Err(error);
        }
        state.queued.push_back(request);
        record_admission_lifecycle(
            observation_owner.as_ref(),
            admission_request_id,
            &tenant_id,
            AdmissionLifecycleState::Enqueued,
            state.queued.len(),
            None,
            None,
        );

        loop {
            if self.admission_is_closed(&state) {
                let removed = self.remove_queued_request_without_admission_locked(&mut state, admission_request_id);
                self.inner.notify_admission_available();
                if removed.is_some() {
                    record_admission_lifecycle(
                        observation_owner.as_ref(),
                        admission_request_id,
                        &tenant_id,
                        AdmissionLifecycleState::Rejected { reason: "admission_closed".to_string() },
                        state.queued.len(),
                        None,
                        None,
                    );
                }
                return Err(service_closed_error(&tenant_id));
            }
            let request_is_admissible = match self.request_is_admissible(&mut state, admission_request_id) {
                Ok(admissible) => admissible,
                Err(error) => {
                    let removed = self.abandon_queued_request_locked(&mut state, admission_request_id);
                    if removed.is_some() {
                        record_admission_lifecycle(
                            observation_owner.as_ref(),
                            admission_request_id,
                            &tenant_id,
                            admission_rejection(&error),
                            state.queued.len(),
                            None,
                            None,
                        );
                    }
                    return Err(error);
                },
            };
            if request_is_admissible
                && let Some(position) =
                    state.queued.iter().position(|queued| queued.admission_request_id == admission_request_id)
            {
                let placed_query = match self.try_place_query(
                    &tenant_id,
                    admission_request_id,
                    grant_policy,
                    #[cfg(feature = "cugraph")]
                    state.queued[position].warm_graph_grant_policy,
                    qos,
                    &state.queued[position].placement_hint,
                    Some(deadline),
                    state.active_queries,
                    &mut slot_notifications,
                ) {
                    Ok(placed_query) => placed_query,
                    Err(error) => {
                        let removed = self.abandon_queued_request_locked(&mut state, admission_request_id);
                        if removed.is_some() {
                            record_admission_lifecycle(
                                observation_owner.as_ref(),
                                admission_request_id,
                                &tenant_id,
                                admission_rejection(&error),
                                state.queued.len(),
                                None,
                                None,
                            );
                        }
                        return Err(error);
                    },
                };
                let Some(placed_query) = placed_query else {
                    if let Err(error) =
                        self.request_adaptive_grant_shrink_for_blocked_request_locked(&state, &state.queued[position])
                    {
                        let removed = self.abandon_queued_request_locked(&mut state, admission_request_id);
                        if removed.is_some() {
                            record_admission_lifecycle(
                                observation_owner.as_ref(),
                                admission_request_id,
                                &tenant_id,
                                admission_rejection(&error),
                                state.queued.len(),
                                None,
                                None,
                            );
                        }
                        return Err(error);
                    }
                    if slot_notifications.has_pending() {
                        drop(state);
                        slot_notifications.notify();
                        state = self.lock_admission()?;
                        if self.admission_is_closed(&state) {
                            continue;
                        }
                    }
                    let now = Instant::now();
                    if now >= deadline {
                        let removed = self.abandon_queued_request_locked(&mut state, admission_request_id);
                        if removed.is_some() {
                            record_admission_lifecycle(
                                observation_owner.as_ref(),
                                admission_request_id,
                                &tenant_id,
                                AdmissionLifecycleState::TimedOut,
                                state.queued.len(),
                                None,
                                None,
                            );
                        }
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
                        let removed = self.abandon_queued_request_locked(&mut state, admission_request_id);
                        if removed.is_some() {
                            record_admission_lifecycle(
                                observation_owner.as_ref(),
                                admission_request_id,
                                &tenant_id,
                                AdmissionLifecycleState::TimedOut,
                                state.queued.len(),
                                None,
                                None,
                            );
                        }
                        return Err(self.memory_grant_wait_timeout_error(&tenant_id, grant_policy, state.queued.len()));
                    }
                    continue;
                };
                let admitted = state.queued.remove(position).expect("admissible request exists");
                if let Err(error) = self.attach_admission_release_wakeup(&placed_query.memory_grant) {
                    record_admission_lifecycle(
                        &admitted.observation,
                        admission_request_id,
                        &tenant_id,
                        admission_rejection(&error),
                        state.queued.len(),
                        None,
                        None,
                    );
                    return Err(error);
                }
                if let Err(error) = self.attach_admission_budget_decrease_wakeup(&placed_query.memory_grant) {
                    record_admission_lifecycle(
                        &admitted.observation,
                        admission_request_id,
                        &tenant_id,
                        admission_rejection(&error),
                        state.queued.len(),
                        None,
                        None,
                    );
                    return Err(error);
                }
                if self.admission_is_closed(&state) {
                    let error = service_closed_error(&tenant_id);
                    record_admission_lifecycle(
                        &admitted.observation,
                        admission_request_id,
                        &tenant_id,
                        admission_rejection(&error),
                        state.queued.len(),
                        None,
                        None,
                    );
                    self.inner.notify_admission_available();
                    return Err(error);
                }
                state.active_queries = state.active_queries.saturating_add(1);
                self.register_active_adaptive_grant(
                    &mut state,
                    admission_request_id,
                    placed_query.device_ordinal,
                    placed_query.grant_policy,
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
                record_admission_lifecycle(
                    &admitted.observation,
                    admission_request_id,
                    &tenant_id,
                    AdmissionLifecycleState::Admitted,
                    state.queued.len(),
                    Some(device_ordinal),
                    Some(device_budget_bytes),
                );
                return Ok(QueryHandle {
                    lifetime: QueryHandleLifetime::new(
                        Arc::clone(&service),
                        tenant_id.clone(),
                        admission_request_id,
                        admitted.observation,
                        admitted.observation_owner,
                        admitted.observation_activity,
                        placed_query.memory_grant,
                        placed_query.device_slot,
                        #[cfg(feature = "cugraph")]
                        placed_query.warm_graph_reservation,
                    ),
                    service,
                    tenant_id,
                    admission_request_id,
                    admission_sequence,
                    admission_wait: requested_at.elapsed(),
                    device_budget_bytes,
                    device_ordinal,
                });
            }

            let now = Instant::now();
            if now >= deadline {
                let removed = self.abandon_queued_request_locked(&mut state, admission_request_id);
                if removed.is_some() {
                    record_admission_lifecycle(
                        observation_owner.as_ref(),
                        admission_request_id,
                        &tenant_id,
                        AdmissionLifecycleState::TimedOut,
                        state.queued.len(),
                        None,
                        None,
                    );
                }
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
                let removed = self.abandon_queued_request_locked(&mut state, admission_request_id);
                if removed.is_some() {
                    record_admission_lifecycle(
                        observation_owner.as_ref(),
                        admission_request_id,
                        &tenant_id,
                        AdmissionLifecycleState::TimedOut,
                        state.queued.len(),
                        None,
                        None,
                    );
                }
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
        self.admit_async_with_request(
            tenant_id,
            QueryAdmissionRequest::new().with_qos(qos).with_grant_request(grant_request),
        )
        .await
    }

    /// Admits a query asynchronously with consolidated request-scoped admission metadata.
    pub async fn admit_async_with_request(
        &self,
        tenant_id: impl Into<TenantId>,
        request: QueryAdmissionRequest,
    ) -> Result<QueryHandle> {
        let tenant_id = tenant_id.into();
        let mut waiter = self.enqueue_admission_with_request(tenant_id, request)?;
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
        self.enqueue_admission_with_request(
            tenant_id,
            QueryAdmissionRequest::new().with_qos(qos).with_grant_request(grant_request),
        )
    }

    /// Enqueues a query admission request with consolidated request-scoped metadata.
    pub fn enqueue_admission_with_request(
        &self,
        tenant_id: impl Into<TenantId>,
        request: QueryAdmissionRequest,
    ) -> Result<QueryAdmissionWaiter> {
        let tenant_id = tenant_id.into();
        let grant_request = request.grant_request;
        #[cfg(feature = "cugraph")]
        let warm_graph_grant_request = request.warm_graph_grant_request;
        let qos = request.qos;
        let placement_hint = request.placement_hint.clone();
        let mut observation_owner = request.into_runtime_observation();
        let admission_request_id =
            AdmissionRequestId::new(self.inner.next_admission_request_id.fetch_add(1, Ordering::Relaxed));
        observation_owner.reject_activity_error(admission_request_id, &tenant_id)?;
        let grant_policy = match self.tenant_grant_policy_for_request(&tenant_id, grant_request) {
            Ok(grant_policy) => grant_policy,
            Err(error) => {
                record_admission_lifecycle(
                    observation_owner.as_ref(),
                    admission_request_id,
                    &tenant_id,
                    admission_rejection(&error),
                    0,
                    None,
                    None,
                );
                return Err(error);
            },
        };
        #[cfg(feature = "cugraph")]
        let warm_graph_grant_policy = match warm_graph_grant_request
            .map(|request| self.tenant_grant_policy_for_request(&tenant_id, request))
            .transpose()
        {
            Ok(grant_policy) => grant_policy,
            Err(error) => {
                record_admission_lifecycle(
                    observation_owner.as_ref(),
                    admission_request_id,
                    &tenant_id,
                    admission_rejection(&error),
                    0,
                    None,
                    None,
                );
                return Err(error);
            },
        };
        let requested_at = Instant::now();
        let deadline = requested_at.checked_add(self.inner.config.admission_wait_timeout).unwrap_or(requested_at);
        let mut state = self.lock_admission()?;
        if self.admission_is_closed(&state) {
            let error = service_closed_error(&tenant_id);
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                &tenant_id,
                admission_rejection(&error),
                state.queued.len(),
                None,
                None,
            );
            return Err(error);
        }
        self.sweep_expired_queued_requests_locked(&mut state);
        if state.queued.len() >= self.inner.config.max_admission_queue_len {
            let error = service_overloaded_error(
                "admission_queue_full",
                &tenant_id,
                state.queued.len(),
                self.inner.config.max_admission_queue_len,
            );
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                &tenant_id,
                admission_rejection(&error),
                state.queued.len(),
                None,
                None,
            );
            return Err(error);
        }
        state.queued.push_back(AdmissionRequest {
            admission_request_id,
            observation: observation_owner.handle.clone(),
            observation_owner: observation_owner.owner.clone(),
            observation_activity: observation_owner.activity.take(),
            tenant_id: tenant_id.clone(),
            grant_policy,
            #[cfg(feature = "cugraph")]
            warm_graph_grant_policy,
            qos,
            placement_hint,
            deadline,
        });
        record_admission_lifecycle(
            observation_owner.as_ref(),
            admission_request_id,
            &tenant_id,
            AdmissionLifecycleState::Enqueued,
            state.queued.len(),
            None,
            None,
        );
        self.inner.notify_admission_available();
        Ok(QueryAdmissionWaiter {
            service: self.clone(),
            tenant_id,
            admission_request_id,
            observation: observation_owner.handle,
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
        self.try_admit_query_with_request(
            tenant_id,
            QueryAdmissionRequest::new().with_qos(qos).with_grant_request(grant_request),
        )
    }

    /// Attempts to admit a query with consolidated request-scoped metadata.
    pub fn try_admit_query_with_request(
        &self,
        tenant_id: impl Into<TenantId>,
        request: QueryAdmissionRequest,
    ) -> Result<Option<QueryHandle>> {
        let tenant_id = tenant_id.into();
        let grant_request = request.grant_request;
        #[cfg(feature = "cugraph")]
        let warm_graph_grant_request = request.warm_graph_grant_request;
        let qos = request.qos;
        let placement_hint = request.placement_hint.clone();
        let mut observation_owner = request.into_runtime_observation();
        let admission_request_id =
            AdmissionRequestId::new(self.inner.next_admission_request_id.fetch_add(1, Ordering::Relaxed));
        observation_owner.reject_activity_error(admission_request_id, &tenant_id)?;
        let grant_policy = match self.tenant_grant_policy_for_request(&tenant_id, grant_request) {
            Ok(grant_policy) => grant_policy,
            Err(error) => {
                record_admission_lifecycle(
                    observation_owner.as_ref(),
                    admission_request_id,
                    &tenant_id,
                    admission_rejection(&error),
                    0,
                    None,
                    None,
                );
                return Err(error);
            },
        };
        #[cfg(feature = "cugraph")]
        let warm_graph_grant_policy = match warm_graph_grant_request
            .map(|request| self.tenant_grant_policy_for_request(&tenant_id, request))
            .transpose()
        {
            Ok(grant_policy) => grant_policy,
            Err(error) => {
                record_admission_lifecycle(
                    observation_owner.as_ref(),
                    admission_request_id,
                    &tenant_id,
                    admission_rejection(&error),
                    0,
                    None,
                    None,
                );
                return Err(error);
            },
        };
        let mut slot_notifications = PendingDeviceSlotNotifications::default();
        let mut state = self.lock_admission()?;
        if self.admission_is_closed(&state) {
            let error = service_closed_error(&tenant_id);
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                &tenant_id,
                admission_rejection(&error),
                state.queued.len(),
                None,
                None,
            );
            return Err(error);
        }
        self.sweep_expired_queued_requests_locked(&mut state);
        if state.active_queries >= self.inner.config.max_concurrent_queries || !state.queued.is_empty() {
            return Ok(None);
        }
        let placed_query = match self.try_place_query(
            &tenant_id,
            admission_request_id,
            grant_policy,
            #[cfg(feature = "cugraph")]
            warm_graph_grant_policy,
            qos,
            &placement_hint,
            None,
            state.active_queries,
            &mut slot_notifications,
        ) {
            Ok(placed_query) => placed_query,
            Err(error) => {
                record_admission_lifecycle(
                    observation_owner.as_ref(),
                    admission_request_id,
                    &tenant_id,
                    admission_rejection(&error),
                    state.queued.len(),
                    None,
                    None,
                );
                return Err(error);
            },
        };
        let Some(placed_query) = placed_query else {
            return Ok(None);
        };
        if let Err(error) = self.attach_admission_release_wakeup(&placed_query.memory_grant) {
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                &tenant_id,
                admission_rejection(&error),
                state.queued.len(),
                None,
                None,
            );
            return Err(error);
        }
        if let Err(error) = self.attach_admission_budget_decrease_wakeup(&placed_query.memory_grant) {
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                &tenant_id,
                admission_rejection(&error),
                state.queued.len(),
                None,
                None,
            );
            return Err(error);
        }
        if self.admission_is_closed(&state) {
            let error = service_closed_error(&tenant_id);
            record_admission_lifecycle(
                observation_owner.as_ref(),
                admission_request_id,
                &tenant_id,
                admission_rejection(&error),
                state.queued.len(),
                None,
                None,
            );
            return Err(error);
        }
        state.active_queries = state.active_queries.saturating_add(1);
        self.register_active_adaptive_grant(
            &mut state,
            admission_request_id,
            placed_query.device_ordinal,
            placed_query.grant_policy,
            &placed_query.memory_grant,
        );
        self.rotate_after_admission(&mut state, &tenant_id);
        let admission_sequence = self.inner.next_admission_sequence.fetch_add(1, Ordering::Relaxed);
        let service = Arc::clone(&self.inner);
        let device_budget_bytes = placed_query.memory_grant.budget_bytes();
        let device_ordinal = placed_query.device_ordinal;
        record_admission_lifecycle(
            observation_owner.as_ref(),
            admission_request_id,
            &tenant_id,
            AdmissionLifecycleState::Admitted,
            state.queued.len(),
            Some(device_ordinal),
            Some(device_budget_bytes),
        );
        Ok(Some(QueryHandle {
            lifetime: QueryHandleLifetime::new(
                Arc::clone(&service),
                tenant_id.clone(),
                admission_request_id,
                observation_owner.handle,
                observation_owner.owner,
                observation_owner.activity,
                placed_query.memory_grant,
                placed_query.device_slot,
                #[cfg(feature = "cugraph")]
                placed_query.warm_graph_reservation,
            ),
            service,
            tenant_id,
            admission_request_id,
            admission_sequence,
            admission_wait: Duration::ZERO,
            device_budget_bytes,
            device_ordinal,
        }))
    }

    /// Closes admission and wakes every queued waiter.
    ///
    /// The atomic gate closes before the best-effort mutex update, so this
    /// signal never waits behind an admission transaction. Commit paths
    /// recheck the gate while holding the admission mutex. Active query
    /// handles keep their slots until dropped or finished.
    pub fn close(&self) -> Result<()> {
        self.inner.admission_open.store(false, Ordering::Release);
        let close_result = match self.inner.admission.try_lock() {
            Ok(mut state) => {
                state.closed = true;
                Ok(())
            },
            Err(std::sync::TryLockError::Poisoned(_)) => Err(QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.query_service.admission",
                "query service admission mutex was poisoned",
            )),
            Err(std::sync::TryLockError::WouldBlock) => Ok(()),
        };
        self.inner.service_epoch.invalidate();
        let stream_pool_close_result =
            self.inner.resources.as_ref().map_or(Ok(()), QueryServiceResources::close_stream_pools);
        self.inner.notify_admission_available();
        close_result.and(stream_pool_close_result)
    }

    /// Closes admission and waits for every admitted handle and queued waiter
    /// to leave the service.
    ///
    /// This is the synchronous teardown boundary used when an async server
    /// future is cancelled: returning guarantees no query cleanup still owns
    /// service runtime resources or the process-wide governor claim.
    pub fn close_and_wait_for_quiescence(&self, timeout: Duration) -> Result<()> {
        let deadline = Instant::now().checked_add(timeout).unwrap_or_else(Instant::now);
        self.close_and_wait_for_quiescence_until(deadline)
    }

    /// Closes admission and waits for complete cleanup under one caller-owned
    /// absolute deadline.
    ///
    /// Time spent acquiring the admission lock or closing the service consumes
    /// this same deadline instead of starting a second grace period.
    pub fn close_and_wait_for_quiescence_until(&self, deadline: Instant) -> Result<()> {
        let operation = "nexus_query_engine.query_service.close_and_wait_for_quiescence";
        self.inner.admission_open.store(false, Ordering::Release);
        self.inner.service_epoch.invalidate();
        let stream_pool_close_result =
            self.inner.resources.as_ref().map_or(Ok(()), QueryServiceResources::close_stream_pools);
        self.inner.notify_admission_available();
        // Local-Parquet cache checkouts may outlive their query handle. Drain
        // cache ownership before waiting for those query-scoped overlays so a
        // pinned escaped output cannot form a shutdown ordering cycle.
        let local_cache_result = self
            .inner
            .resources
            .as_ref()
            .map_or(Ok(()), |resources| resources.close_and_drain_local_parquet_source_caches_until(deadline));
        let mut state = self.lock_admission_until(deadline, operation)?;
        state.closed = true;
        let mut waited_for_quiescence = false;
        loop {
            if state.active_queries == 0 && state.cleanup_in_progress == 0 && state.queued.is_empty() {
                if waited_for_quiescence && crate::runtime::deadline::deadline_expired(deadline) {
                    return Err(query_service_shutdown_timeout(operation, Some(&state), "admission_quiescence"));
                }
                break;
            }
            if crate::runtime::deadline::deadline_expired(deadline) {
                return Err(query_service_shutdown_timeout(operation, Some(&state), "admission_quiescence"));
            }
            let Some(remaining) = crate::runtime::deadline::deadline_remaining(deadline) else {
                return Err(query_service_shutdown_timeout(operation, Some(&state), "admission_quiescence"));
            };
            waited_for_quiescence = true;
            #[cfg(test)]
            self.inner.shutdown_waiters.fetch_add(1, Ordering::AcqRel);
            let waited = self.inner.admission_available.wait_timeout(state, remaining);
            #[cfg(test)]
            self.inner.shutdown_waiters.fetch_sub(1, Ordering::AcqRel);
            let (next, timeout) = waited.map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    operation,
                    "query service admission condition variable was poisoned during shutdown",
                )
            })?;
            state = next;
            if timeout.timed_out() {
                return Err(query_service_shutdown_timeout(operation, Some(&state), "admission_quiescence"));
            }
        }
        drop(state);
        #[cfg(feature = "cugraph")]
        let resident_cache_result = self
            .inner
            .resources
            .as_ref()
            .map_or(Ok(()), |resources| resources.close_and_drain_resident_graph_caches_until(deadline));
        let stream_pool_drain_result = self
            .inner
            .resources
            .as_ref()
            .map_or(Ok(()), |resources| resources.close_and_drain_stream_pools_until(deadline));
        let epoch_result = self.inner.service_epoch.wait_for_idle_until(deadline, operation);
        stream_pool_close_result?;
        local_cache_result?;
        #[cfg(feature = "cugraph")]
        resident_cache_result?;
        stream_pool_drain_result?;
        epoch_result?;
        if crate::runtime::deadline::deadline_expired(deadline) {
            return Err(query_service_shutdown_timeout(operation, None, "final_completion"));
        }
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

    pub(super) fn adaptive_shrink_deficit_bytes(
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
        admission_request_id: AdmissionRequestId,
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
        let mut request = GrantRequest::new(admission_request_id, grant_policy.min_bytes, want_bytes);
        request = request.with_qos(qos);
        if let Some(deadline) = deadline {
            request = request.with_deadline(deadline);
        }
        governor.try_request(request)
    }

    pub(super) fn try_place_query(
        &self,
        tenant_id: &TenantId,
        admission_request_id: AdmissionRequestId,
        grant_policy: TenantGrantPolicy,
        #[cfg(feature = "cugraph")] warm_graph_grant_policy: Option<TenantGrantPolicy>,
        qos: QueryQos,
        placement_hint: &QueryPlacementHint,
        deadline: Option<Instant>,
        active_queries: usize,
        slot_notifications: &mut PendingDeviceSlotNotifications,
    ) -> Result<Option<PlacedQuery>> {
        let Some(resources) = &self.inner.resources else {
            let fallback_device = resolve_fallback_placement_device(self.inner.config.device_ordinal)?;
            let health = cudf::runtime::device_health_snapshot(fallback_device);
            if health.status() == cudf::runtime::DeviceHealthStatus::Unhealthy {
                return Err(unhealthy_placement_device_error(fallback_device, health));
            }
            if !self.fallback_active_query_cap_allows(active_queries) {
                return Ok(None);
            }
            let Some(memory_grant) = self.try_grant_from_governor(
                &self.inner.fallback_memory_governor,
                admission_request_id,
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
                grant_policy,
                #[cfg(feature = "cugraph")]
                warm_graph_reservation: None,
            }));
        };

        let placement_request = PlacementRequest::with_grant_range(grant_policy.min_bytes, grant_policy.want_bytes)
            .with_tenant_id(tenant_id.as_str())
            .with_placement_hint(placement_hint.clone());
        let mut any_device_can_fit_minimum = false;
        let mut healthy_candidates = 0usize;
        let mut first_unhealthy = None;
        for device in self.placement_candidates(resources, &placement_request)? {
            let health = cudf::runtime::device_health_snapshot(device);
            if health.status() == cudf::runtime::DeviceHealthStatus::Unhealthy {
                first_unhealthy.get_or_insert((device, health));
                continue;
            }
            healthy_candidates = healthy_candidates.saturating_add(1);
            let context = resources.devices.context(device)?;
            if context.capacity_bytes() >= grant_policy.min_bytes {
                any_device_can_fit_minimum = true;
            }
            let available_bytes = context.available_bytes()?;
            if available_bytes < grant_policy.min_bytes {
                continue;
            }
            // This helper accounts for the query being selected; compute it
            // before the device slot increments the active-query counter.
            let reserved_for_remaining_slots =
                self.reserved_min_bytes_for_selected_device(resources, device, active_queries)?;
            let Some(device_slot) = context.try_acquire_query_slot(self.inner.config.max_queries_per_device) else {
                continue;
            };
            #[cfg(feature = "cugraph")]
            let warm_graph_reservation = match (warm_graph_grant_policy, placement_hint.resident_graph_key()) {
                (Some(_warm_policy), Some(resident_graph_key)) => {
                    resources.try_reserve_warm_graph(context, resident_graph_key)?
                },
                _ => None,
            };
            #[cfg(feature = "cugraph")]
            let selected_grant_policy = if warm_graph_reservation.is_some() {
                warm_graph_grant_policy.expect("warm reservation requires warm graph grant policy")
            } else {
                grant_policy
            };
            #[cfg(not(feature = "cugraph"))]
            let selected_grant_policy = grant_policy;
            let Some(want_bytes) =
                self.grant_budget_bytes(available_bytes, selected_grant_policy, reserved_for_remaining_slots)
            else {
                slot_notifications.release_slot(device_slot);
                continue;
            };
            let mut request = GrantRequest::new(admission_request_id, selected_grant_policy.min_bytes, want_bytes);
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
                        grant_policy: selected_grant_policy,
                        #[cfg(feature = "cugraph")]
                        warm_graph_reservation,
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
        if healthy_candidates == 0
            && let Some((device, health)) = first_unhealthy
        {
            return Err(unhealthy_placement_device_error(device, health));
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

    pub(super) fn attach_admission_release_wakeup(&self, grant: &MemoryGrant) -> Result<()> {
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
            if guard.cleanup_in_progress != 0 {
                return;
            }
            if guard.queued.is_empty() {
                service.grow_active_adaptive_grants_locked(&mut guard);
            }
            service.inner.notify_admission_available();
        }))
    }

    pub(super) fn attach_admission_budget_decrease_wakeup(&self, grant: &MemoryGrant) -> Result<()> {
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
            if guard.cleanup_in_progress != 0 {
                return;
            }
            if guard.queued.is_empty() {
                service.grow_active_adaptive_grants_locked(&mut guard);
            }
            service.inner.notify_admission_available();
        }))
    }

    pub(super) fn register_active_adaptive_grant(
        &self,
        state: &mut AdmissionState,
        admission_request_id: AdmissionRequestId,
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
            admission_request_id,
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

    pub(super) fn grow_active_adaptive_grants_locked(&self, state: &mut AdmissionState) {
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
            let admission_request_ids = state
                .active_adaptive_grants
                .iter()
                .filter_map(|(admission_request_id, active)| {
                    (active.device_ordinal == device_ordinal).then_some(*admission_request_id)
                })
                .collect::<Vec<_>>();
            for admission_request_id in admission_request_ids {
                let Some(active) = state.active_adaptive_grants.get(&admission_request_id) else {
                    continue;
                };
                let current_bytes = active.memory_grant.budget_bytes();
                let target_bytes = active.target_bytes.max(active.min_bytes);
                if current_bytes >= target_bytes {
                    continue;
                }
                candidates.push(AdaptiveGrowthCandidate {
                    admission_request_id,
                    missing_bytes: target_bytes.saturating_sub(current_bytes),
                    weight: active.memory_grant.qos().budget_weight(),
                });
            }
            for (admission_request_id, allocated_bytes) in
                weighted_adaptive_growth_allocations(&candidates, available_bytes)
            {
                let Some(active) = state.active_adaptive_grants.get(&admission_request_id).cloned() else {
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

    pub(super) fn request_adaptive_grant_shrink_for_blocked_request_locked(
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
            .with_tenant_id(request.tenant_id.as_str())
            .with_placement_hint(request.placement_hint.clone());
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
        let candidates = self.inner.config.placement_policy.candidates(&resources.devices, request)?;
        #[cfg(feature = "cugraph")]
        if let Some(resident_graph_key) = request.placement_hint.resident_graph_key() {
            let mut hit_ordinals = BTreeSet::new();
            for device in &candidates {
                if let Some(cache) = resources.resident_graph_cache_for_device(device.ordinal()) {
                    match cache.contains_resident_key(resident_graph_key) {
                        Ok(true) => {
                            hit_ordinals.insert(device.ordinal());
                        },
                        Ok(false) => {},
                        Err(error) => {
                            eprintln!(
                                "[nexus-query-engine] warning: failed to inspect resident graph cache for placement affinity on device {}: {error}",
                                device.ordinal()
                            );
                        },
                    }
                }
            }
            return Ok(rank_cache_affine_candidates(candidates, &hit_ordinals));
        }
        Ok(candidates)
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
        let mut admission_request_ids = state
            .active_adaptive_grants
            .iter()
            .filter_map(|(admission_request_id, active)| {
                (active.device_ordinal == device_ordinal).then_some(*admission_request_id)
            })
            .collect::<Vec<_>>();
        admission_request_ids.sort_unstable();

        let mut remaining_deficit = deficit_bytes;
        for admission_request_id in admission_request_ids {
            let Some(active) = state.active_adaptive_grants.get(&admission_request_id) else {
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

    pub(super) fn refresh_active_adaptive_shrink_requests_locked(&self, state: &AdmissionState) {
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

    pub(super) fn remove_queued_request_without_admission_locked(
        &self,
        state: &mut AdmissionState,
        admission_request_id: AdmissionRequestId,
    ) -> Option<AdmissionRequest> {
        let removed = remove_queued_request(state, admission_request_id);
        if removed.is_some() {
            self.refresh_active_adaptive_shrink_requests_locked(state);
            if state.queued.is_empty() {
                self.grow_active_adaptive_grants_locked(state);
            }
        }
        removed
    }

    pub(super) fn abandon_queued_request_locked(
        &self,
        state: &mut AdmissionState,
        admission_request_id: AdmissionRequestId,
    ) -> Option<AdmissionRequest> {
        let removed = self.remove_queued_request_without_admission_locked(state, admission_request_id);
        if removed.is_some() {
            self.inner.notify_admission_available();
        }
        removed
    }

    /// Returns one memory-governor report for each installed service device.
    pub fn memory_governor_reports(&self) -> Result<Vec<MemoryGovernorReport>> {
        let Some(resources) = &self.inner.resources else {
            return Ok(vec![self.inner.fallback_memory_governor.report()?]);
        };
        resources.devices.contexts().map(|context| context.memory_governor().report()).collect()
    }

    /// Returns process-wide active-query slot counts for each service device.
    pub fn device_admission_snapshots(&self) -> Vec<QueryServiceDeviceAdmissionSnapshot> {
        let Some(resources) = &self.inner.resources else {
            return Vec::new();
        };
        resources
            .devices
            .contexts()
            .map(|context| QueryServiceDeviceAdmissionSnapshot {
                device_ordinal: context.device().ordinal(),
                process_active_query_slots: context.active_queries(),
            })
            .collect()
    }

    /// Returns one resident graph cache row per configured service device,
    /// including disabled all-zero rows.
    pub fn resident_graph_cache_snapshots(&self) -> Result<Vec<QueryServiceGraphCacheSnapshot>> {
        let Some(resources) = &self.inner.resources else {
            return Ok(vec![QueryServiceGraphCacheSnapshot {
                device_ordinal: self.inner.config.device_ordinal,
                enabled: self.inner.config.resident_graph_cache_max_retained_bytes.is_some(),
                ..QueryServiceGraphCacheSnapshot::default()
            }]);
        };
        resources
            .devices
            .contexts()
            .map(|context| {
                let device_ordinal = context.device().ordinal();
                #[cfg(feature = "cugraph")]
                if let Some(cache) = resources.resident_graph_cache_for_device(device_ordinal) {
                    let snapshot = cache.snapshot()?;
                    return Ok(QueryServiceGraphCacheSnapshot {
                        device_ordinal,
                        enabled: true,
                        entries: snapshot.entries,
                        resident_bytes: snapshot.resident_bytes,
                        in_use_bytes: snapshot.in_use_bytes,
                        hits: snapshot.hits,
                        misses: snapshot.misses,
                        evictions: snapshot.evictions,
                        builds_in_flight: snapshot.builds_in_flight,
                        build_waits: snapshot.build_waits,
                    });
                }
                Ok(QueryServiceGraphCacheSnapshot { device_ordinal, ..QueryServiceGraphCacheSnapshot::default() })
            })
            .collect()
    }

    /// Samples current CUDA and optional NVML telemetry for each service device.
    ///
    /// Telemetry is observability-only and is never consulted by admission or
    /// placement. A failed substrate is represented by status fields in each
    /// snapshot rather than failing diagnostics as a whole.
    pub fn device_telemetry_snapshots(&self) -> Vec<DeviceTelemetrySnapshot> {
        self.device_telemetry_snapshots_with(&CudfNvmlTelemetry)
    }

    fn device_telemetry_snapshots_with(&self, telemetry: &dyn GpuTelemetry) -> Vec<DeviceTelemetrySnapshot> {
        let Some(resources) = &self.inner.resources else {
            return Vec::new();
        };
        super::gpu::inventory::device_telemetry_snapshots_for_ordinals(telemetry, resources.devices.device_ordinals())
    }

    /// Returns one device memory allocation-failure report for each installed service device.
    pub fn device_memory_failure_reports(&self) -> Vec<DeviceMemoryFailureReport> {
        let Some(resources) = &self.inner.resources else {
            return Vec::new();
        };
        resources.devices.memory_failure_reports()
    }

    /// Returns one dedicated GPU execution-pool report per service device.
    pub fn gpu_execution_pool_reports(&self) -> Vec<GpuExecutionPoolSnapshot> {
        let Some(resources) = &self.inner.resources else {
            return Vec::new();
        };
        resources
            .devices
            .device_ordinals()
            .iter()
            .filter_map(|ordinal| resources.gpu_execution_pools.get(ordinal).map(GpuExecutionPool::snapshot))
            .collect()
    }

    /// Returns the exact current and peak query concurrency across all service GPU pools.
    pub fn gpu_execution_query_report(&self) -> GpuExecutionQuerySnapshot {
        self.inner
            .resources
            .as_ref()
            .map(|resources| resources.gpu_execution_query_tracker.snapshot())
            .unwrap_or(GpuExecutionQuerySnapshot { active_query_jobs: 0, peak_active_query_jobs: 0 })
    }

    /// Blocks until every created service runtime has completed background
    /// stream cleanup and returned to its per-device pool.
    ///
    /// This provides an explicit quiescence boundary for warmup/capture and
    /// graceful drain. It does not require unused lazy pool slots to be
    /// created. Active queries cause the call to wait until `timeout` expires.
    pub fn wait_for_stream_pools_idle(&self, timeout: Duration) -> Result<()> {
        let Some(resources) = &self.inner.resources else {
            return Ok(());
        };
        let deadline = Instant::now().checked_add(timeout).unwrap_or_else(Instant::now);
        for context in resources.devices.contexts() {
            context
                .stream_pool()
                .wait_until_idle(deadline, "nexus_query_engine.query_service.wait_for_stream_pools_idle")?;
        }
        Ok(())
    }

    /// Blocks until each pool has only its declared long-lived runtime leases
    /// checked out and every other created runtime has completed cleanup.
    ///
    /// `expected_leased_runtimes` contains `(device_ordinal, count)` pairs.
    /// Omitted configured devices expect zero leases. Duplicate or unknown
    /// ordinals are rejected instead of being silently merged.
    pub fn wait_for_stream_pools_quiescent(
        &self,
        expected_leased_runtimes: &[(usize, usize)],
        timeout: Duration,
    ) -> Result<()> {
        let operation = "nexus_query_engine.query_service.wait_for_stream_pools_quiescent";
        let Some(resources) = &self.inner.resources else {
            if expected_leased_runtimes.iter().all(|(_device_ordinal, expected)| *expected == 0) {
                return Ok(());
            }
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                operation,
                "expected stream-pool leases require query service runtime resources",
            )
            .with_fact("reason", "runtime_resources_unavailable"));
        };
        let mut expected_by_device = BTreeMap::new();
        for &(device_ordinal, expected) in expected_leased_runtimes {
            if expected_by_device.insert(device_ordinal, expected).is_some() {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    operation,
                    "expected stream-pool lease counts contain a duplicate device ordinal",
                )
                .with_fact("device_ordinal", device_ordinal)
                .with_fact("reason", "duplicate_device_ordinal"));
            }
            resources.devices.context_by_ordinal(device_ordinal)?;
        }
        let deadline = Instant::now().checked_add(timeout).unwrap_or_else(Instant::now);
        for context in resources.devices.contexts() {
            let expected = expected_by_device.remove(&context.device().ordinal()).unwrap_or(0);
            context.stream_pool().wait_until_quiescent(expected, deadline, operation)?;
        }
        Ok(())
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

    pub(super) fn lock_admission(&self) -> Result<std::sync::MutexGuard<'_, AdmissionState>> {
        self.inner.admission.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.query_service.admission",
                "query service admission mutex was poisoned",
            )
        })
    }

    pub(super) fn admission_is_closed(&self, state: &AdmissionState) -> bool {
        !self.inner.admission_open.load(Ordering::Acquire) || state.closed
    }

    fn lock_admission_until(
        &self,
        deadline: Instant,
        operation: &'static str,
    ) -> Result<std::sync::MutexGuard<'_, AdmissionState>> {
        match crate::runtime::deadline::mutex_lock_until(&self.inner.admission, deadline) {
            Ok(state) => Ok(state),
            Err(crate::runtime::deadline::DeadlineMutexLockError::Poisoned) => self.lock_admission(),
            Err(crate::runtime::deadline::DeadlineMutexLockError::TimedOut) => {
                Err(query_service_shutdown_timeout(operation, None, "admission_mutex"))
            },
        }
    }

    pub(super) fn request_is_admissible(
        &self,
        state: &mut AdmissionState,
        admission_request_id: AdmissionRequestId,
    ) -> Result<bool> {
        self.sweep_expired_queued_requests_locked(state);
        if state.active_queries >= self.inner.config.max_concurrent_queries {
            return Ok(false);
        }
        let Some(position) = self.select_next_request_position(state)? else {
            return Ok(false);
        };
        Ok(state.queued[position].admission_request_id == admission_request_id)
    }

    fn sweep_expired_queued_requests_locked(&self, state: &mut AdmissionState) {
        let now = Instant::now();
        let mut expired = Vec::new();
        let mut retained = VecDeque::with_capacity(state.queued.len());
        while let Some(request) = state.queued.pop_front() {
            if request.deadline <= now {
                expired.push(request);
            } else {
                retained.push_back(request);
            }
        }
        state.queued = retained;
        if !expired.is_empty() {
            self.refresh_active_adaptive_shrink_requests_locked(state);
            if state.queued.is_empty() {
                self.grow_active_adaptive_grants_locked(state);
            }
            self.inner.notify_admission_available();
            for request in expired {
                record_admission_lifecycle(
                    &request.observation,
                    request.admission_request_id,
                    &request.tenant_id,
                    AdmissionLifecycleState::TimedOut,
                    state.queued.len(),
                    None,
                    None,
                );
            }
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

    pub(super) fn memory_grant_wait_timeout_error(
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

    pub(super) fn rotate_after_admission(&self, state: &mut AdmissionState, tenant_id: &TenantId) {
        if let Some(index) = self.inner.tenant_order.iter().position(|candidate| candidate == tenant_id) {
            state.next_tenant_index = (index + 1) % self.inner.tenant_order.len();
        }
    }
}

fn query_service_shutdown_timeout(
    operation: &'static str,
    state: Option<&AdmissionState>,
    deadline_stage: &'static str,
) -> QueryError {
    let error = QueryError::new(
        QueryErrorCode::ServiceOverloaded,
        operation,
        "timed out waiting for query service shutdown cleanup",
    )
    .with_fact("reason", "query_service_shutdown_timeout")
    .with_fact("deadline_stage", deadline_stage);
    match state {
        Some(state) => error
            .with_fact("active_queries", state.active_queries)
            .with_fact("cleanup_in_progress", state.cleanup_in_progress)
            .with_fact("queued_queries", state.queued.len()),
        None => error,
    }
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
