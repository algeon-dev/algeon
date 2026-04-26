//! QueryService constructors and config preflight validation.

use super::*;

pub(in crate::runtime::query_service::service) fn validate_placement_policy_for_resources(
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

pub(in crate::runtime::query_service::service) fn validate_placement_policy_for_device_ordinals(
    config: &QueryServiceConfig,
    device_ordinals: &[usize],
) -> Result<()> {
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

pub(in crate::runtime::query_service::service) fn validate_query_service_tenants(
    tenants: &[QueryServiceTenantConfig],
) -> Result<()> {
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
        let mut registry_config = DeviceRegistryConfig::new(device_ordinals, stream_pool_size);
        registry_config = registry_config.with_device_memory_manager_base(config.device_memory_manager_base);
        if let Some(device_capacity_bytes) = config.device_capacity_bytes {
            registry_config = registry_config.with_device_capacity_bytes(device_capacity_bytes);
        }
        let devices = DeviceRegistry::install(&NexusGpuInventory, &registry_config)?;
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
    pub(in crate::runtime::query_service) fn new_for_admission_tests(
        config: QueryServiceConfig,
        tenants: Vec<QueryServiceTenantConfig>,
        deployment_io_capability: DeploymentIoCapability,
    ) -> Result<Self> {
        Self::new_admission_only(config, tenants, deployment_io_capability)
    }

    pub(in crate::runtime::query_service::service) fn build(
        config: QueryServiceConfig,
        tenants: Vec<QueryServiceTenantConfig>,
        resources: Option<QueryServiceResources>,
    ) -> Result<Self> {
        let deployment =
            resources.as_ref().map(|resources| &resources.deployment_io_capability).cloned().unwrap_or_else(|| {
                DeploymentIoCapability::from_probe_inputs(
                    NvidiaFsProbeStatus::DriverMissing,
                    KvikioCompatModeStatus::Unset,
                    GdsDriverProbeStatus::ProbeUnimplemented,
                )
            });
        Self::build_with_deployment(config, tenants, resources, deployment)
    }

    pub(in crate::runtime::query_service::service) fn build_with_deployment(
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
}
