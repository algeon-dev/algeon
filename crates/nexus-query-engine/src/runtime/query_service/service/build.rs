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
    pub(crate) fn validate_controller_policy_for_backend(
        config: &QueryServiceConfig,
        device_ordinals: &[usize],
        deployment_io_capability: &DeploymentIoCapability,
    ) -> Result<()> {
        validate_controller_policy_preflight(config, device_ordinals, deployment_io_capability)
    }

    /// Creates a service entry point and initializes service-owned runtime gates.
    #[cfg(test)]
    pub fn new(config: QueryServiceConfig, tenants: Vec<QueryServiceTenantConfig>) -> Result<Self> {
        let device_ordinals = vec![config.device_ordinal];
        Self::new_with_device_ordinals(config, device_ordinals, tenants)
    }

    /// Creates a service entry point over an explicit set of CUDA device ordinals.
    #[cfg(test)]
    pub fn new_with_device_ordinals(
        config: QueryServiceConfig,
        device_ordinals: Vec<usize>,
        tenants: Vec<QueryServiceTenantConfig>,
    ) -> Result<Self> {
        let deployment = DeploymentIoCapability::cached_current_host();
        Self::new_with_device_ordinals_and_deployment(config, device_ordinals, tenants, deployment)
    }

    /// Creates a service entry point with caller-supplied deployment diagnostics.
    #[cfg(test)]
    pub fn new_with_deployment(
        config: QueryServiceConfig,
        tenants: Vec<QueryServiceTenantConfig>,
        deployment_io_capability: DeploymentIoCapability,
    ) -> Result<Self> {
        let device_ordinals = vec![config.device_ordinal];
        Self::new_with_device_ordinals_and_deployment(config, device_ordinals, tenants, deployment_io_capability)
    }

    /// Creates a service entry point over explicit devices with caller-supplied deployment diagnostics.
    #[cfg(test)]
    pub fn new_with_device_ordinals_and_deployment(
        config: QueryServiceConfig,
        device_ordinals: Vec<usize>,
        tenants: Vec<QueryServiceTenantConfig>,
        deployment_io_capability: DeploymentIoCapability,
    ) -> Result<Self> {
        validate_runtime_service_preflight(&config, &device_ordinals, &tenants, &deployment_io_capability)?;
        let device_ordinals = normalize_device_ordinals(&device_ordinals)?;
        let owner_lease = claim_live_backend(&device_ordinals)?;
        let token = owner_lease.token();
        let mut resources = Self::install_runtime_resources(
            &config,
            device_ordinals,
            deployment_io_capability,
            &token,
            None,
            None,
            None,
            None,
        )?;
        drop(token);
        resources.owner_lease = Some(owner_lease);
        Self::build(config, tenants, Some(resources))
    }

    /// Installs the private runtime service under an already-acquired backend
    /// owner token. The caller retains the live-backend lease and must release
    /// it only after destroying the returned service and every derived GPU
    /// resource.
    pub(crate) fn install_for_backend(
        device_ordinals: Vec<usize>,
        controller: GpuBackendControllerDraft,
        deployment_io_capability: DeploymentIoCapability,
        owner: &DeviceOwnerToken<'_>,
        device_profiles: &[GpuBackendDeviceConfig],
        probe: &dyn GpuBackendConstructionProbe,
        construction_journal: &Arc<ConstructionResourceJournal>,
    ) -> Result<Self> {
        let device_ordinals = normalize_device_ordinals(&device_ordinals)?;
        let derived_global_max_active_queries = device_profiles.iter().try_fold(0usize, |sum, profile| {
            sum.checked_add(profile.max_active_queries()).ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.gpu_backend_core.build",
                    "derived global active-query limit overflowed",
                )
                .with_fact("field", "max_active_queries")
                .with_fact("reason", "derived_global_max_overflow")
            })
        })?;
        let (mut config, tenant_drafts) = controller.into_parts();
        config.max_concurrent_queries = derived_global_max_active_queries;
        config.max_queries_per_device = device_profiles.iter().map(GpuBackendDeviceConfig::max_active_queries).max();
        validate_controller_policy_preflight(&config, &device_ordinals, &deployment_io_capability)?;
        let registry_config = device_registry_config(&config, device_ordinals.clone(), Some(device_profiles))?;
        let resolved_plans = DeviceRegistry::resolve_install_plans(&NexusGpuInventory, &registry_config, Some(probe))?;
        let projection = match config.placement_policy {
            QueryPlacementPolicy::Pinned { device_ordinal } => {
                resolved_plans.plans().iter().find(|plan| plan.device.ordinal() == device_ordinal)
            },
            QueryPlacementPolicy::LeastLoaded | QueryPlacementPolicy::BestFit => resolved_plans
                .plans()
                .iter()
                .max_by_key(|plan| (plan.capacity_bytes, std::cmp::Reverse(plan.device.ordinal()))),
        }
        .ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.gpu_backend_core.build",
                "resolved backend has no device capacity projection",
            )
            .with_fact("reason", "device_capacity_projection_missing")
        })?;
        config.device_ordinal = match config.placement_policy {
            QueryPlacementPolicy::Pinned { device_ordinal } => device_ordinal,
            _ => projection.device.ordinal(),
        };
        config.device_capacity_bytes = Some(projection.capacity_bytes);
        config.stream_pool_size = None;
        let tenants = resolve_backend_tenants(&config, device_profiles.len(), tenant_drafts)?;
        validate_backend_controller_against_plans(&config, &tenants, &resolved_plans)?;
        let resources = Self::install_runtime_resources(
            &config,
            device_ordinals,
            deployment_io_capability,
            owner,
            Some(device_profiles),
            Some(probe),
            Some(resolved_plans),
            Some(construction_journal),
        )?;
        Self::build(config, tenants, Some(resources))
    }

    fn install_runtime_resources(
        config: &QueryServiceConfig,
        device_ordinals: Vec<usize>,
        deployment_io_capability: DeploymentIoCapability,
        owner: &DeviceOwnerToken<'_>,
        device_profiles: Option<&[GpuBackendDeviceConfig]>,
        probe: Option<&dyn GpuBackendConstructionProbe>,
        resolved_plans: Option<ResolvedDeviceInstallPlans>,
        construction_journal: Option<&Arc<ConstructionResourceJournal>>,
    ) -> Result<QueryServiceResources> {
        let devices = match resolved_plans {
            Some(resolved_plans) => {
                DeviceRegistry::install_resolved(resolved_plans, owner, probe, construction_journal)?
            },
            None => {
                let registry_config = device_registry_config(config, device_ordinals, device_profiles)?;
                DeviceRegistry::install(&NexusGpuInventory, &registry_config, owner, probe, construction_journal)?
            },
        };
        let gpu_execution_query_tracker = GpuExecutionQueryTracker::new();
        let gpu_execution_worker_count = gpu_execution_worker_count_per_device(&config);
        let gpu_execution_pools = devices
            .contexts()
            .map(|device| {
                let device_ordinal = device.device().ordinal();
                if let Some(probe) = probe {
                    probe.checkpoint(GpuBackendConstructionCheckpoint::BeforeWorkerPool { device_ordinal })?;
                }
                let worker_count = match device_profiles {
                    Some(profiles) => profiles
                        .iter()
                        .find(|profile| profile.device_ordinal() == device_ordinal)
                        .map(GpuBackendDeviceConfig::worker_count)
                        .ok_or_else(|| {
                            QueryError::new(
                                QueryErrorCode::Internal,
                                "nexus_query_engine.query_service.build",
                                "installed device has no backend construction profile",
                            )
                            .with_fact("device_ordinal", device_ordinal)
                            .with_fact("reason", "device_profile_missing")
                        })?,
                    None => gpu_execution_worker_count,
                };
                let pool = GpuExecutionPool::new_with_tracker(
                    device_ordinal,
                    worker_count,
                    gpu_execution_query_tracker.clone(),
                )?;
                if let Some(journal) = construction_journal {
                    journal.record_execution_pool(pool.clone())?;
                }
                if let Some(probe) = probe {
                    probe.checkpoint(GpuBackendConstructionCheckpoint::AfterWorkerPool { device_ordinal })?;
                }
                Ok((device_ordinal, pool))
            })
            .collect::<Result<HashMap<_, _>>>()?;
        let local_parquet_source_caches = local_parquet_source_caches_for_devices(
            &devices,
            config.local_parquet_source_cache_max_retained_bytes,
            probe,
            construction_journal,
        )?;
        #[cfg(feature = "cugraph")]
        let resident_graph_caches = resident_graph_caches_for_devices(
            &devices,
            config.resident_graph_cache_max_retained_bytes,
            resident_graph_cache_max_copies_per_key(&config),
            probe,
            construction_journal,
        )?;
        Ok(QueryServiceResources {
            devices,
            gpu_execution_pools,
            gpu_execution_query_tracker,
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
            owner_lease: None,
        })
    }

    #[cfg(test)]
    pub(in crate::runtime::query_service) fn new_for_admission_tests(
        config: QueryServiceConfig,
        tenants: Vec<QueryServiceTenantConfig>,
        deployment_io_capability: DeploymentIoCapability,
    ) -> Result<Self> {
        Self::build_with_deployment(config, tenants, None, deployment_io_capability)
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
        let resources = resources;
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
        let fallback_memory_governor = MemoryGovernor::new(config.device_ordinal, device_capacity_bytes)?;

        // Epoch idleness is only one owned drain phase. It cannot release the
        // process-wide backend lease while queues, caches, worker pools, or
        // destructors may still retain GPU state. Legacy direct construction
        // keeps its lease in `QueryServiceResources`; core-owned construction
        // keeps it in `ResourceGeneration` and drops it explicitly last.
        let service_epoch = QueryServiceEpoch::new(None);
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

fn device_registry_config(
    config: &QueryServiceConfig,
    device_ordinals: Vec<usize>,
    device_profiles: Option<&[GpuBackendDeviceConfig]>,
) -> Result<DeviceRegistryConfig> {
    // Backend profiles already contain explicit per-device installation
    // bounds. Legacy internal construction retains its historical derived
    // uniform stream-pool size.
    let stream_pool_size = match device_profiles {
        Some(_) => 1,
        None => {
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
            stream_pool_size
        },
    };
    let mut registry_config = DeviceRegistryConfig::new(device_ordinals, stream_pool_size)
        .with_device_memory_manager_base(config.device_memory_manager_base);
    if let Some(device_capacity_bytes) = config.device_capacity_bytes {
        registry_config = registry_config.with_device_capacity_bytes(device_capacity_bytes);
    }
    if let Some(device_profiles) = device_profiles {
        for profile in device_profiles {
            let capacity_policy = match profile.memory_limit() {
                GpuDeviceMemoryLimit::Automatic { headroom_bytes, fallback_bytes, fraction } => {
                    DeviceRegistryCapacityPolicy::Automatic {
                        headroom_bytes,
                        fallback_bytes,
                        fraction,
                        maximum_bytes: None,
                    }
                },
                GpuDeviceMemoryLimit::Explicit { limit_bytes } => {
                    DeviceRegistryCapacityPolicy::Explicit { limit_bytes }
                },
            };
            let memory_partition = match profile.memory_ownership() {
                GpuMemoryOwnership::WholeDeviceExclusive => DeviceRegistryMemoryPartition::WholeDevice,
                GpuMemoryOwnership::StaticPartition { allocation_limit_bytes } => {
                    DeviceRegistryMemoryPartition::StaticPartition {
                        allocation_limit_bytes: usize::try_from(allocation_limit_bytes).map_err(|_error| {
                            QueryError::new(
                                QueryErrorCode::InvalidPlan,
                                "nexus_query_engine.gpu_backend_core.build",
                                "static partition limit cannot be represented by the native allocator",
                            )
                            .with_fact("field", "memory_ownership.allocation_limit_bytes")
                            .with_fact("device_ordinal", profile.device_ordinal())
                            .with_fact("allocation_limit_bytes", allocation_limit_bytes)
                            .with_fact("reason", "allocation_limit_out_of_range")
                        })?,
                    }
                },
            };
            registry_config = registry_config.with_device_profile(
                profile.device_ordinal(),
                DeviceRegistryInstallProfile {
                    stream_pool_size: profile.stream_pool_size(),
                    max_active_queries: Some(profile.max_active_queries()),
                    capacity_policy,
                    backend_reserve_bytes: profile.backend_reserve_bytes(),
                    device_memory_manager_base: config.device_memory_manager_base,
                    memory_partition,
                },
            );
        }
    }
    Ok(registry_config)
}

fn validate_runtime_service_preflight(
    config: &QueryServiceConfig,
    device_ordinals: &[usize],
    tenants: &[QueryServiceTenantConfig],
    deployment_io_capability: &DeploymentIoCapability,
) -> Result<()> {
    // Resource-independent admission must precede process-global allocator
    // installation so those rejections cannot mutate CUDA/RMM state.
    config.validate(deployment_io_capability)?;
    SourceObjectStoreRegistry::validate_profiles(&config.source_storage_profiles)?;
    validate_query_service_tenants(tenants)?;
    let normalized = normalize_device_ordinals(device_ordinals)?;
    validate_placement_policy_for_device_ordinals(config, &normalized)
}

fn validate_controller_policy_preflight(
    config: &QueryServiceConfig,
    device_ordinals: &[usize],
    deployment_io_capability: &DeploymentIoCapability,
) -> Result<()> {
    config.validate(deployment_io_capability)?;
    SourceObjectStoreRegistry::validate_profiles(&config.source_storage_profiles)?;
    let normalized = normalize_device_ordinals(device_ordinals)?;
    validate_placement_policy_for_device_ordinals(config, &normalized)
}

fn resolve_backend_tenants(
    config: &QueryServiceConfig,
    device_count: usize,
    drafts: Vec<GpuBackendTenantDraft>,
) -> Result<Vec<QueryServiceTenantConfig>> {
    let capacity = config.device_capacity_bytes.ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::Internal,
            "nexus_query_engine.gpu_backend_core.build",
            "legacy controller capacity projection was not resolved",
        )
        .with_fact("reason", "controller_capacity_projection_missing")
    })?;
    drafts
        .into_iter()
        .map(|draft| {
            let (tenant_id, minimum_share, preferred_share) = draft.into_parts();
            let minimum = resolve_backend_memory_share("tenant.minimum", minimum_share, capacity, false)?;
            let preferred = match preferred_share {
                GpuBackendMemoryShare::PolicyDefault => {
                    default_backend_preferred_grant(config, device_count, capacity, minimum)
                },
                share => resolve_backend_memory_share("tenant.preferred", share, capacity, true),
            }?;
            if preferred < minimum {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.gpu_backend_core.build",
                    "preferred tenant grant must be at least its minimum",
                )
                .with_fact("field", "tenant.preferred")
                .with_fact("tenant_id", tenant_id.as_str())
                .with_fact("reason", "preferred_below_minimum"));
            }
            Ok(QueryServiceTenantConfig::with_grant_range(tenant_id, minimum, preferred))
        })
        .collect()
}

fn validate_backend_controller_against_plans(
    config: &QueryServiceConfig,
    tenants: &[QueryServiceTenantConfig],
    plans: &ResolvedDeviceInstallPlans,
) -> Result<()> {
    validate_query_service_tenants(tenants)?;
    let eligible = plans
        .plans()
        .iter()
        .filter(|plan| placement_policy_admits_ordinal(config.placement_policy, plan.device.ordinal()));
    let capacities = eligible
        .map(|plan| (plan.capacity_bytes, plan.profile.max_active_queries.or(config.max_queries_per_device)))
        .collect::<Vec<_>>();
    let max_admissible_device_capacity_bytes = capacities.iter().map(|(capacity, _)| *capacity).max().unwrap_or(0);
    let total_admissible_capacity_bytes =
        capacities.iter().map(|(capacity, _)| *capacity).fold(0u64, u64::saturating_add);
    let max_tenant_min_bytes = tenants.iter().map(|tenant| tenant.min_device_budget_bytes).max().unwrap_or(0);
    for tenant in tenants {
        if tenant.min_device_budget_bytes > max_admissible_device_capacity_bytes {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.query_service.tenant.validate",
                "tenant minimum device budget exceeds memory governor capacity",
            )
            .with_fact("tenant_id", tenant.tenant_id.as_str())
            .with_fact("field", "min_device_budget_bytes")
            .with_fact("reason", "min_device_budget_above_capacity")
            .with_fact("device_capacity_bytes", max_admissible_device_capacity_bytes));
        }
    }
    if config.slot_guarantee == SlotGuarantee::GuaranteedMinimum {
        let guaranteed_min_slots = capacities
            .into_iter()
            .map(|(capacity, active_limit)| {
                let memory_slots = if max_tenant_min_bytes == 0 {
                    usize::MAX
                } else {
                    usize::try_from(capacity / max_tenant_min_bytes).unwrap_or(usize::MAX)
                };
                active_limit.map_or(memory_slots, |limit| memory_slots.min(limit))
            })
            .fold(0usize, usize::saturating_add);
        if guaranteed_min_slots < config.max_concurrent_queries {
            let required_capacity_bytes =
                max_tenant_min_bytes.saturating_mul(u64::try_from(config.max_concurrent_queries).unwrap_or(u64::MAX));
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.query_service.tenant.validate",
                "tenant minimum device budgets exceed guaranteed slot capacity",
            )
            .with_fact("field", "min_device_budget_bytes")
            .with_fact("reason", "tenant_minima_exceed_guaranteed_slot_capacity")
            .with_fact("max_tenant_min_device_budget_bytes", max_tenant_min_bytes)
            .with_fact("max_concurrent_queries", config.max_concurrent_queries)
            .with_fact("required_capacity_bytes", required_capacity_bytes)
            .with_fact("device_capacity_bytes", total_admissible_capacity_bytes)
            .with_fact("guaranteed_min_slot_capacity", guaranteed_min_slots));
        }
    }
    Ok(())
}

fn resolve_backend_memory_share(
    field: &'static str,
    share: GpuBackendMemoryShare,
    capacity: u64,
    allow_default: bool,
) -> Result<u64> {
    let bytes = match share {
        GpuBackendMemoryShare::Bytes(bytes) => bytes,
        GpuBackendMemoryShare::Fraction(fraction) => fraction.capacity_bytes(capacity),
        GpuBackendMemoryShare::PolicyDefault if allow_default => capacity,
        GpuBackendMemoryShare::PolicyDefault => {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu_backend_core.build",
                "minimum tenant grant must be concrete",
            )
            .with_fact("field", field)
            .with_fact("reason", "policy_default"));
        },
    };
    if bytes == 0 {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.gpu_backend_core.build",
            "resolved tenant grant must be greater than zero",
        )
        .with_fact("field", field)
        .with_fact("reason", "zero"));
    }
    Ok(bytes)
}

fn default_backend_preferred_grant(
    config: &QueryServiceConfig,
    device_count: usize,
    capacity: u64,
    minimum: u64,
) -> Result<u64> {
    if config.slot_guarantee == SlotGuarantee::Optimistic
        && config.native_pipeline_execution_mode == NativePipelineExecutionMode::StrictBounded
    {
        let eligible_devices = match config.placement_policy {
            QueryPlacementPolicy::Pinned { .. } => 1,
            _ => device_count.max(1),
        };
        let slots_per_device = config.max_queries_per_device.map_or_else(
            || config.max_concurrent_queries.div_ceil(eligible_devices).max(1),
            |cap| config.max_concurrent_queries.div_ceil(eligible_devices).max(1).min(cap),
        );
        let reserved = minimum.saturating_mul(u64::try_from(slots_per_device.saturating_sub(1)).unwrap_or(u64::MAX));
        return Ok(capacity.saturating_sub(reserved).max(minimum));
    }
    Ok(capacity)
}

#[cfg(test)]
mod phase0_tests {
    use super::*;
    use crate::runtime::DeviceFacts;

    #[derive(Debug)]
    struct FrozenPlanInventory(Vec<DeviceFacts>);

    impl GpuInventory for FrozenPlanInventory {
        fn device_count(&self) -> Result<usize> {
            Ok(self.0.len())
        }

        fn facts(&self, device_ordinal: usize) -> Result<DeviceFacts> {
            self.0.get(device_ordinal).cloned().ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.query_service.phase0_test",
                    "missing fake device facts",
                )
            })
        }
    }

    #[derive(Debug, Default)]
    struct FrozenPlanProbe(Mutex<Vec<GpuBackendConstructionCheckpoint>>);

    impl GpuBackendConstructionProbe for FrozenPlanProbe {
        fn checkpoint(&self, checkpoint: GpuBackendConstructionCheckpoint) -> Result<()> {
            self.0.lock().unwrap().push(checkpoint);
            Ok(())
        }
    }

    #[test]
    fn pinned_small_device_rejects_tenant_before_installation_phase() {
        let inventory = FrozenPlanInventory(vec![
            DeviceFacts { device: 0, total_bytes: 100, free_bytes: 100, sm_count: Some(1), utilization_pct: None },
            DeviceFacts { device: 1, total_bytes: 10, free_bytes: 10, sm_count: Some(1), utilization_pct: None },
        ]);
        let whole_device_profile = |limit_bytes| DeviceRegistryInstallProfile {
            stream_pool_size: 1,
            max_active_queries: Some(1),
            capacity_policy: DeviceRegistryCapacityPolicy::Explicit { limit_bytes },
            backend_reserve_bytes: 0,
            device_memory_manager_base: cudf::DeviceMemoryManagerBase::default(),
            memory_partition: DeviceRegistryMemoryPartition::WholeDevice,
        };
        let registry_config = DeviceRegistryConfig::new(vec![0, 1], 1)
            .with_device_profile(0, whole_device_profile(100))
            .with_device_profile(1, whole_device_profile(10));
        let probe = FrozenPlanProbe::default();
        let plans = DeviceRegistry::resolve_install_plans(&inventory, &registry_config, Some(&probe)).unwrap();
        let config = QueryServiceConfig {
            placement_policy: QueryPlacementPolicy::Pinned { device_ordinal: 1 },
            device_ordinal: 1,
            device_capacity_bytes: Some(10),
            max_concurrent_queries: 1,
            max_queries_per_device: Some(1),
            ..QueryServiceConfig::default()
        };
        let tenants = vec![QueryServiceTenantConfig::new("tenant", 20)];

        let error = validate_backend_controller_against_plans(&config, &tenants, &plans).unwrap_err();

        assert_eq!(error.fact_value("reason"), Some("min_device_budget_above_capacity"));
        assert!(
            probe
                .0
                .lock()
                .unwrap()
                .iter()
                .all(|checkpoint| !matches!(checkpoint, GpuBackendConstructionCheckpoint::BeforeMemoryManager { .. }))
        );
    }
}
