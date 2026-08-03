//! QueryService constructors and config preflight validation.

use super::*;

impl QueryService {
    pub(crate) fn validate_controller_policy_for_backend(
        config: &QueryExecutionConfig,
        queue_bounds: &QueueBounds,
        device_ordinals: &[usize],
        deployment_io_capability: &DeploymentIoCapability,
    ) -> Result<()> {
        validate_controller_policy_preflight(config, queue_bounds, device_ordinals, deployment_io_capability)
    }

    /// Creates a service entry point and initializes service-owned runtime gates.
    #[cfg(test)]
    pub fn new(config: QueryExecutionConfig, queue_bounds: QueueBounds) -> Result<Self> {
        let device_ordinals = vec![0];
        Self::new_with_device_ordinals(config, queue_bounds, device_ordinals)
    }

    /// Creates a service entry point over an explicit set of CUDA device ordinals.
    #[cfg(test)]
    pub fn new_with_device_ordinals(
        config: QueryExecutionConfig,
        queue_bounds: QueueBounds,
        device_ordinals: Vec<usize>,
    ) -> Result<Self> {
        let deployment = DeploymentIoCapability::cached_current_host();
        Self::new_with_device_ordinals_and_deployment(config, queue_bounds, device_ordinals, deployment)
    }

    /// Creates a service entry point with caller-supplied deployment diagnostics.
    #[cfg(test)]
    pub fn new_with_deployment(
        config: QueryExecutionConfig,
        queue_bounds: QueueBounds,
        deployment_io_capability: DeploymentIoCapability,
    ) -> Result<Self> {
        let device_ordinals = vec![0];
        Self::new_with_device_ordinals_and_deployment(config, queue_bounds, device_ordinals, deployment_io_capability)
    }

    /// Creates a service entry point over explicit devices with caller-supplied deployment diagnostics.
    #[cfg(test)]
    pub fn new_with_device_ordinals_and_deployment(
        config: QueryExecutionConfig,
        queue_bounds: QueueBounds,
        device_ordinals: Vec<usize>,
        deployment_io_capability: DeploymentIoCapability,
    ) -> Result<Self> {
        validate_runtime_service_preflight(&config, &queue_bounds, &device_ordinals, &deployment_io_capability)?;
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
        Self::build(config, queue_bounds, Some(resources))
    }

    /// Test-only runtime construction over explicit immutable device profiles.
    #[cfg(test)]
    pub(in crate::runtime::query_service) fn new_with_device_profiles_and_deployment_for_test(
        config: QueryExecutionConfig,
        queue_bounds: QueueBounds,
        device_profiles: Vec<GpuBackendDeviceConfig>,
        deployment_io_capability: DeploymentIoCapability,
    ) -> Result<Self> {
        let device_ordinals = device_profiles.iter().map(GpuBackendDeviceConfig::device_ordinal).collect::<Vec<_>>();
        validate_runtime_service_preflight(&config, &queue_bounds, &device_ordinals, &deployment_io_capability)?;
        let device_ordinals = normalize_device_ordinals(&device_ordinals)?;
        let owner_lease = claim_live_backend(&device_ordinals)?;
        let token = owner_lease.token();
        let mut resources = Self::install_runtime_resources(
            &config,
            device_ordinals,
            deployment_io_capability,
            &token,
            Some(&device_profiles),
            None,
            None,
            None,
        )?;
        drop(token);
        resources.owner_lease = Some(owner_lease);
        Self::build(config, queue_bounds, Some(resources))
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
        let (config, queue_bounds) = controller.into_parts();
        validate_controller_policy_preflight(&config, &queue_bounds, &device_ordinals, &deployment_io_capability)?;
        let registry_config = device_registry_config(device_ordinals.clone(), Some(device_profiles))?;
        let resolved_plans = DeviceRegistry::resolve_install_plans(&NexusGpuInventory, &registry_config, Some(probe))?;
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
        Self::build(config, queue_bounds, Some(resources))
    }

    fn install_runtime_resources(
        config: &QueryExecutionConfig,
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
                let registry_config = device_registry_config(device_ordinals, device_profiles)?;
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
                let (root_job_capacity, aux_job_capacity) = match devices.resource_profile(device_ordinal) {
                    Some(profile) => (profile.max_active_attempts, profile.aux_source_work_capacity),
                    None => (worker_count, worker_count),
                };
                let pool = GpuExecutionPool::new_with_tracker_and_classes(
                    device_ordinal,
                    worker_count,
                    root_job_capacity,
                    aux_job_capacity,
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
            owner_lease: None,
        })
    }

    #[cfg(test)]
    pub(in crate::runtime::query_service) fn new_for_admission_tests(
        config: QueryExecutionConfig,
        queue_bounds: QueueBounds,
        deployment_io_capability: DeploymentIoCapability,
        ledger: DeviceLedger,
    ) -> Result<Self> {
        Self::new_for_admission_tests_with_ledger(config, queue_bounds, deployment_io_capability, ledger)
    }

    /// Builds a CUDA-free admission service with an immutable synthetic device
    /// profile. This is test-only so capability placement can be validated
    /// without constructing a CUDA runtime.
    #[cfg(test)]
    pub(in crate::runtime::query_service) fn new_for_admission_tests_with_device_capabilities(
        config: QueryExecutionConfig,
        queue_bounds: QueueBounds,
        deployment_io_capability: DeploymentIoCapability,
        device_capabilities: crate::runtime::DeviceCapabilities,
        ledger: DeviceLedger,
    ) -> Result<Self> {
        let _ = device_capabilities;
        Self::new_for_admission_tests_with_ledger(config, queue_bounds, deployment_io_capability, ledger)
    }

    /// Builds a CUDA-free admission service over an explicit immutable ledger
    /// fixture. The fixture carries capacity, floor, concurrency, source-work,
    /// and capability facts. Execution and queue policy stay independent.
    #[cfg(test)]
    pub(in crate::runtime::query_service) fn new_for_admission_tests_with_ledger(
        config: QueryExecutionConfig,
        queue_bounds: QueueBounds,
        deployment_io_capability: DeploymentIoCapability,
        ledger: DeviceLedger,
    ) -> Result<Self> {
        Self::build_with_deployment_and_fallback_device_capabilities(
            config,
            queue_bounds,
            None,
            deployment_io_capability,
            None,
            Some(ledger),
        )
    }

    pub(in crate::runtime::query_service::service) fn build(
        config: QueryExecutionConfig,
        queue_bounds: QueueBounds,
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
        Self::build_with_deployment(config, queue_bounds, resources, deployment)
    }

    pub(in crate::runtime::query_service::service) fn build_with_deployment(
        config: QueryExecutionConfig,
        queue_bounds: QueueBounds,
        resources: Option<QueryServiceResources>,
        deployment: DeploymentIoCapability,
    ) -> Result<Self> {
        Self::build_with_deployment_and_fallback_device_capabilities(
            config,
            queue_bounds,
            resources,
            deployment,
            None,
            None,
        )
    }

    fn build_with_deployment_and_fallback_device_capabilities(
        config: QueryExecutionConfig,
        queue_bounds: QueueBounds,
        resources: Option<QueryServiceResources>,
        deployment: DeploymentIoCapability,
        fallback_device_capabilities: Option<crate::runtime::DeviceCapabilities>,
        fallback_device_ledger: Option<DeviceLedger>,
    ) -> Result<Self> {
        let resources = resources;
        config.validate(&deployment)?;
        queue_bounds.validate()?;
        SourceObjectStoreRegistry::validate_profiles(&config.source_storage_profiles)?;

        let _ = fallback_device_capabilities;
        let fallback_device_ledger = if resources.is_none() {
            Some(fallback_device_ledger.ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.query_service.config.validate",
                    "admission-only query service requires an explicit device ledger fixture",
                )
                .with_fact("field", "device_ledger")
                .with_fact("reason", "required_for_admission_only_service")
            })?)
        } else {
            None
        };

        // Epoch idleness is only one owned drain phase. It cannot release the
        // process-wide backend lease while queues, caches, worker pools, or
        // destructors may still retain GPU state. Legacy direct construction
        // keeps its lease in `QueryServiceResources`; core-owned construction
        // keeps it in `ResourceGeneration` and drops it explicitly last.
        let service_epoch = QueryServiceEpoch::new(None);
        let inner = Arc::new(QueryServiceInner {
            config,
            queue_bounds,
            fallback_device_ledger,
            resources,
            service_epoch,
            admission_open: AtomicBool::new(true),
            fleet_admission: Mutex::new(()),
            admission: Mutex::new(AdmissionState::default()),
            admission_available: Condvar::new(),
            admission_notify: Notify::new(),
            #[cfg(test)]
            shutdown_waiters: AtomicUsize::new(0),
            admission_transaction_probe: Mutex::new(None),
            #[cfg(test)]
            unknown_isolation_closure_pause: Mutex::new(None),
            #[cfg(test)]
            protection_retarget_failure_after_release: AtomicBool::new(false),
            next_admission_request_id: AtomicU64::new(1),
            next_submission_seq: AtomicU64::new(1),
            next_admission_sequence: AtomicU64::new(1),
        });
        register_shared_device_admission_waiters(&inner)?;
        let service = Self { inner };
        // Grant releases wake this service's queue and any sibling services
        // sharing the device. The hook fires only after the ledger state lock
        // is released; the admission invariant forbids grant drops while the
        // admission mutex is held.
        let ledgers = service
            .inner
            .resources
            .as_ref()
            .map(|resources| {
                resources.devices.contexts().map(|context| Arc::clone(context.ledger())).collect::<Vec<_>>()
            })
            .unwrap_or_else(|| {
                service
                    .inner
                    .fallback_device_ledger
                    .as_ref()
                    .map(|ledger| vec![Arc::new(ledger.clone())])
                    .unwrap_or_default()
            });
        for ledger in ledgers {
            let weak = Arc::downgrade(&service.inner);
            ledger.set_release_hook(Arc::new(move |device_ordinal| {
                notify_shared_device_admission_available(device_ordinal);
                if let Some(inner) = weak.upgrade() {
                    inner.notify_admission_available();
                }
            }))?;
        }
        Ok(service)
    }
}

fn device_registry_config(
    device_ordinals: Vec<usize>,
    device_profiles: Option<&[GpuBackendDeviceConfig]>,
) -> Result<DeviceRegistryConfig> {
    // Backend profiles already contain explicit per-device installation
    // bounds. The test-only direct constructor uses the same concrete
    // per-device source-work default instead of borrowing an execution
    // setting as controller capacity.
    let stream_pool_size = match device_profiles {
        Some(_) => 1,
        None => {
            let default_stream_pool_size = default_stream_pool_size_for_ordinals(&device_ordinals)?;
            let minimum_stream_pool_size = minimum_stream_pool_size(1, 1);
            default_stream_pool_size.max(minimum_stream_pool_size)
        },
    };
    let mut registry_config = DeviceRegistryConfig::new(device_ordinals.clone(), stream_pool_size);
    if device_profiles.is_none() {
        let legacy_capacity_policy = DeviceRegistryCapacityPolicy::Automatic {
            headroom_bytes: registry_config.capacity_headroom_bytes,
            fallback_bytes: registry_config.fallback_capacity_bytes,
            fraction: None,
            maximum_bytes: registry_config.device_capacity_bytes,
        };
        for ordinal in &device_ordinals {
            registry_config = registry_config.with_device_profile(
                *ordinal,
                DeviceRegistryInstallProfile {
                    stream_pool_size,
                    device_capabilities: crate::runtime::DeviceCapabilities::default(),
                    max_active_queries: Some(1),
                    capacity_policy: legacy_capacity_policy,
                    backend_reserve_bytes: 0,
                    attempt_floor_bytes: None,
                    source_work_per_attempt: 1,
                    worker_job_capacity: stream_pool_size,
                    lane_count: stream_pool_size,
                    memory_partition: DeviceRegistryMemoryPartition::WholeDevice,
                },
            );
        }
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
                    device_capabilities: profile.device_capabilities().clone(),
                    max_active_queries: Some(profile.max_active_queries()),
                    capacity_policy,
                    backend_reserve_bytes: profile.backend_reserve_bytes(),
                    attempt_floor_bytes: profile.attempt_floor_bytes(),
                    source_work_per_attempt: profile.source_work_per_attempt(),
                    // Logical worker job-slot capacity follows the frozen
                    // A_d * (1 + S_d) stream formula; physical worker threads
                    // remain the pool's own lazily-grown count.
                    worker_job_capacity: profile.stream_pool_size(),
                    lane_count: profile.lane_count(),
                    memory_partition,
                },
            );
        }
    }
    Ok(registry_config)
}

fn validate_runtime_service_preflight(
    config: &QueryExecutionConfig,
    queue_bounds: &QueueBounds,
    device_ordinals: &[usize],
    deployment_io_capability: &DeploymentIoCapability,
) -> Result<()> {
    // Resource-independent admission must precede process-global allocator
    // installation so those rejections cannot mutate CUDA/RMM state.
    config.validate(deployment_io_capability)?;
    queue_bounds.validate()?;
    SourceObjectStoreRegistry::validate_profiles(&config.source_storage_profiles)?;
    normalize_device_ordinals(device_ordinals)?;
    Ok(())
}

fn validate_controller_policy_preflight(
    config: &QueryExecutionConfig,
    queue_bounds: &QueueBounds,
    device_ordinals: &[usize],
    deployment_io_capability: &DeploymentIoCapability,
) -> Result<()> {
    config.validate(deployment_io_capability)?;
    queue_bounds.validate()?;
    SourceObjectStoreRegistry::validate_profiles(&config.source_storage_profiles)?;
    normalize_device_ordinals(device_ordinals)?;
    Ok(())
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
    fn pinned_small_device_rejects_oversized_floor_before_installation_phase() {
        let inventory = FrozenPlanInventory(vec![
            DeviceFacts { device: 0, total_bytes: 100, free_bytes: 100, sm_count: Some(1), utilization_pct: None },
            DeviceFacts { device: 1, total_bytes: 10, free_bytes: 10, sm_count: Some(1), utilization_pct: None },
        ]);
        let whole_device_profile = |limit_bytes, floor| DeviceRegistryInstallProfile {
            stream_pool_size: 2,
            device_capabilities: crate::runtime::DeviceCapabilities::default(),
            max_active_queries: Some(1),
            capacity_policy: DeviceRegistryCapacityPolicy::Explicit { limit_bytes },
            backend_reserve_bytes: 0,
            attempt_floor_bytes: floor,
            source_work_per_attempt: 1,
            worker_job_capacity: 2,
            lane_count: 1,
            memory_partition: DeviceRegistryMemoryPartition::WholeDevice,
        };
        let registry_config = DeviceRegistryConfig::new(vec![0, 1], 1)
            .with_device_profile(0, whole_device_profile(100, None))
            .with_device_profile(1, whole_device_profile(10, Some(20)));
        let probe = FrozenPlanProbe::default();
        let error = DeviceRegistry::resolve_install_plans(&inventory, &registry_config, Some(&probe)).unwrap_err();

        assert_eq!(error.fact_value("reason"), Some("attempt_floor_exceeds_managed_capacity"));
        assert!(
            probe
                .0
                .lock()
                .unwrap()
                .iter()
                .all(|checkpoint| !matches!(checkpoint, GpuBackendConstructionCheckpoint::BeforeMemoryManager { .. }))
        );
    }

    #[test]
    fn backend_device_capabilities_are_preserved_in_the_registry_install_profile() {
        use crate::runtime::{DeviceCapabilities, DeviceCapability, ReservationKind};

        let capabilities = DeviceCapabilities::only([DeviceCapability::new(
            ReservationKind::OperatorOutput,
            "test_operator_contract",
        )]);
        let device = GpuBackendDeviceConfig::new(0, GpuMemoryOwnership::WholeDeviceExclusive)
            .with_device_capabilities(capabilities.clone());

        let registry = device_registry_config(vec![0], Some(&[device])).unwrap();

        assert_eq!(registry.profile(0).device_capabilities, capabilities);
    }
}
