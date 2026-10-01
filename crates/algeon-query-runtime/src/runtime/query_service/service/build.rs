//! QueryService constructors and config preflight validation.

#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::{
    collections::HashMap,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicU64},
    },
};

use query_planning::error::{QueryError, QueryErrorCode, Result};
use tokio::sync::Notify;

use super::{
    super::{
        admission::AdmissionState,
        config::{QueryExecutionConfig, QueueBounds},
        device_waiters::{notify_shared_device_admission_available, register_shared_device_admission_waiters},
        resources::QueryServiceResources,
    },
    QueryService, QueryServiceInner,
};
#[cfg(test)]
use crate::runtime::backend_core::GpuBackendDeviceProfile;
#[cfg(test)]
use crate::runtime::gpu::GpuInventory;
use crate::runtime::{
    AlgeonGpuInventory, CacheFillByteBudget, DeviceLedger, GpuDeviceMemoryLimit, GpuMemoryOwnership,
    allocator_correlation::AllocatorEventCorrelationIndex,
    backend_core::{
        ConstructionResourceJournal, GpuBackendConstructionCheckpoint, GpuBackendConstructionProbe,
        GpuBackendControllerDraft, GpuBackendDeviceConfig,
    },
    backend_metrics::BackendMetricsStore,
    cache_fill::{CacheFillConcurrencyGate, default_lakehouse_cache_fill_budget_bytes},
    context::{DeviceRuntime, QueryServiceEpoch},
    gpu::{
        DeviceOwnerToken, normalize_device_ordinals,
        registry::{
            DeviceRegistry, DeviceRegistryCapacityPolicy, DeviceRegistryConfig, DeviceRegistryInstallProfile,
            DeviceRegistryMemoryPartition, ResolvedDeviceInstallPlans,
        },
    },
    gpu_execution_pool::{GpuExecutionPool, GpuExecutionQueryTracker},
    object_store::{SourceObjectStoreRegistry, default_object_store_control_runtime},
};
#[cfg(any(test, feature = "test-hooks"))]
use crate::runtime::{backend_core::GpuBackendCoreDraft, gpu::claim_live_backend};

impl QueryService {
    pub(crate) fn validate_controller_policy_for_backend(
        config: &QueryExecutionConfig,
        queue_bounds: &QueueBounds,
        device_ordinals: &[usize],
    ) -> Result<()> {
        validate_controller_policy_preflight(config, queue_bounds, device_ordinals)
    }

    /// Creates a service entry point over explicit devices.
    #[cfg(test)]
    pub(crate) fn new_with_device_ordinals(
        config: QueryExecutionConfig,
        queue_bounds: QueueBounds,
        device_ordinals: Vec<usize>,
    ) -> Result<Self> {
        let device_profiles = device_ordinals
            .into_iter()
            .map(|device_ordinal| {
                GpuBackendDeviceConfig::new(
                    GpuBackendDeviceProfile::new(device_ordinal),
                    GpuMemoryOwnership::WholeDeviceExclusive,
                )
            })
            .collect();
        Self::new_with_device_profiles_for_test(config, queue_bounds, device_profiles)
    }

    /// Test-only runtime construction over explicit immutable device profiles.
    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    pub fn new_with_device_profiles_for_test(
        config: QueryExecutionConfig,
        queue_bounds: QueueBounds,
        device_profiles: Vec<GpuBackendDeviceConfig>,
    ) -> Result<Self> {
        let controller = GpuBackendControllerDraft::new(config, queue_bounds);
        let mut draft = GpuBackendCoreDraft::new(device_profiles, controller.clone());
        draft.normalize_installation_bounds()?;
        draft.validate()?;
        let device_profiles = draft.devices().to_vec();
        let device_ordinals = device_profiles.iter().map(GpuBackendDeviceConfig::device_ordinal).collect::<Vec<_>>();
        let device_ordinals = normalize_device_ordinals(&device_ordinals)?;
        let owner_lease = claim_live_backend(&device_ordinals)?;
        let construction_journal = Arc::new(ConstructionResourceJournal::default());
        let backend_metrics = BackendMetricsStore::new(owner_lease.generation(), &device_ordinals)?;
        let probe = TestConstructionProbe;
        let token = owner_lease.token();
        Self::install_for_backend(
            device_ordinals,
            controller,
            &token,
            &device_profiles,
            &probe,
            &construction_journal,
            &backend_metrics,
        )
    }

    /// Installs the private runtime service under an already-acquired backend
    /// owner token. The caller retains the live-backend lease and must release
    /// it only after destroying the returned service and every derived GPU
    /// resource.
    pub(crate) fn install_for_backend(
        device_ordinals: Vec<usize>,
        controller: GpuBackendControllerDraft,
        owner: &DeviceOwnerToken<'_>,
        device_profiles: &[GpuBackendDeviceConfig],
        probe: &dyn GpuBackendConstructionProbe,
        construction_journal: &Arc<ConstructionResourceJournal>,
        backend_metrics: &BackendMetricsStore,
    ) -> Result<Self> {
        let device_ordinals = normalize_device_ordinals(&device_ordinals)?;
        let (config, queue_bounds) = controller.into_parts();
        validate_controller_policy_preflight(&config, &queue_bounds, &device_ordinals)?;
        let registry_config = device_registry_config(device_ordinals, device_profiles)?;
        let resolved_plans = DeviceRegistry::resolve_install_plans(&AlgeonGpuInventory, &registry_config, Some(probe))?;
        let resources = Self::install_runtime_resources(
            &config,
            owner,
            device_profiles,
            probe,
            resolved_plans,
            construction_journal,
            backend_metrics,
        )?;
        Self::build(config, queue_bounds, Some(resources))
    }

    fn install_runtime_resources(
        config: &QueryExecutionConfig,
        owner: &DeviceOwnerToken<'_>,
        device_profiles: &[GpuBackendDeviceConfig],
        probe: &dyn GpuBackendConstructionProbe,
        resolved_plans: ResolvedDeviceInstallPlans,
        construction_journal: &Arc<ConstructionResourceJournal>,
        backend_metrics: &BackendMetricsStore,
    ) -> Result<QueryServiceResources> {
        let devices = DeviceRegistry::install_resolved(
            resolved_plans,
            owner,
            Some(probe),
            Some(construction_journal),
            Some(backend_metrics),
            &AlgeonGpuInventory,
        )?;
        let gpu_execution_query_tracker = GpuExecutionQueryTracker::new();
        let gpu_execution_pools = devices
            .contexts()
            .map(|device| {
                let device_ordinal = device.device().ordinal();
                probe.checkpoint(GpuBackendConstructionCheckpoint::BeforeWorkerPool { device_ordinal })?;
                let worker_count = device_profiles
                    .iter()
                    .find(|profile| profile.device_ordinal() == device_ordinal)
                    .map(GpuBackendDeviceConfig::worker_count)
                    .ok_or_else(|| {
                        QueryError::new(
                            QueryErrorCode::Internal,
                            "query_engine.query_service.build",
                            "installed device has no backend construction profile",
                        )
                        .with_fact("device_ordinal", device_ordinal)
                        .with_fact("reason", "device_profile_missing")
                    })?;
                let profile = devices.resource_profile(device_ordinal).ok_or_else(|| {
                    QueryError::new(
                        QueryErrorCode::Internal,
                        "query_engine.query_service.build",
                        "installed device has no frozen resource profile",
                    )
                    .with_fact("device_ordinal", device_ordinal)
                    .with_fact("reason", "resource_profile_missing")
                })?;
                let pool = GpuExecutionPool::new_with_tracker_and_classes(
                    device_ordinal,
                    worker_count,
                    profile.max_active_attempts,
                    profile.aux_source_work_capacity,
                    gpu_execution_query_tracker.clone(),
                )?;
                construction_journal.record_execution_pool(pool.clone())?;
                probe.checkpoint(GpuBackendConstructionCheckpoint::AfterWorkerPool { device_ordinal })?;
                Ok((device_ordinal, pool))
            })
            .collect::<Result<HashMap<_, _>>>()?;
        // This allocation owner is constructed only after the registry has
        // created its authoritative per-device ledgers.  It intentionally
        // lives in `QueryServiceResources`, which the backend resource
        // generation owns through close and destruction.
        let resident_caches = QueryServiceResources::install_resident_caches(&devices)?;
        let lakehouse_cache_fill_budget = CacheFillByteBudget::new(
            config.lakehouse_cache_fill.budget_bytes.unwrap_or_else(default_lakehouse_cache_fill_budget_bytes),
        )?;
        let lakehouse_cache_fill_concurrency =
            CacheFillConcurrencyGate::new(config.lakehouse_cache_fill.max_concurrent_ops)?;
        let object_store_control_runtime = default_object_store_control_runtime()?;
        let source_object_stores =
            Arc::new(SourceObjectStoreRegistry::with_profiles(config.source_storage_profiles.clone())?);
        let device_runtimes = gpu_execution_pools
            .iter()
            .map(|(&device_ordinal, gpu_execution_pool)| {
                let device = devices.context_by_ordinal(device_ordinal)?;
                let device_runtime = DeviceRuntime {
                    gpu_execution_pool: gpu_execution_pool.clone(),
                    service_stream_pool: device.stream_pool().clone(),
                    resident_source_cache: resident_caches.source_for_device(device_ordinal),
                    #[cfg(feature = "cugraph")]
                    resident_graph_cache: resident_caches.graph_for_device(device_ordinal),
                    lakehouse_cache_fill_config: config.lakehouse_cache_fill.clone(),
                    lakehouse_cache_fill_budget: lakehouse_cache_fill_budget.clone(),
                    lakehouse_cache_fill_concurrency: lakehouse_cache_fill_concurrency.clone(),
                    object_store_control_runtime: Arc::clone(&object_store_control_runtime),
                    source_object_stores: Arc::clone(&source_object_stores),
                    source_scheduler_policy: config.source_scheduler_policy,
                    source_chunk_planning_mode: config.source_chunk_planning_mode,
                    source_chunk_planning_options: config.source_chunk_planning_options,
                    statistical_aggregate_execution_mode: config.statistical_aggregate_execution_mode,
                    aggregate_strategy_execution_options: config.aggregate_strategy_execution_options,
                    planning_policy: config.planning_policy,
                };
                Ok((device_ordinal, Arc::new(device_runtime)))
            })
            .collect::<Result<HashMap<_, _>>>()?;
        Ok(QueryServiceResources {
            device_runtimes,
            devices,
            gpu_execution_pools,
            resident_caches,
            gpu_execution_query_tracker,
        })
    }

    #[cfg(test)]
    pub(in crate::runtime::query_service) fn new_for_admission_tests(
        config: QueryExecutionConfig,
        queue_bounds: QueueBounds,
        ledger: DeviceLedger,
    ) -> Result<Self> {
        Self::build_with_fallback_device_ledger(config, queue_bounds, None, Some(ledger))
    }

    pub(in crate::runtime::query_service::service) fn build(
        config: QueryExecutionConfig,
        queue_bounds: QueueBounds,
        resources: Option<QueryServiceResources>,
    ) -> Result<Self> {
        Self::build_with_fallback_device_ledger(config, queue_bounds, resources, None)
    }

    fn build_with_fallback_device_ledger(
        config: QueryExecutionConfig,
        queue_bounds: QueueBounds,
        resources: Option<QueryServiceResources>,
        fallback_device_ledger: Option<DeviceLedger>,
    ) -> Result<Self> {
        config.validate()?;
        queue_bounds.validate()?;
        SourceObjectStoreRegistry::validate_profiles(&config.source_storage_profiles)?;

        let fallback_device_ledger = if resources.is_none() {
            Some(fallback_device_ledger.ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "query_engine.query_service.config.validate",
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
        // destructors may still retain GPU state. `ResourceGeneration` keeps
        // that lease and drops it explicitly last.
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
            allocator_event_correlation: Mutex::new(AllocatorEventCorrelationIndex::default()),
            #[cfg(test)]
            shutdown_waiters: AtomicUsize::new(0),
            #[cfg(test)]
            admission_transaction_probe: Mutex::new(None),
            #[cfg(test)]
            direct_placement_resolution_failure: AtomicBool::new(false),
            #[cfg(test)]
            direct_provisional_cleanup_pause: Mutex::new(None),
            #[cfg(test)]
            pending_preflight_resolver_calls: AtomicUsize::new(0),
            #[cfg(test)]
            pending_preflight_pause: Mutex::new(None),
            #[cfg(test)]
            pending_preflight_failure: AtomicBool::new(false),
            next_admission_request_id: AtomicU64::new(1),
            next_submission_seq: AtomicU64::new(1),
            next_admission_sequence: AtomicU64::new(1),
        });
        register_shared_device_admission_waiters(&inner)?;
        let service = Self { inner };
        // Grant releases wake this service's queue and any sibling services
        // sharing the device. Release observers run only after the ledger
        // state lock is released; the admission invariant forbids grant drops
        // while the admission mutex is held.
        let ledgers = service
            .inner
            .resources
            .as_ref()
            .map(|resources| {
                resources.devices.contexts().map(|context| Arc::clone(context.ledger())).collect::<Vec<_>>()
            })
            .unwrap_or_else(|| {
                vec![Arc::new(
                    service
                        .inner
                        .fallback_device_ledger
                        .as_ref()
                        .expect("admission-only service has a fallback device ledger")
                        .clone(),
                )]
            });
        for ledger in ledgers {
            let weak = Arc::downgrade(&service.inner);
            ledger.register_release_observer(Arc::new(move |transition| {
                notify_shared_device_admission_available(transition.snapshot.device_ordinal);
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
    device_profiles: &[GpuBackendDeviceConfig],
) -> Result<DeviceRegistryConfig> {
    let mut registry_config = DeviceRegistryConfig::new(device_ordinals, 1);
    for profile in device_profiles {
        let capacity_policy = match profile.memory_limit() {
            GpuDeviceMemoryLimit::Automatic { headroom_bytes, fallback_bytes, fraction } => {
                DeviceRegistryCapacityPolicy::Automatic { headroom_bytes, fallback_bytes, fraction }
            },
            GpuDeviceMemoryLimit::Explicit { limit_bytes } => DeviceRegistryCapacityPolicy::Explicit { limit_bytes },
        };
        let memory_partition = match profile.memory_ownership() {
            GpuMemoryOwnership::WholeDeviceExclusive => DeviceRegistryMemoryPartition::WholeDevice,
            GpuMemoryOwnership::StaticPartition { allocation_limit_bytes } => {
                DeviceRegistryMemoryPartition::StaticPartition {
                    allocation_limit_bytes: usize::try_from(allocation_limit_bytes).map_err(|_error| {
                        QueryError::new(
                            QueryErrorCode::InvalidPlan,
                            "query_engine.gpu_backend_core.build",
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
                max_active_attempts: profile.max_active_attempts(),
                capacity_policy,
                backend_reserve_bytes: profile.backend_reserve_bytes(),
                cache_cap_bytes: profile.cache_cap_bytes(),
                default_query_limit_bytes: profile.default_query_limit_bytes(),
                source_work_per_attempt: profile.source_work_per_attempt(),
                // Logical worker job-slot capacity follows the frozen
                // A_d * (1 + S_d) stream formula; physical worker threads
                // remain the pool's own lazily-grown count.
                worker_job_capacity: profile.stream_pool_size(),
                lane_count: profile.lane_count(),
                install_telemetry: profile.install_telemetry(),
                memory_partition,
            },
        );
    }
    Ok(registry_config)
}

#[cfg(any(test, feature = "test-hooks"))]
#[derive(Debug)]
struct TestConstructionProbe;

#[cfg(any(test, feature = "test-hooks"))]
impl GpuBackendConstructionProbe for TestConstructionProbe {
    fn checkpoint(&self, _checkpoint: GpuBackendConstructionCheckpoint) -> Result<()> {
        Ok(())
    }
}

fn validate_controller_policy_preflight(
    config: &QueryExecutionConfig,
    queue_bounds: &QueueBounds,
    device_ordinals: &[usize],
) -> Result<()> {
    config.validate()?;
    queue_bounds.validate()?;
    SourceObjectStoreRegistry::validate_profiles(&config.source_storage_profiles)?;
    normalize_device_ordinals(device_ordinals)?;
    Ok(())
}

#[cfg(test)]
mod construction_tests {
    use super::*;
    use crate::runtime::{DeviceCapabilities, DeviceFacts};

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
                    "query_engine.query_service.construction_test",
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
    fn pinned_small_device_rejects_oversized_query_limit_before_installation_phase() {
        let inventory = FrozenPlanInventory(vec![
            DeviceFacts {
                device: 0,
                model_name: None,
                uuid: None,
                physical_total_bytes: None,
                total_bytes: 100,
                free_bytes: 100,
                sm_count: Some(1),
                utilization_pct: None,
            },
            DeviceFacts {
                device: 1,
                model_name: None,
                uuid: None,
                physical_total_bytes: None,
                total_bytes: 10,
                free_bytes: 10,
                sm_count: Some(1),
                utilization_pct: None,
            },
        ]);
        let whole_device_profile = |limit_bytes, default_query_limit_bytes| DeviceRegistryInstallProfile {
            stream_pool_size: 2,
            device_capabilities: DeviceCapabilities::default(),
            max_active_attempts: 1,
            capacity_policy: DeviceRegistryCapacityPolicy::Explicit { limit_bytes },
            backend_reserve_bytes: 0,
            cache_cap_bytes: 0,
            default_query_limit_bytes,
            source_work_per_attempt: 1,
            worker_job_capacity: 2,
            lane_count: 1,
            install_telemetry: true,
            memory_partition: DeviceRegistryMemoryPartition::WholeDevice,
        };
        let registry_config = DeviceRegistryConfig::new(vec![0, 1], 1)
            .with_device_profile(0, whole_device_profile(100, None))
            .with_device_profile(1, whole_device_profile(10, Some(20)));
        let probe = FrozenPlanProbe::default();
        let error = DeviceRegistry::resolve_install_plans(&inventory, &registry_config, Some(&probe)).unwrap_err();

        assert_eq!(error.fact_value("reason"), Some("default_query_limit_exceeds_managed_capacity"));
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
        use crate::runtime::{DeviceCapabilities, DeviceCapability};

        let capabilities = DeviceCapabilities::only([DeviceCapability::new("test_operator")]);
        let device = GpuBackendDeviceConfig::new(
            GpuBackendDeviceProfile::new(0).with_device_capabilities(capabilities.clone()).with_telemetry(false),
            GpuMemoryOwnership::WholeDeviceExclusive,
        );

        let registry = device_registry_config(vec![0], &[device]).unwrap();

        assert_eq!(registry.profile(0).expect("backend profile").device_capabilities, capabilities);
        assert!(!registry.profile(0).expect("backend profile").install_telemetry);
    }
}
