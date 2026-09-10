use crate::runtime::gpu::DeviceLedger;

#[derive(Clone, Debug)]
struct AdmissionTestController {
    service: QueryService,
    synthetic_ticket: Arc<std::sync::Mutex<Option<AdmissionTicket>>>,
}

impl AdmissionTestController {
    fn new(config: QueryExecutionConfig, queue_bounds: QueueBounds) -> Result<Self> {
        Self::new_with_ledger(config, queue_bounds, test_ledger(64 * 1024 * 1024, 1, Default::default()))
    }

    fn new_with_ledger(config: QueryExecutionConfig, queue_bounds: QueueBounds, ledger: DeviceLedger) -> Result<Self> {
        QueryService::new_for_admission_tests(config, queue_bounds, ledger)
            .map(|service| Self { service, synthetic_ticket: Arc::new(std::sync::Mutex::new(None)) })
    }

    fn new_with_device_capabilities(
        config: QueryExecutionConfig,
        queue_bounds: QueueBounds,
        device_capabilities: crate::runtime::DeviceCapabilities,
    ) -> Result<Self> {
        QueryService::new_for_admission_tests(
            config,
            queue_bounds,
            test_ledger(64 * 1024 * 1024, 1, device_capabilities),
        )
        .map(|service| Self { service, synthetic_ticket: Arc::new(std::sync::Mutex::new(None)) })
    }

    fn admit_query(&self) -> Result<AdmissionTestHandle> {
        self.service.admit_query().map(AdmissionTestHandle::new)
    }

    async fn admit_async(&self) -> Result<AdmissionTestHandle> {
        self.service.admit_async().await.map(AdmissionTestHandle::new)
    }

    fn admit_query_with_request(&self, request: QueryAdmissionRequest) -> Result<AdmissionTestHandle> {
        self.service.admit_query_with_request(request).map(AdmissionTestHandle::new)
    }

    async fn admit_async_with_request(&self, request: QueryAdmissionRequest) -> Result<AdmissionTestHandle> {
        self.service.admit_async_with_request(request).await.map(AdmissionTestHandle::new)
    }

    fn try_admit_query(&self) -> Result<Option<AdmissionTestHandle>> {
        self.service.try_admit_query().map(|handle| handle.map(AdmissionTestHandle::new))
    }

    fn try_admit_query_with_request(&self, request: QueryAdmissionRequest) -> Result<Option<AdmissionTestHandle>> {
        self.service.try_admit_query_with_request(request).map(|handle| handle.map(AdmissionTestHandle::new))
    }

    fn enqueue_admission(&self) -> Result<AdmissionTestWaiter> {
        self.service.enqueue_admission().map(AdmissionTestWaiter::new)
    }

    fn enqueue_admission_with_request(&self, request: QueryAdmissionRequest) -> Result<AdmissionTestWaiter> {
        self.service.enqueue_admission_with_request(request).map(AdmissionTestWaiter::new)
    }

    fn admission_snapshot(&self) -> Result<QueryServiceAdmissionSnapshot> {
        self.service.admission_snapshot()
    }

    fn device_ledger_reports(&self) -> Result<Vec<DeviceLedgerReport>> {
        self.service.device_ledger_reports()
    }

    fn queued_len(&self) -> usize {
        self.service.inner.admission.lock().unwrap().queued.len()
    }

    fn queued_len_recovering_poison(&self) -> usize {
        self.service.inner.admission.lock().unwrap_or_else(std::sync::PoisonError::into_inner).queued.len()
    }

    fn pending_preflight_len(&self) -> usize {
        self.service.inner.admission.lock().unwrap_or_else(std::sync::PoisonError::into_inner).pending_preflight.len()
    }

    fn pending_preflight_resolver_calls(&self) -> usize {
        self.service.inner.pending_preflight_resolver_calls()
    }

    fn reset_pending_preflight_resolver_calls(&self) {
        self.service.inner.reset_pending_preflight_resolver_calls();
    }

    fn pause_pending_preflight(&self) -> crate::runtime::query_service::service::TestPauseHandle {
        self.service.inner.pause_pending_preflight_for_test()
    }

    fn fail_next_pending_preflight(&self) {
        self.service.inner.fail_next_pending_preflight_for_test();
    }

    fn poison_admission(&self) {
        let service = self.service.clone();
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _state = service.inner.admission.lock().expect("test acquires admission state");
            panic!("injected query service admission mutex poison");
        }));
    }

    fn close(&self) -> Result<()> {
        self.service.close()
    }

    fn close_device_admission_for_test(&self, device_ordinal: usize, reason: &str) -> Result<()> {
        self.service.close_device_admission_for_test(device_ordinal, reason)
    }

    fn close_and_wait_for_quiescence(&self, timeout: Duration) -> Result<()> {
        self.service.close_and_wait_for_quiescence(timeout)
    }

    fn close_and_wait_for_quiescence_until(&self, deadline: Instant) -> Result<()> {
        self.service.close_and_wait_for_quiescence_until(deadline)
    }

    fn hold_admission_lock(&self) -> Result<AdmissionTestLockGuard<'_>> {
        self.service
            .lock_admission()
            .map(|state| AdmissionTestLockGuard { state, synthetic_ticket: Arc::clone(&self.synthetic_ticket) })
    }

    fn shutdown_waiter_count(&self) -> usize {
        self.service.inner.shutdown_waiters.load(Ordering::Acquire)
    }

    fn notify_admission_available(&self) {
        self.service.inner.notify_admission_available();
    }

    fn expire_waiter(&self, waiter: &mut AdmissionTestWaiter) {
        let expired = Instant::now();
        waiter.waiter.deadline = expired;
        let admission_request_id = waiter.waiter.admission_request_id;
        self.service
            .inner
            .admission
            .lock()
            .expect("admission state")
            .queued
            .iter_mut()
            .find(|request| request.admission_request_id == admission_request_id)
            .expect("queued request")
            .deadline = expired;
    }
}

#[derive(Debug)]
struct AdmissionTestHandle {
    handle: QueryHandle,
}

impl AdmissionTestHandle {
    fn new(handle: QueryHandle) -> Self {
        Self { handle }
    }

    fn admission_request_id(&self) -> AdmissionRequestId {
        self.handle.admission_request_id()
    }

    fn device_budget_bytes(&self) -> u64 {
        self.handle.device_budget_bytes()
    }

    fn device_ordinal(&self) -> usize {
        self.handle.device_ordinal()
    }

    fn grant_identity(&self) -> crate::runtime::gpu::GrantIdentity {
        self.handle.grant_identity().expect("admitted handle owns a live grant identity")
    }

    fn boundary_policy(&self) -> ExecutionBoundaryPolicy {
        self.handle.boundary_policy()
    }

    fn source_scheduler_policy(&self) -> SourceSchedulerPolicy {
        self.handle.source_scheduler_policy()
    }

    fn source_chunk_planning_mode(&self) -> SourceChunkPlanningMode {
        self.handle.source_chunk_planning_mode()
    }

    fn statistical_aggregate_execution_mode(&self) -> StatisticalAggregateExecutionMode {
        self.handle.statistical_aggregate_execution_mode()
    }

    fn dynamic_filter_execution_options(&self) -> nexus_query_planning::dynamic_filter::DynamicFilterExecutionOptions {
        self.handle.dynamic_filter_execution_options()
    }

    fn keyed_hash_join_strategy_options(&self) -> nexus_query_planning::capability::KeyedHashJoinStrategyOptions {
        self.handle.keyed_hash_join_strategy_options()
    }

    fn aggregate_strategy_execution_options(
        &self,
    ) -> nexus_query_planning::capability::AggregateStrategyExecutionOptions {
        self.handle.aggregate_strategy_execution_options()
    }

    fn source_chunk_planning_options(&self) -> nexus_query_planning::chunk::SourceChunkPlanningOptions {
        self.handle.source_chunk_planning_options()
    }

    fn memory_grant(&self) -> Result<MemoryGrant> {
        self.handle.lifetime.memory_grant()
    }

    fn finish(&self) {
        self.handle.finish();
    }
}

#[derive(Debug)]
struct AdmissionTestLockGuard<'a> {
    state: std::sync::MutexGuard<'a, AdmissionState>,
    synthetic_ticket: Arc<std::sync::Mutex<Option<AdmissionTicket>>>,
}

impl AdmissionTestLockGuard<'_> {
    fn set_active_queries(&mut self, active_queries: usize) {
        let mut synthetic_ticket = self.synthetic_ticket.lock().expect("synthetic lifecycle ticket state");
        if active_queries == 0 {
            if let Some(mut ticket) = synthetic_ticket.take() {
                ticket.transition_to_cleanup().expect("synthetic lifecycle ticket enters cleanup exactly once");
                drop(ticket);
            }
            return;
        }
        if synthetic_ticket.is_none() {
            let mut ticket = self.state.lifecycle.begin_provisional(Weak::new());
            ticket.transition_to_active().expect("synthetic lifecycle ticket activates exactly once");
            *synthetic_ticket = Some(ticket);
        }
    }
}

#[derive(Debug)]
struct AdmissionTestWaiter {
    waiter: QueryAdmissionWaiter,
}

impl AdmissionTestWaiter {
    fn new(waiter: QueryAdmissionWaiter) -> Self {
        Self { waiter }
    }

    fn try_admit(&mut self) -> Result<Option<AdmissionTestHandle>> {
        self.waiter.try_admit().map(|handle| handle.map(AdmissionTestHandle::new))
    }

    fn admission_request_id(&self) -> AdmissionRequestId {
        self.waiter.admission_request_id
    }
}

/// Placement-only test evidence carrying no capability or plan-scope facts.
///
/// Grant sizing is entirely ledger/device-profile driven; evidence carries no
/// bytes at all, so this fixture exists only to give tests a concrete
/// `AttemptRequirement` value to attach and to distinguish device entries.
fn requirement() -> AttemptRequirement {
    AttemptRequirement::empty()
}

/// Test evidence distinct from [`requirement`] by its required capability set,
/// for tests asserting that per-device entries are not collapsed together.
fn distinct_capability_requirement() -> AttemptRequirement {
    AttemptRequirement::from_parts(
        vec![nexus_query_planning::admission::ExecutionCapability::new("test_distinct_operator")],
        Vec::new(),
    )
}

fn observed_admission_events(owner: &RuntimeObservationOwner) -> Vec<(AdmissionRequestId, AdmissionLifecycleState)> {
    let collect = |events: &[crate::observability::RuntimeTimelineEvent]| {
        events
            .iter()
            .filter_map(|event| match &event.payload {
                RuntimeObservationPayload::Admission(observation) => {
                    Some((observation.admission_request_id, observation.lifecycle.clone()))
                },
                _ => None,
            })
            .collect()
    };
    if let Some(snapshot) = owner.live_snapshot() {
        collect(snapshot.events())
    } else {
        collect(owner.sealed_snapshot().expect("observation is live or sealed").events())
    }
}
fn wait_for_sealed_observation(
    owner: &RuntimeObservationOwner,
) -> crate::observability::SealedRuntimeObservationSnapshot {
    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        if let Some(snapshot) = owner.sealed_snapshot() {
            return snapshot;
        }
        assert!(Instant::now() < deadline, "timed out waiting for observation sealing");
        std::thread::yield_now();
    }
}
fn non_gds_deployment() -> DeploymentIoCapability {
    DeploymentIoCapability::from_probe_inputs(
        NvidiaFsProbeStatus::DriverMissing,
        KvikioCompatModeStatus::Unset,
        GdsDriverProbeStatus::ProbeUnimplemented,
    )
}
fn test_ledger(
    managed_capacity_bytes: u64,
    max_active_attempts: usize,
    device_capabilities: crate::runtime::DeviceCapabilities,
) -> DeviceLedger {
    test_ledger_for_ordinal(0, managed_capacity_bytes, max_active_attempts, device_capabilities)
}
fn test_ledger_for_ordinal(
    device_ordinal: usize,
    managed_capacity_bytes: u64,
    max_active_attempts: usize,
    device_capabilities: crate::runtime::DeviceCapabilities,
) -> DeviceLedger {
    let max_active_attempts = max_active_attempts.max(1);
    let floor =
        (managed_capacity_bytes / u64::try_from(max_active_attempts.saturating_mul(8)).unwrap_or(u64::MAX)).max(1);
    test_ledger_with_explicit_profile(
        device_ordinal,
        managed_capacity_bytes,
        floor,
        max_active_attempts,
        device_capabilities,
    )
}
fn test_ledger_with_explicit_profile(
    device_ordinal: usize,
    managed_capacity_bytes: u64,
    default_query_limit_bytes: u64,
    max_active_attempts: usize,
    device_capabilities: crate::runtime::DeviceCapabilities,
) -> DeviceLedger {
    let max_active_attempts = max_active_attempts.max(1);
    let source_work = 1usize;
    let slots = max_active_attempts.saturating_mul(source_work.saturating_add(1)).max(max_active_attempts);
    let profile = DeviceResourceProfile::resolve(DeviceResourceEnvelope {
        device_ordinal,
        configured_limit_bytes: managed_capacity_bytes,
        headroom_bytes: 0,
        backend_reserve_bytes: 0,
        default_query_limit_bytes,
        max_active_attempts,
        source_work_per_attempt: source_work,
        stream_pool_size: slots,
        worker_job_capacity: slots,
        device_capabilities,
        cache_cap_bytes: 0,
    })
    .expect("test ledger profile must resolve");
    DeviceLedger::new(profile)
}

/// Build the CUDA-free admission fixture with the exact per-device ledger
/// contract asserted by a test. `QueryExecutionConfig` deliberately carries
/// no device limits, so test-local capacity and floor values belong here.
fn admission_test_service_with_profile(
    config: QueryExecutionConfig,
    queue_bounds: QueueBounds,
    managed_capacity_bytes: u64,
    default_query_limit_bytes: u64,
    max_active_attempts: usize,
) -> AdmissionTestController {
    AdmissionTestController::new_with_ledger(
        config,
        queue_bounds,
        test_ledger_with_explicit_profile(
            0,
            managed_capacity_bytes,
            default_query_limit_bytes,
            max_active_attempts,
            Default::default(),
        ),
    )
    .expect("explicit admission test profile must resolve")
}

fn test_service(max_active_attempts_per_device: usize, max_waiters: usize) -> AdmissionTestController {
    let queue_bounds = QueueBounds { max_waiters, max_wait: Duration::from_millis(500) };
    admission_test_service_with_profile(
        QueryExecutionConfig::default(),
        queue_bounds,
        64 * 1024 * 1024,
        (64 * 1024 * 1024 / u64::try_from(max_active_attempts_per_device.saturating_mul(8)).unwrap_or(u64::MAX)).max(1),
        max_active_attempts_per_device,
    )
}
fn runtime_test_service(max_active_attempts_per_device: usize, max_waiters: usize) -> QueryService {
    let queue_bounds = QueueBounds { max_waiters, max_wait: Duration::from_millis(500) };
    runtime_test_service_with_profiles(
        QueryExecutionConfig::default(),
        queue_bounds,
        vec![runtime_test_device_profile(max_active_attempts_per_device)],
    )
    .unwrap()
}
fn try_guaranteed_runtime_test_service() -> Result<QueryService> {
    try_guaranteed_runtime_test_service_with_concurrency(1)
}
fn try_guaranteed_runtime_test_service_with_concurrency(max_active_attempts_per_device: usize) -> Result<QueryService> {
    runtime_test_service_with_profiles(
        QueryExecutionConfig::default(),
        QueueBounds {
            // Keep the guaranteed-runtime fixture independent of the host
            // inventory. Its regressions request small, deterministic grants.
            max_waiters: 8,
            max_wait: Duration::from_millis(500),
        },
        vec![runtime_test_device_profile(max_active_attempts_per_device)],
    )
}

fn runtime_test_service_with_profiles(
    config: QueryExecutionConfig,
    queue_bounds: QueueBounds,
    device_profiles: Vec<GpuBackendDeviceConfig>,
) -> Result<QueryService> {
    QueryService::new_with_device_profiles_and_deployment_for_test(
        config,
        queue_bounds,
        device_profiles,
        non_gds_deployment(),
    )
}

/// Runtime-backed service tests must install the same explicit per-device
/// contract their assertions exercise.  Host-derived construction silently
/// forces one active attempt and makes capacity assertions depend on the
/// machine running the test.
fn runtime_test_device_profile(max_active_attempts_per_device: usize) -> GpuBackendDeviceConfig {
    const CAPACITY_BYTES: u64 = 64 * 1024 * 1024;

    let max_active_attempts = max_active_attempts_per_device.max(1);
    let default_query_limit_bytes =
        (CAPACITY_BYTES / u64::try_from(max_active_attempts.saturating_mul(8)).unwrap_or(u64::MAX)).max(1);
    // Match `SourceSchedulerPolicy::default()`: every active attempt reserves
    // one root stream plus two bounded auxiliary source-work slots.
    let stream_pool_size = max_active_attempts.saturating_mul(3);

    runtime_test_device_profile_with_limits(
        0,
        CAPACITY_BYTES,
        default_query_limit_bytes,
        max_active_attempts,
        stream_pool_size,
        max_active_attempts,
        max_active_attempts,
    )
}

fn runtime_test_device_profile_with_limits(
    device_ordinal: usize,
    capacity_bytes: u64,
    default_query_limit_bytes: u64,
    max_active_attempts: usize,
    stream_pool_size: usize,
    worker_count: usize,
    worker_lane_count: usize,
) -> GpuBackendDeviceConfig {
    GpuBackendDeviceConfig::new(
        GpuBackendDeviceProfile::new(device_ordinal)
            .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: capacity_bytes })
            .with_default_query_limit_bytes(default_query_limit_bytes)
            .with_source_work_per_attempt(2)
            .with_max_active_attempts(max_active_attempts)
            .with_installation_bounds(GpuBackendDeviceInstallationBounds::new(
                stream_pool_size,
                worker_count,
                worker_lane_count,
            )),
        GpuMemoryOwnership::WholeDeviceExclusive,
    )
}
fn wait_for_queued_len(service: &AdmissionTestController, expected: usize) {
    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        if service.queued_len() >= expected {
            return;
        }
        assert!(Instant::now() < deadline, "timed out waiting for queued admissions");
        std::thread::sleep(Duration::from_millis(5));
    }
}
fn wait_for_exact_queued_len(service: &AdmissionTestController, expected: usize) {
    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        let queued = service.admission_snapshot().unwrap().queued_queries;
        if queued == expected {
            return;
        }
        assert!(Instant::now() < deadline, "timed out waiting for {expected} queued admissions; last queued={queued}");
        std::thread::sleep(Duration::from_millis(5));
    }
}
fn wait_for_pending_preflight_len(service: &AdmissionTestController, expected: usize) {
    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        let pending = service.pending_preflight_len();
        if pending == expected {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {expected} pending preflight reservations; last pending={pending}"
        );
        std::thread::yield_now();
    }
}
fn first_memory_governor_report(service: &AdmissionTestController) -> DeviceLedgerReport {
    let reports = service.device_ledger_reports().unwrap();
    assert_eq!(reports.len(), 1);
    reports.into_iter().next().unwrap()
}
