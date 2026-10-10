use super::*;

/// Admits now, or reports that the request would queue, through the queue
/// waiter; a request that cannot admit leaves the queue with its waiter.
impl QueryService {
    pub(super) fn try_admit_query(&self) -> Result<Option<QueryHandle>> {
        self.try_admit_query_with_request(QueryAdmissionRequest::test_attempt())
    }

    pub(super) fn try_admit_query_with_request(&self, request: QueryAdmissionRequest) -> Result<Option<QueryHandle>> {
        self.enqueue_admission_with_request(request)?.try_admit()
    }
}

/// CUDA-free admission fixture over the controller alone. A placed request
/// activates its lifecycle ticket and keeps its grant; no envelope is built
/// and no query handle is published.
#[derive(Clone, Debug)]
pub(super) struct AdmissionTestController {
    pub(super) controller: Arc<AdmissionController>,
    pub(super) synthetic_ticket: Arc<std::sync::Mutex<Option<AdmissionTicket>>>,
}

impl AdmissionTestController {
    pub(super) fn new(queue_bounds: QueueBounds) -> Result<Self> {
        Self::new_with_ledger(queue_bounds, test_ledger(64 * 1024 * 1024, 1, Default::default()))
    }

    pub(super) fn new_with_ledger(queue_bounds: QueueBounds, ledger: DeviceLedger) -> Result<Self> {
        Self::new_with_ledgers(queue_bounds, [ledger])
    }

    pub(super) fn new_with_ledgers(
        queue_bounds: QueueBounds,
        ledgers: impl IntoIterator<Item = DeviceLedger>,
    ) -> Result<Self> {
        let devices =
            AdmissionDevices::new(ledgers.into_iter().map(|ledger| AdmissionDevice::new(Arc::new(ledger), || None)));
        AdmissionController::new(queue_bounds, devices)
            .map(|controller| Self { controller, synthetic_ticket: Arc::new(std::sync::Mutex::new(None)) })
    }

    pub(super) fn new_with_device_capabilities(
        queue_bounds: QueueBounds,
        device_capabilities: DeviceCapabilities,
    ) -> Result<Self> {
        Self::new_with_ledger(queue_bounds, test_ledger(64 * 1024 * 1024, 1, device_capabilities))
    }

    pub(super) fn admit_query(&self) -> Result<AdmissionTestHandle> {
        self.admit_query_with_request(QueryAdmissionRequest::test_attempt())
    }

    pub(super) async fn admit_async(&self) -> Result<AdmissionTestHandle> {
        self.admit_async_with_request(QueryAdmissionRequest::test_attempt()).await
    }

    pub(super) fn admit_query_with_request(&self, request: QueryAdmissionRequest) -> Result<AdmissionTestHandle> {
        let mut waiter = self.enqueue_admission_with_request(request)?;
        let placed = waiter.waiter.place_blocking()?;
        Ok(AdmissionTestHandle::activate(&self.controller, placed, waiter.test_attempt.take()))
    }

    pub(super) async fn admit_async_with_request(&self, request: QueryAdmissionRequest) -> Result<AdmissionTestHandle> {
        let mut waiter = self.enqueue_admission_with_request(request)?;
        let placed = waiter.waiter.place_async().await?;
        Ok(AdmissionTestHandle::activate(&self.controller, placed, waiter.test_attempt.take()))
    }

    pub(super) fn try_admit_query(&self) -> Result<Option<AdmissionTestHandle>> {
        self.try_admit_query_with_request(QueryAdmissionRequest::test_attempt())
    }

    pub(super) fn try_admit_query_with_request(
        &self,
        request: QueryAdmissionRequest,
    ) -> Result<Option<AdmissionTestHandle>> {
        self.enqueue_admission_with_request(request)?.try_admit()
    }

    pub(super) fn enqueue_admission(&self) -> Result<AdmissionTestWaiter> {
        self.enqueue_admission_with_request(QueryAdmissionRequest::test_attempt())
    }

    pub(super) fn enqueue_admission_with_request(&self, request: QueryAdmissionRequest) -> Result<AdmissionTestWaiter> {
        let submission = self.begin_submission(request)?;
        self.finish_submission(submission)
    }

    /// Holds a pending-preflight slot until [`Self::finish_submission`].
    pub(super) fn begin_submission(&self, request: QueryAdmissionRequest) -> Result<AdmissionTestSubmission> {
        let (request, test_attempt) = request.split_test_attempt();
        let submission = self.controller.begin_submission(request, None, None, None)?;
        Ok(AdmissionTestSubmission { submission, test_attempt })
    }

    pub(super) fn finish_submission(&self, submission: AdmissionTestSubmission) -> Result<AdmissionTestWaiter> {
        let AdmissionTestSubmission { submission, test_attempt } = submission;
        let waiter = self.controller.finish_submission(submission)?;
        Ok(AdmissionTestWaiter { controller: Arc::clone(&self.controller), waiter, test_attempt })
    }

    pub(super) fn finish_submission_with<F>(
        &self,
        submission: AdmissionTestSubmission,
        resolve: F,
    ) -> Result<AdmissionTestWaiter>
    where
        F: FnMut(usize) -> Result<AttemptCapabilityEvidence>,
    {
        let AdmissionTestSubmission { submission, test_attempt } = submission;
        let waiter = self.controller.finish_submission_with(submission, resolve)?;
        Ok(AdmissionTestWaiter { controller: Arc::clone(&self.controller), waiter, test_attempt })
    }

    pub(super) fn admission_snapshot(&self) -> Result<QueryServiceAdmissionSnapshot> {
        self.controller.admission_snapshot()
    }

    pub(super) fn device_ledger_reports(&self) -> Result<Vec<DeviceLedgerReport>> {
        self.controller.device_ledger_reports()
    }

    pub(super) fn queued_len(&self) -> usize {
        self.controller.state.lock().unwrap().queued.len()
    }

    pub(super) fn queued_len_recovering_poison(&self) -> usize {
        self.controller.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner).queued.len()
    }

    pub(super) fn pending_preflight_len(&self) -> usize {
        self.controller.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner).pending_preflight.len()
    }

    pub(super) fn lifecycle_snapshot(&self) -> AdmissionLifecycleSnapshot {
        self.controller.state.lock().expect("admission state").lifecycle.snapshot()
    }

    pub(super) fn poison_admission(&self) {
        let controller = Arc::clone(&self.controller);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _state = controller.state.lock().expect("test acquires admission state");
            panic!("injected query service admission mutex poison");
        }));
    }

    pub(super) fn close(&self) -> Result<()> {
        self.controller.close().finish()
    }

    pub(super) fn close_device_admission_for_test(&self, device_ordinal: usize, reason: &str) -> Result<()> {
        self.controller.close_device_admission_for_test(device_ordinal, reason)
    }

    pub(super) fn close_and_wait_for_quiescence(&self, timeout: Duration) -> Result<()> {
        self.close_and_wait_for_quiescence_until(Instant::now() + timeout)
    }

    pub(super) fn close_and_wait_for_quiescence_until(&self, deadline: Instant) -> Result<()> {
        self.controller.close_gate();
        self.controller.notify_admission_available();
        let admission = self
            .controller
            .wait_for_quiescence_until(deadline, "query_engine.query_service.close_and_wait_for_quiescence");
        QueryServiceDrainOutcome::new(vec![admission]).into_result()
    }

    pub(super) fn hold_admission_lock(&self) -> Result<AdmissionTestLockGuard<'_>> {
        self.controller
            .lock_admission()
            .map(|state| AdmissionTestLockGuard { state, synthetic_ticket: Arc::clone(&self.synthetic_ticket) })
    }

    pub(super) fn shutdown_waiter_count(&self) -> usize {
        self.controller.shutdown_waiters.load(Ordering::Acquire)
    }

    pub(super) fn notify_admission_available(&self) {
        self.controller.notify_admission_available();
    }

    pub(super) fn expire_waiter(&self, waiter: &mut AdmissionTestWaiter) {
        let expired = Instant::now();
        waiter.waiter.deadline = Some(expired);
        let admission_request_id = waiter.waiter.admission_request_id;
        self.controller
            .state
            .lock()
            .expect("admission state")
            .queued
            .iter_mut()
            .find(|request| request.admission_request_id == admission_request_id)
            .expect("queued request")
            .deadline = Some(expired);
    }
}

/// A placed request whose lifecycle ticket is active. Dropping it returns the
/// grant's budget before the ticket leaves the lifecycle, the envelope
/// teardown order.
#[derive(Debug)]
pub(super) struct AdmissionTestHandle {
    pub(super) controller: Arc<AdmissionController>,
    pub(super) device_ordinal: usize,
    pub(super) admitted_device_requirements: BTreeMap<usize, AttemptCapabilityEvidence>,
    pub(super) plan_identities: AdmittedPlanIdentities,
    pub(super) memory_grant: MemoryGrant,
    pub(super) _lifecycle_ticket: AdmissionTicket,
    pub(super) _completion: QueryCompletion,
    pub(super) _test_attempt: Option<AttemptOwner>,
}

impl AdmissionTestHandle {
    pub(super) fn activate(
        controller: &Arc<AdmissionController>,
        placed: PlacedAdmission,
        test_attempt: Option<AttemptOwner>,
    ) -> Self {
        let PlacedAdmission {
            placed_query: PlacedQuery { device_ordinal, memory_grant },
            admitted_device_requirements,
            submitted_plan_capability_evidence,
            completion,
            mut lifecycle_ticket,
            ..
        } = placed;
        let state = controller.state.lock().expect("admission state");
        lifecycle_ticket.transition_to_active().expect("a placed lifecycle ticket activates once");
        drop(state);
        Self {
            controller: Arc::clone(controller),
            device_ordinal,
            admitted_device_requirements,
            plan_identities: AdmittedPlanIdentities::new(submitted_plan_capability_evidence),
            memory_grant,
            _lifecycle_ticket: lifecycle_ticket,
            _completion: completion,
            _test_attempt: test_attempt,
        }
    }

    pub(super) fn admitted_device_requirement(&self, ordinal: usize) -> Option<&AttemptCapabilityEvidence> {
        self.admitted_device_requirements.get(&ordinal)
    }

    pub(super) fn admitted_capability_evidence(&self) -> Option<Arc<AttemptCapabilityEvidence>> {
        self.plan_identities.current()
    }

    pub(super) fn join_fragment(&self, fragment: &AttemptCapabilityEvidence) -> Result<()> {
        self.controller.join_admitted_fragment(self.device_ordinal, &self.plan_identities, fragment)
    }

    pub(super) fn device_budget_bytes(&self) -> u64 {
        self.memory_grant.budget_bytes()
    }

    pub(super) fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    pub(super) fn grant_identity(&self) -> GrantIdentity {
        self.memory_grant.identity()
    }

    pub(super) fn memory_grant(&self) -> MemoryGrant {
        self.memory_grant.clone()
    }
}

#[derive(Debug)]
pub(super) struct AdmissionTestLockGuard<'a> {
    pub(super) state: std::sync::MutexGuard<'a, AdmissionState>,
    pub(super) synthetic_ticket: Arc<std::sync::Mutex<Option<AdmissionTicket>>>,
}

impl AdmissionTestLockGuard<'_> {
    pub(super) fn set_active_queries(&mut self, active_queries: usize) {
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

/// A submission holding its pending-preflight slot.
#[derive(Debug)]
pub(super) struct AdmissionTestSubmission {
    pub(super) submission: PendingSubmission,
    pub(super) test_attempt: Option<AttemptOwner>,
}

#[derive(Debug)]
pub(super) struct AdmissionTestWaiter {
    pub(super) controller: Arc<AdmissionController>,
    pub(super) waiter: AdmissionWaiter,
    pub(super) test_attempt: Option<AttemptOwner>,
}

impl AdmissionTestWaiter {
    pub(super) fn try_admit(&mut self) -> Result<Option<AdmissionTestHandle>> {
        let Some(placed) = self.waiter.try_place()? else {
            return Ok(None);
        };
        Ok(Some(AdmissionTestHandle::activate(&self.controller, placed, self.test_attempt.take())))
    }

    pub(super) fn admission_request_id(&self) -> AdmissionRequestId {
        self.waiter.admission_request_id
    }
}

/// Placement-only test evidence carrying no capability or plan-scope facts.
///
/// Grant sizing is entirely ledger/device-profile driven; evidence carries no
/// bytes at all, so this fixture exists only to give tests a concrete
/// `AttemptCapabilityEvidence` value to attach and to distinguish device entries.
pub(super) fn requirement() -> AttemptCapabilityEvidence {
    AttemptCapabilityEvidence::empty()
}

/// Test evidence distinct from [`requirement`] by its required capability set,
/// for tests asserting that per-device entries are not collapsed together.
pub(super) fn distinct_capability_requirement() -> AttemptCapabilityEvidence {
    AttemptCapabilityEvidence::from_parts(vec![ExecutionCapability::new("test_distinct_operator")], Vec::new())
}

pub(super) fn observed_admission_events(
    observation: &RuntimeObservationHandle,
) -> Vec<(AdmissionRequestId, AdmissionLifecycleState)> {
    let collect = |events: &[RuntimeTimelineEvent]| {
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
    if let Some(snapshot) = observation.live_snapshot() {
        collect(snapshot.events())
    } else {
        collect(observation.sealed_snapshot().expect("observation is live or sealed").events())
    }
}
pub(super) fn wait_for_sealed_observation(observation: &RuntimeObservationHandle) -> SealedRuntimeObservationSnapshot {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(snapshot) = observation.sealed_snapshot() {
            return snapshot;
        }
        assert!(Instant::now() < deadline, "timed out waiting for observation sealing");
        std::thread::yield_now();
    }
}
pub(super) fn test_ledger(
    managed_capacity_bytes: u64,
    max_active_attempts: usize,
    device_capabilities: DeviceCapabilities,
) -> DeviceLedger {
    test_ledger_for_ordinal(0, managed_capacity_bytes, max_active_attempts, device_capabilities)
}
pub(super) fn test_ledger_for_ordinal(
    device_ordinal: usize,
    managed_capacity_bytes: u64,
    max_active_attempts: usize,
    device_capabilities: DeviceCapabilities,
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
    device_capabilities: DeviceCapabilities,
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
/// contract asserted by a test.
pub(super) fn admission_test_service_with_profile(
    queue_bounds: QueueBounds,
    managed_capacity_bytes: u64,
    default_query_limit_bytes: u64,
    max_active_attempts: usize,
) -> AdmissionTestController {
    AdmissionTestController::new_with_ledger(
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

pub(super) fn test_service(max_active_attempts_per_device: usize, max_waiters: usize) -> AdmissionTestController {
    let queue_bounds = QueueBounds { max_waiters, max_wait: Duration::from_millis(500) };
    admission_test_service_with_profile(
        queue_bounds,
        64 * 1024 * 1024,
        (64 * 1024 * 1024 / u64::try_from(max_active_attempts_per_device.saturating_mul(8)).unwrap_or(u64::MAX)).max(1),
        max_active_attempts_per_device,
    )
}
pub(super) fn runtime_test_service(max_active_attempts_per_device: usize, max_waiters: usize) -> QueryService {
    let queue_bounds = QueueBounds { max_waiters, max_wait: Duration::from_millis(500) };
    runtime_test_service_with_profiles(
        QueryExecutionConfig::default(),
        queue_bounds,
        vec![runtime_test_device_profile(max_active_attempts_per_device)],
    )
    .unwrap()
}
pub(super) fn try_guaranteed_runtime_test_service() -> Result<QueryService> {
    try_guaranteed_runtime_test_service_with_concurrency(1)
}
pub(super) fn try_guaranteed_runtime_test_service_with_concurrency(
    max_active_attempts_per_device: usize,
) -> Result<QueryService> {
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

pub(super) fn runtime_test_service_with_profiles(
    config: QueryExecutionConfig,
    queue_bounds: QueueBounds,
    device_profiles: Vec<GpuBackendDeviceConfig>,
) -> Result<QueryService> {
    QueryService::new_with_device_profiles_for_test(config, queue_bounds, device_profiles)
}

/// Runtime-backed service tests must install the same explicit per-device
/// contract their assertions exercise.  Host-derived construction silently
/// forces one active attempt and makes capacity assertions depend on the
/// machine running the test.
pub(super) fn runtime_test_device_profile(max_active_attempts_per_device: usize) -> GpuBackendDeviceConfig {
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
        max_active_attempts + RESERVED_LIFECYCLE_WORKERS,
        max_active_attempts,
    )
}

pub(super) fn runtime_test_device_profile_with_limits(
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
pub(super) fn wait_for_queued_len(service: &AdmissionTestController, expected: usize) {
    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        if service.queued_len() >= expected {
            return;
        }
        assert!(Instant::now() < deadline, "timed out waiting for queued admissions");
        std::thread::sleep(Duration::from_millis(5));
    }
}
pub(super) fn wait_for_exact_queued_len(service: &AdmissionTestController, expected: usize) {
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
pub(super) fn wait_for_pending_preflight_len(service: &AdmissionTestController, expected: usize) {
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
pub(super) fn first_memory_governor_report(service: &AdmissionTestController) -> DeviceLedgerReport {
    let reports = service.device_ledger_reports().unwrap();
    assert_eq!(reports.len(), 1);
    reports.into_iter().next().unwrap()
}
