/// Admits now, or reports that the request would queue, through the queue
/// waiter; a request that cannot admit leaves the queue with its waiter.
impl QueryService {
    fn try_admit_query(&self) -> Result<Option<QueryHandle>> {
        self.try_admit_query_with_request(QueryAdmissionRequest::test_attempt())
    }

    fn try_admit_query_with_request(&self, request: QueryAdmissionRequest) -> Result<Option<QueryHandle>> {
        self.enqueue_admission_with_request(request)?.try_admit()
    }
}

use std::collections::BTreeMap;

use query_planning::{
    admission::ExecutionCapability,
    capability::{AggregateStrategyExecutionOptions, KeyedHashJoinStrategyOptions},
    chunk::SourceChunkPlanningOptions,
    dynamic_filter::DynamicFilterExecutionOptions,
};
use crate::{
    observability::{RuntimeTimelineEvent, SealedRuntimeObservationSnapshot},
    runtime::{
        DeviceCapabilities,
        context::AdmittedPlanIdentities,
        gpu::{DeviceLedger, GrantIdentity},
        gpu_execution_pool::RESERVED_LIFECYCLE_WORKERS,
    },
};

/// CUDA-free admission fixture over the controller alone. A placed request
/// activates its lifecycle ticket and keeps its grant; no envelope is built
/// and no query handle is published.
#[derive(Clone, Debug)]
struct AdmissionTestController {
    controller: Arc<AdmissionController>,
    synthetic_ticket: Arc<std::sync::Mutex<Option<AdmissionTicket>>>,
}

impl AdmissionTestController {
    fn new(queue_bounds: QueueBounds) -> Result<Self> {
        Self::new_with_ledger(queue_bounds, test_ledger(64 * 1024 * 1024, 1, Default::default()))
    }

    fn new_with_ledger(queue_bounds: QueueBounds, ledger: DeviceLedger) -> Result<Self> {
        Self::new_with_ledgers(queue_bounds, [ledger])
    }

    fn new_with_ledgers(queue_bounds: QueueBounds, ledgers: impl IntoIterator<Item = DeviceLedger>) -> Result<Self> {
        let devices =
            AdmissionDevices::new(ledgers.into_iter().map(|ledger| AdmissionDevice::new(Arc::new(ledger), || None)));
        AdmissionController::new(queue_bounds, devices)
            .map(|controller| Self { controller, synthetic_ticket: Arc::new(std::sync::Mutex::new(None)) })
    }

    fn new_with_device_capabilities(queue_bounds: QueueBounds, device_capabilities: DeviceCapabilities) -> Result<Self> {
        Self::new_with_ledger(queue_bounds, test_ledger(64 * 1024 * 1024, 1, device_capabilities))
    }

    fn admit_query(&self) -> Result<AdmissionTestHandle> {
        self.admit_query_with_request(QueryAdmissionRequest::test_attempt())
    }

    async fn admit_async(&self) -> Result<AdmissionTestHandle> {
        self.admit_async_with_request(QueryAdmissionRequest::test_attempt()).await
    }

    fn admit_query_with_request(&self, request: QueryAdmissionRequest) -> Result<AdmissionTestHandle> {
        let mut waiter = self.enqueue_admission_with_request(request)?;
        let placed = waiter.waiter.place_blocking()?;
        Ok(AdmissionTestHandle::activate(&self.controller, placed, waiter.test_attempt.take()))
    }

    async fn admit_async_with_request(&self, request: QueryAdmissionRequest) -> Result<AdmissionTestHandle> {
        let mut waiter = self.enqueue_admission_with_request(request)?;
        let placed = waiter.waiter.place_async().await?;
        Ok(AdmissionTestHandle::activate(&self.controller, placed, waiter.test_attempt.take()))
    }

    fn try_admit_query(&self) -> Result<Option<AdmissionTestHandle>> {
        self.try_admit_query_with_request(QueryAdmissionRequest::test_attempt())
    }

    fn try_admit_query_with_request(&self, request: QueryAdmissionRequest) -> Result<Option<AdmissionTestHandle>> {
        self.enqueue_admission_with_request(request)?.try_admit()
    }

    fn enqueue_admission(&self) -> Result<AdmissionTestWaiter> {
        self.enqueue_admission_with_request(QueryAdmissionRequest::test_attempt())
    }

    fn enqueue_admission_with_request(&self, request: QueryAdmissionRequest) -> Result<AdmissionTestWaiter> {
        let submission = self.begin_submission(request)?;
        self.finish_submission(submission)
    }

    /// Holds a pending-preflight slot until [`Self::finish_submission`].
    fn begin_submission(&self, request: QueryAdmissionRequest) -> Result<AdmissionTestSubmission> {
        let (request, test_attempt) = request.split_test_attempt();
        let submission = self.controller.begin_submission(request, None, None, None)?;
        Ok(AdmissionTestSubmission { submission, test_attempt })
    }

    fn finish_submission(&self, submission: AdmissionTestSubmission) -> Result<AdmissionTestWaiter> {
        let AdmissionTestSubmission { submission, test_attempt } = submission;
        let waiter = self.controller.finish_submission(submission)?;
        Ok(AdmissionTestWaiter { controller: Arc::clone(&self.controller), waiter, test_attempt })
    }

    fn finish_submission_with<F>(&self, submission: AdmissionTestSubmission, resolve: F) -> Result<AdmissionTestWaiter>
    where
        F: FnMut(usize) -> Result<AttemptCapabilityEvidence>,
    {
        let AdmissionTestSubmission { submission, test_attempt } = submission;
        let waiter = self.controller.finish_submission_with(submission, resolve)?;
        Ok(AdmissionTestWaiter { controller: Arc::clone(&self.controller), waiter, test_attempt })
    }

    fn admission_snapshot(&self) -> Result<QueryServiceAdmissionSnapshot> {
        self.controller.admission_snapshot()
    }

    fn device_ledger_reports(&self) -> Result<Vec<DeviceLedgerReport>> {
        self.controller.device_ledger_reports()
    }

    fn queued_len(&self) -> usize {
        self.controller.state.lock().unwrap().queued.len()
    }

    fn queued_len_recovering_poison(&self) -> usize {
        self.controller.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner).queued.len()
    }

    fn pending_preflight_len(&self) -> usize {
        self.controller.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner).pending_preflight.len()
    }

    fn lifecycle_snapshot(&self) -> AdmissionLifecycleSnapshot {
        self.controller.state.lock().expect("admission state").lifecycle.snapshot()
    }

    fn poison_admission(&self) {
        let controller = Arc::clone(&self.controller);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _state = controller.state.lock().expect("test acquires admission state");
            panic!("injected query service admission mutex poison");
        }));
    }

    fn close(&self) -> Result<()> {
        self.controller.close().finish()
    }

    fn close_device_admission_for_test(&self, device_ordinal: usize, reason: &str) -> Result<()> {
        self.controller.close_device_admission_for_test(device_ordinal, reason)
    }

    fn close_and_wait_for_quiescence(&self, timeout: Duration) -> Result<()> {
        self.close_and_wait_for_quiescence_until(Instant::now() + timeout)
    }

    fn close_and_wait_for_quiescence_until(&self, deadline: Instant) -> Result<()> {
        self.controller.close_gate();
        self.controller.notify_admission_available();
        let admission = self
            .controller
            .wait_for_quiescence_until(deadline, "query_engine.query_service.close_and_wait_for_quiescence");
        QueryServiceDrainOutcome::new(vec![admission]).into_result()
    }

    fn hold_admission_lock(&self) -> Result<AdmissionTestLockGuard<'_>> {
        self.controller
            .lock_admission()
            .map(|state| AdmissionTestLockGuard { state, synthetic_ticket: Arc::clone(&self.synthetic_ticket) })
    }

    fn shutdown_waiter_count(&self) -> usize {
        self.controller.shutdown_waiters.load(Ordering::Acquire)
    }

    fn notify_admission_available(&self) {
        self.controller.notify_admission_available();
    }

    fn expire_waiter(&self, waiter: &mut AdmissionTestWaiter) {
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
struct AdmissionTestHandle {
    controller: Arc<AdmissionController>,
    device_ordinal: usize,
    admitted_device_requirements: BTreeMap<usize, AttemptCapabilityEvidence>,
    plan_identities: AdmittedPlanIdentities,
    memory_grant: MemoryGrant,
    _lifecycle_ticket: AdmissionTicket,
    _completion: QueryCompletion,
    _test_attempt: Option<AttemptOwner>,
}

impl AdmissionTestHandle {
    fn activate(
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

    fn admitted_device_requirement(&self, ordinal: usize) -> Option<&AttemptCapabilityEvidence> {
        self.admitted_device_requirements.get(&ordinal)
    }

    fn admitted_capability_evidence(&self) -> Option<Arc<AttemptCapabilityEvidence>> {
        self.plan_identities.current()
    }

    fn join_fragment(&self, fragment: &AttemptCapabilityEvidence) -> Result<()> {
        self.controller.join_admitted_fragment(self.device_ordinal, &self.plan_identities, fragment)
    }

    fn device_budget_bytes(&self) -> u64 {
        self.memory_grant.budget_bytes()
    }

    fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    fn grant_identity(&self) -> GrantIdentity {
        self.memory_grant.identity()
    }

    fn memory_grant(&self) -> MemoryGrant {
        self.memory_grant.clone()
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

/// A submission holding its pending-preflight slot.
#[derive(Debug)]
struct AdmissionTestSubmission {
    submission: PendingSubmission,
    test_attempt: Option<AttemptOwner>,
}

#[derive(Debug)]
struct AdmissionTestWaiter {
    controller: Arc<AdmissionController>,
    waiter: AdmissionWaiter,
    test_attempt: Option<AttemptOwner>,
}

impl AdmissionTestWaiter {
    fn try_admit(&mut self) -> Result<Option<AdmissionTestHandle>> {
        let Some(placed) = self.waiter.try_place()? else {
            return Ok(None);
        };
        Ok(Some(AdmissionTestHandle::activate(&self.controller, placed, self.test_attempt.take())))
    }

    fn admission_request_id(&self) -> AdmissionRequestId {
        self.waiter.admission_request_id
    }
}

/// Placement-only test evidence carrying no capability or plan-scope facts.
///
/// Grant sizing is entirely ledger/device-profile driven; evidence carries no
/// bytes at all, so this fixture exists only to give tests a concrete
/// `AttemptCapabilityEvidence` value to attach and to distinguish device entries.
fn requirement() -> AttemptCapabilityEvidence {
    AttemptCapabilityEvidence::empty()
}

/// Test evidence distinct from [`requirement`] by its required capability set,
/// for tests asserting that per-device entries are not collapsed together.
fn distinct_capability_requirement() -> AttemptCapabilityEvidence {
    AttemptCapabilityEvidence::from_parts(
        vec![ExecutionCapability::new("test_distinct_operator")],
        Vec::new(),
    )
}

fn observed_admission_events(observation: &RuntimeObservationHandle) -> Vec<(AdmissionRequestId, AdmissionLifecycleState)> {
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
fn wait_for_sealed_observation(observation: &RuntimeObservationHandle) -> SealedRuntimeObservationSnapshot {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(snapshot) = observation.sealed_snapshot() {
            return snapshot;
        }
        assert!(Instant::now() < deadline, "timed out waiting for observation sealing");
        std::thread::yield_now();
    }
}
fn test_ledger(
    managed_capacity_bytes: u64,
    max_active_attempts: usize,
    device_capabilities: DeviceCapabilities,
) -> DeviceLedger {
    test_ledger_for_ordinal(0, managed_capacity_bytes, max_active_attempts, device_capabilities)
}
fn test_ledger_for_ordinal(
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
fn admission_test_service_with_profile(
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

fn test_service(max_active_attempts_per_device: usize, max_waiters: usize) -> AdmissionTestController {
    let queue_bounds = QueueBounds { max_waiters, max_wait: Duration::from_millis(500) };
    admission_test_service_with_profile(
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
    QueryService::new_with_device_profiles_for_test(config, queue_bounds, device_profiles)
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
        max_active_attempts + RESERVED_LIFECYCLE_WORKERS,
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
