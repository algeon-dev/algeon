use std::{
    sync::{Arc, Barrier},
    time::{Duration, Instant},
};

use cudf::{DeviceId, DeviceMemoryManager, ErrorKind};
use nexus_query_engine::{
    attempt::{AttemptBegin, AttemptCompletion, AttemptOwner, BeginAttemptRequest},
    error::{QueryError, QueryErrorCode},
    observability::{
        AttemptAdmissionFact, AttemptAllocatorEventSource, AttemptCapacityDisposition, AttemptCleanupDisposition,
        AttemptEvidenceProvenance, AttemptHealthTransition, QueryAttemptReport,
    },
    runtime::{
        AdmittedQueryContext, DeploymentIoCapability, GdsDriverProbeStatus, GpuBackendControllerDraft, GpuBackendCore,
        GpuBackendCoreDraft, GpuBackendDeviceConfig, GpuDeviceMemoryLimit, GpuExecutionJobKind, GpuMemoryOwnership,
        KvikioCompatModeStatus, NvidiaFsProbeStatus, QueryAdmissionRequest, QueryExecutionConfig, QueryHandle,
        QueueBounds,
    },
};

const STATIC_PARTITION_BYTES: u64 = 1024 * 1024 * 1024;
const ATTEMPT_FLOOR_BYTES: u64 = 384 * 1024 * 1024;
const SECOND_REQUEST_DELTA_BYTES: usize = 2 * 1024 * 1024;
const HEALTHY_REUSE_REQUEST_BYTES: usize = 1024 * 1024;

struct AdmittedAttempt {
    owner: AttemptOwner,
    completion: AttemptCompletion,
    handle: QueryHandle,
    context: AdmittedQueryContext,
}

fn non_gds_deployment() -> DeploymentIoCapability {
    DeploymentIoCapability::from_probe_inputs(
        NvidiaFsProbeStatus::DriverMissing,
        KvikioCompatModeStatus::Unset,
        GdsDriverProbeStatus::ProbeUnimplemented,
    )
}

fn test_backend(device: DeviceId) -> GpuBackendCore {
    let controller = GpuBackendControllerDraft::new(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 2, max_wait: Duration::from_secs(10), max_overtakes_per_waiter: 1 },
    );
    let device = GpuBackendDeviceConfig::new(
        device.ordinal(),
        GpuMemoryOwnership::StaticPartition { allocation_limit_bytes: STATIC_PARTITION_BYTES },
    )
    .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: STATIC_PARTITION_BYTES })
    .with_attempt_floor_bytes(ATTEMPT_FLOOR_BYTES)
    .with_source_work_per_attempt(1)
    .with_max_active_queries(2)
    .with_installation_bounds(4, 2, 2);
    GpuBackendCore::builder(GpuBackendCoreDraft::new(vec![device], controller, non_gds_deployment()))
        .build()
        .expect("two-attempt physical allocator-correlation backend builds")
}

fn admit_attempt(core: &GpuBackendCore) -> AdmittedAttempt {
    let AttemptBegin { owner, completion } = core
        .begin_attempt(BeginAttemptRequest::default())
        .expect("backend begins a physical allocator-correlation attempt");
    let attempt = owner.handle();
    attempt.record_planning_started().expect("planning starts");
    attempt.record_planning_finished().expect("planning finishes");
    attempt.set_disposition(nexus_query_engine::observability::AttemptDisposition::Native).expect("native disposition");
    attempt.record_admission_requested().expect("admission request records");

    let handle = core
        .session_capability()
        .admit_query_with_request(QueryAdmissionRequest::new().with_attempt_binding(owner.admission_binding()))
        .expect("same-device attempt admits");
    attempt
        .record_admitted(AttemptAdmissionFact {
            admission_sequence: handle.admission_sequence(),
            admission_wait_ns: handle.admission_wait().as_nanos().try_into().unwrap_or(u64::MAX),
            device_ordinal: handle.device_ordinal(),
            grant_bytes: handle.device_budget_bytes(),
        })
        .expect("admission fact records");
    attempt.record_physical_execution_started().expect("physical execution starts");
    attempt.record_gpu_execution_started().expect("GPU execution starts");
    let context = handle.acquire_context().expect("admitted context binds its immutable domain");

    AdmittedAttempt { owner, completion, handle, context }
}

fn commit_allocation_failure(owner: &AttemptOwner, error: cudf::Error) {
    assert_eq!(
        error.kind(),
        ErrorKind::ResourceExhausted,
        "physical pressure must exhaust the native allocator: {error:?}"
    );
    let error = QueryError::from_cudf(
        "nexus_query_engine.gpu_allocator_event_correlation",
        "real framebuffer pressure rejected the query-domain allocation",
        error,
    );
    assert_eq!(error.code(), QueryErrorCode::CudfResourceExhausted);
    owner.commit_query_error(&error).expect("physical allocator failure terminalizes the attempt");
}

fn assert_exact_physical_event(report: &QueryAttemptReport, requested_bytes: usize) -> u64 {
    assert_eq!(report.cleanup_disposition, AttemptCleanupDisposition::Completed, "physical failure cleanup completes");
    assert_eq!(report.cleanup_failure_count, 0, "physical failure has no synthetic cleanup failure");
    assert!(report.cleanup_failure_count_complete, "cleanup accounting is exact");
    assert_eq!(
        report.capacity_disposition,
        AttemptCapacityDisposition::Released,
        "physical OOM alone preserves healthy reuse"
    );
    assert_eq!(report.health_transition, Some(AttemptHealthTransition::HealthyReuse));

    let evidence = report.resource_evidence.as_ref().expect("admitted physical failure retains resource evidence");
    assert_eq!(
        evidence.allocator_event.provenance,
        AttemptEvidenceProvenance::AttemptExact,
        "allocator callback is joined to this attempt domain rather than retained as a device sample",
    );
    let event = evidence.allocator_event.value.expect("exact allocator callback record is retained");
    assert_eq!(event.source, AttemptAllocatorEventSource::Physical, "only genuine physical exhaustion qualifies");
    assert_eq!(
        event.requested_bytes,
        u64::try_from(requested_bytes).expect("test allocation request fits the callback ABI"),
        "this report must retain its own request size rather than the other live domain's size",
    );
    event.sequence
}

fn wait_for_device_release(core: &GpuBackendCore, device: DeviceId) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let reports = core.session_capability().device_ledger_reports().expect("device ledger remains observable");
        let report = reports
            .iter()
            .find(|report| report.device_ordinal == device.ordinal())
            .expect("configured device ledger remains present");
        if report.active_attempts == 0 && report.grants.is_empty() {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "all grants must release before the physical allocator-correlation timeout: {report:?}"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn domain_cap_bytes(context: &AdmittedQueryContext) -> u64 {
    let usage = context
        .context()
        .attempt_allocation_domain_usage()
        .expect("allocation-domain usage samples")
        .expect("admitted context has one allocation domain");
    assert_eq!(usage.allocation_limit_bytes, usage.ceiling_bytes, "domain limit equals its immutable ceiling");
    usage.ceiling_bytes
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn overlapping_query_domains_keep_real_physical_allocator_events_exact_and_reusable() {
    let device = DeviceId::current().expect("current CUDA device is available");
    let core = test_backend(device);
    let manager_cap_bytes = DeviceMemoryManager::options_for_device(device)
        .expect("service installs the static-partition manager")
        .and_then(|options| options.allocation_limit_bytes)
        .expect("installed manager retains the static-partition limit");
    let manager_cap_bytes = u64::try_from(manager_cap_bytes).expect("manager cap fits the engine accounting type");
    assert_eq!(manager_cap_bytes, STATIC_PARTITION_BYTES);
    let first = admit_attempt(&core);
    let second = admit_attempt(&core);
    assert_eq!(first.handle.device_ordinal(), device.ordinal());
    assert_eq!(second.handle.device_ordinal(), device.ordinal());
    assert_ne!(
        first.context.context().attempt_allocation_domain_id(),
        second.context.context().attempt_allocation_domain_id(),
        "overlapping attempts must own distinct native allocation domains",
    );
    assert_ne!(
        first.handle.device_budget_bytes(),
        second.handle.device_budget_bytes(),
        "overlapping domains retain independently computed immutable grants",
    );
    let first_domain_cap_bytes = domain_cap_bytes(&first.context);
    let second_domain_cap_bytes = domain_cap_bytes(&second.context);
    assert_eq!(first_domain_cap_bytes, first.handle.device_budget_bytes());
    assert_eq!(second_domain_cap_bytes, second.handle.device_budget_bytes());

    let pressure = cudf::testing::memory::PhysicalMemoryPressureLease::establish(device)
        .expect("real device pressure is established without an injected allocator failure");
    assert_eq!(pressure.device(), device, "pressure lease remains on the attempt device");
    let pressure_facts = pressure.facts();
    assert!(
        pressure_facts.free_under_pressure_bytes < pressure_facts.free_before_bytes && pressure_facts.bytes_held > 0,
        "sidecar allocation must measurably reduce actual CUDA free memory",
    );
    let first_request_bytes = pressure_facts
        .request_bytes_above_pressure()
        .expect("pressure facts derive a checked request above measured free memory");
    let second_request_bytes = first_request_bytes
        .checked_add(SECOND_REQUEST_DELTA_BYTES)
        .expect("second physical request stays representable");
    assert_ne!(first_request_bytes, second_request_bytes);
    let first_request_u64 = u64::try_from(first_request_bytes).expect("first request fits u64");
    let second_request_u64 = u64::try_from(second_request_bytes).expect("second request fits u64");
    assert!(
        first_request_u64 < first.handle.device_budget_bytes()
            && first_request_u64 < first_domain_cap_bytes
            && first_request_u64 < manager_cap_bytes,
        "first pressure request stays below its grant, domain, and manager caps so only physical exhaustion can reject it",
    );
    assert!(
        second_request_u64 < second.handle.device_budget_bytes()
            && second_request_u64 < second_domain_cap_bytes
            && second_request_u64 < manager_cap_bytes,
        "second pressure request stays below its grant, domain, and manager caps so only physical exhaustion can reject it",
    );
    assert!(
        first_request_u64.checked_add(second_request_u64).expect("request sum fits u64") < manager_cap_bytes,
        "the concurrent requests fit collectively below the manager cap before physical pressure is applied",
    );

    let first_runtime = first.context.context().runtime().expect("first attempt runtime");
    let second_runtime = second.context.context().runtime().expect("second attempt runtime");
    let first_executor = first.handle.gpu_executor().expect("first GPU executor");
    let second_executor = second.handle.gpu_executor().expect("second GPU executor");
    let barrier = Arc::new(Barrier::new(2));
    let first_barrier = Arc::clone(&barrier);
    let second_barrier = Arc::clone(&barrier);

    let first_allocation = first_executor.run(GpuExecutionJobKind::QueryExecution, async move {
        first_barrier.wait();
        first_runtime.device_buffer(first_request_bytes).map(drop)
    });
    let second_allocation = second_executor.run(GpuExecutionJobKind::QueryExecution, async move {
        second_barrier.wait();
        second_runtime.device_buffer(second_request_bytes).map(drop)
    });
    let (first_allocation, second_allocation) = tokio::join!(first_allocation, second_allocation);
    let first_error = first_allocation
        .expect("first GPU allocation job completes")
        .expect_err("first request exceeds real free memory");
    let second_error = second_allocation
        .expect("second GPU allocation job completes")
        .expect_err("second request exceeds real free memory");

    commit_allocation_failure(&first.owner, first_error);
    commit_allocation_failure(&second.owner, second_error);
    drop(first_executor);
    drop(second_executor);
    drop(first.context);
    drop(second.context);
    first.handle.finish();
    second.handle.finish();
    drop(first.handle);
    drop(second.handle);

    let first_report =
        first.completion.wait_timeout(Duration::from_secs(10)).expect("first physical failure report publishes");
    let second_report =
        second.completion.wait_timeout(Duration::from_secs(10)).expect("second physical failure report publishes");
    let first_sequence = assert_exact_physical_event(&first_report, first_request_bytes);
    let second_sequence = assert_exact_physical_event(&second_report, second_request_bytes);
    assert_ne!(first_sequence, second_sequence, "overlapping callback failures retain unique ring sequences");
    wait_for_device_release(&core, device);

    let release = pressure.release_and_verify().expect("real pressure releases before healthy reuse");
    assert_eq!(release.device, device, "release facts remain device-specific");
    assert!(
        release.recovered_bytes.expect("release records measured CUDA free-memory recovery") > 0,
        "released pressure restores physical capacity for a later query",
    );

    let reuse = core
        .session_capability()
        .admit_query()
        .expect("healthy device admits another query after physical pressure release");
    let reuse_context = reuse.acquire_context().expect("healthy reuse acquires a fresh domain");
    let reuse_buffer = reuse_context
        .context()
        .runtime()
        .expect("healthy reuse runtime")
        .device_buffer(HEALTHY_REUSE_REQUEST_BYTES)
        .expect("healthy reuse allocates after pressure release");
    drop(reuse_buffer);
    drop(reuse_context);
    reuse.finish();
    drop(reuse);
    wait_for_device_release(&core, device);

    let close =
        core.close_and_wait(Duration::from_secs(10)).expect("backend closes after pressure release and healthy reuse");
    assert!(close.gpu_resource_quiescent);
    assert!(close.observation_quiescent);
}
