use std::{sync::Arc, time::Duration};

use cudf::{DeviceId, DeviceMemoryManager, DeviceMemoryManagerBase, DeviceMemoryManagerOptions};
use query_engine::{
    error::QueryErrorCode,
    observability::RuntimeObservationHandle,
    runtime::{
        GpuBackendControllerDraft, GpuBackendCore, GpuBackendCoreDraft, GpuBackendDeviceConfig,
        GpuBackendDeviceInstallationBounds, GpuBackendDeviceProfile, GpuExecutionJobKind, GpuMemoryOwnership,
        QueryAdmissionRequest, QueryExecutionConfig, QueryHandle, QueueBounds, S3SourceStorageProfile,
        SourceStorageProfile,
    },
};
use tokio::{
    spawn,
    sync::Barrier,
    time::{Instant, sleep, timeout},
};

fn canonical_options() -> DeviceMemoryManagerOptions {
    let total_bytes = DeviceId::current().unwrap().available_memory().unwrap().total_bytes;
    let managed_capacity = total_bytes
        .checked_sub(500 * 1024 * 1024)
        .filter(|capacity| *capacity > 0)
        .unwrap_or((256 * 1024 * 1024).min(total_bytes));
    DeviceMemoryManagerOptions {
        base: DeviceMemoryManagerBase::CudaAsync {
            initial_pool_size: None,
            release_threshold: Some(usize::try_from(managed_capacity).unwrap()),
        },
        allocation_limit_bytes: None,
    }
}

fn backend_draft(config: QueryExecutionConfig) -> GpuBackendCoreDraft {
    backend_draft_with_limits_and_queue_bounds(config, QueueBounds::default(), 1)
}

fn backend_draft_with_limits_and_queue_bounds(
    config: QueryExecutionConfig,
    queue_bounds: QueueBounds,
    max_active_attempts: usize,
) -> GpuBackendCoreDraft {
    let device_ordinal = DeviceId::current().expect("current CUDA device").ordinal();
    let source_work_per_attempt = 1;
    let stream_pool_size = max_active_attempts.saturating_mul(1 + source_work_per_attempt).max(1);
    let controller = GpuBackendControllerDraft::new(config, queue_bounds);
    let device = GpuBackendDeviceConfig::new(
        GpuBackendDeviceProfile::new(device_ordinal)
            .with_source_work_per_attempt(source_work_per_attempt)
            .with_max_active_attempts(max_active_attempts)
            .with_installation_bounds(GpuBackendDeviceInstallationBounds::new(
                stream_pool_size,
                max_active_attempts,
                max_active_attempts,
            )),
        GpuMemoryOwnership::WholeDeviceExclusive,
    );
    GpuBackendCoreDraft::new(vec![device], controller)
}

#[test]
fn rejected_configs_and_unidentified_allocator_do_not_mutate_the_allocator() {
    let device = DeviceId::current().unwrap();
    assert_eq!(DeviceMemoryManager::options_for_device(device).unwrap(), None);

    let empty_profile_config = QueryExecutionConfig {
        source_storage_profiles: vec![SourceStorageProfile::s3("  ", S3SourceStorageProfile::default_chain())],
        ..QueryExecutionConfig::default()
    };
    let empty_profile = backend_draft(empty_profile_config).validate().unwrap_err();
    assert_eq!(empty_profile.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(empty_profile.fact_value("reason"), Some("empty_profile_id"));

    let profile = SourceStorageProfile::s3("duplicate", S3SourceStorageProfile::default_chain());
    let duplicate_profile_config = QueryExecutionConfig {
        source_storage_profiles: vec![profile.clone(), profile],
        ..QueryExecutionConfig::default()
    };
    let duplicate_profile = backend_draft(duplicate_profile_config).validate().unwrap_err();
    assert_eq!(duplicate_profile.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(duplicate_profile.fact_value("reason"), Some("duplicate_profile_id"));
    assert_eq!(
        DeviceMemoryManager::options_for_device(device).unwrap(),
        None,
        "resource-independent storage-profile validation must precede allocator installation",
    );

    DeviceMemoryManager::install_for_device(device).unwrap();
    let incompatible = GpuBackendCore::builder(backend_draft(QueryExecutionConfig::default())).build().unwrap_err();

    assert_eq!(incompatible.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(incompatible.fact_value("reason"), Some("device_memory_manager_options_conflict"));
    assert_eq!(incompatible.fact_value("installation_status"), Some("Unidentified"));
    assert_eq!(incompatible.fact_value("installation_retry"), Some("PermanentlyUnusable"));
    assert_eq!(incompatible.fact_value("installation_cause"), Some("OptionsMismatch"));
    assert_eq!(incompatible.fact_value("device_ordinal"), Some(device.ordinal().to_string().as_str()));
    assert_eq!(
        DeviceMemoryManager::options_for_device(device).unwrap(),
        Some(DeviceMemoryManagerOptions::default()),
        "an unidentified allocator must not be replaced by backend construction",
    );
}

#[test]
fn noncanonical_pool_allocator_is_rejected_during_backend_preflight() {
    let device = DeviceId::current().unwrap();
    assert_eq!(DeviceMemoryManager::options_for_device(device).unwrap(), None);

    let installed_options = DeviceMemoryManagerOptions {
        base: DeviceMemoryManagerBase::Pool {
            initial_pool_size: 64 * 1024 * 1024,
            maximum_pool_size: Some(64 * 1024 * 1024),
        },
        allocation_limit_bytes: None,
    };
    DeviceMemoryManager::install_for_device_with_options(device, installed_options).unwrap();

    let conflict = GpuBackendCore::builder(backend_draft(QueryExecutionConfig::default())).build().unwrap_err();

    assert_eq!(conflict.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(conflict.fact_value("reason"), Some("device_memory_manager_options_conflict"));
    assert_eq!(conflict.fact_value("installation_cause"), Some("OptionsMismatch"));
    assert_eq!(conflict.fact_value("existing_options"), Some(format!("{installed_options:?}").as_str()));
    assert_eq!(conflict.fact_value("requested_options"), Some(format!("{:?}", canonical_options()).as_str()));
    assert_eq!(conflict.fact_value("device_ordinal"), Some(device.ordinal().to_string().as_str()));
    assert_eq!(
        DeviceMemoryManager::options_for_device(device).unwrap(),
        Some(installed_options),
        "the failed preflight must not replace a process-global allocator before any query can admit",
    );
}

#[test]
fn canonical_allocator_reuse_preserves_authoritative_installation_identity_across_backend_generations() {
    let device = DeviceId::current().unwrap();
    assert_eq!(DeviceMemoryManager::options_for_device(device).unwrap(), None);

    let first = GpuBackendCore::builder(backend_draft(QueryExecutionConfig::default())).build().unwrap();
    let first_identity = first.identity();
    let first_pools = first.session_capability().gpu_execution_pool_reports().unwrap();
    let [first_pool] = first_pools.as_slice() else {
        panic!("expected one worker pool for the single configured device");
    };
    let first_pool_id = first_pool.pool_id;
    let [first_record] = first.device_startup_records() else {
        panic!("expected one startup record for the single configured device");
    };
    let first_record = first_record.clone();
    let first_installation = DeviceMemoryManager::installation_state_for_device(device).unwrap();

    assert_eq!(first_record.memory_manager_installation, first_installation);
    assert_eq!(first_installation.device_ordinal, device.ordinal());
    assert_eq!(first_installation.status, cudf::DeviceMemoryManagerInstallationStatus::Owned);
    assert_eq!(first_installation.retry, cudf::DeviceMemoryManagerInstallationRetry::Reusable);
    assert!(first_installation.identity.is_some(), "an owned reusable allocator must retain its installation identity");
    assert!(
        first_installation.installation_generation.is_some_and(|generation| generation > 0),
        "an owned reusable allocator must retain its authoritative installation generation: {first_installation:?}"
    );
    assert!(first_record.device_health_generation > 0);
    assert!(first_record.backend_health_generation > 0);
    assert!(first_pool_id > 0, "the first backend generation must publish a stable worker-pool identity");

    let close = first.close_and_wait(Duration::from_secs(10)).unwrap();
    assert!(close.gpu_resource_quiescent);
    assert!(close.observation_quiescent);
    drop(first);

    let second = GpuBackendCore::builder(backend_draft(QueryExecutionConfig::default())).build().unwrap();
    let second_identity = second.identity();
    let second_pools = second.session_capability().gpu_execution_pool_reports().unwrap();
    let [second_pool] = second_pools.as_slice() else {
        panic!("expected one worker pool for the single configured device");
    };
    let second_pool_id = second_pool.pool_id;
    let [second_record] = second.device_startup_records() else {
        panic!("expected one startup record for the single configured device");
    };
    let second_record = second_record.clone();
    let second_installation = DeviceMemoryManager::installation_state_for_device(device).unwrap();

    assert!(
        second_identity.resource_generation > first_identity.resource_generation,
        "a newly published backend must receive a later resource generation"
    );
    assert_ne!(second_identity.resource_core, first_identity.resource_core);
    assert_ne!(second_identity.observation_core, first_identity.observation_core);
    assert_ne!(
        second_pool_id, first_pool_id,
        "a separately published backend generation must construct a distinct worker pool"
    );
    assert_eq!(second_installation, first_installation, "compatible backend generations must reuse the same allocator");
    assert_eq!(second_record.memory_manager_installation, second_installation);
    assert_eq!(second_record.memory_manager_installation.identity, first_record.memory_manager_installation.identity);
    assert_eq!(
        second_record.memory_manager_installation.installation_generation,
        first_record.memory_manager_installation.installation_generation,
        "the process allocator generation is stable when an identical backend reuses it"
    );
    assert_eq!(
        second_record.device_health_generation, first_record.device_health_generation,
        "device health remains bound to the reused allocator installation"
    );
    assert!(
        second_record.backend_health_generation > first_record.backend_health_generation,
        "each published backend retains a distinct authoritative backend generation"
    );

    let close = second.close_and_wait(Duration::from_secs(10)).unwrap();
    assert!(close.gpu_resource_quiescent);
    assert!(close.observation_quiescent);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn canonical_default_allocator_admits_a_hard_query_domain() {
    let core = GpuBackendCore::builder(backend_draft_with_limits_and_queue_bounds(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 2, max_wait: Duration::from_secs(10) },
        1,
    ))
    .build()
    .unwrap();
    let session = core.session_capability();
    let handle = session.admit_query().unwrap();
    let context = handle.acquire_context().unwrap();
    assert!(context.context().attempt_allocation_domain_id().is_some());
    drop(context);
    handle.finish();
    drop(handle);
    drop(session);
    let close = core.close_and_wait(Duration::from_secs(10)).unwrap();
    assert!(close.gpu_resource_quiescent);
    assert!(close.observation_quiescent);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn same_device_attempts_have_distinct_grant_domain_stream_lane_and_job_identities() {
    const FIRST_QUERY_CAP_BYTES: u64 = 64 * 1024 * 1024;
    const SECOND_QUERY_CAP_BYTES: u64 = 32 * 1024 * 1024;

    let device = DeviceId::current().unwrap();
    let core = GpuBackendCore::builder(backend_draft_with_limits_and_queue_bounds(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 2, max_wait: Duration::from_secs(10) },
        2,
    ))
    .build()
    .unwrap();
    let session = core.session_capability();

    let [startup] = core.device_startup_records() else {
        panic!("expected one startup record for the single configured device");
    };
    assert_eq!(startup.device_ordinal, device.ordinal());
    assert_eq!(startup.max_active_attempts, 2);
    assert_eq!(startup.lane_count, 2);
    assert_eq!(startup.resource_profile.root_lane_count, 2);
    assert_eq!(startup.resource_profile.root_stream_count, 2);
    assert_eq!(startup.worker_count, 2);

    let first = Arc::new(
        session
            .admit_query_with_request(QueryAdmissionRequest::test_attempt().with_query_cap_bytes(FIRST_QUERY_CAP_BYTES))
            .unwrap(),
    );
    let second = Arc::new(
        session
            .admit_query_with_request(
                QueryAdmissionRequest::test_attempt().with_query_cap_bytes(SECOND_QUERY_CAP_BYTES),
            )
            .unwrap(),
    );
    assert_eq!(first.device_ordinal(), device.ordinal());
    assert_eq!(second.device_ordinal(), device.ordinal());
    assert_ne!(first.admission_request_id(), second.admission_request_id());
    assert_eq!(first.device_budget_bytes(), FIRST_QUERY_CAP_BYTES);
    assert_eq!(second.device_budget_bytes(), SECOND_QUERY_CAP_BYTES);

    let ledger_reports = session.device_ledger_reports().unwrap();
    let [ledger] = ledger_reports.as_slice() else {
        panic!("expected one device ledger for the single configured device");
    };
    assert_eq!(ledger.device_ordinal, device.ordinal());
    assert_eq!(ledger.max_active_attempts, 2);
    assert_eq!(ledger.active_attempts, 2);
    assert_eq!(ledger.live_grants_bytes, FIRST_QUERY_CAP_BYTES + SECOND_QUERY_CAP_BYTES);
    assert_eq!(ledger.grants.len(), 2);
    let grant_id = |request_id| {
        ledger
            .grants
            .iter()
            .find(|grant| grant.admission_request_id == request_id)
            .unwrap_or_else(|| panic!("missing grant for admission request {request_id:?}"))
            .grant_id
    };
    let first_grant_id = grant_id(first.admission_request_id());
    let second_grant_id = grant_id(second.admission_request_id());
    assert_ne!(first_grant_id, second_grant_id, "simultaneous attempts must own distinct immutable grants");
    assert_eq!(
        ledger.grants.iter().find(|grant| grant.grant_id == first_grant_id).expect("first grant row").grant_bytes,
        FIRST_QUERY_CAP_BYTES,
    );
    assert_eq!(
        ledger.grants.iter().find(|grant| grant.grant_id == second_grant_id).expect("second grant row").grant_bytes,
        SECOND_QUERY_CAP_BYTES,
    );

    let first_context = first.acquire_context().unwrap();
    let second_context = second.acquire_context().unwrap();
    let first_domain_id = first_context.context().attempt_allocation_domain_id().expect("first attempt domain");
    let second_domain_id = second_context.context().attempt_allocation_domain_id().expect("second attempt domain");
    assert_eq!(first_domain_id, first_grant_id);
    assert_eq!(second_domain_id, second_grant_id);
    assert_ne!(first_domain_id, second_domain_id, "simultaneous attempts must own distinct allocation domains");

    let first_runtime = first_context.context().runtime().unwrap();
    let second_runtime = second_context.context().runtime().unwrap();
    let first_stream_id = first_runtime.stream().identity();
    let second_stream_id = second_runtime.stream().identity();
    assert_ne!(first_stream_id, second_stream_id, "simultaneous attempts must own distinct root streams");
    drop(first_runtime);
    drop(second_runtime);

    // Publishing both handles proves that both entries in the two-lane array
    // are held simultaneously: admission acquires one exclusive lane before it
    // can create the root stream/domain envelope, and no lane has been released.
    assert!(
        session
            .enqueue_admission_with_request(
                QueryAdmissionRequest::test_attempt().with_query_cap_bytes(FIRST_QUERY_CAP_BYTES)
            )
            .unwrap()
            .try_admit()
            .unwrap()
            .is_none(),
        "two live attempts must exhaust exactly the two configured attempt lanes",
    );

    let entered = Arc::new(Barrier::new(3));
    let release = Arc::new(Barrier::new(3));
    let spawn_job = |handle: Arc<QueryHandle>, observation: RuntimeObservationHandle| {
        let entered = Arc::clone(&entered);
        let release = Arc::clone(&release);
        spawn(async move {
            handle
                .gpu_executor()
                .unwrap()
                .run_observed_scoped(GpuExecutionJobKind::QueryExecution, observation, move |scope| async move {
                    entered.wait().await;
                    release.wait().await;
                    scope
                })
                .await
                .unwrap()
        })
    };
    let first_job = spawn_job(Arc::clone(&first), first_context.context().runtime_observation().clone());
    let second_job = spawn_job(Arc::clone(&second), second_context.context().runtime_observation().clone());
    timeout(Duration::from_secs(10), entered.wait()).await.expect("both same-device GPU jobs must start concurrently");

    let pool_reports = session.gpu_execution_pool_reports().unwrap();
    let [pool] = pool_reports.as_slice() else {
        panic!("expected one worker pool for the single configured device");
    };
    assert_eq!(pool.device_ordinal, device.ordinal());
    assert_eq!(pool.active_query_jobs, 2);
    assert_eq!(pool.active_jobs, 2);

    timeout(Duration::from_secs(10), release.wait())
        .await
        .expect("both same-device GPU jobs must reach the release barrier");
    let first_scope = first_job.await.unwrap();
    let second_scope = second_job.await.unwrap();
    let first_job_id = first_scope.gpu_job_id.expect("first GPU job id");
    let second_job_id = second_scope.gpu_job_id.expect("second GPU job id");
    assert_ne!(first_job_id, second_job_id, "simultaneous attempts must run distinct GPU jobs");
    let first_worker_id = first_scope.gpu_worker_id.expect("first GPU worker id");
    let second_worker_id = second_scope.gpu_worker_id.expect("second GPU worker id");
    assert_ne!(first_worker_id, second_worker_id, "simultaneous GPU jobs must occupy distinct worker lanes");

    drop(first_context);
    drop(second_context);
    first.finish();
    second.finish();
    drop(first);
    drop(second);
    drop(session);
    let close = core.close_and_wait(Duration::from_secs(10)).unwrap();
    assert!(close.gpu_resource_quiescent);
    assert!(close.observation_quiescent);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn derived_default_caps_overlap_two_attempts_without_per_query_tuning() {
    let device = DeviceId::current().unwrap();
    let core = GpuBackendCore::builder(backend_draft_with_limits_and_queue_bounds(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 3, max_wait: Duration::from_secs(10) },
        2,
    ))
    .build()
    .unwrap();
    let session = core.session_capability();

    let [startup] = core.device_startup_records() else {
        panic!("expected one startup record for the single configured device");
    };
    assert_eq!(startup.device_ordinal, device.ordinal());
    assert_eq!(startup.max_active_attempts, 2);
    let query_share = startup.resource_profile.query_share_bytes();
    let derived = startup.resource_profile.default_query_limit_bytes;
    assert_eq!(derived, query_share / 2, "setting only N derives floor(Q / N) end to end");
    assert!(derived > 0, "a parallel service always derives a usable default grant");

    // No per-query caps: both defaults admit from the parallelism target alone.
    let first = Arc::new(session.admit_query().unwrap());
    let second = Arc::new(session.admit_query().unwrap());
    assert_eq!(first.device_ordinal(), device.ordinal());
    assert_eq!(second.device_ordinal(), device.ordinal());
    assert_eq!(first.device_budget_bytes(), derived);
    assert_eq!(second.device_budget_bytes(), derived);

    let ledger_reports = session.device_ledger_reports().unwrap();
    let [ledger] = ledger_reports.as_slice() else {
        panic!("expected one device ledger for the single configured device");
    };
    assert_eq!(ledger.active_attempts, 2);
    assert_eq!(ledger.live_grants_bytes, derived.saturating_mul(2));
    assert!(
        session
            .enqueue_admission_with_request(QueryAdmissionRequest::test_attempt())
            .unwrap()
            .try_admit()
            .unwrap()
            .is_none(),
        "two derived defaults fill the lanes and the query share; a third queues"
    );

    let first_context = first.acquire_context().unwrap();
    let second_context = second.acquire_context().unwrap();
    let entered = Arc::new(Barrier::new(3));
    let release = Arc::new(Barrier::new(3));
    let spawn_job = |handle: Arc<QueryHandle>, observation: RuntimeObservationHandle| {
        let entered = Arc::clone(&entered);
        let release = Arc::clone(&release);
        spawn(async move {
            handle
                .gpu_executor()
                .unwrap()
                .run_observed_scoped(GpuExecutionJobKind::QueryExecution, observation, move |scope| async move {
                    entered.wait().await;
                    release.wait().await;
                    scope
                })
                .await
                .unwrap()
        })
    };
    let first_job = spawn_job(Arc::clone(&first), first_context.context().runtime_observation().clone());
    let second_job = spawn_job(Arc::clone(&second), second_context.context().runtime_observation().clone());
    timeout(Duration::from_secs(10), entered.wait())
        .await
        .expect("both default-cap GPU jobs must overlap without per-query tuning");

    let pool_reports = session.gpu_execution_pool_reports().unwrap();
    let [pool] = pool_reports.as_slice() else {
        panic!("expected one worker pool for the single configured device");
    };
    assert_eq!(pool.active_query_jobs, 2);

    timeout(Duration::from_secs(10), release.wait())
        .await
        .expect("both default-cap GPU jobs must reach the release barrier");
    first_job.await.unwrap();
    second_job.await.unwrap();

    drop(first_context);
    drop(second_context);
    first.finish();
    second.finish();
    drop(first);
    drop(second);
    drop(session);
    let close = core.close_and_wait(Duration::from_secs(10)).unwrap();
    assert!(close.gpu_resource_quiescent);
    assert!(close.observation_quiescent);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn backend_clone_sessions_share_identity_ledger_worker_pools_and_startup_records() {
    let device = DeviceId::current().unwrap();
    let core = GpuBackendCore::builder(backend_draft_with_limits_and_queue_bounds(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 2, max_wait: Duration::from_secs(10) },
        1,
    ))
    .build()
    .unwrap();
    let cloned_core = core.clone();
    let first_session = core.session_capability();
    let second_session = cloned_core.session_capability();
    let identity = core.identity();

    assert_eq!(cloned_core.identity(), identity);
    assert_eq!(first_session.identity(), identity);
    assert_eq!(second_session.identity(), identity);
    assert!(first_session.same_backend(&second_session));
    assert!(identity.resource_core > 0);
    assert!(identity.observation_core > 0);
    assert!(identity.resource_generation > 0);
    assert_eq!(core.device_startup_records(), cloned_core.device_startup_records());

    let first_device_records = first_session.device_admission_snapshots().unwrap();
    let second_device_records = second_session.device_admission_snapshots().unwrap();
    assert_eq!(first_device_records, second_device_records);
    assert_eq!(first_device_records.len(), 1);
    assert_eq!(first_device_records[0].device_ordinal, device.ordinal());
    assert_eq!(first_device_records[0].process_active_query_slots, 0);

    let first_worker_pools = first_session.gpu_execution_pool_reports().unwrap();
    let second_worker_pools = second_session.gpu_execution_pool_reports().unwrap();
    assert_eq!(first_worker_pools, second_worker_pools);
    assert_eq!(first_worker_pools.len(), 1);
    assert!(first_worker_pools[0].pool_id > 0);
    assert_eq!(first_worker_pools[0].pool_id, second_worker_pools[0].pool_id);
    assert_eq!(first_worker_pools[0].device_ordinal, device.ordinal());
    assert_eq!(first_worker_pools[0].worker_count, 1);

    let first = first_session.admit_query().unwrap();
    let first_sequence = first.admission_sequence();
    let queued_session = second_session.clone();
    let queued =
        spawn(async move { queued_session.admit_async_with_request(QueryAdmissionRequest::test_attempt()).await });

    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let first_snapshot = first_session.admission_snapshot().unwrap();
        let second_snapshot = second_session.admission_snapshot().unwrap();
        if first_snapshot.active_queries == 1 && first_snapshot.queued_queries == 1 {
            assert_eq!(second_snapshot, first_snapshot, "cloned sessions must observe one shared admission ledger");
            break;
        }
        assert!(
            Instant::now() < deadline,
            "second cloned session did not queue behind the first session's admission: first={first_snapshot:?}, second={second_snapshot:?}"
        );
        sleep(Duration::from_millis(25)).await;
    }

    first.finish();
    drop(first);
    let queued = timeout(Duration::from_secs(10), queued)
        .await
        .expect("queued cloned-session admission must wake after the shared slot is released")
        .expect("queued cloned-session task must not panic")
        .expect("queued cloned-session admission must succeed");
    assert!(queued.admission_sequence() > first_sequence);
    queued.finish();
    drop(queued);

    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let snapshot = second_session.admission_snapshot().unwrap();
        if snapshot.active_queries == 0 && snapshot.queued_queries == 0 {
            break;
        }
        assert!(Instant::now() < deadline, "shared admission ledger did not drain: {snapshot:?}");
        sleep(Duration::from_millis(25)).await;
    }

    drop(first_session);
    drop(second_session);
    let close = core.close_and_wait(Duration::from_secs(10)).unwrap();
    assert!(close.gpu_resource_quiescent);
    assert!(close.observation_quiescent);
    drop(cloned_core);
}
