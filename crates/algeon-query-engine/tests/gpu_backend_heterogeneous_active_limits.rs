use std::{
    sync::{Mutex, MutexGuard, OnceLock},
    time::Duration,
};

use query_engine::runtime::{
    DeploymentIoCapability, GdsDriverProbeStatus, GpuBackendControllerDraft, GpuBackendCore, GpuBackendCoreDraft,
    GpuBackendDeviceConfig, GpuBackendDeviceInstallationBounds, GpuBackendDeviceProfile, GpuDeviceMemoryLimit,
    GpuMemoryOwnership, KvikioCompatModeStatus, NvidiaFsProbeStatus, QueryAdmissionRequest, QueryExecutionConfig,
    QueueBounds,
};

const DEVICE_CAPACITY_BYTES: u64 = 64 * 1024 * 1024;

fn live_backend_test_guard() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(())).lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn non_gds_deployment() -> DeploymentIoCapability {
    DeploymentIoCapability::from_probe_inputs(
        NvidiaFsProbeStatus::DriverMissing,
        KvikioCompatModeStatus::Unset,
        GdsDriverProbeStatus::ProbeUnimplemented,
    )
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn heterogeneous_device_active_limits_are_not_collapsed_to_the_largest_cap() {
    let device_count = cudf::DeviceId::count().expect("CUDA device enumeration for heterogeneous backend coverage");
    assert!(device_count >= 2, "heterogeneous backend coverage requires at least two CUDA devices");
    let _guard = live_backend_test_guard();

    let controller = GpuBackendControllerDraft::new(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 8, max_wait: Duration::from_millis(250) },
    );
    // Each device carries an explicit per-device cap here so the two devices
    // can use different parallelism targets in one backend. Without an
    // explicit cap the install derives `floor((M - K) / N)` from the
    // parallelism target alone, which already leaves room for
    // `max_active_attempts` simultaneous grants.
    let device = |device_ordinal, max_active_attempts: usize| {
        GpuBackendDeviceConfig::new(
            GpuBackendDeviceProfile::new(device_ordinal)
                .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: DEVICE_CAPACITY_BYTES })
                .with_default_query_limit_bytes(DEVICE_CAPACITY_BYTES / max_active_attempts as u64)
                .with_max_active_attempts(max_active_attempts)
                .with_installation_bounds(GpuBackendDeviceInstallationBounds::new(
                    max_active_attempts.saturating_mul(3),
                    max_active_attempts,
                    max_active_attempts,
                )),
            GpuMemoryOwnership::WholeDeviceExclusive,
        )
    };
    let core = GpuBackendCore::builder(GpuBackendCoreDraft::new(
        vec![device(0, 2), device(1, 1)],
        controller,
        non_gds_deployment(),
    ))
    .build()
    .unwrap();
    let session = core.session_capability();

    let first =
        session.try_admit_query_with_request(QueryAdmissionRequest::new()).unwrap().expect("first query admits");
    let second =
        session.try_admit_query_with_request(QueryAdmissionRequest::new()).unwrap().expect("second query admits");
    let third =
        session.try_admit_query_with_request(QueryAdmissionRequest::new()).unwrap().expect("third query admits");
    // Least-loaded placement decides the visiting order; the contract here is
    // how many attempts each device ends up holding, not which one is picked
    // first while both are idle.
    let mut placed = [0usize; 2];
    for handle in [&first, &second, &third] {
        placed[handle.device_ordinal()] += 1;
    }
    assert_eq!(placed, [2, 1], "device 0 holds its cap of two and device 1 holds its cap of one");
    assert!(
        session.try_admit_query_with_request(QueryAdmissionRequest::new()).unwrap().is_none(),
        "the device-1 cap remains one instead of inheriting device 0's cap of two",
    );

    let snapshots = session.device_admission_snapshots().unwrap();
    assert_eq!(snapshots.iter().map(|snapshot| snapshot.process_active_query_slots).collect::<Vec<_>>(), vec![2, 1]);

    first.finish();
    second.finish();
    third.finish();
    core.close_and_wait(Duration::from_secs(5)).unwrap();
}

#[test]
#[ignore = "requires at least 2 GPUs"]
fn device_startup_records_carry_ordinal_local_resource_profiles() {
    let device_count = cudf::DeviceId::count().expect("CUDA device enumeration for per-device startup records");
    assert!(device_count >= 2, "per-device startup record coverage requires at least two CUDA devices");
    let _guard = live_backend_test_guard();

    const DEVICE_ONE_LIMIT_BYTES: u64 = 2 * DEVICE_CAPACITY_BYTES;
    let controller = GpuBackendControllerDraft::new(QueryExecutionConfig::default(), QueueBounds::default());
    let device = |device_ordinal, limit_bytes, max_active_attempts| {
        GpuBackendDeviceConfig::new(
            GpuBackendDeviceProfile::new(device_ordinal)
                .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes })
                .with_max_active_attempts(max_active_attempts)
                .with_installation_bounds(GpuBackendDeviceInstallationBounds::new(
                    max_active_attempts.saturating_mul(3),
                    max_active_attempts,
                    max_active_attempts,
                )),
            GpuMemoryOwnership::WholeDeviceExclusive,
        )
    };
    let core = GpuBackendCore::builder(GpuBackendCoreDraft::new(
        vec![device(0, DEVICE_CAPACITY_BYTES, 2), device(1, DEVICE_ONE_LIMIT_BYTES, 1)],
        controller,
        non_gds_deployment(),
    ))
    .build()
    .unwrap();

    let records = core.device_startup_records();
    assert_eq!(records.len(), 2);
    let first = &records[0].resource_profile;
    let second = &records[1].resource_profile;
    assert_eq!(first.device_ordinal, 0);
    assert_eq!(second.device_ordinal, 1);
    assert_eq!(first.managed_capacity_bytes, DEVICE_CAPACITY_BYTES);
    assert_eq!(second.managed_capacity_bytes, DEVICE_ONE_LIMIT_BYTES);
    assert_eq!(first.max_active_attempts, 2);
    assert_eq!(second.max_active_attempts, 1);
    assert_eq!(first.root_lane_count, 2);
    assert_eq!(second.root_lane_count, 1);
    assert_eq!(first.stream_pool_size, 2 * 3);
    assert_eq!(second.stream_pool_size, 3);
    assert_eq!(first.worker_job_capacity, 2 * 3);
    assert_eq!(second.worker_job_capacity, 3);
    // Default query limits are ordinal-local: each resolves to its own
    // managed capacity rather than inheriting a value from device 0.
    assert_eq!(first.default_query_limit_bytes, DEVICE_CAPACITY_BYTES);
    assert_eq!(second.default_query_limit_bytes, DEVICE_ONE_LIMIT_BYTES);
    assert_eq!(first.cache_cap_bytes, 0);
    assert_eq!(second.cache_cap_bytes, 0);

    core.close_and_wait(Duration::from_secs(5)).unwrap();
}
