#![cfg(feature = "test-hooks")]

use std::{
    fs,
    path::PathBuf,
    sync::{Arc, Barrier, OnceLock, atomic::AtomicU64},
    thread,
    time::Duration,
};

use cudf::{CudfErrorFailureSource, DataFrame, DeviceId};
use cudf_sys::test_support;
use query_runtime::runtime::{
    GpuBackendDeviceConfig, GpuBackendDeviceInstallationBounds, GpuBackendDeviceProfile, GpuDeviceMemoryLimit,
    GpuMemoryOwnership, QueryAdmissionRequest, QueryExecutionConfig, QueryService, QueueBounds,
};

fn unique_fixture_path() -> PathBuf {
    static NEXT_ID: OnceLock<AtomicU64> = OnceLock::new();
    let id = NEXT_ID.get_or_init(|| AtomicU64::new(0)).fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!("algeon_stream_provider_parquet_{}_{}.parquet", std::process::id(), id))
}

fn test_service() -> QueryService {
    QueryService::new_with_device_profiles_for_test(
        QueryExecutionConfig::default(),
        QueueBounds { max_waiters: 4, max_wait: Duration::from_secs(5) },
        vec![GpuBackendDeviceConfig::new(
            GpuBackendDeviceProfile::new(0)
                .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 64 * 1024 * 1024 })
                .with_default_query_limit_bytes(4 * 1024 * 1024)
                .with_source_work_per_attempt(2)
                .with_max_active_attempts(2)
                .with_installation_bounds(GpuBackendDeviceInstallationBounds::new(6, 3, 2)),
            GpuMemoryOwnership::WholeDeviceExclusive,
        )],
    )
    .expect("two-attempt runtime service")
}

#[test]
fn same_device_attempt_routes_overlap_parquet_reads_and_drain_after_failure() {
    if DeviceId::count().unwrap_or(0) == 0 {
        eprintln!("skipping attempt stream-provider parquet test; no CUDA device is visible");
        return;
    }

    let path = unique_fixture_path();
    let service = test_service();
    let first = service.admit_query_with_request(QueryAdmissionRequest::test_attempt()).unwrap();
    let second = service.admit_query_with_request(QueryAdmissionRequest::test_attempt()).unwrap();
    assert_eq!(first.device_ordinal(), 0);
    assert_eq!(second.device_ordinal(), 0);
    assert_ne!(first.grant_identity().unwrap(), second.grant_identity().unwrap());
    let second_grant_identity = second.grant_identity().unwrap();

    let first_context = first.acquire_context().unwrap();
    let second_context = second.acquire_context().unwrap();
    let mut second_domain_id = 0;
    for (handle, context) in [(&first, &first_context), (&second, &second_context)] {
        let usage = context.context().attempt_allocation_domain_usage().unwrap().unwrap();
        assert_eq!(usage.ceiling_bytes, handle.device_budget_bytes());
        assert_eq!(usage.allocation_limit_bytes, handle.device_budget_bytes());
        assert_eq!(usage.active_members, 4, "root plus two auxiliary forks and the reserved root fork");
        if handle.grant_identity().unwrap() == second_grant_identity {
            second_domain_id = usage.domain_id;
        }
    }
    assert_ne!(second_domain_id, 0);

    let fixture_runtime = first_context.context().runtime().unwrap();
    let values = (0..262_144).collect::<Vec<i32>>();
    let expected_rows = values.len();
    let frame = DataFrame::from_series([("id", fixture_runtime.series(&values).unwrap())]).unwrap();
    frame.write_parquet(&path).unwrap();
    drop(frame);
    fixture_runtime.synchronize().unwrap();

    let barrier = Arc::new(Barrier::new(2));
    thread::scope(|scope| {
        let first_barrier = Arc::clone(&barrier);
        let first_path = path.clone();
        scope.spawn(move || {
            first_barrier.wait();
            let runtime = first_context.context().runtime().unwrap();
            let read = DataFrame::read_parquet(&runtime, &first_path).unwrap();
            assert_eq!(read.num_rows(), expected_rows);
            let usage = first_context.context().attempt_allocation_domain_usage().unwrap().unwrap();
            assert!(usage.current_bytes <= usage.allocation_limit_bytes);
        });

        let second_barrier = Arc::clone(&barrier);
        let second_path = path.clone();
        scope.spawn(move || {
            second_barrier.wait();
            let runtime = second_context.context().runtime().unwrap();
            unsafe { test_support::cudf_test_query_domain_fail_allocation_after(second_domain_id, 4) };
            let error = DataFrame::read_parquet(&runtime, &second_path)
                .expect_err("injected parquet allocation failure must reach the caller");
            unsafe { test_support::cudf_test_query_domain_reset_allocation_failure() };
            assert_eq!(
                error.ffi_detail().expect("parquet failure keeps native attribution").failure_source,
                CudfErrorFailureSource::Physical
            );
            runtime.synchronize().unwrap();
            let usage = second_context.context().attempt_allocation_domain_usage().unwrap().unwrap();
            assert_eq!(usage.current_allocations, 0);
            assert!(usage.current_bytes <= usage.allocation_limit_bytes);

            let read = DataFrame::read_parquet(&runtime, &second_path).unwrap();
            assert_eq!(read.num_rows(), expected_rows);
        });
    });

    first.end_test_attempt();
    second.end_test_attempt();
    drop(fixture_runtime);
    drop(first);
    drop(second);
    service.close_and_wait_for_quiescence(Duration::from_secs(10)).unwrap();
    assert_eq!(service.admission_snapshot().unwrap().active_queries, 0);
    let reports = service.device_ledger_reports().unwrap();
    assert_eq!(reports.len(), 1);
    assert_eq!(reports[0].active_attempts, 0);
    assert_eq!(reports[0].live_grants_bytes, 0);
    fs::remove_file(path).unwrap();
}
