mod common;

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use cudf::{
    DeviceMemoryManager, ErrorKind, ExecutionBudgetRequest, ExecutionBudgetStatus, Runtime, SortOptions,
    StreamReservationLease, testing::prelude::*,
};

fn assert_send_sync<T: Send + Sync>() {}

fn set_current_device_for_runtime(rt: &Runtime) {
    let status = unsafe { cudf_sys::cudf_cuda_set_device(rt.device().ordinal() as i32) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
}

fn current_device_resource_raw_value() -> usize {
    let mut out = std::ptr::null_mut();
    let status = unsafe { cudf_sys::cudf_test_current_device_resource_raw(&mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS);
    out as usize
}

#[test]
fn default_runtime_uses_device_level_memory_manager_resource() {
    let rt = Runtime::builder().build().unwrap();
    let manager = rt.device_memory_manager().unwrap();

    assert_eq!(
        rt.memory_resource().testing_raw_handle_value(),
        manager.resource().testing_raw_handle_value(),
        "plain runtimes must use the per-device manager so normal execution does not create per-runtime current-MR switching pressure",
    );
}

#[test]
fn device_memory_manager_runtime_uses_one_device_level_resource_with_stream_reservations() {
    assert_send_sync::<DeviceMemoryManager>();
    assert_send_sync::<StreamReservationLease>();

    let rt = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = rt.device_memory_manager().unwrap();
    let manager_mr = manager.resource().testing_raw_handle_value();

    assert_eq!(rt.memory_resource().testing_raw_handle_value(), manager_mr);

    let lease = rt.stream_reservation(42, 64 * 1024 * 1024).unwrap();
    assert_eq!(lease.stream().testing_raw_handle_value(), rt.stream().testing_raw_handle_value());
    assert_eq!(lease.memory_resource().testing_raw_handle_value(), manager_mr);

    let usage = lease.usage().unwrap();
    assert_eq!(usage.reservation_id, 42);
    assert!(usage.attached);
    assert_eq!(usage.allocation_limit_bytes, 64 * 1024 * 1024);
}

#[test]
fn query_execution_context_reserves_stream_against_device_manager_resource() {
    let rt = Runtime::builder().with_device_memory_manager().build().unwrap();
    let query = rt.query_execution();
    let manager_mr = query.device_memory_manager().unwrap().resource().testing_raw_handle_value();

    let lease = query.reserve_stream(77, 32 * 1024 * 1024).unwrap();

    assert_eq!(lease.stream().testing_raw_handle_value(), query.stream().testing_raw_handle_value());
    assert_eq!(lease.memory_resource().testing_raw_handle_value(), manager_mr);

    let usage = lease.usage().unwrap();
    assert_eq!(usage.reservation_id, 77);
    assert!(usage.attached);
    assert_eq!(usage.allocation_limit_bytes, 32 * 1024 * 1024);
}

#[test]
fn concurrent_stream_reservations_share_device_resource_without_counter_crosstalk() {
    let left = Runtime::builder().with_device_memory_manager().build().unwrap();
    let right = Runtime::builder().with_device_memory_manager().build().unwrap();
    let device_mr = left.memory_resource().testing_raw_handle_value();

    assert_eq!(right.memory_resource().testing_raw_handle_value(), device_mr);
    assert_ne!(left.stream().testing_raw_handle_value(), right.stream().testing_raw_handle_value());

    let left_handle = std::thread::spawn(move || {
        let token = left
            .execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("left_query"))
            .unwrap();
        let outcome = left.series_from_value_with_budget(11i32, 1024, &token).unwrap();
        let usage = token.reservation_usage().unwrap();
        assert_eq!(outcome.value().to_vec_strict::<i32>().unwrap()[0], 11);
        assert_eq!(outcome.report().status(), ExecutionBudgetStatus::ReservationEnforced);
        (
            outcome.value().device_allocated_size_bytes().unwrap(),
            usage.peak_bytes as usize,
            usage.current_bytes as usize,
            outcome.value().as_column().testing_execution_domain_mr_value(),
        )
    });

    let right_handle = std::thread::spawn(move || {
        let token = right
            .execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("right_query"))
            .unwrap();
        let outcome = right.series_from_value_with_budget(29i32, 2048, &token).unwrap();
        let usage = token.reservation_usage().unwrap();
        assert_eq!(outcome.value().to_vec_strict::<i32>().unwrap()[0], 29);
        assert_eq!(outcome.report().status(), ExecutionBudgetStatus::ReservationEnforced);
        (
            outcome.value().device_allocated_size_bytes().unwrap(),
            usage.peak_bytes as usize,
            usage.current_bytes as usize,
            outcome.value().as_column().testing_execution_domain_mr_value(),
        )
    });

    let left_usage = left_handle.join().unwrap();
    let right_usage = right_handle.join().unwrap();

    assert_eq!(left_usage.3, device_mr);
    assert_eq!(right_usage.3, device_mr);
    assert!(left_usage.1 >= left_usage.0);
    assert!(right_usage.1 >= right_usage.0);
    assert!(left_usage.2 >= left_usage.0);
    assert!(right_usage.2 >= right_usage.0);
}

#[test]
fn concurrent_reserved_query_work_leaves_current_resource_on_device_adaptor() {
    let left = Runtime::builder().with_device_memory_manager().build().unwrap();
    let right = Runtime::builder().with_device_memory_manager().build().unwrap();
    let device_mr = left.device_memory_manager().unwrap().resource().testing_raw_handle_value();

    assert_eq!(left.memory_resource().testing_raw_handle_value(), device_mr);
    assert_eq!(right.memory_resource().testing_raw_handle_value(), device_mr);
    set_current_device_for_runtime(&left);
    assert_eq!(current_device_resource_raw_value(), device_mr);

    let left_handle = std::thread::spawn(move || {
        set_current_device_for_runtime(&left);
        assert_eq!(current_device_resource_raw_value(), device_mr);
        let token = left
            .execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("left_current_mr"))
            .unwrap();
        let outcome = left.series_from_value_with_budget(3i32, 1024, &token).unwrap();
        assert_eq!(outcome.value().as_column().testing_execution_domain_mr_value(), device_mr);
        assert_eq!(token.memory_resource().testing_raw_handle_value(), device_mr);
        assert_eq!(current_device_resource_raw_value(), device_mr);
    });

    let right_handle = std::thread::spawn(move || {
        set_current_device_for_runtime(&right);
        assert_eq!(current_device_resource_raw_value(), device_mr);
        let token = right
            .execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("right_current_mr"))
            .unwrap();
        let outcome = right.series_from_value_with_budget(5i32, 2048, &token).unwrap();
        assert_eq!(outcome.value().as_column().testing_execution_domain_mr_value(), device_mr);
        assert_eq!(token.memory_resource().testing_raw_handle_value(), device_mr);
        assert_eq!(current_device_resource_raw_value(), device_mr);
    });

    left_handle.join().unwrap();
    right_handle.join().unwrap();
    set_current_device_for_runtime(&Runtime::builder().with_device_memory_manager().build().unwrap());
    assert_eq!(current_device_resource_raw_value(), device_mr);
}

#[test]
fn reservation_usage_releases_retained_output_by_allocation_identity() {
    let rt = Runtime::builder().with_device_memory_manager().build().unwrap();
    let token = rt
        .execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("retained_output"))
        .unwrap();

    let outcome = rt.series_from_value_with_budget(7i32, 4096, &token).unwrap();
    let output_bytes = outcome.value().device_allocated_size_bytes().unwrap();
    let while_retained = token.reservation_usage().unwrap();
    assert!(while_retained.current_bytes as usize >= output_bytes);

    drop(outcome);

    let after_drop = token.reservation_usage().unwrap();
    assert!(
        after_drop.current_bytes < while_retained.current_bytes,
        "dropping retained output must release bytes from the reservation that created it"
    );
}

#[test]
fn reservation_limit_failure_is_stream_local() {
    let small_rt = Runtime::builder().with_device_memory_manager().build().unwrap();
    let large_rt = Runtime::builder().with_device_memory_manager().build().unwrap();
    let small = small_rt.execution_budget_token(ExecutionBudgetRequest::new(1).with_boundary("small_query")).unwrap();
    let large = large_rt
        .execution_budget_token(ExecutionBudgetRequest::new(64 * 1024 * 1024).with_boundary("large_query"))
        .unwrap();

    let err = small_rt.series_from_value_with_budget(1i32, 1024, &small).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::ResourceExhausted);
    assert!(small.reservation_usage().unwrap().failed_allocation_count > 0);

    let outcome = large_rt.series_from_value_with_budget(2i32, 1024, &large).unwrap();
    assert_eq!(outcome.value().to_vec_strict::<i32>().unwrap()[0], 2);
    assert_eq!(large.reservation_usage().unwrap().failed_allocation_count, 0);
}

/// §10.1 invariant contract (replaces the per-op guard contract): with the
/// device memory domain installed, concurrent execution mixing default
/// (manager-backed) runtimes and owned-pool runtimes must never disturb the
/// device's current resource. In the device-domain model nothing swaps it, so
/// any observed change means a per-op swap was reintroduced.
#[test]
fn concurrent_owned_pool_and_manager_runtimes_preserve_device_current_resource() {
    let manager_rt = Runtime::builder().with_device_memory_manager().build().unwrap();
    let device_mr = manager_rt.device_memory_manager().unwrap().resource().testing_raw_handle_value();
    set_current_device_for_runtime(&manager_rt);
    assert_eq!(current_device_resource_raw_value(), device_mr);

    let stop = Arc::new(AtomicBool::new(false));
    let poller = {
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            let mut observations = 0u64;
            while !stop.load(Ordering::Relaxed) {
                assert_eq!(
                    current_device_resource_raw_value(),
                    device_mr,
                    "device current resource changed during concurrent mixed-runtime execution"
                );
                observations += 1;
            }
            observations
        })
    };

    let manager_worker = std::thread::spawn(move || {
        for _ in 0..40 {
            let series = manager_rt.series(&[3i32, 1, 2, 5, 4]).unwrap();
            let sorted = series.sort_values(SortOptions::default()).unwrap();
            assert_eq!(sorted.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3, 4, 5]);
        }
    });
    let owned_pool_worker = std::thread::spawn(move || {
        let rt = Runtime::builder().with_pool_memory_resource(64 * 1024 * 1024, None).build().unwrap();
        for _ in 0..40 {
            let series = rt.series(&[30i32, 10, 20]).unwrap();
            let sorted = series.sort_values(SortOptions::default()).unwrap();
            assert_eq!(sorted.to_vec_strict::<i32>().unwrap(), vec![10, 20, 30]);
        }
    });

    manager_worker.join().unwrap();
    owned_pool_worker.join().unwrap();
    stop.store(true, Ordering::Relaxed);
    let observations = poller.join().unwrap();
    assert!(observations > 0, "invariant poller never observed the current resource");
    assert_eq!(current_device_resource_raw_value(), device_mr);
}

/// §10.3 accounting correctness: two pipelines (two runtimes, each with its
/// own stream reservation) run temporary-heavy work concurrently. Temporaries
/// resolve through the device adaptor and are charged by stream, so each
/// pipeline's reservation must see its own usage (including temporaries) and
/// an over-limit failure must stay local to its pipeline.
#[test]
fn concurrent_temp_heavy_pipelines_keep_reservation_accounting_isolated() {
    const ROWS: usize = 1 << 20;
    const ROW_BYTES: usize = ROWS * std::mem::size_of::<i32>();

    let left = Runtime::builder().with_device_memory_manager().build().unwrap();
    let right = Runtime::builder().with_device_memory_manager().build().unwrap();
    let left_lease = left.stream_reservation(9101, 64 * 1024 * 1024).unwrap();
    let right_lease = right.stream_reservation(9202, 1).unwrap();

    let left_worker = std::thread::spawn(move || {
        let data: Vec<i32> = (0..ROWS as i32).rev().collect();
        let series = left.series(&data).unwrap();
        let sorted = series.sort_values(SortOptions::default()).unwrap();
        assert_eq!(sorted.len(), ROWS);
        let usage = left_lease.usage().unwrap();
        assert_eq!(usage.reservation_id, 9101);
        assert_eq!(usage.failed_allocation_count, 0, "well-budgeted pipeline must not observe failures");
        // Input column + sort temporaries (order map, materialized output) all
        // charge this reservation: the peak must clearly exceed one column.
        assert!(
            usage.peak_bytes as usize >= 2 * ROW_BYTES,
            "sort temporaries were not charged to the owning pipeline's reservation: peak {} < {}",
            usage.peak_bytes,
            2 * ROW_BYTES
        );
        usage.peak_bytes
    });

    let right_worker = std::thread::spawn(move || {
        // A 1-byte budget cannot even hold the input column: the failure must
        // surface in this pipeline only, as a resource-exhausted error.
        let err = right.series(&[1i32, 2, 3]).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::ResourceExhausted);
        let usage = right_lease.usage().unwrap();
        assert_eq!(usage.reservation_id, 9202);
        assert!(usage.failed_allocation_count > 0);
        assert_eq!(usage.current_bytes, 0, "failed allocations must not leak accounted bytes");
    });

    let left_peak = left_worker.join().unwrap();
    right_worker.join().unwrap();
    assert!(left_peak > 0);
}

/// §10.4 outputs routing: explicit memory resources still flow to output
/// allocations after the global lock removal — an owned-pool runtime's
/// outputs carry its pool, while the device current resource (temporaries'
/// route) stays on the device adaptor.
#[test]
fn explicit_output_memory_resource_routing_survives_lock_removal() {
    let manager_rt = Runtime::builder().with_device_memory_manager().build().unwrap();
    let device_mr = manager_rt.device_memory_manager().unwrap().resource().testing_raw_handle_value();

    let owned = Runtime::builder().with_pool_memory_resource(64 * 1024 * 1024, None).build().unwrap();
    let owned_mr = owned.memory_resource().testing_raw_handle_value();
    assert_ne!(owned_mr, device_mr);

    let series = owned.series(&[5i32, 1, 4]).unwrap();
    let sorted = series.sort_values(SortOptions::default()).unwrap();
    assert_eq!(sorted.to_vec_strict::<i32>().unwrap(), vec![1, 4, 5]);
    assert_eq!(
        sorted.as_column().testing_execution_domain_mr_value(),
        owned_mr,
        "outputs must keep routing through the explicitly provided memory resource"
    );

    set_current_device_for_runtime(&owned);
    assert_eq!(
        current_device_resource_raw_value(),
        device_mr,
        "running on an owned-pool runtime must not install its pool as the device current resource"
    );
}
