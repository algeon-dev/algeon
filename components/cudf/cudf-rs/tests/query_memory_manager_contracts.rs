mod common;

use cudf::{
    DeviceMemoryManager, ErrorKind, ExecutionBudgetRequest, ExecutionBudgetStatus, Runtime, StreamReservationLease,
    testing::prelude::*,
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
