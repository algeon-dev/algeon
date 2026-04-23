use cudf::{Runtime, StreamDependencyOutcome, testing::prelude::*};
use rapids_interop::{CudaStreamHandle, ExecutionBinding, InteropErrorKind};

use crate::common::lock_event_observation;

fn reset_event_observation() {
    unsafe { cudf_sys::cudf_test_event_reset_observation() };
}

fn observed_stream_wait_call_count_for_stream_raw(stream_raw: usize) -> i32 {
    let mut out = -1;
    let status = unsafe { cudf_sys::cudf_test_event_stream_wait_call_count_for_stream_raw(stream_raw, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read per-stream wait observation count");
    out
}

fn observed_raw_record_call_count_for_stream_raw(stream_raw: usize) -> i32 {
    let mut out = -1;
    let status = unsafe { cudf_sys::cudf_test_event_raw_record_call_count_for_stream_raw(stream_raw, &mut out) };
    assert_eq!(status, cudf_sys::CUDF_SUCCESS, "failed to read per-stream raw event-record observation count");
    out
}

#[test]
fn runtime_execution_binding_matches_retained_runtime_provenance() {
    let runtime = Runtime::builder().build().unwrap();

    let binding = runtime.execution_binding().unwrap();

    assert_eq!(binding.device(), runtime.device().ordinal() as i32);
    assert_eq!(binding.cuda_stream().as_raw() as usize, runtime.stream().testing_raw_handle_value());
    assert_eq!(binding.cuda_stream().device(), runtime.device().ordinal() as i32);
    assert_eq!(binding.memory_resource().as_raw() as usize, runtime.memory_resource().testing_raw_handle_value());
}

#[test]
fn order_after_execution_binding_noops_for_exact_runtime_provenance() {
    let _observation_lock = lock_event_observation();
    let runtime = Runtime::builder().build().unwrap();
    let producer = runtime.execution_binding().unwrap();

    reset_event_observation();

    let outcome =
        runtime.order_after_execution_binding(producer, "runtime_execution_binding_tests::same_provenance").unwrap();

    assert!(matches!(outcome, StreamDependencyOutcome::NoOp));
    assert_eq!(observed_stream_wait_call_count_for_stream_raw(runtime.stream().testing_raw_handle_value()), 0);
    assert_eq!(observed_raw_record_call_count_for_stream_raw(runtime.stream().testing_raw_handle_value()), 0);
}

#[test]
fn order_after_execution_binding_noops_for_same_stream_different_memory_resource() {
    let _observation_lock = lock_event_observation();
    let runtime = Runtime::builder().build().unwrap();
    let alternate_mr_runtime = Runtime::builder().build().unwrap();
    let producer = ExecutionBinding::explicit(
        runtime.device().ordinal() as i32,
        CudaStreamHandle::from_raw_on_device(
            runtime.stream().testing_raw_handle_value() as *mut std::ffi::c_void,
            runtime.device().ordinal() as i32,
        )
        .unwrap(),
        alternate_mr_runtime.execution_binding().unwrap().memory_resource(),
    );

    reset_event_observation();

    let outcome = runtime
        .order_after_execution_binding(producer, "runtime_execution_binding_tests::same_stream_different_mr")
        .unwrap();

    assert!(matches!(outcome, StreamDependencyOutcome::NoOp));
    assert_eq!(observed_stream_wait_call_count_for_stream_raw(runtime.stream().testing_raw_handle_value()), 0);
    assert_eq!(observed_raw_record_call_count_for_stream_raw(runtime.stream().testing_raw_handle_value()), 0);
}

#[test]
fn order_after_execution_binding_records_barrier_for_same_device_different_stream() {
    let _observation_lock = lock_event_observation();
    let runtime = Runtime::builder().build().unwrap();
    let producer_runtime = Runtime::builder().build().unwrap();
    let producer = producer_runtime.execution_binding().unwrap();
    let producer_stream_raw = producer_runtime.stream().testing_raw_handle_value();
    let consumer_stream_raw = runtime.stream().testing_raw_handle_value();

    reset_event_observation();
    let baseline_record = observed_raw_record_call_count_for_stream_raw(producer_stream_raw);
    let baseline_wait = observed_stream_wait_call_count_for_stream_raw(consumer_stream_raw);

    let outcome = runtime
        .order_after_execution_binding(producer, "runtime_execution_binding_tests::same_device_different_stream")
        .unwrap();

    let barrier = match outcome {
        StreamDependencyOutcome::Barrier(barrier) => barrier,
        other => panic!("expected barrier outcome, got {other:?}"),
    };

    assert_eq!(barrier.device().ordinal(), runtime.device().ordinal());
    assert!(observed_raw_record_call_count_for_stream_raw(producer_stream_raw) > baseline_record);
    assert!(observed_stream_wait_call_count_for_stream_raw(consumer_stream_raw) > baseline_wait);
}

#[test]
fn order_after_execution_binding_rejects_cross_device_bindings_before_cuda_ordering() {
    let _observation_lock = lock_event_observation();
    let runtime = Runtime::builder().build().unwrap();
    let binding = runtime.execution_binding().unwrap();
    let cross_device = ExecutionBinding::explicit(
        binding.device() + 1,
        CudaStreamHandle::from_raw_on_device(binding.cuda_stream().as_raw(), binding.device() + 1).unwrap(),
        binding.memory_resource(),
    );

    reset_event_observation();

    let err = runtime
        .order_after_execution_binding(cross_device, "runtime_execution_binding_tests::cross_device")
        .unwrap_err();

    assert_eq!(err.kind(), InteropErrorKind::ExecutionBindingMismatch);
    assert_eq!(observed_stream_wait_call_count_for_stream_raw(runtime.stream().testing_raw_handle_value()), 0);
    assert_eq!(observed_raw_record_call_count_for_stream_raw(runtime.stream().testing_raw_handle_value()), 0);
}
