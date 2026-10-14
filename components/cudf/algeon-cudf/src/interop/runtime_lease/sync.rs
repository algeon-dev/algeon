use std::{ffi::c_void, sync::Arc};

use rapids_interop::{CompletionWitness, InteropError, InteropErrorKind};

use super::authorities::{ProducerReadinessBinding, SharedCudaCompletionWitnessState, create_producer_owned_event};
use crate::{
    core::error::{self, Result},
    interop::map_cudf_error,
    runtime::{CudaEvent, CudaStream, DeviceId, device_guard::DeviceGuard},
};

pub(super) fn complete_same_device_stream_shared(
    producer_device: DeviceId,
    consumer_stream_raw: *mut c_void,
    consumer_device: DeviceId,
    operation: &'static str,
) -> rapids_interop::Result<CompletionWitness> {
    let shared_event = create_producer_owned_event(producer_device)
        .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    let witness_event_raw = shared_event
        .interop_raw_handle()
        .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    let witness = CompletionWitness::new(Arc::new(SharedCudaCompletionWitnessState::new(shared_event)));
    let _guard = DeviceGuard::new(producer_device)
        .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    error::check_status_in_with_details(
        unsafe { cudf_sys::cudf_raw_cuda_event_record(witness_event_raw, consumer_stream_raw) },
        operation,
        [
            ("completion", "same_device_stream".to_string()),
            ("producer_device", producer_device.ordinal().to_string()),
            ("consumer_device", consumer_device.ordinal().to_string()),
        ],
    )
    .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    Ok(witness)
}

pub(in crate::interop) fn synchronize_same_device_stream_raw(
    stream_raw: *mut c_void,
    stream_device: DeviceId,
    operation: &'static str,
) -> Result<()> {
    // A failed completion setup may already have marked the device unhealthy.
    // Cleanup must still attempt entry and retire pending reads before release.
    let _guard = DeviceGuard::enter_for_teardown(stream_device, operation)?;
    error::check_status_in_with_details(
        unsafe { cudf_sys::cudf_raw_cuda_stream_synchronize(stream_raw) },
        operation,
        [("completion", "same_device_stream".to_string()), ("consumer_device", stream_device.ordinal().to_string())],
    )
}

pub(super) fn order_consumer_stream_after_producer_event_raw(
    producer_event_raw: *mut c_void,
    producer_device: DeviceId,
    consumer_stream_raw: *mut c_void,
    consumer_device: DeviceId,
    operation: &'static str,
) -> std::result::Result<(), InteropError> {
    let _guard = DeviceGuard::new(producer_device)
        .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    error::check_status_in_with_details(
        unsafe { cudf_sys::cudf_raw_cuda_stream_wait_event(consumer_stream_raw, producer_event_raw) },
        operation,
        [
            ("readiness", "device_event_ordered".to_string()),
            ("read_start", "consumer_stream".to_string()),
            ("producer_device", producer_device.ordinal().to_string()),
            ("consumer_device", consumer_device.ordinal().to_string()),
        ],
    )
    .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))
}

pub(super) fn order_consumer_stream_after_producer_stream_raw(
    producer_stream_raw: *mut c_void,
    producer_device: DeviceId,
    consumer_stream_raw: *mut c_void,
    consumer_device: DeviceId,
    operation: &'static str,
) -> std::result::Result<CudaEvent, InteropError> {
    let _guard = DeviceGuard::new(producer_device)
        .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    let barrier = CudaEvent::new_on(producer_device).map_err(|error| {
        map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error)
            .with_context("readiness", "device_stream_ordered")
            .with_context("read_start", "consumer_stream")
    })?;
    let barrier_raw = barrier.interop_raw_handle().map_err(|error| {
        map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error)
            .with_context("readiness", "device_stream_ordered")
            .with_context("read_start", "consumer_stream")
    })?;
    error::check_status_in_with_details(
        unsafe { cudf_sys::cudf_raw_cuda_event_record(barrier_raw, producer_stream_raw) },
        operation,
        [
            ("readiness", "device_stream_ordered".to_string()),
            ("read_start", "consumer_stream".to_string()),
            ("producer_device", producer_device.ordinal().to_string()),
        ],
    )
    .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    error::check_status_in_with_details(
        unsafe { cudf_sys::cudf_raw_cuda_stream_wait_event(consumer_stream_raw, barrier_raw) },
        operation,
        [
            ("readiness", "device_stream_ordered".to_string()),
            ("read_start", "consumer_stream".to_string()),
            ("producer_device", producer_device.ordinal().to_string()),
            ("consumer_device", consumer_device.ordinal().to_string()),
        ],
    )
    .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    Ok(barrier)
}

pub(super) fn order_consumer_stream_after_producer_readiness_raw(
    producer_readiness: ProducerReadinessBinding,
    consumer_stream_raw: *mut c_void,
    consumer_device: DeviceId,
    operation: &'static str,
) -> std::result::Result<Option<CudaEvent>, InteropError> {
    match producer_readiness {
        ProducerReadinessBinding::Stream { raw, device } => order_consumer_stream_after_producer_stream_raw(
            raw,
            device,
            consumer_stream_raw,
            consumer_device,
            operation,
        )
        .map(Some),
        ProducerReadinessBinding::Event { raw, device } => {
            order_consumer_stream_after_producer_event_raw(
                raw,
                device,
                consumer_stream_raw,
                consumer_device,
                operation,
            )?;
            Ok(None)
        },
    }
}

pub(super) fn synchronize_producer_stream_readiness_raw(
    producer_stream_raw: *mut c_void,
    producer_device: DeviceId,
    operation: &'static str,
) -> std::result::Result<(), InteropError> {
    let _guard = DeviceGuard::new(producer_device)
        .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    let barrier = CudaEvent::new_on(producer_device)
        .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    let barrier_raw = barrier
        .interop_raw_handle()
        .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    error::check_status_in_with_details(
        unsafe { cudf_sys::cudf_raw_cuda_event_record(barrier_raw, producer_stream_raw) },
        operation,
        [
            ("readiness", "device_stream_ordered".to_string()),
            ("read_start", "synchronous_by_return".to_string()),
            ("producer_device", producer_device.ordinal().to_string()),
        ],
    )
    .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    barrier.synchronize().map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))
}

pub(super) fn synchronize_producer_event_readiness_raw(
    producer_event_raw: *mut c_void,
    producer_device: DeviceId,
    operation: &'static str,
) -> std::result::Result<(), InteropError> {
    let _guard = DeviceGuard::new(producer_device)
        .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    let stream = CudaStream::new_on(producer_device)
        .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    let stream_raw = stream.interop_raw_handle();
    let barrier = CudaEvent::new_on(producer_device)
        .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    let barrier_raw = barrier
        .interop_raw_handle()
        .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    error::check_status_in_with_details(
        unsafe { cudf_sys::cudf_raw_cuda_stream_wait_event(stream_raw, producer_event_raw) },
        operation,
        [
            ("readiness", "device_event_ordered".to_string()),
            ("read_start", "synchronous_by_return".to_string()),
            ("producer_device", producer_device.ordinal().to_string()),
        ],
    )
    .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    error::check_status_in_with_details(
        unsafe { cudf_sys::cudf_raw_cuda_event_record(barrier_raw, stream_raw) },
        operation,
        [
            ("readiness", "device_event_ordered".to_string()),
            ("read_start", "synchronous_by_return".to_string()),
            ("producer_device", producer_device.ordinal().to_string()),
        ],
    )
    .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    barrier.synchronize().map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))
}

pub(super) fn synchronize_producer_readiness(
    producer_readiness: ProducerReadinessBinding,
    operation: &'static str,
) -> std::result::Result<(), InteropError> {
    match producer_readiness {
        ProducerReadinessBinding::Stream { raw, device } => {
            synchronize_producer_stream_readiness_raw(raw, device, operation)
        },
        ProducerReadinessBinding::Event { raw, device } => {
            synchronize_producer_event_readiness_raw(raw, device, operation)
        },
    }
}
