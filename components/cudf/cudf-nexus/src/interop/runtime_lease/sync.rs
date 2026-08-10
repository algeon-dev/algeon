use std::{ffi::c_void, sync::Arc};

use rapids_interop::{CompletionWitness, CompletionWitnessKind, InteropError, InteropErrorKind};

use super::authorities::{ProducerReadinessBinding, SharedCudaCompletionWitnessState, create_producer_owned_event};
use crate::{
    core::error::{self, Result},
    interop::runtime_release::TransferCompletionAuthority,
    runtime::{CudaEvent, CudaStream, DeviceId, device_guard::DeviceGuard},
};

pub(super) fn complete_same_device_stream_shared(
    producer_device: DeviceId,
    consumer_stream_raw: *mut c_void,
    consumer_device: DeviceId,
    operation: &'static str,
) -> rapids_interop::Result<CompletionWitness> {
    let shared_event = create_producer_owned_event(producer_device)
        .map_err(|error| super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    let witness_event_raw = shared_event
        .interop_raw_handle()
        .map_err(|error| super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    let witness = CompletionWitness::with_backend(
        CompletionWitnessKind::ConsumerStream,
        Arc::new(SharedCudaCompletionWitnessState::new(shared_event)),
    );
    let _guard = DeviceGuard::new(producer_device)
        .map_err(|error| super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    error::check_status_in_with_details(
        unsafe { cudf_sys::cudf_raw_cuda_event_record(witness_event_raw, consumer_stream_raw) },
        operation,
        [
            ("completion", "same_device_stream".to_string()),
            ("producer_device", producer_device.ordinal().to_string()),
            ("consumer_device", consumer_device.ordinal().to_string()),
        ],
    )
    .map_err(|error| super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    Ok(witness)
}

pub(super) fn complete_cross_device_event_shared(
    producer_readiness: ProducerReadinessBinding,
    consumer_event_raw: *mut c_void,
    consumer_device: DeviceId,
    operation: &'static str,
) -> rapids_interop::Result<CompletionWitness> {
    let producer_device = producer_readiness.device();
    let shared_state = create_producer_owned_event(producer_device)
        .map_err(|error| super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    let witness_event_raw = shared_state
        .interop_raw_handle()
        .map_err(|error| super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    let witness = CompletionWitness::with_backend(
        CompletionWitnessKind::ConsumerEvent,
        Arc::new(SharedCudaCompletionWitnessState::new(shared_state)),
    );

    let _guard = DeviceGuard::new(producer_device)
        .map_err(|error| super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    match producer_readiness {
        ProducerReadinessBinding::Stream { raw: producer_stream_raw, .. } => {
            error::check_status_in_with_details(
                unsafe { cudf_sys::cudf_raw_cuda_stream_wait_event(producer_stream_raw, consumer_event_raw) },
                operation,
                [
                    ("completion", "cross_device_event".to_string()),
                    ("producer_device", producer_device.ordinal().to_string()),
                    ("consumer_device", consumer_device.ordinal().to_string()),
                ],
            )
            .map_err(|error| super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
            error::check_status_in_with_details(
                unsafe { cudf_sys::cudf_raw_cuda_event_record(witness_event_raw, producer_stream_raw) },
                operation,
                [
                    ("completion", "cross_device_event".to_string()),
                    ("producer_device", producer_device.ordinal().to_string()),
                    ("consumer_device", consumer_device.ordinal().to_string()),
                ],
            )
            .map_err(|error| super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
        },
        ProducerReadinessBinding::Event { raw: producer_event_raw, .. } => {
            let producer_stream = CudaStream::new().map_err(|error| {
                super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error)
            })?;
            let producer_stream_raw = producer_stream.interop_raw_handle().map_err(|error| {
                super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error)
            })?;
            error::check_status_in_with_details(
                unsafe { cudf_sys::cudf_raw_cuda_stream_wait_event(producer_stream_raw, producer_event_raw) },
                operation,
                [
                    ("readiness", "device_event_ordered".to_string()),
                    ("producer_device", producer_device.ordinal().to_string()),
                ],
            )
            .map_err(|error| super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
            error::check_status_in_with_details(
                unsafe { cudf_sys::cudf_raw_cuda_stream_wait_event(producer_stream_raw, consumer_event_raw) },
                operation,
                [
                    ("completion", "cross_device_event".to_string()),
                    ("readiness", "device_event_ordered".to_string()),
                    ("producer_device", producer_device.ordinal().to_string()),
                    ("consumer_device", consumer_device.ordinal().to_string()),
                ],
            )
            .map_err(|error| super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
            error::check_status_in_with_details(
                unsafe { cudf_sys::cudf_raw_cuda_event_record(witness_event_raw, producer_stream_raw) },
                operation,
                [
                    ("completion", "cross_device_event".to_string()),
                    ("readiness", "device_event_ordered".to_string()),
                    ("producer_device", producer_device.ordinal().to_string()),
                    ("consumer_device", consumer_device.ordinal().to_string()),
                ],
            )
            .map_err(|error| super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
        },
    }
    Ok(witness)
}

pub(in crate::interop) fn synchronize_same_device_stream_raw(
    stream_raw: *mut c_void,
    stream_device: DeviceId,
    operation: &'static str,
) -> Result<()> {
    let event = create_producer_owned_event(stream_device).map_err(|error| error.reanchor(operation))?;
    let event_raw = event.interop_raw_handle().map_err(|error| error.reanchor(operation))?;
    let _guard = DeviceGuard::new(stream_device)?;
    error::check_status_in_with_details(
        unsafe { cudf_sys::cudf_raw_cuda_event_record(event_raw, stream_raw) },
        operation,
        [("completion", "same_device_stream".to_string()), ("consumer_device", stream_device.ordinal().to_string())],
    )?;
    event.synchronize().map_err(|error| error.reanchor(operation).with_detail("completion", "same_device_stream"))
}

pub(in crate::interop) fn synchronize_cross_device_event_raw(
    event_raw: *mut c_void,
    consumer_device: DeviceId,
    operation: &'static str,
) -> Result<()> {
    let _guard = DeviceGuard::new(consumer_device)?;
    let stream = CudaStream::new().map_err(|error| error.reanchor(operation))?;
    let stream_raw = stream.interop_raw_handle().map_err(|error| error.reanchor(operation))?;
    let completion = CudaEvent::new().map_err(|error| error.reanchor(operation))?;
    let completion_raw = completion.interop_raw_handle().map_err(|error| error.reanchor(operation))?;
    error::check_status_in_with_details(
        unsafe { cudf_sys::cudf_raw_cuda_stream_wait_event(stream_raw, event_raw) },
        operation,
        [("completion", "cross_device_event".to_string()), ("consumer_device", consumer_device.ordinal().to_string())],
    )?;
    error::check_status_in_with_details(
        unsafe { cudf_sys::cudf_raw_cuda_event_record(completion_raw, stream_raw) },
        operation,
        [("completion", "cross_device_event".to_string()), ("consumer_device", consumer_device.ordinal().to_string())],
    )?;
    completion.synchronize().map_err(|error| error.reanchor(operation).with_detail("completion", "cross_device_event"))
}

pub(super) fn order_consumer_stream_after_producer_event_raw(
    producer_event_raw: *mut c_void,
    producer_device: DeviceId,
    consumer_stream_raw: *mut c_void,
    consumer_device: DeviceId,
    operation: &'static str,
) -> std::result::Result<(), InteropError> {
    let _guard = DeviceGuard::new(producer_device)
        .map_err(|error| super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
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
    .map_err(|error| super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))
}

pub(super) fn order_consumer_stream_after_producer_stream_raw(
    producer_stream_raw: *mut c_void,
    producer_device: DeviceId,
    consumer_stream_raw: *mut c_void,
    consumer_device: DeviceId,
    operation: &'static str,
) -> std::result::Result<CudaEvent, InteropError> {
    let _guard = DeviceGuard::new(producer_device)
        .map_err(|error| super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    let barrier = CudaEvent::new().map_err(|error| {
        super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error)
            .with_context("readiness", "device_stream_ordered")
            .with_context("read_start", "consumer_stream")
    })?;
    let barrier_raw = barrier.interop_raw_handle().map_err(|error| {
        super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error)
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
    .map_err(|error| super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
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
    .map_err(|error| super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
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
        .map_err(|error| super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    let barrier = CudaEvent::new()
        .map_err(|error| super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    let barrier_raw = barrier
        .interop_raw_handle()
        .map_err(|error| super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    error::check_status_in_with_details(
        unsafe { cudf_sys::cudf_raw_cuda_event_record(barrier_raw, producer_stream_raw) },
        operation,
        [
            ("readiness", "device_stream_ordered".to_string()),
            ("read_start", "synchronous_by_return".to_string()),
            ("producer_device", producer_device.ordinal().to_string()),
        ],
    )
    .map_err(|error| super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    barrier
        .synchronize()
        .map_err(|error| super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))
}

pub(super) fn synchronize_producer_event_readiness_raw(
    producer_event_raw: *mut c_void,
    producer_device: DeviceId,
    operation: &'static str,
) -> std::result::Result<(), InteropError> {
    let _guard = DeviceGuard::new(producer_device)
        .map_err(|error| super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    let stream = CudaStream::new()
        .map_err(|error| super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    let stream_raw = stream
        .interop_raw_handle()
        .map_err(|error| super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    let barrier = CudaEvent::new()
        .map_err(|error| super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    let barrier_raw = barrier
        .interop_raw_handle()
        .map_err(|error| super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    error::check_status_in_with_details(
        unsafe { cudf_sys::cudf_raw_cuda_stream_wait_event(stream_raw, producer_event_raw) },
        operation,
        [
            ("readiness", "device_event_ordered".to_string()),
            ("read_start", "synchronous_by_return".to_string()),
            ("producer_device", producer_device.ordinal().to_string()),
        ],
    )
    .map_err(|error| super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    error::check_status_in_with_details(
        unsafe { cudf_sys::cudf_raw_cuda_event_record(barrier_raw, stream_raw) },
        operation,
        [
            ("readiness", "device_event_ordered".to_string()),
            ("read_start", "synchronous_by_return".to_string()),
            ("producer_device", producer_device.ordinal().to_string()),
        ],
    )
    .map_err(|error| super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    barrier
        .synchronize()
        .map_err(|error| super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))
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

pub(crate) fn synchronize_completion_fallback(
    completion: TransferCompletionAuthority<'_>,
    operation: &'static str,
) -> Result<()> {
    match completion {
        TransferCompletionAuthority::SynchronousByReturn => Ok(()),
        TransferCompletionAuthority::SameDeviceStream { stream } => {
            let _guard = DeviceGuard::new(stream.device())?;
            stream
                .synchronize()
                .map_err(|error| error.reanchor(operation).with_detail("completion", "same_device_stream"))
        },
        TransferCompletionAuthority::CrossDeviceEvent { event } => {
            let _guard = DeviceGuard::new(event.device())?;
            event
                .synchronize()
                .map_err(|error| error.reanchor(operation).with_detail("completion", "cross_device_event"))
        },
    }
}
