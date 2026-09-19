use std::sync::Arc;

use super::*;

pub(crate) fn execution_binding<'a>(
    operation: &'static str,
    domain: &Arc<ExecutionDomain>,
) -> Result<ExecutionBinding<'a>, InteropError> {
    let options = domain
        .enter_execution()
        .map_err(|error| map_cudf_error(InteropErrorKind::ExecutionBindingUnavailable, operation, error))?;
    let stream = domain.stream();
    let mr = domain.memory_resource();
    let stream_ptr = stream
        .interop_raw_handle()
        .map_err(|error| map_cudf_error(InteropErrorKind::ExecutionBindingUnavailable, operation, error))?;
    let mr_ptr = mr
        .interop_raw_handle()
        .map_err(|error| map_cudf_error(InteropErrorKind::ExecutionBindingUnavailable, operation, error))?;
    let device = options.options().device;
    Ok(ExecutionBinding::explicit(
        device,
        CudaStreamHandle::from_raw_on_device(stream_ptr, device)?,
        DeviceMemoryResourceHandle::from_raw(mr_ptr)?,
    ))
}

pub(super) fn prepared_device_readiness<'a>(
    operation: &'static str,
    domain: &Arc<ExecutionDomain>,
) -> Result<(i32, ProducerReadiness<'a>), InteropError> {
    let options = domain
        .enter_execution()
        .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    let stream = domain.stream();
    let stream_ptr = stream
        .interop_raw_handle()
        .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    let device = options.options().device;
    Ok((
        device,
        ProducerReadiness::DeviceStreamOrdered { stream: CudaStreamHandle::from_raw_on_device(stream_ptr, device)? },
    ))
}

pub(super) fn prepared_device_event_readiness<'a>(
    operation: &'static str,
    domain: &Arc<ExecutionDomain>,
    event: &'a CudaEvent,
) -> Result<(i32, ProducerReadiness<'a>), InteropError> {
    let options = domain
        .enter_execution()
        .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    let device = options.options().device;
    let event_device = event.device().ordinal_raw();
    if event_device != device {
        return Err(InteropError::new(
            InteropErrorKind::ReadinessMismatch,
            "prepared export readiness event must live on the producer CUDA device",
        )
        .with_context("operation", operation)
        .with_context("left_device", device.to_string())
        .with_context("right_device", event_device.to_string()));
    }
    let event_ptr = event
        .interop_raw_handle()
        .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    Ok((
        device,
        ProducerReadiness::DeviceEventOrdered { event: CudaEventHandle::from_raw_on_device(event_ptr, event_device)? },
    ))
}
