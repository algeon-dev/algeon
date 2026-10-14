use std::sync::Arc;

use super::{
    BoundBytes, BufferBinding, CudaEvent, CudaEventHandle, CudaStreamHandle, DeviceMemoryResourceHandle,
    ExecutionBinding, ExecutionDomain, InteropError, InteropErrorKind, ProducerReadiness, map_cudf_error,
};

pub(crate) fn execution_binding<'a>(
    operation: &'static str,
    domain: &Arc<ExecutionDomain>,
) -> Result<ExecutionBinding<'a>, InteropError> {
    let options = domain
        .enter_execution()
        .map_err(|error| map_cudf_error(InteropErrorKind::ExecutionBindingUnavailable, operation, error))?;
    let stream = domain.stream();
    let mr = domain.memory_resource();
    let stream_ptr = stream.interop_raw_handle();
    let mr_ptr = mr.interop_raw_handle();
    // SAFETY: runtime accessors borrow this domain or its retained owner, which
    // owns the stream and MR and preserves their device identity.
    Ok(ExecutionBinding::new(
        unsafe { CudaStreamHandle::from_raw_on_device(stream_ptr, options.options().device) }?,
        unsafe { DeviceMemoryResourceHandle::from_raw(mr_ptr) }?,
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
    let stream_ptr = stream.interop_raw_handle();
    let device = options.options().device;
    // SAFETY: the producer column/table borrows or retains this domain, and
    // the domain owns this live stream on the selected device.
    Ok((
        device,
        ProducerReadiness::DeviceStreamOrdered {
            stream: unsafe { CudaStreamHandle::from_raw_on_device(stream_ptr, device) }?,
        },
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
    // SAFETY: the readiness borrows the event and its device was checked above.
    Ok((
        device,
        ProducerReadiness::DeviceEventOrdered {
            event: unsafe { CudaEventHandle::from_raw_on_device(event_ptr, event_device) }?,
        },
    ))
}

pub(super) fn to_static_binding(
    operation: &'static str,
    binding: BufferBinding<'_>,
) -> Result<BufferBinding<'static>, InteropError> {
    // SAFETY: This lifetime erasure is only used while constructing prepared
    // exports rooted in an owned column or table. The raw pointer still refers
    // to producer-owned device memory, and the export keeps that owner alive as
    // its retained owner, so the source `Column`/`Table` outlives every
    // `BufferBinding<'static>` created here.
    Ok(BufferBinding::new(
        binding.id(),
        BoundBytes::from_raw_parts(binding.bytes().base_ptr() as *mut std::ffi::c_void, binding.bytes().byte_len())
            .map_err(|error| error.with_context("operation", operation))?,
        *binding.placement(),
    ))
}

pub(super) fn to_static_bindings(
    operation: &'static str,
    bindings: Vec<BufferBinding<'_>>,
) -> Result<Vec<BufferBinding<'static>>, InteropError> {
    bindings.into_iter().map(|binding| to_static_binding(operation, binding)).collect()
}

pub(super) fn to_static_readiness(
    operation: &'static str,
    readiness: ProducerReadiness<'_>,
) -> Result<ProducerReadiness<'static>, InteropError> {
    // SAFETY: owned export construction retains the column/table domain and,
    // for event readiness, the event itself before exposing this descriptor.
    match readiness {
        ProducerReadiness::DeviceStreamOrdered { stream } => Ok(ProducerReadiness::DeviceStreamOrdered {
            stream: unsafe { CudaStreamHandle::from_raw_on_device(stream.as_raw(), stream.device()) }.map_err(
                |error| error.with_context("operation", operation).with_context("readiness", "device_stream_ordered"),
            )?,
        }),
        ProducerReadiness::DeviceEventOrdered { event } => Ok(ProducerReadiness::DeviceEventOrdered {
            event: unsafe { CudaEventHandle::from_raw_on_device(event.as_raw(), event.device()) }.map_err(|error| {
                error.with_context("operation", operation).with_context("readiness", "device_event_ordered")
            })?,
        }),
    }
}
