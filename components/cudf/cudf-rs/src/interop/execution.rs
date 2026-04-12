use super::*;

pub(super) fn execution_binding<'a>(
    operation: &'static str,
    domain: &Arc<ExecutionDomain>,
) -> Result<ExecutionBinding<'a>, InteropError> {
    let options = domain.explicit_exec_options_raw().expect("execution domain is always explicit");
    let stream = domain.explicit_stream().expect("execution domain is always explicit");
    let mr = domain.explicit_memory_resource().expect("execution domain is always explicit");
    let stream_ptr = stream
        .interop_raw_handle()
        .map_err(|error| map_cudf_error(InteropErrorKind::ExecutionBindingUnavailable, operation, error))?;
    let mr_ptr = mr
        .interop_raw_handle()
        .map_err(|error| map_cudf_error(InteropErrorKind::ExecutionBindingUnavailable, operation, error))?;
    Ok(ExecutionBinding::explicit(
        options.device,
        CudaStreamHandle::from_raw_on_device(stream_ptr, options.device)?,
        DeviceMemoryResourceHandle::from_raw(mr_ptr)?,
    ))
}

pub(super) fn prepared_device_readiness<'a>(
    operation: &'static str,
    domain: &Arc<ExecutionDomain>,
) -> Result<(i32, ProducerReadiness<'a>), InteropError> {
    let options = domain.explicit_exec_options_raw().expect("execution domain is always explicit");
    let stream = domain.explicit_stream().expect("execution domain is always explicit");
    let stream_ptr = stream
        .interop_raw_handle()
        .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    Ok((
        options.device,
        ProducerReadiness::DeviceStreamOrdered {
            stream: CudaStreamHandle::from_raw_on_device(stream_ptr, options.device)?,
        },
    ))
}

pub(super) fn prepared_device_event_readiness<'a>(
    operation: &'static str,
    domain: &Arc<ExecutionDomain>,
    event: &'a CudaEvent,
) -> Result<(i32, ProducerReadiness<'a>), InteropError> {
    let options = domain.explicit_exec_options_raw().expect("execution domain is always explicit");
    let event_device = event.device().ordinal_raw();
    if event_device != options.device {
        return Err(InteropError::new(
            InteropErrorKind::ReadinessMismatch,
            "prepared export readiness event must live on the producer CUDA device",
        )
        .with_context("operation", operation)
        .with_context("producer_device", options.device.to_string())
        .with_context("event_device", event_device.to_string()));
    }
    let event_ptr = event
        .interop_raw_handle()
        .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    Ok((
        options.device,
        ProducerReadiness::DeviceEventOrdered { event: CudaEventHandle::from_raw_on_device(event_ptr, event_device)? },
    ))
}
