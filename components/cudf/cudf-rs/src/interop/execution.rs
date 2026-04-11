use super::*;

pub(super) fn execution_binding<'a>(
    operation: &'static str,
    domain: &Arc<ExecutionDomain>,
) -> Result<ExecutionBinding<'a>, InteropError> {
    match domain.explicit_exec_options_raw() {
        None => Err(InteropError::new(
            InteropErrorKind::AmbientInteropUnsupported,
            "borrowed interop requires explicit execution provenance; ambient/default-only objects are not valid producers",
        )
        .with_context("operation", operation)
        .with_context("policy", "ambient_interop_unsupported")
        .with_context("device_snapshot", domain.device_snapshot().to_string())),
        Some(options) => {
            let stream = domain.explicit_stream().expect("explicit domain missing stream");
            let mr = domain.explicit_memory_resource().expect("explicit domain missing memory resource");
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
        },
    }
}

pub(super) fn prepared_device_readiness<'a>(
    operation: &'static str,
    domain: &Arc<ExecutionDomain>,
) -> Result<(i32, ProducerReadiness<'a>), InteropError> {
    match domain.explicit_exec_options_raw() {
        None => Err(InteropError::new(
            InteropErrorKind::AmbientInteropUnsupported,
            "prepared device export requires explicit execution provenance; ambient/default-only objects are not valid producers",
        )
        .with_context("operation", operation)
        .with_context("policy", "ambient_interop_unsupported")
        .with_context("device_snapshot", domain.device_snapshot().to_string())),
        Some(options) => {
            let stream = domain.explicit_stream().expect("explicit domain missing stream");
            let stream_ptr = stream
                .interop_raw_handle()
                .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
            Ok((
                options.device,
                ProducerReadiness::DeviceStreamOrdered {
                    stream: CudaStreamHandle::from_raw_on_device(stream_ptr, options.device)?,
                },
            ))
        },
    }
}

pub(super) fn prepared_device_event_readiness<'a>(
    operation: &'static str,
    domain: &Arc<ExecutionDomain>,
    event: &'a CudaEvent,
) -> Result<(i32, ProducerReadiness<'a>), InteropError> {
    match domain.explicit_exec_options_raw() {
        None => Err(InteropError::new(
            InteropErrorKind::AmbientInteropUnsupported,
            "prepared device export requires explicit execution provenance; ambient/default-only objects are not valid producers",
        )
        .with_context("operation", operation)
        .with_context("policy", "ambient_interop_unsupported")
        .with_context("device_snapshot", domain.device_snapshot().to_string())),
        Some(options) => {
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
                ProducerReadiness::DeviceEventOrdered {
                    event: CudaEventHandle::from_raw_on_device(event_ptr, event_device)?,
                },
            ))
        },
    }
}
