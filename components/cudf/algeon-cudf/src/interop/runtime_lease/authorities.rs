use std::ffi::c_void;

use rapids_interop::{CompletionWitnessBackend, InteropError, InteropErrorKind, ProducerReadiness};

use super::{SHARED_COMPLETION_WITNESS_READY_OP, SHARED_COMPLETION_WITNESS_SYNC_OP};
use crate::{
    core::error::Result,
    interop::map_cudf_error,
    runtime::{CudaEvent, DeviceId, device_guard::DeviceGuard},
};

pub(super) struct SharedCudaCompletionWitnessState {
    event: CudaEvent,
}

impl SharedCudaCompletionWitnessState {
    pub(super) fn new(event: CudaEvent) -> Self {
        Self { event }
    }

    fn device(&self) -> DeviceId {
        self.event.device()
    }
}

impl CompletionWitnessBackend for SharedCudaCompletionWitnessState {
    fn is_ready(&self) -> rapids_interop::Result<bool> {
        let _guard = DeviceGuard::new(self.device()).map_err(|error| {
            map_cudf_error(InteropErrorKind::ReadinessUnavailable, SHARED_COMPLETION_WITNESS_READY_OP, error)
        })?;
        self.event.is_ready().map_err(|error| {
            map_cudf_error(InteropErrorKind::ReadinessUnavailable, SHARED_COMPLETION_WITNESS_READY_OP, error)
        })
    }

    fn synchronize(&self) -> rapids_interop::Result<()> {
        let _guard = DeviceGuard::new(self.device()).map_err(|error| {
            map_cudf_error(InteropErrorKind::ReadinessUnavailable, SHARED_COMPLETION_WITNESS_SYNC_OP, error)
        })?;
        self.event.synchronize().map_err(|error| {
            map_cudf_error(InteropErrorKind::ReadinessUnavailable, SHARED_COMPLETION_WITNESS_SYNC_OP, error)
        })
    }
}

/// # Safety
///
/// The `raw` pointers are lifetime-erased CUDA stream/event handles. They are
/// valid only while the enclosing shared lease/export that provided them
/// remains alive. Do not store these bindings beyond the immediate interop
/// activation/completion flow.
#[derive(Clone, Copy, Debug)]
pub(super) enum ProducerReadinessBinding {
    Stream { raw: *mut c_void, device: DeviceId },
    Event { raw: *mut c_void, device: DeviceId },
}

pub(super) fn producer_readiness_binding(
    readiness: ProducerReadiness<'_>,
    operation: &'static str,
) -> std::result::Result<ProducerReadinessBinding, InteropError> {
    let device = DeviceId::new(readiness.device() as usize)
        .map_err(|error| map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    Ok(match readiness {
        ProducerReadiness::DeviceStreamOrdered { stream } => {
            ProducerReadinessBinding::Stream { raw: stream.as_raw(), device }
        },
        ProducerReadiness::DeviceEventOrdered { event } => {
            ProducerReadinessBinding::Event { raw: event.as_raw(), device }
        },
    })
}

pub(super) fn create_producer_owned_event(producer_device: DeviceId) -> Result<CudaEvent> {
    let _guard = DeviceGuard::new(producer_device)?;
    CudaEvent::new()
}
