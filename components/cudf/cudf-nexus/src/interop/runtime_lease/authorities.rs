use std::ffi::c_void;

use rapids_interop::{CompletionWitnessBackend, InteropError, InteropErrorKind, LeaseExportContext, LeaseReadiness};

use super::{SHARED_COMPLETION_WITNESS_READY_OP, SHARED_COMPLETION_WITNESS_SYNC_OP};
use crate::{
    core::error::Result,
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
            super::super::map_cudf_error(
                InteropErrorKind::ReadinessUnavailable,
                SHARED_COMPLETION_WITNESS_READY_OP,
                error,
            )
        })?;
        self.event.is_ready().map_err(|error| {
            super::super::map_cudf_error(
                InteropErrorKind::ReadinessUnavailable,
                SHARED_COMPLETION_WITNESS_READY_OP,
                error,
            )
        })
    }

    fn synchronize(&self) -> rapids_interop::Result<()> {
        let _guard = DeviceGuard::new(self.device()).map_err(|error| {
            super::super::map_cudf_error(
                InteropErrorKind::ReadinessUnavailable,
                SHARED_COMPLETION_WITNESS_SYNC_OP,
                error,
            )
        })?;
        self.event.synchronize().map_err(|error| {
            super::super::map_cudf_error(
                InteropErrorKind::ReadinessUnavailable,
                SHARED_COMPLETION_WITNESS_SYNC_OP,
                error,
            )
        })
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
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

impl ProducerReadinessBinding {
    pub(super) fn device(self) -> DeviceId {
        match self {
            Self::Stream { device, .. } | Self::Event { device, .. } => device,
        }
    }
}

pub(super) fn producer_readiness_binding_from_export(
    export: &LeaseExportContext,
    operation: &'static str,
) -> std::result::Result<ProducerReadinessBinding, InteropError> {
    match export.readiness() {
        LeaseReadiness::DeviceStreamOrdered { raw, device } => {
            let device = DeviceId::new(device as usize).map_err(|error| {
                super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error)
            })?;
            Ok(ProducerReadinessBinding::Stream { raw, device })
        },
        LeaseReadiness::HostVisible => Err(InteropError::new(
            InteropErrorKind::ProfileUnsupported,
            "shared CUDA prepared leases require explicit device-ordered producer readiness",
        )
        .with_context("operation", operation)
        .with_context("readiness", "host_visible")),
        LeaseReadiness::DeviceEventOrdered { raw, device } => {
            let device = DeviceId::new(device as usize).map_err(|error| {
                super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error)
            })?;
            Ok(ProducerReadinessBinding::Event { raw, device })
        },
    }
}

pub(super) fn create_producer_owned_event(producer_device: DeviceId) -> Result<CudaEvent> {
    let _guard = DeviceGuard::new(producer_device)?;
    CudaEvent::new()
}
