use std::{ffi::c_void, fmt};

use rapids_interop::{
    CompletionAuthority, CompletionWitnessBackend, CompletionWitnessKind, CudaEventHandle, CudaStreamHandle,
    InteropError, InteropErrorKind, LeaseExportContext, LeaseReadiness, ProducerReadiness, ReadStartAuthority,
};

use super::{
    RUNTIME_COMPLETION_WITNESS_READY_OP, RUNTIME_COMPLETION_WITNESS_SYNC_OP, SHARED_COMPLETION_WITNESS_READY_OP,
    SHARED_COMPLETION_WITNESS_SYNC_OP,
};
use crate::{
    core::error::{self, Result},
    runtime::{CudaEvent, CudaStream, DeviceId, device_guard::DeviceGuard},
};

/// CUDA-specific compatibility wrapper for shared consumer read-start
/// authority.
///
/// This surface exists for callers that already traffic in
/// [`crate::runtime::CudaStream`] handles. It maps onto the shared
/// [`ReadStartAuthority`] contract rather than defining a separate lease model.
#[derive(Clone, Copy)]
pub enum RuntimeReadStartAuthority<'a> {
    /// The producer guarantees read-start completion before activation
    /// returns, so the consumer does not need an explicit stream wait.
    SynchronousByReturn,
    /// The consumer will begin reading on the supplied CUDA stream.
    ConsumerStream { stream: &'a CudaStream },
}

impl<'a> RuntimeReadStartAuthority<'a> {
    /// Use a synchronous-by-return read-start contract.
    pub const fn synchronous_by_return() -> Self {
        Self::SynchronousByReturn
    }

    /// Use a consumer-stream read-start contract.
    pub const fn consumer_stream(stream: &'a CudaStream) -> Self {
        Self::ConsumerStream { stream }
    }
}

impl fmt::Debug for RuntimeReadStartAuthority<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SynchronousByReturn => f.write_str("RuntimeReadStartAuthority::SynchronousByReturn"),
            Self::ConsumerStream { stream } => {
                f.debug_struct("RuntimeReadStartAuthority::ConsumerStream").field("device", &stream.device()).finish()
            },
        }
    }
}

/// CUDA-specific compatibility wrapper for shared consumer completion
/// authority.
///
/// This surface exists for callers that already traffic in
/// [`crate::runtime::CudaStream`] and [`crate::runtime::CudaEvent`] handles.
/// It maps onto the shared [`CompletionAuthority`] contract rather than
/// defining a separate lease model.
#[derive(Clone, Copy)]
pub enum RuntimeCompletionAuthority<'a> {
    /// Completion is guaranteed before the lease-completion call returns.
    SynchronousByReturn,
    /// Completion will be observed on the supplied CUDA stream.
    SameDeviceStream { stream: &'a CudaStream },
    /// Completion will be observed through the supplied CUDA event.
    CrossDeviceEvent { event: &'a CudaEvent },
}

impl<'a> RuntimeCompletionAuthority<'a> {
    /// Use a synchronous-by-return completion contract.
    pub const fn synchronous_by_return() -> Self {
        Self::SynchronousByReturn
    }

    /// Use a same-device stream completion contract.
    pub const fn same_device_stream(stream: &'a CudaStream) -> Self {
        Self::SameDeviceStream { stream }
    }

    /// Use a cross-device event completion contract.
    pub const fn cross_device_event(event: &'a CudaEvent) -> Self {
        Self::CrossDeviceEvent { event }
    }
}

impl fmt::Debug for RuntimeCompletionAuthority<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SynchronousByReturn => f.write_str("RuntimeCompletionAuthority::SynchronousByReturn"),
            Self::SameDeviceStream { stream } => f
                .debug_struct("RuntimeCompletionAuthority::SameDeviceStream")
                .field("device", &stream.device())
                .finish(),
            Self::CrossDeviceEvent { event } => {
                f.debug_struct("RuntimeCompletionAuthority::CrossDeviceEvent").field("device", &event.device()).finish()
            },
        }
    }
}

/// CUDA-backed compatibility witness for the shared completion contract.
///
/// Shared leases complete into [`CompletionWitness`]. The `Runtime*`
/// compatibility surface keeps returning a producer-owned [`CudaEvent`] so
/// existing `cudf-rs` runtime-oriented callers can inspect readiness and
/// synchronize through owned runtime handles while still following the same
/// completion and liveness semantics as the shared contract.
pub struct RuntimeCompletionWitness {
    kind: CompletionWitnessKind,
    event: CudaEvent,
}

impl RuntimeCompletionWitness {
    pub(super) fn new(kind: CompletionWitnessKind, event: CudaEvent) -> Self {
        Self { kind, event }
    }

    /// Return the shared completion-witness kind this runtime wrapper carries.
    pub fn kind(&self) -> CompletionWitnessKind {
        self.kind
    }

    /// Return the CUDA device that owns the producer completion event.
    pub fn device(&self) -> DeviceId {
        self.event.device()
    }

    /// Borrow the producer-owned CUDA event behind this witness.
    pub fn event(&self) -> &CudaEvent {
        &self.event
    }

    /// Check whether the producer-owned completion event has completed.
    pub fn is_ready(&self) -> Result<bool> {
        let _guard = DeviceGuard::new(self.device())?;
        self.event.is_ready().map_err(|error| error.with_operation(RUNTIME_COMPLETION_WITNESS_READY_OP))
    }

    /// Synchronize the calling thread with the producer-owned completion event.
    pub fn synchronize(&self) -> Result<()> {
        let _guard = DeviceGuard::new(self.device())?;
        self.event.synchronize().map_err(|error| error.with_operation(RUNTIME_COMPLETION_WITNESS_SYNC_OP))
    }
}

impl fmt::Debug for RuntimeCompletionWitness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RuntimeCompletionWitness").field("kind", &self.kind).field("device", &self.device()).finish()
    }
}

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

/// CUDA-specific compatibility wrapper for [`rapids_interop::LeaseCompletionResult`].
///
/// This keeps the shared completion outcome vocabulary but swaps the shared
/// witness object for [`RuntimeCompletionWitness`].
pub enum RuntimeLeaseCompletionResult {
    /// Completion produced a runtime-owned witness that can be queried or synchronized later.
    Completed(RuntimeCompletionWitness),
    /// Completion fell back to an immediate synchronization path instead of returning a witness.
    FallbackSynchronized,
}

impl fmt::Debug for RuntimeLeaseCompletionResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Completed(witness) => f.debug_tuple("Completed").field(witness).finish(),
            Self::FallbackSynchronized => f.write_str("FallbackSynchronized"),
        }
    }
}

/// # Safety
///
/// The `raw` pointers are lifetime-erased CUDA stream/event handles. They are
/// valid only while the `&'export Export` reference in `RuntimeActiveLeaseState`
/// is alive. Do not store these bindings beyond the export's lifetime.
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

pub(super) fn producer_readiness_binding(
    readiness: &ProducerReadiness<'_>,
    operation: &'static str,
) -> std::result::Result<ProducerReadinessBinding, InteropError> {
    match readiness {
        ProducerReadiness::DeviceStreamOrdered { stream } => {
            let device = stream.device();
            let device = DeviceId::new(device as usize).map_err(|error| {
                super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error)
            })?;
            Ok(ProducerReadinessBinding::Stream { raw: stream.as_raw(), device })
        },
        ProducerReadiness::HostVisible => Err(InteropError::new(
            InteropErrorKind::ProfileUnsupported,
            "runtime-backed prepared leases require explicit device-ordered producer readiness",
        )
        .with_context("operation", operation)
        .with_context("readiness", "host_visible")),
        ProducerReadiness::DeviceEventOrdered { event } => {
            let device = event.device();
            let device = DeviceId::new(device as usize).map_err(|error| {
                super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error)
            })?;
            Ok(ProducerReadinessBinding::Event { raw: event.as_raw(), device })
        },
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
            "runtime-backed prepared leases require explicit device-ordered producer readiness",
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

pub(super) fn runtime_completion_as_shared<'a>(
    completion: RuntimeCompletionAuthority<'a>,
    operation: &'static str,
) -> std::result::Result<CompletionAuthority<'a>, InteropError> {
    match completion {
        RuntimeCompletionAuthority::SynchronousByReturn => Ok(CompletionAuthority::SynchronousByReturn),
        RuntimeCompletionAuthority::SameDeviceStream { stream } => {
            let raw = stream.interop_raw_handle().map_err(|error| {
                super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error)
            })?;
            Ok(CompletionAuthority::SameDeviceStream {
                stream: CudaStreamHandle::from_raw_on_device(raw, stream.device().ordinal_raw())?,
            })
        },
        RuntimeCompletionAuthority::CrossDeviceEvent { event } => {
            let raw = event.interop_raw_handle().map_err(|error| {
                super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error)
            })?;
            Ok(CompletionAuthority::CrossDeviceEvent {
                event: CudaEventHandle::from_raw_on_device(raw, event.device().ordinal_raw())?,
            })
        },
    }
}

pub(super) fn runtime_read_start_as_shared<'a>(
    read_start: RuntimeReadStartAuthority<'a>,
    operation: &'static str,
) -> std::result::Result<ReadStartAuthority<'a>, InteropError> {
    match read_start {
        RuntimeReadStartAuthority::SynchronousByReturn => Ok(ReadStartAuthority::SynchronousByReturn),
        RuntimeReadStartAuthority::ConsumerStream { stream } => {
            let raw = stream.interop_raw_handle().map_err(|error| {
                super::super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error)
            })?;
            Ok(ReadStartAuthority::ConsumerStream {
                stream: CudaStreamHandle::from_raw_on_device(raw, stream.device().ordinal_raw())?,
            })
        },
    }
}

pub(super) fn create_producer_owned_event(producer_device: DeviceId) -> Result<CudaEvent> {
    let _guard = DeviceGuard::new(producer_device)?;
    CudaEvent::new()
}

pub(super) fn map_runtime_completion_interop_error(operation: &'static str, error: InteropError) -> error::Error {
    error::Error::from(error).with_operation(operation)
}

pub(super) fn runtime_completion_result_from_shared(
    result: rapids_interop::LeaseCompletionResult,
    operation: &'static str,
) -> Result<RuntimeLeaseCompletionResult> {
    match result {
        rapids_interop::LeaseCompletionResult::FallbackSynchronized => {
            Ok(RuntimeLeaseCompletionResult::FallbackSynchronized)
        },
        rapids_interop::LeaseCompletionResult::Completed(shared_witness) => {
            if let Some(state) = shared_witness.downcast_backend_ref::<SharedCudaCompletionWitnessState>() {
                return Ok(RuntimeLeaseCompletionResult::Completed(RuntimeCompletionWitness::new(
                    shared_witness.kind(),
                    state.event.clone(),
                )));
            }

            shared_witness.synchronize().map_err(|error| map_runtime_completion_interop_error(operation, error))?;
            Ok(RuntimeLeaseCompletionResult::FallbackSynchronized)
        },
    }
}
