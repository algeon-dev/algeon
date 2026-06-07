use std::{ffi::c_void, fmt, sync::Arc};

use rapids_interop::{
    ActiveLeaseActivation, ActiveLeaseBackend, CompletionAuthority, CompletionBinding, CompletionWitness,
    CompletionWitnessBackend, CompletionWitnessKind, CudaEventHandle, CudaStreamHandle, IncompleteDropBehavior,
    InteropError, InteropErrorKind, LeaseActivation, LeaseBackend, LeaseExportContext, LeaseReadiness,
    PreparedColumnExport, PreparedTableExport, ProducerReadiness,
};

use crate::{
    core::error::{self, Result},
    runtime::{CudaEvent, CudaStream, DeviceId, device_guard::DeviceGuard},
};

const ACTIVATE_RUNTIME_COLUMN_LEASE_OP: &str = "interop::PreparedColumnExport::try_activate_runtime_read_lease";
const ACTIVATE_RUNTIME_TABLE_LEASE_OP: &str = "interop::PreparedTableExport::try_activate_runtime_read_lease";
const COMPLETE_RUNTIME_COLUMN_LEASE_OP: &str = "interop::RuntimeActiveColumnReadLease::complete";
const COMPLETE_RUNTIME_TABLE_LEASE_OP: &str = "interop::RuntimeActiveTableReadLease::complete";
const DROP_RUNTIME_COLUMN_LEASE_OP: &str = "interop::RuntimeActiveColumnReadLease::drop";
const DROP_RUNTIME_TABLE_LEASE_OP: &str = "interop::RuntimeActiveTableReadLease::drop";
const RUNTIME_COMPLETION_WITNESS_READY_OP: &str = "interop::RuntimeCompletionWitness::is_ready";
const RUNTIME_COMPLETION_WITNESS_SYNC_OP: &str = "interop::RuntimeCompletionWitness::synchronize";
const SHARED_ACTIVATE_LEASE_OP: &str = "interop::PreparedExport::try_activate_read_lease";
const SHARED_COMPLETE_LEASE_OP: &str = "interop::ActiveReadLease::complete";
const SHARED_DROP_LEASE_OP: &str = "interop::ActiveReadLease::drop";
const SHARED_COMPLETION_WITNESS_READY_OP: &str = "interop::CompletionWitness::is_ready";
const SHARED_COMPLETION_WITNESS_SYNC_OP: &str = "interop::CompletionWitness::synchronize";

/// CUDA-specific compatibility wrapper for shared consumer completion
/// authority.
///
/// This surface exists for callers that already traffic in
/// [`crate::runtime::CudaStream`] and [`crate::runtime::CudaEvent`] handles.
/// It maps onto the shared [`CompletionAuthority`] contract rather than
/// defining a separate lease model.
#[derive(Clone, Copy)]
pub enum RuntimeCompletionAuthority<'a> {
    SynchronousByReturn,
    SameDeviceStream { stream: &'a CudaStream },
    CrossDeviceEvent { event: &'a CudaEvent },
}

impl<'a> RuntimeCompletionAuthority<'a> {
    pub const fn synchronous_by_return() -> Self {
        Self::SynchronousByReturn
    }

    pub const fn same_device_stream(stream: &'a CudaStream) -> Self {
        Self::SameDeviceStream { stream }
    }

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
    fn new(kind: CompletionWitnessKind, event: CudaEvent) -> Self {
        Self { kind, event }
    }

    pub fn kind(&self) -> CompletionWitnessKind {
        self.kind
    }

    pub fn device(&self) -> DeviceId {
        self.event.device()
    }

    pub fn event(&self) -> &CudaEvent {
        &self.event
    }

    pub fn is_ready(&self) -> Result<bool> {
        let _guard = DeviceGuard::new(self.device())?;
        self.event.is_ready().map_err(|error| error.with_operation(RUNTIME_COMPLETION_WITNESS_READY_OP))
    }

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

struct SharedCudaCompletionWitnessState {
    event: CudaEvent,
}

impl SharedCudaCompletionWitnessState {
    fn new(event: CudaEvent) -> Self {
        Self { event }
    }

    fn device(&self) -> DeviceId {
        self.event.device()
    }
}

impl CompletionWitnessBackend for SharedCudaCompletionWitnessState {
    fn is_ready(&self) -> rapids_interop::Result<bool> {
        let _guard = DeviceGuard::new(self.device()).map_err(|error| {
            super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, SHARED_COMPLETION_WITNESS_READY_OP, error)
        })?;
        self.event.is_ready().map_err(|error| {
            super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, SHARED_COMPLETION_WITNESS_READY_OP, error)
        })
    }

    fn synchronize(&self) -> rapids_interop::Result<()> {
        let _guard = DeviceGuard::new(self.device()).map_err(|error| {
            super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, SHARED_COMPLETION_WITNESS_SYNC_OP, error)
        })?;
        self.event.synchronize().map_err(|error| {
            super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, SHARED_COMPLETION_WITNESS_SYNC_OP, error)
        })
    }
}

/// CUDA-specific compatibility wrapper for [`rapids_interop::LeaseCompletionResult`].
///
/// This keeps the shared completion outcome vocabulary but swaps the shared
/// witness object for [`RuntimeCompletionWitness`].
pub enum RuntimeLeaseCompletionResult {
    Completed(RuntimeCompletionWitness),
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

#[derive(Clone, Copy, Debug)]
enum ProducerReadinessBinding {
    Stream { raw: *mut c_void, device: DeviceId },
    Event { raw: *mut c_void, device: DeviceId },
}

impl ProducerReadinessBinding {
    fn device(self) -> DeviceId {
        match self {
            Self::Stream { device, .. } | Self::Event { device, .. } => device,
        }
    }
}

fn producer_readiness_binding(
    readiness: &ProducerReadiness<'_>,
    operation: &'static str,
) -> std::result::Result<ProducerReadinessBinding, InteropError> {
    match readiness {
        ProducerReadiness::DeviceStreamOrdered { stream } => {
            let device = stream.device();
            let device = DeviceId::new(device as usize)
                .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
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
            let device = DeviceId::new(device as usize)
                .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
            Ok(ProducerReadinessBinding::Event { raw: event.as_raw(), device })
        },
    }
}

fn producer_readiness_binding_from_export(
    export: &LeaseExportContext,
    operation: &'static str,
) -> std::result::Result<ProducerReadinessBinding, InteropError> {
    match export.readiness() {
        LeaseReadiness::DeviceStreamOrdered { raw, device } => {
            let device = DeviceId::new(device as usize)
                .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
            Ok(ProducerReadinessBinding::Stream { raw, device })
        },
        LeaseReadiness::HostVisible => Err(InteropError::new(
            InteropErrorKind::ProfileUnsupported,
            "runtime-backed prepared leases require explicit device-ordered producer readiness",
        )
        .with_context("operation", operation)
        .with_context("readiness", "host_visible")),
        LeaseReadiness::DeviceEventOrdered { raw, device } => {
            let device = DeviceId::new(device as usize)
                .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
            Ok(ProducerReadinessBinding::Event { raw, device })
        },
    }
}

fn runtime_completion_as_shared<'a>(
    completion: RuntimeCompletionAuthority<'a>,
    operation: &'static str,
) -> std::result::Result<CompletionAuthority<'a>, InteropError> {
    match completion {
        RuntimeCompletionAuthority::SynchronousByReturn => Ok(CompletionAuthority::SynchronousByReturn),
        RuntimeCompletionAuthority::SameDeviceStream { stream } => {
            let raw = stream
                .interop_raw_handle()
                .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
            Ok(CompletionAuthority::SameDeviceStream {
                stream: CudaStreamHandle::from_raw_on_device(raw, stream.device().ordinal_raw())?,
            })
        },
        RuntimeCompletionAuthority::CrossDeviceEvent { event } => {
            let raw = event
                .interop_raw_handle()
                .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
            Ok(CompletionAuthority::CrossDeviceEvent {
                event: CudaEventHandle::from_raw_on_device(raw, event.device().ordinal_raw())?,
            })
        },
    }
}

fn create_producer_owned_witness(
    producer_device: DeviceId,
    completion_kind: CompletionWitnessKind,
) -> Result<RuntimeCompletionWitness> {
    let _guard = DeviceGuard::new(producer_device)?;
    let event = CudaEvent::new()?;
    Ok(RuntimeCompletionWitness::new(completion_kind, event))
}

fn create_producer_owned_event(producer_device: DeviceId) -> Result<CudaEvent> {
    let _guard = DeviceGuard::new(producer_device)?;
    CudaEvent::new()
}

fn complete_same_device_stream(
    producer_device: DeviceId,
    stream: &CudaStream,
    operation: &'static str,
) -> Result<RuntimeCompletionWitness> {
    let witness = create_producer_owned_witness(producer_device, CompletionWitnessKind::ConsumerStream)
        .map_err(|error| error.with_operation(operation))?;
    let _guard = DeviceGuard::new(producer_device)?;
    witness
        .event()
        .record(stream)
        .map_err(|error| error.with_operation(operation).with_context("completion", "same_device_stream"))?;
    Ok(witness)
}

fn complete_cross_device_event(
    producer_readiness: ProducerReadinessBinding,
    event: &CudaEvent,
    operation: &'static str,
) -> Result<RuntimeCompletionWitness> {
    let producer_device = producer_readiness.device();
    let witness = create_producer_owned_witness(producer_device, CompletionWitnessKind::ConsumerEvent)
        .map_err(|error| error.with_operation(operation))?;
    let consumer_event_raw = event
        .interop_raw_handle()
        .map_err(|error| error.with_operation(operation).with_context("completion", "cross_device_event"))?;
    let witness_event_raw = witness
        .event()
        .interop_raw_handle()
        .map_err(|error| error.with_operation(operation).with_context("completion", "cross_device_event"))?;

    let _guard = DeviceGuard::new(producer_device)?;
    match producer_readiness {
        ProducerReadinessBinding::Stream { raw: producer_stream_raw, .. } => {
            error::check_status_in_with(
                unsafe { cudf_sys::cudf_raw_cuda_stream_wait_event(producer_stream_raw, consumer_event_raw) },
                operation,
                [
                    ("completion", "cross_device_event".to_string()),
                    ("producer_device", producer_device.ordinal().to_string()),
                    ("consumer_device", event.device().ordinal().to_string()),
                ],
            )?;
            error::check_status_in_with(
                unsafe { cudf_sys::cudf_raw_cuda_event_record(witness_event_raw, producer_stream_raw) },
                operation,
                [
                    ("completion", "cross_device_event".to_string()),
                    ("producer_device", producer_device.ordinal().to_string()),
                    ("consumer_device", event.device().ordinal().to_string()),
                ],
            )?;
        },
        ProducerReadinessBinding::Event { raw: producer_event_raw, .. } => {
            let producer_stream = CudaStream::new().map_err(|error| error.with_operation(operation))?;
            let producer_stream_raw = producer_stream
                .interop_raw_handle()
                .map_err(|error| error.with_operation(operation).with_context("readiness", "device_event_ordered"))?;
            error::check_status_in_with(
                unsafe { cudf_sys::cudf_raw_cuda_stream_wait_event(producer_stream_raw, producer_event_raw) },
                operation,
                [
                    ("readiness", "device_event_ordered".to_string()),
                    ("producer_device", producer_device.ordinal().to_string()),
                ],
            )?;
            error::check_status_in_with(
                unsafe { cudf_sys::cudf_raw_cuda_stream_wait_event(producer_stream_raw, consumer_event_raw) },
                operation,
                [
                    ("completion", "cross_device_event".to_string()),
                    ("readiness", "device_event_ordered".to_string()),
                    ("producer_device", producer_device.ordinal().to_string()),
                    ("consumer_device", event.device().ordinal().to_string()),
                ],
            )?;
            error::check_status_in_with(
                unsafe { cudf_sys::cudf_raw_cuda_event_record(witness_event_raw, producer_stream_raw) },
                operation,
                [
                    ("completion", "cross_device_event".to_string()),
                    ("readiness", "device_event_ordered".to_string()),
                    ("producer_device", producer_device.ordinal().to_string()),
                    ("consumer_device", event.device().ordinal().to_string()),
                ],
            )?;
        },
    }
    Ok(witness)
}

fn complete_same_device_stream_shared(
    producer_device: DeviceId,
    consumer_stream_raw: *mut c_void,
    consumer_device: DeviceId,
    operation: &'static str,
) -> rapids_interop::Result<CompletionWitness> {
    let shared_event = create_producer_owned_event(producer_device)
        .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    let witness_event_raw = shared_event
        .interop_raw_handle()
        .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    let witness = CompletionWitness::with_backend(
        CompletionWitnessKind::ConsumerStream,
        Arc::new(SharedCudaCompletionWitnessState::new(shared_event)),
    );
    let _guard = DeviceGuard::new(producer_device)
        .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    error::check_status_in_with(
        unsafe { cudf_sys::cudf_raw_cuda_event_record(witness_event_raw, consumer_stream_raw) },
        operation,
        [
            ("completion", "same_device_stream".to_string()),
            ("producer_device", producer_device.ordinal().to_string()),
            ("consumer_device", consumer_device.ordinal().to_string()),
        ],
    )
    .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    Ok(witness)
}

fn complete_cross_device_event_shared(
    producer_readiness: ProducerReadinessBinding,
    consumer_event_raw: *mut c_void,
    consumer_device: DeviceId,
    operation: &'static str,
) -> rapids_interop::Result<CompletionWitness> {
    let producer_device = producer_readiness.device();
    let shared_state = create_producer_owned_event(producer_device)
        .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    let witness_event_raw = shared_state
        .interop_raw_handle()
        .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    let witness = CompletionWitness::with_backend(
        CompletionWitnessKind::ConsumerEvent,
        Arc::new(SharedCudaCompletionWitnessState::new(shared_state)),
    );

    let _guard = DeviceGuard::new(producer_device)
        .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    match producer_readiness {
        ProducerReadinessBinding::Stream { raw: producer_stream_raw, .. } => {
            error::check_status_in_with(
                unsafe { cudf_sys::cudf_raw_cuda_stream_wait_event(producer_stream_raw, consumer_event_raw) },
                operation,
                [
                    ("completion", "cross_device_event".to_string()),
                    ("producer_device", producer_device.ordinal().to_string()),
                    ("consumer_device", consumer_device.ordinal().to_string()),
                ],
            )
            .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
            error::check_status_in_with(
                unsafe { cudf_sys::cudf_raw_cuda_event_record(witness_event_raw, producer_stream_raw) },
                operation,
                [
                    ("completion", "cross_device_event".to_string()),
                    ("producer_device", producer_device.ordinal().to_string()),
                    ("consumer_device", consumer_device.ordinal().to_string()),
                ],
            )
            .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
        },
        ProducerReadinessBinding::Event { raw: producer_event_raw, .. } => {
            let producer_stream = CudaStream::new()
                .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
            let producer_stream_raw = producer_stream
                .interop_raw_handle()
                .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
            error::check_status_in_with(
                unsafe { cudf_sys::cudf_raw_cuda_stream_wait_event(producer_stream_raw, producer_event_raw) },
                operation,
                [
                    ("readiness", "device_event_ordered".to_string()),
                    ("producer_device", producer_device.ordinal().to_string()),
                ],
            )
            .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
            error::check_status_in_with(
                unsafe { cudf_sys::cudf_raw_cuda_stream_wait_event(producer_stream_raw, consumer_event_raw) },
                operation,
                [
                    ("completion", "cross_device_event".to_string()),
                    ("readiness", "device_event_ordered".to_string()),
                    ("producer_device", producer_device.ordinal().to_string()),
                    ("consumer_device", consumer_device.ordinal().to_string()),
                ],
            )
            .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
            error::check_status_in_with(
                unsafe { cudf_sys::cudf_raw_cuda_event_record(witness_event_raw, producer_stream_raw) },
                operation,
                [
                    ("completion", "cross_device_event".to_string()),
                    ("readiness", "device_event_ordered".to_string()),
                    ("producer_device", producer_device.ordinal().to_string()),
                    ("consumer_device", consumer_device.ordinal().to_string()),
                ],
            )
            .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
        },
    }
    Ok(witness)
}

fn synchronize_same_device_stream_raw(
    stream_raw: *mut c_void,
    stream_device: DeviceId,
    operation: &'static str,
) -> Result<()> {
    let event = create_producer_owned_event(stream_device).map_err(|error| error.with_operation(operation))?;
    let event_raw = event.interop_raw_handle().map_err(|error| error.with_operation(operation))?;
    let _guard = DeviceGuard::new(stream_device)?;
    error::check_status_in_with(
        unsafe { cudf_sys::cudf_raw_cuda_event_record(event_raw, stream_raw) },
        operation,
        [("completion", "same_device_stream".to_string()), ("consumer_device", stream_device.ordinal().to_string())],
    )?;
    event
        .synchronize()
        .map_err(|error| error.with_operation(operation).with_context("completion", "same_device_stream"))
}

fn synchronize_cross_device_event_raw(
    event_raw: *mut c_void,
    consumer_device: DeviceId,
    operation: &'static str,
) -> Result<()> {
    let _guard = DeviceGuard::new(consumer_device)?;
    let stream = CudaStream::new().map_err(|error| error.with_operation(operation))?;
    let stream_raw = stream.interop_raw_handle().map_err(|error| error.with_operation(operation))?;
    let completion = CudaEvent::new().map_err(|error| error.with_operation(operation))?;
    let completion_raw = completion.interop_raw_handle().map_err(|error| error.with_operation(operation))?;
    error::check_status_in_with(
        unsafe { cudf_sys::cudf_raw_cuda_stream_wait_event(stream_raw, event_raw) },
        operation,
        [("completion", "cross_device_event".to_string()), ("consumer_device", consumer_device.ordinal().to_string())],
    )?;
    error::check_status_in_with(
        unsafe { cudf_sys::cudf_raw_cuda_event_record(completion_raw, stream_raw) },
        operation,
        [("completion", "cross_device_event".to_string()), ("consumer_device", consumer_device.ordinal().to_string())],
    )?;
    completion
        .synchronize()
        .map_err(|error| error.with_operation(operation).with_context("completion", "cross_device_event"))
}

fn order_consumer_stream_after_producer_event_raw(
    producer_event_raw: *mut c_void,
    producer_device: DeviceId,
    consumer_stream_raw: *mut c_void,
    consumer_device: DeviceId,
    operation: &'static str,
) -> std::result::Result<(), InteropError> {
    let _guard = DeviceGuard::new(producer_device)
        .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    error::check_status_in_with(
        unsafe { cudf_sys::cudf_raw_cuda_stream_wait_event(consumer_stream_raw, producer_event_raw) },
        operation,
        [
            ("readiness", "device_event_ordered".to_string()),
            ("completion", "same_device_stream".to_string()),
            ("producer_device", producer_device.ordinal().to_string()),
            ("consumer_device", consumer_device.ordinal().to_string()),
        ],
    )
    .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))
}

fn order_consumer_stream_after_producer_stream_raw(
    producer_stream_raw: *mut c_void,
    producer_device: DeviceId,
    consumer_stream_raw: *mut c_void,
    consumer_device: DeviceId,
    operation: &'static str,
) -> std::result::Result<CudaEvent, InteropError> {
    let _guard = DeviceGuard::new(producer_device)
        .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    let barrier = CudaEvent::new().map_err(|error| {
        super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error)
            .with_context("readiness", "device_stream_ordered")
            .with_context("completion", "same_device_stream")
    })?;
    let barrier_raw = barrier.interop_raw_handle().map_err(|error| {
        super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error)
            .with_context("readiness", "device_stream_ordered")
            .with_context("completion", "same_device_stream")
    })?;
    error::check_status_in_with(
        unsafe { cudf_sys::cudf_raw_cuda_event_record(barrier_raw, producer_stream_raw) },
        operation,
        [
            ("readiness", "device_stream_ordered".to_string()),
            ("completion", "same_device_stream".to_string()),
            ("producer_device", producer_device.ordinal().to_string()),
        ],
    )
    .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    error::check_status_in_with(
        unsafe { cudf_sys::cudf_raw_cuda_stream_wait_event(consumer_stream_raw, barrier_raw) },
        operation,
        [
            ("readiness", "device_stream_ordered".to_string()),
            ("completion", "same_device_stream".to_string()),
            ("producer_device", producer_device.ordinal().to_string()),
            ("consumer_device", consumer_device.ordinal().to_string()),
        ],
    )
    .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    Ok(barrier)
}

fn order_consumer_stream_after_producer_readiness_raw(
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

fn synchronize_completion_fallback(completion: RuntimeCompletionAuthority<'_>, operation: &'static str) -> Result<()> {
    match completion {
        RuntimeCompletionAuthority::SynchronousByReturn => Ok(()),
        RuntimeCompletionAuthority::SameDeviceStream { stream } => {
            let _guard = DeviceGuard::new(stream.device())?;
            stream
                .synchronize()
                .map_err(|error| error.with_operation(operation).with_context("completion", "same_device_stream"))
        },
        RuntimeCompletionAuthority::CrossDeviceEvent { event } => {
            let _guard = DeviceGuard::new(event.device())?;
            event
                .synchronize()
                .map_err(|error| error.with_operation(operation).with_context("completion", "cross_device_event"))
        },
    }
}

struct SharedCudaLeaseActivation {
    _barrier: Option<CudaEvent>,
}

impl ActiveLeaseActivation for SharedCudaLeaseActivation {}

#[derive(Debug, Clone, Copy)]
struct SharedCudaLeaseBackend {
    incomplete_drop_behavior: IncompleteDropBehavior,
}

impl ActiveLeaseBackend for SharedCudaLeaseBackend {
    fn activate(
        &self,
        export: &LeaseExportContext,
        completion: CompletionBinding,
    ) -> rapids_interop::Result<LeaseActivation> {
        let CompletionBinding::SameDeviceStream { raw, device } = completion else {
            return Ok(LeaseActivation::noop());
        };

        let producer_readiness = producer_readiness_binding_from_export(export, SHARED_ACTIVATE_LEASE_OP)?;
        let consumer_device = DeviceId::new(device as usize).map_err(|error| {
            super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, SHARED_ACTIVATE_LEASE_OP, error)
        })?;
        let barrier = order_consumer_stream_after_producer_readiness_raw(
            producer_readiness,
            raw,
            consumer_device,
            SHARED_ACTIVATE_LEASE_OP,
        )?;

        Ok(match barrier {
            Some(barrier) => {
                LeaseActivation::with_state(Arc::new(SharedCudaLeaseActivation { _barrier: Some(barrier) }))
            },
            None => LeaseActivation::noop(),
        })
    }

    fn incomplete_drop_behavior(&self) -> IncompleteDropBehavior {
        self.incomplete_drop_behavior
    }

    fn complete(
        &self,
        export: &LeaseExportContext,
        completion: CompletionBinding,
    ) -> rapids_interop::Result<rapids_interop::LeaseCompletionResult> {
        match completion {
            CompletionBinding::SynchronousByReturn => Ok(rapids_interop::LeaseCompletionResult::FallbackSynchronized),
            CompletionBinding::SameDeviceStream { raw, device } => {
                let producer_device = export.producer_device().ok_or_else(|| {
                    InteropError::new(
                        InteropErrorKind::ReadinessUnavailable,
                        "same-device stream completion requires a concrete producer CUDA device",
                    )
                    .with_context("operation", SHARED_COMPLETE_LEASE_OP)
                    .with_context("completion", "same_device_stream")
                })?;
                let producer_device = DeviceId::new(producer_device as usize).map_err(|error| {
                    super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, SHARED_COMPLETE_LEASE_OP, error)
                })?;
                let consumer_device = DeviceId::new(device as usize).map_err(|error| {
                    super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, SHARED_COMPLETE_LEASE_OP, error)
                })?;
                complete_same_device_stream_shared(producer_device, raw, consumer_device, SHARED_COMPLETE_LEASE_OP)
                    .map(rapids_interop::LeaseCompletionResult::Completed)
            },
            CompletionBinding::CrossDeviceEvent { raw, device } => {
                let producer_readiness = producer_readiness_binding_from_export(export, SHARED_COMPLETE_LEASE_OP)?;
                let consumer_device = DeviceId::new(device as usize).map_err(|error| {
                    super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, SHARED_COMPLETE_LEASE_OP, error)
                })?;
                complete_cross_device_event_shared(producer_readiness, raw, consumer_device, SHARED_COMPLETE_LEASE_OP)
                    .map(rapids_interop::LeaseCompletionResult::Completed)
            },
        }
    }

    fn drop_incomplete(
        &self,
        _export: &LeaseExportContext,
        completion: CompletionBinding,
    ) -> rapids_interop::Result<()> {
        match completion {
            CompletionBinding::SynchronousByReturn => Ok(()),
            CompletionBinding::SameDeviceStream { raw, device } => {
                let device = DeviceId::new(device as usize).map_err(|error| {
                    super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, SHARED_DROP_LEASE_OP, error)
                })?;
                synchronize_same_device_stream_raw(raw, device, SHARED_DROP_LEASE_OP).map_err(|error| {
                    super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, SHARED_DROP_LEASE_OP, error)
                })
            },
            CompletionBinding::CrossDeviceEvent { raw, device } => {
                let device = DeviceId::new(device as usize).map_err(|error| {
                    super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, SHARED_DROP_LEASE_OP, error)
                })?;
                synchronize_cross_device_event_raw(raw, device, SHARED_DROP_LEASE_OP).map_err(|error| {
                    super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, SHARED_DROP_LEASE_OP, error)
                })
            },
        }
    }
}

fn shared_cuda_lease_backend_with_behavior(incomplete_drop_behavior: IncompleteDropBehavior) -> LeaseBackend {
    LeaseBackend::new(SharedCudaLeaseBackend { incomplete_drop_behavior })
}

pub(crate) fn shared_cuda_lease_backend() -> LeaseBackend {
    shared_cuda_lease_backend_with_behavior(IncompleteDropBehavior::GuaranteedFallback)
}

/// CUDA-backed prepared-lease backend that requires explicit completion.
///
/// This is an opt-in backend for callers that want the real CUDA runtime
/// activation and witness path but do not want abandoned leases to
/// conservatively synchronize during `drop`.
pub fn cuda_explicit_completion_required_lease_backend() -> LeaseBackend {
    shared_cuda_lease_backend_with_behavior(IncompleteDropBehavior::ExplicitCompletionRequired)
}

/// CUDA-specific compatibility wrapper over a shared active column read lease.
///
/// Activation still goes through [`PreparedColumnExport::try_activate_read_lease`]
/// under the hood, so export-root single-active-lease coordination, shared
/// liveness retention, and backend-declared incomplete-drop behavior all stay
/// aligned with the shared contract. This wrapper only changes the completion
/// surface exposed to CUDA-runtime-oriented callers.
#[must_use = "lease must be completed explicitly via .complete(); dropping without completion falls back to conservative synchronization"]
pub struct RuntimeActiveColumnReadLease<'export, 'producer, 'consumer> {
    export: &'export PreparedColumnExport<'producer>,
    completion: RuntimeCompletionAuthority<'consumer>,
    producer_readiness: ProducerReadinessBinding,
    incomplete_drop_behavior: IncompleteDropBehavior,
    shared_lease: Option<super::ActiveColumnReadLease<'export, 'producer, 'consumer>>,
    completed: bool,
}

impl<'export, 'producer, 'consumer> RuntimeActiveColumnReadLease<'export, 'producer, 'consumer> {
    pub fn export(&self) -> &'export PreparedColumnExport<'producer> {
        self.export
    }

    pub fn completion(&self) -> &RuntimeCompletionAuthority<'consumer> {
        &self.completion
    }

    pub fn complete(mut self) -> Result<RuntimeLeaseCompletionResult> {
        let result = match self.completion {
            RuntimeCompletionAuthority::SynchronousByReturn => RuntimeLeaseCompletionResult::FallbackSynchronized,
            RuntimeCompletionAuthority::SameDeviceStream { stream } => {
                RuntimeLeaseCompletionResult::Completed(complete_same_device_stream(
                    self.producer_readiness.device(),
                    stream,
                    COMPLETE_RUNTIME_COLUMN_LEASE_OP,
                )?)
            },
            RuntimeCompletionAuthority::CrossDeviceEvent { event } => RuntimeLeaseCompletionResult::Completed(
                complete_cross_device_event(self.producer_readiness, event, COMPLETE_RUNTIME_COLUMN_LEASE_OP)?,
            ),
        };
        if let Some(shared_lease) = self.shared_lease.take() {
            shared_lease.release_after_external_completion();
        }
        self.completed = true;
        Ok(result)
    }
}

impl Drop for RuntimeActiveColumnReadLease<'_, '_, '_> {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        if matches!(self.incomplete_drop_behavior, IncompleteDropBehavior::GuaranteedFallback) {
            let _ = synchronize_completion_fallback(self.completion, DROP_RUNTIME_COLUMN_LEASE_OP);
        }
    }
}

impl fmt::Debug for RuntimeActiveColumnReadLease<'_, '_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RuntimeActiveColumnReadLease")
            .field("completion", &self.completion)
            .field("producer_device", &self.producer_readiness.device())
            .finish()
    }
}

/// CUDA-specific compatibility wrapper over a shared active table read lease.
///
/// Activation still goes through [`PreparedTableExport::try_activate_read_lease`]
/// under the hood, so export-root single-active-lease coordination, shared
/// liveness retention, and backend-declared incomplete-drop behavior all stay
/// aligned with the shared contract. This wrapper only changes the completion
/// surface exposed to CUDA-runtime-oriented callers.
#[must_use = "lease must be completed explicitly via .complete(); dropping without completion falls back to conservative synchronization"]
pub struct RuntimeActiveTableReadLease<'export, 'producer, 'consumer> {
    export: &'export PreparedTableExport<'producer>,
    completion: RuntimeCompletionAuthority<'consumer>,
    producer_readiness: ProducerReadinessBinding,
    incomplete_drop_behavior: IncompleteDropBehavior,
    shared_lease: Option<super::ActiveTableReadLease<'export, 'producer, 'consumer>>,
    completed: bool,
}

impl<'export, 'producer, 'consumer> RuntimeActiveTableReadLease<'export, 'producer, 'consumer> {
    pub fn export(&self) -> &'export PreparedTableExport<'producer> {
        self.export
    }

    pub fn completion(&self) -> &RuntimeCompletionAuthority<'consumer> {
        &self.completion
    }

    pub fn complete(mut self) -> Result<RuntimeLeaseCompletionResult> {
        let result = match self.completion {
            RuntimeCompletionAuthority::SynchronousByReturn => RuntimeLeaseCompletionResult::FallbackSynchronized,
            RuntimeCompletionAuthority::SameDeviceStream { stream } => RuntimeLeaseCompletionResult::Completed(
                complete_same_device_stream(self.producer_readiness.device(), stream, COMPLETE_RUNTIME_TABLE_LEASE_OP)?,
            ),
            RuntimeCompletionAuthority::CrossDeviceEvent { event } => RuntimeLeaseCompletionResult::Completed(
                complete_cross_device_event(self.producer_readiness, event, COMPLETE_RUNTIME_TABLE_LEASE_OP)?,
            ),
        };
        if let Some(shared_lease) = self.shared_lease.take() {
            shared_lease.release_after_external_completion();
        }
        self.completed = true;
        Ok(result)
    }
}

impl Drop for RuntimeActiveTableReadLease<'_, '_, '_> {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        if matches!(self.incomplete_drop_behavior, IncompleteDropBehavior::GuaranteedFallback) {
            let _ = synchronize_completion_fallback(self.completion, DROP_RUNTIME_TABLE_LEASE_OP);
        }
    }
}

impl fmt::Debug for RuntimeActiveTableReadLease<'_, '_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RuntimeActiveTableReadLease")
            .field("completion", &self.completion)
            .field("producer_device", &self.producer_readiness.device())
            .finish()
    }
}

/// CUDA-specific compatibility activation over a prepared column export.
///
/// Prefer [`PreparedColumnExport::try_activate_read_lease`] for new shared
/// interop APIs. Use this trait when you specifically need the
/// `cudf-rs`-owned runtime witness surface.
pub trait TryActivateRuntimeColumnReadLease {
    fn try_activate_runtime_read_lease<'consumer>(
        &self,
        completion: RuntimeCompletionAuthority<'consumer>,
    ) -> std::result::Result<RuntimeActiveColumnReadLease<'_, '_, 'consumer>, InteropError>;
}

impl<'producer> TryActivateRuntimeColumnReadLease for PreparedColumnExport<'producer> {
    fn try_activate_runtime_read_lease<'consumer>(
        &self,
        completion: RuntimeCompletionAuthority<'consumer>,
    ) -> std::result::Result<RuntimeActiveColumnReadLease<'_, '_, 'consumer>, InteropError> {
        let shared_completion = runtime_completion_as_shared(completion, ACTIVATE_RUNTIME_COLUMN_LEASE_OP)?;
        let shared_lease = self.try_activate_read_lease(shared_completion)?;
        let incomplete_drop_behavior = shared_lease.incomplete_drop_behavior();
        let producer_readiness = producer_readiness_binding(self.readiness(), ACTIVATE_RUNTIME_COLUMN_LEASE_OP)?;
        Ok(RuntimeActiveColumnReadLease {
            export: self,
            completion,
            producer_readiness,
            incomplete_drop_behavior,
            shared_lease: Some(shared_lease),
            completed: false,
        })
    }
}

/// CUDA-specific compatibility activation over a prepared table export.
///
/// Prefer [`PreparedTableExport::try_activate_read_lease`] for new shared
/// interop APIs. Use this trait when you specifically need the
/// `cudf-rs`-owned runtime witness surface.
pub trait TryActivateRuntimeTableReadLease {
    fn try_activate_runtime_read_lease<'consumer>(
        &self,
        completion: RuntimeCompletionAuthority<'consumer>,
    ) -> std::result::Result<RuntimeActiveTableReadLease<'_, '_, 'consumer>, InteropError>;
}

impl<'producer> TryActivateRuntimeTableReadLease for PreparedTableExport<'producer> {
    fn try_activate_runtime_read_lease<'consumer>(
        &self,
        completion: RuntimeCompletionAuthority<'consumer>,
    ) -> std::result::Result<RuntimeActiveTableReadLease<'_, '_, 'consumer>, InteropError> {
        let shared_completion = runtime_completion_as_shared(completion, ACTIVATE_RUNTIME_TABLE_LEASE_OP)?;
        let shared_lease = self.try_activate_read_lease(shared_completion)?;
        let incomplete_drop_behavior = shared_lease.incomplete_drop_behavior();
        let producer_readiness = producer_readiness_binding(self.readiness(), ACTIVATE_RUNTIME_TABLE_LEASE_OP)?;
        Ok(RuntimeActiveTableReadLease {
            export: self,
            completion,
            producer_readiness,
            incomplete_drop_behavior,
            shared_lease: Some(shared_lease),
            completed: false,
        })
    }
}
