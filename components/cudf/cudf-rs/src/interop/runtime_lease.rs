use std::{ffi::c_void, fmt, sync::Arc};

use rapids_interop::{
    ActiveLeaseActivation, ActiveLeaseBackend, CompletionAuthority, CompletionBinding, CompletionWitness,
    CompletionWitnessBackend, CompletionWitnessKind, CudaEventHandle, CudaStreamHandle, IncompleteDropBehavior,
    InteropError, InteropErrorKind, LeaseActivation, LeaseBackend, LeaseExportContext, LeaseReadiness,
    PreparedColumnExport, PreparedTableExport, ProducerReadiness, ReadStartAuthority, ReadStartBinding,
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
    fn new(kind: CompletionWitnessKind, event: CudaEvent) -> Self {
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
/// The `raw` pointers are lifetime-erased CUDA stream/event handles.  They
/// are valid only while the `&'export Export` reference in
/// `RuntimeActiveLeaseState` is alive.  Do not store these bindings beyond
/// the export's lifetime.
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

fn runtime_read_start_as_shared<'a>(
    read_start: RuntimeReadStartAuthority<'a>,
    operation: &'static str,
) -> std::result::Result<ReadStartAuthority<'a>, InteropError> {
    match read_start {
        RuntimeReadStartAuthority::SynchronousByReturn => Ok(ReadStartAuthority::SynchronousByReturn),
        RuntimeReadStartAuthority::ConsumerStream { stream } => {
            let raw = stream
                .interop_raw_handle()
                .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
            Ok(ReadStartAuthority::ConsumerStream {
                stream: CudaStreamHandle::from_raw_on_device(raw, stream.device().ordinal_raw())?,
            })
        },
    }
}

fn create_producer_owned_event(producer_device: DeviceId) -> Result<CudaEvent> {
    let _guard = DeviceGuard::new(producer_device)?;
    CudaEvent::new()
}

fn map_runtime_completion_interop_error(operation: &'static str, error: InteropError) -> error::Error {
    error::Error::from(error).with_operation(operation)
}

fn runtime_completion_result_from_shared(
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
            ("read_start", "consumer_stream".to_string()),
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
            .with_context("read_start", "consumer_stream")
    })?;
    let barrier_raw = barrier.interop_raw_handle().map_err(|error| {
        super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error)
            .with_context("readiness", "device_stream_ordered")
            .with_context("read_start", "consumer_stream")
    })?;
    error::check_status_in_with(
        unsafe { cudf_sys::cudf_raw_cuda_event_record(barrier_raw, producer_stream_raw) },
        operation,
        [
            ("readiness", "device_stream_ordered".to_string()),
            ("read_start", "consumer_stream".to_string()),
            ("producer_device", producer_device.ordinal().to_string()),
        ],
    )
    .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    error::check_status_in_with(
        unsafe { cudf_sys::cudf_raw_cuda_stream_wait_event(consumer_stream_raw, barrier_raw) },
        operation,
        [
            ("readiness", "device_stream_ordered".to_string()),
            ("read_start", "consumer_stream".to_string()),
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

fn synchronize_producer_stream_readiness_raw(
    producer_stream_raw: *mut c_void,
    producer_device: DeviceId,
    operation: &'static str,
) -> std::result::Result<(), InteropError> {
    let _guard = DeviceGuard::new(producer_device)
        .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    let barrier = CudaEvent::new()
        .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    let barrier_raw = barrier
        .interop_raw_handle()
        .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    error::check_status_in_with(
        unsafe { cudf_sys::cudf_raw_cuda_event_record(barrier_raw, producer_stream_raw) },
        operation,
        [
            ("readiness", "device_stream_ordered".to_string()),
            ("read_start", "synchronous_by_return".to_string()),
            ("producer_device", producer_device.ordinal().to_string()),
        ],
    )
    .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    barrier
        .synchronize()
        .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))
}

fn synchronize_producer_event_readiness_raw(
    producer_event_raw: *mut c_void,
    producer_device: DeviceId,
    operation: &'static str,
) -> std::result::Result<(), InteropError> {
    let _guard = DeviceGuard::new(producer_device)
        .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    let stream = CudaStream::new()
        .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    let stream_raw = stream
        .interop_raw_handle()
        .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    let barrier = CudaEvent::new()
        .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    let barrier_raw = barrier
        .interop_raw_handle()
        .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    error::check_status_in_with(
        unsafe { cudf_sys::cudf_raw_cuda_stream_wait_event(stream_raw, producer_event_raw) },
        operation,
        [
            ("readiness", "device_event_ordered".to_string()),
            ("read_start", "synchronous_by_return".to_string()),
            ("producer_device", producer_device.ordinal().to_string()),
        ],
    )
    .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    error::check_status_in_with(
        unsafe { cudf_sys::cudf_raw_cuda_event_record(barrier_raw, stream_raw) },
        operation,
        [
            ("readiness", "device_event_ordered".to_string()),
            ("read_start", "synchronous_by_return".to_string()),
            ("producer_device", producer_device.ordinal().to_string()),
        ],
    )
    .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
    barrier
        .synchronize()
        .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))
}

fn synchronize_producer_readiness(
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
    completion: RuntimeCompletionAuthority<'_>,
    operation: &'static str,
) -> Result<()> {
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

pub(crate) fn synchronize_runtime_release_completion(
    completion: RuntimeCompletionAuthority<'_>,
    release_device: i32,
    operation: &'static str,
) -> std::result::Result<(), InteropError> {
    match completion {
        RuntimeCompletionAuthority::SynchronousByReturn => Err(InteropError::new(
            InteropErrorKind::ProfileUnsupported,
            "runtime-backed owned-transfer release requires a consumer signal completion authority",
        )
        .with_context("operation", operation)
        .with_context("completion", "synchronous_by_return")
        .with_context("release_device", release_device.to_string())),
        RuntimeCompletionAuthority::SameDeviceStream { stream } => {
            let stream_device = stream.device().ordinal_raw() as i32;
            if stream_device != release_device {
                return Err(InteropError::new(
                    InteropErrorKind::ReadinessMismatch,
                    "runtime-backed owned-transfer release stream must live on the release CUDA device",
                )
                .with_context("operation", operation)
                .with_context("completion", "same_device_stream")
                .with_context("release_device", release_device.to_string())
                .with_context("stream_device", stream_device.to_string()));
            }
            let raw = stream
                .interop_raw_handle()
                .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
            synchronize_same_device_stream_raw(raw, stream.device(), operation)
                .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))
        },
        RuntimeCompletionAuthority::CrossDeviceEvent { event } => {
            let raw = event
                .interop_raw_handle()
                .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
            synchronize_cross_device_event_raw(raw, event.device(), operation)
                .map_err(|error| super::map_cudf_error(InteropErrorKind::ReadinessUnavailable, operation, error))
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
        read_start: ReadStartBinding,
        completion: CompletionBinding,
    ) -> rapids_interop::Result<LeaseActivation> {
        let producer_readiness = producer_readiness_binding_from_export(export, SHARED_ACTIVATE_LEASE_OP)?;
        let _ = completion;

        match read_start {
            ReadStartBinding::SynchronousByReturn => {
                synchronize_producer_readiness(producer_readiness, SHARED_ACTIVATE_LEASE_OP)?;
                Ok(LeaseActivation::noop())
            },
            ReadStartBinding::ConsumerStream { raw, device } => {
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
            },
        }
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
///
/// Callers must ensure that every activated lease is completed explicitly via
/// `.complete()` before the producer's underlying buffers are invalidated.
/// Dropping an incomplete lease with this backend does not synchronize, which
/// can cause GPU use-after-free if the consumer is still accessing producer
/// memory.
pub fn cuda_explicit_completion_required_lease_backend() -> LeaseBackend {
    shared_cuda_lease_backend_with_behavior(IncompleteDropBehavior::ExplicitCompletionRequired)
}

trait RuntimeSharedLease {
    fn incomplete_drop_behavior(&self) -> IncompleteDropBehavior;
    fn complete(self) -> rapids_interop::Result<rapids_interop::LeaseCompletionResult>;
}

impl RuntimeSharedLease for super::ActiveColumnReadLease<'_, '_, '_> {
    fn incomplete_drop_behavior(&self) -> IncompleteDropBehavior {
        super::ActiveColumnReadLease::incomplete_drop_behavior(self)
    }

    fn complete(self) -> rapids_interop::Result<rapids_interop::LeaseCompletionResult> {
        super::ActiveColumnReadLease::complete(self)
    }
}

impl RuntimeSharedLease for super::ActiveTableReadLease<'_, '_, '_> {
    fn incomplete_drop_behavior(&self) -> IncompleteDropBehavior {
        super::ActiveTableReadLease::incomplete_drop_behavior(self)
    }

    fn complete(self) -> rapids_interop::Result<rapids_interop::LeaseCompletionResult> {
        super::ActiveTableReadLease::complete(self)
    }
}

struct RuntimeActiveLeaseState<'export, 'consumer, Export, SharedLease: RuntimeSharedLease> {
    export: &'export Export,
    read_start: RuntimeReadStartAuthority<'consumer>,
    completion: RuntimeCompletionAuthority<'consumer>,
    producer_readiness: ProducerReadinessBinding,
    incomplete_drop_behavior: IncompleteDropBehavior,
    shared_lease: Option<SharedLease>,
    complete_operation: &'static str,
    drop_operation: &'static str,
    completed: bool,
}

impl<'export, 'consumer, Export, SharedLease: RuntimeSharedLease>
    RuntimeActiveLeaseState<'export, 'consumer, Export, SharedLease>
{
    fn new(
        export: &'export Export,
        read_start: RuntimeReadStartAuthority<'consumer>,
        completion: RuntimeCompletionAuthority<'consumer>,
        producer_readiness: ProducerReadinessBinding,
        shared_lease: SharedLease,
        complete_operation: &'static str,
        drop_operation: &'static str,
    ) -> Self {
        let incomplete_drop_behavior = shared_lease.incomplete_drop_behavior();
        Self {
            export,
            read_start,
            completion,
            producer_readiness,
            incomplete_drop_behavior,
            shared_lease: Some(shared_lease),
            complete_operation,
            drop_operation,
            completed: false,
        }
    }

    fn export(&self) -> &'export Export {
        self.export
    }

    fn read_start(&self) -> &RuntimeReadStartAuthority<'consumer> {
        &self.read_start
    }

    fn completion(&self) -> &RuntimeCompletionAuthority<'consumer> {
        &self.completion
    }

    fn producer_device(&self) -> DeviceId {
        self.producer_readiness.device()
    }

    fn complete(mut self) -> Result<RuntimeLeaseCompletionResult> {
        let shared_lease = self.shared_lease.take().expect("runtime lease missing shared lease state");
        let shared_result = shared_lease
            .complete()
            .map_err(|error| map_runtime_completion_interop_error(self.complete_operation, error))?;
        let result = runtime_completion_result_from_shared(shared_result, self.complete_operation)?;
        self.completed = true;
        Ok(result)
    }
}

impl<Export, SharedLease: RuntimeSharedLease> Drop for RuntimeActiveLeaseState<'_, '_, Export, SharedLease> {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        if matches!(self.incomplete_drop_behavior, IncompleteDropBehavior::GuaranteedFallback) {
            let _ = synchronize_completion_fallback(self.completion, self.drop_operation);
        }
    }
}

struct RuntimeActiveReadLease<'export, 'consumer, Export, SharedLease>
where SharedLease: RuntimeSharedLease
{
    inner: RuntimeActiveLeaseState<'export, 'consumer, Export, SharedLease>,
}

impl<'export, 'consumer, Export, SharedLease> RuntimeActiveReadLease<'export, 'consumer, Export, SharedLease>
where SharedLease: RuntimeSharedLease
{
    fn new(inner: RuntimeActiveLeaseState<'export, 'consumer, Export, SharedLease>) -> Self {
        Self { inner }
    }

    fn export(&self) -> &'export Export {
        self.inner.export()
    }

    fn read_start(&self) -> &RuntimeReadStartAuthority<'consumer> {
        self.inner.read_start()
    }

    fn completion(&self) -> &RuntimeCompletionAuthority<'consumer> {
        self.inner.completion()
    }

    fn complete(self) -> Result<RuntimeLeaseCompletionResult> {
        self.inner.complete()
    }

    fn fmt_debug(&self, name: &'static str, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct(name)
            .field("read_start", &self.read_start())
            .field("completion", &self.completion())
            .field("producer_device", &self.inner.producer_device())
            .finish()
    }
}

fn activate_runtime_read_lease<'export, 'consumer, Export, SharedLease>(
    export: &'export Export,
    read_start: RuntimeReadStartAuthority<'consumer>,
    completion: RuntimeCompletionAuthority<'consumer>,
    readiness: &ProducerReadiness<'_>,
    activate_operation: &'static str,
    complete_operation: &'static str,
    drop_operation: &'static str,
    activate_shared: impl FnOnce(
        ReadStartAuthority<'consumer>,
        CompletionAuthority<'consumer>,
    ) -> std::result::Result<SharedLease, InteropError>,
) -> std::result::Result<RuntimeActiveReadLease<'export, 'consumer, Export, SharedLease>, InteropError>
where
    SharedLease: RuntimeSharedLease,
{
    let shared_read_start = runtime_read_start_as_shared(read_start, activate_operation)?;
    let shared_completion = runtime_completion_as_shared(completion, activate_operation)?;
    let shared_lease = activate_shared(shared_read_start, shared_completion)?;
    let producer_readiness = producer_readiness_binding(readiness, activate_operation)?;
    Ok(RuntimeActiveReadLease::new(RuntimeActiveLeaseState::new(
        export,
        read_start,
        completion,
        producer_readiness,
        shared_lease,
        complete_operation,
        drop_operation,
    )))
}

macro_rules! impl_runtime_active_read_lease_surface {
    ($lease:ident, $export_ty:ty, $debug_name:literal) => {
        impl<'export, 'producer, 'consumer> $lease<'export, 'producer, 'consumer> {
            pub fn export(&self) -> &'export $export_ty {
                self.inner.export()
            }

            pub fn read_start(&self) -> &RuntimeReadStartAuthority<'consumer> {
                self.inner.read_start()
            }

            pub fn completion(&self) -> &RuntimeCompletionAuthority<'consumer> {
                self.inner.completion()
            }

            pub fn complete(self) -> Result<RuntimeLeaseCompletionResult> {
                self.inner.complete()
            }
        }

        impl fmt::Debug for $lease<'_, '_, '_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.inner.fmt_debug($debug_name, f)
            }
        }
    };
}

macro_rules! impl_try_activate_runtime_read_lease {
    ($trait_name:ident, $export_ty:ty, $lease:ident, $activate_op:expr, $complete_op:expr, $drop_op:expr) => {
        impl<'producer> $trait_name for $export_ty {
            fn try_activate_runtime_read_lease<'consumer>(
                &self,
                read_start: RuntimeReadStartAuthority<'consumer>,
                completion: RuntimeCompletionAuthority<'consumer>,
            ) -> std::result::Result<$lease<'_, '_, 'consumer>, InteropError> {
                let inner = activate_runtime_read_lease(
                    self,
                    read_start,
                    completion,
                    self.readiness(),
                    $activate_op,
                    $complete_op,
                    $drop_op,
                    |shared_read_start, shared_completion| {
                        self.try_activate_read_lease(shared_read_start, shared_completion)
                    },
                )?;
                Ok($lease { inner })
            }
        }
    };
}

/// CUDA-specific compatibility wrapper over a shared active column read lease.
///
/// Activation still goes through [`PreparedColumnExport::try_activate_read_lease`]
/// under the hood, so export-root overlapping-lease coordination, shared
/// liveness retention, and backend-declared incomplete-drop behavior all stay
/// aligned with the shared contract. This wrapper only changes the completion
/// surface exposed to CUDA-runtime-oriented callers.
#[must_use = "lease must be completed explicitly via .complete(); dropping without completion falls back to conservative synchronization"]
pub struct RuntimeActiveColumnReadLease<'export, 'producer, 'consumer> {
    inner: RuntimeActiveReadLease<
        'export,
        'consumer,
        PreparedColumnExport<'producer>,
        super::ActiveColumnReadLease<'export, 'producer, 'consumer>,
    >,
}
impl_runtime_active_read_lease_surface!(
    RuntimeActiveColumnReadLease,
    PreparedColumnExport<'producer>,
    "RuntimeActiveColumnReadLease"
);

/// CUDA-specific compatibility wrapper over a shared active table read lease.
///
/// Activation still goes through [`PreparedTableExport::try_activate_read_lease`]
/// under the hood, so export-root overlapping-lease coordination, shared
/// liveness retention, and backend-declared incomplete-drop behavior all stay
/// aligned with the shared contract. This wrapper only changes the completion
/// surface exposed to CUDA-runtime-oriented callers.
#[must_use = "lease must be completed explicitly via .complete(); dropping without completion falls back to conservative synchronization"]
pub struct RuntimeActiveTableReadLease<'export, 'producer, 'consumer> {
    inner: RuntimeActiveReadLease<
        'export,
        'consumer,
        PreparedTableExport<'producer>,
        super::ActiveTableReadLease<'export, 'producer, 'consumer>,
    >,
}
impl_runtime_active_read_lease_surface!(
    RuntimeActiveTableReadLease,
    PreparedTableExport<'producer>,
    "RuntimeActiveTableReadLease"
);

/// CUDA-specific compatibility activation over a prepared column export.
///
/// Prefer [`PreparedColumnExport::try_activate_read_lease`] for new shared
/// interop APIs. Use this trait when you specifically need the
/// `cudf-rs`-owned runtime witness surface.
pub trait TryActivateRuntimeColumnReadLease {
    fn try_activate_runtime_read_lease<'consumer>(
        &self,
        read_start: RuntimeReadStartAuthority<'consumer>,
        completion: RuntimeCompletionAuthority<'consumer>,
    ) -> std::result::Result<RuntimeActiveColumnReadLease<'_, '_, 'consumer>, InteropError>;
}

impl_try_activate_runtime_read_lease!(
    TryActivateRuntimeColumnReadLease,
    PreparedColumnExport<'producer>,
    RuntimeActiveColumnReadLease,
    ACTIVATE_RUNTIME_COLUMN_LEASE_OP,
    COMPLETE_RUNTIME_COLUMN_LEASE_OP,
    DROP_RUNTIME_COLUMN_LEASE_OP
);

/// CUDA-specific compatibility activation over a prepared table export.
///
/// Prefer [`PreparedTableExport::try_activate_read_lease`] for new shared
/// interop APIs. Use this trait when you specifically need the
/// `cudf-rs`-owned runtime witness surface.
pub trait TryActivateRuntimeTableReadLease {
    fn try_activate_runtime_read_lease<'consumer>(
        &self,
        read_start: RuntimeReadStartAuthority<'consumer>,
        completion: RuntimeCompletionAuthority<'consumer>,
    ) -> std::result::Result<RuntimeActiveTableReadLease<'_, '_, 'consumer>, InteropError>;
}

impl_try_activate_runtime_read_lease!(
    TryActivateRuntimeTableReadLease,
    PreparedTableExport<'producer>,
    RuntimeActiveTableReadLease,
    ACTIVATE_RUNTIME_TABLE_LEASE_OP,
    COMPLETE_RUNTIME_TABLE_LEASE_OP,
    DROP_RUNTIME_TABLE_LEASE_OP
);
