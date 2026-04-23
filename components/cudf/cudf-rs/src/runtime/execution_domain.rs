//! Private retained-provenance substrate for owned GPU objects.
//!
//! In `cudf-rs`, every owned GPU object carries **retained provenance**:
//!
//! - the CUDA device the object is pinned to
//! - the CUDA stream that most recently produced or ordered work for it
//! - the device memory resource (MR) that should back follow-on allocations
//!
//! `ExecutionDomain` is the private bundle that stores that metadata.
//! It is not a general scheduler or execution-policy object. Its job is narrow:
//!
//! - remember retained provenance after an operation returns
//! - supply default stream and memory-resource handles when an operation does
//!   not receive explicit borrowed overrides
//! - provide the synchronization hooks needed for host visibility and
//!   device-side cross-stream ordering
//!
//! Terms used here:
//!
//! - **stream**: a CUDA work queue (`CudaStream`) on which libcudf kernels and
//!   copies are enqueued
//! - **memory resource / MR**: an RMM allocator handle
//!   (`DeviceMemoryResource`) used for device allocations
//! - **retained provenance**: execution metadata kept on the owned result so
//!   later operations know which stream and allocator the object is associated
//!   with
//!
//! `ExecutionDomain` does not by itself decide whether mixed-input operations
//! are legal. Higher-level APIs choose whether to reject, relocate, or
//! synchronize mixed-provenance inputs.
//!
//! Every retained domain is explicit and device-pinned. The C++ shim re-enters
//! the pinned device before resolving stream / MR defaults, so operations
//! execute on the correct GPU regardless of the calling thread's current CUDA
//! device.

use std::{
    collections::HashMap,
    num::NonZeroU64,
    os::raw::c_void,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

use super::{device::DeviceId, device_guard::DeviceGuard, event::CudaEvent, stream::CudaStream};
use crate::core::{
    error::{self, DeviceIdFact, Error, ErrorCode, FACT_LEFT_DEVICE, FACT_RIGHT_DEVICE, Result},
    memory::DeviceMemoryResource,
};

static DEBUG_EXPLICIT_DOMAINS: OnceLock<Mutex<HashMap<u64, Arc<ExecutionDomain>>>> = OnceLock::new();
static NEXT_EXPLICIT_DOMAIN_ID: AtomicU64 = AtomicU64::new(1);

pub(crate) struct ExecutionDomain {
    device: DeviceId,
    stream: Arc<CudaStream>,
    mr: Arc<DeviceMemoryResource>,
    identity: ExecutionIdentity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ExecutionIdentity(NonZeroU64);

impl ExecutionIdentity {
    fn new(value: u64) -> Result<Self> {
        NonZeroU64::new(value).map(Self).ok_or_else(|| {
            Error::new_with(
                ErrorCode::InvariantViolation,
                "execution_domain::ExecutionIdentity::new",
                "debug retained-provenance identity must be non-zero",
            )
            .with_detail("identity", value.to_string())
        })
    }

    fn get(self) -> u64 {
        self.0.get()
    }
}

impl ExecutionDomain {
    fn borrowed_override_binding_mismatch(
        operation: &'static str,
        resource: &'static str,
        input_device: DeviceId,
        override_device: DeviceId,
        message: &'static str,
    ) -> Error {
        Error::new_with(ErrorCode::RuntimeBindingMismatch, operation, message)
            .with_fact(FACT_LEFT_DEVICE, DeviceIdFact(input_device.ordinal_raw()))
            .with_fact(FACT_RIGHT_DEVICE, DeviceIdFact(override_device.ordinal_raw()))
            .with_detail("resource", resource)
            .with_detail("reason", "cross_device_borrowed_override")
            .with_detail(
                "remedy",
                "borrow a stream or memory resource created on the same device as the input Runtime, or relocate the input into that Runtime first",
            )
    }

    fn next_identity() -> Result<ExecutionIdentity> {
        ExecutionIdentity::new(NEXT_EXPLICIT_DOMAIN_ID.fetch_add(1, Ordering::Relaxed))
    }

    pub(crate) fn from_handles_on_device(
        device: DeviceId,
        stream: CudaStream,
        mr: DeviceMemoryResource,
    ) -> Result<Arc<Self>> {
        Ok(Arc::new(Self { device, stream: Arc::new(stream), mr: Arc::new(mr), identity: Self::next_identity()? }))
    }

    pub(crate) fn debug_explicit(identity: u64) -> Result<Arc<Self>> {
        let cache = DEBUG_EXPLICIT_DOMAINS.get_or_init(|| Mutex::new(HashMap::new()));
        let mut cache = cache.lock().map_err(|_| {
            error::invariant_violation(
                "execution_domain::debug_explicit",
                "debug explicit retained-provenance cache poisoned",
            )
            .with_detail("identity", identity.to_string())
        })?;
        if let Some(domain) = cache.get(&identity) {
            return Ok(Arc::clone(domain));
        }

        let device = DeviceId::current().unwrap_or(DeviceId::DEFAULT);
        let stream = Arc::new(CudaStream::new()?);
        let mr = Arc::new(DeviceMemoryResource::cuda()?);
        let domain = Arc::new(Self { device, stream, mr, identity: ExecutionIdentity::new(identity)? });
        cache.insert(identity, Arc::clone(&domain));
        Ok(domain)
    }

    pub(crate) fn debug_kind(&self) -> &'static str {
        "explicit"
    }

    pub(crate) fn debug_id(&self) -> u64 {
        self.identity.get()
    }

    pub(crate) fn debug_label(&self) -> String {
        format!("explicit:{}(device={})", self.identity.get(), self.device.ordinal())
    }

    pub(crate) fn debug_stream_value(&self) -> usize {
        self.stream.debug_handle_value().unwrap_or(0)
    }

    pub(crate) fn debug_mr_value(&self) -> usize {
        self.mr.__debug_raw_handle_value()
    }

    pub(crate) fn device_id(&self) -> DeviceId {
        self.device
    }

    pub(crate) fn device_snapshot(&self) -> i32 {
        self.device.ordinal_raw()
    }

    /// Synchronize this domain's stream so that the host can read device memory
    /// produced on it.
    ///
    /// This is a **host fence** (`cudaStreamSynchronize`): the calling CPU
    /// thread blocks until all GPU work enqueued on the stream completes.
    ///
    /// Use this before copying device data to the host (e.g. `to_host()`,
    /// `Scalar::value()`). For cross-stream device-to-device ordering between
    /// operations, use [`insert_stream_dependency`](Self::insert_stream_dependency).
    pub(crate) fn synchronize_for_host_visibility(&self, operation: &'static str) -> Result<()> {
        self.stream
            .synchronize()
            .map_err(|error| error.reanchor(operation).with_detail("sync_stream", self.debug_label()))
    }

    /// Insert a device-side ordering dependency from `source` into `target`.
    ///
    /// This records a CUDA event on the source stream, then inserts a
    /// `cudaStreamWaitEvent` on the target stream. It does not block the
    /// calling CPU thread.
    pub(crate) fn insert_stream_dependency(
        source: &Arc<Self>,
        target: &Arc<Self>,
        operation: &'static str,
    ) -> Result<()> {
        Self::require_same_device_for_mixed_inputs(source, target, operation)?;
        if Self::same_provenance(source, target) {
            return Ok(());
        }

        let _guard = DeviceGuard::new(source.device).map_err(|e| {
            e.reanchor(operation)
                .with_detail("source_domain", source.debug_label())
                .with_detail("target_domain", target.debug_label())
        })?;
        let barrier = CudaEvent::new().map_err(|e| {
            e.reanchor(operation)
                .with_detail("source_domain", source.debug_label())
                .with_detail("target_domain", target.debug_label())
        })?;
        barrier.record(source.stream.as_ref()).map_err(|e| {
            e.reanchor(operation)
                .with_detail("source_domain", source.debug_label())
                .with_detail("target_domain", target.debug_label())
        })?;
        target.stream.wait_event(&barrier).map_err(|e| {
            e.reanchor(operation)
                .with_detail("source_domain", source.debug_label())
                .with_detail("target_domain", target.debug_label())
        })
    }

    pub(crate) fn explicit_exec_options_raw(&self) -> Option<cudf_sys::cudf_exec_options_t> {
        Some(self.exec_options_raw())
    }

    pub(crate) fn exec_options_raw(&self) -> cudf_sys::cudf_exec_options_t {
        cudf_sys::cudf_exec_options_t {
            struct_size: std::mem::size_of::<cudf_sys::cudf_exec_options_t>(),
            device: self.device.ordinal_raw(),
            flags: cudf_sys::CUDF_EXEC_FLAG_DEVICE_EXPLICIT,
            stream: self.stream.as_raw() as *mut c_void,
            mr: self.mr.as_raw() as *mut c_void,
            reserved: [0; cudf_sys::CUDF_EXEC_OPTIONS_RESERVED_SLOTS],
        }
    }

    pub(crate) fn stream(&self) -> &CudaStream {
        self.stream.as_ref()
    }

    pub(crate) fn memory_resource(&self) -> &DeviceMemoryResource {
        self.mr.as_ref()
    }

    pub(crate) fn resolve_with_borrowed_override(
        input: &Arc<Self>,
        override_stream: Option<&CudaStream>,
        override_mr: Option<&DeviceMemoryResource>,
        operation: &'static str,
    ) -> Result<Arc<Self>> {
        if let Some(os) = override_stream {
            if os.device() != input.device {
                return Err(Self::borrowed_override_binding_mismatch(
                    operation,
                    "stream",
                    input.device,
                    os.device(),
                    "cross-device borrowed stream override: stream was created on a different device than the input runtime",
                ));
            }
        }
        if let Some(omr) = override_mr {
            if omr.device() != input.device {
                return Err(Self::borrowed_override_binding_mismatch(
                    operation,
                    "memory_resource",
                    input.device,
                    omr.device(),
                    "cross-device borrowed memory-resource override: memory resource was created on a different device than the input runtime",
                ));
            }
        }

        Ok(Arc::new(Self {
            device: input.device,
            stream: override_stream.map(|stream| Arc::new(stream.clone())).unwrap_or_else(|| Arc::clone(&input.stream)),
            mr: override_mr.map(|mr| Arc::new(mr.clone())).unwrap_or_else(|| Arc::clone(&input.mr)),
            identity: input.identity,
        }))
    }

    pub(crate) fn same_provenance(left: &Arc<Self>, right: &Arc<Self>) -> bool {
        left.device == right.device
            && left.stream.as_raw() == right.stream.as_raw()
            && left.mr.as_raw() == right.mr.as_raw()
    }

    pub(crate) fn require_same_device_for_mixed_inputs(
        left: &Arc<Self>,
        right: &Arc<Self>,
        operation: &'static str,
    ) -> Result<()> {
        let left_device = left.device_snapshot();
        let right_device = right.device_snapshot();
        if left_device == right_device {
            return Ok(());
        }

        Err(Error::new_with(
            ErrorCode::CrossDeviceInputs,
            operation,
            "cross-device operation requires explicit relocation into one runtime before execution",
        )
        .with_fact(FACT_LEFT_DEVICE, DeviceIdFact(left_device))
        .with_fact(FACT_RIGHT_DEVICE, DeviceIdFact(right_device))
        .with_detail("left_domain", left.debug_label())
        .with_detail("right_domain", right.debug_label())
        .with_detail(
            "remedy",
            "use copy_into_runtime(...) to relocate all inputs into one Runtime before calling this operation",
        ))
    }

    pub(crate) fn require_same_provenance(
        left: &Arc<Self>,
        right: &Arc<Self>,
        operation: &'static str,
        message: &'static str,
        remedy: &'static str,
    ) -> Result<()> {
        if Self::same_provenance(left, right) {
            return Ok(());
        }

        Err(Error::new_with(ErrorCode::CrossProvenanceInputs, operation, message)
            .with_fact(FACT_LEFT_DEVICE, DeviceIdFact(left.device_snapshot()))
            .with_fact(FACT_RIGHT_DEVICE, DeviceIdFact(right.device_snapshot()))
            .with_detail("left_domain", left.debug_label())
            .with_detail("right_domain", right.debug_label())
            .with_detail("remedy", remedy))
    }
}
