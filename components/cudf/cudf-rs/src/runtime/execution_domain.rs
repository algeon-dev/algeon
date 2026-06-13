//! Private retained-provenance substrate for owned GPU objects.
//!
//! `ExecutionDomain` remains the internal type name for now, but its semantics
//! are intentionally narrower than the v1 "execution identity" model:
//!
//! - it records retained provenance for later host-visible synchronization
//!   and device-side cross-stream dependencies
//! - it supplies default stream / MR handles when no explicit operation context
//!   is provided
//! - it does not decide whether mixed-input operations are legal
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
    error::{self, Error, ErrorKind, Result},
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
            Error::new(ErrorKind::InvalidOperation, "debug retained-provenance identity must be non-zero")
                .with_operation("execution_domain::ExecutionIdentity::new")
                .with_context("identity", value.to_string())
        })
    }

    fn get(self) -> u64 {
        self.0.get()
    }
}

impl ExecutionDomain {
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
            .with_context("identity", identity.to_string())
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
            .map_err(|e| e.with_operation(operation).with_context("sync_stream", self.debug_label()))
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
            e.with_operation(operation)
                .with_context("source_domain", source.debug_label())
                .with_context("target_domain", target.debug_label())
        })?;
        let barrier = CudaEvent::new().map_err(|e| {
            e.with_operation(operation)
                .with_context("source_domain", source.debug_label())
                .with_context("target_domain", target.debug_label())
        })?;
        barrier.record(source.stream.as_ref()).map_err(|e| {
            e.with_operation(operation)
                .with_context("source_domain", source.debug_label())
                .with_context("target_domain", target.debug_label())
        })?;
        target.stream.wait_event(&barrier).map_err(|e| {
            e.with_operation(operation)
                .with_context("source_domain", source.debug_label())
                .with_context("target_domain", target.debug_label())
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
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    "cross-device borrowed stream override: stream was created on a different device than the domain",
                )
                .with_operation(operation)
                .with_context("domain_device", input.device.ordinal().to_string())
                .with_context("override_stream_device", os.device().ordinal().to_string()));
            }
        }
        if let Some(omr) = override_mr {
            if omr.device() != input.device {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    "cross-device borrowed MR override: memory resource was created on a different device than the domain",
                )
                .with_operation(operation)
                .with_context("domain_device", input.device.ordinal().to_string())
                .with_context("override_mr_device", omr.device().ordinal().to_string()));
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

        Err(Error::new(
            ErrorKind::InvalidOperation,
            "cross-device operation requires explicit relocation into one runtime before execution",
        )
        .with_operation(operation)
        .with_context("left_domain", left.debug_label())
        .with_context("right_domain", right.debug_label())
        .with_context("left_device", left_device.to_string())
        .with_context("right_device", right_device.to_string())
        .with_context(
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

        Err(Error::new(ErrorKind::InvalidOperation, message)
            .with_operation(operation)
            .with_context("left_domain", left.debug_label())
            .with_context("right_domain", right.debug_label())
            .with_context("left_device", left.device_snapshot().to_string())
            .with_context("right_device", right.device_snapshot().to_string())
            .with_context("remedy", remedy))
    }
}
