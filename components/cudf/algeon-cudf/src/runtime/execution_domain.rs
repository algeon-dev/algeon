//! Private retained-provenance substrate for owned GPU objects.
//!
//! In `cudf`, every owned GPU object carries **retained provenance**:
//!
//! - the CUDA device the object is pinned to
//! - the CUDA stream that most recently produced or ordered work for it
//! - the device memory resource (MR) that should back follow-on allocations
//!
//! `ExecutionDomain` is the private bundle that stores that metadata.
//! Its job is narrow and excludes general scheduling and execution policy:
//!
//! - remember retained provenance after an operation returns
//! - supply explicit stream and memory-resource handles from retained
//!   provenance when an operation does not receive borrowed overrides
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
//! the pinned device and receives non-null stream / MR handles, so production
//! operations execute on the correct GPU regardless of the calling thread's
//! current CUDA device.

use std::sync::Arc;
#[cfg(any(test, feature = "test-hooks"))]
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

use super::{
    binding_extension::BindingExtensions, binding_id::ExecutionBindingId, device::DeviceId, device_guard::DeviceGuard,
    device_health, event::CudaEvent, placement::ExecutionPlacement, stream::CudaStream,
};
#[cfg(any(test, feature = "test-hooks"))]
use crate::core::error;
use crate::core::{
    error::{DeviceIdFact, Error, ErrorCode, FACT_LEFT_DEVICE, FACT_RIGHT_DEVICE, Result},
    memory::DeviceMemoryResource,
};

#[cfg(any(test, feature = "test-hooks"))]
static DEBUG_EXPLICIT_DOMAINS: OnceLock<Mutex<HashMap<u64, Arc<ExecutionDomain>>>> = OnceLock::new();

pub(crate) struct ExecutionDomain {
    device: DeviceId,
    // Binding extensions are dropped before the stream and memory resource.
    // Cross-crate native owners may borrow those handles during teardown.
    binding_extensions: Arc<BindingExtensions>,
    stream: Arc<CudaStream>,
    mr: Arc<DeviceMemoryResource>,
    binding_id: ExecutionBindingId,
    retained_lifetimes: RetainedLifetimes,
    /// Identity that test helpers report for this domain: the explicit
    /// identity of a test domain, or the binding identity otherwise.
    #[cfg(any(test, feature = "test-hooks"))]
    debug_identity: u64,
}

/// Scoped execution options whose lifetime proves owner-device entry.
pub(crate) struct EnteredExecution {
    _entry: DeviceGuard,
    options: cudf_sys::cudf_exec_options_t,
}

impl EnteredExecution {
    /// Borrow the native execution options while retaining the host-thread
    /// owner-device entry proof carried by this value.
    pub(crate) fn options(&self) -> &cudf_sys::cudf_exec_options_t {
        &self.options
    }

    /// Bound the Parquet read and decompression scratch of this execution to
    /// `limit` bytes; `None` leaves reads unbounded.
    pub(crate) fn with_parquet_pass_read_limit(mut self, limit: Option<u64>) -> Self {
        // The shim rejects the flag paired with a zero limit.
        if let Some(limit) = limit.filter(|limit| *limit > 0) {
            self.options.flags |= cudf_sys::CUDF_EXEC_FLAG_PARQUET_PASS_READ_LIMIT;
            self.options.reserved[0] = limit;
        }
        self
    }
}

/// Flat, clone-cheap retained owner set.
///
/// Tokens are deduplicated by allocation identity. Keeping one flat Arc-backed
/// slice avoids recursive provenance chains and bounds repeated inheritance to
/// the number of distinct owners. Derivations that add no new owner leave the
/// set unchanged.
#[derive(Clone, Default)]
struct RetainedLifetimes {
    tokens: Option<Arc<[Arc<dyn Send + Sync>]>>,
}

impl RetainedLifetimes {
    fn iter(&self) -> impl Iterator<Item = &Arc<dyn Send + Sync>> {
        self.tokens.as_deref().into_iter().flatten()
    }

    fn len(&self) -> usize {
        self.tokens.as_ref().map_or(0, |tokens| tokens.len())
    }

    /// Return a new set only when `lifetime` is not already retained.
    fn with_added(&self, lifetime: Arc<dyn Send + Sync>) -> Option<Self> {
        if self.iter().any(|existing| Arc::ptr_eq(existing, &lifetime)) {
            return None;
        }
        let mut tokens = Vec::with_capacity(self.len() + 1);
        tokens.extend(self.iter().cloned());
        tokens.push(lifetime);
        Some(Self { tokens: Some(Arc::from(tokens.into_boxed_slice())) })
    }

    /// Return the flat union only when `other` contributes a distinct token.
    fn union(&self, other: &Self) -> Option<Self> {
        let mut added = Vec::new();
        for candidate in other.iter() {
            let already_retained = self.iter().chain(added.iter()).any(|existing| Arc::ptr_eq(existing, candidate));
            if !already_retained {
                added.push(Arc::clone(candidate));
            }
        }
        if added.is_empty() {
            return None;
        }
        let mut tokens = Vec::with_capacity(self.len() + added.len());
        tokens.extend(self.iter().cloned());
        tokens.extend(added);
        Some(Self { tokens: Some(Arc::from(tokens.into_boxed_slice())) })
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

    pub(crate) fn from_handles_on_device(
        device: DeviceId,
        stream: CudaStream,
        mr: DeviceMemoryResource,
    ) -> Result<Arc<Self>> {
        mr.register_stream_for_retirement(&stream);
        let binding_id = ExecutionBindingId::next()?;
        Ok(Arc::new(Self {
            device,
            binding_extensions: Arc::new(BindingExtensions::default()),
            stream: Arc::new(stream),
            mr: Arc::new(mr),
            binding_id,
            retained_lifetimes: RetainedLifetimes::default(),
            #[cfg(any(test, feature = "test-hooks"))]
            debug_identity: binding_id.get(),
        }))
    }

    #[cfg(any(test, feature = "test-hooks"))]
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

        let device = DeviceId::DEFAULT;
        let stream = Arc::new(CudaStream::new_on(device)?);
        let mr = Arc::new(DeviceMemoryResource::cuda_on(device)?);
        let domain = Arc::new(Self {
            device,
            binding_extensions: Arc::new(BindingExtensions::default()),
            stream,
            mr,
            binding_id: ExecutionBindingId::next()?,
            retained_lifetimes: RetainedLifetimes::default(),
            debug_identity: identity,
        });
        cache.insert(identity, Arc::clone(&domain));
        Ok(domain)
    }

    pub(crate) fn debug_kind(&self) -> &'static str {
        "explicit"
    }

    pub(crate) fn debug_id(&self) -> u64 {
        #[cfg(any(test, feature = "test-hooks"))]
        return self.debug_identity;
        #[cfg(not(any(test, feature = "test-hooks")))]
        self.binding_id.get()
    }

    pub(crate) fn debug_label(&self) -> String {
        format!(
            "domain:{}(binding={},device={},stream={},memory_resource={})",
            self.debug_id(),
            self.binding_id.get(),
            self.device.ordinal(),
            self.debug_stream_value(),
            self.debug_mr_value()
        )
    }

    pub(crate) fn debug_stream_value(&self) -> usize {
        self.stream.__debug_raw_handle_value()
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

    pub(crate) fn placement(&self) -> ExecutionPlacement {
        ExecutionPlacement {
            device: self.device,
            stream_debug: self.debug_stream_value(),
            memory_resource_debug: self.debug_mr_value(),
        }
    }

    pub(crate) fn binding_id(&self) -> ExecutionBindingId {
        self.binding_id
    }

    pub(crate) fn binding_extensions(self: &Arc<Self>) -> &BindingExtensions {
        self.binding_extensions.as_ref()
    }

    /// Synchronize this domain's stream so that the host can read device memory
    /// produced on it.
    ///
    /// This is a **host fence** (`cudaStreamSynchronize`). The calling CPU
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

    /// Insert two-way device-side ordering between a cross-stream input
    /// (`source`) and the consuming operation's output domain (`target`).
    ///
    /// The forward direction happens immediately. A CUDA event is recorded on
    /// the source stream, and the target stream waits on it. The consumer thus
    /// never reads bytes that are still being produced. The reverse direction
    /// is carried by the returned [`ConsumerCompletionFence`]. Hold it across
    /// the FFI call that enqueues the consuming operation. When it drops, it
    /// records a completion event on the target stream and makes the source
    /// stream wait on it. Neither direction blocks the calling CPU thread.
    pub(crate) fn insert_stream_dependency(
        source: &Arc<Self>,
        target: &Arc<Self>,
        operation: &'static str,
    ) -> Result<ConsumerCompletionFence> {
        let mut fence = ConsumerCompletionFence::for_output(target, operation);
        fence.order_input(source)?;
        Ok(fence)
    }

    pub(crate) fn enter_execution(&self) -> Result<EnteredExecution> {
        let entry = DeviceGuard::enter(self.device, "execution_domain::enter")?;
        Ok(EnteredExecution {
            _entry: entry,
            options: cudf_sys::cudf_exec_options_t {
                struct_size: std::mem::size_of::<cudf_sys::cudf_exec_options_t>(),
                device: self.device.ordinal_raw(),
                flags: cudf_sys::CUDF_EXEC_FLAG_DEVICE_EXPLICIT,
                stream: self.stream.as_raw(),
                mr: self.mr.as_raw(),
                reserved: [0; cudf_sys::CUDF_EXEC_OPTIONS_RESERVED_SLOTS],
            },
        })
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
        if let Some(os) = override_stream
            && os.device() != input.device
        {
            return Err(Self::borrowed_override_binding_mismatch(
                operation,
                "stream",
                input.device,
                os.device(),
                "cross-device borrowed stream override: stream was created on a different device than the input runtime",
            ));
        }
        if let Some(omr) = override_mr
            && omr.device() != input.device
        {
            return Err(Self::borrowed_override_binding_mismatch(
                operation,
                "memory_resource",
                input.device,
                omr.device(),
                "cross-device borrowed memory-resource override: memory resource was created on a different device than the input runtime",
            ));
        }

        let stream =
            override_stream.map(|stream| Arc::new(stream.clone())).unwrap_or_else(|| Arc::clone(&input.stream));
        let mr = override_mr.map(|mr| Arc::new(mr.clone())).unwrap_or_else(|| Arc::clone(&input.mr));
        if stream.as_raw() == input.stream.as_raw() && mr.as_raw() == input.mr.as_raw() {
            return Ok(Arc::clone(input));
        }

        mr.register_stream_for_retirement(stream.as_ref());

        Ok(Arc::new(Self {
            device: input.device,
            binding_extensions: Arc::new(BindingExtensions::default()),
            stream,
            mr,
            binding_id: ExecutionBindingId::next()?,
            retained_lifetimes: input.retained_lifetimes.clone(),
            #[cfg(any(test, feature = "test-hooks"))]
            debug_identity: input.debug_identity,
        }))
    }

    pub(crate) fn with_retained_lifetime(
        input: &Arc<Self>,
        retained_lifetime: Arc<dyn Send + Sync>,
    ) -> Result<Arc<Self>> {
        let Some(retained_lifetimes) = input.retained_lifetimes.with_added(retained_lifetime) else {
            return Ok(Arc::clone(input));
        };
        Ok(Arc::new(Self {
            device: input.device,
            binding_extensions: Arc::clone(&input.binding_extensions),
            stream: Arc::clone(&input.stream),
            mr: Arc::clone(&input.mr),
            binding_id: input.binding_id,
            retained_lifetimes,
            #[cfg(any(test, feature = "test-hooks"))]
            debug_identity: input.debug_identity,
        }))
    }

    pub(crate) fn inherit_retained_lifetime(input: &Arc<Self>, source: &Arc<Self>) -> Result<Arc<Self>> {
        let Some(retained_lifetimes) = input.retained_lifetimes.union(&source.retained_lifetimes) else {
            return Ok(Arc::clone(input));
        };
        Ok(Arc::new(Self {
            device: input.device,
            binding_extensions: Arc::clone(&input.binding_extensions),
            stream: Arc::clone(&input.stream),
            mr: Arc::clone(&input.mr),
            binding_id: input.binding_id,
            retained_lifetimes,
            #[cfg(any(test, feature = "test-hooks"))]
            debug_identity: input.debug_identity,
        }))
    }

    #[cfg(any(test, feature = "test-hooks"))]
    pub(crate) fn debug_retained_lifetime_count(&self) -> usize {
        self.retained_lifetimes.len()
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

/// Two-way cross-stream ordering for one consuming operation.
///
/// Forward direction (at [`order_input`](Self::order_input) time): the
/// consuming (output) stream waits for work already enqueued on the input's
/// retained stream, so the consumer never reads bytes that are still being
/// produced.
///
/// Reverse direction (at drop): a completion event is recorded on the output
/// stream and every registered cross-stream input's retained stream waits on
/// it. Without this, dropping an input right after the operation enqueues its
/// stream-ordered deallocation on the input's own stream, and the pool can
/// reuse the block for new work on that stream while the consuming kernel is
/// still reading it (device use-after-free).
///
/// Hold the fence across the FFI call that enqueues the consuming operation:
/// the reverse fence must be recorded *after* that work is on the output
/// stream. Dropping it earlier (or on an error path before the FFI call) is
/// safe but provides no protection for work enqueued afterwards.
///
/// Inputs that already share the output stream need no fencing and are not
/// registered, so the same-provenance fast path stays free of CUDA calls.
///
/// Failure policy: if the device-side fence cannot be recorded at drop time
/// (event creation, record, or wait fails; in practice only after a sticky
/// CUDA error), the drop falls back to host-synchronizing the output stream,
/// which restores safety at the cost of blocking the calling thread. Both
/// failures are reported on stderr. Neither is discarded.
#[must_use = "hold the fence across the consuming FFI call; dropping it earlier breaks consumer-to-owner stream ordering"]
pub(crate) struct ConsumerCompletionFence {
    output: Arc<ExecutionDomain>,
    inputs: Vec<Arc<ExecutionDomain>>,
    operation: &'static str,
}

impl ConsumerCompletionFence {
    pub(crate) fn for_output(output: &Arc<ExecutionDomain>, operation: &'static str) -> Self {
        Self { output: Arc::clone(output), inputs: Vec::new(), operation }
    }

    /// Order `input` into the output domain (forward fence) and register it
    /// for the reverse completion fence if it lives on a different stream.
    pub(crate) fn order_input(&mut self, input: &Arc<ExecutionDomain>) -> Result<()> {
        let operation = self.operation;
        ExecutionDomain::require_same_device_for_mixed_inputs(input, &self.output, operation)?;
        if ExecutionDomain::same_provenance(input, &self.output) {
            return Ok(());
        }

        let decorate = |e: Error| {
            e.reanchor(operation)
                .with_detail("source_domain", input.debug_label())
                .with_detail("target_domain", self.output.debug_label())
        };
        let _guard = DeviceGuard::new(input.device).map_err(decorate)?;
        let barrier = CudaEvent::new_on(input.device).map_err(decorate)?;
        barrier.record(input.stream.as_ref()).map_err(decorate)?;
        self.output.stream.wait_event(&barrier).map_err(decorate)?;

        let cross_stream = input.stream.as_raw() != self.output.stream.as_raw();
        let already_registered = self.inputs.iter().any(|known| known.stream.as_raw() == input.stream.as_raw());
        if cross_stream && !already_registered {
            self.inputs.push(Arc::clone(input));
        }
        Ok(())
    }

    fn fire(&self) -> Result<()> {
        let operation = self.operation;
        let decorate = |e: Error| e.reanchor(operation).with_detail("output_domain", self.output.debug_label());
        let _guard = DeviceGuard::new(self.output.device).map_err(decorate)?;
        let completion = CudaEvent::new_on(self.output.device).map_err(decorate)?;
        completion.record(self.output.stream.as_ref()).map_err(decorate)?;
        for input in &self.inputs {
            input.stream.wait_event(&completion).map_err(|e| {
                e.reanchor(operation)
                    .with_detail("input_domain", input.debug_label())
                    .with_detail("output_domain", self.output.debug_label())
            })?;
        }
        Ok(())
    }
}

impl Drop for ConsumerCompletionFence {
    fn drop(&mut self) {
        if self.inputs.is_empty() {
            return;
        }
        if let Err(error) = self.fire() {
            // The device-side fence could not be recorded (or only partially:
            // fire() stops at the first failure). Returning without a fallback
            // would reopen the use-after-free this fence exists to close, so fall
            // back to a host fence: block until the consuming work on the
            // output stream completes. After that, no input stream can
            // deallocate under an in-flight consumer. Slow but safe.
            let fallback = self.output.synchronize_for_host_visibility(self.operation);
            device_health::report_completion_fence_fallback(self.output.device, self.operation, &error, &fallback);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::device_safety::{DeviceEntryTestCall, DeviceSafetyTestScope};

    #[test]
    fn restore_failure_prevents_later_execution_options_before_cuda_ffi() {
        let runtime = crate::Runtime::new().expect("runtime");
        let owner = runtime.device().ordinal_raw();
        let ambient = owner.saturating_add(1);
        let scope = DeviceSafetyTestScope::install([
            DeviceEntryTestCall::GetDevice { status: cudf_sys::CUDF_SUCCESS, device: ambient },
            DeviceEntryTestCall::SetDevice { expected_target: owner, status: cudf_sys::CUDF_SUCCESS },
            DeviceEntryTestCall::SetDevice { expected_target: ambient, status: cudf_sys::CUDF_ERROR_FATAL_CUDA },
            DeviceEntryTestCall::GetDevice { status: cudf_sys::CUDF_SUCCESS, device: owner },
        ]);

        let entered = runtime.execution_domain().enter_execution().expect("initial entered execution");
        assert_eq!(entered.options().device, owner);
        drop(entered);

        let error = match runtime.execution_domain().enter_execution() {
            Ok(_) => panic!("poisoned thread authority must prevent construction of later execution options"),
            Err(error) => error,
        };
        assert_eq!(error.code(), ErrorCode::FatalCuda);
        assert_eq!(scope.remaining_call_count(), 1, "rejected options must not cross CUDA FFI");
    }

    #[test]
    fn placement_reads_cached_handles_without_querying_the_current_device() {
        let runtime = crate::Runtime::new().expect("runtime");
        let series = runtime.series(&[1i32, 2, 3]).expect("series");
        let scope = DeviceSafetyTestScope::install([DeviceEntryTestCall::GetDevice {
            status: cudf_sys::CUDF_SUCCESS,
            device: runtime.device().ordinal_raw(),
        }]);

        let placement = series.placement();

        assert_eq!(placement.device, runtime.device());
        assert_ne!(placement.stream_debug, 0);
        assert_ne!(placement.memory_resource_debug, 0);
        assert_eq!(scope.remaining_call_count(), 1, "placement must not enter the owner device");
    }
}
