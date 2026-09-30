//! Private retained-provenance substrate for owned GPU objects.
//!
//! In `cudf`, every owned GPU object carries **retained provenance**:
//!
//! - the CUDA device the object is pinned to
//! - the CUDA stream that most recently produced or ordered work for it
//! - the device memory resource (MR) that should back follow-on allocations
//!
//! `ExecutionDomain` is the private bundle that stores that metadata.
//! It is not a general scheduler or execution-policy object. Its job is narrow:
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

#[cfg(any(test, feature = "test-hooks"))]
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};
use std::{
    num::NonZeroU64,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use super::{
    affinity::{ExecutionBindingId, RuntimeAffinityAnchor, RuntimeAffinityId, RuntimeAffinityOwner},
    binding_extension::BindingExtensions,
    device::DeviceId,
    device_guard::DeviceGuard,
    event::CudaEvent,
    placement::ExecutionPlacement,
    stream::CudaStream,
};
#[cfg(any(test, feature = "test-hooks"))]
use crate::core::error;
use crate::core::{
    error::{DeviceIdFact, Error, ErrorCode, FACT_LEFT_DEVICE, FACT_RIGHT_DEVICE, Result},
    memory::DeviceMemoryResource,
};

#[cfg(any(test, feature = "test-hooks"))]
static DEBUG_EXPLICIT_DOMAINS: OnceLock<Mutex<HashMap<u64, Arc<ExecutionDomain>>>> = OnceLock::new();
static NEXT_WORKFLOW_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_BUNDLE_ID: AtomicU64 = AtomicU64::new(1);

pub(crate) struct ExecutionDomain {
    device: DeviceId,
    authority: DeviceExecutionAuthority,
    // Binding extensions are dropped before the stream and memory resource.
    // Cross-crate native owners may borrow those handles during teardown.
    binding_extensions: Arc<BindingExtensions>,
    stream: Arc<CudaStream>,
    mr: Arc<DeviceMemoryResource>,
    workflow_id: ExecutionWorkflowId,
    bundle_id: ExecutionBundleId,
    binding_id: ExecutionBindingId,
    affinity_owner: Arc<RuntimeAffinityOwner>,
    retained_lifetimes: RetainedLifetimes,
    /// Parquet read/decompression scratch limit in bytes, or 0 for no limit.
    parquet_pass_read_limit: u64,
}

/// Immutable device authority shared by all execution-domain FFI entry.
pub(crate) struct DeviceExecutionAuthority {
    device: DeviceId,
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
}

impl DeviceExecutionAuthority {
    fn new(device: DeviceId) -> Self {
        Self { device }
    }

    fn enter(
        &self,
        stream: &CudaStream,
        mr: &DeviceMemoryResource,
        parquet_pass_read_limit: u64,
    ) -> Result<EnteredExecution> {
        let entry = DeviceGuard::enter(self.device, "execution_domain::enter")?;
        let mut flags = cudf_sys::CUDF_EXEC_FLAG_DEVICE_EXPLICIT;
        let mut reserved = [0; cudf_sys::CUDF_EXEC_OPTIONS_RESERVED_SLOTS];
        // The shim rejects the flag paired with a zero limit.
        if parquet_pass_read_limit > 0 {
            flags |= cudf_sys::CUDF_EXEC_FLAG_PARQUET_PASS_READ_LIMIT;
            reserved[0] = parquet_pass_read_limit;
        }
        Ok(EnteredExecution {
            _entry: entry,
            options: cudf_sys::cudf_exec_options_t {
                struct_size: std::mem::size_of::<cudf_sys::cudf_exec_options_t>(),
                device: self.device.ordinal_raw(),
                flags,
                stream: stream.as_raw(),
                mr: mr.as_raw(),
                reserved,
            },
        })
    }
}

/// Flat, clone-cheap retained owner set.
///
/// Tokens are deduplicated by allocation identity. Keeping one flat Arc-backed
/// slice avoids recursive provenance chains and bounds repeated inheritance to
/// the number of distinct owners rather than the number of derivations.
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ExecutionWorkflowId(NonZeroU64);

impl ExecutionWorkflowId {
    fn new(value: u64) -> Result<Self> {
        NonZeroU64::new(value).map(Self).ok_or_else(|| {
            Error::new_with(
                ErrorCode::InvariantViolation,
                "execution_domain::ExecutionWorkflowId::new",
                "execution workflow identity must be non-zero",
            )
            .with_detail("workflow_id", value.to_string())
        })
    }

    fn get(self) -> u64 {
        self.0.get()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ExecutionBundleId(NonZeroU64);

impl ExecutionBundleId {
    fn new(value: u64) -> Result<Self> {
        NonZeroU64::new(value).map(Self).ok_or_else(|| {
            Error::new_with(
                ErrorCode::InvariantViolation,
                "execution_domain::ExecutionBundleId::new",
                "execution bundle identity must be non-zero",
            )
            .with_detail("bundle_id", value.to_string())
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

    fn next_workflow_id() -> Result<ExecutionWorkflowId> {
        ExecutionWorkflowId::new(NEXT_WORKFLOW_ID.fetch_add(1, Ordering::Relaxed))
    }

    fn next_bundle_id() -> Result<ExecutionBundleId> {
        ExecutionBundleId::new(NEXT_BUNDLE_ID.fetch_add(1, Ordering::Relaxed))
    }

    pub(crate) fn from_handles_on_device(
        device: DeviceId,
        stream: CudaStream,
        mr: DeviceMemoryResource,
    ) -> Result<Arc<Self>> {
        mr.register_stream_for_retirement(&stream);
        Ok(Arc::new(Self {
            parquet_pass_read_limit: 0,
            device,
            authority: DeviceExecutionAuthority::new(device),
            binding_extensions: Arc::new(BindingExtensions::default()),
            stream: Arc::new(stream),
            mr: Arc::new(mr),
            workflow_id: Self::next_workflow_id()?,
            bundle_id: Self::next_bundle_id()?,
            binding_id: ExecutionBindingId::next()?,
            affinity_owner: RuntimeAffinityOwner::new()?,
            retained_lifetimes: RetainedLifetimes::default(),
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

        let device = DeviceId::current().unwrap_or(DeviceId::DEFAULT);
        let stream = Arc::new(CudaStream::new()?);
        let mr = Arc::new(DeviceMemoryResource::cuda()?);
        let domain = Arc::new(Self {
            parquet_pass_read_limit: 0,
            device,
            authority: DeviceExecutionAuthority::new(device),
            binding_extensions: Arc::new(BindingExtensions::default()),
            stream,
            mr,
            workflow_id: ExecutionWorkflowId::new(identity)?,
            bundle_id: Self::next_bundle_id()?,
            binding_id: ExecutionBindingId::next()?,
            affinity_owner: RuntimeAffinityOwner::new()?,
            retained_lifetimes: RetainedLifetimes::default(),
        });
        cache.insert(identity, Arc::clone(&domain));
        Ok(domain)
    }

    pub(crate) fn debug_kind(&self) -> &'static str {
        "explicit"
    }

    pub(crate) fn debug_id(&self) -> u64 {
        self.workflow_id.get()
    }

    pub(crate) fn debug_label(&self) -> String {
        format!(
            "explicit:{}(workflow={},bundle={},device={},stream={},memory_resource={})",
            self.workflow_id.get(),
            self.workflow_id.get(),
            self.bundle_id.get(),
            self.device.ordinal(),
            self.debug_stream_value(),
            self.debug_mr_value()
        )
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

    pub(crate) fn placement(&self) -> ExecutionPlacement {
        ExecutionPlacement {
            device: self.device,
            workflow_id: self.workflow_id.get(),
            bundle_id: self.bundle_id.get(),
            stream_debug: self.debug_stream_value(),
            memory_resource_debug: self.debug_mr_value(),
        }
    }

    pub(crate) fn affinity_id(&self) -> RuntimeAffinityId {
        self.affinity_owner.id()
    }

    pub(crate) fn binding_id(&self) -> ExecutionBindingId {
        self.binding_id
    }

    pub(crate) fn binding_extensions(self: &Arc<Self>) -> &BindingExtensions {
        self.binding_extensions.as_ref()
    }

    pub(crate) fn affinity_anchor(&self) -> RuntimeAffinityAnchor {
        RuntimeAffinityAnchor::new(&self.affinity_owner)
    }

    pub(crate) fn affinity_owner(&self) -> &Arc<RuntimeAffinityOwner> {
        &self.affinity_owner
    }

    pub(crate) fn same_affinity(left: &Arc<Self>, right: &Arc<Self>) -> bool {
        left.affinity_owner.id() == right.affinity_owner.id()
            && Arc::ptr_eq(&left.affinity_owner, &right.affinity_owner)
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

    /// Insert two-way device-side ordering between a cross-stream input
    /// (`source`) and the consuming operation's output domain (`target`).
    ///
    /// The forward direction happens immediately: a CUDA event is recorded on
    /// the source stream and the target stream waits on it, so the consumer
    /// never reads bytes that are still being produced. The reverse direction
    /// is carried by the returned [`ConsumerCompletionFence`]: hold it across
    /// the FFI call that enqueues the consuming operation; when it drops it
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
        self.authority.enter(&self.stream, &self.mr, self.parquet_pass_read_limit)
    }

    /// Derives a domain whose Parquet reads decode in passes bounded by
    /// `limit` bytes of read and decompression scratch.
    pub(crate) fn with_parquet_pass_read_limit(input: &Arc<Self>, limit: u64) -> Arc<Self> {
        if limit == input.parquet_pass_read_limit {
            return Arc::clone(input);
        }
        Arc::new(Self {
            device: input.device,
            authority: DeviceExecutionAuthority::new(input.device),
            binding_extensions: Arc::clone(&input.binding_extensions),
            stream: Arc::clone(&input.stream),
            mr: Arc::clone(&input.mr),
            workflow_id: input.workflow_id,
            bundle_id: input.bundle_id,
            binding_id: input.binding_id,
            affinity_owner: Arc::clone(&input.affinity_owner),
            retained_lifetimes: input.retained_lifetimes.clone(),
            parquet_pass_read_limit: limit,
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
            parquet_pass_read_limit: input.parquet_pass_read_limit,
            device: input.device,
            authority: DeviceExecutionAuthority::new(input.device),
            binding_extensions: Arc::new(BindingExtensions::default()),
            stream,
            mr,
            workflow_id: input.workflow_id,
            bundle_id: Self::next_bundle_id()?,
            binding_id: ExecutionBindingId::next()?,
            affinity_owner: Arc::clone(&input.affinity_owner),
            retained_lifetimes: input.retained_lifetimes.clone(),
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
            parquet_pass_read_limit: input.parquet_pass_read_limit,
            device: input.device,
            authority: DeviceExecutionAuthority::new(input.device),
            binding_extensions: Arc::clone(&input.binding_extensions),
            stream: Arc::clone(&input.stream),
            mr: Arc::clone(&input.mr),
            workflow_id: input.workflow_id,
            bundle_id: Self::next_bundle_id()?,
            binding_id: input.binding_id,
            affinity_owner: Arc::clone(&input.affinity_owner),
            retained_lifetimes,
        }))
    }

    pub(crate) fn inherit_retained_lifetime(input: &Arc<Self>, source: &Arc<Self>) -> Result<Arc<Self>> {
        let Some(retained_lifetimes) = input.retained_lifetimes.union(&source.retained_lifetimes) else {
            return Ok(Arc::clone(input));
        };
        Ok(Arc::new(Self {
            parquet_pass_read_limit: input.parquet_pass_read_limit,
            device: input.device,
            authority: DeviceExecutionAuthority::new(input.device),
            binding_extensions: Arc::clone(&input.binding_extensions),
            stream: Arc::clone(&input.stream),
            mr: Arc::clone(&input.mr),
            workflow_id: input.workflow_id,
            bundle_id: Self::next_bundle_id()?,
            binding_id: input.binding_id,
            affinity_owner: Arc::clone(&input.affinity_owner),
            retained_lifetimes,
        }))
    }

    #[cfg(any(test, feature = "test-hooks"))]
    pub(crate) fn debug_retained_lifetime_count(&self) -> usize {
        self.retained_lifetimes.len()
    }

    pub(crate) fn inherit_affinity(input: &Arc<Self>, source: &Arc<Self>) -> Result<Arc<Self>> {
        if input.device != source.device {
            return Err(Error::new_with(
                ErrorCode::RuntimeBindingMismatch,
                "runtime::Runtime::inherit_affinity_from",
                "runtime affinity can only be inherited between runtimes on the same CUDA device",
            )
            .with_fact(FACT_LEFT_DEVICE, DeviceIdFact(input.device.ordinal_raw()))
            .with_fact(FACT_RIGHT_DEVICE, DeviceIdFact(source.device.ordinal_raw()))
            .with_detail("reason", "cross_device_runtime_affinity")
            .with_detail("remedy", "create the companion runtime on the same CUDA device as its base runtime"));
        }
        if Self::same_affinity(input, source) {
            return Ok(Arc::clone(input));
        }
        Ok(Arc::new(Self {
            parquet_pass_read_limit: input.parquet_pass_read_limit,
            device: input.device,
            authority: DeviceExecutionAuthority::new(input.device),
            binding_extensions: Arc::clone(&input.binding_extensions),
            stream: Arc::clone(&input.stream),
            mr: Arc::clone(&input.mr),
            workflow_id: input.workflow_id,
            bundle_id: Self::next_bundle_id()?,
            binding_id: input.binding_id,
            affinity_owner: Arc::clone(&source.affinity_owner),
            retained_lifetimes: input.retained_lifetimes.clone(),
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
/// hand the block to new work on that stream while the consuming kernel is
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
/// failures are reported on stderr; they are never silently ignored.
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
        let barrier = CudaEvent::new().map_err(decorate)?;
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
        let completion = CudaEvent::new().map_err(decorate)?;
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
            // fire() stops at the first failure). Returning silently would
            // reopen the use-after-free this fence exists to close, so fall
            // back to a host fence: block until the consuming work on the
            // output stream completes. After that, no input stream can
            // deallocate under an in-flight consumer. Slow but safe.
            eprintln!(
                "[cudf::ConsumerCompletionFence] warning: failed to order cross-stream input streams after consuming operation {}; falling back to host synchronization: {error}",
                self.operation
            );
            if let Err(sync_error) = self.output.synchronize_for_host_visibility(self.operation) {
                eprintln!(
                    "[cudf::ConsumerCompletionFence] warning: host-synchronization fallback also failed after consuming operation {}; cross-stream input buffers may be reused while the consumer is still reading them: {sync_error}",
                    self.operation
                );
            }
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
}
