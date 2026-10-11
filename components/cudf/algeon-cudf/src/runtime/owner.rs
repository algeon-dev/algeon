//! Public owner for device-pinned default stream, allocator provenance, and retained provenance.

use std::{any::Any, sync::Arc};

use rapids_interop::{ExecutionBinding, InteropError, InteropErrorKind};

use super::{
    Exec,
    binding_extension::{BindingExtensionInitError, BindingExtensionLease},
    binding_id::ExecutionBindingId,
    device::DeviceId,
    device_guard::DeviceGuard,
    event::CudaEvent,
    execution_domain::ExecutionDomain,
    placement::ExecutionPlacement,
    stream::CudaStream,
};
use crate::{
    core::{
        column::{Column, DeviceBuffer},
        error::{self, DeviceIdFact, Error, ErrorCode, FACT_LEFT_DEVICE, FACT_RIGHT_DEVICE, Result},
        memory::{DeviceMemoryManager, DeviceMemoryResource},
        scalar::{Scalar, ScalarValue},
        table::Table,
        types::{CudfType, DataType, TypeId},
    },
    frame::{
        dataframe::DataFrame,
        series::{HostColumnData, Series},
    },
    interop,
};

/// Outcome of ordering this runtime's stream after a producer execution binding.
#[derive(Debug, Clone)]
pub enum StreamDependencyOutcome {
    NoOp,
    Barrier(CudaEvent),
}

fn map_stream_dependency_error(kind: InteropErrorKind, operation: &'static str, error: Error) -> InteropError {
    let message = error.to_string();
    InteropError::new(kind, message).with_context("operation", operation).set_source(error)
}

/// Owns one device-pinned retained-provenance bundle for a `cudf` pipeline.
///
/// Each `Runtime` is pinned to a specific CUDA device (see
/// [`RuntimeBuilder::with_device`]).  All resources created through this
/// runtime live on that device.
///
/// That bundle contains the runtime's default [`CudaStream`] and allocator
/// handle. By default the allocator is the device-level
/// [`DeviceMemoryManager`](crate::core::memory::DeviceMemoryManager)
/// resource for the runtime's CUDA device. Owned GPU objects created through
/// the runtime retain those defaults unless a supported operation resolves
/// same-device borrowed overrides for that call.
///
/// `Runtime` is therefore the canonical place to express a workflow's default
/// execution placement. For query-style budgets, prefer a shared
/// [`crate::QueryMemoryDomain`] member lease. An owned memory resource receives
/// only output allocations; libcudf temporaries resolve through the device's
/// long-lived memory domain. Create an explicit [`DeviceMemoryResource`] first
/// only when you need an advanced allocator handle to reuse across runtimes or
/// borrow through [`crate::Exec`] for one operation.
///
/// When an operation is invoked through this runtime, the runtime's device is
/// re-entered at the FFI boundary before any GPU work is enqueued. This ensures
/// that compute runs on the correct device regardless of the calling thread's
/// current CUDA device. The previous device is restored after each FFI call.
#[derive(Clone)]
pub struct Runtime {
    domain: Arc<ExecutionDomain>,
}

impl std::fmt::Debug for Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Runtime").field("device", &self.device()).field("placement", &self.placement()).finish()
    }
}

/// Builder for [`Runtime`].
pub struct RuntimeBuilder {
    stream: Option<CudaStream>,
    mr_selection: MemoryResourceSelection,
    device: Option<DeviceId>,
}

enum MemoryResourceSelection {
    Owned(DeviceMemoryResource),
    Pool { initial_pool_size: usize, maximum_pool_size: Option<usize> },
    DeviceMemoryManager,
}

impl Default for RuntimeBuilder {
    fn default() -> Self {
        Self { stream: None, mr_selection: MemoryResourceSelection::DeviceMemoryManager, device: None }
    }
}

impl Runtime {
    /// Start building a new explicit runtime.
    pub fn builder() -> RuntimeBuilder {
        RuntimeBuilder::default()
    }

    /// Build a new runtime on [`DeviceId::DEFAULT`] with a new stream and the
    /// device memory manager's resource.
    ///
    /// This is shorthand for `Runtime::builder().build()`.
    pub fn new() -> Result<Self> {
        Self::builder().build()
    }

    /// The CUDA device this runtime is pinned to.
    pub fn device(&self) -> DeviceId {
        self.domain.device_id()
    }

    /// Return a diagnostic summary of this runtime's retained execution placement.
    pub fn placement(&self) -> ExecutionPlacement {
        self.domain.placement()
    }

    /// Stable identity for this runtime's exact stream and memory-resource
    /// binding.
    pub fn binding_id(&self) -> ExecutionBindingId {
        self.domain.binding_id()
    }

    /// Return a counted lease on an extension generation for this exact
    /// stream/MR binding, initializing it if necessary.
    ///
    /// Values are shared by all runtimes derived without changing either raw
    /// execution handle. A runtime with a different stream or memory resource
    /// has a different extension store. Factories and extension destructors
    /// run outside the type-map and lifecycle locks. Values may retain a
    /// `Runtime`: the stable slot holds only a weak generation witness, while
    /// the returned lease keeps the execution domain alive until after value
    /// destruction.
    ///
    /// The returned value is opaque and can only be accessed through its
    /// scoped execution guard.
    pub fn get_or_try_init_binding_extension<T, E>(
        &self,
        operation: &'static str,
        factory: impl FnOnce() -> std::result::Result<T, E>,
    ) -> std::result::Result<BindingExtensionLease<T>, BindingExtensionInitError<E>>
    where
        T: Any + Send + Sync,
    {
        self.domain.binding_extensions().get_or_try_init(Arc::clone(&self.domain), operation, factory)
    }

    /// Synchronize the runtime's default stream.
    pub fn synchronize(&self) -> Result<()> {
        self.stream().synchronize()
    }

    /// Record a completion event on the runtime's default stream without blocking the host.
    ///
    /// The event orders after every operation already enqueued on this stream, including the
    /// reverse completion fences held across FFI calls by resolved executions. Synchronizing the
    /// event later observes the same completion as [`synchronize`](Self::synchronize), but the
    /// producing thread is free to return first while device-side consumers order through the
    /// retained provenance fences. A late GPU failure surfaces at the next synchronizing
    /// observation of the event. This call does not report it. The committer attributes
    /// the failure with its own task facts. The producer's worker facts are not used.
    pub fn record_completion_event(&self) -> Result<CudaEvent> {
        self.stream().record_completion_event()
    }

    /// Access the runtime's owned CUDA stream.
    pub fn stream(&self) -> &CudaStream {
        self.domain.stream()
    }

    /// Access the runtime's owned device memory resource.
    pub fn memory_resource(&self) -> &DeviceMemoryResource {
        self.domain.memory_resource()
    }

    /// Run on this runtime's stream and allocate outputs from its memory resource.
    pub fn exec(&self) -> Exec<'_> {
        Exec::new(self.stream(), self.memory_resource())
    }

    /// This runtime with the stream and memory resource `exec` selects as its
    /// defaults, so values it creates run there. [`Exec::RETAINED`] keeps this
    /// runtime's own.
    pub fn on(&self, exec: Exec<'_>) -> Result<Self> {
        let domain = exec.resolve_output_domain(&self.domain)?;
        Ok(Self { domain })
    }

    /// Resolve the device-level reservation-aware memory manager for this runtime.
    pub fn device_memory_manager(&self) -> Result<DeviceMemoryManager> {
        DeviceMemoryManager::for_device(self.device())
    }

    /// Derive a runtime whose complete retained-provenance object graph keeps
    /// `lifetime` alive. Every scalar, series, table, frame, and runtime
    /// override produced from the returned runtime inherits the same token.
    pub fn with_retained_lifetime(&self, lifetime: Arc<dyn Send + Sync>) -> Result<Self> {
        Ok(Self { domain: ExecutionDomain::with_retained_lifetime(&self.domain, lifetime)? })
    }

    /// Inherit the retained lifetime attached to `source`, if any, while
    /// preserving this runtime's own device, stream, and memory resource.
    pub fn inherit_retained_lifetime_from(&self, source: &Runtime) -> Result<Self> {
        Ok(Self { domain: ExecutionDomain::inherit_retained_lifetime(&self.domain, &source.domain)? })
    }

    /// Derive a runtime with a same-device memory resource.
    ///
    /// The derived runtime retains this runtime's device, stream and lifetime
    /// owners. Its execution binding changes when the allocator changes. Objects created
    /// through it retain that allocator and the inherited lifetime owners.
    /// A resource on another CUDA device is rejected. `operation` supplies
    /// the diagnostic operation label if derivation fails.
    pub fn with_memory_resource_override(&self, mr: &DeviceMemoryResource, operation: &'static str) -> Result<Self> {
        let domain = ExecutionDomain::resolve_with_borrowed_override(&self.domain, None, Some(mr), operation)?;
        Ok(Self { domain })
    }

    /// Borrow this runtime's explicit execution binding for cross-crate interop.
    pub fn execution_binding(&self) -> std::result::Result<ExecutionBinding<'_>, InteropError> {
        interop::execution::execution_binding("runtime::Runtime::execution_binding", self.execution_domain())
    }

    /// Order this runtime's stream after a producer execution binding.
    ///
    /// Same-device same-stream producers are already ordered and return
    /// [`StreamDependencyOutcome::NoOp`]. Same-device different-stream
    /// producers record an event on the producer stream and make the runtime
    /// stream wait on it, returning the barrier so callers can retain it until
    /// dependent work has been enqueued. Cross-device producers are rejected.
    pub fn order_after_execution_binding(
        &self,
        producer: ExecutionBinding<'_>,
        operation: &'static str,
    ) -> std::result::Result<StreamDependencyOutcome, InteropError> {
        let consumer_device = self.device().ordinal() as i32;
        if producer.device() != consumer_device {
            return Err(InteropError::new(
                InteropErrorKind::ExecutionBindingMismatch,
                "producer execution binding must match the runtime CUDA device",
            )
            .with_context("operation", operation)
            .with_context("left_device", producer.device().to_string())
            .with_context("right_device", consumer_device.to_string()));
        }

        let consumer_stream_raw = self.stream().interop_raw_handle();
        if producer.cuda_stream().as_raw() == consumer_stream_raw {
            return Ok(StreamDependencyOutcome::NoOp);
        }

        let _guard = DeviceGuard::new(self.device())
            .map_err(|error| map_stream_dependency_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
        let barrier = CudaEvent::new_on(self.device())
            .map_err(|error| map_stream_dependency_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
        let barrier_raw = barrier
            .interop_raw_handle()
            .map_err(|error| map_stream_dependency_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
        error::check_status_in_with_details(
            unsafe { cudf_sys::cudf_raw_cuda_event_record(barrier_raw, producer.cuda_stream().as_raw()) },
            operation,
            [
                ("readiness", "device_stream_ordered".to_string()),
                ("read_start", "consumer_stream".to_string()),
                ("producer_device", producer.device().to_string()),
                ("consumer_device", consumer_device.to_string()),
            ],
        )
        .map_err(|error| map_stream_dependency_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
        self.stream()
            .wait_event(&barrier)
            .map_err(|error| map_stream_dependency_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;

        Ok(StreamDependencyOutcome::Barrier(barrier))
    }

    /// Create a fixed-width low-level `Column` inside this runtime.
    pub fn column_from_slice<T: CudfType>(&self, data: &[T]) -> Result<Column> {
        Column::from_slice_with_runtime(self, data)
    }

    /// Create a nullable fixed-width low-level `Column` inside this runtime.
    pub fn column_from_slice_nullable<T: CudfType>(&self, data: &[T], validity: &[bool]) -> Result<Column> {
        Column::from_slice_nullable_with_runtime(self, data, validity)
    }

    /// Create a UTF-8 string low-level `Column` inside this runtime.
    pub fn column_from_strs(&self, data: &[&str]) -> Result<Column> {
        Column::from_strs_with_runtime(self, data)
    }

    /// Create a nullable UTF-8 string low-level `Column` inside this runtime.
    pub fn column_from_strs_nullable(&self, data: &[&str], validity: &[bool]) -> Result<Column> {
        Column::from_strs_nullable_with_runtime(self, data, validity)
    }

    /// Create an all-null low-level `Column` inside this runtime.
    pub fn null_column(&self, dtype: impl Into<DataType>, len: usize) -> Result<Column> {
        Column::nulls_with_runtime(self, dtype.into(), len)
    }

    /// Create a device scalar inside this runtime.
    ///
    /// The scalar is allocated on this runtime's stream and memory resource.
    /// Creation does not parse text, apply timezone behavior, or intentionally
    /// synchronize the host before returning. A decimal or day-resolution
    /// temporal value outside its storage range fails with a typed error.
    pub fn scalar(&self, value: impl Into<ScalarValue>) -> Result<Scalar> {
        Scalar::from_scalar_value_in_domain(&value.into(), self.execution_domain())
    }

    /// Create an empty low-level `Column` inside this runtime.
    pub fn empty_column(&self, dtype: impl Into<DataType>) -> Result<Column> {
        Column::empty_with_runtime(self, dtype.into())
    }

    /// Create a low-level `Column` by repeating a scalar inside this runtime.
    ///
    /// If `scalar` already belongs to this runtime's device, same-device
    /// cross-stream ordering is handled with a device-side dependency and the
    /// scalar is copied on this runtime's stream. If `scalar` belongs to a
    /// different CUDA device, cudf materializes the scalar value through
    /// the source execution domain and recreates it in this runtime before
    /// broadcasting, which may synchronize the source stream for host
    /// visibility. The output column is allocated with this runtime's memory
    /// resource and retains this runtime's execution provenance.
    pub fn column_from_scalar(&self, scalar: &Scalar, size: usize) -> Result<Column> {
        Column::from_scalar_with_runtime(self, scalar, size)
    }

    /// Create an empty low-level `Table` inside this runtime.
    pub fn empty_table(&self) -> Result<Table> {
        Table::empty_in_domain(Arc::clone(self.execution_domain()), "runtime::empty_table")
    }

    /// Allocate an owned device buffer on this runtime's CUDA device.
    ///
    /// The contents are uninitialized. For `size_bytes == 0`, the buffer is
    /// valid but empty.
    pub fn device_buffer(&self, size_bytes: usize) -> Result<DeviceBuffer> {
        DeviceBuffer::new_in_runtime(self, size_bytes)
    }

    /// Allocate an owned device buffer on this runtime's CUDA device and copy
    /// a host slice into it.
    pub fn device_buffer_from_host_data<T: CudfType>(&self, data: &[T]) -> Result<DeviceBuffer> {
        DeviceBuffer::from_host_data_in_runtime(self, data)
    }

    /// Create a low-level `Column` by copying from a borrowed device pointer
    /// into this runtime.
    ///
    /// Uses the runtime's stream and memory resource and does not synchronize
    /// before returning; the caller must keep `device_ptr` valid until this
    /// runtime's stream has completed the copy.
    ///
    /// # Safety
    ///
    /// - `device_ptr` must point to a valid device allocation containing at
    ///   least `len` elements of `type_id`.
    /// - Any prior GPU writes to `device_ptr` must be ordered before this
    ///   runtime's stream.
    /// - The pointer must remain valid until this runtime's stream has
    ///   completed the copy.
    pub unsafe fn column_from_device_ptr(
        &self,
        device_ptr: *const std::ffi::c_void,
        len: usize,
        type_id: TypeId,
    ) -> Result<Column> {
        // SAFETY: this method forwards the caller's documented pointer,
        // length, device-ordering, and lifetime guarantees unchanged.
        unsafe { Column::from_device_ptr_with_runtime(self, device_ptr, len, type_id) }
    }

    /// Create a low-level `Column` by adopting an owned `DeviceBuffer`
    /// inside this runtime without a device-to-device copy.
    pub fn column_from_device_buffer(&self, buffer: DeviceBuffer, row_count: usize, type_id: TypeId) -> Result<Column> {
        Column::from_device_buffer_with_runtime(self, buffer, row_count, type_id)
    }

    /// Create a nullable low-level `Column` by adopting owned data and null
    /// mask `DeviceBuffer`s inside this runtime without a device-to-device copy.
    pub fn column_from_device_buffer_nullable(
        &self,
        data_buffer: DeviceBuffer,
        null_mask_buffer: DeviceBuffer,
        null_count: usize,
        row_count: usize,
        type_id: TypeId,
    ) -> Result<Column> {
        Column::from_device_buffer_nullable_with_runtime(
            self,
            data_buffer,
            null_mask_buffer,
            null_count,
            row_count,
            type_id,
        )
    }

    /// Create an unnamed `Series` from a host slice inside this runtime.
    pub fn series<T: HostColumnData>(&self, data: &[T]) -> Result<Series> {
        Series::from_column(T::create_column_in_runtime(self, data)?)
    }

    /// Create an unnamed nullable `Series` from a host slice plus validity
    /// mask inside this runtime.
    pub fn nullable_series<T: HostColumnData>(&self, data: &[T], validity: &[bool]) -> Result<Series> {
        Series::from_column(T::create_column_nullable_in_runtime(self, data, validity)?)
    }

    /// Create a fixed-width `Series` from optional host values inside this runtime.
    ///
    /// `Some(value)` becomes a valid row; `None` becomes a null row. If every
    /// element is `Some`, this uses the non-nullable constructor path and does
    /// not allocate a null mask.
    pub fn series_opt<T: CudfType>(&self, data: &[Option<T>]) -> Result<Series> {
        let (values, validity, has_nulls) = Series::split_optional_host_values(data);
        if has_nulls {
            Series::from_column(self.column_from_slice_nullable(&values, &validity)?)
        } else {
            Series::from_column(self.column_from_slice(&values)?)
        }
    }

    /// Create an unnamed all-null `Series` inside this runtime.
    pub fn nulls(&self, dtype: impl Into<DataType>, len: usize) -> Result<Series> {
        Series::from_column(self.null_column(dtype, len)?)
    }

    /// Create an unnamed empty `Series` inside this runtime.
    pub fn empty_series(&self, dtype: impl Into<DataType>) -> Result<Series> {
        Series::from_column(self.empty_column(dtype)?)
    }

    /// Create an unnamed `Series` by repeating a scalar inside this runtime.
    ///
    /// If `scalar` already belongs to this runtime's device, same-device
    /// cross-stream ordering is handled with a device-side dependency and the
    /// scalar is copied on this runtime's stream. If `scalar` belongs to a
    /// different CUDA device, cudf materializes the scalar value through
    /// the source execution domain and recreates it in this runtime before
    /// broadcasting, which may synchronize the source stream for host
    /// visibility. The output `Series` shares the newly allocated column and
    /// retains this runtime's execution provenance.
    ///
    /// Supports all scalar types except Dictionary32 (no libcudf scalar
    /// subclass exists for dictionary-encoded scalars).
    pub fn series_from_scalar(&self, scalar: &Scalar, size: usize) -> Result<Series> {
        Series::from_column(self.column_from_scalar(scalar, size)?)
    }

    /// Create an unnamed fixed-width `Series` by repeating `value` inside this
    /// runtime.
    pub fn series_from_value<T: CudfType>(&self, value: T, len: usize) -> Result<Series> {
        self.series_from_scalar(&self.scalar(value)?, len)
    }

    /// Create an unnamed fixed-width arithmetic sequence `Series` inside this runtime.
    ///
    /// The output is `[start, start + step, start + 2*step, ...]` with
    /// `size` rows, generated directly on device via libcudf sequence
    /// filling.
    pub fn sequence<T: CudfType>(&self, start: T, step: T, size: usize) -> Result<Series> {
        Series::from_column(Column::from_sequence_with_runtime(self, start, step, size)?)
    }

    /// Create an empty `DataFrame` inside this runtime.
    pub fn empty_dataframe(&self) -> DataFrame {
        DataFrame::empty_in_domain(Arc::clone(self.execution_domain()))
    }

    pub(crate) fn execution_domain(&self) -> &Arc<ExecutionDomain> {
        &self.domain
    }
    #[cfg(any(test, feature = "test-hooks"))]
    pub(crate) fn __debug_execution_domain_id(&self) -> u64 {
        self.domain.debug_id()
    }
    #[cfg(any(test, feature = "test-hooks"))]
    pub(crate) fn __debug_retained_lifetime_count(&self) -> usize {
        self.domain.debug_retained_lifetime_count()
    }
}

impl RuntimeBuilder {
    fn runtime_binding_mismatch(
        resource: &'static str,
        left_device: DeviceId,
        right_device: DeviceId,
        message: &'static str,
    ) -> Error {
        Error::new_with(ErrorCode::RuntimeBindingMismatch, "runtime::RuntimeBuilder::build", message)
            .with_fact(FACT_LEFT_DEVICE, DeviceIdFact(left_device.ordinal_raw()))
            .with_fact(FACT_RIGHT_DEVICE, DeviceIdFact(right_device.ordinal_raw()))
            .with_detail("resource", resource)
            .with_detail("reason", "cross_device_runtime_binding")
            .with_detail(
                "remedy",
                "use stream and memory-resource handles created on one CUDA device, and make with_device(...) match that device when it is set explicitly",
            )
    }

    /// Pin the runtime to a specific CUDA device.
    ///
    /// If not called, the runtime uses the device of the supplied stream or
    /// memory resource, or [`DeviceId::DEFAULT`] when neither is supplied. The
    /// calling thread's current CUDA device is never consulted.
    pub fn with_device(mut self, device: DeviceId) -> Self {
        self.device = Some(device);
        self
    }

    /// Override the runtime's owned stream.
    pub fn with_stream(mut self, stream: CudaStream) -> Self {
        self.stream = Some(stream);
        self
    }

    /// Override the runtime's owned default device memory resource.
    ///
    /// The supplied handle receives only the *output* allocations of operations
    /// that execute in this runtime, matching upstream libcudf's `mr` parameter.
    /// Temporary allocations made inside libcudf kernels always resolve through
    /// the device's long-lived memory domain (the per-device current resource
    /// installed at domain creation) and are accounted per stream; they are never
    /// routed into this resource. Use this when output buffers must live on a
    /// specific allocator (for example, interop buffers passed to an external
    /// system).
    ///
    /// For memory budgeting, do not reach for an owned resource. Attach the
    /// runtime's stream to a [`QueryMemoryDomain`](crate::QueryMemoryDomain)
    /// through [`QueryMemoryDomain::reserve_stream`](crate::QueryMemoryDomain::reserve_stream);
    /// the domain caps outputs *and* temporaries for the pipeline.
    ///
    /// If [`with_device`](Self::with_device) is also set, `mr` must have been
    /// created on that same device. Otherwise, `build()` rejects the
    /// combination as a cross-device binding mismatch.
    pub fn with_memory_resource(mut self, mr: DeviceMemoryResource) -> Self {
        self.mr_selection = MemoryResourceSelection::Owned(mr);
        self
    }

    /// Configure the runtime to build an owned pool memory resource during `build()`.
    ///
    /// The pool is created only after the runtime target device has been
    /// resolved and entered, so this path does not depend on unrelated ambient
    /// current-device state at call time.
    ///
    /// Like [`with_memory_resource`](Self::with_memory_resource), the pool receives
    /// only output allocations. libcudf temporaries route through the device's
    /// memory domain and are accounted per stream. An owned pool is therefore not
    /// a budgeting tool. Use the default device-level memory manager plus a stream
    /// member of a [`QueryMemoryDomain`](crate::QueryMemoryDomain) to bound a
    /// pipeline's memory.
    ///
    /// - `initial_pool_size`: initial pool allocation in bytes.
    /// - `maximum_pool_size`: maximum pool size in bytes, or `None` for unbounded.
    pub fn with_pool_memory_resource(mut self, initial_pool_size: usize, maximum_pool_size: Option<usize>) -> Self {
        self.mr_selection = MemoryResourceSelection::Pool { initial_pool_size, maximum_pool_size };
        self
    }

    /// Use the device-level reservation-aware memory manager as this runtime's default allocator.
    ///
    /// This is the query-backend path. The runtime keeps one explicit stream, and
    /// allocator accounting is controlled by stream-bound reservation leases. The
    /// memory resource is not replaced for each execution.
    pub fn with_device_memory_manager(mut self) -> Self {
        self.mr_selection = MemoryResourceSelection::DeviceMemoryManager;
        self
    }

    /// Build a runtime with one explicit retained provenance bundle.
    ///
    /// If [`with_device`](Self::with_device) was called, resources are created
    /// on that device.  Otherwise the device is inferred from supplied handles,
    /// or defaults to [`DeviceId::DEFAULT`].
    ///
    /// The resulting runtime owns exactly one stream. Unless a memory resource
    /// is selected, it uses the device-level reservation-aware memory manager as
    /// its default memory resource. Borrowed per-operation overrides can
    /// temporarily replace either handle for a supported call, but they do not
    /// mutate this owned bundle.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - An explicit device is given and a supplied handle was created on a
    ///   different device.
    /// - No explicit device is given and both handles were supplied but created
    ///   on different devices.
    pub fn build(self) -> Result<Runtime> {
        let device = self.resolve_device()?;
        let RuntimeBuilder { stream, mr_selection, .. } = self;
        let device_manager = DeviceMemoryManager::for_device(device)?;

        let stream = match stream {
            Some(stream) => stream,
            None => CudaStream::new_on(device)?,
        };
        let mr = match mr_selection {
            MemoryResourceSelection::Owned(mr) => mr,
            MemoryResourceSelection::Pool { initial_pool_size, maximum_pool_size } => {
                DeviceMemoryResource::pool_on(device, initial_pool_size, maximum_pool_size)?
            },
            MemoryResourceSelection::DeviceMemoryManager => device_manager.resource().clone(),
        };
        let domain = ExecutionDomain::from_handles_on_device(device, stream, mr)?;
        Ok(Runtime { domain })
    }

    /// Resolve the target device, validating handle provenance.
    fn resolve_device(&self) -> Result<DeviceId> {
        if let Some(target) = self.device {
            // Explicit device; validate supplied handles agree.
            if let Some(ref stream) = self.stream
                && stream.device() != target
            {
                return Err(Self::runtime_binding_mismatch(
                    "stream",
                    target,
                    stream.device(),
                    "cross-device stream: stream was created on a different device than the runtime target",
                ));
            }
            if let MemoryResourceSelection::Owned(ref mr) = self.mr_selection
                && mr.device() != target
            {
                return Err(Self::runtime_binding_mismatch(
                    "memory_resource",
                    target,
                    mr.device(),
                    "cross-device memory resource: MR was created on a different device than the runtime target",
                ));
            }
            Ok(target)
        } else {
            // No explicit device; infer from handles.
            match (&self.stream, &self.mr_selection) {
                (Some(stream), MemoryResourceSelection::Owned(mr)) => {
                    if stream.device() != mr.device() {
                        return Err(Self::runtime_binding_mismatch(
                            "stream_and_memory_resource",
                            stream.device(),
                            mr.device(),
                            "cross-device handles: stream and memory resource were created on different devices",
                        ));
                    }
                    Ok(stream.device())
                },
                (Some(stream), _) => Ok(stream.device()),
                (None, MemoryResourceSelection::Owned(mr)) => Ok(mr.device()),
                (None, MemoryResourceSelection::Pool { .. } | MemoryResourceSelection::DeviceMemoryManager) => {
                    Ok(DeviceId::DEFAULT)
                },
            }
        }
    }
}
