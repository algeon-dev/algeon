//! Public owner for explicit stream / allocator defaults and retained provenance seeding.

use std::sync::Arc;

use super::{device::DeviceId, device_guard::DeviceGuard, execution_domain::ExecutionDomain, stream::CudaStream};
use crate::{
    core::{
        column::{Column, DeviceBuffer},
        error::{Error, ErrorKind, Result},
        memory::DeviceMemoryResource,
        scalar::Scalar,
        table::Table,
        types::{CudfType, DataType, TypeId},
    },
    frame::{
        dataframe::DataFrame,
        series::{HostColumnData, Series},
    },
};

/// Owns one explicit retained provenance bundle for a `cudf-rs` pipeline.
///
/// Each `Runtime` is pinned to a specific CUDA device (see
/// [`RuntimeBuilder::with_device`]).  All resources created through this
/// runtime live on that device.
///
/// **Multi-GPU execution guarantee:** when an operation is invoked through
/// this runtime, the runtime's device is re-entered at the FFI boundary before
/// any GPU work is enqueued. This ensures that compute runs on the correct
/// device regardless of the calling thread's current CUDA device. The
/// previous device is restored after each FFI call.
#[derive(Clone)]
pub struct Runtime {
    domain: Arc<ExecutionDomain>,
}

// SAFETY: Runtime owns an immutable retained-provenance handle bundle. The
// underlying stream and memory-resource handles are thread-safe wrappers.
unsafe impl Send for Runtime {}
unsafe impl Sync for Runtime {}

impl std::fmt::Debug for Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Runtime")
            .field("device", &self.device())
            .field("retained_provenance", &self.__debug_execution_domain_id())
            .finish()
    }
}

/// Builder for [`Runtime`].
pub struct RuntimeBuilder {
    stream: Option<CudaStream>,
    mr_selection: MemoryResourceSelection,
    device: Option<DeviceId>,
}

enum MemoryResourceSelection {
    DefaultCuda,
    Owned(DeviceMemoryResource),
    Pool { initial_pool_size: usize, maximum_pool_size: Option<usize> },
}

impl Default for RuntimeBuilder {
    fn default() -> Self {
        Self { stream: None, mr_selection: MemoryResourceSelection::DefaultCuda, device: None }
    }
}

impl Runtime {
    /// Start building a new explicit runtime.
    pub fn builder() -> RuntimeBuilder {
        RuntimeBuilder::default()
    }

    /// Build a new explicit runtime with default device, stream, and memory resource selection.
    ///
    /// This is shorthand for `Runtime::builder().build()`.
    pub fn new() -> Result<Self> {
        Self::builder().build()
    }

    /// The CUDA device this runtime is pinned to.
    pub fn device(&self) -> DeviceId {
        self.domain.device().expect("runtime always owns an explicit domain with a device")
    }

    /// Synchronize the runtime's default stream.
    pub fn synchronize(&self) -> Result<()> {
        self.stream().synchronize()
    }

    /// Access the runtime's owned CUDA stream.
    pub fn stream(&self) -> &CudaStream {
        self.domain.explicit_stream().expect("runtime always owns an explicit stream")
    }

    /// Access the runtime's owned device memory resource.
    pub fn memory_resource(&self) -> &DeviceMemoryResource {
        self.domain.explicit_memory_resource().expect("runtime always owns an explicit memory resource")
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

    /// Create a fixed-width scalar inside this runtime.
    pub fn scalar<T: CudfType>(&self, value: T) -> Result<Scalar> {
        Scalar::from_value_in_domain(value, self.execution_domain())
    }

    /// Create a UTF-8 scalar inside this runtime.
    pub fn string_scalar(&self, value: &str) -> Result<Scalar> {
        Scalar::from_str_in_domain(value, self.execution_domain())
    }

    /// Create a null fixed-width scalar inside this runtime.
    pub fn null_scalar<T: CudfType>(&self) -> Result<Scalar> {
        Scalar::null_numeric_in_domain::<T>(self.execution_domain())
    }

    /// Create a null UTF-8 scalar inside this runtime.
    pub fn null_string_scalar(&self) -> Result<Scalar> {
        Scalar::null_string_in_domain(self.execution_domain())
    }

    /// Create an empty low-level `Column` inside this runtime.
    pub fn empty_column(&self, dtype: impl Into<DataType>) -> Result<Column> {
        Column::empty_with_runtime(self, dtype.into())
    }

    /// Create a low-level `Column` by repeating a scalar inside this runtime.
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
        Column::from_device_ptr_with_runtime(self, device_ptr, len, type_id)
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

    /// Create a `Series` from a host slice inside this runtime.
    pub fn series<T: HostColumnData>(&self, name: impl Into<String>, data: &[T]) -> Result<Series> {
        Series::from_named_column(name, T::create_column_in_runtime(self, data)?)
    }

    /// Create a nullable `Series` from a host slice plus validity mask inside this runtime.
    pub fn nullable_series<T: HostColumnData>(
        &self,
        name: impl Into<String>,
        data: &[T],
        validity: &[bool],
    ) -> Result<Series> {
        Series::from_named_column(name, T::create_column_nullable_in_runtime(self, data, validity)?)
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

    /// Create a named fixed-width `Series` from optional host values inside
    /// this runtime.
    pub fn named_series_opt<T: CudfType>(&self, name: impl Into<String>, data: &[Option<T>]) -> Result<Series> {
        self.series_opt(data).map(|series| series.rename(name))
    }

    /// Create an unnamed all-null `Series` inside this runtime.
    pub fn nulls(&self, dtype: impl Into<DataType>, len: usize) -> Result<Series> {
        Series::from_column(self.null_column(dtype, len)?)
    }

    /// Create a named all-null `Series` inside this runtime.
    pub fn named_nulls(&self, name: impl Into<String>, dtype: impl Into<DataType>, len: usize) -> Result<Series> {
        Series::from_named_column(name, self.null_column(dtype, len)?)
    }

    /// Create an unnamed empty `Series` inside this runtime.
    pub fn empty_series(&self, dtype: impl Into<DataType>) -> Result<Series> {
        Series::from_column(self.empty_column(dtype)?)
    }

    /// Create a named empty `Series` inside this runtime.
    pub fn named_empty_series(&self, name: impl Into<String>, dtype: impl Into<DataType>) -> Result<Series> {
        Series::from_named_column(name, self.empty_column(dtype)?)
    }

    /// Create an unnamed `Series` by repeating a scalar inside this runtime.
    ///
    /// If `scalar` does not already belong to this runtime's execution domain,
    /// the scalar is recreated in this runtime first so placement and
    /// allocator selection remain honest.
    ///
    /// Supports all scalar types except Dictionary32 (no libcudf scalar
    /// subclass exists for dictionary-encoded scalars).
    pub fn series_from_scalar(&self, scalar: &Scalar, size: usize) -> Result<Series> {
        Series::from_column(self.column_from_scalar(scalar, size)?)
    }

    /// Create a named `Series` by repeating a scalar inside this runtime.
    ///
    /// If `scalar` does not already belong to this runtime's execution domain,
    /// the scalar is recreated in this runtime first so placement and
    /// allocator selection remain honest.
    ///
    /// Supports all scalar types except Dictionary32 (no libcudf scalar
    /// subclass exists for dictionary-encoded scalars).
    pub fn named_series_from_scalar(&self, name: impl Into<String>, scalar: &Scalar, size: usize) -> Result<Series> {
        Series::from_named_column(name, self.column_from_scalar(scalar, size)?)
    }

    /// Create an unnamed fixed-width arithmetic sequence `Series` inside this runtime.
    ///
    /// The output is `[start, start + step, start + 2*step, ...]` with
    /// `size` rows, generated directly on device via libcudf sequence
    /// filling.
    pub fn sequence<T: CudfType>(&self, start: T, step: T, size: usize) -> Result<Series> {
        Series::from_column(Column::from_sequence_with_runtime(self, start, step, size)?)
    }

    /// Create a named fixed-width arithmetic sequence `Series` inside this runtime.
    ///
    /// The output is `[start, start + step, start + 2*step, ...]` with
    /// `size` rows, generated directly on device via libcudf sequence
    /// filling.
    pub fn named_sequence<T: CudfType>(
        &self,
        name: impl Into<String>,
        start: T,
        step: T,
        size: usize,
    ) -> Result<Series> {
        Series::from_named_column(name, Column::from_sequence_with_runtime(self, start, step, size)?)
    }

    /// Create an empty `DataFrame` inside this runtime.
    pub fn empty_dataframe(&self) -> DataFrame {
        DataFrame::empty_in_domain(Arc::clone(self.execution_domain()))
    }

    pub(crate) fn execution_domain(&self) -> &Arc<ExecutionDomain> {
        &self.domain
    }
    pub(crate) fn __debug_execution_domain_id(&self) -> u64 {
        self.domain.debug_id()
    }
}

impl RuntimeBuilder {
    /// Pin the runtime to a specific CUDA device.
    ///
    /// If not called, the runtime uses the calling thread's current device
    /// at build time (typically device 0).
    pub fn with_device(mut self, device: DeviceId) -> Self {
        self.device = Some(device);
        self
    }

    /// Override the runtime's owned stream.
    pub fn with_stream(mut self, stream: CudaStream) -> Self {
        self.stream = Some(stream);
        self
    }

    /// Override the runtime's owned device memory resource.
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
    /// - `initial_pool_size`: initial pool allocation in bytes.
    /// - `maximum_pool_size`: maximum pool size in bytes, or `None` for unbounded.
    pub fn with_pool_memory_resource(mut self, initial_pool_size: usize, maximum_pool_size: Option<usize>) -> Self {
        self.mr_selection = MemoryResourceSelection::Pool { initial_pool_size, maximum_pool_size };
        self
    }

    /// Build a runtime with one explicit retained provenance bundle.
    ///
    /// If [`with_device`](Self::with_device) was called, resources are created
    /// on that device.  Otherwise the device is inferred from supplied handles,
    /// or defaults to the calling thread's current device.
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

        // Temporarily switch to the target device so that any auto-created
        // stream or memory resource lands on the correct GPU.
        let _guard = DeviceGuard::new(device)?;

        let stream = match stream {
            Some(stream) => stream,
            None => CudaStream::new()?,
        };
        let mr = match mr_selection {
            MemoryResourceSelection::DefaultCuda => DeviceMemoryResource::cuda_on(device)?,
            MemoryResourceSelection::Owned(mr) => mr,
            MemoryResourceSelection::Pool { initial_pool_size, maximum_pool_size } => {
                DeviceMemoryResource::pool_on(device, initial_pool_size, maximum_pool_size)?
            },
        };
        let domain = ExecutionDomain::from_handles_on_device(device, stream, mr)?;
        Ok(Runtime { domain })
    }

    /// Resolve the target device, validating handle provenance.
    fn resolve_device(&self) -> Result<DeviceId> {
        let op = "runtime::RuntimeBuilder::build";

        if let Some(target) = self.device {
            // Explicit device — validate supplied handles agree.
            if let Some(ref stream) = self.stream {
                if stream.device() != target {
                    return Err(Error::new(
                        ErrorKind::InvalidOperation,
                        "cross-device stream: stream was created on a different device than the runtime target",
                    )
                    .with_operation(op)
                    .with_context("target_device", target.ordinal().to_string())
                    .with_context("stream_device", stream.device().ordinal().to_string()));
                }
            }
            if let MemoryResourceSelection::Owned(ref mr) = self.mr_selection {
                if mr.device() != target {
                    return Err(Error::new(
                        ErrorKind::InvalidOperation,
                        "cross-device memory resource: MR was created on a different device than the runtime target",
                    )
                    .with_operation(op)
                    .with_context("target_device", target.ordinal().to_string())
                    .with_context("mr_device", mr.device().ordinal().to_string()));
                }
            }
            Ok(target)
        } else {
            // No explicit device — infer from handles.
            match (&self.stream, &self.mr_selection) {
                (Some(stream), MemoryResourceSelection::Owned(mr)) => {
                    if stream.device() != mr.device() {
                        return Err(Error::new(
                            ErrorKind::InvalidOperation,
                            "cross-device handles: stream and memory resource were created on different devices",
                        )
                        .with_operation(op)
                        .with_context("stream_device", stream.device().ordinal().to_string())
                        .with_context("mr_device", mr.device().ordinal().to_string()));
                    }
                    Ok(stream.device())
                },
                (Some(stream), _) => Ok(stream.device()),
                (None, MemoryResourceSelection::Owned(mr)) => Ok(mr.device()),
                (None, MemoryResourceSelection::DefaultCuda | MemoryResourceSelection::Pool { .. }) => {
                    DeviceId::current()
                },
            }
        }
    }
}
