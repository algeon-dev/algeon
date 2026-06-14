//! Public owner for device-pinned default stream, allocator provenance, and retained provenance.

use std::sync::Arc;

use rapids_interop::{ExecutionBinding, InteropError, InteropErrorKind};

use super::{
    ExecutionOverrides,
    boundary::{BoundaryObserver, BoundaryObserverRef},
    budget::{BudgetedExecutionOutcome, ExecutionBudgetReport, ExecutionBudgetRequest, ExecutionBudgetToken},
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
        error::{self, DeviceIdFact, Error, ErrorCode, ErrorKind, FACT_LEFT_DEVICE, FACT_RIGHT_DEVICE, Result},
        memory::{DeviceMemoryManager, DeviceMemoryResource, StreamReservationLease},
        scalar::Scalar,
        table::Table,
        types::{CudfType, DataType, TimestampResolution, TypeId},
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

fn annotate_runtime_budget_execution_error(
    error: Error,
    token: &ExecutionBudgetToken,
    operation: &'static str,
) -> Error {
    match error.kind() {
        ErrorKind::ResourceExhausted | ErrorKind::DependencyFailure => token.annotate_error(error, operation),
        _ => error.reanchor_if_needed(operation),
    }
}

const RUNTIME_SCALAR_TO_SERIES_TEMPORARY_WORKSPACE_CONTRACT_GAP: &str =
    "libcudf_scalar_to_series_temporary_workspace_preflight_unavailable";
const RUNTIME_DECIMAL_SCALED_NULLS_BUDGETED_CAST_CONTRACT_GAP: &str =
    "libcudf_decimal_scaled_nulls_budgeted_cast_preflight_unavailable";

fn budgeted_runtime_for_operation(
    runtime: &Runtime,
    token: &ExecutionBudgetToken,
    operation: &'static str,
) -> Result<Runtime> {
    runtime
        .with_execution_overrides(token.stream(), token.memory_resource(), operation)
        .map_err(|error| annotate_runtime_budget_execution_error(error, token, operation))
}

fn budgeted_series_report(
    series: &Series,
    token: &ExecutionBudgetToken,
    rows: usize,
    operation: &'static str,
    unresolved_contract: Option<&'static str>,
) -> Result<ExecutionBudgetReport> {
    let output_bytes = series.device_allocated_size_bytes().map_err(|error| error.reanchor(operation))?;
    let mut report = token
        .reservation_enforced_report()
        .with_retained_bytes(output_bytes)
        .with_output_bytes(output_bytes)
        .with_output_rows(rows);
    if let Some(unresolved_contract) = unresolved_contract {
        report = report.with_unresolved_contract(unresolved_contract);
    }
    Ok(report)
}

/// Owns one device-pinned retained-provenance bundle for a `cudf-rs` pipeline.
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
/// execution placement. For query-style budgets, prefer
/// [`Runtime::execution_budget_token`] or [`Runtime::stream_reservation`]
/// instead of constructing a separate memory resource. Create an explicit
/// [`DeviceMemoryResource`] first only when you need an advanced allocator
/// handle to reuse across runtimes or borrow through
/// [`crate::ExecutionOverrides`] for one operation.
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
        f.debug_struct("Runtime").field("device", &self.device()).field("placement", &self.placement()).finish()
    }
}

/// Builder for [`Runtime`].
pub struct RuntimeBuilder {
    stream: Option<CudaStream>,
    mr_selection: MemoryResourceSelection,
    device: Option<DeviceId>,
    boundary_observer: Option<BoundaryObserverRef>,
}

enum MemoryResourceSelection {
    Owned(DeviceMemoryResource),
    Pool { initial_pool_size: usize, maximum_pool_size: Option<usize> },
    DeviceMemoryManager,
}

impl Default for RuntimeBuilder {
    fn default() -> Self {
        Self {
            stream: None,
            mr_selection: MemoryResourceSelection::DeviceMemoryManager,
            device: None,
            boundary_observer: None,
        }
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
        self.domain.device_id()
    }

    /// Return a diagnostic summary of this runtime's retained execution placement.
    pub fn placement(&self) -> ExecutionPlacement {
        self.domain.placement()
    }

    /// Synchronize the runtime's default stream.
    pub fn synchronize(&self) -> Result<()> {
        self.stream().synchronize()
    }

    /// Access the runtime's owned CUDA stream.
    pub fn stream(&self) -> &CudaStream {
        self.domain.stream()
    }

    /// Access the runtime's owned device memory resource.
    pub fn memory_resource(&self) -> &DeviceMemoryResource {
        self.domain.memory_resource()
    }

    /// Create a hard execution-budget token using this runtime's default
    /// stream and a device-level reservation-aware memory manager.
    pub fn execution_budget_token(&self, request: ExecutionBudgetRequest) -> Result<ExecutionBudgetToken> {
        ExecutionBudgetToken::from_runtime(self, request)
    }

    /// Resolve the device-level reservation-aware memory manager for this runtime.
    pub fn device_memory_manager(&self) -> Result<DeviceMemoryManager> {
        DeviceMemoryManager::for_device(self.device())
    }

    /// Attach a stream-bound reservation to this runtime's default stream.
    pub fn stream_reservation(&self, reservation_id: u64, limit_bytes: usize) -> Result<StreamReservationLease> {
        self.device_memory_manager()?.reserve_stream(self.stream(), reservation_id, limit_bytes)
    }

    pub(crate) fn with_memory_resource_override(
        &self,
        mr: &DeviceMemoryResource,
        operation: &'static str,
    ) -> Result<Self> {
        let domain = ExecutionDomain::resolve_with_borrowed_override(&self.domain, None, Some(mr), operation)?;
        Ok(Self { domain })
    }

    pub(crate) fn with_execution_overrides(
        &self,
        stream: &CudaStream,
        mr: &DeviceMemoryResource,
        operation: &'static str,
    ) -> Result<Self> {
        let domain = ExecutionDomain::resolve_with_borrowed_override(&self.domain, Some(stream), Some(mr), operation)?;
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

        let consumer_stream_raw = self
            .stream()
            .interop_raw_handle()
            .map_err(|error| map_stream_dependency_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
        if producer.cuda_stream().as_raw() == consumer_stream_raw {
            return Ok(StreamDependencyOutcome::NoOp);
        }

        let _guard = DeviceGuard::new(self.device())
            .map_err(|error| map_stream_dependency_error(InteropErrorKind::ReadinessUnavailable, operation, error))?;
        let barrier = CudaEvent::new()
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

    /// Create a null scalar for an arbitrary supported [`DataType`].
    ///
    /// This is the dtype-erased counterpart to [`Self::null_scalar`] /
    /// [`Self::null_string_scalar`] and dispatches on the [`TypeId`].
    /// Use it when the target data type is only known at runtime — for
    /// example when building a fill scalar from a column or schema. Fails
    /// with [`ErrorCode::UnsupportedDtype`](crate::core::error::ErrorCode)
    /// for types whose null encoding is not yet exposed by cudf-rs.
    ///
    /// [`DataType`]: crate::DataType
    pub fn null_scalar_with_data_type(&self, dtype: impl Into<DataType>) -> Result<Scalar> {
        Scalar::null_in_domain(dtype.into(), self.execution_domain())
    }

    /// Create a Decimal32 scalar from its raw fixed-point representation.
    ///
    /// `raw` is the stored integer representation and `scale` is the cudf
    /// fixed-point scale. This creates a device-resident scalar in this
    /// runtime's execution domain, using this runtime's stream and memory
    /// resource. It does not parse or format decimal text, and does not
    /// intentionally synchronize the host before returning.
    pub fn decimal32_raw_scalar(&self, raw: i32, scale: i32) -> Result<Scalar> {
        Scalar::decimal_from_raw_in_domain(
            i128::from(raw),
            DataType::fixed_point(TypeId::Decimal32, scale),
            self.execution_domain(),
        )
    }

    /// Create a Decimal64 scalar from its raw fixed-point representation.
    ///
    /// `raw` is the stored integer representation and `scale` is the cudf
    /// fixed-point scale. This creates a device-resident scalar in this
    /// runtime's execution domain, using this runtime's stream and memory
    /// resource. It does not parse or format decimal text, and does not
    /// intentionally synchronize the host before returning.
    pub fn decimal64_raw_scalar(&self, raw: i64, scale: i32) -> Result<Scalar> {
        Scalar::decimal_from_raw_in_domain(
            i128::from(raw),
            DataType::fixed_point(TypeId::Decimal64, scale),
            self.execution_domain(),
        )
    }

    /// Create a Decimal128 scalar from its raw fixed-point representation.
    ///
    /// `raw` is the stored integer representation and `scale` is the cudf
    /// fixed-point scale. This creates a device-resident scalar in this
    /// runtime's execution domain, using this runtime's stream and memory
    /// resource. It does not parse or format decimal text, and does not
    /// intentionally synchronize the host before returning.
    pub fn decimal128_raw_scalar(&self, raw: i128, scale: i32) -> Result<Scalar> {
        Scalar::decimal_from_raw_in_domain(
            raw,
            DataType::fixed_point(TypeId::Decimal128, scale),
            self.execution_domain(),
        )
    }

    /// Create a decimal scalar from its raw fixed-point representation.
    ///
    /// `dtype` must be Decimal32, Decimal64, or Decimal128. For Decimal32 and
    /// Decimal64, `raw` must fit the corresponding physical representation.
    /// This creates a device-resident scalar in this runtime's execution
    /// domain, using this runtime's stream and memory resource. It does not
    /// parse or format decimal text, and does not intentionally synchronize
    /// the host before returning.
    pub fn decimal_raw_scalar(&self, raw: i128, dtype: impl Into<DataType>) -> Result<Scalar> {
        Scalar::decimal_from_raw_in_domain(raw, dtype.into(), self.execution_domain())
    }

    /// Create a timestamp scalar from a raw count in the requested resolution.
    ///
    /// `raw` is the stored epoch-unit representation for `resolution`. This
    /// creates a device-resident scalar in this runtime's execution domain,
    /// using this runtime's stream and memory resource. It does not parse
    /// strings, apply timezone behavior, or intentionally synchronize the host
    /// before returning.
    pub fn timestamp_raw_scalar(&self, raw: i64, resolution: TimestampResolution) -> Result<Scalar> {
        Scalar::timestamp_from_raw_in_domain(raw, resolution.type_id(), self.execution_domain())
    }

    /// Create a duration scalar from a raw count in the requested resolution.
    ///
    /// `raw` is the stored unit-count representation for `resolution`. This
    /// creates a device-resident scalar in this runtime's execution domain,
    /// using this runtime's stream and memory resource. It does not parse
    /// strings, apply formatting behavior, or intentionally synchronize the
    /// host before returning.
    pub fn duration_raw_scalar(&self, raw: i64, resolution: TimestampResolution) -> Result<Scalar> {
        Scalar::duration_from_raw_in_domain(raw, resolution.duration_type_id(), self.execution_domain())
    }

    /// Create a null Decimal32 device scalar with the requested scale.
    pub fn null_decimal32_raw_scalar(&self, scale: i32) -> Result<Scalar> {
        self.null_decimal_raw_scalar(DataType::fixed_point(TypeId::Decimal32, scale))
    }

    /// Create a null Decimal64 device scalar with the requested scale.
    pub fn null_decimal64_raw_scalar(&self, scale: i32) -> Result<Scalar> {
        self.null_decimal_raw_scalar(DataType::fixed_point(TypeId::Decimal64, scale))
    }

    /// Create a null Decimal128 device scalar with the requested scale.
    pub fn null_decimal128_raw_scalar(&self, scale: i32) -> Result<Scalar> {
        self.null_decimal_raw_scalar(DataType::fixed_point(TypeId::Decimal128, scale))
    }

    /// Create a null decimal device scalar.
    ///
    /// `dtype` must be Decimal32, Decimal64, or Decimal128. The scalar is
    /// created in this runtime's execution domain, using this runtime's stream
    /// and memory resource, without intentionally synchronizing the host
    /// before returning.
    pub fn null_decimal_raw_scalar(&self, dtype: impl Into<DataType>) -> Result<Scalar> {
        Scalar::null_decimal_in_domain(dtype.into(), self.execution_domain())
    }

    /// Create a null timestamp device scalar in the requested resolution.
    ///
    /// The scalar is created in this runtime's execution domain, using this
    /// runtime's stream and memory resource, without intentionally
    /// synchronizing the host before returning.
    pub fn null_timestamp_raw_scalar(&self, resolution: TimestampResolution) -> Result<Scalar> {
        Scalar::null_timestamp_in_domain(resolution.type_id(), self.execution_domain())
    }

    /// Create a null duration device scalar in the requested resolution.
    ///
    /// The scalar is created in this runtime's execution domain, using this
    /// runtime's stream and memory resource, without intentionally
    /// synchronizing the host before returning.
    pub fn null_duration_raw_scalar(&self, resolution: TimestampResolution) -> Result<Scalar> {
        Scalar::null_duration_in_domain(resolution.duration_type_id(), self.execution_domain())
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
    /// different CUDA device, cudf-rs materializes the scalar value through
    /// the source execution domain and recreates it in this runtime before
    /// broadcasting, which may synchronize the source stream for host
    /// visibility. The output column is allocated with this runtime's memory
    /// resource and retains this runtime's execution provenance.
    pub fn column_from_scalar(&self, scalar: &Scalar, size: usize) -> Result<Column> {
        Column::from_scalar_with_runtime(self, scalar, size)
    }

    /// Create a low-level `Column` by repeating a scalar under a hard execution-budget token.
    ///
    /// This has the same scalar placement semantics as
    /// [`column_from_scalar`](Self::column_from_scalar), but the scalar
    /// relocation, if needed, and the output column allocation are routed
    /// through the token's stream-bound reservation. The returned report
    /// records the observed output column footprint and row count.
    pub fn column_from_scalar_with_budget(
        &self,
        scalar: &Scalar,
        size: usize,
        token: &ExecutionBudgetToken,
    ) -> Result<BudgetedExecutionOutcome<Column>> {
        let operation = "runtime::column_from_scalar_with_budget";
        let budgeted_runtime = budgeted_runtime_for_operation(self, token, operation)?;
        let output = budgeted_runtime
            .column_from_scalar(scalar, size)
            .map_err(|error| annotate_runtime_budget_execution_error(error, token, operation))?;
        let output_bytes = output.device_allocated_size_bytes().map_err(|error| error.reanchor(operation))?;
        let report = token
            .reservation_enforced_report()
            .with_retained_bytes(output_bytes)
            .with_output_bytes(output_bytes)
            .with_output_rows(size);

        Ok(BudgetedExecutionOutcome::new(output, report))
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

    /// Create an unnamed `Series` from a host slice inside this runtime.
    pub fn series<T: HostColumnData>(&self, data: &[T]) -> Result<Series> {
        Series::from_column(T::create_column_in_runtime(self, data)?)
    }

    /// Create an unnamed `Series` from a host slice using borrowed execution overrides.
    pub fn series_in<T: HostColumnData>(&self, data: &[T], ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        let operation = "runtime::series_in";
        let domain = ctx.resolve_output_domain(&self.domain).map_err(|error| error.reanchor(operation))?;
        let runtime = Self { domain };
        runtime.series(data).map_err(|error| error.reanchor(operation))
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

    /// Create an unnamed all-null `Series` under a hard execution-budget token.
    ///
    /// The all-null column construction and output allocation are routed
    /// through the token's stream-bound reservation. The returned report
    /// records the resulting `Series` footprint and row count.
    pub fn nulls_with_budget(
        &self,
        dtype: impl Into<DataType>,
        len: usize,
        token: &ExecutionBudgetToken,
    ) -> Result<BudgetedExecutionOutcome<Series>> {
        let operation = "runtime::nulls_with_budget";
        let dtype = dtype.into();
        let unresolved_contract = if dtype.id().is_decimal() && dtype.scale() != 0 {
            Some(RUNTIME_DECIMAL_SCALED_NULLS_BUDGETED_CAST_CONTRACT_GAP)
        } else {
            None
        };
        let budgeted_runtime = budgeted_runtime_for_operation(self, token, operation)?;
        let series = budgeted_runtime
            .nulls(dtype, len)
            .map_err(|error| annotate_runtime_budget_execution_error(error, token, operation))?;
        let report = budgeted_series_report(&series, token, len, operation, unresolved_contract)?;
        Ok(BudgetedExecutionOutcome::new(series, report))
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
    /// different CUDA device, cudf-rs materializes the scalar value through
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

    /// Create an unnamed `Series` by repeating a scalar under a hard execution-budget token.
    ///
    /// This is the budgeted counterpart to
    /// [`series_from_scalar`](Self::series_from_scalar). The underlying
    /// scalar-to-column construction is executed through the token's
    /// stream-bound reservation, and the returned report records the observed
    /// output footprint and row count.
    pub fn series_from_scalar_with_budget(
        &self,
        scalar: &Scalar,
        size: usize,
        token: &ExecutionBudgetToken,
    ) -> Result<BudgetedExecutionOutcome<Series>> {
        let operation = "runtime::series_from_scalar_with_budget";
        let budgeted_runtime = budgeted_runtime_for_operation(self, token, operation)?;
        let column = budgeted_runtime
            .column_from_scalar(scalar, size)
            .map_err(|error| annotate_runtime_budget_execution_error(error, token, operation))?;
        let series = Series::from_column(column).map_err(|error| error.reanchor(operation))?;
        let report = budgeted_series_report(&series, token, size, operation, None)?;
        Ok(BudgetedExecutionOutcome::new(series, report))
    }

    /// Create an unnamed fixed-width `Series` by repeating a Rust value under
    /// a hard execution-budget token.
    ///
    /// Both the temporary device scalar and the output column are created in a
    /// reservation-routed execution domain. The report records the retained/output
    /// footprint of the resulting `Series`; it also marks the current scalar
    /// construction/broadcast preflight gap because libcudf does not expose an
    /// exact temporary workspace bound for this fused path.
    pub fn series_from_value_with_budget<T: CudfType>(
        &self,
        value: T,
        len: usize,
        token: &ExecutionBudgetToken,
    ) -> Result<BudgetedExecutionOutcome<Series>> {
        let operation = "runtime::series_from_value_with_budget";
        let budgeted_runtime = budgeted_runtime_for_operation(self, token, operation)?;
        let scalar = budgeted_runtime
            .scalar(value)
            .map_err(|error| annotate_runtime_budget_execution_error(error, token, operation))?;
        let series = budgeted_runtime
            .series_from_scalar(&scalar, len)
            .map_err(|error| annotate_runtime_budget_execution_error(error, token, operation))?;
        let report = budgeted_series_report(
            &series,
            token,
            len,
            operation,
            Some(RUNTIME_SCALAR_TO_SERIES_TEMPORARY_WORKSPACE_CONTRACT_GAP),
        )?;
        Ok(BudgetedExecutionOutcome::new(series, report))
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
    pub(crate) fn __debug_execution_domain_id(&self) -> u64 {
        self.domain.debug_id()
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

    /// Override the runtime's owned default device memory resource.
    ///
    /// **Outputs-only semantics** (matching upstream libcudf's `mr`
    /// parameter): the supplied handle receives the *output* allocations of
    /// operations that execute in this runtime. Temporary allocations made
    /// inside libcudf kernels always resolve through the device's long-lived
    /// memory domain (the per-device current resource installed at domain
    /// creation) and are accounted per stream — they are never routed into
    /// this resource. Use this when output buffers must live on a specific
    /// allocator (for example, interop buffers handed to an external system).
    ///
    /// For memory budgeting, do not reach for an owned resource: attach a
    /// [`StreamReservationLease`](crate::StreamReservationLease) to the
    /// runtime's stream instead (see [`Runtime::stream_reservation`]); it
    /// accounts outputs *and* temporaries for the pipeline.
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
    /// **Outputs-only semantics**: like
    /// [`with_memory_resource`](Self::with_memory_resource), the pool receives
    /// only output allocations; libcudf temporaries route through the
    /// device's memory domain and are accounted per stream. An owned pool is
    /// therefore not a budgeting tool — use the default device-level memory
    /// manager plus stream-bound reservations
    /// ([`Runtime::stream_reservation`]) to bound a pipeline's memory.
    ///
    /// - `initial_pool_size`: initial pool allocation in bytes.
    /// - `maximum_pool_size`: maximum pool size in bytes, or `None` for unbounded.
    pub fn with_pool_memory_resource(mut self, initial_pool_size: usize, maximum_pool_size: Option<usize>) -> Self {
        self.mr_selection = MemoryResourceSelection::Pool { initial_pool_size, maximum_pool_size };
        self
    }

    /// Use the device-level reservation-aware memory manager as this runtime's default allocator.
    ///
    /// This is the query-backend path: the runtime keeps one explicit stream,
    /// while allocator accounting is controlled by stream-bound reservation
    /// leases rather than replacing the memory resource for each execution.
    pub fn with_device_memory_manager(mut self) -> Self {
        self.mr_selection = MemoryResourceSelection::DeviceMemoryManager;
        self
    }

    /// Attach an opt-in boundary diagnostics observer to this runtime.
    ///
    /// The observer is inherited by objects created through the runtime and by
    /// same-device execution domains derived from borrowed stream or
    /// memory-resource overrides. It receives successful host/device and
    /// interop boundary observations synchronously.
    pub fn with_boundary_observer(mut self, observer: impl BoundaryObserver) -> Self {
        self.boundary_observer = Some(Arc::new(observer));
        self
    }

    /// Build a runtime with one explicit retained provenance bundle.
    ///
    /// If [`with_device`](Self::with_device) was called, resources are created
    /// on that device.  Otherwise the device is inferred from supplied handles,
    /// or defaults to the calling thread's current device.
    ///
    /// The resulting runtime owns exactly one default stream and uses the
    /// device-level reservation-aware memory manager as its default memory
    /// resource. Borrowed per-operation overrides can temporarily replace
    /// either handle for a supported call, but they do not mutate this owned
    /// bundle.
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
        let RuntimeBuilder { stream, mr_selection, boundary_observer, .. } = self;

        // Temporarily switch to the target device so that any auto-created
        // stream or memory resource lands on the correct GPU.
        let _guard = DeviceGuard::new(device)?;

        let stream = match stream {
            Some(stream) => stream,
            None => CudaStream::new()?,
        };
        let mr = match mr_selection {
            MemoryResourceSelection::Owned(mr) => mr,
            MemoryResourceSelection::Pool { initial_pool_size, maximum_pool_size } => {
                DeviceMemoryResource::pool_on(device, initial_pool_size, maximum_pool_size)?
            },
            MemoryResourceSelection::DeviceMemoryManager => DeviceMemoryManager::for_device(device)?.resource().clone(),
        };
        let domain = ExecutionDomain::from_handles_on_device(device, stream, mr, boundary_observer)?;
        Ok(Runtime { domain })
    }

    /// Resolve the target device, validating handle provenance.
    fn resolve_device(&self) -> Result<DeviceId> {
        if let Some(target) = self.device {
            // Explicit device — validate supplied handles agree.
            if let Some(ref stream) = self.stream {
                if stream.device() != target {
                    return Err(Self::runtime_binding_mismatch(
                        "stream",
                        target,
                        stream.device(),
                        "cross-device stream: stream was created on a different device than the runtime target",
                    ));
                }
            }
            if let MemoryResourceSelection::Owned(ref mr) = self.mr_selection {
                if mr.device() != target {
                    return Err(Self::runtime_binding_mismatch(
                        "memory_resource",
                        target,
                        mr.device(),
                        "cross-device memory resource: MR was created on a different device than the runtime target",
                    ));
                }
            }
            Ok(target)
        } else {
            // No explicit device — infer from handles.
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
                    DeviceId::current()
                },
            }
        }
    }
}
