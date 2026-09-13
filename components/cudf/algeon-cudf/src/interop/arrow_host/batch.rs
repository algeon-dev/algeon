//! RecordBatch to DataFrame import entry points and DataFrame export.

use std::sync::Arc;

use arrow::{
    array::{Array, ArrayData},
    compute::concat,
    datatypes::{DataType as ArrowDataType, Schema, TimeUnit},
    record_batch::{RecordBatch, RecordBatchOptions},
};

use super::{
    export::{column_view_to_arrow_with_requested_type, map_record_batch_error},
    import::arrow_array_to_column,
    record_host_boundary_observation_in_domain, record_host_export_call, record_host_import_call,
    schema::{ArrowHostField, ArrowHostSchema},
};
use crate::{
    core::{
        DeviceBuffer,
        column::Column,
        error::{self, Error, ErrorCode, Result},
        types::TypeId,
    },
    frame::dataframe::DataFrame,
    hash::fast_index_map,
    internal::resolve::missing_column_error,
    runtime::{
        BoundaryKind, CudaEvent, QueryStageKind, Runtime,
        device_guard::DeviceGuard,
        host_staging::{PinnedSlabLease, pinned_staging_pool},
        stream::CudaStream,
    },
};

impl DataFrame {
    /// Convert this GPU DataFrame to an Arrow [`RecordBatch`] via host copy.
    ///
    /// Each column is copied from GPU to host memory, then wrapped in the
    /// corresponding Arrow array type. The schema (column names and types)
    /// is preserved.
    ///
    /// If you need device-pointer handoff instead of host materialization, use
    /// [`crate::DataFrame::try_prepare_device_arrow_export`] from
    /// [`crate::interop::arrow_device`].
    ///
    /// # Supported types
    ///
    /// - Numeric: `Int8`, `Int16`, `Int32`, `Int64`, `UInt8`, `UInt16`,
    ///   `UInt32`, `UInt64`, `Float32`, `Float64`
    /// - `Bool8`
    /// - `Decimal32`, `Decimal64`, and `Decimal128` as Arrow
    ///   `Decimal128(38, scale)`. Decimal32 and Decimal64 raw values are
    ///   widened to Arrow's i128 decimal storage during the host copy; Decimal128
    ///   values are copied directly into Arrow Decimal128 storage. cuDF stores
    ///   decimal scale but not precision, so this host path uses precision 38.
    /// - `TimestampDays` as Arrow `Date32`
    /// - `TimestampSeconds`, `TimestampMilliseconds`, `TimestampMicroseconds`,
    ///   and `TimestampNanoseconds` as Arrow timestamps without time zone.
    /// - `DurationSeconds`, `DurationMilliseconds`, `DurationMicroseconds`,
    ///   and `DurationNanoseconds` as Arrow durations.
    /// - `String` (converted to Arrow `Utf8`)
    /// - `List` with `String` or numeric children (converted to Arrow `List`)
    ///
    /// # Errors
    ///
    /// Returns an error if any column has an unsupported type, or if the
    /// underlying GPU-to-host copy fails.
    pub fn to_record_batch(&self) -> Result<RecordBatch> {
        self.to_record_batch_with_optional_schema(None)
    }

    pub(crate) fn to_record_batch_with_arrow_schema(&self, schema: &ArrowHostSchema) -> Result<RecordBatch> {
        self.to_record_batch_with_optional_schema(Some(schema))
    }

    fn to_record_batch_with_optional_schema(&self, schema: Option<&ArrowHostSchema>) -> Result<RecordBatch> {
        record_host_export_call();
        let names = self.column_names();
        let rows = self.num_rows();
        let column_count = self.num_columns();
        let placement = self.placement();

        if schema.is_some_and(|schema| schema.len() != column_count) {
            return Err(Error::new_with(
                ErrorCode::InvariantViolation,
                "dataframe::to_record_batch",
                "validated Arrow schema and DataFrame column counts diverged before materialization",
            ));
        }

        let col_map = self.columns_map();
        let columns: Vec<(String, Arc<dyn Array>, bool)> = names
            .iter()
            .enumerate()
            .map(|(index, &name)| {
                let col = col_map.get(name).ok_or_else(|| missing_column_error("dataframe::to_record_batch", name))?;
                let cv = col.view()?;
                let dtype = cv.data_type()?;
                let nullable = cv.nullable()?;
                let requested_type = schema.and_then(|schema| schema.field(index)).map(ArrowHostField::arrow_data_type);
                let arr = column_view_to_arrow_with_requested_type(&cv, dtype, requested_type)?;
                Ok((name.to_string(), arr, nullable))
            })
            .collect::<Result<_>>()?;

        let batch = if columns.is_empty() {
            RecordBatch::try_new_with_options(
                Arc::new(Schema::empty()),
                Vec::new(),
                &RecordBatchOptions::new().with_row_count(Some(rows)),
            )
        } else {
            RecordBatch::try_from_iter_with_nullable(columns)
        }
        .map_err(map_record_batch_error)?;
        let observation = self
            .boundary_observation_for_query_stage(
                BoundaryKind::DeviceToHost,
                QueryStageKind::FinalMaterialize,
                true,
                "dataframe::to_record_batch",
            )
            .map_err(|error| error.reanchor("dataframe::to_record_batch"))?;
        debug_assert_eq!(observation.placement, placement);
        debug_assert_eq!(observation.rows, rows);
        debug_assert_eq!(observation.columns, column_count);
        record_host_boundary_observation_in_domain(self.execution_domain(), observation);
        Ok(batch)
    }

    /// Create a GPU `DataFrame` from an Arrow [`RecordBatch`] via host copy
    /// inside the caller-provided runtime.
    ///
    /// This is the inverse of [`to_record_batch()`](Self::to_record_batch).
    /// Each Arrow array is copied from host memory to the GPU. Column names
    /// and supported logical values are preserved, and all imported columns
    /// are seeded with the caller runtime's retained provenance. Decimal128
    /// imports preserve values and scale, but cuDF does not retain Arrow
    /// precision metadata; re-export normalizes Decimal128 fields to
    /// `Decimal128(38, scale)`.
    ///
    /// This method imports from host-resident `RecordBatch` data. If you
    /// already have Arrow C Device ABI roots on the same CUDA device, prefer
    /// [`crate::DataFrame::try_import_from_device_arrow_table_into_runtime`]
    /// from [`crate::interop::arrow_device`].
    ///
    /// # Supported types
    ///
    /// - Numeric: `Int8`, `Int16`, `Int32`, `Int64`, `UInt8`, `UInt16`,
    ///   `UInt32`, `UInt64`, `Float32`, `Float64`
    /// - `Boolean`
    /// - `Decimal128` (preserves values and scale; precision is re-exported
    ///   as 38)
    /// - `Date32`
    /// - `Timestamp(Second|Millisecond|Microsecond|Nanosecond, _)`
    /// - `Duration(Second|Millisecond|Microsecond|Nanosecond)`
    /// - `Utf8` / `LargeUtf8` / `Utf8View` (converted to cudf `String`)
    ///
    /// # Errors
    ///
    /// Returns an error if any column has an unsupported Arrow type, or if
    /// the host-to-GPU copy fails.
    pub fn from_record_batch(runtime: &Runtime, batch: &RecordBatch) -> Result<Self> {
        runtime.dataframe_from_record_batch(batch)
    }

    /// Create a GPU `DataFrame` from a sequence of Arrow [`RecordBatch`]es
    /// sharing one schema, staging the host copies through reusable pinned
    /// slabs.
    ///
    /// Prefer this over calling [`Self::from_record_batch`] per batch and
    /// concatenating on the device: per-batch imports issue one small
    /// pageable host-to-device copy and one stream synchronization per
    /// column per batch, which serializes process-wide under concurrent
    /// queries. See [`dataframe_from_record_batches`] for the staged-import
    /// contract, the fast-path type set, and the fallback behavior.
    ///
    /// # Errors
    ///
    /// Returns an error if `batches` is empty, batch schemas differ, the
    /// combined row count exceeds cudf's supported column length, or any
    /// column type is unsupported by the host import path.
    pub fn from_record_batches(runtime: &Runtime, batches: &[RecordBatch]) -> Result<Self> {
        runtime.dataframe_from_record_batches(batches)
    }
}

impl Runtime {
    /// Convert an Arrow [`RecordBatch`] to a GPU [`DataFrame`] via host copy
    /// inside this runtime.
    pub fn dataframe_from_record_batch(&self, batch: &RecordBatch) -> Result<DataFrame> {
        dataframe_from_record_batch(self, batch)
    }

    /// Convert same-schema Arrow [`RecordBatch`]es to one GPU [`DataFrame`]
    /// via pinned-staged host copy inside this runtime.
    ///
    /// See [`dataframe_from_record_batches`].
    pub fn dataframe_from_record_batches(&self, batches: &[RecordBatch]) -> Result<DataFrame> {
        dataframe_from_record_batches(self, batches)
    }
}

/// Convert an Arrow [`RecordBatch`] to a GPU [`DataFrame`] via host copy
/// inside the caller-provided runtime.
///
/// Prefer [`Runtime::dataframe_from_record_batch()`] or
/// [`DataFrame::from_record_batch()`] for a more ergonomic API.
///
/// Each Arrow array is copied from host memory to the GPU. Column names and
/// supported logical values are preserved, and imported columns inherit the
/// supplied runtime's retained provenance. Decimal128 imports preserve values
/// and scale, but cuDF does not retain Arrow precision metadata; re-export
/// normalizes Decimal128 fields to `Decimal128(38, scale)`.
///
/// This is the host `RecordBatch` import path, not Arrow C Device ABI import.
/// For same-device device-root handoff, see
/// [`crate::DataFrame::try_import_from_device_arrow_table_into_runtime`].
///
/// # Supported types
///
/// - Numeric: `Int8`, `Int16`, `Int32`, `Int64`, `UInt8`, `UInt16`,
///   `UInt32`, `UInt64`, `Float32`, `Float64`
/// - `Boolean`
/// - `Decimal128` (preserves values and scale; precision is re-exported as 38)
/// - `Date32`
/// - `Timestamp(Second|Millisecond|Microsecond|Nanosecond, _)`
/// - `Duration(Second|Millisecond|Microsecond|Nanosecond)`
/// - `Utf8` / `LargeUtf8` / `Utf8View` (converted to cudf `String`)
///
/// Unsupported Arrow types will return an error.
///
/// # Errors
///
/// Returns an error if any column has an unsupported Arrow type, or if
/// the host-to-GPU copy fails.
pub fn dataframe_from_record_batch(runtime: &Runtime, batch: &RecordBatch) -> Result<DataFrame> {
    record_host_import_call();
    let schema = batch.schema();
    let num_cols = schema.fields().len();

    if num_cols == 0 {
        let frame =
            DataFrame::from_raw_parts(fast_index_map(), batch.num_rows(), Arc::clone(runtime.execution_domain()));
        let observation = frame.boundary_observation_for_query_stage(
            BoundaryKind::HostToDevice,
            QueryStageKind::InteropImport,
            false,
            "dataframe_from_record_batch",
        )?;
        record_host_boundary_observation_in_domain(runtime.execution_domain(), observation);
        return Ok(frame);
    }

    let mut names: Vec<String> = Vec::with_capacity(num_cols);
    let mut columns: Vec<Column> = Vec::with_capacity(num_cols);

    for (i, field) in schema.fields().iter().enumerate() {
        names.push(field.name().clone());
        let arr = batch.column(i);
        let col = arrow_array_to_column(runtime, arr.as_ref(), field.name())?;
        columns.push(col);
    }

    let frame = DataFrame::new(names, columns)?;
    let observation = frame.boundary_observation_for_query_stage(
        BoundaryKind::HostToDevice,
        QueryStageKind::InteropImport,
        false,
        "dataframe_from_record_batch",
    )?;
    record_host_boundary_observation_in_domain(runtime.execution_domain(), observation);
    Ok(frame)
}

/// Convert a sequence of Arrow [`RecordBatch`]es sharing one schema into a
/// single GPU [`DataFrame`] via batched host copy.
///
/// This is the many-batch counterpart of [`dataframe_from_record_batch`],
/// built for streamed producers (one batch per upstream morsel). Importing
/// batch-by-batch issues one device allocation, one small host-to-device
/// copy, and one stream synchronization **per column per batch**; under
/// concurrent queries those small pageable copies serialize process-wide in
/// the CUDA driver. This entry point instead:
///
/// - allocates **one** device buffer per column sized for all batches,
/// - stages source bytes through reusable pinned slabs
///   (`crate::runtime::host_staging`) so the host-to-device copies are true
///   asynchronous DMA, and
/// - waits only on the staging copies' completion events instead of full
///   stream synchronizations per batch.
///
/// Columns whose type or nullability is outside the staged fast path
/// (non-primitive layouts such as strings and booleans, decimals, or any
/// column with nulls) are concatenated on the host with Arrow's `concat`
/// kernel and imported through the same per-column path as
/// [`dataframe_from_record_batch`]; one import per column instead of one
/// per column per batch.
///
/// All batches must share `batches[0]`'s schema. The combined row count must
/// fit cudf's row-count type (`i32`).
///
/// # Errors
///
/// Returns an error if `batches` is empty, schemas differ, the combined row
/// count overflows, a column type is unsupported by the host import path, or
/// a copy fails.
pub fn dataframe_from_record_batches(runtime: &Runtime, batches: &[RecordBatch]) -> Result<DataFrame> {
    if batches.len() != 1 {
        record_host_import_call();
    }
    let Some(first_batch) = batches.first() else {
        return Err(Error::new_with(
            ErrorCode::EmptyInput,
            "dataframe_from_record_batches",
            "at least one record batch is required; build an empty batch from the schema for empty imports",
        ));
    };
    if batches.len() == 1 {
        return dataframe_from_record_batch(runtime, first_batch);
    }
    let schema = first_batch.schema();
    for (batch_index, batch) in batches.iter().enumerate().skip(1) {
        if batch.schema() != schema {
            return Err(Error::new_with(
                ErrorCode::TypeMismatch,
                "dataframe_from_record_batches",
                format!("record batch {batch_index} does not share the first batch's schema"),
            ));
        }
    }

    let total_rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
    if i32::try_from(total_rows).is_err() {
        return Err(Error::new_with(
            ErrorCode::ValueOutOfRange,
            "dataframe_from_record_batches",
            format!("combined row count {total_rows} exceeds cudf's supported column length"),
        ));
    }

    let num_cols = schema.fields().len();
    if num_cols == 0 {
        let frame = DataFrame::from_raw_parts(fast_index_map(), total_rows, Arc::clone(runtime.execution_domain()));
        let observation = frame.boundary_observation_for_query_stage(
            BoundaryKind::HostToDevice,
            QueryStageKind::InteropImport,
            false,
            "dataframe_from_record_batches",
        )?;
        record_host_boundary_observation_in_domain(runtime.execution_domain(), observation);
        return Ok(frame);
    }

    let mut names: Vec<String> = Vec::with_capacity(num_cols);
    let mut columns: Vec<Column> = Vec::with_capacity(num_cols);
    for (i, field) in schema.fields().iter().enumerate() {
        names.push(field.name().clone());
        let arrays: Vec<&dyn Array> = batches.iter().map(|batch| batch.column(i).as_ref()).collect();
        let all_non_null = arrays.iter().all(|array| array.null_count() == 0);
        let column = match staged_import_type_id(field.data_type()) {
            Some(type_id) if all_non_null => {
                let elem_size = field.data_type().primitive_width().ok_or_else(|| {
                    Error::new_with(
                        ErrorCode::UnsupportedDtype,
                        "dataframe_from_record_batches",
                        format!("staged import type {:?} lacks a fixed primitive width", field.data_type()),
                    )
                })?;
                let datas: Vec<ArrayData> = arrays.iter().map(|array| array.to_data()).collect();
                let slices: Vec<&[u8]> = datas
                    .iter()
                    .map(|data| {
                        let start = data.offset() * elem_size;
                        let len = data.len() * elem_size;
                        &data.buffers()[0].as_slice()[start..start + len]
                    })
                    .collect();
                staged_fixed_width_column(runtime, type_id, total_rows, &slices)
                    .map_err(|error| error.reanchor("arrow::dataframe_from_record_batches"))?
            },
            _ => {
                let merged = concat(&arrays).map_err(|error| {
                    error::foreign_invariant_violation(
                        "dataframe_from_record_batches",
                        error.to_string(),
                        "record_batch",
                        "arrow_concat_failed",
                        "ensure batch columns share one Arrow data type and valid layouts",
                    )
                    .with_source(error)
                })?;
                arrow_array_to_column(runtime, merged.as_ref(), field.name())?
            },
        };
        columns.push(column);
    }

    let frame = DataFrame::new(names, columns)?;
    let observation = frame.boundary_observation_for_query_stage(
        BoundaryKind::HostToDevice,
        QueryStageKind::InteropImport,
        false,
        "dataframe_from_record_batches",
    )?;
    record_host_boundary_observation_in_domain(runtime.execution_domain(), observation);
    Ok(frame)
}

/// cudf type id for Arrow types eligible for the staged fixed-width import
/// fast path.
///
/// The fast path requires the Arrow values buffer to be a plain fixed-width
/// array whose bytes can be adopted verbatim as the cudf column data:
/// bit-packed (`Boolean`), offset-based (strings, lists), and
/// scale-carrying (`Decimal128`) types are excluded and take the concat
/// fallback.
fn staged_import_type_id(dt: &ArrowDataType) -> Option<TypeId> {
    match dt {
        ArrowDataType::Int8 => Some(TypeId::Int8),
        ArrowDataType::Int16 => Some(TypeId::Int16),
        ArrowDataType::Int32 => Some(TypeId::Int32),
        ArrowDataType::Int64 => Some(TypeId::Int64),
        ArrowDataType::UInt8 => Some(TypeId::UInt8),
        ArrowDataType::UInt16 => Some(TypeId::UInt16),
        ArrowDataType::UInt32 => Some(TypeId::UInt32),
        ArrowDataType::UInt64 => Some(TypeId::UInt64),
        ArrowDataType::Float32 => Some(TypeId::Float32),
        ArrowDataType::Float64 => Some(TypeId::Float64),
        ArrowDataType::Date32 => Some(TypeId::TimestampDays),
        ArrowDataType::Date64 => Some(TypeId::TimestampMilliseconds),
        ArrowDataType::Time64(TimeUnit::Microsecond) => Some(TypeId::DurationMicroseconds),
        ArrowDataType::Time64(TimeUnit::Nanosecond) => Some(TypeId::DurationNanoseconds),
        ArrowDataType::Timestamp(TimeUnit::Second, _) => Some(TypeId::TimestampSeconds),
        ArrowDataType::Timestamp(TimeUnit::Millisecond, _) => Some(TypeId::TimestampMilliseconds),
        ArrowDataType::Timestamp(TimeUnit::Microsecond, _) => Some(TypeId::TimestampMicroseconds),
        ArrowDataType::Timestamp(TimeUnit::Nanosecond, _) => Some(TypeId::TimestampNanoseconds),
        ArrowDataType::Duration(TimeUnit::Second) => Some(TypeId::DurationSeconds),
        ArrowDataType::Duration(TimeUnit::Millisecond) => Some(TypeId::DurationMilliseconds),
        ArrowDataType::Duration(TimeUnit::Microsecond) => Some(TypeId::DurationMicroseconds),
        ArrowDataType::Duration(TimeUnit::Nanosecond) => Some(TypeId::DurationNanoseconds),
        _ => None,
    }
}

/// Build one non-nullable fixed-width column by staging `slices` through
/// pinned slabs into a single device buffer.
///
/// Uses two slabs in rotation: while one slab's DMA is in flight, the other
/// is filled on the CPU. Each slab is guarded by a CUDA event recorded after
/// its copy; the event is awaited before the slab is refilled and before the
/// leases return the slabs to the pool.
fn staged_fixed_width_column(
    runtime: &Runtime,
    type_id: TypeId,
    total_rows: usize,
    slices: &[&[u8]],
) -> Result<Column> {
    let total_bytes: usize = slices.iter().map(|slice| slice.len()).sum();
    let device_buffer = DeviceBuffer::new_in_runtime(runtime, total_bytes)?;
    if total_bytes > 0 {
        let _device_guard =
            DeviceGuard::new(runtime.device()).map_err(|error| error.reanchor("arrow::staged_fixed_width_column"))?;
        let pool = pinned_staging_pool();
        let mut leases = [pool.acquire()?, pool.acquire()?];
        let events = [CudaEvent::new()?, CudaEvent::new()?];
        let mut event_pending = [false, false];
        let slab_capacity = pool.slab_bytes();
        let stream = runtime.stream();

        let mut current = 0usize;
        let mut fill = 0usize;
        let mut dst_offset = 0usize;
        for slice in slices {
            let mut consumed = 0usize;
            while consumed < slice.len() {
                if fill == slab_capacity {
                    flush_staging_slab(
                        &device_buffer,
                        &mut leases[current],
                        fill,
                        &mut dst_offset,
                        &events[current],
                        stream,
                    )?;
                    event_pending[current] = true;
                    current ^= 1;
                    if event_pending[current] {
                        events[current].synchronize()?;
                        event_pending[current] = false;
                    }
                    fill = 0;
                }
                let take = (slab_capacity - fill).min(slice.len() - consumed);
                leases[current].slab().write_at(fill, &slice[consumed..consumed + take]);
                fill += take;
                consumed += take;
            }
        }
        if fill > 0 {
            flush_staging_slab(&device_buffer, &mut leases[current], fill, &mut dst_offset, &events[current], stream)?;
            event_pending[current] = true;
        }
        debug_assert_eq!(dst_offset, total_bytes);
        for (index, pending) in event_pending.iter().enumerate() {
            if *pending {
                events[index].synchronize()?;
            }
        }
    }
    runtime.column_from_device_buffer(device_buffer, total_rows, type_id)
}

fn flush_staging_slab(
    device_buffer: &DeviceBuffer,
    lease: &mut PinnedSlabLease,
    fill: usize,
    dst_offset: &mut usize,
    event: &CudaEvent,
    stream: &CudaStream,
) -> Result<()> {
    let slab = lease.slab();
    // SAFETY: the slab holds at least `fill` initialized bytes written via
    // `write_at`, and the lease keeps the region alive until the recorded
    // event is awaited.
    let staged = unsafe { std::slice::from_raw_parts(slab.as_ptr(), fill) };
    device_buffer.copy_from_host_range_async(*dst_offset, staged)?;
    event.record(stream)?;
    *dst_offset += fill;
    Ok(())
}
