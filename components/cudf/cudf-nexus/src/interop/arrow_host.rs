//! Arrow `RecordBatch` interop: GPU `DataFrame` ↔ Arrow `RecordBatch`
//! conversion.
//!
//! This module is specifically the Rust `arrow` crate / `RecordBatch` path.
//! It provides a **host-copy** conversion: data is transferred GPU → CPU →
//! Arrow (or Arrow → CPU → GPU).
//!
//! This is deliberately distinct from [`crate::interop::arrow_device`], which
//! implements Arrow C Device ABI handoff for existing device-backed buffers.
//! That device-interop path can preserve device pointers for many supported
//! layouts, but it is a different contract surface from `RecordBatch` and is
//! not the API implemented here.
//!
//! # Example
//!
//! ```no_run
//! use cudf::{DataFrame, Runtime};
//!
//! let rt = Runtime::new().unwrap();
//! let ids = rt.series(&[1i32, 2, 3]).unwrap();
//! let scores = rt.series(&[10.0f64, 20.0, 30.0]).unwrap();
//! let df = DataFrame::from_series([("id", ids), ("score", scores)]).unwrap();
//!
//! // GPU → Arrow RecordBatch (host-copy)
//! let batch = df.to_record_batch().unwrap();
//! assert_eq!(batch.num_rows(), 3);
//!
//! // Arrow RecordBatch → GPU DataFrame (host-copy) inside the explicit runtime
//! let df2 = DataFrame::from_record_batch(&rt, &batch).unwrap();
//! assert_eq!(df2.num_columns(), 2);
//! ```

use std::{
    cell::Cell,
    ffi::c_void,
    ptr,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use arrow::{
    array::*,
    buffer::{NullBuffer, OffsetBuffer, ScalarBuffer},
    datatypes::{DataType as ArrowDataType, Field, FieldRef, Schema, TimeUnit},
    ffi::{FFI_ArrowArray, FFI_ArrowSchema},
    record_batch::{RecordBatch, RecordBatchOptions},
};
use rapids_interop::{
    BoundBytes, BufferBinding, BufferId, ColumnId, ColumnLayout, DataType as InteropDataType, PhysicalValidity,
    Placement, PreparedColumnExport, ProducerReadiness, SchemaNullability, SemanticColumn, TypeId as InteropTypeId,
};

use crate::{
    core::{
        column::{Column, ColumnView},
        error::{self, Error, ErrorCode, FACT_TYPE_ID, Result},
        types::{DataType, TypeId, cudf_null_mask_allocation_size_bytes},
    },
    frame::{dataframe::DataFrame, series::Series},
    internal::{
        hash::fast_index_map,
        resolve::{missing_column_error, validate_no_duplicate_names},
    },
    runtime::{BoundaryKind, BoundaryObservation, QueryStageKind, Runtime, record_boundary_observation_in_domain},
};

pub(crate) mod testing_hooks;

thread_local! {
    static HOST_ARROW_LAST_BOUNDARY_OBSERVATION: Cell<Option<BoundaryObservation>> = const { Cell::new(None) };
    static HOST_ARROW_EXPORT_CALL_BASELINE: Cell<usize> = const { Cell::new(0) };
    static HOST_ARROW_IMPORT_CALL_BASELINE: Cell<usize> = const { Cell::new(0) };
}

static HOST_ARROW_EXPORT_CALL_COUNT: AtomicUsize = AtomicUsize::new(0);
static HOST_ARROW_IMPORT_CALL_COUNT: AtomicUsize = AtomicUsize::new(0);

const DEFAULT_DECIMAL128_ARROW_PRECISION: u8 = 38;

/// Copy-free planner view of one Arrow host field as cudf would import it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrowHostField {
    name: String,
    arrow_dtype: ArrowDataType,
    cudf_dtype: DataType,
    nullable: bool,
}

impl ArrowHostField {
    fn new(name: String, arrow_dtype: ArrowDataType, cudf_dtype: DataType, nullable: bool) -> Self {
        Self { name, arrow_dtype, cudf_dtype, nullable }
    }

    /// Field name in Arrow schema order.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Original Arrow logical type.
    pub fn arrow_data_type(&self) -> &ArrowDataType {
        &self.arrow_dtype
    }

    /// cudf logical type that the host Arrow import path would create.
    pub fn cudf_data_type(&self) -> DataType {
        self.cudf_dtype
    }

    /// Arrow field nullability. cudf physical dtypes do not encode this bit.
    pub fn nullable(&self) -> bool {
        self.nullable
    }
}

/// Copy-free planner view of an Arrow host schema as cudf would import it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrowHostSchema {
    fields: Vec<ArrowHostField>,
    row_count: usize,
}

impl ArrowHostSchema {
    fn new(fields: Vec<ArrowHostField>, row_count: usize) -> Self {
        Self { fields, row_count }
    }

    pub(crate) fn with_row_count(&self, row_count: usize) -> Self {
        Self { fields: self.fields.clone(), row_count }
    }

    /// Validated fields in schema order.
    pub fn fields(&self) -> &[ArrowHostField] {
        &self.fields
    }

    /// Return one validated field by ordinal.
    pub fn field(&self, index: usize) -> Option<&ArrowHostField> {
        self.fields.get(index)
    }

    /// Return one validated field by name.
    pub fn field_by_name(&self, name: &str) -> Option<&ArrowHostField> {
        self.fields.iter().find(|field| field.name() == name)
    }

    /// Number of schema fields.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Whether the schema has no fields.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Logical row count supplied by the caller or observed from a RecordBatch.
    pub fn row_count(&self) -> usize {
        self.row_count
    }
}

/// Convert a supported Arrow host logical type into a cudf logical type.
///
/// This is a copy-free planner/schema helper. It follows the same support
/// envelope as [`DataFrame::from_record_batch`]: numeric types, Boolean,
/// Decimal128, Date32, timestamps, durations with Arrow time units, Utf8,
/// LargeUtf8, Utf8View, top-level `List` / `LargeList` with string-like or
/// numeric children, and positive-width `FixedSizeList` with numeric children.
/// Timestamp timezone metadata is accepted as an Arrow schema contract; cuDF
/// stores the physical epoch values and resolution.
pub fn cudf_dtype_from_arrow_type(dt: &ArrowDataType) -> Result<DataType> {
    cudf_dtype_from_arrow_type_in(dt, "arrow_host::cudf_dtype_from_arrow_type")
}

/// Convert a supported cudf logical type into the Arrow host type used by
/// [`DataFrame::to_record_batch`].
///
/// Decimal32, Decimal64, and Decimal128 export as Arrow Decimal128 with
/// precision 38 because libcudf retains decimal scale but not Arrow precision
/// metadata.
pub fn arrow_type_from_cudf_dtype(dtype: DataType) -> Result<ArrowDataType> {
    arrow_type_from_cudf_dtype_in(dtype, "arrow_host::arrow_type_from_cudf_dtype")
}

/// Return whether this Arrow host type is supported by cudf host import.
pub fn is_supported_arrow_type(dt: &ArrowDataType) -> bool {
    cudf_dtype_from_arrow_type(dt).is_ok()
}

/// Return whether this Arrow host type is a supported numeric cudf type.
pub fn is_supported_arrow_numeric_type(dt: &ArrowDataType) -> bool {
    cudf_dtype_from_arrow_type(dt).map(|dtype| dtype.id().is_numeric()).unwrap_or(false)
}

/// Return whether a cudf logical dtype can materialize into the requested
/// Arrow host type without changing the column's logical values.
///
/// This follows the export side of the host-copy interop contract. It is not
/// the same as Arrow import mapping: Decimal32, Decimal64, and Decimal128 all
/// export as Arrow Decimal128 with the same scale because libcudf retains
/// decimal scale but not Arrow precision metadata.
///
/// Nested cudf dtypes do not retain their child schema in [`DataType`], so
/// this dtype-only predicate deliberately returns `false` for Arrow list
/// types. Owners with a concrete column view, such as `GpuRecordBatch`, must
/// validate the actual child dtype and any fixed-size shape contract.
pub fn cudf_dtype_can_export_to_arrow_type(cudf_dtype: DataType, arrow_type: &ArrowDataType) -> bool {
    match arrow_type {
        ArrowDataType::Decimal128(_, scale) if cudf_dtype.id().is_decimal() => {
            cudf_dtype.scale() == arrow_decimal_scale_to_cudf_scale(*scale)
        },
        ArrowDataType::List(_) | ArrowDataType::LargeList(_) | ArrowDataType::FixedSizeList(_, _)
            if cudf_dtype.id() == TypeId::List =>
        {
            false
        },
        _ => cudf_dtype_from_arrow_type(arrow_type)
            .map(|arrow_cudf_dtype| arrow_cudf_dtype == cudf_dtype)
            .unwrap_or(false),
    }
}

/// Validate an Arrow schema without copying data and return cudf planner metadata.
///
/// Nullability is retained as schema metadata rather than folded into the cudf
/// physical dtype.
pub fn validate_arrow_schema(schema: &Schema, row_count: usize) -> Result<ArrowHostSchema> {
    let operation = "arrow_host::validate_arrow_schema";
    let names = schema.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();
    validate_no_duplicate_names(&names, operation)?;

    let mut fields = Vec::with_capacity(schema.fields().len());
    for field in schema.fields() {
        let cudf_dtype = cudf_dtype_from_arrow_type_in(field.data_type(), operation)
            .map_err(|error| error.with_detail("field_name", field.name().clone()))?;
        fields.push(ArrowHostField::new(
            field.name().clone(),
            field.data_type().clone(),
            cudf_dtype,
            field.is_nullable(),
        ));
    }

    Ok(ArrowHostSchema::new(fields, row_count))
}

/// Validate a RecordBatch schema without copying array data and return cudf planner metadata.
pub fn validate_record_batch_schema(batch: &RecordBatch) -> Result<ArrowHostSchema> {
    validate_arrow_schema(batch.schema().as_ref(), batch.num_rows())
        .map_err(|error| error.reanchor("arrow_host::validate_record_batch_schema"))
}

/// Constructs a typed all-null `List` or `LargeList` series through the Arrow host import contract.
pub fn all_null_list_series(runtime: &Runtime, data_type: &ArrowDataType, rows: usize) -> Result<Series> {
    match data_type {
        ArrowDataType::List(field) | ArrowDataType::LargeList(field) if list_child_is_supported(field) => {},
        _ => {
            return Err(Error::new_with(
                ErrorCode::UnsupportedDtype,
                "arrow_host::all_null_list_series",
                "typed null list construction requires a supported Arrow List or LargeList type",
            )
            .with_detail("arrow_type", format!("{data_type:?}"))
            .with_detail("reason", "unsupported_null_list_type"));
        },
    }
    let array = new_null_array(data_type, rows);
    let column = arrow_array_to_column(runtime, array.as_ref(), "all_null_list")
        .map_err(|error| error.reanchor("arrow_host::all_null_list_series"))?;
    Series::from_column(column).map_err(|error| error.reanchor("arrow_host::all_null_list_series"))
}

/// Returns the device bytes retained when Arrow list arrays are concatenated
/// into one cudf list column.
pub fn estimate_list_import_device_memory_bytes(arrays: &[ArrayRef]) -> Result<usize> {
    let operation = "arrow_host::estimate_list_import_device_memory_bytes";
    let Some(first) = arrays.first() else {
        return Err(Error::new_with(ErrorCode::EmptyInput, operation, "at least one Arrow list array is required"));
    };
    let data_type = first.data_type();
    let child_type = match data_type {
        ArrowDataType::List(field) | ArrowDataType::LargeList(field) if list_child_is_supported(field) => {
            field.data_type()
        },
        _ => {
            return Err(Error::new_with(
                ErrorCode::UnsupportedDtype,
                operation,
                "device import sizing requires supported Arrow List or LargeList arrays",
            )
            .with_detail("arrow_type", format!("{data_type:?}")));
        },
    };

    let mut root_rows = 0usize;
    let mut root_has_nulls = false;
    let mut child_rows = 0usize;
    let mut child_has_nulls = false;
    let mut string_chars = 0usize;
    for array in arrays {
        if array.data_type() != data_type {
            return Err(Error::new_with(
                ErrorCode::TypeMismatch,
                operation,
                "all Arrow list arrays must have the same logical type",
            )
            .with_detail("expected_type", format!("{data_type:?}"))
            .with_detail("actual_type", format!("{:?}", array.data_type())));
        }
        root_rows = checked_size_add(root_rows, array.len(), operation, "summing list root rows")?;
        root_has_nulls |= array.null_count() > 0;
        let (values, start, end) = active_list_child_range(array.as_ref(), operation)?;
        child_rows = checked_size_add(child_rows, end - start, operation, "summing active list child rows")?;
        child_has_nulls |= (start..end).any(|index| values.is_null(index));
        string_chars = checked_size_add(
            string_chars,
            active_string_bytes(values.as_ref(), start, end, child_type, operation)?,
            operation,
            "summing active list string bytes",
        )?;
    }
    if i32::try_from(root_rows).is_err() || i32::try_from(child_rows).is_err() {
        return Err(Error::new_with(
            ErrorCode::ValueOutOfRange,
            operation,
            "Arrow list import exceeds cudf's signed 32-bit row or child-offset range",
        )
        .with_detail("root_rows", root_rows.to_string())
        .with_detail("child_rows", child_rows.to_string()));
    }

    let root_offset_count = checked_size_add(root_rows, 1, operation, "sizing list offset count")?;
    let root_offsets = checked_size_mul(root_offset_count, 4, operation, "sizing list offsets")?;
    let root_validity = if root_has_nulls { cudf_null_mask_allocation_size_bytes(root_rows) } else { 0 };
    let child_data = list_child_import_device_bytes(child_type, child_rows, string_chars, operation)?;
    let child_validity = if child_has_nulls { cudf_null_mask_allocation_size_bytes(child_rows) } else { 0 };
    checked_size_add(
        checked_size_add(root_offsets, root_validity, operation, "sizing list root buffers")?,
        checked_size_add(child_data, child_validity, operation, "sizing list child buffers")?,
        operation,
        "sizing imported list column",
    )
}

fn active_list_child_range(array: &dyn Array, operation: &'static str) -> Result<(ArrayRef, usize, usize)> {
    let bounds = |offsets: &[i64], values: ArrayRef| -> Result<(ArrayRef, usize, usize)> {
        let start =
            usize::try_from(offsets[0]).map_err(|_| size_overflow(operation, "converting list start offset"))?;
        let end = usize::try_from(offsets[offsets.len() - 1])
            .map_err(|_| size_overflow(operation, "converting list end offset"))?;
        Ok((values, start, end))
    };
    match array.data_type() {
        ArrowDataType::List(_) => {
            let list = array.as_any().downcast_ref::<ListArray>().ok_or_else(|| {
                Error::new_with(ErrorCode::TypeMismatch, operation, "Arrow List array had an incompatible runtime type")
            })?;
            let offsets = list.value_offsets().iter().map(|offset| i64::from(*offset)).collect::<Vec<_>>();
            bounds(&offsets, Arc::clone(list.values()))
        },
        ArrowDataType::LargeList(_) => {
            let list = array.as_any().downcast_ref::<LargeListArray>().ok_or_else(|| {
                Error::new_with(
                    ErrorCode::TypeMismatch,
                    operation,
                    "Arrow LargeList array had an incompatible runtime type",
                )
            })?;
            bounds(list.value_offsets(), Arc::clone(list.values()))
        },
        _ => unreachable!("caller validates the list container type"),
    }
}

fn active_string_bytes(
    values: &dyn Array,
    start: usize,
    end: usize,
    child_type: &ArrowDataType,
    operation: &'static str,
) -> Result<usize> {
    match child_type {
        ArrowDataType::Utf8 => {
            let strings = values.as_any().downcast_ref::<StringArray>().ok_or_else(|| {
                Error::new_with(
                    ErrorCode::TypeMismatch,
                    operation,
                    "Arrow Utf8 list child had an incompatible runtime type",
                )
            })?;
            Ok(usize::try_from(strings.value_offsets()[end] - strings.value_offsets()[start])
                .map_err(|_| size_overflow(operation, "converting Utf8 child byte span"))?)
        },
        ArrowDataType::LargeUtf8 => {
            let strings = values.as_any().downcast_ref::<LargeStringArray>().ok_or_else(|| {
                Error::new_with(
                    ErrorCode::TypeMismatch,
                    operation,
                    "Arrow LargeUtf8 list child had an incompatible runtime type",
                )
            })?;
            Ok(usize::try_from(strings.value_offsets()[end] - strings.value_offsets()[start])
                .map_err(|_| size_overflow(operation, "converting LargeUtf8 child byte span"))?)
        },
        ArrowDataType::Utf8View => {
            let strings = values.as_any().downcast_ref::<StringViewArray>().ok_or_else(|| {
                Error::new_with(
                    ErrorCode::TypeMismatch,
                    operation,
                    "Arrow Utf8View list child had an incompatible runtime type",
                )
            })?;
            (start..end).try_fold(0usize, |total, index| {
                checked_size_add(total, strings.value(index).len(), operation, "summing Utf8View child bytes")
            })
        },
        _ => Ok(0),
    }
}

fn list_child_import_device_bytes(
    child_type: &ArrowDataType,
    child_rows: usize,
    string_chars: usize,
    operation: &'static str,
) -> Result<usize> {
    let width = match child_type {
        ArrowDataType::Boolean | ArrowDataType::Int8 | ArrowDataType::UInt8 => Some(1),
        ArrowDataType::Int16 | ArrowDataType::UInt16 => Some(2),
        ArrowDataType::Int32 | ArrowDataType::UInt32 | ArrowDataType::Float32 => Some(4),
        ArrowDataType::Int64 | ArrowDataType::UInt64 | ArrowDataType::Float64 => Some(8),
        ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View => None,
        _ => unreachable!("caller validates the basic list child type"),
    };
    if let Some(width) = width {
        return checked_size_mul(child_rows, width, operation, "sizing fixed-width list child values");
    }
    if child_rows == 0 {
        return Ok(0);
    }
    let child_offset_count = checked_size_add(child_rows, 1, operation, "sizing string child offset count")?;
    checked_size_add(
        checked_size_mul(child_offset_count, 4, operation, "sizing string child offsets")?,
        string_chars,
        operation,
        "sizing string child buffers",
    )
}

/// Conservatively estimate the Arrow buffer capacity retained by a host-copy
/// export before any device values are copied to host.
///
/// The estimate follows the final external Arrow schema rather than the cuDF
/// source representation. In particular, Decimal32/Decimal64 widen to
/// Decimal128, booleans become bit-packed, and string/list offset widths follow
/// Utf8/LargeUtf8 and List/LargeList respectively. Every logical Arrow buffer
/// is rounded to Arrow's 64-byte mutable-buffer allocation quantum, making the
/// result an upper bound for the capacities counted by DataFusion's
/// `get_record_batch_memory_size`.
pub(crate) fn estimate_arrow_host_record_batch_memory_bytes(
    frame: &DataFrame,
    schema: &ArrowHostSchema,
) -> Result<usize> {
    let operation = "arrow_host::estimate_record_batch_memory_bytes";
    if frame.num_columns() != schema.len() {
        return Err(Error::new_with(
            ErrorCode::InvariantViolation,
            operation,
            "validated Arrow schema and DataFrame column counts diverged before host materialization",
        )
        .with_detail("frame_columns", frame.num_columns().to_string())
        .with_detail("schema_fields", schema.len().to_string()));
    }

    // Native operators may represent an empty string/list column without the
    // offset buffers that exist in a non-empty cuDF column. No values can be
    // copied in this case, so size the typed Arrow destination directly from
    // its validated external schema instead of inferring character bytes by
    // subtracting cuDF structural payloads that are legitimately absent.
    if frame.num_rows() == 0 {
        return schema.fields().iter().try_fold(0usize, |total, field| {
            let column_bytes = estimate_empty_arrow_host_column_memory_bytes(field.arrow_data_type(), operation)?;
            checked_size_add(total, column_bytes, operation, "accumulating empty Arrow column capacities")
        });
    }

    let mut total = 0usize;
    for field in schema.fields() {
        let series = frame
            .series(field.name())
            .map_err(|error| error.reanchor(operation).with_column(field.name().to_string()))?;
        let view = series.view().map_err(|error| error.reanchor(operation).with_column(field.name().to_string()))?;
        let payload_bytes = series
            .as_column()
            .payload_size_bytes()
            .map_err(|error| error.reanchor(operation).with_column(field.name().to_string()))?;
        let column_bytes =
            estimate_arrow_host_column_memory_bytes(&view, payload_bytes, field.arrow_data_type(), operation)
                .map_err(|error| error.with_column(field.name().to_string()))?;
        total = checked_size_add(total, column_bytes, operation, "accumulating Arrow column capacities")?;
    }
    Ok(total)
}

fn estimate_empty_arrow_host_column_memory_bytes(arrow_type: &ArrowDataType, operation: &'static str) -> Result<usize> {
    match arrow_type {
        ArrowDataType::Boolean
        | ArrowDataType::Int8
        | ArrowDataType::UInt8
        | ArrowDataType::Int16
        | ArrowDataType::UInt16
        | ArrowDataType::Int32
        | ArrowDataType::UInt32
        | ArrowDataType::Float32
        | ArrowDataType::Date32
        | ArrowDataType::Time32(_)
        | ArrowDataType::Int64
        | ArrowDataType::UInt64
        | ArrowDataType::Float64
        | ArrowDataType::Date64
        | ArrowDataType::Time64(_)
        | ArrowDataType::Timestamp(_, _)
        | ArrowDataType::Duration(_)
        | ArrowDataType::Decimal128(_, _)
        | ArrowDataType::Utf8View => Ok(0),
        ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 => estimate_arrow_string_buffers(0, 0, arrow_type, operation),
        ArrowDataType::List(child) | ArrowDataType::LargeList(child) => {
            let offset_width = if matches!(arrow_type, ArrowDataType::LargeList(_)) { 8 } else { 4 };
            let root_offsets = estimate_offset_capacity(0, offset_width, operation)?;
            let child_buffers = estimate_empty_arrow_host_column_memory_bytes(child.data_type(), operation)?;
            checked_size_add(root_offsets, child_buffers, operation, "sizing empty Arrow list buffers")
        },
        ArrowDataType::FixedSizeList(child, dimension) if *dimension > 0 => {
            estimate_empty_arrow_host_column_memory_bytes(child.data_type(), operation)
        },
        _ => Err(Error::new_with(
            ErrorCode::InvariantViolation,
            operation,
            "validated Arrow host schema contains an unhandled empty materialization type",
        )
        .with_detail("arrow_type", format!("{arrow_type:?}"))),
    }
}

fn estimate_arrow_host_column_memory_bytes(
    view: &ColumnView<'_>,
    cudf_payload_bytes: usize,
    arrow_type: &ArrowDataType,
    operation: &'static str,
) -> Result<usize> {
    let rows = view.len().map_err(|error| error.reanchor(operation))?;
    let nullable = view.nullable().map_err(|error| error.reanchor(operation))?;
    let validity_bytes = estimate_arrow_validity_capacity(rows, nullable, operation)?;

    let values_bytes = match arrow_type {
        ArrowDataType::Boolean => estimate_arrow_buffer_capacity(bitpacked_bytes(rows, operation)?, operation)?,
        ArrowDataType::Int8 | ArrowDataType::UInt8 => estimate_fixed_width_capacity(rows, 1, operation)?,
        ArrowDataType::Int16 | ArrowDataType::UInt16 => estimate_fixed_width_capacity(rows, 2, operation)?,
        ArrowDataType::Int32
        | ArrowDataType::UInt32
        | ArrowDataType::Float32
        | ArrowDataType::Date32
        | ArrowDataType::Time32(_) => estimate_fixed_width_capacity(rows, 4, operation)?,
        ArrowDataType::Int64
        | ArrowDataType::UInt64
        | ArrowDataType::Float64
        | ArrowDataType::Date64
        | ArrowDataType::Time64(_)
        | ArrowDataType::Timestamp(_, _)
        | ArrowDataType::Duration(_) => estimate_fixed_width_capacity(rows, 8, operation)?,
        ArrowDataType::Decimal128(_, _) => estimate_fixed_width_capacity(rows, 16, operation)?,
        ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View => {
            let chars_bytes = cudf_string_chars_bytes(view, cudf_payload_bytes, operation)?;
            estimate_arrow_string_buffers(rows, chars_bytes, arrow_type, operation)?
        },
        ArrowDataType::List(child) | ArrowDataType::LargeList(child) => {
            let child_view = view.list_sliced_child().map_err(|error| error.reanchor(operation))?;
            let child_rows = child_view.len().map_err(|error| error.reanchor(operation))?;
            let child_nullable = child_view.nullable().map_err(|error| error.reanchor(operation))?;

            let root_offset_width = if matches!(arrow_type, ArrowDataType::LargeList(_)) { 8 } else { 4 };
            let root_offsets = estimate_offset_capacity(rows, root_offset_width, operation)?;
            if list_fixed_width_interop_dtype(child.data_type()).is_some() {
                let child_values =
                    estimate_arrow_list_fixed_width_values_capacity(child_rows, child.data_type(), operation)?;
                let child_validity = estimate_arrow_validity_capacity(child_rows, child_nullable, operation)?;
                checked_size_add(
                    checked_size_add(root_offsets, child_values, operation, "sizing Arrow numeric list values")?,
                    child_validity,
                    operation,
                    "sizing Arrow numeric list child validity",
                )?
            } else {
                let chars_bytes = cudf_list_string_chars_bytes(
                    rows,
                    nullable,
                    child_rows,
                    child_nullable,
                    cudf_payload_bytes,
                    operation,
                )?;

                let child_buffers =
                    estimate_arrow_string_buffers(child_rows, chars_bytes, child.data_type(), operation)?;
                let child_validity = estimate_arrow_validity_capacity(child_rows, child_nullable, operation)?;
                checked_size_add(
                    checked_size_add(root_offsets, child_buffers, operation, "sizing Arrow list child buffers")?,
                    child_validity,
                    operation,
                    "sizing Arrow list child validity",
                )?
            }
        },
        ArrowDataType::FixedSizeList(child, dimension) if *dimension > 0 => {
            let child_view = view.list_sliced_child().map_err(|error| error.reanchor(operation))?;
            let child_rows = child_view.len().map_err(|error| error.reanchor(operation))?;
            let child_nullable = child_view.nullable().map_err(|error| error.reanchor(operation))?;
            let dimension = usize::try_from(*dimension).map_err(|_| {
                Error::new_with(ErrorCode::ValueOutOfRange, operation, "FixedSizeList dimension must be positive")
                    .with_detail("dimension", dimension.to_string())
            })?;
            let expected_child_rows = rows.checked_mul(dimension).ok_or_else(|| {
                Error::new_with(ErrorCode::SizeOverflow, operation, "FixedSizeList child row count overflowed usize")
                    .with_detail("rows", rows.to_string())
                    .with_detail("dimension", dimension.to_string())
            })?;
            if child_rows != expected_child_rows {
                return Err(Error::new_with(
                    ErrorCode::ColumnLengthMismatch,
                    operation,
                    "cuDF list rows do not satisfy the requested FixedSizeList dimension",
                )
                .with_fact(crate::core::error::FACT_EXPECTED_ROWS, expected_child_rows)
                .with_fact(crate::core::error::FACT_ACTUAL_ROWS, child_rows)
                .with_detail("dimension", dimension.to_string())
                .with_detail("reason", "fixed_size_list_dimension_mismatch"));
            }
            let child_values =
                estimate_arrow_list_fixed_width_values_capacity(child_rows, child.data_type(), operation)?;
            let child_validity = estimate_arrow_validity_capacity(child_rows, child_nullable, operation)?;
            checked_size_add(child_values, child_validity, operation, "sizing Arrow FixedSizeList child buffers")?
        },
        _ => {
            return Err(Error::new_with(
                ErrorCode::InvariantViolation,
                operation,
                "validated Arrow host schema contains an unhandled materialization type",
            )
            .with_detail("arrow_type", format!("{arrow_type:?}")));
        },
    };

    checked_size_add(values_bytes, validity_bytes, operation, "sizing Arrow column validity")
}

fn cudf_string_chars_bytes(view: &ColumnView<'_>, cudf_payload_bytes: usize, operation: &'static str) -> Result<usize> {
    let rows = view.len().map_err(|error| error.reanchor(operation))?;
    let nullable = view.nullable().map_err(|error| error.reanchor(operation))?;
    let structural = checked_size_add(
        cudf_offset_payload_bytes(rows, operation)?,
        cudf_validity_payload_bytes(rows, nullable, operation)?,
        operation,
        "sizing cuDF string structure",
    )?;
    cudf_payload_bytes.checked_sub(structural).ok_or_else(|| {
        Error::new_with(
            ErrorCode::InvariantViolation,
            operation,
            "cuDF string payload is smaller than its offsets and validity buffers",
        )
        .with_detail("cudf_payload_bytes", cudf_payload_bytes.to_string())
        .with_detail("structural_bytes", structural.to_string())
    })
}

fn estimate_arrow_string_buffers(
    rows: usize,
    chars_bytes: usize,
    arrow_type: &ArrowDataType,
    operation: &'static str,
) -> Result<usize> {
    let descriptor = match arrow_type {
        ArrowDataType::Utf8 => estimate_offset_capacity(rows, 4, operation)?,
        ArrowDataType::LargeUtf8 => estimate_offset_capacity(rows, 8, operation)?,
        ArrowDataType::Utf8View => estimate_fixed_width_capacity(rows, 16, operation)?,
        _ => {
            return Err(Error::new_with(
                ErrorCode::InvariantViolation,
                operation,
                "validated list child is not a supported Arrow string type",
            )
            .with_detail("arrow_type", format!("{arrow_type:?}")));
        },
    };
    let chars = estimate_arrow_buffer_capacity(chars_bytes, operation)?;
    checked_size_add(descriptor, chars, operation, "sizing Arrow string descriptor and values buffers")
}

fn estimate_arrow_list_fixed_width_values_capacity(
    rows: usize,
    arrow_type: &ArrowDataType,
    operation: &'static str,
) -> Result<usize> {
    if matches!(arrow_type, ArrowDataType::Boolean) {
        return estimate_arrow_buffer_capacity(bitpacked_bytes(rows, operation)?, operation);
    }
    let width = match arrow_type {
        ArrowDataType::Int8 | ArrowDataType::UInt8 => 1,
        ArrowDataType::Int16 | ArrowDataType::UInt16 => 2,
        ArrowDataType::Int32 | ArrowDataType::UInt32 | ArrowDataType::Float32 => 4,
        ArrowDataType::Int64 | ArrowDataType::UInt64 | ArrowDataType::Float64 => 8,
        _ => {
            return Err(Error::new_with(
                ErrorCode::InvariantViolation,
                operation,
                "validated numeric list child has an unsupported Arrow type",
            )
            .with_detail("arrow_type", format!("{arrow_type:?}")));
        },
    };
    estimate_fixed_width_capacity(rows, width, operation)
}

fn estimate_fixed_width_capacity(rows: usize, width: usize, operation: &'static str) -> Result<usize> {
    let logical = checked_size_mul(rows, width, operation, "sizing Arrow fixed-width values")?;
    estimate_arrow_buffer_capacity(logical, operation)
}

fn estimate_offset_capacity(rows: usize, width: usize, operation: &'static str) -> Result<usize> {
    let count = rows.checked_add(1).ok_or_else(|| size_overflow(operation, "sizing Arrow offset count"))?;
    let logical = checked_size_mul(count, width, operation, "sizing Arrow offsets")?;
    estimate_arrow_buffer_capacity(logical, operation)
}

fn estimate_arrow_validity_capacity(rows: usize, nullable: bool, operation: &'static str) -> Result<usize> {
    if !nullable {
        return Ok(0);
    }
    estimate_arrow_buffer_capacity(bitpacked_bytes(rows, operation)?, operation)
}

fn estimate_arrow_buffer_capacity(logical_bytes: usize, operation: &'static str) -> Result<usize> {
    if logical_bytes == 0 {
        return Ok(0);
    }
    logical_bytes
        .checked_add(63)
        .map(|bytes| bytes & !63usize)
        .ok_or_else(|| size_overflow(operation, "rounding Arrow buffer capacity to 64 bytes"))
}

fn bitpacked_bytes(rows: usize, operation: &'static str) -> Result<usize> {
    rows.checked_add(7).map(|bits| bits / 8).ok_or_else(|| size_overflow(operation, "sizing Arrow bit-packed buffer"))
}

fn cudf_offset_payload_bytes(rows: usize, operation: &'static str) -> Result<usize> {
    let count = rows.checked_add(1).ok_or_else(|| size_overflow(operation, "sizing cuDF offset count"))?;
    checked_size_mul(count, std::mem::size_of::<i32>(), operation, "sizing cuDF offsets")
}

fn cudf_validity_payload_bytes(rows: usize, nullable: bool, operation: &'static str) -> Result<usize> {
    if !nullable {
        return Ok(0);
    }
    let words = rows
        .checked_add(31)
        .map(|bits| bits / 32)
        .ok_or_else(|| size_overflow(operation, "sizing cuDF validity word count"))?;
    checked_size_mul(words, std::mem::size_of::<u32>(), operation, "sizing cuDF validity payload")
}

fn cudf_list_string_chars_bytes(
    rows: usize,
    nullable: bool,
    child_rows: usize,
    child_nullable: bool,
    cudf_payload_bytes: usize,
    operation: &'static str,
) -> Result<usize> {
    let root_structural = checked_size_add(
        cudf_offset_payload_bytes(rows, operation)?,
        cudf_validity_payload_bytes(rows, nullable, operation)?,
        operation,
        "sizing cuDF list root structure",
    )?;
    let child_structural = if child_rows == 0 {
        0
    } else {
        checked_size_add(
            cudf_offset_payload_bytes(child_rows, operation)?,
            cudf_validity_payload_bytes(child_rows, child_nullable, operation)?,
            operation,
            "sizing cuDF list string child structure",
        )?
    };
    let structural =
        checked_size_add(root_structural, child_structural, operation, "sizing cuDF list and string child structure")?;
    cudf_payload_bytes.checked_sub(structural).ok_or_else(|| {
        Error::new_with(
            ErrorCode::InvariantViolation,
            operation,
            "cuDF List<String> payload is smaller than its structural buffers",
        )
        .with_detail("cudf_payload_bytes", cudf_payload_bytes.to_string())
        .with_detail("structural_bytes", structural.to_string())
    })
}

fn checked_size_add(left: usize, right: usize, operation: &'static str, context: &'static str) -> Result<usize> {
    left.checked_add(right).ok_or_else(|| size_overflow(operation, context))
}

fn checked_size_mul(left: usize, right: usize, operation: &'static str, context: &'static str) -> Result<usize> {
    left.checked_mul(right).ok_or_else(|| size_overflow(operation, context))
}

fn size_overflow(operation: &'static str, context: &'static str) -> Error {
    Error::new_with(ErrorCode::SizeOverflow, operation, "host materialization size accounting overflowed usize")
        .with_detail("size_context", context)
}

fn cudf_dtype_from_arrow_type_in(dt: &ArrowDataType, operation: &'static str) -> Result<DataType> {
    let dtype = match dt {
        ArrowDataType::Int8 => DataType::new(TypeId::Int8),
        ArrowDataType::Int16 => DataType::new(TypeId::Int16),
        ArrowDataType::Int32 => DataType::new(TypeId::Int32),
        ArrowDataType::Int64 => DataType::new(TypeId::Int64),
        ArrowDataType::UInt8 => DataType::new(TypeId::UInt8),
        ArrowDataType::UInt16 => DataType::new(TypeId::UInt16),
        ArrowDataType::UInt32 => DataType::new(TypeId::UInt32),
        ArrowDataType::UInt64 => DataType::new(TypeId::UInt64),
        ArrowDataType::Float32 => DataType::new(TypeId::Float32),
        ArrowDataType::Float64 => DataType::new(TypeId::Float64),
        ArrowDataType::Boolean => DataType::new(TypeId::Bool8),
        ArrowDataType::Date32 => DataType::new(TypeId::TimestampDays),
        ArrowDataType::Date64 => DataType::new(TypeId::TimestampMilliseconds),
        ArrowDataType::Time32(TimeUnit::Second) => DataType::new(TypeId::DurationSeconds),
        ArrowDataType::Time32(TimeUnit::Millisecond) => DataType::new(TypeId::DurationMilliseconds),
        ArrowDataType::Time64(TimeUnit::Microsecond) => DataType::new(TypeId::DurationMicroseconds),
        ArrowDataType::Time64(TimeUnit::Nanosecond) => DataType::new(TypeId::DurationNanoseconds),
        ArrowDataType::Timestamp(unit, _) => cudf_timestamp_dtype(unit),
        ArrowDataType::Duration(TimeUnit::Second) => DataType::new(TypeId::DurationSeconds),
        ArrowDataType::Duration(TimeUnit::Millisecond) => DataType::new(TypeId::DurationMilliseconds),
        ArrowDataType::Duration(TimeUnit::Microsecond) => DataType::new(TypeId::DurationMicroseconds),
        ArrowDataType::Duration(TimeUnit::Nanosecond) => DataType::new(TypeId::DurationNanoseconds),
        ArrowDataType::Decimal128(_, scale) => {
            DataType::fixed_point(TypeId::Decimal128, arrow_decimal_scale_to_cudf_scale(*scale))
        },
        ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View => DataType::new(TypeId::String),
        ArrowDataType::List(field) | ArrowDataType::LargeList(field) if list_child_is_supported(field) => {
            DataType::new(TypeId::List)
        },
        ArrowDataType::FixedSizeList(field, dimension) => {
            if *dimension <= 0 {
                return Err(Error::new_with(
                    ErrorCode::ValueOutOfRange,
                    operation,
                    "Arrow FixedSizeList dimension must be positive for cudf conversion",
                )
                .with_detail("dimension", dimension.to_string())
                .with_detail("reason", "non_positive_fixed_size_list_dimension"));
            }
            if !list_numeric_child_is_supported(field) {
                return Err(Error::new_with(
                    ErrorCode::UnsupportedDtype,
                    operation,
                    format!("unsupported Arrow FixedSizeList child type for cudf conversion: {:?}", field.data_type()),
                )
                .with_detail("arrow_type", format!("{:?}", dt))
                .with_detail("reason", "unsupported_fixed_size_list_child_type"));
            }
            DataType::new(TypeId::List)
        },
        _ => {
            return Err(Error::new_with(
                ErrorCode::UnsupportedDtype,
                operation,
                format!("unsupported Arrow type for cudf conversion: {:?}", dt),
            )
            .with_detail("arrow_type", format!("{:?}", dt))
            .with_detail("reason", "unsupported_arrow_type"));
        },
    };

    Ok(dtype)
}

fn list_string_child_is_supported(field: &FieldRef) -> bool {
    matches!(field.data_type(), ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View)
}

fn list_numeric_child_is_supported(field: &FieldRef) -> bool {
    numeric_interop_dtype(field.data_type()).is_some()
}

fn list_fixed_width_child_is_supported(field: &FieldRef) -> bool {
    list_fixed_width_interop_dtype(field.data_type()).is_some()
}

fn list_child_is_supported(field: &FieldRef) -> bool {
    list_string_child_is_supported(field) || list_fixed_width_child_is_supported(field)
}

fn cudf_timestamp_dtype(unit: &TimeUnit) -> DataType {
    match unit {
        TimeUnit::Second => DataType::new(TypeId::TimestampSeconds),
        TimeUnit::Millisecond => DataType::new(TypeId::TimestampMilliseconds),
        TimeUnit::Microsecond => DataType::new(TypeId::TimestampMicroseconds),
        TimeUnit::Nanosecond => DataType::new(TypeId::TimestampNanoseconds),
    }
}

fn arrow_type_from_cudf_dtype_in(dtype: DataType, operation: &'static str) -> Result<ArrowDataType> {
    match dtype.id() {
        TypeId::Int8 => Ok(ArrowDataType::Int8),
        TypeId::Int16 => Ok(ArrowDataType::Int16),
        TypeId::Int32 => Ok(ArrowDataType::Int32),
        TypeId::Int64 => Ok(ArrowDataType::Int64),
        TypeId::UInt8 => Ok(ArrowDataType::UInt8),
        TypeId::UInt16 => Ok(ArrowDataType::UInt16),
        TypeId::UInt32 => Ok(ArrowDataType::UInt32),
        TypeId::UInt64 => Ok(ArrowDataType::UInt64),
        TypeId::Float32 => Ok(ArrowDataType::Float32),
        TypeId::Float64 => Ok(ArrowDataType::Float64),
        TypeId::Bool8 => Ok(ArrowDataType::Boolean),
        TypeId::TimestampDays => Ok(ArrowDataType::Date32),
        TypeId::TimestampSeconds => Ok(ArrowDataType::Timestamp(TimeUnit::Second, None)),
        TypeId::TimestampMilliseconds => Ok(ArrowDataType::Timestamp(TimeUnit::Millisecond, None)),
        TypeId::TimestampMicroseconds => Ok(ArrowDataType::Timestamp(TimeUnit::Microsecond, None)),
        TypeId::TimestampNanoseconds => Ok(ArrowDataType::Timestamp(TimeUnit::Nanosecond, None)),
        TypeId::DurationSeconds => Ok(ArrowDataType::Duration(TimeUnit::Second)),
        TypeId::DurationMilliseconds => Ok(ArrowDataType::Duration(TimeUnit::Millisecond)),
        TypeId::DurationMicroseconds => Ok(ArrowDataType::Duration(TimeUnit::Microsecond)),
        TypeId::DurationNanoseconds => Ok(ArrowDataType::Duration(TimeUnit::Nanosecond)),
        TypeId::String => Ok(ArrowDataType::Utf8),
        TypeId::Decimal32 | TypeId::Decimal64 | TypeId::Decimal128 => {
            let arrow_scale = cudf_decimal_scale_to_arrow_scale(dtype.scale(), operation)
                .map_err(|error| error.with_fact(FACT_TYPE_ID, dtype.id()))?;
            Ok(ArrowDataType::Decimal128(DEFAULT_DECIMAL128_ARROW_PRECISION, arrow_scale))
        },
        _ => Err(Error::new_with(
            ErrorCode::UnsupportedDtype,
            operation,
            format!("unsupported cudf type for Arrow host conversion: {:?}", dtype.id()),
        )
        .with_fact(FACT_TYPE_ID, dtype.id())),
    }
}

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
        record_host_export_call();
        let names = self.column_names();
        let rows = self.num_rows();
        let column_count = self.num_columns();
        let placement = self.placement();

        let col_map = self.columns_map();
        let columns: Vec<(String, Arc<dyn Array>, bool)> = names
            .iter()
            .map(|&name| {
                let col = col_map.get(name).ok_or_else(|| missing_column_error("dataframe::to_record_batch", name))?;
                let cv = col.view()?;
                let dtype = cv.data_type()?;
                let nullable = cv.nullable()?;
                let arr = column_view_to_arrow(&cv, dtype)?;
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
/// [`dataframe_from_record_batch`] — one import per column instead of one
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
    record_host_import_call();
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
                let datas: Vec<arrow::array::ArrayData> = arrays.iter().map(|array| array.to_data()).collect();
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
                let merged = arrow::compute::concat(&arrays).map_err(|error| {
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
    let device_buffer = crate::core::DeviceBuffer::new_in_runtime(runtime, total_bytes)?;
    if total_bytes > 0 {
        let _device_guard = crate::runtime::device_guard::DeviceGuard::new(runtime.device())
            .map_err(|error| error.reanchor("arrow::staged_fixed_width_column"))?;
        let pool = crate::runtime::host_staging::pinned_staging_pool();
        let mut leases = [pool.acquire()?, pool.acquire()?];
        let events = [crate::runtime::CudaEvent::new()?, crate::runtime::CudaEvent::new()?];
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
    device_buffer: &crate::core::DeviceBuffer,
    lease: &mut crate::runtime::host_staging::PinnedSlabLease,
    fill: usize,
    dst_offset: &mut usize,
    event: &crate::runtime::CudaEvent,
    stream: &crate::runtime::stream::CudaStream,
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

impl TryFrom<TypeId> for ArrowDataType {
    type Error = Error;

    fn try_from(type_id: TypeId) -> Result<Self> {
        if type_id.is_decimal() {
            return Err(Error::new_with(
                ErrorCode::UnsupportedDtype,
                "arrow::TryFrom<TypeId> for ArrowDataType",
                format!("decimal cudf type {:?} requires scale for Arrow host conversion", type_id),
            )
            .with_fact(FACT_TYPE_ID, type_id)
            .with_detail("reason", "decimal_scale_required")
            .with_detail("remedy", "use arrow_host::arrow_type_from_cudf_dtype with DataType::fixed_point"));
        }
        arrow_type_from_cudf_dtype_in(DataType::new(type_id), "arrow::TryFrom<TypeId> for ArrowDataType")
    }
}

impl TryFrom<&ArrowDataType> for TypeId {
    type Error = Error;

    fn try_from(dt: &ArrowDataType) -> Result<Self> {
        cudf_dtype_from_arrow_type_in(dt, "arrow::TryFrom<&ArrowDataType> for TypeId").map(|dtype| dtype.id())
    }
}

/// Convert a single `ColumnView` to an Arrow `Array`.
fn column_view_to_arrow(cv: &ColumnView<'_>, dtype: DataType) -> Result<Arc<dyn Array>> {
    match dtype.id() {
        TypeId::Int8 => numeric_to_arrow::<i8>(cv),
        TypeId::Int16 => numeric_to_arrow::<i16>(cv),
        TypeId::Int32 => numeric_to_arrow::<i32>(cv),
        TypeId::Int64 => numeric_to_arrow::<i64>(cv),
        TypeId::UInt8 => numeric_to_arrow::<u8>(cv),
        TypeId::UInt16 => numeric_to_arrow::<u16>(cv),
        TypeId::UInt32 => numeric_to_arrow::<u32>(cv),
        TypeId::UInt64 => numeric_to_arrow::<u64>(cv),
        TypeId::Float32 => numeric_to_arrow::<f32>(cv),
        TypeId::Float64 => numeric_to_arrow::<f64>(cv),
        TypeId::Bool8 => bool_to_arrow(cv),
        TypeId::String => string_to_arrow(cv),
        TypeId::Decimal32 => decimal_widening_to_arrow::<i32>(cv, dtype),
        TypeId::Decimal64 => decimal_widening_to_arrow::<i64>(cv, dtype),
        TypeId::Decimal128 => decimal128_to_arrow(cv, dtype),
        TypeId::TimestampDays => fixed_width_to_arrow::<i32, arrow::datatypes::Date32Type>(cv),
        TypeId::TimestampSeconds => fixed_width_to_arrow::<i64, arrow::datatypes::TimestampSecondType>(cv),
        TypeId::TimestampMilliseconds => fixed_width_to_arrow::<i64, arrow::datatypes::TimestampMillisecondType>(cv),
        TypeId::TimestampMicroseconds => fixed_width_to_arrow::<i64, arrow::datatypes::TimestampMicrosecondType>(cv),
        TypeId::TimestampNanoseconds => fixed_width_to_arrow::<i64, arrow::datatypes::TimestampNanosecondType>(cv),
        TypeId::DurationSeconds => fixed_width_to_arrow::<i64, arrow::datatypes::DurationSecondType>(cv),
        TypeId::DurationMilliseconds => fixed_width_to_arrow::<i64, arrow::datatypes::DurationMillisecondType>(cv),
        TypeId::DurationMicroseconds => fixed_width_to_arrow::<i64, arrow::datatypes::DurationMicrosecondType>(cv),
        TypeId::DurationNanoseconds => fixed_width_to_arrow::<i64, arrow::datatypes::DurationNanosecondType>(cv),
        TypeId::List => list_to_arrow(cv),
        _ => Err(Error::new_with(
            ErrorCode::UnsupportedDtype,
            "arrow::column_view_to_arrow",
            format!("unsupported cudf type for Arrow conversion: {:?}", dtype.id()),
        )
        .with_fact(FACT_TYPE_ID, dtype.id())),
    }
}

pub(super) fn map_record_batch_error(error: arrow::error::ArrowError) -> Error {
    error::foreign_invariant_violation(
        "dataframe::to_record_batch",
        error.to_string(),
        "record_batch",
        "invalid_arrow_record_batch",
        "ensure Arrow schemas and column arrays have matching row counts and compatible field layouts",
    )
    .with_source(error)
}

pub(super) fn map_numeric_to_arrow_error(error: arrow::error::ArrowError) -> Error {
    error::foreign_invariant_violation(
        "arrow::numeric_to_arrow",
        error.to_string(),
        "arrow_array",
        "invalid_arrow_primitive_array",
        "ensure Arrow primitive values and null buffers have matching logical lengths",
    )
    .with_source(error)
}

fn map_arrow_ffi_export_error(operation: &'static str, error: arrow::error::ArrowError) -> Error {
    error::foreign_invariant_violation(
        operation,
        error.to_string(),
        "arrow_array",
        "invalid_arrow_c_data_export",
        "ensure the Arrow array can be exported through the Arrow C Data Interface",
    )
    .with_source(error)
}

/// Convert a numeric column to the corresponding Arrow primitive array.
fn numeric_to_arrow<T>(cv: &ColumnView<'_>) -> Result<Arc<dyn Array>>
where T: ArrowPrimitiveMapping {
    let data: Vec<T> = cv.to_vec()?;
    let validity = cv.to_host_validity()?;

    match validity {
        None => {
            let arr = PrimitiveArray::<T::ArrowPrimitive>::from_iter_values(data);
            Ok(Arc::new(arr))
        },
        Some(mask) => {
            let null_buffer = arrow::buffer::NullBuffer::from(mask);
            let buffer = arrow::buffer::ScalarBuffer::from(data);
            let arr = PrimitiveArray::<T::ArrowPrimitive>::try_new(buffer, Some(null_buffer))
                .map_err(map_numeric_to_arrow_error)?;
            Ok(Arc::new(arr))
        },
    }
}

fn cudf_decimal_scale_to_arrow_scale(scale: i32, operation: &'static str) -> Result<i8> {
    let arrow_scale = scale.checked_neg().ok_or_else(|| {
        Error::new_with(ErrorCode::UnsupportedDtype, operation, "decimal scale is outside Arrow Decimal128 range")
            .with_detail("cudf_scale", scale.to_string())
            .with_detail("reason", "decimal_scale_overflow")
    })?;
    if arrow_scale > i32::from(DEFAULT_DECIMAL128_ARROW_PRECISION) {
        return Err(Error::new_with(
            ErrorCode::UnsupportedDtype,
            operation,
            "decimal scale exceeds cudf Arrow Decimal128 precision",
        )
        .with_detail("cudf_scale", scale.to_string())
        .with_detail("arrow_scale", arrow_scale.to_string())
        .with_detail("arrow_precision", DEFAULT_DECIMAL128_ARROW_PRECISION.to_string())
        .with_detail("reason", "decimal_scale_exceeds_decimal128_precision"));
    }
    i8::try_from(arrow_scale).map_err(|_| {
        Error::new_with(ErrorCode::UnsupportedDtype, operation, "decimal scale is outside Arrow Decimal128 range")
            .with_detail("cudf_scale", scale.to_string())
            .with_detail("arrow_scale", arrow_scale.to_string())
            .with_detail("reason", "decimal_scale_outside_arrow_i8_range")
    })
}

fn arrow_decimal_scale_to_cudf_scale(scale: i8) -> i32 {
    -i32::from(scale)
}

fn bytes_to_native_vec<T: Copy + Default>(bytes: &[u8]) -> Vec<T> {
    let elem_size = std::mem::size_of::<T>();
    assert_eq!(bytes.len() % elem_size, 0);
    let len = bytes.len() / elem_size;
    let mut out = vec![T::default(); len];
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr().cast::<u8>(), bytes.len());
    }
    out
}

fn native_slice_as_bytes<T>(values: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) }
}

fn fixed_width_host_values<T: Copy + Default>(cv: &ColumnView<'_>) -> Result<Vec<T>> {
    cv.to_fixed_width_host_bytes().map(|bytes| bytes_to_native_vec::<T>(&bytes))
}

fn fixed_width_to_arrow<T, A>(cv: &ColumnView<'_>) -> Result<Arc<dyn Array>>
where
    T: Copy + arrow::datatypes::ArrowNativeType,
    A: arrow::datatypes::ArrowPrimitiveType<Native = T>,
{
    let data = fixed_width_host_values::<T>(cv)?;
    let validity = cv.to_host_validity()?;

    let arr = match validity {
        None => PrimitiveArray::<A>::from_iter_values(data),
        Some(mask) => PrimitiveArray::<A>::from_iter(
            data.into_iter().zip(mask).map(|(value, valid)| if valid { Some(value) } else { None }),
        ),
    };
    Ok(Arc::new(arr))
}

fn list_to_arrow(cv: &ColumnView<'_>) -> Result<Arc<dyn Array>> {
    let operation = "arrow::list_to_arrow";
    let len = cv.len()?;
    let root_offset_raw = cv.offset()?;
    let root_offset = usize::try_from(root_offset_raw).map_err(|_| {
        Error::new_with(ErrorCode::IndexOutOfBounds, operation, "list column offset must be non-negative")
            .with_detail("offset", root_offset_raw.to_string())
            .with_detail("reason", "negative_list_offset")
    })?;
    let root_end = root_offset.checked_add(len).ok_or_else(|| {
        Error::new_with(ErrorCode::ValueOutOfRange, operation, "list active range overflowed usize")
            .with_detail("offset", root_offset.to_string())
            .with_detail("row_count", len.to_string())
            .with_detail("reason", "list_range_overflow")
    })?;

    let offsets_view = cv.child(0)?;
    let offsets_type = offsets_view.data_type()?;
    if offsets_type.id() != TypeId::Int32 {
        return Err(Error::new_with(
            ErrorCode::UnsupportedLayout,
            operation,
            "list offsets child must be Int32 for Arrow host export",
        )
        .with_detail("offsets_type", format!("{:?}", offsets_type.id()))
        .with_detail("reason", "unsupported_list_offsets_type"));
    }
    let host_offsets = offsets_view.to_vec::<i32>()?;
    let empty_cudf_offsets = host_offsets.is_empty() && len == 0 && root_offset == 0;
    if !empty_cudf_offsets && host_offsets.len() <= root_end {
        return Err(Error::new_with(
            ErrorCode::IndexOutOfBounds,
            operation,
            "list offsets child is too small for the active logical range",
        )
        .with_detail("offset", root_offset.to_string())
        .with_detail("row_count", len.to_string())
        .with_detail("offset_count", host_offsets.len().to_string())
        .with_detail("reason", "list_offsets_too_small"));
    }

    let normalized_offsets = if empty_cudf_offsets {
        vec![0]
    } else {
        let base_child_offset = host_offsets[root_offset];
        if base_child_offset < 0 {
            return Err(Error::new_with(ErrorCode::ValueOutOfRange, operation, "list offsets must be non-negative")
                .with_detail("offset_index", root_offset.to_string())
                .with_detail("offset_value", base_child_offset.to_string())
                .with_detail("reason", "negative_list_child_offset"));
        }
        let normalized_offset_count = len.checked_add(1).ok_or_else(|| {
            Error::new_with(ErrorCode::SizeOverflow, operation, "list offset count overflowed usize")
                .with_detail("row_count", len.to_string())
                .with_detail("reason", "list_offset_count_overflow")
        })?;
        let mut normalized_offsets = Vec::with_capacity(normalized_offset_count);
        let mut previous = base_child_offset;
        for (position, raw_offset) in host_offsets[root_offset..=root_end].iter().copied().enumerate() {
            if raw_offset < previous {
                return Err(Error::new_with(
                    ErrorCode::ValueOutOfRange,
                    operation,
                    "list offsets must be non-decreasing",
                )
                .with_detail("offset_index", (root_offset + position).to_string())
                .with_detail("previous_offset", previous.to_string())
                .with_detail("offset_value", raw_offset.to_string())
                .with_detail("reason", "non_monotonic_list_offsets"));
            }
            normalized_offsets.push(raw_offset - base_child_offset);
            previous = raw_offset;
        }
        normalized_offsets
    };

    let child_view = cv.list_sliced_child()?;
    let child_dtype = child_view.data_type()?;
    if child_dtype.id() != TypeId::String && child_dtype.id() != TypeId::Bool8 && !child_dtype.id().is_numeric() {
        return Err(Error::new_with(
            ErrorCode::UnsupportedDtype,
            operation,
            "Arrow host export supports only List<String> and List<fixed-width> columns",
        )
        .with_fact(FACT_TYPE_ID, child_dtype.id())
        .with_detail("reason", "unsupported_list_child_type"));
    }
    let child_array = column_view_to_arrow(&child_view, child_dtype)?;
    let child_nullable = child_view.nullable()?;
    let child_field = Arc::new(Field::new("item", child_array.data_type().clone(), child_nullable));
    let offsets = OffsetBuffer::new(ScalarBuffer::from(normalized_offsets));
    let nulls = cv.to_host_validity()?.and_then(|mask| {
        let nulls = NullBuffer::from(mask);
        (nulls.null_count() > 0).then_some(nulls)
    });
    let array = ListArray::try_new(child_field, offsets, child_array, nulls).map_err(map_list_arrow_error)?;
    Ok(Arc::new(array))
}

fn map_decimal_to_arrow_error(error: arrow::error::ArrowError) -> Error {
    error::foreign_invariant_violation(
        "arrow::decimal128_to_arrow",
        error.to_string(),
        "arrow_array",
        "invalid_arrow_decimal_array",
        "ensure cudf Decimal128 scale fits Arrow Decimal128 metadata",
    )
    .with_source(error)
}

fn map_list_arrow_error(error: arrow::error::ArrowError) -> Error {
    error::foreign_invariant_violation(
        "arrow::list_to_arrow",
        error.to_string(),
        "arrow_array",
        "invalid_arrow_list_array",
        "ensure list offsets, child values, and validity buffers form a valid Arrow ListArray",
    )
    .with_source(error)
}

fn decimal128_to_arrow(cv: &ColumnView<'_>, dtype: DataType) -> Result<Arc<dyn Array>> {
    let arrow_scale = cudf_decimal_scale_to_arrow_scale(dtype.scale(), "arrow::decimal128_to_arrow")
        .map_err(|error| error.with_fact(FACT_TYPE_ID, dtype.id()))?;
    let data = fixed_width_host_values::<i128>(cv)?;
    decimal128_array_from_values(data, cv.to_host_validity()?, arrow_scale)
}

fn decimal_widening_to_arrow<T>(cv: &ColumnView<'_>, dtype: DataType) -> Result<Arc<dyn Array>>
where T: Copy + Default + Into<i128> {
    let arrow_scale = cudf_decimal_scale_to_arrow_scale(dtype.scale(), "arrow::decimal_widening_to_arrow")
        .map_err(|error| error.with_fact(FACT_TYPE_ID, dtype.id()))?;
    let data = fixed_width_host_values::<T>(cv)?.into_iter().map(Into::into).collect::<Vec<_>>();
    decimal128_array_from_values(data, cv.to_host_validity()?, arrow_scale)
}

fn decimal128_array_from_values(
    data: Vec<i128>,
    validity: Option<Vec<bool>>,
    arrow_scale: i8,
) -> Result<Arc<dyn Array>> {
    let arr = match validity {
        None => Decimal128Array::from(data),
        Some(mask) => Decimal128Array::from(
            data.into_iter().zip(mask).map(|(value, valid)| if valid { Some(value) } else { None }).collect::<Vec<_>>(),
        ),
    }
    .with_precision_and_scale(DEFAULT_DECIMAL128_ARROW_PRECISION, arrow_scale)
    .map_err(map_decimal_to_arrow_error)?;

    Ok(Arc::new(arr))
}

pub(super) fn reset_host_interop_observation() {
    HOST_ARROW_EXPORT_CALL_BASELINE.with(|baseline| {
        reset_call_count_baseline(&HOST_ARROW_EXPORT_CALL_COUNT, baseline);
    });
    HOST_ARROW_IMPORT_CALL_BASELINE.with(|baseline| {
        reset_call_count_baseline(&HOST_ARROW_IMPORT_CALL_COUNT, baseline);
    });
    HOST_ARROW_LAST_BOUNDARY_OBSERVATION.with(|last| last.set(None));
}

pub(super) fn host_export_call_count() -> usize {
    HOST_ARROW_EXPORT_CALL_BASELINE.with(|baseline| call_count_since(&HOST_ARROW_EXPORT_CALL_COUNT, baseline))
}

pub(super) fn host_import_call_count() -> usize {
    HOST_ARROW_IMPORT_CALL_BASELINE.with(|baseline| call_count_since(&HOST_ARROW_IMPORT_CALL_COUNT, baseline))
}

fn reset_call_count_baseline(counter: &AtomicUsize, baseline: &Cell<usize>) {
    baseline.set(counter.load(Ordering::Acquire));
}

fn call_count_since(counter: &AtomicUsize, baseline: &Cell<usize>) -> usize {
    counter.load(Ordering::Acquire).saturating_sub(baseline.get())
}

pub(super) fn last_host_boundary_observation() -> Option<BoundaryObservation> {
    HOST_ARROW_LAST_BOUNDARY_OBSERVATION.with(Cell::get)
}

fn record_host_export_call() {
    HOST_ARROW_EXPORT_CALL_COUNT.fetch_add(1, Ordering::AcqRel);
}

fn record_host_import_call() {
    HOST_ARROW_IMPORT_CALL_COUNT.fetch_add(1, Ordering::AcqRel);
}

fn record_host_boundary_observation_in_domain(
    domain: &Arc<crate::runtime::execution_domain::ExecutionDomain>,
    observation: BoundaryObservation,
) {
    HOST_ARROW_LAST_BOUNDARY_OBSERVATION.with(|last| last.set(Some(observation)));
    record_boundary_observation_in_domain(domain, observation);
}

/// Convert a Bool8 column to an Arrow BooleanArray.
fn bool_to_arrow(cv: &ColumnView<'_>) -> Result<Arc<dyn Array>> {
    // Avoid constructing Rust `bool` values directly from raw device bytes.
    let data: Vec<bool> = cv.to_raw_bytes()?.into_iter().map(|byte| byte != 0).collect();
    let validity = cv.to_host_validity()?;

    match validity {
        None => Ok(Arc::new(BooleanArray::from(data))),
        Some(mask) => {
            let null_buffer = arrow::buffer::NullBuffer::from(mask);
            let bool_buffer = arrow::buffer::BooleanBuffer::from(data);
            let arr = BooleanArray::new(bool_buffer, Some(null_buffer));
            Ok(Arc::new(arr))
        },
    }
}

/// Convert a String column to an Arrow StringArray.
fn string_to_arrow(cv: &ColumnView<'_>) -> Result<Arc<dyn Array>> {
    let operation = "arrow::string_to_arrow";
    let data = cv.to_optional_strings()?;
    let chars_bytes = data.iter().try_fold(0usize, |total, value| {
        total.checked_add(value.as_ref().map_or(0, String::len)).ok_or_else(|| {
            Error::new_with(ErrorCode::SizeOverflow, operation, "string character byte count overflowed usize")
                .with_detail("reason", "string_character_count_overflow")
        })
    })?;
    let mut builder = StringBuilder::with_capacity(data.len(), chars_bytes);
    for value in &data {
        builder.append_option(value.as_deref());
    }
    Ok(Arc::new(builder.finish()))
}

/// Convert a single Arrow array to an owned cudf `Column`.
fn arrow_array_to_column(runtime: &Runtime, arr: &dyn Array, field_name: &str) -> Result<Column> {
    let dt = arr.data_type();
    match dt {
        ArrowDataType::Int8 => primitive_to_column::<i8>(runtime, arr),
        ArrowDataType::Int16 => primitive_to_column::<i16>(runtime, arr),
        ArrowDataType::Int32 => primitive_to_column::<i32>(runtime, arr),
        ArrowDataType::Int64 => primitive_to_column::<i64>(runtime, arr),
        ArrowDataType::UInt8 => primitive_to_column::<u8>(runtime, arr),
        ArrowDataType::UInt16 => primitive_to_column::<u16>(runtime, arr),
        ArrowDataType::UInt32 => primitive_to_column::<u32>(runtime, arr),
        ArrowDataType::UInt64 => primitive_to_column::<u64>(runtime, arr),
        ArrowDataType::Float32 => primitive_to_column::<f32>(runtime, arr),
        ArrowDataType::Float64 => primitive_to_column::<f64>(runtime, arr),
        ArrowDataType::Boolean => bool_to_column(runtime, arr),
        ArrowDataType::Date32 => primitive_arrow_to_fixed_width_column::<arrow::datatypes::Date32Type>(
            runtime,
            arr,
            DataType::new(TypeId::TimestampDays),
        ),
        ArrowDataType::Date64 => primitive_arrow_to_fixed_width_column::<arrow::datatypes::Date64Type>(
            runtime,
            arr,
            DataType::new(TypeId::TimestampMilliseconds),
        ),
        ArrowDataType::Time32(TimeUnit::Second) => {
            time32_arrow_to_duration_column::<arrow::datatypes::Time32SecondType>(
                runtime,
                arr,
                DataType::new(TypeId::DurationSeconds),
            )
        },
        ArrowDataType::Time32(TimeUnit::Millisecond) => time32_arrow_to_duration_column::<
            arrow::datatypes::Time32MillisecondType,
        >(
            runtime, arr, DataType::new(TypeId::DurationMilliseconds)
        ),
        ArrowDataType::Time64(TimeUnit::Microsecond) => primitive_arrow_to_fixed_width_column::<
            arrow::datatypes::Time64MicrosecondType,
        >(
            runtime, arr, DataType::new(TypeId::DurationMicroseconds)
        ),
        ArrowDataType::Time64(TimeUnit::Nanosecond) => primitive_arrow_to_fixed_width_column::<
            arrow::datatypes::Time64NanosecondType,
        >(
            runtime, arr, DataType::new(TypeId::DurationNanoseconds)
        ),
        ArrowDataType::Timestamp(unit, _) => timestamp_arrow_to_column(runtime, arr, unit),
        ArrowDataType::Duration(TimeUnit::Second) => primitive_arrow_to_fixed_width_column::<
            arrow::datatypes::DurationSecondType,
        >(runtime, arr, DataType::new(TypeId::DurationSeconds)),
        ArrowDataType::Duration(TimeUnit::Millisecond) => primitive_arrow_to_fixed_width_column::<
            arrow::datatypes::DurationMillisecondType,
        >(
            runtime, arr, DataType::new(TypeId::DurationMilliseconds)
        ),
        ArrowDataType::Duration(TimeUnit::Microsecond) => primitive_arrow_to_fixed_width_column::<
            arrow::datatypes::DurationMicrosecondType,
        >(
            runtime, arr, DataType::new(TypeId::DurationMicroseconds)
        ),
        ArrowDataType::Duration(TimeUnit::Nanosecond) => primitive_arrow_to_fixed_width_column::<
            arrow::datatypes::DurationNanosecondType,
        >(
            runtime, arr, DataType::new(TypeId::DurationNanoseconds)
        ),
        ArrowDataType::Decimal128(_, scale) => {
            let dtype = DataType::fixed_point(TypeId::Decimal128, arrow_decimal_scale_to_cudf_scale(*scale));
            decimal128_array_to_column(runtime, arr, dtype)
        },
        ArrowDataType::List(field) if list_child_is_supported(field) => {
            let list = arr.as_any().downcast_ref::<ListArray>().ok_or_else(|| {
                unexpected_arrow_array_layout_error(
                    "arrow::list_array_to_column",
                    "expected ListArray for List type",
                    dt,
                    arr.data_type(),
                )
            })?;
            list_array_to_column(runtime, field, list.values().as_ref(), list.value_offsets(), list.nulls(), field_name)
        },
        ArrowDataType::LargeList(field) if list_child_is_supported(field) => {
            let list = arr.as_any().downcast_ref::<LargeListArray>().ok_or_else(|| {
                unexpected_arrow_array_layout_error(
                    "arrow::large_list_array_to_column",
                    "expected LargeListArray for LargeList type",
                    dt,
                    arr.data_type(),
                )
            })?;
            list_array_to_column(runtime, field, list.values().as_ref(), list.value_offsets(), list.nulls(), field_name)
        },
        ArrowDataType::FixedSizeList(field, dimension) if *dimension > 0 && list_numeric_child_is_supported(field) => {
            let list = arr.as_any().downcast_ref::<FixedSizeListArray>().ok_or_else(|| {
                unexpected_arrow_array_layout_error(
                    "arrow::fixed_size_list_array_to_column",
                    "expected FixedSizeListArray for FixedSizeList type",
                    dt,
                    arr.data_type(),
                )
            })?;
            fixed_size_list_array_to_column(runtime, field, list, field_name)
        },
        ArrowDataType::Utf8 => string_array_to_column(runtime, arr),
        ArrowDataType::LargeUtf8 => large_string_array_to_column(runtime, arr),
        ArrowDataType::Utf8View => string_view_array_to_column(runtime, arr),
        _ => Err(Error::new_with(
            ErrorCode::UnsupportedDtype,
            "arrow::arrow_array_to_column",
            format!("unsupported Arrow type for column '{}': {:?}", field_name, dt),
        )
        .with_detail("field_name", field_name.to_string())
        .with_detail("arrow_type", format!("{:?}", dt))),
    }
}

fn timestamp_arrow_to_column(runtime: &Runtime, arr: &dyn Array, unit: &TimeUnit) -> Result<Column> {
    match unit {
        TimeUnit::Second => primitive_arrow_to_fixed_width_column::<arrow::datatypes::TimestampSecondType>(
            runtime,
            arr,
            DataType::new(TypeId::TimestampSeconds),
        ),
        TimeUnit::Millisecond => primitive_arrow_to_fixed_width_column::<arrow::datatypes::TimestampMillisecondType>(
            runtime,
            arr,
            DataType::new(TypeId::TimestampMilliseconds),
        ),
        TimeUnit::Microsecond => primitive_arrow_to_fixed_width_column::<arrow::datatypes::TimestampMicrosecondType>(
            runtime,
            arr,
            DataType::new(TypeId::TimestampMicroseconds),
        ),
        TimeUnit::Nanosecond => primitive_arrow_to_fixed_width_column::<arrow::datatypes::TimestampNanosecondType>(
            runtime,
            arr,
            DataType::new(TypeId::TimestampNanoseconds),
        ),
    }
}

fn list_array_to_column<O>(
    runtime: &Runtime,
    child_field: &FieldRef,
    values: &dyn Array,
    offsets: &[O],
    nulls: Option<&NullBuffer>,
    field_name: &str,
) -> Result<Column>
where
    O: Copy + Into<i128>,
{
    let operation = "arrow::list_array_to_column";
    let (normalized_offsets, child_start, child_len) = normalize_list_offsets(offsets, operation, field_name)?;
    let child_values = values.slice(child_start, child_len);
    if list_fixed_width_child_is_supported(child_field) {
        return fixed_width_list_array_to_column(
            runtime,
            child_field,
            child_values.as_ref(),
            normalized_offsets,
            nulls,
            field_name,
            operation,
        );
    }

    let (child_offsets, child_data, child_validity) =
        string_like_array_host_buffers(child_values.as_ref(), child_field.data_type(), operation)?;
    let list_offsets = encode_host_i32_values(&normalized_offsets);

    let root_validity = physical_validity_from_nulls(nulls, BufferId::new(1));
    let child_validity = physical_validity_from_optional_mask(child_validity, BufferId::new(4));

    let mut bindings = Vec::new();
    let mut retained_buffers = Vec::new();
    push_host_binding(operation, &mut bindings, &mut retained_buffers, BufferId::new(0), list_offsets)?;
    if let Some((buffer, _null_count)) = root_validity.buffer.as_ref() {
        push_host_binding(operation, &mut bindings, &mut retained_buffers, BufferId::new(1), Arc::clone(buffer))?;
    }
    push_host_binding(operation, &mut bindings, &mut retained_buffers, BufferId::new(2), child_offsets)?;
    push_host_binding(operation, &mut bindings, &mut retained_buffers, BufferId::new(3), child_data)?;
    if let Some((buffer, _null_count)) = child_validity.buffer.as_ref() {
        push_host_binding(operation, &mut bindings, &mut retained_buffers, BufferId::new(4), Arc::clone(buffer))?;
    }

    let list_column = SemanticColumn::new(
        ColumnId::new(0),
        InteropDataType::new(InteropTypeId::List),
        normalized_offsets.len().saturating_sub(1),
        0,
        root_validity.validity,
        if nulls.is_some() { SchemaNullability::Nullable } else { SchemaNullability::NonNullable },
        ColumnLayout::List { offsets: BufferId::new(0), child: ColumnId::new(1) },
    );
    let child_column = SemanticColumn::new(
        ColumnId::new(1),
        InteropDataType::new(InteropTypeId::String),
        child_len,
        0,
        child_validity.validity,
        if child_field.is_nullable() { SchemaNullability::Nullable } else { SchemaNullability::NonNullable },
        ColumnLayout::BinaryLike { offsets: BufferId::new(2), data: BufferId::new(3) },
    );
    let export = PreparedColumnExport::try_new_with_semantic_columns(
        list_column.clone(),
        vec![list_column, child_column],
        bindings,
        ProducerReadiness::HostVisible,
        super::host_export_diagnostics(),
    )
    .map_err(|error| error::from_interop_in(error, operation))?
    .with_retained_host_buffers(retained_buffers);

    Column::try_import_from_prepared_column_export_into_runtime(&export, runtime)
        .map_err(|error| error::from_interop_in(error, operation))
}

fn fixed_size_list_array_to_column(
    runtime: &Runtime,
    child_field: &FieldRef,
    list: &FixedSizeListArray,
    field_name: &str,
) -> Result<Column> {
    let operation = "arrow::fixed_size_list_array_to_column";
    let dimension = usize::try_from(list.value_length()).map_err(|_| {
        Error::new_with(ErrorCode::ValueOutOfRange, operation, "Arrow FixedSizeList dimension must be positive")
            .with_detail("field_name", field_name.to_string())
            .with_detail("dimension", list.value_length().to_string())
            .with_detail("reason", "non_positive_fixed_size_list_dimension")
    })?;
    if dimension == 0 {
        return Err(Error::new_with(
            ErrorCode::ValueOutOfRange,
            operation,
            "Arrow FixedSizeList dimension must be positive",
        )
        .with_detail("field_name", field_name.to_string())
        .with_detail("dimension", "0")
        .with_detail("reason", "non_positive_fixed_size_list_dimension"));
    }

    let child_len = list.len().checked_mul(dimension).ok_or_else(|| {
        Error::new_with(ErrorCode::SizeOverflow, operation, "Arrow FixedSizeList child length overflowed usize")
            .with_detail("field_name", field_name.to_string())
            .with_detail("row_count", list.len().to_string())
            .with_detail("dimension", dimension.to_string())
            .with_detail("reason", "fixed_size_list_child_len_overflow")
    })?;
    if list.values().len() != child_len {
        return Err(error::foreign_invariant_violation(
            operation,
            "Arrow FixedSizeList values length does not match row count times dimension",
            "arrow_array",
            "fixed_size_list_child_length_mismatch",
            "ensure the FixedSizeList array has a valid child values layout",
        )
        .with_detail("field_name", field_name.to_string())
        .with_detail("child_len", list.values().len().to_string())
        .with_detail("expected_child_len", child_len.to_string()));
    }

    let offset_count = list.len().checked_add(1).ok_or_else(|| {
        Error::new_with(ErrorCode::SizeOverflow, operation, "Arrow FixedSizeList offset count overflowed usize")
            .with_detail("field_name", field_name.to_string())
            .with_detail("row_count", list.len().to_string())
            .with_detail("reason", "fixed_size_list_offset_count_overflow")
    })?;
    let mut normalized_offsets = Vec::with_capacity(offset_count);
    for row in 0..=list.len() {
        let offset = row.checked_mul(dimension).ok_or_else(|| {
            Error::new_with(ErrorCode::SizeOverflow, operation, "Arrow FixedSizeList offset overflowed usize")
                .with_detail("field_name", field_name.to_string())
                .with_detail("row_index", row.to_string())
                .with_detail("dimension", dimension.to_string())
                .with_detail("reason", "fixed_size_list_offset_overflow")
        })?;
        normalized_offsets.push(i32::try_from(offset).map_err(|_| {
            Error::new_with(
                ErrorCode::ValueOutOfRange,
                operation,
                "Arrow FixedSizeList offsets exceed cuDF's Int32 list offset range",
            )
            .with_detail("field_name", field_name.to_string())
            .with_detail("row_index", row.to_string())
            .with_detail("offset_value", offset.to_string())
            .with_detail("reason", "fixed_size_list_offset_i32_overflow")
        })?);
    }

    fixed_width_list_array_to_column(
        runtime,
        child_field,
        list.values().as_ref(),
        normalized_offsets,
        list.nulls(),
        field_name,
        operation,
    )
}

fn fixed_width_list_array_to_column(
    runtime: &Runtime,
    child_field: &FieldRef,
    values: &dyn Array,
    normalized_offsets: Vec<i32>,
    nulls: Option<&NullBuffer>,
    field_name: &str,
    operation: &'static str,
) -> Result<Column> {
    let child_dtype = list_fixed_width_interop_dtype(child_field.data_type()).ok_or_else(|| {
        Error::new_with(ErrorCode::UnsupportedDtype, operation, "Arrow list child must be a supported fixed-width type")
            .with_detail("field_name", field_name.to_string())
            .with_detail("arrow_type", format!("{:?}", child_field.data_type()))
            .with_detail("reason", "unsupported_fixed_width_list_child_type")
    })?;
    let (child_data, child_validity_mask) =
        fixed_width_list_array_host_buffers(values, child_field.data_type(), operation)?;
    let child_len = values.len();
    let list_offsets = encode_host_i32_values(&normalized_offsets);
    let root_validity = physical_validity_from_nulls(nulls, BufferId::new(1));
    let child_validity = physical_validity_from_optional_mask(child_validity_mask, BufferId::new(3));

    let mut bindings = Vec::new();
    let mut retained_buffers = Vec::new();
    push_host_binding(operation, &mut bindings, &mut retained_buffers, BufferId::new(0), list_offsets)?;
    if let Some((buffer, _null_count)) = root_validity.buffer.as_ref() {
        push_host_binding(operation, &mut bindings, &mut retained_buffers, BufferId::new(1), Arc::clone(buffer))?;
    }
    push_host_binding(operation, &mut bindings, &mut retained_buffers, BufferId::new(2), child_data)?;
    if let Some((buffer, _null_count)) = child_validity.buffer.as_ref() {
        push_host_binding(operation, &mut bindings, &mut retained_buffers, BufferId::new(3), Arc::clone(buffer))?;
    }

    let list_column = SemanticColumn::new(
        ColumnId::new(0),
        InteropDataType::new(InteropTypeId::List),
        normalized_offsets.len().saturating_sub(1),
        0,
        root_validity.validity,
        if nulls.is_some() { SchemaNullability::Nullable } else { SchemaNullability::NonNullable },
        ColumnLayout::List { offsets: BufferId::new(0), child: ColumnId::new(1) },
    );
    let child_column = SemanticColumn::new(
        ColumnId::new(1),
        child_dtype,
        child_len,
        0,
        child_validity.validity,
        if child_field.is_nullable() { SchemaNullability::Nullable } else { SchemaNullability::NonNullable },
        ColumnLayout::FixedWidth { values: BufferId::new(2) },
    );
    let export = PreparedColumnExport::try_new_with_semantic_columns(
        list_column.clone(),
        vec![list_column, child_column],
        bindings,
        ProducerReadiness::HostVisible,
        super::host_export_diagnostics(),
    )
    .map_err(|error| error::from_interop_in(error, operation))?
    .with_retained_host_buffers(retained_buffers);

    Column::try_import_from_prepared_column_export_into_runtime(&export, runtime)
        .map_err(|error| error::from_interop_in(error, operation))
}

fn normalize_list_offsets<O>(
    offsets: &[O],
    operation: &'static str,
    field_name: &str,
) -> Result<(Vec<i32>, usize, usize)>
where
    O: Copy + Into<i128>,
{
    if offsets.is_empty() {
        return Err(Error::new_with(ErrorCode::EmptyInput, operation, "Arrow list offsets must not be empty")
            .with_detail("field_name", field_name.to_string())
            .with_detail("reason", "empty_list_offsets"));
    }
    let base = offsets[0].into();
    if base < 0 {
        return Err(Error::new_with(ErrorCode::ValueOutOfRange, operation, "Arrow list offsets must be non-negative")
            .with_detail("field_name", field_name.to_string())
            .with_detail("offset_index", "0")
            .with_detail("offset_value", base.to_string())
            .with_detail("reason", "negative_list_offset"));
    }
    let child_start = usize::try_from(base).map_err(|_| {
        Error::new_with(ErrorCode::ValueOutOfRange, operation, "Arrow list child offset does not fit usize")
            .with_detail("field_name", field_name.to_string())
            .with_detail("offset_value", base.to_string())
            .with_detail("reason", "list_offset_usize_overflow")
    })?;

    let mut normalized = Vec::with_capacity(offsets.len());
    let mut previous = base;
    for (index, raw) in offsets.iter().copied().map(Into::into).enumerate() {
        if raw < previous {
            return Err(Error::new_with(
                ErrorCode::ValueOutOfRange,
                operation,
                "Arrow list offsets must be monotonically non-decreasing",
            )
            .with_detail("field_name", field_name.to_string())
            .with_detail("offset_index", index.to_string())
            .with_detail("previous_offset", previous.to_string())
            .with_detail("offset_value", raw.to_string())
            .with_detail("reason", "non_monotonic_list_offsets"));
        }
        let value = raw - base;
        normalized.push(i32::try_from(value).map_err(|_| {
            Error::new_with(
                ErrorCode::ValueOutOfRange,
                operation,
                "Arrow list offsets exceed cuDF's Int32 list offset range",
            )
            .with_detail("field_name", field_name.to_string())
            .with_detail("offset_index", index.to_string())
            .with_detail("offset_value", raw.to_string())
            .with_detail("reason", "list_offset_i32_overflow")
        })?);
        previous = raw;
    }
    let child_len = usize::try_from(previous - base).map_err(|_| {
        Error::new_with(ErrorCode::ValueOutOfRange, operation, "Arrow list child length does not fit usize")
            .with_detail("field_name", field_name.to_string())
            .with_detail("reason", "list_child_len_usize_overflow")
    })?;
    Ok((normalized, child_start, child_len))
}

struct PreparedValidity {
    validity: PhysicalValidity,
    buffer: Option<(Arc<[u8]>, usize)>,
}

fn physical_validity_from_nulls(nulls: Option<&NullBuffer>, id: BufferId) -> PreparedValidity {
    let Some(nulls) = nulls else {
        return PreparedValidity { validity: PhysicalValidity::NoValidityBuffer, buffer: None };
    };
    physical_validity_from_optional_mask(Some(nulls.iter().collect()), id)
}

fn physical_validity_from_optional_mask(mask: Option<Vec<bool>>, id: BufferId) -> PreparedValidity {
    let Some(mask) = mask else {
        return PreparedValidity { validity: PhysicalValidity::NoValidityBuffer, buffer: None };
    };
    let nulls = NullBuffer::from(mask);
    if nulls.null_count() == 0 {
        return PreparedValidity { validity: PhysicalValidity::NoValidityBuffer, buffer: None };
    }
    let buffer: Arc<[u8]> = Arc::from(nulls.validity().to_vec());
    PreparedValidity {
        validity: PhysicalValidity::ValidityMask { mask: id, null_count: Some(nulls.null_count()) },
        buffer: Some((buffer, nulls.null_count())),
    }
}

fn push_host_binding(
    operation: &'static str,
    bindings: &mut Vec<BufferBinding<'static>>,
    retained_buffers: &mut Vec<Arc<[u8]>>,
    id: BufferId,
    buffer: Arc<[u8]>,
) -> Result<()> {
    let binding = BufferBinding::new(
        id,
        BoundBytes::from_raw_parts(buffer.as_ptr() as *mut c_void, buffer.len())
            .map_err(|error| error::from_interop_in(error, operation))?,
        Placement::Host,
    );
    bindings.push(binding);
    retained_buffers.push(buffer);
    Ok(())
}

fn numeric_interop_dtype(data_type: &ArrowDataType) -> Option<InteropDataType> {
    let type_id = match data_type {
        ArrowDataType::Int8 => InteropTypeId::Int8,
        ArrowDataType::Int16 => InteropTypeId::Int16,
        ArrowDataType::Int32 => InteropTypeId::Int32,
        ArrowDataType::Int64 => InteropTypeId::Int64,
        ArrowDataType::UInt8 => InteropTypeId::UInt8,
        ArrowDataType::UInt16 => InteropTypeId::UInt16,
        ArrowDataType::UInt32 => InteropTypeId::UInt32,
        ArrowDataType::UInt64 => InteropTypeId::UInt64,
        ArrowDataType::Float32 => InteropTypeId::Float32,
        ArrowDataType::Float64 => InteropTypeId::Float64,
        _ => return None,
    };
    Some(InteropDataType::new(type_id))
}

fn list_fixed_width_interop_dtype(data_type: &ArrowDataType) -> Option<InteropDataType> {
    match data_type {
        ArrowDataType::Boolean => Some(InteropDataType::new(InteropTypeId::Bool8)),
        data_type => numeric_interop_dtype(data_type),
    }
}

type FixedWidthListHostBuffers = (Arc<[u8]>, Option<Vec<bool>>);

fn fixed_width_list_array_host_buffers(
    arr: &dyn Array,
    expected_type: &ArrowDataType,
    operation: &'static str,
) -> Result<FixedWidthListHostBuffers> {
    if arr.data_type() != expected_type {
        return Err(Error::new_with(
            ErrorCode::TypeMismatch,
            operation,
            "Arrow fixed-width list child array type must match its child field type",
        )
        .with_detail("expected_arrow_type", format!("{expected_type:?}"))
        .with_detail("actual_arrow_type", format!("{:?}", arr.data_type()))
        .with_detail("reason", "list_child_type_mismatch"));
    }

    macro_rules! primitive_bytes {
        ($arrow_type:ty) => {{
            let values = arr.as_any().downcast_ref::<PrimitiveArray<$arrow_type>>().ok_or_else(|| {
                unexpected_arrow_array_layout_error(
                    operation,
                    "numeric list child has an unexpected Arrow primitive layout",
                    expected_type,
                    arr.data_type(),
                )
            })?;
            Arc::<[u8]>::from(native_slice_as_bytes(values.values().as_ref()))
        }};
    }

    let data = match expected_type {
        ArrowDataType::Boolean => {
            let values = arr.as_any().downcast_ref::<BooleanArray>().ok_or_else(|| {
                unexpected_arrow_array_layout_error(
                    operation,
                    "boolean list child has an unexpected Arrow layout",
                    expected_type,
                    arr.data_type(),
                )
            })?;
            Arc::<[u8]>::from(values.values().iter().map(u8::from).collect::<Vec<_>>())
        },
        ArrowDataType::Int8 => primitive_bytes!(arrow::datatypes::Int8Type),
        ArrowDataType::Int16 => primitive_bytes!(arrow::datatypes::Int16Type),
        ArrowDataType::Int32 => primitive_bytes!(arrow::datatypes::Int32Type),
        ArrowDataType::Int64 => primitive_bytes!(arrow::datatypes::Int64Type),
        ArrowDataType::UInt8 => primitive_bytes!(arrow::datatypes::UInt8Type),
        ArrowDataType::UInt16 => primitive_bytes!(arrow::datatypes::UInt16Type),
        ArrowDataType::UInt32 => primitive_bytes!(arrow::datatypes::UInt32Type),
        ArrowDataType::UInt64 => primitive_bytes!(arrow::datatypes::UInt64Type),
        ArrowDataType::Float32 => primitive_bytes!(arrow::datatypes::Float32Type),
        ArrowDataType::Float64 => primitive_bytes!(arrow::datatypes::Float64Type),
        _ => {
            return Err(Error::new_with(
                ErrorCode::UnsupportedDtype,
                operation,
                "Arrow list host import supports only fixed-width children on this path",
            )
            .with_detail("arrow_type", format!("{expected_type:?}"))
            .with_detail("reason", "unsupported_fixed_width_list_child_type"));
        },
    };
    let validity = (arr.null_count() > 0).then(|| (0..arr.len()).map(|index| arr.is_valid(index)).collect());
    Ok((data, validity))
}

/// Host-side buffers for a string-like array: `(offsets, data, validity)`.
type StringLikeHostBuffers = (Arc<[u8]>, Arc<[u8]>, Option<Vec<bool>>);

fn string_like_array_host_buffers(
    arr: &dyn Array,
    expected_type: &ArrowDataType,
    operation: &'static str,
) -> Result<StringLikeHostBuffers> {
    if arr.data_type() != expected_type {
        return Err(Error::new_with(
            ErrorCode::TypeMismatch,
            operation,
            "Arrow list child array type must match its child field type",
        )
        .with_detail("expected_arrow_type", format!("{expected_type:?}"))
        .with_detail("actual_arrow_type", format!("{:?}", arr.data_type()))
        .with_detail("reason", "list_child_type_mismatch"));
    }
    let strings = string_like_values(arr, expected_type, operation)?;
    let validity = (arr.null_count() > 0).then(|| (0..arr.len()).map(|index| arr.is_valid(index)).collect::<Vec<_>>());
    let offsets = encode_host_string_offsets(operation, &strings)?;
    let data: Arc<[u8]> =
        Arc::from(strings.iter().flat_map(|value| value.as_bytes().iter().copied()).collect::<Vec<_>>());
    Ok((offsets, data, validity))
}

fn string_like_values(arr: &dyn Array, expected_type: &ArrowDataType, operation: &'static str) -> Result<Vec<String>> {
    match expected_type {
        ArrowDataType::Utf8 => {
            let strings = arr.as_any().downcast_ref::<StringArray>().ok_or_else(|| {
                unexpected_arrow_array_layout_error(
                    operation,
                    "expected StringArray for List<Utf8> child",
                    expected_type,
                    arr.data_type(),
                )
            })?;
            Ok((0..strings.len())
                .map(|index| if strings.is_valid(index) { strings.value(index).to_string() } else { String::new() })
                .collect())
        },
        ArrowDataType::LargeUtf8 => {
            let strings = arr.as_any().downcast_ref::<LargeStringArray>().ok_or_else(|| {
                unexpected_arrow_array_layout_error(
                    operation,
                    "expected LargeStringArray for List<LargeUtf8> child",
                    expected_type,
                    arr.data_type(),
                )
            })?;
            Ok((0..strings.len())
                .map(|index| if strings.is_valid(index) { strings.value(index).to_string() } else { String::new() })
                .collect())
        },
        ArrowDataType::Utf8View => {
            let strings = arr.as_string_view_opt().ok_or_else(|| {
                unexpected_arrow_array_layout_error(
                    operation,
                    "expected StringViewArray for List<Utf8View> child",
                    expected_type,
                    arr.data_type(),
                )
            })?;
            Ok((0..strings.len())
                .map(|index| if strings.is_valid(index) { strings.value(index).to_string() } else { String::new() })
                .collect())
        },
        _ => Err(Error::new_with(
            ErrorCode::UnsupportedDtype,
            operation,
            "Arrow list host import currently supports only string-like children",
        )
        .with_detail("arrow_type", format!("{expected_type:?}"))
        .with_detail("reason", "unsupported_list_child_type")),
    }
}

fn encode_host_i32_values(values: &[i32]) -> Arc<[u8]> {
    Arc::from(values.iter().flat_map(|value| value.to_ne_bytes()).collect::<Vec<_>>())
}

fn encode_host_string_offsets(operation: &'static str, strings: &[String]) -> Result<Arc<[u8]>> {
    let mut offsets = Vec::with_capacity(strings.len() + 1);
    let mut total_bytes = 0usize;
    offsets.push(0i32);

    for value in strings {
        total_bytes = total_bytes.checked_add(value.len()).ok_or_else(|| {
            Error::new_with(ErrorCode::ValueOutOfRange, operation, "string byte range overflowed usize")
                .with_detail("reason", "string_byte_len_overflow")
        })?;
        offsets.push(i32::try_from(total_bytes).map_err(|_| {
            Error::new_with(ErrorCode::ValueOutOfRange, operation, "string byte offsets exceed Arrow Utf8 range")
                .with_detail("byte_len", total_bytes.to_string())
                .with_detail("reason", "string_offset_i32_overflow")
        })?);
    }

    Ok(encode_host_i32_values(&offsets))
}

fn unexpected_arrow_array_layout_error(
    operation: &'static str,
    message: &'static str,
    arrow_type: &ArrowDataType,
    actual_arrow_type: &ArrowDataType,
) -> Error {
    error::foreign_invariant_violation(
        operation,
        message,
        "arrow_array",
        "unexpected_arrow_array_layout",
        "ensure Arrow arrays match their declared field layouts",
    )
    .with_detail("arrow_type", format!("{arrow_type:?}"))
    .with_detail("actual_arrow_type", format!("{actual_arrow_type:?}"))
}

/// Convert a primitive Arrow array to a cudf Column.
fn primitive_to_column<T>(runtime: &Runtime, arr: &dyn Array) -> Result<Column>
where T: ArrowPrimitiveMapping {
    let prim = arr.as_primitive::<T::ArrowPrimitive>();
    let values = prim.values().as_ref();

    match prim.nulls() {
        None => runtime.column_from_slice(values),
        Some(null_buf) => {
            let validity: Vec<bool> = null_buf.iter().collect();
            runtime.column_from_slice_nullable(values, &validity)
        },
    }
}

fn primitive_arrow_to_fixed_width_column<A>(runtime: &Runtime, arr: &dyn Array, dtype: DataType) -> Result<Column>
where A: arrow::datatypes::ArrowPrimitiveType {
    let prim = arr.as_primitive::<A>();
    let values = prim.values().as_ref();
    let bytes = native_slice_as_bytes(values);
    let validity = prim.nulls().map(|null_buf| null_buf.iter().collect::<Vec<_>>());
    Column::from_fixed_width_host_bytes_with_runtime(runtime, dtype, bytes, validity.as_deref())
        .map_err(|error| error.reanchor("arrow::arrow_array_to_column"))
}

fn time32_arrow_to_duration_column<A>(runtime: &Runtime, arr: &dyn Array, dtype: DataType) -> Result<Column>
where A: arrow::datatypes::ArrowPrimitiveType<Native = i32> {
    let prim = arr.as_primitive::<A>();
    let values = prim.values().iter().map(|value| i64::from(*value)).collect::<Vec<_>>();
    let bytes = native_slice_as_bytes(&values);
    let validity = prim.nulls().map(|null_buf| null_buf.iter().collect::<Vec<_>>());
    Column::from_fixed_width_host_bytes_with_runtime(runtime, dtype, bytes, validity.as_deref())
        .map_err(|error| error.reanchor("arrow::arrow_array_to_column"))
}

fn decimal128_array_to_column(runtime: &Runtime, arr: &dyn Array, dtype: DataType) -> Result<Column> {
    let decimal = arr.as_primitive::<arrow::datatypes::Decimal128Type>();
    let values = decimal.values().as_ref();
    let bytes = native_slice_as_bytes(values);
    let validity = decimal.nulls().map(|null_buf| null_buf.iter().collect::<Vec<_>>());
    Column::from_fixed_width_host_bytes_with_runtime(runtime, dtype, bytes, validity.as_deref())
        .map_err(|error| error.reanchor("arrow::arrow_array_to_column"))
}

/// Convert a boolean Arrow array to a cudf Column (Bool8).
fn bool_to_column(runtime: &Runtime, arr: &dyn Array) -> Result<Column> {
    let bool_arr = arr.as_boolean();
    // Arrow booleans are bit-packed, while cudf Bool8 is byte-per-value.
    // This path still needs an explicit host-side unpacking step before the
    // required host-to-device copy.
    let values: Vec<bool> = bool_arr.values().iter().collect();

    match bool_arr.nulls() {
        None => runtime.column_from_slice(&values),
        Some(null_buf) => {
            let validity: Vec<bool> = null_buf.iter().collect();
            runtime.column_from_slice_nullable(&values, &validity)
        },
    }
}

fn string_like_arrow_host_column_to_column(
    runtime: &Runtime,
    arr: &dyn Array,
    operation: &'static str,
) -> Result<Column> {
    let data = arr.to_data();
    let ffi_array = FFI_ArrowArray::new(&data);
    let ffi_schema =
        FFI_ArrowSchema::try_from(data.data_type()).map_err(|error| map_arrow_ffi_export_error(operation, error))?;
    let exec_options = runtime.execution_domain().enter_execution()?;
    let mut handle = ptr::null_mut();

    error::check_status_in(
        unsafe {
            cudf_sys::cudf_arrow_host_column_import_ex(
                (&ffi_schema as *const FFI_ArrowSchema).cast::<c_void>(),
                (&ffi_array as *const FFI_ArrowArray).cast::<c_void>(),
                exec_options.options(),
                &mut handle,
            )
        },
        operation,
    )?;

    unsafe { Column::from_raw_with_domain(handle, Arc::clone(runtime.execution_domain())) }
        .map_err(|error| error.reanchor(operation))
}

/// Convert an Arrow StringArray (Utf8) to a cudf STRING Column.
fn string_array_to_column(runtime: &Runtime, arr: &dyn Array) -> Result<Column> {
    let _str_arr = arr.as_any().downcast_ref::<StringArray>().ok_or_else(|| {
        error::foreign_invariant_violation(
            "arrow::string_array_to_column",
            "expected StringArray for Utf8 type",
            "arrow_array",
            "unexpected_arrow_array_layout",
            "ensure Utf8 fields are backed by Arrow StringArray values",
        )
        .with_detail("arrow_type", format!("{:?}", ArrowDataType::Utf8))
        .with_detail("actual_arrow_type", format!("{:?}", arr.data_type()))
    })?;

    string_like_arrow_host_column_to_column(runtime, arr, "arrow::string_array_to_column")
}

/// Convert an Arrow LargeStringArray (LargeUtf8) to a cudf STRING Column.
fn large_string_array_to_column(runtime: &Runtime, arr: &dyn Array) -> Result<Column> {
    let _str_arr = arr.as_any().downcast_ref::<LargeStringArray>().ok_or_else(|| {
        error::foreign_invariant_violation(
            "arrow::large_string_array_to_column",
            "expected LargeStringArray for LargeUtf8 type",
            "arrow_array",
            "unexpected_arrow_array_layout",
            "ensure LargeUtf8 fields are backed by Arrow LargeStringArray values",
        )
        .with_detail("arrow_type", format!("{:?}", ArrowDataType::LargeUtf8))
        .with_detail("actual_arrow_type", format!("{:?}", arr.data_type()))
    })?;

    string_like_arrow_host_column_to_column(runtime, arr, "arrow::large_string_array_to_column")
}

/// Convert an Arrow StringViewArray (Utf8View) to a cudf STRING Column.
fn string_view_array_to_column(runtime: &Runtime, arr: &dyn Array) -> Result<Column> {
    let _str_arr = arr.as_string_view_opt().ok_or_else(|| {
        error::foreign_invariant_violation(
            "arrow::string_view_array_to_column",
            "expected StringViewArray for Utf8View type",
            "arrow_array",
            "unexpected_arrow_array_layout",
            "ensure Utf8View fields are backed by Arrow StringViewArray values",
        )
        .with_detail("arrow_type", format!("{:?}", ArrowDataType::Utf8View))
        .with_detail("actual_arrow_type", format!("{:?}", arr.data_type()))
    })?;

    string_like_arrow_host_column_to_column(runtime, arr, "arrow::string_view_array_to_column")
}

/// Trait to associate a `CudfType` with its Arrow primitive type.
///
/// This avoids needing separate functions for each numeric type.
trait ArrowPrimitiveMapping: crate::core::types::CudfType + arrow::datatypes::ArrowNativeType {
    type ArrowPrimitive: arrow::datatypes::ArrowPrimitiveType<Native = Self>;
}

impl ArrowPrimitiveMapping for i8 {
    type ArrowPrimitive = arrow::datatypes::Int8Type;
}
impl ArrowPrimitiveMapping for i16 {
    type ArrowPrimitive = arrow::datatypes::Int16Type;
}
impl ArrowPrimitiveMapping for i32 {
    type ArrowPrimitive = arrow::datatypes::Int32Type;
}
impl ArrowPrimitiveMapping for i64 {
    type ArrowPrimitive = arrow::datatypes::Int64Type;
}
impl ArrowPrimitiveMapping for u8 {
    type ArrowPrimitive = arrow::datatypes::UInt8Type;
}
impl ArrowPrimitiveMapping for u16 {
    type ArrowPrimitive = arrow::datatypes::UInt16Type;
}
impl ArrowPrimitiveMapping for u32 {
    type ArrowPrimitive = arrow::datatypes::UInt32Type;
}
impl ArrowPrimitiveMapping for u64 {
    type ArrowPrimitive = arrow::datatypes::UInt64Type;
}
impl ArrowPrimitiveMapping for f32 {
    type ArrowPrimitive = arrow::datatypes::Float32Type;
}
impl ArrowPrimitiveMapping for f64 {
    type ArrowPrimitive = arrow::datatypes::Float64Type;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_offset_normalization_rejects_negative_non_monotonic_and_i32_overflow() {
        let negative = normalize_list_offsets(&[-1_i64, 0], "test::list_offsets", "values").unwrap_err();
        assert_eq!(negative.code(), ErrorCode::ValueOutOfRange);
        assert_eq!(negative.detail_value("reason"), Some("negative_list_offset"));

        let non_monotonic = normalize_list_offsets(&[0_i64, 2, 1], "test::list_offsets", "values").unwrap_err();
        assert_eq!(non_monotonic.code(), ErrorCode::ValueOutOfRange);
        assert_eq!(non_monotonic.detail_value("reason"), Some("non_monotonic_list_offsets"));

        let outside_cudf_range =
            normalize_list_offsets(&[0_i64, i64::from(i32::MAX) + 1], "test::list_offsets", "values").unwrap_err();
        assert_eq!(outside_cudf_range.code(), ErrorCode::ValueOutOfRange);
        assert_eq!(outside_cudf_range.status(), crate::core::error::ErrorStatus::Permanent);
        assert_eq!(outside_cudf_range.detail_value("reason"), Some("list_offset_i32_overflow"));
    }
}
