//! cudf column to Arrow array host-copy conversion.

use std::sync::Arc;

use arrow::{
    array::{Array, BooleanArray, Decimal128Array, ListArray, PrimitiveArray, StringBuilder},
    buffer::{BooleanBuffer, Buffer, NullBuffer, OffsetBuffer, ScalarBuffer},
    datatypes::{
        ArrowNativeType, ArrowPrimitiveType, DataType as ArrowDataType, Date32Type, Decimal32Type, Decimal64Type,
        Decimal128Type, DecimalType, DurationMicrosecondType, DurationMillisecondType, DurationNanosecondType,
        DurationSecondType, Field, TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
        TimestampSecondType,
    },
    error::ArrowError,
};

use super::{
    import::ArrowPrimitiveMapping,
    schema::{
        DEFAULT_DECIMAL128_ARROW_PRECISION, arrow_decimal_scale_to_cudf_scale, cudf_decimal_scale_to_arrow_scale,
        validate_arrow_decimal,
    },
};
use crate::core::{
    column::ColumnView,
    error::{self, Error, ErrorCode, FACT_TYPE_ID, Result},
    types::{DataType, TypeId},
};

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
        TypeId::TimestampDays => fixed_width_to_arrow::<i32, Date32Type>(cv),
        TypeId::TimestampSeconds => fixed_width_to_arrow::<i64, TimestampSecondType>(cv),
        TypeId::TimestampMilliseconds => fixed_width_to_arrow::<i64, TimestampMillisecondType>(cv),
        TypeId::TimestampMicroseconds => fixed_width_to_arrow::<i64, TimestampMicrosecondType>(cv),
        TypeId::TimestampNanoseconds => fixed_width_to_arrow::<i64, TimestampNanosecondType>(cv),
        TypeId::DurationSeconds => fixed_width_to_arrow::<i64, DurationSecondType>(cv),
        TypeId::DurationMilliseconds => fixed_width_to_arrow::<i64, DurationMillisecondType>(cv),
        TypeId::DurationMicroseconds => fixed_width_to_arrow::<i64, DurationMicrosecondType>(cv),
        TypeId::DurationNanoseconds => fixed_width_to_arrow::<i64, DurationNanosecondType>(cv),
        TypeId::List => list_to_arrow(cv),
        _ => Err(Error::new_with(
            ErrorCode::UnsupportedDtype,
            "arrow::column_view_to_arrow",
            format!("unsupported cudf type for Arrow conversion: {:?}", dtype.id()),
        )
        .with_fact(FACT_TYPE_ID, dtype.id())),
    }
}

pub(super) fn column_view_to_arrow_with_requested_type(
    cv: &ColumnView<'_>,
    dtype: DataType,
    requested_type: Option<&ArrowDataType>,
) -> Result<Arc<dyn Array>> {
    match (dtype.id(), requested_type) {
        (TypeId::Decimal32, Some(ArrowDataType::Decimal32(precision, scale))) => {
            decimal_to_arrow::<i32, Decimal32Type>(cv, dtype, *precision, *scale)
        },
        (TypeId::Decimal64, Some(ArrowDataType::Decimal64(precision, scale))) => {
            decimal_to_arrow::<i64, Decimal64Type>(cv, dtype, *precision, *scale)
        },
        (
            TypeId::Decimal32 | TypeId::Decimal64 | TypeId::Decimal128,
            Some(ArrowDataType::Decimal128(precision, scale)),
        ) if dtype.scale() == arrow_decimal_scale_to_cudf_scale(*scale) => {
            decimal128_to_arrow_with_precision(cv, dtype, *precision, *scale)
        },
        _ => column_view_to_arrow(cv, dtype),
    }
}

pub(super) fn map_record_batch_error(error: ArrowError) -> Error {
    error::foreign_invariant_violation(
        "dataframe::to_record_batch",
        error.to_string(),
        "record_batch",
        "invalid_arrow_record_batch",
        "ensure Arrow schemas and column arrays have matching row counts and compatible field layouts",
    )
    .with_source(error)
}

pub(super) fn map_numeric_to_arrow_error(error: ArrowError) -> Error {
    error::foreign_invariant_violation(
        "arrow::numeric_to_arrow",
        error.to_string(),
        "arrow_array",
        "invalid_arrow_primitive_array",
        "ensure Arrow primitive values and null buffers have matching logical lengths",
    )
    .with_source(error)
}

pub(super) fn map_arrow_ffi_export_error(operation: &'static str, error: ArrowError) -> Error {
    error::foreign_invariant_violation(
        operation,
        error.to_string(),
        "arrow_array",
        "invalid_arrow_c_data_export",
        "ensure the Arrow array can be exported through the Arrow C Data Interface",
    )
    .with_source(error)
}

/// Arrow null buffer for the active range of `cv`, or `None` when the view has
/// no null mask or no null row.
///
/// cuDF and Arrow both pack validity least-significant-bit first, and the host
/// copy starts the active range at bit 0, so the bytes become the Arrow buffer
/// without unpacking.
fn host_null_buffer(cv: &ColumnView<'_>) -> Result<Option<NullBuffer>> {
    let Some(mask) = cv.to_host_null_mask_bytes()? else {
        return Ok(None);
    };
    let nulls = NullBuffer::new(BooleanBuffer::new(Buffer::from_vec(mask), 0, cv.len()?));
    Ok((nulls.null_count() > 0).then_some(nulls))
}

fn numeric_to_arrow<T>(cv: &ColumnView<'_>) -> Result<Arc<dyn Array>>
where T: ArrowPrimitiveMapping {
    let values = ScalarBuffer::from(cv.to_vec::<T>()?);
    let arr = PrimitiveArray::<T::ArrowPrimitive>::try_new(values, host_null_buffer(cv)?)
        .map_err(map_numeric_to_arrow_error)?;
    Ok(Arc::new(arr))
}

pub(super) fn native_slice_as_bytes<T>(values: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) }
}

fn fixed_width_host_values<T: ArrowNativeType>(cv: &ColumnView<'_>) -> Result<Vec<T>> {
    // SAFETY: `ArrowNativeType` is sealed to integer, float and plain interval
    // structs, for which every bit pattern is a valid value.
    unsafe { cv.to_fixed_width_host_vec::<T>("arrow::fixed_width_host_values") }
}

fn fixed_width_to_arrow<T, A>(cv: &ColumnView<'_>) -> Result<Arc<dyn Array>>
where
    T: ArrowNativeType,
    A: ArrowPrimitiveType<Native = T>,
{
    let values = ScalarBuffer::from(fixed_width_host_values::<T>(cv)?);
    let arr = PrimitiveArray::<A>::try_new(values, host_null_buffer(cv)?).map_err(map_numeric_to_arrow_error)?;
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
    // A sliced view shares its parent's whole offsets child, so only the
    // active `len + 1` offsets are copied.
    let normalized_offsets = if len == 0 {
        vec![0]
    } else {
        let offset_count = offsets_view.len()?;
        if offset_count <= root_end {
            return Err(Error::new_with(
                ErrorCode::IndexOutOfBounds,
                operation,
                "list offsets child is too small for the active logical range",
            )
            .with_detail("offset", root_offset.to_string())
            .with_detail("row_count", len.to_string())
            .with_detail("offset_count", offset_count.to_string())
            .with_detail("reason", "list_offsets_too_small"));
        }
        let active_offsets = offsets_view.slice_with_operation(
            root_offset_raw,
            i32::try_from(len + 1)
                .map_err(|_| error::range_limit_error(operation, "offset_count", "i32::MAX", len + 1))?,
            operation,
        )?;
        let host_offsets = active_offsets.to_vec::<i32>()?;
        let base_child_offset = host_offsets[0];
        if base_child_offset < 0 {
            return Err(Error::new_with(ErrorCode::ValueOutOfRange, operation, "list offsets must be non-negative")
                .with_detail("offset_index", root_offset.to_string())
                .with_detail("offset_value", base_child_offset.to_string())
                .with_detail("reason", "negative_list_child_offset"));
        }
        let mut normalized_offsets = Vec::with_capacity(host_offsets.len());
        let mut previous = base_child_offset;
        for (position, raw_offset) in host_offsets.into_iter().enumerate() {
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
    let nulls = host_null_buffer(cv)?;
    let array = ListArray::try_new(child_field, offsets, child_array, nulls).map_err(map_list_arrow_error)?;
    Ok(Arc::new(array))
}

fn map_decimal_to_arrow_error(error: ArrowError) -> Error {
    error::foreign_invariant_violation(
        "arrow::decimal128_to_arrow",
        error.to_string(),
        "arrow_array",
        "invalid_arrow_decimal_array",
        "ensure cudf Decimal128 scale fits Arrow Decimal128 metadata",
    )
    .with_source(error)
}

fn map_list_arrow_error(error: ArrowError) -> Error {
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
    decimal128_array_from_values(data, host_null_buffer(cv)?, DEFAULT_DECIMAL128_ARROW_PRECISION, arrow_scale)
}

fn decimal_widening_to_arrow<T>(cv: &ColumnView<'_>, dtype: DataType) -> Result<Arc<dyn Array>>
where T: ArrowNativeType + Into<i128> {
    let arrow_scale = cudf_decimal_scale_to_arrow_scale(dtype.scale(), "arrow::decimal_widening_to_arrow")
        .map_err(|error| error.with_fact(FACT_TYPE_ID, dtype.id()))?;
    let data = fixed_width_host_values::<T>(cv)?.into_iter().map(Into::into).collect::<Vec<_>>();
    decimal128_array_from_values(data, host_null_buffer(cv)?, DEFAULT_DECIMAL128_ARROW_PRECISION, arrow_scale)
}

/// Exports a cuDF decimal column as Arrow `Decimal128(precision, scale)` for a
/// schema contract whose scale matches the column, without the
/// `Decimal128(38, scale)` intermediate and Arrow cast.
///
/// cuDF decimals carry no precision, so valid rows are checked against
/// `precision` unless the source width already bounds them. At precision 38 a
/// `Decimal128` column is not checked, matching the default export.
fn decimal128_to_arrow_with_precision(
    cv: &ColumnView<'_>,
    dtype: DataType,
    precision: u8,
    scale: i8,
) -> Result<Arc<dyn Array>> {
    let operation = "arrow::decimal128_to_arrow_with_precision";
    validate_arrow_decimal::<Decimal128Type>(precision, scale, operation)?;
    let (data, source_max_digits): (Vec<i128>, u8) = match dtype.id() {
        // i32 magnitudes stay below 10^10 and i64 magnitudes below 10^19.
        TypeId::Decimal32 => (fixed_width_host_values::<i32>(cv)?.into_iter().map(i128::from).collect(), 10),
        TypeId::Decimal64 => (fixed_width_host_values::<i64>(cv)?.into_iter().map(i128::from).collect(), 19),
        _ => (fixed_width_host_values::<i128>(cv)?, DEFAULT_DECIMAL128_ARROW_PRECISION),
    };
    let nulls = host_null_buffer(cv)?;
    if precision < source_max_digits {
        check_decimal_precision::<Decimal128Type>(&data, nulls.as_ref(), dtype, precision, scale, operation)?;
    }
    decimal128_array_from_values(data, nulls, precision, scale)
}

fn decimal_to_arrow<T, A>(cv: &ColumnView<'_>, dtype: DataType, precision: u8, scale: i8) -> Result<Arc<dyn Array>>
where
    T: ArrowNativeType,
    A: DecimalType<Native = T>,
{
    let expected_scale = arrow_decimal_scale_to_cudf_scale(scale);
    if dtype.scale() != expected_scale {
        return Err(Error::new_with(
            ErrorCode::UnsupportedDtype,
            "arrow::decimal_to_arrow",
            "cudf decimal scale does not match retained Arrow schema",
        )
        .with_fact(FACT_TYPE_ID, dtype.id())
        .with_detail("cudf_scale", dtype.scale().to_string())
        .with_detail("arrow_scale", scale.to_string())
        .with_detail("reason", "decimal_scale_mismatch"));
    }
    validate_arrow_decimal::<A>(precision, scale, "arrow::decimal_to_arrow")?;
    let values = fixed_width_host_values::<T>(cv)?;
    let nulls = host_null_buffer(cv)?;
    // Arrow caps Decimal32 at 9 and Decimal64 at 18 digits, below the 10 and
    // 19 digits their native widths can hold, so every row is checked.
    check_decimal_precision::<A>(&values, nulls.as_ref(), dtype, precision, scale, "arrow::decimal_to_arrow")?;
    let arr = PrimitiveArray::<A>::try_new(ScalarBuffer::from(values), nulls)
        .and_then(|arr| arr.with_precision_and_scale(precision, scale))
        .map_err(map_decimal_to_arrow_error)?;
    Ok(Arc::new(arr))
}

/// Rejects the first valid row whose magnitude does not fit `precision`.
/// Null slots are skipped because cuDF leaves their values unspecified.
fn check_decimal_precision<A: DecimalType>(
    values: &[A::Native],
    nulls: Option<&NullBuffer>,
    dtype: DataType,
    precision: u8,
    scale: i8,
    operation: &'static str,
) -> Result<()> {
    let out_of_range = |row: &usize| !A::is_valid_decimal_precision(values[*row], precision);
    let first_out_of_range = match nulls {
        Some(nulls) => nulls.valid_indices().find(out_of_range),
        None => (0..values.len()).find(out_of_range),
    };
    let Some(row) = first_out_of_range else {
        return Ok(());
    };
    Err(Error::new_with(
        ErrorCode::ValueOutOfRange,
        operation,
        "decimal value exceeds the precision of the Arrow schema contract",
    )
    .with_fact(FACT_TYPE_ID, dtype.id())
    .with_detail("row", row.to_string())
    .with_detail("value", format!("{:?}", values[row]))
    .with_detail("precision", precision.to_string())
    .with_detail("scale", scale.to_string())
    .with_detail("reason", "decimal_precision_overflow"))
}

fn decimal128_array_from_values(
    data: Vec<i128>,
    nulls: Option<NullBuffer>,
    precision: u8,
    scale: i8,
) -> Result<Arc<dyn Array>> {
    let arr = Decimal128Array::try_new(ScalarBuffer::from(data), nulls)
        .and_then(|arr| arr.with_precision_and_scale(precision, scale))
        .map_err(map_decimal_to_arrow_error)?;
    Ok(Arc::new(arr))
}

fn bool_to_arrow(cv: &ColumnView<'_>) -> Result<Arc<dyn Array>> {
    // Null slots may hold any byte, so values are compared with zero instead
    // of being read as Rust `bool`.
    let raw = cv.to_raw_bytes()?;
    let values = BooleanBuffer::collect_bool(raw.len(), |row| raw[row] != 0);
    Ok(Arc::new(BooleanArray::new(values, host_null_buffer(cv)?)))
}

fn string_to_arrow(cv: &ColumnView<'_>) -> Result<Arc<dyn Array>> {
    let operation = "arrow::string_to_arrow";
    let data = cv.to_strings_host_raw(operation)?;
    let nulls = host_null_buffer(cv)?;
    let chars_bytes = data.iter().try_fold(0usize, |total, value| {
        total.checked_add(value.len()).ok_or_else(|| {
            Error::new_with(ErrorCode::SizeOverflow, operation, "string character byte count overflowed usize")
                .with_detail("reason", "string_character_count_overflow")
        })
    })?;
    let mut builder = StringBuilder::with_capacity(data.len(), chars_bytes);
    for (row, value) in data.iter().enumerate() {
        if nulls.as_ref().is_some_and(|nulls| nulls.is_null(row)) {
            builder.append_null();
        } else {
            builder.append_value(value);
        }
    }
    Ok(Arc::new(builder.finish()))
}
