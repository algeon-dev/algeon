//! cudf column to Arrow array host-copy conversion.

use std::sync::Arc;

use arrow::{
    array::{Array, BooleanArray, Decimal128Array, ListArray, PrimitiveArray, StringBuilder},
    buffer::{BooleanBuffer, NullBuffer, OffsetBuffer, ScalarBuffer},
    datatypes::{
        ArrowNativeType, ArrowPrimitiveType, DataType as ArrowDataType, Date32Type, Decimal32Type, Decimal64Type,
        DecimalType, DurationMicrosecondType, DurationMillisecondType, DurationNanosecondType, DurationSecondType,
        Field, TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType,
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
            let null_buffer = NullBuffer::from(mask);
            let buffer = ScalarBuffer::from(data);
            let arr = PrimitiveArray::<T::ArrowPrimitive>::try_new(buffer, Some(null_buffer))
                .map_err(map_numeric_to_arrow_error)?;
            Ok(Arc::new(arr))
        },
    }
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

pub(super) fn native_slice_as_bytes<T>(values: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) }
}

fn fixed_width_host_values<T: Copy + Default>(cv: &ColumnView<'_>) -> Result<Vec<T>> {
    cv.to_fixed_width_host_bytes().map(|bytes| bytes_to_native_vec::<T>(&bytes))
}

fn fixed_width_to_arrow<T, A>(cv: &ColumnView<'_>) -> Result<Arc<dyn Array>>
where
    T: Copy + ArrowNativeType,
    A: ArrowPrimitiveType<Native = T>,
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
    decimal128_array_from_values(data, cv.to_host_validity()?, arrow_scale)
}

fn decimal_widening_to_arrow<T>(cv: &ColumnView<'_>, dtype: DataType) -> Result<Arc<dyn Array>>
where T: Copy + Default + Into<i128> {
    let arrow_scale = cudf_decimal_scale_to_arrow_scale(dtype.scale(), "arrow::decimal_widening_to_arrow")
        .map_err(|error| error.with_fact(FACT_TYPE_ID, dtype.id()))?;
    let data = fixed_width_host_values::<T>(cv)?.into_iter().map(Into::into).collect::<Vec<_>>();
    decimal128_array_from_values(data, cv.to_host_validity()?, arrow_scale)
}

fn decimal_to_arrow<T, A>(cv: &ColumnView<'_>, dtype: DataType, precision: u8, scale: i8) -> Result<Arc<dyn Array>>
where
    T: Copy + Default + ArrowNativeType,
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
    let data = fixed_width_host_values::<T>(cv)?;
    let arr = match cv.to_host_validity()? {
        None => PrimitiveArray::<A>::from_iter_values(data),
        Some(mask) => {
            PrimitiveArray::<A>::from_iter(data.into_iter().zip(mask).map(|(value, valid)| valid.then_some(value)))
        },
    }
    .with_precision_and_scale(precision, scale)
    .map_err(map_decimal_to_arrow_error)?;
    Ok(Arc::new(arr))
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

fn bool_to_arrow(cv: &ColumnView<'_>) -> Result<Arc<dyn Array>> {
    // Avoid constructing Rust `bool` values directly from raw device bytes.
    let data: Vec<bool> = cv.to_raw_bytes()?.into_iter().map(|byte| byte != 0).collect();
    let validity = cv.to_host_validity()?;

    match validity {
        None => Ok(Arc::new(BooleanArray::from(data))),
        Some(mask) => {
            let null_buffer = NullBuffer::from(mask);
            let bool_buffer = BooleanBuffer::from(data);
            let arr = BooleanArray::new(bool_buffer, Some(null_buffer));
            Ok(Arc::new(arr))
        },
    }
}

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
