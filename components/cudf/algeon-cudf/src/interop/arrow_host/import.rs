//! Arrow array to cudf column host-copy conversion.

use std::{ffi::c_void, ptr, sync::Arc};

use arrow::{
    array::*,
    buffer::NullBuffer,
    datatypes::{
        ArrowNativeType, ArrowPrimitiveType, DataType as ArrowDataType, Date32Type, Date64Type, Decimal32Type,
        Decimal64Type, Decimal128Type, DurationMicrosecondType, DurationMillisecondType, DurationNanosecondType,
        DurationSecondType, FieldRef, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type,
        Time32MillisecondType, Time32SecondType, Time64MicrosecondType, Time64NanosecondType, TimeUnit,
        TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType, UInt8Type,
        UInt16Type, UInt32Type, UInt64Type,
    },
    ffi::{FFI_ArrowArray, FFI_ArrowSchema},
};
use rapids_interop::{
    BoundBytes, BufferBinding, BufferId, ColumnId, ColumnLayout, DataType as InteropDataType, PhysicalValidity,
    Placement, PreparedAccess, PreparedColumnExport, SemanticColumn, TypeId as InteropTypeId,
};

use super::{
    estimate::{checked_size_add, checked_size_mul, list_child_import_device_bytes, size_overflow},
    export::{map_arrow_ffi_export_error, native_slice_as_bytes},
    schema::{
        arrow_decimal_scale_to_cudf_scale, list_child_is_supported, list_fixed_width_child_is_supported,
        list_fixed_width_interop_dtype, list_numeric_child_is_supported,
    },
};
use crate::{
    core::{
        column::Column,
        error::{self, Error, ErrorCode, Result},
        types::{CudfType, DataType, TypeId, cudf_null_mask_allocation_size_bytes},
    },
    frame::series::Series,
    runtime::Runtime,
};

pub fn all_null_list_series(runtime: &Runtime, data_type: &ArrowDataType, rows: usize) -> Result<Series> {
    let operation = "arrow_host::all_null_list_series";
    match data_type {
        ArrowDataType::List(field) | ArrowDataType::LargeList(field) if list_child_is_supported(field) => {},
        ArrowDataType::FixedSizeList(field, dimension) if *dimension > 0 && list_numeric_child_is_supported(field) => {
            let dimension = usize::try_from(*dimension)
                .map_err(|_| size_overflow(operation, "converting FixedSizeList dimension"))?;
            let child_rows = rows
                .checked_mul(dimension)
                .ok_or_else(|| size_overflow(operation, "sizing all-null FixedSizeList child rows"))?;
            if i32::try_from(rows).is_err() || i32::try_from(child_rows).is_err() {
                return Err(Error::new_with(
                    ErrorCode::ValueOutOfRange,
                    operation,
                    "all-null FixedSizeList exceeds cudf's signed 32-bit row or child-offset range",
                )
                .with_detail("root_rows", rows.to_string())
                .with_detail("child_rows", child_rows.to_string()));
            }
            let root_offsets = checked_size_mul(
                checked_size_add(rows, 1, operation, "sizing all-null FixedSizeList offset count")?,
                4,
                operation,
                "sizing all-null FixedSizeList offsets",
            )?;
            let child_values = list_child_import_device_bytes(field.data_type(), child_rows, 0, operation)?;
            let _ = checked_size_add(
                checked_size_add(
                    root_offsets,
                    cudf_null_mask_allocation_size_bytes(rows),
                    operation,
                    "sizing all-null FixedSizeList root buffers",
                )?,
                checked_size_add(
                    child_values,
                    cudf_null_mask_allocation_size_bytes(child_rows),
                    operation,
                    "sizing all-null FixedSizeList child buffers",
                )?,
                operation,
                "sizing all-null FixedSizeList column",
            )?;
        },
        _ => {
            return Err(Error::new_with(
                ErrorCode::UnsupportedDtype,
                operation,
                "typed null list construction requires a supported Arrow list carry type",
            )
            .with_detail("arrow_type", format!("{data_type:?}"))
            .with_detail("reason", "unsupported_null_list_type"));
        },
    }
    let array = new_null_array(data_type, rows);
    let column =
        arrow_array_to_column(runtime, array.as_ref(), "all_null_list").map_err(|error| error.reanchor(operation))?;
    Series::from_column(column).map_err(|error| error.reanchor(operation))
}

pub(super) fn arrow_array_to_column(runtime: &Runtime, arr: &dyn Array, field_name: &str) -> Result<Column> {
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
        ArrowDataType::Date32 => {
            primitive_arrow_to_fixed_width_column::<Date32Type>(runtime, arr, DataType::new(TypeId::TimestampDays))
        },
        ArrowDataType::Date64 => primitive_arrow_to_fixed_width_column::<Date64Type>(
            runtime,
            arr,
            DataType::new(TypeId::TimestampMilliseconds),
        ),
        ArrowDataType::Time32(TimeUnit::Second) => {
            time32_arrow_to_duration_column::<Time32SecondType>(runtime, arr, DataType::new(TypeId::DurationSeconds))
        },
        ArrowDataType::Time32(TimeUnit::Millisecond) => time32_arrow_to_duration_column::<Time32MillisecondType>(
            runtime,
            arr,
            DataType::new(TypeId::DurationMilliseconds),
        ),
        ArrowDataType::Time64(TimeUnit::Microsecond) => primitive_arrow_to_fixed_width_column::<Time64MicrosecondType>(
            runtime,
            arr,
            DataType::new(TypeId::DurationMicroseconds),
        ),
        ArrowDataType::Time64(TimeUnit::Nanosecond) => primitive_arrow_to_fixed_width_column::<Time64NanosecondType>(
            runtime,
            arr,
            DataType::new(TypeId::DurationNanoseconds),
        ),
        ArrowDataType::Timestamp(unit, _) => timestamp_arrow_to_column(runtime, arr, unit),
        ArrowDataType::Duration(TimeUnit::Second) => primitive_arrow_to_fixed_width_column::<DurationSecondType>(
            runtime,
            arr,
            DataType::new(TypeId::DurationSeconds),
        ),
        ArrowDataType::Duration(TimeUnit::Millisecond) => primitive_arrow_to_fixed_width_column::<
            DurationMillisecondType,
        >(
            runtime, arr, DataType::new(TypeId::DurationMilliseconds)
        ),
        ArrowDataType::Duration(TimeUnit::Microsecond) => primitive_arrow_to_fixed_width_column::<
            DurationMicrosecondType,
        >(
            runtime, arr, DataType::new(TypeId::DurationMicroseconds)
        ),
        ArrowDataType::Duration(TimeUnit::Nanosecond) => {
            primitive_arrow_to_fixed_width_column::<DurationNanosecondType>(
                runtime,
                arr,
                DataType::new(TypeId::DurationNanoseconds),
            )
        },
        ArrowDataType::Decimal128(_, scale) => {
            let dtype = DataType::fixed_point(TypeId::Decimal128, arrow_decimal_scale_to_cudf_scale(*scale))?;
            decimal128_array_to_column(runtime, arr, dtype)
        },
        ArrowDataType::Decimal32(_, scale) => primitive_arrow_to_fixed_width_column::<Decimal32Type>(
            runtime,
            arr,
            DataType::fixed_point(TypeId::Decimal32, arrow_decimal_scale_to_cudf_scale(*scale))?,
        ),
        ArrowDataType::Decimal64(_, scale) => primitive_arrow_to_fixed_width_column::<Decimal64Type>(
            runtime,
            arr,
            DataType::fixed_point(TypeId::Decimal64, arrow_decimal_scale_to_cudf_scale(*scale))?,
        ),
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
        TimeUnit::Second => primitive_arrow_to_fixed_width_column::<TimestampSecondType>(
            runtime,
            arr,
            DataType::new(TypeId::TimestampSeconds),
        ),
        TimeUnit::Millisecond => primitive_arrow_to_fixed_width_column::<TimestampMillisecondType>(
            runtime,
            arr,
            DataType::new(TypeId::TimestampMilliseconds),
        ),
        TimeUnit::Microsecond => primitive_arrow_to_fixed_width_column::<TimestampMicrosecondType>(
            runtime,
            arr,
            DataType::new(TypeId::TimestampMicroseconds),
        ),
        TimeUnit::Nanosecond => primitive_arrow_to_fixed_width_column::<TimestampNanosecondType>(
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
        ColumnLayout::List { offsets: BufferId::new(0), child: ColumnId::new(1) },
    );
    let child_column = SemanticColumn::new(
        ColumnId::new(1),
        InteropDataType::new(InteropTypeId::String),
        child_len,
        0,
        child_validity.validity,
        ColumnLayout::BinaryLike { offsets: BufferId::new(2), data: BufferId::new(3) },
    );
    let export = PreparedColumnExport::try_new(
        ColumnId::new(0),
        vec![list_column, child_column],
        bindings,
        PreparedAccess::Host,
    )
    .map_err(|error| error::from_interop_in(error, operation))?
    .with_retained_owner(Arc::new(retained_buffers));

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
        ColumnLayout::List { offsets: BufferId::new(0), child: ColumnId::new(1) },
    );
    let child_column = SemanticColumn::new(
        ColumnId::new(1),
        child_dtype,
        child_len,
        0,
        child_validity.validity,
        ColumnLayout::FixedWidth { values: BufferId::new(2) },
    );
    let export = PreparedColumnExport::try_new(
        ColumnId::new(0),
        vec![list_column, child_column],
        bindings,
        PreparedAccess::Host,
    )
    .map_err(|error| error::from_interop_in(error, operation))?
    .with_retained_owner(Arc::new(retained_buffers));

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
        ArrowDataType::Int8 => primitive_bytes!(Int8Type),
        ArrowDataType::Int16 => primitive_bytes!(Int16Type),
        ArrowDataType::Int32 => primitive_bytes!(Int32Type),
        ArrowDataType::Int64 => primitive_bytes!(Int64Type),
        ArrowDataType::UInt8 => primitive_bytes!(UInt8Type),
        ArrowDataType::UInt16 => primitive_bytes!(UInt16Type),
        ArrowDataType::UInt32 => primitive_bytes!(UInt32Type),
        ArrowDataType::UInt64 => primitive_bytes!(UInt64Type),
        ArrowDataType::Float32 => primitive_bytes!(Float32Type),
        ArrowDataType::Float64 => primitive_bytes!(Float64Type),
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
where A: ArrowPrimitiveType {
    let prim = arr.as_primitive::<A>();
    let values = prim.values().as_ref();
    let bytes = native_slice_as_bytes(values);
    let validity = prim.nulls().map(|null_buf| null_buf.iter().collect::<Vec<_>>());
    Column::from_fixed_width_host_bytes_with_runtime(runtime, dtype, bytes, validity.as_deref())
        .map_err(|error| error.reanchor("arrow::arrow_array_to_column"))
}

fn time32_arrow_to_duration_column<A>(runtime: &Runtime, arr: &dyn Array, dtype: DataType) -> Result<Column>
where A: ArrowPrimitiveType<Native = i32> {
    let prim = arr.as_primitive::<A>();
    let values = prim.values().iter().map(|value| i64::from(*value)).collect::<Vec<_>>();
    let bytes = native_slice_as_bytes(&values);
    let validity = prim.nulls().map(|null_buf| null_buf.iter().collect::<Vec<_>>());
    Column::from_fixed_width_host_bytes_with_runtime(runtime, dtype, bytes, validity.as_deref())
        .map_err(|error| error.reanchor("arrow::arrow_array_to_column"))
}

fn decimal128_array_to_column(runtime: &Runtime, arr: &dyn Array, dtype: DataType) -> Result<Column> {
    let decimal = arr.as_primitive::<Decimal128Type>();
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
pub(super) fn string_array_to_column(runtime: &Runtime, arr: &dyn Array) -> Result<Column> {
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
pub(super) fn large_string_array_to_column(runtime: &Runtime, arr: &dyn Array) -> Result<Column> {
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
pub(super) trait ArrowPrimitiveMapping: CudfType + ArrowNativeType {
    type ArrowPrimitive: ArrowPrimitiveType<Native = Self>;
}

impl ArrowPrimitiveMapping for i8 {
    type ArrowPrimitive = Int8Type;
}
impl ArrowPrimitiveMapping for i16 {
    type ArrowPrimitive = Int16Type;
}
impl ArrowPrimitiveMapping for i32 {
    type ArrowPrimitive = Int32Type;
}
impl ArrowPrimitiveMapping for i64 {
    type ArrowPrimitive = Int64Type;
}
impl ArrowPrimitiveMapping for u8 {
    type ArrowPrimitive = UInt8Type;
}
impl ArrowPrimitiveMapping for u16 {
    type ArrowPrimitive = UInt16Type;
}
impl ArrowPrimitiveMapping for u32 {
    type ArrowPrimitive = UInt32Type;
}
impl ArrowPrimitiveMapping for u64 {
    type ArrowPrimitive = UInt64Type;
}
impl ArrowPrimitiveMapping for f32 {
    type ArrowPrimitive = Float32Type;
}
impl ArrowPrimitiveMapping for f64 {
    type ArrowPrimitive = Float64Type;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::error::ErrorStatus;

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
        assert_eq!(outside_cudf_range.status(), ErrorStatus::Permanent);
        assert_eq!(outside_cudf_range.detail_value("reason"), Some("list_offset_i32_overflow"));
    }
}
