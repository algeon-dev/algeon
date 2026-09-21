//! Host import device-memory and Arrow export capacity estimation.

use std::sync::Arc;

use arrow::{array::*, datatypes::DataType as ArrowDataType, record_batch::RecordBatch};

use super::schema::{
    ArrowHostSchema, cudf_dtype_from_arrow_type_in, list_child_is_supported, list_fixed_width_interop_dtype,
    list_numeric_child_is_supported,
};
use crate::{
    core::{
        column::ColumnView,
        error::{self, Error, ErrorCode, Result},
        types::cudf_null_mask_allocation_size_bytes,
    },
    frame::dataframe::DataFrame,
};

/// Returns the device bytes retained when Arrow list arrays are concatenated
/// into one cudf list column.
pub fn estimate_list_import_device_memory_bytes(arrays: &[ArrayRef]) -> Result<usize> {
    list_import_device_allocation_bytes(arrays)?.into_iter().try_fold(0usize, |total, bytes| {
        checked_size_add(total, bytes, "arrow_host::estimate_list_import_device_memory_bytes", "summing list buffers")
    })
}

/// Returns the aligned-live-byte budget required to import same-schema Arrow
/// record batches into one cudf dataframe.
#[doc(hidden)]
pub fn estimate_record_batches_import_required_budget_bytes(batches: &[RecordBatch]) -> Result<usize> {
    let operation = "arrow_host::estimate_record_batches_import_required_budget_bytes";
    let Some(first_batch) = batches.first() else {
        return Err(Error::new_with(ErrorCode::EmptyInput, operation, "at least one record batch is required"));
    };
    let schema = first_batch.schema();
    for (batch_index, batch) in batches.iter().enumerate().skip(1) {
        if batch.schema() != schema {
            return Err(Error::new_with(
                ErrorCode::TypeMismatch,
                operation,
                format!("record batch {batch_index} does not share the first batch's schema"),
            ));
        }
    }

    let total_rows = batches.iter().try_fold(0usize, |total, batch| {
        checked_size_add(total, batch.num_rows(), operation, "summing record batch rows")
    })?;
    if i32::try_from(total_rows).is_err() {
        return Err(Error::new_with(
            ErrorCode::ValueOutOfRange,
            operation,
            "combined row count exceeds cudf's supported column length",
        )
        .with_detail("rows", total_rows.to_string()));
    }

    let mut required = 0usize;
    for (index, field) in schema.fields().iter().enumerate() {
        let arrays = batches.iter().map(|batch| Arc::clone(batch.column(index))).collect::<Vec<_>>();
        let allocations = import_column_device_allocation_bytes(&arrays, field.data_type(), operation)?;
        for bytes in allocations {
            required = checked_size_add(
                required,
                import_allocation_charge(bytes, operation)?,
                operation,
                "summing aligned import allocations",
            )?;
        }
    }
    Ok(required)
}

fn list_import_device_allocation_bytes(arrays: &[ArrayRef]) -> Result<Vec<usize>> {
    let operation = "arrow_host::estimate_list_import_device_memory_bytes";
    let Some(first) = arrays.first() else {
        return Err(Error::new_with(ErrorCode::EmptyInput, operation, "at least one Arrow list array is required"));
    };
    let data_type = first.data_type();
    let child_type = match data_type {
        ArrowDataType::List(field) | ArrowDataType::LargeList(field) if list_child_is_supported(field) => {
            field.data_type()
        },
        ArrowDataType::FixedSizeList(field, dimension) if *dimension > 0 && list_numeric_child_is_supported(field) => {
            field.data_type()
        },
        _ => {
            return Err(Error::new_with(
                ErrorCode::UnsupportedDtype,
                operation,
                "device import sizing requires supported Arrow list carry arrays",
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
    let child_data = list_child_import_device_allocation_bytes(child_type, child_rows, string_chars, operation)?;
    let child_validity = if child_has_nulls { cudf_null_mask_allocation_size_bytes(child_rows) } else { 0 };
    let mut allocations = vec![root_offsets];
    if root_validity > 0 {
        allocations.push(root_validity);
    }
    allocations.extend(child_data);
    if child_validity > 0 {
        allocations.push(child_validity);
    }
    Ok(allocations)
}

fn import_column_device_allocation_bytes(
    arrays: &[ArrayRef],
    data_type: &ArrowDataType,
    operation: &'static str,
) -> Result<Vec<usize>> {
    if matches!(data_type, ArrowDataType::List(_) | ArrowDataType::LargeList(_) | ArrowDataType::FixedSizeList(_, _)) {
        return list_import_device_allocation_bytes(arrays).map_err(|error| error.reanchor(operation));
    }

    let rows = arrays
        .iter()
        .try_fold(0usize, |total, array| checked_size_add(total, array.len(), operation, "summing column rows"))?;
    let has_nulls = arrays.iter().any(|array| array.null_count() > 0);
    let mut allocations =
        if matches!(data_type, ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View) {
            let chars = arrays.iter().try_fold(0usize, |total, array| {
                checked_size_add(
                    total,
                    active_string_bytes(array.as_ref(), 0, array.len(), data_type, operation)?,
                    operation,
                    "summing string character bytes",
                )
            })?;
            let offsets = checked_size_mul(
                checked_size_add(rows, 1, operation, "sizing string offsets")?,
                4,
                operation,
                "sizing string offsets",
            )?;
            vec![offsets, chars]
        } else {
            let dtype = cudf_dtype_from_arrow_type_in(data_type, operation)?;
            let width = dtype.id().size_in_bytes()?.ok_or_else(|| {
                Error::new_with(ErrorCode::UnsupportedDtype, operation, "Arrow import type has no fixed-width layout")
                    .with_detail("arrow_type", format!("{data_type:?}"))
            })?;
            let width = usize::try_from(width).map_err(|_| size_overflow(operation, "converting cudf type width"))?;
            vec![checked_size_mul(rows, width, operation, "sizing fixed-width values")?]
        };
    if has_nulls {
        allocations.push(cudf_null_mask_allocation_size_bytes(rows));
    }
    Ok(allocations)
}

fn import_allocation_charge(bytes: usize, operation: &'static str) -> Result<usize> {
    const CUDA_ALLOCATION_ALIGNMENT_BYTES: usize = 256;
    if bytes == 0 {
        return Ok(0);
    }
    let rounded =
        checked_size_add(bytes, CUDA_ALLOCATION_ALIGNMENT_BYTES - 1, operation, "rounding import allocation")?;
    Ok(rounded / CUDA_ALLOCATION_ALIGNMENT_BYTES * CUDA_ALLOCATION_ALIGNMENT_BYTES)
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
        ArrowDataType::FixedSizeList(_, dimension) => {
            let list = array.as_any().downcast_ref::<FixedSizeListArray>().ok_or_else(|| {
                Error::new_with(
                    ErrorCode::TypeMismatch,
                    operation,
                    "Arrow FixedSizeList array had an incompatible runtime type",
                )
            })?;
            let dimension = usize::try_from(*dimension)
                .map_err(|_| size_overflow(operation, "converting FixedSizeList dimension"))?;
            let expected_child_rows = list
                .len()
                .checked_mul(dimension)
                .ok_or_else(|| size_overflow(operation, "sizing active FixedSizeList child range"))?;
            if list.values().len() != expected_child_rows {
                return Err(error::foreign_invariant_violation(
                    operation,
                    "Arrow FixedSizeList values length does not match its active row span",
                    "arrow_array",
                    "fixed_size_list_child_length_mismatch",
                    "provide a valid sliced FixedSizeList array",
                )
                .with_detail("child_len", list.values().len().to_string())
                .with_detail("expected_child_len", expected_child_rows.to_string()));
            }
            Ok((Arc::clone(list.values()), 0, expected_child_rows))
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

fn list_child_import_device_allocation_bytes(
    child_type: &ArrowDataType,
    child_rows: usize,
    string_chars: usize,
    operation: &'static str,
) -> Result<Vec<usize>> {
    let width = match child_type {
        ArrowDataType::Boolean | ArrowDataType::Int8 | ArrowDataType::UInt8 => Some(1),
        ArrowDataType::Int16 | ArrowDataType::UInt16 => Some(2),
        ArrowDataType::Int32 | ArrowDataType::UInt32 | ArrowDataType::Float32 => Some(4),
        ArrowDataType::Int64 | ArrowDataType::UInt64 | ArrowDataType::Float64 => Some(8),
        ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View => None,
        _ => unreachable!("caller validates the basic list child type"),
    };
    if let Some(width) = width {
        return Ok(vec![checked_size_mul(child_rows, width, operation, "sizing fixed-width list child values")?]);
    }
    if child_rows == 0 {
        return Ok(Vec::new());
    }
    let child_offset_count = checked_size_add(child_rows, 1, operation, "sizing string child offset count")?;
    Ok(vec![checked_size_mul(child_offset_count, 4, operation, "sizing string child offsets")?, string_chars])
}

pub(super) fn list_child_import_device_bytes(
    child_type: &ArrowDataType,
    child_rows: usize,
    string_chars: usize,
    operation: &'static str,
) -> Result<usize> {
    list_child_import_device_allocation_bytes(child_type, child_rows, string_chars, operation)?
        .into_iter()
        .try_fold(0usize, |total, bytes| checked_size_add(total, bytes, operation, "summing list child buffers"))
}

/// Conservatively estimate the Arrow buffer capacity retained by a host-copy
/// export before any device values are copied to host.
///
/// The estimate follows the final external Arrow schema rather than the cuDF
/// source representation. In particular, narrow decimals retain their 4-byte
/// or 8-byte logical width, booleans become bit-packed, and string/list offset
/// widths follow Utf8/LargeUtf8 and List/LargeList respectively. Every logical
/// Arrow buffer is rounded to Arrow's 64-byte mutable-buffer allocation
/// quantum, making the result an upper bound for the capacities counted by
/// DataFusion's `get_record_batch_memory_size`.
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
        | ArrowDataType::Decimal32(_, _)
        | ArrowDataType::Decimal64(_, _)
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
        | ArrowDataType::Time32(_)
        | ArrowDataType::Decimal32(_, _) => estimate_fixed_width_capacity(rows, 4, operation)?,
        ArrowDataType::Int64
        | ArrowDataType::UInt64
        | ArrowDataType::Float64
        | ArrowDataType::Date64
        | ArrowDataType::Time64(_)
        | ArrowDataType::Timestamp(_, _)
        | ArrowDataType::Duration(_)
        | ArrowDataType::Decimal64(_, _) => estimate_fixed_width_capacity(rows, 8, operation)?,
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
            let dimension = usize::try_from(*dimension).map_err(|_| {
                Error::new_with(ErrorCode::ValueOutOfRange, operation, "FixedSizeList dimension must be positive")
                    .with_detail("dimension", dimension.to_string())
            })?;
            let expected_child_rows = rows.checked_mul(dimension).ok_or_else(|| {
                Error::new_with(ErrorCode::SizeOverflow, operation, "FixedSizeList child row count overflowed usize")
                    .with_detail("rows", rows.to_string())
                    .with_detail("dimension", dimension.to_string())
            })?;
            let child_values =
                estimate_arrow_list_fixed_width_values_capacity(expected_child_rows, child.data_type(), operation)?;
            let child_validity =
                estimate_arrow_validity_capacity(expected_child_rows, nullable || child.is_nullable(), operation)?;
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

pub(super) fn checked_size_add(
    left: usize,
    right: usize,
    operation: &'static str,
    context: &'static str,
) -> Result<usize> {
    left.checked_add(right).ok_or_else(|| size_overflow(operation, context))
}

pub(super) fn checked_size_mul(
    left: usize,
    right: usize,
    operation: &'static str,
    context: &'static str,
) -> Result<usize> {
    left.checked_mul(right).ok_or_else(|| size_overflow(operation, context))
}

pub(super) fn size_overflow(operation: &'static str, context: &'static str) -> Error {
    Error::new_with(ErrorCode::SizeOverflow, operation, "host materialization size accounting overflowed usize")
        .with_detail("size_context", context)
}
