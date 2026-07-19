//! Low-level column reshaping operations.

use std::{ptr, sync::Arc};

use crate::{
    core::{
        column::{Column, ColumnView},
        error::{
            self, Error, ErrorCode, FACT_ACTUAL_ROWS, FACT_COLUMN_INDEX, FACT_EXPECTED_ROWS, FACT_LEFT_TYPE_ID,
            FACT_RIGHT_TYPE_ID, Result,
        },
    },
    runtime::{
        execution_context::ExecutionOverrides, execution_domain::ExecutionDomain, execution_resolver::ResolvedExecution,
    },
};

const INTERLEAVE_OPERATION: &str = "reshape::interleave_columns";

fn interleave_requires_input_error() -> Error {
    Error::new_with(ErrorCode::EmptyInput, INTERLEAVE_OPERATION, "interleave requires at least one input column")
        .with_detail("resource", "input_columns")
        .with_detail("reason", "empty_input")
        .with_detail("remedy", "provide at least one equal-length input column")
}

fn checked_interleave_output_size(rows: usize, columns: usize) -> Result<usize> {
    let output_elements = rows.checked_mul(columns).ok_or_else(|| {
        Error::new_with(ErrorCode::SizeOverflow, INTERLEAVE_OPERATION, "interleave output size overflowed usize")
            .with_detail("rows", rows.to_string())
            .with_detail("columns", columns.to_string())
            .with_detail("resource", "output_elements")
            .with_detail("reason", "shape_product_overflow")
            .with_detail("remedy", "reduce the interleave row or column count")
    })?;
    if output_elements > i32::MAX as usize {
        return Err(Error::new_with(
            ErrorCode::ValueOutOfRange,
            INTERLEAVE_OPERATION,
            "interleave output size exceeds cudf size_type maximum",
        )
        .with_detail("rows", rows.to_string())
        .with_detail("columns", columns.to_string())
        .with_detail("output_elements", output_elements.to_string())
        .with_detail("max_output_elements", i32::MAX.to_string())
        .with_detail("resource", "output_elements")
        .with_detail("reason", "output_size_exceeds_cudf_size_type")
        .with_detail("remedy", "reduce the interleave row or column count so their product fits within i32::MAX"));
    }
    Ok(output_elements)
}

fn validate_interleave_inputs(columns: &[&ColumnView<'_>]) -> Result<()> {
    let Some((first, rest)) = columns.split_first() else {
        return Err(interleave_requires_input_error());
    };

    let expected_type = first.data_type().map_err(|error| error.reanchor(INTERLEAVE_OPERATION))?.id();
    let expected_rows = first.len().map_err(|error| error.reanchor(INTERLEAVE_OPERATION))?;
    for (offset, column) in rest.iter().enumerate() {
        let column_index = offset + 1;
        let actual_type = column.data_type().map_err(|error| error.reanchor(INTERLEAVE_OPERATION))?.id();
        if actual_type != expected_type {
            return Err(Error::new_with(
                ErrorCode::TypeMismatch,
                INTERLEAVE_OPERATION,
                "interleave input columns must have the same data type",
            )
            .with_fact(FACT_COLUMN_INDEX, column_index)
            .with_fact(FACT_LEFT_TYPE_ID, expected_type)
            .with_fact(FACT_RIGHT_TYPE_ID, actual_type));
        }

        let actual_rows = column.len().map_err(|error| error.reanchor(INTERLEAVE_OPERATION))?;
        if actual_rows != expected_rows {
            return Err(Error::new_with(
                ErrorCode::ColumnLengthMismatch,
                INTERLEAVE_OPERATION,
                "interleave input columns must have the same row count",
            )
            .with_fact(FACT_COLUMN_INDEX, column_index)
            .with_fact(FACT_EXPECTED_ROWS, expected_rows)
            .with_fact(FACT_ACTUAL_ROWS, actual_rows));
        }
    }
    checked_interleave_output_size(expected_rows, columns.len())?;
    Ok(())
}

fn compatible_execution(columns: &[&ColumnView<'_>]) -> Result<ResolvedExecution> {
    let first = columns.first().ok_or_else(interleave_requires_input_error)?;
    ResolvedExecution::many(
        first.execution_domain(),
        columns[1..].iter().map(|column| column.execution_domain()),
        &ExecutionOverrides::new(),
        INTERLEAVE_OPERATION,
    )
}

/// Interleave equal-length columns into a single row-major column.
///
/// For input columns `[a, b]`, output values are
/// `[a[0], b[0], a[1], b[1], ...]`. All columns must have the same data type
/// and row count.
pub fn interleave_columns(columns: &[&ColumnView<'_>]) -> Result<Column> {
    validate_interleave_inputs(columns)?;
    let execution = compatible_execution(columns)?;
    interleave_columns_with_exec(columns, execution.output_domain(), execution.exec_options())
}

pub(crate) fn interleave_columns_with_exec(
    columns: &[&ColumnView<'_>],
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
) -> Result<Column> {
    validate_interleave_inputs(columns)?;
    let handles = columns.iter().map(|column| column.raw_handle()).collect::<Vec<_>>();
    let column_count = i32::try_from(handles.len()).map_err(|_| {
        error::limit_validation_error(
            INTERLEAVE_OPERATION,
            "number of input columns exceeds i32::MAX",
            "input_columns",
            "column_count",
            "i32::MAX",
            handles.len(),
            "reduce the number of input columns to fit within i32::MAX",
        )
    })?;
    let mut out = ptr::null_mut();
    error::check_status_in_with_details(
        unsafe { cudf_sys::cudf_interleave_columns_ex(handles.as_ptr(), column_count, exec_options, &mut out) },
        INTERLEAVE_OPERATION,
        [("column_count", handles.len().to_string())],
    )?;
    // SAFETY: a successful FFI call transfers ownership of the returned
    // column handle into this Rust owner.
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interleave_output_size_accepts_cudf_size_type_maximum() {
        assert_eq!(checked_interleave_output_size(i32::MAX as usize, 1).unwrap(), i32::MAX as usize);
    }

    #[test]
    fn interleave_output_size_rejects_one_past_cudf_size_type_maximum() {
        let error = checked_interleave_output_size(i32::MAX as usize / 2 + 1, 2)
            .expect_err("interleave output length must fit cudf size_type");
        assert_eq!(error.code(), ErrorCode::ValueOutOfRange);
        assert_eq!(error.operation(), INTERLEAVE_OPERATION);
        assert_eq!(error.detail_value("output_elements"), Some((i32::MAX as usize + 1).to_string().as_str()));
        assert_eq!(error.detail_value("max_output_elements"), Some(i32::MAX.to_string().as_str()));
        assert_eq!(error.detail_value("reason"), Some("output_size_exceeds_cudf_size_type"));
    }

    #[test]
    fn interleave_output_size_overflow_is_structured() {
        let error = checked_interleave_output_size(usize::MAX, 2).expect_err("shape multiplication must overflow");
        assert_eq!(error.code(), ErrorCode::SizeOverflow);
        assert_eq!(error.operation(), INTERLEAVE_OPERATION);
        assert_eq!(error.detail_value("reason"), Some("shape_product_overflow"));
    }
}
