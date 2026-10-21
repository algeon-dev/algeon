//! GPU-accelerated concatenation of columns and tables.
//!
//! # Example
//!
//! ```no_run
//! use cudf::{Exec, Runtime, Series};
//!
//! let rt = Runtime::new().unwrap();
//! let a = rt.series(&[1i32, 2]).unwrap();
//! let b = rt.series(&[3i32, 4, 5]).unwrap();
//!
//! let combined = Series::concat(&[&a, &b], Exec::RETAINED).unwrap();
//! assert_eq!(combined.to_vec_strict::<i32>().unwrap(), vec![1, 2, 3, 4, 5]);
//! ```

use std::{ptr, sync::Arc};

use crate::{
    core::{
        column::{Column, ColumnView},
        error::{self, Error, ErrorCode, FACT_COLUMN_INDEX, FACT_LEFT_TYPE_ID, FACT_RIGHT_TYPE_ID, Result},
        table::{Table, TableView},
        types::TypeId,
    },
    runtime::{execution_context::Exec, execution_domain::ExecutionDomain, execution_resolver::ResolvedExecution},
};

fn checked_len_to_i32(
    len: usize,
    operation: &'static str,
    message: &'static str,
    resource: &'static str,
    value_key: &'static str,
    remedy: &'static str,
) -> Result<i32> {
    i32::try_from(len)
        .map_err(|_| error::limit_validation_error(operation, message, resource, value_key, "i32::MAX", len, remedy))
}

#[cfg(any(test, feature = "test-hooks"))]
fn input_column_count_limit_error(operation: &'static str, column_count: usize) -> Error {
    error::limit_validation_error(
        operation,
        "number of input columns exceeds i32::MAX",
        "input_columns",
        "column_count",
        "i32::MAX",
        column_count,
        "reduce column_count to fit within i32::MAX",
    )
}

#[cfg(any(test, feature = "test-hooks"))]
fn input_table_count_limit_error(operation: &'static str, table_count: usize) -> Error {
    error::limit_validation_error(
        operation,
        "number of input tables exceeds i32::MAX",
        "input_tables",
        "table_count",
        "i32::MAX",
        table_count,
        "reduce table_count to fit within i32::MAX",
    )
}

fn concatenate_requires_input_error(operation: &'static str, input_kind: &'static str) -> Error {
    Error::new_with(
        ErrorCode::UnsupportedOperation,
        operation,
        format!("{operation} requires at least one input {input_kind}"),
    )
    .with_detail("reason", "empty_input")
    .with_detail("remedy", format!("provide at least one input {input_kind}"))
}

fn validate_column_types(columns: &[&ColumnView<'_>], operation: &'static str) -> Result<()> {
    let Some((first, rest)) = columns.split_first() else {
        return Ok(());
    };
    let left_type = first.data_type().map_err(|e| e.reanchor(operation))?.id();

    for (offset, column) in rest.iter().enumerate() {
        let right_type = column.data_type().map_err(|e| e.reanchor(operation))?.id();
        if left_type != right_type {
            return Err(Error::new_with(
                ErrorCode::TypeMismatch,
                operation,
                "all input columns must have the same data type for concatenation",
            )
            .with_fact(FACT_COLUMN_INDEX, offset + 1)
            .with_fact(FACT_LEFT_TYPE_ID, left_type)
            .with_fact(FACT_RIGHT_TYPE_ID, right_type));
        }
    }

    Ok(())
}

fn validate_table_column_types(tables: &[&TableView<'_>], operation: &'static str) -> Result<()> {
    let Some((first, rest)) = tables.split_first() else {
        return Ok(());
    };
    let column_count = first.num_columns().map_err(|e| e.reanchor(operation))?;

    for table in rest {
        let table_column_count = table.num_columns().map_err(|e| e.reanchor(operation))?;
        if table_column_count != column_count {
            return Ok(());
        }

        for index in 0..column_count {
            let left_type =
                first.column(index).and_then(|column| column.data_type()).map_err(|e| e.reanchor(operation))?.id();
            let right_type =
                table.column(index).and_then(|column| column.data_type()).map_err(|e| e.reanchor(operation))?.id();

            if left_type != right_type {
                return Err(Error::new_with(
                    ErrorCode::TypeMismatch,
                    operation,
                    "all input tables must have matching column data types for concatenation",
                )
                .with_fact(FACT_COLUMN_INDEX, index)
                .with_fact(FACT_LEFT_TYPE_ID, left_type)
                .with_fact(FACT_RIGHT_TYPE_ID, right_type));
            }
        }
    }

    Ok(())
}

/// Child index of a libcudf list column's values; child 0 holds its offsets.
const LIST_VALUES_CHILD: usize = 1;

/// Element counts of one column: its length, then the counted children
/// beneath it. A list counts its values child and a struct counts each field.
/// A string column counts no child, since libcudf stores large string
/// characters with 64-bit offsets.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ColumnElementCounts {
    elements: usize,
    children: Vec<ColumnElementCounts>,
}

impl ColumnElementCounts {
    /// Reads the counts from view metadata. With `exact` false, a list counts
    /// its whole values child, which overstates a sliced list and needs no
    /// device read. With `exact` true, a list counts only the values its rows
    /// reference, which reads two offsets per list level from the device.
    fn of(view: &ColumnView<'_>, exact: bool) -> Result<Self> {
        let children = match view.data_type()?.id() {
            TypeId::List => {
                let values = if exact { view.list_sliced_child()? } else { view.child(LIST_VALUES_CHILD)? };
                vec![Self::of(&values, exact)?]
            },
            TypeId::Struct => {
                (0..view.num_children()?).map(|index| Self::of(&view.child(index)?, exact)).collect::<Result<_>>()?
            },
            _ => Vec::new(),
        };
        Ok(Self { elements: view.len()?, children })
    }
}

/// A level of one concatenated column whose summed element count exceeds the
/// per-column limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ConcatenatedElementOverflow {
    /// Rows of the concatenated column.
    rows: u64,
    /// Elements at the overflowing level, summed over the inputs.
    elements: u64,
    /// Nesting depth of that level; 0 is the row count itself.
    depth: usize,
}

impl ConcatenatedElementOverflow {
    /// Largest row count whose elements at this level stay within `limit`,
    /// keeping the concatenated column's elements per row.
    fn row_cap(self, limit: usize) -> u64 {
        let limit = limit as u64;
        let row_cap = u128::from(self.rows) * u128::from(limit) / u128::from(self.elements.max(1));
        u64::try_from(row_cap).unwrap_or(u64::MAX).min(limit)
    }
}

/// The first level, depth first, at which `inputs` (one column's counts per
/// concatenation input) sum above `limit`. Inputs share one type, so their
/// count trees share one shape.
fn first_concatenated_element_overflow(
    inputs: &[ColumnElementCounts],
    limit: usize,
) -> Option<ConcatenatedElementOverflow> {
    fn visit(
        levels: &[&ColumnElementCounts],
        rows: u64,
        depth: usize,
        limit: usize,
    ) -> Option<ConcatenatedElementOverflow> {
        let elements = levels.iter().map(|level| level.elements as u64).fold(0u64, u64::saturating_add);
        if elements > limit as u64 {
            return Some(ConcatenatedElementOverflow { rows, elements, depth });
        }
        let child_count = levels.first().map_or(0, |level| level.children.len());
        (0..child_count).find_map(|index| {
            let children = levels.iter().filter_map(|level| level.children.get(index)).collect::<Vec<_>>();
            visit(&children, rows, depth + 1, limit)
        })
    }
    let roots = inputs.iter().collect::<Vec<_>>();
    let rows = inputs.iter().map(|input| input.elements as u64).fold(0u64, u64::saturating_add);
    visit(&roots, rows, 0, limit)
}

fn concatenated_column_element_limit_error(
    operation: &'static str,
    column: Option<(usize, Option<&str>)>,
    overflow: ConcatenatedElementOverflow,
) -> Error {
    let row_cap = overflow.row_cap(Column::MAX_ELEMENTS);
    let error = Error::new_with(
        ErrorCode::SizeOverflow,
        operation,
        "concatenated column holds more elements than one libcudf column can index",
    );
    let error = match column {
        Some((index, Some(name))) => error.with_fact(FACT_COLUMN_INDEX, index).with_column(name.to_owned()),
        Some((index, None)) => error.with_fact(FACT_COLUMN_INDEX, index),
        None => error,
    };
    error
        .with_detail("element_count", overflow.elements.to_string())
        .with_detail("element_limit", Column::MAX_ELEMENTS.to_string())
        .with_detail("nesting_depth", overflow.depth.to_string())
        .with_detail("row_count", overflow.rows.to_string())
        .with_detail("row_cap", row_cap.to_string())
        .with_detail("reason", "concatenated_column_elements_exceed_cudf_size_type")
        .with_detail("remedy", format!("keep each concatenated relation at or below {row_cap} rows"))
}

/// Rejects a concatenation whose output column would exceed
/// [`Column::MAX_ELEMENTS`] at any level, before libcudf allocates it.
///
/// The first pass reads view metadata only. A list input sliced from a larger
/// column can overstate its count there, so a pass that finds an overflow is
/// repeated with exact counts, which read list offsets from the device.
fn check_concatenated_column_elements(
    inputs: &[&ColumnView<'_>],
    operation: &'static str,
    column: Option<(usize, Option<&str>)>,
) -> Result<()> {
    let counts = |exact| {
        inputs
            .iter()
            .map(|view| ColumnElementCounts::of(view, exact))
            .collect::<Result<Vec<_>>>()
            .map_err(|error| error.reanchor(operation))
    };
    if first_concatenated_element_overflow(&counts(false)?, Column::MAX_ELEMENTS).is_none() {
        return Ok(());
    }
    match first_concatenated_element_overflow(&counts(true)?, Column::MAX_ELEMENTS) {
        Some(overflow) => Err(concatenated_column_element_limit_error(operation, column, overflow)),
        None => Ok(()),
    }
}

fn compatible_column_execution(columns: &[&ColumnView<'_>]) -> Result<ResolvedExecution> {
    let Some(first) = columns.first() else {
        return Err(concatenate_requires_input_error("concatenate::concatenate_columns", "column"));
    };
    let ctx = Exec::RETAINED;
    ResolvedExecution::many(
        first.execution_domain(),
        columns[1..].iter().map(|column| column.execution_domain()),
        ctx,
        "concatenate::concatenate_columns",
    )
}

/// Concatenate multiple columns into a single column.
///
/// All columns must have the same data type.
pub(crate) fn concatenate_columns(columns: &[&ColumnView<'_>]) -> Result<Column> {
    let execution = compatible_column_execution(columns)?;
    concatenate_columns_with_exec(columns, execution.output_domain(), execution.exec_options())
}

pub(crate) fn concatenate_columns_with_exec(
    columns: &[&ColumnView<'_>],
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
) -> Result<Column> {
    validate_column_types(columns, "concatenate::concatenate_columns")?;
    check_concatenated_column_elements(columns, "concatenate::concatenate_columns", None)?;
    let handles: Vec<cudf_sys::cudf_column_view_handle> = columns.iter().map(|c| c.raw_handle()).collect();
    let num_columns = checked_len_to_i32(
        handles.len(),
        "concatenate::concatenate_columns",
        "number of input columns exceeds i32::MAX",
        "input_columns",
        "column_count",
        "reduce column_count to fit within i32::MAX",
    )?;
    let mut out = ptr::null_mut();
    error::check_status_in_with_details(
        // SAFETY: All column view handles are valid (NonNull maintained by ColumnView).
        // handles slice is valid for handles.len() elements.
        unsafe { cudf_sys::cudf_concatenate_columns_ex(handles.as_ptr(), num_columns, exec_options, &mut out) },
        "concatenate::concatenate_columns",
        [("num_columns", handles.len().to_string())],
    )?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    unsafe { Column::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

/// Concatenates `tables` row-wise. `column_names` names the shared columns in
/// order for diagnostics and may be shorter than the column count.
pub(crate) fn concatenate_tables_with_exec(
    tables: &[&TableView<'_>],
    column_names: &[String],
    output_domain: &Arc<ExecutionDomain>,
    exec_options: &cudf_sys::cudf_exec_options_t,
) -> Result<Table> {
    let operation = "concatenate::concatenate_tables";
    validate_table_column_types(tables, operation)?;
    check_concatenated_table_elements(tables, column_names, operation)?;
    let handles: Vec<cudf_sys::cudf_table_view_handle> = tables.iter().map(|t| t.raw_handle()).collect();
    let num_tables = checked_len_to_i32(
        handles.len(),
        "concatenate::concatenate_tables",
        "number of input tables exceeds i32::MAX",
        "input_tables",
        "table_count",
        "reduce table_count to fit within i32::MAX",
    )?;
    let mut out = ptr::null_mut();
    error::check_status_in_with_details(
        // SAFETY: All table view handles are valid (NonNull maintained by TableView).
        // handles slice is valid for handles.len() elements.
        unsafe { cudf_sys::cudf_concatenate_tables_ex(handles.as_ptr(), num_tables, exec_options, &mut out) },
        "concatenate::concatenate_tables",
        [("num_tables", handles.len().to_string())],
    )?;
    // SAFETY: out was returned by a successful FFI call, transferring ownership.
    unsafe { Table::from_raw_with_domain(out, Arc::clone(output_domain)) }
}

/// Applies [`check_concatenated_column_elements`] to each column position of
/// `tables`. Tables whose column counts differ are left to libcudf, which
/// rejects them.
fn check_concatenated_table_elements(
    tables: &[&TableView<'_>],
    column_names: &[String],
    operation: &'static str,
) -> Result<()> {
    let Some(first) = tables.first() else {
        return Ok(());
    };
    let column_count = first.num_columns().map_err(|error| error.reanchor(operation))?;
    for table in tables {
        if table.num_columns().map_err(|error| error.reanchor(operation))? != column_count {
            return Ok(());
        }
    }
    for index in 0..column_count {
        let views = tables
            .iter()
            .map(|table| table.column(index).map_err(|error| error.reanchor(operation)))
            .collect::<Result<Vec<_>>>()?;
        let view_refs = views.iter().collect::<Vec<_>>();
        let name = column_names.get(index).map(String::as_str);
        check_concatenated_column_elements(&view_refs, operation, Some((index, name)))?;
    }
    Ok(())
}

#[cfg(any(test, feature = "test-hooks"))]
pub(crate) fn trigger_concatenate_column_count_limit_error_for_test() -> Result<()> {
    Err(input_column_count_limit_error("concatenate::concatenate_columns", usize::MAX))
}

#[cfg(any(test, feature = "test-hooks"))]
pub(crate) fn trigger_concatenate_table_count_limit_error_for_test() -> Result<()> {
    Err(input_table_count_limit_error("concatenate::concatenate_tables", usize::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::error::{CudfErrorClassification, FactKeyName, classify_cudf_error};

    const VECTOR_WIDTH: usize = 2048;

    fn vector_frame(rows: usize) -> ColumnElementCounts {
        ColumnElementCounts {
            elements: rows,
            children: vec![ColumnElementCounts { elements: rows * VECTOR_WIDTH, children: Vec::new() }],
        }
    }

    fn flat_frame(rows: usize) -> ColumnElementCounts {
        ColumnElementCounts { elements: rows, children: Vec::new() }
    }

    #[test]
    fn concatenated_vector_frames_overflow_at_the_list_child_with_the_width_row_cap() {
        let frames = vec![vector_frame(200_000); 15];

        let overflow = first_concatenated_element_overflow(&frames, Column::MAX_ELEMENTS)
            .expect("3,000,000 rows of 2048 values exceed the per-column limit");
        assert_eq!(
            overflow,
            ConcatenatedElementOverflow { rows: 3_000_000, elements: 3_000_000 * VECTOR_WIDTH as u64, depth: 1 }
        );

        let error = concatenated_column_element_limit_error("concatenate::test", Some((1, Some("fp"))), overflow);
        assert_eq!(classify_cudf_error(&error), CudfErrorClassification::SizeOverflow);
        assert_eq!(error.fact_value(FactKeyName::ColumnName).map(ToString::to_string).as_deref(), Some("fp"));
        assert_eq!(error.detail_value("element_count"), Some("6144000000"));
        assert_eq!(error.detail_value("element_limit"), Some("2147483647"));
        assert_eq!(error.detail_value("row_cap"), Some("1048575"));
    }

    #[test]
    fn concatenation_up_to_the_width_row_cap_is_accepted_and_one_row_more_is_rejected() {
        let row_cap = Column::MAX_ELEMENTS / VECTOR_WIDTH;
        let at_cap = [vector_frame(row_cap - 1), vector_frame(1)];
        assert_eq!(first_concatenated_element_overflow(&at_cap, Column::MAX_ELEMENTS), None);

        let past_cap = [vector_frame(row_cap), vector_frame(1)];
        let overflow = first_concatenated_element_overflow(&past_cap, Column::MAX_ELEMENTS)
            .expect("one row past the cap exceeds the per-column limit");
        assert_eq!(overflow.depth, 1);
        assert_eq!(overflow.row_cap(Column::MAX_ELEMENTS), row_cap as u64);
    }

    #[test]
    fn concatenated_flat_rows_overflow_at_the_row_count() {
        let half = Column::MAX_ELEMENTS / 2 + 1;
        let overflow = first_concatenated_element_overflow(&[flat_frame(half), flat_frame(half)], Column::MAX_ELEMENTS)
            .expect("rows past the limit overflow");

        assert_eq!(overflow.depth, 0);
        assert_eq!(overflow.elements, 2 * half as u64);
        assert_eq!(overflow.row_cap(Column::MAX_ELEMENTS), Column::MAX_ELEMENTS as u64);
    }
}
