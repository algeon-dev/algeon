//! DataFrame reassembly helpers.
//!
//! These convert raw `Table` + name vectors back into `DataFrame` values
//! after GPU kernel execution. They are used by filter, sort, join, and
//! other operations that produce a new `Table` and need to rewrap it with
//! the original (or derived) column names.

use std::sync::Arc;

use indexmap::IndexMap;

use crate::{
    column::Column,
    dataframe::DataFrame,
    error::{Error, ErrorKind, Result},
    table::Table,
};

/// Create a `DataFrame` from a `Table` and a vector of column names.
///
/// The table is decomposed into individual `Arc<Column>` values paired
/// with the given names. This is the standard reassembly path after GPU
/// kernel execution.
///
/// # Errors
///
/// Returns an error if the column count of the table does not match the
/// number of names.
pub(crate) fn dataframe_from_table_and_names(table: Table, names: Vec<String>) -> Result<DataFrame> {
    let view = table.view()?;
    let row_count = if view.num_columns()? > 0 { view.num_rows()? } else { 0 };
    drop(view);

    let cols = table.into_columns()?;
    if cols.len() != names.len() {
        return Err(Error::new(ErrorKind::InvalidOperation, "column count does not match name count")
            .with_operation("internal::reassemble::dataframe_from_table_and_names")
            .with_context("columns_count", cols.len().to_string())
            .with_context("names_count", names.len().to_string()));
    }

    let mut map = IndexMap::with_capacity(names.len());
    for (name, col) in names.into_iter().zip(cols) {
        map.insert(name, Arc::new(col));
    }

    Ok(DataFrame::from_raw_parts(map, row_count))
}

/// Create a `DataFrame` from an iterator of `(name, Column)` pairs.
///
/// Each column is wrapped in an `Arc`. This is a convenience for cases
/// where columns are already separated (e.g., after `Table::into_columns`
/// and manual name assignment).
///
/// # Errors
///
/// Returns an error if columns have mismatched row counts.
pub(crate) fn dataframe_from_named_columns<I>(pairs: I) -> Result<DataFrame>
where I: IntoIterator<Item = (String, Column)> {
    let pairs: Vec<(String, Column)> = pairs.into_iter().collect();

    if pairs.is_empty() {
        return Ok(DataFrame::from_raw_parts(IndexMap::new(), 0));
    }

    let first_rows = pairs[0].1.view()?.len()?;
    for (i, (name, col)) in pairs.iter().enumerate().skip(1) {
        let rows = col.view()?.len()?;
        if rows != first_rows {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                format!("column '{}' has {} rows but first column has {} rows", name, rows, first_rows),
            )
            .with_operation("internal::reassemble::dataframe_from_named_columns")
            .with_context("column_index", i.to_string())
            .with_context("column_rows", rows.to_string())
            .with_context("expected_rows", first_rows.to_string()));
        }
    }

    let mut map = IndexMap::with_capacity(pairs.len());
    for (name, col) in pairs {
        map.insert(name, Arc::new(col));
    }

    Ok(DataFrame::from_raw_parts(map, first_rows))
}
