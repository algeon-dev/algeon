//! DataFrame reassembly helpers.
//!
//! These convert raw `Table` + name vectors back into `DataFrame` values
//! after GPU kernel execution. They are used by filter, sort, join, and
//! other operations that produce a new `Table` and need to rewrap it with
//! the original (or derived) column names.

use std::sync::Arc;

use indexmap::IndexMap;

use crate::{
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

    let domain = crate::dataframe::derive_frame_domain_from_columns(
        &map,
        "internal::reassemble::dataframe_from_table_and_names",
    )?;

    Ok(DataFrame::from_raw_parts(map, row_count as usize, domain))
}
