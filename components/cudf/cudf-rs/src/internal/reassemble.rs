//! DataFrame reassembly helpers.
//!
//! These convert raw `Table` + name vectors back into `DataFrame` values
//! after GPU kernel execution. They are used by filter, sort, join, and
//! other operations that produce a new `Table` and need to rewrap it with
//! the original (or derived) column names.

use std::sync::Arc;

use crate::{
    core::{column::Column, error::Result, schema, table::Table},
    frame::dataframe::{DataFrame, resolve_frame_domain_from_authoritative_source},
    internal::hash::{FastIndexMap, fast_index_map_with_capacity},
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
    let table_domain = Arc::clone(table.execution_domain());

    let cols = table.into_columns()?;
    if cols.len() != names.len() {
        return Err(schema::column_name_count_mismatch_error(
            "internal::reassemble::dataframe_from_table_and_names",
            "column count does not match name count",
            names.len(),
            cols.len(),
        ));
    }

    let mut map: FastIndexMap<String, Arc<Column>> = fast_index_map_with_capacity(names.len());
    for (name, col) in names.into_iter().zip(cols) {
        map.insert(name, Arc::new(col));
    }

    let domain = resolve_frame_domain_from_authoritative_source(
        &map,
        &table_domain,
        "internal::reassemble::dataframe_from_table_and_names",
    )?;

    Ok(DataFrame::from_raw_parts(map, row_count, domain))
}
