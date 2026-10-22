//! Decoded element counts of the read columns of Parquet row groups.
//!
//! A backend column holds at most
//! [`BackendCapabilities::max_column_elements`](crate::backend::BackendCapabilities::max_column_elements)
//! elements at each level: its rows, and the children of every nested list.
//! Chunk planning keeps the largest read column of each chunk, and of each
//! multi-chunk batch, within that limit, so one planned read decodes into one
//! backend table.
//!
//! With a footer, a read column counts the larger of the row group's rows and
//! the `num_values` of each Parquet leaf beneath it. A leaf's `num_values`
//! has one entry per value plus one per null or empty list, which bounds the
//! element count of every list level on that leaf's path. Without a footer, a
//! column counts rows times the product of its `FixedSizeList` widths; a
//! variable-width list then has no count.

use arrow::datatypes::{DataType as ArrowDataType, Schema};
use parquet::file::metadata::ParquetMetaData;

use crate::{parquet_stats::root_column_leaf_indices, source::LocalParquetSource};

/// The read column with the most decoded elements in one row group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ColumnElements {
    /// Element count of that column at its largest level; never below the
    /// row group's row count.
    pub(crate) elements: u64,
    /// Position in the source's `read_columns` of the column holding
    /// `elements`, or `None` when no read column holds more elements than the
    /// row group has rows.
    pub(crate) read_column: Option<usize>,
}

impl ColumnElements {
    fn rows(rows: u64) -> Self {
        Self { elements: rows, read_column: None }
    }

    fn widen(self, read_column: usize, elements: u64) -> Self {
        if elements > self.elements { Self { elements, read_column: Some(read_column) } } else { self }
    }
}

/// The largest read column of every physical row group of `metadata`, from
/// leaf `num_values`. `None` when a read column has no column chunk.
pub(crate) fn row_group_column_elements_from_footer(
    metadata: &ParquetMetaData,
    source: &LocalParquetSource,
) -> Option<Vec<ColumnElements>> {
    if source.read_columns.len() != source.read_column_file_indices.len() {
        return None;
    }
    let footer_schema = metadata.file_metadata().schema_descr();
    let read_column_leaves = source
        .read_column_file_indices
        .iter()
        .map(|root_index| {
            Some(root_column_leaf_indices(footer_schema, *root_index)).filter(|leaves| !leaves.is_empty())
        })
        .collect::<Option<Vec<_>>>()?;
    metadata
        .row_groups()
        .iter()
        .map(|row_group| {
            let rows = u64::try_from(row_group.num_rows()).ok()?;
            read_column_leaves.iter().enumerate().try_fold(
                ColumnElements::rows(rows),
                |largest, (read_column, leaves)| {
                    let elements = leaves.iter().try_fold(0u64, |elements, leaf| {
                        Some(elements.max(u64::try_from(row_group.columns().get(*leaf)?.num_values()).ok()?))
                    })?;
                    Some(largest.widen(read_column, elements))
                },
            )
        })
        .collect()
}

/// The largest read column of a row group of `rows` rows, from the Arrow
/// schema alone. `None` when a read column is a variable-width list or is
/// absent from `schema`, since neither bounds its child elements per row.
pub(crate) fn row_group_column_elements_from_schema(
    schema: &Schema,
    read_columns: &[String],
    rows: u64,
) -> Option<ColumnElements> {
    read_columns.iter().enumerate().try_fold(ColumnElements::rows(rows), |largest, (read_column, column)| {
        let per_row = elements_per_row(schema.field_with_name(column).ok()?.data_type())?;
        Some(largest.widen(read_column, rows.checked_mul(per_row)?))
    })
}

/// Whether every read column has a bounded element count per row, which
/// makes a row count enough to bound a row group's elements.
pub(crate) fn read_columns_bound_elements_per_row(schema: &Schema, read_columns: &[String]) -> bool {
    read_columns
        .iter()
        .all(|column| schema.field_with_name(column).is_ok_and(|field| elements_per_row(field.data_type()).is_some()))
}

/// Largest per-row element count over the levels of one column: the product
/// of nested `FixedSizeList` widths, at least one for the row itself.
fn elements_per_row(data_type: &ArrowDataType) -> Option<u64> {
    match data_type {
        ArrowDataType::FixedSizeList(field, width) => {
            Some(u64::try_from(*width).ok()?.checked_mul(elements_per_row(field.data_type())?)?.max(1))
        },
        ArrowDataType::List(_)
        | ArrowDataType::LargeList(_)
        | ArrowDataType::ListView(_)
        | ArrowDataType::LargeListView(_)
        | ArrowDataType::Map(..) => None,
        ArrowDataType::Struct(fields) => {
            fields.iter().try_fold(1u64, |largest, field| Some(largest.max(elements_per_row(field.data_type())?)))
        },
        _ => Some(1),
    }
}

/// Largest row count at which a row group whose largest column holds
/// `elements` across `rows` rows stays within `limit`, keeping its elements
/// per row.
pub(crate) fn rows_within_element_limit(rows: u64, elements: u64, limit: u64) -> u64 {
    if elements == 0 {
        return limit;
    }
    let rows_within = u128::from(rows) * u128::from(limit) / u128::from(elements);
    u64::try_from(rows_within).unwrap_or(u64::MAX).min(limit)
}
