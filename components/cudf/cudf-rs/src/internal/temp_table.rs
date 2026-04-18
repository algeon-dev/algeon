//! Temporary table-view bundle.
//!
//! [`TempTableView`] wraps a `TableView<'a>` for local call-site clarity when
//! a temporary borrowed table is only needed to feed one low-level libcudf
//! operation.

use std::sync::Arc;

use crate::{
    core::{column::ColumnView, error::Result, table::TableView},
    runtime::execution_domain::ExecutionDomain,
};

/// A temporary wrapper around a borrowed table view.
///
/// Many libcudf kernel wrappers accept a `&TableView`, and some internal
/// helpers prefer to name that temporary value rather than threading a raw
/// `TableView` expression through several branches.
pub struct TempTableView<'a> {
    table: TableView<'a>,
}

impl<'a> TempTableView<'a> {
    pub(crate) fn from_column_views(views: Vec<ColumnView<'a>>) -> Result<Self> {
        Ok(Self {
            table: TableView::from_borrowed_column_views_in_domain(views, None, "temp_table::from_column_views")?,
        })
    }

    pub(crate) fn from_column_views_in_domain(
        views: Vec<ColumnView<'a>>,
        domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Self> {
        Ok(Self { table: TableView::from_borrowed_column_views_in_domain(views, Some(domain), operation)? })
    }

    /// Returns a reference to the underlying `TableView`.
    pub fn table(&self) -> &TableView<'a> {
        &self.table
    }
}
