use std::sync::Arc;

use crate::{
    core::{column::ColumnView, error::Result, table::TableView},
    internal::temp_table::TempTableView,
    runtime::execution_domain::ExecutionDomain,
};

/// Public zero-copy bridge from [`DataFrame`](super::DataFrame) to
/// low-level [`TableView`].
///
/// This advanced helper keeps the intermediate [`ColumnView`] values alive so
/// callers can safely borrow the resulting [`TableView`] and pass it to
/// low-level `cudf::ops::*` APIs without copying device column data.
pub struct DataFrameTableView<'a> {
    bundle: TempTableView<'a>,
}

impl<'a> DataFrameTableView<'a> {
    pub(crate) fn from_column_views_in_domain(
        views: Vec<ColumnView<'a>>,
        domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Self> {
        Ok(Self { bundle: TempTableView::from_column_views_in_domain(views, domain, operation)? })
    }

    /// Returns the borrowed low-level [`TableView`].
    pub fn table(&self) -> &TableView<'a> {
        self.bundle.table()
    }
}

impl std::fmt::Debug for DataFrameTableView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DataFrameTableView").field("table", self.table()).finish()
    }
}
