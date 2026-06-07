use crate::{
    core::{column::ColumnView, error::Result, table::TableView},
    internal::temp_table::TempTableView,
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
    pub(crate) fn from_column_views(views: Vec<ColumnView<'a>>) -> Result<Self> {
        Ok(Self { bundle: TempTableView::from_column_views(views)? })
    }

    /// Returns the borrowed low-level [`TableView`].
    pub fn table(&self) -> &TableView<'a> {
        self.bundle.table()
    }
}
