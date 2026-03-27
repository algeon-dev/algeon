//! Temporary table-view bundle.
//!
//! [`TempTableView`] owns a `Vec<ColumnView<'a>>` alongside a `TableView<'a>`
//! built from equivalent references. This solves the common lifetime problem
//! where callers would otherwise need to keep both values alive separately —
//! callers can hold a single `TempTableView` instead of juggling two values.

use crate::{column::ColumnView, error::Result, table::TableView};

/// A temporary bundle of column views plus a table view.
///
/// Many libcudf kernel wrappers accept a `&TableView`, which is typically
/// created from a temporary `Vec<ColumnView>`. Without this helper, callers
/// need to keep both the vector and the table view alive side by side,
/// leading to awkward `let (_views, tv) = …;` patterns.
///
/// `TempTableView` simply stores both values together so they can be passed
/// around as one unit.
pub(crate) struct TempTableView<'a> {
    /// Column views kept alive for the lifetime of this bundle.
    _views: Vec<ColumnView<'a>>,
    /// The table view composed from equivalent column-view references.
    table: TableView<'a>,
}

impl<'a> TempTableView<'a> {
    /// Build a `TempTableView` from a pre-constructed vector of column views.
    ///
    /// The column views and resulting `TableView` are stored together so
    /// callers can carry them as a single value.
    ///
    /// # Errors
    ///
    /// Returns an error if `TableView::from_column_views` fails (e.g., the
    /// number of views exceeds `i32::MAX`).
    pub(crate) fn from_column_views(views: Vec<ColumnView<'a>>) -> Result<Self> {
        let view_refs: Vec<&ColumnView<'a>> = views.iter().collect();
        let table = TableView::from_column_views(&view_refs)?;
        Ok(Self { _views: views, table })
    }

    /// Returns a reference to the underlying `TableView`.
    pub(crate) fn table(&self) -> &TableView<'a> {
        &self.table
    }
}
