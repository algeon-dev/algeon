//! Self-borrowing temporary table view.
//!
//! [`TempTableView`] owns a `Vec<ColumnView<'a>>` *and* the `TableView<'a>`
//! built from them. This solves the common lifetime problem where column
//! views must outlive the table view they are composed into — callers can
//! hold a single `TempTableView` instead of juggling two separate values.

use crate::{column::ColumnView, error::Result, table::TableView};

/// A temporary table view that owns its column views.
///
/// Many libcudf kernel wrappers accept a `&TableView`, which borrows from
/// one or more `ColumnView`s. Without this helper, callers need to keep the
/// `Vec<ColumnView>` alive alongside the `TableView`, leading to awkward
/// `let (_views, tv) = …;` patterns.
///
/// `TempTableView` bundles both into one value: the column views are stored
/// in `_views` and the table view borrows from them. Because both live in
/// the same struct and `_views` is declared before `table`, Rust's drop
/// order guarantees that `table` is dropped before `_views`.
pub(crate) struct TempTableView<'a> {
    /// Column views that the table view borrows from.
    /// Must be declared before `table` for correct drop order.
    _views: Vec<ColumnView<'a>>,
    /// The table view composed from `_views`.
    table: TableView<'a>,
}

impl<'a> TempTableView<'a> {
    /// Build a `TempTableView` from a pre-constructed vector of column views.
    ///
    /// The column views are moved into the struct and the `TableView` is
    /// built from references to them.
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
