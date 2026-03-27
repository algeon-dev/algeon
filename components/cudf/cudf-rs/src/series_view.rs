//! A named, borrowed view of a GPU column.
//!
//! [`SeriesView`] is the borrow counterpart of [`Series`](crate::Series). It
//! carries an optional column name alongside a [`ColumnView`], and implements
//! `Deref<Target = ColumnView>` so that all low-level column methods are
//! available without qualification.

use std::ops::Deref;

use crate::column::ColumnView;

/// A named, non-owning view of a GPU column.
///
/// `SeriesView` wraps a [`ColumnView`] with an optional name. It implements
/// [`Deref<Target = ColumnView>`], so all `ColumnView` methods (`.len()`,
/// `.data_type()`, `.to_vec()`, `.device_data_ptr()`, etc.) are available
/// directly on `SeriesView` via auto-deref.
///
/// Obtain a `SeriesView` from:
/// - [`DataFrame::column()`](crate::DataFrame::column) — returns a named view
/// - [`Series::view()`](crate::Series::view) — returns a view with the series' name
pub struct SeriesView<'a> {
    name: Option<&'a str>,
    view: ColumnView<'a>,
}

impl<'a> SeriesView<'a> {
    /// Create a new `SeriesView` from an optional name and a `ColumnView`.
    pub(crate) fn new(name: Option<&'a str>, view: ColumnView<'a>) -> Self {
        Self { name, view }
    }

    /// Returns the column name, if any.
    pub fn name(&self) -> Option<&str> {
        self.name
    }

    /// Returns a reference to the underlying `ColumnView`.
    ///
    /// This is useful when you need an explicit `&ColumnView` reference and
    /// auto-deref is not sufficient (e.g., generic contexts).
    pub fn column_view(&self) -> &ColumnView<'a> {
        &self.view
    }
}

impl<'a> Deref for SeriesView<'a> {
    type Target = ColumnView<'a>;

    fn deref(&self) -> &ColumnView<'a> {
        &self.view
    }
}

impl std::fmt::Debug for SeriesView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ds = f.debug_struct("SeriesView");
        ds.field("name", &self.name);
        let dtype_str = self.view.data_type().map(|dt| format!("{:?}", dt.id())).unwrap_or_else(|_| "?".to_string());
        ds.field("dtype", &dtype_str);
        let len = self.view.len().unwrap_or(0);
        ds.field("len", &len);
        ds.finish()
    }
}
