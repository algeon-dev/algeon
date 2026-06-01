//! A named, borrowed view of all columns in a DataFrame.
//!
//! [`DataFrameView`] provides read-only, named access to column views without
//! copying GPU data. It is the borrow counterpart of [`DataFrame`](crate::DataFrame).

use std::sync::Arc;

use crate::{
    column::{Column, ColumnView},
    error::Result,
    series_view::SeriesView,
};

/// A named, non-owning view of a DataFrame's columns.
///
/// `DataFrameView` stores `(name, Arc<Column>)` references in schema order,
/// providing named column lookup and iteration. No GPU data is copied.
///
/// Obtain a `DataFrameView` from [`DataFrame::view()`](crate::DataFrame::view).
pub struct DataFrameView<'a> {
    columns: Vec<(&'a str, &'a Arc<Column>)>,
}

impl<'a> DataFrameView<'a> {
    /// Create a new `DataFrameView` from named column references.
    pub(crate) fn new(columns: Vec<(&'a str, &'a Arc<Column>)>) -> Self {
        Self { columns }
    }

    /// Returns the number of columns.
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// Returns the column names in schema order.
    pub fn column_names(&self) -> Vec<&str> {
        self.columns.iter().map(|(name, _)| *name).collect()
    }

    /// Look up a column by name, returning a [`SeriesView`] if found.
    ///
    /// Each call creates a fresh FFI view handle; the returned `SeriesView`
    /// borrows from the underlying `Column`, not from `self`.
    pub fn column(&self, name: &str) -> Option<SeriesView<'a>> {
        self.columns.iter().find(|(n, _)| *n == name).and_then(|(n, col)| {
            let cv = col.view().ok()?;
            Some(SeriesView::new(Some(n), cv))
        })
    }

    /// Iterate over `(name, &Column)` pairs in schema order.
    ///
    /// Use [`column()`](Self::column) when you need a `SeriesView` with
    /// column-level operations via Deref.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Column)> {
        self.columns.iter().map(|(name, col)| (*name, col.as_ref()))
    }

    /// Iterate over named column views in schema order.
    ///
    /// Each item creates a fresh FFI `ColumnView` handle.
    pub fn iter_views(&self) -> impl Iterator<Item = Result<(&str, ColumnView<'a>)>> + '_ {
        self.columns.iter().map(|(name, col)| {
            let cv = col.view()?;
            Ok((*name, cv))
        })
    }
}
