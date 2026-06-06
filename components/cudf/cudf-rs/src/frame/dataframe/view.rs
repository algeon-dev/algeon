//! A named, borrowed view of all columns in a DataFrame.
//!
//! [`DataFrameView`] provides read-only, named access to column views without
//! copying GPU data. It is the borrow counterpart of [`DataFrame`](crate::DataFrame).
//!
//! # Column access
//!
//! - [`series_view()`](DataFrameView::series_view) — strict access, returns `Err` on miss
//!   (consistent with [`DataFrame::series_view()`](crate::DataFrame::series_view)).
//! - [`get()`](DataFrameView::get) — probe access, returns `Ok(None)` on miss.
//! - [`contains()`](DataFrameView::contains) — existence check, returns `bool`.

use std::sync::Arc;

use crate::{
    core::{
        column::{Column, ColumnView},
        error::{Error, ErrorKind, Result},
    },
    frame::series::SeriesView,
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
    pub(super) fn new(columns: Vec<(&'a str, &'a Arc<Column>)>) -> Self {
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

    /// Look up a column by name, returning a [`SeriesView`].
    ///
    /// Returns `Err` when no column with the given name exists, consistent with
    /// [`DataFrame::series_view()`](crate::DataFrame::series_view).
    ///
    /// Each call creates a fresh FFI view handle; the returned `SeriesView`
    /// borrows from the underlying `Column`, not from `self`.
    pub fn series_view(&self, name: &str) -> Result<SeriesView<'a>> {
        let (n, col) = self.columns.iter().find(|(n, _)| *n == name).ok_or_else(|| {
            Error::new(ErrorKind::InvalidOperation, format!("column '{}' not found in schema", name))
                .with_operation("dataframe_view::series_view")
                .with_context("column_name", name.to_string())
        })?;
        let cv = col.view()?;
        Ok(SeriesView::new(Some(n), cv))
    }

    /// Probe for a column by name, returning `Ok(None)` on miss.
    ///
    /// Use this when a missing column is a normal case, not an error.
    /// For strict access that errors on miss, use
    /// [`series_view()`](Self::series_view).
    ///
    /// Returns `Err` only if the column exists but creating its FFI view fails
    /// (e.g. a GPU error).
    pub fn get(&self, name: &str) -> Result<Option<SeriesView<'a>>> {
        match self.columns.iter().find(|(n, _)| *n == name) {
            None => Ok(None),
            Some((n, col)) => {
                let cv = col.view()?;
                Ok(Some(SeriesView::new(Some(n), cv)))
            },
        }
    }

    /// Returns `true` if a column with the given name exists.
    pub fn contains(&self, name: &str) -> bool {
        self.columns.iter().any(|(n, _)| *n == name)
    }

    /// Core-level iteration. Most users should prefer [`iter_series()`](Self::iter_series).
    ///
    /// Iterates over `(name, &Column)` pairs in schema order.
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

    /// Iterate over named [`SeriesView`]s in schema order.
    ///
    /// This is the high-level counterpart of [`iter()`](Self::iter) and
    /// [`iter_views()`](Self::iter_views). Each item creates a fresh FFI view
    /// handle; no GPU data is copied.
    pub fn iter_series(&self) -> impl Iterator<Item = Result<SeriesView<'a>>> + '_ {
        self.columns.iter().map(|(name, col)| {
            let cv = col.view()?;
            Ok(SeriesView::new(Some(name), cv))
        })
    }
}
