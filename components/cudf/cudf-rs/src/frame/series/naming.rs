use std::sync::Arc;

use super::owned::Series;
use crate::core::{column::Column, error::Result};

impl Series {
    /// Returns the name of this series, if any.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Returns a new `Series` with the given name, sharing the same GPU data.
    ///
    /// **Name authority note:** When a Series is inserted into a DataFrame
    /// via [`DataFrame::with_column`](crate::DataFrame::with_column),
    /// [`DataFrame::assign`](crate::DataFrame::assign), or
    /// [`DataFrame::from_series`](crate::DataFrame::from_series), the
    /// *parameter name* controls the column name in the DataFrame — this
    /// Series name is ignored. `rename` is useful for standalone Series
    /// operations (display, serialization) but does not propagate into
    /// DataFrame schemas.
    ///
    /// # Cost
    /// Metadata-only: zero GPU cost, only an `Arc::clone`.
    pub fn rename(&self, name: impl Into<String>) -> Self {
        Self {
            name: Some(name.into()),
            column: Arc::clone(&self.column),
            cached_len: self.cached_len,
            cached_null_count: self.cached_null_count,
        }
    }

    /// Wrap a result `Column` as a `Series`, inheriting `self`'s name.
    pub(super) fn with_inherited_name(&self, col: Column) -> Result<Self> {
        let mut s = Self::from_column_with_cache(col)?;
        s.name = self.name.clone();
        Ok(s)
    }
}
