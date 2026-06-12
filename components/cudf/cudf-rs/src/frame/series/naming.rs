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
    /// **Name authority note:** `Series` names are metadata, not DataFrame
    /// schema. When a Series is inserted into a DataFrame via
    /// [`DataFrame::from_columns`](crate::DataFrame::from_columns),
    /// [`DataFrame::with_column`](crate::DataFrame::with_column), or
    /// [`DataFrame::assign`](crate::DataFrame::assign), the explicit
    /// DataFrame-side name controls the resulting column name.
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
