use std::sync::Arc;

use super::owned::Series;
use crate::{
    core::{column::Column, error::Result, types::CudfType},
    runtime::Runtime,
};

impl Series {
    /// Create a `Series` from a host slice of fixed-width elements.
    pub fn new<T: CudfType>(name: impl Into<String>, data: &[T]) -> Result<Self> {
        let column = Column::from_slice(data)?;
        let mut s = Self::from_column_with_cache(column)?;
        s.name = Some(name.into());
        Ok(s)
    }

    /// Create a `Series` from a host slice inside the given explicit runtime.
    pub fn new_with_runtime<T: CudfType>(runtime: &Runtime, name: impl Into<String>, data: &[T]) -> Result<Self> {
        let column = Column::from_slice_with_runtime(runtime, data)?;
        let mut s = Self::from_column_with_cache(column)?;
        s.name = Some(name.into());
        Ok(s)
    }

    /// Create a nullable `Series` from host data and a validity mask.
    ///
    /// `validity[i]` is `true` if element `i` is valid (non-null).
    pub fn new_nullable<T: CudfType>(name: impl Into<String>, data: &[T], validity: &[bool]) -> Result<Self> {
        let column = Column::from_slice_nullable(data, validity)?;
        let mut s = Self::from_column_with_cache(column)?;
        s.name = Some(name.into());
        Ok(s)
    }

    /// Create a nullable `Series` inside the given explicit runtime.
    pub fn new_nullable_with_runtime<T: CudfType>(
        runtime: &Runtime,
        name: impl Into<String>,
        data: &[T],
        validity: &[bool],
    ) -> Result<Self> {
        let column = Column::from_slice_nullable_with_runtime(runtime, data, validity)?;
        let mut s = Self::from_column_with_cache(column)?;
        s.name = Some(name.into());
        Ok(s)
    }

    /// Create a `Series` of UTF-8 strings from host data.
    pub fn from_strings(name: impl Into<String>, data: &[&str]) -> Result<Self> {
        let column = Column::from_strings(data)?;
        let mut s = Self::from_column_with_cache(column)?;
        s.name = Some(name.into());
        Ok(s)
    }

    /// Create a `Series` of UTF-8 strings from host data inside the given explicit runtime.
    pub fn from_strings_with_runtime(runtime: &Runtime, name: impl Into<String>, data: &[&str]) -> Result<Self> {
        let column = Column::from_strings_with_runtime(runtime, data)?;
        let mut s = Self::from_column_with_cache(column)?;
        s.name = Some(name.into());
        Ok(s)
    }

    /// Create a nullable `Series` of UTF-8 strings from host data and a validity mask.
    pub fn from_strings_nullable(name: impl Into<String>, data: &[&str], validity: &[bool]) -> Result<Self> {
        let column = Column::from_strings_nullable(data, validity)?;
        let mut s = Self::from_column_with_cache(column)?;
        s.name = Some(name.into());
        Ok(s)
    }

    /// Create a nullable `Series` of UTF-8 strings inside the given explicit runtime.
    pub fn from_strings_nullable_with_runtime(
        runtime: &Runtime,
        name: impl Into<String>,
        data: &[&str],
        validity: &[bool],
    ) -> Result<Self> {
        let column = Column::from_strings_nullable_with_runtime(runtime, data, validity)?;
        let mut s = Self::from_column_with_cache(column)?;
        s.name = Some(name.into());
        Ok(s)
    }

    /// Wrap an owned `Column` as an unnamed `Series`.
    ///
    /// This reads column metadata (length, null count) via FFI at construction
    /// time and caches the values for infallible metadata access.
    pub fn from_column(column: Column) -> Result<Self> {
        Self::from_column_with_cache(column)
    }

    /// Wrap a shared `Arc<Column>` as an unnamed `Series`.
    ///
    /// This reads column metadata (length, null count) via FFI at construction
    /// time and caches the values for infallible metadata access.
    pub fn from_column_arc(column: Arc<Column>) -> Result<Self> {
        let (cached_len, cached_null_count) = {
            let cv = column.view()?;
            (cv.len()?, cv.null_count()? as usize)
        };
        Ok(Self { name: None, column, cached_len, cached_null_count })
    }

    /// Private helper: wrap a `Column` and read its cached metadata.
    pub(super) fn from_column_with_cache(column: Column) -> Result<Self> {
        let (cached_len, cached_null_count) = {
            let cv = column.view()?;
            (cv.len()?, cv.null_count()? as usize)
        };
        Ok(Self { name: None, column: Arc::new(column), cached_len, cached_null_count })
    }
}
