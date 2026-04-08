use std::sync::Arc;

use super::owned::Series;
use crate::core::{
    column::Column,
    error::Result,
    scalar::Scalar,
    types::{CudfType, TypeId},
};

impl Series {
    pub(crate) fn from_named_column(name: impl Into<String>, column: Column) -> Result<Self> {
        let mut s = Self::from_column_with_cache(column)?;
        s.name = Some(name.into());
        Ok(s)
    }

    /// Create a `Series` from a host slice of fixed-width elements.
    pub fn new<T: CudfType>(name: impl Into<String>, data: &[T]) -> Result<Self> {
        let column = Column::from_slice(data)?;
        Self::from_named_column(name, column)
    }

    /// Create a nullable `Series` from host data and a validity mask.
    ///
    /// `validity[i]` is `true` if element `i` is valid (non-null).
    pub fn new_nullable<T: CudfType>(name: impl Into<String>, data: &[T], validity: &[bool]) -> Result<Self> {
        let column = Column::from_slice_nullable(data, validity)?;
        Self::from_named_column(name, column)
    }

    /// Create a `Series` of UTF-8 strings from host data.
    pub fn from_strings(name: impl Into<String>, data: &[&str]) -> Result<Self> {
        let column = Column::from_strings(data)?;
        Self::from_named_column(name, column)
    }

    /// Create a nullable `Series` of UTF-8 strings from host data and a validity mask.
    pub fn from_strings_nullable(name: impl Into<String>, data: &[&str], validity: &[bool]) -> Result<Self> {
        let column = Column::from_strings_nullable(data, validity)?;
        Self::from_named_column(name, column)
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
    ///
    /// # Panics
    ///
    /// Panics if column metadata cannot be read via FFI. This should never
    /// happen for a well-formed `Column` (e.g. one already stored in a
    /// `DataFrame`).
    pub fn from_column_arc(column: Arc<Column>) -> Self {
        let (cached_len, cached_null_count) = {
            let cv = column.view().expect("Column must have a valid view");
            (
                cv.len().expect("Column view must report length"),
                cv.null_count().expect("Column view must report null count"),
            )
        };
        Self { name: None, column, cached_len, cached_null_count }
    }

    /// Create an all-null `Series` of the given type and length.
    ///
    /// Every element is null (invalid). The underlying column is allocated on
    /// the GPU with a zeroed data buffer and an all-null validity mask.
    ///
    /// # Cost
    /// GPU-allocating: one device allocation.
    pub fn nulls(name: impl Into<String>, type_id: TypeId, len: usize) -> Result<Self> {
        let column = Column::nulls(type_id, len)?;
        Self::from_named_column(name, column)
    }

    /// Create a `Series` by repeating a scalar value `size` times.
    ///
    /// If the scalar is null, all elements will be null.
    ///
    /// # Cost
    /// GPU-allocating: one device allocation of `size` elements.
    pub fn from_scalar(name: impl Into<String>, scalar: &Scalar, size: usize) -> Result<Self> {
        let column = Column::from_scalar(scalar, size)?;
        Self::from_named_column(name, column)
    }

    /// Create an empty (zero-length) `Series` of the given type.
    ///
    /// This is equivalent to `Series::nulls(name, type_id, 0)` but may be
    /// slightly cheaper.
    pub fn empty(name: impl Into<String>, type_id: TypeId) -> Result<Self> {
        let column = Column::empty(type_id)?;
        Self::from_named_column(name, column)
    }

    /// Private helper: wrap a `Column` and read its cached metadata.
    pub(super) fn from_column_with_cache(column: Column) -> Result<Self> {
        let (cached_len, cached_null_count) = {
            let cv = column.view()?;
            (cv.len()?, cv.null_count()?)
        };
        Ok(Self { name: None, column: Arc::new(column), cached_len, cached_null_count })
    }
}
