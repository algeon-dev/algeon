use std::sync::Arc;

use super::owned::Series;
use crate::core::{column::Column, error::Result, scalar::Scalar, types::CudfType};

impl Series {
    pub(crate) fn split_optional_host_values<T: CudfType>(data: &[Option<T>]) -> (Vec<T>, Vec<bool>, bool) {
        let mut values = Vec::with_capacity(data.len());
        let mut validity = Vec::with_capacity(data.len());
        let mut has_nulls = false;

        for value in data {
            match value {
                Some(value) => {
                    values.push(*value);
                    validity.push(true);
                },
                None => {
                    has_nulls = true;
                    // SAFETY: `CudfType` is sealed to built-in primitives with
                    // valid all-zero bit patterns.
                    values.push(unsafe { std::mem::zeroed::<T>() });
                    validity.push(false);
                },
            }
        }

        (values, validity, has_nulls)
    }

    /// Wrap an owned `Column` as a `Series`.
    ///
    /// This reads column metadata (length, null count) via FFI at construction
    /// time and caches the values for infallible metadata access.
    pub fn from_column(column: Column) -> Result<Self> {
        Self::from_column_with_cache(column)
    }

    /// Wrap a shared `Arc<Column>` as a `Series`.
    ///
    /// This reads column metadata (length, null count) via FFI at construction
    /// time and caches the values for infallible metadata access.
    pub(crate) fn from_column_arc(column: Arc<Column>) -> Result<Self> {
        let (cached_len, cached_null_count) = {
            let cv = column.view().map_err(|e| e.reanchor("series::from_column_arc"))?;
            (
                cv.len().map_err(|e| e.reanchor("series::from_column_arc"))?,
                cv.null_count().map_err(|e| e.reanchor("series::from_column_arc"))?,
            )
        };
        Ok(Self { column, cached_len, cached_null_count })
    }

    /// Create a `Series` by repeating a scalar value `size` times.
    ///
    /// If the scalar is null, all elements will be null.
    ///
    /// # Cost
    /// GPU-allocating: one device allocation of `size` elements.
    pub fn from_scalar(scalar: &Scalar, size: usize) -> Result<Self> {
        let column = Column::from_scalar(scalar, size)?;
        Self::from_column(column)
    }

    /// Private helper: wrap a `Column` and read its cached metadata.
    pub(super) fn from_column_with_cache(column: Column) -> Result<Self> {
        let (cached_len, cached_null_count) = {
            let cv = column.view()?;
            (cv.len()?, cv.null_count()?)
        };
        Ok(Self { column: Arc::new(column), cached_len, cached_null_count })
    }
}
