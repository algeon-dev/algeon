//! A named, `Arc`-shared GPU column — the primary user-facing column type.
//!
//! `Series` wraps an `Arc<Column>` with an optional name. It is the façade
//! counterpart to the low-level [`Column`] / [`ColumnView`] types.
//!
//! Clone is cheap: it bumps the `Arc` reference count — no GPU data is copied.

use std::sync::Arc;

use crate::{
    column::{Column, ColumnView},
    error::Result,
    scalar::Scalar,
    types::{CudfType, DataType},
};

/// A named, reference-counted GPU column.
///
/// `Series` is the cuDF-like user-facing column type. Internally it holds an
/// `Arc<Column>`, so cloning, renaming, and passing Series between DataFrames
/// is zero-copy on the GPU side.
#[derive(Clone)]
pub struct Series {
    name: Option<String>,
    column: Arc<Column>,
}

impl std::fmt::Debug for Series {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ds = f.debug_struct("Series");
        ds.field("name", &self.name);
        let dtype_str = self.dtype().map(|dt| format!("{:?}", dt.id())).unwrap_or_else(|_| "?".to_string());
        ds.field("dtype", &dtype_str);
        let len = self.len().unwrap_or(-1);
        ds.field("len", &len);
        ds.finish()
    }
}

impl From<Column> for Series {
    fn from(col: Column) -> Self {
        Self::from_column(col)
    }
}

impl Series {
    // -- Constructors --------------------------------------------------------

    /// Create a `Series` from a host slice of fixed-width elements.
    pub fn new<T: CudfType>(name: impl Into<String>, data: &[T]) -> Result<Self> {
        let column = Column::from_slice(data)?;
        Ok(Self { name: Some(name.into()), column: Arc::new(column) })
    }

    /// Create a nullable `Series` from host data and a validity mask.
    ///
    /// `validity[i]` is `true` if element `i` is valid (non-null).
    pub fn new_nullable<T: CudfType>(name: impl Into<String>, data: &[T], validity: &[bool]) -> Result<Self> {
        let column = Column::from_slice_nullable(data, validity)?;
        Ok(Self { name: Some(name.into()), column: Arc::new(column) })
    }

    /// Create a `Series` of UTF-8 strings from host data.
    pub fn from_strings(name: impl Into<String>, data: &[&str]) -> Result<Self> {
        let column = Column::from_strings(data)?;
        Ok(Self { name: Some(name.into()), column: Arc::new(column) })
    }

    /// Create a nullable `Series` of UTF-8 strings from host data and a validity mask.
    pub fn from_strings_nullable(name: impl Into<String>, data: &[&str], validity: &[bool]) -> Result<Self> {
        let column = Column::from_strings_nullable(data, validity)?;
        Ok(Self { name: Some(name.into()), column: Arc::new(column) })
    }

    /// Wrap an owned `Column` as an unnamed `Series`.
    pub fn from_column(column: Column) -> Self {
        Self { name: None, column: Arc::new(column) }
    }

    /// Wrap a shared `Arc<Column>` as an unnamed `Series`.
    pub fn from_column_arc(column: Arc<Column>) -> Self {
        Self { name: None, column }
    }

    // -- Metadata (zero-copy, no GPU work) -----------------------------------

    /// Returns the name of this series, if any.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Returns a new `Series` with the given name, sharing the same GPU data.
    pub fn rename(&self, name: impl Into<String>) -> Self {
        Self { name: Some(name.into()), column: Arc::clone(&self.column) }
    }

    /// Returns the data type of the underlying column.
    pub fn dtype(&self) -> Result<DataType> {
        self.column.view()?.data_type()
    }

    /// Returns the number of elements.
    pub fn len(&self) -> Result<i32> {
        self.column.view()?.len()
    }

    /// Returns `true` if the series has zero elements.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Returns the number of null (invalid) elements.
    pub fn null_count(&self) -> Result<i32> {
        self.column.view()?.null_count()
    }

    // -- Access --------------------------------------------------------------

    /// Returns a reference to the underlying `Column`.
    pub fn column(&self) -> &Column {
        &self.column
    }

    /// Returns the shared `Arc<Column>`.
    pub fn column_arc(&self) -> &Arc<Column> {
        &self.column
    }

    /// Creates a borrowed `ColumnView` for passing to low-level operations.
    pub fn view(&self) -> Result<ColumnView<'_>> {
        self.column.view()
    }

    // -- Scalar comparison ops (B2a) -----------------------------------------

    /// Element-wise `self > scalar`. Returns an unnamed `BOOL8` Series.
    pub fn gt_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        let s = Scalar::from_value(rhs)?;
        let col = self.column.gt_scalar(&s)?;
        Ok(Self::from_column(col))
    }

    /// Element-wise `self < scalar`. Returns an unnamed `BOOL8` Series.
    pub fn lt_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        let s = Scalar::from_value(rhs)?;
        let col = self.column.lt_scalar(&s)?;
        Ok(Self::from_column(col))
    }

    /// Element-wise `self == scalar`. Returns an unnamed `BOOL8` Series.
    pub fn eq_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        let s = Scalar::from_value(rhs)?;
        let col = self.column.eq_scalar(&s)?;
        Ok(Self::from_column(col))
    }

    /// Element-wise `self == string_scalar`. Returns an unnamed `BOOL8` Series.
    pub fn eq_scalar_string(&self, rhs: &str) -> Result<Series> {
        let s = Scalar::from_string(rhs)?;
        let col = self.column.eq_scalar(&s)?;
        Ok(Self::from_column(col))
    }

    // -- Extended scalar comparison ops (B2b) --------------------------------

    /// Element-wise `self != scalar`. Returns an unnamed `BOOL8` Series.
    pub fn ne_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        let s = Scalar::from_value(rhs)?;
        let col = self.column.ne_scalar(&s)?;
        Ok(Self::from_column(col))
    }

    /// Element-wise `self >= scalar`. Returns an unnamed `BOOL8` Series.
    pub fn ge_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        let s = Scalar::from_value(rhs)?;
        let col = self.column.ge_scalar(&s)?;
        Ok(Self::from_column(col))
    }

    /// Element-wise `self <= scalar`. Returns an unnamed `BOOL8` Series.
    pub fn le_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        let s = Scalar::from_value(rhs)?;
        let col = self.column.le_scalar(&s)?;
        Ok(Self::from_column(col))
    }

    // -- Column-vs-column comparison ops (B2b) -------------------------------

    /// Element-wise `self == rhs`. Returns an unnamed `BOOL8` Series.
    pub fn eq(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.eq_col(&rhs.column)?;
        Ok(Self::from_column(col))
    }

    /// Element-wise `self != rhs`. Returns an unnamed `BOOL8` Series.
    pub fn ne(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.ne_col(&rhs.column)?;
        Ok(Self::from_column(col))
    }

    /// Element-wise `self > rhs`. Returns an unnamed `BOOL8` Series.
    pub fn gt(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.gt_col(&rhs.column)?;
        Ok(Self::from_column(col))
    }

    /// Element-wise `self < rhs`. Returns an unnamed `BOOL8` Series.
    pub fn lt(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.lt_col(&rhs.column)?;
        Ok(Self::from_column(col))
    }

    /// Element-wise `self >= rhs`. Returns an unnamed `BOOL8` Series.
    pub fn ge(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.ge_col(&rhs.column)?;
        Ok(Self::from_column(col))
    }

    /// Element-wise `self <= rhs`. Returns an unnamed `BOOL8` Series.
    pub fn le(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.le_col(&rhs.column)?;
        Ok(Self::from_column(col))
    }

    // -- Arithmetic ops (B2b) ------------------------------------------------

    /// Element-wise `self + rhs`. Output type inferred. Returns an unnamed Series.
    pub fn add(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.add(&rhs.column)?;
        Ok(Self::from_column(col))
    }

    /// Element-wise `self - rhs`. Output type inferred. Returns an unnamed Series.
    pub fn sub(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.sub(&rhs.column)?;
        Ok(Self::from_column(col))
    }

    /// Element-wise `self * rhs`. Output type inferred. Returns an unnamed Series.
    pub fn mul(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.mul(&rhs.column)?;
        Ok(Self::from_column(col))
    }

    /// Element-wise `self / rhs`. Output type inferred. Returns an unnamed Series.
    pub fn div(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.div(&rhs.column)?;
        Ok(Self::from_column(col))
    }

    // -- Scalar arithmetic ops (B2b) -----------------------------------------

    /// Element-wise `self + scalar`. Returns an unnamed Series.
    pub fn add_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        let s = Scalar::from_value(rhs)?;
        let col = self.column.add_scalar(&s)?;
        Ok(Self::from_column(col))
    }

    /// Element-wise `self - scalar`. Returns an unnamed Series.
    pub fn sub_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        let s = Scalar::from_value(rhs)?;
        let col = self.column.sub_scalar(&s)?;
        Ok(Self::from_column(col))
    }

    /// Element-wise `self * scalar`. Returns an unnamed Series.
    pub fn mul_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        let s = Scalar::from_value(rhs)?;
        let col = self.column.mul_scalar(&s)?;
        Ok(Self::from_column(col))
    }

    /// Element-wise `self / scalar`. Returns an unnamed Series.
    pub fn div_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        let s = Scalar::from_value(rhs)?;
        let col = self.column.div_scalar(&s)?;
        Ok(Self::from_column(col))
    }
}
