use std::sync::Arc;

use super::view::SeriesView;
use crate::{
    core::{
        column::{Column, ColumnView},
        error::Result,
        scalar::Scalar,
        types::{CudfType, DataType},
    },
    internal::temp_table::TempTableView,
    ops::{
        replace,
        sorting::SortOptions,
        stream_compaction::{self, DuplicateKeepOption, NanEquality, NullEquality},
        unary,
    },
    runtime::{Runtime, execution_domain::ExecutionDomain},
};

/// A named, reference-counted GPU column.
///
/// `Series` is the cuDF-like user-facing column type. Internally it holds an
/// `Arc<Column>`, so cloning, renaming, and passing Series between DataFrames
/// is zero-copy on the GPU side.
///
/// Metadata queries (`len()`, `is_empty()`, `null_count()`) are infallible and
/// free — the values are cached at construction time.
#[derive(Clone)]
pub struct Series {
    pub(super) name: Option<String>,
    pub(super) column: Arc<Column>,
    pub(super) cached_len: usize,
    pub(super) cached_null_count: usize,
}

impl std::fmt::Debug for Series {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ds = f.debug_struct("Series");
        ds.field("name", &self.name);
        let dtype_str = self.dtype().map(|dt| format!("{:?}", dt.id())).unwrap_or_else(|_| "?".to_string());
        ds.field("dtype", &dtype_str);
        ds.field("len", &self.cached_len);
        ds.finish()
    }
}

impl TryFrom<Column> for Series {
    type Error = crate::core::error::Error;

    fn try_from(col: Column) -> Result<Self> {
        Self::from_column(col)
    }
}

/// Trait for types that can be converted into a `Series`, possibly fallibly.
///
/// This is used by [`DataFrame::with_column`] to accept both `Series` and
/// `Column` transparently.
pub trait IntoSeries {
    /// Convert into a `Series`.
    fn into_series(self) -> Result<Series>;
}

impl IntoSeries for Series {
    fn into_series(self) -> Result<Series> {
        Ok(self)
    }
}

impl IntoSeries for Column {
    fn into_series(self) -> Result<Series> {
        Series::from_column(self)
    }
}

impl Series {
    /// Returns the data type of the underlying column.
    pub fn dtype(&self) -> Result<DataType> {
        self.column.view()?.data_type()
    }

    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        self.cached_len
    }

    /// Returns `true` if the series has zero elements.
    pub fn is_empty(&self) -> bool {
        self.cached_len == 0
    }

    /// Returns the number of null (invalid) elements.
    pub fn null_count(&self) -> usize {
        self.cached_null_count
    }

    // -- Access --------------------------------------------------------------

    /// Advanced: drop to the core `Column` type.
    ///
    /// Returns a reference to the underlying `Column`. Use this when you need
    /// to call low-level column operations that are not exposed on `Series`.
    pub fn as_column(&self) -> &Column {
        &self.column
    }

    /// Returns a reference to the shared `Arc<Column>`.
    ///
    /// Crate-internal: used by `DataFrame` to clone the Arc into its column map.
    /// External callers should use [`into_inner`](Self::into_inner) to take
    /// ownership of the `Arc<Column>`.
    pub(in crate::frame) fn column_arc(&self) -> &Arc<Column> {
        &self.column
    }

    /// Consume this Series and return the underlying `Arc<Column>`.
    ///
    /// This is the idiomatic way to unwrap a Series when you no longer need
    /// the name or Series wrapper.
    pub fn into_inner(self) -> Arc<Column> {
        self.column
    }

    /// Creates a borrowed [`SeriesView`] carrying this series' name.
    ///
    /// The returned view implements `Deref<Target = ColumnView>`, so all
    /// low-level column operations are accessible directly.
    pub fn view(&self) -> Result<SeriesView<'_>> {
        let cv = self.column.view()?;
        Ok(SeriesView::new(self.name.as_deref(), cv))
    }

    /// Returns a borrowed bare `ColumnView` for low-level operations.
    ///
    /// # Advanced
    ///
    /// Prefer [`view()`](Self::view) for most uses. Use this method only when
    /// an API specifically requires a bare `ColumnView` without the
    /// `SeriesView` name wrapper.
    pub fn as_column_view(&self) -> Result<ColumnView<'_>> {
        self.column.view()
    }

    /// Copy string data to host as `Vec<Option<String>>`, distinguishing null
    /// from empty string.
    ///
    /// Null positions return `None`; valid positions return `Some(s)`.
    /// Non-nullable columns return all `Some`.
    ///
    /// # Errors
    ///
    /// Returns `TypeError` if the underlying column type is not `STRING`.
    pub fn to_optional_strings(&self) -> Result<Vec<Option<String>>> {
        self.column.view()?.to_optional_strings()
    }

    // -- Scalar comparison ops -----------------------------------------

    /// Element-wise `self > scalar`. Inherits the source Series' name.
    pub fn gt_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        let s = Scalar::from_value(rhs)?;
        let col = self.column.gt_scalar(&s)?;
        self.with_inherited_name(col)
    }

    /// Element-wise `self < scalar`. Inherits the source Series' name.
    pub fn lt_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        let s = Scalar::from_value(rhs)?;
        let col = self.column.lt_scalar(&s)?;
        self.with_inherited_name(col)
    }

    /// Element-wise `self == scalar`. Inherits the source Series' name.
    pub fn eq_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        let s = Scalar::from_value(rhs)?;
        let col = self.column.eq_scalar(&s)?;
        self.with_inherited_name(col)
    }

    // -- String scalar comparison ops ----------------------------------------

    /// Element-wise `self == string_scalar`. Inherits the source Series' name.
    pub fn eq_scalar_str(&self, rhs: &str) -> Result<Series> {
        let s = Scalar::from_string(rhs)?;
        let col = self.column.eq_scalar(&s)?;
        self.with_inherited_name(col)
    }

    /// Element-wise `self != string_scalar`. Inherits the source Series' name.
    pub fn ne_scalar_str(&self, rhs: &str) -> Result<Series> {
        let s = Scalar::from_string(rhs)?;
        let col = self.column.ne_scalar(&s)?;
        self.with_inherited_name(col)
    }

    /// Element-wise `self > string_scalar` (lexicographic). Inherits the source Series' name.
    pub fn gt_scalar_str(&self, rhs: &str) -> Result<Series> {
        let s = Scalar::from_string(rhs)?;
        let col = self.column.gt_scalar(&s)?;
        self.with_inherited_name(col)
    }

    /// Element-wise `self < string_scalar` (lexicographic). Inherits the source Series' name.
    pub fn lt_scalar_str(&self, rhs: &str) -> Result<Series> {
        let s = Scalar::from_string(rhs)?;
        let col = self.column.lt_scalar(&s)?;
        self.with_inherited_name(col)
    }

    /// Element-wise `self >= string_scalar` (lexicographic). Inherits the source Series' name.
    pub fn ge_scalar_str(&self, rhs: &str) -> Result<Series> {
        let s = Scalar::from_string(rhs)?;
        let col = self.column.ge_scalar(&s)?;
        self.with_inherited_name(col)
    }

    /// Element-wise `self <= string_scalar` (lexicographic). Inherits the source Series' name.
    pub fn le_scalar_str(&self, rhs: &str) -> Result<Series> {
        let s = Scalar::from_string(rhs)?;
        let col = self.column.le_scalar(&s)?;
        self.with_inherited_name(col)
    }

    // -- Extended scalar comparison ops --------------------------------

    /// Element-wise `self != scalar`. Inherits the source Series' name.
    pub fn ne_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        let s = Scalar::from_value(rhs)?;
        let col = self.column.ne_scalar(&s)?;
        self.with_inherited_name(col)
    }

    /// Element-wise `self >= scalar`. Inherits the source Series' name.
    pub fn ge_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        let s = Scalar::from_value(rhs)?;
        let col = self.column.ge_scalar(&s)?;
        self.with_inherited_name(col)
    }

    /// Element-wise `self <= scalar`. Inherits the source Series' name.
    pub fn le_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        let s = Scalar::from_value(rhs)?;
        let col = self.column.le_scalar(&s)?;
        self.with_inherited_name(col)
    }

    // -- Column-vs-column comparison ops -------------------------------

    /// Element-wise `self == rhs`. Returns an unnamed `BOOL8` Series.
    pub fn eq(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.eq_col(&rhs.column)?;
        Self::from_column(col)
    }

    /// Element-wise `self != rhs`. Returns an unnamed `BOOL8` Series.
    pub fn ne(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.ne_col(&rhs.column)?;
        Self::from_column(col)
    }

    /// Element-wise `self > rhs`. Returns an unnamed `BOOL8` Series.
    pub fn gt(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.gt_col(&rhs.column)?;
        Self::from_column(col)
    }

    /// Element-wise `self < rhs`. Returns an unnamed `BOOL8` Series.
    pub fn lt(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.lt_col(&rhs.column)?;
        Self::from_column(col)
    }

    /// Element-wise `self >= rhs`. Returns an unnamed `BOOL8` Series.
    pub fn ge(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.ge_col(&rhs.column)?;
        Self::from_column(col)
    }

    /// Element-wise `self <= rhs`. Returns an unnamed `BOOL8` Series.
    pub fn le(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.le_col(&rhs.column)?;
        Self::from_column(col)
    }

    // -- Arithmetic ops ------------------------------------------------

    /// Element-wise `self + rhs`. Output type inferred. Returns an unnamed Series.
    pub fn add(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.add(&rhs.column)?;
        Self::from_column(col)
    }

    /// Element-wise `self - rhs`. Output type inferred. Returns an unnamed Series.
    pub fn sub(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.sub(&rhs.column)?;
        Self::from_column(col)
    }

    /// Element-wise `self * rhs`. Output type inferred. Returns an unnamed Series.
    pub fn mul(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.mul(&rhs.column)?;
        Self::from_column(col)
    }

    /// Element-wise `self / rhs`. Output type inferred. Returns an unnamed Series.
    pub fn div(&self, rhs: &Series) -> Result<Series> {
        let col = self.column.div(&rhs.column)?;
        Self::from_column(col)
    }

    // -- Scalar arithmetic ops -----------------------------------------

    /// Element-wise `self + scalar`. Inherits the source Series' name.
    pub fn add_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        let s = Scalar::from_value(rhs)?;
        let col = self.column.add_scalar(&s)?;
        self.with_inherited_name(col)
    }

    /// Element-wise `self - scalar`. Inherits the source Series' name.
    pub fn sub_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        let s = Scalar::from_value(rhs)?;
        let col = self.column.sub_scalar(&s)?;
        self.with_inherited_name(col)
    }

    /// Element-wise `self * scalar`. Inherits the source Series' name.
    pub fn mul_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        let s = Scalar::from_value(rhs)?;
        let col = self.column.mul_scalar(&s)?;
        self.with_inherited_name(col)
    }

    /// Element-wise `self / scalar`. Inherits the source Series' name.
    pub fn div_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        let s = Scalar::from_value(rhs)?;
        let col = self.column.div_scalar(&s)?;
        self.with_inherited_name(col)
    }
    /// Clone or deep-copy an `Arc<Column>` into the target domain.
    fn clone_or_copy_into_domain(
        column: &Arc<Column>,
        output_domain: &Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Arc<Column>> {
        if ExecutionDomain::same_provenance(column.execution_domain(), output_domain) {
            return Ok(Arc::clone(column));
        }

        column.execution_domain().fence_stream(operation)?;

        column.copy_into_domain(Arc::clone(output_domain), operation).map(Arc::new)
    }

    // -- Convenience methods — unary ops inherit name ---------

    /// Return a BOOL8 mask: `true` where elements are null.
    ///
    /// # Cost
    /// GPU-allocating: produces a new BOOL8 column on the device.
    pub fn is_null(&self) -> Result<Series> {
        let cv = self.column.view()?;
        let col = unary::is_null(&cv)?;
        self.with_inherited_name(col)
    }

    /// Return a BOOL8 mask: `true` where elements are valid (non-null).
    ///
    /// # Cost
    /// GPU-allocating: produces a new BOOL8 column on the device.
    pub fn is_not_null(&self) -> Result<Series> {
        let cv = self.column.view()?;
        let col = unary::is_valid(&cv)?;
        self.with_inherited_name(col)
    }

    /// Cast this series to the given type. Inherits the source Series' name.
    ///
    /// Accepts either a [`DataType`] (canonical, supports decimal scale) or a
    /// [`crate::TypeId`] (convenience shorthand, scale defaults to 0).
    ///
    /// Equivalent to cuDF Python's `Series.astype()`.
    ///
    /// # Cost
    /// GPU-allocating: a new column is produced with the target dtype.
    pub fn astype(&self, target: impl Into<DataType>) -> Result<Series> {
        let dt: DataType = target.into();
        let cv = self.column.view()?;
        let col = unary::cast(&cv, &dt)?;
        self.with_inherited_name(col)
    }

    /// Replace null values with a scalar. Inherits the source Series' name.
    ///
    /// Equivalent to cuDF Python's `Series.fillna(scalar)`.
    ///
    /// # Cost
    /// GPU-allocating: a new column is produced with nulls replaced.
    pub fn fillna<T: CudfType>(&self, value: T) -> Result<Series> {
        let cv = self.column.view()?;
        let scalar = Scalar::from_value(value)?;
        let exec_options = self.column.execution_domain().explicit_exec_options_raw();
        let col = replace::replace_nulls_with_exec_options(&cv, &scalar, exec_options.as_ref(), "series::fillna")?
            .with_execution_domain(Arc::clone(self.column.execution_domain()));
        self.with_inherited_name(col)
    }

    /// Replace null values using corresponding elements from another Series.
    /// Inherits the source Series' name.
    ///
    /// Both series must have the same dtype and length.
    ///
    /// # Cost
    /// GPU-allocating: a new column is produced with nulls replaced.
    pub fn fillna_with_series(&self, other: &Series) -> Result<Series> {
        let output_domain = Arc::clone(self.column.execution_domain());
        let cv = self.column.view()?;
        let other_cv = other.column.view()?;
        let exec_options = output_domain.explicit_exec_options_raw();
        let col = replace::replace_nulls_column_with_exec_options(
            &cv,
            &other_cv,
            exec_options.as_ref(),
            "series::fillna_with_series",
        )?
        .with_execution_domain(output_domain);
        self.with_inherited_name(col)
    }

    /// Replace null values using corresponding elements from another Series,
    /// copying inputs into the target [`Runtime`]'s retained provenance when
    /// necessary.
    ///
    /// Unlike [`fillna_with_series`](Self::fillna_with_series), this method
    /// recovers from mixed retained provenance by deep-copying both series
    /// into the runtime's domain before executing the replace-nulls
    /// operation. Inherits the source Series' name.
    ///
    /// # Cost
    /// GPU-allocating: a new column is produced. If any input is in a
    /// different domain, an additional device-to-device copy per mismatched
    /// series occurs.
    pub fn fillna_with_series_with_runtime(&self, other: &Series, runtime: &Runtime) -> Result<Series> {
        let operation = "series::fillna_with_series_with_runtime";
        let output_domain = Arc::clone(runtime.execution_domain());

        // Copy self and other into the target domain if not already there.
        let self_col = Self::clone_or_copy_into_domain(&self.column, &output_domain, operation)?;
        let other_col = Self::clone_or_copy_into_domain(&other.column, &output_domain, operation)?;

        let cv = self_col.view()?;
        let other_cv = other_col.view()?;
        let exec_options = output_domain.explicit_exec_options_raw();
        let col = replace::replace_nulls_column_with_exec_options(&cv, &other_cv, exec_options.as_ref(), operation)?
            .with_execution_domain(output_domain);
        self.with_inherited_name(col)
    }

    /// Element-wise membership test. Returns a BOOL8 Series where `true`
    /// indicates the element is present in `values`. Inherits the source
    /// Series' name.
    ///
    /// Equivalent to cuDF Python's `Series.isin()`.
    ///
    /// # Cost
    /// GPU-allocating: produces a new BOOL8 column.
    pub fn isin(&self, values: &Series) -> Result<Series> {
        let cv = self.column.view()?;
        let values_cv = values.column.view()?;
        let col = cv.isin(&values_cv)?;
        self.with_inherited_name(col)
    }

    /// Return a Series with duplicate values removed. Inherits the source
    /// Series' name.
    ///
    /// The order of elements in the result is **not** specified.
    ///
    /// Equivalent to cuDF Python's `Series.unique()`.
    ///
    /// # Cost
    /// GPU-allocating: produces a new column containing only distinct values.
    pub fn unique(&self) -> Result<Series> {
        let cv = self.column.view()?;
        let ttv = TempTableView::from_column_views(vec![cv])?;
        let result = stream_compaction::distinct(
            ttv.table(),
            &[0],
            DuplicateKeepOption::KeepAny,
            NullEquality::Equal,
            NanEquality::AllEqual,
        )?
        .with_execution_domain(Arc::clone(self.column.execution_domain()));
        let mut cols = result.into_columns()?;
        self.with_inherited_name(cols.remove(0))
    }

    /// Return the number of distinct non-null values.
    ///
    /// Equivalent to cuDF Python's `Series.nunique()`.
    ///
    /// # Cost
    /// GPU-allocating: internally computes `unique()` and counts rows.
    pub fn nunique(&self) -> Result<usize> {
        let uniq = self.unique()?;
        // Exclude nulls from the count (matches cuDF Python default).
        Ok(uniq.len() - uniq.null_count())
    }

    /// Encode the series as integer codes and a set of unique values.
    ///
    /// Returns `(codes, uniques)` where `codes` is an `Int32` Series mapping
    /// each element to its index in `uniques`.
    ///
    /// Name behavior:
    /// - named input => `"{name}_codes"` and `"{name}_uniques"`
    /// - unnamed input => both outputs stay unnamed
    ///
    /// Equivalent to cuDF Python's `Series.factorize()`.
    ///
    /// # Cost
    /// GPU-allocating: produces two new columns (codes + uniques).
    pub fn factorize(&self) -> Result<(Series, Series)> {
        let cv = self.column.view()?;
        let (codes, uniques) = cv.factorize()?;
        let codes = Self::from_column(codes)?;
        let uniques = Self::from_column(uniques)?;
        if let Some(name) = self.name() {
            Ok((codes.rename(format!("{name}_codes")), uniques.rename(format!("{name}_uniques"))))
        } else {
            Ok((codes, uniques))
        }
    }

    /// Sort the series according to the given options. Inherits the source
    /// Series' name.
    ///
    /// Equivalent to cuDF Python's `Series.sort_values()`.
    ///
    /// # Cost
    /// GPU-allocating: a new sorted column is produced.
    pub fn sort_values(&self, options: SortOptions) -> Result<Series> {
        let (orders, null_orders) = options.resolve(1)?;
        let cv = self.column.view()?;
        let ttv = TempTableView::from_column_views(vec![cv])?;
        let sorted = if options.stable {
            crate::ops::sorting::stable_sort(ttv.table(), Some(&orders), null_orders.as_deref())?
        } else {
            crate::ops::sorting::sort(ttv.table(), Some(&orders), null_orders.as_deref())?
        }
        .with_execution_domain(Arc::clone(self.column.execution_domain()));
        let mut cols = sorted.into_columns()?;
        self.with_inherited_name(cols.remove(0))
    }
}

// ===========================================================================
// Operator overloads: Add, Sub, Mul, Div
// ===========================================================================

/// Implement `std::ops::{Add, Sub, Mul, Div}` for all value/ref combinations.
///
/// Each operator panics on error — use the named method (e.g. `.add()`) for
/// `Result`-based error handling.
macro_rules! impl_series_arithmetic_op {
    ($trait:ident, $method:ident, $named_method:ident) => {
        impl std::ops::$trait for Series {
            type Output = Series;

            /// Panics if the operation fails (e.g., type mismatch). Use the
            /// named method [`Series::$named_method`] for `Result`-based error handling.
            fn $method(self, rhs: Series) -> Series {
                Series::$named_method(&self, &rhs).expect(concat!("Series::", stringify!($named_method), " failed"))
            }
        }

        impl std::ops::$trait<&Series> for Series {
            type Output = Series;

            fn $method(self, rhs: &Series) -> Series {
                Series::$named_method(&self, rhs).expect(concat!("Series::", stringify!($named_method), " failed"))
            }
        }

        impl std::ops::$trait<Series> for &Series {
            type Output = Series;

            fn $method(self, rhs: Series) -> Series {
                Series::$named_method(self, &rhs).expect(concat!("Series::", stringify!($named_method), " failed"))
            }
        }

        impl std::ops::$trait<&Series> for &Series {
            type Output = Series;

            fn $method(self, rhs: &Series) -> Series {
                Series::$named_method(self, rhs).expect(concat!("Series::", stringify!($named_method), " failed"))
            }
        }
    };
}

impl_series_arithmetic_op!(Add, add, add);
impl_series_arithmetic_op!(Sub, sub, sub);
impl_series_arithmetic_op!(Mul, mul, mul);
impl_series_arithmetic_op!(Div, div, div);
