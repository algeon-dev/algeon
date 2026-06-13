//! `Series` convenience methods for reduction, scan, and unary operations.
//!
//! Prefer the `*_value` helpers when you want an ordinary host result from a
//! high-level workflow. They preserve cuDF/libcudf reduction semantics while
//! avoiding an explicit drop into [`Scalar`](crate::core::Scalar).

use super::{context::default_execution_context, owned::Series};
use crate::{
    core::{
        error::{Error, ErrorCode, ErrorKind, Result},
        scalar::Scalar,
        types::{CudfType, DataType, NullPolicy, TypeId},
    },
    ops::{
        reduction::{ReduceAgg, ScanAgg, ScanType},
        unary::UnaryOperator,
    },
    runtime::{ExecutionOverrides, Runtime},
};

/// Infer the output `DataType` for a reduction aggregation, matching cuDF
/// Python behavior.
///
/// Accepts `&DataType` (not bare `TypeId`) so that fixed-point scale is
/// preserved for pass-through arms (min, max, product, decimal sum).
pub(crate) fn infer_reduce_output_type(input: &DataType, agg: &ReduceAgg, operation: &'static str) -> Result<DataType> {
    match agg {
        // sum: int types → Int64, float → same float, Bool8 → Int64,
        //       decimal → same decimal (preserves scale), rest → error
        ReduceAgg::Sum => match input.id() {
            TypeId::Int8
            | TypeId::Int16
            | TypeId::Int32
            | TypeId::Int64
            | TypeId::UInt8
            | TypeId::UInt16
            | TypeId::UInt32
            | TypeId::UInt64
            | TypeId::Bool8 => Ok(DataType::new(TypeId::Int64)),
            TypeId::Float32 => Ok(DataType::new(TypeId::Float32)),
            TypeId::Float64 => Ok(DataType::new(TypeId::Float64)),
            id if id.is_decimal() => Ok(*input),
            _ => Err(Error::new(ErrorKind::InvalidInput, "unsupported type for sum reduction")
                .with_code(ErrorCode::UnsupportedDtype)
                .with_operation(operation)
                .with_context("type_id", format!("{:?}", input.id()).to_lowercase())),
        },

        // min / max / product: same type as input (preserves decimal scale)
        ReduceAgg::Min | ReduceAgg::Max | ReduceAgg::Product => Ok(*input),

        // mean / median / std / var: always Float64
        ReduceAgg::Mean | ReduceAgg::Median | ReduceAgg::Std { .. } | ReduceAgg::Variance { .. } => {
            Ok(DataType::new(TypeId::Float64))
        },

        // any / all: always Bool8
        ReduceAgg::Any | ReduceAgg::All => Ok(DataType::new(TypeId::Bool8)),

        // nunique: libcudf size_type (int32)
        ReduceAgg::NUnique { .. } => Ok(DataType::new(TypeId::Int32)),
    }
}

pub(crate) fn scalar_to_optional_value<T: CudfType>(scalar: Scalar) -> Result<Option<T>> {
    if !scalar.is_valid()? {
        return Ok(None);
    }

    scalar.value::<T>().map(Some)
}

fn scalar_to_usize_count(scalar: Scalar, operation: &'static str) -> Result<usize> {
    if !scalar.is_valid().map_err(|e| e.with_operation(operation))? {
        return Ok(0);
    }

    let count = scalar.value::<i32>().map_err(|e| e.with_operation(operation))?;
    usize::try_from(count).map_err(|_| negative_distinct_count_error(operation, count))
}

fn negative_distinct_count_error(operation: &'static str, count: i32) -> Error {
    Error::new(ErrorKind::InvariantViolation, "distinct count must be >= 0")
        .with_code(ErrorCode::InvariantViolation)
        .with_operation(operation)
        .with_context("distinct_count", count.to_string())
        .with_context("resource", "distinct_count")
        .with_context("reason", "negative_count")
        .with_context("remedy", "ensure the foreign reduction backend returns a non-negative distinct count")
}

pub(crate) fn trigger_nunique_negative_distinct_count_for_test() -> Result<()> {
    let runtime = Runtime::new()?;
    let scalar = runtime.scalar(-1i32)?;
    scalar_to_usize_count(scalar, "series::nunique").map(|_| ())
}

impl Series {
    pub(super) fn nunique_with_context(&self, ctx: &ExecutionOverrides<'_>, operation: &'static str) -> Result<usize> {
        let scalar = self.reduce_in_context(ReduceAgg::NUnique { null_policy: NullPolicy::Exclude }, ctx, operation)?;
        scalar_to_usize_count(scalar, operation)
    }

    // -- Scalar reductions ---------------------------------------------------

    /// Reduce the series to a single scalar: sum of all elements.
    ///
    /// Output type: integer types → `Int64`, float → same float, `Bool8` → `Int64`.
    /// Nulls are excluded from the sum.
    ///
    /// # Cost
    /// GPU computation (synchronizes to return host-accessible scalar).
    pub fn sum(&self) -> Result<Scalar> {
        self.reduce_in_context(ReduceAgg::Sum, &default_execution_context(), "series::sum")
    }

    /// Reduce the series to a host value: sum of all elements.
    ///
    /// This preserves the underlying reduction output type contract. For
    /// example, integer and bool sums materialize as `i64`.
    ///
    /// # Cost
    /// GPU computation plus synchronization for host extraction.
    ///
    /// ```no_run
    /// use cudf::Runtime;
    ///
    /// let rt = Runtime::new().unwrap();
    /// let s = rt.series(&[10i32, 20, 30]).unwrap();
    /// let total = s.sum_value::<i64>().unwrap();
    ///
    /// assert_eq!(total, 60);
    /// ```
    pub fn sum_value<T: CudfType>(&self) -> Result<T> {
        self.sum()?.value::<T>()
    }

    /// Reduce the series to a single scalar: minimum element.
    ///
    /// Output type: same as input. Nulls are excluded.
    pub fn min(&self) -> Result<Scalar> {
        self.reduce_in_context(ReduceAgg::Min, &default_execution_context(), "series::min")
    }

    /// Reduce the series to a host minimum value.
    ///
    /// Returns `None` when the reduction has no valid element to select.
    ///
    /// ```no_run
    /// use cudf::Runtime;
    ///
    /// let rt = Runtime::new().unwrap();
    /// let s = rt.nullable_series(&[10i32, 0, 30], &[true, false, true]).unwrap();
    /// let min = s.min_value::<i32>().unwrap();
    ///
    /// assert_eq!(min, Some(10));
    /// ```
    pub fn min_value<T: CudfType>(&self) -> Result<Option<T>> {
        scalar_to_optional_value(self.min()?)
    }

    /// Reduce the series to a single scalar: maximum element.
    ///
    /// Output type: same as input. Nulls are excluded.
    pub fn max(&self) -> Result<Scalar> {
        self.reduce_in_context(ReduceAgg::Max, &default_execution_context(), "series::max")
    }

    /// Reduce the series to a host maximum value.
    ///
    /// Returns `None` when the reduction has no valid element to select.
    pub fn max_value<T: CudfType>(&self) -> Result<Option<T>> {
        scalar_to_optional_value(self.max()?)
    }

    /// Arithmetic mean of all elements. Always returns `Float64`.
    ///
    /// Nulls are excluded.
    pub fn mean(&self) -> Result<Scalar> {
        self.reduce_in_context(ReduceAgg::Mean, &default_execution_context(), "series::mean")
    }

    /// Reduce the series to a host mean value.
    ///
    /// Returns `None` when the reduction has no valid element to average.
    ///
    /// ```no_run
    /// use cudf::Runtime;
    ///
    /// let rt = Runtime::new().unwrap();
    /// let s = rt.series(&[10i32, 20, 30]).unwrap();
    /// let mean = s.mean_value().unwrap();
    ///
    /// assert_eq!(mean, Some(20.0));
    /// ```
    pub fn mean_value(&self) -> Result<Option<f64>> {
        scalar_to_optional_value(self.mean()?)
    }

    /// Median of all elements. Always returns `Float64`.
    ///
    /// Nulls are excluded.
    pub fn median(&self) -> Result<Scalar> {
        self.reduce_in_context(ReduceAgg::Median, &default_execution_context(), "series::median")
    }

    /// Reduce the series to a host median value.
    ///
    /// Returns `None` when the reduction has no valid element to select.
    pub fn median_value(&self) -> Result<Option<f64>> {
        scalar_to_optional_value(self.median()?)
    }

    /// Standard deviation with the given degrees-of-freedom correction.
    ///
    /// `ddof = 1` gives the sample standard deviation (default in cuDF Python).
    /// Always returns `Float64`. Nulls are excluded.
    pub fn std(&self, ddof: u32) -> Result<Scalar> {
        self.reduce_in_context(ReduceAgg::Std { ddof }, &default_execution_context(), "series::std")
    }

    /// Reduce the series to a host standard deviation value.
    ///
    /// Returns `None` when the reduction has no valid element to aggregate.
    pub fn std_value(&self, ddof: u32) -> Result<Option<f64>> {
        scalar_to_optional_value(self.std(ddof)?)
    }

    /// Variance with the given degrees-of-freedom correction.
    ///
    /// `ddof = 1` gives the sample variance. Always returns `Float64`.
    /// Nulls are excluded.
    pub fn var(&self, ddof: u32) -> Result<Scalar> {
        self.reduce_in_context(ReduceAgg::Variance { ddof }, &default_execution_context(), "series::var")
    }

    /// Reduce the series to a host variance value.
    ///
    /// Returns `None` when the reduction has no valid element to aggregate.
    pub fn var_value(&self, ddof: u32) -> Result<Option<f64>> {
        scalar_to_optional_value(self.var(ddof)?)
    }

    /// `true` if any element is truthy. Returns a `Bool8` scalar.
    ///
    /// Nulls are excluded.
    pub fn any(&self) -> Result<Scalar> {
        self.reduce_in_context(ReduceAgg::Any, &default_execution_context(), "series::any")
    }

    /// Reduce the series to a host `bool`: `true` if any element is truthy.
    ///
    /// When the reduction has no valid element, this preserves the boolean
    /// identity and returns `false`.
    pub fn any_value(&self) -> Result<bool> {
        Ok(scalar_to_optional_value(self.any()?)?.unwrap_or(false))
    }

    /// `true` if all elements are truthy. Returns a `Bool8` scalar.
    ///
    /// Nulls are excluded.
    pub fn all(&self) -> Result<Scalar> {
        self.reduce_in_context(ReduceAgg::All, &default_execution_context(), "series::all")
    }

    /// Reduce the series to a host `bool`: `true` if all elements are truthy.
    ///
    /// When the reduction has no valid element, this preserves the boolean
    /// identity and returns `true`.
    pub fn all_value(&self) -> Result<bool> {
        Ok(scalar_to_optional_value(self.all()?)?.unwrap_or(true))
    }

    /// Product of all elements. Output type: same as input.
    ///
    /// Nulls are excluded.
    pub fn product(&self) -> Result<Scalar> {
        self.reduce_in_context(ReduceAgg::Product, &default_execution_context(), "series::product")
    }

    /// Reduce the series to a host product value.
    ///
    /// Returns `None` when the reduction has no valid element to aggregate.
    pub fn product_value<T: CudfType>(&self) -> Result<Option<T>> {
        scalar_to_optional_value(self.product()?)
    }

    /// Simultaneous min and max in a single pass.
    ///
    /// Returns `(min_scalar, max_scalar)`.
    pub fn minmax(&self) -> Result<(Scalar, Scalar)> {
        self.minmax_with_context(&default_execution_context(), "series::minmax")
    }

    /// Reduce the series to host minimum and maximum values in one pass.
    ///
    /// Each side returns `None` when the reduction has no valid element to
    /// select.
    ///
    /// ```no_run
    /// use cudf::Runtime;
    ///
    /// let rt = Runtime::new().unwrap();
    /// let s = rt.series(&[30i32, 10, 20]).unwrap();
    /// let (min, max) = s.minmax_values::<i32>().unwrap();
    ///
    /// assert_eq!((min, max), (Some(10), Some(30)));
    /// ```
    pub fn minmax_values<T: CudfType>(&self) -> Result<(Option<T>, Option<T>)> {
        let (min, max) = self.minmax()?;
        Ok((scalar_to_optional_value(min)?, scalar_to_optional_value(max)?))
    }

    // -- Scan (cumulative) operations ----------------------------------------

    /// Inclusive cumulative sum.
    ///
    /// Nulls are excluded from the running total (filled with null in output).
    ///
    /// # Cost
    /// GPU-allocating: produces a new column.
    pub fn cumsum(&self) -> Result<Series> {
        self.scan_with_context(
            ScanAgg::Sum,
            ScanType::Inclusive,
            NullPolicy::Exclude,
            &default_execution_context(),
            "series::cumsum",
        )
    }

    /// Inclusive cumulative minimum.
    pub fn cummin(&self) -> Result<Series> {
        self.scan_with_context(
            ScanAgg::Min,
            ScanType::Inclusive,
            NullPolicy::Exclude,
            &default_execution_context(),
            "series::cummin",
        )
    }

    /// Inclusive cumulative maximum.
    pub fn cummax(&self) -> Result<Series> {
        self.scan_with_context(
            ScanAgg::Max,
            ScanType::Inclusive,
            NullPolicy::Exclude,
            &default_execution_context(),
            "series::cummax",
        )
    }

    /// Inclusive cumulative product.
    pub fn cumprod(&self) -> Result<Series> {
        self.scan_with_context(
            ScanAgg::Product,
            ScanType::Inclusive,
            NullPolicy::Exclude,
            &default_execution_context(),
            "series::cumprod",
        )
    }

    // -- Unary operations ----------------------------------------------------

    /// Element-wise NaN check. Returns a `BOOL8` mask.
    ///
    /// Only meaningful for float types; integer types will return all `false`.
    ///
    /// # Cost
    /// GPU-allocating: produces a new BOOL8 column.
    pub fn is_nan(&self) -> Result<Series> {
        self.is_nan_with_context(&default_execution_context(), "series::is_nan")
    }

    /// Element-wise not-NaN check. Returns a `BOOL8` mask.
    pub fn is_not_nan(&self) -> Result<Series> {
        self.is_not_nan_with_context(&default_execution_context(), "series::is_not_nan")
    }

    /// Element-wise absolute value.
    ///
    /// # Cost
    /// GPU-allocating: produces a new column.
    pub fn abs(&self) -> Result<Series> {
        self.unary_with_context(UnaryOperator::Abs, &default_execution_context(), "series::abs")
    }

    /// Element-wise arithmetic negation.
    ///
    /// # Cost
    /// GPU-allocating: produces a new column.
    pub fn neg(&self) -> Result<Series> {
        self.unary_with_context(UnaryOperator::Negate, &default_execution_context(), "series::neg")
    }

    /// Element-wise logical NOT for `Bool8` columns.
    ///
    /// # Cost
    /// GPU-allocating: produces a new BOOL8 column.
    pub fn logical_not(&self) -> Result<Series> {
        self.logical_not_with_context(&default_execution_context(), "series::logical_not")
    }
}
