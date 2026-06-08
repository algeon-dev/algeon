//! `Series` convenience methods for reduction, scan, and unary operations.
//!
//! Prefer the `*_value` helpers when you want an ordinary host result from a
//! high-level workflow. They preserve cuDF/libcudf reduction semantics while
//! avoiding an explicit drop into [`Scalar`](crate::core::Scalar).

use super::owned::Series;
use crate::{
    core::{
        error::{Error, ErrorKind, Result},
        scalar::Scalar,
        types::{CudfType, DataType, NullPolicy, TypeId},
    },
    ops::{
        reduction::{self, ReduceAgg, ScanAgg, ScanType},
        unary::{self, UnaryOperator},
    },
};

/// Infer the output `DataType` for a reduction aggregation, matching cuDF
/// Python behavior.
///
/// Accepts `&DataType` (not bare `TypeId`) so that fixed-point scale is
/// preserved for pass-through arms (min, max, product, decimal sum).
pub(crate) fn infer_reduce_output_type(input: &DataType, agg: &ReduceAgg) -> Result<DataType> {
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
            _ => Err(Error::new(ErrorKind::TypeError, "unsupported type for sum reduction")
                .with_operation("series::sum")
                .with_context("input_type", format!("{:?}", input.id()))),
        },

        // min / max / product: same type as input (preserves decimal scale)
        ReduceAgg::Min | ReduceAgg::Max | ReduceAgg::Product => Ok(*input),

        // mean / median / std / var: always Float64
        ReduceAgg::Mean | ReduceAgg::Median | ReduceAgg::Std { .. } | ReduceAgg::Variance { .. } => {
            Ok(DataType::new(TypeId::Float64))
        },

        // any / all: always Bool8
        ReduceAgg::Any | ReduceAgg::All => Ok(DataType::new(TypeId::Bool8)),
    }
}

pub(crate) fn scalar_to_optional_value<T: CudfType>(scalar: Scalar) -> Result<Option<T>> {
    if !scalar.is_valid()? {
        return Ok(None);
    }

    scalar.value::<T>().map(Some)
}

impl Series {
    // -- Scalar reductions ---------------------------------------------------

    /// Reduce the series to a single scalar: sum of all elements.
    ///
    /// Output type: integer types → `Int64`, float → same float, `Bool8` → `Int64`.
    /// Nulls are excluded from the sum.
    ///
    /// # Cost
    /// GPU computation (synchronizes to return host-accessible scalar).
    pub fn sum(&self) -> Result<Scalar> {
        let cv = self.column.view()?;
        let dt = self.dtype()?;
        let out_type = infer_reduce_output_type(&dt, &ReduceAgg::Sum)?;
        reduction::reduce(&cv, &ReduceAgg::Sum, &out_type)
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
    /// use cudf::Series;
    ///
    /// let s = Series::new::<i32>("x", &[10, 20, 30]).unwrap();
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
        let cv = self.column.view()?;
        let dt = self.dtype()?;
        let out_type = infer_reduce_output_type(&dt, &ReduceAgg::Min)?;
        reduction::reduce(&cv, &ReduceAgg::Min, &out_type)
    }

    /// Reduce the series to a host minimum value.
    ///
    /// Returns `None` when the reduction has no valid element to select.
    ///
    /// ```no_run
    /// use cudf::Series;
    ///
    /// let s =
    ///     Series::new_nullable::<i32>("x", &[10, 0, 30], &[true, false, true])
    ///         .unwrap();
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
        let cv = self.column.view()?;
        let dt = self.dtype()?;
        let out_type = infer_reduce_output_type(&dt, &ReduceAgg::Max)?;
        reduction::reduce(&cv, &ReduceAgg::Max, &out_type)
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
        let cv = self.column.view()?;
        let out_type = DataType::new(TypeId::Float64);
        reduction::reduce(&cv, &ReduceAgg::Mean, &out_type)
    }

    /// Reduce the series to a host mean value.
    ///
    /// Returns `None` when the reduction has no valid element to average.
    ///
    /// ```no_run
    /// use cudf::Series;
    ///
    /// let s = Series::new::<i32>("x", &[10, 20, 30]).unwrap();
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
        let cv = self.column.view()?;
        let out_type = DataType::new(TypeId::Float64);
        reduction::reduce(&cv, &ReduceAgg::Median, &out_type)
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
        let cv = self.column.view()?;
        let out_type = DataType::new(TypeId::Float64);
        reduction::reduce(&cv, &ReduceAgg::Std { ddof }, &out_type)
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
        let cv = self.column.view()?;
        let out_type = DataType::new(TypeId::Float64);
        reduction::reduce(&cv, &ReduceAgg::Variance { ddof }, &out_type)
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
        let cv = self.column.view()?;
        let out_type = DataType::new(TypeId::Bool8);
        reduction::reduce(&cv, &ReduceAgg::Any, &out_type)
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
        let cv = self.column.view()?;
        let out_type = DataType::new(TypeId::Bool8);
        reduction::reduce(&cv, &ReduceAgg::All, &out_type)
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
        let cv = self.column.view()?;
        let dt = self.dtype()?;
        let out_type = infer_reduce_output_type(&dt, &ReduceAgg::Product)?;
        reduction::reduce(&cv, &ReduceAgg::Product, &out_type)
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
        let cv = self.column.view()?;
        reduction::minmax(&cv)
    }

    /// Reduce the series to host minimum and maximum values in one pass.
    ///
    /// Each side returns `None` when the reduction has no valid element to
    /// select.
    ///
    /// ```no_run
    /// use cudf::Series;
    ///
    /// let s = Series::new::<i32>("x", &[30, 10, 20]).unwrap();
    /// let (min, max) = s.minmax_values::<i32>().unwrap();
    ///
    /// assert_eq!((min, max), (Some(10), Some(30)));
    /// ```
    pub fn minmax_values<T: CudfType>(&self) -> Result<(Option<T>, Option<T>)> {
        let (min, max) = self.minmax()?;
        Ok((scalar_to_optional_value(min)?, scalar_to_optional_value(max)?))
    }

    // -- Scan (cumulative) operations ----------------------------------------

    /// Inclusive cumulative sum. Inherits the source Series' name.
    ///
    /// Nulls are excluded from the running total (filled with null in output).
    ///
    /// # Cost
    /// GPU-allocating: produces a new column.
    pub fn cumsum(&self) -> Result<Series> {
        let cv = self.column.view()?;
        let col = reduction::scan(&cv, ScanAgg::Sum, ScanType::Inclusive, NullPolicy::Exclude)?;
        self.with_inherited_name(col)
    }

    /// Inclusive cumulative minimum. Inherits the source Series' name.
    pub fn cummin(&self) -> Result<Series> {
        let cv = self.column.view()?;
        let col = reduction::scan(&cv, ScanAgg::Min, ScanType::Inclusive, NullPolicy::Exclude)?;
        self.with_inherited_name(col)
    }

    /// Inclusive cumulative maximum. Inherits the source Series' name.
    pub fn cummax(&self) -> Result<Series> {
        let cv = self.column.view()?;
        let col = reduction::scan(&cv, ScanAgg::Max, ScanType::Inclusive, NullPolicy::Exclude)?;
        self.with_inherited_name(col)
    }

    /// Inclusive cumulative product. Inherits the source Series' name.
    pub fn cumprod(&self) -> Result<Series> {
        let cv = self.column.view()?;
        let col = reduction::scan(&cv, ScanAgg::Product, ScanType::Inclusive, NullPolicy::Exclude)?;
        self.with_inherited_name(col)
    }

    // -- Unary operations ----------------------------------------------------

    /// Element-wise NaN check. Returns a `BOOL8` mask. Inherits name.
    ///
    /// Only meaningful for float types; integer types will return all `false`.
    ///
    /// # Cost
    /// GPU-allocating: produces a new BOOL8 column.
    pub fn is_nan(&self) -> Result<Series> {
        let cv = self.column.view()?;
        let col = unary::is_nan(&cv)?;
        self.with_inherited_name(col)
    }

    /// Element-wise not-NaN check. Returns a `BOOL8` mask. Inherits name.
    pub fn is_not_nan(&self) -> Result<Series> {
        let cv = self.column.view()?;
        let col = unary::is_not_nan(&cv)?;
        self.with_inherited_name(col)
    }

    /// Element-wise absolute value. Inherits the source Series' name.
    ///
    /// # Cost
    /// GPU-allocating: produces a new column.
    pub fn abs(&self) -> Result<Series> {
        let cv = self.column.view()?;
        let col = unary::unary_operation(&cv, UnaryOperator::Abs)?;
        self.with_inherited_name(col)
    }

    /// Element-wise arithmetic negation. Inherits the source Series' name.
    ///
    /// # Cost
    /// GPU-allocating: produces a new column.
    pub fn neg(&self) -> Result<Series> {
        let cv = self.column.view()?;
        let col = unary::unary_operation(&cv, UnaryOperator::Negate)?;
        self.with_inherited_name(col)
    }

    /// Element-wise logical NOT for `Bool8` columns. Inherits the source
    /// Series' name.
    ///
    /// # Cost
    /// GPU-allocating: produces a new BOOL8 column.
    pub fn logical_not(&self) -> Result<Series> {
        let cv = self.column.view()?;
        let col = unary::logical_not(&cv)?;
        self.with_inherited_name(col)
    }
}
