//! Series facade methods for reduction, scan, and unary operations.

use super::owned::Series;
use crate::{
    core::{
        error::{Error, ErrorKind, Result},
        scalar::Scalar,
        types::{DataType, NullPolicy, TypeId},
    },
    ops::{
        reduction::{self, ReduceAggregation, ScanAggregation, ScanType},
        unary::{self, UnaryOperator},
    },
};

/// Infer the output `DataType` for a reduction aggregation, matching cuDF
/// Python behavior.
///
/// Accepts `&DataType` (not bare `TypeId`) so that fixed-point scale is
/// preserved for pass-through arms (min, max, product, decimal sum).
fn infer_reduce_output_type(input: &DataType, agg: &ReduceAggregation) -> Result<DataType> {
    match agg {
        // sum: int types → Int64, float → same float, Bool8 → Int64,
        //       decimal → same decimal (preserves scale), rest → error
        ReduceAggregation::Sum => match input.id() {
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
        ReduceAggregation::Min | ReduceAggregation::Max | ReduceAggregation::Product => Ok(*input),

        // mean / median / std / var: always Float64
        ReduceAggregation::Mean
        | ReduceAggregation::Median
        | ReduceAggregation::Std { .. }
        | ReduceAggregation::Variance { .. } => Ok(DataType::new(TypeId::Float64)),

        // any / all: always Bool8
        ReduceAggregation::Any | ReduceAggregation::All => Ok(DataType::new(TypeId::Bool8)),
    }
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
        let out_type = infer_reduce_output_type(&dt, &ReduceAggregation::Sum)?;
        reduction::reduce(&cv, &ReduceAggregation::Sum, &out_type)
    }

    /// Reduce the series to a single scalar: minimum element.
    ///
    /// Output type: same as input. Nulls are excluded.
    pub fn min(&self) -> Result<Scalar> {
        let cv = self.column.view()?;
        let dt = self.dtype()?;
        let out_type = infer_reduce_output_type(&dt, &ReduceAggregation::Min)?;
        reduction::reduce(&cv, &ReduceAggregation::Min, &out_type)
    }

    /// Reduce the series to a single scalar: maximum element.
    ///
    /// Output type: same as input. Nulls are excluded.
    pub fn max(&self) -> Result<Scalar> {
        let cv = self.column.view()?;
        let dt = self.dtype()?;
        let out_type = infer_reduce_output_type(&dt, &ReduceAggregation::Max)?;
        reduction::reduce(&cv, &ReduceAggregation::Max, &out_type)
    }

    /// Arithmetic mean of all elements. Always returns `Float64`.
    ///
    /// Nulls are excluded.
    pub fn mean(&self) -> Result<Scalar> {
        let cv = self.column.view()?;
        let out_type = DataType::new(TypeId::Float64);
        reduction::reduce(&cv, &ReduceAggregation::Mean, &out_type)
    }

    /// Median of all elements. Always returns `Float64`.
    ///
    /// Nulls are excluded.
    pub fn median(&self) -> Result<Scalar> {
        let cv = self.column.view()?;
        let out_type = DataType::new(TypeId::Float64);
        reduction::reduce(&cv, &ReduceAggregation::Median, &out_type)
    }

    /// Standard deviation with the given degrees-of-freedom correction.
    ///
    /// `ddof = 1` gives the sample standard deviation (default in cuDF Python).
    /// Always returns `Float64`. Nulls are excluded.
    pub fn std(&self, ddof: i32) -> Result<Scalar> {
        let cv = self.column.view()?;
        let out_type = DataType::new(TypeId::Float64);
        reduction::reduce(&cv, &ReduceAggregation::Std { ddof }, &out_type)
    }

    /// Variance with the given degrees-of-freedom correction.
    ///
    /// `ddof = 1` gives the sample variance. Always returns `Float64`.
    /// Nulls are excluded.
    pub fn var(&self, ddof: i32) -> Result<Scalar> {
        let cv = self.column.view()?;
        let out_type = DataType::new(TypeId::Float64);
        reduction::reduce(&cv, &ReduceAggregation::Variance { ddof }, &out_type)
    }

    /// `true` if any element is truthy. Returns a `Bool8` scalar.
    ///
    /// Nulls are excluded.
    pub fn any(&self) -> Result<Scalar> {
        let cv = self.column.view()?;
        let out_type = DataType::new(TypeId::Bool8);
        reduction::reduce(&cv, &ReduceAggregation::Any, &out_type)
    }

    /// `true` if all elements are truthy. Returns a `Bool8` scalar.
    ///
    /// Nulls are excluded.
    pub fn all(&self) -> Result<Scalar> {
        let cv = self.column.view()?;
        let out_type = DataType::new(TypeId::Bool8);
        reduction::reduce(&cv, &ReduceAggregation::All, &out_type)
    }

    /// Product of all elements. Output type: same as input.
    ///
    /// Nulls are excluded.
    pub fn product(&self) -> Result<Scalar> {
        let cv = self.column.view()?;
        let dt = self.dtype()?;
        let out_type = infer_reduce_output_type(&dt, &ReduceAggregation::Product)?;
        reduction::reduce(&cv, &ReduceAggregation::Product, &out_type)
    }

    /// Simultaneous min and max in a single pass.
    ///
    /// Returns `(min_scalar, max_scalar)`.
    pub fn minmax(&self) -> Result<(Scalar, Scalar)> {
        let cv = self.column.view()?;
        reduction::minmax(&cv)
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
        let col = reduction::scan(&cv, ScanAggregation::Sum, ScanType::Inclusive, NullPolicy::Exclude)?;
        self.with_inherited_name(col)
    }

    /// Inclusive cumulative minimum. Inherits the source Series' name.
    pub fn cummin(&self) -> Result<Series> {
        let cv = self.column.view()?;
        let col = reduction::scan(&cv, ScanAggregation::Min, ScanType::Inclusive, NullPolicy::Exclude)?;
        self.with_inherited_name(col)
    }

    /// Inclusive cumulative maximum. Inherits the source Series' name.
    pub fn cummax(&self) -> Result<Series> {
        let cv = self.column.view()?;
        let col = reduction::scan(&cv, ScanAggregation::Max, ScanType::Inclusive, NullPolicy::Exclude)?;
        self.with_inherited_name(col)
    }

    /// Inclusive cumulative product. Inherits the source Series' name.
    pub fn cumprod(&self) -> Result<Series> {
        let cv = self.column.view()?;
        let col = reduction::scan(&cv, ScanAggregation::Product, ScanType::Inclusive, NullPolicy::Exclude)?;
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
