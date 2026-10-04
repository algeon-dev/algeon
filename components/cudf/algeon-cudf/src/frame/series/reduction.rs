//! `Series` convenience methods for reduction, scan, and unary operations.
//!
//! Prefer the `*_value` helpers when you want an ordinary host result from a
//! high-level workflow. They preserve cuDF/libcudf reduction semantics while
//! avoiding an explicit drop into [`Scalar`](crate::core::Scalar).

use super::{context::SeriesOps, owned::Series};
#[cfg(any(test, feature = "test-hooks"))]
use crate::runtime::Runtime;
use crate::{
    core::{
        error::{self, Error, ErrorCode, ErrorKind, FACT_TYPE_ID, Result},
        scalar::Scalar,
        types::{CudfType, DataType, NullPolicy, TypeId},
    },
    ops::{
        reduction::{ReduceAgg, ScanAgg, ScanType},
        unary::UnaryOperator,
    },
    runtime::{BudgetedExecutionOutcome, Exec, ExecutionBudgetToken},
};

/// Conservative output-shape facts for a prepared scalar reduction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalarReductionOutputShape {
    output_type: DataType,
    output_rows: usize,
}

impl ScalarReductionOutputShape {
    fn new(output_type: DataType) -> Self {
        Self { output_type, output_rows: 1 }
    }

    /// Data type of the scalar produced by the reduction.
    pub fn output_type(&self) -> DataType {
        self.output_type
    }

    /// Number of logical rows produced by a scalar reduction.
    pub fn output_rows(&self) -> usize {
        self.output_rows
    }
}

/// Result of a prepared scalar reduction.
pub struct ScalarReductionOutput {
    scalar: Scalar,
}

impl ScalarReductionOutput {
    fn new(scalar: Scalar) -> Self {
        Self { scalar }
    }

    /// Borrow the produced scalar.
    pub fn scalar(&self) -> &Scalar {
        &self.scalar
    }

    /// Consume this output and return the produced scalar.
    pub fn into_scalar(self) -> Scalar {
        self.scalar
    }

    /// Number of logical rows produced by a scalar reduction.
    pub fn output_rows(&self) -> usize {
        1
    }
}

/// Prepared scalar reduction over a retained input series.
///
/// Preparation validates the reduction output type and records the stable
/// scalar output shape. Execution then runs where an [`Exec`] selects or
/// under a hard execution-budget token.
pub struct SeriesPreparedScalarReduction<'a> {
    input: &'a Series,
    aggregation: ReduceAgg,
    output_shape: ScalarReductionOutputShape,
}

impl<'a> SeriesPreparedScalarReduction<'a> {
    fn new(input: &'a Series, aggregation: ReduceAgg, output_shape: ScalarReductionOutputShape) -> Self {
        Self { input, aggregation, output_shape }
    }

    /// Borrow the retained input series.
    pub fn input(&self) -> &Series {
        self.input
    }

    /// Borrow the reduction aggregation.
    pub fn aggregation(&self) -> &ReduceAgg {
        &self.aggregation
    }

    /// Data type of the scalar produced by this prepared reduction.
    pub fn output_type(&self) -> DataType {
        self.output_shape.output_type()
    }

    /// Borrow the prepared output-shape facts.
    pub fn output_shape(&self) -> &ScalarReductionOutputShape {
        &self.output_shape
    }

    /// Execute the prepared scalar reduction.
    pub fn execute(&self, exec: Exec<'_>) -> Result<ScalarReductionOutput> {
        self.execute_in_context(exec, "series::prepared_scalar_reduction::execute", |error| error)
    }

    /// Execute the prepared scalar reduction with a hard execution-budget token.
    ///
    /// Execution allocations are routed through the token's stream-bound
    /// reservation. The report records the fixed scalar output row count. It
    /// does not estimate scalar retained/output bytes.
    pub fn execute_with_budget(
        &self,
        token: &ExecutionBudgetToken,
    ) -> Result<BudgetedExecutionOutcome<ScalarReductionOutput>> {
        let operation = "series::prepared_scalar_reduction::execute_with_budget";
        let ctx = token.exec();
        let output = self.execute_in_context(ctx, operation, |error| {
            annotate_scalar_reduction_budget_execution_error(error, token, operation)
        })?;
        let report = token.reservation_enforced_report().with_output_rows(output.output_rows());

        Ok(BudgetedExecutionOutcome::new(output, report))
    }

    fn execute_in_context<F>(
        &self,
        ctx: Exec<'_>,
        operation: &'static str,
        map_execution_error: F,
    ) -> Result<ScalarReductionOutput>
    where
        F: FnOnce(Error) -> Error,
    {
        self.input
            .reduce_prepared_in_context(
                &self.aggregation,
                &self.output_shape.output_type,
                ctx,
                operation,
                map_execution_error,
            )
            .map(ScalarReductionOutput::new)
            .map_err(|error| error.reanchor_if_needed(operation))
    }
}

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
            _ => Err(Error::new_with(ErrorCode::UnsupportedDtype, operation, "unsupported type for sum reduction")
                .with_fact(FACT_TYPE_ID, input.id())),
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
    if !scalar.is_valid().map_err(|e| e.reanchor(operation))? {
        return Ok(0);
    }

    let count = scalar.value::<i32>().map_err(|e| e.reanchor(operation))?;
    usize::try_from(count).map_err(|_| negative_distinct_count_error(operation, count))
}

fn negative_distinct_count_error(operation: &'static str, count: i32) -> Error {
    error::foreign_invariant_violation(
        operation,
        "distinct count must be >= 0",
        "distinct_count",
        "negative_count",
        "ensure the foreign reduction backend returns a non-negative distinct count",
    )
    .with_detail("distinct_count", count.to_string())
}

#[cfg(any(test, feature = "test-hooks"))]
pub(crate) fn trigger_nunique_negative_distinct_count_for_test() -> Result<()> {
    let runtime = Runtime::new()?;
    let scalar = runtime.scalar(-1i32)?;
    scalar_to_usize_count(scalar, "series::nunique").map(|_| ())
}

impl Series {
    /// Prepare a scalar reduction over this series.
    ///
    /// This validates and records the reduction's output type and fixed scalar
    /// output shape without claiming any libcudf temporary-workspace preflight
    /// coverage.
    pub fn prepare_scalar_reduction(&self, agg: ReduceAgg) -> Result<SeriesPreparedScalarReduction<'_>> {
        let operation = "series::prepare_scalar_reduction";
        let input_type = self.dtype().map_err(|error| error.reanchor_if_needed(operation))?;
        let output_type = infer_reduce_output_type(&input_type, &agg, operation)?;
        Ok(SeriesPreparedScalarReduction::new(self, agg, ScalarReductionOutputShape::new(output_type)))
    }

    pub(super) fn nunique_with_context(&self, ctx: Exec<'_>, operation: &'static str) -> Result<usize> {
        let scalar = self.reduce_in_context(ReduceAgg::NUnique { null_policy: NullPolicy::Exclude }, ctx, operation)?;
        scalar_to_usize_count(scalar, operation)
    }

    // -- Scalar reductions ---------------------------------------------------

    // -- Scan (cumulative) operations ----------------------------------------

    // -- Unary operations ----------------------------------------------------
}

fn annotate_scalar_reduction_budget_execution_error(
    error: Error,
    token: &ExecutionBudgetToken,
    operation: &'static str,
) -> Error {
    match error.kind() {
        ErrorKind::ResourceExhausted | ErrorKind::DependencyFailure => token.annotate_error(error, operation),
        _ => error.reanchor_if_needed(operation),
    }
}

impl SeriesOps<'_> {
    /// Reduce the series to a single scalar: sum of all elements.
    ///
    /// Output type: integer types → `Int64`, float → same float, `Bool8` → `Int64`.
    /// Nulls are excluded from the sum.
    ///
    /// # Cost
    /// GPU computation (synchronizes to return host-accessible scalar).
    pub fn sum(&self) -> Result<Scalar> {
        self.series.reduce_in_context(ReduceAgg::Sum, self.ctx, "series::sum")
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
    /// let total = s.on(rt.exec()).sum_value::<i64>().unwrap();
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
        self.series.reduce_in_context(ReduceAgg::Min, self.ctx, "series::min")
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
    /// let min = s.on(rt.exec()).min_value::<i32>().unwrap();
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
        self.series.reduce_in_context(ReduceAgg::Max, self.ctx, "series::max")
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
        self.series.reduce_in_context(ReduceAgg::Mean, self.ctx, "series::mean")
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
    /// let mean = s.on(rt.exec()).mean_value().unwrap();
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
        self.series.reduce_in_context(ReduceAgg::Median, self.ctx, "series::median")
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
        self.series.reduce_in_context(ReduceAgg::Std { ddof }, self.ctx, "series::std")
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
        self.series.reduce_in_context(ReduceAgg::Variance { ddof }, self.ctx, "series::var")
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
        self.series.reduce_in_context(ReduceAgg::Any, self.ctx, "series::any")
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
        self.series.reduce_in_context(ReduceAgg::All, self.ctx, "series::all")
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
        self.series.reduce_in_context(ReduceAgg::Product, self.ctx, "series::product")
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
        self.series.minmax_with_context(self.ctx, "series::minmax")
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
    /// let (min, max) = s.on(rt.exec()).minmax_values::<i32>().unwrap();
    ///
    /// assert_eq!((min, max), (Some(10), Some(30)));
    /// ```
    pub fn minmax_values<T: CudfType>(&self) -> Result<(Option<T>, Option<T>)> {
        let (min, max) = self.minmax()?;
        Ok((scalar_to_optional_value(min)?, scalar_to_optional_value(max)?))
    }

    /// Inclusive cumulative sum.
    ///
    /// Nulls are excluded from the running total (filled with null in output).
    ///
    /// # Cost
    /// GPU-allocating: produces a new column.
    pub fn cumsum(&self) -> Result<Series> {
        self.series.scan_with_context(
            ScanAgg::Sum,
            ScanType::Inclusive,
            NullPolicy::Exclude,
            self.ctx,
            "series::cumsum",
        )
    }

    /// Inclusive cumulative minimum.
    pub fn cummin(&self) -> Result<Series> {
        self.series.scan_with_context(
            ScanAgg::Min,
            ScanType::Inclusive,
            NullPolicy::Exclude,
            self.ctx,
            "series::cummin",
        )
    }

    /// Inclusive cumulative maximum.
    pub fn cummax(&self) -> Result<Series> {
        self.series.scan_with_context(
            ScanAgg::Max,
            ScanType::Inclusive,
            NullPolicy::Exclude,
            self.ctx,
            "series::cummax",
        )
    }

    /// Inclusive cumulative product.
    pub fn cumprod(&self) -> Result<Series> {
        self.series.scan_with_context(
            ScanAgg::Product,
            ScanType::Inclusive,
            NullPolicy::Exclude,
            self.ctx,
            "series::cumprod",
        )
    }

    /// Element-wise NaN check. Returns a `BOOL8` mask.
    ///
    /// Only meaningful for float types; integer types will return all `false`.
    ///
    /// # Cost
    /// GPU-allocating: produces a new BOOL8 column.
    pub fn is_nan(&self) -> Result<Series> {
        self.series.is_nan_with_context(self.ctx, "series::is_nan")
    }

    /// Element-wise not-NaN check. Returns a `BOOL8` mask.
    pub fn is_not_nan(&self) -> Result<Series> {
        self.series.is_not_nan_with_context(self.ctx, "series::is_not_nan")
    }

    /// Element-wise absolute value.
    ///
    /// # Cost
    /// GPU-allocating: produces a new column.
    pub fn abs(&self) -> Result<Series> {
        self.series.unary_with_context(UnaryOperator::Abs, self.ctx, "series::abs")
    }

    /// Element-wise arithmetic negation.
    ///
    /// # Cost
    /// GPU-allocating: produces a new column.
    pub fn neg(&self) -> Result<Series> {
        self.series.unary_with_context(UnaryOperator::Negate, self.ctx, "series::neg")
    }

    /// Element-wise logical NOT for `Bool8` columns.
    ///
    /// # Cost
    /// GPU-allocating: produces a new BOOL8 column.
    pub fn logical_not(&self) -> Result<Series> {
        self.series.logical_not_with_context(self.ctx, "series::logical_not")
    }
}
