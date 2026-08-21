//! `Series` convenience methods for reduction, scan, and unary operations.
//!
//! Prefer the `*_value` helpers when you want an ordinary host result from a
//! high-level workflow. They preserve cuDF/libcudf reduction semantics while
//! avoiding an explicit drop into [`Scalar`](crate::core::Scalar).

use super::{context::default_execution_context, owned::Series};
use crate::{
    core::{
        error::{self, Error, ErrorCode, FACT_TYPE_ID, Result},
        scalar::Scalar,
        types::{CudfType, DataType, NullPolicy, TypeId},
    },
    ops::{
        reduction::{ReduceAgg, ScanAgg, ScanType},
        unary::UnaryOperator,
    },
    runtime::{BudgetedExecutionOutcome, ExecutionBudgetToken, ExecutionOverrides, Runtime},
};

const SCALAR_REDUCTION_TEMPORARY_WORKSPACE_CONTRACT_GAP: &str =
    "libcudf_scalar_reduction_temporary_workspace_preflight_unavailable";

const DISTINCT_WORKSPACE_PREFLIGHT_CONTRACT: &str = "libcudf_distinct_count_workspace_preflight";

/// Allocation-free native workspace bound for one non-null fixed-width distinct key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScalarReductionDistinctWorkspacePreflight {
    preprocessing_retained_bytes: usize,
    static_set_storage_bytes: usize,
    insertion_count_bytes: usize,
    active_workspace_peak_bytes: usize,
}

/// Allocation-free hard bound for the additional workspace owned by one grouped NUnique value column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupedNuniqueWorkspacePreflight {
    argument_sanitation_bytes: usize,
    distinct_state_bytes: usize,
    active_workspace_peak_bytes: usize,
}

impl GroupedNuniqueWorkspacePreflight {
    /// Resolve the grouped NUnique workspace bound from row and retained-column metadata.
    pub fn for_metadata(num_rows: usize, argument_retained_bytes: usize) -> Result<Self> {
        let operation = "groupby::nunique_workspace_preflight";
        let distinct =
            ScalarReductionDistinctWorkspacePreflight::for_metadata(num_rows, DataType::new(TypeId::Int64), 0)
                .map_err(|error| error.reanchor_if_needed(operation))?;
        let row_flags_bytes = num_rows.checked_mul(std::mem::size_of::<i32>()).ok_or_else(|| {
            Error::new_with(ErrorCode::SizeOverflow, operation, "grouped NUnique row flags overflow usize")
                .with_detail("reason", "row_flags_overflowed_usize")
        })?;
        if distinct.static_set_storage_bytes() < row_flags_bytes {
            return Err(Error::new_with(
                ErrorCode::ForeignInvariantViolation,
                operation,
                "distinct-set storage does not dominate grouped NUnique row flags",
            )
            .with_detail("reason", "distinct_state_does_not_dominate_row_flags")
            .with_detail("distinct_state_bytes", distinct.static_set_storage_bytes().to_string())
            .with_detail("row_flags_bytes", row_flags_bytes.to_string()));
        }
        let distinct_state_bytes = distinct.active_workspace_peak_bytes();
        let active_workspace_peak_bytes =
            argument_retained_bytes.checked_add(distinct_state_bytes).ok_or_else(|| {
                Error::new_with(ErrorCode::SizeOverflow, operation, "grouped NUnique workspace overflows usize")
                    .with_detail("reason", "workspace_sum_overflowed_usize")
            })?;
        Ok(Self {
            argument_sanitation_bytes: argument_retained_bytes,
            distinct_state_bytes,
            active_workspace_peak_bytes,
        })
    }

    /// Bytes reserved for worst-case nullable or variable-width argument sanitation.
    pub fn argument_sanitation_bytes(self) -> usize {
        self.argument_sanitation_bytes
    }

    /// Native distinct-state bound that dominates the per-row NUnique flag vector.
    pub fn distinct_state_bytes(self) -> usize {
        self.distinct_state_bytes
    }

    /// Checked sum of simultaneously live grouped NUnique workspace components.
    pub fn active_workspace_peak_bytes(self) -> usize {
        self.active_workspace_peak_bytes
    }
}

impl ScalarReductionDistinctWorkspacePreflight {
    /// Resolve the libcudf-owned workspace contract from input metadata.
    pub fn for_metadata(num_rows: usize, key_type: DataType, null_count: usize) -> Result<Self> {
        let operation = "series::scalar_reduction_distinct_workspace_preflight";
        if num_rows > i32::MAX as usize {
            return Err(Error::new_with(
                ErrorCode::SizeOverflow,
                operation,
                "distinct-count workspace preflight rows exceed cudf::size_type",
            )
            .with_detail("reason", "row_count_exceeds_cudf_size_type")
            .with_detail("row_count", num_rows.to_string()));
        }
        if null_count != 0 {
            return Err(Error::new_with(
                ErrorCode::NullUnexpected,
                operation,
                "distinct-count workspace preflight requires a non-null key",
            )
            .with_detail("reason", "nullable_key_unsupported")
            .with_detail("null_count", null_count.to_string()));
        }
        if !distinct_workspace_key_supported(key_type.id()) {
            return Err(Error::new_with(
                ErrorCode::UnsupportedDtype,
                operation,
                "distinct-count workspace preflight requires a fixed-width non-nested key",
            )
            .with_fact(FACT_TYPE_ID, key_type.id())
            .with_detail("reason", "key_type_not_fixed_width"));
        }
        let mut raw = cudf_sys::cudf_distinct_count_workspace_preflight_result_t {
            preprocessing_retained_bytes: 0,
            static_set_storage_bytes: 0,
            insertion_count_bytes: 0,
            active_workspace_peak_bytes: 0,
        };
        let status = unsafe {
            cudf_sys::cudf_distinct_count_workspace_preflight(
                error::usize_to_i64(num_rows, operation, "num_rows")?,
                key_type.id().to_raw(),
                error::usize_to_i64(null_count, operation, "null_count")?,
                1,
                &mut raw,
            )
        };
        error::check_status_in(status, operation)?;
        let checked = |bytes: u64, field: &'static str| {
            usize::try_from(bytes).map_err(|_| {
                Error::new_with(ErrorCode::SizeOverflow, operation, "native distinct workspace bytes exceed usize")
                    .with_detail("reason", "native_byte_count_exceeds_usize")
                    .with_detail("resource", field)
                    .with_detail("native_bytes", bytes.to_string())
            })
        };
        let preprocessing_retained_bytes = checked(raw.preprocessing_retained_bytes, "preprocessing_retained_bytes")?;
        let static_set_storage_bytes = checked(raw.static_set_storage_bytes, "static_set_storage_bytes")?;
        let insertion_count_bytes = checked(raw.insertion_count_bytes, "insertion_count_bytes")?;
        let active_workspace_peak_bytes = checked(raw.active_workspace_peak_bytes, "active_workspace_peak_bytes")?;
        let expected_peak = preprocessing_retained_bytes
            .checked_add(static_set_storage_bytes)
            .and_then(|bytes| bytes.checked_add(insertion_count_bytes))
            .ok_or_else(|| {
                Error::new_with(
                    ErrorCode::SizeOverflow,
                    operation,
                    "distinct-count workspace component sum overflowed usize",
                )
                .with_detail("reason", "component_sum_overflowed_usize")
            })?;
        if active_workspace_peak_bytes != expected_peak {
            return Err(Error::new_with(
                ErrorCode::ForeignInvariantViolation,
                operation,
                "native distinct workspace peak is not the checked component sum",
            )
            .with_detail("reason", "active_workspace_peak_component_sum_mismatch")
            .with_detail("expected_peak_bytes", expected_peak.to_string())
            .with_detail("reported_peak_bytes", active_workspace_peak_bytes.to_string()));
        }
        Ok(Self {
            preprocessing_retained_bytes,
            static_set_storage_bytes,
            insertion_count_bytes,
            active_workspace_peak_bytes,
        })
    }

    /// Bytes retained by row preprocessing while the hash set is active.
    pub fn preprocessing_retained_bytes(self) -> usize {
        self.preprocessing_retained_bytes
    }

    /// Bytes allocated by the pinned cuco static-set storage owner.
    pub fn static_set_storage_bytes(self) -> usize {
        self.static_set_storage_bytes
    }

    /// Bytes for cuco's synchronous insertion-result counter.
    pub fn insertion_count_bytes(self) -> usize {
        self.insertion_count_bytes
    }

    /// Checked sum of all simultaneously live distinct workspace components.
    pub fn active_workspace_peak_bytes(self) -> usize {
        self.active_workspace_peak_bytes
    }
}

fn distinct_workspace_key_supported(type_id: TypeId) -> bool {
    !matches!(type_id, TypeId::Empty | TypeId::Dictionary32 | TypeId::String | TypeId::List | TypeId::Struct)
}

fn distinct_workspace_context_facts(
    preflight: Option<ScalarReductionDistinctWorkspacePreflight>,
) -> Vec<(&'static str, String)> {
    let mut facts = vec![(
        "distinct_workspace_preflight_status",
        if preflight.is_some() { "supported" } else { "unresolved_unsupported_shape" }.to_string(),
    )];
    if let Some(preflight) = preflight {
        facts.extend([
            ("distinct_workspace_preflight_contract", DISTINCT_WORKSPACE_PREFLIGHT_CONTRACT.to_string()),
            ("distinct_workspace_preprocessing_retained_bytes", preflight.preprocessing_retained_bytes().to_string()),
            ("distinct_workspace_static_set_storage_bytes", preflight.static_set_storage_bytes().to_string()),
            ("distinct_workspace_insertion_count_bytes", preflight.insertion_count_bytes().to_string()),
            ("distinct_workspace_active_peak_bytes", preflight.active_workspace_peak_bytes().to_string()),
        ]);
    }
    facts
}

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
/// scalar output shape. Execution may then be routed through a default
/// execution context or a hard execution-budget token.
pub struct SeriesPreparedScalarReduction<'a> {
    input: &'a Series,
    aggregation: ReduceAgg,
    output_shape: ScalarReductionOutputShape,
    distinct_workspace_preflight: Option<ScalarReductionDistinctWorkspacePreflight>,
}

impl<'a> SeriesPreparedScalarReduction<'a> {
    fn new(
        input: &'a Series,
        aggregation: ReduceAgg,
        output_shape: ScalarReductionOutputShape,
        distinct_workspace_preflight: Option<ScalarReductionDistinctWorkspacePreflight>,
    ) -> Self {
        Self { input, aggregation, output_shape, distinct_workspace_preflight }
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

    /// Exact native workspace owner contract for supported `NUnique` inputs.
    pub fn distinct_workspace_preflight(&self) -> Option<ScalarReductionDistinctWorkspacePreflight> {
        self.distinct_workspace_preflight
    }

    /// Execute the prepared scalar reduction.
    pub fn execute(&self) -> Result<ScalarReductionOutput> {
        self.execute_in_context(&default_execution_context(), "series::prepared_scalar_reduction::execute", |error| {
            error
        })
    }

    /// Execute the prepared scalar reduction with a hard execution-budget token.
    ///
    /// Execution allocations are routed through the token's stream-bound
    /// reservation. The report records the fixed scalar output row count and
    /// the unresolved libcudf temporary-workspace preflight contract. It does
    /// not estimate scalar retained/output bytes.
    pub fn execute_with_budget(
        &self,
        token: &ExecutionBudgetToken,
    ) -> Result<BudgetedExecutionOutcome<ScalarReductionOutput>> {
        let operation = "series::prepared_scalar_reduction::execute_with_budget";
        let ctx = token.execution_overrides();
        let output = self.execute_in_context(&ctx, operation, |error| {
            annotate_scalar_reduction_budget_execution_error(error, token, operation)
        })?;
        let mut report = token
            .reservation_enforced_report()
            .with_output_rows(output.output_rows())
            .with_context_facts(distinct_workspace_context_facts(self.distinct_workspace_preflight));
        if self.distinct_workspace_preflight.is_none() {
            report = report.with_unresolved_contract(SCALAR_REDUCTION_TEMPORARY_WORKSPACE_CONTRACT_GAP);
        }

        Ok(BudgetedExecutionOutcome::new(output, report))
    }

    fn execute_in_context<F>(
        &self,
        ctx: &ExecutionOverrides<'_>,
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
        let distinct_workspace_preflight = match &agg {
            ReduceAgg::NUnique { .. }
                if self.null_count() == 0 && distinct_workspace_key_supported(input_type.id()) =>
            {
                Some(ScalarReductionDistinctWorkspacePreflight::for_metadata(self.len(), input_type, 0)?)
            },
            _ => None,
        };
        Ok(SeriesPreparedScalarReduction::new(
            self,
            agg,
            ScalarReductionOutputShape::new(output_type),
            distinct_workspace_preflight,
        ))
    }

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

fn annotate_scalar_reduction_budget_execution_error(
    error: Error,
    token: &ExecutionBudgetToken,
    operation: &'static str,
) -> Error {
    match error.kind() {
        crate::core::error::ErrorKind::ResourceExhausted | crate::core::error::ErrorKind::DependencyFailure => {
            token.annotate_error(error, operation)
        },
        _ => error.reanchor_if_needed(operation),
    }
}
