use std::sync::Arc;

use super::{
    owned::Series,
    reduction::{infer_reduce_output_type, scalar_to_optional_value},
};
use crate::{
    core::{
        column::copy_view_to_owned,
        error::{Error, ErrorKind, Result},
        scalar::Scalar,
        types::{CudfType, DataType, NullPolicy, TimestampResolution, TypeId},
    },
    internal::temp_table::TempTableView,
    ops::{
        binary::{self, BinaryOperator},
        dictionary,
        reduction::{self, ReduceAggregation, ScanAggregation, ScanType},
        replace::{self, FillPolicy},
        sorting::SortOptions,
        stream_compaction::{self, DuplicateKeepOption, NanEquality, NullEquality},
        strings::{self, StripSide},
        unary::{self, UnaryOperator},
    },
    runtime::{ExecutionContext, execution_domain::ExecutionDomain},
};

/// Generate a simple context helper that resolves the output domain, gets a column view,
/// calls the given op with exec_options, sets the output domain, and inherits the name.
macro_rules! series_context_op {
    // Unary with fixed operation string
    (unary fn $method:ident($($param:ident : $pty:ty),*), $operation:literal,
     |$cv:ident, $exec:ident, $op:ident| $call:expr) => {
        fn $method(&self, $($param: $pty,)* ctx: &ExecutionContext<'_>) -> Result<Series> {
            let $op = $operation;
            let output_domain = self.resolve_context_output_domain(ctx, $op)?;
            let $cv = self.column.view()?;
            let $exec = output_domain.explicit_exec_options_raw();
            let col = $call?.with_execution_domain(output_domain);
            self.with_inherited_name(col)
        }
    };
    // Unary with caller-supplied operation string
    (unary fn $method:ident($($param:ident : $pty:ty),*),
     caller_op |$cv:ident, $exec:ident, $op:ident| $call:expr) => {
        fn $method(&self, $($param: $pty,)* ctx: &ExecutionContext<'_>, $op: &'static str) -> Result<Series> {
            let output_domain = self.resolve_context_output_domain(ctx, $op)?;
            let $cv = self.column.view()?;
            let $exec = output_domain.explicit_exec_options_raw();
            let col = $call?.with_execution_domain(output_domain);
            self.with_inherited_name(col)
        }
    };
    // Binary (two-Series): adds require_same_device + fence for the other Series
    (binary fn $method:ident($other:ident : &Series),
     $operation:literal, |$cv:ident, $other_cv:ident, $exec:ident, $op:ident| $call:expr) => {
        fn $method(
            &self,
            $other: &Series,
            ctx: &ExecutionContext<'_>,
        ) -> Result<Series> {
            let $op = $operation;
            let output_domain = self.resolve_context_output_domain(ctx, $op)?;
            ExecutionDomain::require_same_device_for_mixed_inputs(
                self.column.execution_domain(),
                $other.column.execution_domain(),
                $op,
            )?;
            if !ExecutionDomain::same_provenance($other.column.execution_domain(), &output_domain) {
                $other.column.execution_domain().fence_stream($op)?;
            }
            let $cv = self.column.view()?;
            let $other_cv = $other.column.view()?;
            let $exec = output_domain.explicit_exec_options_raw();
            let col = $call?.with_execution_domain(output_domain);
            self.with_inherited_name(col)
        }
    };
}

/// Scoped borrowed execution override for one `Series` operation chain.
#[must_use]
pub struct SeriesExecutionScope<'series, 'ctx, 'handles> {
    series: &'series Series,
    ctx: &'ctx ExecutionContext<'handles>,
}

impl Series {
    /// Borrow a per-operation execution override for subsequent high-level calls.
    pub fn in_context<'series, 'ctx, 'handles>(
        &'series self,
        ctx: &'ctx ExecutionContext<'handles>,
    ) -> SeriesExecutionScope<'series, 'ctx, 'handles> {
        SeriesExecutionScope { series: self, ctx }
    }

    fn resolve_context_output_domain(
        &self,
        ctx: &ExecutionContext<'_>,
        operation: &'static str,
    ) -> Result<Arc<ExecutionDomain>> {
        let output_domain = ctx.resolve_output_domain(self.column.execution_domain())?;
        if !ExecutionDomain::same_provenance(self.column.execution_domain(), &output_domain) {
            self.column.execution_domain().fence_stream(operation)?;
        }
        Ok(output_domain)
    }

    // ── Hand-written helpers (unique control flow) ──────────────────────

    fn sort_values_in_context(&self, options: SortOptions, ctx: &ExecutionContext<'_>) -> Result<Series> {
        let operation = "series::in_context::sort_values";
        let output_domain = self.resolve_context_output_domain(ctx, operation)?;
        let (orders, null_orders) = options.resolve(1)?;
        let cv = self.column.view()?;
        let ttv = TempTableView::from_column_views(vec![cv])?;
        let exec_options = output_domain.explicit_exec_options_raw();
        let sorted = match exec_options.as_ref() {
            Some(exec_options) => crate::ops::sorting::sort_table_impl(
                ttv.table(),
                Some(&orders),
                null_orders.as_deref(),
                options.stable,
                Some(exec_options),
                operation,
            )?,
            None => {
                if options.stable {
                    crate::ops::sorting::stable_sort(ttv.table(), Some(&orders), null_orders.as_deref())?
                } else {
                    crate::ops::sorting::sort(ttv.table(), Some(&orders), null_orders.as_deref())?
                }
            },
        }
        .with_execution_domain(output_domain);
        let mut cols = sorted.into_columns()?;
        self.with_inherited_name(cols.remove(0))
    }

    fn to_datetime_in_context(
        &self,
        resolution: Option<TimestampResolution>,
        ctx: &ExecutionContext<'_>,
    ) -> Result<Series> {
        let resolution = resolution.unwrap_or(TimestampResolution::Microseconds);
        let source_type = self.dtype()?.id();
        if source_type.is_numeric() {
            let duration =
                self.cast_in_context(resolution.duration_type_id(), ctx, "series::in_context::to_datetime_duration")?;
            duration.cast_in_context(resolution.type_id(), ctx, "series::in_context::to_datetime")
        } else {
            self.cast_in_context(resolution.type_id(), ctx, "series::in_context::to_datetime")
        }
    }

    fn factorize_in_context(&self, ctx: &ExecutionContext<'_>) -> Result<(Series, Series)> {
        let operation = "series::in_context::factorize";
        let output_domain = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let exec_options = output_domain.explicit_exec_options_raw();
        let dict = dictionary::encode_impl(&cv, TypeId::Int32, exec_options.as_ref(), operation)?
            .with_execution_domain(Arc::clone(&output_domain));
        let dict_view = dict.view()?;
        let indices_view = dictionary::indices(&dict_view)?;
        let keys_view = dictionary::keys(&dict_view)?;

        let codes = Series::from_column(copy_view_to_owned(&indices_view)?)?;
        let uniques = Series::from_column(copy_view_to_owned(&keys_view)?)?;
        if let Some(name) = self.name() {
            Ok((codes.rename(format!("{name}_codes")), uniques.rename(format!("{name}_uniques"))))
        } else {
            Ok((codes, uniques))
        }
    }

    fn fillna_policy_in_context(&self, policy: FillPolicy, ctx: &ExecutionContext<'_>) -> Result<Series> {
        let operation = match policy {
            FillPolicy::Preceding => "series::in_context::ffill",
            FillPolicy::Following => "series::in_context::bfill",
        };
        let output_domain = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let exec_options = output_domain.explicit_exec_options_raw();
        let col = replace::replace_nulls_policy_with_exec_options(&cv, policy, exec_options.as_ref(), operation)?
            .with_execution_domain(output_domain);
        self.with_inherited_name(col)
    }

    fn unique_in_context(&self, ctx: &ExecutionContext<'_>) -> Result<Series> {
        let operation = "series::in_context::unique";
        let output_domain = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let ttv = TempTableView::from_column_views(vec![cv])?;
        let exec_options = output_domain.explicit_exec_options_raw();
        let result = stream_compaction::distinct_impl(
            ttv.table(),
            &[0],
            DuplicateKeepOption::KeepAny,
            NullEquality::Equal,
            NanEquality::AllEqual,
            exec_options.as_ref(),
            operation,
        )?
        .with_execution_domain(output_domain);
        let mut cols = result.into_columns()?;
        self.with_inherited_name(cols.remove(0))
    }

    fn reduce_in_context(
        &self,
        agg: ReduceAggregation,
        ctx: &ExecutionContext<'_>,
        operation: &'static str,
    ) -> Result<Scalar> {
        let output_domain = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let input_type = self.dtype().map_err(|e| e.with_operation(operation))?;
        let output_type = infer_reduce_output_type(&input_type, &agg).map_err(|e| e.with_operation(operation))?;
        let exec_options = output_domain.explicit_exec_options_raw();
        reduction::reduce_impl(&cv, &agg, &output_type, exec_options.as_ref(), operation)
            .map(|scalar| scalar.with_execution_domain(output_domain))
    }

    fn minmax_in_context(&self, ctx: &ExecutionContext<'_>) -> Result<(Scalar, Scalar)> {
        let operation = "series::in_context::minmax";
        let output_domain = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let exec_options = output_domain.explicit_exec_options_raw();
        reduction::minmax_impl(&cv, exec_options.as_ref(), operation).map(|(min, max)| {
            (min.with_execution_domain(Arc::clone(&output_domain)), max.with_execution_domain(output_domain))
        })
    }

    // ── Macro-generated unary helpers ───────────────────────────────────

    series_context_op!(unary fn is_nan_in_context(), "series::in_context::is_nan",
        |cv, exec, op| unary::is_nan_impl(&cv, exec.as_ref(), op));

    series_context_op!(unary fn is_not_nan_in_context(), "series::in_context::is_not_nan",
        |cv, exec, op| unary::is_not_nan_impl(&cv, exec.as_ref(), op));

    series_context_op!(unary fn logical_not_in_context(), "series::in_context::logical_not",
        |cv, exec, op| unary::unary_operation_impl(&cv, UnaryOperator::Not, exec.as_ref(), op));

    series_context_op!(unary fn is_null_in_context(), "series::in_context::is_null",
        |cv, exec, op| unary::is_null_impl(&cv, exec.as_ref(), op));

    series_context_op!(unary fn is_not_null_in_context(), "series::in_context::is_not_null",
        |cv, exec, op| unary::is_valid_impl(&cv, exec.as_ref(), op));

    // Unary helpers with caller-supplied operation string
    series_context_op!(unary fn unary_in_context(op: UnaryOperator),
        caller_op |cv, exec, operation| unary::unary_operation_impl(&cv, op, exec.as_ref(), operation));

    series_context_op!(unary fn scan_in_context(agg: ScanAggregation, scan_type: ScanType, null_policy: NullPolicy),
        caller_op |cv, exec, operation| reduction::scan_impl(&cv, agg, scan_type, null_policy, exec.as_ref(), operation));

    series_context_op!(unary fn string_unary_in_context(
        op: fn(&crate::core::column::ColumnView<'_>, Option<&cudf_sys::cudf_exec_options_t>, &'static str) -> Result<crate::core::column::Column>),
        caller_op |cv, exec, operation| op(&cv, exec.as_ref(), operation));

    fn strip_in_context(
        &self,
        side: StripSide,
        to_strip: &str,
        ctx: &ExecutionContext<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        let output_domain = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let scalar = Scalar::from_string_in_domain(to_strip, &output_domain)?;
        let exec_options = output_domain.explicit_exec_options_raw();
        let col = strings::strip_impl(&cv, side, &scalar, exec_options.as_ref(), operation)?
            .with_execution_domain(output_domain);
        self.with_inherited_name(col)
    }

    fn string_scalar_in_context(
        &self,
        target: &str,
        ctx: &ExecutionContext<'_>,
        operation: &'static str,
        op: fn(
            &crate::core::column::ColumnView<'_>,
            &Scalar,
            Option<&cudf_sys::cudf_exec_options_t>,
            &'static str,
        ) -> Result<crate::core::column::Column>,
    ) -> Result<Series> {
        let output_domain = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let scalar = Scalar::from_string_in_domain(target, &output_domain)?;
        let exec_options = output_domain.explicit_exec_options_raw();
        let col = op(&cv, &scalar, exec_options.as_ref(), operation)?.with_execution_domain(output_domain);
        self.with_inherited_name(col)
    }

    // ── Cast (special: `impl Into<DataType>` needs local conversion) ───

    fn cast_in_context(
        &self,
        target: impl Into<DataType>,
        ctx: &ExecutionContext<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        let output_domain = self.resolve_context_output_domain(ctx, operation)?;
        let target = target.into();
        let cv = self.column.view()?;
        let exec_options = output_domain.explicit_exec_options_raw();
        let col =
            unary::cast_impl(&cv, &target, exec_options.as_ref(), operation)?.with_execution_domain(output_domain);
        self.with_inherited_name(col)
    }

    // ── Fillna with scalar (generic over T: CudfType) ──────────────────

    fn fillna_in_context<T: CudfType>(&self, value: T, ctx: &ExecutionContext<'_>) -> Result<Series> {
        let operation = "series::in_context::fillna";
        let output_domain = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let scalar = Scalar::from_value_in_domain(value, &output_domain)?;
        let exec_options = output_domain.explicit_exec_options_raw();
        let col = replace::replace_nulls_with_exec_options(&cv, &scalar, exec_options.as_ref(), operation)?
            .with_execution_domain(output_domain);
        self.with_inherited_name(col)
    }

    // ── Macro-generated binary helpers ──────────────────────────────────

    series_context_op!(binary fn fillna_with_series_in_context(other: &Series),
        "series::in_context::fillna_with_series",
        |cv, other_cv, exec, op| replace::replace_nulls_column_with_exec_options(&cv, &other_cv, exec.as_ref(), op));

    series_context_op!(binary fn isin_in_context(values: &Series),
        "series::in_context::isin",
        |cv, values_cv, exec, op| crate::ops::search::contains_impl(&values_cv, &cv, exec.as_ref(), op));

    // ── Binary with type inference (hand-written — output type depends on both inputs) ──

    fn binary_series_in_context(
        &self,
        other: &Series,
        op: BinaryOperator,
        ctx: &ExecutionContext<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        let output_domain = self.resolve_context_output_domain(ctx, operation)?;
        ExecutionDomain::require_same_device_for_mixed_inputs(
            self.column.execution_domain(),
            other.column.execution_domain(),
            operation,
        )?;
        if !ExecutionDomain::same_provenance(other.column.execution_domain(), &output_domain) {
            other.column.execution_domain().fence_stream(operation)?;
        }
        let lhs = self.column.view()?;
        let rhs = other.column.view()?;
        let output_type = binary::infer_binary_output_type(lhs.data_type()?.id(), rhs.data_type()?.id(), op)?;
        let exec_options = output_domain.explicit_exec_options_raw();
        let col = binary::binary_operation_vv_impl(&lhs, &rhs, op, &output_type, exec_options.as_ref(), operation)?
            .with_execution_domain(output_domain);
        Series::from_column(col)
    }

    fn binary_scalar_in_context<T: CudfType>(
        &self,
        value: T,
        op: BinaryOperator,
        ctx: &ExecutionContext<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        let output_domain = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let scalar = Scalar::from_value_in_domain(value, &output_domain)?;
        let output_type = binary::infer_binary_output_type(cv.data_type()?.id(), scalar.data_type()?.id(), op)?;
        let exec_options = output_domain.explicit_exec_options_raw();
        let col = binary::binary_operation_vs_impl(&cv, &scalar, op, &output_type, exec_options.as_ref(), operation)?
            .with_execution_domain(output_domain);
        self.with_inherited_name(col)
    }

    fn binary_string_scalar_in_context(
        &self,
        value: &str,
        op: BinaryOperator,
        ctx: &ExecutionContext<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        let output_domain = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let scalar = Scalar::from_string_in_domain(value, &output_domain)?;
        let output_type = binary::infer_binary_output_type(cv.data_type()?.id(), scalar.data_type()?.id(), op)?;
        let exec_options = output_domain.explicit_exec_options_raw();
        let col = binary::binary_operation_vs_impl(&cv, &scalar, op, &output_type, exec_options.as_ref(), operation)?
            .with_execution_domain(output_domain);
        self.with_inherited_name(col)
    }
}

impl<'series, 'ctx, 'handles> SeriesExecutionScope<'series, 'ctx, 'handles> {
    /// Sort using the borrowed execution override.
    pub fn sort_values(&self, options: SortOptions) -> Result<Series> {
        self.series.sort_values_in_context(options, self.ctx)
    }

    /// Replace null values with a scalar using the borrowed execution override.
    pub fn fillna<T: CudfType>(&self, value: T) -> Result<Series> {
        self.series.fillna_in_context(value, self.ctx)
    }

    /// Replace null values using another series with the borrowed execution override.
    pub fn fillna_with_series(&self, other: &Series) -> Result<Series> {
        self.series.fillna_with_series_in_context(other, self.ctx)
    }

    /// Forward fill using the borrowed execution override.
    pub fn ffill(&self) -> Result<Series> {
        self.series.fillna_policy_in_context(FillPolicy::Preceding, self.ctx)
    }

    /// Backward fill using the borrowed execution override.
    pub fn bfill(&self) -> Result<Series> {
        self.series.fillna_policy_in_context(FillPolicy::Following, self.ctx)
    }

    /// Reduce to a scalar sum using the borrowed execution override.
    pub fn sum(&self) -> Result<Scalar> {
        self.series.reduce_in_context(ReduceAggregation::Sum, self.ctx, "series::in_context::sum")
    }

    /// Reduce to a host sum value using the borrowed execution override.
    pub fn sum_value<T: CudfType>(&self) -> Result<T> {
        self.sum()?.value::<T>()
    }

    /// Reduce to a scalar minimum using the borrowed execution override.
    pub fn min(&self) -> Result<Scalar> {
        self.series.reduce_in_context(ReduceAggregation::Min, self.ctx, "series::in_context::min")
    }

    /// Reduce to a host minimum value using the borrowed execution override.
    pub fn min_value<T: CudfType>(&self) -> Result<Option<T>> {
        scalar_to_optional_value(self.min()?)
    }

    /// Reduce to a scalar maximum using the borrowed execution override.
    pub fn max(&self) -> Result<Scalar> {
        self.series.reduce_in_context(ReduceAggregation::Max, self.ctx, "series::in_context::max")
    }

    /// Reduce to a host maximum value using the borrowed execution override.
    pub fn max_value<T: CudfType>(&self) -> Result<Option<T>> {
        scalar_to_optional_value(self.max()?)
    }

    /// Reduce to a scalar mean using the borrowed execution override.
    pub fn mean(&self) -> Result<Scalar> {
        self.series.reduce_in_context(ReduceAggregation::Mean, self.ctx, "series::in_context::mean")
    }

    /// Reduce to a host mean value using the borrowed execution override.
    pub fn mean_value(&self) -> Result<Option<f64>> {
        scalar_to_optional_value(self.mean()?)
    }

    /// Reduce to a scalar median using the borrowed execution override.
    pub fn median(&self) -> Result<Scalar> {
        self.series.reduce_in_context(ReduceAggregation::Median, self.ctx, "series::in_context::median")
    }

    /// Reduce to a host median value using the borrowed execution override.
    pub fn median_value(&self) -> Result<Option<f64>> {
        scalar_to_optional_value(self.median()?)
    }

    /// Reduce to a scalar standard deviation using the borrowed execution override.
    pub fn std(&self, ddof: i32) -> Result<Scalar> {
        self.series.reduce_in_context(ReduceAggregation::Std { ddof }, self.ctx, "series::in_context::std")
    }

    /// Reduce to a host standard deviation value using the borrowed execution override.
    pub fn std_value(&self, ddof: i32) -> Result<Option<f64>> {
        scalar_to_optional_value(self.std(ddof)?)
    }

    /// Reduce to a scalar variance using the borrowed execution override.
    pub fn var(&self, ddof: i32) -> Result<Scalar> {
        self.series.reduce_in_context(ReduceAggregation::Variance { ddof }, self.ctx, "series::in_context::var")
    }

    /// Reduce to a host variance value using the borrowed execution override.
    pub fn var_value(&self, ddof: i32) -> Result<Option<f64>> {
        scalar_to_optional_value(self.var(ddof)?)
    }

    /// Reduce to a scalar logical-any using the borrowed execution override.
    pub fn any(&self) -> Result<Scalar> {
        self.series.reduce_in_context(ReduceAggregation::Any, self.ctx, "series::in_context::any")
    }

    /// Reduce to a host bool logical-any using the borrowed execution override.
    pub fn any_value(&self) -> Result<bool> {
        Ok(scalar_to_optional_value(self.any()?)?.unwrap_or(false))
    }

    /// Reduce to a scalar logical-all using the borrowed execution override.
    pub fn all(&self) -> Result<Scalar> {
        self.series.reduce_in_context(ReduceAggregation::All, self.ctx, "series::in_context::all")
    }

    /// Reduce to a host bool logical-all using the borrowed execution override.
    pub fn all_value(&self) -> Result<bool> {
        Ok(scalar_to_optional_value(self.all()?)?.unwrap_or(true))
    }

    /// Reduce to a scalar product using the borrowed execution override.
    pub fn product(&self) -> Result<Scalar> {
        self.series.reduce_in_context(ReduceAggregation::Product, self.ctx, "series::in_context::product")
    }

    /// Reduce to a host product value using the borrowed execution override.
    pub fn product_value<T: CudfType>(&self) -> Result<Option<T>> {
        scalar_to_optional_value(self.product()?)
    }

    /// Reduce to a min/max pair using the borrowed execution override.
    pub fn minmax(&self) -> Result<(Scalar, Scalar)> {
        self.series.minmax_in_context(self.ctx)
    }

    /// Reduce to host min/max values using the borrowed execution override.
    pub fn minmax_values<T: CudfType>(&self) -> Result<(Option<T>, Option<T>)> {
        let (min, max) = self.minmax()?;
        Ok((scalar_to_optional_value(min)?, scalar_to_optional_value(max)?))
    }

    /// Inclusive cumulative sum using the borrowed execution override.
    pub fn cumsum(&self) -> Result<Series> {
        self.series.scan_in_context(
            ScanAggregation::Sum,
            ScanType::Inclusive,
            NullPolicy::Exclude,
            self.ctx,
            "series::in_context::cumsum",
        )
    }

    /// Inclusive cumulative minimum using the borrowed execution override.
    pub fn cummin(&self) -> Result<Series> {
        self.series.scan_in_context(
            ScanAggregation::Min,
            ScanType::Inclusive,
            NullPolicy::Exclude,
            self.ctx,
            "series::in_context::cummin",
        )
    }

    /// Inclusive cumulative maximum using the borrowed execution override.
    pub fn cummax(&self) -> Result<Series> {
        self.series.scan_in_context(
            ScanAggregation::Max,
            ScanType::Inclusive,
            NullPolicy::Exclude,
            self.ctx,
            "series::in_context::cummax",
        )
    }

    /// Inclusive cumulative product using the borrowed execution override.
    pub fn cumprod(&self) -> Result<Series> {
        self.series.scan_in_context(
            ScanAggregation::Product,
            ScanType::Inclusive,
            NullPolicy::Exclude,
            self.ctx,
            "series::in_context::cumprod",
        )
    }

    /// Element-wise NaN check using the borrowed execution override.
    pub fn is_nan(&self) -> Result<Series> {
        self.series.is_nan_in_context(self.ctx)
    }

    /// Element-wise not-NaN check using the borrowed execution override.
    pub fn is_not_nan(&self) -> Result<Series> {
        self.series.is_not_nan_in_context(self.ctx)
    }

    /// Element-wise absolute value using the borrowed execution override.
    pub fn abs(&self) -> Result<Series> {
        self.series.unary_in_context(UnaryOperator::Abs, self.ctx, "series::in_context::abs")
    }

    /// Element-wise arithmetic negation using the borrowed execution override.
    pub fn neg(&self) -> Result<Series> {
        self.series.unary_in_context(UnaryOperator::Negate, self.ctx, "series::in_context::neg")
    }

    /// Element-wise logical NOT using the borrowed execution override.
    pub fn logical_not(&self) -> Result<Series> {
        self.series.logical_not_in_context(self.ctx)
    }

    /// Return a BOOL8 mask: `true` where elements are null.
    pub fn is_null(&self) -> Result<Series> {
        self.series.is_null_in_context(self.ctx)
    }

    /// Return a BOOL8 mask: `true` where elements are valid.
    pub fn is_not_null(&self) -> Result<Series> {
        self.series.is_not_null_in_context(self.ctx)
    }

    /// Cast using the borrowed execution override.
    pub fn astype(&self, target: impl Into<DataType>) -> Result<Series> {
        self.series.cast_in_context(target, self.ctx, "series::in_context::astype")
    }

    /// Convert numeric values to timestamps using the borrowed execution override.
    pub fn to_datetime(&self, resolution: Option<TimestampResolution>) -> Result<Series> {
        self.series.to_datetime_in_context(resolution, self.ctx)
    }

    /// Cast to a numeric type using the borrowed execution override.
    pub fn to_numeric(&self, target: TypeId) -> Result<Series> {
        if !target.is_numeric() {
            return Err(Error::new(ErrorKind::InvalidOperation, format!("target type {target:?} is not numeric"))
                .with_operation("series::in_context::to_numeric"));
        }
        self.series.cast_in_context(target, self.ctx, "series::in_context::to_numeric")
    }

    /// Element-wise equality-to-scalar using the borrowed execution override.
    pub fn eq_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        self.series.binary_scalar_in_context(rhs, BinaryOperator::Equal, self.ctx, "series::in_context::eq_scalar")
    }

    /// Element-wise inequality-to-scalar using the borrowed execution override.
    pub fn ne_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        self.series.binary_scalar_in_context(rhs, BinaryOperator::NotEqual, self.ctx, "series::in_context::ne_scalar")
    }

    /// Element-wise greater-than-scalar using the borrowed execution override.
    pub fn gt_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        self.series.binary_scalar_in_context(rhs, BinaryOperator::Greater, self.ctx, "series::in_context::gt_scalar")
    }

    /// Element-wise less-than-scalar using the borrowed execution override.
    pub fn lt_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        self.series.binary_scalar_in_context(rhs, BinaryOperator::Less, self.ctx, "series::in_context::lt_scalar")
    }

    /// Element-wise greater-than-or-equal-to-scalar using the borrowed execution override.
    pub fn ge_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        self.series.binary_scalar_in_context(
            rhs,
            BinaryOperator::GreaterEqual,
            self.ctx,
            "series::in_context::ge_scalar",
        )
    }

    /// Element-wise less-than-or-equal-to-scalar using the borrowed execution override.
    pub fn le_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        self.series.binary_scalar_in_context(rhs, BinaryOperator::LessEqual, self.ctx, "series::in_context::le_scalar")
    }

    /// Element-wise equality using the borrowed execution override.
    pub fn eq(&self, rhs: &Series) -> Result<Series> {
        self.series.binary_series_in_context(rhs, BinaryOperator::Equal, self.ctx, "series::in_context::eq")
    }

    /// Element-wise inequality using the borrowed execution override.
    pub fn ne(&self, rhs: &Series) -> Result<Series> {
        self.series.binary_series_in_context(rhs, BinaryOperator::NotEqual, self.ctx, "series::in_context::ne")
    }

    /// Element-wise greater-than using the borrowed execution override.
    pub fn gt(&self, rhs: &Series) -> Result<Series> {
        self.series.binary_series_in_context(rhs, BinaryOperator::Greater, self.ctx, "series::in_context::gt")
    }

    /// Element-wise less-than using the borrowed execution override.
    pub fn lt(&self, rhs: &Series) -> Result<Series> {
        self.series.binary_series_in_context(rhs, BinaryOperator::Less, self.ctx, "series::in_context::lt")
    }

    /// Element-wise greater-than-or-equal using the borrowed execution override.
    pub fn ge(&self, rhs: &Series) -> Result<Series> {
        self.series.binary_series_in_context(rhs, BinaryOperator::GreaterEqual, self.ctx, "series::in_context::ge")
    }

    /// Element-wise less-than-or-equal using the borrowed execution override.
    pub fn le(&self, rhs: &Series) -> Result<Series> {
        self.series.binary_series_in_context(rhs, BinaryOperator::LessEqual, self.ctx, "series::in_context::le")
    }

    /// Element-wise addition using the borrowed execution override.
    pub fn add(&self, rhs: &Series) -> Result<Series> {
        self.series.binary_series_in_context(rhs, BinaryOperator::Add, self.ctx, "series::in_context::add")
    }

    /// Element-wise addition-by-scalar using the borrowed execution override.
    pub fn add_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        self.series.binary_scalar_in_context(rhs, BinaryOperator::Add, self.ctx, "series::in_context::add_scalar")
    }

    /// Element-wise subtraction using the borrowed execution override.
    pub fn sub(&self, rhs: &Series) -> Result<Series> {
        self.series.binary_series_in_context(rhs, BinaryOperator::Sub, self.ctx, "series::in_context::sub")
    }

    /// Element-wise subtraction-by-scalar using the borrowed execution override.
    pub fn sub_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        self.series.binary_scalar_in_context(rhs, BinaryOperator::Sub, self.ctx, "series::in_context::sub_scalar")
    }

    /// Element-wise multiplication using the borrowed execution override.
    pub fn mul(&self, rhs: &Series) -> Result<Series> {
        self.series.binary_series_in_context(rhs, BinaryOperator::Mul, self.ctx, "series::in_context::mul")
    }

    /// Element-wise multiplication-by-scalar using the borrowed execution override.
    pub fn mul_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        self.series.binary_scalar_in_context(rhs, BinaryOperator::Mul, self.ctx, "series::in_context::mul_scalar")
    }

    /// Element-wise division using the borrowed execution override.
    pub fn div(&self, rhs: &Series) -> Result<Series> {
        self.series.binary_series_in_context(rhs, BinaryOperator::Div, self.ctx, "series::in_context::div")
    }

    /// Element-wise division-by-scalar using the borrowed execution override.
    pub fn div_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        self.series.binary_scalar_in_context(rhs, BinaryOperator::Div, self.ctx, "series::in_context::div_scalar")
    }

    /// Element-wise modulo using the borrowed execution override.
    pub fn rem(&self, rhs: &Series) -> Result<Series> {
        self.series.binary_series_in_context(rhs, BinaryOperator::Mod, self.ctx, "series::in_context::rem")
    }

    /// Element-wise modulo-by-scalar using the borrowed execution override.
    pub fn rem_scalar<T: CudfType>(&self, rhs: T) -> Result<Series> {
        self.series.binary_scalar_in_context(rhs, BinaryOperator::Mod, self.ctx, "series::in_context::rem_scalar")
    }

    /// Element-wise membership test using the borrowed execution override.
    pub fn isin(&self, values: &Series) -> Result<Series> {
        self.series.isin_in_context(values, self.ctx)
    }

    /// Remove duplicates using the borrowed execution override.
    pub fn unique(&self) -> Result<Series> {
        self.series.unique_in_context(self.ctx)
    }

    /// Count distinct non-null values using the borrowed execution override.
    pub fn nunique(&self) -> Result<usize> {
        let unique = self.unique()?;
        Ok(unique.len() - unique.null_count())
    }

    /// Factorize using the borrowed execution override.
    pub fn factorize(&self) -> Result<(Series, Series)> {
        self.series.factorize_in_context(self.ctx)
    }

    /// Lowercase strings using the borrowed execution override.
    pub fn str_lower(&self) -> Result<Series> {
        self.series.string_unary_in_context(strings::to_lower_impl, self.ctx, "series::in_context::str_lower")
    }

    /// Uppercase strings using the borrowed execution override.
    pub fn str_upper(&self) -> Result<Series> {
        self.series.string_unary_in_context(strings::to_upper_impl, self.ctx, "series::in_context::str_upper")
    }

    /// Literal substring containment using the borrowed execution override.
    pub fn str_contains(&self, pat: &str) -> Result<Series> {
        self.series.string_scalar_in_context(
            pat,
            self.ctx,
            "series::in_context::str_contains",
            strings::str_contains_impl,
        )
    }

    /// Equality-to-string-scalar using the borrowed execution override.
    pub fn eq_scalar_str(&self, rhs: &str) -> Result<Series> {
        self.series.binary_string_scalar_in_context(
            rhs,
            BinaryOperator::Equal,
            self.ctx,
            "series::in_context::eq_scalar_str",
        )
    }

    /// Inequality-to-string-scalar using the borrowed execution override.
    pub fn ne_scalar_str(&self, rhs: &str) -> Result<Series> {
        self.series.binary_string_scalar_in_context(
            rhs,
            BinaryOperator::NotEqual,
            self.ctx,
            "series::in_context::ne_scalar_str",
        )
    }

    /// Greater-than-string-scalar using the borrowed execution override.
    pub fn gt_scalar_str(&self, rhs: &str) -> Result<Series> {
        self.series.binary_string_scalar_in_context(
            rhs,
            BinaryOperator::Greater,
            self.ctx,
            "series::in_context::gt_scalar_str",
        )
    }

    /// Less-than-string-scalar using the borrowed execution override.
    pub fn lt_scalar_str(&self, rhs: &str) -> Result<Series> {
        self.series.binary_string_scalar_in_context(
            rhs,
            BinaryOperator::Less,
            self.ctx,
            "series::in_context::lt_scalar_str",
        )
    }

    /// Greater-than-or-equal-to-string-scalar using the borrowed execution override.
    pub fn ge_scalar_str(&self, rhs: &str) -> Result<Series> {
        self.series.binary_string_scalar_in_context(
            rhs,
            BinaryOperator::GreaterEqual,
            self.ctx,
            "series::in_context::ge_scalar_str",
        )
    }

    /// Less-than-or-equal-to-string-scalar using the borrowed execution override.
    pub fn le_scalar_str(&self, rhs: &str) -> Result<Series> {
        self.series.binary_string_scalar_in_context(
            rhs,
            BinaryOperator::LessEqual,
            self.ctx,
            "series::in_context::le_scalar_str",
        )
    }

    /// Starts-with test using the borrowed execution override.
    pub fn str_startswith(&self, pat: &str) -> Result<Series> {
        self.series.string_scalar_in_context(
            pat,
            self.ctx,
            "series::in_context::str_startswith",
            strings::starts_with_impl,
        )
    }

    /// Ends-with test using the borrowed execution override.
    pub fn str_endswith(&self, pat: &str) -> Result<Series> {
        self.series.string_scalar_in_context(pat, self.ctx, "series::in_context::str_endswith", strings::ends_with_impl)
    }

    /// Strip whitespace using the borrowed execution override.
    pub fn str_strip(&self) -> Result<Series> {
        self.series.strip_in_context(StripSide::Both, "", self.ctx, "series::in_context::str_strip")
    }

    /// Strip leading whitespace using the borrowed execution override.
    pub fn str_lstrip(&self) -> Result<Series> {
        self.series.strip_in_context(StripSide::Left, "", self.ctx, "series::in_context::str_lstrip")
    }

    /// Strip trailing whitespace using the borrowed execution override.
    pub fn str_rstrip(&self) -> Result<Series> {
        self.series.strip_in_context(StripSide::Right, "", self.ctx, "series::in_context::str_rstrip")
    }

    /// Strip specific characters using the borrowed execution override.
    pub fn str_strip_chars(&self, side: StripSide, to_strip: &str) -> Result<Series> {
        self.series.strip_in_context(side, to_strip, self.ctx, "series::in_context::str_strip_chars")
    }

    /// Character counts using the borrowed execution override.
    pub fn str_len(&self) -> Result<Series> {
        self.series.string_unary_in_context(strings::count_characters_impl, self.ctx, "series::in_context::str_len")
    }

    /// Byte counts using the borrowed execution override.
    pub fn str_byte_count(&self) -> Result<Series> {
        self.series.string_unary_in_context(strings::count_bytes_impl, self.ctx, "series::in_context::str_byte_count")
    }
}
