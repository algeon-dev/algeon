use std::sync::Arc;

use super::{
    owned::{SERIES_VALUE_COUNTS_COUNT_NAME, SERIES_VALUE_COUNTS_FALLBACK_NAME, Series},
    reduction::infer_reduce_output_type,
};
#[cfg(any(test, feature = "test-hooks"))]
use crate::runtime::Runtime;
use crate::{
    core::{
        column::{Column, ColumnView},
        error::{Error, ErrorCode, FACT_LEFT_TYPE_ID, FACT_RIGHT_TYPE_ID, Result},
        row_range,
        scalar::{Scalar, ScalarValue},
        table::TableView,
        types::{CudfType, DataType, NullPolicy, TimestampResolution, TypeId},
    },
    frame::{
        dataframe::DataFrame,
        factorization::{Codebook, DictionaryEncodingOptions, DictionaryKeyOrder, EncodedSeries, Factorized},
    },
    ops::{
        binary::{self, BinaryOperator},
        concatenate,
        copying::gather_impl,
        datetime::{self, DatetimeComponent},
        factorization,
        filter::{apply_boolean_mask_impl, validate_boolean_mask},
        groupby::{
            AggregationRequest, GroupByAgg, GroupByAggregationOutput, groupby_aggregate_impl,
            value_counts_result_columns,
        },
        reduction::{self, ReduceAgg, ScanAgg, ScanType},
        replace::{self, FillPolicy},
        search::contains_impl,
        sorting::{SortExecution, SortKeyConfig, SortOptions, sort_table_impl},
        stream_compaction::{self, DuplicateKeepOption, NanEquality, NullEquality},
        strings::{self, StringSliceOptions, StripSide},
        unary::{self, UnaryOperator},
    },
    runtime::{
        Exec,
        execution_domain::{ConsumerCompletionFence, ExecutionDomain},
        execution_resolver::ResolvedExecution,
    },
};

type StringScalarColumnOp =
    fn(&ColumnView<'_>, &Scalar, &Arc<ExecutionDomain>, &cudf_sys::cudf_exec_options_t, &'static str) -> Result<Column>;

fn factorization_key_order(order: DictionaryKeyOrder) -> factorization::KeyOrder {
    match order {
        DictionaryKeyOrder::Sorted => factorization::KeyOrder::Sorted,
        DictionaryKeyOrder::FirstSeen => factorization::KeyOrder::FirstSeen,
    }
}

/// Generate a simple context helper that resolves the output domain, gets a
/// column view, and wraps the result as a `Series`.
macro_rules! series_context_op {
    (unary fn $method:ident($($param:ident : $pty:ty),*), $operation:literal,
     |$cv:ident, $domain:ident, $exec:ident, $op:ident| $call:expr) => {
        fn $method(&self, $($param: $pty,)* ctx: Exec<'_>) -> Result<Series> {
            let $op = $operation;
            let ($domain, _consumer_fence) = self.resolve_context_output_domain(ctx, $op)?;
            let $cv = self.column.view()?;
            let $exec = $domain.enter_execution()?;
            let col = $call?;
            Series::from_column(col)
        }
    };
    (unary fn $method:ident($($param:ident : $pty:ty),*),
     caller_op |$cv:ident, $domain:ident, $exec:ident, $op:ident| $call:expr) => {
        fn $method(&self, $($param: $pty,)* ctx: Exec<'_>, $op: &'static str) -> Result<Series> {
            let ($domain, _consumer_fence) = self.resolve_context_output_domain(ctx, $op)?;
            let $cv = self.column.view()?;
            let $exec = $domain.enter_execution()?;
            let col = $call?;
            Series::from_column(col)
        }
    };
    (binary fn $method:ident($other:ident : &Series),
     $operation:literal, |$cv:ident, $other_cv:ident, $domain:ident, $exec:ident, $op:ident| $call:expr) => {
        fn $method(
            &self,
            $other: &Series,
            ctx: Exec<'_>,
        ) -> Result<Series> {
            let $op = $operation;
            let ($domain, mut consumer_fence) = self.resolve_context_output_domain(ctx, $op)?;
            ExecutionDomain::require_same_device_for_mixed_inputs(
                self.column.execution_domain(),
                $other.column.execution_domain(),
                $op,
            )?;
            consumer_fence.order_input($other.column.execution_domain())?;
            let $cv = self.column.view()?;
            let $other_cv = $other.column.view()?;
            let $exec = $domain.enter_execution()?;
            let col = $call?;
            Series::from_column(col)
        }
    };
}

fn value_counts_from_groupby_output(gb: GroupByAggregationOutput, operation: &'static str) -> Result<DataFrame> {
    let (key_col, count_col) = value_counts_result_columns(gb, operation)?;
    DataFrame::from_series([
        (SERIES_VALUE_COUNTS_FALLBACK_NAME.to_string(), Series::from_column(key_col)?),
        (SERIES_VALUE_COUNTS_COUNT_NAME.to_string(), Series::from_column(count_col)?),
    ])
    .map_err(|e| e.reanchor(operation))
}

#[cfg(any(test, feature = "test-hooks"))]
fn range_testing_series() -> Result<Series> {
    let runtime = Runtime::new()?;
    runtime.series(&[1i32, 2, 3])
}

#[cfg(any(test, feature = "test-hooks"))]
pub(crate) fn trigger_series_row_range_start_limit_error_for_test() -> Result<Series> {
    range_testing_series()?.__debug_gather_range_for_test(usize::MAX, 1, "series::slice")
}

#[cfg(any(test, feature = "test-hooks"))]
pub(crate) fn trigger_series_row_range_end_limit_error_for_test() -> Result<Series> {
    range_testing_series()?.__debug_gather_range_for_test(i32::MAX as usize, 1, "series::slice")
}

#[cfg(any(test, feature = "test-hooks"))]
pub(crate) fn trigger_series_row_range_overflow_error_for_test() -> Result<Series> {
    range_testing_series()?.__debug_gather_range_for_test(i32::MAX as usize, usize::MAX, "series::slice")
}

/// The GPU operations of one `Series`, run where an [`Exec`] selects.
#[must_use]
pub struct SeriesOps<'a> {
    pub(super) series: &'a Series,
    pub(super) ctx: Exec<'a>,
}

impl Series {
    /// The GPU operations of this series, run where `exec` selects.
    pub fn on<'a>(&'a self, exec: Exec<'a>) -> SeriesOps<'a> {
        SeriesOps { series: self, ctx: exec }
    }
}

impl Series {
    /// Resolve the output domain for a context op and order this series'
    /// retained stream into it.
    ///
    /// The returned [`ConsumerCompletionFence`] must be held across the FFI
    /// call that consumes this series; additional inputs are ordered through
    /// [`ConsumerCompletionFence::order_input`] so one fence covers them all.
    fn resolve_context_output_domain(
        &self,
        ctx: Exec<'_>,
        operation: &'static str,
    ) -> Result<(Arc<ExecutionDomain>, ConsumerCompletionFence)> {
        let output_domain = ctx.resolve_output_domain(self.column.execution_domain())?;
        let consumer_fence =
            ExecutionDomain::insert_stream_dependency(self.column.execution_domain(), &output_domain, operation)?;
        Ok((output_domain, consumer_fence))
    }

    // These helpers are hand-written because their control flow is unique.

    /// Concatenate `series` in order, running where `exec` selects; the
    /// output provenance resolves from the first series.
    pub fn concat(series: &[&Series], exec: Exec<'_>) -> Result<Series> {
        let operation = "series::concat";
        let ctx = exec;
        let Some(first) = series.first() else {
            return Err(Error::new_with(
                ErrorCode::UnsupportedOperation,
                operation,
                "cannot concatenate empty slice of Series",
            )
            .with_detail("reason", "empty_input")
            .with_detail("remedy", "provide at least one input series"));
        };
        let output_domain = ctx.resolve_output_domain(first.column.execution_domain())?;
        let execution = ResolvedExecution::with_output_domain(
            output_domain,
            series.iter().map(|series| series.column.execution_domain()),
            operation,
        )?;
        let views = series
            .iter()
            .map(|series| series.column.view().map_err(|error| error.reanchor(operation)))
            .collect::<Result<Vec<_>>>()?;
        let view_refs = views.iter().collect::<Vec<&ColumnView<'_>>>();
        let col =
            concatenate::concatenate_columns_with_exec(&view_refs, execution.output_domain(), execution.exec_options())
                .map_err(|error| error.reanchor(operation))?;
        Series::from_column(col)
    }

    pub(super) fn sort_values_with_context(
        &self,
        options: SortOptions,
        ctx: Exec<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let (orders, null_orders) = options.resolve(1, operation)?;
        let cv = self.column.view()?;
        let ttv = TableView::from_borrowed_column_views_in_domain(vec![cv], None, operation)?;
        let exec_options = output_domain.enter_execution()?;
        let sorted = sort_table_impl(
            &ttv,
            SortKeyConfig::new(Some(&orders), null_orders.as_deref()),
            options.stable,
            SortExecution::new(&output_domain, exec_options.options(), operation),
        )?;
        let mut cols = sorted.into_columns()?;
        Series::from_column(cols.remove(0))
    }

    pub(super) fn to_datetime_with_context(
        &self,
        resolution: Option<TimestampResolution>,
        ctx: Exec<'_>,
        duration_operation: &'static str,
        operation: &'static str,
    ) -> Result<Series> {
        let resolution = resolution.unwrap_or(TimestampResolution::Microseconds);
        let source_type = self.dtype()?.id();
        if source_type.is_numeric() || source_type.is_boolean() {
            let duration = self.cast_in_context(resolution.duration_type_id(), ctx, duration_operation)?;
            duration.cast_in_context(resolution.type_id(), ctx, operation)
        } else {
            self.cast_in_context(resolution.type_id(), ctx, operation)
        }
    }

    pub(crate) fn build_codebook_in_domain(
        &self,
        output_domain: Arc<ExecutionDomain>,
        options: DictionaryEncodingOptions,
        operation: &'static str,
    ) -> Result<Codebook> {
        let cv = self.column.view()?;
        let exec_options = output_domain.enter_execution()?;
        let keys = factorization::keys_impl(
            &cv,
            TypeId::Int32,
            factorization_key_order(options.key_order_mode()),
            &output_domain,
            exec_options.options(),
            operation,
        )?;
        let keys = Series::from_column(keys)?;
        Codebook::from_keys(keys)
    }

    pub(super) fn codebook_with_context(&self, ctx: Exec<'_>, operation: &'static str) -> Result<Codebook> {
        self.codebook_with_options_in_context(ctx, DictionaryEncodingOptions::new(), operation)
    }

    pub(super) fn codebook_with_options_in_context(
        &self,
        ctx: Exec<'_>,
        options: DictionaryEncodingOptions,
        operation: &'static str,
    ) -> Result<Codebook> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        self.build_codebook_in_domain(output_domain, options, operation)
    }

    pub(super) fn factorize_with_context(&self, ctx: Exec<'_>, operation: &'static str) -> Result<Factorized> {
        self.factorize_with_options_in_context(ctx, DictionaryEncodingOptions::new(), operation)
    }

    pub(super) fn factorize_with_options_in_context(
        &self,
        ctx: Exec<'_>,
        options: DictionaryEncodingOptions,
        operation: &'static str,
    ) -> Result<Factorized> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let exec_options = output_domain.enter_execution()?;
        let parts = factorization::factorize_impl(
            &cv,
            TypeId::Int32,
            factorization_key_order(options.key_order_mode()),
            &output_domain,
            exec_options.options(),
            operation,
        )?;
        let row_codes = Series::from_column(parts.row_codes)?;
        let keys = Series::from_column(parts.keys)?;
        let row_codes = EncodedSeries::new(row_codes);
        let codebook = Codebook::from_keys(keys)?;
        Ok(Factorized::new(row_codes, codebook))
    }

    pub(super) fn fillna_policy_with_context(
        &self,
        policy: FillPolicy,
        ctx: Exec<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let exec_options = output_domain.enter_execution()?;
        let col = replace::replace_nulls_policy_with_exec_options(
            &cv,
            policy,
            &output_domain,
            exec_options.options(),
            operation,
        )?;
        Series::from_column(col)
    }

    pub(super) fn unique_with_context(&self, ctx: Exec<'_>, operation: &'static str) -> Result<Series> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let ttv = TableView::from_borrowed_column_views_in_domain(vec![cv], None, operation)?;
        let exec_options = output_domain.enter_execution()?;
        let result = stream_compaction::distinct_impl(
            &ttv,
            stream_compaction::DistinctConfig::new(
                &[0],
                DuplicateKeepOption::KeepAny,
                NullEquality::Equal,
                NanEquality::AllEqual,
            ),
            stream_compaction::StreamCompactionExecution::new(&output_domain, exec_options.options(), operation),
        )?;
        let mut cols = result.into_columns()?;
        Series::from_column(cols.remove(0))
    }

    pub(super) fn head_with_context(&self, n: usize, ctx: Exec<'_>, operation: &'static str) -> Result<Series> {
        let total = self.len();
        let take = n.min(total);
        self.gather_range_in_context_inner(0, take, ctx, operation)
    }

    pub(super) fn tail_with_context(&self, n: usize, ctx: Exec<'_>, operation: &'static str) -> Result<Series> {
        let total = self.len();
        let take = n.min(total);
        let start = total - take;
        self.gather_range_in_context_inner(start, take, ctx, operation)
    }

    pub(super) fn slice_with_context(
        &self,
        offset: usize,
        length: usize,
        ctx: Exec<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        let total = self.len();
        let start = offset.min(total);
        let take = length.min(total - start);
        self.gather_range_in_context_inner(start, take, ctx, operation)
    }

    fn gather_range_in_context_inner(
        &self,
        start: usize,
        len: usize,
        ctx: Exec<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let map = row_range::gather_map_in_domain(start, len, &output_domain, operation)?;
        let map_view = map.view()?;

        let cv = self.column.view().map_err(|e| e.reanchor(operation))?;
        let ttv = TableView::from_borrowed_column_views_in_domain(vec![cv], None, operation)?;
        let exec_options = output_domain.enter_execution()?;
        let result = gather_impl(
            &ttv,
            &map_view,
            cudf_sys::CUDF_OUT_OF_BOUNDS_POLICY_DONT_CHECK,
            "dont_check",
            &output_domain,
            exec_options.options(),
            operation,
        )?;
        let mut cols = result.into_columns()?;
        Series::from_column(cols.remove(0))
    }

    pub(super) fn gather_with_context(
        &self,
        indices: &Series,
        ctx: Exec<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        let (output_domain, mut consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let idx_cv = indices.view().map_err(|e| e.reanchor(operation))?;
        ExecutionDomain::require_same_device_for_mixed_inputs(
            self.column.execution_domain(),
            idx_cv.execution_domain(),
            operation,
        )?;
        consumer_fence.order_input(idx_cv.execution_domain())?;

        let cv = self.column.view().map_err(|e| e.reanchor(operation))?;
        let ttv = TableView::from_borrowed_column_views_in_domain(vec![cv], None, operation)?;
        let exec_options = output_domain.enter_execution()?;
        let result = gather_impl(
            &ttv,
            &idx_cv,
            cudf_sys::CUDF_OUT_OF_BOUNDS_POLICY_NULLIFY,
            "nullify",
            &output_domain,
            exec_options.options(),
            operation,
        )?;
        let mut cols = result.into_columns()?;
        Series::from_column(cols.remove(0))
    }

    pub(super) fn filter_with_context(&self, mask: &Series, ctx: Exec<'_>, operation: &'static str) -> Result<Series> {
        let (output_domain, mut consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let mask_cv = mask.view().map_err(|e| e.reanchor(operation))?;
        validate_boolean_mask(&mask_cv, self.len(), "series length", operation)?;

        ExecutionDomain::require_same_device_for_mixed_inputs(
            self.column.execution_domain(),
            mask_cv.execution_domain(),
            operation,
        )?;
        consumer_fence.order_input(mask_cv.execution_domain())?;

        let cv = self.column.view().map_err(|e| e.reanchor(operation))?;
        let ttv = TableView::from_borrowed_column_views_in_domain(vec![cv], None, operation)?;
        let exec_options = output_domain.enter_execution()?;
        let result = apply_boolean_mask_impl(&ttv, &mask_cv, &output_domain, exec_options.options(), operation)?;
        let mut cols = result.into_columns()?;
        Series::from_column(cols.remove(0))
    }

    pub(super) fn dropna_with_context(&self, ctx: Exec<'_>, operation: &'static str) -> Result<Series> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view().map_err(|e| e.reanchor(operation))?;
        let ttv = TableView::from_borrowed_column_views_in_domain(vec![cv], None, operation)?;
        let exec_options = output_domain.enter_execution()?;
        let result =
            stream_compaction::drop_nulls_impl(&ttv, &[0], 1, &output_domain, exec_options.options(), operation)?;
        let mut cols = result.into_columns()?;
        Series::from_column(cols.remove(0))
    }

    pub(super) fn drop_duplicates_with_context(&self, ctx: Exec<'_>, operation: &'static str) -> Result<Series> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view().map_err(|e| e.reanchor(operation))?;
        let ttv = TableView::from_borrowed_column_views_in_domain(vec![cv], None, operation)?;
        let exec_options = output_domain.enter_execution()?;
        let result = stream_compaction::distinct_impl(
            &ttv,
            stream_compaction::DistinctConfig::new(
                &[0],
                DuplicateKeepOption::KeepAny,
                NullEquality::Equal,
                NanEquality::AllEqual,
            ),
            stream_compaction::StreamCompactionExecution::new(&output_domain, exec_options.options(), operation),
        )?;
        let mut cols = result.into_columns()?;
        Series::from_column(cols.remove(0))
    }

    pub(super) fn value_counts_with_context(&self, ctx: Exec<'_>, operation: &'static str) -> Result<DataFrame> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;

        let keys_view = self.column.view().map_err(|e| e.reanchor(operation))?;
        let values_view = self.column.view().map_err(|e| e.reanchor(operation))?;
        let ttv = TableView::from_borrowed_column_views_in_domain(vec![keys_view], None, operation)?;
        let aggs = [GroupByAgg::Count { include_nulls: false }];
        let requests = [AggregationRequest { values: &values_view, aggregations: &aggs }];
        let exec_options = output_domain.enter_execution()?;
        let gb = groupby_aggregate_impl(
            &ttv,
            &requests,
            NullPolicy::Exclude,
            false,
            &output_domain,
            exec_options.options(),
            operation,
        )?;
        value_counts_from_groupby_output(gb, operation)
    }

    pub(super) fn reduce_in_context(&self, agg: ReduceAgg, ctx: Exec<'_>, operation: &'static str) -> Result<Scalar> {
        let input_type = self.dtype().map_err(|e| e.reanchor(operation))?;
        let output_type = infer_reduce_output_type(&input_type, &agg, operation)?;
        self.reduce_prepared_in_context(&agg, &output_type, ctx, operation, |error| error)
    }

    pub(super) fn reduce_prepared_in_context<F>(
        &self,
        agg: &ReduceAgg,
        output_type: &DataType,
        ctx: Exec<'_>,
        operation: &'static str,
        map_reduction_error: F,
    ) -> Result<Scalar>
    where
        F: FnOnce(Error) -> Error,
    {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let exec_options = output_domain.enter_execution()?;
        reduction::reduce_impl_with_status_error_mapper(
            &cv,
            agg,
            output_type,
            &output_domain,
            exec_options.options(),
            operation,
            map_reduction_error,
        )
    }

    pub(super) fn minmax_with_context(&self, ctx: Exec<'_>, operation: &'static str) -> Result<(Scalar, Scalar)> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let exec_options = output_domain.enter_execution()?;
        reduction::minmax_impl(&cv, &output_domain, exec_options.options(), operation)
    }

    pub(super) fn is_nan_with_context(&self, ctx: Exec<'_>, operation: &'static str) -> Result<Series> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let exec_options = output_domain.enter_execution()?;
        let col = unary::is_nan_impl(&cv, &output_domain, exec_options.options(), operation)?;
        Series::from_column(col)
    }

    pub(super) fn is_not_nan_with_context(&self, ctx: Exec<'_>, operation: &'static str) -> Result<Series> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let exec_options = output_domain.enter_execution()?;
        let col = unary::is_not_nan_impl(&cv, &output_domain, exec_options.options(), operation)?;
        Series::from_column(col)
    }

    pub(super) fn logical_not_with_context(&self, ctx: Exec<'_>, operation: &'static str) -> Result<Series> {
        self.unary_in_context(UnaryOperator::Not, ctx, operation)
    }

    pub(super) fn is_null_with_context(&self, ctx: Exec<'_>, operation: &'static str) -> Result<Series> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let exec_options = output_domain.enter_execution()?;
        let col = unary::is_null_impl(&cv, &output_domain, exec_options.options(), operation)?;
        Series::from_column(col)
    }

    pub(super) fn is_not_null_with_context(&self, ctx: Exec<'_>, operation: &'static str) -> Result<Series> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let exec_options = output_domain.enter_execution()?;
        let col = unary::is_valid_impl(&cv, &output_domain, exec_options.options(), operation)?;
        Series::from_column(col)
    }

    series_context_op!(unary fn unary_in_context(op: UnaryOperator),
    caller_op |cv, output_domain, exec, operation| unary::unary_operation_impl(
        &cv,
        op,
        &output_domain,
        exec.options(),
        operation
    ));

    pub(super) fn unary_with_context(
        &self,
        op: UnaryOperator,
        ctx: Exec<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        self.unary_in_context(op, ctx, operation)
    }

    fn scan_in_context(
        &self,
        agg: ScanAgg,
        scan_type: ScanType,
        null_policy: NullPolicy,
        ctx: Exec<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let exec_options = output_domain.enter_execution()?;
        let col =
            reduction::scan_impl(&cv, agg, scan_type, null_policy, &output_domain, exec_options.options(), operation)?;
        Series::from_column(col)
    }

    pub(super) fn scan_with_context(
        &self,
        agg: ScanAgg,
        scan_type: ScanType,
        null_policy: NullPolicy,
        ctx: Exec<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        self.scan_in_context(agg, scan_type, null_policy, ctx, operation)
    }

    series_context_op!(unary fn string_unary_in_context(
    op: fn(
        &ColumnView<'_>,
        &Arc<ExecutionDomain>,
        &cudf_sys::cudf_exec_options_t,
        &'static str,
    ) -> Result<Column>),
    caller_op |cv, output_domain, exec, operation| op(
        &cv,
        &output_domain,
        exec.options(),
        operation
    ));

    pub(super) fn string_unary_with_context(
        &self,
        op: fn(&ColumnView<'_>, &Arc<ExecutionDomain>, &cudf_sys::cudf_exec_options_t, &'static str) -> Result<Column>,
        ctx: Exec<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        self.string_unary_in_context(op, ctx, operation)
    }

    pub(super) fn replace_with_backrefs_output_size_in_context(
        &self,
        pattern: &str,
        replacement: &str,
        ctx: Exec<'_>,
        operation: &'static str,
    ) -> Result<strings::ReplaceWithBackrefsOutputSizePreflight> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let exec_options = output_domain.enter_execution()?;
        strings::replace_with_backrefs_output_size_impl(&cv, pattern, replacement, exec_options.options(), operation)
    }

    pub(super) fn datetime_component_with_context(
        &self,
        component: DatetimeComponent,
        ctx: Exec<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let exec_options = output_domain.enter_execution()?;
        let col = datetime::extract_component_impl(&cv, component, &output_domain, exec_options.options(), operation)?;
        Series::from_column(col)
    }

    pub(super) fn strip_in_context(
        &self,
        side: StripSide,
        to_strip: &str,
        ctx: Exec<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let scalar = Scalar::from_str_in_domain(to_strip, &output_domain)?;
        let exec_options = output_domain.enter_execution()?;
        let col = strings::strip_impl(&cv, side, &scalar, &output_domain, exec_options.options(), operation)?;
        Series::from_column(col)
    }

    pub(super) fn string_scalar_in_context(
        &self,
        target: &str,
        ctx: Exec<'_>,
        operation: &'static str,
        op: StringScalarColumnOp,
    ) -> Result<Series> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let scalar = Scalar::from_str_in_domain(target, &output_domain)?;
        let exec_options = output_domain.enter_execution()?;
        let col = op(&cv, &scalar, &output_domain, exec_options.options(), operation)?;
        Series::from_column(col)
    }

    pub(super) fn like_in_context(
        &self,
        pattern: &str,
        escape: Option<&str>,
        ctx: Exec<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let exec_options = output_domain.enter_execution()?;
        let col = strings::like_impl(&cv, pattern, escape, &output_domain, exec_options.options(), operation)?;
        Series::from_column(col)
    }

    pub(super) fn string_slice_in_context(
        &self,
        options: StringSliceOptions,
        ctx: Exec<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let exec_options = output_domain.enter_execution()?;
        let col = strings::slice_impl(&cv, options, &output_domain, exec_options.options(), operation)?;
        Series::from_column(col)
    }

    pub(super) fn try_to_integers_in_context(
        &self,
        output_type: &DataType,
        ctx: Exec<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let exec_options = output_domain.enter_execution()?;
        let col = strings::try_to_integers_impl(&cv, output_type, &output_domain, exec_options.options(), operation)?;
        Series::from_column(col)
    }

    // Cast is hand-written because its `impl Into<DataType>` target needs a local conversion.

    pub(super) fn cast_in_context(
        &self,
        target: impl Into<DataType>,
        ctx: Exec<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let target = target.into();
        let cv = self.column.view()?;
        let exec_options = output_domain.enter_execution()?;
        let col = unary::cast_impl(&cv, &target, &output_domain, exec_options.options(), operation)?;
        Series::from_column(col)
    }

    pub(super) fn fillna_with_context<T: CudfType>(
        &self,
        value: T,
        ctx: Exec<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let scalar = Scalar::from_value_in_domain(value, &output_domain)?;
        let exec_options = output_domain.enter_execution()?;
        let col =
            replace::replace_nulls_with_exec_options(&cv, &scalar, &output_domain, exec_options.options(), operation)?;
        Series::from_column(col)
    }

    pub(super) fn fillna_with_series_with_context(
        &self,
        other: &Series,
        ctx: Exec<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        let (output_domain, mut consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        ExecutionDomain::require_same_device_for_mixed_inputs(
            self.column.execution_domain(),
            other.column.execution_domain(),
            operation,
        )?;
        consumer_fence.order_input(other.column.execution_domain())?;
        let cv = self.column.view()?;
        let other_cv = other.column.view()?;
        let exec_options = output_domain.enter_execution()?;
        let col = replace::replace_nulls_column_with_exec_options(
            &cv,
            &other_cv,
            &output_domain,
            exec_options.options(),
            operation,
        )?;
        Series::from_column(col)
    }

    pub(super) fn isin_with_context(&self, values: &Series, ctx: Exec<'_>, operation: &'static str) -> Result<Series> {
        let (output_domain, mut consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let values_cv = values.column.view()?;
        let input_type = cv.data_type()?.id();
        let values_type = values_cv.data_type()?.id();
        let annotate_type_context = |e: Error| {
            e.with_detail("input_type_id", format!("{:?}", input_type).to_lowercase())
                .with_detail("values_type_id", format!("{:?}", values_type).to_lowercase())
        };

        if input_type != values_type {
            return Err(Error::new_with(
                ErrorCode::TypeMismatch,
                operation,
                "isin requires both inputs to have the same dtype",
            )
            .with_fact(FACT_LEFT_TYPE_ID, input_type)
            .with_fact(FACT_RIGHT_TYPE_ID, values_type));
        }

        ExecutionDomain::require_same_device_for_mixed_inputs(
            self.column.execution_domain(),
            values.column.execution_domain(),
            operation,
        )
        .map_err(annotate_type_context)?;
        consumer_fence.order_input(values.column.execution_domain()).map_err(annotate_type_context)?;
        let exec_options = output_domain.enter_execution()?;
        let col = contains_impl(&values_cv, &cv, &output_domain, exec_options.options(), operation)
            .map_err(annotate_type_context)?;
        Series::from_column(col)
    }

    // Hand-written because the output type depends on both inputs.
    pub(super) fn binary_series_in_context(
        &self,
        other: &Series,
        op: BinaryOperator,
        ctx: Exec<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        let (output_domain, mut consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        ExecutionDomain::require_same_device_for_mixed_inputs(
            self.column.execution_domain(),
            other.column.execution_domain(),
            operation,
        )?;
        consumer_fence.order_input(other.column.execution_domain())?;
        let lhs = self.column.view()?;
        let rhs = other.column.view()?;
        let output_type = binary::infer_binary_output_dtype_in(lhs.data_type()?, rhs.data_type()?, op, operation)?;
        let exec_options = output_domain.enter_execution()?;
        let col = binary::binary_operation_vv_impl(
            &lhs,
            &rhs,
            op,
            &output_type,
            &output_domain,
            exec_options.options(),
            operation,
        )?;
        Series::from_column(col)
    }

    pub(super) fn binary_scalar_in_context(
        &self,
        value: impl Into<ScalarValue>,
        op: BinaryOperator,
        ctx: Exec<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let scalar = Scalar::from_scalar_value_in_domain(&value.into(), &output_domain)?;
        let output_type = binary::infer_binary_output_dtype_in(cv.data_type()?, scalar.data_type()?, op, operation)?;
        let exec_options = output_domain.enter_execution()?;
        let col = binary::binary_operation_vs_impl(
            &cv,
            &scalar,
            op,
            &output_type,
            &output_domain,
            exec_options.options(),
            operation,
        )?;
        Series::from_column(col)
    }

    #[cfg(any(test, feature = "test-hooks"))]
    pub(crate) fn __debug_gather_range_for_test(
        &self,
        start: usize,
        len: usize,
        operation: &'static str,
    ) -> Result<Series> {
        self.gather_range_in_context_inner(start, len, Exec::RETAINED, operation)
    }
}
