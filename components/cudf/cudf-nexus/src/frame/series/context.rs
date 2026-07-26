use std::sync::Arc;

use super::{
    owned::{
        SERIES_VALUE_COUNTS_COUNT_NAME, SERIES_VALUE_COUNTS_FALLBACK_NAME, Series, unsupported_numeric_target_error,
    },
    reduction::{infer_reduce_output_type, scalar_to_optional_value},
};
use crate::{
    core::{
        column::ColumnView,
        error::{Error, ErrorCode, FACT_LEFT_TYPE_ID, FACT_RIGHT_TYPE_ID, Result},
        row_range,
        scalar::{IntoRuntimeScalar, Scalar},
        types::{CudfType, DataType, NullPolicy, TimestampResolution, TypeId},
    },
    frame::factorization::{Codebook, DictionaryEncodingOptions, DictionaryKeyOrder, EncodedSeries, Factorized},
    internal::temp_table::TempTableView,
    ops::{
        binary::{self, BinaryOperator},
        concatenate,
        datetime::{self, DatetimeComponent},
        factorization,
        groupby::GroupByAgg,
        reduction::{self, ReduceAgg, ScanAgg, ScanType},
        replace::{self, FillPolicy},
        sorting::SortOptions,
        stream_compaction::{self, DuplicateKeepOption, NanEquality, NullEquality},
        strings::{self, StringSliceOptions, StripSide},
        unary::{self, UnaryOperator},
    },
    runtime::{
        ExecutionOverrides, Runtime,
        execution_domain::{ConsumerCompletionFence, ExecutionDomain},
        execution_resolver::ResolvedExecution,
    },
};

type StringScalarColumnOp = fn(
    &crate::core::column::ColumnView<'_>,
    &Scalar,
    &Arc<ExecutionDomain>,
    &cudf_sys::cudf_exec_options_t,
    &'static str,
) -> Result<crate::core::column::Column>;

fn factorization_key_order(order: DictionaryKeyOrder) -> factorization::KeyOrder {
    match order {
        DictionaryKeyOrder::Sorted => factorization::KeyOrder::Sorted,
        DictionaryKeyOrder::FirstSeen => factorization::KeyOrder::FirstSeen,
    }
}

/// Generate a simple context helper that resolves the output domain, gets a
/// column view, and wraps the result as a `Series`.
macro_rules! series_context_op {
    // Unary with fixed operation string
    (unary fn $method:ident($($param:ident : $pty:ty),*), $operation:literal,
     |$cv:ident, $domain:ident, $exec:ident, $op:ident| $call:expr) => {
        fn $method(&self, $($param: $pty,)* ctx: &ExecutionOverrides<'_>) -> Result<Series> {
            let $op = $operation;
            let ($domain, _consumer_fence) = self.resolve_context_output_domain(ctx, $op)?;
            let $cv = self.column.view()?;
            let $exec = $domain.enter_execution()?;
            let col = $call?;
            Series::from_column(col)
        }
    };
    // Unary with caller-supplied operation string
    (unary fn $method:ident($($param:ident : $pty:ty),*),
     caller_op |$cv:ident, $domain:ident, $exec:ident, $op:ident| $call:expr) => {
        fn $method(&self, $($param: $pty,)* ctx: &ExecutionOverrides<'_>, $op: &'static str) -> Result<Series> {
            let ($domain, _consumer_fence) = self.resolve_context_output_domain(ctx, $op)?;
            let $cv = self.column.view()?;
            let $exec = $domain.enter_execution()?;
            let col = $call?;
            Series::from_column(col)
        }
    };
    // Binary (two-Series): adds require_same_device + fence for the other Series
    (binary fn $method:ident($other:ident : &Series),
     $operation:literal, |$cv:ident, $other_cv:ident, $domain:ident, $exec:ident, $op:ident| $call:expr) => {
        fn $method(
            &self,
            $other: &Series,
            ctx: &ExecutionOverrides<'_>,
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

fn value_counts_from_groupby_output(
    gb: crate::ops::groupby::GroupByAggregationOutput,
    operation: &'static str,
) -> Result<crate::DataFrame> {
    let (key_col, count_col) = crate::ops::groupby::value_counts_result_columns(gb, operation)?;
    crate::DataFrame::from_series([
        (SERIES_VALUE_COUNTS_FALLBACK_NAME.to_string(), Series::from_column(key_col)?),
        (SERIES_VALUE_COUNTS_COUNT_NAME.to_string(), Series::from_column(count_col)?),
    ])
    .map_err(|e| e.reanchor(operation))
}

macro_rules! series_forward_in {
    ($(#[$meta:meta])* fn $method:ident($($param:ident : $pty:ty),*) -> $ret:ty => $target:ident;) => {
        $(#[$meta])*
        pub fn $method(&self, $($param: $pty,)* ctx: &ExecutionOverrides<'_>) -> $ret {
            self.$target($($param,)* ctx)
        }
    };
    ($( $(#[$meta:meta])* fn $method:ident($($param:ident : $pty:ty),*) -> $ret:ty => $target:ident; )*) => {
        $(
            series_forward_in!(
                $(#[$meta])*
                fn $method($($param : $pty),*) -> $ret => $target;
            );
        )*
    };
}

macro_rules! series_forward_in_generic {
    ($(#[$meta:meta])* fn $method:ident<$gen:ident : $bound:path>($($param:ident : $pty:ty),*) -> $ret:ty => $target:ident;) => {
        $(#[$meta])*
        pub fn $method<$gen: $bound>(&self, $($param: $pty,)* ctx: &ExecutionOverrides<'_>) -> $ret {
            self.$target($($param,)* ctx)
        }
    };
    ($( $(#[$meta:meta])* fn $method:ident<$gen:ident : $bound:path>($($param:ident : $pty:ty),*) -> $ret:ty => $target:ident; )*) => {
        $(
            series_forward_in_generic!(
                $(#[$meta])*
                fn $method<$gen : $bound>($($param : $pty),*) -> $ret => $target;
            );
        )*
    };
}

macro_rules! series_reduce_in {
    ($(#[$meta:meta])* fn $method:ident($($param:ident : $pty:ty),*) => $agg:expr, $operation:literal;) => {
        $(#[$meta])*
        pub fn $method(&self, $($param: $pty,)* ctx: &ExecutionOverrides<'_>) -> Result<Scalar> {
            self.reduce_in_context($agg, ctx, $operation)
        }
    };
    ($( $(#[$meta:meta])* fn $method:ident($($param:ident : $pty:ty),*) => $agg:expr, $operation:literal; )*) => {
        $(
            series_reduce_in!(
                $(#[$meta])*
                fn $method($($param : $pty),*) => $agg, $operation;
            );
        )*
    };
}

macro_rules! series_scan_in {
    ($(#[$meta:meta])* fn $method:ident() => $agg:expr, $operation:literal;) => {
        $(#[$meta])*
        pub fn $method(&self, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
            self.scan_in_context($agg, ScanType::Inclusive, NullPolicy::Exclude, ctx, $operation)
        }
    };
    ($( $(#[$meta:meta])* fn $method:ident() => $agg:expr, $operation:literal; )*) => {
        $(
            series_scan_in!(
                $(#[$meta])*
                fn $method() => $agg, $operation;
            );
        )*
    };
}

macro_rules! series_unary_in {
    ($(#[$meta:meta])* fn $method:ident() => $target:ident;) => {
        $(#[$meta])*
        pub fn $method(&self, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
            self.$target(ctx)
        }
    };
    ($( $(#[$meta:meta])* fn $method:ident() => $target:ident; )*) => {
        $(
            series_unary_in!(
                $(#[$meta])*
                fn $method() => $target;
            );
        )*
    };
}

macro_rules! series_fillna_policy_in {
    ($(#[$meta:meta])* fn $method:ident() => $policy:expr;) => {
        $(#[$meta])*
        pub fn $method(&self, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
            self.fillna_policy_in_context($policy, ctx)
        }
    };
    ($( $(#[$meta:meta])* fn $method:ident() => $policy:expr; )*) => {
        $(
            series_fillna_policy_in!(
                $(#[$meta])*
                fn $method() => $policy;
            );
        )*
    };
}

macro_rules! series_binary_series_in {
    ($(#[$meta:meta])* fn $method:ident($rhs:ident : &Series) => $op:expr, $operation:literal;) => {
        $(#[$meta])*
        pub fn $method(&self, $rhs: &Series, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
            self.binary_series_in_context($rhs, $op, ctx, $operation)
        }
    };
    ($( $(#[$meta:meta])* fn $method:ident($rhs:ident : &Series) => $op:expr, $operation:literal; )*) => {
        $(
            series_binary_series_in!(
                $(#[$meta])*
                fn $method($rhs : &Series) => $op, $operation;
            );
        )*
    };
}

macro_rules! series_binary_scalar_in {
    ($(#[$meta:meta])* fn $method:ident<$gen:ident : $bound:path>($rhs:ident : $pty:ty) => $op:expr, $operation:literal;) => {
        $(#[$meta])*
        pub fn $method<$gen: $bound>(&self, $rhs: $pty, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
            self.binary_scalar_in_context($rhs, $op, ctx, $operation)
        }
    };
    ($( $(#[$meta:meta])* fn $method:ident<$gen:ident : $bound:path>($rhs:ident : $pty:ty) => $op:expr, $operation:literal; )*) => {
        $(
            series_binary_scalar_in!(
                $(#[$meta])*
                fn $method<$gen : $bound>($rhs : $pty) => $op, $operation;
            );
        )*
    };
}

pub(super) fn default_execution_context() -> ExecutionOverrides<'static> {
    ExecutionOverrides::new()
}

fn range_testing_series() -> Result<Series> {
    let runtime = Runtime::new()?;
    runtime.series(&[1i32, 2, 3])
}

pub(crate) fn trigger_series_row_range_start_limit_error_for_test() -> Result<Series> {
    range_testing_series()?.__debug_gather_range_for_test(usize::MAX, 1, "series::slice")
}

pub(crate) fn trigger_series_row_range_end_limit_error_for_test() -> Result<Series> {
    range_testing_series()?.__debug_gather_range_for_test(i32::MAX as usize, 1, "series::slice")
}

pub(crate) fn trigger_series_row_range_overflow_error_for_test() -> Result<Series> {
    range_testing_series()?.__debug_gather_range_for_test(i32::MAX as usize, usize::MAX, "series::slice")
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
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<(Arc<ExecutionDomain>, ConsumerCompletionFence)> {
        let output_domain = ctx.resolve_output_domain(self.column.execution_domain())?;
        let consumer_fence =
            ExecutionDomain::insert_stream_dependency(self.column.execution_domain(), &output_domain, operation)?;
        Ok((output_domain, consumer_fence))
    }

    // ── Hand-written helpers (unique control flow) ──────────────────────

    /// Concatenate multiple series using a borrowed per-operation execution override.
    pub fn concat_in(series: &[&Series], ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        let operation = "series::with_execution::concat";
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
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let (orders, null_orders) = options.resolve(1, operation)?;
        let cv = self.column.view()?;
        let ttv = TempTableView::from_column_views(vec![cv])?;
        let exec_options = output_domain.enter_execution()?;
        let sorted = crate::ops::sorting::sort_table_impl(
            ttv.table(),
            crate::ops::sorting::SortKeyConfig::new(Some(&orders), null_orders.as_deref()),
            options.stable,
            crate::ops::sorting::SortExecution::new(&output_domain, exec_options.options(), operation),
        )?
        .with_execution_domain(output_domain);
        let mut cols = sorted.into_columns()?;
        Series::from_column(cols.remove(0))
    }

    fn sort_values_in_context(&self, options: SortOptions, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.sort_values_with_context(options, ctx, "series::with_execution::sort_values")
    }

    pub(super) fn to_datetime_with_context(
        &self,
        resolution: Option<TimestampResolution>,
        ctx: &ExecutionOverrides<'_>,
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

    fn to_datetime_in_context(
        &self,
        resolution: Option<TimestampResolution>,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<Series> {
        self.to_datetime_with_context(
            resolution,
            ctx,
            "series::with_execution::to_datetime_duration",
            "series::with_execution::to_datetime",
        )
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

    pub(super) fn codebook_with_context(
        &self,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<Codebook> {
        self.codebook_with_options_in_context(ctx, DictionaryEncodingOptions::new(), operation)
    }

    pub(super) fn codebook_with_options_in_context(
        &self,
        ctx: &ExecutionOverrides<'_>,
        options: DictionaryEncodingOptions,
        operation: &'static str,
    ) -> Result<Codebook> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        self.build_codebook_in_domain(output_domain, options, operation)
    }

    fn codebook_in_context(&self, ctx: &ExecutionOverrides<'_>) -> Result<Codebook> {
        self.codebook_with_context(ctx, "series::with_execution::codebook")
    }

    pub(super) fn factorize_with_context(
        &self,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<Factorized> {
        self.factorize_with_options_in_context(ctx, DictionaryEncodingOptions::new(), operation)
    }

    pub(super) fn factorize_with_options_in_context(
        &self,
        ctx: &ExecutionOverrides<'_>,
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

    fn factorize_in_context(&self, ctx: &ExecutionOverrides<'_>) -> Result<Factorized> {
        self.factorize_with_context(ctx, "series::with_execution::factorize")
    }

    pub(super) fn fillna_policy_with_context(
        &self,
        policy: FillPolicy,
        ctx: &ExecutionOverrides<'_>,
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

    fn fillna_policy_in_context(&self, policy: FillPolicy, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        let operation = match policy {
            FillPolicy::Preceding => "series::with_execution::ffill",
            FillPolicy::Following => "series::with_execution::bfill",
        };
        self.fillna_policy_with_context(policy, ctx, operation)
    }

    pub(super) fn unique_with_context(&self, ctx: &ExecutionOverrides<'_>, operation: &'static str) -> Result<Series> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let ttv = TempTableView::from_column_views(vec![cv])?;
        let exec_options = output_domain.enter_execution()?;
        let result = stream_compaction::distinct_impl(
            ttv.table(),
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

    fn unique_in_context(&self, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.unique_with_context(ctx, "series::with_execution::unique")
    }

    pub(super) fn head_with_context(
        &self,
        n: usize,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        let total = self.len();
        let take = n.min(total);
        self.gather_range_in_context_inner(0, take, ctx, operation)
    }

    fn head_in_context(&self, n: usize, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.head_with_context(n, ctx, "series::with_execution::head")
    }

    pub(super) fn tail_with_context(
        &self,
        n: usize,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        let total = self.len();
        let take = n.min(total);
        let start = total - take;
        self.gather_range_in_context_inner(start, take, ctx, operation)
    }

    fn tail_in_context(&self, n: usize, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.tail_with_context(n, ctx, "series::with_execution::tail")
    }

    pub(super) fn slice_with_context(
        &self,
        offset: usize,
        length: usize,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        let total = self.len();
        let start = offset.min(total);
        let take = length.min(total - start);
        self.gather_range_in_context_inner(start, take, ctx, operation)
    }

    fn slice_in_context(&self, offset: usize, length: usize, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.slice_with_context(offset, length, ctx, "series::with_execution::slice")
    }

    fn gather_range_in_context_inner(
        &self,
        start: usize,
        len: usize,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        if len == 0 {
            let cv = self.column.view().map_err(|e| e.reanchor(operation))?;
            let empty =
                crate::core::column::Column::empty_with_dtype_in_domain(cv.data_type()?, output_domain, operation)?;
            return Series::from_column(empty);
        }

        let map = row_range::gather_map_in_domain(start, len, &output_domain, operation)?;
        let map_view = map.view()?;

        let cv = self.column.view().map_err(|e| e.reanchor(operation))?;
        let ttv = TempTableView::from_column_views(vec![cv])?;
        let exec_options = output_domain.enter_execution()?;
        let result = crate::ops::copying::gather_impl(
            ttv.table(),
            &map_view,
            cudf_sys::cudf_out_of_bounds_policy_t::CUDF_OUT_OF_BOUNDS_POLICY_DONT_CHECK,
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
        ctx: &ExecutionOverrides<'_>,
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

        if idx_cv.is_empty()? {
            let cv = self.column.view().map_err(|e| e.reanchor(operation))?;
            let empty =
                crate::core::column::Column::empty_with_dtype_in_domain(cv.data_type()?, output_domain, operation)?;
            return Series::from_column(empty);
        }

        let cv = self.column.view().map_err(|e| e.reanchor(operation))?;
        let ttv = TempTableView::from_column_views(vec![cv])?;
        let exec_options = output_domain.enter_execution()?;
        let result = crate::ops::copying::gather_impl(
            ttv.table(),
            &idx_cv,
            cudf_sys::cudf_out_of_bounds_policy_t::CUDF_OUT_OF_BOUNDS_POLICY_NULLIFY,
            "nullify",
            &output_domain,
            exec_options.options(),
            operation,
        )?;
        let mut cols = result.into_columns()?;
        Series::from_column(cols.remove(0))
    }

    fn gather_in_context(&self, indices: &Series, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.gather_with_context(indices, ctx, "series::with_execution::gather")
    }

    pub(super) fn filter_with_context(
        &self,
        mask: &Series,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        let (output_domain, mut consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let mask_cv = mask.view().map_err(|e| e.reanchor(operation))?;
        crate::ops::filter::validate_boolean_mask(&mask_cv, self.len(), "series length", operation)?;

        ExecutionDomain::require_same_device_for_mixed_inputs(
            self.column.execution_domain(),
            mask_cv.execution_domain(),
            operation,
        )?;
        consumer_fence.order_input(mask_cv.execution_domain())?;

        let cv = self.column.view().map_err(|e| e.reanchor(operation))?;
        let ttv = TempTableView::from_column_views(vec![cv])?;
        let exec_options = output_domain.enter_execution()?;
        let result = crate::ops::filter::apply_boolean_mask_impl(
            ttv.table(),
            &mask_cv,
            &output_domain,
            exec_options.options(),
            operation,
        )?;
        let mut cols = result.into_columns()?;
        Series::from_column(cols.remove(0))
    }

    fn filter_in_context(&self, mask: &Series, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.filter_with_context(mask, ctx, "series::with_execution::filter")
    }

    pub(super) fn dropna_with_context(&self, ctx: &ExecutionOverrides<'_>, operation: &'static str) -> Result<Series> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view().map_err(|e| e.reanchor(operation))?;
        let ttv = TempTableView::from_column_views(vec![cv])?;
        let exec_options = output_domain.enter_execution()?;
        let result = stream_compaction::drop_nulls_impl(
            ttv.table(),
            &[0],
            1,
            &output_domain,
            exec_options.options(),
            operation,
        )?;
        let mut cols = result.into_columns()?;
        Series::from_column(cols.remove(0))
    }

    fn dropna_in_context(&self, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.dropna_with_context(ctx, "series::with_execution::dropna")
    }

    pub(super) fn drop_duplicates_with_context(
        &self,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view().map_err(|e| e.reanchor(operation))?;
        let ttv = TempTableView::from_column_views(vec![cv])?;
        let exec_options = output_domain.enter_execution()?;
        let result = stream_compaction::distinct_impl(
            ttv.table(),
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

    fn drop_duplicates_in_context(&self, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.drop_duplicates_with_context(ctx, "series::with_execution::drop_duplicates")
    }

    pub(super) fn value_counts_with_context(
        &self,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<crate::DataFrame> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;

        let keys_view = self.column.view().map_err(|e| e.reanchor(operation))?;
        let values_view = self.column.view().map_err(|e| e.reanchor(operation))?;
        let ttv = TempTableView::from_column_views(vec![keys_view])?;
        let aggs = [GroupByAgg::Count { include_nulls: false }];
        let requests = [crate::ops::groupby::AggregationRequest { values: &values_view, aggregations: &aggs }];
        let exec_options = output_domain.enter_execution()?;
        let gb = crate::ops::groupby::groupby_aggregate_impl(
            ttv.table(),
            &requests,
            NullPolicy::Exclude,
            false,
            &output_domain,
            exec_options.options(),
            operation,
        )?;
        value_counts_from_groupby_output(gb, operation)
    }

    fn value_counts_in_context(&self, ctx: &ExecutionOverrides<'_>) -> Result<crate::DataFrame> {
        self.value_counts_with_context(ctx, "series::with_execution::value_counts")
    }

    pub(super) fn reduce_in_context(
        &self,
        agg: ReduceAgg,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<Scalar> {
        let input_type = self.dtype().map_err(|e| e.reanchor(operation))?;
        let output_type = infer_reduce_output_type(&input_type, &agg, operation)?;
        self.reduce_prepared_in_context(&agg, &output_type, ctx, operation, |error| error)
    }

    pub(super) fn reduce_prepared_in_context<F>(
        &self,
        agg: &ReduceAgg,
        output_type: &DataType,
        ctx: &ExecutionOverrides<'_>,
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

    pub(super) fn minmax_with_context(
        &self,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<(Scalar, Scalar)> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let exec_options = output_domain.enter_execution()?;
        reduction::minmax_impl(&cv, &output_domain, exec_options.options(), operation)
    }

    fn minmax_in_context(&self, ctx: &ExecutionOverrides<'_>) -> Result<(Scalar, Scalar)> {
        self.minmax_with_context(ctx, "series::with_execution::minmax")
    }

    // ── Macro-generated unary helpers ───────────────────────────────────

    pub(super) fn is_nan_with_context(&self, ctx: &ExecutionOverrides<'_>, operation: &'static str) -> Result<Series> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let exec_options = output_domain.enter_execution()?;
        let col = unary::is_nan_impl(&cv, &output_domain, exec_options.options(), operation)?;
        Series::from_column(col)
    }

    fn is_nan_in_context(&self, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.is_nan_with_context(ctx, "series::with_execution::is_nan")
    }

    pub(super) fn is_not_nan_with_context(
        &self,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let exec_options = output_domain.enter_execution()?;
        let col = unary::is_not_nan_impl(&cv, &output_domain, exec_options.options(), operation)?;
        Series::from_column(col)
    }

    fn is_not_nan_in_context(&self, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.is_not_nan_with_context(ctx, "series::with_execution::is_not_nan")
    }

    pub(super) fn logical_not_with_context(
        &self,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        self.unary_in_context(UnaryOperator::Not, ctx, operation)
    }

    fn logical_not_in_context(&self, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.logical_not_with_context(ctx, "series::with_execution::logical_not")
    }

    pub(super) fn is_null_with_context(&self, ctx: &ExecutionOverrides<'_>, operation: &'static str) -> Result<Series> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let exec_options = output_domain.enter_execution()?;
        let col = unary::is_null_impl(&cv, &output_domain, exec_options.options(), operation)?;
        Series::from_column(col)
    }

    fn is_null_in_context(&self, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.is_null_with_context(ctx, "series::with_execution::is_null")
    }

    pub(super) fn is_not_null_with_context(
        &self,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let exec_options = output_domain.enter_execution()?;
        let col = unary::is_valid_impl(&cv, &output_domain, exec_options.options(), operation)?;
        Series::from_column(col)
    }

    fn is_not_null_in_context(&self, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.is_not_null_with_context(ctx, "series::with_execution::is_not_null")
    }

    // Unary helpers with caller-supplied operation string
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
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        self.unary_in_context(op, ctx, operation)
    }

    fn scan_in_context(
        &self,
        agg: ScanAgg,
        scan_type: ScanType,
        null_policy: NullPolicy,
        ctx: &ExecutionOverrides<'_>,
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
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        self.scan_in_context(agg, scan_type, null_policy, ctx, operation)
    }

    series_context_op!(unary fn string_unary_in_context(
    op: fn(
        &crate::core::column::ColumnView<'_>,
        &Arc<ExecutionDomain>,
        &cudf_sys::cudf_exec_options_t,
        &'static str,
    ) -> Result<crate::core::column::Column>),
    caller_op |cv, output_domain, exec, operation| op(
        &cv,
        &output_domain,
        exec.options(),
        operation
    ));

    pub(super) fn string_unary_with_context(
        &self,
        op: fn(
            &crate::core::column::ColumnView<'_>,
            &Arc<ExecutionDomain>,
            &cudf_sys::cudf_exec_options_t,
            &'static str,
        ) -> Result<crate::core::column::Column>,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        self.string_unary_in_context(op, ctx, operation)
    }

    pub(super) fn replace_with_backrefs_output_size_in_context(
        &self,
        pattern: &str,
        replacement: &str,
        ctx: &ExecutionOverrides<'_>,
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
        ctx: &ExecutionOverrides<'_>,
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
        ctx: &ExecutionOverrides<'_>,
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
        ctx: &ExecutionOverrides<'_>,
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
        ctx: &ExecutionOverrides<'_>,
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
        ctx: &ExecutionOverrides<'_>,
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
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let exec_options = output_domain.enter_execution()?;
        let col = strings::try_to_integers_impl(&cv, output_type, &output_domain, exec_options.options(), operation)?;
        Series::from_column(col)
    }

    // ── Cast (special: `impl Into<DataType>` needs local conversion) ───

    pub(super) fn cast_in_context(
        &self,
        target: impl Into<DataType>,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let target = target.into();
        let cv = self.column.view()?;
        let exec_options = output_domain.enter_execution()?;
        let col = unary::cast_impl(&cv, &target, &output_domain, exec_options.options(), operation)?;
        Series::from_column(col)
    }

    // ── Fillna with scalar (generic over T: CudfType) ──────────────────

    pub(super) fn fillna_with_context<T: CudfType>(
        &self,
        value: T,
        ctx: &ExecutionOverrides<'_>,
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

    fn fillna_in_context<T: CudfType>(&self, value: T, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.fillna_with_context(value, ctx, "series::with_execution::fillna")
    }

    // ── Macro-generated binary helpers ──────────────────────────────────

    pub(super) fn fillna_with_series_with_context(
        &self,
        other: &Series,
        ctx: &ExecutionOverrides<'_>,
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

    fn fillna_with_series_in_context(&self, other: &Series, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.fillna_with_series_with_context(other, ctx, "series::with_execution::fillna_with_series")
    }

    pub(super) fn isin_with_context(
        &self,
        values: &Series,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<Series> {
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
        let col = crate::ops::search::contains_impl(&values_cv, &cv, &output_domain, exec_options.options(), operation)
            .map_err(annotate_type_context)?;
        Series::from_column(col)
    }

    fn isin_in_context(&self, values: &Series, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.isin_with_context(values, ctx, "series::with_execution::isin")
    }

    // ── Binary with type inference (hand-written — output type depends on both inputs) ──

    pub(super) fn binary_series_in_context(
        &self,
        other: &Series,
        op: BinaryOperator,
        ctx: &ExecutionOverrides<'_>,
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

    pub(super) fn binary_scalar_in_context<T: IntoRuntimeScalar>(
        &self,
        value: T,
        op: BinaryOperator,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        let (output_domain, _consumer_fence) = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let scalar = value.into_runtime_scalar(&output_domain)?;
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

    pub(crate) fn __debug_gather_range_for_test(
        &self,
        start: usize,
        len: usize,
        operation: &'static str,
    ) -> Result<Series> {
        self.gather_range_in_context_inner(start, len, &default_execution_context(), operation)
    }
}

impl Series {
    series_forward_in!(
        /// Return the first `n` rows using the explicit execution-context override.
        fn head_in(n: usize) -> Result<Series> => head_in_context;

        /// Return the last `n` rows using the explicit execution-context override.
        fn tail_in(n: usize) -> Result<Series> => tail_in_context;

        /// Return `length` rows starting at `offset` using the explicit execution-context override.
        fn slice_in(offset: usize, length: usize) -> Result<Series> => slice_in_context;

        /// Gather rows using the explicit execution-context override.
        fn gather_in(indices: &Series) -> Result<Series> => gather_in_context;

        /// Filter rows using the explicit execution-context override.
        fn filter_in(mask: &Series) -> Result<Series> => filter_in_context;

        /// Sort using the explicit execution-context override.
        fn sort_values_in(options: SortOptions) -> Result<Series> => sort_values_in_context;

        /// Drop null rows using the explicit execution-context override.
        fn dropna_in() -> Result<Series> => dropna_in_context;

        /// Remove duplicate rows using the explicit execution-context override.
        fn drop_duplicates_in() -> Result<Series> => drop_duplicates_in_context;

        /// Count the frequency of each distinct non-null value using the explicit execution-context override.
        fn value_counts_in() -> Result<crate::DataFrame> => value_counts_in_context;

        /// Replace null values using another series with the explicit execution-context override.
        fn fillna_with_series_in(other: &Series) -> Result<Series> => fillna_with_series_in_context;

        /// Reduce to a min/max pair using the explicit execution-context override.
        fn minmax_in() -> Result<(Scalar, Scalar)> => minmax_in_context;

        /// Convert numeric values to timestamps using the explicit execution-context override.
        fn to_datetime_in(resolution: Option<TimestampResolution>) -> Result<Series> => to_datetime_in_context;

        /// Element-wise membership test using the explicit execution-context override.
        fn isin_in(values: &Series) -> Result<Series> => isin_in_context;

        /// Remove duplicates using the explicit execution-context override.
        fn unique_in() -> Result<Series> => unique_in_context;

        /// Build a reusable codebook using the explicit execution-context override.
        fn codebook_in() -> Result<Codebook> => codebook_in_context;

        /// Factorize using the explicit execution-context override.
        fn factorize_in() -> Result<Factorized> => factorize_in_context;
    );

    series_forward_in_generic!(
        /// Replace null values with a scalar using the explicit execution-context override.
        fn fillna_in<T: CudfType>(value: T) -> Result<Series> => fillna_in_context;
    );

    series_fillna_policy_in!(
        /// Forward fill using the explicit execution-context override.
        fn ffill_in() => FillPolicy::Preceding;

        /// Backward fill using the explicit execution-context override.
        fn bfill_in() => FillPolicy::Following;
    );

    series_reduce_in!(
        /// Reduce to a scalar sum using the explicit execution-context override.
        fn sum_in() => ReduceAgg::Sum, "series::sum_in";

        /// Reduce to a scalar minimum using the explicit execution-context override.
        fn min_in() => ReduceAgg::Min, "series::min_in";

        /// Reduce to a scalar maximum using the explicit execution-context override.
        fn max_in() => ReduceAgg::Max, "series::max_in";

        /// Reduce to a scalar mean using the explicit execution-context override.
        fn mean_in() => ReduceAgg::Mean, "series::mean_in";

        /// Reduce to a scalar median using the explicit execution-context override.
        fn median_in() => ReduceAgg::Median, "series::median_in";

        /// Reduce to a scalar standard deviation using the explicit execution-context override.
        fn std_in(ddof: u32) => ReduceAgg::Std { ddof }, "series::std_in";

        /// Reduce to a scalar variance using the explicit execution-context override.
        fn var_in(ddof: u32) => ReduceAgg::Variance { ddof }, "series::var_in";

        /// Reduce to a scalar logical-any using the explicit execution-context override.
        fn any_in() => ReduceAgg::Any, "series::any_in";

        /// Reduce to a scalar logical-all using the explicit execution-context override.
        fn all_in() => ReduceAgg::All, "series::all_in";

        /// Reduce to a scalar product using the explicit execution-context override.
        fn product_in() => ReduceAgg::Product, "series::product_in";
    );

    /// Reduce to a host sum value using the explicit execution-context override.
    pub fn sum_value_in<T: CudfType>(&self, ctx: &ExecutionOverrides<'_>) -> Result<T> {
        self.sum_in(ctx)?.value::<T>()
    }

    /// Reduce to a host minimum value using the explicit execution-context override.
    pub fn min_value_in<T: CudfType>(&self, ctx: &ExecutionOverrides<'_>) -> Result<Option<T>> {
        scalar_to_optional_value(self.min_in(ctx)?)
    }

    /// Reduce to a host maximum value using the explicit execution-context override.
    pub fn max_value_in<T: CudfType>(&self, ctx: &ExecutionOverrides<'_>) -> Result<Option<T>> {
        scalar_to_optional_value(self.max_in(ctx)?)
    }

    /// Reduce to a host mean value using the explicit execution-context override.
    pub fn mean_value_in(&self, ctx: &ExecutionOverrides<'_>) -> Result<Option<f64>> {
        scalar_to_optional_value(self.mean_in(ctx)?)
    }

    /// Reduce to a host median value using the explicit execution-context override.
    pub fn median_value_in(&self, ctx: &ExecutionOverrides<'_>) -> Result<Option<f64>> {
        scalar_to_optional_value(self.median_in(ctx)?)
    }

    /// Reduce to a host standard deviation value using the explicit execution-context override.
    pub fn std_value_in(&self, ddof: u32, ctx: &ExecutionOverrides<'_>) -> Result<Option<f64>> {
        scalar_to_optional_value(self.std_in(ddof, ctx)?)
    }

    /// Reduce to a host variance value using the explicit execution-context override.
    pub fn var_value_in(&self, ddof: u32, ctx: &ExecutionOverrides<'_>) -> Result<Option<f64>> {
        scalar_to_optional_value(self.var_in(ddof, ctx)?)
    }

    /// Reduce to a host bool logical-any using the explicit execution-context override.
    pub fn any_value_in(&self, ctx: &ExecutionOverrides<'_>) -> Result<bool> {
        Ok(scalar_to_optional_value(self.any_in(ctx)?)?.unwrap_or(false))
    }

    /// Reduce to a host bool logical-all using the explicit execution-context override.
    pub fn all_value_in(&self, ctx: &ExecutionOverrides<'_>) -> Result<bool> {
        Ok(scalar_to_optional_value(self.all_in(ctx)?)?.unwrap_or(true))
    }

    /// Reduce to a host product value using the explicit execution-context override.
    pub fn product_value_in<T: CudfType>(&self, ctx: &ExecutionOverrides<'_>) -> Result<Option<T>> {
        scalar_to_optional_value(self.product_in(ctx)?)
    }

    /// Reduce to host min/max values using the explicit execution-context override.
    pub fn minmax_values_in<T: CudfType>(&self, ctx: &ExecutionOverrides<'_>) -> Result<(Option<T>, Option<T>)> {
        let (min, max) = self.minmax_in(ctx)?;
        Ok((scalar_to_optional_value(min)?, scalar_to_optional_value(max)?))
    }

    series_scan_in!(
        /// Inclusive cumulative sum using the explicit execution-context override.
        fn cumsum_in() => ScanAgg::Sum, "series::cumsum_in";

        /// Inclusive cumulative minimum using the explicit execution-context override.
        fn cummin_in() => ScanAgg::Min, "series::cummin_in";

        /// Inclusive cumulative maximum using the explicit execution-context override.
        fn cummax_in() => ScanAgg::Max, "series::cummax_in";

        /// Inclusive cumulative product using the explicit execution-context override.
        fn cumprod_in() => ScanAgg::Product, "series::cumprod_in";
    );

    series_unary_in!(
        /// Element-wise NaN check using the explicit execution-context override.
        fn is_nan_in() => is_nan_in_context;

        /// Element-wise not-NaN check using the explicit execution-context override.
        fn is_not_nan_in() => is_not_nan_in_context;

        /// Element-wise logical NOT using the explicit execution-context override.
        fn logical_not_in() => logical_not_in_context;

        /// Return a BOOL8 mask: `true` where elements are null.
        fn is_null_in() => is_null_in_context;

        /// Return a BOOL8 mask: `true` where elements are valid.
        fn is_not_null_in() => is_not_null_in_context;
    );

    /// Element-wise absolute value using the explicit execution-context override.
    pub fn abs_in(&self, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.unary_in_context(UnaryOperator::Abs, ctx, "series::abs_in")
    }

    /// Element-wise arithmetic negation using the explicit execution-context override.
    pub fn neg_in(&self, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.unary_in_context(UnaryOperator::Negate, ctx, "series::neg_in")
    }

    /// Element-wise logical AND using the explicit execution-context override.
    pub fn logical_and_in(&self, rhs: &Series, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        let operation = "series::logical_and_in";
        self.require_bool_binary_operands(rhs, operation)?;
        self.binary_series_in_context(rhs, BinaryOperator::LogicalAnd, ctx, operation)
    }

    /// Element-wise logical OR using the explicit execution-context override.
    pub fn logical_or_in(&self, rhs: &Series, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        let operation = "series::logical_or_in";
        self.require_bool_binary_operands(rhs, operation)?;
        self.binary_series_in_context(rhs, BinaryOperator::LogicalOr, ctx, operation)
    }

    /// Cast using the explicit execution-context override.
    pub fn astype_in(&self, target: impl Into<DataType>, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        self.cast_in_context(target, ctx, "series::astype_in")
    }

    /// Cast to a numeric type using the explicit execution-context override.
    pub fn to_numeric_in(&self, target: TypeId, ctx: &ExecutionOverrides<'_>) -> Result<Series> {
        if !target.is_numeric() && !target.is_boolean() {
            return Err(unsupported_numeric_target_error("series::to_numeric_in", target));
        }
        self.cast_in_context(target, ctx, "series::to_numeric_in")
    }

    series_binary_scalar_in!(
        /// Element-wise equality-to-scalar using the explicit execution-context override.
        fn eq_scalar_in<T: IntoRuntimeScalar>(rhs: T) => BinaryOperator::Equal, "series::eq_scalar_in";

        /// Element-wise inequality-to-scalar using the explicit execution-context override.
        fn ne_scalar_in<T: IntoRuntimeScalar>(rhs: T) => BinaryOperator::NotEqual, "series::ne_scalar_in";

        /// Element-wise greater-than-scalar using the explicit execution-context override.
        fn gt_scalar_in<T: IntoRuntimeScalar>(rhs: T) => BinaryOperator::Greater, "series::gt_scalar_in";

        /// Element-wise less-than-scalar using the explicit execution-context override.
        fn lt_scalar_in<T: IntoRuntimeScalar>(rhs: T) => BinaryOperator::Less, "series::lt_scalar_in";

        /// Element-wise greater-than-or-equal-to-scalar using the explicit execution-context override.
        fn ge_scalar_in<T: IntoRuntimeScalar>(rhs: T) => BinaryOperator::GreaterEqual, "series::ge_scalar_in";

        /// Element-wise less-than-or-equal-to-scalar using the explicit execution-context override.
        fn le_scalar_in<T: IntoRuntimeScalar>(rhs: T) => BinaryOperator::LessEqual, "series::le_scalar_in";

        /// Element-wise addition-by-scalar using the explicit execution-context override.
        fn add_scalar_in<T: CudfType>(rhs: T) => BinaryOperator::Add, "series::add_scalar_in";

        /// Element-wise subtraction-by-scalar using the explicit execution-context override.
        fn sub_scalar_in<T: CudfType>(rhs: T) => BinaryOperator::Sub, "series::sub_scalar_in";

        /// Element-wise multiplication-by-scalar using the explicit execution-context override.
        fn mul_scalar_in<T: CudfType>(rhs: T) => BinaryOperator::Mul, "series::mul_scalar_in";

        /// Element-wise division-by-scalar using the explicit execution-context override.
        fn div_scalar_in<T: CudfType>(rhs: T) => BinaryOperator::Div, "series::div_scalar_in";

        /// Element-wise modulo-by-scalar using the explicit execution-context override.
        fn rem_scalar_in<T: CudfType>(rhs: T) => BinaryOperator::Mod, "series::rem_scalar_in";
    );

    series_binary_series_in!(
        /// Element-wise equality using the explicit execution-context override.
        fn eq_in(rhs: &Series) => BinaryOperator::Equal, "series::eq_in";

        /// Element-wise inequality using the explicit execution-context override.
        fn ne_in(rhs: &Series) => BinaryOperator::NotEqual, "series::ne_in";

        /// Element-wise greater-than using the explicit execution-context override.
        fn gt_in(rhs: &Series) => BinaryOperator::Greater, "series::gt_in";

        /// Element-wise less-than using the explicit execution-context override.
        fn lt_in(rhs: &Series) => BinaryOperator::Less, "series::lt_in";

        /// Element-wise greater-than-or-equal using the explicit execution-context override.
        fn ge_in(rhs: &Series) => BinaryOperator::GreaterEqual, "series::ge_in";

        /// Element-wise less-than-or-equal using the explicit execution-context override.
        fn le_in(rhs: &Series) => BinaryOperator::LessEqual, "series::le_in";

        /// Element-wise addition using the explicit execution-context override.
        fn add_in(rhs: &Series) => BinaryOperator::Add, "series::add_in";

        /// Element-wise subtraction using the explicit execution-context override.
        fn sub_in(rhs: &Series) => BinaryOperator::Sub, "series::sub_in";

        /// Element-wise multiplication using the explicit execution-context override.
        fn mul_in(rhs: &Series) => BinaryOperator::Mul, "series::mul_in";

        /// Element-wise division using the explicit execution-context override.
        fn div_in(rhs: &Series) => BinaryOperator::Div, "series::div_in";

        /// Element-wise modulo using the explicit execution-context override.
        fn rem_in(rhs: &Series) => BinaryOperator::Mod, "series::rem_in";
    );

    /// Count distinct non-null values using the explicit execution-context override.
    pub fn nunique_in(&self, ctx: &ExecutionOverrides<'_>) -> Result<usize> {
        self.nunique_with_context(ctx, "series::nunique_in")
    }

    /// Build a reusable codebook using the explicit execution-context override and explicit encoding policy.
    pub fn codebook_with_in(
        &self,
        options: DictionaryEncodingOptions,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<Codebook> {
        self.codebook_with_options_in_context(ctx, options, "series::codebook_with_in")
    }

    /// Factorize using the explicit execution-context override and explicit encoding policy.
    pub fn factorize_with_in(
        &self,
        options: DictionaryEncodingOptions,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<Factorized> {
        self.factorize_with_options_in_context(ctx, options, "series::factorize_with_in")
    }
}
