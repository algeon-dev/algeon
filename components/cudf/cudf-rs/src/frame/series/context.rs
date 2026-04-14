use std::sync::Arc;

use super::{
    owned::{SERIES_VALUE_COUNTS_COUNT_NAME, SERIES_VALUE_COUNTS_FALLBACK_NAME, Series},
    reduction::{infer_reduce_output_type, scalar_to_optional_value},
};
use crate::{
    core::{
        column::copy_view_to_owned,
        error::{Error, ErrorKind, Result},
        scalar::Scalar,
        types::{CudfType, DataType, NullPolicy, TimestampResolution, TypeId},
    },
    frame::factorization::{Codebook, EncodedSeries, Factorized},
    internal::temp_table::TempTableView,
    ops::{
        binary::{self, BinaryOperator},
        dictionary,
        groupby::{AggSpec, GroupByAgg},
        reduction::{self, ReduceAgg, ScanAgg, ScanType},
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
     |$cv:ident, $domain:ident, $exec:ident, $op:ident| $call:expr) => {
        fn $method(&self, $($param: $pty,)* ctx: &ExecutionContext<'_>) -> Result<Series> {
            let $op = $operation;
            let $domain = self.resolve_context_output_domain(ctx, $op)?;
            let $cv = self.column.view()?;
            let $exec = $domain.explicit_exec_options_raw();
            let col = $call?;
            self.with_inherited_name(col)
        }
    };
    // Unary with caller-supplied operation string
    (unary fn $method:ident($($param:ident : $pty:ty),*),
     caller_op |$cv:ident, $domain:ident, $exec:ident, $op:ident| $call:expr) => {
        fn $method(&self, $($param: $pty,)* ctx: &ExecutionContext<'_>, $op: &'static str) -> Result<Series> {
            let $domain = self.resolve_context_output_domain(ctx, $op)?;
            let $cv = self.column.view()?;
            let $exec = $domain.explicit_exec_options_raw();
            let col = $call?;
            self.with_inherited_name(col)
        }
    };
    // Binary (two-Series): adds require_same_device + fence for the other Series
    (binary fn $method:ident($other:ident : &Series),
     $operation:literal, |$cv:ident, $other_cv:ident, $domain:ident, $exec:ident, $op:ident| $call:expr) => {
        fn $method(
            &self,
            $other: &Series,
            ctx: &ExecutionContext<'_>,
        ) -> Result<Series> {
            let $op = $operation;
            let $domain = self.resolve_context_output_domain(ctx, $op)?;
            ExecutionDomain::require_same_device_for_mixed_inputs(
                self.column.execution_domain(),
                $other.column.execution_domain(),
                $op,
            )?;
            if !ExecutionDomain::same_provenance($other.column.execution_domain(), &$domain) {
                $other.column.execution_domain().fence_stream($op)?;
            }
            let $cv = self.column.view()?;
            let $other_cv = $other.column.view()?;
            let $exec = $domain.explicit_exec_options_raw();
            let col = $call?;
            self.with_inherited_name(col)
        }
    };
}

macro_rules! series_scope_forward {
    ($(#[$meta:meta])* fn $method:ident($($param:ident : $pty:ty),*) -> $ret:ty => $target:ident;) => {
        $(#[$meta])*
        pub fn $method(&self, $($param: $pty),*) -> $ret {
            self.series.$target($($param,)* self.ctx)
        }
    };
    ($( $(#[$meta:meta])* fn $method:ident($($param:ident : $pty:ty),*) -> $ret:ty => $target:ident; )*) => {
        $(
            series_scope_forward!(
                $(#[$meta])*
                fn $method($($param : $pty),*) -> $ret => $target;
            );
        )*
    };
}

macro_rules! series_scope_forward_generic {
    ($(#[$meta:meta])* fn $method:ident<$gen:ident : $bound:path>($($param:ident : $pty:ty),*) -> $ret:ty => $target:ident;) => {
        $(#[$meta])*
        pub fn $method<$gen: $bound>(&self, $($param: $pty),*) -> $ret {
            self.series.$target($($param,)* self.ctx)
        }
    };
    ($( $(#[$meta:meta])* fn $method:ident<$gen:ident : $bound:path>($($param:ident : $pty:ty),*) -> $ret:ty => $target:ident; )*) => {
        $(
            series_scope_forward_generic!(
                $(#[$meta])*
                fn $method<$gen : $bound>($($param : $pty),*) -> $ret => $target;
            );
        )*
    };
}

macro_rules! series_scope_reduce {
    ($(#[$meta:meta])* fn $method:ident($($param:ident : $pty:ty),*) => $agg:expr, $operation:literal;) => {
        $(#[$meta])*
        pub fn $method(&self, $($param: $pty),*) -> Result<Scalar> {
            self.series.reduce_in_context($agg, self.ctx, $operation)
        }
    };
    ($( $(#[$meta:meta])* fn $method:ident($($param:ident : $pty:ty),*) => $agg:expr, $operation:literal; )*) => {
        $(
            series_scope_reduce!(
                $(#[$meta])*
                fn $method($($param : $pty),*) => $agg, $operation;
            );
        )*
    };
}

macro_rules! series_scope_scan {
    ($(#[$meta:meta])* fn $method:ident() => $agg:expr, $operation:literal;) => {
        $(#[$meta])*
        pub fn $method(&self) -> Result<Series> {
            self.series.scan_in_context($agg, ScanType::Inclusive, NullPolicy::Exclude, self.ctx, $operation)
        }
    };
    ($( $(#[$meta:meta])* fn $method:ident() => $agg:expr, $operation:literal; )*) => {
        $(
            series_scope_scan!(
                $(#[$meta])*
                fn $method() => $agg, $operation;
            );
        )*
    };
}

macro_rules! series_scope_unary {
    ($(#[$meta:meta])* fn $method:ident() => $target:ident;) => {
        $(#[$meta])*
        pub fn $method(&self) -> Result<Series> {
            self.series.$target(self.ctx)
        }
    };
    ($( $(#[$meta:meta])* fn $method:ident() => $target:ident; )*) => {
        $(
            series_scope_unary!(
                $(#[$meta])*
                fn $method() => $target;
            );
        )*
    };
}

macro_rules! series_scope_fillna_policy {
    ($(#[$meta:meta])* fn $method:ident() => $policy:expr;) => {
        $(#[$meta])*
        pub fn $method(&self) -> Result<Series> {
            self.series.fillna_policy_in_context($policy, self.ctx)
        }
    };
    ($( $(#[$meta:meta])* fn $method:ident() => $policy:expr; )*) => {
        $(
            series_scope_fillna_policy!(
                $(#[$meta])*
                fn $method() => $policy;
            );
        )*
    };
}

macro_rules! series_scope_binary_series {
    ($(#[$meta:meta])* fn $method:ident($rhs:ident : &Series) => $op:expr, $operation:literal;) => {
        $(#[$meta])*
        pub fn $method(&self, $rhs: &Series) -> Result<Series> {
            self.series.binary_series_in_context($rhs, $op, self.ctx, $operation)
        }
    };
    ($( $(#[$meta:meta])* fn $method:ident($rhs:ident : &Series) => $op:expr, $operation:literal; )*) => {
        $(
            series_scope_binary_series!(
                $(#[$meta])*
                fn $method($rhs : &Series) => $op, $operation;
            );
        )*
    };
}

macro_rules! series_scope_binary_scalar {
    ($(#[$meta:meta])* fn $method:ident<$gen:ident : $bound:path>($rhs:ident : $pty:ty) => $op:expr, $operation:literal;) => {
        $(#[$meta])*
        pub fn $method<$gen: $bound>(&self, $rhs: $pty) -> Result<Series> {
            self.series.binary_scalar_in_context($rhs, $op, self.ctx, $operation)
        }
    };
    ($( $(#[$meta:meta])* fn $method:ident<$gen:ident : $bound:path>($rhs:ident : $pty:ty) => $op:expr, $operation:literal; )*) => {
        $(
            series_scope_binary_scalar!(
                $(#[$meta])*
                fn $method<$gen : $bound>($rhs : $pty) => $op, $operation;
            );
        )*
    };
}

macro_rules! series_scope_binary_string_scalar {
    ($(#[$meta:meta])* fn $method:ident($rhs:ident : &str) => $op:expr, $operation:literal;) => {
        $(#[$meta])*
        pub fn $method(&self, $rhs: &str) -> Result<Series> {
            self.series.binary_string_scalar_in_context($rhs, $op, self.ctx, $operation)
        }
    };
    ($( $(#[$meta:meta])* fn $method:ident($rhs:ident : &str) => $op:expr, $operation:literal; )*) => {
        $(
            series_scope_binary_string_scalar!(
                $(#[$meta])*
                fn $method($rhs : &str) => $op, $operation;
            );
        )*
    };
}

macro_rules! series_scope_string_unary {
    ($(#[$meta:meta])* fn $method:ident() => $target:path, $operation:literal;) => {
        $(#[$meta])*
        pub fn $method(&self) -> Result<Series> {
            self.series.string_unary_in_context($target, self.ctx, $operation)
        }
    };
    ($( $(#[$meta:meta])* fn $method:ident() => $target:path, $operation:literal; )*) => {
        $(
            series_scope_string_unary!(
                $(#[$meta])*
                fn $method() => $target, $operation;
            );
        )*
    };
}

macro_rules! series_scope_string_scalar {
    ($(#[$meta:meta])* fn $method:ident($pat:ident : &str) => $target:path, $operation:literal;) => {
        $(#[$meta])*
        pub fn $method(&self, $pat: &str) -> Result<Series> {
            self.series.string_scalar_in_context($pat, self.ctx, $operation, $target)
        }
    };
    ($( $(#[$meta:meta])* fn $method:ident($pat:ident : &str) => $target:path, $operation:literal; )*) => {
        $(
            series_scope_string_scalar!(
                $(#[$meta])*
                fn $method($pat : &str) => $target, $operation;
            );
        )*
    };
}

/// Scoped borrowed execution override for one `Series` operation chain.
#[must_use]
pub struct SeriesExecutionScope<'series, 'ctx, 'handles> {
    series: &'series Series,
    ctx: &'ctx ExecutionContext<'handles>,
}

/// Scoped string namespace for one `SeriesExecutionScope` operation chain.
#[must_use]
pub struct SeriesExecutionScopeStringNamespace<'series, 'ctx, 'handles> {
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
        let sorted = crate::ops::sorting::sort_table_impl(
            ttv.table(),
            Some(&orders),
            null_orders.as_deref(),
            options.stable,
            &output_domain,
            exec_options.as_ref(),
            operation,
        )?
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
        if source_type.is_numeric() || source_type.is_boolean() {
            let duration =
                self.cast_in_context(resolution.duration_type_id(), ctx, "series::in_context::to_datetime_duration")?;
            duration.cast_in_context(resolution.type_id(), ctx, "series::in_context::to_datetime")
        } else {
            self.cast_in_context(resolution.type_id(), ctx, "series::in_context::to_datetime")
        }
    }

    pub(crate) fn build_codebook_in_domain(
        &self,
        output_domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Codebook> {
        let cv = self.column.view()?;
        let exec_options = output_domain.explicit_exec_options_raw();
        let dict = dictionary::encode_impl(&cv, TypeId::Int32, &output_domain, exec_options.as_ref(), operation)?;
        let dict_view = dict.view()?;
        let keys_view = dictionary::keys(&dict_view)?;
        let keys = Series::from_column(copy_view_to_owned(&keys_view)?)?;
        let keys = match self.name() {
            Some(name) => keys.rename(format!("{name}_keys")),
            None => keys,
        };
        Codebook::from_keys_with_fallback(keys, self.name().map(str::to_string))
    }

    fn codebook_in_context(&self, ctx: &ExecutionContext<'_>) -> Result<Codebook> {
        let operation = "series::in_context::codebook";
        let output_domain = self.resolve_context_output_domain(ctx, operation)?;
        self.build_codebook_in_domain(output_domain, operation)
    }

    fn factorize_in_context(&self, ctx: &ExecutionContext<'_>) -> Result<Factorized> {
        let operation = "series::in_context::factorize";
        let output_domain = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let exec_options = output_domain.explicit_exec_options_raw();
        let dict = dictionary::encode_impl(&cv, TypeId::Int32, &output_domain, exec_options.as_ref(), operation)?;
        let dict_view = dict.view()?;
        let indices_view = dictionary::indices(&dict_view)?;
        let keys_view = dictionary::keys(&dict_view)?;

        let row_codes = Series::from_column(copy_view_to_owned(&indices_view)?)?;
        let keys = Series::from_column(copy_view_to_owned(&keys_view)?)?;
        let (row_codes, keys) = match self.name() {
            Some(name) => (row_codes.rename(format!("{name}_codes")), keys.rename(format!("{name}_keys"))),
            None => (row_codes, keys),
        };
        let row_codes = EncodedSeries::new(row_codes, self.name().map(str::to_string));
        let codebook = Codebook::from_keys_with_fallback(keys, self.name().map(str::to_string))?;
        Ok(Factorized::new(row_codes, codebook))
    }

    fn fillna_policy_in_context(&self, policy: FillPolicy, ctx: &ExecutionContext<'_>) -> Result<Series> {
        let operation = match policy {
            FillPolicy::Preceding => "series::in_context::ffill",
            FillPolicy::Following => "series::in_context::bfill",
        };
        let output_domain = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let exec_options = output_domain.explicit_exec_options_raw();
        let col = replace::replace_nulls_policy_with_exec_options(
            &cv,
            policy,
            &output_domain,
            exec_options.as_ref(),
            operation,
        )?;
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
            &output_domain,
            exec_options.as_ref(),
            operation,
        )?;
        let mut cols = result.into_columns()?;
        self.with_inherited_name(cols.remove(0))
    }

    fn head_in_context(&self, n: usize, ctx: &ExecutionContext<'_>) -> Result<Series> {
        let operation = "series::in_context::head";
        let (df, column_name) = self.as_single_column_dataframe(operation)?;
        let result = df.in_context(ctx).head(n).map_err(|e| e.with_operation(operation))?;
        self.restore_single_column_result(result, &column_name, operation)
    }

    fn tail_in_context(&self, n: usize, ctx: &ExecutionContext<'_>) -> Result<Series> {
        let operation = "series::in_context::tail";
        let (df, column_name) = self.as_single_column_dataframe(operation)?;
        let result = df.in_context(ctx).tail(n).map_err(|e| e.with_operation(operation))?;
        self.restore_single_column_result(result, &column_name, operation)
    }

    fn slice_in_context(&self, offset: usize, length: usize, ctx: &ExecutionContext<'_>) -> Result<Series> {
        let operation = "series::in_context::slice";
        let (df, column_name) = self.as_single_column_dataframe(operation)?;
        let result = df.in_context(ctx).slice(offset, length).map_err(|e| e.with_operation(operation))?;
        self.restore_single_column_result(result, &column_name, operation)
    }

    fn gather_in_context(&self, indices: &Series, ctx: &ExecutionContext<'_>) -> Result<Series> {
        let operation = "series::in_context::gather";
        let (df, column_name) = self.as_single_column_dataframe(operation)?;
        let result = df.in_context(ctx).gather(indices).map_err(|e| e.with_operation(operation))?;
        self.restore_single_column_result(result, &column_name, operation)
    }

    fn filter_in_context(&self, mask: &Series, ctx: &ExecutionContext<'_>) -> Result<Series> {
        let operation = "series::in_context::filter";
        let (df, column_name) = self.as_single_column_dataframe(operation)?;
        let result = df.in_context(ctx).filter(mask).map_err(|e| e.with_operation(operation))?;
        self.restore_single_column_result(result, &column_name, operation)
    }

    fn dropna_in_context(&self, ctx: &ExecutionContext<'_>) -> Result<Series> {
        let operation = "series::in_context::dropna";
        let (df, column_name) = self.as_single_column_dataframe(operation)?;
        let result =
            df.in_context(ctx).dropna(crate::DropNaOptions::default()).map_err(|e| e.with_operation(operation))?;
        self.restore_single_column_result(result, &column_name, operation)
    }

    fn drop_duplicates_in_context(&self, ctx: &ExecutionContext<'_>) -> Result<Series> {
        let operation = "series::in_context::drop_duplicates";
        let (df, column_name) = self.as_single_column_dataframe(operation)?;
        let result = df
            .in_context(ctx)
            .drop_duplicates(crate::DropDuplicatesOptions::default())
            .map_err(|e| e.with_operation(operation))?;
        self.restore_single_column_result(result, &column_name, operation)
    }

    fn value_counts_in_context(&self, ctx: &ExecutionContext<'_>) -> Result<crate::DataFrame> {
        let operation = "series::in_context::value_counts";
        let (df, column_name) = self.as_single_column_dataframe(operation)?;
        let value_name =
            self.name().map(str::to_string).unwrap_or_else(|| SERIES_VALUE_COUNTS_FALLBACK_NAME.to_string());

        let mut counts = df
            .groupby([column_name.as_str()])
            .in_context(ctx)
            .agg(&[AggSpec::new(column_name.as_str(), GroupByAgg::Count { include_nulls: false })
                .with_alias(SERIES_VALUE_COUNTS_COUNT_NAME)])
            .map_err(|e| e.with_operation(operation))?;

        if value_name != column_name {
            counts = counts.rename_column(&column_name, &value_name).map_err(|e| e.with_operation(operation))?;
        }

        Ok(counts)
    }

    fn reduce_in_context(&self, agg: ReduceAgg, ctx: &ExecutionContext<'_>, operation: &'static str) -> Result<Scalar> {
        let output_domain = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let input_type = self.dtype().map_err(|e| e.with_operation(operation))?;
        let output_type = infer_reduce_output_type(&input_type, &agg).map_err(|e| e.with_operation(operation))?;
        let exec_options = output_domain.explicit_exec_options_raw();
        reduction::reduce_impl(&cv, &agg, &output_type, &output_domain, exec_options.as_ref(), operation)
    }

    fn minmax_in_context(&self, ctx: &ExecutionContext<'_>) -> Result<(Scalar, Scalar)> {
        let operation = "series::in_context::minmax";
        let output_domain = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let exec_options = output_domain.explicit_exec_options_raw();
        reduction::minmax_impl(&cv, &output_domain, exec_options.as_ref(), operation)
    }

    // ── Macro-generated unary helpers ───────────────────────────────────

    series_context_op!(unary fn is_nan_in_context(), "series::in_context::is_nan",
        |cv, output_domain, exec, op| unary::is_nan_impl(&cv, &output_domain, exec.as_ref(), op));

    series_context_op!(unary fn is_not_nan_in_context(), "series::in_context::is_not_nan",
        |cv, output_domain, exec, op| unary::is_not_nan_impl(&cv, &output_domain, exec.as_ref(), op));

    series_context_op!(unary fn logical_not_in_context(), "series::in_context::logical_not",
        |cv, output_domain, exec, op| unary::unary_operation_impl(&cv, UnaryOperator::Not, &output_domain, exec.as_ref(), op));

    series_context_op!(unary fn is_null_in_context(), "series::in_context::is_null",
        |cv, output_domain, exec, op| unary::is_null_impl(&cv, &output_domain, exec.as_ref(), op));

    series_context_op!(unary fn is_not_null_in_context(), "series::in_context::is_not_null",
        |cv, output_domain, exec, op| unary::is_valid_impl(&cv, &output_domain, exec.as_ref(), op));

    // Unary helpers with caller-supplied operation string
    series_context_op!(unary fn unary_in_context(op: UnaryOperator),
        caller_op |cv, output_domain, exec, operation| unary::unary_operation_impl(&cv, op, &output_domain, exec.as_ref(), operation));

    series_context_op!(unary fn scan_in_context(agg: ScanAgg, scan_type: ScanType, null_policy: NullPolicy),
        caller_op |cv, output_domain, exec, operation| reduction::scan_impl(&cv, agg, scan_type, null_policy, &output_domain, exec.as_ref(), operation));

    series_context_op!(unary fn string_unary_in_context(
        op: fn(
            &crate::core::column::ColumnView<'_>,
            &Arc<ExecutionDomain>,
            Option<&cudf_sys::cudf_exec_options_t>,
            &'static str,
        ) -> Result<crate::core::column::Column>),
        caller_op |cv, output_domain, exec, operation| op(&cv, &output_domain, exec.as_ref(), operation));

    fn strip_in_context(
        &self,
        side: StripSide,
        to_strip: &str,
        ctx: &ExecutionContext<'_>,
        operation: &'static str,
    ) -> Result<Series> {
        let output_domain = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let scalar = Scalar::from_str_in_domain(to_strip, &output_domain)?;
        let exec_options = output_domain.explicit_exec_options_raw();
        let col = strings::strip_impl(&cv, side, &scalar, &output_domain, exec_options.as_ref(), operation)?;
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
            &Arc<ExecutionDomain>,
            Option<&cudf_sys::cudf_exec_options_t>,
            &'static str,
        ) -> Result<crate::core::column::Column>,
    ) -> Result<Series> {
        let output_domain = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let scalar = Scalar::from_str_in_domain(target, &output_domain)?;
        let exec_options = output_domain.explicit_exec_options_raw();
        let col = op(&cv, &scalar, &output_domain, exec_options.as_ref(), operation)?;
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
        let col = unary::cast_impl(&cv, &target, &output_domain, exec_options.as_ref(), operation)?;
        self.with_inherited_name(col)
    }

    // ── Fillna with scalar (generic over T: CudfType) ──────────────────

    fn fillna_in_context<T: CudfType>(&self, value: T, ctx: &ExecutionContext<'_>) -> Result<Series> {
        let operation = "series::in_context::fillna";
        let output_domain = self.resolve_context_output_domain(ctx, operation)?;
        let cv = self.column.view()?;
        let scalar = Scalar::from_value_in_domain(value, &output_domain)?;
        let exec_options = output_domain.explicit_exec_options_raw();
        let col =
            replace::replace_nulls_with_exec_options(&cv, &scalar, &output_domain, exec_options.as_ref(), operation)?;
        self.with_inherited_name(col)
    }

    // ── Macro-generated binary helpers ──────────────────────────────────

    series_context_op!(binary fn fillna_with_series_in_context(other: &Series),
        "series::in_context::fillna_with_series",
        |cv, other_cv, output_domain, exec, op| replace::replace_nulls_column_with_exec_options(&cv, &other_cv, &output_domain, exec.as_ref(), op));

    series_context_op!(binary fn isin_in_context(values: &Series),
        "series::in_context::isin",
        |cv, values_cv, output_domain, exec, op| crate::ops::search::contains_impl(&values_cv, &cv, &output_domain, exec.as_ref(), op));

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
        let col = binary::binary_operation_vv_impl(
            &lhs,
            &rhs,
            op,
            &output_type,
            &output_domain,
            exec_options.as_ref(),
            operation,
        )?;
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
        let col = binary::binary_operation_vs_impl(
            &cv,
            &scalar,
            op,
            &output_type,
            &output_domain,
            exec_options.as_ref(),
            operation,
        )?;
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
        let scalar = Scalar::from_str_in_domain(value, &output_domain)?;
        let output_type = binary::infer_binary_output_type(cv.data_type()?.id(), scalar.data_type()?.id(), op)?;
        let exec_options = output_domain.explicit_exec_options_raw();
        let col = binary::binary_operation_vs_impl(
            &cv,
            &scalar,
            op,
            &output_type,
            &output_domain,
            exec_options.as_ref(),
            operation,
        )?;
        self.with_inherited_name(col)
    }
}

impl<'series, 'ctx, 'handles> SeriesExecutionScope<'series, 'ctx, 'handles> {
    /// Enter the string namespace for string-specific scoped operations.
    pub fn str(&self) -> SeriesExecutionScopeStringNamespace<'series, 'ctx, 'handles> {
        SeriesExecutionScopeStringNamespace { series: self.series, ctx: self.ctx }
    }

    series_scope_forward!(
        /// Return the first `n` rows using the borrowed execution override.
        fn head(n: usize) -> Result<Series> => head_in_context;

        /// Return the last `n` rows using the borrowed execution override.
        fn tail(n: usize) -> Result<Series> => tail_in_context;

        /// Return `length` rows starting at `offset` using the borrowed execution override.
        fn slice(offset: usize, length: usize) -> Result<Series> => slice_in_context;

        /// Gather rows using the borrowed execution override.
        fn gather(indices: &Series) -> Result<Series> => gather_in_context;

        /// Filter rows using the borrowed execution override.
        fn filter(mask: &Series) -> Result<Series> => filter_in_context;

        /// Sort using the borrowed execution override.
        fn sort_values(options: SortOptions) -> Result<Series> => sort_values_in_context;

        /// Drop null rows using the borrowed execution override.
        fn dropna() -> Result<Series> => dropna_in_context;

        /// Remove duplicate rows using the borrowed execution override.
        fn drop_duplicates() -> Result<Series> => drop_duplicates_in_context;

        /// Count the frequency of each distinct non-null value using the borrowed execution override.
        fn value_counts() -> Result<crate::DataFrame> => value_counts_in_context;

        /// Replace null values using another series with the borrowed execution override.
        fn fillna_with_series(other: &Series) -> Result<Series> => fillna_with_series_in_context;

        /// Reduce to a min/max pair using the borrowed execution override.
        fn minmax() -> Result<(Scalar, Scalar)> => minmax_in_context;

        /// Convert numeric values to timestamps using the borrowed execution override.
        fn to_datetime(resolution: Option<TimestampResolution>) -> Result<Series> => to_datetime_in_context;

        /// Element-wise membership test using the borrowed execution override.
        fn isin(values: &Series) -> Result<Series> => isin_in_context;

        /// Remove duplicates using the borrowed execution override.
        fn unique() -> Result<Series> => unique_in_context;

        /// Build a reusable codebook using the borrowed execution override.
        fn codebook() -> Result<Codebook> => codebook_in_context;

        /// Factorize using the borrowed execution override.
        fn factorize() -> Result<Factorized> => factorize_in_context;
    );

    series_scope_forward_generic!(
        /// Replace null values with a scalar using the borrowed execution override.
        fn fillna<T: CudfType>(value: T) -> Result<Series> => fillna_in_context;
    );

    series_scope_fillna_policy!(
        /// Forward fill using the borrowed execution override.
        fn ffill() => FillPolicy::Preceding;

        /// Backward fill using the borrowed execution override.
        fn bfill() => FillPolicy::Following;
    );

    series_scope_reduce!(
        /// Reduce to a scalar sum using the borrowed execution override.
        fn sum() => ReduceAgg::Sum, "series::in_context::sum";

        /// Reduce to a scalar minimum using the borrowed execution override.
        fn min() => ReduceAgg::Min, "series::in_context::min";

        /// Reduce to a scalar maximum using the borrowed execution override.
        fn max() => ReduceAgg::Max, "series::in_context::max";

        /// Reduce to a scalar mean using the borrowed execution override.
        fn mean() => ReduceAgg::Mean, "series::in_context::mean";

        /// Reduce to a scalar median using the borrowed execution override.
        fn median() => ReduceAgg::Median, "series::in_context::median";

        /// Reduce to a scalar standard deviation using the borrowed execution override.
        fn std(ddof: u32) => ReduceAgg::Std { ddof }, "series::in_context::std";

        /// Reduce to a scalar variance using the borrowed execution override.
        fn var(ddof: u32) => ReduceAgg::Variance { ddof }, "series::in_context::var";

        /// Reduce to a scalar logical-any using the borrowed execution override.
        fn any() => ReduceAgg::Any, "series::in_context::any";

        /// Reduce to a scalar logical-all using the borrowed execution override.
        fn all() => ReduceAgg::All, "series::in_context::all";

        /// Reduce to a scalar product using the borrowed execution override.
        fn product() => ReduceAgg::Product, "series::in_context::product";
    );

    /// Reduce to a host sum value using the borrowed execution override.
    pub fn sum_value<T: CudfType>(&self) -> Result<T> {
        self.sum()?.value::<T>()
    }

    /// Reduce to a host minimum value using the borrowed execution override.
    pub fn min_value<T: CudfType>(&self) -> Result<Option<T>> {
        scalar_to_optional_value(self.min()?)
    }

    /// Reduce to a host maximum value using the borrowed execution override.
    pub fn max_value<T: CudfType>(&self) -> Result<Option<T>> {
        scalar_to_optional_value(self.max()?)
    }

    /// Reduce to a host mean value using the borrowed execution override.
    pub fn mean_value(&self) -> Result<Option<f64>> {
        scalar_to_optional_value(self.mean()?)
    }

    /// Reduce to a host median value using the borrowed execution override.
    pub fn median_value(&self) -> Result<Option<f64>> {
        scalar_to_optional_value(self.median()?)
    }

    /// Reduce to a host standard deviation value using the borrowed execution override.
    pub fn std_value(&self, ddof: u32) -> Result<Option<f64>> {
        scalar_to_optional_value(self.std(ddof)?)
    }

    /// Reduce to a host variance value using the borrowed execution override.
    pub fn var_value(&self, ddof: u32) -> Result<Option<f64>> {
        scalar_to_optional_value(self.var(ddof)?)
    }

    /// Reduce to a host bool logical-any using the borrowed execution override.
    pub fn any_value(&self) -> Result<bool> {
        Ok(scalar_to_optional_value(self.any()?)?.unwrap_or(false))
    }

    /// Reduce to a host bool logical-all using the borrowed execution override.
    pub fn all_value(&self) -> Result<bool> {
        Ok(scalar_to_optional_value(self.all()?)?.unwrap_or(true))
    }

    /// Reduce to a host product value using the borrowed execution override.
    pub fn product_value<T: CudfType>(&self) -> Result<Option<T>> {
        scalar_to_optional_value(self.product()?)
    }

    /// Reduce to host min/max values using the borrowed execution override.
    pub fn minmax_values<T: CudfType>(&self) -> Result<(Option<T>, Option<T>)> {
        let (min, max) = self.minmax()?;
        Ok((scalar_to_optional_value(min)?, scalar_to_optional_value(max)?))
    }

    series_scope_scan!(
        /// Inclusive cumulative sum using the borrowed execution override.
        fn cumsum() => ScanAgg::Sum, "series::in_context::cumsum";

        /// Inclusive cumulative minimum using the borrowed execution override.
        fn cummin() => ScanAgg::Min, "series::in_context::cummin";

        /// Inclusive cumulative maximum using the borrowed execution override.
        fn cummax() => ScanAgg::Max, "series::in_context::cummax";

        /// Inclusive cumulative product using the borrowed execution override.
        fn cumprod() => ScanAgg::Product, "series::in_context::cumprod";
    );

    series_scope_unary!(
        /// Element-wise NaN check using the borrowed execution override.
        fn is_nan() => is_nan_in_context;

        /// Element-wise not-NaN check using the borrowed execution override.
        fn is_not_nan() => is_not_nan_in_context;

        /// Element-wise logical NOT using the borrowed execution override.
        fn logical_not() => logical_not_in_context;

        /// Return a BOOL8 mask: `true` where elements are null.
        fn is_null() => is_null_in_context;

        /// Return a BOOL8 mask: `true` where elements are valid.
        fn is_not_null() => is_not_null_in_context;
    );

    /// Element-wise absolute value using the borrowed execution override.
    pub fn abs(&self) -> Result<Series> {
        self.series.unary_in_context(UnaryOperator::Abs, self.ctx, "series::in_context::abs")
    }

    /// Element-wise arithmetic negation using the borrowed execution override.
    pub fn neg(&self) -> Result<Series> {
        self.series.unary_in_context(UnaryOperator::Negate, self.ctx, "series::in_context::neg")
    }

    /// Element-wise logical AND using the borrowed execution override.
    pub fn logical_and(&self, rhs: &Series) -> Result<Series> {
        let operation = "series::in_context::logical_and";
        self.series.require_bool_binary_operands(rhs, operation)?;
        self.series.binary_series_in_context(rhs, BinaryOperator::LogicalAnd, self.ctx, operation)
    }

    /// Element-wise logical OR using the borrowed execution override.
    pub fn logical_or(&self, rhs: &Series) -> Result<Series> {
        let operation = "series::in_context::logical_or";
        self.series.require_bool_binary_operands(rhs, operation)?;
        self.series.binary_series_in_context(rhs, BinaryOperator::LogicalOr, self.ctx, operation)
    }

    /// Cast using the borrowed execution override.
    pub fn astype(&self, target: impl Into<DataType>) -> Result<Series> {
        self.series.cast_in_context(target, self.ctx, "series::in_context::astype")
    }

    /// Cast to a numeric type using the borrowed execution override.
    pub fn to_numeric(&self, target: TypeId) -> Result<Series> {
        if !target.is_numeric() && !target.is_boolean() {
            return Err(Error::new(ErrorKind::NotSupported, format!("target type {target:?} is not numeric"))
                .with_operation("series::in_context::to_numeric"));
        }
        self.series.cast_in_context(target, self.ctx, "series::in_context::to_numeric")
    }

    series_scope_binary_scalar!(
        /// Element-wise equality-to-scalar using the borrowed execution override.
        fn eq_scalar<T: CudfType>(rhs: T) => BinaryOperator::Equal, "series::in_context::eq_scalar";

        /// Element-wise inequality-to-scalar using the borrowed execution override.
        fn ne_scalar<T: CudfType>(rhs: T) => BinaryOperator::NotEqual, "series::in_context::ne_scalar";

        /// Element-wise greater-than-scalar using the borrowed execution override.
        fn gt_scalar<T: CudfType>(rhs: T) => BinaryOperator::Greater, "series::in_context::gt_scalar";

        /// Element-wise less-than-scalar using the borrowed execution override.
        fn lt_scalar<T: CudfType>(rhs: T) => BinaryOperator::Less, "series::in_context::lt_scalar";

        /// Element-wise greater-than-or-equal-to-scalar using the borrowed execution override.
        fn ge_scalar<T: CudfType>(rhs: T) => BinaryOperator::GreaterEqual, "series::in_context::ge_scalar";

        /// Element-wise less-than-or-equal-to-scalar using the borrowed execution override.
        fn le_scalar<T: CudfType>(rhs: T) => BinaryOperator::LessEqual, "series::in_context::le_scalar";

        /// Element-wise addition-by-scalar using the borrowed execution override.
        fn add_scalar<T: CudfType>(rhs: T) => BinaryOperator::Add, "series::in_context::add_scalar";

        /// Element-wise subtraction-by-scalar using the borrowed execution override.
        fn sub_scalar<T: CudfType>(rhs: T) => BinaryOperator::Sub, "series::in_context::sub_scalar";

        /// Element-wise multiplication-by-scalar using the borrowed execution override.
        fn mul_scalar<T: CudfType>(rhs: T) => BinaryOperator::Mul, "series::in_context::mul_scalar";

        /// Element-wise division-by-scalar using the borrowed execution override.
        fn div_scalar<T: CudfType>(rhs: T) => BinaryOperator::Div, "series::in_context::div_scalar";

        /// Element-wise modulo-by-scalar using the borrowed execution override.
        fn rem_scalar<T: CudfType>(rhs: T) => BinaryOperator::Mod, "series::in_context::rem_scalar";
    );

    series_scope_binary_series!(
        /// Element-wise equality using the borrowed execution override.
        fn eq(rhs: &Series) => BinaryOperator::Equal, "series::in_context::eq";

        /// Element-wise inequality using the borrowed execution override.
        fn ne(rhs: &Series) => BinaryOperator::NotEqual, "series::in_context::ne";

        /// Element-wise greater-than using the borrowed execution override.
        fn gt(rhs: &Series) => BinaryOperator::Greater, "series::in_context::gt";

        /// Element-wise less-than using the borrowed execution override.
        fn lt(rhs: &Series) => BinaryOperator::Less, "series::in_context::lt";

        /// Element-wise greater-than-or-equal using the borrowed execution override.
        fn ge(rhs: &Series) => BinaryOperator::GreaterEqual, "series::in_context::ge";

        /// Element-wise less-than-or-equal using the borrowed execution override.
        fn le(rhs: &Series) => BinaryOperator::LessEqual, "series::in_context::le";

        /// Element-wise addition using the borrowed execution override.
        fn add(rhs: &Series) => BinaryOperator::Add, "series::in_context::add";

        /// Element-wise subtraction using the borrowed execution override.
        fn sub(rhs: &Series) => BinaryOperator::Sub, "series::in_context::sub";

        /// Element-wise multiplication using the borrowed execution override.
        fn mul(rhs: &Series) => BinaryOperator::Mul, "series::in_context::mul";

        /// Element-wise division using the borrowed execution override.
        fn div(rhs: &Series) => BinaryOperator::Div, "series::in_context::div";

        /// Element-wise modulo using the borrowed execution override.
        fn rem(rhs: &Series) => BinaryOperator::Mod, "series::in_context::rem";
    );

    /// Count distinct non-null values using the borrowed execution override.
    pub fn nunique(&self) -> Result<usize> {
        let unique = self.unique()?;
        Ok(unique.len() - unique.null_count())
    }

    series_scope_binary_string_scalar!(
        /// Equality-to-string-scalar using the borrowed execution override.
        fn eq_scalar_str(rhs: &str) => BinaryOperator::Equal, "series::in_context::eq_scalar_str";

        /// Inequality-to-string-scalar using the borrowed execution override.
        fn ne_scalar_str(rhs: &str) => BinaryOperator::NotEqual, "series::in_context::ne_scalar_str";

        /// Greater-than-string-scalar using the borrowed execution override.
        fn gt_scalar_str(rhs: &str) => BinaryOperator::Greater, "series::in_context::gt_scalar_str";

        /// Less-than-string-scalar using the borrowed execution override.
        fn lt_scalar_str(rhs: &str) => BinaryOperator::Less, "series::in_context::lt_scalar_str";

        /// Greater-than-or-equal-to-string-scalar using the borrowed execution override.
        fn ge_scalar_str(rhs: &str) => BinaryOperator::GreaterEqual, "series::in_context::ge_scalar_str";

        /// Less-than-or-equal-to-string-scalar using the borrowed execution override.
        fn le_scalar_str(rhs: &str) => BinaryOperator::LessEqual, "series::in_context::le_scalar_str";
    );
}

impl SeriesExecutionScopeStringNamespace<'_, '_, '_> {
    series_scope_string_unary!(
        /// Lowercase strings using the borrowed execution override.
        fn lower() => strings::to_lower_impl, "series::in_context::str::lower";

        /// Uppercase strings using the borrowed execution override.
        fn upper() => strings::to_upper_impl, "series::in_context::str::upper";

        /// Character counts using the borrowed execution override.
        fn len() => strings::count_characters_impl, "series::in_context::str::len";

        /// Byte counts using the borrowed execution override.
        fn byte_count() => strings::count_bytes_impl, "series::in_context::str::byte_count";
    );

    series_scope_string_scalar!(
        /// Literal substring containment using the borrowed execution override.
        fn contains(pat: &str) => strings::str_contains_impl, "series::in_context::str::contains";

        /// Starts-with test using the borrowed execution override.
        fn starts_with(pat: &str) => strings::starts_with_impl, "series::in_context::str::starts_with";

        /// Ends-with test using the borrowed execution override.
        fn ends_with(pat: &str) => strings::ends_with_impl, "series::in_context::str::ends_with";
    );

    /// Strip whitespace using the borrowed execution override.
    pub fn strip(&self) -> Result<Series> {
        self.series.strip_in_context(StripSide::Both, "", self.ctx, "series::in_context::str::strip")
    }

    /// Strip leading whitespace using the borrowed execution override.
    pub fn lstrip(&self) -> Result<Series> {
        self.series.strip_in_context(StripSide::Left, "", self.ctx, "series::in_context::str::lstrip")
    }

    /// Strip trailing whitespace using the borrowed execution override.
    pub fn rstrip(&self) -> Result<Series> {
        self.series.strip_in_context(StripSide::Right, "", self.ctx, "series::in_context::str::rstrip")
    }

    /// Strip specific characters using the borrowed execution override.
    pub fn strip_chars(&self, side: StripSide, to_strip: &str) -> Result<Series> {
        self.series.strip_in_context(side, to_strip, self.ctx, "series::in_context::str::strip_chars")
    }
}
