//! High-level DataFrame groupby API.

use std::sync::Arc;

use smallvec::SmallVec;
use typed_builder::TypedBuilder;

use super::{
    agg::{AggregationRequest, GroupByAgg, NullPolicy, groupby_aggregate_impl},
    errors::unsupported_groupby_validation_error,
    output::take_groupby_result_columns_for_output,
};
use crate::{
    core::{
        column::{Column, ColumnView},
        error::Result,
    },
    frame::dataframe::{DataFrame, group_keys::require_non_empty_group_keys},
    internal::resolve::missing_column_error,
    runtime::{execution_context::ExecutionOverrides, execution_domain::ExecutionDomain},
};

const GROUPBY_VALUE_VIEW_INLINE_CAPACITY: usize = 8;
const GROUPBY_REQUEST_META_INLINE_CAPACITY: usize = 8;
const GROUPBY_RESULT_NAME_INLINE_CAPACITY: usize = 8;
const GROUPBY_OUTPUT_ORDER_INLINE_CAPACITY: usize = 8;
const GROUPBY_AGG_SLICE_INLINE_CAPACITY: usize = 8;
const GROUPBY_REQUEST_INLINE_CAPACITY: usize = 8;

macro_rules! groupby_shortcuts {
    ($(#[$meta:meta])* fn $method:ident($columns:ident) => $agg:expr;) => {
        $(#[$meta])*
        pub fn $method<I, S>(&self, $columns: I) -> Result<DataFrame>
        where
            I: IntoIterator<Item = S>,
            S: AsRef<str>,
        {
            let specs: Vec<AggSpec> = $columns.into_iter().map(|col| AggSpec::new(col.as_ref(), $agg)).collect();
            self.agg(&specs)
        }
    };
    ($( $(#[$meta:meta])* fn $method:ident($columns:ident) => $agg:expr; )*) => {
        $(
            groupby_shortcuts!(
                $(#[$meta])*
                fn $method($columns) => $agg;
            );
        )*
    };
}

impl GroupByAgg {
    /// Return the default output column suffix for this aggregation.
    fn suffix(&self) -> &'static str {
        match self {
            GroupByAgg::Sum => "sum",
            GroupByAgg::Min => "min",
            GroupByAgg::Max => "max",
            GroupByAgg::Mean => "mean",
            GroupByAgg::Count { .. } => "count",
            GroupByAgg::Median => "median",
            GroupByAgg::NUnique => "nunique",
            GroupByAgg::Std { .. } => "std",
            GroupByAgg::Variance { .. } => "var",
        }
    }
}

/// A single aggregation specification: which column, which aggregation,
/// and an optional output alias.
///
/// If no alias is set the output column is named `"{column}_{agg}"`,
/// e.g. `"price_sum"`.
#[derive(Debug, Clone)]
pub struct AggSpec {
    column: String,
    agg: GroupByAgg,
    alias: Option<String>,
}

impl AggSpec {
    /// Create a new aggregation specification for `column` with `agg`.
    pub fn new(column: impl Into<String>, agg: GroupByAgg) -> Self {
        Self { column: column.into(), agg, alias: None }
    }

    /// Set an output alias for this aggregation result column.
    pub fn with_alias(mut self, alias: impl Into<String>) -> Self {
        self.alias = Some(alias.into());
        self
    }

    /// The output column name: alias if set, otherwise `"{column}_{suffix}"`.
    fn output_name(&self) -> String {
        match &self.alias {
            Some(alias) => alias.clone(),
            None => format!("{}_{}", self.column, self.agg.suffix()),
        }
    }
}

/// Options for the high-level groupby operation.
#[derive(Debug, Clone, TypedBuilder)]
pub struct GroupByOptions {
    #[builder(default)]
    include_null_keys: bool,
}

impl Default for GroupByOptions {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl GroupByOptions {
    pub(crate) fn null_policy(&self) -> NullPolicy {
        if self.include_null_keys { NullPolicy::Include } else { NullPolicy::Exclude }
    }
}

/// A pending groupby operation on a [`DataFrame`][crate::DataFrame].
///
/// Created via [`DataFrame::groupby`][crate::DataFrame::groupby].
/// Call [`.agg()`][GroupBy::agg] with [`AggSpec`] items, or use the shortcut
/// methods [`.sum()`][GroupBy::sum], [`.mean()`][GroupBy::mean],
/// [`.count()`][GroupBy::count] to execute.
pub struct GroupBy<'a> {
    df: &'a DataFrame,
    keys: Vec<String>,
    options: GroupByOptions,
}

/// Scoped borrowed execution override for one groupby operation chain.
#[must_use]
pub struct GroupByExecutionScope<'a, 'ctx, 'handles> {
    groupby: GroupBy<'a>,
    ctx: &'ctx ExecutionOverrides<'handles>,
}

impl<'a> GroupBy<'a> {
    /// Create a new `GroupBy` for the given DataFrame and key column names.
    pub(crate) fn new(df: &'a DataFrame, keys: Vec<String>) -> Self {
        Self { df, keys, options: GroupByOptions::default() }
    }

    /// Override the default groupby options.
    pub fn options(mut self, options: GroupByOptions) -> Self {
        self.options = options;
        self
    }

    /// Borrow a per-operation execution override for subsequent groupby calls.
    pub fn with_execution<'ctx, 'handles>(
        self,
        ctx: &'ctx ExecutionOverrides<'handles>,
    ) -> GroupByExecutionScope<'a, 'ctx, 'handles> {
        GroupByExecutionScope { groupby: self, ctx }
    }

    /// Execute the groupby with the given aggregation specifications.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `specs` is empty
    /// - Any key or value column name is not in the DataFrame's schema
    pub fn agg(&self, specs: &[AggSpec]) -> Result<DataFrame> {
        let output_domain = Arc::clone(self.df.execution_domain());
        self.agg_impl(specs, None, output_domain, "groupby::GroupBy::agg")
    }

    /// Execute the groupby with a borrowed per-operation execution override.
    pub(crate) fn agg_in_context(&self, specs: &[AggSpec], ctx: &ExecutionOverrides<'_>) -> Result<DataFrame> {
        let output_domain = ctx.resolve_output_domain(self.df.execution_domain())?;
        self.agg_impl(
            specs,
            output_domain.explicit_exec_options_raw(),
            output_domain,
            "groupby::GroupBy::with_execution::agg",
        )
    }
    pub(crate) fn __debug_agg_in_context(&self, specs: &[AggSpec], ctx: &ExecutionOverrides<'_>) -> Result<DataFrame> {
        self.agg_in_context(specs, ctx)
    }

    fn agg_impl(
        &self,
        specs: &[AggSpec],
        exec_options: Option<cudf_sys::cudf_exec_options_t>,
        output_domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<DataFrame> {
        use crate::internal::{
            hash::{FastHashMap, fast_hash_map_with_capacity},
            resolve::{build_borrowed_table_view_from_columns, resolve_column_names},
        };

        ExecutionDomain::insert_stream_dependency(self.df.execution_domain(), &output_domain, operation)?;

        require_non_empty_group_keys(self.keys.len(), operation)?;

        if specs.is_empty() {
            return Err(unsupported_groupby_validation_error(
                operation,
                "at least one aggregation spec is required",
                "empty_agg_specs",
                "provide at least one AggSpec or use a shortcut like .sum([...])",
            ));
        }

        let columns = self.df.columns_map();

        let key_refs: SmallVec<[&str; GROUPBY_REQUEST_META_INLINE_CAPACITY]> =
            self.keys.iter().map(|s| s.as_str()).collect();
        let key_indices = resolve_column_names(columns, &key_refs, operation)?;
        drop(key_refs);

        for spec in specs {
            if !columns.contains_key(spec.column.as_str()) {
                return Err(missing_column_error(operation, &spec.column));
            }
        }

        let keys_table = build_borrowed_table_view_from_columns(columns, &key_indices)?;

        let mut value_views: SmallVec<[ColumnView<'_>; GROUPBY_VALUE_VIEW_INLINE_CAPACITY]> = SmallVec::new();
        let mut request_meta: SmallVec<[(usize, Vec<GroupByAgg>); GROUPBY_REQUEST_META_INLINE_CAPACITY]> =
            SmallVec::new();
        let mut result_names: SmallVec<[String; GROUPBY_RESULT_NAME_INLINE_CAPACITY]> = SmallVec::new();

        let mut col_name_to_view_idx: FastHashMap<String, usize> = fast_hash_map_with_capacity(specs.len());
        let mut view_idx_to_group_idx: FastHashMap<usize, usize> = fast_hash_map_with_capacity(specs.len());
        let mut output_order: SmallVec<[(usize, usize); GROUPBY_OUTPUT_ORDER_INLINE_CAPACITY]> = SmallVec::new();

        for spec in specs {
            let view_idx = if let Some(&idx) = col_name_to_view_idx.get(spec.column.as_str()) {
                idx
            } else {
                let arc_col =
                    columns.get(spec.column.as_str()).ok_or_else(|| missing_column_error(operation, &spec.column))?;
                let idx = value_views.len();
                value_views.push(arc_col.view()?);
                col_name_to_view_idx.insert(spec.column.clone(), idx);
                idx
            };

            let group_idx = if let Some(&idx) = view_idx_to_group_idx.get(&view_idx) {
                idx
            } else {
                let idx = request_meta.len();
                request_meta.push((view_idx, Vec::new()));
                view_idx_to_group_idx.insert(view_idx, idx);
                idx
            };

            let agg_idx_within_group = request_meta[group_idx].1.len();
            request_meta[group_idx].1.push(spec.agg.clone());
            output_order.push((group_idx, agg_idx_within_group));

            result_names.push(spec.output_name());
        }

        let agg_slices: SmallVec<[Vec<GroupByAgg>; GROUPBY_AGG_SLICE_INLINE_CAPACITY]> =
            request_meta.iter().map(|(_, aggs)| aggs.clone()).collect();

        let requests: SmallVec<[AggregationRequest<'_>; GROUPBY_REQUEST_INLINE_CAPACITY]> = request_meta
            .iter()
            .enumerate()
            .map(|(i, (view_idx, _))| AggregationRequest {
                values: &value_views[*view_idx],
                aggregations: &agg_slices[i],
            })
            .collect();

        let gb_result = if let Some(exec_options) = exec_options.as_ref() {
            groupby_aggregate_impl(
                keys_table.table(),
                &requests,
                self.options.null_policy(),
                &output_domain,
                Some(exec_options),
                operation,
            )?
        } else {
            let exec_options = output_domain.exec_options_raw();
            groupby_aggregate_impl(
                keys_table.table(),
                &requests,
                self.options.null_policy(),
                &output_domain,
                Some(&exec_options),
                operation,
            )?
        };

        let grouped_results: Vec<Vec<Option<Column>>> =
            gb_result.results.into_iter().map(|cols| cols.into_iter().map(Some).collect()).collect();
        let all_result_cols = take_groupby_result_columns_for_output(grouped_results, &output_order, operation)?;

        let result_keys_cols = gb_result.keys.into_columns()?;

        let mut final_cols: Vec<Column> = Vec::with_capacity(result_keys_cols.len() + all_result_cols.len());
        final_cols.extend(result_keys_cols);
        final_cols.extend(all_result_cols);

        let mut schema_names: Vec<String> = Vec::with_capacity(self.keys.len() + result_names.len());
        schema_names.extend(self.keys.iter().cloned());
        schema_names.extend(result_names);

        DataFrame::new(schema_names, final_cols)
    }

    groupby_shortcuts!(
        /// Shortcut: compute the sum of each named column per group.
        fn sum(columns) => GroupByAgg::Sum;

        /// Shortcut: compute the mean of each named column per group.
        fn mean(columns) => GroupByAgg::Mean;

        /// Shortcut: count non-null values of each named column per group.
        fn count(columns) => GroupByAgg::Count { include_nulls: false };

        /// Shortcut: compute the median of each named column per group.
        fn median(columns) => GroupByAgg::Median;

        /// Shortcut: count unique non-null values of each named column per group.
        fn nunique(columns) => GroupByAgg::NUnique;

        /// Shortcut: compute the sample standard deviation (ddof=1) of each named column per group.
        fn std(columns) => GroupByAgg::Std { ddof: 1 };

        /// Shortcut: compute the sample variance (ddof=1) of each named column per group.
        fn var(columns) => GroupByAgg::Variance { ddof: 1 };
    );
}

impl<'a, 'ctx, 'handles> GroupByExecutionScope<'a, 'ctx, 'handles> {
    /// Execute the groupby with the given aggregation specifications.
    pub fn agg(&self, specs: &[AggSpec]) -> Result<DataFrame> {
        self.groupby.agg_in_context(specs, self.ctx)
    }

    groupby_shortcuts!(
        /// Shortcut: compute the sum of each named column per group.
        fn sum(columns) => GroupByAgg::Sum;

        /// Shortcut: compute the mean of each named column per group.
        fn mean(columns) => GroupByAgg::Mean;

        /// Shortcut: count non-null values of each named column per group.
        fn count(columns) => GroupByAgg::Count { include_nulls: false };

        /// Shortcut: compute the median of each named column per group.
        fn median(columns) => GroupByAgg::Median;

        /// Shortcut: count unique non-null values of each named column per group.
        fn nunique(columns) => GroupByAgg::NUnique;

        /// Shortcut: compute the sample standard deviation (ddof=1) of each named column per group.
        fn std(columns) => GroupByAgg::Std { ddof: 1 };

        /// Shortcut: compute the sample variance (ddof=1) of each named column per group.
        fn var(columns) => GroupByAgg::Variance { ddof: 1 };
    );
}
