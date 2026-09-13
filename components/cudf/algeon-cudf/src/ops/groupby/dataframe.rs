//! High-level DataFrame groupby API.

use std::{ptr, ptr::NonNull, sync::Arc};

use smallvec::SmallVec;
use typed_builder::TypedBuilder;

use super::{
    agg::{
        AggregationRequest, GroupByAgg, NullPolicy, PreparedGroupBy, PreparedGroupByFacts, groupby_aggregate_impl,
        groupby_aggregate_impl_with_prepared, prepared_groupby_facts_raw,
    },
    errors::unsupported_groupby_validation_error,
    output::take_groupby_result_columns_for_output,
    shift::{GroupByShiftRequest, groupby_shift_impl},
};
use crate::{
    core::{
        column::{Column, ColumnView},
        error::{self, Error, ErrorKind, Result, invariant_violation},
        native_teardown::drop_device_owned_handle,
        scalar::Scalar,
    },
    frame::dataframe::{DataFrame, InfoDetail, group_keys::require_non_empty_group_keys},
    internal::resolve::missing_column_error,
    runtime::{
        BudgetedExecutionOutcome, ExecutionBudgetToken, execution_context::ExecutionOverrides,
        execution_domain::ExecutionDomain, execution_resolver::ResolvedExecution,
    },
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

    /// Input column to aggregate.
    pub fn input_column(&self) -> &str {
        &self.column
    }

    /// Aggregation function and parameters.
    pub fn agg(&self) -> &GroupByAgg {
        &self.agg
    }

    /// Explicit output alias, if one was provided.
    pub fn alias(&self) -> Option<&str> {
        self.alias.as_deref()
    }

    /// The output column name: alias if set, otherwise `"{column}_{suffix}"`.
    pub fn resolved_output_name(&self) -> String {
        match &self.alias {
            Some(alias) => alias.clone(),
            None => format!("{}_{}", self.column, self.agg.suffix()),
        }
    }
}

/// Options for the high-level groupby operation.
#[derive(Debug, Clone, TypedBuilder)]
pub struct GroupByOptions {
    /// Include null key rows as their own groups. Defaults to `false`, which
    /// excludes rows containing null group keys.
    #[builder(default)]
    include_null_keys: bool,
    /// Assert that every key column is already sorted ascending with nulls
    /// after non-null values. Defaults to `false`; setting this incorrectly
    /// violates cuDF's sorted-key groupby contract.
    #[builder(default)]
    keys_are_sorted: bool,
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

    pub(crate) fn keys_are_sorted(&self) -> bool {
        self.keys_are_sorted
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

/// Owned high-level prepared groupby state for a DataFrame.
///
/// The value retains the input [`DataFrame`] so the key columns borrowed by
/// the underlying cuDF groupby handle remain alive. Preparing this value calls
/// cuDF's sort-backed `get_groups` path; callers should treat it as an
/// explicit `prepared_sort_groupby` strategy, not as a free diagnostic probe.
pub struct PreparedDataFrameGroupBy {
    df: DataFrame,
    keys: Vec<String>,
    options: GroupByOptions,
    handle: NonNull<cudf_sys::cudf_prepared_groupby_handle_state>,
    keys_domain: Arc<ExecutionDomain>,
    prepared_facts: PreparedGroupByFacts,
}

impl PreparedDataFrameGroupBy {
    /// Facts observed when this prepared groupby handle was created.
    pub fn prepared_facts(&self) -> PreparedGroupByFacts {
        self.prepared_facts
    }

    /// Re-run cuDF grouping preparation and return direct grouped-key facts.
    pub fn prepare_groups(&self) -> Result<PreparedGroupByFacts> {
        prepared_groupby_facts_raw(
            self.handle.as_ptr(),
            &self.keys_domain,
            "groupby::PreparedDataFrameGroupBy::prepare_groups",
        )
    }

    /// Execute aggregation requests through this prepared groupby handle.
    pub fn agg(&self, specs: &[AggSpec]) -> Result<DataFrame> {
        let ctx = ExecutionOverrides::new();
        self.agg_impl_with_groupby_error_mapper(specs, &ctx, "groupby::PreparedDataFrameGroupBy::agg", |error| error)
    }

    /// Execute aggregation requests through this prepared groupby handle with a
    /// hard execution-budget token.
    pub fn agg_with_budget(
        &self,
        specs: &[AggSpec],
        token: &ExecutionBudgetToken,
    ) -> Result<BudgetedExecutionOutcome<DataFrame>> {
        let operation = "groupby::PreparedDataFrameGroupBy::agg_with_budget";
        let ctx = token.execution_overrides();
        let output = self
            .agg_impl_with_groupby_error_mapper(specs, &ctx, operation, |error| {
                annotate_groupby_budget_execution_error(error, token, operation)
            })
            .map_err(|error| error.reanchor_if_needed(operation))?;
        let output_bytes = dataframe_device_allocated_size_bytes(&output, operation)?;
        let output_rows = output.num_rows();
        let facts = self.prepared_facts();
        let report = token
            .reservation_enforced_report()
            .with_retained_bytes(output_bytes)
            .with_output_bytes(output_bytes)
            .with_output_rows(output_rows)
            .with_context_facts([
                ("prepared_groupby_strategy", "prepared_sort_groupby".to_string()),
                ("prepared_groupby_num_groups", facts.num_groups.to_string()),
                ("prepared_groupby_grouped_key_output_bytes", facts.grouped_key_output_bytes.to_string()),
                ("prepared_groupby_max_group_size", facts.max_group_size.to_string()),
            ]);

        Ok(BudgetedExecutionOutcome::new(output, report))
    }

    fn agg_impl_with_groupby_error_mapper<F>(
        &self,
        specs: &[AggSpec],
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
        map_groupby_error: F,
    ) -> Result<DataFrame>
    where
        F: FnOnce(Error) -> Error,
    {
        use crate::{
            hash::{FastHashMap, fast_hash_map_with_capacity},
            internal::resolve::resolve_column_names,
        };

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
        let _key_indices = resolve_column_names(columns, &key_refs, operation)?;

        for spec in specs {
            if !columns.contains_key(spec.column.as_str()) {
                return Err(missing_column_error(operation, &spec.column));
            }
        }

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

            result_names.push(spec.resolved_output_name());
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

        let execution = ResolvedExecution::many(
            &self.keys_domain,
            requests.iter().map(|request| request.values.execution_domain()),
            ctx,
            operation,
        )?;
        let output_domain = Arc::clone(execution.output_domain());

        let gb_result = groupby_aggregate_impl_with_prepared(
            self.handle.as_ptr(),
            &requests,
            self.options.null_policy(),
            self.options.keys_are_sorted(),
            &output_domain,
            execution.exec_options(),
            operation,
        )
        .map_err(map_groupby_error)?;

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
}

impl Drop for PreparedDataFrameGroupBy {
    fn drop(&mut self) {
        drop_device_owned_handle(
            self.handle.as_ptr(),
            self.keys_domain.device_id(),
            "PreparedDataFrameGroupBy::drop",
            |owned| unsafe { cudf_sys::cudf_prepared_groupby_destroy(owned) },
        );
    }
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
        self.agg_impl(specs, output_domain, "groupby::GroupBy::agg")
    }

    /// Prepare reusable grouping state and return cuDF-reported cardinality.
    ///
    /// This runs cuDF's grouping preparation for this DataFrame's key columns
    /// and returns only direct facts from that operation. It does not estimate
    /// temporary workspace or retained groupby-state memory.
    pub fn prepare_groups(&self) -> Result<PreparedGroupByFacts> {
        let ctx = ExecutionOverrides::new();
        self.prepare_groups_impl(&ctx, "groupby::GroupBy::prepare_groups")
    }

    /// Build explicit prepared sort-groupby state under a hard execution
    /// budget.
    ///
    /// This is a reusable execution strategy object, not a transparent
    /// diagnostic probe: construction calls cuDF's `get_groups({})` path to
    /// initialize sort-backed grouping state and records exact grouped-key
    /// output facts. The grouped-key output is released before this returns.
    pub fn prepare_sort_groupby_with_budget(
        &self,
        token: &ExecutionBudgetToken,
    ) -> Result<BudgetedExecutionOutcome<PreparedDataFrameGroupBy>> {
        let operation = "groupby::GroupBy::prepare_sort_groupby_with_budget";
        let ctx = token.execution_overrides();
        let prepared = self
            .prepare_sort_groupby_impl(&ctx, operation)
            .map_err(|error| annotate_groupby_budget_execution_error(error, token, operation))?;
        let facts = prepared.prepared_facts();
        let report = token.reservation_enforced_report().with_output_rows(facts.num_groups).with_context_facts([
            ("prepared_groupby_strategy", "prepared_sort_groupby".to_string()),
            ("prepared_groupby_num_groups", facts.num_groups.to_string()),
            ("prepared_groupby_grouped_key_output_bytes", facts.grouped_key_output_bytes.to_string()),
            ("prepared_groupby_max_group_size", facts.max_group_size.to_string()),
            ("prepared_groupby_memory_facts_status", "grouped_key_output_only_retained_state_unavailable".to_string()),
        ]);

        Ok(BudgetedExecutionOutcome::new(prepared, report))
    }

    /// Execute the groupby with a hard execution-budget token.
    ///
    /// This has the same aggregation semantics and validation as
    /// [`agg`](Self::agg), but routes execution allocations through the
    /// token's execution overrides. The returned report records the observed
    /// output DataFrame device footprint after execution.
    pub fn agg_with_budget(
        &self,
        specs: &[AggSpec],
        token: &ExecutionBudgetToken,
    ) -> Result<BudgetedExecutionOutcome<DataFrame>> {
        let operation = "groupby::GroupBy::agg_with_budget";
        let ctx = token.execution_overrides();
        let output_domain = ctx
            .resolve_output_domain(self.df.execution_domain())
            .map_err(|error| error.reanchor_if_needed(operation))?;
        let output = self
            .agg_impl_with_groupby_error_mapper(specs, output_domain, operation, |error| {
                annotate_groupby_budget_execution_error(error, token, operation)
            })
            .map_err(|error| error.reanchor_if_needed(operation))?;
        let output_bytes = dataframe_device_allocated_size_bytes(&output, operation)?;
        let output_rows = output.num_rows();
        let report = token
            .reservation_enforced_report()
            .with_retained_bytes(output_bytes)
            .with_output_bytes(output_bytes)
            .with_output_rows(output_rows);

        Ok(BudgetedExecutionOutcome::new(output, report))
    }

    /// Execute the groupby with a borrowed per-operation execution override.
    pub(crate) fn agg_in_context(&self, specs: &[AggSpec], ctx: &ExecutionOverrides<'_>) -> Result<DataFrame> {
        let output_domain = ctx.resolve_output_domain(self.df.execution_domain())?;
        self.agg_impl(specs, output_domain, "groupby::GroupBy::with_execution::agg")
    }
    /// Prepare reusable grouping state using a borrowed per-operation
    /// execution override.
    pub(crate) fn prepare_groups_in_context(&self, ctx: &ExecutionOverrides<'_>) -> Result<PreparedGroupByFacts> {
        self.prepare_groups_impl(ctx, "groupby::GroupBy::with_execution::prepare_groups")
    }
    pub(crate) fn __debug_agg_in_context(&self, specs: &[AggSpec], ctx: &ExecutionOverrides<'_>) -> Result<DataFrame> {
        self.agg_in_context(specs, ctx)
    }
    fn prepare_groups_impl(
        &self,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<PreparedGroupByFacts> {
        use crate::internal::resolve::{build_borrowed_table_view_from_columns, resolve_column_names};

        require_non_empty_group_keys(self.keys.len(), operation)?;

        let columns = self.df.columns_map();
        let key_refs: SmallVec<[&str; GROUPBY_REQUEST_META_INLINE_CAPACITY]> =
            self.keys.iter().map(|s| s.as_str()).collect();
        let key_indices = resolve_column_names(columns, &key_refs, operation)?;

        let keys_table = build_borrowed_table_view_from_columns(columns, &key_indices)?;
        let prepared = if self.options.keys_are_sorted() {
            PreparedGroupBy::new_sorted_in(&keys_table, self.options.null_policy(), ctx)
        } else {
            PreparedGroupBy::new_in(&keys_table, self.options.null_policy(), ctx)
        }?;
        prepared.prepare_groups()
    }

    fn prepare_sort_groupby_impl(
        &self,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<PreparedDataFrameGroupBy> {
        use crate::internal::resolve::{build_borrowed_table_view_from_columns, resolve_column_names};

        require_non_empty_group_keys(self.keys.len(), operation)?;

        let retained_df = self.df.clone();
        let (handle, keys_domain) = {
            let columns = retained_df.columns_map();
            let key_refs: SmallVec<[&str; GROUPBY_REQUEST_META_INLINE_CAPACITY]> =
                self.keys.iter().map(|s| s.as_str()).collect();
            let key_indices = resolve_column_names(columns, &key_refs, operation)?;

            let keys_table = build_borrowed_table_view_from_columns(columns, &key_indices)?;
            let execution = ResolvedExecution::unary(keys_table.execution_domain(), ctx, operation)?;
            let mut handle = ptr::null_mut();
            error::check_status_in_with_details(
                unsafe {
                    cudf_sys::cudf_prepared_groupby_create_ex(
                        keys_table.raw_handle(),
                        self.options.null_policy().to_raw(),
                        i32::from(self.options.keys_are_sorted()),
                        &mut handle,
                        execution.exec_options(),
                    )
                },
                operation,
                [
                    ("null_policy", (self.options.null_policy() as i32).to_string()),
                    ("keys_are_sorted", self.options.keys_are_sorted().to_string()),
                    ("execution_authority", "retained_dataframe_keys".to_string()),
                ],
            )?;
            (
                error::require_non_null(handle, operation, "prepared dataframe groupby")?,
                Arc::clone(execution.output_domain()),
            )
        };

        let prepared_facts = match prepared_groupby_facts_raw(handle.as_ptr(), &keys_domain, operation) {
            Ok(facts) => facts,
            Err(error) => {
                drop_device_owned_handle(
                    handle.as_ptr(),
                    keys_domain.device_id(),
                    "groupby::PreparedDataFrameGroupBy::new::cleanup",
                    |owned| unsafe { cudf_sys::cudf_prepared_groupby_destroy(owned) },
                );
                return Err(error);
            },
        };

        Ok(PreparedDataFrameGroupBy {
            df: retained_df,
            keys: self.keys.clone(),
            options: self.options.clone(),
            handle,
            keys_domain,
            prepared_facts,
        })
    }

    fn agg_impl(
        &self,
        specs: &[AggSpec],
        output_domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<DataFrame> {
        self.agg_impl_with_groupby_error_mapper(specs, output_domain, operation, |error| error)
    }

    fn agg_impl_with_groupby_error_mapper<F>(
        &self,
        specs: &[AggSpec],
        output_domain: Arc<ExecutionDomain>,
        operation: &'static str,
        map_groupby_error: F,
    ) -> Result<DataFrame>
    where
        F: FnOnce(Error) -> Error,
    {
        use crate::{
            hash::{FastHashMap, fast_hash_map_with_capacity},
            internal::resolve::{build_borrowed_table_view_from_columns, resolve_column_names},
        };

        let execution =
            ResolvedExecution::with_output_domain(Arc::clone(&output_domain), [self.df.execution_domain()], operation)?;
        let output_domain = Arc::clone(execution.output_domain());

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

            result_names.push(spec.resolved_output_name());
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

        let gb_result = groupby_aggregate_impl(
            &keys_table,
            &requests,
            self.options.null_policy(),
            self.options.keys_are_sorted(),
            &output_domain,
            execution.exec_options(),
            operation,
        )
        .map_err(map_groupby_error)?;

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

    /// Apply a per-group row shift to each column named by `specs`.
    ///
    /// This is the building block for SQL `LAG(value, offset, default)` and
    /// `LEAD(value, offset, default)` window functions. Within each group
    /// libcudf shifts the column by the requested offset and fills
    /// indeterminate output rows with `fill_value`. The returned
    /// [`DataFrame`] contains the key columns followed by one shifted column
    /// per spec, in spec order. Row order within each group matches the input
    /// row order, but the inter-group order is libcudf-defined and may differ
    /// across calls.
    ///
    /// # Errors
    ///
    /// Returns an error if `specs` is empty, a referenced column does not
    /// exist, or a `ShiftSpec`'s `fill_value` does not have the same data
    /// type as the corresponding column.
    pub fn shift(&self, specs: &[ShiftSpec]) -> Result<DataFrame> {
        let output_domain = Arc::clone(self.df.execution_domain());
        self.shift_impl(specs, output_domain, false, "groupby::GroupBy::shift")
    }

    /// Apply a per-group row shift with a borrowed per-operation execution
    /// override.
    pub(crate) fn shift_in_context(&self, specs: &[ShiftSpec], ctx: &ExecutionOverrides<'_>) -> Result<DataFrame> {
        let output_domain = ctx.resolve_output_domain(self.df.execution_domain())?;
        self.shift_impl(specs, output_domain, false, "groupby::GroupBy::with_execution::shift")
    }

    /// Apply a per-group row shift, asserting that key columns are already
    /// ascending with nulls after.
    ///
    /// Use this overload when keys come from a prior sort step so libcudf can
    /// skip its internal key sort. Passing keys that are not actually sorted
    /// produces undefined output ordering within groups.
    pub fn shift_sorted(&self, specs: &[ShiftSpec]) -> Result<DataFrame> {
        let output_domain = Arc::clone(self.df.execution_domain());
        self.shift_impl(specs, output_domain, true, "groupby::GroupBy::shift_sorted")
    }

    fn shift_impl(
        &self,
        specs: &[ShiftSpec],
        output_domain: Arc<ExecutionDomain>,
        keys_are_sorted: bool,
        operation: &'static str,
    ) -> Result<DataFrame> {
        use crate::internal::resolve::{build_borrowed_table_view_from_columns, resolve_column_names};

        // Fill scalars may come from a different runtime / stream than the
        // dataframe, so include each `spec.fill_value.execution_domain()` in
        // the resolver inputs. Without this, libcudf would receive the
        // scalars on the output stream with no same-device check or stream
        // dependency, matching the contract that `copying::shift` and the
        // low-level `groupby_shift` already enforce.
        let secondaries: Vec<&Arc<ExecutionDomain>> = std::iter::once(self.df.execution_domain())
            .chain(specs.iter().map(|spec| spec.fill_value.execution_domain()))
            .collect();
        let execution = ResolvedExecution::with_output_domain(Arc::clone(&output_domain), secondaries, operation)?;
        let output_domain = Arc::clone(execution.output_domain());

        require_non_empty_group_keys(self.keys.len(), operation)?;

        if specs.is_empty() {
            return Err(unsupported_groupby_validation_error(
                operation,
                "at least one shift spec is required",
                "empty_shift_specs",
                "provide at least one ShiftSpec to GroupBy::shift",
            ));
        }

        let columns = self.df.columns_map();
        for spec in specs {
            if !columns.contains_key(spec.column.as_str()) {
                return Err(missing_column_error(operation, &spec.column));
            }
        }

        let key_refs: SmallVec<[&str; GROUPBY_REQUEST_META_INLINE_CAPACITY]> =
            self.keys.iter().map(|s| s.as_str()).collect();
        let key_indices = resolve_column_names(columns, &key_refs, operation)?;
        let keys_table = build_borrowed_table_view_from_columns(columns, &key_indices)?;

        let mut value_views: SmallVec<[ColumnView<'_>; GROUPBY_VALUE_VIEW_INLINE_CAPACITY]> =
            SmallVec::with_capacity(specs.len());
        for spec in specs {
            let arc_col =
                columns.get(spec.column.as_str()).ok_or_else(|| missing_column_error(operation, &spec.column))?;
            value_views.push(arc_col.view()?);
        }

        let requests: SmallVec<[GroupByShiftRequest<'_>; GROUPBY_VALUE_VIEW_INLINE_CAPACITY]> = value_views
            .iter()
            .zip(specs.iter())
            .map(|(values, spec)| GroupByShiftRequest { values, offset: spec.offset, fill_value: &spec.fill_value })
            .collect();

        let result = groupby_shift_impl(
            &keys_table,
            &requests,
            self.options.null_policy(),
            keys_are_sorted,
            &output_domain,
            execution.exec_options(),
            operation,
        )?;

        let result_keys_cols = result.keys.into_columns()?;
        let result_values_cols = result.values.into_columns()?;

        let mut final_cols: Vec<Column> = Vec::with_capacity(result_keys_cols.len() + result_values_cols.len());
        final_cols.extend(result_keys_cols);
        final_cols.extend(result_values_cols);

        let mut schema_names: Vec<String> = Vec::with_capacity(self.keys.len() + specs.len());
        schema_names.extend(self.keys.iter().cloned());
        schema_names.extend(specs.iter().map(|spec| spec.resolved_output_name()));

        DataFrame::new(schema_names, final_cols)
    }
}

/// Specification for one column in [`GroupBy::shift`].
///
/// `offset` matches libcudf's `cudf::size_type` convention: positive offsets
/// move each row toward larger indices within its group (SQL `LAG`), and
/// negative offsets move each row toward smaller indices (SQL `LEAD`).
/// `fill_value` materializes the indeterminate output rows that fall outside
/// the group; it must have the same data type as the column. A null-valued
/// fill scalar forces the output column to be nullable.
#[derive(Debug)]
pub struct ShiftSpec {
    column: String,
    offset: i32,
    fill_value: Scalar,
    alias: Option<String>,
}

impl ShiftSpec {
    /// Create a new shift specification for `column`.
    pub fn new(column: impl Into<String>, offset: i32, fill_value: Scalar) -> Self {
        Self { column: column.into(), offset, fill_value, alias: None }
    }

    /// Set an output column alias.
    pub fn with_alias(mut self, alias: impl Into<String>) -> Self {
        self.alias = Some(alias.into());
        self
    }

    /// Input column to shift.
    pub fn input_column(&self) -> &str {
        &self.column
    }

    /// Shift offset.
    pub fn offset(&self) -> i32 {
        self.offset
    }

    /// Fill value used for indeterminate output rows.
    pub fn fill_value(&self) -> &Scalar {
        &self.fill_value
    }

    /// Explicit output alias, if one was provided.
    pub fn alias(&self) -> Option<&str> {
        self.alias.as_deref()
    }

    /// Output column name: alias if set, otherwise `{column}_shift`.
    pub fn resolved_output_name(&self) -> String {
        match &self.alias {
            Some(alias) => alias.clone(),
            None => format!("{}_shift", self.column),
        }
    }
}

fn annotate_groupby_budget_execution_error(
    error: Error,
    token: &ExecutionBudgetToken,
    operation: &'static str,
) -> Error {
    match error.kind() {
        ErrorKind::ResourceExhausted | ErrorKind::DependencyFailure => token.annotate_error(error, operation),
        _ => error.reanchor_if_needed(operation),
    }
}

fn dataframe_device_allocated_size_bytes(frame: &DataFrame, operation: &'static str) -> Result<usize> {
    let info = frame.info_with(InfoDetail::DEVICE_ALLOCATED).map_err(|error| error.reanchor_if_needed(operation))?;
    let Some(memory) = info.memory_accounting else {
        return Err(invariant_violation(
            operation,
            "InfoDetail::DEVICE_ALLOCATED must populate dataframe memory accounting",
        ));
    };
    memory.device_allocated_bytes.ok_or_else(|| {
        invariant_violation(operation, "InfoDetail::DEVICE_ALLOCATED must populate dataframe device-allocated bytes")
    })
}

impl GroupByExecutionScope<'_, '_, '_> {
    /// Execute the groupby with the given aggregation specifications.
    pub fn agg(&self, specs: &[AggSpec]) -> Result<DataFrame> {
        self.groupby.agg_in_context(specs, self.ctx)
    }

    /// Prepare reusable grouping state and return cuDF-reported cardinality
    /// using this scope's execution override.
    pub fn prepare_groups(&self) -> Result<PreparedGroupByFacts> {
        self.groupby.prepare_groups_in_context(self.ctx)
    }

    /// Apply a per-group row shift using this scope's execution override.
    pub fn shift(&self, specs: &[ShiftSpec]) -> Result<DataFrame> {
        self.groupby.shift_in_context(specs, self.ctx)
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
