use std::sync::Arc;

use smallvec::SmallVec;

use super::{fillna_ops::DropDuplicatesOptions, group_keys::require_non_empty_group_keys, owned::DataFrame};
use crate::{
    core::{
        column::{Column, position_sequence_in_domain},
        error::{self, Result},
        table::TableView,
    },
    frame::series::Series,
    hash::FastIndexMap,
    internal::resolve::{build_borrowed_table_view_from_columns, resolve_column_names},
    ops::{
        groupby::{GroupByScanRequest, NullPolicy, groupby_scan_sorted},
        join::MergeOptions,
        reduction::ScanAgg,
        sorting::{self, SortOptions},
        stream_compaction::DuplicateKeepOption,
    },
    runtime::{
        BudgetedExecutionOutcome, ExecutionBudgetToken, execution_context::ExecutionOverrides,
        execution_domain::ExecutionDomain, execution_resolver::ResolvedExecution,
    },
};

const SORT_KEY_INLINE_CAPACITY: usize = 4;
const PREPARED_SORT_MATERIALIZED_OUTPUT_RESERVATION_CONTRACT: &str =
    "prepared_sort_materialized_output_reservation_size_bytes";

fn unique_temp_column_name(df: &DataFrame, base: &str) -> String {
    if !df.contains(base) {
        return base.to_string();
    }

    let mut suffix = 1usize;
    loop {
        let candidate = format!("{base}{suffix}");
        if !df.contains(&candidate) {
            return candidate;
        }
        suffix += 1;
    }
}

/// Retained DataFrame-level sorted-order plan.
///
/// This type owns the input [`DataFrame`] and retains a device-side INT32
/// sorted-order column. It is intended for bounded query engines that need to
/// split final sort into explicit phases:
///
/// - reserve and then retain the sorted-order row-index column,
/// - reserve exact materialized output bytes through the gather sizing path,
/// - materialize the sorted DataFrame by gathering the retained input.
///
/// This contract covers the retained sorted-order column and the materialized
/// gather output.
#[derive(Debug, Clone)]
pub struct DataFramePreparedSort {
    input: DataFrame,
    sorted_order: Series,
}

impl DataFramePreparedSort {
    pub(crate) fn new(input: DataFrame, sorted_order: Series) -> Self {
        Self { input, sorted_order }
    }

    /// Borrow the retained input DataFrame.
    pub fn input(&self) -> &DataFrame {
        &self.input
    }

    /// Borrow the retained INT32 sorted-order gather map.
    pub fn sorted_order(&self) -> &Series {
        &self.sorted_order
    }

    /// Consume this wrapper and return the retained input and sorted-order map.
    pub fn into_parts(self) -> (DataFrame, Series) {
        (self.input, self.sorted_order)
    }

    /// Number of rows in the materialized sorted output.
    pub fn output_rows(&self) -> usize {
        self.sorted_order.len()
    }

    /// Returns `true` when the materialized sorted output would contain no rows.
    pub fn is_empty(&self) -> bool {
        self.sorted_order.is_empty()
    }

    /// Return exact bytes required for a sorted-order INT32 row-index column.
    ///
    /// This pre-allocation helper performs no allocation and rejects row counts
    /// above libcudf's `cudf::size_type` / `INT32_MAX` limit. It only covers
    /// the retained sorted-order column, not libcudf sort temporary workspace.
    pub fn sorted_order_reservation_size_bytes_for_len(output_rows: usize) -> Result<usize> {
        sorting::sorted_order_reservation_size_bytes_for_len(output_rows)
            .map_err(|error| error.reanchor("dataframe::prepared_sort::sorted_order_reservation_size_bytes_for_len"))
    }

    /// Return exact bytes required for this prepared sort's retained
    /// sorted-order column.
    ///
    /// This is metadata-only and does not allocate. It does not include libcudf
    /// sort temporary workspace.
    pub fn sorted_order_reservation_size_bytes(&self) -> Result<usize> {
        Self::sorted_order_reservation_size_bytes_for_len(self.output_rows())
            .map_err(|error| error.reanchor("dataframe::prepared_sort::sorted_order_reservation_size_bytes"))
    }

    /// Return bytes retained by the prepared INT32 sorted-order column.
    ///
    /// This excludes the retained input DataFrame and excludes any libcudf sort
    /// temporary workspace used while producing the sorted order.
    pub fn retained_sorted_order_device_allocated_size_bytes(&self) -> Result<usize> {
        self.sorted_order.device_allocated_size_bytes().map_err(|error| {
            error.reanchor("dataframe::prepared_sort::retained_sorted_order_device_allocated_size_bytes")
        })
    }

    /// Return exact bytes retained by the materialized sorted DataFrame.
    ///
    /// This uses the same lower-level gather device-allocation sizing path as
    /// join-index materialization. For non-empty inputs it executes the gather
    /// sizing contract and releases any temporary result before returning. The
    /// result does not include the retained sorted-order column and does not
    /// include libcudf sort temporary workspace.
    pub fn materialized_output_reservation_size_bytes(&self) -> Result<usize> {
        let ctx = ExecutionOverrides::new();
        self.materialized_output_reservation_size_bytes_in_context(
            &ctx,
            "dataframe::prepared_sort::materialized_output_reservation_size_bytes",
        )
    }

    /// Return exact materialized output bytes using a borrowed execution
    /// override.
    pub fn materialized_output_reservation_size_bytes_in(&self, ctx: &ExecutionOverrides<'_>) -> Result<usize> {
        self.materialized_output_reservation_size_bytes_in_context(
            ctx,
            "dataframe::prepared_sort::materialized_output_reservation_size_bytes_in",
        )
    }

    /// Gather the retained input DataFrame using the retained sorted order.
    pub fn materialize(&self) -> Result<DataFrame> {
        self.input.gather(&self.sorted_order).map_err(|error| error.reanchor("dataframe::prepared_sort::materialize"))
    }

    /// Gather the retained input DataFrame using a borrowed execution override.
    pub fn materialize_in(&self, ctx: &ExecutionOverrides<'_>) -> Result<DataFrame> {
        self.input
            .gather_in_context(&self.sorted_order, ctx)
            .map_err(|error| error.reanchor("dataframe::prepared_sort::materialize_in"))
    }

    /// Gather the retained input DataFrame using a hard budget token.
    ///
    /// The gathered output allocation is routed through the token's
    /// stream-bound reservation. The returned report records the retained
    /// sorted-order bytes and exact materialized output bytes.
    pub fn materialize_with_budget(&self, token: &ExecutionBudgetToken) -> Result<BudgetedExecutionOutcome<DataFrame>> {
        let operation = "dataframe::prepared_sort::materialize_with_budget";
        let ctx = token.execution_overrides();
        let retained_bytes = self
            .retained_sorted_order_device_allocated_size_bytes()
            .map_err(|error| token.annotate_error(error, operation))?;
        let output_bytes = self
            .materialized_output_reservation_size_bytes_in(&ctx)
            .map_err(|error| token.annotate_error(error, operation))?;
        let output = self.materialize_in(&ctx).map_err(|error| token.annotate_error(error, operation))?;
        let report =
            token.reservation_enforced_report().with_retained_bytes(retained_bytes).with_output_bytes(output_bytes);

        Ok(BudgetedExecutionOutcome::new(output, report))
    }

    /// Alias for [`materialize`](Self::materialize).
    pub fn execute(&self) -> Result<DataFrame> {
        self.materialize().map_err(|error| error.reanchor("dataframe::prepared_sort::execute"))
    }

    /// Alias for [`materialize_with_budget`](Self::materialize_with_budget).
    pub fn execute_with_budget(&self, token: &ExecutionBudgetToken) -> Result<BudgetedExecutionOutcome<DataFrame>> {
        self.materialize_with_budget(token)
            .map_err(|error| error.reanchor("dataframe::prepared_sort::execute_with_budget"))
    }

    fn materialized_output_reservation_size_bytes_in_context(
        &self,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<usize> {
        self.input
            .gather_device_allocated_size_in_context(
                &self.sorted_order,
                ctx,
                operation,
                PREPARED_SORT_MATERIALIZED_OUTPUT_RESERVATION_CONTRACT,
            )
            .map_err(|error| error.reanchor(operation))
    }
}

impl DataFrame {
    /// Sort using a borrowed per-operation execution override.
    pub(crate) fn sort_values_in_context<I, S>(
        &self,
        by: I,
        options: SortOptions,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let output_domain = ctx.resolve_output_domain(&self.domain)?;
        self.sort_values_impl(by, options, output_domain, "dataframe::with_execution::sort_values")
    }
    pub(crate) fn __debug_sort_values_in_context<I, S>(
        &self,
        by: I,
        options: SortOptions,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.sort_values_in_context(by, options, ctx)
    }

    /// Sort rows using cuDF-style [`SortOptions`].
    pub fn sort_values<I, S>(&self, by: I, options: SortOptions) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.sort_values_impl(by, options, Arc::clone(&self.domain), "dataframe::sort_values")
    }

    /// Prepare a retained sorted-order gather map for a later DataFrame sort.
    ///
    /// This resolves and validates keys/options with the same user-facing
    /// rules as [`sort_values`](Self::sort_values), then retains only the INT32
    /// sorted-order row-index column. Use
    /// [`DataFramePreparedSort::materialized_output_reservation_size_bytes`]
    /// to reserve the exact gathered output bytes before calling
    /// [`DataFramePreparedSort::materialize`].
    ///
    /// This contract covers the retained sorted-order bytes and the
    /// materialized gather output sizing, and retains the unresolved
    /// libcudf sort temporary-workspace contract.
    pub fn prepare_sort_values<I, S>(&self, by: I, options: SortOptions) -> Result<DataFramePreparedSort>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.prepare_sort_values_impl(by, options, Arc::clone(&self.domain), "dataframe::prepare_sort_values")
    }

    /// Prepare a retained sorted-order gather map using a hard budget token.
    ///
    /// Sorted-order creation is routed through the token's stream-bound
    /// reservation. The returned report records the retained sorted-order
    /// bytes and the unresolved libcudf sort temporary-workspace contract.
    pub fn prepare_sort_values_with_budget<I, S>(
        &self,
        by: I,
        options: SortOptions,
        token: &ExecutionBudgetToken,
    ) -> Result<BudgetedExecutionOutcome<DataFramePreparedSort>>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let operation = "dataframe::prepare_sort_values_with_budget";
        let ctx = token.execution_overrides();
        let prepared = self
            .prepare_sort_values_in_context(by, options, &ctx)
            .map_err(|error| token.annotate_error(error, operation))?;
        let retained_bytes = prepared
            .retained_sorted_order_device_allocated_size_bytes()
            .map_err(|error| token.annotate_error(error, operation))?;
        let report = token.reservation_enforced_report().with_retained_bytes(retained_bytes);

        Ok(BudgetedExecutionOutcome::new(prepared, report))
    }

    fn sort_values_impl<I, S>(
        &self,
        by: I,
        options: SortOptions,
        output_domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let execution = ResolvedExecution::with_output_domain(Arc::clone(&output_domain), [&self.domain], operation)?;

        let by: SmallVec<[String; SORT_KEY_INLINE_CAPACITY]> = by.into_iter().map(|s| s.as_ref().to_string()).collect();
        let by_refs: SmallVec<[&str; SORT_KEY_INLINE_CAPACITY]> = by.iter().map(|s| s.as_str()).collect();
        let key_indices = resolve_column_names(&self.columns, &by_refs, operation)?;
        let (orders, null_orders) = options.resolve(by_refs.len(), operation)?;
        let null_order_ref = null_orders.as_deref();

        let all_in_order =
            key_indices.len() == self.columns.len() && key_indices.iter().enumerate().all(|(i, &idx)| idx == i);

        let tv = self.build_table_view()?;
        let result_table = if all_in_order {
            sorting::sort_table_impl(
                &tv,
                sorting::SortKeyConfig::new(Some(&orders), null_order_ref),
                options.stable,
                sorting::SortExecution::new(&output_domain, execution.exec_options(), operation),
            )?
        } else {
            let keys_table = tv.select_columns(&key_indices)?;
            sorting::sort_by_key_impl(
                &tv,
                &keys_table,
                sorting::SortKeyConfig::new(Some(&orders), null_order_ref),
                options.stable,
                sorting::SortExecution::new(&output_domain, execution.exec_options(), operation),
            )?
        }
        .with_execution_domain(output_domain);

        Self::from_table_with_names(result_table, self.column_names_owned())
    }

    pub(crate) fn prepare_sort_values_in_context<I, S>(
        &self,
        by: I,
        options: SortOptions,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<DataFramePreparedSort>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let output_domain = ctx.resolve_output_domain(&self.domain)?;
        self.prepare_sort_values_impl(by, options, output_domain, "dataframe::with_execution::prepare_sort_values")
    }

    fn prepare_sort_values_impl<I, S>(
        &self,
        by: I,
        options: SortOptions,
        output_domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<DataFramePreparedSort>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let execution = ResolvedExecution::with_output_domain(Arc::clone(&output_domain), [&self.domain], operation)?;

        let by: SmallVec<[String; SORT_KEY_INLINE_CAPACITY]> = by.into_iter().map(|s| s.as_ref().to_string()).collect();
        let by_refs: SmallVec<[&str; SORT_KEY_INLINE_CAPACITY]> = by.iter().map(|s| s.as_str()).collect();
        let key_indices = resolve_column_names(&self.columns, &by_refs, operation)?;
        let (mut orders, mut null_orders) = options.resolve(by_refs.len(), operation)?;

        let stable_row_position = if options.stable {
            orders.push(sorting::Order::Ascending);
            if let Some(null_orders) = null_orders.as_mut() {
                null_orders.push(sorting::NullOrder::After);
            }
            Some(position_sequence_in_domain(self.row_count, &output_domain, operation)?)
        } else {
            None
        };

        let stable_key_columns;
        let keys_table = if stable_row_position.is_some() {
            stable_key_columns = self.sort_key_columns_in_domain(&key_indices, &output_domain, operation)?;
            Self::build_stable_sort_key_table_for_sorted_order(
                &stable_key_columns,
                stable_row_position.as_ref(),
                operation,
            )?
        } else {
            Self::build_sort_key_table_for_sorted_order(
                self.columns_map(),
                &key_indices,
                stable_row_position.as_ref(),
                operation,
            )?
        };
        let sorted_order = sorting::sorted_order_impl(
            &keys_table,
            sorting::SortKeyConfig::new(Some(&orders), null_orders.as_deref()),
            sorting::SortExecution::new(&output_domain, execution.exec_options(), operation),
        )?
        .with_execution_domain(output_domain);
        let sorted_order = Series::from_column(sorted_order)
            .map_err(|error| error.reanchor(operation).with_detail("resource", "sorted_order"))?;

        Ok(DataFramePreparedSort::new(self.clone(), sorted_order))
    }

    fn sort_key_columns_in_domain(
        &self,
        key_indices: &[usize],
        output_domain: &Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Vec<Arc<Column>>> {
        let should_copy = !ExecutionDomain::same_provenance(&self.domain, output_domain);
        let mut key_columns = Vec::with_capacity(key_indices.len());
        for &idx in key_indices {
            let (_, column) = self.columns.get_index(idx).expect("resolved sort key index remains valid");
            let column = if should_copy {
                Arc::new(column.copy_into_domain(Arc::clone(output_domain), operation)?)
            } else {
                Arc::clone(column)
            };
            key_columns.push(column);
        }
        Ok(key_columns)
    }

    fn build_stable_sort_key_table_for_sorted_order<'a>(
        key_columns: &'a [Arc<Column>],
        stable_row_position: Option<&'a Column>,
        operation: &'static str,
    ) -> Result<TableView<'a>> {
        let mut views = Vec::with_capacity(key_columns.len() + usize::from(stable_row_position.is_some()));
        for column in key_columns {
            views.push(column.view()?);
        }
        if let Some(stable_row_position) = stable_row_position {
            views.push(stable_row_position.view()?);
        }

        TableView::from_borrowed_column_views_in_domain(views, None, operation)
    }

    fn build_sort_key_table_for_sorted_order<'a>(
        columns: &'a FastIndexMap<String, Arc<Column>>,
        key_indices: &[usize],
        stable_row_position: Option<&'a Column>,
        operation: &'static str,
    ) -> Result<TableView<'a>> {
        let mut views = Vec::with_capacity(key_indices.len() + usize::from(stable_row_position.is_some()));
        for &idx in key_indices {
            let (_, column) = columns.get_index(idx).expect("resolved sort key index remains valid");
            views.push(column.view()?);
        }
        if let Some(stable_row_position) = stable_row_position {
            views.push(stable_row_position.view()?);
        }

        if views.is_empty() {
            build_borrowed_table_view_from_columns(columns, &[])
        } else {
            TableView::from_borrowed_column_views_in_domain(views, None, operation)
        }
    }

    pub fn nlargest(&self, n: usize, column: &str) -> Result<Self> {
        self.sort_values([column], SortOptions::new().with_descending(vec![true]))?.head(n)
    }

    pub fn nsmallest(&self, n: usize, column: &str) -> Result<Self> {
        self.sort_values([column], SortOptions::new())?.head(n)
    }

    /// Return zero-based `Int32` row positions aligned to the current frame order.
    pub fn row_positions(&self) -> Result<Series> {
        Series::from_column(position_sequence_in_domain(self.row_count, &self.domain, "dataframe::row_positions")?)
    }

    /// Return a zero-based per-group row number for already-sorted rows.
    pub fn group_cumcount<I, S>(&self, keys: I) -> Result<Series>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let operation = "dataframe::group_cumcount";
        let keys: SmallVec<[String; SORT_KEY_INLINE_CAPACITY]> =
            keys.into_iter().map(|key| key.as_ref().to_string()).collect();
        require_non_empty_group_keys(keys.len(), operation)?;

        let key_refs: SmallVec<[&str; SORT_KEY_INLINE_CAPACITY]> = keys.iter().map(|key| key.as_str()).collect();
        resolve_column_names(&self.columns, &key_refs, operation)?;

        let row_idx_name = unique_temp_column_name(self, "__cudf_group_cumcount_row_idx__");
        let group_start_name = unique_temp_column_name(self, "__cudf_group_cumcount_group_start__");
        let row_idx = Series::from_column(position_sequence_in_domain(self.row_count, &self.domain, operation)?)?;
        let with_row_idx = self.with_column(&row_idx_name, row_idx)?;

        let first_rows = with_row_idx.drop_duplicates(
            DropDuplicatesOptions::builder().subset(key_refs.as_slice()).keep(DuplicateKeepOption::KeepFirst).build(),
        )?;
        let mut first_row_columns: SmallVec<[&str; SORT_KEY_INLINE_CAPACITY + 1]> = SmallVec::new();
        first_row_columns.extend(key_refs.iter().copied());
        first_row_columns.push(row_idx_name.as_str());
        let first_rows = first_rows.select(first_row_columns)?;
        let first_rows = first_rows
            .with_column(&group_start_name, first_rows.series(&row_idx_name)?)?
            .without_columns([row_idx_name.as_str()])?;

        let aligned = with_row_idx
            .merge(&first_rows, MergeOptions::new().how_left().on(key_refs.iter().copied()))?
            .sort_values([row_idx_name.as_str()], SortOptions::new().with_stable(true))?;

        let rank = aligned.series(&row_idx_name)?.sub(&aligned.series(&group_start_name)?)?;
        Ok(rank)
    }

    /// Return a per-group inclusive cumulative scan for already group-sorted rows.
    ///
    /// Rows with equal `keys` must be contiguous and in the desired scan order.
    /// The returned series is row-aligned to this frame.
    pub fn group_scan_sorted<I, S>(&self, keys: I, value: &str, agg: ScanAgg) -> Result<Series>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let operation = "dataframe::group_scan_sorted";
        let keys: SmallVec<[String; SORT_KEY_INLINE_CAPACITY]> =
            keys.into_iter().map(|key| key.as_ref().to_string()).collect();
        require_non_empty_group_keys(keys.len(), operation)?;

        let key_refs: SmallVec<[&str; SORT_KEY_INLINE_CAPACITY]> = keys.iter().map(|key| key.as_str()).collect();
        let key_indices = resolve_column_names(&self.columns, &key_refs, operation)?;
        let value_indices = resolve_column_names(&self.columns, &[value], operation)?;
        let value_column = self.columns.get_index(value_indices[0]).expect("resolved scan value index remains valid");

        let keys_table = build_borrowed_table_view_from_columns(&self.columns, &key_indices)?;
        let value_view = value_column.1.view()?;
        let aggregations = [agg];
        let requests = [GroupByScanRequest { values: &value_view, aggregations: &aggregations }];
        let mut output = groupby_scan_sorted(&keys_table, &requests, NullPolicy::Include)?;
        let result = output.results.pop().and_then(|mut columns| columns.pop()).ok_or_else(|| {
            error::invariant_violation(operation, "grouped scan returned no result column")
                .with_detail("requested_aggregations", "1")
        })?;
        Series::from_column(result)
    }
}
