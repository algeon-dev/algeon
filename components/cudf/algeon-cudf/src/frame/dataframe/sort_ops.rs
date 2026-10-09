use std::sync::Arc;

use smallvec::SmallVec;

use super::{
    context::FrameOps, fillna_ops::DropDuplicatesOptions, group_keys::require_non_empty_group_keys, owned::DataFrame,
};
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
        stream_compaction::{self, DuplicateKeepOption, StreamCompactionExecution},
    },
    runtime::{
        BudgetedExecutionOutcome, CompleteReport, ExecutionBudgetReport, ExecutionBudgetToken, execution_context::Exec,
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
    /// the retained sorted-order column and does not cover libcudf sort
    /// temporary workspace.
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
    pub fn materialized_output_reservation_size_bytes(&self, ctx: Exec<'_>) -> Result<usize> {
        self.materialized_output_reservation_size_bytes_in_context(
            ctx,
            "dataframe::prepared_sort::materialized_output_reservation_size_bytes",
        )
    }

    /// Gather the retained input DataFrame using the retained sorted order.
    pub fn materialize(&self, ctx: Exec<'_>) -> Result<DataFrame> {
        self.input
            .gather_in_context(&self.sorted_order, ctx)
            .map_err(|error| error.reanchor("dataframe::prepared_sort::materialize"))
    }

    /// Materialize rows after removing consecutive duplicate keys in the
    /// prepared order. Equal keys must already be adjacent in that order.
    /// First and last retention preserve the prepared order, including stable
    /// ties. Null and NaN keys compare equal. The prepared row order is released
    /// before gathering retained payload rows.
    pub fn materialize_drop_consecutive_duplicates(
        self,
        opts: DropDuplicatesOptions,
        ctx: Exec<'_>,
    ) -> Result<DataFrame> {
        let operation = "dataframe::prepared_sort::materialize_drop_consecutive_duplicates";
        let output_domain = ctx.resolve_output_domain(&self.input.domain)?;
        let execution = ResolvedExecution::with_output_domain(
            Arc::clone(&output_domain),
            [&self.input.domain, self.sorted_order.as_column().execution_domain()],
            operation,
        )?;
        let subset = if opts.subset.is_empty() { self.input.column_names_owned() } else { opts.subset };
        let subset_refs = subset.iter().map(String::as_str).collect::<Vec<_>>();
        let key_indices = resolve_column_names(&self.input.columns, &subset_refs, operation)?;
        let selected = {
            let input = self.input.build_table_view()?;
            let row_order = self.sorted_order.view()?;
            let selected = stream_compaction::ordered_unique_indices_impl(
                &input,
                &row_order,
                &key_indices,
                opts.keep,
                StreamCompactionExecution::new(&output_domain, execution.exec_options(), operation),
            )?;
            Series::from_column(selected)?
        };
        // The selected indices replace the prepared permutation before the payload allocation.
        drop(self.sorted_order);
        self.input.gather_in_context(&selected, ctx).map_err(|error| error.reanchor(operation))
    }

    /// Gather the retained input DataFrame using a hard budget token.
    ///
    /// The gathered output allocation is routed through the token's
    /// stream-bound reservation. The returned report records the retained
    /// sorted-order bytes and exact materialized output bytes.
    pub fn materialize_with_budget(&self, token: &ExecutionBudgetToken) -> Result<BudgetedExecutionOutcome<DataFrame>> {
        let operation = "dataframe::prepared_sort::materialize_with_budget";
        let ctx = token.exec();
        let retained_bytes = self
            .retained_sorted_order_device_allocated_size_bytes()
            .map_err(|error| token.annotate_error(error, operation))?;
        let output_bytes = self
            .materialized_output_reservation_size_bytes(ctx)
            .map_err(|error| token.annotate_error(error, operation))?;
        let output = self.materialize(ctx).map_err(|error| token.annotate_error(error, operation))?;
        let report =
            token.reservation_enforced_report().with_retained_bytes(retained_bytes).with_output_bytes(output_bytes);

        Ok(BudgetedExecutionOutcome::new(output, report))
    }

    fn materialized_output_reservation_size_bytes_in_context(
        &self,
        ctx: Exec<'_>,
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
    pub(crate) fn sort_values_in_context<I, S>(&self, by: I, options: SortOptions, ctx: Exec<'_>) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let output_domain = ctx.resolve_output_domain(&self.domain)?;
        self.sort_values_impl(by, options, output_domain, "dataframe::sort_values")
    }
    #[cfg(any(test, feature = "test-hooks"))]
    pub(crate) fn __debug_sort_values_in_context<I, S>(
        &self,
        by: I,
        options: SortOptions,
        ctx: Exec<'_>,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.sort_values_in_context(by, options, ctx)
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
        };

        Self::from_table_with_names(result_table, self.column_names_owned())
    }

    pub(crate) fn prepare_sort_values_in_context<I, S>(
        &self,
        by: I,
        options: SortOptions,
        ctx: Exec<'_>,
    ) -> Result<DataFramePreparedSort>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let output_domain = ctx.resolve_output_domain(&self.domain)?;
        self.prepare_sort_values_impl(by, options, output_domain, "dataframe::prepare_sort_values")
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
        let (orders, null_orders) = options.resolve(by_refs.len(), operation)?;

        let sorted_order = if options.stable && key_indices.is_empty() {
            // A zero-key stable sort preserves every input row; libcudf's
            // zero-column sorted order is empty even for a nonempty input.
            position_sequence_in_domain(self.row_count, &output_domain, operation)?
        } else {
            let keys_table = Self::build_sort_key_table_for_sorted_order(self.columns_map(), &key_indices, operation)?;
            sorting::sorted_order_impl(
                &keys_table,
                sorting::SortKeyConfig::new(Some(&orders), null_orders.as_deref()),
                options.stable,
                sorting::SortExecution::new(&output_domain, execution.exec_options(), operation),
            )?
        };
        let sorted_order = Series::from_column(sorted_order)
            .map_err(|error| error.reanchor(operation).with_detail("resource", "sorted_order"))?;

        Ok(DataFramePreparedSort::new(self.clone(), sorted_order))
    }

    fn build_sort_key_table_for_sorted_order<'a>(
        columns: &'a FastIndexMap<String, Arc<Column>>,
        key_indices: &[usize],
        operation: &'static str,
    ) -> Result<TableView<'a>> {
        let mut views = Vec::with_capacity(key_indices.len());
        for &idx in key_indices {
            let (_, column) = columns.get_index(idx).expect("resolved sort key index remains valid");
            views.push(column.view()?);
        }
        if views.is_empty() {
            build_borrowed_table_view_from_columns(columns, &[])
        } else {
            TableView::from_borrowed_column_views_in_domain(views, None, operation)
        }
    }
}

impl FrameOps<'_> {
    /// The `n` rows with the largest `column` values, in descending order.
    pub fn nlargest(&self, n: usize, column: &str) -> Result<DataFrame> {
        self.sort_values([column], SortOptions::new().with_descending(vec![true]))?.on(Exec::RETAINED).head(n)
    }

    /// The `n` rows with the smallest `column` values, in ascending order.
    pub fn nsmallest(&self, n: usize, column: &str) -> Result<DataFrame> {
        self.sort_values([column], SortOptions::new())?.on(Exec::RETAINED).head(n)
    }

    /// Return zero-based `Int32` row positions aligned to the current frame order.
    pub fn row_positions(&self) -> Result<Series> {
        let output_domain = self.ctx.resolve_output_domain(&self.df.domain)?;
        Series::from_column(position_sequence_in_domain(self.df.row_count, &output_domain, "dataframe::row_positions")?)
    }

    /// Return a zero-based per-group row number for already-sorted rows.
    ///
    /// The result depends only on the `keys` columns, so no other column is
    /// read or copied. A key listed more than once groups as if listed once.
    pub fn group_cumcount<I, S>(&self, keys: I) -> Result<Series>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let operation = "dataframe::group_cumcount";
        let keys: SmallVec<[String; SORT_KEY_INLINE_CAPACITY]> =
            keys.into_iter().map(|key| key.as_ref().to_string()).collect();
        require_non_empty_group_keys(keys.len(), operation)?;

        let mut key_refs: SmallVec<[&str; SORT_KEY_INLINE_CAPACITY]> = SmallVec::new();
        for key in &keys {
            if !key_refs.contains(&key.as_str()) {
                key_refs.push(key.as_str());
            }
        }
        resolve_column_names(&self.df.columns, &key_refs, operation)?;

        let key_frame = self.df.select(key_refs.iter().copied())?;
        let key_ops = key_frame.on(self.ctx);
        let row_idx_name = unique_temp_column_name(&key_frame, "__cudf_group_cumcount_row_idx__");
        let group_start_name = unique_temp_column_name(&key_frame, "__cudf_group_cumcount_group_start__");
        let with_row_idx = key_ops.with_column(&row_idx_name, key_ops.row_positions()?)?;

        let first_rows = with_row_idx.on(Exec::RETAINED).drop_duplicates(
            DropDuplicatesOptions::builder().subset(key_refs.as_slice()).keep(DuplicateKeepOption::KeepFirst).build(),
        )?;
        let first_rows = first_rows
            .with_column(&group_start_name, first_rows.series(&row_idx_name)?)?
            .without_columns([row_idx_name.as_str()])?;

        let aligned = with_row_idx
            .on(Exec::RETAINED)
            .merge(&first_rows, MergeOptions::new().how_left().on(key_refs.iter().copied()))?
            .on(Exec::RETAINED)
            .sort_values([row_idx_name.as_str()], SortOptions::new().with_stable(true))?;

        aligned.series(&row_idx_name)?.on(Exec::RETAINED).sub(&aligned.series(&group_start_name)?)
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
        let columns = &self.df.columns;
        let keys: SmallVec<[String; SORT_KEY_INLINE_CAPACITY]> =
            keys.into_iter().map(|key| key.as_ref().to_string()).collect();
        require_non_empty_group_keys(keys.len(), operation)?;

        let key_refs: SmallVec<[&str; SORT_KEY_INLINE_CAPACITY]> = keys.iter().map(|key| key.as_str()).collect();
        let key_indices = resolve_column_names(columns, &key_refs, operation)?;
        let value_indices = resolve_column_names(columns, &[value], operation)?;
        let value_column = columns.get_index(value_indices[0]).expect("resolved scan value index remains valid");

        let keys_table = build_borrowed_table_view_from_columns(columns, &key_indices)?;
        let value_view = value_column.1.view()?;
        let aggregations = [agg];
        let requests = [GroupByScanRequest { values: &value_view, aggregations: &aggregations }];
        let mut output = groupby_scan_sorted(&keys_table, &requests, NullPolicy::Include, self.ctx)?;
        let result = output.results.pop().and_then(|mut columns| columns.pop()).ok_or_else(|| {
            error::invariant_violation(operation, "grouped scan returned no result column")
                .with_detail("requested_aggregations", "1")
        })?;
        Series::from_column(result)
    }
}

/// A prepared sort retains its sorted order; materializing it is a separate
/// operation.
impl CompleteReport for DataFramePreparedSort {
    fn complete_report(&self, report: ExecutionBudgetReport) -> Result<ExecutionBudgetReport> {
        Ok(report.with_retained_bytes(self.retained_sorted_order_device_allocated_size_bytes()?))
    }
}
