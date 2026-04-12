use typed_builder::TypedBuilder;

/// How [`DataFrame::dropna`] decides which rows to drop when `thresh` is unset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropNaHow {
    /// Drop rows when any inspected column is null.
    Any,
    /// Drop rows only when all inspected columns are null.
    All,
}

impl Default for DropNaHow {
    fn default() -> Self {
        Self::Any
    }
}

/// Options for [`DataFrame::dropna`].
///
/// Controls which rows are dropped based on null values. Mirrors the
/// `dropna()` API from cuDF Python.
///
/// # Defaults
///
/// - `how` = any (drop if **any** subset column is null)
/// - `subset` = all columns
/// - `thresh` = `None` (when set, overrides `how`)
#[derive(Debug, Clone, TypedBuilder)]
pub struct DropNaOptions {
    /// Column names to inspect. Empty means all columns.
    #[builder(default, setter(transform = |columns: &[&str]| columns.iter().map(|s| s.to_string()).collect()))]
    subset: Vec<String>,
    /// Minimum number of non-null values in subset required to keep the row.
    /// `None` means derive from `how`.
    #[builder(default, setter(transform = |thresh: i32| Some(thresh)))]
    thresh: Option<i32>,
    /// How rows should be dropped when `thresh` is unset.
    #[builder(default)]
    how: DropNaHow,
}

impl Default for DropNaOptions {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl DropNaOptions {
    /// Create default options: how=any, subset=all columns, no threshold override.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Options for [`DataFrame::drop_duplicates`].
///
/// Controls deduplication behavior. Mirrors `drop_duplicates()` from cuDF
/// Python.
///
/// # Defaults
///
/// - `subset` = all columns
/// - `keep` = [`DuplicateKeepOption::KeepAny`]
#[derive(Debug, Clone, TypedBuilder)]
pub struct DropDuplicatesOptions {
    /// Column names to use as deduplication keys. Empty means all columns.
    #[builder(default, setter(transform = |columns: &[&str]| columns.iter().map(|s| s.to_string()).collect()))]
    subset: Vec<String>,
    /// Which duplicate occurrence to keep.
    #[builder(default = DuplicateKeepOption::KeepAny)]
    keep: DuplicateKeepOption,
}

impl Default for DropDuplicatesOptions {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl DropDuplicatesOptions {
    /// Create default options: all columns, keep any.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Options for [`DataFrame::melt`] (unpivot: wide → long format).
///
/// Transforms a "wide" DataFrame into "long" format by taking one or more
/// columns and turning them into rows. Mirrors the `melt()` API from cuDF
/// Python / pandas.
///
/// # Defaults
///
/// - `id_vars` = empty (no identifier columns)
/// - `value_vars` = empty (defaults to all columns not in `id_vars`)
/// - `var_name` = `"variable"`
/// - `value_name` = `"value"`
#[derive(Debug, Clone, TypedBuilder)]
pub struct MeltOptions {
    /// Columns to use as identifier variables (kept as-is, tiled K times).
    #[builder(default, setter(transform = |columns: &[&str]| columns.iter().map(|s| s.to_string()).collect()))]
    id_vars: Vec<String>,
    /// Columns to unpivot into the `value` column. Empty means all columns
    /// not in `id_vars`.
    #[builder(default, setter(transform = |columns: &[&str]| columns.iter().map(|s| s.to_string()).collect()))]
    value_vars: Vec<String>,
    /// Name for the column that stores the original column names.
    #[builder(default = "variable".to_string(), setter(transform = |name: &str| name.to_string()))]
    var_name: String,
    /// Name for the column that stores the unpivoted values.
    #[builder(default = "value".to_string(), setter(transform = |name: &str| name.to_string()))]
    value_name: String,
}

impl Default for MeltOptions {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl MeltOptions {
    /// Create default options: no id_vars, all columns as value_vars,
    /// var_name = "variable", value_name = "value".
    pub fn new() -> Self {
        Self::default()
    }
}

use std::sync::Arc;

use smallvec::SmallVec;

use super::owned::{DataFrame, derive_frame_domain_from_columns};
use crate::{
    core::{
        column::{Column, ColumnView},
        error::{Error, ErrorKind, Result},
        scalar::Scalar,
        types::{CudfType, DataType},
    },
    frame::series::{IntoSeries, Series},
    internal::{
        hash::{FastHashSet, fast_index_map_with_capacity},
        resolve::{missing_column_error, resolve_column_names},
    },
    ops::{
        self, copying, filter, replace,
        sorting::{self, SortOptions},
        stream_compaction::{self, DuplicateKeepOption, NanEquality, NullEquality},
        unary,
    },
    runtime::{execution_context::ExecutionContext, execution_domain::ExecutionDomain},
};

const SORT_KEY_INLINE_CAPACITY: usize = 4;

/// Trait for values that can be normalized into a `with_columns()` item.
///
/// This supports either a bare [`Expr`](crate::ops::Expr) or an explicit
/// `(name, expr)` pair.
#[doc(hidden)]
pub trait IntoWithColumnExpr {
    /// Convert into a normalized `with_columns()` item.
    fn into_with_column_expr(self) -> (Option<String>, ops::expr::Expr);
}

impl IntoWithColumnExpr for ops::expr::Expr {
    fn into_with_column_expr(self) -> (Option<String>, ops::expr::Expr) {
        (None, self)
    }
}

impl<S> IntoWithColumnExpr for (S, ops::expr::Expr)
where S: Into<String>
{
    fn into_with_column_expr(self) -> (Option<String>, ops::expr::Expr) {
        (Some(self.0.into()), self.1)
    }
}

impl DataFrame {
    /// Sort using a borrowed per-operation execution override.
    pub(crate) fn sort_values_in_context<I, S>(
        &self,
        by: I,
        options: SortOptions,
        ctx: &ExecutionContext<'_>,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let output_domain = ctx.resolve_output_domain(&self.domain)?;
        self.sort_values_impl(
            by,
            options,
            output_domain.explicit_exec_options_raw(),
            output_domain,
            "dataframe::in_context::sort_values",
        )
    }

    #[doc(hidden)]
    pub fn __debug_sort_values_in_context<I, S>(
        &self,
        by: I,
        options: SortOptions,
        ctx: &ExecutionContext<'_>,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.sort_values_in_context(by, options, ctx)
    }

    /// Gather rows using a borrowed per-operation execution override.
    pub(crate) fn gather_in_context(&self, indices: &Series, ctx: &ExecutionContext<'_>) -> Result<Self> {
        let output_domain = ctx.resolve_output_domain(&self.domain)?;
        let cv = indices.as_column_view()?;
        self.gather_by_column_view_impl(
            &cv,
            output_domain.explicit_exec_options_raw(),
            output_domain,
            "dataframe::in_context::gather",
        )
    }

    #[doc(hidden)]
    pub fn __debug_gather_in_context(&self, indices: &Series, ctx: &ExecutionContext<'_>) -> Result<Self> {
        self.gather_in_context(indices, ctx)
    }

    /// Filter rows using a borrowed per-operation execution override.
    pub(crate) fn filter_in_context(&self, mask: &Series, ctx: &ExecutionContext<'_>) -> Result<Self> {
        let output_domain = ctx.resolve_output_domain(&self.domain)?;
        let mask_view = self.validate_series_mask(mask)?;
        self.filter_by_mask_impl(
            &mask_view,
            output_domain.explicit_exec_options_raw(),
            output_domain,
            "dataframe::in_context::filter",
        )
    }

    #[doc(hidden)]
    pub fn __debug_filter_in_context(&self, mask: &Series, ctx: &ExecutionContext<'_>) -> Result<Self> {
        self.filter_in_context(mask, ctx)
    }

    /// Return the first `n` rows using a borrowed per-operation execution override.
    pub(crate) fn head_in_context(&self, n: usize, ctx: &ExecutionContext<'_>) -> Result<Self> {
        let total = self.num_rows();
        let take = n.min(total);
        self.gather_range_in_context(0, take, ctx, "dataframe::in_context::head")
    }

    /// Return the last `n` rows using a borrowed per-operation execution override.
    pub(crate) fn tail_in_context(&self, n: usize, ctx: &ExecutionContext<'_>) -> Result<Self> {
        let total = self.num_rows();
        let take = n.min(total);
        let start = total - take;
        self.gather_range_in_context(start, take, ctx, "dataframe::in_context::tail")
    }

    /// Return `length` rows starting at `offset` using a borrowed per-operation execution override.
    pub(crate) fn slice_in_context(&self, offset: usize, length: usize, ctx: &ExecutionContext<'_>) -> Result<Self> {
        let total = self.num_rows();
        let start = offset.min(total);
        let take = length.min(total - start);
        self.gather_range_in_context(start, take, ctx, "dataframe::in_context::slice")
    }

    /// Return the first `n` rows.
    ///
    /// If `n >= num_rows`, all rows are returned. The schema is preserved.
    ///
    /// # Cost
    ///
    /// GPU-allocating: always produces a new table with gathered rows via FFI.
    /// GPU allocation dominates runtime cost, so a consuming variant would
    /// provide negligible benefit.
    pub fn head(&self, n: usize) -> Result<Self> {
        let total = self.num_rows();
        let take = n.min(total);
        self.gather_range(0, take)
    }

    /// Return the last `n` rows.
    ///
    /// If `n >= num_rows`, all rows are returned. The schema is preserved.
    ///
    /// # Cost
    ///
    /// GPU-allocating: always produces a new table with gathered rows via FFI.
    /// GPU allocation dominates runtime cost, so a consuming variant would
    /// provide negligible benefit.
    pub fn tail(&self, n: usize) -> Result<Self> {
        let total = self.num_rows();
        let take = n.min(total);
        let start = total - take;
        self.gather_range(start, take)
    }

    /// Return `length` rows starting at `offset`.
    ///
    /// Both `offset` and `length` are clamped to the number of rows, so
    /// out-of-range values never produce an error — they simply yield fewer
    /// (or zero) rows. The schema is preserved.
    pub fn slice(&self, offset: usize, length: usize) -> Result<Self> {
        let total = self.num_rows();
        let start = offset.min(total);
        let take = length.min(total - start);
        self.gather_range(start, take)
    }

    /// Gather (select) rows by an index `Series`.
    ///
    /// Produces a new `DataFrame` whose row `i` equals `self[indices[i]]`.
    /// The `indices` series must be `Int32`. Out-of-bounds indices produce
    /// null rows (uses [`copying::gather`] which nullifies out-of-bounds indices).
    /// The schema is preserved.
    ///
    /// Advanced low-level workflows should use [`DataFrame::table_view`]
    /// together with [`crate::ops::copying::gather`] directly.
    ///
    /// # Errors
    ///
    /// Returns an error if `indices` is not `Int32`.
    pub fn gather(&self, indices: &Series) -> Result<Self> {
        let cv = indices.as_column_view()?;
        let output_domain = Arc::clone(&self.domain);
        self.gather_by_column_view_impl(
            &cv,
            output_domain.explicit_exec_options_raw(),
            output_domain,
            "dataframe::gather",
        )
    }

    // -- Filter, sort, deduplicate -------------------------------------------

    /// Filter rows using a `Series` boolean mask (cuDF-style high-level API).
    ///
    /// Rows where `mask` is `true` are kept; `false` or null rows are
    /// dropped. The schema is preserved.
    ///
    /// Advanced low-level workflows should use [`DataFrame::table_view`]
    /// together with [`crate::ops::filter::apply_boolean_mask`] directly.
    ///
    /// # Errors
    ///
    /// - `TypeError` if `mask` dtype is not `Bool8`.
    /// - `InvalidOperation` if `mask` length does not match the DataFrame row count.
    pub fn filter(&self, mask: &Series) -> Result<Self> {
        let mask_view = self.validate_series_mask(mask)?;
        let output_domain = Arc::clone(&self.domain);
        self.filter_by_mask_impl(
            &mask_view,
            output_domain.explicit_exec_options_raw(),
            output_domain,
            "dataframe::filter",
        )
    }

    /// Filter rows to those where `column` values appear in a `Series`.
    ///
    /// Convenience wrapper around [`crate::ops::search::contains`] +
    /// [`DataFrame::filter`], equivalent to pandas `df[df[column].isin(values)]`.
    ///
    /// `values` is the haystack (the set to test membership against).
    /// The column and values must have the same data type.
    ///
    /// Advanced low-level workflows should use [`DataFrame::table_view`]
    /// together with [`crate::ops::search`] and [`crate::ops::filter`]
    /// directly instead of going through a `DataFrame` method that accepts
    /// raw borrowed views.
    ///
    /// # Errors
    ///
    /// Returns an error if `column` is not in the schema, or if the
    /// `contains` / `filter` operation fails.
    pub fn filter_isin(&self, column: &str, values: &Series) -> Result<Self> {
        let operation = "dataframe::filter_isin";
        let cv = values.as_column_view()?;
        let col = self.columns.get(column).ok_or_else(|| missing_column_error(operation, column))?;
        let col_view = col.view()?;
        let output_domain = Arc::clone(&self.domain);
        ExecutionDomain::require_same_device_for_mixed_inputs(&self.domain, cv.execution_domain(), operation)?;
        if !ExecutionDomain::same_provenance(cv.execution_domain(), &output_domain) {
            cv.execution_domain().fence_stream(operation)?;
        }
        let exec_options = output_domain.explicit_exec_options_raw();
        let mask = ops::search::contains_impl(&cv, &col_view, exec_options.as_ref(), operation)?
            .with_execution_domain(Arc::clone(&output_domain));
        let mask_view = mask.view()?;
        self.filter_by_mask_impl(&mask_view, output_domain.explicit_exec_options_raw(), output_domain, operation)
    }

    /// Validate that a `Series` is a valid boolean mask for this DataFrame.
    ///
    /// Returns the `ColumnView` if valid, or an appropriate error.
    fn validate_series_mask<'a>(&self, mask: &'a Series) -> Result<ColumnView<'a>> {
        let mask_view = mask.as_column_view()?;
        self.validate_mask_view(&mask_view, "dataframe::filter")?;
        Ok(mask_view)
    }

    /// Validate that a low-level column view is a valid boolean mask.
    fn validate_mask_view(&self, mask: &ColumnView<'_>, operation: &'static str) -> Result<()> {
        let mask_dtype = mask.data_type()?;
        if mask_dtype.id() != crate::core::types::TypeId::Bool8 {
            return Err(Error::new(
                ErrorKind::TypeError,
                format!("filter mask must be Bool8, got {:?}", mask_dtype.id()),
            )
            .with_operation(operation)
            .with_context("mask_dtype", format!("{:?}", mask_dtype.id())));
        }
        let mask_len = mask.len()?;
        if mask_len != self.row_count {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                format!("filter mask length ({}) does not match DataFrame row count ({})", mask_len, self.row_count),
            )
            .with_operation(operation)
            .with_context("mask_length", mask_len.to_string())
            .with_context("row_count", self.row_count.to_string()));
        }
        Ok(())
    }

    /// Sort rows using cuDF-style [`SortOptions`].
    ///
    /// This is the high-level equivalent of [`sort`](Self::sort), accepting
    /// column names and a `SortOptions` struct instead of raw `&[Order]`
    /// slices.
    ///
    /// # Option vector lengths
    ///
    /// `descending` and `nulls_first` vectors in `SortOptions` must be
    /// either empty (meaning defaults for all keys) or exactly equal in
    /// length to the number of `by` columns. Any other length produces an
    /// `InvalidOperation` error.
    ///
    /// # Cost
    ///
    /// Produces a new GPU-allocated output table.
    ///
    /// If `by` is exactly all columns in schema order, cuDF sorts the table
    /// directly.
    ///
    /// Otherwise, `sort_values()` builds a temporary borrowed keys view and
    /// calls the keyed sort primitive. This does not deep-copy the key
    /// columns, but it still allocates temporary view handles plus the sorted
    /// output.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cudf::{DataFrame, Series, SortOptions};
    ///
    /// let ids = Series::new::<i32>("id", &[3, 1, 2]).unwrap();
    /// let df = DataFrame::from_named_pairs([("id", ids)]).unwrap();
    ///
    /// let sorted = df
    ///     .sort_values(&["id"], SortOptions::new().with_descending(vec![true]))
    ///     .unwrap();
    /// ```
    pub fn sort_values<I, S>(&self, by: I, options: SortOptions) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.sort_values_impl(by, options, None, Arc::clone(&self.domain), "dataframe::sort_values")
    }

    fn sort_values_impl<I, S>(
        &self,
        by: I,
        options: SortOptions,
        exec_options: Option<cudf_sys::cudf_exec_options_t>,
        output_domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        if !ExecutionDomain::same_provenance(&self.domain, &output_domain) {
            self.domain.fence_stream(operation)?;
        }

        let by: SmallVec<[String; SORT_KEY_INLINE_CAPACITY]> = by.into_iter().map(|s| s.as_ref().to_string()).collect();
        let by_refs: SmallVec<[&str; SORT_KEY_INLINE_CAPACITY]> = by.iter().map(|s| s.as_str()).collect();
        let key_indices = resolve_column_names(&self.columns, &by_refs, operation)?;
        let (orders, null_orders) = options.resolve(by_refs.len())?;

        let null_order_ref = null_orders.as_deref();

        // Check whether `by` names are exactly all columns in table order.
        let all_in_order =
            key_indices.len() == self.columns.len() && key_indices.iter().enumerate().all(|(i, &idx)| idx == i);

        let tv = self.build_table_view()?;

        let result_table = if all_in_order {
            if let Some(exec_options) = exec_options.as_ref() {
                sorting::sort_table_impl(
                    tv.table(),
                    Some(&orders),
                    null_order_ref,
                    options.stable,
                    Some(exec_options),
                    operation,
                )?
            } else {
                let sort_fn = if options.stable { sorting::stable_sort } else { sorting::sort };
                sort_fn(tv.table(), Some(&orders), null_order_ref)?
            }
        } else {
            // Project the sort keys directly from the already-built full
            // table view, then reorder the full table according to those keys.
            let keys_table = tv.table().select_columns(&key_indices)?;
            if let Some(exec_options) = exec_options.as_ref() {
                sorting::sort_by_key_impl(
                    tv.table(),
                    &keys_table,
                    Some(&orders),
                    null_order_ref,
                    options.stable,
                    Some(exec_options),
                    operation,
                )?
            } else {
                let sort_by_key_fn = if options.stable { sorting::stable_sort_by_key } else { sorting::sort_by_key };
                sort_by_key_fn(tv.table(), &keys_table, Some(&orders), null_order_ref)?
            }
        }
        .with_execution_domain(output_domain);

        Self::from_table_with_names(result_table, self.column_names_owned())
    }

    /// Return the `n` rows with the largest values in `column`.
    ///
    /// Equivalent to `self.sort_values(&[column], SortOptions::new().with_descending(vec![true]))?.head(n)`.
    ///
    /// # Errors
    ///
    /// Returns an error if `column` is not in the schema.
    pub fn nlargest(&self, n: usize, column: &str) -> Result<Self> {
        self.sort_values([column], SortOptions::new().with_descending(vec![true]))?.head(n)
    }

    /// Return the `n` rows with the smallest values in `column`.
    ///
    /// Equivalent to `self.sort_values(&[column], SortOptions::new())?.head(n)`.
    ///
    /// # Errors
    ///
    /// Returns an error if `column` is not in the schema.
    pub fn nsmallest(&self, n: usize, column: &str) -> Result<Self> {
        self.sort_values([column], SortOptions::new())?.head(n)
    }

    /// Remove duplicate rows based on a subset of columns (internal).
    ///
    /// The order of rows in the output is **not** guaranteed. The full
    /// schema (all columns) is preserved.
    ///
    /// This is the low-level implementation behind [`drop_duplicates`](Self::drop_duplicates).
    /// External callers should use `drop_duplicates` with [`DropDuplicatesOptions`] instead.
    pub(crate) fn distinct(&self, subset: &[&str], keep: DuplicateKeepOption) -> Result<Self> {
        let output_domain = Arc::clone(&self.domain);
        self.distinct_impl(
            subset,
            keep,
            output_domain.explicit_exec_options_raw(),
            output_domain,
            "dataframe::distinct",
        )
    }

    /// Drop rows where the specified columns contain nulls.
    ///
    /// A row is kept only if at least `keep_threshold` of the `subset`
    /// columns are non-null. For the common "drop any null" behavior, set
    /// `keep_threshold` equal to `subset.len()`.
    ///
    /// # Errors
    ///
    /// Returns an error if any name in `subset` is not in the schema.
    pub(crate) fn drop_nulls(&self, subset: &[&str], keep_threshold: i32) -> Result<Self> {
        let output_domain = Arc::clone(&self.domain);
        self.drop_nulls_impl(
            subset,
            keep_threshold,
            output_domain.explicit_exec_options_raw(),
            output_domain,
            "dataframe::drop_nulls",
        )
    }

    /// Concatenate two DataFrames row-wise.
    ///
    /// Both DataFrames must have the same column names in the same order
    /// and matching column types.
    ///
    /// # Errors
    ///
    /// Returns an error if the schemas do not match (different names, count,
    /// order, or types).
    pub fn concat(&self, other: &DataFrame) -> Result<Self> {
        self.concat_many_impl(&[other], None, None, "dataframe::concat")
    }

    /// Concatenate multiple DataFrames row-wise.
    ///
    /// All DataFrames must have the same column names in the same order
    /// and matching column types. An empty `others` slice returns a clone
    /// of `self`.
    ///
    /// This is more efficient than chaining multiple [`concat`](Self::concat)
    /// calls because it passes all table views to a single FFI call.
    ///
    /// # Errors
    ///
    /// Returns an error if any DataFrame's schema does not match `self`
    /// (different names, count, order, or types).
    pub fn concat_many(&self, others: &[&DataFrame]) -> Result<Self> {
        self.concat_many_impl(others, None, None, "dataframe::concat_many")
    }

    /// Concatenate two DataFrames row-wise using a borrowed per-operation execution override.
    pub(crate) fn concat_in_context(&self, other: &DataFrame, ctx: &ExecutionContext<'_>) -> Result<Self> {
        self.concat_many_in_context(&[other], ctx)
    }

    /// Concatenate multiple DataFrames row-wise using a borrowed per-operation execution override.
    pub(crate) fn concat_many_in_context(&self, others: &[&DataFrame], ctx: &ExecutionContext<'_>) -> Result<Self> {
        let output_domain = ctx.resolve_output_domain(&self.domain)?;
        let operation = "dataframe::in_context::concat_many";
        // Fence self if needed.
        if !ExecutionDomain::same_provenance(&self.domain, &output_domain) {
            self.domain.fence_stream(operation)?;
        }
        // Fence each other if needed.
        for other in others {
            if !ExecutionDomain::same_provenance(&other.domain, &output_domain) {
                other.domain.fence_stream(operation)?;
            }
        }
        self.concat_many_impl(others, output_domain.explicit_exec_options_raw(), Some(output_domain), operation)
    }

    fn concat_many_impl(
        &self,
        others: &[&DataFrame],
        exec_options: Option<cudf_sys::cudf_exec_options_t>,
        forced_output_domain: Option<Arc<ExecutionDomain>>,
        operation: &'static str,
    ) -> Result<Self> {
        if others.is_empty() {
            return Ok(self.clone());
        }

        let output_domain = forced_output_domain.unwrap_or_else(|| Arc::clone(&self.domain));
        let exec_options = exec_options.or_else(|| output_domain.explicit_exec_options_raw());

        // Validate device compatibility and fence streams for all inputs.
        for other in others {
            ExecutionDomain::require_same_device_for_mixed_inputs(&self.domain, &other.domain, operation)?;
        }
        if !ExecutionDomain::same_provenance(&self.domain, &output_domain) {
            self.domain.fence_stream(operation)?;
        }
        for other in others {
            if !ExecutionDomain::same_provenance(&other.domain, &output_domain) {
                other.domain.fence_stream(operation)?;
            }
        }

        // Validate schemas match: same names, same order, same types.
        for other in others {
            self.validate_concat_schema(other, operation)?;
        }

        // Build all table views and pass them in a single FFI call.
        let self_tv = self.build_table_view()?;
        let other_tvs: Vec<_> = others.iter().map(|df| df.build_table_view()).collect::<Result<_>>()?;

        let mut table_refs: Vec<&crate::core::table::TableView<'_>> = Vec::with_capacity(1 + others.len());
        table_refs.push(self_tv.table());
        for tv in &other_tvs {
            table_refs.push(tv.table());
        }

        let result_table = ops::concatenate::concatenate_tables_with_exec(&table_refs, exec_options.as_ref())?
            .with_execution_domain(output_domain);

        Self::from_table_with_names(result_table, self.column_names_owned())
    }

    /// Validate that `other` has the same schema (column count, names, types) as `self`.
    fn validate_concat_schema(&self, other: &DataFrame, operation: &'static str) -> Result<()> {
        if self.columns.len() != other.columns.len() {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "cannot concatenate DataFrames with different column counts",
            )
            .with_operation(operation)
            .with_context("left_columns", self.columns.len().to_string())
            .with_context("right_columns", other.columns.len().to_string()));
        }

        for (left_name, right_name) in self.columns.keys().zip(other.columns.keys()) {
            if left_name != right_name {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    format!("schema mismatch: column '{}' vs '{}'", left_name, right_name),
                )
                .with_operation(operation)
                .with_context("left_column", left_name.clone())
                .with_context("right_column", right_name.clone()));
            }
        }

        for (i, ((name, left_col), (_, right_col))) in self.columns.iter().zip(other.columns.iter()).enumerate() {
            let left_type = left_col.view()?.data_type()?.id();
            let right_type = right_col.view()?.data_type()?.id();
            if left_type != right_type {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    "cannot concatenate DataFrames with mismatched column types",
                )
                .with_operation(operation)
                .with_context("column_index", i.to_string())
                .with_context("column_name", name.to_string())
                .with_context("left_type_id", format!("{:?}", left_type).to_lowercase())
                .with_context("right_type_id", format!("{:?}", right_type).to_lowercase()));
            }
        }

        Ok(())
    }
    /// Return a new DataFrame with a single column added or replaced.
    ///
    /// The `name` parameter controls the column name in the resulting
    /// DataFrame. The input's own [`Series::name()`] is **ignored** —
    /// DataFrame is the schema authority for column naming.
    ///
    /// Use [`assign`](Self::assign) for batch addition of multiple columns,
    /// or [`with_columns`](Self::with_columns) for expression-based computed
    /// columns.
    ///
    /// # Cost
    /// Unchanged columns are shared via `Arc::clone` — no GPU copy.
    ///
    /// # Errors
    /// Returns an error if the new column's row count does not match the
    /// DataFrame's existing row count.
    pub fn with_column(&self, name: &str, col: impl IntoSeries) -> Result<Self> {
        let series: Series = col.into_series()?;
        self.validate_series_insertion_domain(&series, "dataframe::with_column", name)?;
        let col_rows = series.len();
        let expected_rows = if self.num_columns() > 0 || self.row_count > 0 { Some(self.row_count) } else { None };

        if let Some(self_rows) = expected_rows {
            if col_rows != self_rows {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    format!("new column has {} rows but DataFrame has {} rows", col_rows, self_rows),
                )
                .with_operation("dataframe::with_column")
                .with_context("column_name", name.to_string())
                .with_context("column_rows", col_rows.to_string())
                .with_context("dataframe_rows", self_rows.to_string()));
            }
        }

        let mut new_columns = self.columns.clone();
        new_columns.insert(name.to_string(), Arc::clone(series.column_arc()));

        let row_count = expected_rows.unwrap_or(col_rows);

        let domain = derive_frame_domain_from_columns(&new_columns, "dataframe::with_column")?;
        Ok(Self { columns: new_columns, row_count, domain })
    }

    /// Return a new DataFrame with multiple columns added or replaced from
    /// `(name, Series)` pairs.
    ///
    /// The pair key controls the column name in the resulting DataFrame. The
    /// input Series' own [`Series::name()`] is **ignored** — DataFrame is the
    /// schema authority for column naming.
    ///
    /// For each pair, if `name` already exists the column is replaced;
    /// otherwise it is appended. All unchanged columns are shared via
    /// `Arc::clone` without copying GPU data.
    ///
    /// # Duplicate name handling
    ///
    /// If the same output name appears more than once in `pairs`, the **last
    /// value wins** — earlier entries for that name are silently overwritten.
    /// This matches the semantics of Python's `DataFrame.assign(**kwargs)` and
    /// is intentional: `assign` is a batch "set these columns" operation, not a
    /// schema-definition operation. Methods that define schema shape
    /// ([`select`](Self::select), [`rename_columns`](Self::rename_columns))
    /// reject duplicates with an error.
    ///
    /// If `self` has zero columns but a non-zero logical row count (for
    /// example after [`drop`](Self::drop) or [`select`](Self::select) removes
    /// every column), inserted Series must still match that row count. Only a
    /// truly empty `0 x 0` DataFrame lets the first inserted Series establish
    /// the row count.
    ///
    /// # Errors
    ///
    /// Returns an error if any Series has a different row count than the
    /// DataFrame.
    pub fn assign<I, S>(&self, pairs: I) -> Result<Self>
    where
        I: IntoIterator<Item = (S, Series)>,
        S: AsRef<str>,
    {
        let mut new_columns = self.columns.clone();
        let mut expected_rows = if self.num_columns() > 0 || self.row_count > 0 { Some(self.row_count) } else { None };

        for (name, series) in pairs {
            let name = name.as_ref();
            let series_rows = series.len();

            if let Some(self_rows) = expected_rows {
                if series_rows != self_rows {
                    return Err(Error::new(
                        ErrorKind::InvalidOperation,
                        format!("series '{}' has {} rows but DataFrame has {} rows", name, series_rows, self_rows),
                    )
                    .with_operation("dataframe::assign")
                    .with_context("column_name", name.to_string())
                    .with_context("series_rows", series_rows.to_string())
                    .with_context("dataframe_rows", self_rows.to_string()));
                }
            } else {
                expected_rows = Some(series_rows);
            }

            self.validate_series_insertion_domain(&series, "dataframe::assign", name)?;
            let column = Arc::clone(series.column_arc());

            new_columns.insert(name.to_string(), column);
        }

        let row_count = expected_rows.unwrap_or(self.row_count);
        let domain = derive_frame_domain_from_columns(&new_columns, "dataframe::assign")?;
        Ok(Self { columns: new_columns, row_count, domain })
    }

    /// Evaluate expressions against this DataFrame and add/replace the
    /// resulting columns.
    ///
    /// Use this for declarative computed columns. For pre-computed Series,
    /// use [`with_column`](Self::with_column) (single) or
    /// [`assign`](Self::assign) (batch).
    ///
    /// All expressions are evaluated against the **original** state of `self`
    /// — not an incrementally mutated copy. This ensures that expressions can
    /// reference columns that other expressions will overwrite, and the order
    /// of evaluation does not matter.
    ///
    /// Each item must resolve to an output name. You can provide either:
    ///
    /// - a bare expression, which uses [`Expr::output_name()`](crate::ops::expr::Expr::output_name)
    /// - an explicit `(name, expr)` pair, which always uses the provided name
    ///
    /// Bare expressions can still get their names via
    /// [`.alias()`](crate::ops::expr::Expr::alias), by being a bare
    /// [`col()`](crate::ops::expr::col) reference, or by being a single-input
    /// transform that inherits its input name (for example `cast`, `is_null`,
    /// `fillna`, cumulative scans, and string unary operations). Broadcast
    /// reductions do **not** inherit an output name; give them an alias or an
    /// explicit `(name, expr)` pair. Items without a resolvable name produce
    /// an error **before** any GPU work.
    ///
    /// # Errors
    ///
    /// Returns an error if any item lacks an output name, or if
    /// evaluation of any expression fails.
    pub fn with_columns<I, T>(&self, exprs: I) -> Result<Self>
    where
        I: IntoIterator<Item = T>,
        T: IntoWithColumnExpr,
    {
        let exprs: Vec<(Option<String>, ops::expr::Expr)> =
            exprs.into_iter().map(IntoWithColumnExpr::into_with_column_expr).collect();

        // Validate all names up-front before doing any GPU work.
        let names: Vec<String> = exprs
            .iter()
            .map(|(explicit_name, expr)| {
                explicit_name.as_deref().or_else(|| expr.output_name()).map(str::to_string).ok_or_else(|| {
                    Error::new(ErrorKind::InvalidOperation, "expression has no output name; use .alias() to name it")
                        .with_operation("dataframe::with_columns")
                })
            })
            .collect::<Result<_>>()?;

        // Evaluate all expressions against the original DataFrame.
        let results: Vec<Series> = exprs.iter().map(|(_, expr)| expr.eval(self)).collect::<Result<_>>()?;

        // Build the new DataFrame.
        let pairs: Vec<(&str, Series)> = names.iter().map(String::as_str).zip(results).collect();
        self.assign(pairs)
    }

    /// Return a new DataFrame with null values in the named column replaced
    /// by a scalar value.
    ///
    /// All other columns are preserved unchanged. The scalar type must be
    /// compatible with the column type.
    ///
    /// # Errors
    ///
    /// Returns an error if `column` is not in the schema.
    pub(crate) fn replace_nulls(&self, column: &str, value: &Scalar) -> Result<Self> {
        let operation = "dataframe::replace_nulls";
        let col = self.columns.get(column).ok_or_else(|| missing_column_error(operation, column))?;

        ExecutionDomain::require_same_device_and_fence_secondary(
            col.execution_domain(),
            value.execution_domain(),
            operation,
        )?;
        let cv = col.view()?;
        let exec_options = col.execution_domain().explicit_exec_options_raw();
        let new_col = replace::replace_nulls_with_exec_options(&cv, value, exec_options.as_ref(), operation)?
            .with_execution_domain(Arc::clone(col.execution_domain()));

        let mut new_columns = self.columns.clone();
        new_columns.insert(column.to_string(), Arc::new(new_col));

        let domain = derive_frame_domain_from_columns(&new_columns, operation)?;
        Ok(Self { columns: new_columns, row_count: self.row_count, domain })
    }

    /// Return a new DataFrame with the named column cast to a different type.
    ///
    /// All other columns are preserved unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error if `column` is not in the schema, or if the cast
    /// is unsupported.
    pub fn cast_column(&self, column: &str, target_type: impl Into<DataType>) -> Result<Self> {
        let target_type = target_type.into();
        let col = self.columns.get(column).ok_or_else(|| missing_column_error("dataframe::cast_column", column))?;

        let cv = col.view()?;
        let new_col = unary::cast(&cv, &target_type)?.with_execution_domain(Arc::clone(col.execution_domain()));

        let mut new_columns = self.columns.clone();
        new_columns.insert(column.to_string(), Arc::new(new_col));

        let domain = derive_frame_domain_from_columns(&new_columns, "dataframe::cast_column")?;
        Ok(Self { columns: new_columns, row_count: self.row_count, domain })
    }

    // -- High-level API: cuDF-style naming --------------------------------

    /// Drop rows containing null values.
    ///
    /// Equivalent to cuDF Python's `DataFrame.dropna()`.
    ///
    /// - `how=any` (default): drop rows where **any** subset column is null.
    /// - `how=all`: drop rows only when **all** subset columns are null.
    /// - `thresh`: if set, keep rows with at least `thresh` non-null subset values
    ///   (overrides `how`).
    /// - `subset`: columns to inspect; defaults to all columns.
    ///
    /// # Cost
    /// GPU-allocating: produces a new table with the surviving rows.
    ///
    /// # Errors
    ///
    /// Returns an error if any column in `subset` is not in the schema.
    pub fn dropna(&self, opts: DropNaOptions) -> Result<Self> {
        let subset_names: Vec<&str> =
            if opts.subset.is_empty() { self.column_names() } else { opts.subset.iter().map(|s| s.as_str()).collect() };

        let subset_len = i32::try_from(subset_names.len()).map_err(|_| {
            Error::new(ErrorKind::InvalidOperation, "subset length exceeds i32::MAX")
                .with_operation("dataframe::dropna")
        })?;

        let threshold = if let Some(t) = opts.thresh {
            t
        } else {
            match opts.how {
                DropNaHow::All => {
                    // "all": drop only when every subset column is null → keep if ≥1 non-null.
                    1
                },
                DropNaHow::Any => {
                    // "any": drop if any subset column is null → keep if all non-null.
                    subset_len
                },
            }
        };

        self.drop_nulls(&subset_names, threshold).map_err(|e| e.with_operation("dataframe::dropna"))
    }

    /// Drop rows containing null values using a borrowed per-operation execution override.
    pub(crate) fn dropna_in_context(&self, opts: DropNaOptions, ctx: &ExecutionContext<'_>) -> Result<Self> {
        let subset_names: Vec<&str> =
            if opts.subset.is_empty() { self.column_names() } else { opts.subset.iter().map(|s| s.as_str()).collect() };

        let subset_len = i32::try_from(subset_names.len()).map_err(|_| {
            Error::new(ErrorKind::InvalidOperation, "subset length exceeds i32::MAX")
                .with_operation("dataframe::in_context::dropna")
        })?;

        let threshold = if let Some(t) = opts.thresh {
            t
        } else {
            match opts.how {
                DropNaHow::All => 1,
                DropNaHow::Any => subset_len,
            }
        };

        let output_domain = ctx.resolve_output_domain(&self.domain)?;
        self.drop_nulls_impl(
            &subset_names,
            threshold,
            output_domain.explicit_exec_options_raw(),
            output_domain,
            "dataframe::in_context::dropna",
        )
    }

    /// Remove duplicate rows.
    ///
    /// Equivalent to cuDF Python's `DataFrame.drop_duplicates()`.
    ///
    /// - `subset`: columns to use as deduplication keys; defaults to all columns.
    /// - `keep`: which occurrence to keep
    ///   ([`DuplicateKeepOption::KeepAny`] by default).
    ///
    /// # Cost
    /// GPU-allocating: produces a new table with duplicates removed.
    ///
    /// # Errors
    ///
    /// Returns an error if any column in `subset` is not in the schema.
    pub fn drop_duplicates(&self, opts: DropDuplicatesOptions) -> Result<Self> {
        let subset_names: Vec<&str> =
            if opts.subset.is_empty() { self.column_names() } else { opts.subset.iter().map(|s| s.as_str()).collect() };

        self.distinct(&subset_names, opts.keep).map_err(|e| e.with_operation("dataframe::drop_duplicates"))
    }

    /// Remove duplicate rows using a borrowed per-operation execution override.
    pub(crate) fn drop_duplicates_in_context(
        &self,
        opts: DropDuplicatesOptions,
        ctx: &ExecutionContext<'_>,
    ) -> Result<Self> {
        let subset_names: Vec<&str> =
            if opts.subset.is_empty() { self.column_names() } else { opts.subset.iter().map(|s| s.as_str()).collect() };

        let output_domain = ctx.resolve_output_domain(&self.domain)?;
        self.distinct_impl(
            &subset_names,
            opts.keep,
            output_domain.explicit_exec_options_raw(),
            output_domain,
            "dataframe::in_context::drop_duplicates",
        )
    }

    /// Replace null values in a single column with a plain Rust value.
    ///
    /// Equivalent to cuDF Python's `DataFrame.fillna({column: value})`.
    ///
    /// Returns a new DataFrame with the named column's nulls replaced.
    /// All other columns are preserved unchanged by sharing their existing
    /// `Arc<Column>` handles.
    ///
    /// # Cost
    /// GPU-allocating: a new column is produced for the filled column.
    ///
    /// # Errors
    ///
    /// Returns an error if `column` is not in the schema.
    pub fn fillna<T: CudfType>(&self, column: &str, value: T) -> Result<Self> {
        let scalar = Scalar::from_value_in_domain(value, self.execution_domain())?;
        self.fillna_with_scalar(column, &scalar)
    }

    /// Replace null values in a single column with a pre-constructed [`Scalar`].
    ///
    /// This is the advanced entry point. Prefer [`fillna`](Self::fillna)
    /// with a plain Rust value for the cuDF-style API.
    ///
    /// Returns a new DataFrame with the named column's nulls replaced.
    /// All other columns are preserved unchanged by sharing their existing
    /// `Arc<Column>` handles.
    ///
    /// # Cost
    /// GPU-allocating: a new column is produced for the filled column.
    ///
    /// # Errors
    ///
    /// Returns an error if `column` is not in the schema.
    pub fn fillna_with_scalar(&self, column: &str, value: &Scalar) -> Result<Self> {
        self.replace_nulls(column, value).map_err(|e| e.with_operation("dataframe::fillna_with_scalar"))
    }

    /// Replace null values in a single column using corresponding elements
    /// from a [`Series`].
    ///
    /// Both the target column and `fill_series` must have compatible types and
    /// matching lengths.
    ///
    /// Returns a new DataFrame with the named column's nulls replaced.
    /// All other columns are preserved unchanged by sharing their existing
    /// `Arc<Column>` handles.
    ///
    /// # Cost
    /// GPU-allocating: a new column is produced for the filled column.
    ///
    /// # Errors
    ///
    /// Returns an error if `column` is not in the schema, or if the types
    /// or lengths are incompatible.
    pub fn fillna_with_series(&self, column: &str, fill_series: &Series) -> Result<Self> {
        let operation = "dataframe::fillna_with_series";
        let col = self.columns.get(column).ok_or_else(|| missing_column_error(operation, column))?;

        let output_domain = Arc::clone(col.execution_domain());
        let cv = col.view()?;
        let fill_cv = fill_series.as_column_view()?;
        ExecutionDomain::require_same_device_for_mixed_inputs(&output_domain, fill_cv.execution_domain(), operation)?;
        if !ExecutionDomain::same_provenance(fill_cv.execution_domain(), &output_domain) {
            fill_cv.execution_domain().fence_stream(operation)?;
        }
        let exec_options = output_domain.explicit_exec_options_raw();
        let new_col = replace::replace_nulls_column_with_exec_options(&cv, &fill_cv, exec_options.as_ref(), operation)?
            .with_execution_domain(output_domain);

        let mut new_columns = self.columns.clone();
        new_columns.insert(column.to_string(), Arc::new(new_col));

        let domain = derive_frame_domain_from_columns(&new_columns, operation)?;
        Ok(Self { columns: new_columns, row_count: self.row_count, domain })
    }
}

impl DataFrame {
    fn distinct_impl(
        &self,
        subset: &[&str],
        keep: DuplicateKeepOption,
        exec_options: Option<cudf_sys::cudf_exec_options_t>,
        output_domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Self> {
        if !ExecutionDomain::same_provenance(&self.domain, &output_domain) {
            self.domain.fence_stream(operation)?;
        }

        let key_indices = resolve_column_names(&self.columns, subset, operation)?;
        let tv = self.build_table_view()?;
        let result_table = stream_compaction::distinct_impl(
            tv.table(),
            &key_indices,
            keep,
            NullEquality::Equal,
            NanEquality::AllEqual,
            exec_options.as_ref(),
            operation,
        )?
        .with_execution_domain(output_domain);
        Self::from_table_with_names(result_table, self.column_names_owned())
    }

    fn drop_nulls_impl(
        &self,
        subset: &[&str],
        keep_threshold: i32,
        exec_options: Option<cudf_sys::cudf_exec_options_t>,
        output_domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Self> {
        if !ExecutionDomain::same_provenance(&self.domain, &output_domain) {
            self.domain.fence_stream(operation)?;
        }

        let key_indices = resolve_column_names(&self.columns, subset, operation)?;
        let tv = self.build_table_view()?;
        let result_table = stream_compaction::drop_nulls_impl(
            tv.table(),
            &key_indices,
            keep_threshold,
            exec_options.as_ref(),
            operation,
        )?
        .with_execution_domain(output_domain);
        Self::from_table_with_names(result_table, self.column_names_owned())
    }

    fn gather_by_column_view_impl(
        &self,
        indices: &ColumnView<'_>,
        exec_options: Option<cudf_sys::cudf_exec_options_t>,
        output_domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Self> {
        ExecutionDomain::require_same_device_for_mixed_inputs(&self.domain, indices.execution_domain(), operation)?;
        if !ExecutionDomain::same_provenance(&self.domain, &output_domain) {
            self.domain.fence_stream(operation)?;
        }
        if !ExecutionDomain::same_provenance(indices.execution_domain(), &output_domain) {
            indices.execution_domain().fence_stream(operation)?;
        }

        if indices.is_empty()? {
            return self.empty_with_same_schema_in_domain(output_domain);
        }
        let tv = self.build_table_view()?;
        let result_table = copying::gather_impl(
            tv.table(),
            indices,
            cudf_sys::cudf_out_of_bounds_policy_t::CUDF_OUT_OF_BOUNDS_POLICY_NULLIFY,
            "nullify",
            exec_options.as_ref(),
            operation,
        )?
        .with_execution_domain(output_domain);
        Self::from_table_with_names(result_table, self.column_names_owned())
    }

    fn filter_by_mask_impl(
        &self,
        mask: &ColumnView<'_>,
        exec_options: Option<cudf_sys::cudf_exec_options_t>,
        output_domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Self> {
        ExecutionDomain::require_same_device_for_mixed_inputs(&self.domain, mask.execution_domain(), operation)?;
        if !ExecutionDomain::same_provenance(&self.domain, &output_domain) {
            self.domain.fence_stream(operation)?;
        }
        if !ExecutionDomain::same_provenance(mask.execution_domain(), &output_domain) {
            mask.execution_domain().fence_stream(operation)?;
        }

        self.validate_mask_view(mask, operation)?;
        let tv = self.build_table_view()?;
        let result_table = filter::apply_boolean_mask_impl(tv.table(), mask, exec_options.as_ref(), operation)?
            .with_execution_domain(output_domain);
        Self::from_table_with_names(result_table, self.column_names_owned())
    }

    /// Internal: gather a contiguous range of rows into a new `DataFrame`,
    /// preserving the schema.
    fn gather_range(&self, start: usize, len: usize) -> Result<Self> {
        if len == 0 {
            return self.empty_with_same_schema();
        }

        let output_domain = Arc::clone(&self.domain);
        self.gather_range_impl(
            start,
            len,
            output_domain.explicit_exec_options_raw(),
            output_domain,
            "dataframe::gather_range",
        )
    }

    fn gather_range_in_context(
        &self,
        start: usize,
        len: usize,
        ctx: &ExecutionContext<'_>,
        operation: &'static str,
    ) -> Result<Self> {
        let output_domain = ctx.resolve_output_domain(&self.domain)?;
        self.gather_range_impl(start, len, output_domain.explicit_exec_options_raw(), output_domain, operation)
    }

    fn gather_range_impl(
        &self,
        start: usize,
        len: usize,
        exec_options: Option<cudf_sys::cudf_exec_options_t>,
        output_domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Self> {
        if len == 0 {
            return self.empty_with_same_schema_in_domain(output_domain);
        }

        let end = start + len; // callers guarantee no overflow
        let start_i32: i32 = start.try_into().map_err(|_| {
            Error::new(ErrorKind::InvalidOperation, "gather_range: start exceeds i32::MAX").with_operation(operation)
        })?;
        let end_i32: i32 = end.try_into().map_err(|_| {
            Error::new(ErrorKind::InvalidOperation, "gather_range: start + len exceeds i32::MAX")
                .with_operation(operation)
        })?;
        let indices: Vec<i32> = (start_i32..end_i32).collect();
        let map = Column::from_slice_in_domain(&indices, &output_domain)?;
        let map_view = map.view()?;
        ExecutionDomain::require_same_device_for_mixed_inputs(&self.domain, map_view.execution_domain(), operation)?;
        if !ExecutionDomain::same_provenance(&self.domain, &output_domain) {
            self.domain.fence_stream(operation)?;
        }

        let tv = self.build_table_view()?;
        // Indices are constructed from `start..start+len` which the callers
        // have already validated against row_count, and the i32 conversion
        // above ensures no truncation, so DONT_CHECK is safe here.
        let result_table = copying::gather_impl(
            tv.table(),
            &map_view,
            cudf_sys::cudf_out_of_bounds_policy_t::CUDF_OUT_OF_BOUNDS_POLICY_DONT_CHECK,
            "dont_check",
            exec_options.as_ref(),
            operation,
        )?
        .with_execution_domain(output_domain);

        Self::from_table_with_names(result_table, self.column_names_owned())
    }

    /// Internal: build an empty DataFrame with the same schema (column names
    /// and types, zero rows).
    ///
    /// This helper is for row-space operations that produce no rows at all,
    /// such as empty gathers or zero-length slices. Unlike schema-only
    /// operations such as [`select`](Self::select) and [`drop`](Self::drop),
    /// it intentionally resets the logical row count to zero.
    fn empty_with_same_schema(&self) -> Result<Self> {
        self.empty_with_same_schema_in_domain(Arc::clone(&self.domain))
    }

    fn empty_with_same_schema_in_domain(&self, domain: Arc<ExecutionDomain>) -> Result<Self> {
        let mut new_columns = fast_index_map_with_capacity(self.columns.len());
        for (name, col) in &self.columns {
            let cv = col.view()?;
            let type_id = cv.data_type()?.id();
            new_columns
                .insert(name.clone(), Arc::new(Column::empty(type_id)?.with_execution_domain(Arc::clone(&domain))));
        }
        Ok(Self { columns: new_columns, row_count: 0, domain })
    }

    /// Unpivot a DataFrame from wide format to long format.
    ///
    /// Takes a set of id columns (kept as-is) and a set of value columns
    /// (turned into rows). Produces a DataFrame with columns:
    /// `[...id_vars, var_name, value_name]`.
    ///
    /// Equivalent to cuDF Python's `DataFrame.melt()` / `cudf.melt()`.
    ///
    /// # Algorithm (pure Rust, no C++ kernel)
    ///
    /// 1. Tile each id_var column K times (K = number of value_vars)
    /// 2. Build a `variable` string column by repeating value_var names
    /// 3. Concatenate all value_var columns into a single `value` column
    ///
    /// # Cost
    /// GPU-allocating: creates K copies of each id column and concatenates
    /// the value columns.
    pub fn melt(&self, opts: MeltOptions) -> Result<Self> {
        let id_vars = &opts.id_vars;
        let value_vars: Vec<String> = if opts.value_vars.is_empty() {
            let id_set: FastHashSet<&str> = id_vars.iter().map(|s| s.as_str()).collect();
            self.columns.keys().filter(|name| !id_set.contains(name.as_str())).cloned().collect()
        } else {
            opts.value_vars.clone()
        };

        // Validate: id_vars must exist
        for name in id_vars {
            if !self.columns.contains_key(name) {
                return Err(missing_column_error("dataframe::melt", name));
            }
        }
        // Validate: value_vars must exist
        for name in &value_vars {
            if !self.columns.contains_key(name) {
                return Err(missing_column_error("dataframe::melt", name));
            }
        }
        // Validate: id_vars and value_vars must not overlap (L5)
        let id_set: FastHashSet<&str> = id_vars.iter().map(|s| s.as_str()).collect();
        for name in &value_vars {
            if id_set.contains(name.as_str()) {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    format!("column '{name}' appears in both id_vars and value_vars"),
                )
                .with_operation("dataframe::melt")
                .with_context("column_name", name.clone()));
            }
        }
        // Validate: var_name and value_name must not collide with id_vars (L6)
        if id_set.contains(opts.var_name.as_str()) {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                format!("var_name '{}' collides with an id_var column", opts.var_name),
            )
            .with_operation("dataframe::melt")
            .with_context("var_name", opts.var_name.clone()));
        }
        if id_set.contains(opts.value_name.as_str()) {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                format!("value_name '{}' collides with an id_var column", opts.value_name),
            )
            .with_operation("dataframe::melt")
            .with_context("value_name", opts.value_name.clone()));
        }

        let n = self.row_count;
        let k = value_vars.len();

        let mut result_columns: Vec<(&str, Series)> = Vec::new();

        // Step 1: Tile id columns K times
        for id_name in id_vars {
            let id_series = self.series(id_name)?;
            if k == 0 {
                let empty = Series::from_column(
                    Column::empty(id_series.dtype()?.id())?.with_execution_domain(Arc::clone(&self.domain)),
                )?
                .rename(id_name);
                result_columns.push((id_name, empty));
            } else if k == 1 {
                result_columns.push((id_name, id_series));
            } else {
                let copies: Vec<&Series> = (0..k).map(|_| &id_series).collect();
                let tiled = Series::concat(&copies)?;
                result_columns.push((id_name, tiled));
            }
        }

        // Step 2: Build the "variable" string column
        // Each value_var name is repeated N times, in order
        let var_strings: Vec<&str> = value_vars.iter().flat_map(|name| std::iter::repeat_n(name.as_str(), n)).collect();
        let var_series =
            Series::from_column(Column::from_strs_in_domain(&var_strings, &self.domain)?)?.rename(&opts.var_name);
        result_columns.push((&opts.var_name, var_series));

        // Step 3: Concatenate value columns into a single "value" column
        if value_vars.is_empty() {
            // No value vars → empty value column. Use Int32 as default type.
            let empty_series = Series::from_column(
                Column::empty(crate::TypeId::Int32)?.with_execution_domain(Arc::clone(&self.domain)),
            )?
            .rename(&opts.value_name);
            result_columns.push((&opts.value_name, empty_series));
        } else {
            let value_series_list: Vec<Series> = value_vars
                .iter()
                .map(|name| {
                    let s = self.series(name)?;
                    Ok(s.rename(&opts.value_name))
                })
                .collect::<Result<Vec<_>>>()?;
            let refs: Vec<&Series> = value_series_list.iter().collect();
            let concat_values = Series::concat(&refs)?;
            result_columns.push((&opts.value_name, concat_values));
        }

        DataFrame::from_named_pairs(result_columns)
    }
}
