use typed_builder::TypedBuilder;

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
    /// If `true`, drop only when **all** subset columns are null.
    /// If `false` (default), drop when **any** subset column is null.
    #[builder(default)]
    how_all: bool,
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
pub struct DistinctOptions {
    /// Column names to use as deduplication keys. Empty means all columns.
    #[builder(default, setter(transform = |columns: &[&str]| columns.iter().map(|s| s.to_string()).collect()))]
    subset: Vec<String>,
    /// Which duplicate occurrence to keep.
    #[builder(default = DuplicateKeepOption::KeepAny)]
    keep: DuplicateKeepOption,
}

impl Default for DistinctOptions {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl DistinctOptions {
    /// Create default options: all columns, keep any.
    pub fn new() -> Self {
        Self::default()
    }
}

use std::sync::Arc;

use indexmap::IndexMap;
use smallvec::SmallVec;

use super::owned::{DataFrame, derive_frame_domain_from_columns, resolve_frame_domain_from_authoritative_source};
use crate::{
    core::{
        column::{Column, ColumnView},
        error::{Error, ErrorKind, Result},
        scalar::Scalar,
        types::{CudfType, DataType},
    },
    frame::series::{IntoSeries, Series, SeriesView},
    internal::resolve::resolve_column_names,
    ops::{
        self, copying, filter, replace,
        sorting::{self, SortOptions},
        stream_compaction::{self, DuplicateKeepOption, NanEquality, NullEquality},
        unary,
    },
    runtime::{Runtime, execution_context::ExecutionContext, execution_domain::ExecutionDomain},
};

const SORT_KEY_INLINE_CAPACITY: usize = 4;

impl DataFrame {
    /// Sort using a borrowed per-operation execution override.
    pub fn sort_values_with_context<I, S>(
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
            "dataframe::sort_values_with_context",
        )
    }

    #[doc(hidden)]
    pub fn __debug_sort_values_with_execution_context<I, S>(
        &self,
        by: I,
        options: SortOptions,
        ctx: &ExecutionContext<'_>,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.sort_values_with_context(by, options, ctx)
    }

    /// Gather rows using a borrowed per-operation execution override.
    pub fn gather_with_context(&self, indices: &Series, ctx: &ExecutionContext<'_>) -> Result<Self> {
        let output_domain = ctx.resolve_output_domain(&self.domain)?;
        let cv = indices.as_column_view()?;
        self.gather_by_column_view_impl(
            &cv,
            output_domain.explicit_exec_options_raw(),
            output_domain,
            "dataframe::gather_with_context",
        )
    }

    #[doc(hidden)]
    pub fn __debug_gather_with_execution_context(&self, indices: &Series, ctx: &ExecutionContext<'_>) -> Result<Self> {
        self.gather_with_context(indices, ctx)
    }

    /// Filter rows using a borrowed per-operation execution override.
    pub fn filter_with_context(&self, mask: &Series, ctx: &ExecutionContext<'_>) -> Result<Self> {
        let output_domain = ctx.resolve_output_domain(&self.domain)?;
        let mask_view = self.validate_series_mask(mask)?;
        self.filter_by_mask_impl(
            &mask_view,
            output_domain.explicit_exec_options_raw(),
            output_domain,
            "dataframe::filter_with_context",
        )
    }

    #[doc(hidden)]
    pub fn __debug_filter_with_execution_context(&self, mask: &Series, ctx: &ExecutionContext<'_>) -> Result<Self> {
        self.filter_with_context(mask, ctx)
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
    /// # Errors
    ///
    /// Returns an error if `indices` is not `Int32`.
    pub fn gather(&self, indices: &Series) -> Result<Self> {
        let cv = indices.as_column_view()?;
        self.gather_by_column_view(&cv)
    }

    /// Gather (select) rows by a low-level index `ColumnView`.
    ///
    /// This is the advanced entry point. Prefer [`gather`](Self::gather)
    /// with a `Series` for the cuDF-style API.
    ///
    /// # Errors
    ///
    /// Returns an error if `indices` is not `Int32`.
    pub fn gather_by_column_view(&self, indices: &ColumnView<'_>) -> Result<Self> {
        let output_domain = Arc::clone(&self.domain);
        self.gather_by_column_view_impl(
            indices,
            output_domain.explicit_exec_options_raw(),
            output_domain,
            "dataframe::gather_by_column_view",
        )
    }

    // -- Filter, sort, deduplicate -------------------------------------------

    /// Filter rows using a `Series` boolean mask (cuDF-style high-level API).
    ///
    /// Rows where `mask` is `true` are kept; `false` or null rows are
    /// dropped. The schema is preserved.
    ///
    /// # Errors
    ///
    /// - `TypeError` if `mask` dtype is not `Bool8`.
    /// - `InvalidOperation` if `mask` length does not match the DataFrame row count.
    pub fn filter(&self, mask: &Series) -> Result<Self> {
        let mask_view = self.validate_series_mask(mask)?;
        self.filter_by_mask(&mask_view)
    }

    /// Filter rows using a low-level `ColumnView` boolean mask.
    ///
    /// This is the low-level entry point. Prefer [`filter`](Self::filter)
    /// with a `Series` mask for the cuDF-style API.
    ///
    /// # Errors
    ///
    /// Returns an error if `mask` is not `Bool8`, or if its row count
    /// does not match the DataFrame's row count.
    ///
    /// # Cost
    ///
    /// GPU-allocating: always produces a new filtered table via FFI.
    /// GPU allocation dominates runtime cost, so a consuming variant would
    /// provide negligible benefit.
    pub fn filter_by_mask(&self, mask: &ColumnView<'_>) -> Result<Self> {
        let output_domain = Arc::clone(&self.domain);
        self.filter_by_mask_impl(
            mask,
            output_domain.explicit_exec_options_raw(),
            output_domain,
            "dataframe::filter_by_mask",
        )
    }

    /// Filter rows to those where `column` values appear in a `Series`.
    ///
    /// Convenience wrapper around [`crate::ops::search::contains`] +
    /// [`DataFrame::filter_by_mask`], equivalent to pandas `df[df[column].isin(values)]`.
    ///
    /// `values` is the haystack (the set to test membership against).
    /// The column and values must have the same data type.
    ///
    /// # Errors
    ///
    /// Returns an error if `column` is not in the schema, or if the
    /// `contains` / `filter` operation fails.
    pub fn filter_isin(&self, column: &str, values: &Series) -> Result<Self> {
        let cv = values.as_column_view()?;
        self.filter_isin_by_column_view(column, &cv)
    }

    /// Filter rows to those where `column` values appear in `values`.
    ///
    /// This is the advanced entry point that accepts a raw `ColumnView`.
    /// Prefer [`filter_isin`](Self::filter_isin) with a `Series` for the
    /// cuDF-style API.
    ///
    /// # Errors
    ///
    /// Returns an error if `column` is not in the schema, or if the
    /// `contains` / `filter` operation fails.
    pub fn filter_isin_by_column_view(&self, column: &str, values: &ColumnView<'_>) -> Result<Self> {
        let col_view = self.series_view(column)?;
        let output_domain = Arc::clone(&self.domain);
        if !ExecutionDomain::same_provenance(values.execution_domain(), &output_domain) {
            values.execution_domain().fence_stream("dataframe::filter_isin_by_column_view")?;
        }
        let exec_options = output_domain.explicit_exec_options_raw();
        let mask = ops::search::contains_with_exec_options(
            values,
            &col_view,
            exec_options.as_ref(),
            "dataframe::filter_isin_by_column_view",
        )?
        .with_execution_domain(Arc::clone(&output_domain));
        let mask_view = mask.view()?;
        self.filter_by_mask_impl(
            &mask_view,
            output_domain.explicit_exec_options_raw(),
            output_domain,
            "dataframe::filter_isin_by_column_view",
        )
    }

    /// Validate that a `Series` is a valid boolean mask for this DataFrame.
    ///
    /// Returns the `ColumnView` if valid, or an appropriate error.
    fn validate_series_mask<'a>(&self, mask: &'a Series) -> Result<SeriesView<'a>> {
        let mask_view = mask.view()?;
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
    /// let df = DataFrame::from_columns([("id", ids)]).unwrap();
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
                sorting::sort_with_exec_options(
                    tv.table(),
                    Some(&orders),
                    null_order_ref,
                    options.stable,
                    exec_options,
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
                sorting::sort_by_key_with_exec_options(
                    tv.table(),
                    &keys_table,
                    Some(&orders),
                    null_order_ref,
                    options.stable,
                    exec_options,
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
    /// External callers should use `drop_duplicates` with [`DistinctOptions`] instead.
    pub(crate) fn distinct(&self, subset: &[&str], keep: DuplicateKeepOption) -> Result<Self> {
        let key_indices = resolve_column_names(&self.columns, subset, "dataframe::distinct")?;
        let key_i32: Vec<i32> = key_indices.iter().map(|&i| i as i32).collect();
        let tv = self.build_table_view()?;
        let result_table =
            stream_compaction::distinct(tv.table(), &key_i32, keep, NullEquality::Equal, NanEquality::AllEqual)?
                .with_execution_domain(Arc::clone(&self.domain));
        Self::from_table_with_names(result_table, self.column_names_owned())
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
    pub fn drop_nulls(&self, subset: &[&str], keep_threshold: i32) -> Result<Self> {
        let key_indices = resolve_column_names(&self.columns, subset, "dataframe::drop_nulls")?;
        let key_i32: Vec<i32> = key_indices.iter().map(|&i| i as i32).collect();
        let tv = self.build_table_view()?;
        let result_table = stream_compaction::drop_nulls(tv.table(), &key_i32, keep_threshold)?
            .with_execution_domain(Arc::clone(&self.domain));
        Self::from_table_with_names(result_table, self.column_names_owned())
    }

    /// Drop rows where **any** of the specified `subset` columns contain nulls.
    ///
    /// Equivalent to `self.drop_nulls(subset, subset.len() as i32)` — every
    /// column in `subset` must be non-null for a row to be kept.
    ///
    /// # Cost
    ///
    /// GPU-allocating: always produces a new table with nulls removed via FFI.
    /// GPU allocation dominates runtime cost, so a consuming variant would
    /// provide negligible benefit.
    ///
    /// # Errors
    ///
    /// Returns an error if any name in `subset` is not in the schema.
    pub fn drop_nulls_any(&self, subset: &[&str]) -> Result<Self> {
        let subset_count = subset.len();
        let keep_threshold = i32::try_from(subset_count).map_err(|_| {
            Error::new(ErrorKind::InvalidOperation, "subset length exceeds i32::MAX")
                .with_operation("dataframe::drop_nulls_any")
                .with_context("subset_count", subset_count.to_string())
        })?;

        self.drop_nulls(subset, keep_threshold).map_err(|e| {
            e.with_operation("dataframe::drop_nulls_any").with_context("subset_count", subset_count.to_string())
        })
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
        self.concat_impl(other, None, None, "dataframe::concat")
    }

    /// Concatenate two DataFrames row-wise into the given target runtime.
    pub fn concat_with_runtime(&self, other: &DataFrame, runtime: &Runtime) -> Result<Self> {
        let output_domain = Arc::clone(runtime.execution_domain());
        self.synchronize_for_forced_output_domain(other, &output_domain, "dataframe::concat_with_runtime")?;
        self.concat_impl(
            other,
            output_domain.explicit_exec_options_raw(),
            Some(output_domain),
            "dataframe::concat_with_runtime",
        )
    }

    /// Concatenate two DataFrames row-wise using a borrowed per-operation execution override.
    pub fn concat_with_context(&self, other: &DataFrame, ctx: &ExecutionContext<'_>) -> Result<Self> {
        let output_domain = ctx.resolve_output_domain(&self.domain)?;
        self.synchronize_for_forced_output_domain(other, &output_domain, "dataframe::concat_with_context")?;
        self.concat_impl(
            other,
            output_domain.explicit_exec_options_raw(),
            Some(output_domain),
            "dataframe::concat_with_context",
        )
    }

    fn concat_impl(
        &self,
        other: &DataFrame,
        exec_options: Option<cudf_sys::cudf_exec_options_t>,
        forced_output_domain: Option<Arc<ExecutionDomain>>,
        operation: &'static str,
    ) -> Result<Self> {
        let output_domain = forced_output_domain.unwrap_or_else(|| Arc::clone(&self.domain));
        let exec_options = exec_options.or_else(|| output_domain.explicit_exec_options_raw());

        if !ExecutionDomain::same_provenance(&self.domain, &output_domain) {
            self.domain.fence_stream(operation)?;
        }
        if !ExecutionDomain::same_provenance(&other.domain, &output_domain) {
            other.domain.fence_stream(operation)?;
        }

        // Validate schemas match: same names, same order.
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

        // Validate matching types column-by-column up front for clearer errors.
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

        let left_tv = self.build_table_view()?;
        let right_tv = other.build_table_view()?;
        let result_table = ops::concatenate::concatenate_tables_with_exec(
            &[left_tv.table(), right_tv.table()],
            exec_options.as_ref(),
        )?
        .with_execution_domain(output_domain);

        Self::from_table_with_names(result_table, self.column_names_owned())
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
        if self.columns.is_empty() {
            ExecutionDomain::require_same_provenance(
                &self.domain,
                series.as_column().execution_domain(),
                "dataframe::with_column",
            )?;
        }
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

    /// Return a new DataFrame with the given column added or replaced,
    /// copying inputs into the target runtime when needed.
    ///
    /// Any source column whose retained domain does not already match
    /// `runtime` is deep-copied into that runtime before reassembly. This
    /// makes mixed-domain recovery explicit for metadata-style operations that
    /// would otherwise only be able to Arc-share incompatible inputs.
    pub fn with_column_with_runtime(&self, name: &str, col: impl IntoSeries, runtime: &Runtime) -> Result<Self> {
        let operation = "dataframe::with_column_with_runtime";
        let series: Series = col.into_series()?;
        let col_rows = series.len();
        let expected_rows = if self.num_columns() > 0 || self.row_count > 0 { Some(self.row_count) } else { None };

        if let Some(self_rows) = expected_rows {
            if col_rows != self_rows {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    format!("new column has {} rows but DataFrame has {} rows", col_rows, self_rows),
                )
                .with_operation(operation)
                .with_context("column_name", name.to_string())
                .with_context("column_rows", col_rows.to_string())
                .with_context("dataframe_rows", self_rows.to_string()));
            }
        }

        let output_domain = Arc::clone(runtime.execution_domain());
        let mut new_columns = self.clone_or_copy_columns_into_domain(&output_domain, operation)?;
        let copied = Self::clone_or_copy_series_into_domain(&series, &output_domain, operation, name)?;
        new_columns.insert(name.to_string(), copied);

        let row_count = expected_rows.unwrap_or(col_rows);
        let domain = resolve_frame_domain_from_authoritative_source(&new_columns, &output_domain, operation)?;
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

            let column = if self.columns.is_empty() && self.domain.explicit_stream().is_some() {
                Self::clone_or_copy_series_into_domain(&series, &self.domain, "dataframe::assign", name)?
            } else {
                Arc::clone(series.column_arc())
            };

            new_columns.insert(name.to_string(), column);
        }

        let row_count = expected_rows.unwrap_or(self.row_count);
        let domain = derive_frame_domain_from_columns(&new_columns, "dataframe::assign")?;
        Ok(Self { columns: new_columns, row_count, domain })
    }

    /// Return a new DataFrame with multiple columns added or replaced,
    /// copying inputs into the target runtime when needed.
    ///
    /// Existing columns are preserved without copying GPU data when they
    /// already belong to `runtime`; otherwise they are deep-copied into that
    /// runtime before the final frame is assembled.
    pub fn assign_with_runtime<I, S>(&self, pairs: I, runtime: &Runtime) -> Result<Self>
    where
        I: IntoIterator<Item = (S, Series)>,
        S: AsRef<str>,
    {
        let operation = "dataframe::assign_with_runtime";
        let output_domain = Arc::clone(runtime.execution_domain());
        let mut new_columns = self.clone_or_copy_columns_into_domain(&output_domain, operation)?;
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
                    .with_operation(operation)
                    .with_context("column_name", name.to_string())
                    .with_context("series_rows", series_rows.to_string())
                    .with_context("dataframe_rows", self_rows.to_string()));
                }
            } else {
                expected_rows = Some(series_rows);
            }

            let copied = Self::clone_or_copy_series_into_domain(&series, &output_domain, operation, name)?;
            new_columns.insert(name.to_string(), copied);
        }

        let row_count = expected_rows.unwrap_or(self.row_count);
        let domain = resolve_frame_domain_from_authoritative_source(&new_columns, &output_domain, operation)?;
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
    /// Each expression must have a determinable output name: either via
    /// [`.alias()`](crate::ops::expr::Expr::alias) or by being a bare
    /// [`col()`](crate::ops::expr::col) reference. Expressions without a name
    /// produce an error **before** any GPU work.
    ///
    /// # Errors
    ///
    /// Returns an error if any expression lacks an output name, or if
    /// evaluation of any expression fails.
    pub fn with_columns<I>(&self, exprs: I) -> Result<Self>
    where I: IntoIterator<Item = ops::expr::Expr> {
        let exprs: Vec<ops::expr::Expr> = exprs.into_iter().collect();

        // Validate all names up-front before doing any GPU work.
        let names: Vec<&str> = exprs
            .iter()
            .map(|e| {
                e.output_name().ok_or_else(|| {
                    Error::new(ErrorKind::InvalidOperation, "expression has no output name; use .alias() to name it")
                        .with_operation("dataframe::with_columns")
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // Evaluate all expressions against the original DataFrame.
        let results: Vec<Series> = exprs.iter().map(|e| e.eval(self)).collect::<Result<Vec<_>>>()?;

        // Build the new DataFrame.
        let pairs: Vec<(&str, Series)> = names.into_iter().zip(results).collect();
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
    pub fn replace_nulls(&self, column: &str, value: &Scalar) -> Result<Self> {
        let col = self.columns.get(column).ok_or_else(|| {
            Error::new(ErrorKind::InvalidOperation, format!("column '{}' not found in schema", column))
                .with_operation("dataframe::replace_nulls")
                .with_context("column_name", column.to_string())
        })?;

        let cv = col.view()?;
        let exec_options = col.execution_domain().explicit_exec_options_raw();
        let new_col =
            replace::replace_nulls_with_exec_options(&cv, value, exec_options.as_ref(), "dataframe::replace_nulls")?
                .with_execution_domain(Arc::clone(col.execution_domain()));

        let mut new_columns = self.columns.clone();
        new_columns.insert(column.to_string(), Arc::new(new_col));

        let domain = derive_frame_domain_from_columns(&new_columns, "dataframe::replace_nulls")?;
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
    pub fn cast_column(&self, column: &str, target_type: &DataType) -> Result<Self> {
        let col = self.columns.get(column).ok_or_else(|| {
            Error::new(ErrorKind::InvalidOperation, format!("column '{}' not found in schema", column))
                .with_operation("dataframe::cast_column")
                .with_context("column_name", column.to_string())
        })?;

        let cv = col.view()?;
        let new_col = unary::cast(&cv, target_type)?.with_execution_domain(Arc::clone(col.execution_domain()));

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
        } else if opts.how_all {
            // "all": drop only when every subset column is null → keep if ≥1 non-null.
            1
        } else {
            // "any": drop if any subset column is null → keep if all non-null.
            subset_len
        };

        self.drop_nulls(&subset_names, threshold).map_err(|e| e.with_operation("dataframe::dropna"))
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
    pub fn drop_duplicates(&self, opts: DistinctOptions) -> Result<Self> {
        let subset_names: Vec<&str> =
            if opts.subset.is_empty() { self.column_names() } else { opts.subset.iter().map(|s| s.as_str()).collect() };

        self.distinct(&subset_names, opts.keep).map_err(|e| e.with_operation("dataframe::drop_duplicates"))
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
        let scalar = Scalar::from_value(value)?;
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
        let col = self.columns.get(column).ok_or_else(|| {
            Error::new(ErrorKind::InvalidOperation, format!("column '{}' not found in schema", column))
                .with_operation("dataframe::fillna_with_series")
                .with_context("column_name", column.to_string())
        })?;

        let output_domain = Arc::clone(col.execution_domain());
        let cv = col.view()?;
        let fill_cv = fill_series.as_column_view()?;
        let exec_options = output_domain.explicit_exec_options_raw();
        let new_col = replace::replace_nulls_column_with_exec_options(
            &cv,
            &fill_cv,
            exec_options.as_ref(),
            "dataframe::fillna_with_series",
        )?
        .with_execution_domain(output_domain);

        let mut new_columns = self.columns.clone();
        new_columns.insert(column.to_string(), Arc::new(new_col));

        let domain = derive_frame_domain_from_columns(&new_columns, "dataframe::fillna_with_series")?;
        Ok(Self { columns: new_columns, row_count: self.row_count, domain })
    }

    /// Replace null values in a column using a [`Series`], copying inputs
    /// into the target [`Runtime`]'s retained provenance when necessary.
    ///
    /// Unlike [`fillna_with_series`](Self::fillna_with_series), this method
    /// recovers from mixed retained provenance by deep-copying the target
    /// column and `fill_series` into the runtime's domain before executing
    /// the replace-nulls operation.
    ///
    /// # Cost
    /// GPU-allocating: a new column is produced. If any input is in a
    /// different domain, an additional device-to-device copy per mismatched
    /// column occurs.
    pub fn fillna_with_series_with_runtime(
        &self,
        column: &str,
        fill_series: &Series,
        runtime: &Runtime,
    ) -> Result<Self> {
        let operation = "dataframe::fillna_with_series_with_runtime";
        if !self.columns.contains_key(column) {
            return Err(Error::new(ErrorKind::InvalidOperation, format!("column '{}' not found in schema", column))
                .with_operation(operation)
                .with_context("column_name", column.to_string()));
        }

        let output_domain = Arc::clone(runtime.execution_domain());

        // Copy all existing columns into the target domain. After this the
        // target column is guaranteed to be in the output domain.
        let mut new_columns = self.clone_or_copy_columns_into_domain(&output_domain, operation)?;

        // Copy the fill series into the target domain.
        let fill_copied = Self::clone_or_copy_series_into_domain(fill_series, &output_domain, operation, column)?;

        // Perform replace-nulls on the target column. Scoped to drop views
        // before mutating new_columns.
        let replaced = {
            let target_col = new_columns.get(column).expect("column validated above");
            let cv = target_col.view()?;
            let fill_cv = fill_copied.view()?;
            let exec_options = output_domain.explicit_exec_options_raw();
            replace::replace_nulls_column_with_exec_options(&cv, &fill_cv, exec_options.as_ref(), operation)?
                .with_execution_domain(Arc::clone(&output_domain))
        };

        new_columns.insert(column.to_string(), Arc::new(replaced));

        let domain = resolve_frame_domain_from_authoritative_source(&new_columns, &output_domain, operation)?;
        Ok(Self { columns: new_columns, row_count: self.row_count, domain })
    }
}

impl DataFrame {
    fn gather_by_column_view_impl(
        &self,
        indices: &ColumnView<'_>,
        exec_options: Option<cudf_sys::cudf_exec_options_t>,
        output_domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Self> {
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
        let result_table = copying::gather_with_exec_options(tv.table(), indices, exec_options.as_ref(), operation)?
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
        if !ExecutionDomain::same_provenance(&self.domain, &output_domain) {
            self.domain.fence_stream(operation)?;
        }
        if !ExecutionDomain::same_provenance(mask.execution_domain(), &output_domain) {
            mask.execution_domain().fence_stream(operation)?;
        }

        self.validate_mask_view(mask, operation)?;
        let tv = self.build_table_view()?;
        let result_table =
            filter::apply_boolean_mask_with_exec_options(tv.table(), mask, exec_options.as_ref(), operation)?
                .with_execution_domain(output_domain);
        Self::from_table_with_names(result_table, self.column_names_owned())
    }

    /// Internal: gather a contiguous range of rows into a new `DataFrame`,
    /// preserving the schema.
    fn gather_range(&self, start: usize, len: usize) -> Result<Self> {
        if len == 0 {
            return self.empty_with_same_schema();
        }

        let indices: Vec<i32> = (start as i32..(start + len) as i32).collect();
        let map = Column::from_slice(&indices)?;
        let tv = self.build_table_view()?;
        let exec_options = self.domain.explicit_exec_options_raw();
        // SAFETY: indices are constructed from `start..start+len` which the
        // callers have already validated against row_count.
        let result_table = unsafe {
            copying::gather_unchecked_with_exec_options(
                tv.table(),
                &map.view()?,
                exec_options.as_ref(),
                "dataframe::gather_range",
            )?
        }
        .with_execution_domain(Arc::clone(&self.domain));

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
        let mut new_columns = IndexMap::with_capacity(self.columns.len());
        for (name, col) in &self.columns {
            let cv = col.view()?;
            let type_id = cv.data_type()?.id();
            new_columns
                .insert(name.clone(), Arc::new(Column::empty(type_id)?.with_execution_domain(Arc::clone(&domain))));
        }
        Ok(Self { columns: new_columns, row_count: 0, domain })
    }
}
