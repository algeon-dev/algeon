use std::sync::Arc;

use typed_builder::TypedBuilder;

use super::{context::FrameOps, owned::DataFrame};
use crate::{
    core::{
        column::{Column, ColumnView},
        error::Result,
        scalar::Scalar,
        schema,
        types::CudfType,
    },
    frame::series::Series,
    internal::resolve::{missing_column_error, resolve_column_names},
    ops::{
        replace,
        stream_compaction::{self, DuplicateKeepOption, NanEquality, NullEquality},
    },
    runtime::{execution_context::Exec, execution_domain::ExecutionDomain, execution_resolver::ResolvedExecution},
};

/// How [`FrameOps::dropna`](crate::FrameOps::dropna) decides which rows to drop when `thresh` is unset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DropNaHow {
    /// Drop rows when any inspected column is null.
    #[default]
    Any,
    /// Drop rows only when all inspected columns are null.
    All,
}

/// Options for [`FrameOps::dropna`](crate::FrameOps::dropna).
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
    pub(super) subset: Vec<String>,
    /// Minimum number of non-null values in subset required to keep the row.
    /// `None` means derive from `how`.
    #[builder(default, setter(transform = |thresh: i32| Some(thresh)))]
    pub(super) thresh: Option<i32>,
    /// How rows should be dropped when `thresh` is unset.
    #[builder(default)]
    pub(super) how: DropNaHow,
}

impl Default for DropNaOptions {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl DropNaOptions {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Options for [`FrameOps::drop_duplicates`](crate::FrameOps::drop_duplicates).
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
    pub(super) subset: Vec<String>,
    /// Which duplicate occurrence to keep.
    #[builder(default = DuplicateKeepOption::KeepAny)]
    pub(super) keep: DuplicateKeepOption,
}

impl Default for DropDuplicatesOptions {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl DropDuplicatesOptions {
    pub fn new() -> Self {
        Self::default()
    }
}

#[cfg(any(test, feature = "test-hooks"))]
pub(crate) fn trigger_dataframe_dropna_subset_count_limit_error_for_test() -> Result<()> {
    Err(schema::column_name_count_limit_error("dataframe::dropna", "subset length exceeds i32::MAX", usize::MAX))
}

impl DataFrame {
    fn resolve_subset_names<'a>(&'a self, subset: &'a [String]) -> Vec<&'a str> {
        if subset.is_empty() { self.column_names() } else { subset.iter().map(String::as_str).collect() }
    }

    pub(crate) fn dropna_in_context(&self, opts: DropNaOptions, ctx: Exec<'_>) -> Result<Self> {
        let subset_names = self.resolve_subset_names(&opts.subset);

        let subset_len = i32::try_from(subset_names.len()).map_err(|_| {
            schema::column_name_count_limit_error(
                "dataframe::dropna",
                "subset length exceeds i32::MAX",
                subset_names.len(),
            )
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
        self.drop_nulls_impl(&subset_names, threshold, output_domain, "dataframe::drop_nulls")
            .map_err(|e| e.reanchor("dataframe::dropna"))
    }

    pub(crate) fn drop_duplicates_in_context(&self, opts: DropDuplicatesOptions, ctx: Exec<'_>) -> Result<Self> {
        let subset_names = self.resolve_subset_names(&opts.subset);

        let output_domain = ctx.resolve_output_domain(&self.domain)?;
        self.distinct_impl(&subset_names, opts.keep, output_domain, "dataframe::distinct")
            .map_err(|e| e.reanchor("dataframe::drop_duplicates"))
    }

    pub(crate) fn drop_consecutive_duplicates_in_context(
        &self,
        opts: DropDuplicatesOptions,
        ctx: Exec<'_>,
    ) -> Result<Self> {
        let subset_names = self.resolve_subset_names(&opts.subset);

        let output_domain = ctx.resolve_output_domain(&self.domain)?;
        self.unique_impl(&subset_names, opts.keep, output_domain, "dataframe::drop_consecutive_duplicates")
    }

    fn fillna_column_in_context(
        &self,
        column: &str,
        fill_input: &Arc<ExecutionDomain>,
        ctx: Exec<'_>,
        operation: &'static str,
        fill: impl FnOnce(&ColumnView<'_>, &ResolvedExecution) -> Result<Column>,
    ) -> Result<Self> {
        let output_domain = ctx.resolve_output_domain(&self.domain)?;
        let col = self.columns.get(column).ok_or_else(|| missing_column_error(operation, column))?;
        let cv = col.view()?;
        let execution = ResolvedExecution::with_output_domain(
            Arc::clone(&output_domain),
            [cv.execution_domain(), fill_input],
            operation,
        )?;
        let filled = Series::from_column(fill(&cv, &execution)?)?;
        self.with_column_in_domain(column, filled, output_domain, operation)
    }
}

impl DataFrame {
    fn distinct_impl(
        &self,
        subset: &[&str],
        keep: DuplicateKeepOption,
        output_domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Self> {
        let execution = ResolvedExecution::with_output_domain(Arc::clone(&output_domain), [&self.domain], operation)?;
        let output_domain = Arc::clone(execution.output_domain());

        let key_indices = resolve_column_names(&self.columns, subset, operation)?;
        let tv = self.build_table_view()?;
        let result_table = stream_compaction::distinct_impl(
            &tv,
            stream_compaction::DistinctConfig::new(&key_indices, keep, NullEquality::Equal, NanEquality::AllEqual),
            stream_compaction::StreamCompactionExecution::new(&output_domain, execution.exec_options(), operation),
        )?;
        Self::from_table_with_names(result_table, self.column_names_owned())
    }

    fn unique_impl(
        &self,
        subset: &[&str],
        keep: DuplicateKeepOption,
        output_domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Self> {
        let execution = ResolvedExecution::with_output_domain(Arc::clone(&output_domain), [&self.domain], operation)?;
        let output_domain = Arc::clone(execution.output_domain());

        let key_indices = resolve_column_names(&self.columns, subset, operation)?;
        let tv = self.build_table_view()?;
        let result_table = stream_compaction::unique_impl(
            &tv,
            &key_indices,
            keep,
            NullEquality::Equal,
            stream_compaction::StreamCompactionExecution::new(&output_domain, execution.exec_options(), operation),
        )?;
        Self::from_table_with_names(result_table, self.column_names_owned())
    }

    fn drop_nulls_impl(
        &self,
        subset: &[&str],
        keep_threshold: i32,
        output_domain: Arc<ExecutionDomain>,
        operation: &'static str,
    ) -> Result<Self> {
        let execution = ResolvedExecution::with_output_domain(Arc::clone(&output_domain), [&self.domain], operation)?;
        let output_domain = Arc::clone(execution.output_domain());

        let key_indices = resolve_column_names(&self.columns, subset, operation)?;
        let tv = self.build_table_view()?;
        let result_table = stream_compaction::drop_nulls_impl(
            &tv,
            &key_indices,
            keep_threshold,
            &output_domain,
            execution.exec_options(),
            operation,
        )?;
        Self::from_table_with_names(result_table, self.column_names_owned())
    }
}

impl FrameOps<'_> {
    /// Replace null values in a single column with a plain Rust value.
    ///
    /// Equivalent to cuDF Python's `DataFrame.fillna({column: value})`.
    pub fn fillna<T: CudfType>(&self, column: &str, value: T) -> Result<DataFrame> {
        let output_domain = self.ctx.resolve_output_domain(&self.df.domain)?;
        let scalar = Scalar::from_value_in_domain(value, &output_domain)?;
        self.fillna_with_scalar(column, &scalar).map_err(|error| error.reanchor("dataframe::fillna"))
    }

    /// Replace null values in a single column with a pre-constructed [`Scalar`].
    pub fn fillna_with_scalar(&self, column: &str, value: &Scalar) -> Result<DataFrame> {
        let operation = "dataframe::fillna_with_scalar";
        self.df.fillna_column_in_context(column, value.execution_domain(), self.ctx, operation, |cv, execution| {
            replace::replace_nulls_with_exec_options(
                cv,
                value,
                execution.output_domain(),
                execution.exec_options(),
                operation,
            )
        })
    }

    /// Replace null values in a single column using corresponding elements
    /// from a [`Series`].
    pub fn fillna_with_series(&self, column: &str, fill_series: &Series) -> Result<DataFrame> {
        let operation = "dataframe::fillna_with_series";
        let fill_cv = fill_series.view()?;
        self.df.fillna_column_in_context(column, fill_cv.execution_domain(), self.ctx, operation, |cv, execution| {
            replace::replace_nulls_column_with_exec_options(
                cv,
                &fill_cv,
                execution.output_domain(),
                execution.exec_options(),
                operation,
            )
        })
    }
}
