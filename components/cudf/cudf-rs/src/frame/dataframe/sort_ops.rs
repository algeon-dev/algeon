use std::sync::Arc;

use smallvec::SmallVec;

use super::{fillna_ops::DropDuplicatesOptions, owned::DataFrame};
use crate::{
    core::{
        column::Column,
        error::{Error, ErrorKind, Result},
    },
    frame::series::Series,
    internal::resolve::resolve_column_names,
    ops::{
        join::MergeOptions,
        sorting::{self, SortOptions},
        stream_compaction::DuplicateKeepOption,
    },
    runtime::{execution_context::ExecutionContext, execution_domain::ExecutionDomain},
};

const SORT_KEY_INLINE_CAPACITY: usize = 4;

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

    /// Sort rows using cuDF-style [`SortOptions`].
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
                    &output_domain,
                    Some(exec_options),
                    operation,
                )?
            } else {
                let sort_fn = if options.stable { sorting::stable_sort } else { sorting::sort };
                sort_fn(tv.table(), Some(&orders), null_order_ref)?
            }
        } else {
            let keys_table = tv.table().select_columns(&key_indices)?;
            if let Some(exec_options) = exec_options.as_ref() {
                sorting::sort_by_key_impl(
                    tv.table(),
                    &keys_table,
                    Some(&orders),
                    null_order_ref,
                    options.stable,
                    &output_domain,
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

    pub fn nlargest(&self, n: usize, column: &str) -> Result<Self> {
        self.sort_values([column], SortOptions::new().with_descending(vec![true]))?.head(n)
    }

    pub fn nsmallest(&self, n: usize, column: &str) -> Result<Self> {
        self.sort_values([column], SortOptions::new())?.head(n)
    }

    /// Return a zero-based per-group row number for already-sorted rows.
    pub fn group_cumcount<I, S>(&self, keys: I, name: &str) -> Result<Series>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let operation = "dataframe::group_cumcount";
        let keys: SmallVec<[String; SORT_KEY_INLINE_CAPACITY]> =
            keys.into_iter().map(|key| key.as_ref().to_string()).collect();
        if keys.is_empty() {
            return Err(
                Error::new(ErrorKind::InvalidOperation, "at least one group key is required").with_operation(operation)
            );
        }

        let key_refs: SmallVec<[&str; SORT_KEY_INLINE_CAPACITY]> = keys.iter().map(|key| key.as_str()).collect();
        resolve_column_names(&self.columns, &key_refs, operation)?;

        let row_idx_name = unique_temp_column_name(self, "__cudf_group_cumcount_row_idx__");
        let group_start_name = unique_temp_column_name(self, "__cudf_group_cumcount_group_start__");
        let row_idx = Series::from_named_column(
            row_idx_name.clone(),
            Column::from_sequence_in_domain(0i32, 1i32, self.row_count, Arc::clone(&self.domain), operation)?,
        )?;
        let with_row_idx = self.with_column(&row_idx_name, row_idx)?;

        let first_rows = with_row_idx.drop_duplicates(
            DropDuplicatesOptions::builder().subset(key_refs.as_slice()).keep(DuplicateKeepOption::KeepFirst).build(),
        )?;
        let mut first_row_columns: SmallVec<[&str; SORT_KEY_INLINE_CAPACITY + 1]> = SmallVec::new();
        first_row_columns.extend(key_refs.iter().copied());
        first_row_columns.push(row_idx_name.as_str());
        let first_rows = first_rows.select(first_row_columns)?;
        let first_rows = first_rows
            .with_column(&group_start_name, first_rows.series(&row_idx_name)?.rename(&group_start_name))?
            .without_columns([row_idx_name.as_str()])?;

        let aligned = with_row_idx
            .merge(&first_rows, MergeOptions::new().how_left().on(key_refs.iter().copied()))?
            .sort_values([row_idx_name.as_str()], SortOptions::new().with_stable(true))?;

        let rank = aligned.series(&row_idx_name)?.sub(&aligned.series(&group_start_name)?)?;
        Ok(rank.rename(name))
    }
}
