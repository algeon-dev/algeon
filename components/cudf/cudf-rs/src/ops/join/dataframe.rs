use std::sync::Arc;

use crate::{
    core::{
        column::Column,
        error::{Error, ErrorKind, Result},
    },
    frame::dataframe::{DataFrame, DataFrameJoinSides, derive_frame_domain_from_columns},
    internal::{
        hash::{FastHashSet, FastIndexMap, fast_index_map_with_capacity},
        resolve::{
            build_borrowed_table_view_from_columns, payload_indices, resolve_column_names, validate_no_duplicate_names,
        },
    },
    ops::{join, replace},
    runtime::{execution_context::ExecutionOverrides, execution_domain::ExecutionDomain},
};

impl DataFrame {
    /// Merge using a borrowed per-operation execution override.
    pub(crate) fn merge_in_context(
        &self,
        right: &DataFrame,
        options: join::MergeOptions,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<Self> {
        let output_domain = ctx.resolve_output_domain(self.execution_domain())?;
        self.merge_impl(
            right,
            options,
            output_domain.explicit_exec_options_raw(),
            Some(output_domain),
            "dataframe::with_execution::merge",
        )
    }

    /// Merge-split using a borrowed per-operation execution override.
    pub(crate) fn merge_split_in_context(
        &self,
        right: &DataFrame,
        options: join::MergeOptions,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<DataFrameJoinSides> {
        let operation: &'static str = "dataframe::with_execution::merge_split";
        let (left_on_strs, right_on_strs, _shared_keys) = resolve_merge_keys(&options, operation)?;
        let left_on_refs: Vec<&str> = left_on_strs.iter().map(|s| s.as_str()).collect();
        let right_on_refs: Vec<&str> = right_on_strs.iter().map(|s| s.as_str()).collect();

        if left_on_refs.len() != right_on_refs.len() {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "left_on and right_on must have the same number of key columns",
            )
            .with_operation(operation)
            .with_context("left_on_count", left_on_refs.len().to_string())
            .with_context("right_on_count", right_on_refs.len().to_string()));
        }

        let output_domain = ctx.resolve_output_domain(self.execution_domain())?;
        let exec_options = output_domain.explicit_exec_options_raw();
        self.join_split_on_impl_with_null_eq_and_context(
            right,
            &left_on_refs,
            &right_on_refs,
            options.join_type(),
            options.null_eq(),
            exec_options.as_ref(),
            Some(output_domain),
            operation,
        )
    }
    pub(crate) fn __debug_merge_in_context(
        &self,
        right: &DataFrame,
        options: join::MergeOptions,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<Self> {
        self.merge_in_context(right, options, ctx)
    }
    // -- Joins (merge) -------------------------------------------------------

    /// Merge (join) two DataFrames with cuDF-style options.
    ///
    /// This is the unified high-level entry point for joins. Configure the merge
    /// via [`MergeOptions`](crate::MergeOptions).
    ///
    /// # Key modes
    ///
    /// - `on(keys)` — shared key column names present in both sides. Output
    ///   contains one shared key column per key.
    /// - `left_on(keys)` + `right_on(keys)` — asymmetric key names. Both key
    ///   columns are preserved in the output.
    ///
    /// # Suffix handling
    ///
    /// When `on()` is used: if a left payload and right payload column share
    /// the same name, both are suffixed with the configured suffixes (default
    /// `"_left"` / `"_right"`).
    ///
    /// When `left_on()` / `right_on()` is used: both key columns are preserved,
    /// and conflicting payload names get the same suffix treatment.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No keys are specified (`on` or `left_on`+`right_on`)
    /// - `left_on` is set without `right_on` or vice versa
    /// - `left_on` and `right_on` have different lengths
    /// - A key column is not found in its respective DataFrame
    pub fn merge(&self, right: &DataFrame, options: join::MergeOptions) -> Result<Self> {
        self.merge_impl(right, options, None, None, "dataframe::merge")
    }

    fn merge_impl(
        &self,
        right: &DataFrame,
        options: join::MergeOptions,
        exec_options: Option<cudf_sys::cudf_exec_options_t>,
        forced_output_domain: Option<Arc<ExecutionDomain>>,
        operation: &'static str,
    ) -> Result<Self> {
        let operation: &'static str = operation;

        // Resolve key names from options.
        let (left_on_strs, right_on_strs, shared_keys) = resolve_merge_keys(&options, operation)?;

        let left_on_refs: Vec<&str> = left_on_strs.iter().map(|s| s.as_str()).collect();
        let right_on_refs: Vec<&str> = right_on_strs.iter().map(|s| s.as_str()).collect();

        // Validate key counts match.
        if left_on_refs.len() != right_on_refs.len() {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "left_on and right_on must have the same number of key columns",
            )
            .with_operation(operation)
            .with_context("left_on_count", left_on_refs.len().to_string())
            .with_context("right_on_count", right_on_refs.len().to_string()));
        }

        let join_type = options.join_type();

        // Execute the split join with null_equality from options.
        let split = self.join_split_on_impl_with_null_eq_and_context(
            right,
            &left_on_refs,
            &right_on_refs,
            join_type,
            options.null_eq(),
            exec_options.as_ref(),
            forced_output_domain,
            operation,
        )?;

        // Reassemble with high-level suffix logic.
        merge_join_result_with_suffixes(
            split,
            &left_on_refs,
            shared_keys,
            shared_keys && join_type == join::JoinType::Full,
            options.suffix_left(),
            options.suffix_right(),
            operation,
        )
    }

    /// Merge (join) two DataFrames, returning split left/right results.
    ///
    /// Like [`merge`](Self::merge) but returns a [`DataFrameJoinSides`] with
    /// separate left and right `DataFrame`s instead of merging columns into one.
    /// Each side contains `[key_columns..., payload_columns...]`.
    ///
    /// # Errors
    ///
    /// Same error conditions as [`merge`](Self::merge).
    pub fn merge_split(&self, right: &DataFrame, options: join::MergeOptions) -> Result<DataFrameJoinSides> {
        let operation: &'static str = "dataframe::merge_split";

        let (left_on_strs, right_on_strs, _shared_keys) = resolve_merge_keys(&options, operation)?;

        let left_on_refs: Vec<&str> = left_on_strs.iter().map(|s| s.as_str()).collect();
        let right_on_refs: Vec<&str> = right_on_strs.iter().map(|s| s.as_str()).collect();

        if left_on_refs.len() != right_on_refs.len() {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "left_on and right_on must have the same number of key columns",
            )
            .with_operation(operation)
            .with_context("left_on_count", left_on_refs.len().to_string())
            .with_context("right_on_count", right_on_refs.len().to_string()));
        }

        self.join_split_on_impl_with_null_eq_and_context(
            right,
            &left_on_refs,
            &right_on_refs,
            options.join_type(),
            options.null_eq(),
            None,
            None,
            operation,
        )
    }
    /// Internal split-join dispatch with configurable null equality.
    #[allow(clippy::too_many_arguments)]
    fn join_split_on_impl_with_null_eq_and_context(
        &self,
        right: &DataFrame,
        left_on: &[&str],
        right_on: &[&str],
        join_type: join::JoinType,
        null_equality: join::NullEquality,
        exec_options: Option<&cudf_sys::cudf_exec_options_t>,
        forced_output_domain: Option<Arc<ExecutionDomain>>,
        operation: &'static str,
    ) -> Result<DataFrameJoinSides> {
        let output_domain = forced_output_domain.unwrap_or_else(|| Arc::clone(self.execution_domain()));

        ExecutionDomain::require_same_device_for_mixed_inputs(
            self.execution_domain(),
            right.execution_domain(),
            operation,
        )?;
        if !ExecutionDomain::same_provenance(self.execution_domain(), &output_domain) {
            self.execution_domain().fence_stream(operation)?;
        }
        if !ExecutionDomain::same_provenance(right.execution_domain(), &output_domain) {
            right.execution_domain().fence_stream(operation)?;
        }

        if left_on.len() != right_on.len() {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "left_on and right_on must have the same number of key columns",
            )
            .with_operation(operation)
            .with_context("left_on_count", left_on.len().to_string())
            .with_context("right_on_count", right_on.len().to_string()));
        }

        // Resolve key column names in both sides.
        let left_key_indices = resolve_column_names(self.columns_map(), left_on, operation)?;
        let right_key_indices = resolve_column_names(right.columns_map(), right_on, operation)?;

        // Compute payload indices (all columns not in keys, preserving order).
        let left_payload_indices = payload_indices(self.columns_map(), &left_key_indices);
        let right_payload_indices = payload_indices(right.columns_map(), &right_key_indices);

        // Build borrowed key and payload table views over the existing columns.
        let lk_table = build_borrowed_table_view_from_columns(self.columns_map(), &left_key_indices)?;
        let rk_table = build_borrowed_table_view_from_columns(right.columns_map(), &right_key_indices)?;
        let lp_table = build_borrowed_table_view_from_columns(self.columns_map(), &left_payload_indices)?;
        let rp_table = build_borrowed_table_view_from_columns(right.columns_map(), &right_payload_indices)?;

        let result = if let Some(exec_options) = exec_options {
            join::join_impl(
                lk_table.table(),
                rk_table.table(),
                lp_table.table(),
                rp_table.table(),
                null_equality,
                join_type,
                &output_domain,
                Some(exec_options),
                operation,
            )?
        } else {
            match join_type {
                join::JoinType::Inner => join::inner_join(
                    lk_table.table(),
                    rk_table.table(),
                    lp_table.table(),
                    rp_table.table(),
                    null_equality,
                )?,
                join::JoinType::Left => join::left_join(
                    lk_table.table(),
                    rk_table.table(),
                    lp_table.table(),
                    rp_table.table(),
                    null_equality,
                )?,
                join::JoinType::Full => join::full_join(
                    lk_table.table(),
                    rk_table.table(),
                    lp_table.table(),
                    rp_table.table(),
                    null_equality,
                )?,
            }
        };

        // Build schemas: left side uses left key names, right side uses right key names.
        let left_schema_names = build_join_schema_names(self.columns_map(), &left_key_indices, &left_payload_indices);
        let right_schema_names =
            build_join_schema_names(right.columns_map(), &right_key_indices, &right_payload_indices);

        Ok(DataFrameJoinSides {
            left: DataFrame::from_table(result.left, left_schema_names)?,
            right: DataFrame::from_table(result.right, right_schema_names)?,
        })
    }
}

/// Build the schema name list for a join result side:
/// key column names (in `on` order) followed by payload column names.
fn build_join_schema_names(
    columns: &FastIndexMap<String, Arc<Column>>,
    key_indices: &[usize],
    payload_indices: &[usize],
) -> Vec<String> {
    let keys_iter = columns.keys().collect::<Vec<_>>();

    let mut names = Vec::with_capacity(key_indices.len() + payload_indices.len());
    for &idx in key_indices {
        names.push(keys_iter[idx].clone());
    }
    for &idx in payload_indices {
        names.push(keys_iter[idx].clone());
    }
    names
}

/// Coalesce two columns by preferring non-null values from `left`; if null,
/// take from `right`. Runs entirely on GPU via `replace_nulls_column`.
fn coalesce_columns(left: &Column, right: &Column, operation: &'static str) -> Result<Column> {
    let output_domain = Arc::clone(left.execution_domain());
    let lcv = left.view()?;
    let rcv = right.view()?;

    let ldt = lcv.data_type()?;
    let rdt = rcv.data_type()?;
    if ldt != rdt {
        return Err(Error::new(ErrorKind::InvalidOperation, "cannot coalesce columns with different types")
            .with_operation(operation)
            .with_context("left_type_id", format!("{:?}", ldt.id()).to_lowercase())
            .with_context("right_type_id", format!("{:?}", rdt.id()).to_lowercase()));
    }

    let exec_options = output_domain.explicit_exec_options_raw();
    replace::replace_nulls_column_with_exec_options(&lcv, &rcv, &output_domain, exec_options.as_ref(), operation)
}

// ---------------------------------------------------------------------------
// High-level merge helpers
// ---------------------------------------------------------------------------

/// Resolve the key names from `MergeOptions`, returning `(left_on, right_on, shared)`.
///
/// `shared` is `true` when `on()` was used (both sides share the same key names),
/// meaning the output should emit one shared key column per key.
fn resolve_merge_keys(
    options: &join::MergeOptions,
    operation: &'static str,
) -> Result<(Vec<String>, Vec<String>, bool)> {
    match (options.on_keys(), options.left_on_keys(), options.right_on_keys()) {
        // on() mode: shared key names
        (Some(on), None, None) => {
            let keys: Vec<String> = on.to_vec();
            Ok((keys.clone(), keys, true))
        },
        // left_on() + right_on() mode: asymmetric keys
        (None, Some(lo), Some(ro)) => Ok((lo.to_vec(), ro.to_vec(), false)),
        // left_on without right_on
        (None, Some(_), None) => {
            Err(Error::new(ErrorKind::InvalidOperation, "left_on was specified without right_on; set both or use on()")
                .with_operation(operation))
        },
        // right_on without left_on
        (None, None, Some(_)) => {
            Err(Error::new(ErrorKind::InvalidOperation, "right_on was specified without left_on; set both or use on()")
                .with_operation(operation))
        },
        // on() together with left_on/right_on is ambiguous — treat as error
        (Some(_), _, _) => Err(Error::new(
            ErrorKind::InvalidOperation,
            "on() cannot be combined with left_on()/right_on(); use one or the other",
        )
        .with_operation(operation)),
        // Nothing specified
        (None, None, None) => {
            Err(Error::new(ErrorKind::InvalidOperation, "no join keys specified; use on() or left_on() + right_on()")
                .with_operation(operation))
        },
    }
}

/// Reassemble a split join result into a single DataFrame using high-level suffix rules.
///
/// When `shared_keys` is true (from `on()`):
///   - Emit one shared key column per key (coalesced for full join).
///   - Conflicting payload names get `suffix_left` / `suffix_right`.
///
/// When `shared_keys` is false (from `left_on()`/`right_on()`):
///   - Both key columns are preserved (left key name, right key name).
///   - Conflicting payload names get suffixes.
fn merge_join_result_with_suffixes(
    split: DataFrameJoinSides,
    left_on: &[&str],
    shared_keys: bool,
    coalesce_keys: bool,
    suffix_left: &str,
    suffix_right: &str,
    operation: &'static str,
) -> Result<DataFrame> {
    let DataFrameJoinSides { left, right } = split;
    let key_count = left_on.len();
    let row_count = left.num_rows().max(right.num_rows());

    let left_names = left.column_names_owned();
    let right_names = right.column_names_owned();
    let left_cols: Vec<Arc<Column>> = left.columns_map().values().cloned().collect();
    let right_cols: Vec<Arc<Column>> = right.columns_map().values().cloned().collect();

    if key_count > left_cols.len() || key_count > right_cols.len() {
        return Err(Error::new(ErrorKind::InvalidOperation, "join key count exceeds result column count")
            .with_operation(operation)
            .with_context("num_keys", key_count.to_string())
            .with_context("left_columns", left_cols.len().to_string())
            .with_context("right_columns", right_cols.len().to_string()));
    }

    // Collect left and right payload names (after key columns).
    let left_payload_names: Vec<&str> = left_names[key_count..].iter().map(|s| s.as_str()).collect();
    let right_payload_names: Vec<&str> = right_names[key_count..].iter().map(|s| s.as_str()).collect();

    // Find which payload names collide.
    let left_payload_set: FastHashSet<&str> = left_payload_names.iter().copied().collect();
    let right_payload_set: FastHashSet<&str> = right_payload_names.iter().copied().collect();
    let conflicting: FastHashSet<&str> = left_payload_set.intersection(&right_payload_set).copied().collect();

    let mut out_cols: Vec<Arc<Column>> = Vec::new();
    let mut out_names: Vec<String> = Vec::new();

    // Only shared-key full joins should coalesce key columns. For left_on/right_on
    // joins, the contract is to preserve both key columns separately.
    for i in 0..key_count {
        let key_col = if coalesce_keys {
            Arc::new(coalesce_columns(&left_cols[i], &right_cols[i], operation)?)
        } else {
            Arc::clone(&left_cols[i])
        };
        out_cols.push(key_col);
        out_names.push(left_names[i].clone());
    }

    // Left payload.
    for (idx, name) in left_payload_names.iter().enumerate() {
        let col_idx = key_count + idx;
        out_cols.push(Arc::clone(&left_cols[col_idx]));
        if conflicting.contains(name) {
            out_names.push(format!("{}{}", name, suffix_left));
        } else {
            out_names.push(name.to_string());
        }
    }

    if !shared_keys {
        // Emit right key columns for asymmetric join.
        for i in 0..key_count {
            out_cols.push(Arc::clone(&right_cols[i]));
            out_names.push(right_names[i].clone());
        }
    }

    // Right payload.
    for (idx, name) in right_payload_names.iter().enumerate() {
        let col_idx = key_count + idx;
        out_cols.push(Arc::clone(&right_cols[col_idx]));
        if conflicting.contains(name) {
            out_names.push(format!("{}{}", name, suffix_right));
        } else {
            out_names.push(name.to_string());
        }
    }

    validate_no_duplicate_names(&out_names, operation)?;

    let mut columns = fast_index_map_with_capacity(out_names.len());
    for (name, col) in out_names.into_iter().zip(out_cols) {
        columns.insert(name, col);
    }

    let domain = derive_frame_domain_from_columns(&columns, operation)?;
    Ok(DataFrame::from_raw_parts(columns, row_count, domain))
}
