use std::sync::Arc;

use crate::{
    core::{
        column::Column,
        error::{Error, ErrorCode, FACT_KEYS_COUNT, FACT_LEFT_TYPE_ID, FACT_RIGHT_TYPE_ID, Result},
    },
    frame::{
        dataframe::{
            DataFrame, DataFrameJoinIndices, DataFrameJoinOptions, DataFrameJoinSides, derive_frame_domain_from_columns,
        },
        series::Series,
    },
    internal::{
        hash::{FastHashSet, FastIndexMap, fast_index_map_with_capacity},
        resolve::{
            build_borrowed_table_view_from_columns, payload_indices, resolve_column_names, validate_no_duplicate_names,
        },
    },
    ops::{join, replace},
    runtime::{
        Runtime, execution_context::ExecutionOverrides, execution_domain::ExecutionDomain,
        execution_resolver::ResolvedExecution,
    },
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
            return Err(merge_key_count_mismatch_error(operation, left_on_refs.len(), right_on_refs.len()));
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

    /// Produce left semi join indices using a borrowed per-operation execution override.
    pub(crate) fn left_semi_join_indices_in_context(
        &self,
        right: &DataFrame,
        options: DataFrameJoinOptions,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<DataFrameJoinIndices> {
        self.single_side_join_indices_impl(
            right,
            options,
            true,
            Some(ctx),
            "dataframe::with_execution::left_semi_join_indices",
        )
    }

    /// Produce left anti join indices using a borrowed per-operation execution override.
    pub(crate) fn left_anti_join_indices_in_context(
        &self,
        right: &DataFrame,
        options: DataFrameJoinOptions,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<DataFrameJoinIndices> {
        self.single_side_join_indices_impl(
            right,
            options,
            false,
            Some(ctx),
            "dataframe::with_execution::left_anti_join_indices",
        )
    }

    /// Left semi join using a borrowed per-operation execution override.
    pub(crate) fn left_semi_join_in_context(
        &self,
        right: &DataFrame,
        options: DataFrameJoinOptions,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<Self> {
        let indices = self.left_semi_join_indices_in_context(right, options, ctx)?;
        self.materialize_join_indices_in_context(
            &indices,
            self.column_names_owned(),
            ctx,
            "dataframe::with_execution::left_semi_join",
        )
    }

    /// Left anti join using a borrowed per-operation execution override.
    pub(crate) fn left_anti_join_in_context(
        &self,
        right: &DataFrame,
        options: DataFrameJoinOptions,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<Self> {
        let indices = self.left_anti_join_indices_in_context(right, options, ctx)?;
        self.materialize_join_indices_in_context(
            &indices,
            self.column_names_owned(),
            ctx,
            "dataframe::with_execution::left_anti_join",
        )
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
            return Err(merge_key_count_mismatch_error(operation, left_on_refs.len(), right_on_refs.len()));
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
            return Err(merge_key_count_mismatch_error(operation, left_on_refs.len(), right_on_refs.len()));
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

    /// Return device-side gather indices for the left rows that have a match in `right`.
    ///
    /// This is an index-producing query-planning helper. It allocates only an
    /// `INT32` device gather map and does not copy payload columns. Duplicate
    /// matches on the right side do not duplicate left rows.
    pub fn left_semi_join_indices(
        &self,
        right: &DataFrame,
        options: DataFrameJoinOptions,
    ) -> Result<DataFrameJoinIndices> {
        self.single_side_join_indices_impl(right, options, true, None, "dataframe::left_semi_join_indices")
    }

    /// Return device-side gather indices for the left rows that have no match in `right`.
    ///
    /// This is an index-producing query-planning helper. It allocates only an
    /// `INT32` device gather map and does not copy payload columns.
    pub fn left_anti_join_indices(
        &self,
        right: &DataFrame,
        options: DataFrameJoinOptions,
    ) -> Result<DataFrameJoinIndices> {
        self.single_side_join_indices_impl(right, options, false, None, "dataframe::left_anti_join_indices")
    }

    /// Materialize selected columns by gathering rows with join indices.
    ///
    /// This performs a GPU gather of the selected columns and therefore copies
    /// device buffers for those output columns. It is not a zero-copy schema
    /// projection. The row count of the output is the length of `indices`.
    pub fn materialize_join_indices<I, S>(&self, indices: &DataFrameJoinIndices, columns: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.materialize_join_indices_impl(indices, columns, None, "dataframe::materialize_join_indices")
    }

    /// Left semi join, returning only left-side columns.
    ///
    /// This helper first computes left-side join indices and then gathers the
    /// left frame's columns. The gather copies selected device rows into new
    /// columns. Right-side payload columns are never included, and duplicate
    /// right matches do not duplicate left rows.
    pub fn left_semi_join(&self, right: &DataFrame, options: DataFrameJoinOptions) -> Result<Self> {
        let indices = self.left_semi_join_indices(right, options)?;
        self.materialize_join_indices_impl(&indices, self.column_names_owned(), None, "dataframe::left_semi_join")
    }

    /// Left anti join, returning only left-side columns.
    ///
    /// This helper first computes left-side join indices and then gathers the
    /// left frame's columns. The gather copies selected device rows into new
    /// columns. Right-side payload columns are never included.
    pub fn left_anti_join(&self, right: &DataFrame, options: DataFrameJoinOptions) -> Result<Self> {
        let indices = self.left_anti_join_indices(right, options)?;
        self.materialize_join_indices_impl(&indices, self.column_names_owned(), None, "dataframe::left_anti_join")
    }

    fn single_side_join_indices_impl(
        &self,
        right: &DataFrame,
        options: DataFrameJoinOptions,
        semi_join: bool,
        ctx: Option<&ExecutionOverrides<'_>>,
        operation: &'static str,
    ) -> Result<DataFrameJoinIndices> {
        let (left_on_strs, right_on_strs) = resolve_dataframe_join_keys(&options, operation)?;
        let left_on_refs: Vec<&str> = left_on_strs.iter().map(|s| s.as_str()).collect();
        let right_on_refs: Vec<&str> = right_on_strs.iter().map(|s| s.as_str()).collect();

        if left_on_refs.len() != right_on_refs.len() {
            return Err(merge_key_count_mismatch_error(operation, left_on_refs.len(), right_on_refs.len()));
        }

        let left_key_indices = resolve_column_names(self.columns_map(), &left_on_refs, operation)?;
        let right_key_indices = resolve_column_names(right.columns_map(), &right_on_refs, operation)?;
        let lk_table = build_borrowed_table_view_from_columns(self.columns_map(), &left_key_indices)?;
        let rk_table = build_borrowed_table_view_from_columns(right.columns_map(), &right_key_indices)?;

        let raw_indices = match (semi_join, ctx) {
            (true, Some(ctx)) => {
                join::left_semi_join_indices_in(lk_table.table(), rk_table.table(), options.null_eq(), ctx)?
            },
            (false, Some(ctx)) => {
                join::left_anti_join_indices_in(lk_table.table(), rk_table.table(), options.null_eq(), ctx)?
            },
            (true, None) => join::left_semi_join_indices(lk_table.table(), rk_table.table(), options.null_eq())?,
            (false, None) => join::left_anti_join_indices(lk_table.table(), rk_table.table(), options.null_eq())?,
        };

        let indices = Series::from_column(raw_indices.indices)
            .map_err(|e| e.reanchor(operation).with_detail("resource", "join_indices"))?;
        Ok(DataFrameJoinIndices::new(indices))
    }

    pub(crate) fn materialize_join_indices_in_context<I, S>(
        &self,
        indices: &DataFrameJoinIndices,
        columns: I,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.materialize_join_indices_impl(indices, columns, Some(ctx), operation)
    }

    fn materialize_join_indices_impl<I, S>(
        &self,
        indices: &DataFrameJoinIndices,
        columns: I,
        ctx: Option<&ExecutionOverrides<'_>>,
        operation: &'static str,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let selected = self.select(columns).map_err(|e| e.reanchor(operation))?;
        match ctx {
            Some(ctx) => selected.gather_in_context(indices.indices(), ctx).map_err(|e| e.reanchor(operation)),
            None => selected.gather(indices.indices()).map_err(|e| e.reanchor(operation)),
        }
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
        let execution = ResolvedExecution::with_output_domain(
            Arc::clone(&output_domain),
            [self.execution_domain(), right.execution_domain()],
            operation,
        )?;
        let output_domain = Arc::clone(execution.output_domain());
        let resolved_exec_options = exec_options.unwrap_or(execution.exec_options());

        if left_on.len() != right_on.len() {
            return Err(merge_key_count_mismatch_error(operation, left_on.len(), right_on.len()));
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

        let result = join::join_impl(
            lk_table.table(),
            rk_table.table(),
            lp_table.table(),
            rp_table.table(),
            null_equality,
            join_type,
            &output_domain,
            Some(resolved_exec_options),
            operation,
        )?;

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
        return Err(Error::new_with(
            ErrorCode::TypeMismatch,
            operation,
            "cannot coalesce columns with different types",
        )
        .with_fact(FACT_LEFT_TYPE_ID, ldt.id())
        .with_fact(FACT_RIGHT_TYPE_ID, rdt.id()));
    }

    let execution = ResolvedExecution::with_output_domain(
        Arc::clone(&output_domain),
        [lcv.execution_domain(), rcv.execution_domain()],
        operation,
    )?;
    replace::replace_nulls_column_with_exec_options(
        &lcv,
        &rcv,
        execution.output_domain(),
        Some(execution.exec_options()),
        operation,
    )
}

// ---------------------------------------------------------------------------
// High-level merge helpers
// ---------------------------------------------------------------------------

/// Resolve the key names from `MergeOptions`, returning `(left_on, right_on, shared)`.
///
/// `shared` is `true` when `on()` was used (both sides share the same key names),
/// meaning the output should emit one shared key column per key.
fn merge_validation_error(
    operation: &'static str,
    message: &'static str,
    reason: &'static str,
    remedy: &'static str,
) -> Error {
    Error::new_with(ErrorCode::UnsupportedOperation, operation, message)
        .with_detail("reason", reason)
        .with_detail("remedy", remedy)
}

fn merge_key_count_mismatch_error(operation: &'static str, left_count: usize, right_count: usize) -> Error {
    merge_validation_error(
        operation,
        "left_on and right_on must have the same number of key columns",
        "join_key_count_mismatch",
        "set left_on() and right_on() with the same number of key columns",
    )
    .with_detail("left_on_count", left_count.to_string())
    .with_detail("right_on_count", right_count.to_string())
}

pub(super) fn trigger_merge_reassembly_key_count_error_for_test_impl() -> Result<DataFrame> {
    let runtime = Runtime::new()?;
    let left = DataFrame::from_series([("id", runtime.series(&[1i32])?)])?;
    let right = DataFrame::from_series([("id", runtime.series(&[1i32])?)])?;

    merge_join_result_with_suffixes(
        DataFrameJoinSides { left, right },
        &["id", "unexpected_extra_key"],
        true,
        false,
        "_left",
        "_right",
        "dataframe::merge",
    )
}

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
        (None, Some(_), None) => Err(merge_validation_error(
            operation,
            "left_on was specified without right_on; set both or use on()",
            "left_on_without_right_on",
            "set right_on() or use on()",
        )),
        // right_on without left_on
        (None, None, Some(_)) => Err(merge_validation_error(
            operation,
            "right_on was specified without left_on; set both or use on()",
            "right_on_without_left_on",
            "set left_on() or use on()",
        )),
        // on() together with left_on/right_on is ambiguous — treat as error
        (Some(_), _, _) => Err(merge_validation_error(
            operation,
            "on() cannot be combined with left_on()/right_on(); use one or the other",
            "ambiguous_join_key_configuration",
            "use on() or left_on() + right_on(), not both",
        )),
        // Nothing specified
        (None, None, None) => Err(merge_validation_error(
            operation,
            "no join keys specified; use on() or left_on() + right_on()",
            "missing_join_keys",
            "set on() or left_on() + right_on()",
        )),
    }
}

fn resolve_dataframe_join_keys(
    options: &DataFrameJoinOptions,
    operation: &'static str,
) -> Result<(Vec<String>, Vec<String>)> {
    match (options.on_keys(), options.left_on_keys(), options.right_on_keys()) {
        (Some(on), None, None) => {
            let keys: Vec<String> = on.to_vec();
            Ok((keys.clone(), keys))
        },
        (None, Some(lo), Some(ro)) => Ok((lo.to_vec(), ro.to_vec())),
        (None, Some(_), None) => Err(merge_validation_error(
            operation,
            "left_on was specified without right_on; set both or use on()",
            "left_on_without_right_on",
            "set right_on() or use on()",
        )),
        (None, None, Some(_)) => Err(merge_validation_error(
            operation,
            "right_on was specified without left_on; set both or use on()",
            "right_on_without_left_on",
            "set left_on() or use on()",
        )),
        (Some(_), _, _) => Err(merge_validation_error(
            operation,
            "on() cannot be combined with left_on()/right_on(); use one or the other",
            "ambiguous_join_key_configuration",
            "use on() or left_on() + right_on(), not both",
        )),
        (None, None, None) => Err(merge_validation_error(
            operation,
            "no join keys specified; use on() or left_on() + right_on()",
            "missing_join_keys",
            "set on() or left_on() + right_on()",
        )),
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
        return Err(merge_validation_error(
            operation,
            "join key count exceeds result column count",
            "join_result_key_count_exceeds_result_columns",
            "report this as a library bug; join result schema should contain at least one column per key on each side",
        )
        .with_fact(FACT_KEYS_COUNT, key_count)
        .with_detail("left_columns", left_cols.len().to_string())
        .with_detail("right_columns", right_cols.len().to_string()));
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
