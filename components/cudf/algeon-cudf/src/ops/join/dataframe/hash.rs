use std::sync::Arc;

use super::*;

impl DataFrame {
    /// Return exact device bytes that must be reserved before building a
    /// reusable hash-join artifact from this DataFrame's named key columns.
    ///
    /// This is a planning contract for bounded engines. It returns the exact
    /// retained bytes for the reusable hash table and build-key device views,
    /// without building or allocating retained payload. libcudf includes retained
    /// null-payload sanitization for flat nullable string keys; nested build
    /// keys remain unsupported.
    pub fn hash_join_pre_build_reservation_size_bytes<I, S>(
        &self,
        build_on: I,
        options: DataFrameHashJoinOptions,
    ) -> Result<usize>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let ctx = ExecutionOverrides::new();
        self.hash_join_pre_build_reservation_size_bytes_in_context(build_on, options, &ctx)
    }

    pub(crate) fn hash_join_pre_build_reservation_size_bytes_in_context<I, S>(
        &self,
        build_on: I,
        options: DataFrameHashJoinOptions,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<usize>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let operation = "dataframe::hash_join_pre_build_reservation_size_bytes";
        let build_keys = build_on.into_iter().map(|key| key.as_ref().to_string()).collect::<Vec<_>>();
        validate_hash_join_key_names(&build_keys, operation)?;

        let build_key_refs = build_keys.iter().map(|key| key.as_str()).collect::<Vec<_>>();
        let build_key_indices = resolve_column_names(self.columns_map(), &build_key_refs, operation)?;
        let build_key_table = build_borrowed_table_view_from_columns(self.columns_map(), &build_key_indices)?;
        match join::HashJoin::pre_build_reservation_size_bytes_for_view(
            &build_key_table,
            options.to_hash_join_options(),
            ctx,
            operation,
        ) {
            Ok(bytes) => Ok(bytes),
            Err(error)
                if join::is_hash_join_retained_preprocessing_reservation_gap(&error)
                    && join::hash_join_retained_preprocessing_fallback_supported(&build_key_table)? =>
            {
                let build_key_frame = self.select(build_keys.clone()).map_err(|error| error.reanchor(operation))?;
                let build_key_table = build_key_frame.to_table_copy().map_err(|error| error.reanchor(operation))?;
                join::HashJoin::pre_build_reservation_size_bytes_in(
                    &build_key_table,
                    options.to_hash_join_options(),
                    ctx,
                )
                .map_err(|error| error.reanchor_if_needed(operation))
            },
            Err(error) => Err(error),
        }
    }

    /// Build a reusable hash-join artifact from this DataFrame's named key columns.
    pub fn build_hash_join<I, S>(&self, build_on: I, options: DataFrameHashJoinOptions) -> Result<DataFrameHashJoin>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let ctx = ExecutionOverrides::new();
        self.build_hash_join_in_context(build_on, options, &ctx)
    }

    pub(crate) fn build_hash_join_in_context<I, S>(
        &self,
        build_on: I,
        options: DataFrameHashJoinOptions,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<DataFrameHashJoin>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let operation = "dataframe::build_hash_join";
        let build_keys = build_on.into_iter().map(|key| key.as_ref().to_string()).collect::<Vec<_>>();
        validate_hash_join_key_names(&build_keys, operation)?;

        let build_key_refs = build_keys.iter().map(|key| key.as_str()).collect::<Vec<_>>();
        resolve_column_names(self.columns_map(), &build_key_refs, operation)?;
        let build_key_frame = self.select(build_keys.clone()).map_err(|error| error.reanchor(operation))?;
        let build_key_table = build_key_frame.to_table_copy().map_err(|error| error.reanchor(operation))?;
        let join = join::HashJoin::new_in(build_key_table, options.to_hash_join_options(), ctx)
            .map_err(|error| error.reanchor(operation))?;

        Ok(DataFrameHashJoin { join, build: self.clone(), build_keys, options })
    }

    /// Merge using a borrowed per-operation execution override.
    pub(crate) fn merge_in_context(
        &self,
        right: &DataFrame,
        options: join::MergeOptions,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<Self> {
        let output_domain = ctx.resolve_output_domain(self.execution_domain())?;
        self.merge_impl(right, options, Some(output_domain), "dataframe::with_execution::merge")
    }

    /// Merge-split using a borrowed per-operation execution override.
    pub(crate) fn merge_split_in_context(
        &self,
        right: &DataFrame,
        options: join::MergeOptions,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<DataFrameJoinSides> {
        let operation: &'static str = "dataframe::with_execution::merge_split";
        let (left_on_strs, right_on_strs, _) =
            resolve_join_keys(options.on_keys(), options.left_on_keys(), options.right_on_keys(), operation)?;
        let left_on_refs: Vec<&str> = left_on_strs.iter().map(|s| s.as_str()).collect();
        let right_on_refs: Vec<&str> = right_on_strs.iter().map(|s| s.as_str()).collect();

        let output_domain = ctx.resolve_output_domain(self.execution_domain())?;
        self.join_split_on_impl_with_null_eq_and_context(
            right,
            JoinSplitKeys { left_on: &left_on_refs, right_on: &right_on_refs },
            JoinSplitCall {
                join_type: options.join_type(),
                null_equality: options.null_eq(),
                forced_output_domain: Some(output_domain),
                operation,
            },
        )
    }
    #[cfg(any(test, feature = "test-hooks"))]
    pub(crate) fn __debug_merge_in_context(
        &self,
        right: &DataFrame,
        options: join::MergeOptions,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<Self> {
        self.merge_in_context(right, options, ctx)
    }

    /// Produce inner join row-index pairs using a borrowed per-operation execution override.
    pub(crate) fn inner_join_indices_in_context(
        &self,
        right: &DataFrame,
        options: DataFrameJoinOptions,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<DataFrameJoinPairIndices> {
        self.two_side_join_indices_impl(
            right,
            options,
            join::JoinType::Inner,
            Some(ctx),
            "dataframe::with_execution::inner_join_indices",
        )
    }

    /// Produce left join row-index pairs using a borrowed per-operation execution override.
    pub(crate) fn left_join_indices_in_context(
        &self,
        right: &DataFrame,
        options: DataFrameJoinOptions,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<DataFrameJoinPairIndices> {
        self.two_side_join_indices_impl(
            right,
            options,
            join::JoinType::Left,
            Some(ctx),
            "dataframe::with_execution::left_join_indices",
        )
    }

    /// Produce full join row-index pairs using a borrowed per-operation execution override.
    pub(crate) fn full_join_indices_in_context(
        &self,
        right: &DataFrame,
        options: DataFrameJoinOptions,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<DataFrameJoinPairIndices> {
        self.two_side_join_indices_impl(
            right,
            options,
            join::JoinType::Full,
            Some(ctx),
            "dataframe::with_execution::full_join_indices",
        )
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
    /// - `on(keys)`: shared key column names present in both sides. Output
    ///   contains one shared key column per key.
    /// - `left_on(keys)` + `right_on(keys)`: asymmetric key names. Both key
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
        self.merge_impl(right, options, None, "dataframe::merge")
    }

    fn merge_impl(
        &self,
        right: &DataFrame,
        options: join::MergeOptions,
        forced_output_domain: Option<Arc<ExecutionDomain>>,
        operation: &'static str,
    ) -> Result<Self> {
        let operation: &'static str = operation;

        // Resolve key names from options.
        let (left_on_strs, right_on_strs, shared_keys) =
            resolve_join_keys(options.on_keys(), options.left_on_keys(), options.right_on_keys(), operation)?;

        let left_on_refs: Vec<&str> = left_on_strs.iter().map(|s| s.as_str()).collect();
        let right_on_refs: Vec<&str> = right_on_strs.iter().map(|s| s.as_str()).collect();

        let join_type = options.join_type();

        let split = self.join_split_on_impl_with_null_eq_and_context(
            right,
            JoinSplitKeys { left_on: &left_on_refs, right_on: &right_on_refs },
            JoinSplitCall { join_type, null_equality: options.null_eq(), forced_output_domain, operation },
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

        let (left_on_strs, right_on_strs, _) =
            resolve_join_keys(options.on_keys(), options.left_on_keys(), options.right_on_keys(), operation)?;

        let left_on_refs: Vec<&str> = left_on_strs.iter().map(|s| s.as_str()).collect();
        let right_on_refs: Vec<&str> = right_on_strs.iter().map(|s| s.as_str()).collect();

        self.join_split_on_impl_with_null_eq_and_context(
            right,
            JoinSplitKeys { left_on: &left_on_refs, right_on: &right_on_refs },
            JoinSplitCall {
                join_type: options.join_type(),
                null_equality: options.null_eq(),
                forced_output_domain: None,
                operation,
            },
        )
    }

    /// Return device-side gather index pairs for an inner equality join.
    ///
    /// This is an index-producing query-planning helper. It allocates only the
    /// left and right `INT32` device gather maps and does not copy payload
    /// columns. Duplicate key matches produce one row pair per match.
    pub fn inner_join_indices(
        &self,
        right: &DataFrame,
        options: DataFrameJoinOptions,
    ) -> Result<DataFrameJoinPairIndices> {
        self.two_side_join_indices_impl(right, options, join::JoinType::Inner, None, "dataframe::inner_join_indices")
    }

    /// Return device-side gather index pairs for a left equality join.
    ///
    /// Misses on the right side contain libcudf's no-match sentinel. Use
    /// [`materialize_join_indices`](Self::materialize_join_indices) to gather
    /// selected right-side columns with nullification for those misses.
    pub fn left_join_indices(
        &self,
        right: &DataFrame,
        options: DataFrameJoinOptions,
    ) -> Result<DataFrameJoinPairIndices> {
        self.two_side_join_indices_impl(right, options, join::JoinType::Left, None, "dataframe::left_join_indices")
    }

    /// Return device-side gather index pairs for a full equality join.
    ///
    /// Misses on either side contain libcudf's no-match sentinel. Use
    /// [`materialize_join_indices`](Self::materialize_join_indices) to gather
    /// selected columns with nullification for unmatched rows.
    pub fn full_join_indices(
        &self,
        right: &DataFrame,
        options: DataFrameJoinOptions,
    ) -> Result<DataFrameJoinPairIndices> {
        self.two_side_join_indices_impl(right, options, join::JoinType::Full, None, "dataframe::full_join_indices")
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

    /// Return the exact retained device allocation footprint of
    /// [`materialize_join_indices`](Self::materialize_join_indices) for the
    /// selected payload columns.
    ///
    /// This is a lower-level reservation contract for bounded engines. It is
    /// implemented in the cuDF layer by materializing non-empty projected
    /// gather results, observing the table allocation footprint, and releasing
    /// the temporary result before returning. Empty gather maps return zero
    /// without executing the gather. Callers should reserve the returned byte
    /// count before performing the durable materialization.
    ///
    /// Cost note: unlike
    /// [`hash_join_pre_build_reservation_size_bytes`](Self::hash_join_pre_build_reservation_size_bytes),
    /// this contract executes the non-empty gather end-to-end and discards the
    /// result. Its cost is comparable to
    /// [`materialize_join_indices`](Self::materialize_join_indices) itself, so
    /// reserve-then-materialize roughly doubles the gather work.
    ///
    /// Doc-only deprecation: this helper is retained for legacy callers that
    /// need an exact post-gather allocation measurement. It is an executable
    /// preflight, not a closed-form bound, and can allocate the full gather
    /// working set before the caller has reserved the returned bytes. New
    /// bounded execution paths should prefer an explicit closed-form
    /// reservation contract and then reconcile the observed retained size after
    /// materialization.
    pub fn materialize_join_indices_reservation_size_bytes<I, S>(
        &self,
        indices: &DataFrameJoinIndices,
        columns: I,
    ) -> Result<usize>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.materialize_join_indices_reservation_size_bytes_impl(
            indices,
            columns,
            None,
            "dataframe::materialize_join_indices_reservation_size_bytes",
        )
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

    fn two_side_join_indices_impl(
        &self,
        right: &DataFrame,
        options: DataFrameJoinOptions,
        join_type: join::JoinType,
        ctx: Option<&ExecutionOverrides<'_>>,
        operation: &'static str,
    ) -> Result<DataFrameJoinPairIndices> {
        let (left_on_strs, right_on_strs, _) =
            resolve_join_keys(options.on_keys(), options.left_on_keys(), options.right_on_keys(), operation)?;
        let left_on_refs: Vec<&str> = left_on_strs.iter().map(|s| s.as_str()).collect();
        let right_on_refs: Vec<&str> = right_on_strs.iter().map(|s| s.as_str()).collect();

        let left_key_indices = resolve_column_names(self.columns_map(), &left_on_refs, operation)?;
        let right_key_indices = resolve_column_names(right.columns_map(), &right_on_refs, operation)?;
        let lk_table = build_borrowed_table_view_from_columns(self.columns_map(), &left_key_indices)?;
        let rk_table = build_borrowed_table_view_from_columns(right.columns_map(), &right_key_indices)?;

        let raw_indices = match (join_type, ctx) {
            (join::JoinType::Inner, Some(ctx)) => {
                join::inner_join_indices_in(&lk_table, &rk_table, options.null_eq(), ctx)?
            },
            (join::JoinType::Left, Some(ctx)) => {
                join::left_join_indices_in(&lk_table, &rk_table, options.null_eq(), ctx)?
            },
            (join::JoinType::Full, Some(ctx)) => {
                join::full_join_indices_in(&lk_table, &rk_table, options.null_eq(), ctx)?
            },
            (join::JoinType::Inner, None) => join::inner_join_indices(&lk_table, &rk_table, options.null_eq())?,
            (join::JoinType::Left, None) => join::left_join_indices(&lk_table, &rk_table, options.null_eq())?,
            (join::JoinType::Full, None) => join::full_join_indices(&lk_table, &rk_table, options.null_eq())?,
        };

        let left = Series::from_column(raw_indices.left)
            .map_err(|e| e.reanchor(operation).with_detail("resource", "left_join_indices"))?;
        let right = Series::from_column(raw_indices.right)
            .map_err(|e| e.reanchor(operation).with_detail("resource", "right_join_indices"))?;
        Ok(DataFrameJoinPairIndices::new(left, right))
    }

    fn single_side_join_indices_impl(
        &self,
        right: &DataFrame,
        options: DataFrameJoinOptions,
        semi_join: bool,
        ctx: Option<&ExecutionOverrides<'_>>,
        operation: &'static str,
    ) -> Result<DataFrameJoinIndices> {
        let (left_on_strs, right_on_strs, _) =
            resolve_join_keys(options.on_keys(), options.left_on_keys(), options.right_on_keys(), operation)?;
        let left_on_refs: Vec<&str> = left_on_strs.iter().map(|s| s.as_str()).collect();
        let right_on_refs: Vec<&str> = right_on_strs.iter().map(|s| s.as_str()).collect();

        let left_key_indices = resolve_column_names(self.columns_map(), &left_on_refs, operation)?;
        let right_key_indices = resolve_column_names(right.columns_map(), &right_on_refs, operation)?;
        let lk_table = build_borrowed_table_view_from_columns(self.columns_map(), &left_key_indices)?;
        let rk_table = build_borrowed_table_view_from_columns(right.columns_map(), &right_key_indices)?;

        let raw_indices = match (semi_join, ctx) {
            (true, Some(ctx)) => join::left_semi_join_indices_in(&lk_table, &rk_table, options.null_eq(), ctx)?,
            (false, Some(ctx)) => join::left_anti_join_indices_in(&lk_table, &rk_table, options.null_eq(), ctx)?,
            (true, None) => join::left_semi_join_indices(&lk_table, &rk_table, options.null_eq())?,
            (false, None) => join::left_anti_join_indices(&lk_table, &rk_table, options.null_eq())?,
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

    pub(crate) fn materialize_join_indices_reservation_size_bytes_in_context<I, S>(
        &self,
        indices: &DataFrameJoinIndices,
        columns: I,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<usize>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.materialize_join_indices_reservation_size_bytes_impl(indices, columns, Some(ctx), operation)
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

    fn materialize_join_indices_reservation_size_bytes_impl<I, S>(
        &self,
        indices: &DataFrameJoinIndices,
        columns: I,
        ctx: Option<&ExecutionOverrides<'_>>,
        operation: &'static str,
    ) -> Result<usize>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let selected = self.select(columns).map_err(|e| e.reanchor(operation))?;
        match ctx {
            Some(ctx) => selected.gather_device_allocated_size_in_context(
                indices.indices(),
                ctx,
                operation,
                MATERIALIZE_JOIN_INDICES_RESERVATION_CONTRACT,
            ),
            None => {
                let default_ctx = ExecutionOverrides::new();
                selected.gather_device_allocated_size_in_context(
                    indices.indices(),
                    &default_ctx,
                    operation,
                    MATERIALIZE_JOIN_INDICES_RESERVATION_CONTRACT,
                )
            },
        }
        .map_err(|e| e.reanchor(operation))
    }

    /// Internal split-join dispatch with configurable null equality.
    fn join_split_on_impl_with_null_eq_and_context(
        &self,
        right: &DataFrame,
        keys: JoinSplitKeys<'_, '_>,
        call: JoinSplitCall,
    ) -> Result<DataFrameJoinSides> {
        let operation = call.operation;
        let join_type = call.join_type;
        let null_equality = call.null_equality;
        let output_domain = call.forced_output_domain.unwrap_or_else(|| Arc::clone(self.execution_domain()));
        let execution = ResolvedExecution::with_output_domain(
            Arc::clone(&output_domain),
            [self.execution_domain(), right.execution_domain()],
            operation,
        )?;
        let output_domain = Arc::clone(execution.output_domain());

        // Resolve key column names in both sides.
        let left_key_indices = resolve_column_names(self.columns_map(), keys.left_on, operation)?;
        let right_key_indices = resolve_column_names(right.columns_map(), keys.right_on, operation)?;

        // Compute payload indices (all columns not in keys, preserving order).
        let left_payload_indices = payload_indices(self.columns_map(), &left_key_indices);
        let right_payload_indices = payload_indices(right.columns_map(), &right_key_indices);

        // Build borrowed key and payload table views over the existing columns.
        let lk_table = build_borrowed_table_view_from_columns(self.columns_map(), &left_key_indices)?;
        let rk_table = build_borrowed_table_view_from_columns(right.columns_map(), &right_key_indices)?;
        let lp_table = build_borrowed_table_view_from_columns(self.columns_map(), &left_payload_indices)?;
        let rp_table = build_borrowed_table_view_from_columns(right.columns_map(), &right_payload_indices)?;

        let result = join::join_impl(
            join::SplitJoinTables {
                left_keys: &lk_table,
                right_keys: &rk_table,
                left_payload: &lp_table,
                right_payload: &rp_table,
            },
            null_equality,
            join_type,
            join::JoinCallContext::new(&output_domain, execution.exec_options(), operation),
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

impl DataFrameHashJoin {
    /// Probe this reusable build artifact with inner-join semantics.
    pub fn inner_probe<I, S>(&self, probe: &DataFrame, probe_on: I) -> Result<DataFrameHashJoinPairIndices>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let ctx = ExecutionOverrides::new();
        self.inner_probe_in(probe, probe_on, &ctx)
    }

    /// Probe this reusable build artifact with inner-join semantics using an
    /// exact caller-supplied output row count.
    pub fn inner_probe_with_output_size<I, S>(
        &self,
        probe: &DataFrame,
        probe_on: I,
        output_size: usize,
    ) -> Result<DataFrameHashJoinPairIndices>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let ctx = ExecutionOverrides::new();
        self.inner_probe_with_output_size_in(probe, probe_on, output_size, &ctx)
    }

    /// Probe this reusable build artifact with left-join semantics.
    pub fn left_probe<I, S>(&self, probe: &DataFrame, probe_on: I) -> Result<DataFrameHashJoinPairIndices>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let ctx = ExecutionOverrides::new();
        self.left_probe_in(probe, probe_on, &ctx)
    }

    /// Probe this reusable build artifact with left-join semantics using an
    /// exact caller-supplied output row count.
    pub fn left_probe_with_output_size<I, S>(
        &self,
        probe: &DataFrame,
        probe_on: I,
        output_size: usize,
    ) -> Result<DataFrameHashJoinPairIndices>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let ctx = ExecutionOverrides::new();
        self.left_probe_with_output_size_in(probe, probe_on, output_size, &ctx)
    }

    /// Probe this reusable build artifact with full-join semantics.
    pub fn full_probe<I, S>(&self, probe: &DataFrame, probe_on: I) -> Result<DataFrameHashJoinPairIndices>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let ctx = ExecutionOverrides::new();
        self.full_probe_in(probe, probe_on, &ctx)
    }

    /// Probe this reusable build artifact with full-join semantics using an
    /// exact caller-supplied output row count.
    pub fn full_probe_with_output_size<I, S>(
        &self,
        probe: &DataFrame,
        probe_on: I,
        output_size: usize,
    ) -> Result<DataFrameHashJoinPairIndices>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let ctx = ExecutionOverrides::new();
        self.full_probe_with_output_size_in(probe, probe_on, output_size, &ctx)
    }

    /// Return the exact output row count for an inner probe.
    pub fn inner_probe_size<I, S>(&self, probe: &DataFrame, probe_on: I) -> Result<usize>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let ctx = ExecutionOverrides::new();
        self.inner_probe_size_in(probe, probe_on, &ctx)
    }

    /// Return the exact output row count for a left probe.
    pub fn left_probe_size<I, S>(&self, probe: &DataFrame, probe_on: I) -> Result<usize>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let ctx = ExecutionOverrides::new();
        self.left_probe_size_in(probe, probe_on, &ctx)
    }

    /// Return the exact output row count for a full probe.
    pub fn full_probe_size<I, S>(&self, probe: &DataFrame, probe_on: I) -> Result<usize>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let ctx = ExecutionOverrides::new();
        self.full_probe_size_in(probe, probe_on, &ctx)
    }

    /// Probe this reusable build artifact with inner-join semantics using a borrowed execution override.
    pub fn inner_probe_in<I, S>(
        &self,
        probe: &DataFrame,
        probe_on: I,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<DataFrameHashJoinPairIndices>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.probe_indices_with_context(
            probe,
            probe_on,
            join::JoinType::Inner,
            None,
            ctx,
            "dataframe::hash_join::inner_probe",
        )
    }

    /// Probe this reusable build artifact with inner-join semantics using a
    /// borrowed execution override and an exact caller-supplied output row count.
    pub fn inner_probe_with_output_size_in<I, S>(
        &self,
        probe: &DataFrame,
        probe_on: I,
        output_size: usize,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<DataFrameHashJoinPairIndices>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.probe_indices_with_context(
            probe,
            probe_on,
            join::JoinType::Inner,
            Some(output_size),
            ctx,
            "dataframe::hash_join::inner_probe_with_output_size",
        )
    }

    /// Return the exact output row count for an inner probe using a borrowed execution override.
    pub fn inner_probe_size_in<I, S>(
        &self,
        probe: &DataFrame,
        probe_on: I,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<usize>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.probe_size_with_context(
            probe,
            probe_on,
            join::JoinType::Inner,
            ctx,
            "dataframe::hash_join::inner_probe_size",
        )
    }

    /// Probe this reusable build artifact with left-join semantics using a borrowed execution override.
    pub fn left_probe_in<I, S>(
        &self,
        probe: &DataFrame,
        probe_on: I,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<DataFrameHashJoinPairIndices>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.probe_indices_with_context(
            probe,
            probe_on,
            join::JoinType::Left,
            None,
            ctx,
            "dataframe::hash_join::left_probe",
        )
    }

    /// Probe this reusable build artifact with left-join semantics using a
    /// borrowed execution override and an exact caller-supplied output row count.
    pub fn left_probe_with_output_size_in<I, S>(
        &self,
        probe: &DataFrame,
        probe_on: I,
        output_size: usize,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<DataFrameHashJoinPairIndices>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.probe_indices_with_context(
            probe,
            probe_on,
            join::JoinType::Left,
            Some(output_size),
            ctx,
            "dataframe::hash_join::left_probe_with_output_size",
        )
    }

    /// Return the exact output row count for a left probe using a borrowed execution override.
    pub fn left_probe_size_in<I, S>(
        &self,
        probe: &DataFrame,
        probe_on: I,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<usize>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.probe_size_with_context(
            probe,
            probe_on,
            join::JoinType::Left,
            ctx,
            "dataframe::hash_join::left_probe_size",
        )
    }

    /// Probe this reusable build artifact with full-join semantics using a borrowed execution override.
    pub fn full_probe_in<I, S>(
        &self,
        probe: &DataFrame,
        probe_on: I,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<DataFrameHashJoinPairIndices>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.probe_indices_with_context(
            probe,
            probe_on,
            join::JoinType::Full,
            None,
            ctx,
            "dataframe::hash_join::full_probe",
        )
    }

    /// Probe this reusable build artifact with full-join semantics using a
    /// borrowed execution override and an exact caller-supplied output row count.
    pub fn full_probe_with_output_size_in<I, S>(
        &self,
        probe: &DataFrame,
        probe_on: I,
        output_size: usize,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<DataFrameHashJoinPairIndices>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.probe_indices_with_context(
            probe,
            probe_on,
            join::JoinType::Full,
            Some(output_size),
            ctx,
            "dataframe::hash_join::full_probe_with_output_size",
        )
    }

    /// Return the exact output row count for a full probe using a borrowed execution override.
    pub fn full_probe_size_in<I, S>(
        &self,
        probe: &DataFrame,
        probe_on: I,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<usize>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.probe_size_with_context(
            probe,
            probe_on,
            join::JoinType::Full,
            ctx,
            "dataframe::hash_join::full_probe_size",
        )
    }

    /// Materialize selected columns from the probe DataFrame.
    pub fn materialize_probe<I, S>(
        &self,
        probe: &DataFrame,
        indices: &DataFrameHashJoinPairIndices,
        columns: I,
    ) -> Result<DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        probe
            .materialize_join_indices(indices.probe(), columns)
            .map_err(|error| error.reanchor("dataframe::hash_join::materialize_probe"))
    }

    /// Materialize selected columns from the retained build DataFrame.
    pub fn materialize_build<I, S>(&self, indices: &DataFrameHashJoinPairIndices, columns: I) -> Result<DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.build
            .materialize_join_indices(indices.build(), columns)
            .map_err(|error| error.reanchor("dataframe::hash_join::materialize_build"))
    }

    /// Materialize selected probe and build columns.
    pub fn materialize<PI, PS, BI, BS>(
        &self,
        probe: &DataFrame,
        indices: &DataFrameHashJoinPairIndices,
        probe_columns: PI,
        build_columns: BI,
    ) -> Result<DataFrameHashJoinMaterialized>
    where
        PI: IntoIterator<Item = PS>,
        PS: AsRef<str>,
        BI: IntoIterator<Item = BS>,
        BS: AsRef<str>,
    {
        Ok(DataFrameHashJoinMaterialized {
            probe: self.materialize_probe(probe, indices, probe_columns)?,
            build: self.materialize_build(indices, build_columns)?,
        })
    }

    fn probe_indices_with_context<I, S>(
        &self,
        probe: &DataFrame,
        probe_on: I,
        join_type: join::JoinType,
        output_size: Option<usize>,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<DataFrameHashJoinPairIndices>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let probe_key_table = resolve_probe_key_table(probe, probe_on, self.build_keys.len(), operation)?;
        let raw_indices = match (join_type, output_size) {
            (join::JoinType::Inner, None) => self.join.inner_join_indices_in(&probe_key_table, ctx)?,
            (join::JoinType::Left, None) => self.join.left_join_indices_in(&probe_key_table, ctx)?,
            (join::JoinType::Full, None) => self.join.full_join_indices_in(&probe_key_table, ctx)?,
            (join::JoinType::Inner, Some(output_size)) => {
                self.join.inner_join_indices_with_output_size_in(&probe_key_table, output_size, ctx)?
            },
            (join::JoinType::Left, Some(output_size)) => {
                self.join.left_join_indices_with_output_size_in(&probe_key_table, output_size, ctx)?
            },
            (join::JoinType::Full, Some(output_size)) => {
                self.join.full_join_indices_with_output_size_in(&probe_key_table, output_size, ctx)?
            },
        };
        let probe = Series::from_column(raw_indices.left)
            .map_err(|error| error.reanchor(operation).with_detail("resource", "probe_join_indices"))?;
        let build = Series::from_column(raw_indices.right)
            .map_err(|error| error.reanchor(operation).with_detail("resource", "build_join_indices"))?;
        Ok(DataFrameHashJoinPairIndices::new(probe, build))
    }

    fn probe_size_with_context<I, S>(
        &self,
        probe: &DataFrame,
        probe_on: I,
        join_type: join::JoinType,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<usize>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let probe_key_table = resolve_probe_key_table(probe, probe_on, self.build_keys.len(), operation)?;
        let size = match join_type {
            join::JoinType::Inner => self.join.inner_join_size_in(&probe_key_table, ctx)?,
            join::JoinType::Left => self.join.left_join_size_in(&probe_key_table, ctx)?,
            join::JoinType::Full => self.join.full_join_size_in(&probe_key_table, ctx)?,
        };
        Ok(size)
    }
}
