use super::*;

impl DataFrame {
    /// Return exact device bytes that must be reserved before building a
    /// reusable filtered-join artifact from this DataFrame's named key columns.
    pub fn filtered_join_pre_build_reservation_size_bytes<I, S>(
        &self,
        build_on: I,
        options: DataFrameFilteredJoinOptions,
    ) -> Result<usize>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let ctx = ExecutionOverrides::new();
        self.filtered_join_pre_build_reservation_size_bytes_in_context(build_on, options, &ctx)
    }

    pub(crate) fn filtered_join_pre_build_reservation_size_bytes_in_context<I, S>(
        &self,
        build_on: I,
        options: DataFrameFilteredJoinOptions,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<usize>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let operation = "dataframe::filtered_join_pre_build_reservation_size_bytes";
        let build_keys = build_on.into_iter().map(|key| key.as_ref().to_string()).collect::<Vec<_>>();
        validate_hash_join_key_names(&build_keys, operation)?;

        let build_key_refs = build_keys.iter().map(|key| key.as_str()).collect::<Vec<_>>();
        let build_key_indices = resolve_column_names(self.columns_map(), &build_key_refs, operation)?;
        let build_key_table = build_borrowed_table_view_from_columns(self.columns_map(), &build_key_indices)?;
        join::FilteredJoin::pre_build_reservation_size_bytes_for_view(
            build_key_table.table(),
            options.to_filtered_join_options(),
            ctx,
            operation,
        )
    }

    /// Build a reusable filtered-join artifact from this DataFrame's named key columns.
    pub fn build_filtered_join<I, S>(
        &self,
        build_on: I,
        options: DataFrameFilteredJoinOptions,
    ) -> Result<DataFrameFilteredJoin>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let ctx = ExecutionOverrides::new();
        self.build_filtered_join_in_context(build_on, options, &ctx)
    }

    pub(crate) fn build_filtered_join_in_context<I, S>(
        &self,
        build_on: I,
        options: DataFrameFilteredJoinOptions,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<DataFrameFilteredJoin>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let operation = "dataframe::build_filtered_join";
        let build_keys = build_on.into_iter().map(|key| key.as_ref().to_string()).collect::<Vec<_>>();
        validate_hash_join_key_names(&build_keys, operation)?;

        let build_key_refs = build_keys.iter().map(|key| key.as_str()).collect::<Vec<_>>();
        resolve_column_names(self.columns_map(), &build_key_refs, operation)?;
        let build_key_frame = self.select(build_keys.clone()).map_err(|error| error.reanchor(operation))?;
        let build_key_table = build_key_frame.to_table_copy().map_err(|error| error.reanchor(operation))?;
        let join = join::FilteredJoin::new_in(build_key_table, options.to_filtered_join_options(), ctx)
            .map_err(|error| error.reanchor(operation))?;

        Ok(DataFrameFilteredJoin { join, build: self.clone(), build_keys, options })
    }
    /// Filter existing two-sided join-index pairs with a predicate.
    pub fn filter_join_indices(
        &self,
        right: &DataFrame,
        left_indices: &DataFrameJoinIndices,
        right_indices: &DataFrameJoinIndices,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
        join_type: join::JoinType,
    ) -> Result<DataFrameJoinPairIndices> {
        let ctx = ExecutionOverrides::new();
        self.filter_join_indices_in_context(right, left_indices, right_indices, predicate, options, join_type, &ctx)
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn filter_join_indices_in_context(
        &self,
        right: &DataFrame,
        left_indices: &DataFrameJoinIndices,
        right_indices: &DataFrameJoinIndices,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
        join_type: join::JoinType,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<DataFrameJoinPairIndices> {
        let operation = "dataframe::with_execution::filter_join_indices";
        let (left_table, right_table) = predicate_tables(self, right, &options, operation)?;
        let left_indices = left_indices.indices().as_column().view().map_err(|e| e.reanchor(operation))?;
        let right_indices = right_indices.indices().as_column().view().map_err(|e| e.reanchor(operation))?;
        let raw = join::filter_join_indices_in(
            left_table.table(),
            right_table.table(),
            &left_indices,
            &right_indices,
            predicate,
            join_type,
            ctx,
        )?;
        pair_indices_from_raw(raw, operation)
    }
}

impl DataFrameFilteredJoin {
    /// Probe this reusable build artifact with left semi semantics.
    pub fn left_semi_probe<I, S>(&self, probe: &DataFrame, probe_on: I) -> Result<DataFrameJoinIndices>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let ctx = ExecutionOverrides::new();
        self.left_semi_probe_in(probe, probe_on, &ctx)
    }

    /// Probe this reusable build artifact with left anti semantics.
    pub fn left_anti_probe<I, S>(&self, probe: &DataFrame, probe_on: I) -> Result<DataFrameJoinIndices>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let ctx = ExecutionOverrides::new();
        self.left_anti_probe_in(probe, probe_on, &ctx)
    }

    /// Begin a retained left semi probe without materializing its gather map.
    pub fn begin_left_semi_probe<I, S>(&self, probe: &DataFrame, probe_on: I) -> Result<DataFrameFilteredJoinProbeState>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let ctx = ExecutionOverrides::new();
        self.begin_left_semi_probe_in(probe, probe_on, &ctx)
    }

    /// Begin a retained left anti probe without materializing its gather map.
    pub fn begin_left_anti_probe<I, S>(&self, probe: &DataFrame, probe_on: I) -> Result<DataFrameFilteredJoinProbeState>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let ctx = ExecutionOverrides::new();
        self.begin_left_anti_probe_in(probe, probe_on, &ctx)
    }

    /// Return the exact output row count for a left semi probe.
    pub fn left_semi_probe_size<I, S>(&self, probe: &DataFrame, probe_on: I) -> Result<usize>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let ctx = ExecutionOverrides::new();
        self.left_semi_probe_size_in(probe, probe_on, &ctx)
    }

    /// Return the exact output row count for a left anti probe.
    pub fn left_anti_probe_size<I, S>(&self, probe: &DataFrame, probe_on: I) -> Result<usize>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let ctx = ExecutionOverrides::new();
        self.left_anti_probe_size_in(probe, probe_on, &ctx)
    }

    /// Probe this reusable build artifact with left semi semantics using an
    /// exact caller-supplied output row count.
    pub fn left_semi_probe_with_output_size<I, S>(
        &self,
        probe: &DataFrame,
        probe_on: I,
        output_size: usize,
    ) -> Result<DataFrameJoinIndices>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let ctx = ExecutionOverrides::new();
        self.left_semi_probe_with_output_size_in(probe, probe_on, output_size, &ctx)
    }

    /// Probe this reusable build artifact with left anti semantics using an
    /// exact caller-supplied output row count.
    pub fn left_anti_probe_with_output_size<I, S>(
        &self,
        probe: &DataFrame,
        probe_on: I,
        output_size: usize,
    ) -> Result<DataFrameJoinIndices>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let ctx = ExecutionOverrides::new();
        self.left_anti_probe_with_output_size_in(probe, probe_on, output_size, &ctx)
    }

    /// Probe this reusable build artifact with left semi semantics using a borrowed execution override.
    pub fn left_semi_probe_in<I, S>(
        &self,
        probe: &DataFrame,
        probe_on: I,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<DataFrameJoinIndices>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.probe_indices_with_context(probe, probe_on, true, None, ctx, "dataframe::filtered_join::left_semi_probe")
    }

    /// Probe this reusable build artifact with left anti semantics using a borrowed execution override.
    pub fn left_anti_probe_in<I, S>(
        &self,
        probe: &DataFrame,
        probe_on: I,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<DataFrameJoinIndices>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.probe_indices_with_context(probe, probe_on, false, None, ctx, "dataframe::filtered_join::left_anti_probe")
    }

    /// Begin a retained left semi probe using a borrowed execution override.
    pub fn begin_left_semi_probe_in<I, S>(
        &self,
        probe: &DataFrame,
        probe_on: I,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<DataFrameFilteredJoinProbeState>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.begin_probe_state_with_context(
            probe,
            probe_on,
            true,
            ctx,
            "dataframe::filtered_join::begin_left_semi_probe",
        )
    }

    /// Begin a retained left anti probe using a borrowed execution override.
    pub fn begin_left_anti_probe_in<I, S>(
        &self,
        probe: &DataFrame,
        probe_on: I,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<DataFrameFilteredJoinProbeState>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.begin_probe_state_with_context(
            probe,
            probe_on,
            false,
            ctx,
            "dataframe::filtered_join::begin_left_anti_probe",
        )
    }

    /// Return the exact output row count for a left semi probe using a borrowed execution override.
    pub fn left_semi_probe_size_in<I, S>(
        &self,
        probe: &DataFrame,
        probe_on: I,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<usize>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.probe_size_with_context(probe, probe_on, true, ctx, "dataframe::filtered_join::left_semi_probe_size")
    }

    /// Return the exact output row count for a left anti probe using a borrowed execution override.
    pub fn left_anti_probe_size_in<I, S>(
        &self,
        probe: &DataFrame,
        probe_on: I,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<usize>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.probe_size_with_context(probe, probe_on, false, ctx, "dataframe::filtered_join::left_anti_probe_size")
    }

    /// Probe this reusable build artifact with left semi semantics using a
    /// borrowed execution override and an exact caller-supplied output row count.
    pub fn left_semi_probe_with_output_size_in<I, S>(
        &self,
        probe: &DataFrame,
        probe_on: I,
        output_size: usize,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<DataFrameJoinIndices>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.probe_indices_with_context(
            probe,
            probe_on,
            true,
            Some(output_size),
            ctx,
            "dataframe::filtered_join::left_semi_probe_with_output_size",
        )
    }

    /// Probe this reusable build artifact with left anti semantics using a
    /// borrowed execution override and an exact caller-supplied output row count.
    pub fn left_anti_probe_with_output_size_in<I, S>(
        &self,
        probe: &DataFrame,
        probe_on: I,
        output_size: usize,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<DataFrameJoinIndices>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.probe_indices_with_context(
            probe,
            probe_on,
            false,
            Some(output_size),
            ctx,
            "dataframe::filtered_join::left_anti_probe_with_output_size",
        )
    }

    /// Materialize selected columns from the probe DataFrame.
    pub fn materialize_probe<I, S>(
        &self,
        probe: &DataFrame,
        indices: &DataFrameJoinIndices,
        columns: I,
    ) -> Result<DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        probe
            .materialize_join_indices(indices, columns)
            .map_err(|error| error.reanchor("dataframe::filtered_join::materialize_probe"))
    }

    fn probe_indices_with_context<I, S>(
        &self,
        probe: &DataFrame,
        probe_on: I,
        semi_join: bool,
        output_size: Option<usize>,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<DataFrameJoinIndices>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let probe_keys = probe_on.into_iter().map(|key| key.as_ref().to_string()).collect::<Vec<_>>();
        validate_hash_join_key_names(&probe_keys, operation)?;
        if probe_keys.len() != self.build_keys.len() {
            return Err(merge_key_count_mismatch_error(operation, probe_keys.len(), self.build_keys.len()));
        }

        let probe_key_refs = probe_keys.iter().map(|key| key.as_str()).collect::<Vec<_>>();
        let probe_key_indices = resolve_column_names(probe.columns_map(), &probe_key_refs, operation)?;
        let probe_key_table = build_borrowed_table_view_from_columns(probe.columns_map(), &probe_key_indices)?;
        let raw_indices = match (semi_join, output_size) {
            (true, Some(output_size)) => {
                self.join.left_semi_join_indices_with_output_size_in(probe_key_table.table(), output_size, ctx)?
            },
            (false, Some(output_size)) => {
                self.join.left_anti_join_indices_with_output_size_in(probe_key_table.table(), output_size, ctx)?
            },
            (true, None) => {
                let state = self.join.begin_left_semi_probe_in(probe_key_table.table(), ctx)?;
                state.materialize_indices_in(ctx)?
            },
            (false, None) => {
                let state = self.join.begin_left_anti_probe_in(probe_key_table.table(), ctx)?;
                state.materialize_indices_in(ctx)?
            },
        };
        single_side_indices_from_raw(raw_indices, operation)
    }

    fn begin_probe_state_with_context<I, S>(
        &self,
        probe: &DataFrame,
        probe_on: I,
        semi_join: bool,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<DataFrameFilteredJoinProbeState>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let probe_keys = probe_on.into_iter().map(|key| key.as_ref().to_string()).collect::<Vec<_>>();
        validate_hash_join_key_names(&probe_keys, operation)?;
        if probe_keys.len() != self.build_keys.len() {
            return Err(merge_key_count_mismatch_error(operation, probe_keys.len(), self.build_keys.len()));
        }

        let probe_key_refs = probe_keys.iter().map(|key| key.as_str()).collect::<Vec<_>>();
        let probe_key_indices = resolve_column_names(probe.columns_map(), &probe_key_refs, operation)?;
        let probe_key_table = build_borrowed_table_view_from_columns(probe.columns_map(), &probe_key_indices)?;
        let state = if semi_join {
            self.join.begin_left_semi_probe_in(probe_key_table.table(), ctx)?
        } else {
            self.join.begin_left_anti_probe_in(probe_key_table.table(), ctx)?
        };
        Ok(DataFrameFilteredJoinProbeState { state })
    }

    fn probe_size_with_context<I, S>(
        &self,
        probe: &DataFrame,
        probe_on: I,
        semi_join: bool,
        ctx: &ExecutionOverrides<'_>,
        operation: &'static str,
    ) -> Result<usize>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let probe_keys = probe_on.into_iter().map(|key| key.as_ref().to_string()).collect::<Vec<_>>();
        validate_hash_join_key_names(&probe_keys, operation)?;
        if probe_keys.len() != self.build_keys.len() {
            return Err(merge_key_count_mismatch_error(operation, probe_keys.len(), self.build_keys.len()));
        }

        let probe_key_refs = probe_keys.iter().map(|key| key.as_str()).collect::<Vec<_>>();
        let probe_key_indices = resolve_column_names(probe.columns_map(), &probe_key_refs, operation)?;
        let probe_key_table = build_borrowed_table_view_from_columns(probe.columns_map(), &probe_key_indices)?;
        if semi_join {
            self.join.left_semi_join_size_in(probe_key_table.table(), ctx)
        } else {
            self.join.left_anti_join_size_in(probe_key_table.table(), ctx)
        }
    }
}
