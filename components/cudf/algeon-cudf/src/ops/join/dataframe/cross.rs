use super::*;

impl DataFrame {
    /// Create a validated projected cross-join plan for bounded execution.
    ///
    /// The returned plan checks output names, source columns, and cartesian
    /// row count once. Its reservation and materialization methods operate
    /// only on the selected output columns.
    pub fn projected_cross_join_plan(
        &self,
        right: &DataFrame,
        options: DataFrameCrossJoinOptions,
    ) -> Result<DataFrameProjectedCrossJoinPlan> {
        self.projected_cross_join_plan_impl(right, options, "dataframe::projected_cross_join_plan")
    }

    pub(crate) fn cross_join_projected_in_context(
        &self,
        right: &DataFrame,
        options: DataFrameCrossJoinOptions,
        ctx: Exec<'_>,
    ) -> Result<Self> {
        let operation = "dataframe::cross_join_projected";
        let plan = self.projected_cross_join_plan_impl(right, options, operation)?;
        plan.materialize_in_context(ctx, operation)
    }

    fn projected_cross_join_plan_impl(
        &self,
        right: &DataFrame,
        options: DataFrameCrossJoinOptions,
        operation: &'static str,
    ) -> Result<DataFrameProjectedCrossJoinPlan> {
        let left_names = options.left_column_names();
        let right_names = options.right_column_names();
        if left_names.is_empty() && right_names.is_empty() {
            return Err(merge_validation_error(
                operation,
                "projected cross join requires at least one selected output column",
                "empty_cross_join_projection",
                "select at least one left or right output column",
            ));
        }
        if left_names.len() != options.left_output_names().len() {
            return Err(merge_key_count_mismatch_error(operation, left_names.len(), options.left_output_names().len()));
        }
        if right_names.len() != options.right_output_names().len() {
            return Err(merge_key_count_mismatch_error(
                operation,
                right_names.len(),
                options.right_output_names().len(),
            ));
        }

        let output_names =
            options.left_output_names().iter().chain(options.right_output_names()).cloned().collect::<Vec<_>>();
        validate_no_duplicate_names(&output_names, operation)?;

        let left_refs = string_refs(left_names);
        let right_refs = string_refs(right_names);
        let left_indices = resolve_column_names(self.columns_map(), &left_refs, operation)?;
        let right_indices = resolve_column_names(right.columns_map(), &right_refs, operation)?;
        let output_rows =
            join::checked_cross_join_output_rows_for_counts(self.num_rows(), right.num_rows(), operation)?;

        Ok(DataFrameProjectedCrossJoinPlan {
            left: self.clone(),
            right: right.clone(),
            left_indices,
            right_indices,
            output_names,
            output_rows,
        })
    }
}

impl DataFrameProjectedCrossJoinPlan {
    /// Checked cartesian output row count.
    pub fn output_rows(&self) -> usize {
        self.output_rows
    }

    /// Output column names in materialized result order.
    pub fn output_column_names(&self) -> &[String] {
        &self.output_names
    }

    /// Return exact bytes retained by the projected output table.
    ///
    /// This performs a lower-level dry-run materialization under the default
    /// retained execution placement and releases the temporary result.
    pub fn reservation_size_bytes(&self, ctx: Exec<'_>) -> Result<usize> {
        self.reservation_size_bytes_in_context(ctx, "dataframe::projected_cross_join_plan::reservation_size_bytes")
    }

    /// Materialize the projected cross join.
    pub fn materialize(&self, ctx: Exec<'_>) -> Result<DataFrame> {
        self.materialize_in_context(ctx, "dataframe::projected_cross_join_plan::materialize")
    }

    fn reservation_size_bytes_in_context(&self, ctx: Exec<'_>, operation: &'static str) -> Result<usize> {
        let execution = self.resolve_selected_execution(ctx, operation)?;
        let left_table = if self.left_indices.is_empty() {
            None
        } else {
            Some(build_borrowed_table_view_from_columns(self.left.columns_map(), &self.left_indices)?)
        };
        let right_table = if self.right_indices.is_empty() {
            None
        } else {
            Some(build_borrowed_table_view_from_columns(self.right.columns_map(), &self.right_indices)?)
        };

        join::projected_cross_join_device_allocated_size_impl(
            left_table.as_ref(),
            right_table.as_ref(),
            self.left.num_rows(),
            self.right.num_rows(),
            execution.exec_options(),
            operation,
            PROJECTED_CROSS_JOIN_RESERVATION_CONTRACT,
        )
        .map_err(|error| error.reanchor(operation))
    }

    pub(crate) fn materialize_in_context(&self, ctx: Exec<'_>, operation: &'static str) -> Result<DataFrame> {
        let execution = self.resolve_selected_execution(ctx, operation)?;
        let left_table = if self.left_indices.is_empty() {
            None
        } else {
            Some(build_borrowed_table_view_from_columns(self.left.columns_map(), &self.left_indices)?)
        };
        let right_table = if self.right_indices.is_empty() {
            None
        } else {
            Some(build_borrowed_table_view_from_columns(self.right.columns_map(), &self.right_indices)?)
        };

        let table = join::projected_cross_join_impl(
            left_table.as_ref(),
            right_table.as_ref(),
            self.left.num_rows(),
            self.right.num_rows(),
            execution.output_domain(),
            execution.exec_options(),
            operation,
        )
        .map_err(|error| error.reanchor(operation))?;
        DataFrame::from_table(table, self.output_names.clone()).map_err(|error| error.reanchor(operation))
    }

    fn resolve_selected_execution(&self, ctx: Exec<'_>, operation: &'static str) -> Result<ResolvedExecution> {
        match (self.left_indices.is_empty(), self.right_indices.is_empty()) {
            (false, false) => {
                ResolvedExecution::binary(self.left.execution_domain(), self.right.execution_domain(), ctx, operation)
            },
            (false, true) => ResolvedExecution::unary(self.left.execution_domain(), ctx, operation),
            (true, false) => ResolvedExecution::unary(self.right.execution_domain(), ctx, operation),
            (true, true) => Err(merge_validation_error(
                operation,
                "projected cross join requires at least one selected output column",
                "empty_cross_join_projection",
                "select at least one left or right output column",
            )),
        }
    }
}
