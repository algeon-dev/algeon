use super::*;

impl DataFrame {
    /// Conditional inner join indices for rows whose AST predicate evaluates true.
    pub fn conditional_inner_join_indices(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
    ) -> Result<DataFrameJoinPairIndices> {
        let ctx = ExecutionOverrides::new();
        self.conditional_join_indices_in_context(right, predicate, options, join::JoinType::Inner, &ctx)
    }

    /// Exact output row count for a conditional inner join.
    pub fn conditional_inner_join_size(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
    ) -> Result<usize> {
        let ctx = ExecutionOverrides::new();
        self.conditional_join_size_in_context(
            right,
            predicate,
            options,
            join::SizedConditionalPairJoinType::Inner,
            &ctx,
        )
    }

    /// Conditional inner join indices using an exact caller-supplied output row count.
    pub fn conditional_inner_join_indices_with_output_size(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
        output_size: usize,
    ) -> Result<DataFrameJoinPairIndices> {
        let ctx = ExecutionOverrides::new();
        self.conditional_join_indices_with_output_size_in_context(
            right,
            predicate,
            options,
            join::SizedConditionalPairJoinType::Inner,
            output_size,
            &ctx,
        )
    }

    /// Conditional left join indices for rows whose AST predicate evaluates true.
    pub fn conditional_left_join_indices(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
    ) -> Result<DataFrameJoinPairIndices> {
        let ctx = ExecutionOverrides::new();
        self.conditional_join_indices_in_context(right, predicate, options, join::JoinType::Left, &ctx)
    }

    /// Exact output row count for a conditional left join.
    pub fn conditional_left_join_size(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
    ) -> Result<usize> {
        let ctx = ExecutionOverrides::new();
        self.conditional_join_size_in_context(right, predicate, options, join::SizedConditionalPairJoinType::Left, &ctx)
    }

    /// Conditional left join indices using an exact caller-supplied output row count.
    pub fn conditional_left_join_indices_with_output_size(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
        output_size: usize,
    ) -> Result<DataFrameJoinPairIndices> {
        let ctx = ExecutionOverrides::new();
        self.conditional_join_indices_with_output_size_in_context(
            right,
            predicate,
            options,
            join::SizedConditionalPairJoinType::Left,
            output_size,
            &ctx,
        )
    }

    /// Conditional full join indices for rows whose AST predicate evaluates true.
    pub fn conditional_full_join_indices(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
    ) -> Result<DataFrameJoinPairIndices> {
        let ctx = ExecutionOverrides::new();
        self.conditional_join_indices_in_context(right, predicate, options, join::JoinType::Full, &ctx)
    }

    /// Begin retained size-data computation for a conditional full join.
    pub fn conditional_full_join_size_data<'a>(
        &self,
        right: &DataFrame,
        predicate: &'a AstExpression,
        options: DataFramePredicateJoinOptions,
    ) -> Result<DataFrameConditionalFullJoinSizeData<'a>> {
        let ctx = ExecutionOverrides::new();
        self.conditional_full_join_size_data_in_context(right, predicate, options, &ctx)
    }

    /// Conditional left semi join indices.
    pub fn conditional_left_semi_join_indices(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
    ) -> Result<DataFrameJoinIndices> {
        let ctx = ExecutionOverrides::new();
        self.conditional_single_side_join_indices_in_context(right, predicate, options, true, &ctx)
    }

    /// Exact output row count for a conditional left semi join.
    pub fn conditional_left_semi_join_size(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
    ) -> Result<usize> {
        let ctx = ExecutionOverrides::new();
        self.conditional_single_side_join_size_in_context(right, predicate, options, true, &ctx)
    }

    /// Conditional left semi join indices using an exact caller-supplied output row count.
    pub fn conditional_left_semi_join_indices_with_output_size(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
        output_size: usize,
    ) -> Result<DataFrameJoinIndices> {
        let ctx = ExecutionOverrides::new();
        self.conditional_single_side_join_indices_with_output_size_in_context(
            right,
            predicate,
            options,
            true,
            output_size,
            &ctx,
        )
    }

    /// Conditional left anti join indices.
    pub fn conditional_left_anti_join_indices(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
    ) -> Result<DataFrameJoinIndices> {
        let ctx = ExecutionOverrides::new();
        self.conditional_single_side_join_indices_in_context(right, predicate, options, false, &ctx)
    }

    /// Exact output row count for a conditional left anti join.
    pub fn conditional_left_anti_join_size(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
    ) -> Result<usize> {
        let ctx = ExecutionOverrides::new();
        self.conditional_single_side_join_size_in_context(right, predicate, options, false, &ctx)
    }

    /// Conditional left anti join indices using an exact caller-supplied output row count.
    pub fn conditional_left_anti_join_indices_with_output_size(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
        output_size: usize,
    ) -> Result<DataFrameJoinIndices> {
        let ctx = ExecutionOverrides::new();
        self.conditional_single_side_join_indices_with_output_size_in_context(
            right,
            predicate,
            options,
            false,
            output_size,
            &ctx,
        )
    }

    pub(crate) fn conditional_join_indices_in_context(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
        join_type: join::JoinType,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<DataFrameJoinPairIndices> {
        let operation = "dataframe::with_execution::conditional_join_indices";
        let (left_table, right_table) = predicate_tables(self, right, &options, operation)?;
        let raw = match join_type {
            join::JoinType::Inner => {
                join::conditional_inner_join_indices_in(left_table.table(), right_table.table(), predicate, ctx)?
            },
            join::JoinType::Left => {
                join::conditional_left_join_indices_in(left_table.table(), right_table.table(), predicate, ctx)?
            },
            join::JoinType::Full => {
                join::conditional_full_join_indices_in(left_table.table(), right_table.table(), predicate, ctx)?
            },
        };
        pair_indices_from_raw(raw, operation)
    }

    pub(crate) fn conditional_full_join_size_data_in_context<'a>(
        &self,
        right: &DataFrame,
        predicate: &'a AstExpression,
        options: DataFramePredicateJoinOptions,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<DataFrameConditionalFullJoinSizeData<'a>> {
        let operation = "dataframe::with_execution::conditional_full_join_size_data";
        let (left_table, right_table) = predicate_tables(self, right, &options, operation)?;
        let state =
            join::conditional_full_join_begin_size_data_in(left_table.table(), right_table.table(), predicate, ctx)?;
        Ok(DataFrameConditionalFullJoinSizeData { state, _left: self.clone(), _right: right.clone() })
    }

    pub(crate) fn conditional_join_indices_with_output_size_in_context(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
        join_type: join::SizedConditionalPairJoinType,
        output_size: usize,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<DataFrameJoinPairIndices> {
        let operation = "dataframe::with_execution::conditional_join_indices_with_output_size";
        let (left_table, right_table) = predicate_tables(self, right, &options, operation)?;
        let raw = match join_type {
            join::SizedConditionalPairJoinType::Inner => join::conditional_inner_join_indices_with_output_size_in(
                left_table.table(),
                right_table.table(),
                predicate,
                output_size,
                ctx,
            )?,
            join::SizedConditionalPairJoinType::Left => join::conditional_left_join_indices_with_output_size_in(
                left_table.table(),
                right_table.table(),
                predicate,
                output_size,
                ctx,
            )?,
        };
        pair_indices_from_raw(raw, operation)
    }

    pub(crate) fn conditional_join_size_in_context(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
        join_type: join::SizedConditionalPairJoinType,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<usize> {
        let operation = "dataframe::with_execution::conditional_join_size";
        let (left_table, right_table) = predicate_tables(self, right, &options, operation)?;
        match join_type {
            join::SizedConditionalPairJoinType::Inner => {
                join::conditional_inner_join_size_in(left_table.table(), right_table.table(), predicate, ctx)
            },
            join::SizedConditionalPairJoinType::Left => {
                join::conditional_left_join_size_in(left_table.table(), right_table.table(), predicate, ctx)
            },
        }
    }

    pub(crate) fn conditional_single_side_join_indices_in_context(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
        semi_join: bool,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<DataFrameJoinIndices> {
        let operation = "dataframe::with_execution::conditional_single_side_join_indices";
        let (left_table, right_table) = predicate_tables(self, right, &options, operation)?;
        let raw = if semi_join {
            join::conditional_left_semi_join_indices_in(left_table.table(), right_table.table(), predicate, ctx)?
        } else {
            join::conditional_left_anti_join_indices_in(left_table.table(), right_table.table(), predicate, ctx)?
        };
        single_side_indices_from_raw(raw, operation)
    }

    pub(crate) fn conditional_single_side_join_indices_with_output_size_in_context(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
        semi_join: bool,
        output_size: usize,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<DataFrameJoinIndices> {
        let operation = "dataframe::with_execution::conditional_single_side_join_indices_with_output_size";
        let (left_table, right_table) = predicate_tables(self, right, &options, operation)?;
        let raw = if semi_join {
            join::conditional_left_semi_join_indices_with_output_size_in(
                left_table.table(),
                right_table.table(),
                predicate,
                output_size,
                ctx,
            )?
        } else {
            join::conditional_left_anti_join_indices_with_output_size_in(
                left_table.table(),
                right_table.table(),
                predicate,
                output_size,
                ctx,
            )?
        };
        single_side_indices_from_raw(raw, operation)
    }

    pub(crate) fn conditional_single_side_join_size_in_context(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
        semi_join: bool,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<usize> {
        let operation = "dataframe::with_execution::conditional_single_side_join_size";
        let (left_table, right_table) = predicate_tables(self, right, &options, operation)?;
        if semi_join {
            join::conditional_left_semi_join_size_in(left_table.table(), right_table.table(), predicate, ctx)
        } else {
            join::conditional_left_anti_join_size_in(left_table.table(), right_table.table(), predicate, ctx)
        }
    }
}
