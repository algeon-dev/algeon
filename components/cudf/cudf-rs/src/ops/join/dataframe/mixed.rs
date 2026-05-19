use super::*;

impl DataFrame {
    /// Mixed equality-plus-predicate inner join indices.
    pub fn mixed_inner_join_indices(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFrameMixedJoinOptions,
    ) -> Result<DataFrameJoinPairIndices> {
        let ctx = ExecutionOverrides::new();
        self.mixed_join_indices_in_context(right, predicate, options, join::JoinType::Inner, &ctx)
    }

    /// Mixed equality-plus-predicate left join indices.
    pub fn mixed_left_join_indices(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFrameMixedJoinOptions,
    ) -> Result<DataFrameJoinPairIndices> {
        let ctx = ExecutionOverrides::new();
        self.mixed_join_indices_in_context(right, predicate, options, join::JoinType::Left, &ctx)
    }

    /// Begin retained size-data computation for a mixed inner join.
    pub fn mixed_inner_join_size_data<'a>(
        &self,
        right: &DataFrame,
        predicate: &'a AstExpression,
        options: DataFrameMixedJoinOptions,
    ) -> Result<DataFrameMixedJoinSizeData<'a>> {
        let ctx = ExecutionOverrides::new();
        self.mixed_join_size_data_in_context(right, predicate, options, join::SizedMixedJoinType::Inner, &ctx)
    }

    /// Begin retained size-data computation for a mixed left join.
    pub fn mixed_left_join_size_data<'a>(
        &self,
        right: &DataFrame,
        predicate: &'a AstExpression,
        options: DataFrameMixedJoinOptions,
    ) -> Result<DataFrameMixedJoinSizeData<'a>> {
        let ctx = ExecutionOverrides::new();
        self.mixed_join_size_data_in_context(right, predicate, options, join::SizedMixedJoinType::Left, &ctx)
    }

    /// Begin retained size-data computation for a mixed full join.
    pub fn mixed_full_join_size_data<'a>(
        &self,
        right: &DataFrame,
        predicate: &'a AstExpression,
        options: DataFrameMixedJoinOptions,
    ) -> Result<DataFrameMixedJoinSizeData<'a>> {
        let ctx = ExecutionOverrides::new();
        self.mixed_join_size_data_in_context(right, predicate, options, join::SizedMixedJoinType::Full, &ctx)
    }

    /// Mixed equality-plus-predicate full join indices.
    pub fn mixed_full_join_indices(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFrameMixedJoinOptions,
    ) -> Result<DataFrameJoinPairIndices> {
        let ctx = ExecutionOverrides::new();
        self.mixed_join_indices_in_context(right, predicate, options, join::JoinType::Full, &ctx)
    }

    /// Mixed equality-plus-predicate left semi join indices.
    pub fn mixed_left_semi_join_indices(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFrameMixedJoinOptions,
    ) -> Result<DataFrameJoinIndices> {
        let ctx = ExecutionOverrides::new();
        self.mixed_single_side_join_indices_in_context(right, predicate, options, true, &ctx)
    }

    /// Begin retained size-data computation for a mixed left semi join.
    pub fn mixed_left_semi_join_size_data<'a>(
        &self,
        right: &DataFrame,
        predicate: &'a AstExpression,
        options: DataFrameMixedJoinOptions,
    ) -> Result<DataFrameMixedSingleSideJoinSizeData<'a>> {
        let ctx = ExecutionOverrides::new();
        self.mixed_single_side_join_size_data_in_context(right, predicate, options, true, &ctx)
    }

    /// Mixed equality-plus-predicate left anti join indices.
    pub fn mixed_left_anti_join_indices(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFrameMixedJoinOptions,
    ) -> Result<DataFrameJoinIndices> {
        let ctx = ExecutionOverrides::new();
        self.mixed_single_side_join_indices_in_context(right, predicate, options, false, &ctx)
    }

    /// Begin retained size-data computation for a mixed left anti join.
    pub fn mixed_left_anti_join_size_data<'a>(
        &self,
        right: &DataFrame,
        predicate: &'a AstExpression,
        options: DataFrameMixedJoinOptions,
    ) -> Result<DataFrameMixedSingleSideJoinSizeData<'a>> {
        let ctx = ExecutionOverrides::new();
        self.mixed_single_side_join_size_data_in_context(right, predicate, options, false, &ctx)
    }
    pub(crate) fn mixed_join_indices_in_context(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFrameMixedJoinOptions,
        join_type: join::JoinType,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<DataFrameJoinPairIndices> {
        let operation = "dataframe::with_execution::mixed_join_indices";
        let tables = mixed_join_tables(self, right, &options, operation)?;
        let raw = match join_type {
            join::JoinType::Inner => join::mixed_inner_join_indices_in(
                tables.left_equality.table(),
                tables.right_equality.table(),
                tables.left_conditional.table(),
                tables.right_conditional.table(),
                predicate,
                options.null_eq(),
                ctx,
            )?,
            join::JoinType::Left => join::mixed_left_join_indices_in(
                tables.left_equality.table(),
                tables.right_equality.table(),
                tables.left_conditional.table(),
                tables.right_conditional.table(),
                predicate,
                options.null_eq(),
                ctx,
            )?,
            join::JoinType::Full => join::mixed_full_join_indices_in(
                tables.left_equality.table(),
                tables.right_equality.table(),
                tables.left_conditional.table(),
                tables.right_conditional.table(),
                predicate,
                options.null_eq(),
                ctx,
            )?,
        };
        pair_indices_from_raw(raw, operation)
    }

    pub(crate) fn mixed_join_size_data_in_context<'a>(
        &self,
        right: &DataFrame,
        predicate: &'a AstExpression,
        options: DataFrameMixedJoinOptions,
        join_type: join::SizedMixedJoinType,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<DataFrameMixedJoinSizeData<'a>> {
        let operation = "dataframe::with_execution::mixed_join_size_data";
        let tables = mixed_join_tables(self, right, &options, operation)?;
        let state = match join_type {
            join::SizedMixedJoinType::Inner => join::mixed_inner_join_begin_size_data_in(
                tables.left_equality.table(),
                tables.right_equality.table(),
                tables.left_conditional.table(),
                tables.right_conditional.table(),
                predicate,
                options.null_eq(),
                ctx,
            )?,
            join::SizedMixedJoinType::Left => join::mixed_left_join_begin_size_data_in(
                tables.left_equality.table(),
                tables.right_equality.table(),
                tables.left_conditional.table(),
                tables.right_conditional.table(),
                predicate,
                options.null_eq(),
                ctx,
            )?,
            join::SizedMixedJoinType::Full => join::mixed_full_join_begin_size_data_in(
                tables.left_equality.table(),
                tables.right_equality.table(),
                tables.left_conditional.table(),
                tables.right_conditional.table(),
                predicate,
                options.null_eq(),
                ctx,
            )?,
        };
        Ok(DataFrameMixedJoinSizeData { state, _left: self.clone(), _right: right.clone() })
    }

    pub(crate) fn mixed_single_side_join_size_data_in_context<'a>(
        &self,
        right: &DataFrame,
        predicate: &'a AstExpression,
        options: DataFrameMixedJoinOptions,
        semi_join: bool,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<DataFrameMixedSingleSideJoinSizeData<'a>> {
        let operation = "dataframe::with_execution::mixed_single_side_join_size_data";
        let tables = mixed_join_tables(self, right, &options, operation)?;
        let state = if semi_join {
            join::mixed_left_semi_join_begin_size_data_in(
                tables.left_equality.table(),
                tables.right_equality.table(),
                tables.left_conditional.table(),
                tables.right_conditional.table(),
                predicate,
                options.null_eq(),
                ctx,
            )?
        } else {
            join::mixed_left_anti_join_begin_size_data_in(
                tables.left_equality.table(),
                tables.right_equality.table(),
                tables.left_conditional.table(),
                tables.right_conditional.table(),
                predicate,
                options.null_eq(),
                ctx,
            )?
        };
        Ok(DataFrameMixedSingleSideJoinSizeData { state, _left: self.clone(), _right: right.clone() })
    }

    pub(crate) fn mixed_single_side_join_indices_in_context(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFrameMixedJoinOptions,
        semi_join: bool,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<DataFrameJoinIndices> {
        let operation = "dataframe::with_execution::mixed_single_side_join_indices";
        let tables = mixed_join_tables(self, right, &options, operation)?;
        let raw = if semi_join {
            join::mixed_left_semi_join_indices_in(
                tables.left_equality.table(),
                tables.right_equality.table(),
                tables.left_conditional.table(),
                tables.right_conditional.table(),
                predicate,
                options.null_eq(),
                ctx,
            )?
        } else {
            join::mixed_left_anti_join_indices_in(
                tables.left_equality.table(),
                tables.right_equality.table(),
                tables.left_conditional.table(),
                tables.right_conditional.table(),
                predicate,
                options.null_eq(),
                ctx,
            )?
        };
        single_side_indices_from_raw(raw, operation)
    }
}
