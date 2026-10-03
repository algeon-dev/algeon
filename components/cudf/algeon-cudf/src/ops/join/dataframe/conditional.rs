use super::*;

impl DataFrame {
    pub(crate) fn conditional_join_indices_in_context(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
        join_type: join::JoinType,
        ctx: Exec<'_>,
    ) -> Result<DataFrameJoinPairIndices> {
        let operation = "dataframe::conditional_join_indices";
        let (left_table, right_table) = predicate_tables(self, right, &options, operation)?;
        let raw = match join_type {
            join::JoinType::Inner => {
                join::conditional_inner_join_indices_in(&left_table, &right_table, predicate, ctx)?
            },
            join::JoinType::Left => join::conditional_left_join_indices_in(&left_table, &right_table, predicate, ctx)?,
            join::JoinType::Full => join::conditional_full_join_indices_in(&left_table, &right_table, predicate, ctx)?,
        };
        pair_indices_from_raw(raw, operation)
    }

    pub(crate) fn conditional_full_join_size_data_in_context<'a>(
        &self,
        right: &DataFrame,
        predicate: &'a AstExpression,
        options: DataFramePredicateJoinOptions,
        ctx: Exec<'_>,
    ) -> Result<DataFrameConditionalFullJoinSizeData<'a>> {
        let operation = "dataframe::conditional_full_join_size_data";
        let (left_table, right_table) = predicate_tables(self, right, &options, operation)?;
        let state = join::conditional_full_join_begin_size_data_in(&left_table, &right_table, predicate, ctx)?;
        Ok(DataFrameConditionalFullJoinSizeData { state, _left: self.clone(), _right: right.clone() })
    }

    pub(crate) fn conditional_join_indices_with_output_size_in_context(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
        join_type: join::SizedConditionalPairJoinType,
        output_size: usize,
        ctx: Exec<'_>,
    ) -> Result<DataFrameJoinPairIndices> {
        let operation = "dataframe::conditional_join_indices_with_output_size";
        let (left_table, right_table) = predicate_tables(self, right, &options, operation)?;
        let raw = match join_type {
            join::SizedConditionalPairJoinType::Inner => join::conditional_inner_join_indices_with_output_size_in(
                &left_table,
                &right_table,
                predicate,
                output_size,
                ctx,
            )?,
            join::SizedConditionalPairJoinType::Left => join::conditional_left_join_indices_with_output_size_in(
                &left_table,
                &right_table,
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
        ctx: Exec<'_>,
    ) -> Result<usize> {
        let operation = "dataframe::conditional_join_size";
        let (left_table, right_table) = predicate_tables(self, right, &options, operation)?;
        match join_type {
            join::SizedConditionalPairJoinType::Inner => {
                join::conditional_inner_join_size_in(&left_table, &right_table, predicate, ctx)
            },
            join::SizedConditionalPairJoinType::Left => {
                join::conditional_left_join_size_in(&left_table, &right_table, predicate, ctx)
            },
        }
    }

    pub(crate) fn conditional_single_side_join_indices_in_context(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFramePredicateJoinOptions,
        semi_join: bool,
        ctx: Exec<'_>,
    ) -> Result<DataFrameJoinIndices> {
        let operation = "dataframe::conditional_single_side_join_indices";
        let (left_table, right_table) = predicate_tables(self, right, &options, operation)?;
        let raw = if semi_join {
            join::conditional_left_semi_join_indices_in(&left_table, &right_table, predicate, ctx)?
        } else {
            join::conditional_left_anti_join_indices_in(&left_table, &right_table, predicate, ctx)?
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
        ctx: Exec<'_>,
    ) -> Result<DataFrameJoinIndices> {
        let operation = "dataframe::conditional_single_side_join_indices_with_output_size";
        let (left_table, right_table) = predicate_tables(self, right, &options, operation)?;
        let raw = if semi_join {
            join::conditional_left_semi_join_indices_with_output_size_in(
                &left_table,
                &right_table,
                predicate,
                output_size,
                ctx,
            )?
        } else {
            join::conditional_left_anti_join_indices_with_output_size_in(
                &left_table,
                &right_table,
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
        ctx: Exec<'_>,
    ) -> Result<usize> {
        let operation = "dataframe::conditional_single_side_join_size";
        let (left_table, right_table) = predicate_tables(self, right, &options, operation)?;
        if semi_join {
            join::conditional_left_semi_join_size_in(&left_table, &right_table, predicate, ctx)
        } else {
            join::conditional_left_anti_join_size_in(&left_table, &right_table, predicate, ctx)
        }
    }
}
