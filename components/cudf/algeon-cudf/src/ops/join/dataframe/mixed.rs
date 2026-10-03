use super::*;

impl DataFrame {
    pub(crate) fn mixed_join_indices_in_context(
        &self,
        right: &DataFrame,
        predicate: &AstExpression,
        options: DataFrameMixedJoinOptions,
        join_type: join::JoinType,
        ctx: Exec<'_>,
    ) -> Result<DataFrameJoinPairIndices> {
        let operation = "dataframe::mixed_join_indices";
        let tables = mixed_join_tables(self, right, &options, operation)?;
        let raw = match join_type {
            join::JoinType::Inner => join::mixed_inner_join_indices_in(
                &tables.left_equality,
                &tables.right_equality,
                &tables.left_conditional,
                &tables.right_conditional,
                predicate,
                options.null_eq(),
                ctx,
            )?,
            join::JoinType::Left => join::mixed_left_join_indices_in(
                &tables.left_equality,
                &tables.right_equality,
                &tables.left_conditional,
                &tables.right_conditional,
                predicate,
                options.null_eq(),
                ctx,
            )?,
            join::JoinType::Full => join::mixed_full_join_indices_in(
                &tables.left_equality,
                &tables.right_equality,
                &tables.left_conditional,
                &tables.right_conditional,
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
        ctx: Exec<'_>,
    ) -> Result<DataFrameMixedJoinSizeData<'a>> {
        let operation = "dataframe::mixed_join_size_data";
        let tables = mixed_join_tables(self, right, &options, operation)?;
        let state = match join_type {
            join::SizedMixedJoinType::Inner => join::mixed_inner_join_begin_size_data_in(
                &tables.left_equality,
                &tables.right_equality,
                &tables.left_conditional,
                &tables.right_conditional,
                predicate,
                options.null_eq(),
                ctx,
            )?,
            join::SizedMixedJoinType::Left => join::mixed_left_join_begin_size_data_in(
                &tables.left_equality,
                &tables.right_equality,
                &tables.left_conditional,
                &tables.right_conditional,
                predicate,
                options.null_eq(),
                ctx,
            )?,
            join::SizedMixedJoinType::Full => join::mixed_full_join_begin_size_data_in(
                &tables.left_equality,
                &tables.right_equality,
                &tables.left_conditional,
                &tables.right_conditional,
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
        ctx: Exec<'_>,
    ) -> Result<DataFrameMixedSingleSideJoinSizeData<'a>> {
        let operation = "dataframe::mixed_single_side_join_size_data";
        let tables = mixed_join_tables(self, right, &options, operation)?;
        let state = if semi_join {
            join::mixed_left_semi_join_begin_size_data_in(
                &tables.left_equality,
                &tables.right_equality,
                &tables.left_conditional,
                &tables.right_conditional,
                predicate,
                options.null_eq(),
                ctx,
            )?
        } else {
            join::mixed_left_anti_join_begin_size_data_in(
                &tables.left_equality,
                &tables.right_equality,
                &tables.left_conditional,
                &tables.right_conditional,
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
        ctx: Exec<'_>,
    ) -> Result<DataFrameJoinIndices> {
        let operation = "dataframe::mixed_single_side_join_indices";
        let tables = mixed_join_tables(self, right, &options, operation)?;
        let raw = if semi_join {
            join::mixed_left_semi_join_indices_in(
                &tables.left_equality,
                &tables.right_equality,
                &tables.left_conditional,
                &tables.right_conditional,
                predicate,
                options.null_eq(),
                ctx,
            )?
        } else {
            join::mixed_left_anti_join_indices_in(
                &tables.left_equality,
                &tables.right_equality,
                &tables.left_conditional,
                &tables.right_conditional,
                predicate,
                options.null_eq(),
                ctx,
            )?
        };
        single_side_indices_from_raw(raw, operation)
    }
}
