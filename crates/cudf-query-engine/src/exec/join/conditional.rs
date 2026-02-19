use super::*;

pub(super) fn execute_conditional_join_indices(
    input: &JoinIndexInput<'_, '_>,
    node: &JoinNode,
) -> Result<EquiJoinIndices> {
    let predicate = lower_join_residual_predicate(input.context, node, input.residual_columns)?;
    match node.spec.join_type {
        JoinType::Inner => {
            let pairs = input
                .left
                .with_execution(input.overrides)
                .conditional_inner_join_indices(
                    input.right,
                    &predicate,
                    dataframe_predicate_join_options(input.residual_columns),
                )
                .map_err(|error| join_index_error(error, node))?;
            Ok(EquiJoinIndices::Pair { left: pairs.left().clone(), right: pairs.right().clone() })
        },
        JoinType::Left => {
            let pairs = input
                .left
                .with_execution(input.overrides)
                .conditional_left_join_indices(
                    input.right,
                    &predicate,
                    dataframe_predicate_join_options(input.residual_columns),
                )
                .map_err(|error| join_index_error(error, node))?;
            Ok(EquiJoinIndices::Pair { left: pairs.left().clone(), right: pairs.right().clone() })
        },
        JoinType::Full => {
            let pairs = input
                .left
                .with_execution(input.overrides)
                .conditional_full_join_indices(
                    input.right,
                    &predicate,
                    dataframe_predicate_join_options(input.residual_columns),
                )
                .map_err(|error| join_index_error(error, node))?;
            Ok(EquiJoinIndices::Pair { left: pairs.left().clone(), right: pairs.right().clone() })
        },
        JoinType::LeftSemi => input
            .left
            .with_execution(input.overrides)
            .conditional_left_semi_join_indices(
                input.right,
                &predicate,
                dataframe_predicate_join_options(input.residual_columns),
            )
            .map(EquiJoinIndices::LeftOnly)
            .map_err(|error| join_index_error(error, node)),
        JoinType::LeftAnti => input
            .left
            .with_execution(input.overrides)
            .conditional_left_anti_join_indices(
                input.right,
                &predicate,
                dataframe_predicate_join_options(input.residual_columns),
            )
            .map(EquiJoinIndices::LeftOnly)
            .map_err(|error| join_index_error(error, node)),
        JoinType::Cross | JoinType::LeftMark => Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "cudf_query_engine.join.execute",
            "cross and left mark joins are not conditional join-index operations",
        )
        .with_fact("join_type", node.spec.join_type)),
    }
}
