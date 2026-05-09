use super::*;

pub(super) fn execute_mixed_join_indices(
    input: &JoinIndexInput<'_, '_>,
    node: &JoinNode,
    _scratch: &mut ExecutionScratch,
    metrics: &mut QueryMetrics,
) -> Result<JoinIndexResult> {
    match node.spec.join_type {
        JoinType::Inner | JoinType::Left => {
            let predicate = lower_join_residual_predicate(input.context, node, input.residual_columns)?;
            let options = dataframe_mixed_join_options(
                input.left_keys,
                input.right_keys,
                input.residual_columns,
                node.spec.null_equality,
            );
            let size_data = match node.spec.join_type {
                JoinType::Inner => input.left.with_execution(input.overrides).mixed_inner_join_size_data(
                    input.right,
                    &predicate,
                    options,
                ),
                JoinType::Left => input.left.with_execution(input.overrides).mixed_left_join_size_data(
                    input.right,
                    &predicate,
                    options,
                ),
                JoinType::Full | JoinType::LeftSemi | JoinType::LeftAnti | JoinType::Cross => {
                    unreachable!("matched above")
                },
            }
            .map_err(|error| join_index_error(error, node))?;
            let size_data_bytes =
                size_data.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
                    QueryError::from_cudf(
                        "cudf_query_engine.join.mixed_size_data_bytes",
                        "failed to observe native mixed join size-data memory",
                        error,
                    )
                    .with_fact("join_type", node.spec.join_type)
                })?;
            metrics.record_mixed_join_size_data_bytes(size_data_bytes);
            let pairs =
                size_data.materialize_indices_in(input.overrides).map_err(|error| join_index_error(error, node))?;
            Ok(JoinIndexResult::without_hash_table(EquiJoinIndices::Pair {
                left: pairs.left().clone(),
                right: pairs.right().clone(),
            })
            .with_mixed_size_data(size_data_bytes))
        },
        JoinType::Full => {
            let predicate = lower_join_match_predicate(input.context, node, input.residual_columns)?;
            let pairs = input
                .left
                .with_execution(input.overrides)
                .conditional_full_join_indices(
                    input.right,
                    &predicate,
                    dataframe_predicate_join_options(input.residual_columns),
                )
                .map_err(|error| join_index_error(error, node))?;
            Ok(JoinIndexResult::without_hash_table(EquiJoinIndices::Pair {
                left: pairs.left().clone(),
                right: pairs.right().clone(),
            }))
        },
        JoinType::LeftSemi => {
            let predicate = lower_join_residual_predicate(input.context, node, input.residual_columns)?;
            let options = dataframe_mixed_join_options(
                input.left_keys,
                input.right_keys,
                input.residual_columns,
                node.spec.null_equality,
            );
            input
                .left
                .with_execution(input.overrides)
                .mixed_left_semi_join_indices(input.right, &predicate, options)
                .map(EquiJoinIndices::LeftOnly)
                .map(JoinIndexResult::without_hash_table)
                .map_err(|error| join_index_error(error, node))
        },
        JoinType::LeftAnti => {
            let predicate = lower_join_residual_predicate(input.context, node, input.residual_columns)?;
            let options = dataframe_mixed_join_options(
                input.left_keys,
                input.right_keys,
                input.residual_columns,
                node.spec.null_equality,
            );
            input
                .left
                .with_execution(input.overrides)
                .mixed_left_anti_join_indices(input.right, &predicate, options)
                .map(EquiJoinIndices::LeftOnly)
                .map(JoinIndexResult::without_hash_table)
                .map_err(|error| join_index_error(error, node))
        },
        JoinType::Cross => Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "cudf_query_engine.join.execute",
            "cross joins are not mixed join-index operations",
        )
        .with_fact("join_type", node.spec.join_type)),
    }
}
