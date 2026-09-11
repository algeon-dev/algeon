use super::{
    BoundedCompiledPreparedJoinProbeState, BoundedExecutionAttempt, CompiledJoinExecutionStrategy, JoinNode, NodeId,
    ResidentGpuFrame, Result, execute_compiled_keyless_join_materialize,
    execute_compiled_keyless_outer_cross_join_materialize, execute_compiled_mixed_join_materialize,
    execute_compiled_one_shot_equi_join_materialize, execute_compiled_partitioned_hash_equi_join_materialize,
    execute_compiled_partitioned_mixed_keyed_residual_join_materialize,
    execute_compiled_projected_cross_join_materialize, unsupported_bounded_shape,
};

pub(super) fn execute_compiled_prepared_join_materialize(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    state: BoundedCompiledPreparedJoinProbeState,
) -> Result<ResidentGpuFrame> {
    match state.strategy {
        CompiledJoinExecutionStrategy::KeylessConditional { join_type } => {
            execute_compiled_keyless_join_materialize(execution, node_id, node, join_type, state.left, state.right)
        },
        CompiledJoinExecutionStrategy::KeylessOuterCross => {
            execute_compiled_keyless_outer_cross_join_materialize(execution, node_id, node, state.left, state.right)
        },
        CompiledJoinExecutionStrategy::ProjectedCross => {
            execute_compiled_projected_cross_join_materialize(execution, node_id, node, state.left, state.right)
        },
        CompiledJoinExecutionStrategy::MixedKeyedResidual => {
            execute_compiled_mixed_join_materialize(execution, node_id, node, state.left, state.right)
        },
        CompiledJoinExecutionStrategy::OneShotEqui => {
            execute_compiled_one_shot_equi_join_materialize(execution, node_id, node, state.left, state.right)
        },
        CompiledJoinExecutionStrategy::PartitionedHashEqui => execute_compiled_partitioned_hash_equi_join_materialize(
            execution,
            node_id,
            node,
            state.left,
            state.right,
            state.strategy_options,
        ),
        CompiledJoinExecutionStrategy::PartitionedMixedKeyedResidual => {
            execute_compiled_partitioned_mixed_keyed_residual_join_materialize(
                execution,
                node_id,
                node,
                state.left,
                state.right,
                state.strategy_options,
            )
        },
        CompiledJoinExecutionStrategy::HashEqui | CompiledJoinExecutionStrategy::FilteredSemiAnti => {
            Err(unsupported_bounded_shape(
                "compiled prepared join state received an incompatible strategy",
                "compiled_join_strategy_state_mismatch",
            ))
        },
    }
}
