use super::{
    BoundedCompiledPreparedJoinProbeState, BoundedExecutionAttempt, JoinNode, JoinStrategy, NodeId, ResidentGpuFrame,
    Result, bounded_invariant_violation, execute_compiled_keyless_join_materialize,
    execute_compiled_keyless_outer_cross_join_materialize, execute_compiled_mixed_join_materialize,
    execute_compiled_one_shot_equi_join_materialize, execute_compiled_partitioned_hash_equi_join_materialize,
    execute_compiled_partitioned_mixed_keyed_residual_join_materialize,
    execute_compiled_projected_cross_join_materialize,
};

pub(super) fn execute_compiled_prepared_join_materialize(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    state: BoundedCompiledPreparedJoinProbeState,
) -> Result<ResidentGpuFrame> {
    match state.strategy {
        JoinStrategy::KeylessConditional { join_type } => {
            execute_compiled_keyless_join_materialize(execution, node_id, node, join_type, state.left, state.right)
        },
        JoinStrategy::KeylessOuterCross => {
            execute_compiled_keyless_outer_cross_join_materialize(execution, node_id, node, state.left, state.right)
        },
        JoinStrategy::ProjectedCross => {
            execute_compiled_projected_cross_join_materialize(execution, node_id, node, state.left, state.right)
        },
        JoinStrategy::MixedKeyedResidual => {
            execute_compiled_mixed_join_materialize(execution, node_id, node, state.left, state.right)
        },
        JoinStrategy::OneShotEqui => {
            execute_compiled_one_shot_equi_join_materialize(execution, node_id, node, state.left, state.right)
        },
        JoinStrategy::PartitionedHashEqui => execute_compiled_partitioned_hash_equi_join_materialize(
            execution,
            node_id,
            node,
            state.left,
            state.right,
            state.strategy_options,
        ),
        JoinStrategy::PartitionedMixedKeyedResidual => {
            execute_compiled_partitioned_mixed_keyed_residual_join_materialize(
                execution,
                node_id,
                node,
                state.left,
                state.right,
                state.strategy_options,
            )
        },
        JoinStrategy::ReusableHashEqui | JoinStrategy::FilteredSemiAnti => Err(bounded_invariant_violation(
            "compiled prepared join state received an incompatible strategy",
            "compiled_join_strategy_state_mismatch",
        )),
    }
}
