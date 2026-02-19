use super::*;

pub(super) fn execute_compiled_prepared_join_materialize(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    state: BoundedCompiledPreparedJoinProbeState,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    match state.strategy {
        CompiledJoinExecutionStrategy::KeylessConditional { join_type } => execute_compiled_keyless_join_materialize(
            context,
            node_id,
            node,
            join_type,
            state.left,
            state.right,
            reservations,
            metrics,
        ),
        CompiledJoinExecutionStrategy::KeylessOuterCross => execute_compiled_keyless_outer_cross_join_materialize(
            context,
            node_id,
            node,
            state.left,
            state.right,
            reservations,
            metrics,
        ),
        CompiledJoinExecutionStrategy::ProjectedCross => execute_compiled_projected_cross_join_materialize(
            context,
            node_id,
            node,
            state.left,
            state.right,
            reservations,
            metrics,
        ),
        CompiledJoinExecutionStrategy::MixedKeyedResidual => execute_compiled_mixed_join_materialize(
            context,
            node_id,
            node,
            state.left,
            state.right,
            reservations,
            metrics,
        ),
        CompiledJoinExecutionStrategy::OneShotEqui => execute_compiled_one_shot_equi_join_materialize(
            context,
            node_id,
            node,
            state.left,
            state.right,
            reservations,
            metrics,
        ),
        CompiledJoinExecutionStrategy::PartitionedHashEqui => execute_compiled_partitioned_hash_equi_join_materialize(
            context,
            node_id,
            node,
            state.left,
            state.right,
            reservations,
            metrics,
        ),
        CompiledJoinExecutionStrategy::PartitionedMixedKeyedResidual => {
            execute_compiled_partitioned_mixed_keyed_residual_join_materialize(
                context,
                node_id,
                node,
                state.left,
                state.right,
                reservations,
                metrics,
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
