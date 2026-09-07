use std::time::Instant;

use super::{
    BoundedExecutionAttempt, BoundedJoinSideFrame, CompiledJoinIndices, CompiledJoinMaterializationRequest,
    CompiledKeylessWork, DataFrame, ExecutionOverrides, JoinNode, JoinSide, KeylessConditionalJoinKind, NodeId,
    ResidentGpuFrame, Result, cudf_query_error, dataframe_predicate_join_options, execution_overrides,
    join_index_error, lower_join_residual_predicate, materialize_compiled_indices_output, residual_input_columns,
};

pub(super) fn execute_compiled_keyless_join_materialize(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    join_type: KeylessConditionalJoinKind,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
) -> Result<ResidentGpuFrame> {
    let residual_columns = residual_input_columns(node)?;
    let overrides_lease = execution_overrides(execution.context())?;
    let overrides = overrides_lease.overrides();
    let predicate = lower_join_residual_predicate(execution.context(), node, &residual_columns)?;
    let options = dataframe_predicate_join_options(&residual_columns);
    let timer = Instant::now();
    let (size_data_bytes, expected_output_rows, work) = if join_type == KeylessConditionalJoinKind::Full {
        let size_data = left
            .frame
            .with_execution(&overrides)
            .conditional_full_join_size_data(&right.frame, &predicate, options)
            .map_err(|error| join_index_error(error, node))?;
        let expected_output_rows = size_data.output_size() as u64;
        let size_data_bytes = size_data.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
            cudf_query_error(
                "nexus_query_engine.bounded.join.conditional_full_size_data_bytes",
                "failed to observe compiled conditional full join size-data memory",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
        })?;
        execution.metrics_mut().record_mixed_join_size_data_bytes(size_data_bytes);
        (size_data_bytes, expected_output_rows, CompiledKeylessWork::Full { size_data: Box::new(size_data) })
    } else {
        let output_size = keyless_conditional_join_output_size(
            join_type,
            &left.frame,
            &right.frame,
            &predicate,
            options.clone(),
            &overrides,
            node,
        )?;
        (0, output_size as u64, CompiledKeylessWork::Sized { output_size, options })
    };
    let indices = match work {
        CompiledKeylessWork::Full { size_data } => {
            let indices = (*size_data).materialize_indices_in(&overrides);
            let indices = indices.map_err(|error| join_index_error(error, node))?;
            CompiledJoinIndices::Pair { left: indices.left().clone(), right: indices.right().clone() }
        },
        CompiledKeylessWork::Sized { output_size, options } => keyless_conditional_join_indices_with_output_size(
            join_type,
            &left.frame,
            &right.frame,
            &predicate,
            options,
            output_size,
            &overrides,
            node,
        )?,
    };
    let gather_map_bytes = indices.device_allocated_size_bytes(node)?;
    materialize_compiled_indices_output(
        execution,
        node,
        CompiledJoinMaterializationRequest {
            node_id,
            execution_strategy: "bounded_keyless_conditional_join",
            key_count: 0,
            residual_predicate: true,
            build_side: JoinSide::Left,
            probe_side: JoinSide::Right,
            left,
            right,
            indices,
            mixed_size_data_bytes: size_data_bytes,
            filtered_probe_state_bytes: 0,
            expected_output_rows,
            gather_map_bytes,
            timer,
            one_shot_index_bytes: None,
        },
    )
}

pub(super) fn keyless_conditional_join_output_size(
    join_type: KeylessConditionalJoinKind,
    left: &DataFrame,
    right: &DataFrame,
    predicate: &cudf::ops::ast::AstExpression,
    options: cudf::DataFramePredicateJoinOptions,
    overrides: &ExecutionOverrides<'_>,
    node: &JoinNode,
) -> Result<usize> {
    let result = match join_type {
        KeylessConditionalJoinKind::Inner => {
            left.with_execution(overrides).conditional_inner_join_size(right, predicate, options)
        },
        KeylessConditionalJoinKind::Left => {
            left.with_execution(overrides).conditional_left_join_size(right, predicate, options)
        },
        KeylessConditionalJoinKind::Full => unreachable!("conditional full joins use retained size data"),
        KeylessConditionalJoinKind::LeftSemi => {
            left.with_execution(overrides).conditional_left_semi_join_size(right, predicate, options)
        },
        KeylessConditionalJoinKind::LeftAnti => {
            left.with_execution(overrides).conditional_left_anti_join_size(right, predicate, options)
        },
    };
    result.map_err(|error| join_index_error(error, node))
}

#[expect(
    clippy::too_many_arguments,
    reason = "bounded execution passes operator-occurrence inputs explicitly beside the attempt state"
)]
pub(super) fn keyless_conditional_join_indices_with_output_size(
    join_type: KeylessConditionalJoinKind,
    left: &DataFrame,
    right: &DataFrame,
    predicate: &cudf::ops::ast::AstExpression,
    options: cudf::DataFramePredicateJoinOptions,
    output_size: usize,
    overrides: &ExecutionOverrides<'_>,
    node: &JoinNode,
) -> Result<CompiledJoinIndices> {
    match join_type {
        KeylessConditionalJoinKind::Inner => {
            let indices = left
                .with_execution(overrides)
                .conditional_inner_join_indices_with_output_size(right, predicate, options, output_size)
                .map_err(|error| join_index_error(error, node))?;
            Ok(CompiledJoinIndices::Pair { left: indices.left().clone(), right: indices.right().clone() })
        },
        KeylessConditionalJoinKind::Left => {
            let indices = left
                .with_execution(overrides)
                .conditional_left_join_indices_with_output_size(right, predicate, options, output_size)
                .map_err(|error| join_index_error(error, node))?;
            Ok(CompiledJoinIndices::Pair { left: indices.left().clone(), right: indices.right().clone() })
        },
        KeylessConditionalJoinKind::Full => unreachable!("conditional full joins use retained size data"),
        KeylessConditionalJoinKind::LeftSemi => left
            .with_execution(overrides)
            .conditional_left_semi_join_indices_with_output_size(right, predicate, options, output_size)
            .map(CompiledJoinIndices::LeftOnly)
            .map_err(|error| join_index_error(error, node)),
        KeylessConditionalJoinKind::LeftAnti => left
            .with_execution(overrides)
            .conditional_left_anti_join_indices_with_output_size(right, predicate, options, output_size)
            .map(CompiledJoinIndices::LeftOnly)
            .map_err(|error| join_index_error(error, node)),
    }
}
