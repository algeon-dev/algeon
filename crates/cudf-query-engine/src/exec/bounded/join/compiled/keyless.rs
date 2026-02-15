use super::*;

pub(super) fn execute_compiled_keyless_join_materialize(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    join_type: KeylessConditionalJoinKind,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let residual_columns = residual_input_columns(node)?;
    let overrides = execution_overrides(context);
    let predicate = lower_join_residual_predicate(context, node, &residual_columns)?;
    let options = dataframe_predicate_join_options(&residual_columns);
    let timer = Instant::now();
    let (size_data_bytes, expected_output_rows, gather_map_reservation_bytes, work) = if join_type
        == KeylessConditionalJoinKind::Full
    {
        let size_data = left
            .frame
            .with_execution(&overrides)
            .conditional_full_join_size_data(&right.frame, &predicate, options)
            .map_err(|error| join_index_error(error, node))?;
        let expected_output_rows = size_data.output_size() as u64;
        let size_data_bytes = size_data.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.bounded.join.conditional_full_size_data_bytes",
                "failed to observe compiled conditional full join size-data memory",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
        })?;
        let size_data_reservation = reservations.try_reserve(
            bounded_join_request(
                ReservationKind::MixedSizeData,
                "join_mixed_size_data",
                size_data_bytes,
                reservations.budget(),
                node_id,
            ),
            metrics,
        )?;
        metrics.record_mixed_join_size_data_bytes(size_data_bytes);
        let gather_map_reservation_bytes =
            size_data.gather_map_reservation_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.bounded.join.conditional_full_gather_map_reservation",
                    "failed to size compiled conditional full join gather maps",
                    error,
                )
                .with_fact("join_type", node.spec.join_type)
            })?;
        (
            size_data_bytes,
            expected_output_rows,
            gather_map_reservation_bytes,
            CompiledKeylessWork::Full {
                size_data: Box::new(size_data),
                size_data_reservation: Box::new(size_data_reservation),
            },
        )
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
        let gather_map_reservation_bytes =
            keyless_conditional_join_gather_map_reservation_size_bytes(join_type, node, output_size)?;
        (0, output_size as u64, gather_map_reservation_bytes, CompiledKeylessWork::Sized { output_size, options })
    };
    let gather_map_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::GatherMap,
            "join_gather_map",
            gather_map_reservation_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;
    let indices = match work {
        CompiledKeylessWork::Full { size_data, size_data_reservation } => {
            let indices = (*size_data).materialize_indices_in(&overrides);
            reservations.release(*size_data_reservation, metrics);
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
    materialize_compiled_indices_output(
        context,
        node_id,
        node,
        "bounded_keyless_conditional_join",
        0,
        true,
        JoinSide::Left,
        JoinSide::Right,
        left,
        right,
        indices,
        size_data_bytes,
        0,
        expected_output_rows,
        gather_map_reservation,
        timer,
        None,
        reservations,
        metrics,
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

pub(super) fn keyless_conditional_join_gather_map_reservation_size_bytes(
    join_type: KeylessConditionalJoinKind,
    node: &JoinNode,
    output_size: usize,
) -> Result<u64> {
    let result = match join_type {
        KeylessConditionalJoinKind::Inner | KeylessConditionalJoinKind::Left => {
            cudf::DataFrameJoinPairIndices::reservation_size_bytes_for_len(output_size)
        },
        KeylessConditionalJoinKind::Full => unreachable!("conditional full joins use retained size data"),
        KeylessConditionalJoinKind::LeftSemi | KeylessConditionalJoinKind::LeftAnti => {
            cudf::DataFrameJoinIndices::reservation_size_bytes_for_len(output_size)
        },
    };
    result.map(|bytes| bytes as u64).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.bounded.join.gather_map_reservation",
            "failed to size compiled conditional join gather maps",
            error,
        )
        .with_fact("join_type", node.spec.join_type)
    })
}

#[allow(clippy::too_many_arguments)]
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
