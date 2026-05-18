use super::*;

pub(super) fn execute_compiled_projected_cross_join_materialize(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    execute_compiled_projected_cross_join_materialize_with_strategy(
        context,
        node_id,
        node,
        left,
        right,
        "bounded_projected_cross_join",
        reservations,
        metrics,
    )
}

fn execute_compiled_projected_cross_join_materialize_with_strategy(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    execution_strategy: &'static str,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let options = dataframe_cross_join_options(&node.spec.output_projection);
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let plan = left.frame.projected_cross_join_plan(&right.frame, options).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.bounded.join.projected_cross_plan",
            "failed to create compiled bounded projected cross join plan",
            error,
        )
        .with_fact("join_type", node.spec.join_type)
    })?;
    let expected_output_rows = plan.output_rows() as u64;
    let output_reservation_bytes =
        plan.reservation_size_bytes_in(&overrides).map(|bytes| bytes as u64).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.bounded.join.projected_cross_output_reservation",
                "failed to size compiled bounded projected cross join output",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
        })?;
    let output_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::JoinOutput,
            "join_output",
            output_reservation_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;
    let output = plan.materialize_in(&overrides).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.bounded.join.projected_cross_materialize",
            "failed to materialize compiled bounded projected cross join output",
            error,
        )
        .with_fact("join_type", node.spec.join_type)
    })?;
    let output = reorder_projected_cross_join_output(output, &node.spec.output_projection)?;
    metrics.operator_time += timer.elapsed();
    let output_rows = output.num_rows() as u64;
    debug_assert_eq!(output_rows, expected_output_rows);
    let output_bytes = observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), metrics)?;
    let build_side = node.spec.build_side;
    let probe_side = probe_side(build_side);
    metrics.record_operator_metrics(crate::metrics::OperatorMetrics::join(
        node_id,
        crate::metrics::JoinOperatorMetrics {
            join_type: node.spec.join_type.to_string(),
            execution_strategy: execution_strategy.to_string(),
            key_count: 0,
            residual_predicate: false,
            null_equality: node.spec.null_equality.to_string(),
            build_side: build_side.to_string(),
            probe_side: probe_side.to_string(),
            build_rows: side_metric(build_side, left.rows, right.rows),
            build_bytes: side_metric(build_side, left.bytes, right.bytes),
            probe_rows: side_metric(probe_side, left.rows, right.rows),
            probe_bytes: side_metric(probe_side, left.bytes, right.bytes),
            hash_table_bytes: 0,
            filtered_probe_state_bytes: 0,
            mixed_size_data_bytes: 0,
            gather_map_bytes: 0,
            output_reservation_bytes,
            output_rows,
            output_bytes,
        },
    ));
    drop(left);
    drop(right);
    Ok(ResidentGpuFrame { frame: GpuDataFrame::new(output), reservation: output_reservation })
}

pub(super) fn execute_compiled_keyless_outer_cross_join_materialize(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    match node.spec.join_type {
        JoinType::Left if left.rows == 0 || right.rows > 0 => {
            execute_compiled_projected_cross_join_materialize_with_strategy(
                context,
                node_id,
                node,
                left,
                right,
                "bounded_keyless_outer_cross_join",
                reservations,
                metrics,
            )
        },
        JoinType::Full if left.rows == 0 && right.rows == 0 => {
            execute_compiled_projected_cross_join_materialize_with_strategy(
                context,
                node_id,
                node,
                left,
                right,
                "bounded_keyless_outer_cross_join",
                reservations,
                metrics,
            )
        },
        JoinType::Full if left.rows > 0 && right.rows > 0 => {
            execute_compiled_projected_cross_join_materialize_with_strategy(
                context,
                node_id,
                node,
                left,
                right,
                "bounded_keyless_outer_cross_join",
                reservations,
                metrics,
            )
        },
        JoinType::Left => execute_compiled_keyless_outer_cross_join_from_indices(
            context,
            node_id,
            node,
            left,
            right,
            PreservedKeylessOuterSide::Left,
            reservations,
            metrics,
        ),
        JoinType::Full if left.rows > 0 => execute_compiled_keyless_outer_cross_join_from_indices(
            context,
            node_id,
            node,
            left,
            right,
            PreservedKeylessOuterSide::Left,
            reservations,
            metrics,
        ),
        JoinType::Full => execute_compiled_keyless_outer_cross_join_from_indices(
            context,
            node_id,
            node,
            left,
            right,
            PreservedKeylessOuterSide::Right,
            reservations,
            metrics,
        ),
        _ => Err(unsupported_bounded_shape(
            "compiled keyless outer cross join execution supports predicate-free left and full joins",
            "keyless_outer_cross_join_requires_left_or_full",
        )
        .with_fact("join_type", node.spec.join_type)
        .with_fact("key_count", node.spec.keys.len())
        .with_fact("residual_predicate", node.spec.residual_predicate.is_some())),
    }
}

#[derive(Debug, Clone, Copy)]
enum PreservedKeylessOuterSide {
    Left,
    Right,
}

fn execute_compiled_keyless_outer_cross_join_from_indices(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    preserved_side: PreservedKeylessOuterSide,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let output_size = match preserved_side {
        PreservedKeylessOuterSide::Left => usize::try_from(left.rows),
        PreservedKeylessOuterSide::Right => usize::try_from(right.rows),
    }
    .map_err(|_| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.bounded.join.keyless_outer_cross",
            "keyless outer cross join output row count exceeds usize",
        )
        .with_fact("join_type", node.spec.join_type)
    })?;
    if output_size > i32::MAX as usize {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.bounded.join.keyless_outer_cross",
            "keyless outer cross join gather-map row count exceeds INT32_MAX",
        )
        .with_fact("join_type", node.spec.join_type)
        .with_fact("output_rows", output_size));
    }

    let gather_map_reservation_bytes = cudf::DataFrameJoinPairIndices::reservation_size_bytes_for_len(output_size)
        .map(|bytes| bytes as u64)
        .map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.bounded.join.keyless_outer_cross_gather_map_reservation",
                "failed to size keyless outer cross join gather maps",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
        })?;
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
    let timer = Instant::now();
    let indices = keyless_outer_cross_join_indices(context, preserved_side, output_size, node)?;
    materialize_compiled_indices_output(
        context,
        node_id,
        node,
        "bounded_keyless_outer_cross_join",
        0,
        false,
        node.spec.build_side,
        probe_side(node.spec.build_side),
        left,
        right,
        indices,
        0,
        0,
        output_size as u64,
        gather_map_reservation,
        timer,
        None,
        reservations,
        metrics,
    )
}

fn keyless_outer_cross_join_indices(
    context: &ExecutionContext,
    preserved_side: PreservedKeylessOuterSide,
    output_size: usize,
    node: &JoinNode,
) -> Result<CompiledJoinIndices> {
    let sequence = || {
        context.runtime().sequence(0i32, 1i32, output_size).map(cudf::DataFrameJoinIndices::from_series).map_err(
            |error| {
                QueryError::from_cudf(
                    "cudf_query_engine.bounded.join.keyless_outer_cross_sequence",
                    "failed to create preserved-side keyless outer cross gather map",
                    error,
                )
                .with_fact("join_type", node.spec.join_type)
            },
        )
    };
    let null_supplier = || {
        let sentinel = context.runtime().scalar(-1i32).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.bounded.join.keyless_outer_cross_null_supplier_scalar",
                "failed to create keyless outer cross null-supplier gather sentinel",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
        })?;
        context
            .runtime()
            .series_from_scalar(&sentinel, output_size)
            .map(cudf::DataFrameJoinIndices::from_series)
            .map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.bounded.join.keyless_outer_cross_null_supplier",
                    "failed to create null-supplying keyless outer cross gather map",
                    error,
                )
                .with_fact("join_type", node.spec.join_type)
            })
    };
    match preserved_side {
        PreservedKeylessOuterSide::Left => Ok(CompiledJoinIndices::Pair { left: sequence()?, right: null_supplier()? }),
        PreservedKeylessOuterSide::Right => {
            Ok(CompiledJoinIndices::Pair { left: null_supplier()?, right: sequence()? })
        },
    }
}
