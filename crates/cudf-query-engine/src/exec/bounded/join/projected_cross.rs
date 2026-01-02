use super::{
    common::{bounded_join_request, prepare_bounded_join_side_input},
    *,
};

pub(super) fn execute_bounded_root_projected_cross_join(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: ResidentGpuFrame,
    right: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    validate_executable_join_subset(node)?;
    if !node.is_bounded_root_projected_cross_join_compatible() {
        return Err(unsupported_bounded_shape(
            "bounded root projected cross join execution requires a keyless cross join with an explicit output projection",
            "bounded_join_plan_shape_unsupported",
        )
        .with_fact("join_type", node.spec.join_type)
        .with_fact("key_count", node.spec.keys.len())
        .with_fact("residual_predicate", node.spec.residual_predicate.is_some())
        .with_fact("output_projection_len", node.spec.output_projection.len()));
    }

    let left_columns = projected_output_columns(&node.spec.output_projection, JoinSide::Left);
    let right_columns = projected_output_columns(&node.spec.output_projection, JoinSide::Right);
    let build_side = node.spec.build_side;
    let probe_side = probe_side(build_side);
    let (left, right) = match build_side {
        JoinSide::Left => {
            let left = prepare_bounded_join_side_input(
                left,
                &left_columns,
                JoinSide::Left,
                node_id,
                build_side,
                reservations,
                metrics,
            )?;
            let right = prepare_bounded_join_side_input(
                right,
                &right_columns,
                JoinSide::Right,
                node_id,
                build_side,
                reservations,
                metrics,
            )?;
            (left, right)
        },
        JoinSide::Right => {
            let right = prepare_bounded_join_side_input(
                right,
                &right_columns,
                JoinSide::Right,
                node_id,
                build_side,
                reservations,
                metrics,
            )?;
            let left = prepare_bounded_join_side_input(
                left,
                &left_columns,
                JoinSide::Left,
                node_id,
                build_side,
                reservations,
                metrics,
            )?;
            (left, right)
        },
    };
    let left_rows = left.rows;
    let right_rows = right.rows;
    let left_bytes = left.bytes;
    let right_bytes = right.bytes;

    let options = dataframe_cross_join_options(&node.spec.output_projection);
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let plan = left.frame.projected_cross_join_plan(&right.frame, options).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.bounded.join.projected_cross_plan",
            "failed to create bounded native projected cross join plan",
            error,
        )
        .with_fact("join_type", node.spec.join_type)
    })?;
    let expected_output_rows = plan.output_rows() as u64;
    let output_reservation_bytes =
        plan.reservation_size_bytes_in(&overrides).map(|bytes| bytes as u64).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.bounded.join.projected_cross_output_reservation",
                "failed to size bounded native projected cross join output",
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
            "failed to materialize bounded native projected cross join output",
            error,
        )
        .with_fact("join_type", node.spec.join_type)
    })?;
    let output = reorder_projected_cross_join_output(output, &node.spec.output_projection)?;
    metrics.operator_time += timer.elapsed();

    let output_rows = output.num_rows() as u64;
    debug_assert_eq!(output_rows, expected_output_rows);
    let output_bytes = observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), metrics)?;
    metrics.record_operator_metrics(crate::metrics::OperatorMetrics::join(
        node_id,
        crate::metrics::JoinOperatorMetrics {
            join_type: node.spec.join_type.to_string(),
            execution_strategy: "bounded_projected_cross_join".to_string(),
            key_count: 0,
            residual_predicate: false,
            null_equality: node.spec.null_equality.to_string(),
            build_side: build_side.to_string(),
            probe_side: probe_side.to_string(),
            build_rows: side_metric(build_side, left_rows, right_rows),
            build_bytes: side_metric(build_side, left_bytes, right_bytes),
            probe_rows: side_metric(probe_side, left_rows, right_rows),
            probe_bytes: side_metric(probe_side, left_bytes, right_bytes),
            hash_table_bytes: 0,
            filtered_probe_state_bytes: 0,
            mixed_size_data_bytes: 0,
            gather_map_bytes: 0,
            output_rows,
            output_bytes,
        },
    ));

    drop(left);
    drop(right);

    Ok(CommittedPartial { frame: GpuDataFrame::new(output), reservation: output_reservation })
}
