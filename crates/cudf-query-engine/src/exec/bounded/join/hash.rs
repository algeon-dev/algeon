use super::{
    common::{bounded_join_request, join_side_output_reservation_bytes, prepare_bounded_join_side_input},
    *,
};

pub(super) fn execute_bounded_root_hash_join(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: ResidentGpuFrame,
    right: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    validate_executable_join_subset(node)?;
    if !node.is_bounded_root_hash_join_compatible() {
        return Err(unsupported_bounded_shape(
            "bounded root join execution currently supports reusable hash inner, left, and full equi joins",
            "bounded_join_plan_shape_unsupported",
        )
        .with_fact("join_type", node.spec.join_type)
        .with_fact("build_side", node.spec.build_side)
        .with_fact("key_count", node.spec.keys.len())
        .with_fact("residual_predicate", node.spec.residual_predicate.is_some())
        .with_fact("output_projection_len", node.spec.output_projection.len()));
    }

    let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
    let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
    let left_columns = projected_input_columns(&left_keys, &node.spec.output_projection, [], JoinSide::Left);
    let right_columns = projected_input_columns(&right_keys, &node.spec.output_projection, [], JoinSide::Right);
    let build_side = node.spec.build_side;
    let probe_side = probe_side(build_side);
    let overrides = execution_overrides(context);
    let hash_options =
        cudf::DataFrameHashJoinOptions::new().null_equality(lower_null_equality(node.spec.null_equality));
    let timer = Instant::now();
    let (left, right, hash_join, hash_table_reservation, hash_table_bytes, left_indices_from_probe) = match build_side {
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
            let (hash_join, hash_table_reservation, hash_table_bytes) = reserve_and_build_bounded_hash_join(
                context,
                node_id,
                node,
                &left.frame,
                &left_keys,
                hash_options.clone(),
                build_side,
                reservations,
                metrics,
                &overrides,
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
            (left, right, hash_join, hash_table_reservation, hash_table_bytes, false)
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
            let (hash_join, hash_table_reservation, hash_table_bytes) = reserve_and_build_bounded_hash_join(
                context,
                node_id,
                node,
                &right.frame,
                &right_keys,
                hash_options.clone(),
                build_side,
                reservations,
                metrics,
                &overrides,
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
            (left, right, hash_join, hash_table_reservation, hash_table_bytes, true)
        },
    };
    let original_left_rows = left.rows;
    let original_right_rows = right.rows;
    let left_bytes = left.bytes;
    let right_bytes = right.bytes;

    let (probe, probe_keys) = match build_side {
        JoinSide::Left => (&right.frame, &right_keys),
        JoinSide::Right => (&left.frame, &left_keys),
    };
    let build_bytes = side_metric(build_side, left_bytes, right_bytes);
    let probe_bytes = side_metric(probe_side, left_bytes, right_bytes);

    let output_size = match node.spec.join_type {
        JoinType::Inner => hash_join.hash_join().inner_probe_size_in(probe, probe_keys, &overrides),
        JoinType::Left => hash_join.hash_join().left_probe_size_in(probe, probe_keys, &overrides),
        JoinType::Full => hash_join.hash_join().full_probe_size_in(probe, probe_keys, &overrides),
        JoinType::LeftSemi | JoinType::LeftAnti | JoinType::Cross => unreachable!("validated above"),
    }
    .map_err(|error| join_index_error(error, node))?;
    let gather_map_reservation_bytes =
        hash_join.hash_join().probe_gather_map_reservation_size_bytes(output_size).map(|bytes| bytes as u64).map_err(
            |error| {
                QueryError::from_cudf(
                    "cudf_query_engine.bounded.join.gather_map_reservation",
                    "failed to size bounded native reusable hash join gather maps",
                    error,
                )
                .with_fact("join_type", node.spec.join_type)
                .with_fact("build_side", build_side)
            },
        )?;
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
    let indices = match node.spec.join_type {
        JoinType::Inner => {
            hash_join.hash_join().inner_probe_with_output_size_in(probe, probe_keys, output_size, &overrides)
        },
        JoinType::Left => {
            hash_join.hash_join().left_probe_with_output_size_in(probe, probe_keys, output_size, &overrides)
        },
        JoinType::Full => {
            hash_join.hash_join().full_probe_with_output_size_in(probe, probe_keys, output_size, &overrides)
        },
        JoinType::LeftSemi | JoinType::LeftAnti | JoinType::Cross => unreachable!("validated above"),
    }
    .map_err(|error| join_index_error(error, node))?;
    let gather_map_bytes = indices.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.bounded.join.gather_map_bytes",
            "failed to observe bounded native join gather-map memory",
            error,
        )
    })?;
    let (left_indices, right_indices) = if left_indices_from_probe {
        (indices.probe().clone(), indices.build().clone())
    } else {
        (indices.build().clone(), indices.probe().clone())
    };
    let left_projection = projected_output_columns(&node.spec.output_projection, JoinSide::Left);
    let right_projection = projected_output_columns(&node.spec.output_projection, JoinSide::Right);
    let left_output_bytes =
        join_side_output_reservation_bytes(&left.frame, &left_indices, &left_projection, &overrides, JoinSide::Left)?;
    let right_output_bytes = join_side_output_reservation_bytes(
        &right.frame,
        &right_indices,
        &right_projection,
        &overrides,
        JoinSide::Right,
    )?;
    let output_reservation_bytes = left_output_bytes.saturating_add(right_output_bytes);
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
    let output = materialize_join_output_projection(
        &left.frame,
        &right.frame,
        &left_indices,
        &right_indices,
        &overrides,
        &node.spec.output_projection,
    )?;
    metrics.operator_time += timer.elapsed();

    let output_rows = output.num_rows() as u64;
    let output_bytes = observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), metrics)?;
    metrics.record_operator_metrics(crate::metrics::OperatorMetrics::join(
        node_id,
        crate::metrics::JoinOperatorMetrics {
            join_type: node.spec.join_type.to_string(),
            execution_strategy: "bounded_reusable_hash_join".to_string(),
            key_count: node.spec.keys.len(),
            residual_predicate: false,
            null_equality: node.spec.null_equality.to_string(),
            build_side: build_side.to_string(),
            probe_side: probe_side.to_string(),
            build_rows: side_metric(build_side, original_left_rows, original_right_rows),
            build_bytes,
            probe_rows: side_metric(probe_side, original_left_rows, original_right_rows),
            probe_bytes,
            hash_table_bytes,
            filtered_probe_state_bytes: 0,
            mixed_size_data_bytes: 0,
            gather_map_bytes,
            output_rows,
            output_bytes,
        },
    ));

    // `left_indices`/`right_indices` are `Series::clone`s of `indices.probe()` /
    // `indices.build()`; the underlying columns are Arc-shared, so device memory
    // is only released once every Arc holder drops. All three must be released
    // before `reservations.release(gather_map_reservation)` for the budget to
    // reflect actual residency.
    drop(left_indices);
    drop(right_indices);
    drop(indices);
    reservations.release(gather_map_reservation);
    drop(hash_join);
    reservations.release(hash_table_reservation);
    drop(left);
    drop(right);

    Ok(CommittedPartial { frame: GpuDataFrame::new(output), reservation: output_reservation })
}

#[allow(clippy::too_many_arguments)]
fn reserve_and_build_bounded_hash_join(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    build: &DataFrame,
    build_keys: &[String],
    hash_options: cudf::DataFrameHashJoinOptions,
    build_side: JoinSide,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    overrides: &ExecutionOverrides<'_>,
) -> Result<(GpuHashJoinBuildHandle, ActiveReservation, u64)> {
    let hash_table_reservation_bytes = build
        .with_execution(overrides)
        .hash_join_pre_build_reservation_size_bytes(build_keys, hash_options.clone())
        .map(|bytes| bytes as u64)
        .map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.bounded.join.hash_table_reservation",
                "failed to size bounded native reusable hash join build",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
            .with_fact("build_side", build_side)
        })?;
    let hash_table_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::HashTable,
            "join_hash_table",
            hash_table_reservation_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;
    let hash_join = build.with_execution(overrides).build_hash_join(build_keys, hash_options).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.bounded.join.build_hash_join",
            "failed to build bounded native reusable hash join",
            error,
        )
        .with_fact("join_type", node.spec.join_type)
        .with_fact("key_count", node.spec.keys.len())
        .with_fact("build_side", build_side)
    })?;
    let hash_join = GpuHashJoinBuildHandle::new(context, hash_join, node.spec.null_equality).map_err(|error| {
        error
            .with_fact("join_type", node.spec.join_type)
            .with_fact("key_count", node.spec.keys.len())
            .with_fact("build_side", build_side)
    })?;
    let hash_table_bytes = hash_join.device_allocated_size_bytes();
    metrics.record_hash_join_build();
    metrics.record_hash_join_hash_table_bytes(hash_table_bytes);
    Ok((hash_join, hash_table_reservation, hash_table_bytes))
}
