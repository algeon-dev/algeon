use super::*;

pub(super) fn execute_compiled_filtered_join_build_from_frame(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    input: GpuDataFrame,
    input_reservation: &ActiveReservation,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedCompiledFilteredJoinBuildState> {
    validate_executable_join_subset(node)?;
    if !keyed_filtered_semi_anti_join_capable(node) {
        return Err(unsupported_bounded_shape(
            "compiled filtered join build requires a filtered semi/anti/mark join node",
            "compiled_join_strategy_unsupported",
        ));
    }
    let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
    let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
    let left_columns = projected_input_columns(&left_keys, &node.spec.output_projection, [], JoinSide::Left);
    let right_columns = projected_input_columns(&right_keys, &node.spec.output_projection, [], JoinSide::Right);
    let build = prepare_bounded_join_side_input_from_shared_frame(
        input,
        input_reservation,
        &right_columns,
        JoinSide::Right,
        node_id,
        JoinSide::Right,
        reservations,
        metrics,
    )?;
    let overrides = execution_overrides(context);
    let filtered_options =
        cudf::DataFrameFilteredJoinOptions::new().null_equality(lower_null_equality(node.spec.null_equality));
    let hash_table_reservation_bytes = build
        .frame
        .with_execution(&overrides)
        .filtered_join_pre_build_reservation_size_bytes(&right_keys, filtered_options)
        .map(|bytes| bytes as u64)
        .map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.bounded.join.filtered_hash_table_reservation",
                "failed to size compiled bounded filtered join build",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
            .with_fact("build_side", JoinSide::Right)
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
    let timer = Instant::now();
    let filtered_join =
        build.frame.with_execution(&overrides).build_filtered_join(&right_keys, filtered_options).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.bounded.join.build_filtered_join",
                "failed to build compiled bounded filtered join",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
            .with_fact("key_count", node.spec.keys.len())
            .with_fact("build_side", JoinSide::Right)
        })?;
    let filtered_join =
        GpuFilteredJoinBuildHandle::new(context, filtered_join, node.spec.null_equality).map_err(|error| {
            error
                .with_fact("join_type", node.spec.join_type)
                .with_fact("key_count", node.spec.keys.len())
                .with_fact("build_side", JoinSide::Right)
        })?;
    let hash_table_bytes = filtered_join.device_allocated_size_bytes();
    metrics.record_filtered_join_build();
    metrics.record_filtered_join_hash_table_bytes(hash_table_bytes);
    Ok(BoundedCompiledFilteredJoinBuildState {
        left_keys,
        left_columns,
        build,
        filtered_join,
        hash_table_reservation,
        hash_table_bytes,
        timer,
    })
}

pub(super) fn execute_compiled_filtered_join_build_from_resident_frame(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    input: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedCompiledFilteredJoinBuildState> {
    validate_executable_join_subset(node)?;
    if !keyed_filtered_semi_anti_join_capable(node) {
        return Err(unsupported_bounded_shape(
            "compiled filtered join build requires a filtered semi/anti/mark join node",
            "compiled_join_strategy_unsupported",
        ));
    }
    let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
    let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
    let left_columns = projected_input_columns(&left_keys, &node.spec.output_projection, [], JoinSide::Left);
    let right_columns = projected_input_columns(&right_keys, &node.spec.output_projection, [], JoinSide::Right);
    let build = prepare_bounded_join_side_input_from_resident_frame(
        input,
        &right_columns,
        JoinSide::Right,
        node_id,
        JoinSide::Right,
        reservations,
        metrics,
    )?;
    let overrides = execution_overrides(context);
    let filtered_options =
        cudf::DataFrameFilteredJoinOptions::new().null_equality(lower_null_equality(node.spec.null_equality));
    let hash_table_reservation_bytes = build
        .frame
        .with_execution(&overrides)
        .filtered_join_pre_build_reservation_size_bytes(&right_keys, filtered_options)
        .map(|bytes| bytes as u64)
        .map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.bounded.join.filtered_hash_table_reservation",
                "failed to size compiled bounded filtered join build",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
            .with_fact("build_side", JoinSide::Right)
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
    let timer = Instant::now();
    let filtered_join =
        build.frame.with_execution(&overrides).build_filtered_join(&right_keys, filtered_options).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.bounded.join.build_filtered_join",
                "failed to build compiled bounded filtered join",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
            .with_fact("key_count", node.spec.keys.len())
            .with_fact("build_side", JoinSide::Right)
        })?;
    let filtered_join =
        GpuFilteredJoinBuildHandle::new(context, filtered_join, node.spec.null_equality).map_err(|error| {
            error
                .with_fact("join_type", node.spec.join_type)
                .with_fact("key_count", node.spec.keys.len())
                .with_fact("build_side", JoinSide::Right)
        })?;
    let hash_table_bytes = filtered_join.device_allocated_size_bytes();
    metrics.record_filtered_join_build();
    metrics.record_filtered_join_hash_table_bytes(hash_table_bytes);
    Ok(BoundedCompiledFilteredJoinBuildState {
        left_keys,
        left_columns,
        build,
        filtered_join,
        hash_table_reservation,
        hash_table_bytes,
        timer,
    })
}

pub(super) fn execute_compiled_filtered_join_probe_from_frame(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    build: BoundedCompiledFilteredJoinBuildState,
    input: GpuDataFrame,
    input_reservation: &ActiveReservation,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedCompiledFilteredJoinProbeState> {
    let probe = prepare_bounded_join_side_input_from_shared_frame(
        input,
        input_reservation,
        &build.left_columns,
        JoinSide::Left,
        node_id,
        JoinSide::Right,
        reservations,
        metrics,
    )?;
    let overrides = execution_overrides(context);
    let probe_state = match node.spec.join_type {
        JoinType::LeftSemi | JoinType::LeftMark => {
            build.filtered_join.filtered_join().begin_left_semi_probe_in(&probe.frame, &build.left_keys, &overrides)
        },
        JoinType::LeftAnti => {
            build.filtered_join.filtered_join().begin_left_anti_probe_in(&probe.frame, &build.left_keys, &overrides)
        },
        JoinType::Inner | JoinType::Left | JoinType::Full | JoinType::Cross => unreachable!("validated above"),
    }
    .map_err(|error| join_index_error(error, node))?;
    let probe_state_bytes = probe_state.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.bounded.join.filtered_probe_state_bytes",
            "failed to observe compiled bounded filtered join probe-state memory",
            error,
        )
        .with_fact("join_type", node.spec.join_type)
        .with_fact("build_side", JoinSide::Right)
    })?;
    let probe_state_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::FilteredProbeState,
            "join_filtered_probe_state",
            probe_state_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;
    metrics.record_filtered_join_probe_state_bytes(probe_state_bytes);
    Ok(BoundedCompiledFilteredJoinProbeState { build, probe, probe_state, probe_state_reservation, probe_state_bytes })
}

pub(super) fn execute_compiled_filtered_join_probe_from_resident_frame(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    build: BoundedCompiledFilteredJoinBuildState,
    input: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedCompiledFilteredJoinProbeState> {
    let probe = prepare_bounded_join_side_input_from_resident_frame(
        input,
        &build.left_columns,
        JoinSide::Left,
        node_id,
        JoinSide::Right,
        reservations,
        metrics,
    )?;
    let overrides = execution_overrides(context);
    let probe_state = match node.spec.join_type {
        JoinType::LeftSemi | JoinType::LeftMark => {
            build.filtered_join.filtered_join().begin_left_semi_probe_in(&probe.frame, &build.left_keys, &overrides)
        },
        JoinType::LeftAnti => {
            build.filtered_join.filtered_join().begin_left_anti_probe_in(&probe.frame, &build.left_keys, &overrides)
        },
        JoinType::Inner | JoinType::Left | JoinType::Full | JoinType::Cross => unreachable!("validated above"),
    }
    .map_err(|error| join_index_error(error, node))?;
    let probe_state_bytes = probe_state.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.bounded.join.filtered_probe_state_bytes",
            "failed to observe compiled bounded filtered join probe-state memory",
            error,
        )
        .with_fact("join_type", node.spec.join_type)
        .with_fact("build_side", JoinSide::Right)
    })?;
    let probe_state_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::FilteredProbeState,
            "join_filtered_probe_state",
            probe_state_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;
    metrics.record_filtered_join_probe_state_bytes(probe_state_bytes);
    Ok(BoundedCompiledFilteredJoinProbeState { build, probe, probe_state, probe_state_reservation, probe_state_bytes })
}

pub(super) fn execute_compiled_filtered_join_probe_chunk_from_resident_frame(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    build: &BoundedCompiledFilteredJoinBuildState,
    input: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let probe = prepare_bounded_join_side_input_from_resident_frame(
        input,
        &build.left_columns,
        JoinSide::Left,
        node_id,
        JoinSide::Right,
        reservations,
        metrics,
    )?;
    let overrides = execution_overrides(context);
    let probe_state = match node.spec.join_type {
        JoinType::LeftSemi | JoinType::LeftMark => {
            build.filtered_join.filtered_join().begin_left_semi_probe_in(&probe.frame, &build.left_keys, &overrides)
        },
        JoinType::LeftAnti => {
            build.filtered_join.filtered_join().begin_left_anti_probe_in(&probe.frame, &build.left_keys, &overrides)
        },
        JoinType::Inner | JoinType::Left | JoinType::Full | JoinType::Cross => unreachable!("validated above"),
    }
    .map_err(|error| join_index_error(error, node))?;
    let probe_state_bytes = probe_state.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.bounded.join.filtered_probe_state_bytes",
            "failed to observe compiled streamed filtered join probe-state memory",
            error,
        )
        .with_fact("join_type", node.spec.join_type)
        .with_fact("build_side", JoinSide::Right)
    })?;
    let probe_state_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::FilteredProbeState,
            "join_filtered_probe_state",
            probe_state_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;
    metrics.record_filtered_join_probe_state_bytes(probe_state_bytes);

    let gather_map_reservation_bytes =
        probe_state.probe_gather_map_reservation_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.bounded.join.filtered_gather_map_reservation",
                "failed to size compiled streamed filtered join gather map",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
            .with_fact("build_side", JoinSide::Right)
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
    let indices = match probe_state.materialize_indices_in(&overrides) {
        Ok(indices) => indices,
        Err(error) => {
            reservations.release(gather_map_reservation, metrics);
            reservations.release(probe_state_reservation, metrics);
            drop(probe);
            return Err(join_index_error(error, node));
        },
    };
    reservations.release(probe_state_reservation, metrics);
    let gather_map_bytes = indices.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.bounded.join.filtered_gather_map_bytes",
            "failed to observe compiled streamed filtered join gather-map memory",
            error,
        )
    })?;
    let output = if node.spec.join_type == JoinType::LeftMark {
        let mark_column_name = node.spec.mark_column_name.as_deref().unwrap_or("mark");
        // Mark join: estimate reservation as probe size (all rows kept) + mark column overhead.
        let output_reservation_bytes = probe.bytes.saturating_add(probe.rows.saturating_mul(1));
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
        let output = materialize_mark_join_output_projection(
            context,
            &probe.frame,
            &indices,
            &node.spec.output_projection,
            mark_column_name,
        )?;
        let output_rows = output.num_rows() as u64;
        let output_bytes = observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), metrics)?;
        metrics.record_operator_metrics(crate::metrics::OperatorMetrics::join(
            node_id,
            crate::metrics::JoinOperatorMetrics {
                join_type: node.spec.join_type.to_string(),
                execution_strategy: "bounded_reusable_filtered_join".to_string(),
                key_count: node.spec.keys.len(),
                residual_predicate: false,
                null_equality: node.spec.null_equality.to_string(),
                build_side: JoinSide::Right.to_string(),
                probe_side: JoinSide::Left.to_string(),
                build_rows: build.build.rows,
                build_bytes: build.build.bytes,
                probe_rows: probe.rows,
                probe_bytes: probe.bytes,
                hash_table_bytes: build.hash_table_bytes,
                filtered_probe_state_bytes: probe_state_bytes,
                mixed_size_data_bytes: 0,
                gather_map_bytes,
                output_reservation_bytes,
                output_rows,
                output_bytes,
            },
        ));
        drop(indices);
        reservations.release(gather_map_reservation, metrics);
        drop(probe);
        ResidentGpuFrame { frame: GpuDataFrame::new(output), reservation: output_reservation }
    } else {
        let left_projection =
            projected_output_columns_for_retained_sizing(&node.spec.output_projection, JoinSide::Left);
        let output_reservation_bytes = join_side_output_upper_bound_bytes(
            context,
            node_id,
            &probe.frame,
            indices.len() as u64,
            &left_projection,
            JoinSide::Left,
            reservations,
            metrics,
        )?;
        let mut output_reservation = reservations.try_reserve(
            bounded_join_request(
                ReservationKind::JoinOutput,
                "join_output",
                output_reservation_bytes,
                reservations.budget(),
                node_id,
            ),
            metrics,
        )?;
        let output = materialize_left_only_join_output_projection(
            &probe.frame,
            &indices,
            &overrides,
            &node.spec.output_projection,
            node.spec.join_type,
        )?;
        let output_rows = output.num_rows() as u64;
        let output_bytes = observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), metrics)?;
        reconcile_join_output_upper_bound(node_id, &mut output_reservation, output_bytes, reservations, metrics)?;
        metrics.record_operator_metrics(crate::metrics::OperatorMetrics::join(
            node_id,
            crate::metrics::JoinOperatorMetrics {
                join_type: node.spec.join_type.to_string(),
                execution_strategy: "bounded_reusable_filtered_join".to_string(),
                key_count: node.spec.keys.len(),
                residual_predicate: false,
                null_equality: node.spec.null_equality.to_string(),
                build_side: JoinSide::Right.to_string(),
                probe_side: JoinSide::Left.to_string(),
                build_rows: build.build.rows,
                build_bytes: build.build.bytes,
                probe_rows: probe.rows,
                probe_bytes: probe.bytes,
                hash_table_bytes: build.hash_table_bytes,
                filtered_probe_state_bytes: probe_state_bytes,
                mixed_size_data_bytes: 0,
                gather_map_bytes,
                output_reservation_bytes,
                output_rows,
                output_bytes,
            },
        ));
        drop(indices);
        reservations.release(gather_map_reservation, metrics);
        drop(probe);
        ResidentGpuFrame { frame: GpuDataFrame::new(output), reservation: output_reservation }
    };
    Ok(output)
}

pub(super) fn execute_compiled_filtered_join_materialize(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    state: BoundedCompiledFilteredJoinProbeState,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let BoundedCompiledFilteredJoinProbeState { build, probe, probe_state, probe_state_reservation, probe_state_bytes } =
        state;
    let overrides = execution_overrides(context);
    let gather_map_reservation_bytes =
        probe_state.probe_gather_map_reservation_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.bounded.join.filtered_gather_map_reservation",
                "failed to size compiled bounded filtered join gather map",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
            .with_fact("build_side", JoinSide::Right)
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
    let indices = probe_state.materialize_indices_in(&overrides).map_err(|error| join_index_error(error, node))?;
    reservations.release(probe_state_reservation, metrics);
    let gather_map_bytes = indices.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.bounded.join.filtered_gather_map_bytes",
            "failed to observe compiled bounded filtered join gather-map memory",
            error,
        )
    })?;
    let output = if node.spec.join_type == JoinType::LeftMark {
        let mark_column_name = node.spec.mark_column_name.as_deref().unwrap_or("mark");
        let output_reservation_bytes = probe.bytes.saturating_add(probe.rows.saturating_mul(1));
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
        let output = materialize_mark_join_output_projection(
            context,
            &probe.frame,
            &indices,
            &node.spec.output_projection,
            mark_column_name,
        )?;
        metrics.operator_time += build.timer.elapsed();
        let output_rows = output.num_rows() as u64;
        let output_bytes = observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), metrics)?;
        metrics.record_operator_metrics(crate::metrics::OperatorMetrics::join(
            node_id,
            crate::metrics::JoinOperatorMetrics {
                join_type: node.spec.join_type.to_string(),
                execution_strategy: "bounded_reusable_filtered_join".to_string(),
                key_count: node.spec.keys.len(),
                residual_predicate: false,
                null_equality: node.spec.null_equality.to_string(),
                build_side: JoinSide::Right.to_string(),
                probe_side: JoinSide::Left.to_string(),
                build_rows: build.build.rows,
                build_bytes: build.build.bytes,
                probe_rows: probe.rows,
                probe_bytes: probe.bytes,
                hash_table_bytes: build.hash_table_bytes,
                filtered_probe_state_bytes: probe_state_bytes,
                mixed_size_data_bytes: 0,
                gather_map_bytes,
                output_reservation_bytes,
                output_rows,
                output_bytes,
            },
        ));
        drop(indices);
        reservations.release(gather_map_reservation, metrics);
        drop(build.filtered_join);
        reservations.release(build.hash_table_reservation, metrics);
        drop(probe);
        drop(build.build);
        ResidentGpuFrame { frame: GpuDataFrame::new(output), reservation: output_reservation }
    } else {
        let left_projection =
            projected_output_columns_for_retained_sizing(&node.spec.output_projection, JoinSide::Left);
        let output_reservation_bytes = join_side_output_upper_bound_bytes(
            context,
            node_id,
            &probe.frame,
            indices.len() as u64,
            &left_projection,
            JoinSide::Left,
            reservations,
            metrics,
        )?;
        let mut output_reservation = reservations.try_reserve(
            bounded_join_request(
                ReservationKind::JoinOutput,
                "join_output",
                output_reservation_bytes,
                reservations.budget(),
                node_id,
            ),
            metrics,
        )?;
        let output = materialize_left_only_join_output_projection(
            &probe.frame,
            &indices,
            &overrides,
            &node.spec.output_projection,
            node.spec.join_type,
        )?;
        metrics.operator_time += build.timer.elapsed();
        let output_rows = output.num_rows() as u64;
        let output_bytes = observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), metrics)?;
        reconcile_join_output_upper_bound(node_id, &mut output_reservation, output_bytes, reservations, metrics)?;
        metrics.record_operator_metrics(crate::metrics::OperatorMetrics::join(
            node_id,
            crate::metrics::JoinOperatorMetrics {
                join_type: node.spec.join_type.to_string(),
                execution_strategy: "bounded_reusable_filtered_join".to_string(),
                key_count: node.spec.keys.len(),
                residual_predicate: false,
                null_equality: node.spec.null_equality.to_string(),
                build_side: JoinSide::Right.to_string(),
                probe_side: JoinSide::Left.to_string(),
                build_rows: build.build.rows,
                build_bytes: build.build.bytes,
                probe_rows: probe.rows,
                probe_bytes: probe.bytes,
                hash_table_bytes: build.hash_table_bytes,
                filtered_probe_state_bytes: probe_state_bytes,
                mixed_size_data_bytes: 0,
                gather_map_bytes,
                output_reservation_bytes,
                output_rows,
                output_bytes,
            },
        ));
        drop(indices);
        reservations.release(gather_map_reservation, metrics);
        drop(build.filtered_join);
        reservations.release(build.hash_table_reservation, metrics);
        drop(probe);
        drop(build.build);
        ResidentGpuFrame { frame: GpuDataFrame::new(output), reservation: output_reservation }
    };
    Ok(output)
}

pub(super) fn release_compiled_filtered_join_build_state(
    build: BoundedCompiledFilteredJoinBuildState,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) {
    let BoundedCompiledFilteredJoinBuildState { build, filtered_join, hash_table_reservation, .. } = build;
    drop(filtered_join);
    reservations.release(hash_table_reservation, metrics);
    release_bounded_join_side_frame(build, reservations, metrics);
}

pub(super) fn release_compiled_filtered_join_probe_state(
    probe: BoundedCompiledFilteredJoinProbeState,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) {
    let BoundedCompiledFilteredJoinProbeState { build, probe, probe_state, probe_state_reservation, .. } = probe;
    drop(probe_state);
    reservations.release(probe_state_reservation, metrics);
    release_compiled_filtered_join_build_state(build, reservations, metrics);
    release_bounded_join_side_frame(probe, reservations, metrics);
}
