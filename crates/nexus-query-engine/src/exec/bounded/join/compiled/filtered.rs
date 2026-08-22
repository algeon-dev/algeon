use super::*;

pub(super) fn execute_compiled_filtered_join_build_from_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    input: GpuDataFrame,
    input_reservation: &ActiveReservation,
) -> Result<BoundedCompiledFilteredJoinBuildState> {
    validate_executable_join_subset(node)?;
    if !keyed_filtered_semi_anti_join_capable(node) {
        return Err(unsupported_bounded_shape(
            "compiled filtered join build requires a filtered semi/anti/mark join node",
            "compiled_join_strategy_unsupported",
        ));
    }
    let setup = filtered_join_build_setup(node);
    let build = prepare_bounded_join_side_input_from_shared_frame(
        execution,
        input,
        input_reservation,
        &setup.right_columns,
        JoinSide::Right,
        node_id,
        JoinSide::Right,
    )?;
    build_filtered_join_artifact(execution, node_id, node, setup, build)
}

pub(super) fn execute_compiled_filtered_join_build_from_resident_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    input: ResidentGpuFrame,
) -> Result<BoundedCompiledFilteredJoinBuildState> {
    validate_executable_join_subset(node)?;
    if !keyed_filtered_semi_anti_join_capable(node) {
        return Err(unsupported_bounded_shape(
            "compiled filtered join build requires a filtered semi/anti/mark join node",
            "compiled_join_strategy_unsupported",
        ));
    }
    let setup = filtered_join_build_setup(node);
    let build = prepare_bounded_join_side_input_from_resident_frame(
        execution,
        input,
        &setup.right_columns,
        JoinSide::Right,
        node_id,
        JoinSide::Right,
    )?;
    build_filtered_join_artifact(execution, node_id, node, setup, build)
}

struct FilteredJoinBuildSetup {
    left_keys: Vec<String>,
    right_keys: Vec<String>,
    left_columns: Vec<String>,
    right_columns: Vec<String>,
}

fn filtered_join_build_setup(node: &JoinNode) -> FilteredJoinBuildSetup {
    let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
    let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
    let left_columns = projected_input_columns(&left_keys, &node.spec.output_projection, [], JoinSide::Left);
    let right_columns = projected_input_columns(&right_keys, &node.spec.output_projection, [], JoinSide::Right);
    FilteredJoinBuildSetup { left_keys, right_keys, left_columns, right_columns }
}

fn build_filtered_join_artifact(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    setup: FilteredJoinBuildSetup,
    build: BoundedJoinSideFrame,
) -> Result<BoundedCompiledFilteredJoinBuildState> {
    let FilteredJoinBuildSetup { left_keys, right_keys, left_columns, .. } = setup;
    let input_build_rows = build.rows;
    let input_build_bytes = build.bytes;
    let build = deduplicate_filtered_join_build_keys(execution, node_id, build, &right_keys)?;
    let overrides_lease = execution_overrides(execution.context())?;
    let overrides = overrides_lease.overrides();
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
    let hash_table_reservation = execution.try_reserve(bounded_join_request(
        ReservationKind::HashTable,
        "join_hash_table",
        hash_table_reservation_bytes,
        execution.reservations().budget(),
        node_id,
    ))?;
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
    let filtered_join = GpuFilteredJoinBuildHandle::new(execution.context(), filtered_join, node.spec.null_equality)
        .map_err(|error| {
            error
                .with_fact("join_type", node.spec.join_type)
                .with_fact("key_count", node.spec.keys.len())
                .with_fact("build_side", JoinSide::Right)
        })?;
    let hash_table_bytes = filtered_join.device_allocated_size_bytes();
    execution.metrics_mut().record_filtered_join_build();
    execution.metrics_mut().record_filtered_join_hash_table_bytes(hash_table_bytes);
    Ok(BoundedCompiledFilteredJoinBuildState {
        left_keys,
        left_columns,
        build,
        input_build_rows,
        input_build_bytes,
        filtered_join,
        hash_table_reservation,
        hash_table_bytes,
        timer,
    })
}

fn deduplicate_filtered_join_build_keys(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    build: BoundedJoinSideFrame,
    right_keys: &[String],
) -> Result<BoundedJoinSideFrame> {
    let key_columns = right_keys.iter().map(String::as_str).collect::<Vec<_>>();
    let mut output_reservation = match execution.try_reserve(bounded_join_request(
        ReservationKind::OperatorOutput,
        "join_filtered_build_key_deduplicate",
        build.bytes,
        execution.reservations().budget(),
        node_id,
    )) {
        Ok(reservation) => reservation,
        Err(error) => {
            release_bounded_join_side_frame(execution, build);
            return Err(error);
        },
    };
    let token = match execution_budget_token_for_boundary(
        execution.context(),
        execution.reservations().available_bytes_including(&output_reservation),
        "join_filtered_build_key_deduplicate",
        "nexus_query_engine.bounded.join.filtered_build_key_deduplicate.budget_token",
        "failed to create filtered join build-key deduplication budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            execution.release(output_reservation);
            release_bounded_join_side_frame(execution, build);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let budget_overrides = token.execution_overrides();
    let output = match build.frame.with_execution(&budget_overrides).drop_duplicates(
        DropDuplicatesOptions::builder().subset(key_columns.as_slice()).keep(DuplicateKeepOption::KeepAny).build(),
    ) {
        Ok(output) => output,
        Err(error) => {
            execution.release(output_reservation);
            release_bounded_join_side_frame(execution, build);
            return Err(attach_budget_token_failure_facts(
                QueryError::from_cudf(
                    "nexus_query_engine.bounded.join.filtered_build_key_deduplicate",
                    "failed to deduplicate compiled filtered join build keys",
                    error,
                ),
                &token,
            )
            .with_fact("key_count", right_keys.len()));
        },
    };
    let output_bytes = match observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), execution.metrics_mut()) {
        Ok(bytes) => bytes,
        Err(error) => {
            execution.release(output_reservation);
            release_bounded_join_side_frame(execution, build);
            return Err(error);
        },
    };
    if output_bytes > output_reservation.bytes() {
        let delta = output_bytes.saturating_sub(output_reservation.bytes());
        if let Err(error) = execution.try_grow(
            &mut output_reservation,
            bounded_join_request(
                ReservationKind::OperatorOutput,
                "join_filtered_build_key_deduplicate",
                delta,
                execution.reservations().budget(),
                node_id,
            ),
        ) {
            execution.release(output_reservation);
            release_bounded_join_side_frame(execution, build);
            return Err(error);
        }
    } else {
        execution.shrink_to_observed(&mut output_reservation, output_bytes);
    }
    let output_rows = output.num_rows() as u64;
    release_bounded_join_side_frame(execution, build);
    Ok(BoundedJoinSideFrame { frame: output, _reservation: output_reservation, rows: output_rows, bytes: output_bytes })
}

pub(super) fn execute_compiled_filtered_join_probe_from_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    build: BoundedCompiledFilteredJoinBuildState,
    input: GpuDataFrame,
    input_reservation: &ActiveReservation,
) -> Result<BoundedCompiledFilteredJoinProbeState> {
    let probe = prepare_bounded_join_side_input_from_shared_frame(
        execution,
        input,
        input_reservation,
        &build.left_columns,
        JoinSide::Left,
        node_id,
        JoinSide::Right,
    )?;
    let overrides_lease = execution_overrides(execution.context())?;
    let overrides = overrides_lease.overrides();
    let (probe_state, probe_state_reservation, probe_state_bytes) = begin_filtered_join_probe_state(
        execution,
        node_id,
        node,
        &build,
        &probe,
        &overrides,
        "failed to observe compiled bounded filtered join probe-state memory",
    )?;
    Ok(BoundedCompiledFilteredJoinProbeState { build, probe, probe_state, probe_state_reservation, probe_state_bytes })
}

pub(super) fn execute_compiled_filtered_join_probe_from_resident_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    build: BoundedCompiledFilteredJoinBuildState,
    input: ResidentGpuFrame,
) -> Result<BoundedCompiledFilteredJoinProbeState> {
    let probe = prepare_bounded_join_side_input_from_resident_frame(
        execution,
        input,
        &build.left_columns,
        JoinSide::Left,
        node_id,
        JoinSide::Right,
    )?;
    let overrides_lease = execution_overrides(execution.context())?;
    let overrides = overrides_lease.overrides();
    let (probe_state, probe_state_reservation, probe_state_bytes) = begin_filtered_join_probe_state(
        execution,
        node_id,
        node,
        &build,
        &probe,
        &overrides,
        "failed to observe compiled bounded filtered join probe-state memory",
    )?;
    Ok(BoundedCompiledFilteredJoinProbeState { build, probe, probe_state, probe_state_reservation, probe_state_bytes })
}

fn begin_filtered_join_probe_state(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    build: &BoundedCompiledFilteredJoinBuildState,
    probe: &BoundedJoinSideFrame,
    overrides: &ExecutionOverrides<'_>,
    probe_state_bytes_message: &'static str,
) -> Result<(cudf::DataFrameFilteredJoinProbeState, ActiveReservation, u64)> {
    let probe_state = match node.spec.join_type {
        JoinType::LeftSemi | JoinType::LeftMark => {
            build.filtered_join.filtered_join().begin_left_semi_probe_in(&probe.frame, &build.left_keys, overrides)
        },
        JoinType::LeftAnti => {
            build.filtered_join.filtered_join().begin_left_anti_probe_in(&probe.frame, &build.left_keys, overrides)
        },
        JoinType::Inner | JoinType::Left | JoinType::Full | JoinType::Cross => unreachable!("validated above"),
    }
    .map_err(|error| join_index_error(error, node))?;
    let probe_state_bytes = probe_state.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.bounded.join.filtered_probe_state_bytes",
            probe_state_bytes_message,
            error,
        )
        .with_fact("join_type", node.spec.join_type)
        .with_fact("build_side", JoinSide::Right)
    })?;
    let probe_state_reservation = execution.try_reserve(bounded_join_request(
        ReservationKind::FilteredProbeState,
        "join_filtered_probe_state",
        probe_state_bytes,
        execution.reservations().budget(),
        node_id,
    ))?;
    execution.metrics_mut().record_filtered_join_probe_state_bytes(probe_state_bytes);
    Ok((probe_state, probe_state_reservation, probe_state_bytes))
}

pub(super) fn execute_compiled_filtered_join_probe_chunk_from_resident_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    build: &BoundedCompiledFilteredJoinBuildState,
    input: ResidentGpuFrame,
) -> Result<ResidentGpuFrame> {
    let probe = prepare_bounded_join_side_input_from_resident_frame(
        execution,
        input,
        &build.left_columns,
        JoinSide::Left,
        node_id,
        JoinSide::Right,
    )?;
    let overrides_lease = execution_overrides(execution.context())?;
    let overrides = overrides_lease.overrides();
    let (probe_state, probe_state_reservation, probe_state_bytes) = begin_filtered_join_probe_state(
        execution,
        node_id,
        node,
        build,
        &probe,
        &overrides,
        "failed to observe compiled streamed filtered join probe-state memory",
    )?;

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
    let gather_map_reservation = execution.try_reserve(bounded_join_request(
        ReservationKind::GatherMap,
        "join_gather_map",
        gather_map_reservation_bytes,
        execution.reservations().budget(),
        node_id,
    ))?;
    let indices = match probe_state.materialize_indices_in(&overrides) {
        Ok(indices) => indices,
        Err(error) => {
            execution.release(gather_map_reservation);
            execution.release(probe_state_reservation);
            return Err(join_index_error(error, node));
        },
    };
    execution.release(probe_state_reservation);
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
        let output_reservation = execution.try_reserve(bounded_join_request(
            ReservationKind::JoinOutput,
            "join_output",
            output_reservation_bytes,
            execution.reservations().budget(),
            node_id,
        ))?;
        let output = materialize_mark_join_output_projection(
            execution.context(),
            &probe.frame,
            &indices,
            &node.spec.output_projection,
            mark_column_name,
        )?;
        let output_rows = output.num_rows() as u64;
        let output_bytes = observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), execution.metrics_mut())?;
        record_filtered_join_output_metrics(
            execution.metrics_mut(),
            node_id,
            node,
            build,
            &probe,
            probe_state_bytes,
            gather_map_bytes,
            output_reservation_bytes,
            output_rows,
            output_bytes,
        );
        execution.release(gather_map_reservation);
        ResidentGpuFrame { frame: GpuDataFrame::new(output), reservation: output_reservation }
    } else {
        let left_projection =
            projected_output_columns_for_retained_sizing(&node.spec.output_projection, JoinSide::Left);
        let output_reservation_bytes = join_side_output_upper_bound_bytes(
            execution,
            node_id,
            &probe.frame,
            indices.len() as u64,
            &left_projection,
            JoinSide::Left,
        )?;
        let mut output_reservation = execution.try_reserve(bounded_join_request(
            ReservationKind::JoinOutput,
            "join_output",
            output_reservation_bytes,
            execution.reservations().budget(),
            node_id,
        ))?;
        let output = materialize_left_only_join_output_projection(
            &probe.frame,
            &indices,
            &overrides,
            &node.spec.output_projection,
            node.spec.join_type,
        )?;
        let output_rows = output.num_rows() as u64;
        let output_bytes = observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), execution.metrics_mut())?;
        reconcile_join_output_upper_bound(execution, node_id, &mut output_reservation, output_bytes)?;
        record_filtered_join_output_metrics(
            execution.metrics_mut(),
            node_id,
            node,
            build,
            &probe,
            probe_state_bytes,
            gather_map_bytes,
            output_reservation_bytes,
            output_rows,
            output_bytes,
        );
        execution.release(gather_map_reservation);
        ResidentGpuFrame { frame: GpuDataFrame::new(output), reservation: output_reservation }
    };
    Ok(output)
}

pub(super) fn execute_compiled_filtered_join_materialize(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    state: BoundedCompiledFilteredJoinProbeState,
) -> Result<ResidentGpuFrame> {
    let BoundedCompiledFilteredJoinProbeState { build, probe, probe_state, probe_state_reservation, probe_state_bytes } =
        state;
    let overrides_lease = execution_overrides(execution.context())?;
    let overrides = overrides_lease.overrides();
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
    let gather_map_reservation = execution.try_reserve(bounded_join_request(
        ReservationKind::GatherMap,
        "join_gather_map",
        gather_map_reservation_bytes,
        execution.reservations().budget(),
        node_id,
    ))?;
    let indices = probe_state.materialize_indices_in(&overrides).map_err(|error| join_index_error(error, node))?;
    execution.release(probe_state_reservation);
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
        let output_reservation = execution.try_reserve(bounded_join_request(
            ReservationKind::JoinOutput,
            "join_output",
            output_reservation_bytes,
            execution.reservations().budget(),
            node_id,
        ))?;
        let output = materialize_mark_join_output_projection(
            execution.context(),
            &probe.frame,
            &indices,
            &node.spec.output_projection,
            mark_column_name,
        )?;
        execution.metrics_mut().operator_time += build.timer.elapsed();
        let output_rows = output.num_rows() as u64;
        let output_bytes = observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), execution.metrics_mut())?;
        record_filtered_join_output_metrics(
            execution.metrics_mut(),
            node_id,
            node,
            &build,
            &probe,
            probe_state_bytes,
            gather_map_bytes,
            output_reservation_bytes,
            output_rows,
            output_bytes,
        );
        execution.release(gather_map_reservation);
        execution.release(build.hash_table_reservation);
        ResidentGpuFrame { frame: GpuDataFrame::new(output), reservation: output_reservation }
    } else {
        let left_projection =
            projected_output_columns_for_retained_sizing(&node.spec.output_projection, JoinSide::Left);
        let output_reservation_bytes = join_side_output_upper_bound_bytes(
            execution,
            node_id,
            &probe.frame,
            indices.len() as u64,
            &left_projection,
            JoinSide::Left,
        )?;
        let mut output_reservation = execution.try_reserve(bounded_join_request(
            ReservationKind::JoinOutput,
            "join_output",
            output_reservation_bytes,
            execution.reservations().budget(),
            node_id,
        ))?;
        let output = materialize_left_only_join_output_projection(
            &probe.frame,
            &indices,
            &overrides,
            &node.spec.output_projection,
            node.spec.join_type,
        )?;
        execution.metrics_mut().operator_time += build.timer.elapsed();
        let output_rows = output.num_rows() as u64;
        let output_bytes = observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), execution.metrics_mut())?;
        reconcile_join_output_upper_bound(execution, node_id, &mut output_reservation, output_bytes)?;
        record_filtered_join_output_metrics(
            execution.metrics_mut(),
            node_id,
            node,
            &build,
            &probe,
            probe_state_bytes,
            gather_map_bytes,
            output_reservation_bytes,
            output_rows,
            output_bytes,
        );
        execution.release(gather_map_reservation);
        execution.release(build.hash_table_reservation);
        ResidentGpuFrame { frame: GpuDataFrame::new(output), reservation: output_reservation }
    };
    Ok(output)
}

fn record_filtered_join_output_metrics(
    metrics: &mut QueryMetrics,
    node_id: NodeId,
    node: &JoinNode,
    build: &BoundedCompiledFilteredJoinBuildState,
    probe: &BoundedJoinSideFrame,
    probe_state_bytes: u64,
    gather_map_bytes: u64,
    output_reservation_bytes: u64,
    output_rows: u64,
    output_bytes: u64,
) {
    metrics.record_operator_metrics(crate::metrics::OperatorMetrics::join(
        node_id,
        crate::metrics::JoinOperatorMetrics {
            query_id: None,
            join_type: node.spec.join_type.to_string(),
            execution_strategy: "bounded_reusable_filtered_join".to_string(),
            key_count: node.spec.keys.len(),
            residual_predicate: false,
            null_equality: node.spec.null_equality.to_string(),
            build_side: JoinSide::Right.to_string(),
            probe_side: JoinSide::Left.to_string(),
            build_rows: build.input_build_rows,
            build_bytes: build.input_build_bytes,
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
}

pub(super) fn release_compiled_filtered_join_build_state(
    execution: &mut BoundedExecutionAttempt<'_>,
    build: BoundedCompiledFilteredJoinBuildState,
) {
    let BoundedCompiledFilteredJoinBuildState { build, filtered_join: _, hash_table_reservation, .. } = build;
    execution.release(hash_table_reservation);
    release_bounded_join_side_frame(execution, build);
}

pub(super) fn release_compiled_filtered_join_probe_state(
    execution: &mut BoundedExecutionAttempt<'_>,
    probe: BoundedCompiledFilteredJoinProbeState,
) {
    let BoundedCompiledFilteredJoinProbeState { build, probe, probe_state: _, probe_state_reservation, .. } = probe;
    execution.release(probe_state_reservation);
    release_compiled_filtered_join_build_state(execution, build);
    release_bounded_join_side_frame(execution, probe);
}
