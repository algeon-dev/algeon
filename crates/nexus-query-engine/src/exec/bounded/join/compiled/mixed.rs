use super::*;

pub(super) fn execute_compiled_mixed_join_materialize(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let residual_columns = residual_input_columns(node)?;
    let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
    let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
    let overrides_lease = execution_overrides(context)?;
    let overrides = overrides_lease.overrides();
    let predicate = lower_join_residual_predicate(context, node, &residual_columns)?;
    let options = dataframe_mixed_join_options(&left_keys, &right_keys, &residual_columns, node.spec.null_equality);
    let timer = Instant::now();
    let size_data = match node.spec.join_type {
        JoinType::Inner => left
            .frame
            .with_execution(&overrides)
            .mixed_inner_join_size_data(&right.frame, &predicate, options)
            .map(CompiledMixedSizeData::Pair),
        JoinType::Left => left
            .frame
            .with_execution(&overrides)
            .mixed_left_join_size_data(&right.frame, &predicate, options)
            .map(CompiledMixedSizeData::Pair),
        JoinType::Full => left
            .frame
            .with_execution(&overrides)
            .mixed_full_join_size_data(&right.frame, &predicate, options)
            .map(CompiledMixedSizeData::Pair),
        JoinType::LeftSemi => left
            .frame
            .with_execution(&overrides)
            .mixed_left_semi_join_size_data(&right.frame, &predicate, options)
            .map(CompiledMixedSizeData::LeftOnly),
        JoinType::LeftAnti => left
            .frame
            .with_execution(&overrides)
            .mixed_left_anti_join_size_data(&right.frame, &predicate, options)
            .map(CompiledMixedSizeData::LeftOnly),
        JoinType::Cross | JoinType::LeftMark => unreachable!("validated above"),
    }
    .map_err(|error| join_index_error(error, node))?;
    let expected_output_rows = size_data.output_size() as u64;
    let size_data_bytes = size_data.device_allocated_size_bytes(node)?;
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
    let gather_map_reservation_bytes = size_data.gather_map_reservation_size_bytes(node)?;
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
    let indices = size_data.materialize_indices_in(&overrides, node)?;
    reservations.release(size_data_reservation, metrics);
    materialize_compiled_indices_output(
        context,
        node,
        CompiledJoinMaterializationRequest {
            node_id,
            execution_strategy: "bounded_mixed_keyed_residual_join",
            key_count: node.spec.keys.len(),
            residual_predicate: true,
            build_side: node.spec.build_side,
            probe_side: probe_side(node.spec.build_side),
            left,
            right,
            indices,
            mixed_size_data_bytes: size_data_bytes,
            filtered_probe_state_bytes: 0,
            expected_output_rows,
            gather_map_reservation,
            timer,
            one_shot_index_bytes: None,
        },
        reservations,
        metrics,
    )
}

pub(super) fn execute_compiled_mixed_join_probe_from_stream(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    build: BoundedCompiledPreparedJoinBuildState,
    stream: BoundedJoinLeafLocalParquetSourceStream,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedCompiledJoinProbeState> {
    if node.spec.join_type != JoinType::Inner {
        release_bounded_join_side_frame(build.input, reservations, metrics);
        return Err(unsupported_bounded_shape(
            "streamed compiled mixed keyed residual join probe supports inner joins only",
            "compiled_join_strategy_unsupported",
        )
        .with_fact("join_type", node.spec.join_type)
        .with_fact("build_side", node.spec.build_side));
    }

    let probe_side = probe_side(build.side);
    let probe_columns = compiled_join_input_columns(node, build.strategy, probe_side)?;
    let mut outputs = Vec::new();
    let stream_result = stream_bounded_join_leaf_local_parquet_source_serial(
        context,
        stream,
        reservations,
        metrics,
        |input, reservations, metrics| {
            let probe = prepare_bounded_join_side_input_from_resident_frame(
                input,
                &probe_columns,
                probe_side,
                node_id,
                build.side,
                reservations,
                metrics,
            )?;
            let output = materialize_compiled_mixed_join_probe_chunk(
                context,
                node_id,
                node,
                build.side,
                &build.input,
                probe,
                reservations,
                metrics,
            )?;
            outputs.push(output);
            Ok(())
        },
    );
    match stream_result {
        Ok(()) => Ok(BoundedCompiledJoinProbeState::MixedStreamed { build, outputs }),
        Err(error) => {
            release_streamed_join_outputs_reverse(reservations, outputs, metrics);
            release_bounded_join_side_frame(build.input, reservations, metrics);
            Err(error)
        },
    }
}

fn materialize_compiled_mixed_join_probe_chunk(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    build_side: JoinSide,
    build: &BoundedJoinSideFrame,
    probe: BoundedJoinSideFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let build = alias_compiled_mixed_join_build_chunk(node_id, build, reservations, metrics)?;
    let probe_side = probe_side(build_side);
    let (left, right) = orient_prepared_inputs(build_side, build, probe_side, probe)?;
    execute_compiled_mixed_join_materialize(context, node_id, node, left, right, reservations, metrics)
}

fn alias_compiled_mixed_join_build_chunk(
    node_id: NodeId,
    build: &BoundedJoinSideFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedJoinSideFrame> {
    let reservation = reservations.alias_reservation(
        &build._reservation,
        bounded_join_request(
            ReservationKind::BuildSide,
            "join_build_side_stream_alias",
            build.bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    );
    Ok(BoundedJoinSideFrame {
        frame: build.frame.clone(),
        _reservation: reservation,
        rows: build.rows,
        bytes: build.bytes,
    })
}
