use super::*;

pub(super) fn validate_one_shot_equi_join(node: &JoinNode) -> Result<()> {
    validate_executable_join_subset(node)?;
    if matches!(node.spec.join_type, JoinType::Inner | JoinType::Left | JoinType::Full)
        && !node.spec.keys.is_empty()
        && (node.spec.residual_predicate.is_none() || node.spec.join_type == JoinType::Inner)
        && (node.spec.join_type != JoinType::Left || node.spec.build_side == JoinSide::Right)
    {
        return Ok(());
    }
    Err(unsupported_bounded_shape(
        "compiled one-shot equi join execution supports keyed inner joins with optional residual predicates and keyed left/full joins without residual predicates",
        "compiled_join_strategy_unsupported",
    )
    .with_fact("join_type", node.spec.join_type)
    .with_fact("build_side", node.spec.build_side)
    .with_fact("key_count", node.spec.keys.len())
    .with_fact("residual_predicate", node.spec.residual_predicate.is_some())
    .with_fact("output_projection_len", node.spec.output_projection.len()))
}

// The remaining prepared materializers intentionally mirror the root bounded
// join primitives, but start after build/probe input preparation.
pub(super) fn execute_compiled_one_shot_equi_join_materialize(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    validate_one_shot_equi_join(node)?;
    let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
    let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
    let options = dataframe_join_options(&left_keys, &right_keys, node.spec.null_equality);
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let pair_indices = match node.spec.join_type {
        JoinType::Inner => left.frame.with_execution(&overrides).inner_join_indices(&right.frame, options),
        JoinType::Left => left.frame.with_execution(&overrides).left_join_indices(&right.frame, options),
        JoinType::Full => left.frame.with_execution(&overrides).full_join_indices(&right.frame, options),
        JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark | JoinType::Cross => {
            unreachable!("validated above")
        },
    }
    .map_err(|error| join_index_error(error, node))?;
    let index_bytes = pair_indices.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.bounded.join.one_shot_index_bytes",
            "failed to observe compiled one-shot join index memory",
            error,
        )
        .with_fact("join_type", node.spec.join_type)
    })?;
    let gather_map_reservation = match reservations.try_reserve(
        bounded_join_request(
            ReservationKind::GatherMap,
            "join_one_shot_indices",
            index_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            drop(pair_indices);
            return Err(error);
        },
    };
    let output_rows = pair_indices.len() as u64;
    let indices = CompiledJoinIndices::Pair { left: pair_indices.left().clone(), right: pair_indices.right().clone() };
    drop(pair_indices);
    if node.spec.residual_predicate.is_some() {
        return materialize_compiled_one_shot_residual_output(
            context,
            node_id,
            node,
            left,
            right,
            indices,
            output_rows,
            gather_map_reservation,
            timer,
            index_bytes,
            reservations,
            metrics,
        );
    }
    materialize_compiled_indices_output(
        context,
        node_id,
        node,
        "bounded_one_shot_equi_join",
        node.spec.keys.len(),
        false,
        node.spec.build_side,
        probe_side(node.spec.build_side),
        left,
        right,
        indices,
        0,
        0,
        output_rows,
        gather_map_reservation,
        timer,
        Some(index_bytes),
        reservations,
        metrics,
    )
}

pub(super) fn execute_compiled_one_shot_equi_join_probe_from_stream(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    build: BoundedCompiledPreparedJoinBuildState,
    stream: BoundedJoinLeafLocalParquetSourceStream,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedCompiledJoinProbeState> {
    validate_one_shot_equi_join(node)?;
    if node.spec.residual_predicate.is_some() {
        return Err(unsupported_bounded_shape(
            "streamed compiled one-shot equi join probe does not support residual predicates",
            "compiled_join_strategy_unsupported",
        ));
    }

    let probe_side = probe_side(build.side);
    let probe_columns = compiled_join_input_columns(context, node, build.strategy, probe_side)?;
    let mut outputs = Vec::new();
    let mut summary = StreamedOneShotJoinSummary::new();
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
            let output = materialize_compiled_one_shot_equi_join_probe_chunk(
                context,
                node_id,
                node,
                build.side,
                &build.input,
                probe,
                reservations,
                metrics,
                &mut summary,
            )?;
            outputs.push(output);
            Ok(())
        },
    );
    match stream_result {
        Ok(()) => Ok(BoundedCompiledJoinProbeState::OneShotStreamed { build, outputs, summary }),
        Err(error) => {
            release_streamed_join_outputs_reverse(reservations, outputs, metrics);
            Err(error)
        },
    }
}
pub(super) fn materialize_compiled_one_shot_equi_join_probe_chunk(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    build_side: JoinSide,
    build: &BoundedJoinSideFrame,
    probe: BoundedJoinSideFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    summary: &mut StreamedOneShotJoinSummary,
) -> Result<ResidentGpuFrame> {
    let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
    let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
    let options = dataframe_join_options(&left_keys, &right_keys, node.spec.null_equality);
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let (left, right) = match build_side {
        JoinSide::Left => (build, &probe),
        JoinSide::Right => (&probe, build),
    };
    let pair_indices = match node.spec.join_type {
        JoinType::Inner => left.frame.with_execution(&overrides).inner_join_indices(&right.frame, options),
        JoinType::Left => left.frame.with_execution(&overrides).left_join_indices(&right.frame, options),
        JoinType::Full => left.frame.with_execution(&overrides).full_join_indices(&right.frame, options),
        JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark | JoinType::Cross => {
            unreachable!("validated above")
        },
    }
    .map_err(|error| join_index_error(error, node))?;
    let index_bytes = pair_indices.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.bounded.join.one_shot_index_bytes",
            "failed to observe compiled streamed one-shot join index memory",
            error,
        )
        .with_fact("join_type", node.spec.join_type)
    })?;
    let gather_map_reservation = match reservations.try_reserve(
        bounded_join_request(
            ReservationKind::GatherMap,
            "join_one_shot_indices",
            index_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            drop(pair_indices);
            drop(probe);
            return Err(error);
        },
    };
    let output_rows = pair_indices.len() as u64;
    let indices = CompiledJoinIndices::Pair { left: pair_indices.left().clone(), right: pair_indices.right().clone() };
    drop(pair_indices);
    super::super::common::trace_preflight_entry(
        "materialize_compiled_one_shot_equi_join_probe_chunk",
        "streamed_one_shot",
        node_id,
        reservations,
        index_bytes,
        output_rows,
        left.rows,
        left.bytes,
        right.rows,
        right.bytes,
    );
    let output_reservation_bytes = match indices.output_reservation_size_bytes(
        &left.frame,
        &right.frame,
        &overrides,
        &node.spec.output_projection,
    ) {
        Ok(bytes) => {
            super::super::common::trace_preflight_success(
                "materialize_compiled_one_shot_equi_join_probe_chunk",
                node_id,
                reservations,
                bytes,
            );
            bytes
        },
        Err(error) => {
            super::super::common::trace_preflight_failed(
                "materialize_compiled_one_shot_equi_join_probe_chunk",
                node_id,
                reservations,
                &error,
            );
            drop(indices);
            reservations.release(gather_map_reservation, metrics);
            drop(probe);
            return Err(error);
        },
    };
    let mut output_reservation = match reservations.try_reserve(
        bounded_join_request(
            ReservationKind::JoinOutput,
            "join_output",
            output_reservation_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            drop(indices);
            reservations.release(gather_map_reservation, metrics);
            drop(probe);
            return Err(error);
        },
    };
    let output = match indices.materialize_output(
        &left.frame,
        &right.frame,
        &overrides,
        &node.spec.output_projection,
        node.spec.join_type,
    ) {
        Ok(output) => output,
        Err(error) => {
            drop(indices);
            reservations.release(gather_map_reservation, metrics);
            reservations.release(output_reservation, metrics);
            drop(probe);
            return Err(error);
        },
    };
    metrics.operator_time += timer.elapsed();
    let output_bytes = match observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(output);
            drop(indices);
            reservations.release(gather_map_reservation, metrics);
            reservations.release(output_reservation, metrics);
            drop(probe);
            return Err(error);
        },
    };
    if output_bytes > output_reservation.bytes() {
        let grow_request = bounded_join_request(
            ReservationKind::JoinOutput,
            "join_output",
            output_bytes.saturating_sub(output_reservation.bytes()),
            reservations.budget(),
            node_id,
        );
        if let Err(error) = reservations.try_grow(&mut output_reservation, grow_request, metrics) {
            drop(output);
            drop(indices);
            reservations.release(gather_map_reservation, metrics);
            reservations.release(output_reservation, metrics);
            drop(probe);
            return Err(error);
        }
    } else {
        reservations.shrink_to_observed(&mut output_reservation, output_bytes, metrics);
    }
    drop(indices);
    reservations.release(gather_map_reservation, metrics);
    let probe_rows = probe.rows;
    let probe_bytes = probe.bytes;
    drop(probe);
    summary.record_chunk(probe_rows, probe_bytes, index_bytes, output_reservation_bytes, output_rows, output_bytes);
    Ok(ResidentGpuFrame { frame: GpuDataFrame::new(output), reservation: output_reservation })
}
