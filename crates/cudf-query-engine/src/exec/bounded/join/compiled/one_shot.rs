use super::*;
use crate::exec::bounded::compiled::ResidentFrameStreamSource;

pub(in crate::exec::bounded) struct OneShotEquiJoinFrameStreamProbeState {
    pub(in crate::exec::bounded) build: BoundedCompiledPreparedJoinBuildState,
    pub(in crate::exec::bounded) inputs: ResidentFrameStreamPayload,
    probe_columns: Vec<String>,
}

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

pub(super) fn execute_compiled_one_shot_equi_join_probe_from_frame_stream(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    build: BoundedCompiledPreparedJoinBuildState,
    inputs: ResidentFrameStreamPayload,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedCompiledJoinProbeState> {
    validate_one_shot_equi_join(node)?;
    if node.spec.residual_predicate.is_some() {
        inputs.release_all(reservations, metrics);
        release_bounded_join_side_frame(build.input, reservations, metrics);
        return Err(unsupported_bounded_shape(
            "streamed compiled one-shot equi join probe does not support residual predicates",
            "compiled_join_strategy_unsupported",
        ));
    }
    let probe_side = probe_side(build.side);
    let probe_columns = match compiled_join_input_columns(context, node, build.strategy, probe_side) {
        Ok(columns) => columns,
        Err(error) => {
            inputs.release_all(reservations, metrics);
            release_bounded_join_side_frame(build.input, reservations, metrics);
            return Err(error);
        },
    };
    let _ = node_id;
    Ok(BoundedCompiledJoinProbeState::OneShotLazyStream(OneShotEquiJoinFrameStreamProbeState {
        build,
        inputs,
        probe_columns,
    }))
}

pub(super) fn one_shot_equi_join_frame_stream_probe_to_payload(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    probe: OneShotEquiJoinFrameStreamProbeState,
) -> ResidentFrameStreamPayload {
    let max_per_frame_reservation_bytes = probe.inputs.max_per_frame_reservation_bytes();
    let aggregate_reservation_bytes = probe.inputs.aggregate_reservation_bytes();
    let aggregate_num_rows = probe.inputs.num_rows();
    let num_columns = node.spec.output_projection.len();
    ResidentFrameStreamPayload::from_producer(
        max_per_frame_reservation_bytes,
        aggregate_reservation_bytes,
        aggregate_num_rows,
        num_columns,
        Vec::new(),
        Box::new(OneShotEquiJoinProbeStreamSource {
            context: context.clone(),
            node_id,
            node: node.clone(),
            build: Some(probe.build),
            inputs: Some(probe.inputs),
            probe_columns: probe.probe_columns,
            summary: StreamedOneShotJoinSummary::new(),
            metrics_recorded: false,
        }),
    )
}

struct OneShotEquiJoinProbeStreamSource {
    context: ExecutionContext,
    node_id: NodeId,
    node: JoinNode,
    build: Option<BoundedCompiledPreparedJoinBuildState>,
    inputs: Option<ResidentFrameStreamPayload>,
    probe_columns: Vec<String>,
    summary: StreamedOneShotJoinSummary,
    metrics_recorded: bool,
}

impl OneShotEquiJoinProbeStreamSource {
    fn record_success_metrics(&mut self, metrics: &mut QueryMetrics) {
        if self.metrics_recorded {
            return;
        }
        self.metrics_recorded = true;
        let Some(build) = self.build.as_ref() else { return };
        record_streamed_one_shot_join_metrics(
            self.node_id,
            &self.node,
            build.side,
            probe_side(build.side),
            build.input.rows,
            build.input.bytes,
            &self.summary,
            metrics,
        );
    }
}

impl ResidentFrameStreamSource for OneShotEquiJoinProbeStreamSource {
    fn next_frame(
        &mut self,
        reservations: &mut ReservationTracker,
        metrics: &mut QueryMetrics,
    ) -> Option<Result<ResidentGpuFrame>> {
        let inputs = self.inputs.as_mut()?;
        let mut input = match inputs.next_yielded_frame(reservations, metrics) {
            Some(Ok(input)) => input,
            Some(Err(error)) => return Some(Err(error)),
            None => {
                self.record_success_metrics(metrics);
                self.release(reservations, metrics);
                return None;
            },
        };
        let input = input.take_for_release();
        let build = self.build.as_ref().expect("one-shot join build side released before stream end");
        let probe_side = probe_side(build.side);
        let probe = match prepare_bounded_join_side_input_from_resident_frame(
            input,
            &self.probe_columns,
            probe_side,
            self.node_id,
            build.side,
            reservations,
            metrics,
        ) {
            Ok(probe) => probe,
            Err(error) => return Some(Err(error)),
        };
        Some(materialize_compiled_one_shot_equi_join_probe_chunk(
            &self.context,
            self.node_id,
            &self.node,
            build.side,
            &build.input,
            probe,
            reservations,
            metrics,
            &mut self.summary,
        ))
    }

    fn release(&mut self, reservations: &mut ReservationTracker, metrics: &mut QueryMetrics) {
        if let Some(input) = self.inputs.take() {
            input.release_all(reservations, metrics);
        }
        if let Some(build) = self.build.take() {
            release_bounded_join_side_frame(build.input, reservations, metrics);
        }
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
    let output_reservation_bytes = match indices.output_upper_bound_size_bytes(
        context,
        node_id,
        &left.frame,
        &right.frame,
        output_rows,
        &node.spec.output_projection,
        reservations,
        metrics,
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
    reservations.reclaim_source_cache_for_external_working_set(reservations.budget(), metrics)?;
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
    if let Err(error) =
        reconcile_join_output_upper_bound(node_id, &mut output_reservation, output_bytes, reservations, metrics)
    {
        drop(output);
        drop(indices);
        reservations.release(gather_map_reservation, metrics);
        reservations.release(output_reservation, metrics);
        drop(probe);
        return Err(error);
    }
    drop(indices);
    reservations.release(gather_map_reservation, metrics);
    let probe_rows = probe.rows;
    let probe_bytes = probe.bytes;
    drop(probe);
    summary.record_chunk(probe_rows, probe_bytes, index_bytes, output_reservation_bytes, output_rows, output_bytes);
    Ok(ResidentGpuFrame { frame: GpuDataFrame::new(output), reservation: output_reservation })
}
