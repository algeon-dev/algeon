use std::collections::{BTreeMap, BTreeSet};

use super::{
    common::{
        BoundedJoinSideFrame, bounded_join_request, join_side_output_reservation_bytes,
        prepare_bounded_join_side_input_from_resident_frame, prepare_bounded_join_side_input_from_shared_frame,
    },
    *,
};
use crate::{
    capability::{KeylessConditionalJoinKind, keyed_filtered_semi_anti_join_capable},
    exec::bounded::strict::CompiledJoinExecutionStrategy,
    expr::{ColumnQualifier, ColumnRef},
};

pub(in crate::exec::bounded) enum BoundedCompiledJoinBuildState {
    Hash(BoundedRootHashJoinBuildState),
    Filtered(BoundedCompiledFilteredJoinBuildState),
    Prepared(BoundedCompiledPreparedJoinBuildState),
}

pub(in crate::exec::bounded) enum BoundedCompiledJoinProbeState {
    Hash(BoundedRootHashJoinProbeState),
    HashStreamed {
        build: BoundedRootHashJoinBuildState,
        outputs: Vec<ResidentGpuFrame>,
    },
    Filtered(BoundedCompiledFilteredJoinProbeState),
    FilteredStreamed {
        build: BoundedCompiledFilteredJoinBuildState,
        outputs: Vec<ResidentGpuFrame>,
    },
    Prepared(BoundedCompiledPreparedJoinProbeState),
    OneShotStreamed {
        build: BoundedCompiledPreparedJoinBuildState,
        outputs: Vec<ResidentGpuFrame>,
        summary: StreamedOneShotJoinSummary,
    },
}

pub(in crate::exec::bounded) struct BoundedCompiledPreparedJoinBuildState {
    strategy: CompiledJoinExecutionStrategy,
    side: JoinSide,
    input: BoundedJoinSideFrame,
}

pub(in crate::exec::bounded) struct BoundedCompiledPreparedJoinProbeState {
    strategy: CompiledJoinExecutionStrategy,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
}

pub(in crate::exec::bounded) struct BoundedCompiledFilteredJoinBuildState {
    left_keys: Vec<String>,
    left_columns: Vec<String>,
    build: BoundedJoinSideFrame,
    filtered_join: GpuFilteredJoinBuildHandle,
    hash_table_reservation: ActiveReservation,
    hash_table_bytes: u64,
    timer: Instant,
}

pub(in crate::exec::bounded) struct BoundedCompiledFilteredJoinProbeState {
    build: BoundedCompiledFilteredJoinBuildState,
    probe: BoundedJoinSideFrame,
    probe_state: cudf::DataFrameFilteredJoinProbeState,
    probe_state_reservation: ActiveReservation,
    probe_state_bytes: u64,
}

pub(in crate::exec::bounded) struct StreamedOneShotJoinSummary {
    probe_rows: u64,
    probe_bytes: u64,
    index_bytes: u64,
    output_reservation_bytes: u64,
    output_rows: u64,
    output_bytes: u64,
}

pub(in crate::exec::bounded) enum BoundedCompiledJoinMaterializeOutput {
    Frame(Box<ResidentGpuFrame>),
    Stream(Vec<ResidentGpuFrame>),
}

impl StreamedOneShotJoinSummary {
    fn new() -> Self {
        Self {
            probe_rows: 0,
            probe_bytes: 0,
            index_bytes: 0,
            output_reservation_bytes: 0,
            output_rows: 0,
            output_bytes: 0,
        }
    }

    fn record_chunk(
        &mut self,
        probe_rows: u64,
        probe_bytes: u64,
        index_bytes: u64,
        output_reservation_bytes: u64,
        output_rows: u64,
        output_bytes: u64,
    ) {
        self.probe_rows = self.probe_rows.saturating_add(probe_rows);
        self.probe_bytes = self.probe_bytes.saturating_add(probe_bytes);
        self.index_bytes = self.index_bytes.saturating_add(index_bytes);
        self.output_reservation_bytes = self.output_reservation_bytes.saturating_add(output_reservation_bytes);
        self.output_rows = self.output_rows.saturating_add(output_rows);
        self.output_bytes = self.output_bytes.saturating_add(output_bytes);
    }
}

pub(in crate::exec::bounded) fn execute_bounded_compiled_join_build_from_frame(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    strategy: CompiledJoinExecutionStrategy,
    input: GpuDataFrame,
    input_reservation: &ActiveReservation,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedCompiledJoinBuildState> {
    match strategy {
        CompiledJoinExecutionStrategy::HashEqui => execute_bounded_root_hash_join_build_from_frame(
            context,
            node_id,
            node,
            input,
            input_reservation,
            reservations,
            metrics,
        )
        .map(BoundedCompiledJoinBuildState::Hash),
        CompiledJoinExecutionStrategy::FilteredSemiAnti => execute_compiled_filtered_join_build_from_frame(
            context,
            node_id,
            node,
            input,
            input_reservation,
            reservations,
            metrics,
        )
        .map(BoundedCompiledJoinBuildState::Filtered),
        CompiledJoinExecutionStrategy::OneShotEqui
        | CompiledJoinExecutionStrategy::KeylessConditional { .. }
        | CompiledJoinExecutionStrategy::ProjectedCross
        | CompiledJoinExecutionStrategy::MixedKeyedResidual => {
            let side = node.spec.build_side;
            let columns = compiled_join_input_columns(context, node, strategy, side)?;
            let input = prepare_bounded_join_side_input_from_shared_frame(
                input,
                input_reservation,
                &columns,
                side,
                node_id,
                node.spec.build_side,
                reservations,
                metrics,
            )?;
            Ok(BoundedCompiledJoinBuildState::Prepared(BoundedCompiledPreparedJoinBuildState { strategy, side, input }))
        },
    }
}

pub(in crate::exec::bounded) fn execute_bounded_compiled_join_build_from_resident_frame(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    strategy: CompiledJoinExecutionStrategy,
    input: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedCompiledJoinBuildState> {
    match strategy {
        CompiledJoinExecutionStrategy::HashEqui => execute_bounded_root_hash_join_build_from_resident_frame(
            context,
            node_id,
            node,
            input,
            reservations,
            metrics,
        )
        .map(BoundedCompiledJoinBuildState::Hash),
        CompiledJoinExecutionStrategy::FilteredSemiAnti => execute_compiled_filtered_join_build_from_resident_frame(
            context,
            node_id,
            node,
            input,
            reservations,
            metrics,
        )
        .map(BoundedCompiledJoinBuildState::Filtered),
        CompiledJoinExecutionStrategy::OneShotEqui
        | CompiledJoinExecutionStrategy::KeylessConditional { .. }
        | CompiledJoinExecutionStrategy::ProjectedCross
        | CompiledJoinExecutionStrategy::MixedKeyedResidual => {
            let side = node.spec.build_side;
            let columns = compiled_join_input_columns(context, node, strategy, side)?;
            let input = prepare_bounded_join_side_input_from_resident_frame(
                input,
                &columns,
                side,
                node_id,
                node.spec.build_side,
                reservations,
                metrics,
            )?;
            Ok(BoundedCompiledJoinBuildState::Prepared(BoundedCompiledPreparedJoinBuildState { strategy, side, input }))
        },
    }
}

pub(in crate::exec::bounded) fn execute_bounded_compiled_join_probe_from_frame(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    build: BoundedCompiledJoinBuildState,
    input: GpuDataFrame,
    input_reservation: &ActiveReservation,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedCompiledJoinProbeState> {
    match build {
        BoundedCompiledJoinBuildState::Hash(build) => execute_bounded_root_hash_join_probe_from_frame(
            context,
            node_id,
            node,
            build,
            input,
            input_reservation,
            reservations,
            metrics,
        )
        .map(BoundedCompiledJoinProbeState::Hash),
        BoundedCompiledJoinBuildState::Filtered(build) => execute_compiled_filtered_join_probe_from_frame(
            context,
            node_id,
            node,
            build,
            input,
            input_reservation,
            reservations,
            metrics,
        )
        .map(BoundedCompiledJoinProbeState::Filtered),
        BoundedCompiledJoinBuildState::Prepared(build) => {
            let side = probe_side(build.side);
            let columns = compiled_join_input_columns(context, node, build.strategy, side)?;
            let probe = prepare_bounded_join_side_input_from_shared_frame(
                input,
                input_reservation,
                &columns,
                side,
                node_id,
                node.spec.build_side,
                reservations,
                metrics,
            )?;
            let (left, right) = orient_prepared_inputs(build.side, build.input, side, probe)?;
            Ok(BoundedCompiledJoinProbeState::Prepared(BoundedCompiledPreparedJoinProbeState {
                strategy: build.strategy,
                left,
                right,
            }))
        },
    }
}

pub(in crate::exec::bounded) fn execute_bounded_compiled_join_probe_from_resident_frame(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    build: BoundedCompiledJoinBuildState,
    input: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedCompiledJoinProbeState> {
    match build {
        BoundedCompiledJoinBuildState::Hash(build) => execute_bounded_root_hash_join_probe_from_resident_frame(
            context,
            node_id,
            node,
            build,
            input,
            reservations,
            metrics,
        )
        .map(BoundedCompiledJoinProbeState::Hash),
        BoundedCompiledJoinBuildState::Filtered(build) => execute_compiled_filtered_join_probe_from_resident_frame(
            context,
            node_id,
            node,
            build,
            input,
            reservations,
            metrics,
        )
        .map(BoundedCompiledJoinProbeState::Filtered),
        BoundedCompiledJoinBuildState::Prepared(build) => {
            let side = probe_side(build.side);
            let columns = compiled_join_input_columns(context, node, build.strategy, side)?;
            let probe = prepare_bounded_join_side_input_from_resident_frame(
                input,
                &columns,
                side,
                node_id,
                node.spec.build_side,
                reservations,
                metrics,
            )?;
            let (left, right) = orient_prepared_inputs(build.side, build.input, side, probe)?;
            Ok(BoundedCompiledJoinProbeState::Prepared(BoundedCompiledPreparedJoinProbeState {
                strategy: build.strategy,
                left,
                right,
            }))
        },
    }
}

pub(in crate::exec::bounded) fn execute_bounded_compiled_join_probe_from_stream(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    build: BoundedCompiledJoinBuildState,
    stream: BoundedJoinLeafLocalParquetSourceStream,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedCompiledJoinProbeState> {
    match build {
        BoundedCompiledJoinBuildState::Hash(build) => {
            let mut outputs = Vec::new();
            let stream_result = stream_bounded_join_leaf_local_parquet_source(
                context,
                stream,
                reservations,
                metrics,
                |input, reservations, metrics| {
                    let output = execute_bounded_root_hash_join_probe_chunk_from_resident_frame(
                        context,
                        node_id,
                        node,
                        &build,
                        input,
                        reservations,
                        metrics,
                    )?;
                    outputs.push(output);
                    Ok(())
                },
            );
            match stream_result {
                Ok(()) => Ok(BoundedCompiledJoinProbeState::HashStreamed { build, outputs }),
                Err(error) => {
                    release_streamed_join_outputs_reverse(reservations, outputs, metrics);
                    release_bounded_root_hash_join_build_state(build, reservations, metrics);
                    Err(error)
                },
            }
        },
        BoundedCompiledJoinBuildState::Prepared(build)
            if build.strategy == CompiledJoinExecutionStrategy::OneShotEqui
                && node.spec.residual_predicate.is_none() =>
        {
            execute_compiled_one_shot_equi_join_probe_from_stream(
                context,
                node_id,
                node,
                build,
                stream,
                reservations,
                metrics,
            )
        },
        other => {
            let input =
                materialize_bounded_join_leaf_local_parquet_source_stream(context, stream, reservations, metrics)?;
            execute_bounded_compiled_join_probe_from_resident_frame(
                context,
                node_id,
                node,
                other,
                input,
                reservations,
                metrics,
            )
        },
    }
}

pub(in crate::exec::bounded) fn execute_bounded_compiled_join_probe_from_frame_stream(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    build: BoundedCompiledJoinBuildState,
    inputs: Vec<ResidentGpuFrame>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedCompiledJoinProbeState> {
    match build {
        BoundedCompiledJoinBuildState::Hash(build) => {
            let mut outputs = Vec::new();
            for input in inputs {
                match execute_bounded_root_hash_join_probe_chunk_from_resident_frame(
                    context,
                    node_id,
                    node,
                    &build,
                    input,
                    reservations,
                    metrics,
                ) {
                    Ok(output) => outputs.push(output),
                    Err(error) => {
                        release_streamed_join_outputs_reverse(reservations, outputs, metrics);
                        release_bounded_root_hash_join_build_state(build, reservations, metrics);
                        return Err(error);
                    },
                }
            }
            Ok(BoundedCompiledJoinProbeState::HashStreamed { build, outputs })
        },
        BoundedCompiledJoinBuildState::Filtered(build) => {
            let mut outputs = Vec::new();
            for input in inputs {
                match execute_compiled_filtered_join_probe_chunk_from_resident_frame(
                    context,
                    node_id,
                    node,
                    &build,
                    input,
                    reservations,
                    metrics,
                ) {
                    Ok(output) => outputs.push(output),
                    Err(error) => {
                        release_streamed_join_outputs_reverse(reservations, outputs, metrics);
                        release_compiled_filtered_join_build_state(build, reservations, metrics);
                        return Err(error);
                    },
                }
            }
            Ok(BoundedCompiledJoinProbeState::FilteredStreamed { build, outputs })
        },
        other => {
            release_streamed_join_outputs_reverse(reservations, inputs, metrics);
            drop(other);
            Err(unsupported_bounded_shape(
                "compiled join probe cannot consume a resident frame stream for this join strategy",
                "compiled_join_strategy_unsupported",
            ))
        },
    }
}

pub(in crate::exec::bounded) fn execute_bounded_compiled_join_materialize(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    probe: BoundedCompiledJoinProbeState,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    match probe {
        BoundedCompiledJoinProbeState::Hash(probe) => {
            execute_bounded_root_hash_join_materialize(context, node_id, node, probe, reservations, metrics)
        },
        BoundedCompiledJoinProbeState::HashStreamed { build, outputs } => {
            release_bounded_root_hash_join_build_state(build, reservations, metrics);
            concatenate_streamed_hash_join_outputs(context, node_id, outputs, reservations, metrics)
        },
        BoundedCompiledJoinProbeState::Filtered(probe) => {
            execute_compiled_filtered_join_materialize(context, node_id, node, probe, reservations, metrics)
        },
        BoundedCompiledJoinProbeState::FilteredStreamed { build, outputs } => {
            release_compiled_filtered_join_build_state(build, reservations, metrics);
            concatenate_streamed_hash_join_outputs(context, node_id, outputs, reservations, metrics)
        },
        BoundedCompiledJoinProbeState::Prepared(probe) => {
            execute_compiled_prepared_join_materialize(context, node_id, node, probe, reservations, metrics)
        },
        BoundedCompiledJoinProbeState::OneShotStreamed { build, outputs, summary } => {
            execute_compiled_streamed_one_shot_join_materialize(
                context,
                node_id,
                node,
                build,
                outputs,
                summary,
                reservations,
                metrics,
            )
        },
    }
}

pub(in crate::exec::bounded) fn execute_bounded_compiled_join_materialize_preserving_stream(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    probe: BoundedCompiledJoinProbeState,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedCompiledJoinMaterializeOutput> {
    match probe {
        BoundedCompiledJoinProbeState::FilteredStreamed { build, outputs } => {
            release_compiled_filtered_join_build_state(build, reservations, metrics);
            Ok(BoundedCompiledJoinMaterializeOutput::Stream(outputs))
        },
        BoundedCompiledJoinProbeState::OneShotStreamed { build, outputs, summary } => {
            let build_side = build.side;
            let probe_side = probe_side(build_side);
            let build_rows = build.input.rows;
            let build_bytes = build.input.bytes;
            drop(build);
            record_streamed_one_shot_join_metrics(
                node_id,
                node,
                build_side,
                probe_side,
                build_rows,
                build_bytes,
                &summary,
                metrics,
            );
            Ok(BoundedCompiledJoinMaterializeOutput::Stream(outputs))
        },
        other => execute_bounded_compiled_join_materialize(context, node_id, node, other, reservations, metrics)
            .map(|frame| BoundedCompiledJoinMaterializeOutput::Frame(Box::new(frame))),
    }
}

fn concatenate_streamed_hash_join_outputs(
    context: &ExecutionContext,
    node_id: NodeId,
    mut outputs: Vec<ResidentGpuFrame>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    if outputs.len() == 1 {
        return Ok(outputs.pop().expect("single streamed join output checked"));
    }
    if outputs.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.bounded.join.streamed_hash_output",
            "streamed hash join probe produced no output chunks",
        )
        .with_fact("node_id", node_id));
    }

    let output_reservation_bytes =
        outputs.iter().fold(0u64, |total, output| total.saturating_add(output.reservation.bytes()));
    let mut output_reservation = match reservations.try_reserve(
        bounded_join_request(
            ReservationKind::JoinOutput,
            "join_output_stream_concat",
            output_reservation_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            release_streamed_join_outputs_reverse(reservations, outputs, metrics);
            return Err(error);
        },
    };

    let mut input_reservations = Vec::with_capacity(outputs.len());
    let mut frames = Vec::with_capacity(outputs.len());
    for output in outputs {
        frames.push(output.frame.into_frame());
        input_reservations.push(output.reservation);
    }
    let chunk_count = input_reservations.len();
    let first_frame = frames.remove(0);
    let rest_frame_refs = frames.iter().collect::<Vec<_>>();
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let output = match first_frame.with_execution(&overrides).concat_many(&rest_frame_refs) {
        Ok(output) => output,
        Err(error) => {
            drop(rest_frame_refs);
            drop(frames);
            drop(first_frame);
            release_streamed_join_reservations_reverse(reservations, input_reservations, metrics);
            reservations.release(output_reservation, metrics);
            return Err(QueryError::from_cudf(
                "cudf_query_engine.bounded.join.streamed_hash_output_concat",
                "failed to concatenate streamed bounded hash join output chunks",
                error,
            )
            .with_fact("node_id", node_id)
            .with_fact("chunk_count", chunk_count));
        },
    };
    metrics.operator_time += timer.elapsed();

    let output_frame = GpuDataFrame::new(output);
    let output_bytes = match observe_device_memory_bytes(&output_frame, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(output_frame);
            drop(rest_frame_refs);
            drop(frames);
            drop(first_frame);
            release_streamed_join_reservations_reverse(reservations, input_reservations, metrics);
            reservations.release(output_reservation, metrics);
            return Err(error);
        },
    };
    if output_bytes > output_reservation.bytes() {
        let delta = output_bytes.saturating_sub(output_reservation.bytes());
        if let Err(error) = reservations.try_grow(
            &mut output_reservation,
            bounded_join_request(
                ReservationKind::JoinOutput,
                "join_output_stream_concat",
                delta,
                reservations.budget(),
                node_id,
            ),
            metrics,
        ) {
            drop(output_frame);
            drop(rest_frame_refs);
            drop(frames);
            drop(first_frame);
            release_streamed_join_reservations_reverse(reservations, input_reservations, metrics);
            reservations.release(output_reservation, metrics);
            return Err(error);
        }
    } else {
        reservations.shrink_to_observed(&mut output_reservation, output_bytes, metrics);
    }
    drop(rest_frame_refs);
    drop(frames);
    drop(first_frame);
    release_streamed_join_reservations_reverse(reservations, input_reservations, metrics);
    Ok(ResidentGpuFrame { frame: output_frame, reservation: output_reservation })
}

fn release_streamed_join_outputs_reverse(
    reservations: &mut ReservationTracker,
    mut outputs: Vec<ResidentGpuFrame>,
    metrics: &mut QueryMetrics,
) {
    while let Some(output) = outputs.pop() {
        drop(output.frame);
        reservations.release(output.reservation, metrics);
    }
}

fn release_streamed_join_reservations_reverse(
    reservations: &mut ReservationTracker,
    mut active: Vec<ActiveReservation>,
    metrics: &mut QueryMetrics,
) {
    while let Some(reservation) = active.pop() {
        reservations.release(reservation, metrics);
    }
}

fn execute_compiled_filtered_join_build_from_frame(
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
            "compiled filtered join build requires a filtered semi/anti join node",
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
                "cudf_query_engine.bounded.join.filtered_hash_table_reservation",
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
                "cudf_query_engine.bounded.join.build_filtered_join",
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

fn execute_compiled_filtered_join_build_from_resident_frame(
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
            "compiled filtered join build requires a filtered semi/anti join node",
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
                "cudf_query_engine.bounded.join.filtered_hash_table_reservation",
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
                "cudf_query_engine.bounded.join.build_filtered_join",
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

fn execute_compiled_filtered_join_probe_from_frame(
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
        JoinType::LeftSemi => {
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
            "cudf_query_engine.bounded.join.filtered_probe_state_bytes",
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

fn execute_compiled_filtered_join_probe_from_resident_frame(
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
        JoinType::LeftSemi => {
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
            "cudf_query_engine.bounded.join.filtered_probe_state_bytes",
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

fn execute_compiled_filtered_join_probe_chunk_from_resident_frame(
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
        JoinType::LeftSemi => {
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
            "cudf_query_engine.bounded.join.filtered_probe_state_bytes",
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
                "cudf_query_engine.bounded.join.filtered_gather_map_reservation",
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
            "cudf_query_engine.bounded.join.filtered_gather_map_bytes",
            "failed to observe compiled streamed filtered join gather-map memory",
            error,
        )
    })?;
    let left_projection = projected_output_columns(&node.spec.output_projection, JoinSide::Left);
    let output_reservation_bytes =
        join_side_output_reservation_bytes(&probe.frame, &indices, &left_projection, &overrides, JoinSide::Left)?;
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
    let output =
        materialize_left_only_join_output_projection(&probe.frame, &indices, &overrides, &node.spec.output_projection)?;
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
    Ok(ResidentGpuFrame { frame: GpuDataFrame::new(output), reservation: output_reservation })
}

fn execute_compiled_filtered_join_materialize(
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
                "cudf_query_engine.bounded.join.filtered_gather_map_reservation",
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
            "cudf_query_engine.bounded.join.filtered_gather_map_bytes",
            "failed to observe compiled bounded filtered join gather-map memory",
            error,
        )
    })?;
    let left_projection = projected_output_columns(&node.spec.output_projection, JoinSide::Left);
    let output_reservation_bytes =
        join_side_output_reservation_bytes(&probe.frame, &indices, &left_projection, &overrides, JoinSide::Left)?;
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
    let output =
        materialize_left_only_join_output_projection(&probe.frame, &indices, &overrides, &node.spec.output_projection)?;
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
    Ok(ResidentGpuFrame { frame: GpuDataFrame::new(output), reservation: output_reservation })
}

fn release_compiled_filtered_join_build_state(
    build: BoundedCompiledFilteredJoinBuildState,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) {
    let BoundedCompiledFilteredJoinBuildState { build, filtered_join, hash_table_reservation, .. } = build;
    drop(filtered_join);
    reservations.release(hash_table_reservation, metrics);
    drop(build);
}

fn execute_compiled_prepared_join_materialize(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    state: BoundedCompiledPreparedJoinProbeState,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    match state.strategy {
        CompiledJoinExecutionStrategy::KeylessConditional { join_type } => execute_compiled_keyless_join_materialize(
            context,
            node_id,
            node,
            join_type,
            state.left,
            state.right,
            reservations,
            metrics,
        ),
        CompiledJoinExecutionStrategy::ProjectedCross => execute_compiled_projected_cross_join_materialize(
            context,
            node_id,
            node,
            state.left,
            state.right,
            reservations,
            metrics,
        ),
        CompiledJoinExecutionStrategy::MixedKeyedResidual => execute_compiled_mixed_join_materialize(
            context,
            node_id,
            node,
            state.left,
            state.right,
            reservations,
            metrics,
        ),
        CompiledJoinExecutionStrategy::OneShotEqui => execute_compiled_one_shot_equi_join_materialize(
            context,
            node_id,
            node,
            state.left,
            state.right,
            reservations,
            metrics,
        ),
        CompiledJoinExecutionStrategy::HashEqui | CompiledJoinExecutionStrategy::FilteredSemiAnti => {
            Err(unsupported_bounded_shape(
                "compiled prepared join state received an incompatible strategy",
                "compiled_join_strategy_state_mismatch",
            ))
        },
    }
}

fn compiled_join_input_columns(
    context: &ExecutionContext,
    node: &JoinNode,
    strategy: CompiledJoinExecutionStrategy,
    side: JoinSide,
) -> Result<Vec<String>> {
    match strategy {
        CompiledJoinExecutionStrategy::KeylessConditional { .. } => {
            let residual_columns = residual_input_columns(node)?;
            Ok(projected_input_columns(
                &[],
                &node.spec.output_projection,
                residual_columns.names(side).iter().map(String::as_str),
                side,
            ))
        },
        CompiledJoinExecutionStrategy::ProjectedCross => {
            Ok(projected_output_columns(&node.spec.output_projection, side))
        },
        CompiledJoinExecutionStrategy::MixedKeyedResidual => {
            let residual_columns = residual_input_columns(node)?;
            let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
            let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
            let keys = if side == JoinSide::Left { &left_keys } else { &right_keys };
            Ok(projected_input_columns(
                keys,
                &node.spec.output_projection,
                residual_columns.names(side).iter().map(String::as_str),
                side,
            ))
        },
        CompiledJoinExecutionStrategy::OneShotEqui => {
            let residual_columns = residual_input_columns(node)?;
            let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
            let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
            let keys = if side == JoinSide::Left { &left_keys } else { &right_keys };
            Ok(projected_input_columns(
                keys,
                &node.spec.output_projection,
                residual_columns.names(side).iter().map(String::as_str),
                side,
            ))
        },
        CompiledJoinExecutionStrategy::HashEqui | CompiledJoinExecutionStrategy::FilteredSemiAnti => {
            let _ = context;
            Err(unsupported_bounded_shape(
                "compiled prepared join input columns require a prepared join strategy",
                "compiled_join_strategy_state_mismatch",
            ))
        },
    }
}

fn orient_prepared_inputs(
    build_side: JoinSide,
    build: BoundedJoinSideFrame,
    probe_side: JoinSide,
    probe: BoundedJoinSideFrame,
) -> Result<(BoundedJoinSideFrame, BoundedJoinSideFrame)> {
    match (build_side, probe_side) {
        (JoinSide::Left, JoinSide::Right) => Ok((build, probe)),
        (JoinSide::Right, JoinSide::Left) => Ok((probe, build)),
        _ => Err(unsupported_bounded_shape(
            "compiled prepared join inputs must contain exactly one left and one right side",
            "compiled_join_input_orientation_invalid",
        )),
    }
}

fn validate_one_shot_equi_join(node: &JoinNode) -> Result<()> {
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
fn execute_compiled_keyless_join_materialize(
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

fn execute_compiled_mixed_join_materialize(
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
    let overrides = execution_overrides(context);
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
        JoinType::Cross => unreachable!("validated above"),
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
        node_id,
        node,
        "bounded_mixed_keyed_residual_join",
        node.spec.keys.len(),
        true,
        node.spec.build_side,
        probe_side(node.spec.build_side),
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

fn execute_compiled_one_shot_equi_join_materialize(
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
        JoinType::LeftSemi | JoinType::LeftAnti | JoinType::Cross => unreachable!("validated above"),
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

fn execute_compiled_one_shot_equi_join_probe_from_stream(
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

#[allow(clippy::too_many_arguments)]
fn materialize_compiled_one_shot_equi_join_probe_chunk(
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
        JoinType::LeftSemi | JoinType::LeftAnti | JoinType::Cross => unreachable!("validated above"),
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
    let output_reservation_bytes = match indices.output_reservation_size_bytes(
        &left.frame,
        &right.frame,
        &overrides,
        &node.spec.output_projection,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
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
    let output = match indices.materialize_output(&left.frame, &right.frame, &overrides, &node.spec.output_projection) {
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

#[allow(clippy::too_many_arguments)]
fn execute_compiled_streamed_one_shot_join_materialize(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    build: BoundedCompiledPreparedJoinBuildState,
    outputs: Vec<ResidentGpuFrame>,
    summary: StreamedOneShotJoinSummary,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let build_side = build.side;
    let probe_side = probe_side(build_side);
    let build_rows = build.input.rows;
    let build_bytes = build.input.bytes;
    drop(build);

    let output = concatenate_streamed_hash_join_outputs(context, node_id, outputs, reservations, metrics)?;
    record_streamed_one_shot_join_metrics(
        node_id,
        node,
        build_side,
        probe_side,
        build_rows,
        build_bytes,
        &summary,
        metrics,
    );
    Ok(output)
}

fn record_streamed_one_shot_join_metrics(
    node_id: NodeId,
    node: &JoinNode,
    build_side: JoinSide,
    probe_side: JoinSide,
    build_rows: u64,
    build_bytes: u64,
    summary: &StreamedOneShotJoinSummary,
    metrics: &mut QueryMetrics,
) {
    metrics.record_one_shot_join();
    metrics.record_one_shot_join_index_bytes(summary.index_bytes);
    metrics.record_one_shot_join_materialization_bytes(summary.output_reservation_bytes);
    metrics.record_one_shot_join_output_bytes(summary.output_bytes);
    metrics.record_operator_metrics(crate::metrics::OperatorMetrics::join(
        node_id,
        crate::metrics::JoinOperatorMetrics {
            join_type: node.spec.join_type.to_string(),
            execution_strategy: "bounded_one_shot_equi_join".to_string(),
            key_count: node.spec.keys.len(),
            residual_predicate: false,
            null_equality: node.spec.null_equality.to_string(),
            build_side: build_side.to_string(),
            probe_side: probe_side.to_string(),
            build_rows,
            build_bytes,
            probe_rows: summary.probe_rows,
            probe_bytes: summary.probe_bytes,
            hash_table_bytes: 0,
            filtered_probe_state_bytes: 0,
            mixed_size_data_bytes: 0,
            gather_map_bytes: summary.index_bytes,
            output_reservation_bytes: summary.output_reservation_bytes,
            output_rows: summary.output_rows,
            output_bytes: summary.output_bytes,
        },
    ));
}

#[allow(clippy::too_many_arguments)]
fn materialize_compiled_one_shot_residual_output(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    indices: CompiledJoinIndices,
    candidate_rows: u64,
    gather_map_reservation: ActiveReservation,
    timer: Instant,
    index_bytes: u64,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let overrides = execution_overrides(context);
    let residual_candidate = one_shot_residual_candidate_projection(node)?;
    let gather_map_bytes = indices.device_allocated_size_bytes(node)?;
    let candidate_reservation_bytes =
        indices.output_reservation_size_bytes(&left.frame, &right.frame, &overrides, &residual_candidate.projection)?;
    metrics.record_one_shot_join();
    metrics.record_one_shot_join_index_bytes(index_bytes);
    metrics.record_one_shot_join_materialization_bytes(candidate_reservation_bytes);
    let candidate_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::JoinOutput,
            "join_residual_candidate_output",
            candidate_reservation_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;
    let candidate =
        indices.materialize_output(&left.frame, &right.frame, &overrides, &residual_candidate.projection)?;
    let candidate_output_rows = candidate.num_rows() as u64;
    debug_assert_eq!(candidate_output_rows, candidate_rows);
    drop(indices);
    reservations.release(gather_map_reservation, metrics);

    let mut filtered_reservation = match reservations.try_reserve(
        bounded_join_request(
            ReservationKind::JoinOutput,
            "join_residual_filter_output",
            candidate_reservation_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            drop(candidate);
            reservations.release(candidate_reservation, metrics);
            return Err(error);
        },
    };
    let token = match execution_budget_token_for_boundary(
        context,
        reservations.available_bytes_including(&filtered_reservation),
        "join_residual_filter_output",
        "cudf_query_engine.bounded.join.one_shot_residual_filter.budget_token",
        "failed to create bounded native one-shot residual filter budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            drop(candidate);
            reservations.release(filtered_reservation, metrics);
            reservations.release(candidate_reservation, metrics);
            return Err(error);
        },
    };
    let predicate = lower_expr(&residual_candidate.predicate)?;
    let filter_overrides = token.execution_overrides();
    let filtered = match candidate.with_execution(&filter_overrides).filter_expr(predicate) {
        Ok(filtered) => filtered,
        Err(error) => {
            drop(candidate);
            reservations.release(filtered_reservation, metrics);
            reservations.release(candidate_reservation, metrics);
            return Err(attach_budget_token_failure_facts(
                QueryError::from_cudf(
                    "cudf_query_engine.bounded.join.one_shot_residual_filter",
                    "failed to filter compiled one-shot residual join candidates",
                    error,
                )
                .with_fact("join_type", node.spec.join_type)
                .with_fact("key_count", node.spec.keys.len())
                .with_fact("build_side", node.spec.build_side),
                &token,
            ));
        },
    };
    drop(candidate);
    reservations.release(candidate_reservation, metrics);

    let output =
        filtered.select(residual_candidate.final_output_names.iter().map(String::as_str)).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.bounded.join.one_shot_residual_output_projection",
                "failed to project compiled one-shot residual join output columns",
                error,
            )
        })?;
    drop(filtered);
    metrics.operator_time += timer.elapsed();
    let output_rows = output.num_rows() as u64;
    let output_bytes = observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), metrics)?;
    metrics.record_one_shot_join_output_bytes(output_bytes);
    if output_bytes > filtered_reservation.bytes() {
        let grow_request = bounded_join_request(
            ReservationKind::JoinOutput,
            "join_residual_filter_output",
            output_bytes.saturating_sub(filtered_reservation.bytes()),
            reservations.budget(),
            node_id,
        );
        if let Err(error) = reservations.try_grow(&mut filtered_reservation, grow_request, metrics) {
            drop(output);
            reservations.release(filtered_reservation, metrics);
            drop(left);
            drop(right);
            return Err(error);
        }
    } else {
        reservations.shrink_to_observed(&mut filtered_reservation, output_bytes, metrics);
    }
    let final_output_reservation_bytes = filtered_reservation.bytes();
    let build_side = node.spec.build_side;
    let probe_side = probe_side(build_side);
    metrics.record_operator_metrics(crate::metrics::OperatorMetrics::join(
        node_id,
        crate::metrics::JoinOperatorMetrics {
            join_type: node.spec.join_type.to_string(),
            execution_strategy: "bounded_one_shot_equi_join".to_string(),
            key_count: node.spec.keys.len(),
            residual_predicate: true,
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
            gather_map_bytes,
            output_reservation_bytes: final_output_reservation_bytes,
            output_rows,
            output_bytes,
        },
    ));
    drop(left);
    drop(right);
    Ok(ResidentGpuFrame { frame: GpuDataFrame::new(output), reservation: filtered_reservation })
}

struct OneShotResidualCandidateProjection {
    projection: Vec<JoinOutputColumn>,
    predicate: Expr,
    final_output_names: Vec<String>,
}

fn one_shot_residual_candidate_projection(node: &JoinNode) -> Result<OneShotResidualCandidateProjection> {
    let residual_columns = residual_input_columns(node)?;
    let mut projection = node.spec.output_projection.clone();
    let final_output_names = node.spec.output_projection.iter().map(|column| column.output_name.clone()).collect();
    let mut used_names = projection.iter().map(|column| column.output_name.clone()).collect::<BTreeSet<_>>();
    let mut residual_output_names = BTreeMap::new();

    for side in [JoinSide::Left, JoinSide::Right] {
        for (index, name) in residual_columns.names(side).iter().enumerate() {
            let output_name = residual_candidate_column_name(side, index, name, &mut used_names);
            residual_output_names.insert((residual_candidate_side_key(side), name.clone()), output_name.clone());
            let input_index = match side {
                JoinSide::Left => 0,
                JoinSide::Right => 1,
            };
            projection.push(JoinOutputColumn::new(
                side,
                ColumnRef::input(input_index, name.clone(), output_name.clone()),
                output_name,
            ));
        }
    }

    let predicate = rewrite_join_residual_for_candidate(
        node.spec.residual_predicate.as_ref().ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.bounded.join.one_shot_residual_projection",
                "one-shot residual projection requires a residual predicate",
            )
        })?,
        &residual_output_names,
    )?;

    Ok(OneShotResidualCandidateProjection { projection, predicate, final_output_names })
}

fn residual_candidate_column_name(
    side: JoinSide,
    index: usize,
    name: &str,
    used_names: &mut BTreeSet<String>,
) -> String {
    let side_label = match side {
        JoinSide::Left => "left",
        JoinSide::Right => "right",
    };
    let mut sanitized = name.chars().map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' }).collect::<String>();
    if sanitized.is_empty() {
        sanitized.push_str("column");
    }
    let base = format!("__cudf_one_shot_residual_{side_label}_{index}_{sanitized}");
    let mut candidate = base.clone();
    let mut suffix = 0_usize;
    while !used_names.insert(candidate.clone()) {
        suffix += 1;
        candidate = format!("{base}_{suffix}");
    }
    candidate
}

fn residual_candidate_side_key(side: JoinSide) -> u8 {
    match side {
        JoinSide::Left => 0,
        JoinSide::Right => 1,
    }
}

fn rewrite_join_residual_for_candidate(
    expr: &Expr,
    residual_output_names: &BTreeMap<(u8, String), String>,
) -> Result<Expr> {
    match expr {
        Expr::Column(column) => {
            let side = match column.qualifier() {
                Some(ColumnQualifier::Input { index: 0 }) => JoinSide::Left,
                Some(ColumnQualifier::Input { index: 1 }) => JoinSide::Right,
                qualifier => {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedExpression,
                        "cudf_query_engine.bounded.join.one_shot_residual_projection",
                        "one-shot residual predicates require input-qualified column references",
                    )
                    .with_fact("column", column.name())
                    .with_fact("qualifier", qualifier.map(ToString::to_string).unwrap_or_else(|| "none".to_string())));
                },
            };
            let name = residual_output_names
                .get(&(residual_candidate_side_key(side), column.name().to_string()))
                .ok_or_else(|| {
                    QueryError::new(
                        QueryErrorCode::InvalidPlan,
                        "cudf_query_engine.bounded.join.one_shot_residual_projection",
                        "one-shot residual predicate column was not materialized into the candidate frame",
                    )
                    .with_fact("side", side)
                    .with_fact("column", column.name())
                })?;
            Ok(Expr::column(name.clone()))
        },
        Expr::Literal(value) => Ok(Expr::Literal(value.clone())),
        Expr::Not(input) => Ok(Expr::logical_not(rewrite_join_residual_for_candidate(input, residual_output_names)?)),
        Expr::IsNull(input) => Ok(Expr::is_null(rewrite_join_residual_for_candidate(input, residual_output_names)?)),
        Expr::IsNotNull(input) => {
            Ok(Expr::is_not_null(rewrite_join_residual_for_candidate(input, residual_output_names)?))
        },
        Expr::Cast { expr, target_type } => {
            Ok(Expr::cast(rewrite_join_residual_for_candidate(expr, residual_output_names)?, target_type.clone()))
        },
        Expr::Binary { left, op, right } => Ok(Expr::binary(
            rewrite_join_residual_for_candidate(left, residual_output_names)?,
            *op,
            rewrite_join_residual_for_candidate(right, residual_output_names)?,
        )),
        Expr::Like { expr, pattern, escape, negated } => Ok(Expr::Like {
            expr: Box::new(rewrite_join_residual_for_candidate(expr, residual_output_names)?),
            pattern: pattern.clone(),
            escape: escape.clone(),
            negated: *negated,
        }),
        Expr::Substring { expr, start, length } => {
            Ok(Expr::substring(rewrite_join_residual_for_candidate(expr, residual_output_names)?, *start, *length))
        },
        Expr::InList { expr, list, negated } => Ok(Expr::InList {
            expr: Box::new(rewrite_join_residual_for_candidate(expr, residual_output_names)?),
            list: list
                .iter()
                .map(|item| rewrite_join_residual_for_candidate(item, residual_output_names))
                .collect::<Result<Vec<_>>>()?,
            negated: *negated,
        }),
        Expr::Case { branches, else_expr } => Ok(Expr::case(
            branches
                .iter()
                .map(|branch| {
                    Ok(crate::expr::CaseWhen {
                        when: rewrite_join_residual_for_candidate(&branch.when, residual_output_names)?,
                        then: rewrite_join_residual_for_candidate(&branch.then, residual_output_names)?,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            rewrite_join_residual_for_candidate(else_expr, residual_output_names)?,
        )),
        Expr::Extract { field, expr } => {
            Ok(Expr::extract(*field, rewrite_join_residual_for_candidate(expr, residual_output_names)?))
        },
        Expr::Alias { expr, alias } => {
            Ok(rewrite_join_residual_for_candidate(expr, residual_output_names)?.alias(alias.clone()))
        },
    }
}

fn execute_compiled_projected_cross_join_materialize(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
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
            execution_strategy: "bounded_projected_cross_join".to_string(),
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

enum CompiledKeylessWork<'a> {
    Full {
        size_data: Box<cudf::DataFrameConditionalFullJoinSizeData<'a>>,
        size_data_reservation: Box<ActiveReservation>,
    },
    Sized {
        output_size: usize,
        options: cudf::DataFramePredicateJoinOptions,
    },
}

enum CompiledMixedSizeData<'a> {
    Pair(cudf::DataFrameMixedJoinSizeData<'a>),
    LeftOnly(cudf::DataFrameMixedSingleSideJoinSizeData<'a>),
}

impl CompiledMixedSizeData<'_> {
    fn output_size(&self) -> usize {
        match self {
            Self::Pair(size_data) => size_data.output_size(),
            Self::LeftOnly(size_data) => size_data.output_size(),
        }
    }

    fn device_allocated_size_bytes(&self, node: &JoinNode) -> Result<u64> {
        let result = match self {
            Self::Pair(size_data) => size_data.device_allocated_size_bytes(),
            Self::LeftOnly(size_data) => size_data.device_allocated_size_bytes(),
        };
        result.map(|bytes| bytes as u64).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.bounded.join.mixed_size_data_bytes",
                "failed to observe compiled bounded mixed join size-data memory",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
        })
    }

    fn gather_map_reservation_size_bytes(&self, node: &JoinNode) -> Result<u64> {
        let result = match self {
            Self::Pair(size_data) => size_data.gather_map_reservation_size_bytes(),
            Self::LeftOnly(size_data) => size_data.gather_map_reservation_size_bytes(),
        };
        result.map(|bytes| bytes as u64).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.bounded.join.mixed_gather_map_reservation",
                "failed to size compiled bounded mixed join gather maps",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
        })
    }

    fn materialize_indices_in(
        self,
        overrides: &ExecutionOverrides<'_>,
        node: &JoinNode,
    ) -> Result<CompiledJoinIndices> {
        match self {
            Self::Pair(size_data) => size_data
                .materialize_indices_in(overrides)
                .map(|indices| CompiledJoinIndices::Pair {
                    left: indices.left().clone(),
                    right: indices.right().clone(),
                })
                .map_err(|error| join_index_error(error, node)),
            Self::LeftOnly(size_data) => size_data
                .materialize_indices_in(overrides)
                .map(CompiledJoinIndices::LeftOnly)
                .map_err(|error| join_index_error(error, node)),
        }
    }
}

enum CompiledJoinIndices {
    Pair { left: cudf::DataFrameJoinIndices, right: cudf::DataFrameJoinIndices },
    LeftOnly(cudf::DataFrameJoinIndices),
}

impl CompiledJoinIndices {
    fn device_allocated_size_bytes(&self, node: &JoinNode) -> Result<u64> {
        match self {
            Self::Pair { left, right } => {
                let left_bytes = left.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
                    QueryError::from_cudf(
                        "cudf_query_engine.bounded.join.gather_map_bytes",
                        "failed to observe compiled bounded join left gather-map memory",
                        error,
                    )
                    .with_fact("join_type", node.spec.join_type)
                })?;
                let right_bytes = right.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
                    QueryError::from_cudf(
                        "cudf_query_engine.bounded.join.gather_map_bytes",
                        "failed to observe compiled bounded join right gather-map memory",
                        error,
                    )
                    .with_fact("join_type", node.spec.join_type)
                })?;
                Ok(left_bytes.saturating_add(right_bytes))
            },
            Self::LeftOnly(indices) => {
                indices.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
                    QueryError::from_cudf(
                        "cudf_query_engine.bounded.join.gather_map_bytes",
                        "failed to observe compiled bounded join gather-map memory",
                        error,
                    )
                    .with_fact("join_type", node.spec.join_type)
                })
            },
        }
    }

    fn output_reservation_size_bytes(
        &self,
        left: &DataFrame,
        right: &DataFrame,
        overrides: &ExecutionOverrides<'_>,
        projection: &[JoinOutputColumn],
    ) -> Result<u64> {
        match self {
            Self::Pair { left: left_indices, right: right_indices } => {
                let left_projection = projected_output_columns(projection, JoinSide::Left);
                let right_projection = projected_output_columns(projection, JoinSide::Right);
                let left_bytes = join_side_output_reservation_bytes(
                    left,
                    left_indices,
                    &left_projection,
                    overrides,
                    JoinSide::Left,
                )?;
                let right_bytes = join_side_output_reservation_bytes(
                    right,
                    right_indices,
                    &right_projection,
                    overrides,
                    JoinSide::Right,
                )?;
                Ok(left_bytes.saturating_add(right_bytes))
            },
            Self::LeftOnly(indices) => {
                let left_projection = projected_output_columns(projection, JoinSide::Left);
                join_side_output_reservation_bytes(left, indices, &left_projection, overrides, JoinSide::Left)
            },
        }
    }

    fn materialize_output(
        &self,
        left: &DataFrame,
        right: &DataFrame,
        overrides: &ExecutionOverrides<'_>,
        projection: &[JoinOutputColumn],
    ) -> Result<DataFrame> {
        match self {
            Self::Pair { left: left_indices, right: right_indices } => {
                materialize_join_output_projection(left, right, left_indices, right_indices, overrides, projection)
            },
            Self::LeftOnly(indices) => {
                materialize_left_only_join_output_projection(left, indices, overrides, projection)
            },
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn materialize_compiled_indices_output(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    execution_strategy: &'static str,
    key_count: usize,
    residual_predicate: bool,
    build_side: JoinSide,
    probe_side: JoinSide,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    indices: CompiledJoinIndices,
    mixed_size_data_bytes: u64,
    filtered_probe_state_bytes: u64,
    expected_output_rows: u64,
    gather_map_reservation: ActiveReservation,
    timer: Instant,
    one_shot_index_bytes: Option<u64>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let overrides = execution_overrides(context);
    let gather_map_bytes = indices.device_allocated_size_bytes(node)?;
    let output_reservation_bytes =
        indices.output_reservation_size_bytes(&left.frame, &right.frame, &overrides, &node.spec.output_projection)?;
    if let Some(index_bytes) = one_shot_index_bytes {
        metrics.record_one_shot_join();
        metrics.record_one_shot_join_index_bytes(index_bytes);
        metrics.record_one_shot_join_materialization_bytes(output_reservation_bytes);
    }
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
    let output = indices.materialize_output(&left.frame, &right.frame, &overrides, &node.spec.output_projection)?;
    metrics.operator_time += timer.elapsed();
    let output_rows = output.num_rows() as u64;
    debug_assert_eq!(output_rows, expected_output_rows);
    let output_bytes = observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), metrics)?;
    if one_shot_index_bytes.is_some() {
        metrics.record_one_shot_join_output_bytes(output_bytes);
    }
    metrics.record_operator_metrics(crate::metrics::OperatorMetrics::join(
        node_id,
        crate::metrics::JoinOperatorMetrics {
            join_type: node.spec.join_type.to_string(),
            execution_strategy: execution_strategy.to_string(),
            key_count,
            residual_predicate,
            null_equality: node.spec.null_equality.to_string(),
            build_side: build_side.to_string(),
            probe_side: probe_side.to_string(),
            build_rows: side_metric(build_side, left.rows, right.rows),
            build_bytes: side_metric(build_side, left.bytes, right.bytes),
            probe_rows: side_metric(probe_side, left.rows, right.rows),
            probe_bytes: side_metric(probe_side, left.bytes, right.bytes),
            hash_table_bytes: 0,
            filtered_probe_state_bytes,
            mixed_size_data_bytes,
            gather_map_bytes,
            output_reservation_bytes,
            output_rows,
            output_bytes,
        },
    ));
    drop(indices);
    reservations.release(gather_map_reservation, metrics);
    drop(left);
    drop(right);
    Ok(ResidentGpuFrame { frame: GpuDataFrame::new(output), reservation: output_reservation })
}

fn keyless_conditional_join_output_size(
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

fn keyless_conditional_join_gather_map_reservation_size_bytes(
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
fn keyless_conditional_join_indices_with_output_size(
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
