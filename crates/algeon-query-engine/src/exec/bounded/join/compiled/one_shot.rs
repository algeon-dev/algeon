use std::time::Instant;

use super::{
    BoundedCompiledJoinProbeState, BoundedCompiledPreparedJoinBuildState, BoundedExecutionAttempt,
    BoundedJoinLeafLocalParquetSourceStream, BoundedJoinSideFrame, CompiledEqualityJoinPreflightIndices,
    CompiledJoinMaterializationRequest, ExecutionMetrics, GpuDataFrame, JoinNode, JoinSide, JoinType, NodeId,
    ResidentFrameStreamPayload, ResidentGpuFrame, Result, StreamedOneShotJoinSummary, bounded_invariant_violation,
    compiled_join_input_columns, execution_overrides, materialize_compiled_indices_output,
    materialize_compiled_keyed_equality_join_indices_with_preflight, materialize_compiled_one_shot_residual_output,
    observe_device_memory_bytes, orient_prepared_inputs, prepare_bounded_join_side_input_from_resident_frame,
    probe_side, record_one_shot_join_chunk_metrics, record_streamed_one_shot_join_metrics,
    release_bounded_join_side_frame, stream_bounded_join_leaf_local_parquet_source_serial,
    validate_executable_join_subset,
};
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
    Err(bounded_invariant_violation(
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
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
) -> Result<ResidentGpuFrame> {
    validate_one_shot_equi_join(node)?;
    let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
    let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
    let overrides_lease = execution_overrides(execution.context())?;
    let overrides = overrides_lease.overrides();
    let timer = Instant::now();
    let preflight_indices = match materialize_compiled_keyed_equality_join_indices_with_preflight(
        execution,
        node,
        &left,
        &right,
        node.spec.build_side,
        &left_keys,
        &right_keys,
        &overrides,
    ) {
        Ok(indices) => indices,
        Err(error) => {
            release_bounded_join_side_frame(left);
            release_bounded_join_side_frame(right);
            return Err(error);
        },
    };
    let CompiledEqualityJoinPreflightIndices { indices, index_bytes, output_rows } = preflight_indices;
    if node.spec.residual_predicate.is_some() {
        let (output, chunk_metrics) = materialize_compiled_one_shot_residual_output(
            execution,
            node_id,
            node,
            left,
            right,
            indices,
            output_rows,
            index_bytes,
            timer,
        )?;
        record_one_shot_join_chunk_metrics(
            node_id,
            node,
            node.spec.build_side,
            probe_side(node.spec.build_side),
            &chunk_metrics,
            execution.metrics(),
        );
        return Ok(output);
    }
    materialize_compiled_indices_output(
        execution,
        node,
        CompiledJoinMaterializationRequest {
            node_id,
            execution_strategy: "bounded_one_shot_equi_join",
            key_count: node.spec.keys.len(),
            residual_predicate: false,
            build_side: node.spec.build_side,
            probe_side: probe_side(node.spec.build_side),
            left,
            right,
            indices,
            mixed_size_data_bytes: 0,
            filtered_probe_state_bytes: 0,
            expected_output_rows: output_rows,
            gather_map_bytes: index_bytes,
            timer,
            one_shot_index_bytes: Some(index_bytes),
        },
    )
}

pub(super) fn execute_compiled_one_shot_equi_join_probe_from_stream(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    build: BoundedCompiledPreparedJoinBuildState,
    stream: BoundedJoinLeafLocalParquetSourceStream,
) -> Result<BoundedCompiledJoinProbeState> {
    validate_one_shot_equi_join(node)?;

    let probe_side = probe_side(build.side);
    let probe_columns = compiled_join_input_columns(node, build.strategy, probe_side)?;
    let mut outputs = Vec::new();
    let mut summary = StreamedOneShotJoinSummary::new();
    let stream_result = stream_bounded_join_leaf_local_parquet_source_serial(execution, stream, |input, execution| {
        let probe = prepare_bounded_join_side_input_from_resident_frame(execution, input, &probe_columns, probe_side)?;
        let output = materialize_compiled_one_shot_equi_join_probe_chunk(
            execution,
            node_id,
            node,
            build.side,
            &build.input,
            probe,
            &mut summary,
        )?;
        outputs.push(output);
        Ok(())
    });
    match stream_result {
        Ok(()) => Ok(BoundedCompiledJoinProbeState::OneShotStreamed { build, outputs, summary }),
        Err(error) => {
            release_bounded_join_side_frame(build.input);
            Err(error)
        },
    }
}

pub(super) fn execute_compiled_one_shot_equi_join_probe_from_frame_stream(
    execution: &mut BoundedExecutionAttempt<'_>,
    node: &JoinNode,
    build: BoundedCompiledPreparedJoinBuildState,
    inputs: ResidentFrameStreamPayload,
) -> Result<BoundedCompiledJoinProbeState> {
    validate_one_shot_equi_join(node)?;
    let probe_side = probe_side(build.side);
    let probe_columns = match compiled_join_input_columns(node, build.strategy, probe_side) {
        Ok(columns) => columns,
        Err(error) => {
            inputs.release_all(execution);
            release_bounded_join_side_frame(build.input);
            return Err(error);
        },
    };
    Ok(BoundedCompiledJoinProbeState::OneShotLazyStream(OneShotEquiJoinFrameStreamProbeState {
        build,
        inputs,
        probe_columns,
    }))
}

pub(super) fn one_shot_equi_join_frame_stream_probe_to_payload(
    node_id: NodeId,
    node: &JoinNode,
    probe: OneShotEquiJoinFrameStreamProbeState,
) -> ResidentFrameStreamPayload {
    let max_frame_bytes_hint = probe.inputs.max_frame_bytes_hint();
    let total_bytes_hint = probe.inputs.total_bytes_hint();
    let aggregate_num_rows = probe.inputs.num_rows();
    let num_columns = node.spec.output_projection.len();
    ResidentFrameStreamPayload::from_producer(
        max_frame_bytes_hint,
        total_bytes_hint,
        aggregate_num_rows,
        num_columns,
        Box::new(OneShotEquiJoinProbeStreamSource {
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
    node_id: NodeId,
    node: JoinNode,
    build: Option<BoundedCompiledPreparedJoinBuildState>,
    inputs: Option<ResidentFrameStreamPayload>,
    probe_columns: Vec<String>,
    summary: StreamedOneShotJoinSummary,
    metrics_recorded: bool,
}

impl OneShotEquiJoinProbeStreamSource {
    fn record_success_metrics(&mut self, metrics: &ExecutionMetrics) {
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
    fn next_frame(&mut self, execution: &mut BoundedExecutionAttempt<'_>) -> Option<Result<ResidentGpuFrame>> {
        let inputs = self.inputs.as_mut()?;
        let mut input = match inputs.next_yielded_frame(execution) {
            Some(Ok(input)) => input,
            Some(Err(error)) => return Some(Err(error)),
            None => {
                self.record_success_metrics(execution.metrics());
                self.release(execution);
                return None;
            },
        };
        let input = input.take_for_release();
        let build = self.build.as_ref().expect("one-shot join build side released before stream end");
        let probe_side = probe_side(build.side);
        let probe = match prepare_bounded_join_side_input_from_resident_frame(
            execution,
            input,
            &self.probe_columns,
            probe_side,
        ) {
            Ok(probe) => probe,
            Err(error) => return Some(Err(error)),
        };
        Some(materialize_compiled_one_shot_equi_join_probe_chunk(
            execution,
            self.node_id,
            &self.node,
            build.side,
            &build.input,
            probe,
            &mut self.summary,
        ))
    }

    fn release(&mut self, execution: &mut BoundedExecutionAttempt<'_>) {
        if let Some(input) = self.inputs.take() {
            input.release_all(execution);
        }
        if let Some(build) = self.build.take() {
            release_bounded_join_side_frame(build.input);
        }
    }
}

pub(super) fn materialize_compiled_one_shot_equi_join_probe_chunk(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    build_side: JoinSide,
    build: &BoundedJoinSideFrame,
    probe: BoundedJoinSideFrame,
    summary: &mut StreamedOneShotJoinSummary,
) -> Result<ResidentGpuFrame> {
    let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
    let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
    let overrides_lease = execution_overrides(execution.context())?;
    let overrides = overrides_lease.overrides();
    let timer = Instant::now();
    let (left_input, right_input) = match build_side {
        JoinSide::Left => (build, &probe),
        JoinSide::Right => (&probe, build),
    };
    let preflight_indices = match materialize_compiled_keyed_equality_join_indices_with_preflight(
        execution,
        node,
        left_input,
        right_input,
        build_side,
        &left_keys,
        &right_keys,
        &overrides,
    ) {
        Ok(indices) => indices,
        Err(error) => {
            release_bounded_join_side_frame(probe);
            return Err(error);
        },
    };
    let CompiledEqualityJoinPreflightIndices { indices, index_bytes, output_rows } = preflight_indices;
    let probe_rows = probe.rows;
    let probe_bytes = probe.bytes;
    if node.spec.residual_predicate.is_some() {
        let build_alias = alias_compiled_one_shot_build_chunk(build);
        let probe_side = probe_side(build_side);
        let (left, right) = orient_prepared_inputs(build_side, build_alias, probe_side, probe)?;
        let (output, chunk_metrics) = materialize_compiled_one_shot_residual_output(
            execution,
            node_id,
            node,
            left,
            right,
            indices,
            output_rows,
            index_bytes,
            timer,
        )?;
        summary.record_chunk_metrics(&chunk_metrics);
        return Ok(output);
    }
    let (left, right) = match build_side {
        JoinSide::Left => (build, &probe),
        JoinSide::Right => (&probe, build),
    };
    let output_reservation_bytes = match indices.output_upper_bound_size_bytes(
        execution,
        node_id,
        &left.frame,
        &right.frame,
        output_rows,
        &node.spec.output_projection,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Err(error);
        },
    };
    let output = match indices.materialize_output(
        &left.frame,
        &right.frame,
        &overrides,
        &node.spec.output_projection,
        node.spec.join_type,
        node.spec.datafusion_order_preserved_side,
    ) {
        Ok(output) => output,
        Err(error) => {
            return Err(error);
        },
    };
    execution.metrics().add_operator_time(timer.elapsed());
    let output_bytes = match observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), execution.metrics()) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Err(error);
        },
    };
    summary.record_chunk(
        probe_rows,
        probe_bytes,
        index_bytes,
        output_reservation_bytes,
        output_reservation_bytes,
        output_rows,
        output_bytes,
    );
    Ok(ResidentGpuFrame::with_device_bytes(GpuDataFrame::new(output), output_bytes))
}

fn alias_compiled_one_shot_build_chunk(build: &BoundedJoinSideFrame) -> BoundedJoinSideFrame {
    BoundedJoinSideFrame { frame: build.frame.clone(), rows: build.rows, bytes: build.bytes }
}
