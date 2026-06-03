use std::collections::{BTreeMap, BTreeSet, VecDeque};

use super::{
    common::{
        BoundedJoinSideFrame, bounded_join_request, join_side_output_upper_bound_bytes,
        prepare_bounded_join_side_input_from_resident_frame, prepare_bounded_join_side_input_from_shared_frame,
        projected_output_columns_for_retained_sizing, reconcile_join_output_upper_bound,
        release_bounded_join_side_frame,
    },
    *,
};
use crate::{
    capability::{
        KeylessConditionalJoinKind, LARGE_KEYED_HASH_BUILD_BYTES_THRESHOLD, LARGE_KEYED_HASH_BUILD_ROW_THRESHOLD,
        LARGE_KEYED_HASH_PROBE_BYTES_THRESHOLD, SMALL_REUSABLE_HASH_BUILD_BYTES_THRESHOLD,
        SMALL_REUSABLE_HASH_BUILD_ROW_THRESHOLD, keyed_filtered_semi_anti_join_capable,
    },
    exec::bounded::{
        compiled::{ResidentFrameStreamPayload, ResidentFrameStreamSource},
        strict::CompiledJoinExecutionStrategy,
    },
    expr::{ColumnQualifier, ColumnRef},
};

mod cross;
mod filtered;
mod keyless;
mod mixed;
mod one_shot;
mod partitioned;
mod prepared;
mod residual;
mod streamed;
pub(in crate::exec) mod transaction;

use cross::*;
use filtered::*;
use keyless::*;
use mixed::*;
use one_shot::*;
use partitioned::*;
use prepared::*;
use residual::*;
use streamed::*;
use transaction::*;

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
    HashLazyStream(RootHashJoinFrameStreamProbeState),
    Filtered(BoundedCompiledFilteredJoinProbeState),
    FilteredStreamed {
        build: BoundedCompiledFilteredJoinBuildState,
        outputs: Vec<ResidentGpuFrame>,
    },
    Prepared(BoundedCompiledPreparedJoinProbeState),
    MixedStreamed {
        build: BoundedCompiledPreparedJoinBuildState,
        outputs: Vec<ResidentGpuFrame>,
    },
    OneShotStreamed {
        build: BoundedCompiledPreparedJoinBuildState,
        outputs: Vec<ResidentGpuFrame>,
        summary: StreamedOneShotJoinSummary,
    },
    OneShotLazyStream(OneShotEquiJoinFrameStreamProbeState),
    PartitionedStreamed {
        build_side: JoinSide,
        build_rows: u64,
        build_bytes: u64,
        outputs: Vec<ResidentGpuFrame>,
        summary: StreamedPartitionedJoinSummary,
    },
    PartitionedLazyStream(PartitionedHashEquiJoinFrameStreamProbeState),
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

pub(in crate::exec::bounded) struct RootHashJoinFrameStreamProbeState {
    build: BoundedRootHashJoinBuildState,
    inputs: ResidentFrameStreamPayload,
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

pub(in crate::exec::bounded) struct StreamedPartitionedJoinSummary {
    partition_count: usize,
    partition_target_bytes: u64,
    build_partitioned_bytes: u64,
    probe_partition_batch_count: usize,
    probe_partitioned_bytes: u64,
    probe_rows: u64,
    probe_bytes: u64,
    index_bytes: u64,
    mixed_size_data_bytes: u64,
    output_reservation_bytes: u64,
    output_rows: u64,
    output_bytes: u64,
}

struct PartitionedJoinSide {
    frame: DataFrame,
    reservation: ActiveReservation,
    offsets: Vec<usize>,
}

pub(in crate::exec::bounded) struct PartitionedHashEquiJoinFrameStreamProbeState {
    build_side: JoinSide,
    build_rows: u64,
    build_bytes: u64,
    build_partitioned: PartitionedJoinSide,
    build_partitioned_bytes: u64,
    inputs: ResidentFrameStreamPayload,
    partition_count: usize,
    partition_target_bytes: u64,
    probe_columns: Vec<String>,
    probe_keys: Vec<String>,
}

struct PartitionedJoinPartitionOutput {
    frame: ResidentGpuFrame,
    index_bytes: u64,
    mixed_size_data_bytes: u64,
    output_reservation_bytes: u64,
    output_bytes: u64,
    output_rows: u64,
}

pub(in crate::exec::bounded) enum BoundedCompiledJoinMaterializeOutput {
    Frame(Box<ResidentGpuFrame>),
    Stream(Vec<ResidentGpuFrame>),
    LazyStream(ResidentFrameStreamPayload),
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

impl StreamedPartitionedJoinSummary {
    fn new(partition_count: usize, partition_target_bytes: u64, build_partitioned_bytes: u64) -> Self {
        Self {
            partition_count,
            partition_target_bytes,
            build_partitioned_bytes,
            probe_partition_batch_count: 0,
            probe_partitioned_bytes: 0,
            probe_rows: 0,
            probe_bytes: 0,
            index_bytes: 0,
            mixed_size_data_bytes: 0,
            output_reservation_bytes: 0,
            output_rows: 0,
            output_bytes: 0,
        }
    }

    fn record_probe_partitioned_chunk(&mut self, probe_rows: u64, probe_bytes: u64, probe_partitioned_bytes: u64) {
        self.probe_partition_batch_count = self.probe_partition_batch_count.saturating_add(self.partition_count);
        self.probe_partitioned_bytes = self.probe_partitioned_bytes.saturating_add(probe_partitioned_bytes);
        self.probe_rows = self.probe_rows.saturating_add(probe_rows);
        self.probe_bytes = self.probe_bytes.saturating_add(probe_bytes);
    }

    fn record_output(&mut self, output: &PartitionedJoinPartitionOutput) {
        self.index_bytes = self.index_bytes.saturating_add(output.index_bytes);
        self.mixed_size_data_bytes = self.mixed_size_data_bytes.saturating_add(output.mixed_size_data_bytes);
        self.output_reservation_bytes = self.output_reservation_bytes.saturating_add(output.output_reservation_bytes);
        self.output_rows = self.output_rows.saturating_add(output.output_rows);
        self.output_bytes = self.output_bytes.saturating_add(output.output_bytes);
    }
}

fn root_hash_join_frame_stream_probe_to_payload(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    probe: RootHashJoinFrameStreamProbeState,
) -> ResidentFrameStreamPayload {
    let max_per_frame_reservation_bytes = probe.inputs.max_per_frame_reservation_bytes();
    let aggregate_reservation_bytes = probe.inputs.aggregate_reservation_bytes();
    let aggregate_num_rows = probe.inputs.num_rows();
    let num_columns = node.spec.output_projection.len();
    let source = RootHashJoinProbeStreamSource {
        context: context.clone(),
        node_id,
        node: node.clone(),
        build: Some(probe.build),
        inputs: Some(probe.inputs),
        pending_outputs: VecDeque::new(),
    };
    ResidentFrameStreamPayload::from_producer(
        max_per_frame_reservation_bytes,
        aggregate_reservation_bytes,
        aggregate_num_rows,
        num_columns,
        Vec::new(),
        Box::new(source),
    )
}

struct RootHashJoinProbeStreamSource {
    context: ExecutionContext,
    node_id: NodeId,
    node: JoinNode,
    build: Option<BoundedRootHashJoinBuildState>,
    inputs: Option<ResidentFrameStreamPayload>,
    pending_outputs: VecDeque<ResidentGpuFrame>,
}

impl ResidentFrameStreamSource for RootHashJoinProbeStreamSource {
    fn next_frame(
        &mut self,
        reservations: &mut ReservationTracker,
        metrics: &mut QueryMetrics,
    ) -> Option<Result<ResidentGpuFrame>> {
        loop {
            if let Some(output) = self.pending_outputs.pop_front() {
                return Some(Ok(output));
            }
            let inputs = self.inputs.as_mut()?;
            let mut input = match inputs.next_yielded_frame(reservations, metrics) {
                Some(Ok(input)) => input,
                Some(Err(error)) => return Some(Err(error)),
                None => {
                    self.release(reservations, metrics);
                    return None;
                },
            };
            let input = input.take_for_release();
            let build = self.build.as_ref().expect("root hash join build state released before stream end");
            let outputs = match execute_bounded_root_hash_join_probe_chunk_stream_from_resident_frame(
                &self.context,
                self.node_id,
                &self.node,
                build,
                input,
                reservations,
                metrics,
            ) {
                Ok(outputs) => outputs,
                Err(error) => return Some(Err(error)),
            };
            self.pending_outputs.extend(outputs);
        }
    }

    fn release(&mut self, reservations: &mut ReservationTracker, metrics: &mut QueryMetrics) {
        while let Some(output) = self.pending_outputs.pop_front() {
            reservations.release(output.reservation, metrics);
            drop(output.frame);
        }
        if let Some(input) = self.inputs.take() {
            input.release_all(reservations, metrics);
        }
        if let Some(build) = self.build.take() {
            release_bounded_root_hash_join_build_state(build, reservations, metrics);
        }
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
        | CompiledJoinExecutionStrategy::PartitionedHashEqui
        | CompiledJoinExecutionStrategy::PartitionedMixedKeyedResidual
        | CompiledJoinExecutionStrategy::KeylessConditional { .. }
        | CompiledJoinExecutionStrategy::KeylessOuterCross
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
        | CompiledJoinExecutionStrategy::PartitionedHashEqui
        | CompiledJoinExecutionStrategy::PartitionedMixedKeyedResidual
        | CompiledJoinExecutionStrategy::KeylessConditional { .. }
        | CompiledJoinExecutionStrategy::KeylessOuterCross
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
    if stream.chunk_count() == 0 {
        let input = materialize_bounded_join_leaf_local_parquet_source_stream(context, stream, reservations, metrics)?;
        return execute_bounded_compiled_join_probe_from_resident_frame(
            context,
            node_id,
            node,
            build,
            input,
            reservations,
            metrics,
        );
    }

    match build {
        BoundedCompiledJoinBuildState::Hash(build) => {
            let mut outputs = Vec::new();
            let stream_result = stream_bounded_join_leaf_local_parquet_source(
                context,
                stream,
                reservations,
                metrics,
                |input, reservations, metrics| {
                    let chunk_outputs = execute_bounded_root_hash_join_probe_chunk_stream_from_resident_frame(
                        context,
                        node_id,
                        node,
                        &build,
                        input,
                        reservations,
                        metrics,
                    )?;
                    outputs.extend(chunk_outputs);
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
            if matches!(
                build.strategy,
                CompiledJoinExecutionStrategy::OneShotEqui | CompiledJoinExecutionStrategy::PartitionedHashEqui
            ) && node.spec.residual_predicate.is_none() =>
        {
            if build.strategy == CompiledJoinExecutionStrategy::PartitionedHashEqui {
                if partitioned_streaming_probe_supported(node) {
                    return execute_compiled_partitioned_hash_equi_join_probe_from_stream(
                        context,
                        node_id,
                        node,
                        build,
                        stream,
                        reservations,
                        metrics,
                    );
                }
                let input =
                    materialize_bounded_join_leaf_local_parquet_source_stream(context, stream, reservations, metrics)?;
                return execute_bounded_compiled_join_probe_from_resident_frame(
                    context,
                    node_id,
                    node,
                    BoundedCompiledJoinBuildState::Prepared(build),
                    input,
                    reservations,
                    metrics,
                );
            }
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
        BoundedCompiledJoinBuildState::Prepared(build)
            if build.strategy == CompiledJoinExecutionStrategy::MixedKeyedResidual
                && node.spec.join_type == JoinType::Inner =>
        {
            execute_compiled_mixed_join_probe_from_stream(context, node_id, node, build, stream, reservations, metrics)
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
    mut inputs: ResidentFrameStreamPayload,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedCompiledJoinProbeState> {
    match build {
        BoundedCompiledJoinBuildState::Hash(build) => {
            Ok(BoundedCompiledJoinProbeState::HashLazyStream(RootHashJoinFrameStreamProbeState { build, inputs }))
        },
        BoundedCompiledJoinBuildState::Filtered(build) => {
            let mut outputs = Vec::new();
            while let Some(input) = inputs.next_yielded_frame(reservations, metrics) {
                let mut input = match input {
                    Ok(input) => input,
                    Err(error) => {
                        inputs.release_all(reservations, metrics);
                        release_streamed_join_outputs_reverse(reservations, outputs, metrics);
                        release_compiled_filtered_join_build_state(build, reservations, metrics);
                        return Err(error);
                    },
                };
                let input = input.take_for_release();
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
                        inputs.release_all(reservations, metrics);
                        release_streamed_join_outputs_reverse(reservations, outputs, metrics);
                        release_compiled_filtered_join_build_state(build, reservations, metrics);
                        return Err(error);
                    },
                }
            }
            inputs.release_producer_reservations(reservations, metrics);
            Ok(BoundedCompiledJoinProbeState::FilteredStreamed { build, outputs })
        },
        BoundedCompiledJoinBuildState::Prepared(build)
            if build.strategy == CompiledJoinExecutionStrategy::PartitionedHashEqui
                && node.spec.residual_predicate.is_none() =>
        {
            if inputs.max_per_frame_reservation_bytes() <= partitioned_join_target_bytes_for_context(context) / 2 {
                execute_compiled_one_shot_equi_join_probe_from_frame_stream(
                    context,
                    node_id,
                    node,
                    build,
                    inputs,
                    reservations,
                    metrics,
                )
            } else {
                execute_compiled_partitioned_hash_equi_join_probe_from_frame_stream(
                    context,
                    node_id,
                    node,
                    build,
                    inputs,
                    reservations,
                    metrics,
                )
            }
        },
        other => {
            let inputs = match inputs.into_remaining_frames_owned(reservations, metrics) {
                Ok(inputs) => inputs,
                Err(error) => {
                    release_bounded_compiled_join_build_state(other, reservations, metrics);
                    return Err(error);
                },
            };
            let input = match concatenate_streamed_hash_join_outputs(context, node_id, inputs, reservations, metrics) {
                Ok(input) => input,
                Err(error) => {
                    release_bounded_compiled_join_build_state(other, reservations, metrics);
                    return Err(error);
                },
            };
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
        BoundedCompiledJoinProbeState::HashLazyStream(probe) => {
            let stream = root_hash_join_frame_stream_probe_to_payload(context, node_id, node, probe);
            let outputs = stream.into_remaining_frames_owned(reservations, metrics)?;
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
        BoundedCompiledJoinProbeState::MixedStreamed { build, outputs } => {
            release_bounded_join_side_frame(build.input, reservations, metrics);
            concatenate_streamed_hash_join_outputs(context, node_id, outputs, reservations, metrics)
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
        BoundedCompiledJoinProbeState::OneShotLazyStream(probe) => {
            let stream = one_shot_equi_join_frame_stream_probe_to_payload(context, node_id, node, probe);
            let outputs = stream.into_remaining_frames_owned(reservations, metrics)?;
            concatenate_streamed_hash_join_outputs(context, node_id, outputs, reservations, metrics)
        },
        BoundedCompiledJoinProbeState::PartitionedStreamed {
            build_side,
            build_rows,
            build_bytes,
            outputs,
            summary,
        } => execute_compiled_streamed_partitioned_hash_join_materialize(
            context,
            node_id,
            node,
            build_side,
            build_rows,
            build_bytes,
            outputs,
            summary,
            reservations,
            metrics,
        ),
        BoundedCompiledJoinProbeState::PartitionedLazyStream(probe) => {
            let stream = partitioned_hash_equi_join_frame_stream_probe_to_payload(context, node_id, node, probe);
            let outputs = stream.into_remaining_frames_owned(reservations, metrics)?;
            concatenate_streamed_hash_join_outputs(context, node_id, outputs, reservations, metrics)
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
        BoundedCompiledJoinProbeState::HashStreamed { build, outputs } => {
            release_bounded_root_hash_join_build_state(build, reservations, metrics);
            Ok(BoundedCompiledJoinMaterializeOutput::Stream(outputs))
        },
        BoundedCompiledJoinProbeState::Hash(probe) => Ok(BoundedCompiledJoinMaterializeOutput::Stream(
            execute_bounded_root_hash_join_materialize_preserving_stream(
                context,
                node_id,
                node,
                probe,
                reservations,
                metrics,
            )?,
        )),
        BoundedCompiledJoinProbeState::HashLazyStream(probe) => Ok(BoundedCompiledJoinMaterializeOutput::LazyStream(
            root_hash_join_frame_stream_probe_to_payload(context, node_id, node, probe),
        )),
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
        BoundedCompiledJoinProbeState::MixedStreamed { build, outputs } => {
            release_bounded_join_side_frame(build.input, reservations, metrics);
            Ok(BoundedCompiledJoinMaterializeOutput::Stream(outputs))
        },
        BoundedCompiledJoinProbeState::OneShotLazyStream(probe) => {
            Ok(BoundedCompiledJoinMaterializeOutput::LazyStream(one_shot_equi_join_frame_stream_probe_to_payload(
                context, node_id, node, probe,
            )))
        },
        BoundedCompiledJoinProbeState::PartitionedStreamed {
            build_side,
            build_rows,
            build_bytes,
            outputs,
            summary,
        } => {
            record_streamed_partitioned_hash_join_metrics(
                node_id,
                node,
                build_side,
                probe_side(build_side),
                build_rows,
                build_bytes,
                &summary,
                metrics,
            );
            Ok(BoundedCompiledJoinMaterializeOutput::Stream(outputs))
        },
        BoundedCompiledJoinProbeState::PartitionedLazyStream(probe) => {
            Ok(BoundedCompiledJoinMaterializeOutput::LazyStream(
                partitioned_hash_equi_join_frame_stream_probe_to_payload(context, node_id, node, probe),
            ))
        },
        BoundedCompiledJoinProbeState::Prepared(probe)
            if probe.strategy == CompiledJoinExecutionStrategy::PartitionedHashEqui =>
        {
            execute_compiled_partitioned_hash_equi_join_materialize_preserving_stream(
                context,
                node_id,
                node,
                probe.left,
                probe.right,
                reservations,
                metrics,
            )
        },
        BoundedCompiledJoinProbeState::Prepared(probe)
            if probe.strategy == CompiledJoinExecutionStrategy::PartitionedMixedKeyedResidual =>
        {
            execute_compiled_partitioned_mixed_keyed_residual_join_materialize_preserving_stream(
                context,
                node_id,
                node,
                probe.left,
                probe.right,
                reservations,
                metrics,
            )
        },
        other => execute_bounded_compiled_join_materialize(context, node_id, node, other, reservations, metrics)
            .map(|frame| BoundedCompiledJoinMaterializeOutput::Frame(Box::new(frame))),
    }
}

pub(in crate::exec::bounded) fn release_bounded_compiled_join_build_state(
    build: BoundedCompiledJoinBuildState,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) {
    match build {
        BoundedCompiledJoinBuildState::Hash(build) => {
            release_bounded_root_hash_join_build_state(build, reservations, metrics);
        },
        BoundedCompiledJoinBuildState::Filtered(build) => {
            release_compiled_filtered_join_build_state(build, reservations, metrics);
        },
        BoundedCompiledJoinBuildState::Prepared(build) => {
            let BoundedCompiledPreparedJoinBuildState { input, .. } = build;
            release_bounded_join_side_frame(input, reservations, metrics);
        },
    }
}

pub(in crate::exec::bounded) fn release_bounded_compiled_join_probe_state(
    probe: BoundedCompiledJoinProbeState,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) {
    match probe {
        BoundedCompiledJoinProbeState::Hash(probe) => {
            release_bounded_root_hash_join_probe_state(probe, reservations, metrics);
        },
        BoundedCompiledJoinProbeState::HashStreamed { build, outputs } => {
            release_streamed_join_outputs_reverse(reservations, outputs, metrics);
            release_bounded_root_hash_join_build_state(build, reservations, metrics);
        },
        BoundedCompiledJoinProbeState::HashLazyStream(probe) => {
            probe.inputs.release_all(reservations, metrics);
            release_bounded_root_hash_join_build_state(probe.build, reservations, metrics);
        },
        BoundedCompiledJoinProbeState::Filtered(probe) => {
            release_compiled_filtered_join_probe_state(probe, reservations, metrics);
        },
        BoundedCompiledJoinProbeState::FilteredStreamed { build, outputs } => {
            release_streamed_join_outputs_reverse(reservations, outputs, metrics);
            release_compiled_filtered_join_build_state(build, reservations, metrics);
        },
        BoundedCompiledJoinProbeState::Prepared(probe) => {
            let BoundedCompiledPreparedJoinProbeState { left, right, .. } = probe;
            release_bounded_join_side_frame(left, reservations, metrics);
            release_bounded_join_side_frame(right, reservations, metrics);
        },
        BoundedCompiledJoinProbeState::MixedStreamed { build, outputs } => {
            release_streamed_join_outputs_reverse(reservations, outputs, metrics);
            release_bounded_join_side_frame(build.input, reservations, metrics);
        },
        BoundedCompiledJoinProbeState::OneShotStreamed { build, outputs, .. } => {
            release_streamed_join_outputs_reverse(reservations, outputs, metrics);
            release_bounded_compiled_join_build_state(
                BoundedCompiledJoinBuildState::Prepared(build),
                reservations,
                metrics,
            );
        },
        BoundedCompiledJoinProbeState::OneShotLazyStream(probe) => {
            probe.inputs.release_all(reservations, metrics);
            release_bounded_join_side_frame(probe.build.input, reservations, metrics);
        },
        BoundedCompiledJoinProbeState::PartitionedStreamed { outputs, .. } => {
            release_streamed_join_outputs_reverse(reservations, outputs, metrics);
        },
        BoundedCompiledJoinProbeState::PartitionedLazyStream(probe) => {
            probe.inputs.release_all(reservations, metrics);
            release_partitioned_join_side(probe.build_partitioned, reservations, metrics);
        },
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
            "nexus_query_engine.bounded.join.streamed_hash_output",
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
                "nexus_query_engine.bounded.join.streamed_hash_output_concat",
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
        CompiledJoinExecutionStrategy::KeylessOuterCross => {
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
        CompiledJoinExecutionStrategy::OneShotEqui | CompiledJoinExecutionStrategy::PartitionedHashEqui => {
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
        CompiledJoinExecutionStrategy::PartitionedMixedKeyedResidual => {
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
                "nexus_query_engine.bounded.join.mixed_size_data_bytes",
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
                "nexus_query_engine.bounded.join.mixed_gather_map_reservation",
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
                        "nexus_query_engine.bounded.join.gather_map_bytes",
                        "failed to observe compiled bounded join left gather-map memory",
                        error,
                    )
                    .with_fact("join_type", node.spec.join_type)
                })?;
                let right_bytes = right.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
                    QueryError::from_cudf(
                        "nexus_query_engine.bounded.join.gather_map_bytes",
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
                        "nexus_query_engine.bounded.join.gather_map_bytes",
                        "failed to observe compiled bounded join gather-map memory",
                        error,
                    )
                    .with_fact("join_type", node.spec.join_type)
                })
            },
        }
    }

    fn output_upper_bound_size_bytes(
        &self,
        context: &ExecutionContext,
        node_id: NodeId,
        left: &DataFrame,
        right: &DataFrame,
        expected_output_rows: u64,
        projection: &[JoinOutputColumn],
        reservations: &mut ReservationTracker,
        metrics: &mut QueryMetrics,
    ) -> Result<u64> {
        match self {
            Self::Pair { .. } => {
                let left_projection = projected_output_columns_for_retained_sizing(projection, JoinSide::Left);
                let right_projection = projected_output_columns_for_retained_sizing(projection, JoinSide::Right);
                let left_bytes = join_side_output_upper_bound_bytes(
                    context,
                    node_id,
                    left,
                    expected_output_rows,
                    &left_projection,
                    JoinSide::Left,
                    reservations,
                    metrics,
                )?;
                let right_bytes = join_side_output_upper_bound_bytes(
                    context,
                    node_id,
                    right,
                    expected_output_rows,
                    &right_projection,
                    JoinSide::Right,
                    reservations,
                    metrics,
                )?;
                Ok(left_bytes.saturating_add(right_bytes))
            },
            Self::LeftOnly(_) => {
                let left_projection = projected_output_columns_for_retained_sizing(projection, JoinSide::Left);
                join_side_output_upper_bound_bytes(
                    context,
                    node_id,
                    left,
                    expected_output_rows,
                    &left_projection,
                    JoinSide::Left,
                    reservations,
                    metrics,
                )
            },
        }
    }

    fn materialize_output(
        &self,
        left: &DataFrame,
        right: &DataFrame,
        overrides: &ExecutionOverrides<'_>,
        projection: &[JoinOutputColumn],
        join_type: JoinType,
    ) -> Result<DataFrame> {
        match self {
            Self::Pair { left: left_indices, right: right_indices } => materialize_join_output_projection(
                left,
                right,
                left_indices,
                right_indices,
                overrides,
                projection,
                join_type,
            ),
            Self::LeftOnly(indices) => {
                materialize_left_only_join_output_projection(left, indices, overrides, projection, join_type)
            },
        }
    }
}
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
    let mut txn = JoinMaterializationTxn::with_sides(left, right);
    txn.set_gather_map_reservation(gather_map_reservation);
    let overrides = execution_overrides(context);
    txn.set_indices(indices);
    let gather_map_bytes = match txn.indices().device_allocated_size_bytes(node) {
        Ok(bytes) => bytes,
        Err(error) => {
            txn.rollback(reservations, metrics);
            return Err(error);
        },
    };
    let (left, right) = txn.sides();
    super::common::trace_preflight_entry(
        "materialize_compiled_indices_output",
        execution_strategy,
        node_id,
        reservations,
        gather_map_bytes,
        expected_output_rows,
        left.rows,
        left.bytes,
        right.rows,
        right.bytes,
    );
    let output_reservation_bytes = match txn.indices().output_upper_bound_size_bytes(
        context,
        node_id,
        &left.frame,
        &right.frame,
        expected_output_rows,
        &node.spec.output_projection,
        reservations,
        metrics,
    ) {
        Ok(bytes) => {
            super::common::trace_preflight_success("materialize_compiled_indices_output", node_id, reservations, bytes);
            bytes
        },
        Err(error) => {
            super::common::trace_preflight_failed("materialize_compiled_indices_output", node_id, reservations, &error);
            txn.rollback(reservations, metrics);
            return Err(error);
        },
    };
    if let Some(index_bytes) = one_shot_index_bytes {
        metrics.record_one_shot_join();
        metrics.record_one_shot_join_index_bytes(index_bytes);
        metrics.record_one_shot_join_materialization_bytes(output_reservation_bytes);
    }
    let output_reservation = match reservations.try_reserve(
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
            txn.rollback(reservations, metrics);
            return Err(error);
        },
    };
    txn.set_output_reservation(output_reservation);
    if let Err(error) = reservations.reclaim_source_cache_for_external_working_set(reservations.budget(), metrics) {
        txn.rollback(reservations, metrics);
        return Err(error);
    }
    let (left, right) = txn.sides();
    let output = match txn.indices().materialize_output(
        &left.frame,
        &right.frame,
        &overrides,
        &node.spec.output_projection,
        node.spec.join_type,
    ) {
        Ok(output) => output,
        Err(error) => {
            txn.rollback(reservations, metrics);
            return Err(error);
        },
    };
    metrics.operator_time += timer.elapsed();
    let output_rows = output.num_rows() as u64;
    debug_assert_eq!(output_rows, expected_output_rows);
    let output_bytes = match observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(output);
            txn.rollback(reservations, metrics);
            return Err(error);
        },
    };
    if one_shot_index_bytes.is_some() {
        metrics.record_one_shot_join_output_bytes(output_bytes);
    }
    if let Err(error) =
        reconcile_join_output_upper_bound(node_id, txn.output_reservation_mut(), output_bytes, reservations, metrics)
    {
        drop(output);
        txn.rollback(reservations, metrics);
        return Err(error);
    }
    let (left, right) = txn.sides();
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
    txn.release_gather_map_reservation(reservations, metrics);
    Ok(txn.commit_frame(GpuDataFrame::new(output), reservations, metrics))
}
