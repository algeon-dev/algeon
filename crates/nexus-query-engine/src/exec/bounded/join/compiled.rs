use std::collections::{BTreeMap, BTreeSet, VecDeque};

use super::{
    common::{
        BoundedJoinSideFrame, bounded_join_request, join_side_output_upper_bound_bytes,
        prepare_bounded_join_side_input_from_resident_frame, prepare_bounded_join_side_input_from_shared_frame,
        projected_output_columns_for_retained_sizing, reconcile_join_output_upper_bound,
        release_bounded_join_side_frame, reserve_full_join_size_preflight_workspace,
    },
    *,
};
use crate::{
    admission::CompiledJoinExecutionStrategy,
    capability::{KeyedHashJoinStrategyOptions, KeylessConditionalJoinKind, keyed_filtered_semi_anti_join_capable},
    exec::bounded::compiled::{ResidentFrameStreamPayload, ResidentFrameStreamSource},
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
pub(in crate::exec) use transaction::JoinMaterializationTxn;

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

impl BoundedCompiledJoinProbeState {
    pub(in crate::exec::bounded) fn materialize_admission_bytes(&self, stream_target_bytes: u64) -> u64 {
        match self {
            Self::Hash(probe) => probe.materialize_admission_bytes(stream_target_bytes),
            Self::HashStreamed { outputs, .. }
            | Self::FilteredStreamed { outputs, .. }
            | Self::MixedStreamed { outputs, .. }
            | Self::OneShotStreamed { outputs, .. }
            | Self::PartitionedStreamed { outputs, .. } => {
                outputs.iter().map(|output| output.reservation.bytes()).max().unwrap_or(0)
            },
            Self::HashLazyStream(probe) => probe.inputs.max_per_frame_reservation_bytes(),
            Self::OneShotLazyStream(probe) => probe.inputs.max_per_frame_reservation_bytes(),
            Self::PartitionedLazyStream(probe) => probe.partition_target_bytes,
            Self::Filtered(_) | Self::Prepared(_) => 0,
        }
    }
}

pub(in crate::exec::bounded) struct BoundedCompiledPreparedJoinBuildState {
    strategy: CompiledJoinExecutionStrategy,
    strategy_options: KeyedHashJoinStrategyOptions,
    side: JoinSide,
    input: BoundedJoinSideFrame,
}

pub(in crate::exec::bounded) struct BoundedCompiledPreparedJoinProbeState {
    strategy: CompiledJoinExecutionStrategy,
    strategy_options: KeyedHashJoinStrategyOptions,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
}

pub(in crate::exec::bounded) struct RootHashJoinFrameStreamProbeState {
    build: BoundedRootHashJoinBuildState,
    inputs: ResidentFrameStreamPayload,
}

fn stateless_streaming_probe_supported(node: &JoinNode) -> bool {
    matches!(node.spec.join_type, JoinType::Inner)
        || (node.spec.join_type == JoinType::Left && node.spec.build_side == JoinSide::Right)
}

pub(in crate::exec::bounded) struct BoundedCompiledFilteredJoinBuildState {
    left_keys: Vec<String>,
    left_columns: Vec<String>,
    build: BoundedJoinSideFrame,
    input_build_rows: u64,
    input_build_bytes: u64,
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
    materialization_bytes: u64,
    output_reservation_bytes: u64,
    output_rows: u64,
    output_bytes: u64,
}

struct OneShotJoinChunkMetrics {
    build_rows: u64,
    build_bytes: u64,
    probe_rows: u64,
    probe_bytes: u64,
    index_bytes: u64,
    materialization_bytes: u64,
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
    strategy_options: KeyedHashJoinStrategyOptions,
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
            materialization_bytes: 0,
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
        materialization_bytes: u64,
        output_reservation_bytes: u64,
        output_rows: u64,
        output_bytes: u64,
    ) {
        self.probe_rows = self.probe_rows.saturating_add(probe_rows);
        self.probe_bytes = self.probe_bytes.saturating_add(probe_bytes);
        self.index_bytes = self.index_bytes.saturating_add(index_bytes);
        self.materialization_bytes = self.materialization_bytes.saturating_add(materialization_bytes);
        self.output_reservation_bytes = self.output_reservation_bytes.saturating_add(output_reservation_bytes);
        self.output_rows = self.output_rows.saturating_add(output_rows);
        self.output_bytes = self.output_bytes.saturating_add(output_bytes);
    }

    fn record_chunk_metrics(&mut self, metrics: &OneShotJoinChunkMetrics) {
        self.record_chunk(
            metrics.probe_rows,
            metrics.probe_bytes,
            metrics.index_bytes,
            metrics.materialization_bytes,
            metrics.output_reservation_bytes,
            metrics.output_rows,
            metrics.output_bytes,
        );
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
    node_id: NodeId,
    node: &JoinNode,
    probe: RootHashJoinFrameStreamProbeState,
) -> ResidentFrameStreamPayload {
    let max_per_frame_reservation_bytes = probe.inputs.max_per_frame_reservation_bytes();
    let aggregate_reservation_bytes = probe.inputs.aggregate_reservation_bytes();
    let aggregate_num_rows = probe.inputs.num_rows();
    let num_columns = node.spec.output_projection.len();
    let source = RootHashJoinProbeStreamSource {
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
    node_id: NodeId,
    node: JoinNode,
    build: Option<BoundedRootHashJoinBuildState>,
    inputs: Option<ResidentFrameStreamPayload>,
    pending_outputs: VecDeque<ResidentGpuFrame>,
}

impl ResidentFrameStreamSource for RootHashJoinProbeStreamSource {
    fn next_frame(&mut self, execution: &mut BoundedExecutionAttempt<'_>) -> Option<Result<ResidentGpuFrame>> {
        loop {
            if let Some(output) = self.pending_outputs.pop_front() {
                return Some(Ok(output));
            }
            let inputs = self.inputs.as_mut()?;
            let mut input = match inputs.next_yielded_frame(execution) {
                Some(Ok(input)) => input,
                Some(Err(error)) => return Some(Err(error)),
                None => {
                    self.release(execution);
                    return None;
                },
            };
            let input = input.take_for_release();
            let build = self.build.as_ref().expect("root hash join build state released before stream end");
            let outputs = match execute_bounded_root_hash_join_probe_chunk_stream_from_resident_frame(
                execution,
                self.node_id,
                &self.node,
                build,
                input,
            ) {
                Ok(outputs) => outputs,
                Err(error) => return Some(Err(error)),
            };
            self.pending_outputs.extend(outputs);
        }
    }

    fn release(&mut self, execution: &mut BoundedExecutionAttempt<'_>) {
        while let Some(output) = self.pending_outputs.pop_front() {
            let ResidentGpuFrame { reservation, .. } = output;
            execution.release(reservation);
        }
        if let Some(input) = self.inputs.take() {
            input.release_all(execution);
        }
        if let Some(build) = self.build.take() {
            release_bounded_root_hash_join_build_state(execution, build);
        }
    }
}

pub(in crate::exec::bounded) fn execute_bounded_compiled_join_build_from_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    strategy: CompiledJoinExecutionStrategy,
    strategy_options: KeyedHashJoinStrategyOptions,
    input: GpuDataFrame,
    input_reservation: &ActiveReservation,
) -> Result<BoundedCompiledJoinBuildState> {
    match strategy {
        CompiledJoinExecutionStrategy::HashEqui => {
            execute_bounded_root_hash_join_build_from_frame(execution, node_id, node, input, input_reservation)
                .map(BoundedCompiledJoinBuildState::Hash)
        },
        CompiledJoinExecutionStrategy::FilteredSemiAnti => {
            execute_compiled_filtered_join_build_from_frame(execution, node_id, node, input, input_reservation)
                .map(BoundedCompiledJoinBuildState::Filtered)
        },
        CompiledJoinExecutionStrategy::OneShotEqui
        | CompiledJoinExecutionStrategy::PartitionedHashEqui
        | CompiledJoinExecutionStrategy::PartitionedMixedKeyedResidual
        | CompiledJoinExecutionStrategy::KeylessConditional { .. }
        | CompiledJoinExecutionStrategy::KeylessOuterCross
        | CompiledJoinExecutionStrategy::ProjectedCross
        | CompiledJoinExecutionStrategy::MixedKeyedResidual => {
            let side = node.spec.build_side;
            let columns = compiled_join_input_columns(node, strategy, side)?;
            let input = prepare_bounded_join_side_input_from_shared_frame(
                execution,
                input,
                input_reservation,
                &columns,
                side,
                node_id,
                node.spec.build_side,
            )?;
            Ok(BoundedCompiledJoinBuildState::Prepared(BoundedCompiledPreparedJoinBuildState {
                strategy,
                strategy_options,
                side,
                input,
            }))
        },
    }
}

pub(in crate::exec::bounded) fn execute_bounded_compiled_join_build_from_resident_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    strategy: CompiledJoinExecutionStrategy,
    strategy_options: KeyedHashJoinStrategyOptions,
    input: ResidentGpuFrame,
) -> Result<BoundedCompiledJoinBuildState> {
    match strategy {
        CompiledJoinExecutionStrategy::HashEqui => {
            execute_bounded_root_hash_join_build_from_resident_frame(execution, node_id, node, input)
                .map(BoundedCompiledJoinBuildState::Hash)
        },
        CompiledJoinExecutionStrategy::FilteredSemiAnti => {
            execute_compiled_filtered_join_build_from_resident_frame(execution, node_id, node, input)
                .map(BoundedCompiledJoinBuildState::Filtered)
        },
        CompiledJoinExecutionStrategy::OneShotEqui
        | CompiledJoinExecutionStrategy::PartitionedHashEqui
        | CompiledJoinExecutionStrategy::PartitionedMixedKeyedResidual
        | CompiledJoinExecutionStrategy::KeylessConditional { .. }
        | CompiledJoinExecutionStrategy::KeylessOuterCross
        | CompiledJoinExecutionStrategy::ProjectedCross
        | CompiledJoinExecutionStrategy::MixedKeyedResidual => {
            let side = node.spec.build_side;
            let columns = compiled_join_input_columns(node, strategy, side)?;
            let input = prepare_bounded_join_side_input_from_resident_frame(
                execution,
                input,
                &columns,
                side,
                node_id,
                node.spec.build_side,
            )?;
            Ok(BoundedCompiledJoinBuildState::Prepared(BoundedCompiledPreparedJoinBuildState {
                strategy,
                strategy_options,
                side,
                input,
            }))
        },
    }
}

pub(in crate::exec::bounded) fn execute_bounded_compiled_join_probe_from_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    build: BoundedCompiledJoinBuildState,
    input: GpuDataFrame,
    input_reservation: &ActiveReservation,
) -> Result<BoundedCompiledJoinProbeState> {
    match build {
        BoundedCompiledJoinBuildState::Hash(build) => {
            execute_bounded_root_hash_join_probe_from_frame(execution, node_id, node, build, input, input_reservation)
                .map(BoundedCompiledJoinProbeState::Hash)
        },
        BoundedCompiledJoinBuildState::Filtered(build) => {
            execute_compiled_filtered_join_probe_from_frame(execution, node_id, node, build, input, input_reservation)
                .map(BoundedCompiledJoinProbeState::Filtered)
        },
        BoundedCompiledJoinBuildState::Prepared(build) => {
            let side = probe_side(build.side);
            let columns = compiled_join_input_columns(node, build.strategy, side)?;
            let probe = prepare_bounded_join_side_input_from_shared_frame(
                execution,
                input,
                input_reservation,
                &columns,
                side,
                node_id,
                node.spec.build_side,
            )?;
            let (left, right) = orient_prepared_inputs(build.side, build.input, side, probe)?;
            Ok(BoundedCompiledJoinProbeState::Prepared(BoundedCompiledPreparedJoinProbeState {
                strategy: build.strategy,
                strategy_options: build.strategy_options,
                left,
                right,
            }))
        },
    }
}

pub(in crate::exec::bounded) fn execute_bounded_compiled_join_probe_from_resident_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    build: BoundedCompiledJoinBuildState,
    input: ResidentGpuFrame,
) -> Result<BoundedCompiledJoinProbeState> {
    match build {
        BoundedCompiledJoinBuildState::Hash(build) => {
            execute_bounded_root_hash_join_probe_from_resident_frame(execution, node_id, node, build, input)
                .map(BoundedCompiledJoinProbeState::Hash)
        },
        BoundedCompiledJoinBuildState::Filtered(build) => {
            execute_compiled_filtered_join_probe_from_resident_frame(execution, node_id, node, build, input)
                .map(BoundedCompiledJoinProbeState::Filtered)
        },
        BoundedCompiledJoinBuildState::Prepared(build) => {
            let side = probe_side(build.side);
            let columns = compiled_join_input_columns(node, build.strategy, side)?;
            let probe = prepare_bounded_join_side_input_from_resident_frame(
                execution,
                input,
                &columns,
                side,
                node_id,
                node.spec.build_side,
            )?;
            let (left, right) = orient_prepared_inputs(build.side, build.input, side, probe)?;
            Ok(BoundedCompiledJoinProbeState::Prepared(BoundedCompiledPreparedJoinProbeState {
                strategy: build.strategy,
                strategy_options: build.strategy_options,
                left,
                right,
            }))
        },
    }
}

pub(in crate::exec::bounded) fn execute_bounded_compiled_join_probe_from_stream(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    build: BoundedCompiledJoinBuildState,
    stream: BoundedJoinLeafLocalParquetSourceStream,
) -> Result<BoundedCompiledJoinProbeState> {
    if stream.chunk_count() == 0 {
        let input = materialize_bounded_join_leaf_local_parquet_source_stream(execution, stream)?;
        return execute_bounded_compiled_join_probe_from_resident_frame(execution, node_id, node, build, input);
    }

    match build {
        BoundedCompiledJoinBuildState::Hash(build) if stateless_streaming_probe_supported(node) => {
            let mut outputs = Vec::new();
            let stream_result = stream_bounded_join_leaf_local_parquet_source(execution, stream, |input, execution| {
                let chunk_outputs = execute_bounded_root_hash_join_probe_chunk_stream_from_resident_frame(
                    execution, node_id, node, &build, input,
                )?;
                outputs.extend(chunk_outputs);
                Ok(())
            });
            match stream_result {
                Ok(()) => Ok(BoundedCompiledJoinProbeState::HashStreamed { build, outputs }),
                Err(error) => {
                    release_streamed_join_outputs_reverse(execution, outputs);
                    release_bounded_root_hash_join_build_state(execution, build);
                    Err(error)
                },
            }
        },
        BoundedCompiledJoinBuildState::Prepared(build)
            if build.strategy == CompiledJoinExecutionStrategy::OneShotEqui
                && stateless_streaming_probe_supported(node) =>
        {
            execute_compiled_one_shot_equi_join_probe_from_stream(execution, node_id, node, build, stream)
        },
        BoundedCompiledJoinBuildState::Prepared(build)
            if build.strategy == CompiledJoinExecutionStrategy::PartitionedHashEqui
                && node.spec.residual_predicate.is_none()
                && stateless_streaming_probe_supported(node) =>
        {
            if partitioned_streaming_probe_supported(node) {
                return execute_compiled_partitioned_hash_equi_join_probe_from_stream(
                    execution, node_id, node, build, stream,
                );
            }
            let input = materialize_bounded_join_leaf_local_parquet_source_stream(execution, stream)?;
            execute_bounded_compiled_join_probe_from_resident_frame(
                execution,
                node_id,
                node,
                BoundedCompiledJoinBuildState::Prepared(build),
                input,
            )
        },
        BoundedCompiledJoinBuildState::Prepared(build)
            if build.strategy == CompiledJoinExecutionStrategy::MixedKeyedResidual
                && node.spec.join_type == JoinType::Inner =>
        {
            execute_compiled_mixed_join_probe_from_stream(execution, node_id, node, build, stream)
        },
        other => {
            let input = materialize_bounded_join_leaf_local_parquet_source_stream(execution, stream)?;
            execute_bounded_compiled_join_probe_from_resident_frame(execution, node_id, node, other, input)
        },
    }
}

pub(in crate::exec::bounded) fn execute_bounded_compiled_join_probe_from_frame_stream(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    build: BoundedCompiledJoinBuildState,
    mut inputs: ResidentFrameStreamPayload,
) -> Result<BoundedCompiledJoinProbeState> {
    match build {
        BoundedCompiledJoinBuildState::Hash(build) if stateless_streaming_probe_supported(node) => {
            Ok(BoundedCompiledJoinProbeState::HashLazyStream(RootHashJoinFrameStreamProbeState { build, inputs }))
        },
        BoundedCompiledJoinBuildState::Filtered(build) => {
            let mut outputs = Vec::new();
            while let Some(input) = inputs.next_yielded_frame(execution) {
                let mut input = match input {
                    Ok(input) => input,
                    Err(error) => {
                        inputs.release_all(execution);
                        release_streamed_join_outputs_reverse(execution, outputs);
                        release_compiled_filtered_join_build_state(execution, build);
                        return Err(error);
                    },
                };
                let input = input.take_for_release();
                match execute_compiled_filtered_join_probe_chunk_from_resident_frame(
                    execution, node_id, node, &build, input,
                ) {
                    Ok(output) => outputs.push(output),
                    Err(error) => {
                        inputs.release_all(execution);
                        release_streamed_join_outputs_reverse(execution, outputs);
                        release_compiled_filtered_join_build_state(execution, build);
                        return Err(error);
                    },
                }
            }
            inputs.release_producer_reservations(execution);
            Ok(BoundedCompiledJoinProbeState::FilteredStreamed { build, outputs })
        },
        BoundedCompiledJoinBuildState::Prepared(build)
            if build.strategy == CompiledJoinExecutionStrategy::OneShotEqui
                && stateless_streaming_probe_supported(node) =>
        {
            execute_compiled_one_shot_equi_join_probe_from_frame_stream(execution, node, build, inputs)
        },
        BoundedCompiledJoinBuildState::Prepared(build)
            if build.strategy == CompiledJoinExecutionStrategy::PartitionedHashEqui
                && node.spec.residual_predicate.is_none()
                && stateless_streaming_probe_supported(node) =>
        {
            execute_compiled_partitioned_hash_equi_join_probe_from_frame_stream(execution, node_id, node, build, inputs)
        },
        other => {
            let inputs = match inputs.into_remaining_frames_owned(execution) {
                Ok(inputs) => inputs,
                Err(error) => {
                    release_bounded_compiled_join_build_state(execution, other);
                    return Err(error);
                },
            };
            let input = match concatenate_streamed_hash_join_outputs(execution, node_id, inputs) {
                Ok(input) => input,
                Err(error) => {
                    release_bounded_compiled_join_build_state(execution, other);
                    return Err(error);
                },
            };
            execute_bounded_compiled_join_probe_from_resident_frame(execution, node_id, node, other, input)
        },
    }
}

pub(in crate::exec::bounded) fn execute_bounded_compiled_join_materialize(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    probe: BoundedCompiledJoinProbeState,
) -> Result<ResidentGpuFrame> {
    match probe {
        BoundedCompiledJoinProbeState::Hash(probe) => {
            execute_bounded_root_hash_join_materialize(execution, node_id, node, probe)
        },
        BoundedCompiledJoinProbeState::HashStreamed { build, outputs } => {
            release_bounded_root_hash_join_build_state(execution, build);
            concatenate_streamed_hash_join_outputs(execution, node_id, outputs)
        },
        BoundedCompiledJoinProbeState::HashLazyStream(probe) => {
            let stream = root_hash_join_frame_stream_probe_to_payload(node_id, node, probe);
            let outputs = stream.into_remaining_frames_owned(execution)?;
            concatenate_streamed_hash_join_outputs(execution, node_id, outputs)
        },
        BoundedCompiledJoinProbeState::Filtered(probe) => {
            execute_compiled_filtered_join_materialize(execution, node_id, node, probe)
        },
        BoundedCompiledJoinProbeState::FilteredStreamed { build, outputs } => {
            release_compiled_filtered_join_build_state(execution, build);
            concatenate_streamed_hash_join_outputs(execution, node_id, outputs)
        },
        BoundedCompiledJoinProbeState::Prepared(probe) => {
            execute_compiled_prepared_join_materialize(execution, node_id, node, probe)
        },
        BoundedCompiledJoinProbeState::MixedStreamed { build, outputs } => {
            release_bounded_join_side_frame(execution, build.input);
            concatenate_streamed_hash_join_outputs(execution, node_id, outputs)
        },
        BoundedCompiledJoinProbeState::OneShotStreamed { build, outputs, summary } => {
            execute_compiled_streamed_one_shot_join_materialize(execution, node_id, node, build, outputs, summary)
        },
        BoundedCompiledJoinProbeState::OneShotLazyStream(probe) => {
            let stream = one_shot_equi_join_frame_stream_probe_to_payload(node_id, node, probe);
            let outputs = stream.into_remaining_frames_owned(execution)?;
            concatenate_streamed_hash_join_outputs(execution, node_id, outputs)
        },
        BoundedCompiledJoinProbeState::PartitionedStreamed {
            build_side,
            build_rows,
            build_bytes,
            outputs,
            summary,
        } => execute_compiled_streamed_partitioned_hash_join_materialize(
            execution,
            node_id,
            node,
            build_side,
            build_rows,
            build_bytes,
            outputs,
            summary,
        ),
        BoundedCompiledJoinProbeState::PartitionedLazyStream(probe) => {
            let stream = partitioned_hash_equi_join_frame_stream_probe_to_payload(node_id, node, probe);
            let outputs = stream.into_remaining_frames_owned(execution)?;
            concatenate_streamed_hash_join_outputs(execution, node_id, outputs)
        },
    }
}

pub(in crate::exec::bounded) fn execute_bounded_compiled_join_materialize_preserving_stream(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    probe: BoundedCompiledJoinProbeState,
) -> Result<BoundedCompiledJoinMaterializeOutput> {
    match probe {
        BoundedCompiledJoinProbeState::HashStreamed { build, outputs } => {
            release_bounded_root_hash_join_build_state(execution, build);
            Ok(BoundedCompiledJoinMaterializeOutput::Stream(outputs))
        },
        BoundedCompiledJoinProbeState::Hash(probe) => Ok(BoundedCompiledJoinMaterializeOutput::LazyStream(
            execute_bounded_root_hash_join_materialize_preserving_stream(execution, node_id, node, probe)?,
        )),
        BoundedCompiledJoinProbeState::HashLazyStream(probe) => Ok(BoundedCompiledJoinMaterializeOutput::LazyStream(
            root_hash_join_frame_stream_probe_to_payload(node_id, node, probe),
        )),
        BoundedCompiledJoinProbeState::FilteredStreamed { build, outputs } => {
            release_compiled_filtered_join_build_state(execution, build);
            Ok(BoundedCompiledJoinMaterializeOutput::Stream(outputs))
        },
        BoundedCompiledJoinProbeState::OneShotStreamed { build, outputs, summary } => {
            let BoundedCompiledPreparedJoinBuildState { side: build_side, input, .. } = build;
            let probe_side = probe_side(build_side);
            let build_rows = input.rows;
            let build_bytes = input.bytes;
            release_bounded_join_side_frame(execution, input);
            record_streamed_one_shot_join_metrics(
                node_id,
                node,
                build_side,
                probe_side,
                build_rows,
                build_bytes,
                &summary,
                execution.metrics_mut(),
            );
            Ok(BoundedCompiledJoinMaterializeOutput::Stream(outputs))
        },
        BoundedCompiledJoinProbeState::MixedStreamed { build, outputs } => {
            release_bounded_join_side_frame(execution, build.input);
            Ok(BoundedCompiledJoinMaterializeOutput::Stream(outputs))
        },
        BoundedCompiledJoinProbeState::OneShotLazyStream(probe) => {
            Ok(BoundedCompiledJoinMaterializeOutput::LazyStream(one_shot_equi_join_frame_stream_probe_to_payload(
                node_id, node, probe,
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
                execution.metrics_mut(),
            );
            Ok(BoundedCompiledJoinMaterializeOutput::Stream(outputs))
        },
        BoundedCompiledJoinProbeState::PartitionedLazyStream(probe) => {
            Ok(BoundedCompiledJoinMaterializeOutput::LazyStream(
                partitioned_hash_equi_join_frame_stream_probe_to_payload(node_id, node, probe),
            ))
        },
        BoundedCompiledJoinProbeState::Prepared(probe)
            if probe.strategy == CompiledJoinExecutionStrategy::PartitionedHashEqui =>
        {
            execute_compiled_partitioned_hash_equi_join_materialize_preserving_stream(
                execution,
                node_id,
                node,
                probe.left,
                probe.right,
                probe.strategy_options,
            )
        },
        BoundedCompiledJoinProbeState::Prepared(probe)
            if probe.strategy == CompiledJoinExecutionStrategy::PartitionedMixedKeyedResidual =>
        {
            execute_compiled_partitioned_mixed_keyed_residual_join_materialize_preserving_stream(
                execution,
                node_id,
                node,
                probe.left,
                probe.right,
                probe.strategy_options,
            )
        },
        other => execute_bounded_compiled_join_materialize(execution, node_id, node, other)
            .map(|frame| BoundedCompiledJoinMaterializeOutput::Frame(Box::new(frame))),
    }
}

pub(in crate::exec::bounded) fn release_bounded_compiled_join_build_state(
    execution: &mut BoundedExecutionAttempt<'_>,
    build: BoundedCompiledJoinBuildState,
) {
    match build {
        BoundedCompiledJoinBuildState::Hash(build) => {
            release_bounded_root_hash_join_build_state(execution, build);
        },
        BoundedCompiledJoinBuildState::Filtered(build) => {
            release_compiled_filtered_join_build_state(execution, build);
        },
        BoundedCompiledJoinBuildState::Prepared(build) => {
            let BoundedCompiledPreparedJoinBuildState { input, .. } = build;
            release_bounded_join_side_frame(execution, input);
        },
    }
}

pub(in crate::exec::bounded) fn release_bounded_compiled_join_probe_state(
    execution: &mut BoundedExecutionAttempt<'_>,
    probe: BoundedCompiledJoinProbeState,
) {
    match probe {
        BoundedCompiledJoinProbeState::Hash(probe) => {
            release_bounded_root_hash_join_probe_state(execution, probe);
        },
        BoundedCompiledJoinProbeState::HashStreamed { build, outputs } => {
            release_streamed_join_outputs_reverse(execution, outputs);
            release_bounded_root_hash_join_build_state(execution, build);
        },
        BoundedCompiledJoinProbeState::HashLazyStream(probe) => {
            probe.inputs.release_all(execution);
            release_bounded_root_hash_join_build_state(execution, probe.build);
        },
        BoundedCompiledJoinProbeState::Filtered(probe) => {
            release_compiled_filtered_join_probe_state(execution, probe);
        },
        BoundedCompiledJoinProbeState::FilteredStreamed { build, outputs } => {
            release_streamed_join_outputs_reverse(execution, outputs);
            release_compiled_filtered_join_build_state(execution, build);
        },
        BoundedCompiledJoinProbeState::Prepared(probe) => {
            let BoundedCompiledPreparedJoinProbeState { left, right, .. } = probe;
            release_bounded_join_side_frame(execution, left);
            release_bounded_join_side_frame(execution, right);
        },
        BoundedCompiledJoinProbeState::MixedStreamed { build, outputs } => {
            release_streamed_join_outputs_reverse(execution, outputs);
            release_bounded_join_side_frame(execution, build.input);
        },
        BoundedCompiledJoinProbeState::OneShotStreamed { build, outputs, .. } => {
            release_streamed_join_outputs_reverse(execution, outputs);
            release_bounded_compiled_join_build_state(execution, BoundedCompiledJoinBuildState::Prepared(build));
        },
        BoundedCompiledJoinProbeState::OneShotLazyStream(probe) => {
            probe.inputs.release_all(execution);
            release_bounded_join_side_frame(execution, probe.build.input);
        },
        BoundedCompiledJoinProbeState::PartitionedStreamed { outputs, .. } => {
            release_streamed_join_outputs_reverse(execution, outputs);
        },
        BoundedCompiledJoinProbeState::PartitionedLazyStream(probe) => {
            probe.inputs.release_all(execution);
            release_partitioned_join_side(execution, probe.build_partitioned);
        },
    }
}

fn concatenate_streamed_hash_join_outputs(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    mut outputs: Vec<ResidentGpuFrame>,
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
    let mut output_reservation = match execution.try_reserve(bounded_join_request(
        ReservationKind::JoinOutput,
        "join_output_stream_concat",
        output_reservation_bytes,
        execution.reservations().budget(),
        node_id,
    )) {
        Ok(reservation) => reservation,
        Err(error) => {
            release_streamed_join_outputs_reverse(execution, outputs);
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
    let overrides_lease = execution_overrides(execution.context())?;
    let overrides = overrides_lease.overrides();
    let timer = Instant::now();
    let output = match first_frame.with_execution(&overrides).concat_many(&rest_frame_refs) {
        Ok(output) => output,
        Err(error) => {
            release_streamed_join_reservations_reverse(execution, input_reservations);
            execution.release(output_reservation);
            return Err(QueryError::from_cudf(
                "nexus_query_engine.bounded.join.streamed_hash_output_concat",
                "failed to concatenate streamed bounded hash join output chunks",
                error,
            )
            .with_fact("node_id", node_id)
            .with_fact("chunk_count", chunk_count));
        },
    };
    execution.metrics_mut().operator_time += timer.elapsed();

    let output_frame = GpuDataFrame::new(output);
    let output_bytes = match observe_device_memory_bytes(&output_frame, execution.metrics_mut()) {
        Ok(bytes) => bytes,
        Err(error) => {
            release_streamed_join_reservations_reverse(execution, input_reservations);
            execution.release(output_reservation);
            return Err(error);
        },
    };
    if output_bytes > output_reservation.bytes() {
        let delta = output_bytes.saturating_sub(output_reservation.bytes());
        if let Err(error) = execution.try_grow(
            &mut output_reservation,
            bounded_join_request(
                ReservationKind::JoinOutput,
                "join_output_stream_concat",
                delta,
                execution.reservations().budget(),
                node_id,
            ),
        ) {
            release_streamed_join_reservations_reverse(execution, input_reservations);
            execution.release(output_reservation);
            return Err(error);
        }
    } else {
        execution.shrink_to_observed(&mut output_reservation, output_bytes);
    }
    release_streamed_join_reservations_reverse(execution, input_reservations);
    Ok(ResidentGpuFrame { frame: output_frame, reservation: output_reservation })
}

fn release_streamed_join_outputs_reverse(
    execution: &mut BoundedExecutionAttempt<'_>,
    mut outputs: Vec<ResidentGpuFrame>,
) {
    while let Some(output) = outputs.pop() {
        let ResidentGpuFrame { reservation, .. } = output;
        execution.release(reservation);
    }
}

fn release_streamed_join_reservations_reverse(
    execution: &mut BoundedExecutionAttempt<'_>,
    mut active: Vec<ActiveReservation>,
) {
    while let Some(reservation) = active.pop() {
        execution.release(reservation);
    }
}

fn compiled_join_input_columns(
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

struct CompiledEqualityJoinPreflightIndices {
    indices: CompiledJoinIndices,
    gather_map_reservation: ActiveReservation,
    index_bytes: u64,
    output_rows: u64,
}

fn materialize_compiled_keyed_equality_join_indices_with_preflight(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    left: &BoundedJoinSideFrame,
    right: &BoundedJoinSideFrame,
    build_side: JoinSide,
    left_keys: &[String],
    right_keys: &[String],
    reservation_label: &'static str,
    overrides: &ExecutionOverrides<'_>,
) -> Result<CompiledEqualityJoinPreflightIndices> {
    let (build, build_keys, probe, probe_keys) = match build_side {
        JoinSide::Left => (left, left_keys, right, right_keys),
        JoinSide::Right => (right, right_keys, left, left_keys),
    };
    let hash_options =
        cudf::DataFrameHashJoinOptions::new().null_equality(lower_null_equality(node.spec.null_equality));
    let (hash_join, hash_table_reservation, _hash_table_bytes) = super::hash::reserve_and_build_bounded_hash_join(
        execution,
        node_id,
        node,
        &build.frame,
        build_keys,
        hash_options,
        build_side,
        overrides,
        "join_keyed_equality_preflight_hash_table",
        false,
    )?;
    let left_join_output_rows = match compiled_hash_join_left_probe_size_for_full_size_preflight(
        node,
        hash_join.hash_join(),
        &probe.frame,
        probe_keys,
        overrides,
    ) {
        Ok(rows) => rows as u64,
        Err(error) => {
            release_temporary_compiled_hash_join(execution, hash_join, hash_table_reservation);
            return Err(error);
        },
    };
    let full_size_preflight_workspace =
        match reserve_full_join_size_preflight_workspace(execution, node_id, node, build.rows, left_join_output_rows) {
            Ok(reservation) => reservation,
            Err(error) => {
                release_temporary_compiled_hash_join(execution, hash_join, hash_table_reservation);
                return Err(error);
            },
        };
    let output_size =
        match compiled_hash_join_probe_size(node, hash_join.hash_join(), &probe.frame, probe_keys, overrides) {
            Ok(output_size) => {
                if let Some(reservation) = full_size_preflight_workspace {
                    execution.release(reservation);
                }
                output_size
            },
            Err(error) => {
                if let Some(reservation) = full_size_preflight_workspace {
                    execution.release(reservation);
                }
                release_temporary_compiled_hash_join(execution, hash_join, hash_table_reservation);
                return Err(error);
            },
        };
    let gather_map_reservation_bytes =
        match hash_join.hash_join().probe_gather_map_reservation_size_bytes(output_size).map(|bytes| bytes as u64) {
            Ok(bytes) => bytes,
            Err(error) => {
                release_temporary_compiled_hash_join(execution, hash_join, hash_table_reservation);
                return Err(QueryError::from_cudf(
                    "nexus_query_engine.bounded.join.compiled_keyed_equality_gather_map_reservation",
                    "failed to size compiled bounded equality join gather maps from exact output-size preflight",
                    error,
                )
                .with_fact("join_type", node.spec.join_type)
                .with_fact("build_side", build_side));
            },
        };
    let gather_map_reservation = match execution.try_reserve(bounded_join_request(
        ReservationKind::GatherMap,
        reservation_label,
        gather_map_reservation_bytes,
        execution.reservations().budget(),
        node_id,
    )) {
        Ok(reservation) => reservation,
        Err(error) => {
            release_temporary_compiled_hash_join(execution, hash_join, hash_table_reservation);
            return Err(error);
        },
    };
    let hash_indices = match compiled_hash_join_probe_with_output_size(
        node,
        hash_join.hash_join(),
        &probe.frame,
        probe_keys,
        output_size,
        overrides,
    ) {
        Ok(indices) => indices,
        Err(error) => {
            execution.release(gather_map_reservation);
            release_temporary_compiled_hash_join(execution, hash_join, hash_table_reservation);
            return Err(error);
        },
    };
    release_temporary_compiled_hash_join(execution, hash_join, hash_table_reservation);

    let index_bytes = match hash_indices.device_allocated_size_bytes().map(|bytes| bytes as u64) {
        Ok(bytes) => bytes,
        Err(error) => {
            execution.release(gather_map_reservation);
            return Err(QueryError::from_cudf(
                "nexus_query_engine.bounded.join.compiled_keyed_equality_index_bytes",
                "failed to observe compiled bounded equality join gather-map memory",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
            .with_fact("build_side", build_side));
        },
    };
    if index_bytes > gather_map_reservation.bytes() {
        execution.release(gather_map_reservation);
        return Err(QueryError::new(
            QueryErrorCode::MemoryReservationFailed,
            "nexus_query_engine.bounded.join.compiled_keyed_equality_index_preflight",
            "compiled bounded equality join gather maps exceeded the exact output-size preflight reservation",
        )
        .with_fact("node_id", node_id)
        .with_fact("join_type", node.spec.join_type)
        .with_fact("build_side", build_side)
        .with_fact("reserved_bytes", gather_map_reservation_bytes)
        .with_fact("observed_bytes", index_bytes)
        .with_fact("output_rows", output_size));
    }

    let indices = match build_side {
        JoinSide::Left => {
            CompiledJoinIndices::Pair { left: hash_indices.build().clone(), right: hash_indices.probe().clone() }
        },
        JoinSide::Right => {
            CompiledJoinIndices::Pair { left: hash_indices.probe().clone(), right: hash_indices.build().clone() }
        },
    };
    Ok(CompiledEqualityJoinPreflightIndices {
        indices,
        gather_map_reservation,
        index_bytes,
        output_rows: output_size as u64,
    })
}

fn compiled_hash_join_left_probe_size_for_full_size_preflight(
    node: &JoinNode,
    hash_join: &cudf::DataFrameHashJoin,
    probe: &DataFrame,
    probe_keys: &[String],
    overrides: &ExecutionOverrides<'_>,
) -> Result<usize> {
    if node.spec.join_type != JoinType::Full {
        return Ok(0);
    }
    hash_join.left_probe_size_in(probe, probe_keys, overrides).map_err(|error| join_index_error(error, node))
}

fn release_temporary_compiled_hash_join(
    execution: &mut BoundedExecutionAttempt<'_>,
    _hash_join: GpuHashJoinBuildHandle,
    hash_table_reservation: ActiveReservation,
) {
    execution.release(hash_table_reservation);
}

fn compiled_hash_join_probe_size(
    node: &JoinNode,
    hash_join: &cudf::DataFrameHashJoin,
    probe: &DataFrame,
    probe_keys: &[String],
    overrides: &ExecutionOverrides<'_>,
) -> Result<usize> {
    match node.spec.join_type {
        JoinType::Inner => hash_join.inner_probe_size_in(probe, probe_keys, overrides),
        JoinType::Left => hash_join.left_probe_size_in(probe, probe_keys, overrides),
        JoinType::Full => hash_join.full_probe_size_in(probe, probe_keys, overrides),
        JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark | JoinType::Cross => {
            unreachable!("validated above")
        },
    }
    .map_err(|error| join_index_error(error, node))
}

fn compiled_hash_join_probe_with_output_size(
    node: &JoinNode,
    hash_join: &cudf::DataFrameHashJoin,
    probe: &DataFrame,
    probe_keys: &[String],
    output_size: usize,
    overrides: &ExecutionOverrides<'_>,
) -> Result<cudf::DataFrameHashJoinPairIndices> {
    match node.spec.join_type {
        JoinType::Inner => hash_join.inner_probe_with_output_size_in(probe, probe_keys, output_size, overrides),
        JoinType::Left => hash_join.left_probe_with_output_size_in(probe, probe_keys, output_size, overrides),
        JoinType::Full => hash_join.full_probe_with_output_size_in(probe, probe_keys, output_size, overrides),
        JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark | JoinType::Cross => {
            unreachable!("validated above")
        },
    }
    .map_err(|error| join_index_error(error, node))
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
        execution: &mut BoundedExecutionAttempt<'_>,
        node_id: NodeId,
        left: &DataFrame,
        right: &DataFrame,
        expected_output_rows: u64,
        projection: &[JoinOutputColumn],
    ) -> Result<u64> {
        match self {
            Self::Pair { .. } => {
                let left_projection = projected_output_columns_for_retained_sizing(projection, JoinSide::Left);
                let right_projection = projected_output_columns_for_retained_sizing(projection, JoinSide::Right);
                let left_bytes = join_side_output_upper_bound_bytes(
                    execution,
                    node_id,
                    left,
                    expected_output_rows,
                    &left_projection,
                    JoinSide::Left,
                )?;
                let right_bytes = join_side_output_upper_bound_bytes(
                    execution,
                    node_id,
                    right,
                    expected_output_rows,
                    &right_projection,
                    JoinSide::Right,
                )?;
                Ok(left_bytes.saturating_add(right_bytes))
            },
            Self::LeftOnly(_) => {
                let left_projection = projected_output_columns_for_retained_sizing(projection, JoinSide::Left);
                join_side_output_upper_bound_bytes(
                    execution,
                    node_id,
                    left,
                    expected_output_rows,
                    &left_projection,
                    JoinSide::Left,
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
        preserved_side: JoinSide,
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
                preserved_side,
            ),
            Self::LeftOnly(indices) => {
                materialize_left_only_join_output_projection(left, indices, overrides, projection, join_type)
            },
        }
    }
}
struct CompiledJoinMaterializationRequest {
    node_id: NodeId,
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
}

fn materialize_compiled_indices_output(
    execution: &mut BoundedExecutionAttempt<'_>,
    node: &JoinNode,
    request: CompiledJoinMaterializationRequest,
) -> Result<ResidentGpuFrame> {
    let CompiledJoinMaterializationRequest {
        node_id,
        execution_strategy,
        key_count,
        residual_predicate,
        build_side,
        probe_side,
        left,
        right,
        indices,
        mixed_size_data_bytes,
        filtered_probe_state_bytes,
        expected_output_rows,
        gather_map_reservation,
        timer,
        one_shot_index_bytes,
    } = request;
    let mut txn = JoinMaterializationTxn::with_sides(left, right);
    txn.set_gather_map_reservation(gather_map_reservation);
    let overrides_lease = execution_overrides(execution.context())?;
    let overrides = overrides_lease.overrides();
    txn.set_indices(indices);
    let gather_map_bytes = match txn.indices().device_allocated_size_bytes(node) {
        Ok(bytes) => bytes,
        Err(error) => {
            txn.rollback(execution);
            return Err(error);
        },
    };
    let (left, right) = txn.sides();
    let output_reservation_bytes = match txn.indices().output_upper_bound_size_bytes(
        execution,
        node_id,
        &left.frame,
        &right.frame,
        expected_output_rows,
        &node.spec.output_projection,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            txn.rollback(execution);
            return Err(error);
        },
    };
    if let Some(index_bytes) = one_shot_index_bytes {
        execution.metrics_mut().record_one_shot_join();
        execution.metrics_mut().record_one_shot_join_index_bytes(index_bytes);
        execution.metrics_mut().record_one_shot_join_materialization_bytes(output_reservation_bytes);
    }
    let output_reservation = match execution.try_reserve(bounded_join_request(
        ReservationKind::JoinOutput,
        "join_output",
        output_reservation_bytes,
        execution.reservations().budget(),
        node_id,
    )) {
        Ok(reservation) => reservation,
        Err(error) => {
            txn.rollback(execution);
            return Err(error);
        },
    };
    txn.set_output_reservation(output_reservation);
    let (left, right) = txn.sides();
    let output = match txn.indices().materialize_output(
        &left.frame,
        &right.frame,
        &overrides,
        &node.spec.output_projection,
        node.spec.join_type,
        node.spec.datafusion_order_preserved_side,
    ) {
        Ok(output) => output,
        Err(error) => {
            txn.rollback(execution);
            return Err(error);
        },
    };
    execution.metrics_mut().operator_time += timer.elapsed();
    let output_rows = output.num_rows() as u64;
    debug_assert_eq!(output_rows, expected_output_rows);
    let output_bytes = match observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), execution.metrics_mut()) {
        Ok(bytes) => bytes,
        Err(error) => {
            txn.rollback(execution);
            return Err(error);
        },
    };
    if one_shot_index_bytes.is_some() {
        execution.metrics_mut().record_one_shot_join_output_bytes(output_bytes);
    }
    if let Err(error) =
        reconcile_join_output_upper_bound(execution, node_id, txn.output_reservation_mut(), output_bytes)
    {
        txn.rollback(execution);
        return Err(error);
    }
    let (left, right) = txn.sides();
    execution.metrics_mut().record_operator_metrics(crate::metrics::OperatorMetrics::join(
        node_id,
        crate::metrics::JoinOperatorMetrics {
            query_id: None,
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
    txn.release_gather_map_reservation(execution);
    Ok(txn.commit_frame(execution, GpuDataFrame::new(output)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::JoinSpec;

    fn join_node(join_type: JoinType, build_side: JoinSide) -> JoinNode {
        JoinNode {
            inputs: [NodeId::from_index_for_internal_use(0), NodeId::from_index_for_internal_use(1)],
            spec: JoinSpec::new(join_type, Vec::new()).with_build_side(build_side),
        }
    }

    #[test]
    fn stateless_streaming_probe_rejects_outer_join_shapes_that_reemit_build_rows() {
        assert!(stateless_streaming_probe_supported(&join_node(JoinType::Inner, JoinSide::Left)));
        assert!(stateless_streaming_probe_supported(&join_node(JoinType::Inner, JoinSide::Right)));
        assert!(stateless_streaming_probe_supported(&join_node(JoinType::Left, JoinSide::Right)));

        assert!(!stateless_streaming_probe_supported(&join_node(JoinType::Left, JoinSide::Left)));
        assert!(!stateless_streaming_probe_supported(&join_node(JoinType::Full, JoinSide::Left)));
        assert!(!stateless_streaming_probe_supported(&join_node(JoinType::Full, JoinSide::Right)));
    }
}
