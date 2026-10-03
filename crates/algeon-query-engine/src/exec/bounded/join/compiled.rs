use std::{
    collections::{BTreeMap, VecDeque},
    time::Instant,
};

use super::{
    common::{
        BoundedJoinSideFrame, join_side_output_upper_bound_bytes, prepare_bounded_join_side_input_from_resident_frame,
        projected_output_columns_for_retained_sizing, release_bounded_join_side_frame,
    },
    cudf_query_error, *,
};
use crate::{
    admission::CompiledJoinExecutionStrategy,
    capability::{KeyedHashJoinStrategyOptions, KeylessConditionalJoinKind, keyed_filtered_semi_anti_join_capable},
    exec::bounded::compiled::{ResidentFrameStreamPayload, ResidentFrameStreamSource, lanes::AggregateLaneDispatch},
    expr::{ColumnQualifier, ColumnRef},
    metrics::{JoinOperatorMetrics, OperatorMetrics},
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
pub(in crate::exec::bounded) use partitioned::{
    BoundedJoinBuildInputStream, BoundedJoinBuildStreamObservation, BoundedJoinBuildStreamOutcome,
    PartitionedJoinBuildCommitment, consume_bounded_join_build_input_stream, partitioned_join_build_commitment,
};
use prepared::*;
use residual::*;
use streamed::*;
pub(in crate::exec) use transaction::JoinMaterializationTxn;

pub(in crate::exec::bounded) enum BoundedCompiledJoinBuildState {
    Hash(BoundedRootHashJoinBuildState),
    Filtered(BoundedCompiledFilteredJoinBuildState),
    Prepared(BoundedCompiledPreparedJoinBuildState),
    Partitioned(BoundedCompiledPartitionedJoinBuildState),
}

// Build side already hash-partitioned while its input streamed in; every
// probe chunk pairs partition by partition with these frames.
pub(in crate::exec::bounded) struct BoundedCompiledPartitionedJoinBuildState {
    pub(in crate::exec::bounded::join::compiled) strategy_options: KeyedHashJoinStrategyOptions,
    pub(in crate::exec::bounded::join::compiled) side: JoinSide,
    pub(in crate::exec::bounded::join::compiled) rows: u64,
    pub(in crate::exec::bounded::join::compiled) bytes: u64,
    pub(in crate::exec::bounded::join::compiled) partition_count: usize,
    pub(in crate::exec::bounded::join::compiled) partition_target_bytes: u64,
    pub(in crate::exec::bounded::join::compiled) partitioned: PartitionedJoinSide,
    pub(in crate::exec::bounded::join::compiled) input_chunks: usize,
}

impl BoundedCompiledPartitionedJoinBuildState {
    pub(in crate::exec::bounded) fn partitions(&self) -> &[DataFrame] {
        match &self.partitioned {
            PartitionedJoinSide::Split { partitions, .. } => partitions,
            PartitionedJoinSide::Contiguous { .. } => {
                unreachable!("streamed build partitions are stored as split partitions")
            },
        }
    }
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
    MixedLeftMatched(BoundedCompiledMixedLeftJoinProbeState),
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

pub(in crate::exec::bounded) struct BoundedCompiledMixedLeftJoinProbeState {
    left: BoundedJoinSideFrame,
    finalized: cudf::DataFrameMixedLeftJoinFinalized,
    probe_rows: u64,
    probe_bytes: u64,
    state_bytes: u64,
    operator_time: std::time::Duration,
}

pub(in crate::exec::bounded) struct RootHashJoinFrameStreamProbeState {
    build: BoundedRootHashJoinBuildState,
    inputs: ResidentFrameStreamPayload,
}

pub(in crate::exec::bounded) fn stateless_streaming_probe_supported(node: &JoinNode) -> bool {
    matches!(node.spec.join_type, JoinType::Inner)
        || (node.spec.join_type == JoinType::Left && node.spec.build_side == JoinSide::Right)
}

pub(in crate::exec::bounded) struct BoundedCompiledFilteredJoinBuildState {
    pub(in crate::exec::bounded) left_keys: Vec<String>,
    pub(in crate::exec::bounded) left_columns: Vec<String>,
    pub(in crate::exec::bounded) build: BoundedJoinSideFrame,
    pub(in crate::exec::bounded) input_build_rows: u64,
    pub(in crate::exec::bounded) input_build_bytes: u64,
    pub(in crate::exec::bounded) filtered_join: GpuFilteredJoinBuildHandle,
    pub(in crate::exec::bounded) hash_table_bytes: u64,
    pub(in crate::exec::bounded) null_aware_probe: NullAwareAntiProbe,
    pub(in crate::exec::bounded) timer: Instant,
}

pub(in crate::exec::bounded) struct BoundedCompiledFilteredJoinProbeState {
    build: BoundedCompiledFilteredJoinBuildState,
    probe: BoundedJoinSideFrame,
    probe_state: cudf::DataFrameFilteredJoinProbeState,
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

// Hash-partitioned join input. `Contiguous` is one partitioned frame with
// partition offsets (the shape `hash_partition_by` returns); `Split` keeps one
// frame per partition so a build side consumed chunk by chunk never needs a
// whole-side concat.
enum PartitionedJoinSide {
    Contiguous { frame: DataFrame, device_bytes: u64, offsets: Vec<usize> },
    Split { partitions: Vec<DataFrame>, device_bytes: u64 },
}

impl PartitionedJoinSide {
    fn device_bytes(&self) -> u64 {
        match self {
            Self::Contiguous { device_bytes, .. } | Self::Split { device_bytes, .. } => *device_bytes,
        }
    }
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

    #[expect(
        clippy::too_many_arguments,
        reason = "strategy-decision facts are reported individually; grouping them would be a metrics DTO"
    )]
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
    let max_frame_bytes_hint = probe.inputs.max_frame_bytes_hint();
    let total_bytes_hint = probe.inputs.total_bytes_hint();
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
        max_frame_bytes_hint,
        total_bytes_hint,
        aggregate_num_rows,
        num_columns,
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
            let input = match inputs.next_frame(execution) {
                Some(Ok(input)) => input,
                Some(Err(error)) => return Some(Err(error)),
                None => {
                    self.release(execution);
                    return None;
                },
            };
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
        self.pending_outputs.clear();
        if let Some(input) = self.inputs.take() {
            input.release_all(execution);
        }
        if let Some(build) = self.build.take() {
            release_bounded_root_hash_join_build_state(build);
        }
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
            execute_bounded_root_hash_join_build_from_resident_frame(execution, node, input)
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
            let input = prepare_bounded_join_side_input_from_resident_frame(execution, input, &columns, side)?;
            Ok(BoundedCompiledJoinBuildState::Prepared(BoundedCompiledPreparedJoinBuildState {
                strategy,
                strategy_options,
                side,
                input,
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
            execute_compiled_filtered_join_probe_from_resident_frame(execution, node, build, input)
                .map(BoundedCompiledJoinProbeState::Filtered)
        },
        BoundedCompiledJoinBuildState::Prepared(build) => {
            if mixed_left_match_probe_supported(node, build.strategy) {
                return execute_compiled_mixed_left_join_probe_from_frame_stream(
                    execution,
                    node,
                    build,
                    ResidentFrameStreamPayload::new(vec![input]),
                );
            }
            let side = probe_side(build.side);
            let columns = compiled_join_input_columns(node, build.strategy, side)?;
            let probe = prepare_bounded_join_side_input_from_resident_frame(execution, input, &columns, side)?;
            let (left, right) = orient_prepared_inputs(build.side, build.input, side, probe)?;
            Ok(BoundedCompiledJoinProbeState::Prepared(BoundedCompiledPreparedJoinProbeState {
                strategy: build.strategy,
                strategy_options: build.strategy_options,
                left,
                right,
            }))
        },
        build @ BoundedCompiledJoinBuildState::Partitioned(_) => execute_bounded_compiled_join_probe_from_frame_stream(
            execution,
            node_id,
            node,
            build,
            ResidentFrameStreamPayload::new(vec![input]),
            None,
        ),
    }
}

/// Runs the probe stage over a deferred Parquet leaf.
///
/// `lazy_output` says the join output is consumed as a stream, so stateless
/// probes pull leaf chunks on demand and yield outputs one by one instead of
/// collecting every output chunk before the materialize task runs. Partitioned
/// build sides always probe lazily: their partition outputs are already
/// bounded per probe chunk.
pub(in crate::exec::bounded) fn execute_bounded_compiled_join_probe_from_stream(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    build: BoundedCompiledJoinBuildState,
    stream: BoundedJoinLeafLocalParquetSourceStream,
    lazy_output: bool,
) -> Result<BoundedCompiledJoinProbeState> {
    if stream.chunk_count() == 0 {
        let input = materialize_bounded_join_leaf_local_parquet_source_stream(execution, stream)?;
        return execute_bounded_compiled_join_probe_from_resident_frame(execution, node_id, node, build, input);
    }

    let lazy_probe_supported = match &build {
        BoundedCompiledJoinBuildState::Hash(_) => lazy_output && stateless_streaming_probe_supported(node),
        BoundedCompiledJoinBuildState::Prepared(build) => {
            mixed_left_match_probe_supported(node, build.strategy)
                || (lazy_output
                    && stateless_streaming_probe_supported(node)
                    && matches!(
                        build.strategy,
                        CompiledJoinExecutionStrategy::OneShotEqui | CompiledJoinExecutionStrategy::PartitionedHashEqui
                    )
                    && node.spec.residual_predicate.is_none())
        },
        BoundedCompiledJoinBuildState::Partitioned(_) => true,
        BoundedCompiledJoinBuildState::Filtered(_) => false,
    };
    if lazy_probe_supported {
        let inputs = bounded_join_leaf_local_parquet_source_frame_stream(execution, stream);
        return execute_bounded_compiled_join_probe_from_frame_stream(execution, node_id, node, build, inputs, None);
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
                    release_bounded_root_hash_join_build_state(build);
                    Err(error)
                },
            }
        },
        // Semi/anti/mark membership is decided per probe row, so leaf chunks
        // stream through the filtered build exactly like a frame stream does
        // instead of materializing the whole leaf first.
        BoundedCompiledJoinBuildState::Filtered(build) => {
            let mut outputs = Vec::new();
            let stream_result = stream_bounded_join_leaf_local_parquet_source(execution, stream, |input, execution| {
                let output = execute_compiled_filtered_join_probe_chunk_from_resident_frame(
                    execution, node_id, node, &build, input,
                )?;
                outputs.push(output);
                Ok(())
            });
            match stream_result {
                Ok(()) => Ok(BoundedCompiledJoinProbeState::FilteredStreamed { build, outputs }),
                Err(error) => {
                    release_compiled_filtered_join_build_state(build);
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
    outer: Option<&AggregateLaneDispatch>,
) -> Result<BoundedCompiledJoinProbeState> {
    match build {
        BoundedCompiledJoinBuildState::Prepared(build) if mixed_left_match_probe_supported(node, build.strategy) => {
            execute_compiled_mixed_left_join_probe_from_frame_stream(execution, node, build, inputs)
        },
        BoundedCompiledJoinBuildState::Hash(build) if stateless_streaming_probe_supported(node) => {
            // Reusable build with parallel probe chunks: one table built once,
            // many probe chunks share it on disjoint lanes with private
            // workspace. Eager multi-frame inputs probe in parallel and merge
            // by concatenation (chunk-independent for inner/probe-preserved
            // left); single-chunk and lazy inputs keep the lazy stream path.
            if hash_parallel_probe_supported(node) && inputs.len() > 1 {
                match parallel_hash_probe_frame_stream(execution, node_id, node, &build, inputs, outer) {
                    Ok(outputs) => Ok(BoundedCompiledJoinProbeState::HashStreamed { build, outputs }),
                    Err(error) => {
                        release_bounded_root_hash_join_build_state(build);
                        Err(error)
                    },
                }
            } else {
                Ok(BoundedCompiledJoinProbeState::HashLazyStream(RootHashJoinFrameStreamProbeState { build, inputs }))
            }
        },
        BoundedCompiledJoinBuildState::Filtered(build) => {
            // One filtered build serves many probe chunks; membership is per
            // probe row, so parallel chunks merge by concatenation. Eager
            // multi-frame inputs use lanes, otherwise keep the serial stream
            // drain.
            if filtered_parallel_probe_supported(node) && inputs.len() > 1 {
                let frames = match inputs.into_remaining_frames_owned(execution) {
                    Ok(frames) => frames,
                    Err(error) => {
                        release_compiled_filtered_join_build_state(build);
                        return Err(error);
                    },
                };
                match parallel_filtered_probe_frames(execution, node_id, node, &build, frames, outer) {
                    Ok(outputs) => Ok(BoundedCompiledJoinProbeState::FilteredStreamed { build, outputs }),
                    Err(error) => {
                        release_compiled_filtered_join_build_state(build);
                        Err(error)
                    },
                }
            } else {
                let mut outputs = Vec::new();
                while let Some(input) = inputs.next_frame(execution) {
                    let input = match input {
                        Ok(input) => input,
                        Err(error) => {
                            inputs.release_all(execution);
                            release_compiled_filtered_join_build_state(build);
                            return Err(error);
                        },
                    };
                    match execute_compiled_filtered_join_probe_chunk_from_resident_frame(
                        execution, node_id, node, &build, input,
                    ) {
                        Ok(output) => outputs.push(output),
                        Err(error) => {
                            inputs.release_all(execution);
                            release_compiled_filtered_join_build_state(build);
                            return Err(error);
                        },
                    }
                }
                inputs.release_producer(execution);
                Ok(BoundedCompiledJoinProbeState::FilteredStreamed { build, outputs })
            }
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
        BoundedCompiledJoinBuildState::Partitioned(build) => {
            execute_compiled_partitioned_hash_equi_join_probe_from_split_build_frame_stream(
                execution, node_id, node, build, inputs,
            )
        },
        other => {
            // Serial probe path for strategies that require exclusive access
            // or a global complement (build-preserved outer, full, residual,
            // mixed/keyless/cross). The reason is recorded beside the
            // parallel-probe decisions so the fallback stays visible in the
            // same observation stream.
            let serial_reason = serial_probe_reason(node);
            execution.metrics().record_join_strategy_decision(
                node_id,
                &BTreeMap::from([("probe", "serial".to_string()), ("serial_probe_reason", serial_reason.to_string())]),
            );
            let inputs = match inputs.into_remaining_frames_owned(execution) {
                Ok(inputs) => inputs,
                Err(error) => {
                    release_bounded_compiled_join_build_state(other);
                    return Err(error);
                },
            };
            let input = match concatenate_streamed_hash_join_outputs(execution, node_id, inputs) {
                Ok(input) => input,
                Err(error) => {
                    release_bounded_compiled_join_build_state(other);
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
    outer: Option<&AggregateLaneDispatch>,
) -> Result<ResidentGpuFrame> {
    match probe {
        BoundedCompiledJoinProbeState::Hash(probe) => {
            execute_bounded_root_hash_join_materialize(execution, node_id, node, probe)
        },
        BoundedCompiledJoinProbeState::HashStreamed { build, outputs } => {
            release_bounded_root_hash_join_build_state(build);
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
            release_compiled_filtered_join_build_state(build);
            concatenate_streamed_hash_join_outputs(execution, node_id, outputs)
        },
        BoundedCompiledJoinProbeState::Prepared(probe) => {
            execute_compiled_prepared_join_materialize(execution, node_id, node, probe)
        },
        BoundedCompiledJoinProbeState::MixedLeftMatched(probe) => {
            execute_compiled_mixed_left_join_materialize(execution, node_id, node, probe)
        },
        BoundedCompiledJoinProbeState::MixedStreamed { build, outputs } => {
            release_bounded_join_side_frame(build.input);
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
            // Independent probe chunks share one split build on the same
            // dispatcher; partitions within a chunk stay serial, preserving
            // split/retry and skew handling. Single-chunk and lazy inputs
            // keep the serial lazy drain.
            if partitioned_parallel_probe_supported(node) && probe.inputs.len() > 1 {
                match parallel_partitioned_lazy_drain(execution, node_id, node, probe, outer) {
                    Ok(outputs) => concatenate_streamed_hash_join_outputs(execution, node_id, outputs),
                    Err(error) => Err(error),
                }
            } else {
                let stream = partitioned_hash_equi_join_frame_stream_probe_to_payload(node_id, node, probe);
                let outputs = stream.into_remaining_frames_owned(execution)?;
                concatenate_streamed_hash_join_outputs(execution, node_id, outputs)
            }
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
            release_bounded_root_hash_join_build_state(build);
            Ok(BoundedCompiledJoinMaterializeOutput::Stream(outputs))
        },
        BoundedCompiledJoinProbeState::Hash(probe) => Ok(BoundedCompiledJoinMaterializeOutput::LazyStream(
            execute_bounded_root_hash_join_materialize_preserving_stream(execution, node_id, node, probe)?,
        )),
        BoundedCompiledJoinProbeState::HashLazyStream(probe) => Ok(BoundedCompiledJoinMaterializeOutput::LazyStream(
            root_hash_join_frame_stream_probe_to_payload(node_id, node, probe),
        )),
        BoundedCompiledJoinProbeState::FilteredStreamed { build, outputs } => {
            release_compiled_filtered_join_build_state(build);
            Ok(BoundedCompiledJoinMaterializeOutput::Stream(outputs))
        },
        BoundedCompiledJoinProbeState::OneShotStreamed { build, outputs, summary } => {
            let BoundedCompiledPreparedJoinBuildState { side: build_side, input, .. } = build;
            let probe_side = probe_side(build_side);
            let build_rows = input.rows;
            let build_bytes = input.bytes;
            release_bounded_join_side_frame(input);
            record_streamed_one_shot_join_metrics(
                node_id,
                node,
                build_side,
                probe_side,
                build_rows,
                build_bytes,
                &summary,
                execution.metrics(),
            );
            Ok(BoundedCompiledJoinMaterializeOutput::Stream(outputs))
        },
        BoundedCompiledJoinProbeState::MixedStreamed { build, outputs } => {
            release_bounded_join_side_frame(build.input);
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
                execution.metrics(),
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
        other => execute_bounded_compiled_join_materialize(execution, node_id, node, other, None)
            .map(|frame| BoundedCompiledJoinMaterializeOutput::Frame(Box::new(frame))),
    }
}

pub(in crate::exec::bounded) fn release_bounded_compiled_join_build_state(build: BoundedCompiledJoinBuildState) {
    match build {
        BoundedCompiledJoinBuildState::Hash(build) => {
            release_bounded_root_hash_join_build_state(build);
        },
        BoundedCompiledJoinBuildState::Filtered(build) => {
            release_compiled_filtered_join_build_state(build);
        },
        BoundedCompiledJoinBuildState::Prepared(build) => {
            let BoundedCompiledPreparedJoinBuildState { input, .. } = build;
            release_bounded_join_side_frame(input);
        },
        BoundedCompiledJoinBuildState::Partitioned(build) => {
            release_partitioned_join_side(build.partitioned);
        },
    }
}

pub(in crate::exec::bounded) fn release_bounded_compiled_join_probe_state(
    execution: &mut BoundedExecutionAttempt<'_>,
    probe: BoundedCompiledJoinProbeState,
) {
    match probe {
        BoundedCompiledJoinProbeState::Hash(probe) => {
            release_bounded_root_hash_join_probe_state(probe);
        },
        BoundedCompiledJoinProbeState::HashStreamed { build, .. } => {
            release_bounded_root_hash_join_build_state(build);
        },
        BoundedCompiledJoinProbeState::HashLazyStream(probe) => {
            probe.inputs.release_all(execution);
            release_bounded_root_hash_join_build_state(probe.build);
        },
        BoundedCompiledJoinProbeState::Filtered(probe) => {
            release_compiled_filtered_join_probe_state(probe);
        },
        BoundedCompiledJoinProbeState::FilteredStreamed { build, .. } => {
            release_compiled_filtered_join_build_state(build);
        },
        BoundedCompiledJoinProbeState::Prepared(probe) => {
            let BoundedCompiledPreparedJoinProbeState { left, right, .. } = probe;
            release_bounded_join_side_frame(left);
            release_bounded_join_side_frame(right);
        },
        BoundedCompiledJoinProbeState::MixedLeftMatched(probe) => {
            release_compiled_mixed_left_join_probe_state(probe);
        },
        BoundedCompiledJoinProbeState::MixedStreamed { build, .. } => {
            release_bounded_join_side_frame(build.input);
        },
        BoundedCompiledJoinProbeState::OneShotStreamed { build, .. } => {
            release_bounded_compiled_join_build_state(BoundedCompiledJoinBuildState::Prepared(build));
        },
        BoundedCompiledJoinProbeState::OneShotLazyStream(probe) => {
            probe.inputs.release_all(execution);
            release_bounded_join_side_frame(probe.build.input);
        },
        BoundedCompiledJoinProbeState::PartitionedStreamed { .. } => {},
        BoundedCompiledJoinProbeState::PartitionedLazyStream(probe) => {
            probe.inputs.release_all(execution);
            release_partitioned_join_side(probe.build_partitioned);
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
            "query_engine.bounded.join.streamed_hash_output",
            "streamed hash join probe produced no output chunks",
        )
        .with_fact("node_id", node_id));
    }

    let chunk_count = outputs.len();
    let mut frames = Vec::with_capacity(outputs.len());
    for output in outputs {
        frames.push(output.frame.into_frame());
    }
    let first_frame = frames.remove(0);
    let rest_frame_refs = frames.iter().collect::<Vec<_>>();
    let overrides_lease = execution_overrides(execution.context())?;
    let overrides = overrides_lease.overrides();
    let timer = Instant::now();
    let output = match first_frame.with_execution(&overrides).concat_many(&rest_frame_refs) {
        Ok(output) => output,
        Err(error) => {
            return Err(cudf_query_error(
                "query_engine.bounded.join.streamed_hash_output_concat",
                "failed to concatenate streamed bounded hash join output chunks",
                error,
            )
            .with_fact("node_id", node_id)
            .with_fact("chunk_count", chunk_count));
        },
    };
    execution.metrics().add_operator_time(timer.elapsed());

    let output_frame = GpuDataFrame::new(output);
    let output_bytes = match observe_device_memory_bytes(&output_frame, execution.metrics()) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Err(error);
        },
    };
    Ok(ResidentGpuFrame::with_device_bytes(output_frame, output_bytes))
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
            Err(bounded_invariant_violation(
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
        _ => Err(bounded_invariant_violation(
            "compiled prepared join inputs must contain exactly one left and one right side",
            "compiled_join_input_orientation_invalid",
        )),
    }
}

enum CompiledKeylessWork<'a> {
    Full { size_data: Box<cudf::DataFrameConditionalFullJoinSizeData<'a>> },
    Sized { output_size: usize, options: cudf::DataFramePredicateJoinOptions },
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
            cudf_query_error(
                "query_engine.bounded.join.mixed_size_data_bytes",
                "failed to observe compiled bounded mixed join size-data memory",
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
    index_bytes: u64,
    output_rows: u64,
}

#[expect(
    clippy::too_many_arguments,
    reason = "bounded execution passes operator-occurrence inputs explicitly beside the attempt state"
)]
fn materialize_compiled_keyed_equality_join_indices_with_preflight(
    execution: &mut BoundedExecutionAttempt<'_>,
    node: &JoinNode,
    left: &BoundedJoinSideFrame,
    right: &BoundedJoinSideFrame,
    build_side: JoinSide,
    left_keys: &[String],
    right_keys: &[String],
    overrides: &ExecutionOverrides<'_>,
) -> Result<CompiledEqualityJoinPreflightIndices> {
    let (build, build_keys, probe, probe_keys) = match build_side {
        JoinSide::Left => (left, left_keys, right, right_keys),
        JoinSide::Right => (right, right_keys, left, left_keys),
    };
    let hash_options =
        cudf::DataFrameHashJoinOptions::new().null_equality(lower_null_equality(node.spec.null_equality));
    let (hash_join, _hash_table_bytes) = super::hash::build_bounded_hash_join(
        execution,
        node,
        &build.frame,
        build_keys,
        hash_options,
        build_side,
        overrides,
        false,
    )?;
    let output_size = compiled_hash_join_probe_size(node, hash_join.hash_join(), &probe.frame, probe_keys, overrides)?;
    let hash_indices = compiled_hash_join_probe_with_output_size(
        node,
        hash_join.hash_join(),
        &probe.frame,
        probe_keys,
        output_size,
        overrides,
    )?;

    let index_bytes = match hash_indices.device_allocated_size_bytes().map(|bytes| bytes as u64) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Err(cudf_query_error(
                "query_engine.bounded.join.compiled_keyed_equality_index_bytes",
                "failed to observe compiled bounded equality join gather-map memory",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
            .with_fact("build_side", build_side));
        },
    };
    let indices = match build_side {
        JoinSide::Left => {
            CompiledJoinIndices::Pair { left: hash_indices.build().clone(), right: hash_indices.probe().clone() }
        },
        JoinSide::Right => {
            CompiledJoinIndices::Pair { left: hash_indices.probe().clone(), right: hash_indices.build().clone() }
        },
    };
    Ok(CompiledEqualityJoinPreflightIndices { indices, index_bytes, output_rows: output_size as u64 })
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
                    cudf_query_error(
                        "query_engine.bounded.join.gather_map_bytes",
                        "failed to observe compiled bounded join left gather-map memory",
                        error,
                    )
                    .with_fact("join_type", node.spec.join_type)
                })?;
                let right_bytes = right.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
                    cudf_query_error(
                        "query_engine.bounded.join.gather_map_bytes",
                        "failed to observe compiled bounded join right gather-map memory",
                        error,
                    )
                    .with_fact("join_type", node.spec.join_type)
                })?;
                Ok(left_bytes.saturating_add(right_bytes))
            },
            Self::LeftOnly(indices) => {
                indices.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
                    cudf_query_error(
                        "query_engine.bounded.join.gather_map_bytes",
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
    gather_map_bytes: u64,
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
        gather_map_bytes,
        timer,
        one_shot_index_bytes,
    } = request;
    let mut txn = JoinMaterializationTxn::with_sides(left, right);
    let overrides_lease = execution_overrides(execution.context())?;
    let overrides = overrides_lease.overrides();
    txn.set_indices(indices);
    let observed_gather_map_bytes = match txn.indices().device_allocated_size_bytes(node) {
        Ok(bytes) => bytes,
        Err(error) => {
            txn.rollback();
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
            txn.rollback();
            return Err(error);
        },
    };
    if let Some(index_bytes) = one_shot_index_bytes {
        execution.metrics().record_one_shot_join();
        execution.metrics().record_one_shot_join_index_bytes(index_bytes);
        execution.metrics().record_one_shot_join_materialization_bytes(output_reservation_bytes);
    }
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
            txn.rollback();
            return Err(error);
        },
    };
    execution.metrics().add_operator_time(timer.elapsed());
    let output_rows = output.num_rows() as u64;
    debug_assert_eq!(output_rows, expected_output_rows);
    let output_bytes = match observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), execution.metrics()) {
        Ok(bytes) => bytes,
        Err(error) => {
            txn.rollback();
            return Err(error);
        },
    };
    if one_shot_index_bytes.is_some() {
        execution.metrics().record_one_shot_join_output_bytes(output_bytes);
    }
    debug_assert_eq!(observed_gather_map_bytes, gather_map_bytes);
    let (left, right) = txn.sides();
    execution.metrics().record_operator_metrics(OperatorMetrics::join(
        node_id,
        JoinOperatorMetrics {
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
    Ok(txn.commit_frame(GpuDataFrame::new(output), output_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::JoinSpec;

    fn join_node(join_type: JoinType, build_side: JoinSide) -> JoinNode {
        JoinNode {
            inputs: [NodeId::from_index_for_test(0), NodeId::from_index_for_test(1)],
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
