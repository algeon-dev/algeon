//! Build, probe and materialize states of a compiled bounded join.
//!
//! The pipeline compiler expands one native join into a build task, a probe
//! task and a materialize task with a build-before-probe barrier
//! (`website/docs/design/native-lowering.mdx`, "What the GPU compiler
//! produces"). This module owns the state passed between those tasks and the
//! dispatch over `JoinStrategy`; the strategy submodules own one join shape
//! each, and `hash` owns the reusable hash join they all fall back to.
//!
//! Every state is consumed by value. A function that returns `Err` has already
//! dropped or released the build frames, probe frames, stream producers and
//! gather maps it was given, so a caller never releases twice; the `release_*`
//! functions exist for the interpreter paths where a task's consumer never
//! runs (cancellation, an earlier task's failure). A build frame is released
//! before the probe outputs are concatenated, so the hash table is not part of
//! the concatenation's peak.

use std::{
    collections::{BTreeMap, VecDeque},
    time::Instant,
};

use cudf::ops::ast::AstExpression;
use query_planning::{
    capability::{JoinForm, JoinStrategy, KeyedHashJoinStrategyOptions, KeylessConditionalJoinKind},
    expr::{ColumnQualifier, ColumnRef},
};
use query_runtime::metrics::{JoinOperatorMetrics, OperatorMetrics};

use super::{
    BoundedExecutionAttempt, BoundedJoinLeafLocalParquetSourceStream, BoundedPartitionRowPolicy,
    BoundedPartitionTargetPolicy, BoundedRootHashJoinBuildState, BoundedRootHashJoinProbeState, DataFrame,
    DropDuplicatesOptions, DuplicateKeepOption, Exec, ExecutionContext, ExecutionMetrics, Expr, GpuDataFrame,
    GpuFilteredJoinBuildHandle, JoinNode, JoinOutputColumn, JoinSide, JoinType, MemoryControlPolicy, NodeId,
    NullAwareAntiProbe, QueryError, QueryErrorCode, ResidentGpuFrame, Result, RetryFacts,
    attach_budget_token_failure_facts, bounded_invariant_violation,
    bounded_join_leaf_local_parquet_source_frame_stream,
    common::{
        BoundedJoinSideFrame, join_side_output_upper_bound_bytes, prepare_bounded_join_side_input_from_resident_frame,
        projected_output_columns_for_retained_sizing, release_bounded_join_side_frame, sum_bounds,
    },
    cudf_query_error, dataframe_cross_join_options, dataframe_mixed_join_options, dataframe_predicate_join_options,
    datafusion_order_columns, execute_bounded_root_hash_join_build_from_resident_frame,
    execute_bounded_root_hash_join_materialize, execute_bounded_root_hash_join_materialize_preserving_stream,
    execute_bounded_root_hash_join_probe_chunk_stream_from_resident_frame,
    execute_bounded_root_hash_join_probe_from_resident_frame, execution_budget_token_for_boundary,
    hash_parallel_probe_supported, join_index_error, lower_expr, lower_join_residual_predicate, lower_null_equality,
    materialize_bounded_join_leaf_local_parquet_source_stream, materialize_join_output_projection,
    materialize_left_only_join_output_projection, materialize_mark_join_output_projection, null_aware_anti_probe,
    null_aware_anti_probe_input, observe_device_memory_bytes, parallel_hash_probe_frame_stream, probe_side,
    projected_input_columns, projected_output_columns, release_bounded_root_hash_join_build_state,
    release_bounded_root_hash_join_probe_state, reorder_projected_cross_join_output, residual_input_columns,
    serial_probe_reason, side_metric, stream_bounded_join_leaf_local_parquet_source,
    stream_bounded_join_leaf_local_parquet_source_serial, validate_executable_join_subset,
};
use crate::exec::bounded::compiled::{
    ResidentFrameStreamPayload, ResidentFrameStreamSource, lanes::AggregateLaneDispatch,
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
use cross::{execute_compiled_keyless_outer_cross_join_materialize, execute_compiled_projected_cross_join_materialize};
use filtered::{
    execute_compiled_filtered_join_build_from_resident_frame, execute_compiled_filtered_join_materialize,
    execute_compiled_filtered_join_probe_chunk_from_resident_frame,
    execute_compiled_filtered_join_probe_from_resident_frame, filtered_parallel_probe_supported,
    parallel_filtered_probe_frames, release_compiled_filtered_join_build_state,
    release_compiled_filtered_join_probe_state,
};
use keyless::execute_compiled_keyless_join_materialize;
use mixed::{
    execute_compiled_mixed_join_materialize, execute_compiled_mixed_join_probe_from_stream,
    execute_compiled_mixed_left_join_materialize, execute_compiled_mixed_left_join_probe_from_frame_stream,
    release_compiled_mixed_left_join_probe_state,
};
use one_shot::{
    OneShotEquiJoinFrameStreamProbeState, execute_compiled_one_shot_equi_join_materialize,
    execute_compiled_one_shot_equi_join_probe_from_frame_stream, execute_compiled_one_shot_equi_join_probe_from_stream,
    one_shot_equi_join_frame_stream_probe_to_payload,
};
pub(in crate::exec::bounded) use partitioned::{
    BoundedJoinBuildInputStream, BoundedJoinBuildStreamObservation, BoundedJoinBuildStreamOutcome,
    PartitionedJoinBuildCommitment, consume_bounded_join_build_input_stream, partitioned_join_build_commitment,
};
use partitioned::{
    execute_compiled_partitioned_hash_equi_join_materialize,
    execute_compiled_partitioned_hash_equi_join_materialize_preserving_stream,
    execute_compiled_partitioned_hash_equi_join_probe_from_frame_stream,
    execute_compiled_partitioned_hash_equi_join_probe_from_split_build_frame_stream,
    execute_compiled_partitioned_hash_equi_join_probe_from_stream,
    execute_compiled_partitioned_mixed_keyed_residual_join_materialize,
    execute_compiled_partitioned_mixed_keyed_residual_join_materialize_preserving_stream,
    parallel_partitioned_lazy_drain, partitioned_hash_equi_join_frame_stream_probe_to_payload,
    partitioned_parallel_probe_supported, partitioned_streaming_probe_supported, release_partitioned_join_side,
};
use prepared::execute_compiled_prepared_join_materialize;
use residual::materialize_compiled_one_shot_residual_output;
use streamed::{
    execute_compiled_streamed_one_shot_join_materialize, execute_compiled_streamed_partitioned_hash_join_materialize,
    record_one_shot_join_chunk_metrics, record_streamed_one_shot_join_metrics,
    record_streamed_partitioned_hash_join_metrics,
};
pub(in crate::exec) use transaction::JoinMaterializationTxn;

/// Output of the build task, retained across every probe of the same join.
///
/// Which variant the build produces follows the planner's `JoinStrategy`.
/// `Hash` and `Filtered` already hold a cuDF hash table or membership
/// structure; `Prepared` holds only the projected build frame because its
/// strategy needs both inputs in one cuDF call; `Partitioned` holds the build
/// side already split by key hash while its input streamed in.
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

/// Output of the probe task, consumed by the materialize task.
///
/// Three shapes recur across the strategies. A plain variant (`Hash`,
/// `Filtered`, `Prepared`, `MixedLeftMatched`) holds gather maps or match
/// state over whole inputs and materializes in one call. A `*Streamed` variant
/// holds output frames already materialized one probe chunk at a time, plus
/// the build state only so that materialize can release it. A `*LazyStream`
/// variant holds the undrained input stream together with the build; output
/// is produced on pull, and the build is released when the stream ends.
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
    /// The split build was already released by the probe; only the output
    /// frames and the metrics summary remain.
    PartitionedStreamed {
        build_side: JoinSide,
        build_rows: u64,
        build_bytes: u64,
        outputs: Vec<ResidentGpuFrame>,
        summary: StreamedPartitionedJoinSummary,
    },
    PartitionedLazyStream(PartitionedHashEquiJoinFrameStreamProbeState),
}

/// Build side of a strategy that joins both inputs in one cuDF call (one-shot,
/// partitioned, mixed, keyless and cross). The frame holds only the columns
/// `compiled_join_input_columns` selects, so build retention is bounded by the
/// projection and does not grow with the input width.
pub(in crate::exec::bounded) struct BoundedCompiledPreparedJoinBuildState {
    strategy: JoinStrategy,
    strategy_options: KeyedHashJoinStrategyOptions,
    side: JoinSide,
    input: BoundedJoinSideFrame,
    /// A semi or anti join building its left side runs the mixed residual
    /// join whose probe keeps per-left-row match state.
    left_match: bool,
}

impl BoundedCompiledPreparedJoinBuildState {
    fn mixed_left_match(&self) -> bool {
        self.left_match && self.strategy == JoinStrategy::MixedKeyedResidual
    }
}

pub(in crate::exec::bounded) struct BoundedCompiledPreparedJoinProbeState {
    strategy: JoinStrategy,
    strategy_options: KeyedHashJoinStrategyOptions,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
}

/// Semi or anti mixed join after every probe chunk has been folded into
/// cuDF's left-row match state. `finalized` is the complement over the whole
/// left input, which is why this shape cannot stream its output per chunk.
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

/// Join shapes whose probe chunks are independent of one another.
///
/// For an inner join, or a left join whose build is the right input, each
/// probe row's output depends only on the build table, so chunk outputs
/// concatenate into the join result. A left join built on its left input and
/// every full join must emit the build rows no probe chunk matched, which needs
/// a matched complement over all chunks and keeps those shapes on the one-call
/// path.
pub(in crate::exec::bounded) fn stateless_streaming_probe_supported(node: &JoinNode) -> bool {
    matches!(node.spec.join_type, JoinType::Inner)
        || (node.spec.join_type == JoinType::Left && node.spec.build_side == JoinSide::Right)
}

/// Build side of a semi, anti or mark join: a membership structure over the
/// right input that each probe chunk tests its rows against.
///
/// Fields are visible across the join owner so the parallel probe path can
/// share one immutable build among lanes. `null_aware_probe` is decided once
/// from the complete right input, so every probe chunk applies the same
/// `NOT IN` rewrite. `timer` starts at the build and is read by materialize so
/// operator time covers build, every probe and the final concatenation.
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
    materialization_bytes: Option<u64>,
    output_reservation_bytes: Option<u64>,
    output_rows: u64,
    output_bytes: u64,
}

struct OneShotJoinChunkMetrics {
    build_rows: u64,
    build_bytes: u64,
    probe_rows: u64,
    probe_bytes: u64,
    index_bytes: u64,
    materialization_bytes: Option<u64>,
    output_reservation_bytes: Option<u64>,
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
    output_reservation_bytes: Option<u64>,
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
    output_reservation_bytes: Option<u64>,
    output_bytes: u64,
    output_rows: u64,
}

/// Result of a materialize task whose consumer accepts chunked output.
///
/// `Stream` carries chunks that already exist on the device; `LazyStream`
/// carries a producer that still owns the build and probes on pull, so a
/// consumer that stops early must release it through the payload.
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
            materialization_bytes: Some(0),
            output_reservation_bytes: Some(0),
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
        materialization_bytes: Option<u64>,
        output_reservation_bytes: Option<u64>,
        output_rows: u64,
        output_bytes: u64,
    ) {
        self.probe_rows = self.probe_rows.saturating_add(probe_rows);
        self.probe_bytes = self.probe_bytes.saturating_add(probe_bytes);
        self.index_bytes = self.index_bytes.saturating_add(index_bytes);
        self.materialization_bytes = sum_bounds(self.materialization_bytes, materialization_bytes);
        self.output_reservation_bytes = sum_bounds(self.output_reservation_bytes, output_reservation_bytes);
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
            output_reservation_bytes: Some(0),
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
        self.output_reservation_bytes = sum_bounds(self.output_reservation_bytes, output.output_reservation_bytes);
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

/// Lazy producer that probes a reusable hash build with each pulled input
/// frame. One input chunk can yield several output slices, queued in
/// `pending_outputs` and yielded before the next input is pulled. The
/// build and the input stream are released when the input is exhausted and
/// again, idempotently, by the payload's `release`.
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

/// Runs the build task over a resident input frame.
///
/// Reusable and filtered strategies build their cuDF structure here; the
/// remaining strategies only project the build side and defer the join to the
/// probe or materialize task. `input` is consumed on every path.
pub(in crate::exec::bounded) fn execute_bounded_compiled_join_build_from_resident_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    form: JoinForm,
    strategy: JoinStrategy,
    strategy_options: KeyedHashJoinStrategyOptions,
    input: ResidentGpuFrame,
) -> Result<BoundedCompiledJoinBuildState> {
    match strategy {
        JoinStrategy::ReusableHashEqui => {
            execute_bounded_root_hash_join_build_from_resident_frame(execution, node, input)
                .map(BoundedCompiledJoinBuildState::Hash)
        },
        JoinStrategy::FilteredSemiAnti => {
            execute_compiled_filtered_join_build_from_resident_frame(execution, node_id, node, input)
                .map(BoundedCompiledJoinBuildState::Filtered)
        },
        JoinStrategy::OneShotEqui
        | JoinStrategy::PartitionedHashEqui
        | JoinStrategy::PartitionedMixedKeyedResidual
        | JoinStrategy::KeylessConditional { .. }
        | JoinStrategy::KeylessOuterCross
        | JoinStrategy::ProjectedCross
        | JoinStrategy::MixedKeyedResidual => {
            let side = node.spec.build_side;
            let columns = compiled_join_input_columns(node, strategy, side)?;
            let input = prepare_bounded_join_side_input_from_resident_frame(execution, input, &columns, side)?;
            Ok(BoundedCompiledJoinBuildState::Prepared(BoundedCompiledPreparedJoinBuildState {
                strategy,
                strategy_options,
                side,
                input,
                left_match: form == JoinForm::KeyedResidual { left_match: true },
            }))
        },
    }
}

/// Runs the probe task over one resident probe frame.
///
/// Partitioned builds and mixed semi/anti builds are forwarded to the frame
/// stream path wrapped as a one-frame stream, so each of those strategies has
/// one probe implementation. On `Err` the build state has been released.
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
            if build.mixed_left_match() {
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
/// `lazy_output` says the join output is consumed as a stream. Stateless
/// probes pull leaf chunks on demand and yield outputs one by one. Outputs are
/// not collected as a whole before the materialize task runs. Partitioned
/// build sides always probe lazily: their partition outputs are already
/// bounded per probe chunk.
///
/// A leaf with zero planned chunks (every row group pruned) is materialized
/// into its empty frame first, so the schema still reaches the join. On `Err`
/// the build state has been released.
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
            build.mixed_left_match()
                || (lazy_output
                    && stateless_streaming_probe_supported(node)
                    && matches!(build.strategy, JoinStrategy::OneShotEqui | JoinStrategy::PartitionedHashEqui)
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
        // stream through the filtered build the same way a frame stream does.
        // The whole leaf is not materialized first.
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
            if build.strategy == JoinStrategy::OneShotEqui && stateless_streaming_probe_supported(node) =>
        {
            execute_compiled_one_shot_equi_join_probe_from_stream(execution, node_id, node, build, stream)
        },
        BoundedCompiledJoinBuildState::Prepared(build)
            if build.strategy == JoinStrategy::PartitionedHashEqui
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
            if build.strategy == JoinStrategy::MixedKeyedResidual && node.spec.join_type == JoinType::Inner =>
        {
            execute_compiled_mixed_join_probe_from_stream(execution, node_id, node, build, stream)
        },
        other => {
            let input = materialize_bounded_join_leaf_local_parquet_source_stream(execution, stream)?;
            execute_bounded_compiled_join_probe_from_resident_frame(execution, node_id, node, other, input)
        },
    }
}

/// Runs the probe task over a resident frame stream.
///
/// An eager stream with more than one frame probes its chunks in parallel on
/// execution lanes when the strategy's outputs concatenate chunk by chunk
/// (`probe_parallel`); `outer` lends the enclosing compiled run's lane
/// dispatcher for that. Lazy streams and single frames keep a serial or
/// pull-driven probe. Strategies that need both whole inputs drain the stream
/// into one frame first and record why in the join strategy decision. On
/// `Err` the build state and the stream have been released.
pub(in crate::exec::bounded) fn execute_bounded_compiled_join_probe_from_frame_stream(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    build: BoundedCompiledJoinBuildState,
    mut inputs: ResidentFrameStreamPayload,
    outer: Option<&AggregateLaneDispatch>,
) -> Result<BoundedCompiledJoinProbeState> {
    match build {
        BoundedCompiledJoinBuildState::Prepared(build) if build.mixed_left_match() => {
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
            if build.strategy == JoinStrategy::OneShotEqui && stateless_streaming_probe_supported(node) =>
        {
            execute_compiled_one_shot_equi_join_probe_from_frame_stream(execution, node, build, inputs)
        },
        BoundedCompiledJoinBuildState::Prepared(build)
            if build.strategy == JoinStrategy::PartitionedHashEqui
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

/// Runs the materialize task to a single resident output frame.
///
/// Lazy streams are drained here and streamed outputs are concatenated. In
/// both cases the build state is released before the concatenation runs, so
/// the table does not count toward the concatenation's peak allocation.
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

/// Materialize for a consumer that accepts chunked output, such as a
/// downstream aggregate or a streaming resident sink.
///
/// Streamed and lazy probe states are returned without concatenation, and the
/// whole-input partitioned strategies produce their partition outputs as a
/// stream. Any other state falls back to the single-frame materialize.
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
        BoundedCompiledJoinProbeState::Prepared(probe) if probe.strategy == JoinStrategy::PartitionedHashEqui => {
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
            if probe.strategy == JoinStrategy::PartitionedMixedKeyedResidual =>
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

/// Releases a build whose probe task will not run.
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

/// Releases a probe state whose materialize task will not run. Lazy streams
/// need the execution attempt because their producers release through it;
/// output frames held by the streamed variants are released by `Drop`.
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

/// Concatenates per-chunk outputs into one frame. A probe over at least one
/// chunk always yields a chunk output, possibly empty, so an empty list means
/// the driver skipped the probe. The function reports that case as a plan error
/// and does not return an empty frame.
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
    let exec_runtime = execution.context().runtime()?;
    let overrides = exec_runtime.exec();
    let timer = Instant::now();
    let output = match first_frame.on(overrides).concat_many(&rest_frame_refs) {
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

/// Columns one side must retain for `strategy`: its keys, the residual
/// predicate's inputs and its share of the output projection, in input order.
/// The reusable and filtered strategies select inside their own build step and
/// are rejected here as an invariant violation.
fn compiled_join_input_columns(node: &JoinNode, strategy: JoinStrategy, side: JoinSide) -> Result<Vec<String>> {
    match strategy {
        JoinStrategy::KeylessConditional { .. } => {
            let residual_columns = residual_input_columns(node)?;
            Ok(projected_input_columns(
                &[],
                &node.spec.output_projection,
                residual_columns.names(side).iter().map(String::as_str),
                side,
            ))
        },
        JoinStrategy::ProjectedCross => Ok(projected_output_columns(&node.spec.output_projection, side)),
        JoinStrategy::KeylessOuterCross => Ok(projected_output_columns(&node.spec.output_projection, side)),
        JoinStrategy::MixedKeyedResidual => {
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
        JoinStrategy::OneShotEqui | JoinStrategy::PartitionedHashEqui => {
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
        JoinStrategy::PartitionedMixedKeyedResidual => {
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
        JoinStrategy::ReusableHashEqui | JoinStrategy::FilteredSemiAnti => Err(bounded_invariant_violation(
            "compiled prepared join input columns require a prepared join strategy",
            "compiled_join_strategy_state_mismatch",
        )),
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

/// Sizing state of a keyless (conditional) join between its size pass and its
/// index pass. A full conditional join keeps cuDF's size-data object because
/// the index kernel reuses it; the other join types only need the output row
/// count to allocate their gather maps.
enum CompiledKeylessWork<'a> {
    Full { size_data: Box<cudf::DataFrameConditionalFullJoinSizeData<'a>> },
    Sized { output_size: usize, options: cudf::DataFramePredicateJoinOptions },
}

/// Lowered predicate and options of one keyed mixed join, oriented for cuDF.
///
/// cuDF builds a mixed join's equality hash table on the right table it
/// receives. An inner or full join planned to build on its left input runs
/// with the two inputs exchanged, and
/// [`CompiledMixedSizeData::materialize_indices_in`] returns the gather maps in
/// plan orientation.
struct CompiledMixedJoinCall {
    predicate: AstExpression,
    options: cudf::DataFrameMixedJoinOptions,
    inputs_exchanged: bool,
}

impl CompiledMixedJoinCall {
    fn lower(context: &ExecutionContext, node: &JoinNode) -> Result<Self> {
        let inputs_exchanged =
            node.spec.build_side == JoinSide::Left && matches!(node.spec.join_type, JoinType::Inner | JoinType::Full);
        let residual_columns = residual_input_columns(node)?;
        let residual_columns =
            if inputs_exchanged { residual_columns.with_exchanged_cudf_inputs() } else { residual_columns };
        let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
        let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
        let predicate = lower_join_residual_predicate(context, node, &residual_columns)?;
        let options = dataframe_mixed_join_options(&left_keys, &right_keys, &residual_columns, node.spec.null_equality);
        Ok(Self { predicate, options, inputs_exchanged })
    }

    fn size_data<'a>(
        &'a self,
        node: &JoinNode,
        left: &DataFrame,
        right: &DataFrame,
        overrides: Exec<'_>,
    ) -> Result<CompiledMixedSizeData<'a>> {
        let (cudf_left, cudf_right) = if self.inputs_exchanged { (right, left) } else { (left, right) };
        let (predicate, options) = (&self.predicate, self.options.clone());
        let pair = |size_data| CompiledMixedSizeData::Pair { size_data, inputs_exchanged: self.inputs_exchanged };
        match node.spec.join_type {
            JoinType::Inner => {
                cudf_left.on(overrides).mixed_inner_join_size_data(cudf_right, predicate, options).map(pair)
            },
            JoinType::Left => {
                cudf_left.on(overrides).mixed_left_join_size_data(cudf_right, predicate, options).map(pair)
            },
            JoinType::Full => {
                cudf_left.on(overrides).mixed_full_join_size_data(cudf_right, predicate, options).map(pair)
            },
            JoinType::LeftSemi => cudf_left
                .on(overrides)
                .mixed_left_semi_join_size_data(cudf_right, predicate, options)
                .map(CompiledMixedSizeData::LeftOnly),
            JoinType::LeftAnti { .. } => cudf_left
                .on(overrides)
                .mixed_left_anti_join_size_data(cudf_right, predicate, options)
                .map(CompiledMixedSizeData::LeftOnly),
            JoinType::Cross | JoinType::LeftMark { .. } => unreachable!("validated by the mixed join strategy"),
        }
        .map_err(|error| join_index_error(error, node))
    }
}

enum CompiledMixedSizeData<'a> {
    /// `inputs_exchanged` marks size data computed with the plan's right input
    /// as cuDF's left table.
    Pair {
        size_data: cudf::DataFrameMixedJoinSizeData<'a>,
        inputs_exchanged: bool,
    },
    LeftOnly(cudf::DataFrameMixedSingleSideJoinSizeData<'a>),
}

impl CompiledMixedSizeData<'_> {
    fn output_size(&self) -> usize {
        match self {
            Self::Pair { size_data, .. } => size_data.output_size(),
            Self::LeftOnly(size_data) => size_data.output_size(),
        }
    }

    fn device_allocated_size_bytes(&self, node: &JoinNode) -> Result<u64> {
        let result = match self {
            Self::Pair { size_data, .. } => size_data.device_allocated_size_bytes(),
            Self::LeftOnly(size_data) => size_data.device_allocated_size_bytes(),
        };
        result.map(|bytes| bytes as u64).map_err(|error| {
            cudf_query_error(
                "query_engine.bounded.join.mixed_size_data_bytes",
                "failed to observe compiled bounded mixed join size-data memory",
                error,
            )
            .with_fact("join_type", &node.spec.join_type)
        })
    }

    fn materialize_indices_in(self, overrides: Exec<'_>, node: &JoinNode) -> Result<CompiledJoinIndices> {
        match self {
            Self::Pair { size_data, inputs_exchanged } => size_data
                .materialize_indices(overrides)
                .map(|indices| {
                    let (cudf_left, cudf_right) = (indices.left().clone(), indices.right().clone());
                    if inputs_exchanged {
                        CompiledJoinIndices::Pair { left: cudf_right, right: cudf_left }
                    } else {
                        CompiledJoinIndices::Pair { left: cudf_left, right: cudf_right }
                    }
                })
                .map_err(|error| join_index_error(error, node)),
            Self::LeftOnly(size_data) => size_data
                .materialize_indices(overrides)
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

/// Computes the gather maps of a one-shot equality join.
///
/// The hash table is built for this call only; cuDF's size pass runs first and
/// the probe then allocates the gather maps at their exact length and does not
/// grow them. The returned indices do not depend on the table, which is
/// dropped on return. Indices come back in plan orientation whichever side
/// was built.
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
    overrides: Exec<'_>,
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
            .with_fact("join_type", &node.spec.join_type)
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
    overrides: Exec<'_>,
) -> Result<usize> {
    match node.spec.join_type {
        JoinType::Inner => hash_join.inner_probe_size(probe, probe_keys, overrides),
        JoinType::Left => hash_join.left_probe_size(probe, probe_keys, overrides),
        JoinType::Full => hash_join.full_probe_size(probe, probe_keys, overrides),
        JoinType::LeftSemi | JoinType::LeftAnti { .. } | JoinType::LeftMark { .. } | JoinType::Cross => {
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
    overrides: Exec<'_>,
) -> Result<cudf::DataFrameHashJoinPairIndices> {
    match node.spec.join_type {
        JoinType::Inner => hash_join.inner_probe_with_output_size(probe, probe_keys, output_size, overrides),
        JoinType::Left => hash_join.left_probe_with_output_size(probe, probe_keys, output_size, overrides),
        JoinType::Full => hash_join.full_probe_with_output_size(probe, probe_keys, output_size, overrides),
        JoinType::LeftSemi | JoinType::LeftAnti { .. } | JoinType::LeftMark { .. } | JoinType::Cross => {
            unreachable!("validated above")
        },
    }
    .map_err(|error| join_index_error(error, node))
}

/// Gather maps in plan orientation. `LeftOnly` is the semi and anti shape,
/// whose output gathers from the left input alone.
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
                    .with_fact("join_type", &node.spec.join_type)
                })?;
                let right_bytes = right.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
                    cudf_query_error(
                        "query_engine.bounded.join.gather_map_bytes",
                        "failed to observe compiled bounded join right gather-map memory",
                        error,
                    )
                    .with_fact("join_type", &node.spec.join_type)
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
                    .with_fact("join_type", &node.spec.join_type)
                })
            },
        }
    }

    /// Closed-form bound on the gathered output, or `None` when some projected
    /// column's dtype has no bound. The bound is reported as the output
    /// reservation in the join metrics; it reserves nothing.
    fn output_upper_bound_size_bytes(
        &self,
        execution: &mut BoundedExecutionAttempt<'_>,
        node_id: NodeId,
        left: &DataFrame,
        right: &DataFrame,
        expected_output_rows: u64,
        projection: &[JoinOutputColumn],
    ) -> Result<Option<u64>> {
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
                Ok(sum_bounds(left_bytes, right_bytes))
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
        overrides: Exec<'_>,
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
/// Inputs of the shared gather step: both prepared sides, the gather maps,
/// and the allocation facts the strategy observed on the way here.
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
    /// Started by the strategy before its size pass, so the operator time
    /// recorded here spans sizing, indices and the gather.
    timer: Instant,
    /// `Some` marks a one-shot strategy, which also records the one-shot
    /// metrics family.
    one_shot_index_bytes: Option<u64>,
}

/// Gathers the output projection through the indices and records the join
/// operator metrics.
///
/// Both sides and the indices live in a `JoinMaterializationTxn` from here on:
/// any failure rolls them back, and success releases them when the output
/// frame is committed. The gather's row count must equal the size pass's
/// `expected_output_rows`; the debug assertions check that the two cuDF passes
/// agreed.
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
    let exec_runtime = execution.context().runtime()?;
    let overrides = exec_runtime.exec();
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
        if let Some(bytes) = output_reservation_bytes {
            execution.metrics().record_one_shot_join_materialization_bytes(bytes);
        }
    }
    let (left, right) = txn.sides();
    let output = match txn.indices().materialize_output(
        &left.frame,
        &right.frame,
        overrides,
        &node.spec.output_projection,
        node.spec.join_type.clone(),
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
    use query_planning::plan::JoinSpec;

    use super::*;

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
