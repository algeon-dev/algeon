//! Hash-partitioned equi join (`PartitionedHashEqui`): keyed inner, left (build side right) and
//! full joins without a residual predicate whose build or probe side is too large for one pass.
//!
//! Three entry points cover the input shapes.
//!
//! - Two resident frames from a `Prepared` state (`..._materialize`): both sides are
//!   hash-partitioned and `run_partitioned_join_driver` joins the partition pairs. This path also
//!   serves full joins and left joins that build the left side, which cannot stream.
//! - A deferred Parquet leaf probe (`..._probe_from_stream`): the resident build frame is
//!   partitioned once, then each leaf chunk is partitioned to match and joined, and the outputs
//!   are collected into a `PartitionedStreamed` state.
//! - A frame-stream probe, or a build side that was partitioned while it streamed in
//!   (`..._probe_from_frame_stream`, `..._probe_from_split_build_frame_stream`): a lazy
//!   `PartitionedLazyStream` state. A pull joins one probe piece against every build partition
//!   (`PartitionedHashEquiJoinProbeStreamSource`). Several eager chunks can instead be probed in
//!   parallel lanes against one shared build (`parallel_partitioned_lazy_drain`).
//!
//! Only inner joins and left joins that preserve the probe side stream, because their output for
//! one probe chunk does not depend on the other chunks (`partitioned_streaming_probe_supported`).
//!
//! Every partition pair runs the one-shot sequence of `one_shot.rs`: hash table over the build
//! partition, probe-size call, probe with that size, gather. A resident join probes each partition
//! once, so its table is built for the pair and dropped before the gather. A streamed probe keeps
//! the table of each build partition from the first probe piece on and joins every later piece
//! against it (`hash_tables.rs`). A partition whose table is not kept defers its probe rows and
//! builds a table for each batch of them (`join_deferred_probe_batch`); a probe of one piece joins
//! it with a table built for the pair. A retry below a whole build partition builds its own table
//! for each child.
//!
//! The partition count follows `partitioned_join_partition_count` (see `partitioned/mod.rs`); for a
//! stream probe the probe size is an estimate, and an unknown estimate leaves the build size to
//! decide.
//!
//! A streamed probe holds its build side as one frame per partition, with the kept tables and the
//! deferred probe slices, until the probe finishes, and the current probe piece in partitioned
//! form until its last partition is joined or deferred. The partition outputs of one piece,
//! together with the deferred batches it caused to be joined, are concatenated into one frame. A
//! downstream consumer therefore sees at most as many frames as this join sees probe pieces, plus
//! one for the batches joined at the end of the input.

use std::sync::Arc;

use query_runtime::runtime::resident::GpuHashJoinBuildHandle;

use super::{
    BTreeMap, BoundedCompiledJoinMaterializeOutput, BoundedCompiledJoinProbeState,
    BoundedCompiledPartitionedJoinBuildState, BoundedCompiledPreparedJoinBuildState, BoundedExecutionAttempt,
    BoundedJoinLeafLocalParquetSourceStream, BoundedJoinSideFrame, BoundedPartitionRowPolicy,
    CompiledEqualityJoinPreflightIndices, DeferredProbePartitions, DeferredProbeSlices, GpuDataFrame, Instant,
    JoinMaterializationTxn, JoinNode, JoinSide, JoinStrategy, KeyedHashJoinStrategyOptions, NodeId,
    PARTITIONED_HASH_EQUI_STRATEGY, PartitionedHashEquiJoinFrameStreamProbeState, PartitionedJoinDriverPlan,
    PartitionedJoinHashTables, PartitionedJoinPartitionOutput, PartitionedJoinProbeChunk, PartitionedJoinSide,
    ResidentFrameStreamPayload, ResidentFrameStreamSource, ResidentGpuFrame, Result, RetainedPartitionHashTables,
    StreamedPartitionedJoinSummary, VecDeque, bounded_invariant_violation, compiled_join_input_columns,
    compiled_keyed_equality_hash_join_options, concatenate_streamed_hash_join_outputs, cudf_query_error,
    hash_partition_join_side, join_keys_for_side, materialize_partitioned_join_partition_with_retry,
    observe_device_memory_bytes, partition_row_range, partitioned_join_partition_count,
    partitioned_join_target_bytes_for_context, partitioned_streaming_probe_supported,
    prepare_bounded_join_side_input_from_resident_frame, probe_compiled_keyed_equality_join_indices, probe_side,
    record_partitioned_join_operator_metrics, record_streamed_partitioned_hash_join_metrics,
    release_partitioned_join_side, run_partitioned_join_driver, single_partition_join_side,
    slice_partition_join_side_range, split_partitioned_join_side, stream_bounded_join_leaf_local_parquet_source_serial,
    validate_partitioned_hash_equi_join, with_partition_side,
};
use crate::exec::bounded::{compiled::lanes::AggregateLaneDispatch, join::hash::build_bounded_hash_join};

/// Joins two resident sides partition by partition and concatenates the partition outputs into one
/// frame.
pub(in crate::exec::bounded::join::compiled) fn execute_compiled_partitioned_hash_equi_join_materialize(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    strategy_options: KeyedHashJoinStrategyOptions,
) -> Result<ResidentGpuFrame> {
    match execute_compiled_partitioned_hash_equi_join_materialize_output(
        execution,
        node_id,
        node,
        left,
        right,
        strategy_options,
        false,
    )? {
        BoundedCompiledJoinMaterializeOutput::Frame(frame) => Ok(*frame),
        BoundedCompiledJoinMaterializeOutput::Stream(_) | BoundedCompiledJoinMaterializeOutput::LazyStream(_) => {
            unreachable!("non-preserving partitioned join returns frame")
        },
    }
}

/// Runs the same join but returns the partition outputs as separate frames
/// (`BoundedCompiledJoinMaterializeOutput::Stream`) so the consumer sees partition-sized frames.
/// The outputs are computed eagerly; only the final concatenation is skipped.
pub(in crate::exec::bounded::join::compiled) fn execute_compiled_partitioned_hash_equi_join_materialize_preserving_stream(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    strategy_options: KeyedHashJoinStrategyOptions,
) -> Result<BoundedCompiledJoinMaterializeOutput> {
    execute_compiled_partitioned_hash_equi_join_materialize_output(
        execution,
        node_id,
        node,
        left,
        right,
        strategy_options,
        true,
    )
}

/// Partitions both sides, joins every pair through the driver, and reports the join metrics.
///
/// The partition count comes from the two sides' bytes. The driver consumes both frames and
/// releases the partitioned sides on success and on error. Each partition is probed once, so no
/// table is kept. Metrics are recorded after concatenation; the reported output bytes then
/// reflect the concatenated frame when `preserve_stream` is false.
fn execute_compiled_partitioned_hash_equi_join_materialize_output(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    strategy_options: KeyedHashJoinStrategyOptions,
    preserve_stream: bool,
) -> Result<BoundedCompiledJoinMaterializeOutput> {
    validate_partitioned_hash_equi_join(node)?;
    let partition_count = partitioned_join_partition_count(execution.context(), left.bytes, right.bytes);
    let plan = PartitionedJoinDriverPlan {
        node_id,
        node,
        strategy: PARTITIONED_HASH_EQUI_STRATEGY,
        strategy_options,
        partition_count,
        partition_target_bytes: partitioned_join_target_bytes_for_context(execution.context()),
    };
    let mut hash_table_builds = 0;
    let mut result =
        run_partitioned_join_driver(execution, plan, left, right, |execution, node_id, node, left, right, _| {
            materialize_partitioned_hash_equi_join_partition(
                execution,
                node_id,
                node,
                left,
                right,
                None,
                &mut hash_table_builds,
            )
        })?;
    result.summary.hash_table_build_count = hash_table_builds;
    let mut stream_outputs = Some(result.outputs);
    let output = if preserve_stream {
        None
    } else {
        let outputs = stream_outputs.take().expect("partitioned join outputs should be available");
        let output = concatenate_streamed_hash_join_outputs(execution, node_id, outputs)?;
        result.summary.observe_concatenated_output(&output);
        Some(output)
    };
    record_partitioned_join_operator_metrics(plan, result.original, &result.summary, execution.metrics());
    if let Some(output) = output {
        Ok(BoundedCompiledJoinMaterializeOutput::Frame(Box::new(output)))
    } else {
        Ok(BoundedCompiledJoinMaterializeOutput::Stream(
            stream_outputs.expect("partitioned stream outputs should be available"),
        ))
    }
}

/// Joins one partition pair with the one-shot sequence and returns its output with sizing facts.
///
/// `left` and `right` are the driver's slices of the stored partitions. A split partition read in
/// full shares its columns with the stored partition and any other slice is a copy. The transaction
/// therefore releases only this call's references, and the stored partitions stay intact for a
/// retry after a retryable resource error. Every failure path rolls the transaction back before the
/// error returns. `output_reservation_bytes` is an upper bound recorded for observation; nothing is
/// reserved.
///
/// `kept_table` is the table kept over the whole build partition that the build slice is. Without
/// one, a table is built over the build slice for this pair, counted in `hash_table_builds` and
/// the query's hash-join build count, and dropped on return.
fn materialize_partitioned_hash_equi_join_partition(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    kept_table: Option<&GpuHashJoinBuildHandle>,
    hash_table_builds: &mut u64,
) -> Result<PartitionedJoinPartitionOutput> {
    let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
    let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
    let build_side = node.spec.build_side;
    let exec_runtime = execution.context().runtime()?;
    let overrides = exec_runtime.exec();
    let timer = Instant::now();
    let mut txn = JoinMaterializationTxn::with_sides(left, right);
    let (left, right) = txn.sides();
    let (build, build_keys, probe, probe_keys) = match build_side {
        JoinSide::Left => (left, &left_keys, right, &right_keys),
        JoinSide::Right => (right, &right_keys, left, &left_keys),
    };
    let preflight_indices = match kept_table {
        Some(hash_join) => probe_compiled_keyed_equality_join_indices(
            node,
            hash_join.hash_join(),
            build_side,
            &probe.frame,
            probe_keys,
            overrides,
        ),
        None => match build_bounded_hash_join(
            execution,
            node,
            &build.frame,
            build_keys,
            compiled_keyed_equality_hash_join_options(node),
            build_side,
            overrides,
            false,
        ) {
            Ok((hash_join, _)) => {
                *hash_table_builds += 1;
                execution.metrics().record_hash_join_build();
                probe_compiled_keyed_equality_join_indices(
                    node,
                    hash_join.hash_join(),
                    build_side,
                    &probe.frame,
                    probe_keys,
                    overrides,
                )
            },
            Err(error) => Err(error),
        },
    };
    let preflight_indices = match preflight_indices {
        Ok(indices) => indices,
        Err(error) => {
            txn.rollback();
            return Err(error);
        },
    };
    let CompiledEqualityJoinPreflightIndices { indices, index_bytes, output_rows } = preflight_indices;
    txn.set_indices(indices);
    let (left, right) = txn.sides();
    let output_reservation_bytes = match txn.indices().output_upper_bound_size_bytes(
        execution,
        node_id,
        &left.frame,
        &right.frame,
        output_rows,
        &node.spec.output_projection,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            txn.rollback();
            return Err(error);
        },
    };
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
    let output_bytes = match observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), execution.metrics()) {
        Ok(bytes) => bytes,
        Err(error) => {
            txn.rollback();
            return Err(error);
        },
    };
    Ok(PartitionedJoinPartitionOutput {
        frame: txn.commit_frame(GpuDataFrame::new(output), output_bytes),
        index_bytes,
        mixed_size_data_bytes: 0,
        output_reservation_bytes,
        output_bytes,
        output_rows,
    })
}

/// Where a probe piece's rows of a partition without a kept table go.
enum ProbeDeferral<'a> {
    /// Joined with the piece, against a table built for the pair. A probe of one piece uses this.
    Off,
    /// Held by the probe owner and joined in batches (`DeferredProbePartitions`).
    Held(&'a mut DeferredProbePartitions),
    /// Returned by an execution lane to the coordinator, which holds them.
    Export(&'a mut DeferredProbeSlices),
}

/// Joins one hash-partitioned probe piece against the build partitions and returns the piece's
/// output as one frame, or `None` when the piece produced no output because all of its rows were
/// deferred.
///
/// Each partition with a kept table, and every partition when `deferral` is `Off`, runs under the
/// driver's retry. A partition without a kept table otherwise has its rows copied out and deferred;
/// with `Held`, the batches the arriving slices push past the deferred share are joined here and
/// their outputs belong to this piece's frame. The partition outputs are concatenated, so the
/// concatenation holds the piece's output twice at its peak. Table builds are added to `summary`
/// whatever the outcome; the output counters and the deferred slices only once the piece's frame
/// exists, because a leaf chunk that fails is read again in smaller pieces.
fn join_partitioned_probe_piece(
    execution: &mut BoundedExecutionAttempt<'_>,
    plan: PartitionedJoinDriverPlan<'_>,
    build_partitioned: &PartitionedJoinSide,
    tables: &RetainedPartitionHashTables,
    probe_partitioned: &PartitionedJoinSide,
    mut deferral: ProbeDeferral<'_>,
    summary: &mut StreamedPartitionedJoinSummary,
) -> Result<Option<ResidentGpuFrame>> {
    let (left_partitioned, right_partitioned) = match plan.node.spec.build_side {
        JoinSide::Left => (build_partitioned, probe_partitioned),
        JoinSide::Right => (probe_partitioned, build_partitioned),
    };
    let mut arriving =
        (!matches!(deferral, ProbeDeferral::Off)).then(|| DeferredProbeSlices::new(plan.partition_count));
    let mut hash_table_builds = 0;
    let mut outputs = Vec::new();
    let mut joined = Ok(());
    for partition in 0..plan.partition_count {
        if let Some(arriving) = arriving.as_mut()
            && tables.table(partition).is_none()
        {
            match partition_row_range(probe_partitioned, partition).and_then(|range| {
                slice_partition_join_side_range(
                    execution,
                    probe_side(plan.node.spec.build_side),
                    probe_partitioned,
                    partition,
                    range,
                )
            }) {
                Ok(slice) => arriving.push(partition, slice),
                Err(error) => {
                    joined = Err(error);
                    break;
                },
            }
            continue;
        }
        let mut materialize = |execution: &mut BoundedExecutionAttempt<'_>,
                               node_id: NodeId,
                               node: &JoinNode,
                               left: BoundedJoinSideFrame,
                               right: BoundedJoinSideFrame,
                               whole_build_partition: Option<usize>| {
            materialize_partitioned_hash_equi_join_partition(
                execution,
                node_id,
                node,
                left,
                right,
                whole_build_partition.and_then(|partition| tables.table(partition)),
                &mut hash_table_builds,
            )
        };
        match materialize_partitioned_join_partition_with_retry(
            execution,
            plan,
            left_partitioned,
            right_partitioned,
            partition,
            &mut materialize,
        ) {
            Ok(partition_outputs) => outputs.extend(partition_outputs),
            Err(error) => {
                joined = Err(error);
                break;
            },
        }
    }
    let mut admission = None;
    if joined.is_ok()
        && let ProbeDeferral::Held(deferred) = &deferral
        && let Some(arriving) = arriving.take()
    {
        match deferred.admit(arriving, |partition, slices| {
            join_deferred_probe_batch(execution, plan, build_partitioned, partition, slices, &mut hash_table_builds)
        }) {
            Ok((batch_outputs, pending)) => {
                outputs.extend(batch_outputs);
                admission = Some(pending);
            },
            Err(error) => joined = Err(error),
        }
    }
    summary.record_hash_table_builds(hash_table_builds);
    joined?;
    let frame = commit_partitioned_probe_outputs(execution, plan.node_id, outputs, summary)?;
    match &mut deferral {
        ProbeDeferral::Held(deferred) => {
            if let Some(admission) = admission {
                deferred.commit(admission)?;
            }
        },
        ProbeDeferral::Export(exported) => {
            if let Some(arriving) = arriving {
                exported.append(arriving);
            }
        },
        ProbeDeferral::Off => {},
    }
    Ok(frame)
}

/// Joins one batch of deferred probe slices of `partition` against that build partition, with a
/// table built for the batch, under the driver's retry.
///
/// The slices are concatenated into one probe frame unless there is only one. The slices and the
/// concatenation are both resident until the concatenation returns, and the probe frame until the
/// batch is joined.
fn join_deferred_probe_batch(
    execution: &mut BoundedExecutionAttempt<'_>,
    plan: PartitionedJoinDriverPlan<'_>,
    build_partitioned: &PartitionedJoinSide,
    partition: usize,
    slices: Vec<BoundedJoinSideFrame>,
    hash_table_builds: &mut u64,
) -> Result<Vec<PartitionedJoinPartitionOutput>> {
    let probe = concatenate_deferred_probe_slices(execution, plan, partition, slices)?;
    let probe_partitioned = single_partition_join_side(probe, partition, plan.partition_count);
    let (left_partitioned, right_partitioned) = match plan.node.spec.build_side {
        JoinSide::Left => (build_partitioned, &probe_partitioned),
        JoinSide::Right => (&probe_partitioned, build_partitioned),
    };
    let outputs = materialize_partitioned_join_partition_with_retry(
        execution,
        plan,
        left_partitioned,
        right_partitioned,
        partition,
        &mut |execution: &mut BoundedExecutionAttempt<'_>,
              node_id: NodeId,
              node: &JoinNode,
              left: BoundedJoinSideFrame,
              right: BoundedJoinSideFrame,
              _whole_build_partition: Option<usize>| {
            materialize_partitioned_hash_equi_join_partition(
                execution,
                node_id,
                node,
                left,
                right,
                None,
                hash_table_builds,
            )
        },
    );
    release_partitioned_join_side(probe_partitioned);
    outputs
}

fn concatenate_deferred_probe_slices(
    execution: &mut BoundedExecutionAttempt<'_>,
    plan: PartitionedJoinDriverPlan<'_>,
    partition: usize,
    mut slices: Vec<BoundedJoinSideFrame>,
) -> Result<BoundedJoinSideFrame> {
    if slices.len() == 1 {
        return Ok(slices.pop().expect("one deferred probe slice checked"));
    }
    let slice_count = slices.len();
    let rows = slices.iter().map(|slice| slice.rows).sum();
    let frames = slices.into_iter().map(|slice| slice.frame).collect::<Vec<_>>();
    let Some((first, rest)) = frames.split_first() else {
        return Err(bounded_invariant_violation(
            "a deferred partitioned join probe batch has no slices",
            "partitioned_join_deferred_probe_empty",
        )
        .with_fact("partition", partition));
    };
    let rest = rest.iter().collect::<Vec<_>>();
    let exec_runtime = execution.context().runtime()?;
    let overrides = exec_runtime.exec();
    let frame = first.on(overrides).concat_many(&rest).map_err(|error| {
        with_partition_side(
            cudf_query_error(
                "query_engine.bounded.join.deferred_probe_concat",
                "failed to concatenate the deferred probe slices of a partitioned join partition",
                error,
            ),
            probe_side(plan.node.spec.build_side),
            "join_deferred_probe_concat",
        )
        .with_fact("node_id", plan.node_id)
        .with_fact("partition", partition)
        .with_fact("slice_count", slice_count)
    })?;
    let bytes = observe_device_memory_bytes(&GpuDataFrame::new(frame.clone()), execution.metrics())?;
    Ok(BoundedJoinSideFrame { frame, rows, bytes })
}

/// Joins every batch still deferred at the end of the probe input and returns their outputs as one
/// frame, or `None` when nothing was held.
fn join_deferred_probe_remainder(
    execution: &mut BoundedExecutionAttempt<'_>,
    plan: PartitionedJoinDriverPlan<'_>,
    build_partitioned: &PartitionedJoinSide,
    deferred: DeferredProbePartitions,
    summary: &mut StreamedPartitionedJoinSummary,
) -> Result<Option<ResidentGpuFrame>> {
    let mut hash_table_builds = 0;
    let outputs = deferred.join_all(|partition, slices| {
        join_deferred_probe_batch(execution, plan, build_partitioned, partition, slices, &mut hash_table_builds)
    });
    summary.record_hash_table_builds(hash_table_builds);
    commit_partitioned_probe_outputs(execution, plan.node_id, outputs?, summary)
}

/// Concatenates partition outputs into one frame and adds them to `summary` once the frame exists.
fn commit_partitioned_probe_outputs(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    outputs: Vec<PartitionedJoinPartitionOutput>,
    summary: &mut StreamedPartitionedJoinSummary,
) -> Result<Option<ResidentGpuFrame>> {
    if outputs.is_empty() {
        return Ok(None);
    }
    let mut committed = summary.clone();
    let frames = outputs
        .into_iter()
        .map(|output| {
            committed.record_output(&output);
            output.frame
        })
        .collect::<Vec<_>>();
    let frame = concatenate_streamed_hash_join_outputs(execution, node_id, frames)?;
    *summary = committed;
    Ok(Some(frame))
}

/// Partitions the resident build frame once and joins a deferred Parquet leaf chunk by chunk.
///
/// The partition count is sized from the build bytes and the leaf's estimated read bytes (zero when
/// unknown) and is at least the hash-frame-stream minimum. The build partitions are copied into
/// one frame each. When the leaf has more than one chunk their tables are kept, and the rows of
/// partitions without one are deferred. Each chunk is projected, hash-partitioned to the same
/// count, and joined as one probe piece (`join_partitioned_probe_piece`). The chunk's partitioned
/// copy is released after the piece. The leaf stream may call the consumer again with a smaller
/// piece of the same chunk after a retryable resource error. A chunk's output, summary counters
/// and deferred slices are therefore published only after the piece succeeded. A piece whose rows
/// were all deferred adds no output; the batches still deferred after the last chunk add one.
///
/// The build partitions, the kept tables and the deferred slices are released before returning,
/// on success and on error. The result is a `PartitionedStreamed` state that holds outputs and a
/// summary, with no GPU state besides the outputs.
pub(in crate::exec::bounded::join::compiled) fn execute_compiled_partitioned_hash_equi_join_probe_from_stream(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    build: BoundedCompiledPreparedJoinBuildState,
    stream: BoundedJoinLeafLocalParquetSourceStream,
) -> Result<BoundedCompiledJoinProbeState> {
    validate_partitioned_hash_equi_join(node)?;
    if !partitioned_streaming_probe_supported(node) {
        return Err(bounded_invariant_violation(
            "streamed compiled partitioned hash join probe supports inner joins and left joins with a right build side",
            "compiled_join_strategy_unsupported",
        )
        .with_fact("join_type", &node.spec.join_type)
        .with_fact("build_side", node.spec.build_side));
    }
    let build_side = build.side;
    let probe_side = probe_side(build_side);
    let build_rows = build.input.rows;
    let build_bytes = build.input.bytes;
    let probe_estimated_bytes = stream.estimated_source_read_bytes().unwrap_or(0);
    let partition_count =
        partitioned_join_partition_count(execution.context(), build.input.bytes, probe_estimated_bytes)
            .max(BoundedPartitionRowPolicy::default().hash_frame_stream_min_partitions());
    let partition_target_bytes = partitioned_join_target_bytes_for_context(execution.context());
    let build_keys = join_keys_for_side(node, build_side);
    let probe_keys = join_keys_for_side(node, probe_side);
    let exec_runtime = execution.context().runtime()?;
    let overrides = exec_runtime.exec();
    let build_partitioned =
        hash_partition_join_side(execution, build_side, build.input, &build_keys, partition_count, overrides)?;
    let build_partitioned = split_partitioned_join_side(execution, build_side, build_partitioned)?;
    let build_partitioned_bytes = build_partitioned.device_bytes();
    let mut outputs = Vec::new();
    let mut summary =
        StreamedPartitionedJoinSummary::new(partition_count, partition_target_bytes, build_partitioned_bytes);
    let plan = PartitionedJoinDriverPlan {
        node_id,
        node,
        strategy: PARTITIONED_HASH_EQUI_STRATEGY,
        strategy_options: build.strategy_options,
        partition_count,
        partition_target_bytes,
    };
    execution.metrics().record_join_strategy_decision(
        node_id,
        &BTreeMap::from([
            ("join_strategy", "partitioned_hash_equi".to_string()),
            ("partition_count", partition_count.to_string()),
            ("partition_target_bytes", partition_target_bytes.to_string()),
            ("build_partition_batch_count", partition_count.to_string()),
            ("build_partitioned_bytes", build_partitioned_bytes.to_string()),
        ]),
    );

    let probe_columns = match compiled_join_input_columns(node, JoinStrategy::PartitionedHashEqui, probe_side) {
        Ok(columns) => columns,
        Err(error) => {
            release_partitioned_join_side(build_partitioned);
            return Err(error);
        },
    };
    let mut hash_tables = PartitionedJoinHashTables::undecided();
    let tables = match hash_tables.decide(
        execution,
        node,
        build_side,
        &build_partitioned,
        stream.chunk_count() > 1,
        &mut summary,
    ) {
        Ok(tables) => tables,
        Err(error) => {
            release_partitioned_join_side(build_partitioned);
            return Err(error);
        },
    };
    let mut deferred =
        if stream.chunk_count() > 1 { DeferredProbePartitions::new(execution, partition_count) } else { None };
    let stream_result = stream_bounded_join_leaf_local_parquet_source_serial(execution, stream, |input, execution| {
        let probe = prepare_bounded_join_side_input_from_resident_frame(execution, input, &probe_columns, probe_side)?;
        let probe_rows = probe.rows;
        let probe_bytes = probe.bytes;
        let probe_partitioned =
            hash_partition_join_side(execution, probe_side, probe, &probe_keys, partition_count, overrides)?;
        let probe_partitioned_bytes = probe_partitioned.device_bytes();
        let output = join_partitioned_probe_piece(
            execution,
            plan,
            &build_partitioned,
            &tables,
            &probe_partitioned,
            deferred.as_mut().map_or(ProbeDeferral::Off, ProbeDeferral::Held),
            &mut summary,
        );
        release_partitioned_join_side(probe_partitioned);
        outputs.extend(output?);
        summary.record_probe_partitioned_chunk(probe_rows, probe_bytes, probe_partitioned_bytes);
        Ok(())
    });
    let remainder = match (stream_result, deferred) {
        (Ok(()), Some(deferred)) => {
            join_deferred_probe_remainder(execution, plan, &build_partitioned, deferred, &mut summary)
        },
        (Ok(()), None) => Ok(None),
        (Err(error), _) => Err(error),
    };
    release_partitioned_join_side(build_partitioned);
    outputs.extend(remainder?);
    execution.metrics().record_join_strategy_decision(
        node_id,
        &BTreeMap::from([
            ("join_strategy", "partitioned_hash_equi".to_string()),
            ("partition_count", partition_count.to_string()),
            ("partition_target_bytes", partition_target_bytes.to_string()),
            ("probe_partition_batch_count", summary.probe_partition_batch_count.to_string()),
            ("probe_partitioned_bytes", summary.probe_partitioned_bytes.to_string()),
        ]),
    );
    Ok(BoundedCompiledJoinProbeState::PartitionedStreamed { build_side, build_rows, build_bytes, outputs, summary })
}

/// Partitions the resident build frame once and returns a lazy `PartitionedLazyStream` state. No
/// probe chunk is joined yet.
///
/// The inputs' largest frame size stands in for the probe size when sizing the partition count.
/// The partitions are copied into one frame each, which their kept tables refer to. If that copy
/// fails or the probe columns cannot be resolved, the inputs and the build partitions are released
/// here.
pub(in crate::exec::bounded::join::compiled) fn execute_compiled_partitioned_hash_equi_join_probe_from_frame_stream(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    build: BoundedCompiledPreparedJoinBuildState,
    inputs: ResidentFrameStreamPayload,
) -> Result<BoundedCompiledJoinProbeState> {
    validate_partitioned_hash_equi_join(node)?;
    if !partitioned_streaming_probe_supported(node) {
        return Err(bounded_invariant_violation(
            "streamed compiled partitioned hash join probe supports inner joins and left joins with a right build side",
            "compiled_join_strategy_unsupported",
        )
        .with_fact("join_type", &node.spec.join_type)
        .with_fact("build_side", node.spec.build_side));
    }
    let build_side = build.side;
    let probe_side = probe_side(build_side);
    let build_rows = build.input.rows;
    let build_bytes = build.input.bytes;
    // An unknown probe size leaves the build size to decide the partitions.
    let probe_estimated_bytes = inputs.max_frame_bytes_hint().unwrap_or(0);
    let partition_count =
        partitioned_join_partition_count(execution.context(), build.input.bytes, probe_estimated_bytes)
            .max(BoundedPartitionRowPolicy::default().hash_frame_stream_min_partitions());
    let partition_target_bytes = partitioned_join_target_bytes_for_context(execution.context());
    let build_keys = join_keys_for_side(node, build_side);
    let probe_keys = join_keys_for_side(node, probe_side);
    let exec_runtime = execution.context().runtime()?;
    let overrides = exec_runtime.exec();
    let build_partitioned =
        match hash_partition_join_side(execution, build_side, build.input, &build_keys, partition_count, overrides)
            .and_then(|partitioned| split_partitioned_join_side(execution, build_side, partitioned))
        {
            Ok(partitioned) => partitioned,
            Err(error) => {
                inputs.release_all(execution);
                return Err(error);
            },
        };
    let build_partitioned_bytes = build_partitioned.device_bytes();
    execution.metrics().record_join_strategy_decision(
        node_id,
        &BTreeMap::from([
            ("join_strategy", "partitioned_hash_equi".to_string()),
            ("partition_count", partition_count.to_string()),
            ("partition_target_bytes", partition_target_bytes.to_string()),
            ("build_partition_batch_count", partition_count.to_string()),
            ("build_partitioned_bytes", build_partitioned_bytes.to_string()),
        ]),
    );

    let probe_columns = match compiled_join_input_columns(node, JoinStrategy::PartitionedHashEqui, probe_side) {
        Ok(columns) => columns,
        Err(error) => {
            inputs.release_all(execution);
            release_partitioned_join_side(build_partitioned);
            return Err(error);
        },
    };
    Ok(BoundedCompiledJoinProbeState::PartitionedLazyStream(PartitionedHashEquiJoinFrameStreamProbeState {
        build_side,
        build_rows,
        build_bytes,
        build_partitioned,
        hash_tables: PartitionedJoinHashTables::undecided(),
        build_partitioned_bytes,
        inputs,
        partition_count,
        partition_target_bytes,
        strategy_options: build.strategy_options,
        probe_columns,
        probe_keys,
    }))
}

/// Returns a lazy probe over a build side that was partitioned while its input streamed in.
///
/// The partition count is fixed by the build, and probe chunks are partitioned to match it. Takes
/// ownership of the build partitions and the inputs: every early return releases both, and on
/// success they move into the returned `PartitionedLazyStream` state.
pub(in crate::exec::bounded::join::compiled) fn execute_compiled_partitioned_hash_equi_join_probe_from_split_build_frame_stream(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    build: BoundedCompiledPartitionedJoinBuildState,
    inputs: ResidentFrameStreamPayload,
) -> Result<BoundedCompiledJoinProbeState> {
    if let Err(error) = validate_partitioned_hash_equi_join(node) {
        inputs.release_all(execution);
        release_partitioned_join_side(build.partitioned);
        return Err(error);
    }
    if !partitioned_streaming_probe_supported(node) {
        inputs.release_all(execution);
        release_partitioned_join_side(build.partitioned);
        return Err(bounded_invariant_violation(
            "streamed compiled partitioned hash join probe supports inner joins and left joins with a right build side",
            "compiled_join_strategy_unsupported",
        )
        .with_fact("join_type", &node.spec.join_type)
        .with_fact("build_side", node.spec.build_side));
    }
    let build_side = build.side;
    let probe_side = probe_side(build_side);
    let probe_columns = match compiled_join_input_columns(node, JoinStrategy::PartitionedHashEqui, probe_side) {
        Ok(columns) => columns,
        Err(error) => {
            inputs.release_all(execution);
            release_partitioned_join_side(build.partitioned);
            return Err(error);
        },
    };
    let build_partitioned_bytes = build.partitioned.device_bytes();
    execution.metrics().record_join_strategy_decision(
        node_id,
        &BTreeMap::from([
            ("join_strategy", "partitioned_hash_equi".to_string()),
            ("partition_count", build.partition_count.to_string()),
            ("partition_target_bytes", build.partition_target_bytes.to_string()),
            ("build_partition_batch_count", build.partition_count.to_string()),
            ("build_input_chunks", build.input_chunks.to_string()),
            ("observed_build_rows", build.rows.to_string()),
            ("build_side_bytes", build.bytes.to_string()),
            ("build_partitioned_bytes", build_partitioned_bytes.to_string()),
        ]),
    );
    Ok(BoundedCompiledJoinProbeState::PartitionedLazyStream(PartitionedHashEquiJoinFrameStreamProbeState {
        build_side,
        build_rows: build.rows,
        build_bytes: build.bytes,
        build_partitioned: build.partitioned,
        hash_tables: PartitionedJoinHashTables::undecided(),
        build_partitioned_bytes,
        inputs,
        partition_count: build.partition_count,
        partition_target_bytes: build.partition_target_bytes,
        strategy_options: build.strategy_options,
        probe_columns,
        probe_keys: join_keys_for_side(node, probe_side),
    }))
}

/// Releases a lazy probe that will not be drained: the unread inputs, the build partitions and the
/// kept tables, whose charges return to the attempt.
pub(in crate::exec::bounded::join::compiled) fn release_partitioned_hash_equi_join_frame_stream_probe_state(
    execution: &mut BoundedExecutionAttempt<'_>,
    probe: PartitionedHashEquiJoinFrameStreamProbeState,
) {
    let PartitionedHashEquiJoinFrameStreamProbeState { build_partitioned, inputs, .. } = probe;
    inputs.release_all(execution);
    release_partitioned_join_side(build_partitioned);
}

/// Wraps a lazy probe state in a `ResidentFrameStreamPayload` whose producer joins one probe piece
/// per pull. Each yielded frame is one input piece's output, so the payload's frame size hint is the
/// inputs' hint.
pub(in crate::exec::bounded::join::compiled) fn partitioned_hash_equi_join_frame_stream_probe_to_payload(
    node_id: NodeId,
    node: &JoinNode,
    probe: PartitionedHashEquiJoinFrameStreamProbeState,
) -> ResidentFrameStreamPayload {
    let max_frame_bytes_hint = probe.inputs.max_frame_bytes_hint();
    let total_bytes_hint = probe.inputs.total_bytes_hint();
    let aggregate_num_rows = probe.inputs.num_rows();
    let num_columns = node.spec.output_projection.len();
    let source = PartitionedHashEquiJoinProbeStreamSource::new(node_id, node, probe, true);
    ResidentFrameStreamPayload::from_producer(
        max_frame_bytes_hint,
        total_bytes_hint,
        aggregate_num_rows,
        num_columns,
        Box::new(source),
    )
}

/// Producer behind a lazy partitioned probe. Each pull yields one frame: the output of the next
/// probe piece that produced any, or at the end of the input the batches still deferred.
///
/// Per probe piece, `load_next_probe_piece` reads the next chunk, prepares it and hash-partitions
/// it, and `join_partitioned_probe_piece` joins it against the build partitions. The first piece
/// decides the kept tables and whether rows of partitions without one are deferred; with no other
/// piece to follow, it keeps no table and defers nothing. A piece whose rows were all deferred
/// yields nothing, and the pull goes on to the next piece. A chunk that fails to partition under a
/// resource limit is split into row ranges (`pending_probe_chunk`) and its pieces are partitioned
/// and joined in order.
///
/// The source retains the build partitions, the kept tables, the deferred slices and the unread
/// inputs until `release`. The owning payload calls it after the stream drains, on early stop and
/// on error, and the source also releases itself at exhaustion and on its own errors. A source that
/// `reports_metrics` records the join metric row once, at exhaustion; a lane's source leaves its
/// summary to the coordinator and returns its deferred slices to it.
struct PartitionedHashEquiJoinProbeStreamSource {
    node_id: NodeId,
    node: JoinNode,
    build_side: JoinSide,
    build_rows: u64,
    build_bytes: u64,
    build_partitioned: Option<PartitionedJoinSide>,
    hash_tables: PartitionedJoinHashTables,
    inputs: Option<ResidentFrameStreamPayload>,
    partition_count: usize,
    partition_target_bytes: u64,
    strategy_options: KeyedHashJoinStrategyOptions,
    probe_columns: Vec<String>,
    probe_keys: Vec<String>,
    pending_probe_chunk: Option<PartitionedJoinProbeChunk>,
    deferral: ProbeSourceDeferral,
    summary: StreamedPartitionedJoinSummary,
    reports_metrics: bool,
    metrics_recorded: bool,
}

/// What a lazy partitioned probe does with the rows of partitions without a kept table.
enum ProbeSourceDeferral {
    /// Decided with the kept tables at the first probe piece.
    Undecided,
    Off,
    Held(DeferredProbePartitions),
    /// A lane's slices for the coordinator. They are the lane's output, so `release` keeps them.
    Export(DeferredProbeSlices),
}

impl ProbeSourceDeferral {
    fn piece_deferral(&mut self) -> ProbeDeferral<'_> {
        match self {
            Self::Undecided | Self::Off => ProbeDeferral::Off,
            Self::Held(deferred) => ProbeDeferral::Held(deferred),
            Self::Export(slices) => ProbeDeferral::Export(slices),
        }
    }
}

impl PartitionedHashEquiJoinProbeStreamSource {
    fn new(
        node_id: NodeId,
        node: &JoinNode,
        probe: PartitionedHashEquiJoinFrameStreamProbeState,
        reports_metrics: bool,
    ) -> Self {
        Self {
            node_id,
            node: node.clone(),
            build_side: probe.build_side,
            build_rows: probe.build_rows,
            build_bytes: probe.build_bytes,
            build_partitioned: Some(probe.build_partitioned),
            hash_tables: probe.hash_tables,
            inputs: Some(probe.inputs),
            partition_count: probe.partition_count,
            partition_target_bytes: probe.partition_target_bytes,
            strategy_options: probe.strategy_options,
            probe_columns: probe.probe_columns,
            probe_keys: probe.probe_keys,
            pending_probe_chunk: None,
            deferral: ProbeSourceDeferral::Undecided,
            summary: StreamedPartitionedJoinSummary::new(
                probe.partition_count,
                probe.partition_target_bytes,
                probe.build_partitioned_bytes,
            ),
            reports_metrics,
            metrics_recorded: false,
        }
    }

    fn record_success_metrics(&mut self, execution: &mut BoundedExecutionAttempt<'_>) {
        if !self.reports_metrics || self.metrics_recorded {
            return;
        }
        self.metrics_recorded = true;
        record_streamed_partitioned_hash_join_metrics(
            self.node_id,
            &self.node,
            self.build_side,
            probe_side(self.build_side),
            self.build_rows,
            self.build_bytes,
            &self.summary,
            execution.metrics(),
        );
    }

    /// Hash-partitions the next probe piece. `None` means the inputs are exhausted; they are
    /// released then, so a later call does not pull them again.
    fn load_next_probe_piece(
        &mut self,
        execution: &mut BoundedExecutionAttempt<'_>,
    ) -> Option<Result<PartitionedJoinSide>> {
        if self.pending_probe_chunk.is_none() {
            let inputs = self.inputs.as_mut()?;
            let input = match inputs.next_frame(execution) {
                Some(Ok(input)) => input,
                Some(Err(error)) => return Some(Err(error)),
                None => {
                    if let Some(inputs) = self.inputs.take() {
                        inputs.release_all(execution);
                    }
                    return None;
                },
            };
            let probe = match prepare_bounded_join_side_input_from_resident_frame(
                execution,
                input,
                &self.probe_columns,
                probe_side(self.build_side),
            ) {
                Ok(probe) => probe,
                Err(error) => return Some(Err(error)),
            };
            self.pending_probe_chunk = Some(PartitionedJoinProbeChunk::new(probe));
        }
        let plan = PartitionedJoinDriverPlan {
            node_id: self.node_id,
            node: &self.node,
            strategy: PARTITIONED_HASH_EQUI_STRATEGY,
            strategy_options: self.strategy_options,
            partition_count: self.partition_count,
            partition_target_bytes: self.partition_target_bytes,
        };
        let pending = self.pending_probe_chunk.as_mut().expect("probe retry input should be present");
        let (probe_partitioned, probe_rows, probe_bytes) =
            match pending.next_partitioned(execution, plan, &self.probe_keys) {
                Ok(Some(output)) => output,
                Ok(None) => unreachable!("a drained probe chunk is released after its final partitioning result"),
                Err(error) => return Some(Err(error)),
            };
        if pending.is_drained() {
            self.pending_probe_chunk = None;
        }
        let probe_partitioned_bytes = probe_partitioned.device_bytes();
        self.summary.record_probe_partitioned_chunk(probe_rows, probe_bytes, probe_partitioned_bytes);
        Some(Ok(probe_partitioned))
    }

    /// Joins one partitioned probe piece, deciding the kept tables and the deferral at the first
    /// piece. Another piece can follow while a retry range or an input frame remains.
    fn join_probe_piece(
        &mut self,
        execution: &mut BoundedExecutionAttempt<'_>,
        probe_partitioned: &PartitionedJoinSide,
    ) -> Result<Option<ResidentGpuFrame>> {
        let more_pieces =
            self.pending_probe_chunk.is_some() || self.inputs.as_ref().is_some_and(|inputs| !inputs.is_empty());
        let build_partitioned =
            self.build_partitioned.as_ref().expect("partitioned hash join build side released before stream end");
        let tables = self.hash_tables.decide(
            execution,
            &self.node,
            self.build_side,
            build_partitioned,
            more_pieces,
            &mut self.summary,
        )?;
        if matches!(self.deferral, ProbeSourceDeferral::Undecided) {
            self.deferral = match more_pieces.then(|| DeferredProbePartitions::new(execution, self.partition_count)) {
                Some(Some(deferred)) => ProbeSourceDeferral::Held(deferred),
                Some(None) | None => ProbeSourceDeferral::Off,
            };
        }
        let plan = PartitionedJoinDriverPlan {
            node_id: self.node_id,
            node: &self.node,
            strategy: PARTITIONED_HASH_EQUI_STRATEGY,
            strategy_options: self.strategy_options,
            partition_count: self.partition_count,
            partition_target_bytes: self.partition_target_bytes,
        };
        join_partitioned_probe_piece(
            execution,
            plan,
            build_partitioned,
            &tables,
            probe_partitioned,
            self.deferral.piece_deferral(),
            &mut self.summary,
        )
    }

    /// Joins the batches still held once the input is exhausted. Deferral ends here, so a later
    /// call returns `None`. Slices a lane exports stay in place for the lane to return.
    fn join_deferred_remainder(
        &mut self,
        execution: &mut BoundedExecutionAttempt<'_>,
    ) -> Result<Option<ResidentGpuFrame>> {
        let deferred = match std::mem::replace(&mut self.deferral, ProbeSourceDeferral::Off) {
            ProbeSourceDeferral::Held(deferred) => deferred,
            other => {
                self.deferral = other;
                return Ok(None);
            },
        };
        let build_partitioned =
            self.build_partitioned.as_ref().expect("partitioned hash join build side released before stream end");
        let plan = PartitionedJoinDriverPlan {
            node_id: self.node_id,
            node: &self.node,
            strategy: PARTITIONED_HASH_EQUI_STRATEGY,
            strategy_options: self.strategy_options,
            partition_count: self.partition_count,
            partition_target_bytes: self.partition_target_bytes,
        };
        join_deferred_probe_remainder(execution, plan, build_partitioned, deferred, &mut self.summary)
    }
}

impl ResidentFrameStreamSource for PartitionedHashEquiJoinProbeStreamSource {
    fn next_frame(&mut self, execution: &mut BoundedExecutionAttempt<'_>) -> Option<Result<ResidentGpuFrame>> {
        loop {
            if let Err(error) =
                execution.context().check_cancellation("query_engine.exec.bounded.partitioned_join_probe_stream")
            {
                self.release(execution);
                return Some(Err(error));
            }
            let output = match self.load_next_probe_piece(execution) {
                Some(Ok(probe_partitioned)) => {
                    let output = self.join_probe_piece(execution, &probe_partitioned);
                    release_partitioned_join_side(probe_partitioned);
                    output
                },
                Some(Err(error)) => Err(error),
                None => match self.join_deferred_remainder(execution) {
                    Ok(None) => {
                        self.record_success_metrics(execution);
                        self.release(execution);
                        return None;
                    },
                    output => output,
                },
            };
            match output {
                Ok(Some(output)) => return Some(Ok(output)),
                Ok(None) => {},
                Err(error) => {
                    self.release(execution);
                    return Some(Err(error));
                },
            }
        }
    }

    fn release(&mut self, execution: &mut BoundedExecutionAttempt<'_>) {
        self.pending_probe_chunk = None;
        if !matches!(self.deferral, ProbeSourceDeferral::Export(_)) {
            self.deferral = ProbeSourceDeferral::Off;
        }
        if let Some(input) = self.inputs.take() {
            input.release_all(execution);
        }
        if let Some(build_partitioned) = self.build_partitioned.take() {
            release_partitioned_join_side(build_partitioned);
        }
        self.hash_tables = PartitionedJoinHashTables::undecided();
    }
}

/// Whether partitioned probe chunks may run in parallel sharing one split build.
///
/// Partitioned streaming supports inner and probe-preserved left joins; each
/// probe chunk is partitioned to match the build and probed partition by
/// partition. Chunks are independent (concatenation preserves inner and
/// probe-preserved semantics); partitions within a chunk stay serial in the
/// lane, preserving the existing split/retry and skew handling. A single huge
/// key still follows the existing split or typed failure path; adding lanes
/// never assumes the max partition shrinks.
pub(in crate::exec::bounded) fn partitioned_parallel_probe_supported(node: &JoinNode) -> bool {
    partitioned_streaming_probe_supported(node)
}

struct PartitionedProbeWorkState {
    pending: VecDeque<ResidentGpuFrame>,
    outputs: Vec<ResidentGpuFrame>,
    deferred: Option<DeferredProbePartitions>,
    summary: StreamedPartitionedJoinSummary,
}

/// One lane's chunk: its output frames, its counters, and the slices it deferred for the
/// coordinator.
type PartitionedProbeChunkOutput = (Vec<ResidentGpuFrame>, StreamedPartitionedJoinSummary, Option<DeferredProbeSlices>);

/// Minimal Send+Sync view of a split partitioned build shared across lanes.
///
/// Partitions are immutable `DataFrame` clones behind `Arc` columns, and the kept tables are
/// `Arc`-shared cuDF hash joins that probes only read. Each lane supplies its own stream and
/// workspace. No new `unsafe` impl.
struct SharedPartitionedBuild {
    build_side: JoinSide,
    build_rows: u64,
    build_bytes: u64,
    partitioned: PartitionedJoinSide,
    tables: RetainedPartitionHashTables,
    build_partitioned_bytes: u64,
    partition_count: usize,
    partition_target_bytes: u64,
    strategy_options: KeyedHashJoinStrategyOptions,
    probe_columns: Vec<String>,
    probe_keys: Vec<String>,
    /// Whether lanes return the rows of partitions without a kept table to the coordinator
    /// instead of joining them.
    exports_deferred: bool,
}

impl SharedPartitionedBuild {
    fn from_state(
        state: &PartitionedHashEquiJoinFrameStreamProbeState,
        tables: RetainedPartitionHashTables,
        exports_deferred: bool,
    ) -> Self {
        Self {
            build_side: state.build_side,
            build_rows: state.build_rows,
            build_bytes: state.build_bytes,
            partitioned: clone_partitioned_side(&state.build_partitioned),
            tables,
            build_partitioned_bytes: state.build_partitioned_bytes,
            partition_count: state.partition_count,
            partition_target_bytes: state.partition_target_bytes,
            strategy_options: state.strategy_options,
            probe_columns: state.probe_columns.clone(),
            probe_keys: state.probe_keys.clone(),
            exports_deferred,
        }
    }

    /// A one-chunk probe state for a lane. The partitions and the tables are clones that share the
    /// coordinator's device buffers.
    fn lane_probe(&self, chunk: ResidentGpuFrame) -> PartitionedHashEquiJoinFrameStreamProbeState {
        PartitionedHashEquiJoinFrameStreamProbeState {
            build_side: self.build_side,
            build_rows: self.build_rows,
            build_bytes: self.build_bytes,
            build_partitioned: clone_partitioned_side(&self.partitioned),
            hash_tables: PartitionedJoinHashTables::shared(self.tables.clone()),
            build_partitioned_bytes: self.build_partitioned_bytes,
            inputs: ResidentFrameStreamPayload::new(vec![chunk]),
            partition_count: self.partition_count,
            partition_target_bytes: self.partition_target_bytes,
            strategy_options: self.strategy_options,
            probe_columns: self.probe_columns.clone(),
            probe_keys: self.probe_keys.clone(),
        }
    }
}

/// Clones the frame handles. Cloned frames share their columns. The clone adds no device memory and
/// frees none when dropped while another handle lives.
fn clone_partitioned_side(side: &PartitionedJoinSide) -> PartitionedJoinSide {
    match side {
        PartitionedJoinSide::Contiguous { frame, device_bytes, offsets } => PartitionedJoinSide::Contiguous {
            frame: frame.clone(),
            device_bytes: *device_bytes,
            offsets: offsets.clone(),
        },
        PartitionedJoinSide::Split { partitions, device_bytes } => {
            PartitionedJoinSide::Split { partitions: partitions.to_vec(), device_bytes: *device_bytes }
        },
    }
}

/// Drain a partitioned lazy stream's eager chunks in parallel.
///
/// The coordinator decides the kept tables first, on its own stream, and the lanes then share them
/// with the split build; each lane drains exactly one probe chunk through the lazy source (partition,
/// join the piece, one frame per piece). A lane returns the rows of partitions without a kept table
/// as slices. The coordinator holds them as a serial probe would, and after each committed chunk
/// joins on its own stream the batches past the attempt's deferred share; the batches still held
/// after the last chunk are joined then. A coordinator that cannot defer, itself a lane attempt,
/// leaves each lane to join those partitions with a table built for its pair. Lane outputs keep
/// chunk order, and the batch outputs follow the chunk whose commit joined them. The lanes'
/// summaries are merged into one join metric row, recorded once the run succeeded. The `outer`
/// dispatcher contract matches the hash parallel probe.
///
/// The build partitions, the kept tables and the deferred slices are released once the dispatcher
/// has joined every lane job, on success and on error. The frames still queued at a failure drop
/// with this call's scope.
pub(in crate::exec::bounded) fn parallel_partitioned_lazy_drain(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    mut probe: PartitionedHashEquiJoinFrameStreamProbeState,
    outer: Option<&AggregateLaneDispatch>,
) -> Result<Vec<ResidentGpuFrame>> {
    let inputs = std::mem::replace(&mut probe.inputs, ResidentFrameStreamPayload::new(Vec::new()));
    let frames = inputs.into_remaining_frames_owned(execution)?;
    if frames.len() <= 1 {
        return serial_partitioned_single_stream(execution, node_id, node, probe, frames.into());
    }
    let mut summary = StreamedPartitionedJoinSummary::new(
        probe.partition_count,
        probe.partition_target_bytes,
        probe.build_partitioned_bytes,
    );
    let tables =
        match probe.hash_tables.decide(execution, node, probe.build_side, &probe.build_partitioned, true, &mut summary)
        {
            Ok(tables) => tables,
            Err(error) => {
                release_partitioned_hash_equi_join_frame_stream_probe_state(execution, probe);
                return Err(error);
            },
        };
    let build_side = probe.build_side;
    let build_rows = probe.build_rows;
    let build_bytes = probe.build_bytes;
    let deferred = DeferredProbePartitions::new(execution, probe.partition_count);
    let shared = Arc::new(SharedPartitionedBuild::from_state(&probe, tables, deferred.is_some()));
    let plan = PartitionedJoinDriverPlan {
        node_id,
        node,
        strategy: PARTITIONED_HASH_EQUI_STRATEGY,
        strategy_options: probe.strategy_options,
        partition_count: probe.partition_count,
        partition_target_bytes: probe.partition_target_bytes,
    };
    let node_owned = node.clone();
    let mut state = PartitionedProbeWorkState { pending: frames.into(), outputs: Vec::new(), deferred, summary };
    let chunk_count = state.pending.len();
    let mut run_on = |dispatch: &AggregateLaneDispatch, execution: &mut BoundedExecutionAttempt<'_>| {
        let width = dispatch.width();
        execution.metrics().record_join_strategy_decision(
            node_id,
            &BTreeMap::from([
                ("parallel_probe", "partitioned".to_string()),
                ("parallel_probe_chunks", chunk_count.to_string()),
                ("parallel_probe_width", width.to_string()),
            ]),
        );
        dispatch.execute_work(
            execution,
            &mut state,
            |state, _| Ok(state.pending.pop_front()),
            {
                let shared = Arc::clone(&shared);
                let node_owned = node_owned.clone();
                move |lane_execution, chunk| {
                    partitioned_probe_one_chunk(lane_execution, node_id, &node_owned, &shared, chunk)
                }
            },
            |state, execution, (mut chunk_outputs, lane_summary, exported)| {
                state.outputs.append(&mut chunk_outputs);
                state.summary.merge_probe_counters(&lane_summary);
                let (Some(deferred), Some(exported)) = (state.deferred.as_mut(), exported) else {
                    return Ok(());
                };
                let mut hash_table_builds = 0;
                let admitted = deferred.admit(exported, |partition, slices| {
                    join_deferred_probe_batch(
                        execution,
                        plan,
                        &shared.partitioned,
                        partition,
                        slices,
                        &mut hash_table_builds,
                    )
                });
                state.summary.record_hash_table_builds(hash_table_builds);
                let (batch_outputs, admission) = admitted?;
                deferred.commit(admission)?;
                for output in batch_outputs {
                    state.summary.record_output(&output);
                    state.outputs.push(output.frame);
                }
                Ok(())
            },
        )
    };
    let mut result = match outer {
        Some(dispatch) => run_on(dispatch, execution),
        None => {
            AggregateLaneDispatch::with_available_lanes(execution, |dispatch, execution| run_on(dispatch, execution))
        },
    };
    if let Some(deferred) = state.deferred.take()
        && result.is_ok()
    {
        let mut hash_table_builds = 0;
        let outputs = deferred.join_all(|partition, slices| {
            join_deferred_probe_batch(execution, plan, &shared.partitioned, partition, slices, &mut hash_table_builds)
        });
        state.summary.record_hash_table_builds(hash_table_builds);
        match outputs {
            Ok(outputs) => {
                for output in outputs {
                    state.summary.record_output(&output);
                    state.outputs.push(output.frame);
                }
            },
            Err(error) => result = Err(error),
        }
    }
    release_partitioned_hash_equi_join_frame_stream_probe_state(execution, probe);
    result?;
    record_streamed_partitioned_hash_join_metrics(
        node_id,
        node,
        build_side,
        probe_side(build_side),
        build_rows,
        build_bytes,
        &state.summary,
        execution.metrics(),
    );
    Ok(state.outputs)
}

/// Drains a probe of at most one chunk through the lazy source on the calling thread, without
/// lanes.
fn serial_partitioned_single_stream(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    probe: PartitionedHashEquiJoinFrameStreamProbeState,
    mut frames: VecDeque<ResidentGpuFrame>,
) -> Result<Vec<ResidentGpuFrame>> {
    // Single-chunk fallback preserves the exact serial lazy path: rebuild a
    // one-stream probe state and drain it without lanes.
    let mut outputs = Vec::new();
    let mut probe = probe;
    probe.inputs = ResidentFrameStreamPayload::new(frames.drain(..).collect());
    let stream = partitioned_hash_equi_join_frame_stream_probe_to_payload(node_id, node, probe);
    outputs.extend(stream.into_remaining_frames_owned(execution)?);
    Ok(outputs)
}

/// Lane body: joins one probe chunk against the shared build and its kept tables, and returns the
/// chunk's frames with the lane's counters and, when the build `exports_deferred`, the chunk's rows
/// of partitions without a kept table.
///
/// The chunk goes through the same lazy source as a serial probe, drained to completion, with the
/// kept tables already decided. The cloned partitions and tables alias the coordinator's. Releasing
/// the lane's source therefore frees nothing the other lanes read; only the tables the lane built
/// for single pairs and the slices it returns are its own.
fn partitioned_probe_one_chunk(
    lane_execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    shared: &SharedPartitionedBuild,
    chunk: ResidentGpuFrame,
) -> Result<PartitionedProbeChunkOutput> {
    let mut source = PartitionedHashEquiJoinProbeStreamSource::new(node_id, node, shared.lane_probe(chunk), false);
    if shared.exports_deferred {
        source.deferral = ProbeSourceDeferral::Export(DeferredProbeSlices::new(shared.partition_count));
    }
    let mut frames = Vec::new();
    while let Some(frame) = source.next_frame(lane_execution) {
        frames.push(frame?);
    }
    let exported = match std::mem::replace(&mut source.deferral, ProbeSourceDeferral::Off) {
        ProbeSourceDeferral::Export(slices) => Some(slices),
        ProbeSourceDeferral::Undecided | ProbeSourceDeferral::Off | ProbeSourceDeferral::Held(_) => None,
    };
    Ok((frames, source.summary, exported))
}
