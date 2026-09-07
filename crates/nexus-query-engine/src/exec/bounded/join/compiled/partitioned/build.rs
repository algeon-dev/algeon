use super::{
    BoundedCompiledPartitionedJoinBuildState, BoundedExecutionAttempt, BoundedJoinLeafLocalParquetSourceStream,
    BoundedJoinSideFrame, BoundedPartitionRowPolicy, CompiledJoinExecutionStrategy, DataFrame, GpuDataFrame, JoinNode,
    JoinSide, KeyedHashJoinStrategyOptions, NodeId, PartitionedJoinSide, QueryError, QueryErrorCode,
    ResidentFrameStreamPayload, ResidentGpuFrame, Result, compiled_join_input_columns, cudf_query_error,
    execution_overrides, hash_partition_join_side, join_keys_for_side,
    materialize_bounded_join_leaf_local_parquet_source_stream, observe_device_memory_bytes,
    partitioned_join_partition_count, partitioned_join_target_bytes_for_context,
    prepare_bounded_join_side_input_from_resident_frame, release_bounded_join_side_frame,
    stream_bounded_join_leaf_local_parquet_source,
};

// Decision to consume the rest of a build input stream partition by partition.
// The strategy owner (compiled executor admission) makes it; this module only
// sizes and stores partitions.
#[derive(Clone, Copy)]
pub(in crate::exec::bounded) struct PartitionedJoinBuildCommitment {
    pub(in crate::exec::bounded) strategy_options: KeyedHashJoinStrategyOptions,
    pub(in crate::exec::bounded) partition_count: usize,
    pub(in crate::exec::bounded) partition_target_bytes: u64,
}

// Build input consumed so far while the strategy is still undecided.
#[derive(Clone, Copy)]
pub(in crate::exec::bounded) struct BoundedJoinBuildStreamObservation {
    pub(in crate::exec::bounded) rows: u64,
    pub(in crate::exec::bounded) bytes: u64,
    pub(in crate::exec::bounded) chunks: usize,
}

pub(in crate::exec::bounded) enum BoundedJoinBuildInputStream {
    Leaf(Box<BoundedJoinLeafLocalParquetSourceStream>),
    Frames(ResidentFrameStreamPayload),
}

pub(in crate::exec::bounded) enum BoundedJoinBuildStreamOutcome {
    Materialized(ResidentGpuFrame),
    Partitioned(BoundedCompiledPartitionedJoinBuildState),
}

type PartitionedJoinBuildSettle<'a> = &'a mut dyn FnMut(
    &mut BoundedExecutionAttempt<'_>,
    Option<BoundedJoinBuildStreamObservation>,
) -> Result<Option<PartitionedJoinBuildCommitment>>;

pub(in crate::exec::bounded) fn partitioned_join_build_commitment(
    execution: &BoundedExecutionAttempt<'_>,
    strategy_options: KeyedHashJoinStrategyOptions,
    build_basis_bytes: Option<u64>,
    probe_bytes: Option<u64>,
) -> PartitionedJoinBuildCommitment {
    let partition_count =
        partitioned_join_partition_count(execution.context(), build_basis_bytes.unwrap_or(0), probe_bytes.unwrap_or(0))
            .max(BoundedPartitionRowPolicy::default().hash_frame_stream_min_partitions());
    PartitionedJoinBuildCommitment {
        strategy_options,
        partition_count,
        partition_target_bytes: partitioned_join_target_bytes_for_context(execution.context()),
    }
}

/// Consumes a join build input one chunk at a time.
///
/// Chunks are projected to the join input columns as they arrive when the node
/// is keyed-hash capable (`keyed_hash_inputs`), so every keyed strategy can
/// still be selected afterwards. `settle` is asked before the first chunk and
/// after every accumulated chunk; once it commits, accumulated and later
/// chunks are hash-partitioned into per-partition slices and the result is a
/// split build side. Without a commitment the chunks are concatenated into one
/// frame for the existing strategy selection.
pub(in crate::exec::bounded) fn consume_bounded_join_build_input_stream(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    input: BoundedJoinBuildInputStream,
    keyed_hash_inputs: bool,
    settle: PartitionedJoinBuildSettle<'_>,
) -> Result<BoundedJoinBuildStreamOutcome> {
    let side = node.spec.build_side;
    let columns = if keyed_hash_inputs {
        Some(compiled_join_input_columns(node, CompiledJoinExecutionStrategy::PartitionedHashEqui, side)?)
    } else {
        None
    };
    let mut accumulator = JoinBuildAccumulator::Pending { frames: Vec::new(), rows: 0, bytes: 0 };
    if columns.is_some()
        && let Some(commitment) = settle(execution, None)?
    {
        accumulator.commit(execution, node_id, node, commitment, None)?;
    }
    let stream_result = {
        let mut consume = |input: ResidentGpuFrame, execution: &mut BoundedExecutionAttempt<'_>| {
            let prepared = match columns.as_deref() {
                Some(columns) => prepare_bounded_join_side_input_from_resident_frame(execution, input, columns, side)?,
                None => {
                    let rows = input.frame.num_rows() as u64;
                    let bytes = input.device_bytes();
                    BoundedJoinSideFrame { frame: input.frame.into_frame(), rows, bytes }
                },
            };
            // A chunk that fails here is re-read in smaller pieces by the leaf
            // stream, so the accumulator must still hold exactly the chunks
            // before it: nothing of this chunk is kept unless the whole step
            // succeeds.
            let observation = match &mut accumulator {
                JoinBuildAccumulator::Pending { frames, rows, bytes } => {
                    if columns.is_none() {
                        *rows = rows.saturating_add(prepared.rows);
                        *bytes = bytes.saturating_add(prepared.bytes);
                        frames.push(prepared);
                        return Ok(());
                    }
                    BoundedJoinBuildStreamObservation {
                        rows: rows.saturating_add(prepared.rows),
                        bytes: bytes.saturating_add(prepared.bytes),
                        chunks: frames.len() + 1,
                    }
                },
                JoinBuildAccumulator::Partitioned(partitioned) => {
                    return partitioned.push(execution, node_id, node, prepared);
                },
            };
            match settle(execution, Some(observation))? {
                Some(commitment) => accumulator.commit(execution, node_id, node, commitment, Some(prepared)),
                None => {
                    accumulator.push_pending(prepared);
                    Ok(())
                },
            }
        };
        match input {
            BoundedJoinBuildInputStream::Leaf(stream) => {
                if stream.chunk_count() == 0 {
                    let frame = materialize_bounded_join_leaf_local_parquet_source_stream(execution, *stream)?;
                    return Ok(BoundedJoinBuildStreamOutcome::Materialized(frame));
                }
                stream_bounded_join_leaf_local_parquet_source(execution, *stream, &mut consume)
            },
            BoundedJoinBuildInputStream::Frames(mut inputs) => {
                let mut result = Ok(());
                while let Some(input) = inputs.next_yielded_frame(execution) {
                    let mut input = match input {
                        Ok(input) => input,
                        Err(error) => {
                            result = Err(error);
                            break;
                        },
                    };
                    let input = input.take_for_release();
                    if let Err(error) = consume(input, execution) {
                        result = Err(error);
                        break;
                    }
                }
                match &result {
                    Ok(()) => inputs.release_producer(execution),
                    Err(_) => inputs.release_all(execution),
                }
                result
            },
        }
    };
    match stream_result {
        Ok(()) => accumulator.finish(execution, node_id, node),
        Err(error) => {
            accumulator.release();
            Err(error)
        },
    }
}

enum JoinBuildAccumulator {
    Pending { frames: Vec<BoundedJoinSideFrame>, rows: u64, bytes: u64 },
    Partitioned(Box<PartitionedJoinBuildAccumulator>),
}

impl JoinBuildAccumulator {
    fn push_pending(&mut self, prepared: BoundedJoinSideFrame) {
        let Self::Pending { frames, rows, bytes } = self else {
            unreachable!("only a pending build accumulator keeps whole chunks")
        };
        *rows = rows.saturating_add(prepared.rows);
        *bytes = bytes.saturating_add(prepared.bytes);
        frames.push(prepared);
    }

    // Partitions every pending chunk plus `current`. The pending chunks share
    // their columns with the partitioner, so a cap rejection leaves them (and
    // this accumulator) exactly as they were and drops only `current`, which
    // the leaf stream re-reads in smaller pieces.
    fn commit(
        &mut self,
        execution: &mut BoundedExecutionAttempt<'_>,
        node_id: NodeId,
        node: &JoinNode,
        commitment: PartitionedJoinBuildCommitment,
        current: Option<BoundedJoinSideFrame>,
    ) -> Result<()> {
        let Self::Pending { frames, .. } = self else {
            unreachable!("only a pending build accumulator commits to partitions")
        };
        let side = node.spec.build_side;
        let mut partitioned = PartitionedJoinBuildAccumulator::new(side, commitment, join_keys_for_side(node, side));
        for frame in frames.iter() {
            let shared = BoundedJoinSideFrame { frame: frame.frame.clone(), rows: frame.rows, bytes: frame.bytes };
            partitioned.push(execution, node_id, node, shared)?;
        }
        if let Some(current) = current {
            partitioned.push(execution, node_id, node, current)?;
        }
        let pending = std::mem::take(frames);
        *self = Self::Partitioned(Box::new(partitioned));
        for frame in pending.into_iter().rev() {
            release_bounded_join_side_frame(frame);
        }
        Ok(())
    }

    fn finish(
        self,
        execution: &mut BoundedExecutionAttempt<'_>,
        node_id: NodeId,
        node: &JoinNode,
    ) -> Result<BoundedJoinBuildStreamOutcome> {
        match self {
            Self::Pending { frames, .. } => concatenate_join_build_input_frames(execution, node_id, node, frames)
                .map(BoundedJoinBuildStreamOutcome::Materialized),
            Self::Partitioned(partitioned) => {
                partitioned.finish(execution, node_id, node).map(BoundedJoinBuildStreamOutcome::Partitioned)
            },
        }
    }

    fn release(self) {
        match self {
            Self::Pending { frames, .. } => {
                for frame in frames.into_iter().rev() {
                    release_bounded_join_side_frame(frame);
                }
            },
            Self::Partitioned(_) => {},
        }
    }
}

fn concatenate_join_build_input_frames(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    mut frames: Vec<BoundedJoinSideFrame>,
) -> Result<ResidentGpuFrame> {
    if frames.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.bounded.join.build_input_stream",
            "join build input stream produced no input frames",
        )
        .with_fact("node_id", node_id)
        .with_fact("build_side", node.spec.build_side));
    }
    if frames.len() == 1 {
        let BoundedJoinSideFrame { frame, bytes, .. } = frames.pop().expect("single frame checked");
        return Ok(ResidentGpuFrame::with_device_bytes(GpuDataFrame::new(frame), bytes));
    }
    let chunk_count = frames.len();
    let mut inputs = frames.into_iter().map(|frame| frame.frame).collect::<Vec<_>>();
    let first = inputs.remove(0);
    let rest = inputs.iter().collect::<Vec<_>>();
    let overrides_lease = execution_overrides(execution.context())?;
    let overrides = overrides_lease.overrides();
    let output = first.with_execution(&overrides).concat_many(&rest).map_err(|error| {
        cudf_query_error(
            "nexus_query_engine.bounded.join.build_input_concat",
            "failed to concatenate bounded join build input chunks",
            error,
        )
        .with_fact("node_id", node_id)
        .with_fact("build_side", node.spec.build_side)
        .with_fact("chunk_count", chunk_count)
    })?;
    let output = GpuDataFrame::new(output);
    let bytes = observe_device_memory_bytes(&output, execution.metrics_mut())?;
    Ok(ResidentGpuFrame::with_device_bytes(output, bytes))
}

// Per-partition slices of the build input consumed so far. Every chunk is
// hash-partitioned once and its partition ranges copied out, so the chunk and
// its partitioned copy are released before the next chunk arrives.
struct PartitionedJoinBuildAccumulator {
    commitment: PartitionedJoinBuildCommitment,
    side: JoinSide,
    keys: Vec<String>,
    slices: Vec<Vec<DataFrame>>,
    empty_template: Option<DataFrame>,
    rows: u64,
    bytes: u64,
    partitioned_bytes: u64,
    chunks: usize,
}

impl PartitionedJoinBuildAccumulator {
    fn new(side: JoinSide, commitment: PartitionedJoinBuildCommitment, keys: Vec<String>) -> Self {
        Self {
            commitment,
            side,
            keys,
            slices: (0..commitment.partition_count).map(|_| Vec::new()).collect(),
            empty_template: None,
            rows: 0,
            bytes: 0,
            partitioned_bytes: 0,
            chunks: 0,
        }
    }

    fn push(
        &mut self,
        execution: &mut BoundedExecutionAttempt<'_>,
        node_id: NodeId,
        node: &JoinNode,
        input: BoundedJoinSideFrame,
    ) -> Result<()> {
        let partition_count = self.commitment.partition_count;
        let overrides_lease = execution_overrides(execution.context())?;
        let overrides = overrides_lease.overrides();
        if self.empty_template.is_none() {
            let template = input.frame.with_execution(&overrides).slice(0, 0).map_err(|error| {
                cudf_query_error(
                    "nexus_query_engine.bounded.join.build_partition_slice",
                    "failed to create an empty build partition template",
                    error,
                )
                .with_fact("node_id", node_id)
                .with_fact("build_side", self.side)
            })?;
            self.empty_template = Some(template);
        }
        let chunk_index = self.chunks;
        let input_rows = input.rows;
        let input_bytes = input.bytes;
        let partitioned =
            hash_partition_join_side(execution, self.side, input, &self.keys, partition_count, &overrides)?;
        let PartitionedJoinSide::Contiguous { frame, offsets, .. } = partitioned else {
            unreachable!("hash partitioning a build chunk yields a contiguous partitioned frame")
        };
        let mut chunk_slices = Vec::with_capacity(partition_count);
        let mut chunk_partitioned_bytes = 0u64;
        for partition in 0..partition_count {
            let start = offsets[partition];
            let len = offsets[partition + 1].saturating_sub(start);
            if len == 0 {
                continue;
            }
            let slice = frame.with_execution(&overrides).slice(start, len).map_err(|error| {
                cudf_query_error(
                    "nexus_query_engine.bounded.join.build_partition_slice",
                    "failed to copy a build chunk partition into its partition accumulator",
                    error,
                )
                .with_fact("node_id", node_id)
                .with_fact("join_type", node.spec.join_type)
                .with_fact("build_side", self.side)
                .with_fact("partition", partition)
                .with_fact("partition_count", partition_count)
                .with_fact("partition_rows", len)
                .with_fact("build_chunk_index", chunk_index)
                .with_fact("accumulated_build_rows", self.rows)
                .with_fact("accumulated_partitioned_bytes", self.partitioned_bytes)
            })?;
            let slice_bytes = observe_device_memory_bytes(&GpuDataFrame::new(slice.clone()), execution.metrics_mut())?;
            chunk_partitioned_bytes = chunk_partitioned_bytes.saturating_add(slice_bytes);
            chunk_slices.push((partition, slice));
        }
        for (partition, slice) in chunk_slices {
            self.slices[partition].push(slice);
        }
        self.rows = self.rows.saturating_add(input_rows);
        self.bytes = self.bytes.saturating_add(input_bytes);
        self.partitioned_bytes = self.partitioned_bytes.saturating_add(chunk_partitioned_bytes);
        self.chunks += 1;
        Ok(())
    }

    fn finish(
        self,
        execution: &mut BoundedExecutionAttempt<'_>,
        node_id: NodeId,
        node: &JoinNode,
    ) -> Result<BoundedCompiledPartitionedJoinBuildState> {
        let Self { commitment, side, slices, empty_template, rows, bytes, chunks, .. } = self;
        let Some(empty_template) = empty_template else {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.bounded.join.build_input_stream",
                "join build input stream produced no input frames",
            )
            .with_fact("node_id", node_id)
            .with_fact("build_side", side));
        };
        let partition_count = commitment.partition_count;
        let overrides_lease = execution_overrides(execution.context())?;
        let overrides = overrides_lease.overrides();
        let mut partitions: Vec<DataFrame> = Vec::with_capacity(partition_count);
        let mut device_bytes = 0u64;
        for (partition, mut partition_slices) in slices.into_iter().enumerate() {
            let frame = match partition_slices.len() {
                0 => empty_template.clone(),
                1 => partition_slices.pop().expect("single slice checked"),
                slice_count => {
                    let partition_rows = partition_slices.iter().map(DataFrame::num_rows).sum::<usize>();
                    let first = partition_slices.remove(0);
                    let output = {
                        let rest = partition_slices.iter().collect::<Vec<_>>();
                        first.with_execution(&overrides).concat_many(&rest)
                    }
                    .map_err(|error| {
                        cudf_query_error(
                            "nexus_query_engine.bounded.join.build_partition_concat",
                            "failed to concatenate accumulated build partition slices",
                            error,
                        )
                        .with_fact("node_id", node_id)
                        .with_fact("join_type", node.spec.join_type)
                        .with_fact("build_side", side)
                        .with_fact("partition", partition)
                        .with_fact("partition_count", partition_count)
                        .with_fact("partition_rows", partition_rows)
                        .with_fact("slice_count", slice_count)
                        .with_fact("build_rows", rows)
                        .with_fact("build_chunks", chunks)
                    })?;
                    // The slices are dead once the partition frame exists; free
                    // them before the next partition concatenates on the domain.
                    drop(partition_slices);
                    drop(first);
                    output
                },
            };
            let frame_bytes = observe_device_memory_bytes(&GpuDataFrame::new(frame.clone()), execution.metrics_mut())?;
            device_bytes = device_bytes.saturating_add(frame_bytes);
            partitions.push(frame);
        }
        Ok(BoundedCompiledPartitionedJoinBuildState {
            strategy_options: commitment.strategy_options,
            side,
            rows,
            bytes,
            partition_count,
            partition_target_bytes: commitment.partition_target_bytes,
            partitioned: PartitionedJoinSide::Split { partitions, device_bytes },
            input_chunks: chunks,
        })
    }
}
