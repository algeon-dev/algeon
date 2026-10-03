use super::{
    BTreeMap, BoundedCompiledJoinMaterializeOutput, BoundedCompiledJoinProbeState,
    BoundedCompiledPartitionedJoinBuildState, BoundedCompiledPreparedJoinBuildState, BoundedExecutionAttempt,
    BoundedJoinLeafLocalParquetSourceStream, BoundedJoinSideFrame, BoundedPartitionRowPolicy,
    CompiledEqualityJoinPreflightIndices, GpuDataFrame, Instant, JoinMaterializationTxn, JoinNode, JoinSide,
    JoinStrategy, KeyedHashJoinStrategyOptions, NodeId, PARTITIONED_HASH_EQUI_STRATEGY,
    PartitionedHashEquiJoinFrameStreamProbeState, PartitionedJoinDriverPlan, PartitionedJoinPartitionOutput,
    PartitionedJoinProbeChunk, PartitionedJoinSide, ResidentFrameStreamPayload, ResidentFrameStreamSource,
    ResidentGpuFrame, Result, StreamedPartitionedJoinSummary, VecDeque, bounded_invariant_violation,
    compiled_join_input_columns, concatenate_streamed_hash_join_outputs, execution_overrides, hash_partition_join_side,
    join_keys_for_side, materialize_compiled_keyed_equality_join_indices_with_preflight,
    materialize_partitioned_join_partition_with_retry, observe_device_memory_bytes, partitioned_join_partition_count,
    partitioned_join_target_bytes_for_context, partitioned_streaming_probe_supported,
    prepare_bounded_join_side_input_from_resident_frame, probe_side, record_partitioned_join_operator_metrics,
    record_streamed_partitioned_hash_join_metrics, release_partitioned_join_side, run_partitioned_join_driver,
    stream_bounded_join_leaf_local_parquet_source_serial, validate_partitioned_hash_equi_join,
};
use crate::exec::bounded::{compiled::lanes::AggregateLaneDispatch, release_bounded_compiled_join_probe_state};

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
    let mut result =
        run_partitioned_join_driver(execution, plan, left, right, materialize_partitioned_hash_equi_join_partition)?;
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

fn materialize_partitioned_hash_equi_join_partition(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
) -> Result<PartitionedJoinPartitionOutput> {
    let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
    let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
    let overrides_lease = execution_overrides(execution.context())?;
    let overrides = overrides_lease.overrides();
    let timer = Instant::now();
    let mut txn = JoinMaterializationTxn::with_sides(left, right);
    let (left, right) = txn.sides();
    let preflight_indices = match materialize_compiled_keyed_equality_join_indices_with_preflight(
        execution,
        node,
        left,
        right,
        node.spec.build_side,
        &left_keys,
        &right_keys,
        &overrides,
    ) {
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
        &overrides,
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
    let overrides_lease = execution_overrides(execution.context())?;
    let overrides = overrides_lease.overrides();
    let build_partitioned =
        hash_partition_join_side(execution, build_side, build.input, &build_keys, partition_count, &overrides)?;
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

    let probe_columns = compiled_join_input_columns(node, JoinStrategy::PartitionedHashEqui, probe_side)?;
    let stream_result = stream_bounded_join_leaf_local_parquet_source_serial(execution, stream, |input, execution| {
        let probe = prepare_bounded_join_side_input_from_resident_frame(execution, input, &probe_columns, probe_side)?;
        let probe_rows = probe.rows;
        let probe_bytes = probe.bytes;
        let probe_partitioned =
            hash_partition_join_side(execution, probe_side, probe, &probe_keys, partition_count, &overrides)?;
        let probe_partitioned_bytes = probe_partitioned.device_bytes();
        // A chunk that fails here is re-read in smaller pieces by the leaf
        // stream, so its outputs and summary are kept only after every
        // partition succeeds; a failed chunk's outputs drop before the retry.
        let mut chunk_outputs = Vec::new();
        for partition in 0..partition_count {
            let (left_partitioned, right_partitioned) = match build_side {
                JoinSide::Left => (&build_partitioned, &probe_partitioned),
                JoinSide::Right => (&probe_partitioned, &build_partitioned),
            };
            match materialize_partitioned_join_partition_with_retry(
                execution,
                plan,
                left_partitioned,
                right_partitioned,
                partition,
                &mut materialize_partitioned_hash_equi_join_partition,
            ) {
                Ok(partition_outputs) => chunk_outputs.extend(partition_outputs),
                Err(error) => {
                    release_partitioned_join_side(probe_partitioned);
                    return Err(error);
                },
            }
        }
        release_partitioned_join_side(probe_partitioned);
        summary.record_probe_partitioned_chunk(probe_rows, probe_bytes, probe_partitioned_bytes);
        for output in chunk_outputs {
            summary.record_output(&output);
            outputs.push(output.frame);
        }
        Ok(())
    });
    match stream_result {
        Ok(()) => {
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
            release_partitioned_join_side(build_partitioned);
            Ok(BoundedCompiledJoinProbeState::PartitionedStreamed {
                build_side,
                build_rows,
                build_bytes,
                outputs,
                summary,
            })
        },
        Err(error) => {
            release_partitioned_join_side(build_partitioned);
            Err(error)
        },
    }
}

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
    let overrides_lease = execution_overrides(execution.context())?;
    let overrides = overrides_lease.overrides();
    let build_partitioned =
        hash_partition_join_side(execution, build_side, build.input, &build_keys, partition_count, &overrides)?;
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
        build_partitioned_bytes,
        inputs,
        partition_count,
        partition_target_bytes,
        strategy_options: build.strategy_options,
        probe_columns,
        probe_keys,
    }))
}

// Probe a build side that was partitioned while its input streamed in. The
// partition count is fixed by the build; probe chunks are partitioned to match.
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
        build_partitioned_bytes,
        inputs,
        partition_count: build.partition_count,
        partition_target_bytes: build.partition_target_bytes,
        strategy_options: build.strategy_options,
        probe_columns,
        probe_keys: join_keys_for_side(node, probe_side),
    }))
}

pub(in crate::exec::bounded::join::compiled) fn partitioned_hash_equi_join_frame_stream_probe_to_payload(
    node_id: NodeId,
    node: &JoinNode,
    probe: PartitionedHashEquiJoinFrameStreamProbeState,
) -> ResidentFrameStreamPayload {
    let max_frame_bytes_hint = Some(probe.partition_target_bytes);
    let total_bytes_hint = probe.inputs.total_bytes_hint();
    let aggregate_num_rows = probe.inputs.num_rows();
    let num_columns = node.spec.output_projection.len();
    let source = PartitionedHashEquiJoinProbeStreamSource {
        node_id,
        node: node.clone(),
        build_side: probe.build_side,
        build_rows: probe.build_rows,
        build_bytes: probe.build_bytes,
        build_partitioned: Some(probe.build_partitioned),
        inputs: Some(probe.inputs),
        partition_count: probe.partition_count,
        partition_target_bytes: probe.partition_target_bytes,
        strategy_options: probe.strategy_options,
        probe_columns: probe.probe_columns,
        probe_keys: probe.probe_keys,
        active_probe_partitioned: None,
        pending_probe_chunk: None,
        pending_outputs: VecDeque::new(),
        next_partition: 0,
        summary: StreamedPartitionedJoinSummary::new(
            probe.partition_count,
            probe.partition_target_bytes,
            probe.build_partitioned_bytes,
        ),
        metrics_recorded: false,
    };
    ResidentFrameStreamPayload::from_producer(
        max_frame_bytes_hint,
        total_bytes_hint,
        aggregate_num_rows,
        num_columns,
        Box::new(source),
    )
}

struct PartitionedHashEquiJoinProbeStreamSource {
    node_id: NodeId,
    node: JoinNode,
    build_side: JoinSide,
    build_rows: u64,
    build_bytes: u64,
    build_partitioned: Option<PartitionedJoinSide>,
    inputs: Option<ResidentFrameStreamPayload>,
    partition_count: usize,
    partition_target_bytes: u64,
    strategy_options: KeyedHashJoinStrategyOptions,
    probe_columns: Vec<String>,
    probe_keys: Vec<String>,
    active_probe_partitioned: Option<PartitionedJoinSide>,
    pending_probe_chunk: Option<PartitionedJoinProbeChunk>,
    pending_outputs: VecDeque<ResidentGpuFrame>,
    next_partition: usize,
    summary: StreamedPartitionedJoinSummary,
    metrics_recorded: bool,
}

impl PartitionedHashEquiJoinProbeStreamSource {
    fn record_success_metrics(&mut self, execution: &mut BoundedExecutionAttempt<'_>) {
        if self.metrics_recorded {
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

    fn load_next_probe_chunk(&mut self, execution: &mut BoundedExecutionAttempt<'_>) -> Option<Result<bool>> {
        if self.pending_probe_chunk.is_none() {
            let inputs = self.inputs.as_mut()?;
            let input = match inputs.next_frame(execution) {
                Some(Ok(input)) => input,
                Some(Err(error)) => return Some(Err(error)),
                None => return None,
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
        let pending = self.pending_probe_chunk.as_mut().expect("probe retry input should be present");
        let plan = PartitionedJoinDriverPlan {
            node_id: self.node_id,
            node: &self.node,
            strategy: PARTITIONED_HASH_EQUI_STRATEGY,
            strategy_options: self.strategy_options,
            partition_count: self.partition_count,
            partition_target_bytes: self.partition_target_bytes,
        };
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
        self.active_probe_partitioned = Some(probe_partitioned);
        self.next_partition = 0;
        Some(Ok(true))
    }

    fn materialize_next_active_partition(&mut self, execution: &mut BoundedExecutionAttempt<'_>) -> Result<()> {
        let partition = self.next_partition;
        self.next_partition += 1;
        let is_last_active_partition = self.next_partition >= self.partition_count;
        let build_partitioned =
            self.build_partitioned.as_ref().expect("partitioned hash join build side released before stream end");
        let probe_partitioned = self
            .active_probe_partitioned
            .as_ref()
            .expect("partitioned hash join probe chunk missing before stream end");
        let (left_partitioned, right_partitioned) = match self.build_side {
            JoinSide::Left => (build_partitioned, probe_partitioned),
            JoinSide::Right => (probe_partitioned, build_partitioned),
        };
        let plan = PartitionedJoinDriverPlan {
            node_id: self.node_id,
            node: &self.node,
            strategy: PARTITIONED_HASH_EQUI_STRATEGY,
            strategy_options: self.strategy_options,
            partition_count: self.partition_count,
            partition_target_bytes: self.partition_target_bytes,
        };
        let outputs = materialize_partitioned_join_partition_with_retry(
            execution,
            plan,
            left_partitioned,
            right_partitioned,
            partition,
            &mut materialize_partitioned_hash_equi_join_partition,
        )?;
        if is_last_active_partition {
            self.release_active_probe_partitioned();
        }
        for output in outputs {
            self.summary.record_output(&output);
            self.pending_outputs.push_back(output.frame);
        }
        Ok(())
    }

    fn release_active_probe_partitioned(&mut self) {
        if let Some(probe_partitioned) = self.active_probe_partitioned.take() {
            release_partitioned_join_side(probe_partitioned);
        }
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
            if let Some(output) = self.pending_outputs.pop_front() {
                return Some(Ok(output));
            }
            if self.active_probe_partitioned.is_some() {
                if self.next_partition < self.partition_count {
                    match self.materialize_next_active_partition(execution) {
                        Ok(()) => continue,
                        Err(error) => {
                            self.release(execution);
                            return Some(Err(error));
                        },
                    }
                }
                self.release_active_probe_partitioned();
            }

            match self.load_next_probe_chunk(execution) {
                Some(Ok(true)) => {},
                Some(Ok(false)) => unreachable!("partitioned hash stream probe chunk loading is boolean-true only"),
                Some(Err(error)) => {
                    self.release(execution);
                    return Some(Err(error));
                },
                None => {
                    self.record_success_metrics(execution);
                    self.release(execution);
                    return None;
                },
            }
        }
    }

    fn release(&mut self, execution: &mut BoundedExecutionAttempt<'_>) {
        self.pending_outputs.clear();
        self.release_active_probe_partitioned();
        self.pending_probe_chunk = None;
        if let Some(input) = self.inputs.take() {
            input.release_all(execution);
        }
        if let Some(build_partitioned) = self.build_partitioned.take() {
            release_partitioned_join_side(build_partitioned);
        }
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
}

/// Minimal Send+Sync view of a split partitioned build shared across lanes.
///
/// Partitions are immutable `DataFrame` clones behind `Arc` columns; each
/// lane supplies its own stream and workspace. No new `unsafe` impl.
struct SharedPartitionedBuild {
    build_side: JoinSide,
    build_rows: u64,
    build_bytes: u64,
    partitioned: PartitionedJoinSide,
    partition_count: usize,
    partition_target_bytes: u64,
    strategy_options: KeyedHashJoinStrategyOptions,
}

impl SharedPartitionedBuild {
    fn from_state(state: &PartitionedHashEquiJoinFrameStreamProbeState) -> Self {
        Self {
            build_side: state.build_side,
            build_rows: state.build_rows,
            build_bytes: state.build_bytes,
            partitioned: clone_partitioned_side(&state.build_partitioned),
            partition_count: state.partition_count,
            partition_target_bytes: state.partition_target_bytes,
            strategy_options: state.strategy_options,
        }
    }
}

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
/// The caller retains ownership of `probe` while lanes run; each lane clones
/// the shared split build and drains exactly one probe chunk through the
/// existing single-chunk lazy pipeline (partition, probe partitions
/// serially, collect). Outputs keep chunk order; the build is constructed
/// once regardless of chunk count. The `outer` dispatcher contract matches
/// the hash parallel probe.
pub(in crate::exec::bounded) fn parallel_partitioned_lazy_drain(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    mut probe: PartitionedHashEquiJoinFrameStreamProbeState,
    outer: Option<&AggregateLaneDispatch>,
) -> Result<Vec<ResidentGpuFrame>> {
    // Take probe inputs out first so `probe` stays usable for the shared-view
    // construction; the frames move into lane work while partitions live on
    // in `shared` through `Arc` columns.
    let inputs = std::mem::replace(&mut probe.inputs, ResidentFrameStreamPayload::new(Vec::new()));
    let frames = inputs.into_remaining_frames_owned(execution)?;
    if frames.len() <= 1 {
        return serial_partitioned_single_stream(execution, node_id, node, probe, frames.into());
    }
    let shared = std::sync::Arc::new(SharedPartitionedBuild::from_state(&probe));
    // `probe` (with empty inputs) is no longer needed; its build partitions
    // live on in `shared` through `Arc` columns while lanes run.
    drop(probe);
    let node_owned = node.clone();
    let mut state = PartitionedProbeWorkState { pending: frames.into(), outputs: Vec::new() };
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
                let shared = std::sync::Arc::clone(&shared);
                let node_owned = node_owned.clone();
                move |lane_execution, chunk| {
                    partitioned_probe_one_chunk(lane_execution, node_id, &node_owned, &shared, chunk)
                }
            },
            |state, _, mut chunk_outputs| {
                state.outputs.append(&mut chunk_outputs);
                Ok(())
            },
        )
    };
    match outer {
        Some(dispatch) => run_on(dispatch, execution)?,
        None => {
            AggregateLaneDispatch::with_available_lanes(execution, |dispatch, execution| run_on(dispatch, execution))?
        },
    }
    Ok(state.outputs)
}

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

fn partitioned_probe_one_chunk(
    lane_execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    shared: &SharedPartitionedBuild,
    chunk: ResidentGpuFrame,
) -> Result<Vec<ResidentGpuFrame>> {
    let build = BoundedCompiledPartitionedJoinBuildState {
        strategy_options: shared.strategy_options,
        side: shared.build_side,
        rows: shared.build_rows,
        bytes: shared.build_bytes,
        partition_count: shared.partition_count,
        partition_target_bytes: shared.partition_target_bytes,
        partitioned: clone_partitioned_side(&shared.partitioned),
        input_chunks: 1,
    };
    let single = ResidentFrameStreamPayload::new(vec![chunk]);
    let lazy = execute_compiled_partitioned_hash_equi_join_probe_from_split_build_frame_stream(
        lane_execution,
        node_id,
        node,
        build,
        single,
    )?;
    let inner = match lazy {
        BoundedCompiledJoinProbeState::PartitionedLazyStream(inner) => inner,
        other => {
            // Split-build single-chunk probes always return lazy; any other
            // variant would mean the strategy changed under us.
            release_bounded_compiled_join_probe_state(lane_execution, other);
            return Err(bounded_invariant_violation(
                "partitioned parallel probe expected a lazy stream for its chunk",
                "partitioned_parallel_probe_unexpected_state",
            ));
        },
    };
    let stream = partitioned_hash_equi_join_frame_stream_probe_to_payload(node_id, node, inner);
    stream.into_remaining_frames_owned(lane_execution)
}
