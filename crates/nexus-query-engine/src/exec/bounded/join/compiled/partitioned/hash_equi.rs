use super::*;

pub(in crate::exec::bounded::join::compiled) fn execute_compiled_partitioned_hash_equi_join_materialize(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    strategy_options: KeyedHashJoinStrategyOptions,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    match execute_compiled_partitioned_hash_equi_join_materialize_output(
        context,
        node_id,
        node,
        left,
        right,
        strategy_options,
        false,
        reservations,
        metrics,
    )? {
        BoundedCompiledJoinMaterializeOutput::Frame(frame) => Ok(*frame),
        BoundedCompiledJoinMaterializeOutput::Stream(_) | BoundedCompiledJoinMaterializeOutput::LazyStream(_) => {
            unreachable!("non-preserving partitioned join returns frame")
        },
    }
}

pub(in crate::exec::bounded::join::compiled) fn execute_compiled_partitioned_hash_equi_join_materialize_preserving_stream(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    strategy_options: KeyedHashJoinStrategyOptions,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedCompiledJoinMaterializeOutput> {
    execute_compiled_partitioned_hash_equi_join_materialize_output(
        context,
        node_id,
        node,
        left,
        right,
        strategy_options,
        true,
        reservations,
        metrics,
    )
}

fn execute_compiled_partitioned_hash_equi_join_materialize_output(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    strategy_options: KeyedHashJoinStrategyOptions,
    preserve_stream: bool,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedCompiledJoinMaterializeOutput> {
    validate_partitioned_hash_equi_join(node)?;
    let partition_count = partitioned_join_partition_count(context, left.bytes, right.bytes);
    let plan = PartitionedJoinDriverPlan {
        context,
        node_id,
        node,
        strategy: PARTITIONED_HASH_EQUI_STRATEGY,
        strategy_options,
        partition_count,
        partition_target_bytes: partitioned_join_target_bytes_for_context(context),
    };
    let mut result = run_partitioned_join_driver(
        plan,
        left,
        right,
        reservations,
        metrics,
        materialize_partitioned_hash_equi_join_partition,
    )?;
    let mut stream_outputs = Some(result.outputs);
    let output = if preserve_stream {
        None
    } else {
        let outputs = stream_outputs.take().expect("partitioned join outputs should be available");
        let output = concatenate_streamed_hash_join_outputs(context, node_id, outputs, reservations, metrics)?;
        result.summary.observe_concatenated_output(&output);
        Some(output)
    };
    record_partitioned_join_operator_metrics(plan, result.original, &result.summary, metrics);
    if let Some(output) = output {
        Ok(BoundedCompiledJoinMaterializeOutput::Frame(Box::new(output)))
    } else {
        Ok(BoundedCompiledJoinMaterializeOutput::Stream(
            stream_outputs.expect("partitioned stream outputs should be available"),
        ))
    }
}

fn materialize_partitioned_hash_equi_join_partition(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<PartitionedJoinPartitionOutput> {
    let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
    let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
    let overrides_lease = execution_overrides(context)?;
    let overrides = overrides_lease.overrides();
    let timer = Instant::now();
    let mut txn = JoinMaterializationTxn::with_sides(left, right);
    let (left, right) = txn.sides();
    let preflight_indices = match materialize_compiled_keyed_equality_join_indices_with_preflight(
        context,
        node_id,
        node,
        left,
        right,
        node.spec.build_side,
        &left_keys,
        &right_keys,
        "join_partition_indices",
        reservations,
        metrics,
        &overrides,
    ) {
        Ok(indices) => indices,
        Err(error) => {
            txn.rollback(reservations, metrics);
            return Err(error);
        },
    };
    let CompiledEqualityJoinPreflightIndices { indices, gather_map_reservation, index_bytes, output_rows } =
        preflight_indices;
    txn.set_gather_map_reservation(gather_map_reservation);
    txn.set_indices(indices);
    let (left, right) = txn.sides();
    super::super::common::trace_preflight_entry(
        "partitioned_keyed_partition",
        "partitioned_keyed",
        node_id,
        reservations,
        0,
        output_rows,
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
        output_rows,
        &node.spec.output_projection,
        reservations,
        metrics,
    ) {
        Ok(bytes) => {
            super::super::common::trace_preflight_success("partitioned_keyed_partition", node_id, reservations, bytes);
            bytes
        },
        Err(error) => {
            super::super::common::trace_preflight_failed("partitioned_keyed_partition", node_id, reservations, &error);
            txn.rollback(reservations, metrics);
            return Err(error);
        },
    };
    let output_reservation = match reservations.try_reserve(
        bounded_join_request(
            ReservationKind::JoinOutput,
            "join_partition_output",
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
            txn.rollback(reservations, metrics);
            return Err(error);
        },
    };
    metrics.operator_time += timer.elapsed();
    let output_bytes = match observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            txn.rollback(reservations, metrics);
            return Err(error);
        },
    };
    if let Err(error) =
        reconcile_join_output_upper_bound(node_id, txn.output_reservation_mut(), output_bytes, reservations, metrics)
    {
        txn.rollback(reservations, metrics);
        return Err(error);
    }
    txn.release_gather_map_reservation(reservations, metrics);
    Ok(PartitionedJoinPartitionOutput {
        frame: txn.commit_frame(GpuDataFrame::new(output), reservations, metrics),
        index_bytes,
        mixed_size_data_bytes: 0,
        output_reservation_bytes,
        output_bytes,
        output_rows,
    })
}

pub(in crate::exec::bounded::join::compiled) fn execute_compiled_partitioned_hash_equi_join_probe_from_stream(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    build: BoundedCompiledPreparedJoinBuildState,
    stream: BoundedJoinLeafLocalParquetSourceStream,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedCompiledJoinProbeState> {
    validate_partitioned_hash_equi_join(node)?;
    if !partitioned_streaming_probe_supported(node) {
        return Err(unsupported_bounded_shape(
            "streamed compiled partitioned hash join probe supports inner joins and left joins with a right build side",
            "compiled_join_strategy_unsupported",
        )
        .with_fact("join_type", node.spec.join_type)
        .with_fact("build_side", node.spec.build_side));
    }
    let build_side = build.side;
    let probe_side = probe_side(build_side);
    let build_rows = build.input.rows;
    let build_bytes = build.input.bytes;
    let probe_estimated_bytes = stream.estimated_source_read_bytes().unwrap_or(0);
    let partition_count = partitioned_join_partition_count(context, build.input.bytes, probe_estimated_bytes)
        .max(BoundedPartitionRowPolicy::default().hash_frame_stream_min_partitions());
    let partition_target_bytes = partitioned_join_target_bytes_for_context(context);
    let build_keys = join_keys_for_side(node, build_side);
    let probe_keys = join_keys_for_side(node, probe_side);
    let overrides_lease = execution_overrides(context)?;
    let overrides = overrides_lease.overrides();
    let build_partitioned = hash_partition_join_side(
        node_id,
        build_side,
        build.input,
        &build_keys,
        partition_count,
        &overrides,
        reservations,
        metrics,
    )?;
    let build_partitioned_bytes = build_partitioned.reservation.bytes();
    let mut outputs = Vec::new();
    let mut summary =
        StreamedPartitionedJoinSummary::new(partition_count, partition_target_bytes, build_partitioned_bytes);
    let plan = PartitionedJoinDriverPlan {
        context,
        node_id,
        node,
        strategy: PARTITIONED_HASH_EQUI_STRATEGY,
        strategy_options: build.strategy_options,
        partition_count,
        partition_target_bytes,
    };
    metrics.record_join_strategy_decision(
        node_id,
        &BTreeMap::from([
            ("join_strategy", "partitioned_hash_equi".to_string()),
            ("partition_count", partition_count.to_string()),
            ("partition_target_bytes", partition_target_bytes.to_string()),
            ("build_partition_batch_count", partition_count.to_string()),
            ("build_partitioned_bytes", build_partitioned_bytes.to_string()),
        ]),
    );

    let probe_columns =
        compiled_join_input_columns(node, CompiledJoinExecutionStrategy::PartitionedHashEqui, probe_side)?;
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
                build_side,
                reservations,
                metrics,
            )?;
            let probe_rows = probe.rows;
            let probe_bytes = probe.bytes;
            let probe_partitioned = hash_partition_join_side(
                node_id,
                probe_side,
                probe,
                &probe_keys,
                partition_count,
                &overrides,
                reservations,
                metrics,
            )?;
            let probe_partitioned_bytes = probe_partitioned.reservation.bytes();
            summary.record_probe_partitioned_chunk(probe_rows, probe_bytes, probe_partitioned_bytes);
            for partition in 0..partition_count {
                let (left_partitioned, right_partitioned) = match build_side {
                    JoinSide::Left => (&build_partitioned, &probe_partitioned),
                    JoinSide::Right => (&probe_partitioned, &build_partitioned),
                };
                match materialize_partitioned_join_partition_with_retry(
                    plan,
                    left_partitioned,
                    right_partitioned,
                    partition,
                    reservations,
                    metrics,
                    &mut materialize_partitioned_hash_equi_join_partition,
                ) {
                    Ok(partition_outputs) => {
                        for output in partition_outputs {
                            summary.record_output(&output);
                            outputs.push(output.frame);
                        }
                    },
                    Err(error) => {
                        release_partitioned_join_side(probe_partitioned, reservations, metrics);
                        return Err(error);
                    },
                }
            }
            release_partitioned_join_side(probe_partitioned, reservations, metrics);
            Ok(())
        },
    );
    match stream_result {
        Ok(()) => {
            metrics.record_join_strategy_decision(
                node_id,
                &BTreeMap::from([
                    ("join_strategy", "partitioned_hash_equi".to_string()),
                    ("partition_count", partition_count.to_string()),
                    ("partition_target_bytes", partition_target_bytes.to_string()),
                    ("probe_partition_batch_count", summary.probe_partition_batch_count.to_string()),
                    ("probe_partitioned_bytes", summary.probe_partitioned_bytes.to_string()),
                ]),
            );
            release_partitioned_join_side(build_partitioned, reservations, metrics);
            Ok(BoundedCompiledJoinProbeState::PartitionedStreamed {
                build_side,
                build_rows,
                build_bytes,
                outputs,
                summary,
            })
        },
        Err(error) => {
            release_streamed_join_outputs_reverse(reservations, outputs, metrics);
            release_partitioned_join_side(build_partitioned, reservations, metrics);
            Err(error)
        },
    }
}

pub(in crate::exec::bounded::join::compiled) fn execute_compiled_partitioned_hash_equi_join_probe_from_frame_stream(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    build: BoundedCompiledPreparedJoinBuildState,
    inputs: ResidentFrameStreamPayload,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedCompiledJoinProbeState> {
    validate_partitioned_hash_equi_join(node)?;
    if !partitioned_streaming_probe_supported(node) {
        return Err(unsupported_bounded_shape(
            "streamed compiled partitioned hash join probe supports inner joins and left joins with a right build side",
            "compiled_join_strategy_unsupported",
        )
        .with_fact("join_type", node.spec.join_type)
        .with_fact("build_side", node.spec.build_side));
    }
    let build_side = build.side;
    let probe_side = probe_side(build_side);
    let build_rows = build.input.rows;
    let build_bytes = build.input.bytes;
    let probe_estimated_bytes = inputs.max_per_frame_reservation_bytes();
    let partition_count = partitioned_join_partition_count(context, build.input.bytes, probe_estimated_bytes)
        .max(BoundedPartitionRowPolicy::default().hash_frame_stream_min_partitions());
    let partition_target_bytes = partitioned_join_target_bytes_for_context(context);
    let build_keys = join_keys_for_side(node, build_side);
    let probe_keys = join_keys_for_side(node, probe_side);
    let overrides_lease = execution_overrides(context)?;
    let overrides = overrides_lease.overrides();
    let build_partitioned = hash_partition_join_side(
        node_id,
        build_side,
        build.input,
        &build_keys,
        partition_count,
        &overrides,
        reservations,
        metrics,
    )?;
    let build_partitioned_bytes = build_partitioned.reservation.bytes();
    metrics.record_join_strategy_decision(
        node_id,
        &BTreeMap::from([
            ("join_strategy", "partitioned_hash_equi".to_string()),
            ("partition_count", partition_count.to_string()),
            ("partition_target_bytes", partition_target_bytes.to_string()),
            ("build_partition_batch_count", partition_count.to_string()),
            ("build_partitioned_bytes", build_partitioned_bytes.to_string()),
        ]),
    );

    let probe_columns =
        match compiled_join_input_columns(node, CompiledJoinExecutionStrategy::PartitionedHashEqui, probe_side) {
            Ok(columns) => columns,
            Err(error) => {
                inputs.release_all(reservations, metrics);
                release_partitioned_join_side(build_partitioned, reservations, metrics);
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

pub(in crate::exec::bounded::join::compiled) fn partitioned_hash_equi_join_frame_stream_probe_to_payload(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    probe: PartitionedHashEquiJoinFrameStreamProbeState,
) -> ResidentFrameStreamPayload {
    let max_per_frame_reservation_bytes = probe.partition_target_bytes;
    let aggregate_reservation_bytes = probe.inputs.aggregate_reservation_bytes();
    let aggregate_num_rows = probe.inputs.num_rows();
    let num_columns = node.spec.output_projection.len();
    let source = PartitionedHashEquiJoinProbeStreamSource {
        context: context.clone(),
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
        max_per_frame_reservation_bytes,
        aggregate_reservation_bytes,
        aggregate_num_rows,
        num_columns,
        Vec::new(),
        Box::new(source),
    )
}

struct PartitionedHashEquiJoinProbeStreamSource {
    context: ExecutionContext,
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
    pending_outputs: VecDeque<ResidentGpuFrame>,
    next_partition: usize,
    summary: StreamedPartitionedJoinSummary,
    metrics_recorded: bool,
}

impl PartitionedHashEquiJoinProbeStreamSource {
    fn record_success_metrics(&mut self, metrics: &mut QueryMetrics) {
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
            metrics,
        );
    }

    fn load_next_probe_chunk(
        &mut self,
        reservations: &mut ReservationTracker,
        metrics: &mut QueryMetrics,
    ) -> Option<Result<bool>> {
        let inputs = self.inputs.as_mut()?;
        let mut input = match inputs.next_yielded_frame(reservations, metrics) {
            Some(Ok(input)) => input,
            Some(Err(error)) => return Some(Err(error)),
            None => return None,
        };
        let input = input.take_for_release();
        let probe_side = probe_side(self.build_side);
        let probe = match prepare_bounded_join_side_input_from_resident_frame(
            input,
            &self.probe_columns,
            probe_side,
            self.node_id,
            self.build_side,
            reservations,
            metrics,
        ) {
            Ok(probe) => probe,
            Err(error) => return Some(Err(error)),
        };
        let probe_rows = probe.rows;
        let probe_bytes = probe.bytes;
        let overrides_lease = match execution_overrides(&self.context) {
            Ok(overrides) => overrides,
            Err(error) => return Some(Err(error)),
        };
        let overrides = overrides_lease.overrides();
        let probe_partitioned = match hash_partition_join_side(
            self.node_id,
            probe_side,
            probe,
            &self.probe_keys,
            self.partition_count,
            &overrides,
            reservations,
            metrics,
        ) {
            Ok(partitioned) => partitioned,
            Err(error) => return Some(Err(error)),
        };
        let probe_partitioned_bytes = probe_partitioned.reservation.bytes();
        self.summary.record_probe_partitioned_chunk(probe_rows, probe_bytes, probe_partitioned_bytes);
        self.active_probe_partitioned = Some(probe_partitioned);
        self.next_partition = 0;
        Some(Ok(true))
    }

    fn materialize_next_active_partition(
        &mut self,
        reservations: &mut ReservationTracker,
        metrics: &mut QueryMetrics,
    ) -> Result<()> {
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
            context: &self.context,
            node_id: self.node_id,
            node: &self.node,
            strategy: PARTITIONED_HASH_EQUI_STRATEGY,
            strategy_options: self.strategy_options,
            partition_count: self.partition_count,
            partition_target_bytes: self.partition_target_bytes,
        };
        let outputs = materialize_partitioned_join_partition_with_retry(
            plan,
            left_partitioned,
            right_partitioned,
            partition,
            reservations,
            metrics,
            &mut materialize_partitioned_hash_equi_join_partition,
        )?;
        if is_last_active_partition {
            self.release_active_probe_partitioned(reservations, metrics);
        }
        for output in outputs {
            self.summary.record_output(&output);
            self.pending_outputs.push_back(output.frame);
        }
        Ok(())
    }

    fn release_active_probe_partitioned(&mut self, reservations: &mut ReservationTracker, metrics: &mut QueryMetrics) {
        if let Some(probe_partitioned) = self.active_probe_partitioned.take() {
            release_partitioned_join_side(probe_partitioned, reservations, metrics);
        }
    }
}

impl ResidentFrameStreamSource for PartitionedHashEquiJoinProbeStreamSource {
    fn next_frame(
        &mut self,
        reservations: &mut ReservationTracker,
        metrics: &mut QueryMetrics,
    ) -> Option<Result<ResidentGpuFrame>> {
        loop {
            if let Some(output) = self.pending_outputs.pop_front() {
                return Some(Ok(output));
            }
            if self.active_probe_partitioned.is_some() {
                if self.next_partition < self.partition_count {
                    match self.materialize_next_active_partition(reservations, metrics) {
                        Ok(()) => continue,
                        Err(error) => {
                            self.release(reservations, metrics);
                            return Some(Err(error));
                        },
                    }
                }
                self.release_active_probe_partitioned(reservations, metrics);
            }

            match self.load_next_probe_chunk(reservations, metrics) {
                Some(Ok(true)) => {},
                Some(Ok(false)) => unreachable!("partitioned hash stream probe chunk loading is boolean-true only"),
                Some(Err(error)) => {
                    self.release(reservations, metrics);
                    return Some(Err(error));
                },
                None => {
                    self.record_success_metrics(metrics);
                    self.release(reservations, metrics);
                    return None;
                },
            }
        }
    }

    fn release(&mut self, reservations: &mut ReservationTracker, metrics: &mut QueryMetrics) {
        release_streamed_join_outputs_reverse(reservations, self.pending_outputs.drain(..).collect(), metrics);
        self.release_active_probe_partitioned(reservations, metrics);
        if let Some(input) = self.inputs.take() {
            input.release_all(reservations, metrics);
        }
        if let Some(build_partitioned) = self.build_partitioned.take() {
            release_partitioned_join_side(build_partitioned, reservations, metrics);
        }
    }
}
