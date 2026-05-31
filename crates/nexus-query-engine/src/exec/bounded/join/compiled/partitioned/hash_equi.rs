use super::*;

pub(in crate::exec::bounded::join::compiled) fn execute_compiled_partitioned_hash_equi_join_materialize(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    match execute_compiled_partitioned_hash_equi_join_materialize_output(
        context,
        node_id,
        node,
        left,
        right,
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
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedCompiledJoinMaterializeOutput> {
    execute_compiled_partitioned_hash_equi_join_materialize_output(
        context,
        node_id,
        node,
        left,
        right,
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
    preserve_stream: bool,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedCompiledJoinMaterializeOutput> {
    validate_partitioned_hash_equi_join(node)?;
    let original_left_rows = left.rows;
    let original_left_bytes = left.bytes;
    let original_right_rows = right.rows;
    let original_right_bytes = right.bytes;
    let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
    let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
    let partition_count = partitioned_join_partition_count(context, left.bytes, right.bytes);
    let partition_target_bytes = match context.memory_control_policy() {
        MemoryControlPolicy::Bounded { device_budget_bytes, .. } => {
            partitioned_join_target_partition_bytes(device_budget_bytes)
        },
        _ => 0,
    };
    metrics.record_join_strategy_decision(
        node_id,
        &BTreeMap::from([
            ("join_strategy", "partitioned_hash_equi".to_string()),
            ("join_type", node.spec.join_type.to_string()),
            ("build_side", node.spec.build_side.to_string()),
            ("key_count", node.spec.keys.len().to_string()),
            ("observed_build_rows", side_metric(node.spec.build_side, left.rows, right.rows).to_string()),
            ("build_side_bytes", side_metric(node.spec.build_side, left.bytes, right.bytes).to_string()),
            ("probe_side_bytes", side_metric(probe_side(node.spec.build_side), left.bytes, right.bytes).to_string()),
            ("probe_reuse_count", "1".to_string()),
            (
                "reusable_hash_join_rejected_reason",
                partitioned_join_reusable_rejected_reason(
                    side_metric(node.spec.build_side, left.rows, right.rows),
                    side_metric(node.spec.build_side, left.bytes, right.bytes),
                    side_metric(probe_side(node.spec.build_side), left.bytes, right.bytes),
                )
                .to_string(),
            ),
            ("partition_count", partition_count.to_string()),
            ("partition_target_bytes", partition_target_bytes.to_string()),
        ]),
    );
    let overrides = execution_overrides(context);
    let left_partitioned = hash_partition_join_side(
        context,
        node_id,
        JoinSide::Left,
        left,
        &left_keys,
        partition_count,
        &overrides,
        reservations,
        metrics,
    )?;
    let right_partitioned = match hash_partition_join_side(
        context,
        node_id,
        JoinSide::Right,
        right,
        &right_keys,
        partition_count,
        &overrides,
        reservations,
        metrics,
    ) {
        Ok(partitioned) => partitioned,
        Err(error) => {
            release_partitioned_join_side(left_partitioned, reservations, metrics);
            return Err(error);
        },
    };
    let left_partitioned_bytes = left_partitioned.reservation.bytes();
    let right_partitioned_bytes = right_partitioned.reservation.bytes();
    metrics.record_join_strategy_decision(
        node_id,
        &BTreeMap::from([
            ("join_strategy", "partitioned_hash_equi".to_string()),
            ("partition_count", partition_count.to_string()),
            ("partition_target_bytes", partition_target_bytes.to_string()),
            ("build_partition_batch_count", partition_count.to_string()),
            ("probe_partition_batch_count", partition_count.to_string()),
            (
                "build_partitioned_bytes",
                side_metric(node.spec.build_side, left_partitioned_bytes, right_partitioned_bytes).to_string(),
            ),
            (
                "probe_partitioned_bytes",
                side_metric(probe_side(node.spec.build_side), left_partitioned_bytes, right_partitioned_bytes)
                    .to_string(),
            ),
        ]),
    );

    let mut outputs = Vec::with_capacity(partition_count);
    let mut total_index_bytes = 0u64;
    let mut total_output_bytes = 0u64;
    let mut total_output_rows = 0u64;
    let mut total_output_reservation_bytes = 0u64;
    for partition in 0..partition_count {
        let left_part = match slice_partition_join_side(
            context,
            node_id,
            JoinSide::Left,
            node.spec.build_side,
            &left_partitioned,
            partition,
            reservations,
            metrics,
        ) {
            Ok(partition) => partition,
            Err(error) => {
                release_streamed_join_outputs_reverse(reservations, outputs, metrics);
                release_partitioned_join_side(left_partitioned, reservations, metrics);
                release_partitioned_join_side(right_partitioned, reservations, metrics);
                return Err(error);
            },
        };
        let right_part = match slice_partition_join_side(
            context,
            node_id,
            JoinSide::Right,
            node.spec.build_side,
            &right_partitioned,
            partition,
            reservations,
            metrics,
        ) {
            Ok(partition) => partition,
            Err(error) => {
                drop(left_part);
                release_streamed_join_outputs_reverse(reservations, outputs, metrics);
                release_partitioned_join_side(left_partitioned, reservations, metrics);
                release_partitioned_join_side(right_partitioned, reservations, metrics);
                return Err(error);
            },
        };
        match materialize_partitioned_hash_equi_join_partition(
            context,
            node_id,
            node,
            left_part,
            right_part,
            reservations,
            metrics,
        ) {
            Ok(output) => {
                total_index_bytes = total_index_bytes.saturating_add(output.index_bytes);
                total_output_rows = total_output_rows.saturating_add(output.output_rows);
                total_output_bytes = total_output_bytes.saturating_add(output.output_bytes);
                total_output_reservation_bytes =
                    total_output_reservation_bytes.saturating_add(output.output_reservation_bytes);
                outputs.push(output.frame);
            },
            Err(error) => {
                release_streamed_join_outputs_reverse(reservations, outputs, metrics);
                release_partitioned_join_side(left_partitioned, reservations, metrics);
                release_partitioned_join_side(right_partitioned, reservations, metrics);
                return Err(error);
            },
        }
    }
    release_partitioned_join_side(left_partitioned, reservations, metrics);
    release_partitioned_join_side(right_partitioned, reservations, metrics);
    let mut stream_outputs = Some(outputs);
    let output = if preserve_stream {
        None
    } else {
        let outputs = stream_outputs.take().expect("partitioned join outputs should be available");
        let output = concatenate_streamed_hash_join_outputs(context, node_id, outputs, reservations, metrics)?;
        let output_bytes = output.reservation.bytes();
        total_output_bytes = total_output_bytes.max(output_bytes);
        Some(output)
    };
    metrics.record_operator_metrics(crate::metrics::OperatorMetrics::join(
        node_id,
        crate::metrics::JoinOperatorMetrics {
            join_type: node.spec.join_type.to_string(),
            execution_strategy: "bounded_partitioned_hash_equi_join".to_string(),
            key_count: node.spec.keys.len(),
            residual_predicate: false,
            null_equality: node.spec.null_equality.to_string(),
            build_side: node.spec.build_side.to_string(),
            probe_side: probe_side(node.spec.build_side).to_string(),
            build_rows: side_metric(node.spec.build_side, original_left_rows, original_right_rows),
            build_bytes: side_metric(node.spec.build_side, original_left_bytes, original_right_bytes),
            probe_rows: side_metric(probe_side(node.spec.build_side), original_left_rows, original_right_rows),
            probe_bytes: side_metric(probe_side(node.spec.build_side), original_left_bytes, original_right_bytes),
            hash_table_bytes: 0,
            filtered_probe_state_bytes: 0,
            mixed_size_data_bytes: 0,
            gather_map_bytes: total_index_bytes,
            output_reservation_bytes: total_output_reservation_bytes,
            output_rows: total_output_rows,
            output_bytes: total_output_bytes,
        },
    ));
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
    let options = dataframe_join_options(&left_keys, &right_keys, node.spec.null_equality);
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let pair_indices = match node.spec.join_type {
        JoinType::Inner => left.frame.with_execution(&overrides).inner_join_indices(&right.frame, options),
        JoinType::Left => left.frame.with_execution(&overrides).left_join_indices(&right.frame, options),
        JoinType::Full => left.frame.with_execution(&overrides).full_join_indices(&right.frame, options),
        JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark | JoinType::Cross => {
            unreachable!("validated above")
        },
    }
    .map_err(|error| join_index_error(error, node))?;
    let index_bytes = pair_indices.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.bounded.join.partitioned_index_bytes",
            "failed to observe compiled partitioned join index memory",
            error,
        )
        .with_fact("join_type", node.spec.join_type)
    })?;
    let gather_map_reservation = match reservations.try_reserve(
        bounded_join_request(
            ReservationKind::GatherMap,
            "join_partition_indices",
            index_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            drop(pair_indices);
            drop(left);
            drop(right);
            return Err(error);
        },
    };
    let output_rows = pair_indices.len() as u64;
    let indices = CompiledJoinIndices::Pair { left: pair_indices.left().clone(), right: pair_indices.right().clone() };
    drop(pair_indices);
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
    let output_reservation_bytes = match indices.output_upper_bound_size_bytes(
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
            drop(indices);
            reservations.release(gather_map_reservation, metrics);
            drop(left);
            drop(right);
            return Err(error);
        },
    };
    let mut output_reservation = match reservations.try_reserve(
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
            drop(indices);
            reservations.release(gather_map_reservation, metrics);
            drop(left);
            drop(right);
            return Err(error);
        },
    };
    reservations.reclaim_source_cache_for_external_working_set(reservations.budget(), metrics)?;
    let output = match indices.materialize_output(
        &left.frame,
        &right.frame,
        &overrides,
        &node.spec.output_projection,
        node.spec.join_type,
    ) {
        Ok(output) => output,
        Err(error) => {
            drop(indices);
            reservations.release(gather_map_reservation, metrics);
            reservations.release(output_reservation, metrics);
            drop(left);
            drop(right);
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
            drop(left);
            drop(right);
            return Err(error);
        },
    };
    if let Err(error) =
        reconcile_join_output_upper_bound(node_id, &mut output_reservation, output_bytes, reservations, metrics)
    {
        drop(output);
        drop(indices);
        reservations.release(gather_map_reservation, metrics);
        reservations.release(output_reservation, metrics);
        drop(left);
        drop(right);
        return Err(error);
    }
    drop(indices);
    reservations.release(gather_map_reservation, metrics);
    drop(left);
    drop(right);
    Ok(PartitionedJoinPartitionOutput {
        frame: ResidentGpuFrame { frame: GpuDataFrame::new(output), reservation: output_reservation },
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
        .max(PARTITIONED_HASH_FRAME_STREAM_MIN_PARTITIONS);
    let partition_target_bytes = partitioned_join_target_bytes_for_context(context);
    let build_keys = join_keys_for_side(node, build_side);
    let probe_keys = join_keys_for_side(node, probe_side);
    let overrides = execution_overrides(context);
    let build_partitioned = hash_partition_join_side(
        context,
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
        compiled_join_input_columns(context, node, CompiledJoinExecutionStrategy::PartitionedHashEqui, probe_side)?;
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
                context,
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
                let build_part = match slice_partition_join_side(
                    context,
                    node_id,
                    build_side,
                    build_side,
                    &build_partitioned,
                    partition,
                    reservations,
                    metrics,
                ) {
                    Ok(partition) => partition,
                    Err(error) => {
                        release_partitioned_join_side(probe_partitioned, reservations, metrics);
                        return Err(error);
                    },
                };
                let probe_part = match slice_partition_join_side(
                    context,
                    node_id,
                    probe_side,
                    build_side,
                    &probe_partitioned,
                    partition,
                    reservations,
                    metrics,
                ) {
                    Ok(partition) => partition,
                    Err(error) => {
                        drop(build_part);
                        release_partitioned_join_side(probe_partitioned, reservations, metrics);
                        return Err(error);
                    },
                };
                let (left_part, right_part) = match build_side {
                    JoinSide::Left => (build_part, probe_part),
                    JoinSide::Right => (probe_part, build_part),
                };
                match materialize_partitioned_hash_equi_join_partition(
                    context,
                    node_id,
                    node,
                    left_part,
                    right_part,
                    reservations,
                    metrics,
                ) {
                    Ok(output) => {
                        summary.record_output(&output);
                        outputs.push(output.frame);
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
        .max(PARTITIONED_HASH_FRAME_STREAM_MIN_PARTITIONS);
    let partition_target_bytes = partitioned_join_target_bytes_for_context(context);
    let build_keys = join_keys_for_side(node, build_side);
    let probe_keys = join_keys_for_side(node, probe_side);
    let overrides = execution_overrides(context);
    let build_partitioned = hash_partition_join_side(
        context,
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

    let probe_columns = match compiled_join_input_columns(
        context,
        node,
        CompiledJoinExecutionStrategy::PartitionedHashEqui,
        probe_side,
    ) {
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
        probe_columns: probe.probe_columns,
        probe_keys: probe.probe_keys,
        active_probe_partitioned: None,
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
    probe_columns: Vec<String>,
    probe_keys: Vec<String>,
    active_probe_partitioned: Option<PartitionedJoinSide>,
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
        let overrides = execution_overrides(&self.context);
        let probe_partitioned = match hash_partition_join_side(
            &self.context,
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
    ) -> Result<ResidentGpuFrame> {
        let partition = self.next_partition;
        self.next_partition += 1;
        let is_last_active_partition = self.next_partition >= self.partition_count;
        let build_partitioned =
            self.build_partitioned.as_ref().expect("partitioned hash join build side released before stream end");
        let probe_partitioned = self
            .active_probe_partitioned
            .as_ref()
            .expect("partitioned hash join probe chunk missing before stream end");
        let probe_side = probe_side(self.build_side);
        let build_part = slice_partition_join_side(
            &self.context,
            self.node_id,
            self.build_side,
            self.build_side,
            build_partitioned,
            partition,
            reservations,
            metrics,
        )?;
        let probe_part = match slice_partition_join_side(
            &self.context,
            self.node_id,
            probe_side,
            self.build_side,
            probe_partitioned,
            partition,
            reservations,
            metrics,
        ) {
            Ok(partition) => partition,
            Err(error) => {
                drop(build_part);
                return Err(error);
            },
        };
        let (left_part, right_part) = match self.build_side {
            JoinSide::Left => (build_part, probe_part),
            JoinSide::Right => (probe_part, build_part),
        };
        if is_last_active_partition {
            self.release_active_probe_partitioned(reservations, metrics);
        }
        let output = materialize_partitioned_hash_equi_join_partition(
            &self.context,
            self.node_id,
            &self.node,
            left_part,
            right_part,
            reservations,
            metrics,
        )
        .map_err(|error| error.with_fact("partition", partition))?;
        self.summary.record_output(&output);
        Ok(output.frame)
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
            if self.active_probe_partitioned.is_some() {
                if self.next_partition < self.partition_count {
                    return Some(self.materialize_next_active_partition(reservations, metrics));
                }
                self.release_active_probe_partitioned(reservations, metrics);
            }

            match self.load_next_probe_chunk(reservations, metrics) {
                Some(Ok(true)) => {},
                Some(Ok(false)) => unreachable!("partitioned hash stream probe chunk loading is boolean-true only"),
                Some(Err(error)) => return Some(Err(error)),
                None => {
                    self.record_success_metrics(metrics);
                    self.release(reservations, metrics);
                    return None;
                },
            }
        }
    }

    fn release(&mut self, reservations: &mut ReservationTracker, metrics: &mut QueryMetrics) {
        self.release_active_probe_partitioned(reservations, metrics);
        if let Some(input) = self.inputs.take() {
            input.release_all(reservations, metrics);
        }
        if let Some(build_partitioned) = self.build_partitioned.take() {
            release_partitioned_join_side(build_partitioned, reservations, metrics);
        }
    }
}
