use super::*;

pub(in crate::exec::bounded::join::compiled) fn execute_compiled_partitioned_mixed_keyed_residual_join_materialize(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    match execute_compiled_partitioned_mixed_keyed_residual_join_materialize_output(
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
            unreachable!("non-preserving partitioned mixed join returns frame")
        },
    }
}

pub(in crate::exec::bounded::join::compiled) fn execute_compiled_partitioned_mixed_keyed_residual_join_materialize_preserving_stream(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedCompiledJoinMaterializeOutput> {
    execute_compiled_partitioned_mixed_keyed_residual_join_materialize_output(
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

fn execute_compiled_partitioned_mixed_keyed_residual_join_materialize_output(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    preserve_stream: bool,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedCompiledJoinMaterializeOutput> {
    validate_partitioned_mixed_keyed_residual_join(node)?;
    let original_left_rows = left.rows;
    let original_left_bytes = left.bytes;
    let original_right_rows = right.rows;
    let original_right_bytes = right.bytes;
    let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
    let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
    let partition_count =
        partitioned_mixed_join_partition_count(context, left.rows, left.bytes, right.rows, right.bytes);
    let partition_target_bytes = partitioned_join_target_bytes_for_context(context);
    metrics.record_join_strategy_decision(
        node_id,
        &BTreeMap::from([
            ("join_strategy", "partitioned_mixed_keyed_residual".to_string()),
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
            ("join_strategy", "partitioned_mixed_keyed_residual".to_string()),
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

    if preserve_stream {
        let forecast = match forecast_partitioned_mixed_join_stream(
            context,
            node_id,
            node,
            &left_partitioned,
            &right_partitioned,
            partition_count,
            reservations,
            metrics,
        ) {
            Ok(forecast) => forecast,
            Err(error) => {
                release_partitioned_join_side(left_partitioned, reservations, metrics);
                release_partitioned_join_side(right_partitioned, reservations, metrics);
                return Err(error);
            },
        };
        let left_partitioned_bytes = left_partitioned.reservation.bytes();
        let right_partitioned_bytes = right_partitioned.reservation.bytes();
        let stream = PartitionedMixedJoinStreamSource {
            context: context.clone(),
            node_id,
            node: node.clone(),
            left_partitioned: Some(left_partitioned),
            right_partitioned: Some(right_partitioned),
            partition_count,
            next_partition: 0,
            original_left_rows,
            original_left_bytes,
            original_right_rows,
            original_right_bytes,
            left_partitioned_bytes,
            right_partitioned_bytes,
            total_index_bytes: 0,
            total_mixed_size_data_bytes: 0,
            total_output_bytes: 0,
            total_output_rows: 0,
            total_output_reservation_bytes: 0,
            metrics_recorded: false,
        };
        let payload = ResidentFrameStreamPayload::from_producer(
            forecast.max_per_frame_reservation_bytes,
            forecast.aggregate_reservation_bytes,
            forecast.aggregate_num_rows,
            node.spec.output_projection.len(),
            Vec::new(),
            Box::new(stream),
        );
        return Ok(BoundedCompiledJoinMaterializeOutput::LazyStream(payload));
    }

    let mut outputs = Vec::with_capacity(partition_count);
    let mut total_index_bytes = 0u64;
    let mut total_mixed_size_data_bytes = 0u64;
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
        match materialize_partitioned_mixed_keyed_residual_join_partition(
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
                total_mixed_size_data_bytes = total_mixed_size_data_bytes.saturating_add(output.mixed_size_data_bytes);
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
    let output = concatenate_streamed_hash_join_outputs(context, node_id, outputs, reservations, metrics)?;
    let output_bytes = output.reservation.bytes();
    total_output_bytes = total_output_bytes.max(output_bytes);
    metrics.record_operator_metrics(crate::metrics::OperatorMetrics::join(
        node_id,
        crate::metrics::JoinOperatorMetrics {
            join_type: node.spec.join_type.to_string(),
            execution_strategy: "bounded_partitioned_mixed_keyed_residual_join".to_string(),
            key_count: node.spec.keys.len(),
            residual_predicate: true,
            null_equality: node.spec.null_equality.to_string(),
            build_side: node.spec.build_side.to_string(),
            probe_side: probe_side(node.spec.build_side).to_string(),
            build_rows: side_metric(node.spec.build_side, original_left_rows, original_right_rows),
            build_bytes: side_metric(node.spec.build_side, original_left_bytes, original_right_bytes),
            probe_rows: side_metric(probe_side(node.spec.build_side), original_left_rows, original_right_rows),
            probe_bytes: side_metric(probe_side(node.spec.build_side), original_left_bytes, original_right_bytes),
            hash_table_bytes: 0,
            filtered_probe_state_bytes: 0,
            mixed_size_data_bytes: total_mixed_size_data_bytes,
            gather_map_bytes: total_index_bytes,
            output_reservation_bytes: total_output_reservation_bytes,
            output_rows: total_output_rows,
            output_bytes: total_output_bytes,
        },
    ));
    Ok(BoundedCompiledJoinMaterializeOutput::Frame(Box::new(output)))
}

fn materialize_partitioned_mixed_keyed_residual_join_partition(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<PartitionedJoinPartitionOutput> {
    let residual_columns = residual_input_columns(node)?;
    let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
    let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
    let overrides = execution_overrides(context);
    let predicate = lower_join_residual_predicate(context, node, &residual_columns)?;
    let options = dataframe_mixed_join_options(&left_keys, &right_keys, &residual_columns, node.spec.null_equality);
    let timer = Instant::now();
    let size_data = match node.spec.join_type {
        JoinType::Inner => left
            .frame
            .with_execution(&overrides)
            .mixed_inner_join_size_data(&right.frame, &predicate, options)
            .map(CompiledMixedSizeData::Pair),
        JoinType::Left => left
            .frame
            .with_execution(&overrides)
            .mixed_left_join_size_data(&right.frame, &predicate, options)
            .map(CompiledMixedSizeData::Pair),
        JoinType::Full => left
            .frame
            .with_execution(&overrides)
            .mixed_full_join_size_data(&right.frame, &predicate, options)
            .map(CompiledMixedSizeData::Pair),
        JoinType::LeftSemi => left
            .frame
            .with_execution(&overrides)
            .mixed_left_semi_join_size_data(&right.frame, &predicate, options)
            .map(CompiledMixedSizeData::LeftOnly),
        JoinType::LeftAnti => left
            .frame
            .with_execution(&overrides)
            .mixed_left_anti_join_size_data(&right.frame, &predicate, options)
            .map(CompiledMixedSizeData::LeftOnly),
        JoinType::Cross | JoinType::LeftMark => unreachable!("validated above"),
    }
    .map_err(|error| join_index_error(error, node))?;
    let output_rows = size_data.output_size() as u64;
    let mixed_size_data_bytes = size_data.device_allocated_size_bytes(node)?;
    let size_data_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::MixedSizeData,
            "join_partition_mixed_size_data",
            mixed_size_data_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;
    metrics.record_mixed_join_size_data_bytes(mixed_size_data_bytes);
    let gather_map_reservation_bytes = size_data.gather_map_reservation_size_bytes(node)?;
    let gather_map_reservation = match reservations.try_reserve(
        bounded_join_request(
            ReservationKind::GatherMap,
            "join_partition_mixed_indices",
            gather_map_reservation_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            reservations.release(size_data_reservation, metrics);
            drop(left);
            drop(right);
            return Err(error);
        },
    };
    let indices = match size_data.materialize_indices_in(&overrides, node) {
        Ok(indices) => indices,
        Err(error) => {
            reservations.release(gather_map_reservation, metrics);
            reservations.release(size_data_reservation, metrics);
            drop(left);
            drop(right);
            return Err(error);
        },
    };
    reservations.release(size_data_reservation, metrics);
    let index_bytes = match indices.device_allocated_size_bytes(node) {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(indices);
            reservations.release(gather_map_reservation, metrics);
            drop(left);
            drop(right);
            return Err(error);
        },
    };
    let forecast_output_reservation_bytes = match partitioned_mixed_join_output_upper_bound_bytes(
        context,
        node_id,
        node,
        &left,
        &right,
        output_rows,
        reservations,
        metrics,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(indices);
            reservations.release(gather_map_reservation, metrics);
            drop(left);
            drop(right);
            return Err(error);
        },
    };
    super::super::common::trace_preflight_entry(
        "partitioned_mixed_partition",
        "partitioned_mixed",
        node_id,
        reservations,
        index_bytes,
        output_rows,
        left.rows,
        left.bytes,
        right.rows,
        right.bytes,
    );
    let output_reservation_bytes = forecast_output_reservation_bytes;
    super::super::common::trace_preflight_success(
        "partitioned_mixed_partition",
        node_id,
        reservations,
        output_reservation_bytes,
    );
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
        mixed_size_data_bytes,
        output_reservation_bytes,
        output_bytes,
        output_rows,
    })
}

fn partitioned_mixed_join_output_upper_bound_bytes(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: &BoundedJoinSideFrame,
    right: &BoundedJoinSideFrame,
    expected_output_rows: u64,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<u64> {
    let left_projection = projected_output_columns_for_retained_sizing(&node.spec.output_projection, JoinSide::Left);
    let right_projection = projected_output_columns_for_retained_sizing(&node.spec.output_projection, JoinSide::Right);
    let left_bytes = join_side_output_upper_bound_bytes(
        context,
        node_id,
        &left.frame,
        expected_output_rows,
        &left_projection,
        JoinSide::Left,
        reservations,
        metrics,
    )?;
    let right_bytes = join_side_output_upper_bound_bytes(
        context,
        node_id,
        &right.frame,
        expected_output_rows,
        &right_projection,
        JoinSide::Right,
        reservations,
        metrics,
    )?;
    Ok(left_bytes.saturating_add(right_bytes))
}

struct PartitionedMixedJoinStreamForecast {
    max_per_frame_reservation_bytes: u64,
    aggregate_reservation_bytes: u64,
    aggregate_num_rows: u64,
}

fn forecast_partitioned_mixed_join_stream(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left_partitioned: &PartitionedJoinSide,
    right_partitioned: &PartitionedJoinSide,
    partition_count: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<PartitionedMixedJoinStreamForecast> {
    let mut max_per_frame_reservation_bytes = 0u64;
    let mut aggregate_reservation_bytes = 0u64;
    let mut aggregate_num_rows = 0u64;
    for partition in 0..partition_count {
        let left_part = slice_partition_join_side(
            context,
            node_id,
            JoinSide::Left,
            node.spec.build_side,
            left_partitioned,
            partition,
            reservations,
            metrics,
        )?;
        let right_part = match slice_partition_join_side(
            context,
            node_id,
            JoinSide::Right,
            node.spec.build_side,
            right_partitioned,
            partition,
            reservations,
            metrics,
        ) {
            Ok(partition) => partition,
            Err(error) => {
                drop(left_part);
                return Err(error);
            },
        };
        let forecast = match forecast_partitioned_mixed_join_partition(
            context,
            node_id,
            node,
            &left_part,
            &right_part,
            reservations,
            metrics,
        ) {
            Ok(forecast) => forecast,
            Err(error) => {
                drop(left_part);
                drop(right_part);
                return Err(error);
            },
        };
        aggregate_num_rows = aggregate_num_rows.saturating_add(forecast.output_rows);
        aggregate_reservation_bytes = aggregate_reservation_bytes.saturating_add(forecast.output_reservation_bytes);
        max_per_frame_reservation_bytes = max_per_frame_reservation_bytes.max(forecast.output_reservation_bytes);
        drop(left_part);
        drop(right_part);
    }
    Ok(PartitionedMixedJoinStreamForecast {
        max_per_frame_reservation_bytes,
        aggregate_reservation_bytes,
        aggregate_num_rows,
    })
}

struct PartitionedMixedJoinPartitionForecast {
    output_rows: u64,
    output_reservation_bytes: u64,
}

fn forecast_partitioned_mixed_join_partition(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: &BoundedJoinSideFrame,
    right: &BoundedJoinSideFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<PartitionedMixedJoinPartitionForecast> {
    let residual_columns = residual_input_columns(node)?;
    let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
    let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
    let overrides = execution_overrides(context);
    let predicate = lower_join_residual_predicate(context, node, &residual_columns)?;
    let options = dataframe_mixed_join_options(&left_keys, &right_keys, &residual_columns, node.spec.null_equality);
    let size_data = match node.spec.join_type {
        JoinType::Inner => left
            .frame
            .with_execution(&overrides)
            .mixed_inner_join_size_data(&right.frame, &predicate, options)
            .map(CompiledMixedSizeData::Pair),
        JoinType::Left => left
            .frame
            .with_execution(&overrides)
            .mixed_left_join_size_data(&right.frame, &predicate, options)
            .map(CompiledMixedSizeData::Pair),
        JoinType::Full => left
            .frame
            .with_execution(&overrides)
            .mixed_full_join_size_data(&right.frame, &predicate, options)
            .map(CompiledMixedSizeData::Pair),
        JoinType::LeftSemi => left
            .frame
            .with_execution(&overrides)
            .mixed_left_semi_join_size_data(&right.frame, &predicate, options)
            .map(CompiledMixedSizeData::LeftOnly),
        JoinType::LeftAnti => left
            .frame
            .with_execution(&overrides)
            .mixed_left_anti_join_size_data(&right.frame, &predicate, options)
            .map(CompiledMixedSizeData::LeftOnly),
        JoinType::Cross | JoinType::LeftMark => unreachable!("validated above"),
    }
    .map_err(|error| join_index_error(error, node))?;
    let output_rows = size_data.output_size() as u64;
    let mixed_size_data_bytes = size_data.device_allocated_size_bytes(node)?;
    let size_data_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::MixedSizeData,
            "join_partition_mixed_size_data_forecast",
            mixed_size_data_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;
    drop(size_data);
    reservations.release(size_data_reservation, metrics);
    let output_reservation_bytes = partitioned_mixed_join_output_upper_bound_bytes(
        context,
        node_id,
        node,
        left,
        right,
        output_rows,
        reservations,
        metrics,
    )?;
    Ok(PartitionedMixedJoinPartitionForecast { output_rows, output_reservation_bytes })
}

struct PartitionedMixedJoinStreamSource {
    context: ExecutionContext,
    node_id: NodeId,
    node: JoinNode,
    left_partitioned: Option<PartitionedJoinSide>,
    right_partitioned: Option<PartitionedJoinSide>,
    partition_count: usize,
    next_partition: usize,
    original_left_rows: u64,
    original_left_bytes: u64,
    original_right_rows: u64,
    original_right_bytes: u64,
    left_partitioned_bytes: u64,
    right_partitioned_bytes: u64,
    total_index_bytes: u64,
    total_mixed_size_data_bytes: u64,
    total_output_bytes: u64,
    total_output_rows: u64,
    total_output_reservation_bytes: u64,
    metrics_recorded: bool,
}

impl PartitionedMixedJoinStreamSource {
    fn record_success_metrics(&mut self, metrics: &mut QueryMetrics) {
        if self.metrics_recorded {
            return;
        }
        self.metrics_recorded = true;
        metrics.record_join_strategy_decision(
            self.node_id,
            &BTreeMap::from([
                ("join_strategy", "partitioned_mixed_keyed_residual".to_string()),
                ("partition_count", self.partition_count.to_string()),
                ("partition_target_bytes", partitioned_join_target_bytes_for_context(&self.context).to_string()),
                ("build_partition_batch_count", self.partition_count.to_string()),
                ("probe_partition_batch_count", self.partition_count.to_string()),
                (
                    "build_partitioned_bytes",
                    side_metric(self.node.spec.build_side, self.left_partitioned_bytes, self.right_partitioned_bytes)
                        .to_string(),
                ),
                (
                    "probe_partitioned_bytes",
                    side_metric(
                        probe_side(self.node.spec.build_side),
                        self.left_partitioned_bytes,
                        self.right_partitioned_bytes,
                    )
                    .to_string(),
                ),
            ]),
        );
        metrics.record_operator_metrics(crate::metrics::OperatorMetrics::join(
            self.node_id,
            crate::metrics::JoinOperatorMetrics {
                join_type: self.node.spec.join_type.to_string(),
                execution_strategy: "bounded_partitioned_mixed_keyed_residual_join".to_string(),
                key_count: self.node.spec.keys.len(),
                residual_predicate: true,
                null_equality: self.node.spec.null_equality.to_string(),
                build_side: self.node.spec.build_side.to_string(),
                probe_side: probe_side(self.node.spec.build_side).to_string(),
                build_rows: side_metric(self.node.spec.build_side, self.original_left_rows, self.original_right_rows),
                build_bytes: side_metric(
                    self.node.spec.build_side,
                    self.original_left_bytes,
                    self.original_right_bytes,
                ),
                probe_rows: side_metric(
                    probe_side(self.node.spec.build_side),
                    self.original_left_rows,
                    self.original_right_rows,
                ),
                probe_bytes: side_metric(
                    probe_side(self.node.spec.build_side),
                    self.original_left_bytes,
                    self.original_right_bytes,
                ),
                hash_table_bytes: 0,
                filtered_probe_state_bytes: 0,
                mixed_size_data_bytes: self.total_mixed_size_data_bytes,
                gather_map_bytes: self.total_index_bytes,
                output_reservation_bytes: self.total_output_reservation_bytes,
                output_rows: self.total_output_rows,
                output_bytes: self.total_output_bytes,
            },
        ));
    }
}

impl ResidentFrameStreamSource for PartitionedMixedJoinStreamSource {
    fn next_frame(
        &mut self,
        reservations: &mut ReservationTracker,
        metrics: &mut QueryMetrics,
    ) -> Option<Result<ResidentGpuFrame>> {
        if self.next_partition >= self.partition_count {
            self.record_success_metrics(metrics);
            self.release(reservations, metrics);
            return None;
        }
        let partition = self.next_partition;
        self.next_partition += 1;
        let left_partitioned =
            self.left_partitioned.as_ref().expect("left partitioned side released before stream end");
        let right_partitioned =
            self.right_partitioned.as_ref().expect("right partitioned side released before stream end");
        let left_part = match slice_partition_join_side(
            &self.context,
            self.node_id,
            JoinSide::Left,
            self.node.spec.build_side,
            left_partitioned,
            partition,
            reservations,
            metrics,
        ) {
            Ok(partition) => partition,
            Err(error) => return Some(Err(error)),
        };
        let right_part = match slice_partition_join_side(
            &self.context,
            self.node_id,
            JoinSide::Right,
            self.node.spec.build_side,
            right_partitioned,
            partition,
            reservations,
            metrics,
        ) {
            Ok(partition) => partition,
            Err(error) => {
                drop(left_part);
                return Some(Err(error));
            },
        };
        match materialize_partitioned_mixed_keyed_residual_join_partition(
            &self.context,
            self.node_id,
            &self.node,
            left_part,
            right_part,
            reservations,
            metrics,
        ) {
            Ok(output) => {
                self.total_index_bytes = self.total_index_bytes.saturating_add(output.index_bytes);
                self.total_mixed_size_data_bytes =
                    self.total_mixed_size_data_bytes.saturating_add(output.mixed_size_data_bytes);
                self.total_output_rows = self.total_output_rows.saturating_add(output.output_rows);
                self.total_output_bytes = self.total_output_bytes.saturating_add(output.output_bytes);
                self.total_output_reservation_bytes =
                    self.total_output_reservation_bytes.saturating_add(output.output_reservation_bytes);
                Some(Ok(output.frame))
            },
            Err(error) => Some(Err(error)),
        }
    }

    fn release(&mut self, reservations: &mut ReservationTracker, metrics: &mut QueryMetrics) {
        if let Some(left) = self.left_partitioned.take() {
            release_partitioned_join_side(left, reservations, metrics);
        }
        if let Some(right) = self.right_partitioned.take() {
            release_partitioned_join_side(right, reservations, metrics);
        }
    }
}
