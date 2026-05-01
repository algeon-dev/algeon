use super::*;

pub(in crate::exec::bounded::join::compiled) fn execute_compiled_partitioned_mixed_keyed_residual_join_materialize(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    strategy_options: KeyedHashJoinStrategyOptions,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    match execute_compiled_partitioned_mixed_keyed_residual_join_materialize_output(
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
    strategy_options: KeyedHashJoinStrategyOptions,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<BoundedCompiledJoinMaterializeOutput> {
    execute_compiled_partitioned_mixed_keyed_residual_join_materialize_output(
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

fn execute_compiled_partitioned_mixed_keyed_residual_join_materialize_output(
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
    validate_partitioned_mixed_keyed_residual_join(node)?;
    let original = PartitionedJoinInputStats {
        left_rows: left.rows,
        left_bytes: left.bytes,
        right_rows: right.rows,
        right_bytes: right.bytes,
    };
    let partition_count =
        partitioned_mixed_join_partition_count(context, left.rows, left.bytes, right.rows, right.bytes);
    let plan = PartitionedJoinDriverPlan {
        context,
        node_id,
        node,
        strategy: PARTITIONED_MIXED_KEYED_RESIDUAL_STRATEGY,
        strategy_options,
        partition_count,
        partition_target_bytes: partitioned_join_target_bytes_for_context(context),
    };

    if preserve_stream {
        record_partitioned_join_selection(plan, original, metrics);
        let partitioned = partition_join_sides(plan, left, right, reservations, metrics)?;
        let left_partitioned_bytes = partitioned.left.reservation.bytes();
        let right_partitioned_bytes = partitioned.right.reservation.bytes();
        record_partitioned_join_sides(plan, left_partitioned_bytes, right_partitioned_bytes, metrics);
        let PartitionedJoinSides { left: left_partitioned, right: right_partitioned } = partitioned;
        let stream = PartitionedMixedJoinStreamSource {
            context: context.clone(),
            node_id,
            node: node.clone(),
            left_partitioned: Some(left_partitioned),
            right_partitioned: Some(right_partitioned),
            strategy_options,
            partition_count,
            next_partition: 0,
            pending_outputs: VecDeque::new(),
            original,
            output_summary: PartitionedJoinOutputSummary::default(),
            metrics_recorded: false,
        };
        let payload = ResidentFrameStreamPayload::from_producer(
            plan.partition_target_bytes,
            0,
            0,
            node.spec.output_projection.len(),
            Vec::new(),
            Box::new(stream),
        );
        return Ok(BoundedCompiledJoinMaterializeOutput::LazyStream(payload));
    }

    let mut result = run_partitioned_join_driver(
        plan,
        left,
        right,
        reservations,
        metrics,
        materialize_partitioned_mixed_keyed_residual_join_partition,
    )?;
    let output = concatenate_streamed_hash_join_outputs(context, node_id, result.outputs, reservations, metrics)?;
    result.summary.observe_concatenated_output(&output);
    record_partitioned_join_operator_metrics(plan, result.original, &result.summary, metrics);
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
    let overrides_lease = execution_overrides(context)?;
    let overrides = overrides_lease.overrides();
    let predicate = lower_join_residual_predicate(context, node, &residual_columns)?;
    let options = dataframe_mixed_join_options(&left_keys, &right_keys, &residual_columns, node.spec.null_equality);
    let timer = Instant::now();
    let mut txn = JoinMaterializationTxn::with_sides(left, right);
    let (left, right) = txn.sides();
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
    .map_err(|error| join_index_error(error, node));
    let size_data = match size_data {
        Ok(size_data) => size_data,
        Err(error) => {
            txn.rollback(reservations, metrics);
            return Err(error);
        },
    };
    let output_rows = size_data.output_size() as u64;
    let mixed_size_data_bytes = match size_data.device_allocated_size_bytes(node) {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(size_data);
            txn.rollback(reservations, metrics);
            return Err(error);
        },
    };
    let size_data_reservation = match reservations.try_reserve(
        bounded_join_request(
            ReservationKind::MixedSizeData,
            "join_partition_mixed_size_data",
            mixed_size_data_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            drop(size_data);
            txn.rollback(reservations, metrics);
            return Err(error);
        },
    };
    txn.set_mixed_size_data_reservation(size_data_reservation);
    metrics.record_mixed_join_size_data_bytes(mixed_size_data_bytes);
    let gather_map_reservation_bytes = match size_data.gather_map_reservation_size_bytes(node) {
        Ok(bytes) => bytes,
        Err(error) => {
            drop(size_data);
            txn.rollback(reservations, metrics);
            return Err(error);
        },
    };
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
            drop(size_data);
            txn.rollback(reservations, metrics);
            return Err(error);
        },
    };
    txn.set_gather_map_reservation(gather_map_reservation);
    let indices = match size_data.materialize_indices_in(&overrides, node) {
        Ok(indices) => indices,
        Err(error) => {
            txn.rollback(reservations, metrics);
            return Err(error);
        },
    };
    txn.release_mixed_size_data_reservation(reservations, metrics);
    txn.set_indices(indices);
    let index_bytes = match txn.indices().device_allocated_size_bytes(node) {
        Ok(bytes) => bytes,
        Err(error) => {
            txn.rollback(reservations, metrics);
            return Err(error);
        },
    };
    let (left, right) = txn.sides();
    let forecast_output_reservation_bytes = match partitioned_mixed_join_output_upper_bound_bytes(
        context,
        node_id,
        node,
        left,
        right,
        output_rows,
        reservations,
        metrics,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            txn.rollback(reservations, metrics);
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
            drop(output);
            txn.rollback(reservations, metrics);
            return Err(error);
        },
    };
    if let Err(error) =
        reconcile_join_output_upper_bound(node_id, txn.output_reservation_mut(), output_bytes, reservations, metrics)
    {
        drop(output);
        txn.rollback(reservations, metrics);
        return Err(error);
    }
    txn.release_gather_map_reservation(reservations, metrics);
    Ok(PartitionedJoinPartitionOutput {
        frame: txn.commit_frame(GpuDataFrame::new(output), reservations, metrics),
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

struct PartitionedMixedJoinStreamSource {
    context: ExecutionContext,
    node_id: NodeId,
    node: JoinNode,
    left_partitioned: Option<PartitionedJoinSide>,
    right_partitioned: Option<PartitionedJoinSide>,
    strategy_options: KeyedHashJoinStrategyOptions,
    partition_count: usize,
    next_partition: usize,
    pending_outputs: VecDeque<ResidentGpuFrame>,
    original: PartitionedJoinInputStats,
    output_summary: PartitionedJoinOutputSummary,
    metrics_recorded: bool,
}

impl PartitionedMixedJoinStreamSource {
    fn record_success_metrics(&mut self, metrics: &mut QueryMetrics) {
        if self.metrics_recorded {
            return;
        }
        self.metrics_recorded = true;
        let plan = PartitionedJoinDriverPlan {
            context: &self.context,
            node_id: self.node_id,
            node: &self.node,
            strategy: PARTITIONED_MIXED_KEYED_RESIDUAL_STRATEGY,
            strategy_options: self.strategy_options,
            partition_count: self.partition_count,
            partition_target_bytes: partitioned_join_target_bytes_for_context(&self.context),
        };
        record_partitioned_join_operator_metrics(plan, self.original, &self.output_summary, metrics);
    }
}

impl ResidentFrameStreamSource for PartitionedMixedJoinStreamSource {
    fn next_frame(
        &mut self,
        reservations: &mut ReservationTracker,
        metrics: &mut QueryMetrics,
    ) -> Option<Result<ResidentGpuFrame>> {
        loop {
            if let Some(output) = self.pending_outputs.pop_front() {
                return Some(Ok(output));
            }
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
            let plan = PartitionedJoinDriverPlan {
                context: &self.context,
                node_id: self.node_id,
                node: &self.node,
                strategy: PARTITIONED_MIXED_KEYED_RESIDUAL_STRATEGY,
                strategy_options: self.strategy_options,
                partition_count: self.partition_count,
                partition_target_bytes: partitioned_join_target_bytes_for_context(&self.context),
            };
            match materialize_partitioned_join_partition_with_retry(
                plan,
                left_partitioned,
                right_partitioned,
                partition,
                reservations,
                metrics,
                &mut materialize_partitioned_mixed_keyed_residual_join_partition,
            ) {
                Ok(outputs) => {
                    for output in outputs {
                        self.output_summary.record_partition_output(&output);
                        self.pending_outputs.push_back(output.frame);
                    }
                },
                Err(error) => {
                    self.release(reservations, metrics);
                    return Some(Err(error));
                },
            }
        }
    }

    fn release(&mut self, reservations: &mut ReservationTracker, metrics: &mut QueryMetrics) {
        release_streamed_join_outputs_reverse(reservations, self.pending_outputs.drain(..).collect(), metrics);
        if let Some(left) = self.left_partitioned.take() {
            release_partitioned_join_side(left, reservations, metrics);
        }
        if let Some(right) = self.right_partitioned.take() {
            release_partitioned_join_side(right, reservations, metrics);
        }
    }
}
