use super::{
    BoundedCompiledJoinMaterializeOutput, BoundedExecutionAttempt, BoundedJoinSideFrame, CompiledMixedSizeData,
    GpuDataFrame, Instant, JoinMaterializationTxn, JoinNode, JoinSide, JoinType, KeyedHashJoinStrategyOptions, NodeId,
    PARTITIONED_MIXED_KEYED_RESIDUAL_STRATEGY, PartitionedJoinDriverPlan, PartitionedJoinInputStats,
    PartitionedJoinOutputSummary, PartitionedJoinPartitionOutput, PartitionedJoinSide, PartitionedJoinSides,
    ResidentFrameStreamPayload, ResidentFrameStreamSource, ResidentGpuFrame, Result, VecDeque,
    concatenate_streamed_hash_join_outputs, dataframe_mixed_join_options, join_index_error,
    join_side_output_upper_bound_bytes, lower_join_residual_predicate,
    materialize_partitioned_join_partition_with_retry, observe_device_memory_bytes, partition_join_sides,
    partitioned_join_target_bytes_for_context, partitioned_mixed_join_partition_count,
    projected_output_columns_for_retained_sizing, record_partitioned_join_operator_metrics,
    record_partitioned_join_selection, record_partitioned_join_sides, release_partitioned_join_side,
    residual_input_columns, run_partitioned_join_driver, sum_bounds, validate_partitioned_mixed_keyed_residual_join,
};

pub(in crate::exec::bounded::join::compiled) fn execute_compiled_partitioned_mixed_keyed_residual_join_materialize(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    strategy_options: KeyedHashJoinStrategyOptions,
) -> Result<ResidentGpuFrame> {
    match execute_compiled_partitioned_mixed_keyed_residual_join_materialize_output(
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
            unreachable!("non-preserving partitioned mixed join returns frame")
        },
    }
}

pub(in crate::exec::bounded::join::compiled) fn execute_compiled_partitioned_mixed_keyed_residual_join_materialize_preserving_stream(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    strategy_options: KeyedHashJoinStrategyOptions,
) -> Result<BoundedCompiledJoinMaterializeOutput> {
    execute_compiled_partitioned_mixed_keyed_residual_join_materialize_output(
        execution,
        node_id,
        node,
        left,
        right,
        strategy_options,
        true,
    )
}

fn execute_compiled_partitioned_mixed_keyed_residual_join_materialize_output(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    strategy_options: KeyedHashJoinStrategyOptions,
    preserve_stream: bool,
) -> Result<BoundedCompiledJoinMaterializeOutput> {
    validate_partitioned_mixed_keyed_residual_join(node)?;
    let original = PartitionedJoinInputStats {
        left_rows: left.rows,
        left_bytes: left.bytes,
        right_rows: right.rows,
        right_bytes: right.bytes,
    };
    let partition_count =
        partitioned_mixed_join_partition_count(execution.context(), left.rows, left.bytes, right.rows, right.bytes);
    let plan = PartitionedJoinDriverPlan {
        node_id,
        node,
        strategy: PARTITIONED_MIXED_KEYED_RESIDUAL_STRATEGY,
        strategy_options,
        partition_count,
        partition_target_bytes: partitioned_join_target_bytes_for_context(execution.context()),
    };

    if preserve_stream {
        record_partitioned_join_selection(plan, original, execution.metrics());
        let partitioned = partition_join_sides(execution, plan, left, right)?;
        let left_partitioned_bytes = partitioned.left.device_bytes();
        let right_partitioned_bytes = partitioned.right.device_bytes();
        record_partitioned_join_sides(plan, left_partitioned_bytes, right_partitioned_bytes, execution.metrics());
        let PartitionedJoinSides { left: left_partitioned, right: right_partitioned } = partitioned;
        let stream = PartitionedMixedJoinStreamSource {
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
            Some(plan.partition_target_bytes),
            None,
            0,
            node.spec.output_projection.len(),
            Box::new(stream),
        );
        return Ok(BoundedCompiledJoinMaterializeOutput::LazyStream(payload));
    }

    let mut result = run_partitioned_join_driver(
        execution,
        plan,
        left,
        right,
        materialize_partitioned_mixed_keyed_residual_join_partition,
    )?;
    let output = concatenate_streamed_hash_join_outputs(execution, node_id, result.outputs)?;
    result.summary.observe_concatenated_output(&output);
    record_partitioned_join_operator_metrics(plan, result.original, &result.summary, execution.metrics());
    Ok(BoundedCompiledJoinMaterializeOutput::Frame(Box::new(output)))
}

fn materialize_partitioned_mixed_keyed_residual_join_partition(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
) -> Result<PartitionedJoinPartitionOutput> {
    let residual_columns = residual_input_columns(node)?;
    let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
    let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
    let exec_runtime = execution.context().runtime()?;
    let overrides = exec_runtime.exec();
    let predicate = lower_join_residual_predicate(execution.context(), node, &residual_columns)?;
    let options = dataframe_mixed_join_options(&left_keys, &right_keys, &residual_columns, node.spec.null_equality);
    let timer = Instant::now();
    let mut txn = JoinMaterializationTxn::with_sides(left, right);
    let (left, right) = txn.sides();
    let size_data = match node.spec.join_type {
        JoinType::Inner => left
            .frame
            .on(overrides)
            .mixed_inner_join_size_data(&right.frame, &predicate, options)
            .map(CompiledMixedSizeData::Pair),
        JoinType::Left => left
            .frame
            .on(overrides)
            .mixed_left_join_size_data(&right.frame, &predicate, options)
            .map(CompiledMixedSizeData::Pair),
        JoinType::Full => left
            .frame
            .on(overrides)
            .mixed_full_join_size_data(&right.frame, &predicate, options)
            .map(CompiledMixedSizeData::Pair),
        JoinType::LeftSemi => left
            .frame
            .on(overrides)
            .mixed_left_semi_join_size_data(&right.frame, &predicate, options)
            .map(CompiledMixedSizeData::LeftOnly),
        JoinType::LeftAnti { .. } => left
            .frame
            .on(overrides)
            .mixed_left_anti_join_size_data(&right.frame, &predicate, options)
            .map(CompiledMixedSizeData::LeftOnly),
        JoinType::Cross | JoinType::LeftMark { .. } => unreachable!("validated above"),
    }
    .map_err(|error| join_index_error(error, node));
    let size_data = match size_data {
        Ok(size_data) => size_data,
        Err(error) => {
            txn.rollback();
            return Err(error);
        },
    };
    let output_rows = size_data.output_size() as u64;
    let mixed_size_data_bytes = match size_data.device_allocated_size_bytes(node) {
        Ok(bytes) => bytes,
        Err(error) => {
            txn.rollback();
            return Err(error);
        },
    };
    execution.metrics().record_mixed_join_size_data_bytes(mixed_size_data_bytes);
    let indices = match size_data.materialize_indices_in(overrides, node) {
        Ok(indices) => indices,
        Err(error) => {
            txn.rollback();
            return Err(error);
        },
    };
    txn.set_indices(indices);
    let index_bytes = match txn.indices().device_allocated_size_bytes(node) {
        Ok(bytes) => bytes,
        Err(error) => {
            txn.rollback();
            return Err(error);
        },
    };
    let (left, right) = txn.sides();
    let forecast_output_reservation_bytes =
        match partitioned_mixed_join_output_upper_bound_bytes(execution, node_id, node, left, right, output_rows) {
            Ok(bytes) => bytes,
            Err(error) => {
                txn.rollback();
                return Err(error);
            },
        };
    let output_reservation_bytes = forecast_output_reservation_bytes;
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
        mixed_size_data_bytes,
        output_reservation_bytes,
        output_bytes,
        output_rows,
    })
}

fn partitioned_mixed_join_output_upper_bound_bytes(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    left: &BoundedJoinSideFrame,
    right: &BoundedJoinSideFrame,
    expected_output_rows: u64,
) -> Result<Option<u64>> {
    let left_projection = projected_output_columns_for_retained_sizing(&node.spec.output_projection, JoinSide::Left);
    let right_projection = projected_output_columns_for_retained_sizing(&node.spec.output_projection, JoinSide::Right);
    let left_bytes = join_side_output_upper_bound_bytes(
        execution,
        node_id,
        &left.frame,
        expected_output_rows,
        &left_projection,
        JoinSide::Left,
    )?;
    let right_bytes = join_side_output_upper_bound_bytes(
        execution,
        node_id,
        &right.frame,
        expected_output_rows,
        &right_projection,
        JoinSide::Right,
    )?;
    Ok(sum_bounds(left_bytes, right_bytes))
}

struct PartitionedMixedJoinStreamSource {
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
    fn record_success_metrics(&mut self, execution: &mut BoundedExecutionAttempt<'_>) {
        if self.metrics_recorded {
            return;
        }
        self.metrics_recorded = true;
        let plan = PartitionedJoinDriverPlan {
            node_id: self.node_id,
            node: &self.node,
            strategy: PARTITIONED_MIXED_KEYED_RESIDUAL_STRATEGY,
            strategy_options: self.strategy_options,
            partition_count: self.partition_count,
            partition_target_bytes: partitioned_join_target_bytes_for_context(execution.context()),
        };
        record_partitioned_join_operator_metrics(plan, self.original, &self.output_summary, execution.metrics());
    }
}

impl ResidentFrameStreamSource for PartitionedMixedJoinStreamSource {
    fn next_frame(&mut self, execution: &mut BoundedExecutionAttempt<'_>) -> Option<Result<ResidentGpuFrame>> {
        loop {
            if let Some(output) = self.pending_outputs.pop_front() {
                return Some(Ok(output));
            }
            if self.next_partition >= self.partition_count {
                self.record_success_metrics(execution);
                self.release(execution);
                return None;
            }
            let partition = self.next_partition;
            self.next_partition += 1;
            let left_partitioned =
                self.left_partitioned.as_ref().expect("left partitioned side released before stream end");
            let right_partitioned =
                self.right_partitioned.as_ref().expect("right partitioned side released before stream end");
            let plan = PartitionedJoinDriverPlan {
                node_id: self.node_id,
                node: &self.node,
                strategy: PARTITIONED_MIXED_KEYED_RESIDUAL_STRATEGY,
                strategy_options: self.strategy_options,
                partition_count: self.partition_count,
                partition_target_bytes: partitioned_join_target_bytes_for_context(execution.context()),
            };
            match materialize_partitioned_join_partition_with_retry(
                execution,
                plan,
                left_partitioned,
                right_partitioned,
                partition,
                &mut materialize_partitioned_mixed_keyed_residual_join_partition,
            ) {
                Ok(outputs) => {
                    for output in outputs {
                        self.output_summary.record_partition_output(&output);
                        self.pending_outputs.push_back(output.frame);
                    }
                },
                Err(error) => {
                    self.release(execution);
                    return Some(Err(error));
                },
            }
        }
    }

    fn release(&mut self, _execution: &mut BoundedExecutionAttempt<'_>) {
        self.pending_outputs.clear();
        if let Some(left) = self.left_partitioned.take() {
            release_partitioned_join_side(left);
        }
        if let Some(right) = self.right_partitioned.take() {
            release_partitioned_join_side(right);
        }
    }
}
