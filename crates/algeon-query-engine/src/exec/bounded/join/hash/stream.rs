use std::collections::VecDeque;

use query_runtime::metrics::{JoinOperatorMetrics, OperatorMetrics};

use super::{
    BoundedExecutionAttempt, BoundedJoinSideFrame, BoundedPartitionTargetPolicy, BoundedRootHashJoinBuildState,
    BoundedRootHashJoinProbeState, ExecutionContext, ExecutionMetrics, Instant, JoinNode, JoinSide, NodeId, QueryError,
    ResidentFrameStreamSource, ResidentGpuFrame, Result, join_side_output_upper_bound_bytes,
    materialize::{materialize_bounded_root_hash_join_index_slice, release_root_hash_join_stream_outputs},
    probe::reusable_hash_join_execution_strategy,
    projected_output_columns_for_retained_sizing, release_bounded_join_side_frame,
    release_bounded_root_hash_join_build_state, side_metric, sum_bounds,
};
use crate::exec::retry::{is_retryable_resource_error, record_retryable_resource_error};

#[expect(
    clippy::too_many_arguments,
    reason = "bounded execution passes operator-occurrence inputs explicitly beside the attempt state"
)]
pub(in crate::exec::bounded::join::hash) fn materialize_bounded_root_hash_join_probe_stream(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    build: &BoundedRootHashJoinBuildState,
    probe: BoundedJoinSideFrame,
    indices: cudf::DataFrameHashJoinPairIndices,
    gather_map_bytes: u64,
    timer: Instant,
) -> Result<Vec<ResidentGpuFrame>> {
    let exec_runtime = execution.context().runtime()?;
    let overrides = exec_runtime.exec();
    let build_side = build.build_side;
    let probe_side = build.probe_side;
    let hash_table_bytes = build.hash_table_bytes;
    let (left, right) = match build_side {
        JoinSide::Left => (&build.build, &probe),
        JoinSide::Right => (&probe, &build.build),
    };
    let original_left_rows = left.rows;
    let original_right_rows = right.rows;
    let left_bytes = left.bytes;
    let right_bytes = right.bytes;
    let build_bytes = side_metric(build_side, left_bytes, right_bytes);
    let probe_bytes = side_metric(probe_side, left_bytes, right_bytes);
    let (left_indices, right_indices) = if build.left_indices_from_probe {
        (indices.probe().clone(), indices.build().clone())
    } else {
        (indices.build().clone(), indices.probe().clone())
    };
    let output_rows = indices.build().len() as u64;
    let left_projection = projected_output_columns_for_retained_sizing(&node.spec.output_projection, JoinSide::Left);
    let right_projection = projected_output_columns_for_retained_sizing(&node.spec.output_projection, JoinSide::Right);
    let left_output_bytes = match join_side_output_upper_bound_bytes(
        execution,
        node_id,
        &left.frame,
        output_rows,
        &left_projection,
        JoinSide::Left,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Err(error);
        },
    };
    let right_output_bytes = match join_side_output_upper_bound_bytes(
        execution,
        node_id,
        &right.frame,
        output_rows,
        &right_projection,
        JoinSide::Right,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Err(error);
        },
    };
    let output_reservation_bytes = sum_bounds(left_output_bytes, right_output_bytes);
    let mut outputs = Vec::new();
    let mut cumulative_output_reservation_bytes = Some(0);
    let mut cumulative_output_bytes = 0u64;
    let mut emitted_rows = 0u64;
    let mut offset = 0usize;
    let output_len = indices.build().len();
    if output_len == 0 {
        match materialize_bounded_root_hash_join_index_slice(
            execution,
            node_id,
            node,
            left,
            right,
            &left_indices,
            &right_indices,
            0,
            0,
            overrides,
        ) {
            Ok(slice_outputs) => {
                for output in slice_outputs {
                    cumulative_output_reservation_bytes =
                        sum_bounds(cumulative_output_reservation_bytes, output.output_reservation_bytes);
                    cumulative_output_bytes = cumulative_output_bytes.saturating_add(output.output_bytes);
                    outputs.push(output.frame);
                }
            },
            Err(error) => {
                return Err(error);
            },
        }
    }
    while offset < output_len {
        let remaining_rows = (output_len - offset) as u64;
        let remaining_output_reservation_bytes =
            proportional_join_stream_bytes(output_reservation_bytes, remaining_rows, output_rows);
        let chunk_target_bytes = reusable_hash_join_stream_target_output_bytes(execution);
        let chunk_rows = reusable_hash_join_stream_chunk_rows(
            remaining_rows,
            remaining_output_reservation_bytes,
            chunk_target_bytes,
        );
        let take = chunk_rows.min(output_len - offset);
        match materialize_bounded_root_hash_join_index_slice(
            execution,
            node_id,
            node,
            left,
            right,
            &left_indices,
            &right_indices,
            offset,
            take,
            overrides,
        ) {
            Ok(slice_outputs) => {
                for output in slice_outputs {
                    cumulative_output_reservation_bytes =
                        sum_bounds(cumulative_output_reservation_bytes, output.output_reservation_bytes);
                    cumulative_output_bytes = cumulative_output_bytes.saturating_add(output.output_bytes);
                    emitted_rows = emitted_rows.saturating_add(output.output_rows);
                    outputs.push(output.frame);
                }
                offset += take;
            },
            Err(error) => {
                release_root_hash_join_stream_outputs(outputs);
                return Err(error);
            },
        }
    }
    execution.metrics().add_operator_time(timer.elapsed());
    execution.metrics().record_operator_metrics(OperatorMetrics::join(
        node_id,
        JoinOperatorMetrics {
            query_id: None,
            join_type: node.spec.join_type.to_string(),
            execution_strategy: reusable_hash_join_execution_strategy(node, build).to_string(),
            key_count: node.spec.keys.len(),
            residual_predicate: false,
            null_equality: node.spec.null_equality.to_string(),
            build_side: build_side.to_string(),
            probe_side: probe_side.to_string(),
            build_rows: side_metric(build_side, original_left_rows, original_right_rows),
            build_bytes,
            probe_rows: side_metric(probe_side, original_left_rows, original_right_rows),
            probe_bytes,
            hash_table_bytes,
            filtered_probe_state_bytes: 0,
            mixed_size_data_bytes: 0,
            gather_map_bytes,
            output_reservation_bytes: cumulative_output_reservation_bytes,
            output_rows: emitted_rows,
            output_bytes: cumulative_output_bytes,
        },
    ));
    Ok(outputs)
}

pub(in crate::exec::bounded::join::hash) struct RootHashJoinMaterializeStreamSource {
    pub(in crate::exec::bounded::join::hash) context: ExecutionContext,
    pub(in crate::exec::bounded::join::hash) node_id: NodeId,
    pub(in crate::exec::bounded::join::hash) node: JoinNode,
    pub(in crate::exec::bounded::join::hash) build: Option<BoundedRootHashJoinBuildState>,
    pub(in crate::exec::bounded::join::hash) probe: Option<BoundedJoinSideFrame>,
    pub(in crate::exec::bounded::join::hash) indices: Option<cudf::DataFrameHashJoinPairIndices>,
    pub(in crate::exec::bounded::join::hash) left_indices: Option<cudf::DataFrameJoinIndices>,
    pub(in crate::exec::bounded::join::hash) right_indices: Option<cudf::DataFrameJoinIndices>,
    pub(in crate::exec::bounded::join::hash) gather_map_bytes: u64,
    pub(in crate::exec::bounded::join::hash) output_reservation_bytes: Option<u64>,
    pub(in crate::exec::bounded::join::hash) timer: Option<Instant>,
    pub(in crate::exec::bounded::join::hash) pending_outputs: VecDeque<ResidentGpuFrame>,
    pub(in crate::exec::bounded::join::hash) offset: usize,
    pub(in crate::exec::bounded::join::hash) output_len: usize,
    pub(in crate::exec::bounded::join::hash) emitted_empty_output: bool,
    pub(in crate::exec::bounded::join::hash) cumulative_output_reservation_bytes: Option<u64>,
    pub(in crate::exec::bounded::join::hash) cumulative_output_bytes: u64,
    pub(in crate::exec::bounded::join::hash) emitted_rows: u64,
}

impl RootHashJoinMaterializeStreamSource {
    pub(in crate::exec::bounded::join::hash) fn new(
        context: &ExecutionContext,
        node_id: NodeId,
        node: &JoinNode,
        state: BoundedRootHashJoinProbeState,
    ) -> Self {
        let output_len = state.indices.build().len();
        let (left_indices, right_indices) = if state.build.left_indices_from_probe {
            (state.indices.probe().clone(), state.indices.build().clone())
        } else {
            (state.indices.build().clone(), state.indices.probe().clone())
        };
        Self {
            context: context.clone(),
            node_id,
            node: node.clone(),
            timer: Some(state.build.timer),
            build: Some(state.build),
            probe: Some(state.probe),
            indices: Some(state.indices),
            left_indices: Some(left_indices),
            right_indices: Some(right_indices),
            gather_map_bytes: state.gather_map_bytes,
            output_reservation_bytes: state.output_reservation_bytes,
            pending_outputs: VecDeque::new(),
            offset: 0,
            output_len,
            emitted_empty_output: false,
            cumulative_output_reservation_bytes: Some(0),
            cumulative_output_bytes: 0,
            emitted_rows: 0,
        }
    }

    pub(in crate::exec::bounded::join::hash) fn materialize_slice(
        &mut self,
        execution: &mut BoundedExecutionAttempt<'_>,
        offset: usize,
        take: usize,
    ) -> Result<()> {
        let build = self.build.as_ref().expect("root hash join build state released before stream end");
        let probe = self.probe.as_ref().expect("root hash join probe frame released before stream end");
        let left_indices = self.left_indices.as_ref().expect("root hash join left indices released before stream end");
        let right_indices =
            self.right_indices.as_ref().expect("root hash join right indices released before stream end");
        let (left, right) = match build.build_side {
            JoinSide::Left => (&build.build, probe),
            JoinSide::Right => (probe, &build.build),
        };
        let exec_runtime = &self.context.runtime()?;
        let overrides = exec_runtime.exec();
        let slice_outputs = materialize_bounded_root_hash_join_index_slice(
            execution,
            self.node_id,
            &self.node,
            left,
            right,
            left_indices,
            right_indices,
            offset,
            take,
            overrides,
        )?;
        for output in slice_outputs {
            self.cumulative_output_reservation_bytes =
                sum_bounds(self.cumulative_output_reservation_bytes, output.output_reservation_bytes);
            self.cumulative_output_bytes = self.cumulative_output_bytes.saturating_add(output.output_bytes);
            self.emitted_rows = self.emitted_rows.saturating_add(output.output_rows);
            self.pending_outputs.push_back(output.frame);
        }
        Ok(())
    }

    pub(in crate::exec::bounded::join::hash) fn finish_success(&mut self, execution: &mut BoundedExecutionAttempt<'_>) {
        if let (Some(timer), Some(build), Some(probe)) = (self.timer.take(), self.build.as_ref(), self.probe.as_ref()) {
            let build_side = build.build_side;
            let probe_side = build.probe_side;
            let (left, right) = match build_side {
                JoinSide::Left => (&build.build, probe),
                JoinSide::Right => (probe, &build.build),
            };
            let original_left_rows = left.rows;
            let original_right_rows = right.rows;
            let left_bytes = left.bytes;
            let right_bytes = right.bytes;
            execution.metrics().add_operator_time(timer.elapsed());
            execution.metrics().record_operator_metrics(OperatorMetrics::join(
                self.node_id,
                JoinOperatorMetrics {
                    query_id: None,
                    join_type: self.node.spec.join_type.to_string(),
                    execution_strategy: reusable_hash_join_execution_strategy(&self.node, build).to_string(),
                    key_count: self.node.spec.keys.len(),
                    residual_predicate: false,
                    null_equality: self.node.spec.null_equality.to_string(),
                    build_side: build_side.to_string(),
                    probe_side: probe_side.to_string(),
                    build_rows: side_metric(build_side, original_left_rows, original_right_rows),
                    build_bytes: side_metric(build_side, left_bytes, right_bytes),
                    probe_rows: side_metric(probe_side, original_left_rows, original_right_rows),
                    probe_bytes: side_metric(probe_side, left_bytes, right_bytes),
                    hash_table_bytes: build.hash_table_bytes,
                    filtered_probe_state_bytes: 0,
                    mixed_size_data_bytes: 0,
                    gather_map_bytes: self.gather_map_bytes,
                    output_reservation_bytes: self.cumulative_output_reservation_bytes,
                    output_rows: self.emitted_rows,
                    output_bytes: self.cumulative_output_bytes,
                },
            ));
        }
        self.release_owned();
    }

    pub(in crate::exec::bounded::join::hash) fn release_owned(&mut self) {
        self.pending_outputs.clear();
        self.left_indices = None;
        self.right_indices = None;
        self.indices = None;
        if let Some(probe) = self.probe.take() {
            release_bounded_join_side_frame(probe);
        }
        if let Some(build) = self.build.take() {
            release_bounded_root_hash_join_build_state(build);
        }
    }
}

impl ResidentFrameStreamSource for RootHashJoinMaterializeStreamSource {
    fn next_frame(&mut self, execution: &mut BoundedExecutionAttempt<'_>) -> Option<Result<ResidentGpuFrame>> {
        loop {
            if let Some(output) = self.pending_outputs.pop_front() {
                return Some(Ok(output));
            }
            if self.output_len == 0 {
                if self.emitted_empty_output {
                    self.finish_success(execution);
                    return None;
                }
                self.emitted_empty_output = true;
                return match self.materialize_slice(execution, 0, 0) {
                    Ok(()) => continue,
                    Err(error) => Some(Err(error)),
                };
            }
            if self.offset >= self.output_len {
                self.finish_success(execution);
                return None;
            }
            let remaining_rows = (self.output_len - self.offset) as u64;
            let remaining_output_reservation_bytes =
                proportional_join_stream_bytes(self.output_reservation_bytes, remaining_rows, self.output_len as u64);
            let chunk_target_bytes = reusable_hash_join_stream_target_output_bytes(execution);
            let chunk_rows = reusable_hash_join_stream_chunk_rows(
                remaining_rows,
                remaining_output_reservation_bytes,
                chunk_target_bytes,
            );
            let take = chunk_rows.min(self.output_len - self.offset);
            let offset = self.offset;
            match self.materialize_slice(execution, offset, take) {
                Ok(()) => self.offset += take,
                Err(error) => return Some(Err(error)),
            }
        }
    }

    fn release(&mut self, _execution: &mut BoundedExecutionAttempt<'_>) {
        self.release_owned();
    }
}

pub(in crate::exec::bounded::join::hash) fn is_retryable_join_stream_slice_error(error: &QueryError) -> bool {
    is_retryable_resource_error(error)
}

pub(in crate::exec::bounded::join::hash) fn record_join_stream_slice_retry_error(
    error: &QueryError,
    metrics: &ExecutionMetrics,
) {
    record_retryable_resource_error(error, metrics);
}

/// Output rows per slice. A join whose output has no byte bound, or fits the
/// target, materializes in one slice; the slice retry splits it if its
/// allocation fails.
pub(in crate::exec::bounded::join::hash) fn reusable_hash_join_stream_chunk_rows(
    output_rows: u64,
    output_reservation_bytes: Option<u64>,
    target_bytes: u64,
) -> usize {
    match output_reservation_bytes {
        Some(bytes) if output_rows > 0 && bytes > target_bytes => {
            let rows = (target_bytes.saturating_mul(output_rows) / bytes).max(1);
            usize::try_from(rows).unwrap_or(usize::MAX)
        },
        _ => usize::try_from(output_rows.max(1)).unwrap_or(usize::MAX),
    }
}

/// The share of `total_bytes` that `rows` of `total_rows` carry.
pub(in crate::exec::bounded::join::hash) fn proportional_join_stream_bytes(
    total_bytes: Option<u64>,
    rows: u64,
    total_rows: u64,
) -> Option<u64> {
    let total_bytes = total_bytes?;
    if total_rows == 0 {
        return Some(total_bytes);
    }
    Some(((u128::from(total_bytes) * u128::from(rows)).div_ceil(u128::from(total_rows))).min(u128::from(u64::MAX))
        as u64)
}

pub(in crate::exec::bounded) fn reusable_hash_join_stream_target_output_bytes(
    execution: &BoundedExecutionAttempt<'_>,
) -> u64 {
    BoundedPartitionTargetPolicy::default().reusable_hash_join_stream_target_output_bytes(
        execution.context().memory_control_policy(),
        execution.device_budget_bytes(),
    )
}
