use super::*;

pub(in crate::exec::bounded::join::hash) fn materialize_bounded_root_hash_join_probe_stream(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    build: &BoundedRootHashJoinBuildState,
    probe: BoundedJoinSideFrame,
    indices: cudf::DataFrameHashJoinPairIndices,
    gather_map_reservation: ActiveReservation,
    gather_map_bytes: u64,
    timer: Instant,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Vec<ResidentGpuFrame>> {
    let overrides_lease = execution_overrides(context)?;
    let overrides = overrides_lease.overrides();
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
        context,
        node_id,
        &left.frame,
        output_rows,
        &left_projection,
        JoinSide::Left,
        reservations,
        metrics,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            reservations.release(gather_map_reservation, metrics);
            return Err(error);
        },
    };
    let right_output_bytes = match join_side_output_upper_bound_bytes(
        context,
        node_id,
        &right.frame,
        output_rows,
        &right_projection,
        JoinSide::Right,
        reservations,
        metrics,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            reservations.release(gather_map_reservation, metrics);
            return Err(error);
        },
    };
    let output_reservation_bytes = left_output_bytes.saturating_add(right_output_bytes);
    let mut outputs = Vec::new();
    let mut cumulative_output_reservation_bytes = 0u64;
    let mut cumulative_output_bytes = 0u64;
    let mut emitted_rows = 0u64;
    let mut offset = 0usize;
    let output_len = indices.build().len();
    if output_len == 0 {
        match materialize_bounded_root_hash_join_index_slice(
            context,
            node_id,
            node,
            left,
            right,
            &left_indices,
            &right_indices,
            0,
            0,
            &overrides,
            reservations,
            metrics,
        ) {
            Ok(slice_outputs) => {
                for output in slice_outputs {
                    cumulative_output_reservation_bytes =
                        cumulative_output_reservation_bytes.saturating_add(output.output_reservation_bytes);
                    cumulative_output_bytes = cumulative_output_bytes.saturating_add(output.output_bytes);
                    outputs.push(output.frame);
                }
            },
            Err(error) => {
                reservations.release(gather_map_reservation, metrics);
                return Err(error);
            },
        }
    }
    while offset < output_len {
        let remaining_rows = u64::try_from(output_len - offset).unwrap_or(u64::MAX);
        let remaining_output_reservation_bytes =
            proportional_join_stream_bytes(output_reservation_bytes, remaining_rows, output_rows);
        let chunk_target_bytes = reusable_hash_join_stream_target_output_bytes(context, reservations);
        let chunk_rows = reusable_hash_join_stream_chunk_rows(
            remaining_rows,
            remaining_output_reservation_bytes,
            chunk_target_bytes,
        );
        let take = chunk_rows.min(output_len - offset);
        match materialize_bounded_root_hash_join_index_slice(
            context,
            node_id,
            node,
            left,
            right,
            &left_indices,
            &right_indices,
            offset,
            take,
            &overrides,
            reservations,
            metrics,
        ) {
            Ok(slice_outputs) => {
                for output in slice_outputs {
                    cumulative_output_reservation_bytes =
                        cumulative_output_reservation_bytes.saturating_add(output.output_reservation_bytes);
                    cumulative_output_bytes = cumulative_output_bytes.saturating_add(output.output_bytes);
                    emitted_rows = emitted_rows.saturating_add(output.output_rows);
                    outputs.push(output.frame);
                }
                offset += take;
            },
            Err(error) => {
                release_root_hash_join_stream_outputs(outputs, reservations, metrics);
                reservations.release(gather_map_reservation, metrics);
                return Err(error);
            },
        }
    }
    metrics.operator_time += timer.elapsed();
    metrics.record_operator_metrics(crate::metrics::OperatorMetrics::join(
        node_id,
        crate::metrics::JoinOperatorMetrics {
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
    reservations.release(gather_map_reservation, metrics);
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
    pub(in crate::exec::bounded::join::hash) gather_map_reservation: Option<ActiveReservation>,
    pub(in crate::exec::bounded::join::hash) gather_map_bytes: u64,
    pub(in crate::exec::bounded::join::hash) output_reservation_bytes: u64,
    pub(in crate::exec::bounded::join::hash) timer: Option<Instant>,
    pub(in crate::exec::bounded::join::hash) pending_outputs: VecDeque<ResidentGpuFrame>,
    pub(in crate::exec::bounded::join::hash) offset: usize,
    pub(in crate::exec::bounded::join::hash) output_len: usize,
    pub(in crate::exec::bounded::join::hash) emitted_empty_output: bool,
    pub(in crate::exec::bounded::join::hash) cumulative_output_reservation_bytes: u64,
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
            gather_map_reservation: Some(state.gather_map_reservation),
            gather_map_bytes: state.gather_map_bytes,
            output_reservation_bytes: state.output_reservation_bytes,
            pending_outputs: VecDeque::new(),
            offset: 0,
            output_len,
            emitted_empty_output: false,
            cumulative_output_reservation_bytes: 0,
            cumulative_output_bytes: 0,
            emitted_rows: 0,
        }
    }

    pub(in crate::exec::bounded::join::hash) fn materialize_slice(
        &mut self,
        offset: usize,
        take: usize,
        reservations: &mut ReservationTracker,
        metrics: &mut QueryMetrics,
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
        let overrides_lease = execution_overrides(&self.context)?;
        let overrides = overrides_lease.overrides();
        let slice_outputs = materialize_bounded_root_hash_join_index_slice(
            &self.context,
            self.node_id,
            &self.node,
            left,
            right,
            left_indices,
            right_indices,
            offset,
            take,
            &overrides,
            reservations,
            metrics,
        )?;
        for output in slice_outputs {
            self.cumulative_output_reservation_bytes =
                self.cumulative_output_reservation_bytes.saturating_add(output.output_reservation_bytes);
            self.cumulative_output_bytes = self.cumulative_output_bytes.saturating_add(output.output_bytes);
            self.emitted_rows = self.emitted_rows.saturating_add(output.output_rows);
            self.pending_outputs.push_back(output.frame);
        }
        Ok(())
    }

    pub(in crate::exec::bounded::join::hash) fn finish_success(
        &mut self,
        reservations: &mut ReservationTracker,
        metrics: &mut QueryMetrics,
    ) {
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
            metrics.operator_time += timer.elapsed();
            metrics.record_operator_metrics(crate::metrics::OperatorMetrics::join(
                self.node_id,
                crate::metrics::JoinOperatorMetrics {
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
        self.release_owned(reservations, metrics);
    }

    pub(in crate::exec::bounded::join::hash) fn release_owned(
        &mut self,
        reservations: &mut ReservationTracker,
        metrics: &mut QueryMetrics,
    ) {
        while let Some(output) = self.pending_outputs.pop_back() {
            let ResidentGpuFrame { reservation, .. } = output;
            reservations.release(reservation, metrics);
        }
        self.left_indices = None;
        self.right_indices = None;
        self.indices = None;
        if let Some(reservation) = self.gather_map_reservation.take() {
            reservations.release(reservation, metrics);
        }
        if let Some(probe) = self.probe.take() {
            release_bounded_join_side_frame(probe, reservations, metrics);
        }
        if let Some(build) = self.build.take() {
            release_bounded_root_hash_join_build_state(build, reservations, metrics);
        }
    }
}

impl ResidentFrameStreamSource for RootHashJoinMaterializeStreamSource {
    fn next_frame(
        &mut self,
        reservations: &mut ReservationTracker,
        metrics: &mut QueryMetrics,
    ) -> Option<Result<ResidentGpuFrame>> {
        loop {
            if let Some(output) = self.pending_outputs.pop_front() {
                return Some(Ok(output));
            }
            if self.output_len == 0 {
                if self.emitted_empty_output {
                    self.finish_success(reservations, metrics);
                    return None;
                }
                self.emitted_empty_output = true;
                return match self.materialize_slice(0, 0, reservations, metrics) {
                    Ok(()) => continue,
                    Err(error) => Some(Err(error)),
                };
            }
            if self.offset >= self.output_len {
                self.finish_success(reservations, metrics);
                return None;
            }
            let remaining_rows = u64::try_from(self.output_len - self.offset).unwrap_or(u64::MAX);
            let remaining_output_reservation_bytes =
                proportional_join_stream_bytes(self.output_reservation_bytes, remaining_rows, self.output_len as u64);
            let chunk_target_bytes = reusable_hash_join_stream_target_output_bytes(&self.context, reservations);
            let chunk_rows = reusable_hash_join_stream_chunk_rows(
                remaining_rows,
                remaining_output_reservation_bytes,
                chunk_target_bytes,
            );
            let take = chunk_rows.min(self.output_len - self.offset);
            let offset = self.offset;
            match self.materialize_slice(offset, take, reservations, metrics) {
                Ok(()) => self.offset += take,
                Err(error) => return Some(Err(error)),
            }
        }
    }

    fn release(&mut self, reservations: &mut ReservationTracker, metrics: &mut QueryMetrics) {
        self.release_owned(reservations, metrics);
    }
}

pub(in crate::exec::bounded::join::hash) fn is_retryable_join_stream_slice_error(error: &QueryError) -> bool {
    crate::exec::retry::is_retryable_resource_error(error)
}

pub(in crate::exec::bounded::join::hash) fn record_join_stream_slice_retry_error(
    error: &QueryError,
    metrics: &mut QueryMetrics,
) {
    crate::exec::retry::record_retryable_resource_error(error, metrics);
}

pub(in crate::exec::bounded::join::hash) fn reusable_hash_join_stream_chunk_rows(
    output_rows: u64,
    output_reservation_bytes: u64,
    target_bytes: u64,
) -> usize {
    if output_rows == 0 || output_reservation_bytes <= target_bytes {
        return usize::try_from(output_rows.max(1)).unwrap_or(usize::MAX);
    }
    let rows = target_bytes.saturating_mul(output_rows).checked_div(output_reservation_bytes).unwrap_or(1).max(1);
    usize::try_from(rows).unwrap_or(usize::MAX)
}

pub(in crate::exec::bounded::join::hash) fn proportional_join_stream_bytes(
    total_bytes: u64,
    rows: u64,
    total_rows: u64,
) -> u64 {
    if total_rows == 0 {
        return total_bytes;
    }
    ((u128::from(total_bytes) * u128::from(rows)).div_ceil(u128::from(total_rows))).min(u128::from(u64::MAX)) as u64
}

pub(in crate::exec::bounded) fn reusable_hash_join_stream_target_output_bytes(
    context: &ExecutionContext,
    reservations: &ReservationTracker,
) -> u64 {
    BoundedPartitionTargetPolicy::default()
        .reusable_hash_join_stream_target_output_bytes(context.memory_control_policy(), reservations.remaining_bytes())
}
