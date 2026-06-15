use super::*;

pub(super) const PARTITIONED_HASH_EQUI_STRATEGY: PartitionedJoinStrategy = PartitionedJoinStrategy {
    strategy_label: "partitioned_hash_equi",
    execution_strategy: "bounded_partitioned_hash_equi_join",
    residual_predicate: false,
};

pub(super) const PARTITIONED_MIXED_KEYED_RESIDUAL_STRATEGY: PartitionedJoinStrategy = PartitionedJoinStrategy {
    strategy_label: "partitioned_mixed_keyed_residual",
    execution_strategy: "bounded_partitioned_mixed_keyed_residual_join",
    residual_predicate: true,
};

// Static labels that bind a partitioned join path to metrics and reports.
#[derive(Clone, Copy)]
pub(super) struct PartitionedJoinStrategy {
    strategy_label: &'static str,
    execution_strategy: &'static str,
    residual_predicate: bool,
}

// Immutable inputs and strategy knobs for one partitioned join run.
#[derive(Clone, Copy)]
pub(super) struct PartitionedJoinDriverPlan<'a> {
    pub(super) context: &'a ExecutionContext,
    pub(super) node_id: NodeId,
    pub(super) node: &'a JoinNode,
    pub(super) strategy: PartitionedJoinStrategy,
    pub(super) strategy_options: KeyedHashJoinStrategyOptions,
    pub(super) partition_count: usize,
    pub(super) partition_target_bytes: u64,
}

// Buffered partition outputs before final concatenation and metric reporting.
pub(super) struct PartitionedJoinBufferedOutput {
    pub(super) outputs: Vec<ResidentGpuFrame>,
    pub(super) original: PartitionedJoinInputStats,
    pub(super) summary: PartitionedJoinOutputSummary,
}

// Original side cardinalities retained after partitioning for join metrics.
#[derive(Clone, Copy)]
pub(super) struct PartitionedJoinInputStats {
    pub(super) left_rows: u64,
    pub(super) left_bytes: u64,
    pub(super) right_rows: u64,
    pub(super) right_bytes: u64,
}

// Accumulates per-partition output costs for the final join metric row.
#[derive(Default)]
pub(super) struct PartitionedJoinOutputSummary {
    pub(super) index_bytes: u64,
    pub(super) mixed_size_data_bytes: u64,
    pub(super) output_reservation_bytes: u64,
    pub(super) output_rows: u64,
    pub(super) output_bytes: u64,
}

impl PartitionedJoinOutputSummary {
    pub(super) fn record_partition_output(&mut self, output: &PartitionedJoinPartitionOutput) {
        self.index_bytes = self.index_bytes.saturating_add(output.index_bytes);
        self.mixed_size_data_bytes = self.mixed_size_data_bytes.saturating_add(output.mixed_size_data_bytes);
        self.output_reservation_bytes = self.output_reservation_bytes.saturating_add(output.output_reservation_bytes);
        self.output_rows = self.output_rows.saturating_add(output.output_rows);
        self.output_bytes = self.output_bytes.saturating_add(output.output_bytes);
    }

    pub(super) fn observe_concatenated_output(&mut self, output: &ResidentGpuFrame) {
        self.output_bytes = self.output_bytes.max(output.reservation.bytes());
    }
}

pub(super) fn run_partitioned_join_driver<F>(
    plan: PartitionedJoinDriverPlan<'_>,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    mut materialize_partition: F,
) -> Result<PartitionedJoinBufferedOutput>
where
    F: FnMut(
        &ExecutionContext,
        NodeId,
        &JoinNode,
        BoundedJoinSideFrame,
        BoundedJoinSideFrame,
        &mut ReservationTracker,
        &mut QueryMetrics,
    ) -> Result<PartitionedJoinPartitionOutput>,
{
    let original = PartitionedJoinInputStats {
        left_rows: left.rows,
        left_bytes: left.bytes,
        right_rows: right.rows,
        right_bytes: right.bytes,
    };
    record_partitioned_join_selection(plan, original, metrics);
    let partitioned = partition_join_sides(plan, left, right, reservations, metrics)?;
    let left_partitioned_bytes = partitioned.left.reservation.bytes();
    let right_partitioned_bytes = partitioned.right.reservation.bytes();
    record_partitioned_join_sides(plan, left_partitioned_bytes, right_partitioned_bytes, metrics);

    let mut outputs = Vec::with_capacity(plan.partition_count);
    let mut summary = PartitionedJoinOutputSummary::default();
    let PartitionedJoinSides { left: left_partitioned, right: right_partitioned } = partitioned;
    for partition in 0..plan.partition_count {
        let left_part = match slice_partition_join_side(
            plan.context,
            plan.node_id,
            JoinSide::Left,
            plan.node.spec.build_side,
            &left_partitioned,
            partition,
            reservations,
            metrics,
        ) {
            Ok(partition) => partition,
            Err(error) => {
                release_partitioned_join_driver_state(
                    Some(left_partitioned),
                    Some(right_partitioned),
                    outputs,
                    reservations,
                    metrics,
                );
                return Err(error);
            },
        };
        let right_part = match slice_partition_join_side(
            plan.context,
            plan.node_id,
            JoinSide::Right,
            plan.node.spec.build_side,
            &right_partitioned,
            partition,
            reservations,
            metrics,
        ) {
            Ok(partition) => partition,
            Err(error) => {
                release_bounded_join_side_frame(left_part, reservations, metrics);
                release_partitioned_join_driver_state(
                    Some(left_partitioned),
                    Some(right_partitioned),
                    outputs,
                    reservations,
                    metrics,
                );
                return Err(error);
            },
        };
        match materialize_partition(plan.context, plan.node_id, plan.node, left_part, right_part, reservations, metrics)
        {
            Ok(output) => {
                summary.record_partition_output(&output);
                outputs.push(output.frame);
            },
            Err(error) => {
                release_partitioned_join_driver_state(
                    Some(left_partitioned),
                    Some(right_partitioned),
                    outputs,
                    reservations,
                    metrics,
                );
                return Err(error);
            },
        }
    }
    release_partitioned_join_side(left_partitioned, reservations, metrics);
    release_partitioned_join_side(right_partitioned, reservations, metrics);
    Ok(PartitionedJoinBufferedOutput { outputs, original, summary })
}

pub(super) fn partition_join_sides(
    plan: PartitionedJoinDriverPlan<'_>,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<PartitionedJoinSides> {
    let left_keys = join_keys_for_side(plan.node, JoinSide::Left);
    let right_keys = join_keys_for_side(plan.node, JoinSide::Right);
    let overrides = execution_overrides(plan.context);
    let left_partitioned = hash_partition_join_side(
        plan.context,
        plan.node_id,
        JoinSide::Left,
        left,
        &left_keys,
        plan.partition_count,
        &overrides,
        reservations,
        metrics,
    )?;
    let right_partitioned = match hash_partition_join_side(
        plan.context,
        plan.node_id,
        JoinSide::Right,
        right,
        &right_keys,
        plan.partition_count,
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
    Ok(PartitionedJoinSides { left: left_partitioned, right: right_partitioned })
}

// Owns both partitioned sides so error paths can release them together.
pub(super) struct PartitionedJoinSides {
    pub(super) left: PartitionedJoinSide,
    pub(super) right: PartitionedJoinSide,
}

pub(super) fn record_partitioned_join_selection(
    plan: PartitionedJoinDriverPlan<'_>,
    original: PartitionedJoinInputStats,
    metrics: &mut QueryMetrics,
) {
    metrics.record_join_strategy_decision(
        plan.node_id,
        &BTreeMap::from([
            ("join_strategy", plan.strategy.strategy_label.to_string()),
            ("join_type", plan.node.spec.join_type.to_string()),
            ("build_side", plan.node.spec.build_side.to_string()),
            ("key_count", plan.node.spec.keys.len().to_string()),
            (
                "observed_build_rows",
                side_metric(plan.node.spec.build_side, original.left_rows, original.right_rows).to_string(),
            ),
            (
                "build_side_bytes",
                side_metric(plan.node.spec.build_side, original.left_bytes, original.right_bytes).to_string(),
            ),
            (
                "probe_side_bytes",
                side_metric(probe_side(plan.node.spec.build_side), original.left_bytes, original.right_bytes)
                    .to_string(),
            ),
            ("probe_reuse_count", "1".to_string()),
            (
                "reusable_hash_join_rejected_reason",
                partitioned_join_reusable_rejected_reason(
                    side_metric(plan.node.spec.build_side, original.left_rows, original.right_rows),
                    side_metric(plan.node.spec.build_side, original.left_bytes, original.right_bytes),
                    side_metric(probe_side(plan.node.spec.build_side), original.left_bytes, original.right_bytes),
                    plan.strategy_options,
                )
                .to_string(),
            ),
            ("partition_count", plan.partition_count.to_string()),
            ("partition_target_bytes", plan.partition_target_bytes.to_string()),
        ]),
    );
}

pub(super) fn record_partitioned_join_sides(
    plan: PartitionedJoinDriverPlan<'_>,
    left_partitioned_bytes: u64,
    right_partitioned_bytes: u64,
    metrics: &mut QueryMetrics,
) {
    metrics.record_join_strategy_decision(
        plan.node_id,
        &BTreeMap::from([
            ("join_strategy", plan.strategy.strategy_label.to_string()),
            ("partition_count", plan.partition_count.to_string()),
            ("partition_target_bytes", plan.partition_target_bytes.to_string()),
            ("build_partition_batch_count", plan.partition_count.to_string()),
            ("probe_partition_batch_count", plan.partition_count.to_string()),
            (
                "build_partitioned_bytes",
                side_metric(plan.node.spec.build_side, left_partitioned_bytes, right_partitioned_bytes).to_string(),
            ),
            (
                "probe_partitioned_bytes",
                side_metric(probe_side(plan.node.spec.build_side), left_partitioned_bytes, right_partitioned_bytes)
                    .to_string(),
            ),
        ]),
    );
}

pub(super) fn record_partitioned_join_operator_metrics(
    plan: PartitionedJoinDriverPlan<'_>,
    original: PartitionedJoinInputStats,
    summary: &PartitionedJoinOutputSummary,
    metrics: &mut QueryMetrics,
) {
    metrics.record_operator_metrics(crate::metrics::OperatorMetrics::join(
        plan.node_id,
        crate::metrics::JoinOperatorMetrics {
            join_type: plan.node.spec.join_type.to_string(),
            execution_strategy: plan.strategy.execution_strategy.to_string(),
            key_count: plan.node.spec.keys.len(),
            residual_predicate: plan.strategy.residual_predicate,
            null_equality: plan.node.spec.null_equality.to_string(),
            build_side: plan.node.spec.build_side.to_string(),
            probe_side: probe_side(plan.node.spec.build_side).to_string(),
            build_rows: side_metric(plan.node.spec.build_side, original.left_rows, original.right_rows),
            build_bytes: side_metric(plan.node.spec.build_side, original.left_bytes, original.right_bytes),
            probe_rows: side_metric(probe_side(plan.node.spec.build_side), original.left_rows, original.right_rows),
            probe_bytes: side_metric(probe_side(plan.node.spec.build_side), original.left_bytes, original.right_bytes),
            hash_table_bytes: 0,
            filtered_probe_state_bytes: 0,
            mixed_size_data_bytes: summary.mixed_size_data_bytes,
            gather_map_bytes: summary.index_bytes,
            output_reservation_bytes: summary.output_reservation_bytes,
            output_rows: summary.output_rows,
            output_bytes: summary.output_bytes,
        },
    ));
}

fn release_partitioned_join_driver_state(
    left: Option<PartitionedJoinSide>,
    right: Option<PartitionedJoinSide>,
    outputs: Vec<ResidentGpuFrame>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) {
    release_streamed_join_outputs_reverse(reservations, outputs, metrics);
    if let Some(left) = left {
        release_partitioned_join_side(left, reservations, metrics);
    }
    if let Some(right) = right {
        release_partitioned_join_side(right, reservations, metrics);
    }
}
