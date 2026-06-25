use super::*;

// Retry subdivision constants are deliberately fixed because they describe the
// deterministic shape of already-admitted partitioned-join work, not a new
// admission policy or report surface. Runtime knobs still bound the retry depth;
// these values keep each refinement predictable and independent of memory
// estimates that may have just proven too optimistic.
//
// Repartition retry always bisects a failed partition. A fixed fan-out of two
// matches the source-chunk retry shape, limits live retry state, and preserves a
// stable left-to-right output order.
const PARTITIONED_JOIN_RETRY_REPARTITION_COUNT: usize = 2;
// Retry repartitioning stops at the same logical partition ceiling as initial
// partition planning. Keeping this cap fixed prevents runaway repartition trees
// while preserving the established bounded partition-count contract.
const PARTITIONED_JOIN_MAX_LOGICAL_PARTITIONS: usize = 1024;
// `min_retry_chunk_row_groups` is a source-row-group knob. Joins split resident
// row ranges, so this fixed row unit maps that existing knob onto a row floor
// without adding another public policy field.
const PARTITIONED_JOIN_MIN_PROBE_ROWS_PER_RETRY_UNIT: usize = 1024;

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

#[derive(Clone, Copy)]
struct PartitionedJoinRetryPolicy {
    max_attempts: usize,
    min_probe_rows: usize,
}

#[derive(Clone, Copy)]
struct PartitionedJoinWorkSlice {
    root_partition: usize,
    partition: usize,
    logical_partition_count: usize,
    left: PartitionedJoinRowRange,
    right: PartitionedJoinRowRange,
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
        match materialize_partitioned_join_partition_with_retry(
            plan,
            &left_partitioned,
            &right_partitioned,
            partition,
            reservations,
            metrics,
            &mut materialize_partition,
        ) {
            Ok(partition_outputs) => {
                for output in partition_outputs {
                    summary.record_partition_output(&output);
                    outputs.push(output.frame);
                }
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

pub(super) fn materialize_partitioned_join_partition_with_retry<F>(
    plan: PartitionedJoinDriverPlan<'_>,
    left_partitioned: &PartitionedJoinSide,
    right_partitioned: &PartitionedJoinSide,
    partition: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    materialize_partition: &mut F,
) -> Result<Vec<PartitionedJoinPartitionOutput>>
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
    let work = PartitionedJoinWorkSlice {
        root_partition: partition,
        partition,
        logical_partition_count: plan.partition_count,
        left: partition_row_range(left_partitioned, partition)?,
        right: partition_row_range(right_partitioned, partition)?,
    };
    materialize_partitioned_join_work_with_retry(
        plan,
        left_partitioned,
        right_partitioned,
        work,
        partitioned_join_retry_policy(plan.context),
        1,
        reservations,
        metrics,
        materialize_partition,
    )
}

fn materialize_partitioned_join_work_with_retry<F>(
    plan: PartitionedJoinDriverPlan<'_>,
    left_partitioned: &PartitionedJoinSide,
    right_partitioned: &PartitionedJoinSide,
    work: PartitionedJoinWorkSlice,
    policy: PartitionedJoinRetryPolicy,
    attempt: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    materialize_partition: &mut F,
) -> Result<Vec<PartitionedJoinPartitionOutput>>
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
    plan.context.check_cancellation("nexus_query_engine.exec.bounded.partitioned_join_retry")?;
    match materialize_partitioned_join_work_once(
        plan,
        left_partitioned,
        right_partitioned,
        work,
        reservations,
        metrics,
        materialize_partition,
    ) {
        Ok(output) => Ok(vec![output]),
        Err(error) if crate::exec::retry::is_retryable_resource_error(&error) => {
            retry_partitioned_join_work_after_error(
                plan,
                left_partitioned,
                right_partitioned,
                work,
                policy,
                attempt,
                error,
                reservations,
                metrics,
                materialize_partition,
            )
        },
        Err(error) => Err(error),
    }
}

fn retry_partitioned_join_work_after_error<F>(
    plan: PartitionedJoinDriverPlan<'_>,
    left_partitioned: &PartitionedJoinSide,
    right_partitioned: &PartitionedJoinSide,
    work: PartitionedJoinWorkSlice,
    policy: PartitionedJoinRetryPolicy,
    attempt: usize,
    error: QueryError,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    materialize_partition: &mut F,
) -> Result<Vec<PartitionedJoinPartitionOutput>>
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
    record_partitioned_join_retry_error(plan.context, &error, metrics);
    if attempt >= policy.max_attempts {
        return Err(attach_partitioned_join_retry_error(
            error,
            plan,
            work,
            attempt,
            policy,
            "retry_attempts_exhausted",
        ));
    }

    if !partitioned_join_retry_failure_requires_repartition(plan.node, &error)
        && partitioned_join_probe_split_supported(plan.node)
        && let Some((left_work, right_work)) = split_partitioned_join_probe_work(plan.node, work, policy.min_probe_rows)
    {
        return materialize_partitioned_join_retry_children(
            plan,
            left_partitioned,
            right_partitioned,
            [left_work, right_work],
            policy,
            attempt + 1,
            reservations,
            metrics,
            materialize_partition,
        );
    }

    if can_repartition_partitioned_join_work(work) {
        return materialize_repartitioned_join_work_with_retry(
            plan,
            left_partitioned,
            right_partitioned,
            work,
            policy,
            attempt + 1,
            reservations,
            metrics,
            materialize_partition,
        )
        .map_err(|retry_error| {
            if crate::exec::retry::is_retryable_resource_error(&retry_error) {
                attach_partitioned_join_retry_error(
                    retry_error,
                    plan,
                    work,
                    attempt + 1,
                    policy,
                    "partition_repartition_retry_failed",
                )
            } else {
                retry_error
            }
        });
    }

    let retry_action = if partitioned_join_probe_split_supported(plan.node) {
        "fail_min_join_partition"
    } else {
        "join_probe_split_unsupported"
    };
    Err(attach_partitioned_join_retry_error(error, plan, work, attempt, policy, retry_action))
}

fn record_partitioned_join_retry_error(context: &ExecutionContext, error: &QueryError, metrics: &mut QueryMetrics) {
    crate::exec::retry::record_retryable_resource_error(error, metrics);
    if context.native_pipeline_execution_mode() == crate::mode::NativePipelineExecutionMode::Adaptive {
        metrics.record_adaptive_runtime_action_counts(1, 0, 0);
    }
}

fn materialize_partitioned_join_retry_children<F>(
    plan: PartitionedJoinDriverPlan<'_>,
    left_partitioned: &PartitionedJoinSide,
    right_partitioned: &PartitionedJoinSide,
    children: [PartitionedJoinWorkSlice; 2],
    policy: PartitionedJoinRetryPolicy,
    attempt: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    materialize_partition: &mut F,
) -> Result<Vec<PartitionedJoinPartitionOutput>>
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
    let mut outputs = Vec::new();
    for child in children {
        match materialize_partitioned_join_work_with_retry(
            plan,
            left_partitioned,
            right_partitioned,
            child,
            policy,
            attempt,
            reservations,
            metrics,
            materialize_partition,
        ) {
            Ok(mut child_outputs) => outputs.append(&mut child_outputs),
            Err(error) => {
                release_partitioned_join_partition_outputs(outputs, reservations, metrics);
                return Err(error);
            },
        }
    }
    Ok(outputs)
}

fn materialize_repartitioned_join_work_with_retry<F>(
    plan: PartitionedJoinDriverPlan<'_>,
    left_partitioned: &PartitionedJoinSide,
    right_partitioned: &PartitionedJoinSide,
    work: PartitionedJoinWorkSlice,
    policy: PartitionedJoinRetryPolicy,
    attempt: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    materialize_partition: &mut F,
) -> Result<Vec<PartitionedJoinPartitionOutput>>
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
    let repartitioned =
        repartition_partitioned_join_work(plan, left_partitioned, right_partitioned, work, reservations, metrics)?;
    let PartitionedJoinSides { left: retry_left, right: retry_right } = repartitioned;
    let mut outputs = Vec::new();
    for partition in 0..PARTITIONED_JOIN_RETRY_REPARTITION_COUNT {
        let child = match repartitioned_join_child_work(work, &retry_left, &retry_right, partition) {
            Ok(child) => child,
            Err(error) => {
                release_partitioned_join_partition_outputs(outputs, reservations, metrics);
                release_partitioned_join_side(retry_left, reservations, metrics);
                release_partitioned_join_side(retry_right, reservations, metrics);
                return Err(error);
            },
        };
        match materialize_partitioned_join_work_with_retry(
            plan,
            &retry_left,
            &retry_right,
            child,
            policy,
            attempt,
            reservations,
            metrics,
            materialize_partition,
        ) {
            Ok(mut partition_outputs) => outputs.append(&mut partition_outputs),
            Err(error) => {
                release_partitioned_join_partition_outputs(outputs, reservations, metrics);
                release_partitioned_join_side(retry_left, reservations, metrics);
                release_partitioned_join_side(retry_right, reservations, metrics);
                return Err(error);
            },
        }
    }
    release_partitioned_join_side(retry_left, reservations, metrics);
    release_partitioned_join_side(retry_right, reservations, metrics);
    Ok(outputs)
}

fn repartition_partitioned_join_work(
    plan: PartitionedJoinDriverPlan<'_>,
    left_partitioned: &PartitionedJoinSide,
    right_partitioned: &PartitionedJoinSide,
    work: PartitionedJoinWorkSlice,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<PartitionedJoinSides> {
    let left = slice_partition_join_side_range(
        plan.context,
        plan.node_id,
        JoinSide::Left,
        plan.node.spec.build_side,
        left_partitioned,
        work.partition,
        work.left,
        reservations,
        metrics,
    )?;
    let right = match slice_partition_join_side_range(
        plan.context,
        plan.node_id,
        JoinSide::Right,
        plan.node.spec.build_side,
        right_partitioned,
        work.partition,
        work.right,
        reservations,
        metrics,
    ) {
        Ok(right) => right,
        Err(error) => {
            release_bounded_join_side_frame(left, reservations, metrics);
            return Err(error);
        },
    };
    let left_keys = join_keys_for_side(plan.node, JoinSide::Left);
    let right_keys = join_keys_for_side(plan.node, JoinSide::Right);
    let overrides = execution_overrides(plan.context);
    let left_partitioned = match hash_partition_join_side(
        plan.context,
        plan.node_id,
        JoinSide::Left,
        left,
        &left_keys,
        PARTITIONED_JOIN_RETRY_REPARTITION_COUNT,
        &overrides,
        reservations,
        metrics,
    ) {
        Ok(partitioned) => partitioned,
        Err(error) => {
            release_bounded_join_side_frame(right, reservations, metrics);
            return Err(error);
        },
    };
    let right_partitioned = match hash_partition_join_side(
        plan.context,
        plan.node_id,
        JoinSide::Right,
        right,
        &right_keys,
        PARTITIONED_JOIN_RETRY_REPARTITION_COUNT,
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

fn materialize_partitioned_join_work_once<F>(
    plan: PartitionedJoinDriverPlan<'_>,
    left_partitioned: &PartitionedJoinSide,
    right_partitioned: &PartitionedJoinSide,
    work: PartitionedJoinWorkSlice,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    materialize_partition: &mut F,
) -> Result<PartitionedJoinPartitionOutput>
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
    let left_part = slice_partition_join_side_range(
        plan.context,
        plan.node_id,
        JoinSide::Left,
        plan.node.spec.build_side,
        left_partitioned,
        work.partition,
        work.left,
        reservations,
        metrics,
    )?;
    let right_part = match slice_partition_join_side_range(
        plan.context,
        plan.node_id,
        JoinSide::Right,
        plan.node.spec.build_side,
        right_partitioned,
        work.partition,
        work.right,
        reservations,
        metrics,
    ) {
        Ok(partition) => partition,
        Err(error) => {
            release_bounded_join_side_frame(left_part, reservations, metrics);
            return Err(error);
        },
    };
    materialize_partition(plan.context, plan.node_id, plan.node, left_part, right_part, reservations, metrics).map_err(
        |error| {
            error
                .with_fact_if_absent("partition", work.root_partition)
                .with_fact_if_absent("logical_partition_count", work.logical_partition_count)
        },
    )
}

fn partitioned_join_retry_policy(context: &ExecutionContext) -> PartitionedJoinRetryPolicy {
    let MemoryControlPolicy::Bounded { max_retry_attempts, min_retry_chunk_row_groups, .. } =
        context.memory_control_policy()
    else {
        return PartitionedJoinRetryPolicy { max_attempts: 1, min_probe_rows: 1 };
    };
    PartitionedJoinRetryPolicy {
        max_attempts: max_retry_attempts.max(1),
        min_probe_rows: min_retry_chunk_row_groups
            .max(1)
            .saturating_mul(PARTITIONED_JOIN_MIN_PROBE_ROWS_PER_RETRY_UNIT),
    }
}

fn partitioned_join_probe_split_supported(node: &JoinNode) -> bool {
    matches!(node.spec.join_type, JoinType::Inner)
        || (node.spec.build_side == JoinSide::Right
            && matches!(node.spec.join_type, JoinType::Left | JoinType::LeftSemi | JoinType::LeftAnti))
}

fn partitioned_join_retry_failure_requires_repartition(node: &JoinNode, error: &QueryError) -> bool {
    let build_boundary = match node.spec.build_side {
        JoinSide::Left => "join_partition_left_side",
        JoinSide::Right => "join_partition_right_side",
    };
    error.fact_value("memory_boundary") == Some(build_boundary)
        || error.fact_value("budget_boundary") == Some(build_boundary)
        || error.fact_value("side") == Some(partitioned_join_side_label(node.spec.build_side))
}

fn partitioned_join_side_label(side: JoinSide) -> &'static str {
    match side {
        JoinSide::Left => "left",
        JoinSide::Right => "right",
    }
}

fn split_partitioned_join_probe_work(
    node: &JoinNode,
    work: PartitionedJoinWorkSlice,
    min_probe_rows: usize,
) -> Option<(PartitionedJoinWorkSlice, PartitionedJoinWorkSlice)> {
    let probe = probe_side(node.spec.build_side);
    let probe_range = match probe {
        JoinSide::Left => work.left,
        JoinSide::Right => work.right,
    };
    if probe_range.len <= min_probe_rows {
        return None;
    }
    let left_len = probe_range.len / 2;
    let right_len = probe_range.len.saturating_sub(left_len);
    if left_len == 0 || right_len == 0 {
        return None;
    }

    let left_probe = PartitionedJoinRowRange { offset: probe_range.offset, len: left_len };
    let right_probe = PartitionedJoinRowRange { offset: probe_range.offset.saturating_add(left_len), len: right_len };
    let mut left_work = work;
    let mut right_work = work;
    match probe {
        JoinSide::Left => {
            left_work.left = left_probe;
            right_work.left = right_probe;
        },
        JoinSide::Right => {
            left_work.right = left_probe;
            right_work.right = right_probe;
        },
    }
    Some((left_work, right_work))
}

fn can_repartition_partitioned_join_work(work: PartitionedJoinWorkSlice) -> bool {
    work.logical_partition_count < PARTITIONED_JOIN_MAX_LOGICAL_PARTITIONS && (work.left.len > 1 || work.right.len > 1)
}

fn repartitioned_join_child_work(
    parent: PartitionedJoinWorkSlice,
    left: &PartitionedJoinSide,
    right: &PartitionedJoinSide,
    partition: usize,
) -> Result<PartitionedJoinWorkSlice> {
    Ok(PartitionedJoinWorkSlice {
        root_partition: parent.root_partition,
        partition,
        logical_partition_count: parent
            .logical_partition_count
            .saturating_mul(PARTITIONED_JOIN_RETRY_REPARTITION_COUNT)
            .min(PARTITIONED_JOIN_MAX_LOGICAL_PARTITIONS),
        left: partition_row_range(left, partition)?,
        right: partition_row_range(right, partition)?,
    })
}

fn partitioned_join_retry_facts(
    _plan: PartitionedJoinDriverPlan<'_>,
    _work: PartitionedJoinWorkSlice,
    _attempt: usize,
    _policy: PartitionedJoinRetryPolicy,
    retry_action: &'static str,
) -> RetryFacts {
    RetryFacts::new().with_retry_action(retry_action).with_resume_boundary("partitioned_join")
}

fn attach_partitioned_join_retry_error(
    error: QueryError,
    plan: PartitionedJoinDriverPlan<'_>,
    work: PartitionedJoinWorkSlice,
    attempt: usize,
    policy: PartitionedJoinRetryPolicy,
    retry_action: &'static str,
) -> QueryError {
    partitioned_join_retry_facts(plan, work, attempt, policy, retry_action)
        .attach_to_error(error)
        .with_fact_if_absent("partition", work.root_partition)
        .with_fact_if_absent("join_type", plan.node.spec.join_type)
        .with_fact_if_absent("build_side", plan.node.spec.build_side)
        .with_fact_if_absent("retry_attempt", attempt)
        .with_fact_if_absent("max_retry_attempts", policy.max_attempts)
        .with_fact_if_absent("logical_partition_count", work.logical_partition_count)
        .with_fact_if_absent("left_partition_rows", work.left.len)
        .with_fact_if_absent("right_partition_rows", work.right.len)
        .with_fact_if_absent("min_retry_probe_rows", policy.min_probe_rows)
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
            query_id: None,
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

fn release_partitioned_join_partition_outputs(
    mut outputs: Vec<PartitionedJoinPartitionOutput>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) {
    let mut frames = Vec::with_capacity(outputs.len());
    while let Some(output) = outputs.pop() {
        frames.push(output.frame);
    }
    release_streamed_join_outputs_reverse(reservations, frames, metrics);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::JoinSpec;

    fn retry_join_node(join_type: JoinType, build_side: JoinSide) -> JoinNode {
        JoinNode {
            inputs: [NodeId::from_index_for_internal_use(1), NodeId::from_index_for_internal_use(2)],
            spec: JoinSpec::new(join_type, Vec::new()).with_build_side(build_side),
        }
    }

    fn retry_work(left_rows: usize, right_rows: usize, logical_partition_count: usize) -> PartitionedJoinWorkSlice {
        PartitionedJoinWorkSlice {
            root_partition: 3,
            partition: 3,
            logical_partition_count,
            left: PartitionedJoinRowRange { offset: 0, len: left_rows },
            right: PartitionedJoinRowRange { offset: 0, len: right_rows },
        }
    }

    fn retryable_error_with_fact(key: &'static str, value: &'static str) -> QueryError {
        QueryError::new(
            QueryErrorCode::MemoryReservationFailed,
            "nexus_query_engine.bounded.join.partitioned_retry_test",
            "injected retryable partitioned join retry test error",
        )
        .with_fact(key, value)
    }

    #[test]
    fn probe_split_splits_left_probe_for_left_join_with_right_build() {
        let node = retry_join_node(JoinType::Left, JoinSide::Right);
        let work = retry_work(9, 4, 8);

        let (left, right) = split_partitioned_join_probe_work(&node, work, 1).expect("left probe should split");

        assert_eq!(left.left.offset, 0);
        assert_eq!(left.left.len, 4);
        assert_eq!(right.left.offset, 4);
        assert_eq!(right.left.len, 5);
        assert_eq!(left.right.offset, 0);
        assert_eq!(left.right.len, 4);
        assert_eq!(right.right.offset, 0);
        assert_eq!(right.right.len, 4);
    }

    #[test]
    fn probe_split_splits_right_probe_for_inner_join_with_left_build() {
        let node = retry_join_node(JoinType::Inner, JoinSide::Left);
        let work = retry_work(4, 10, 8);

        let (left, right) = split_partitioned_join_probe_work(&node, work, 1).expect("right probe should split");

        assert_eq!(left.right.offset, 0);
        assert_eq!(left.right.len, 5);
        assert_eq!(right.right.offset, 5);
        assert_eq!(right.right.len, 5);
        assert_eq!(left.left.len, 4);
        assert_eq!(right.left.len, 4);
    }

    #[test]
    fn probe_split_support_is_limited_to_stateless_probe_preserved_shapes() {
        assert!(partitioned_join_probe_split_supported(&retry_join_node(JoinType::Inner, JoinSide::Left)));
        assert!(partitioned_join_probe_split_supported(&retry_join_node(JoinType::Left, JoinSide::Right)));
        assert!(partitioned_join_probe_split_supported(&retry_join_node(JoinType::LeftSemi, JoinSide::Right)));
        assert!(partitioned_join_probe_split_supported(&retry_join_node(JoinType::LeftAnti, JoinSide::Right)));
        assert!(!partitioned_join_probe_split_supported(&retry_join_node(JoinType::Full, JoinSide::Right)));
        assert!(!partitioned_join_probe_split_supported(&retry_join_node(JoinType::Left, JoinSide::Left)));
    }

    #[test]
    fn build_side_retry_failures_go_to_repartition_before_probe_split() {
        let left_build = retry_join_node(JoinType::Inner, JoinSide::Left);
        assert!(partitioned_join_retry_failure_requires_repartition(
            &left_build,
            &retryable_error_with_fact("memory_boundary", "join_partition_left_side")
        ));
        assert!(!partitioned_join_retry_failure_requires_repartition(
            &left_build,
            &retryable_error_with_fact("memory_boundary", "join_partition_right_side")
        ));

        let right_build = retry_join_node(JoinType::Inner, JoinSide::Right);
        assert!(partitioned_join_retry_failure_requires_repartition(
            &right_build,
            &retryable_error_with_fact("side", "right")
        ));
        assert!(!partitioned_join_retry_failure_requires_repartition(
            &right_build,
            &retryable_error_with_fact("side", "left")
        ));
    }

    #[test]
    fn retry_repartition_is_bounded_by_logical_partition_cap_and_row_floor() {
        assert!(can_repartition_partitioned_join_work(retry_work(1, 2, 8)));
        assert!(!can_repartition_partitioned_join_work(retry_work(1, 1, 8)));
        assert!(!can_repartition_partitioned_join_work(retry_work(100, 100, PARTITIONED_JOIN_MAX_LOGICAL_PARTITIONS)));
    }
}
