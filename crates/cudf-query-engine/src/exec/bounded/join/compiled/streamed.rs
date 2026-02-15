use super::*;

pub(super) fn execute_compiled_streamed_one_shot_join_materialize(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    build: BoundedCompiledPreparedJoinBuildState,
    outputs: Vec<ResidentGpuFrame>,
    summary: StreamedOneShotJoinSummary,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let build_side = build.side;
    let probe_side = probe_side(build_side);
    let build_rows = build.input.rows;
    let build_bytes = build.input.bytes;
    drop(build);

    let output = concatenate_streamed_hash_join_outputs(context, node_id, outputs, reservations, metrics)?;
    record_streamed_one_shot_join_metrics(
        node_id,
        node,
        build_side,
        probe_side,
        build_rows,
        build_bytes,
        &summary,
        metrics,
    );
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute_compiled_streamed_partitioned_hash_join_materialize(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    build_side: JoinSide,
    build_rows: u64,
    build_bytes: u64,
    outputs: Vec<ResidentGpuFrame>,
    summary: StreamedPartitionedJoinSummary,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let output = concatenate_streamed_hash_join_outputs(context, node_id, outputs, reservations, metrics)?;
    record_streamed_partitioned_hash_join_metrics(
        node_id,
        node,
        build_side,
        probe_side(build_side),
        build_rows,
        build_bytes,
        &summary,
        metrics,
    );
    Ok(output)
}

pub(super) fn record_streamed_one_shot_join_metrics(
    node_id: NodeId,
    node: &JoinNode,
    build_side: JoinSide,
    probe_side: JoinSide,
    build_rows: u64,
    build_bytes: u64,
    summary: &StreamedOneShotJoinSummary,
    metrics: &mut QueryMetrics,
) {
    metrics.record_one_shot_join();
    metrics.record_one_shot_join_index_bytes(summary.index_bytes);
    metrics.record_one_shot_join_materialization_bytes(summary.output_reservation_bytes);
    metrics.record_one_shot_join_output_bytes(summary.output_bytes);
    metrics.record_operator_metrics(crate::metrics::OperatorMetrics::join(
        node_id,
        crate::metrics::JoinOperatorMetrics {
            join_type: node.spec.join_type.to_string(),
            execution_strategy: "bounded_one_shot_equi_join".to_string(),
            key_count: node.spec.keys.len(),
            residual_predicate: false,
            null_equality: node.spec.null_equality.to_string(),
            build_side: build_side.to_string(),
            probe_side: probe_side.to_string(),
            build_rows,
            build_bytes,
            probe_rows: summary.probe_rows,
            probe_bytes: summary.probe_bytes,
            hash_table_bytes: 0,
            filtered_probe_state_bytes: 0,
            mixed_size_data_bytes: 0,
            gather_map_bytes: summary.index_bytes,
            output_reservation_bytes: summary.output_reservation_bytes,
            output_rows: summary.output_rows,
            output_bytes: summary.output_bytes,
        },
    ));
}

pub(super) fn record_streamed_partitioned_hash_join_metrics(
    node_id: NodeId,
    node: &JoinNode,
    build_side: JoinSide,
    probe_side: JoinSide,
    build_rows: u64,
    build_bytes: u64,
    summary: &StreamedPartitionedJoinSummary,
    metrics: &mut QueryMetrics,
) {
    metrics.record_operator_metrics(crate::metrics::OperatorMetrics::join(
        node_id,
        crate::metrics::JoinOperatorMetrics {
            join_type: node.spec.join_type.to_string(),
            execution_strategy: "bounded_partitioned_hash_equi_join".to_string(),
            key_count: node.spec.keys.len(),
            residual_predicate: false,
            null_equality: node.spec.null_equality.to_string(),
            build_side: build_side.to_string(),
            probe_side: probe_side.to_string(),
            build_rows,
            build_bytes,
            probe_rows: summary.probe_rows,
            probe_bytes: summary.probe_bytes,
            hash_table_bytes: 0,
            filtered_probe_state_bytes: 0,
            mixed_size_data_bytes: summary.mixed_size_data_bytes,
            gather_map_bytes: summary.index_bytes,
            output_reservation_bytes: summary.output_reservation_bytes,
            output_rows: summary.output_rows,
            output_bytes: summary.output_bytes,
        },
    ));
    metrics.record_join_strategy_decision(
        node_id,
        &BTreeMap::from([
            ("join_strategy", "partitioned_hash_equi".to_string()),
            ("partition_count", summary.partition_count.to_string()),
            ("partition_target_bytes", summary.partition_target_bytes.to_string()),
            ("build_partition_batch_count", summary.partition_count.to_string()),
            ("probe_partition_batch_count", summary.probe_partition_batch_count.to_string()),
            ("build_partitioned_bytes", summary.build_partitioned_bytes.to_string()),
            ("probe_partitioned_bytes", summary.probe_partitioned_bytes.to_string()),
        ]),
    );
}
