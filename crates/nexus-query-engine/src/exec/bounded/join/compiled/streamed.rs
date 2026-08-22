use super::*;

pub(super) fn execute_compiled_streamed_one_shot_join_materialize(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    build: BoundedCompiledPreparedJoinBuildState,
    outputs: Vec<ResidentGpuFrame>,
    summary: StreamedOneShotJoinSummary,
) -> Result<ResidentGpuFrame> {
    let BoundedCompiledPreparedJoinBuildState { side: build_side, input, .. } = build;
    let probe_side = probe_side(build_side);
    let build_rows = input.rows;
    let build_bytes = input.bytes;
    release_bounded_join_side_frame(execution, input);

    let output = concatenate_streamed_hash_join_outputs(execution, node_id, outputs)?;
    record_streamed_one_shot_join_metrics(
        node_id,
        node,
        build_side,
        probe_side,
        build_rows,
        build_bytes,
        &summary,
        execution.metrics_mut(),
    );
    Ok(output)
}
#[expect(
    clippy::too_many_arguments,
    reason = "bounded execution passes operator-occurrence inputs explicitly beside the attempt state"
)]
pub(super) fn execute_compiled_streamed_partitioned_hash_join_materialize(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    build_side: JoinSide,
    build_rows: u64,
    build_bytes: u64,
    outputs: Vec<ResidentGpuFrame>,
    summary: StreamedPartitionedJoinSummary,
) -> Result<ResidentGpuFrame> {
    let output = concatenate_streamed_hash_join_outputs(execution, node_id, outputs)?;
    record_streamed_partitioned_hash_join_metrics(
        node_id,
        node,
        build_side,
        probe_side(build_side),
        build_rows,
        build_bytes,
        &summary,
        execution.metrics_mut(),
    );
    Ok(output)
}

#[expect(
    clippy::too_many_arguments,
    reason = "strategy-decision facts are reported individually; grouping them would be a metrics DTO"
)]
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
    metrics.record_one_shot_join_materialization_bytes(summary.materialization_bytes);
    metrics.record_one_shot_join_output_bytes(summary.output_bytes);
    metrics.record_operator_metrics(crate::metrics::OperatorMetrics::join(
        node_id,
        crate::metrics::JoinOperatorMetrics {
            query_id: None,
            join_type: node.spec.join_type.to_string(),
            execution_strategy: "bounded_one_shot_equi_join".to_string(),
            key_count: node.spec.keys.len(),
            residual_predicate: node.spec.residual_predicate.is_some(),
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

pub(super) fn record_one_shot_join_chunk_metrics(
    node_id: NodeId,
    node: &JoinNode,
    build_side: JoinSide,
    probe_side: JoinSide,
    chunk: &OneShotJoinChunkMetrics,
    metrics: &mut QueryMetrics,
) {
    metrics.record_one_shot_join();
    metrics.record_one_shot_join_index_bytes(chunk.index_bytes);
    metrics.record_one_shot_join_materialization_bytes(chunk.materialization_bytes);
    metrics.record_one_shot_join_output_bytes(chunk.output_bytes);
    metrics.record_operator_metrics(crate::metrics::OperatorMetrics::join(
        node_id,
        crate::metrics::JoinOperatorMetrics {
            query_id: None,
            join_type: node.spec.join_type.to_string(),
            execution_strategy: "bounded_one_shot_equi_join".to_string(),
            key_count: node.spec.keys.len(),
            residual_predicate: node.spec.residual_predicate.is_some(),
            null_equality: node.spec.null_equality.to_string(),
            build_side: build_side.to_string(),
            probe_side: probe_side.to_string(),
            build_rows: chunk.build_rows,
            build_bytes: chunk.build_bytes,
            probe_rows: chunk.probe_rows,
            probe_bytes: chunk.probe_bytes,
            hash_table_bytes: 0,
            filtered_probe_state_bytes: 0,
            mixed_size_data_bytes: 0,
            gather_map_bytes: chunk.index_bytes,
            output_reservation_bytes: chunk.output_reservation_bytes,
            output_rows: chunk.output_rows,
            output_bytes: chunk.output_bytes,
        },
    ));
}

#[expect(
    clippy::too_many_arguments,
    reason = "strategy-decision facts are reported individually; grouping them would be a metrics DTO"
)]
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
            query_id: None,
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
