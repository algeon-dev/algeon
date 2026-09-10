use super::{
    BoundedExecutionAttempt, CommittedPartial, GpuDataFrame, NodeId, Result, final_sort_runtime_facts,
    observe_device_memory_bytes,
};

pub(super) fn reserve_final_sort_output_bytes(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    output_bytes: u64,
    attempt: usize,
    max_attempts: usize,
) -> Result<u64> {
    let _ = (execution, node_id, attempt, max_attempts);
    Ok(output_bytes)
}

pub(super) fn finalize_final_sort_output_reservation(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    frame: GpuDataFrame,
    _estimated_output_bytes: u64,
    attempt: usize,
    max_attempts: usize,
) -> Result<CommittedPartial> {
    let output_bytes = observe_device_memory_bytes(&frame, execution.metrics_mut()).map_err(|error| {
        final_sort_runtime_facts("final_sort_output", node_id, attempt, max_attempts, execution.metrics_mut(), error)
    })?;

    Ok(CommittedPartial::with_device_bytes(frame, output_bytes))
}
