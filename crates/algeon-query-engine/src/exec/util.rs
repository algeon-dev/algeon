#[cfg(any(feature = "cuvs", feature = "cuml"))]
use super::NodeId;
use super::{
    Array, ExecutionMetrics, GpuDataFrame, InfoDetail, LimitSpec, QueryError, QueryErrorCode, RecordBatch, Result,
    cudf_query_error,
};
pub(super) use crate::util::comma_separated_usize;

pub(super) fn observe_device_memory_bytes(frame: &GpuDataFrame, metrics: &ExecutionMetrics) -> Result<u64> {
    let info = frame.frame().info_with(InfoDetail::DEVICE_ALLOCATED).map_err(|error| {
        cudf_query_error(
            "query_engine.metrics.device_memory",
            "failed to record native device memory diagnostics",
            error,
        )
    })?;
    let bytes = info.memory_accounting.and_then(|accounting| accounting.device_allocated_bytes).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::Internal,
            "query_engine.metrics.device_memory",
            "cudf dataframe memory accounting did not include device-allocated bytes",
        )
        .with_fact("rows", frame.num_rows())
        .with_fact("columns", frame.num_columns())
    })?;
    metrics.record_device_memory_observation(bytes as u64);
    Ok(bytes as u64)
}

pub(super) fn record_device_memory_observation(frame: &GpuDataFrame, metrics: &ExecutionMetrics) -> Result<()> {
    observe_device_memory_bytes(frame, metrics).map(|_| ())
}

pub(super) fn limit_slice(row_count: usize, spec: LimitSpec) -> Result<(usize, usize)> {
    let offset = spec.skip.min(row_count);
    let Some(fetch) = spec.fetch else {
        return Ok((offset, row_count.saturating_sub(offset)));
    };
    let Some(end) = spec.skip.checked_add(fetch) else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.limit.validate",
            "limit skip and fetch overflow usize",
        )
        .with_fact("skip", spec.skip)
        .with_fact("fetch", fetch));
    };
    let length = end.min(row_count).saturating_sub(offset);
    Ok((offset, length))
}

pub(super) fn record_batch_memory_size(batch: &RecordBatch) -> usize {
    batch.columns().iter().map(|array| array.get_array_memory_size()).sum()
}

#[cfg(any(feature = "cuvs", feature = "cuml"))]
pub(super) fn invalid_node_reference(id: NodeId, message: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::InvalidPlan, "query_engine.plan.execute", message).with_fact("node", id)
}

/// A bounded execution precondition that admission proved no longer holds:
/// the compiled strategy, input orientation, or native state the executor
/// relies on is not what the admitted plan described.
pub(super) fn bounded_invariant_violation(message: impl Into<String>, reason: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::InternalInvariantViolation, "query_engine.bounded.plan", message)
        .with_fact("reason", reason)
}
