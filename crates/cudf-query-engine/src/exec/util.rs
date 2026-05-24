use super::*;

pub(super) fn execution_overrides(context: &ExecutionContext) -> ExecutionOverrides<'_> {
    ExecutionOverrides::new().with_stream(context.stream()).with_memory_resource(context.memory_resource())
}

pub(super) fn observe_device_memory_bytes(frame: &GpuDataFrame, metrics: &mut QueryMetrics) -> Result<u64> {
    let info = frame.frame().info_with(InfoDetail::DEVICE_ALLOCATED).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.metrics.device_memory",
            "failed to record native device memory diagnostics",
            error,
        )
    })?;
    let bytes = info.memory_accounting.and_then(|accounting| accounting.device_allocated_bytes).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::Internal,
            "cudf_query_engine.metrics.device_memory",
            "cudf dataframe memory accounting did not include device-allocated bytes",
        )
        .with_fact("rows", frame.num_rows())
        .with_fact("columns", frame.num_columns())
    })?;
    metrics.record_device_memory_observation(bytes as u64);
    Ok(bytes as u64)
}

pub(super) fn record_device_memory_observation(frame: &GpuDataFrame, metrics: &mut QueryMetrics) -> Result<()> {
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
            "cudf_query_engine.limit.validate",
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

pub(super) fn invalid_node_reference(id: NodeId, message: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::InvalidPlan, "cudf_query_engine.plan.execute", message).with_fact("node", id)
}

pub(super) fn unsupported_bounded_shape(message: impl Into<String>, reason: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::UnsupportedPlanShape, "cudf_query_engine.bounded.plan", message)
        .with_fact("reason", reason)
}

pub(super) fn comma_separated_usize(values: &[usize]) -> String {
    let mut output = String::new();
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        output.push_str(&value.to_string());
    }
    output
}
