use super::*;

/// Read a caller-supplied GPU-resident source by plan-local input name.
///
/// The handle must match the declared source schema and the same runtime/device
/// placement as the execution context. The returned frame stays on device.
pub fn read_gpu_resident_source(
    context: &ExecutionContext,
    source: &crate::source::GpuResidentSource,
    inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let handle = inputs.get(&source.input_name).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.gpu_resident_source.read",
            "GPU-resident source input was not provided to native execution",
        )
        .with_fact("input_name", source.input_name.as_str())
    })?;

    if handle.schema().as_ref() != source.schema.as_ref() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.gpu_resident_source.read",
            "GPU-resident source schema does not match the declared source schema",
        )
        .with_fact("input_name", source.input_name.as_str())
        .with_fact("declared_fields", source.schema.fields().len())
        .with_fact("actual_fields", handle.schema().fields().len()));
    }

    let context_runtime = context.shared_runtime()?;
    if !cudf::Runtime::shares_affinity(handle.runtime(), &context_runtime) {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.gpu_resident_source.read",
            "GPU-resident source runtime ownership does not match the execution context",
        )
        .with_fact("input_name", source.input_name.as_str())
        .with_fact("source_device", handle.runtime().device().ordinal())
        .with_fact("context_device", context.runtime()?.device().ordinal()));
    }

    if handle.device_placement() != context.device_placement()
        || handle.placement().device != context.runtime()?.device()
    {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.gpu_resident_source.read",
            "GPU-resident source device placement does not match the execution context",
        )
        .with_fact("input_name", source.input_name.as_str())
        .with_fact("source_device", handle.placement().device.ordinal())
        .with_fact("context_device", context.runtime()?.device().ordinal()));
    }

    metrics.boundary.record_gpu_resident_input_handoff();
    Ok(handle.clone().into_gpu_frame())
}
