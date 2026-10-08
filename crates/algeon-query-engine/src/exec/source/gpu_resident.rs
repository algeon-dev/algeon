use query_planning::source::GpuResidentSource;

use super::{
    ExecutionContext, ExecutionMetrics, GpuDataFrame, GpuResidentInputMap, QueryError, QueryErrorCode, Result,
};

/// Read a caller-supplied GPU-resident source by plan-local input name.
///
/// The handle must match the declared source schema and device placement. A
/// handle from another admitted attempt on the same device is borrowed
/// directly. Its resident ownership keeps the producer allocation alive, so
/// the consumer neither copies nor charges those bytes to its own cap.
///
/// Fails with `InvalidPlan` when the input is missing, the schema differs from the declared one
/// (compared as a whole, including nullability), or the handle belongs to another device. All
/// checks run before any metric is recorded, so a rejected handle leaves no input facts. The
/// returned frame is ordered on the consumer's stream.
pub(in crate::exec) fn read_gpu_resident_source(
    context: &ExecutionContext,
    source: &GpuResidentSource,
    inputs: &GpuResidentInputMap,
    metrics: &ExecutionMetrics,
) -> Result<GpuDataFrame> {
    let handle = inputs.get(&source.input_name).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.gpu_resident_source.read",
            "GPU-resident source input was not provided to native execution",
        )
        .with_fact("input_name", source.input_name.as_str())
    })?;

    if handle.schema().as_ref() != source.schema.as_ref() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.gpu_resident_source.read",
            "GPU-resident source schema does not match the declared source schema",
        )
        .with_fact("input_name", source.input_name.as_str())
        .with_fact("declared_fields", source.schema.fields().len())
        .with_fact("actual_fields", handle.schema().fields().len()));
    }

    let context_runtime = context.shared_runtime()?;
    // Runtime bindings may differ across admitted attempts. Device affinity
    // remains mandatory because resident ownership cannot cross devices.
    if handle.runtime().device() != context_runtime.device() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.gpu_resident_source.read",
            "GPU-resident source runtime device does not match the execution context",
        )
        .with_fact("input_name", source.input_name.as_str())
        .with_fact("source_device", handle.runtime().device().ordinal())
        .with_fact("context_device", context_runtime.device().ordinal()));
    }

    if handle.placement().device != context_runtime.device() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.gpu_resident_source.read",
            "GPU-resident source device placement does not match the execution context",
        )
        .with_fact("input_name", source.input_name.as_str())
        .with_fact("source_device", handle.placement().device.ordinal())
        .with_fact("context_device", context_runtime.device().ordinal()));
    }

    metrics.record_input_batch(handle.num_rows() as u64);

    metrics.record_gpu_resident_input_handoff();
    handle.read_on(context)
}
