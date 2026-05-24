use super::*;

/// Read a caller-supplied GPU-resident source by plan-local input name.
///
/// The handle must match the declared source schema and device placement. A
/// handle from another exact runtime binding is relocated before it reaches an
/// operator. In particular, a source from another admitted attempt is copied
/// into the consuming attempt's immutable allocation domain rather than being
/// accepted merely because the two runtimes share device affinity.
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
    // A different runtime affinity is expected for a separately admitted
    // attempt on the same device. It is a relocation boundary, not a reason
    // to reject the input. Cross-device inputs still cannot be repaired by a
    // local device-to-device copy.
    if handle.runtime().device() != context_runtime.device() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.gpu_resident_source.read",
            "GPU-resident source runtime device does not match the execution context",
        )
        .with_fact("input_name", source.input_name.as_str())
        .with_fact("source_device", handle.runtime().device().ordinal())
        .with_fact("context_device", context_runtime.device().ordinal()));
    }

    if handle.device_placement() != context.device_placement() || handle.placement().device != context_runtime.device()
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

    metrics.record_input_batch(handle.num_rows() as u64);

    // Device affinity permits transfer but does not establish allocation ownership.
    let same_context_binding = handle.runtime().binding_id() == context_runtime.binding_id();
    if same_context_binding {
        metrics.boundary.record_gpu_resident_input_handoff();
        return Ok(handle.clone().into_gpu_frame());
    }

    let target_runtime = if context.attempt_allocation_domain_id().is_some() {
        std::sync::Arc::new(attempt_domain_handoff_runtime(context, &context_runtime, source.input_name.as_str())?)
    } else {
        std::sync::Arc::clone(&context_runtime)
    };
    let frame = handle.frame().copy_into_runtime(target_runtime.as_ref()).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.gpu_resident_source.read",
            "failed to relocate GPU-resident source into the consuming execution domain",
            error,
        )
        .with_fact("input_name", source.input_name.as_str())
        .with_fact("source_binding_id", handle.runtime().binding_id())
        .with_fact("target_binding_id", target_runtime.binding_id())
        .with_fact("attempt_allocation_domain_id", context.attempt_allocation_domain_id().unwrap_or_default())
    })?;
    let frame = GpuDataFrame::new(frame);
    let copied_bytes = frame.device_allocated_size_bytes().map_err(|error| {
        error
            .with_fact("input_name", source.input_name.as_str())
            .with_fact("source_binding_id", handle.runtime().binding_id())
            .with_fact("target_binding_id", target_runtime.binding_id())
    })?;
    metrics.boundary.record_gpu_resident_input_handoff();
    metrics.boundary.record_device_to_device_copy(copied_bytes);
    Ok(frame)
}

fn attempt_domain_handoff_runtime(
    context: &ExecutionContext,
    context_runtime: &cudf::Runtime,
    input_name: &str,
) -> Result<cudf::Runtime> {
    let operation = "nexus_query_engine.gpu_resident_source.read";
    let allocation_domain_id = context
        .attempt_allocation_domain_id()
        .expect("attempt-domain handoff only runs for an admitted allocation domain");
    let token = context
        .execution_budget_token(
            cudf::ExecutionBudgetRequest::new(usize::MAX).with_boundary("gpu_resident_source_handoff"),
        )
        .map_err(|error| {
            error
                .with_fact_if_absent("input_name", input_name)
                .with_fact_if_absent("attempt_allocation_domain_id", allocation_domain_id)
                .with_fact_if_absent("budget_boundary", "gpu_resident_source_handoff")
        })?;
    cudf::Runtime::builder()
        .with_device(context_runtime.device())
        .with_stream(token.stream().clone())
        .with_memory_resource(token.memory_resource().clone())
        .build()
        .map_err(|error| {
            QueryError::from_cudf(
                operation,
                "failed to build an attempt-domain runtime for GPU-resident source handoff",
                error,
            )
            .with_fact("input_name", input_name)
            .with_fact("attempt_allocation_domain_id", allocation_domain_id)
            .with_fact("budget_boundary", "gpu_resident_source_handoff")
        })?
        .inherit_retained_lifetime_from(context_runtime)
        .map_err(|error| {
            QueryError::from_cudf(
                operation,
                "failed to retain the consuming attempt while relocating a GPU-resident source",
                error,
            )
            .with_fact("input_name", input_name)
            .with_fact("attempt_allocation_domain_id", allocation_domain_id)
            .with_fact("budget_boundary", "gpu_resident_source_handoff")
        })
}
