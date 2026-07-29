use super::*;
pub(super) fn edge_execution_config(
    plan: &dyn ExecutionPlan,
    default_execution_config: &NexusNativeExecutionConfig,
) -> NexusNativeExecutionConfig {
    if let Some(native) = as_nexus_native_exec(plan) {
        return native.execution_config().clone();
    }
    if let Some(native) = as_nexus_native_composed_exec(plan) {
        return native.execution_config().clone();
    }
    default_execution_config.clone()
}

#[cfg(test)]
pub(super) fn cugraph_execution_config(
    plan: &dyn ExecutionPlan,
    default_execution_config: &NexusNativeExecutionConfig,
) -> NexusNativeExecutionConfig {
    let execution_config = edge_execution_config(plan, default_execution_config);
    if let Some(ticket) = default_execution_config.query_admission_ticket() {
        execution_config.with_existing_query_admission_ticket(ticket)
    } else {
        execution_config.with_plan_query_admission_ticket()
    }
}

pub(super) fn cugraph_execution_config_for_task_context(
    plan: &dyn ExecutionPlan,
    default_execution_config: &NexusNativeExecutionConfig,
    task_context: &TaskContext,
) -> NexusNativeExecutionConfig {
    let execution_config = edge_execution_config(plan, default_execution_config);
    if let Some(ticket) = query_admission_ticket_from_task_context(task_context) {
        execution_config.with_existing_query_admission_ticket(ticket)
    } else if let Some(ticket) = default_execution_config.query_admission_ticket() {
        execution_config.with_existing_query_admission_ticket(ticket)
    } else {
        execution_config.with_plan_query_admission_ticket()
    }
}

pub(super) fn cugraph_execution_config_with_graph_aware_ticket(
    spec: &CugraphAlgorithmSpec,
    edge_source: &dyn ExecutionPlan,
    edge_normalize_inserted: bool,
    placement_hint: QueryPlacementHint,
    execution_config: NexusNativeExecutionConfig,
) -> Result<NexusNativeExecutionConfig> {
    let graph_model_ask =
        if execution_config.requested_grant_bytes().is_some() || execution_config.query_admission_source().is_none() {
            None
        } else {
            cugraph_graph_model_ask_bytes(spec, edge_source, edge_normalize_inserted, &execution_config)?
        };
    execution_config.with_graph_aware_plan_query_admission_ticket(graph_model_ask, placement_hint).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.graph_ticket_ask",
            "failed to compute cuGraph SQL graph-aware query admission ticket ask",
            error,
        ))
    })
}

pub(super) fn cugraph_graph_placement_hint(
    spec: &CugraphAlgorithmSpec,
    graph_residency: Option<&GraphResidencyDescriptor>,
    edge_normalize_inserted: bool,
    predicate_sources: &[Arc<dyn ExecutionPlan>],
) -> QueryPlacementHint {
    let Some(graph_residency) = graph_residency else {
        return QueryPlacementHint::default();
    };
    let native_spec = native_graph_spec_for_placement_hint(spec, edge_normalize_inserted, predicate_sources);
    planned_resident_graph_key(graph_residency, &native_spec)
        .map(QueryPlacementHint::for_resident_graph_key)
        .unwrap_or_default()
}

pub(super) fn cugraph_graph_model_ask_bytes(
    spec: &CugraphAlgorithmSpec,
    edge_source: &dyn ExecutionPlan,
    edge_normalize_inserted: bool,
    execution_config: &NexusNativeExecutionConfig,
) -> Result<Option<GraphAdmissionModelAsk>> {
    let Some(native_spec) = native_graph_spec_for_memory_estimate(spec, edge_normalize_inserted) else {
        return Ok(None);
    };
    if !native_edge_source_allows_graph_aware_ticket(edge_source) {
        return Ok(None);
    }
    let calibration_override = execution_config.graph_memory_calibration_override();
    let raw_input_edges = edge_source_output_rows(edge_source)?.or(calibration_override.raw_input_edges);
    let Some(raw_input_edges) = raw_input_edges else {
        return Ok(None);
    };
    let input_edges = cugraph_effective_input_edges(raw_input_edges, spec, edge_normalize_inserted);
    let device_total_memory_bytes =
        execution_config.query_admission_source().map(|source| source.max_device_total_memory_bytes());
    let estimate = graph_memory_estimate_for_edge_list(
        &native_spec,
        &edge_source.schema(),
        input_edges,
        Some(raw_input_edges),
        calibration_override,
        device_total_memory_bytes,
    )
    .map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.graph_ticket_ask",
            "failed to estimate cuGraph SQL graph memory from planning metadata",
            error,
        ))
    })?;
    Ok(Some(GraphAdmissionModelAsk::new(estimate.requested_bytes, estimate.execution_bytes)))
}

pub(super) fn edge_source_output_rows(edge_source: &dyn ExecutionPlan) -> Result<Option<u64>> {
    if let Some(rows) = exact_execution_plan_output_rows(edge_source)? {
        return Ok(Some(rows));
    }
    Ok(native_edge_source_output_rows(edge_source))
}

/// Returns cardinality evidence that is exact in the physical DataFusion
/// plan. Inexact optimizer estimates must never be used for strict bounded
/// graph admission.
pub(super) fn exact_execution_plan_output_rows(plan: &dyn ExecutionPlan) -> Result<Option<u64>> {
    let statistics = plan.partition_statistics(None)?;
    if let Precision::Exact(rows) = statistics.num_rows {
        return Ok(u64::try_from(rows).ok());
    }
    let mut total = 0_u64;
    for partition in 0..plan.properties().output_partitioning().partition_count() {
        let statistics = plan.partition_statistics(Some(partition))?;
        let Precision::Exact(rows) = statistics.num_rows else {
            return Ok(native_query_plan_hard_row_upper_bound(plan));
        };
        let Some(rows) = u64::try_from(rows).ok() else {
            return Ok(native_query_plan_hard_row_upper_bound(plan));
        };
        let Some(next_total) = total.checked_add(rows) else {
            return Ok(native_query_plan_hard_row_upper_bound(plan));
        };
        total = next_total;
    }
    Ok(Some(total))
}

fn native_query_plan_hard_row_upper_bound(plan: &dyn ExecutionPlan) -> Option<u64> {
    if let Some(native) = as_nexus_native_exec(plan) {
        return nexus_query_engine::graph_memory::query_plan_hard_row_upper_bound(native.native_plan());
    }
    if let Some(native) = as_nexus_native_composed_exec(plan) {
        return nexus_query_engine::graph_memory::query_plan_hard_row_upper_bound(native.native_plan());
    }
    if let Some(native) = plan.downcast_ref::<NexusNativeGpuProducerExec>() {
        return nexus_query_engine::graph_memory::query_plan_hard_row_upper_bound(native.native_plan());
    }
    None
}

pub(super) fn native_edge_source_output_rows(edge_source: &dyn ExecutionPlan) -> Option<u64> {
    if !native_edge_source_allows_graph_aware_ticket(edge_source) {
        return None;
    }
    graph_edge_source_output_rows(edge_source)
}

pub(super) fn native_edge_source_allows_graph_aware_ticket(edge_source: &dyn ExecutionPlan) -> bool {
    if let Some(native) = as_nexus_native_exec(edge_source) {
        return native_plan_is_graph_ticket_light_edge_source(native.native_plan());
    }
    if let Some(native) = as_nexus_native_composed_exec(edge_source) {
        return native_plan_is_graph_ticket_light_edge_source(native.native_plan());
    }
    if let Some(native) = edge_source.downcast_ref::<NexusNativeGpuProducerExec>() {
        return native_plan_is_graph_ticket_light_edge_source(native.native_plan());
    }
    true
}

pub(super) fn graph_edge_source_output_rows(edge_source: &dyn ExecutionPlan) -> Option<u64> {
    if let Some(native) = as_nexus_native_exec(edge_source) {
        return graph_edge_list_output_rows_for_plan(native.native_plan());
    }
    if let Some(native) = as_nexus_native_composed_exec(edge_source) {
        return graph_edge_list_output_rows_for_plan(native.native_plan());
    }
    if let Some(native) = edge_source.downcast_ref::<NexusNativeGpuProducerExec>() {
        return graph_edge_list_output_rows_for_plan(native.native_plan());
    }
    None
}

pub(super) fn native_plan_is_graph_ticket_light_edge_source(native_plan: &nexus_query_engine::plan::QueryPlan) -> bool {
    native_plan.nodes().iter().all(|node| {
        matches!(node.kind, PlanNode::Source(_) | PlanNode::Filter(_) | PlanNode::Projection(_) | PlanNode::Limit(_))
    })
}

pub(super) fn cugraph_effective_input_edges(
    raw_input_edges: u64,
    spec: &CugraphAlgorithmSpec,
    edge_normalize_inserted: bool,
) -> u64 {
    raw_input_edges.saturating_mul(if edge_normalize_inserted && !spec.graph_options.directed { 2 } else { 1 })
}

pub(super) fn task_context_with_cugraph_execution_ticket(
    execution_config: &NexusNativeExecutionConfig,
    task_context: Arc<TaskContext>,
) -> Arc<TaskContext> {
    if let Some(ticket) = execution_config.query_admission_ticket() {
        task_context_with_query_admission_ticket(task_context, ticket)
    } else {
        task_context
    }
}

pub(super) fn validate_standalone_bounded_cugraph_sources(
    config: &NexusNativeExecutionConfig,
    edge_source: &dyn ExecutionPlan,
    predicate_sources: &[Arc<dyn ExecutionPlan>],
) -> Result<()> {
    if !config.is_bounded_without_admission() {
        return Ok(());
    }
    validate_standalone_bounded_source_contains_no_native_child("edge", edge_source)?;
    for predicate in predicate_sources {
        validate_standalone_bounded_source_contains_no_native_child("predicate", predicate.as_ref())?;
    }
    Ok(())
}

pub(super) fn validate_standalone_bounded_source_contains_no_native_child(
    source_kind: &'static str,
    plan: &dyn ExecutionPlan,
) -> Result<()> {
    if plan_contains_nexus_native(plan) {
        return Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "datafusion_nexus.cugraph_sql.admission",
            "cuGraph execution with native child plans requires an installed NexusGpuBackend",
        )
        .with_log_safe_fact("reason", "gpu_backend_required")
        .with_fact("source_kind", source_kind)
        .with_log_safe_fact("remediation", "install_nexus_gpu_backend")
        .into());
    }
    Ok(())
}

pub(super) fn plan_contains_nexus_native(plan: &dyn ExecutionPlan) -> bool {
    plan.downcast_ref::<NexusNativeExec>().is_some()
        || plan.downcast_ref::<NexusNativeComposedExec>().is_some()
        || plan.downcast_ref::<NexusNativeGpuProducerExec>().is_some()
        || plan.downcast_ref::<CugraphAlgoExec>().is_some()
        || plan.children().iter().any(|child| plan_contains_nexus_native(child.as_ref()))
}

pub(super) fn as_nexus_native_exec(plan: &dyn ExecutionPlan) -> Option<&NexusNativeExec> {
    if let Some(native) = plan.downcast_ref::<NexusNativeExec>() {
        return Some(native);
    }
    let children = cooperative_exec_children(plan)?;
    let [child] = children.as_slice() else {
        return None;
    };
    child.downcast_ref::<NexusNativeExec>()
}

pub(super) fn as_nexus_native_composed_exec(plan: &dyn ExecutionPlan) -> Option<&NexusNativeComposedExec> {
    if let Some(native) = plan.downcast_ref::<NexusNativeComposedExec>() {
        return Some(native);
    }
    let children = cooperative_exec_children(plan)?;
    let [child] = children.as_slice() else {
        return None;
    };
    child.downcast_ref::<NexusNativeComposedExec>()
}

pub(super) fn cooperative_exec_children(plan: &dyn ExecutionPlan) -> Option<Vec<&Arc<dyn ExecutionPlan>>> {
    (plan.name() == "CooperativeExec").then(|| plan.children())
}
