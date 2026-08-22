use nexus_query_engine::admission::QueryPlanCapabilityProof;

use super::*;
use crate::{
    attempt::NexusAttemptExtension,
    native_exec::{
        NexusNativeComposedExec, NexusNativeComposedGpuProducerExec, admission::QueryAdmissionTicket,
        composed_admission_capability_proof,
    },
};

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
    if let Some(native) = plan.downcast_ref::<NexusNativeComposedGpuProducerExec>() {
        return native.execution_config().clone();
    }
    default_execution_config.clone()
}

pub(super) async fn exact_composed_edge_source_gpu_plan(
    edge_source: &dyn ExecutionPlan,
    task_context: &TaskContext,
) -> Result<Option<nexus_query_engine::plan::QueryPlan>> {
    let Some(composed) = as_nexus_native_composed_exec(edge_source) else {
        return Ok(None);
    };
    let plan = crate::native_exec::lakehouse::resolve_query_plan(
        composed.execution_config(),
        composed.native_plan().clone(),
        task_context,
    )
    .await;
    Ok(Some(plan.clone_with_gpu_resident_sink()))
}

pub(super) fn cugraph_execution_config_after_staging(
    spec: &CugraphAlgorithmSpec,
    edge_source: &dyn ExecutionPlan,
    predicate_sources: &[Arc<dyn ExecutionPlan>],
    input_row_bounds: &CugraphInputRowBounds,
    result_options: &CugraphResultPlanOptions,
    edge_normalize_inserted: bool,
    default_execution_config: &NexusNativeExecutionConfig,
    task_context: &TaskContext,
    exact_edge_source_gpu_plan: Option<&nexus_query_engine::plan::QueryPlan>,
) -> Result<(NexusNativeExecutionConfig, nexus_query_engine::plan::QueryPlan)> {
    let execution_config =
        edge_execution_config(edge_source, default_execution_config).without_query_admission_ticket();
    // The nested state that lowers a cuGraph edge relation deliberately has
    // no admission authority: the graph root must be the only admission
    // owner once footer metadata is resolved.  Its lowered child still owns
    // execution policy, but restore the outer root's source here so this
    // post-hydration boundary can mint that one ticket.
    let execution_config = if execution_config.has_query_admission_authority() {
        execution_config
    } else if let Some(source) = default_execution_config.query_admission_source() {
        execution_config.with_query_admission_source(source)
    } else {
        execution_config
    };
    let (execution_config, graph_execution_plan) = cugraph_execution_config_with_graph_aware_ticket(
        spec,
        edge_source,
        predicate_sources,
        input_row_bounds,
        result_options,
        edge_normalize_inserted,
        execution_config,
        exact_edge_source_gpu_plan,
    )?;
    Ok((bind_cugraph_ticket_to_attempt(execution_config, task_context)?, graph_execution_plan))
}

fn bind_cugraph_ticket_to_attempt(
    execution_config: NexusNativeExecutionConfig,
    task_context: &TaskContext,
) -> Result<NexusNativeExecutionConfig> {
    let Some(attempt) = task_context.session_config().options().extensions.get::<NexusAttemptExtension>() else {
        return Ok(execution_config);
    };
    let Some(ticket) = execution_config.query_admission_ticket() else {
        return Ok(execution_config);
    };
    let ticket = QueryAdmissionTicket::new_attempt_bound(
        ticket.source(),
        ticket.attempt_capability_proof_for_rebind(),
        attempt.ticket_binding(),
        ticket.metrics_capture(),
    );
    attempt.bind_ticket(Arc::clone(&ticket));
    Ok(execution_config.with_existing_query_admission_ticket(ticket))
}

pub(super) fn cugraph_execution_config_with_graph_aware_ticket(
    spec: &CugraphAlgorithmSpec,
    edge_source: &dyn ExecutionPlan,
    predicate_sources: &[Arc<dyn ExecutionPlan>],
    input_row_bounds: &CugraphInputRowBounds,
    result_options: &CugraphResultPlanOptions,
    edge_normalize_inserted: bool,
    execution_config: NexusNativeExecutionConfig,
    exact_edge_source_gpu_plan: Option<&nexus_query_engine::plan::QueryPlan>,
) -> Result<(NexusNativeExecutionConfig, nexus_query_engine::plan::QueryPlan)> {
    let edge_row_upper_bound = if execution_config.query_admission_source().is_none() {
        None
    } else {
        input_row_bounds.edge_row_upper_bound()
    };
    let edge_vertex_upper_bound = edge_source_graph_vertex_estimate(edge_source).map(|estimate| estimate.count);
    let (graph_execution_plan, graph_execution_proof) = cugraph_execution_capability_proof_with_edge_row_upper_bound(
        spec,
        edge_source.schema(),
        predicate_sources.iter().map(|source| source.schema()).collect(),
        input_row_bounds.predicate_row_upper_bounds(),
        result_options,
        edge_normalize_inserted,
        edge_row_upper_bound,
        edge_vertex_upper_bound,
    )?;
    let graph_model_ask = if execution_config.query_admission_source().is_none() {
        None
    } else {
        cugraph_graph_model_ask_bytes(&graph_execution_plan, edge_source, &execution_config)?
    };
    // A cuGraph attempt executes its graph operation and may also execute
    // lowered native edge/predicate fragments under the same ticket. Keep one
    // aggregate proof for that whole invocation so the engine can verify each
    // fragment at execution time without treating the graph proof as blanket
    // authorization for a different native plan.
    let mut fragment_proofs = vec![graph_execution_proof];
    if let Some(composed_plan) = exact_edge_source_gpu_plan {
        let composed = as_nexus_native_composed_exec(edge_source).ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_nexus.cugraph_sql.graph_ticket_ask",
                "the staged composed edge plan does not match its execution source",
            )
            .with_fact("reason", "composed_edge_source_plan_missing")
        })?;
        let composed_proof =
            composed_admission_capability_proof(composed_plan, composed.gpu_inputs()).map_err(|error| {
                Error::new(
                    ErrorCode::InvalidExecutionPlan,
                    "datafusion_nexus.cugraph_sql.graph_ticket_ask",
                    "failed to derive the complete composed edge-source admission proof",
                )
                .with_source(error)
            })?;
        fragment_proofs.push(composed_proof);
    } else {
        collect_native_fragment_capability_proofs(edge_source, &mut fragment_proofs, true, false)?;
    }
    for predicate_source in predicate_sources {
        collect_native_fragment_capability_proofs(predicate_source.as_ref(), &mut fragment_proofs, false, true)?;
    }
    let execution_config = execution_config
        .with_graph_aware_plan_query_admission_ticket(graph_model_ask, fragment_proofs)
        .map_err(|error| {
            DataFusionError::from(Error::from_native(
                "datafusion_nexus.cugraph_sql.graph_ticket_ask",
                "failed to compute cuGraph SQL graph-aware query admission ticket ask",
                error,
            ))
        })?;
    Ok((execution_config, graph_execution_plan))
}

pub(super) fn validate_cugraph_graph_owner_preflight(
    spec: &CugraphAlgorithmSpec,
    edge_source: &dyn ExecutionPlan,
    predicate_sources: &[Arc<dyn ExecutionPlan>],
    input_row_bounds: &CugraphInputRowBounds,
    result_options: &CugraphResultPlanOptions,
    edge_normalize_inserted: bool,
    execution_config: &NexusNativeExecutionConfig,
) -> Result<()> {
    if execution_config.query_admission_source().is_none() {
        return Ok(());
    }
    let edge_vertex_upper_bound = edge_source_graph_vertex_estimate(edge_source).map(|estimate| estimate.count);
    let (graph_execution_plan, _) = cugraph_execution_capability_proof_with_edge_row_upper_bound(
        spec,
        edge_source.schema(),
        predicate_sources.iter().map(|source| source.schema()).collect(),
        input_row_bounds.predicate_row_upper_bounds(),
        result_options,
        edge_normalize_inserted,
        input_row_bounds.edge_row_upper_bound(),
        edge_vertex_upper_bound,
    )?;
    let _ = cugraph_graph_model_ask_bytes(&graph_execution_plan, edge_source, execution_config)?;
    Ok(())
}

fn collect_native_fragment_capability_proofs(
    plan: &dyn ExecutionPlan,
    proofs: &mut Vec<QueryPlanCapabilityProof>,
    is_graph_input_fragment: bool,
    is_cugraph_side_input_fragment: bool,
) -> Result<()> {
    let native_plan = lowered_native_edge_plan(plan);

    // DataFusion's scheduling-only CooperativeExec is transparently unwrapped
    // by `lowered_native_edge_plan` and by the execution path. Descending
    // through it would issue a second proof for the same native edge frame:
    // only the outer proof is the graph input's exact owner. Other wrappers
    // retain their children because composed GPU producers execute those
    // distinct fragments themselves.
    if !(plan.name() == "CooperativeExec" && native_plan.is_some()) {
        for child in plan.children() {
            collect_native_fragment_capability_proofs(child.as_ref(), proofs, false, false)?;
        }
    }

    // `execute_edge_source_to_gpu` converts every native edge fragment to a
    // GPU-resident sink before submitting it to the engine. Derive the proof
    // from that exact execution shape: a host-sink proof is intentionally not
    // broad enough to authorize the additional GPU-output contract.
    if let Some(native_plan) = native_plan {
        let native_plan = native_plan.clone_with_gpu_resident_sink();
        let proof = if is_graph_input_fragment {
            QueryPlanCapabilityProof::for_cugraph_graph_input_plan(&native_plan)
        } else if is_cugraph_side_input_fragment {
            Some(QueryPlanCapabilityProof::try_for_cugraph_side_input_plan(&native_plan).map_err(|error| {
                DataFusionError::from(Error::from_native(
                    "datafusion_nexus.cugraph_sql.graph_ticket_ask",
                    "failed to derive native cuGraph side-input admission evidence",
                    error,
                ))
            })?)
        } else {
            QueryPlanCapabilityProof::for_query_plan(&native_plan)
        };
        if let Some(proof) = proof {
            proofs.push(proof);
        }
    }
    Ok(())
}

/// Returns the lowered plan carried by every native edge-wrapper shape.  Keep
/// graph admission's row, vertex, capability, and native-boundary checks on
/// this one wrapper set so a new producer form cannot silently lose pricing
/// evidence while retaining graph execution authority.
pub(super) fn lowered_native_edge_plan(plan: &dyn ExecutionPlan) -> Option<&nexus_query_engine::plan::QueryPlan> {
    if let Some(native) = as_nexus_native_exec(plan) {
        return Some(native.native_plan());
    }
    if let Some(native) = as_nexus_native_composed_exec(plan) {
        return Some(native.native_plan());
    }
    if let Some(native) = plan.downcast_ref::<NexusNativeGpuProducerExec>() {
        return Some(native.native_plan());
    }
    plan.downcast_ref::<NexusNativeComposedGpuProducerExec>().map(NexusNativeComposedGpuProducerExec::native_plan)
}

pub(super) fn cugraph_graph_model_ask_bytes(
    graph_plan: &nexus_query_engine::plan::QueryPlan,
    edge_source: &dyn ExecutionPlan,
    execution_config: &NexusNativeExecutionConfig,
) -> Result<Option<GraphAdmissionModelAsk>> {
    if !native_edge_source_allows_graph_aware_ticket(edge_source) {
        return Ok(None);
    }
    let device_total_memory_bytes =
        execution_config.query_admission_source().map(|source| source.max_device_total_memory_bytes());
    let evidence = nexus_query_engine::graph_memory::graph_memory_bounded_admission_evidence_for_plan(
        graph_plan,
        GraphMemoryCalibrationOverride::none(),
        device_total_memory_bytes,
    )
    .map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.graph_ticket_ask",
            "failed to derive cuGraph SQL graph admission evidence from planning metadata",
            error,
        ))
    })?;
    Ok(Some(GraphAdmissionModelAsk::from_graph_memory_admission_evidence(evidence)))
}

pub(super) fn edge_source_output_rows(edge_source: &dyn ExecutionPlan) -> Result<Option<u64>> {
    if let Some(source) = edge_source.downcast_ref::<datafusion::datasource::source::DataSourceExec>()
        && let Some(rows) = crate::native::exact_row_count_for_datafusion_local_parquet_scan(source)
    {
        return Ok(Some(rows));
    }
    if let Some(rows) = native_edge_source_output_rows(edge_source) {
        return Ok(Some(rows));
    }
    exact_execution_plan_output_rows(edge_source)
}

/// Returns cardinality evidence that is exact in the physical DataFusion
/// plan. Inexact optimizer estimates must never be used for strict bounded
/// graph admission.
pub(super) fn exact_execution_plan_output_rows(plan: &dyn ExecutionPlan) -> Result<Option<u64>> {
    let statistics_context = StatisticsContext::new();
    let partition_count = plan.properties().output_partitioning().partition_count();
    if partition_count <= 1 {
        let statistics = statistics_context.compute(plan, &StatisticsArgs::new())?;
        if let Precision::Exact(rows) = statistics.num_rows {
            return Ok(Some(rows as u64));
        }
    }

    // Do not treat the plan-level `Precision::Exact` field as authoritative
    // for a parallel edge scan. DataFusion may populate it from one physical
    // fragment; a strict graph ticket needs either all partition facts or
    // stronger native/footer evidence selected above.
    let mut total = 0_u64;
    for partition in 0..partition_count {
        let statistics = statistics_context.compute(plan, &StatisticsArgs::new().with_partition(Some(partition)))?;
        let Precision::Exact(rows) = statistics.num_rows else {
            return Ok(native_query_plan_hard_row_upper_bound(plan));
        };
        let rows = rows as u64;
        let Some(next_total) = total.checked_add(rows) else {
            return Ok(native_query_plan_hard_row_upper_bound(plan));
        };
        total = next_total;
    }
    Ok(Some(total))
}

fn native_query_plan_hard_row_upper_bound(plan: &dyn ExecutionPlan) -> Option<u64> {
    lowered_native_edge_plan(plan).and_then(nexus_query_engine::graph_memory::query_plan_hard_row_upper_bound)
}

pub(super) fn native_edge_source_output_rows(edge_source: &dyn ExecutionPlan) -> Option<u64> {
    if !native_edge_source_allows_graph_aware_ticket(edge_source) {
        return None;
    }
    native_query_plan_hard_row_upper_bound(edge_source)
}

pub(super) fn native_edge_source_allows_graph_aware_ticket(edge_source: &dyn ExecutionPlan) -> bool {
    lowered_native_edge_plan(edge_source).is_none_or(native_plan_is_graph_ticket_light_edge_source)
}

pub(super) fn edge_source_graph_vertex_estimate(edge_source: &dyn ExecutionPlan) -> Option<GraphVertexEstimate> {
    if let Some(source) = edge_source.downcast_ref::<datafusion::datasource::source::DataSourceExec>() {
        return crate::native::graph_vertex_upper_bound_for_datafusion_local_parquet_scan(source)
            .map(GraphVertexEstimate::graph_metadata);
    }
    lowered_native_edge_plan(edge_source)
        .and_then(query_plan_graph_vertex_upper_bound)
        .map(GraphVertexEstimate::graph_metadata)
}

pub(super) fn native_plan_is_graph_ticket_light_edge_source(native_plan: &nexus_query_engine::plan::QueryPlan) -> bool {
    native_plan.nodes().iter().all(|node| {
        matches!(node.kind, PlanNode::Source(_) | PlanNode::Filter(_) | PlanNode::Projection(_) | PlanNode::Limit(_))
    })
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
    if config.has_query_admission_authority() {
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
        || plan.downcast_ref::<NexusNativeComposedGpuProducerExec>().is_some()
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
