use std::sync::Arc;

use datafusion::datasource::source::DataSourceExec;
use query_engine::{
    admission::{AdmittedPlan, AttemptCapabilityEvidence},
    exec::admit_query_plan,
    plan::{QueryPlan, row_bounds::query_plan_hard_row_upper_bound},
};

use super::{
    AlgeonNativeExec, AlgeonNativeExecutionConfig, AlgeonNativeGpuProducerExec, CooperativeExec, CugraphAlgoExec,
    CugraphAlgorithmSpec, CugraphInputRowBounds, CugraphPlanDefinition, CugraphResultPlanOptions, DataFusionError,
    Error, ErrorCode, ExecutionPlan, PlanNode, Precision, RepartitionExec, Result, StatisticsArgs, StatisticsContext,
    TaskContext, cugraph_execution_capability_evidence, gpu_resident_producer_hard_output_row_upper_bound,
    task_context_with_query_admission_ticket,
};
use crate::{
    native::{
        exact_row_count_for_datafusion_local_parquet_scan, graph_vertex_upper_bound_for_datafusion_local_parquet_scan,
    },
    native_exec::{
        AlgeonNativeComposedExec, AlgeonNativeComposedGpuProducerExec,
        admission::{attempt_execution_required_error, invocation_admission_ticket},
        composed_admission_capability_evidence, gpu_resident_producer_input_name,
        lakehouse::resolve_query_plan,
    },
    planner::native::Invocation,
};

pub(super) fn edge_execution_config(
    plan: &dyn ExecutionPlan,
    default_execution_config: &AlgeonNativeExecutionConfig,
) -> AlgeonNativeExecutionConfig {
    if let Some(native) = as_algeon_native_exec(plan) {
        return native.execution_config().clone();
    }
    if let Some(native) = as_algeon_native_composed_exec(plan) {
        return native.execution_config().clone();
    }
    if let Some(native) = plan.downcast_ref::<AlgeonNativeComposedGpuProducerExec>() {
        return native.execution_config().clone();
    }
    default_execution_config.clone()
}

pub(super) async fn exact_native_edge_source_gpu_plan(
    edge_source: &dyn ExecutionPlan,
    task_context: &TaskContext,
) -> Result<Option<Arc<AdmittedPlan>>> {
    let (execution_config, native_plan) = if let Some(native) = as_algeon_native_exec(edge_source) {
        (native.execution_config(), native.native_plan())
    } else if let Some(composed) = as_algeon_native_composed_exec(edge_source) {
        (composed.execution_config(), composed.native_plan())
    } else {
        return Ok(None);
    };
    admit_native_edge_source_gpu_plan(execution_config, native_plan, task_context).await.map(Some)
}

/// Resolves a native edge source into the GPU-resident form cuGraph reads and
/// admits that form once; its node was admitted in host form.
pub(super) async fn admit_native_edge_source_gpu_plan(
    execution_config: &AlgeonNativeExecutionConfig,
    native_plan: &QueryPlan,
    task_context: &TaskContext,
) -> Result<Arc<AdmittedPlan>> {
    let plan =
        resolve_query_plan(execution_config, native_plan.clone(), task_context).await.clone_with_gpu_resident_sink();
    admit_query_plan(&plan, execution_config.statistical_aggregate_execution_mode(), execution_config.planning_policy())
        .map_err(|error| {
            DataFusionError::from(Error::from_native(
                "algeon_datafusion.cugraph_sql.edge_source",
                "the GPU-resident cuGraph edge source is not admitted",
                error,
            ))
        })
}

#[expect(clippy::too_many_arguments, reason = "execution inputs are distinct domain values that share no owner")]
pub(super) fn cugraph_execution_config_after_staging(
    spec: &CugraphAlgorithmSpec,
    edge_source: &dyn ExecutionPlan,
    predicate_sources: &[Arc<dyn ExecutionPlan>],
    input_row_bounds: &CugraphInputRowBounds,
    result_options: &CugraphResultPlanOptions,
    edge_normalize_inserted: bool,
    default_execution_config: &AlgeonNativeExecutionConfig,
    task_context: &TaskContext,
    exact_edge_source_gpu_plan: Option<&QueryPlan>,
) -> Result<(AlgeonNativeExecutionConfig, QueryPlan)> {
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
    compile_cugraph_admission_ticket(
        spec,
        edge_source,
        predicate_sources,
        input_row_bounds,
        result_options,
        edge_normalize_inserted,
        execution_config,
        exact_edge_source_gpu_plan,
        task_context,
    )
}

#[expect(clippy::too_many_arguments, reason = "execution inputs are distinct domain values that share no owner")]
fn compile_cugraph_admission_ticket(
    spec: &CugraphAlgorithmSpec,
    edge_source: &dyn ExecutionPlan,
    predicate_sources: &[Arc<dyn ExecutionPlan>],
    input_row_bounds: &CugraphInputRowBounds,
    result_options: &CugraphResultPlanOptions,
    edge_normalize_inserted: bool,
    execution_config: AlgeonNativeExecutionConfig,
    exact_edge_source_gpu_plan: Option<&QueryPlan>,
    task_context: &TaskContext,
) -> Result<(AlgeonNativeExecutionConfig, QueryPlan)> {
    let edge_row_upper_bound = if execution_config.query_admission_source().is_none() {
        None
    } else {
        input_row_bounds.edge_row_upper_bound()
    };
    let definition = CugraphPlanDefinition {
        spec: spec.clone(),
        edge_schema: edge_source.schema(),
        predicate_schemas: predicate_sources.iter().map(|source| source.schema()).collect(),
        predicate_row_upper_bounds: input_row_bounds.predicate_row_upper_bounds().to_vec(),
        result_options: result_options.clone(),
        edge_normalize_inserted,
        edge_row_upper_bound,
        edge_vertex_upper_bound: edge_source_graph_vertex_estimate(edge_source),
    };
    let (graph_execution_plan, graph_execution_evidence) = cugraph_execution_capability_evidence(&definition)?;
    let mut fragment_evidence_items = vec![graph_execution_evidence];
    if let Some(composed_plan) = exact_edge_source_gpu_plan {
        if let Some(composed) = as_algeon_native_composed_exec(edge_source) {
            let composed_evidence = composed_admission_capability_evidence(composed_plan, composed.gpu_inputs())
                .map_err(|error| {
                    Error::new(
                        ErrorCode::InvalidExecutionPlan,
                        "algeon_datafusion.cugraph_sql.graph_ticket_ask",
                        "failed to derive the complete composed edge-source admission evidence",
                    )
                    .with_source(error)
                })?;
            fragment_evidence_items.push(composed_evidence);
        } else {
            let edge_evidence = AttemptCapabilityEvidence::for_query_plan(composed_plan);
            fragment_evidence_items.push(edge_evidence);
        }
    } else {
        collect_native_fragment_capability_evidences(edge_source, &mut fragment_evidence_items)?;
    }
    for predicate_source in predicate_sources {
        collect_native_fragment_capability_evidences(predicate_source.as_ref(), &mut fragment_evidence_items)?;
    }
    let ambient_ticket = match invocation_admission_ticket(task_context) {
        Some(ticket) => Some(ticket),
        None => match (Invocation::of_task(task_context), execution_config.query_admission_source()) {
            (Some(invocation), Some(source)) if invocation.attempt().is_some() => {
                let evidence =
                    AttemptCapabilityEvidence::aggregate(fragment_evidence_items.clone()).map_err(|error| {
                        DataFusionError::from(Error::from_native(
                            "algeon_datafusion.cugraph_sql.graph_ticket_ask",
                            "failed to compute cuGraph SQL graph-aware query admission ticket ask",
                            error,
                        ))
                    })?;
                match invocation.attempt_admission_ticket(source, Some(evidence)) {
                    Some((ticket, true)) => {
                        return Ok((
                            execution_config.with_existing_query_admission_ticket(ticket),
                            graph_execution_plan,
                        ));
                    },
                    // Another fragment of this invocation minted the ticket first.
                    Some((ticket, false)) => Some(ticket),
                    None => None,
                }
            },
            _ => None,
        },
    };
    if let Some(ticket) = ambient_ticket {
        if let Some(source) = execution_config.query_admission_source()
            && !source.same_backend(&ticket.source())
        {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "algeon_datafusion.cugraph_sql.graph_ticket_ask",
                "the graph occurrence and invocation admission ticket belong to different backends",
            )
            .with_log_safe_fact("reason", "graph_occurrence_admission_backend_mismatch")
            .into());
        }
        let existing_evidence = ticket.attempt_capability_evidence_for_rebind();
        let graph_is_covered = existing_evidence
            .as_ref()
            .is_some_and(|evidence| evidence.validate_covers_plan(&graph_execution_plan).is_ok());
        let edge_is_covered = exact_edge_source_gpu_plan.is_none_or(|edge_plan| {
            existing_evidence.as_ref().is_some_and(|evidence| {
                if let Some(composed) = as_algeon_native_composed_exec(edge_source) {
                    let occurrence_ids = composed
                        .gpu_inputs()
                        .iter()
                        .filter_map(|input| gpu_resident_producer_input_name(input.as_ref()))
                        .collect::<Vec<_>>();
                    occurrence_ids.len() == composed.gpu_inputs().len()
                        && evidence.validate_composed_occurrences(edge_plan, &occurrence_ids).is_ok()
                } else {
                    evidence.validate_covers_plan(edge_plan).is_ok()
                }
            })
        });
        if !graph_is_covered || !edge_is_covered {
            ticket.extend_capability_evidence_before_admission(fragment_evidence_items).map_err(|error| {
                DataFusionError::from(Error::from_native(
                    "algeon_datafusion.cugraph_sql.graph_ticket_ask",
                    "failed to bind prepared graph evidence before invocation admission",
                    error,
                ))
            })?;
        }
        return Ok((execution_config.with_existing_query_admission_ticket(ticket), graph_execution_plan));
    }
    if execution_config.query_admission_source().is_some() {
        return Err(attempt_execution_required_error("algeon_datafusion.cugraph_sql.graph_ticket_ask").into());
    }
    Ok((execution_config, graph_execution_plan))
}

pub(super) fn validate_cugraph_graph_owner_preflight(
    spec: &CugraphAlgorithmSpec,
    edge_source: &dyn ExecutionPlan,
    predicate_sources: &[Arc<dyn ExecutionPlan>],
    input_row_bounds: &CugraphInputRowBounds,
    result_options: &CugraphResultPlanOptions,
    edge_normalize_inserted: bool,
    execution_config: &AlgeonNativeExecutionConfig,
) -> Result<()> {
    if execution_config.query_admission_source().is_none() {
        return Ok(());
    }
    let definition = CugraphPlanDefinition {
        spec: spec.clone(),
        edge_schema: edge_source.schema(),
        predicate_schemas: predicate_sources.iter().map(|source| source.schema()).collect(),
        predicate_row_upper_bounds: input_row_bounds.predicate_row_upper_bounds().to_vec(),
        result_options: result_options.clone(),
        edge_normalize_inserted,
        edge_row_upper_bound: input_row_bounds.edge_row_upper_bound(),
        edge_vertex_upper_bound: edge_source_graph_vertex_estimate(edge_source),
    };
    let _ = cugraph_execution_capability_evidence(&definition)?;
    Ok(())
}

fn collect_native_fragment_capability_evidences(
    plan: &dyn ExecutionPlan,
    evidence_items: &mut Vec<AttemptCapabilityEvidence>,
) -> Result<()> {
    if let Some(composed) = as_algeon_native_composed_exec(plan) {
        let native_plan = composed.native_plan().clone_with_gpu_resident_sink();
        let composed_evidence =
            composed_admission_capability_evidence(&native_plan, composed.gpu_inputs()).map_err(|error| {
                Error::new(
                    ErrorCode::InvalidExecutionPlan,
                    "algeon_datafusion.cugraph_sql.graph_ticket_ask",
                    "failed to derive complete composed side-input admission evidence",
                )
                .with_source(error)
            })?;
        evidence_items.push(composed_evidence);
        return Ok(());
    }

    let native_plan = lowered_native_edge_plan(plan);

    // DataFusion's scheduling-only CooperativeExec is transparently unwrapped
    // by `lowered_native_edge_plan` and by the execution path. Descending
    // through it would issue a second evidence for the same native edge frame:
    // only the outer evidence is the graph input's exact owner. Other wrappers
    // retain their children because composed GPU producers execute those
    // distinct fragments themselves.
    if !(plan.name() == "CooperativeExec" && native_plan.is_some()) {
        for child in plan.children() {
            collect_native_fragment_capability_evidences(child.as_ref(), evidence_items)?;
        }
    }

    // `execute_edge_source_to_gpu` converts every native edge fragment to a
    // GPU-resident sink before submitting it to the engine. Derive the evidence
    // from that exact execution shape: a host-sink evidence is intentionally not
    // broad enough to authorize the additional GPU-output contract.
    if let Some(native_plan) = native_plan {
        let native_plan = native_plan.clone_with_gpu_resident_sink();
        evidence_items.push(AttemptCapabilityEvidence::for_query_plan(&native_plan));
    }
    Ok(())
}

/// Returns the lowered plan carried by every native edge-wrapper shape.  Keep
/// graph admission's row, vertex, capability, and native-boundary checks on
/// this one wrapper set so a new producer form cannot silently lose pricing
/// evidence while retaining graph execution authority.
pub(super) fn lowered_native_edge_plan(plan: &dyn ExecutionPlan) -> Option<&QueryPlan> {
    if let Some(native) = as_algeon_native_exec(plan) {
        return Some(native.native_plan());
    }
    if let Some(native) = as_algeon_native_composed_exec(plan) {
        return Some(native.native_plan());
    }
    if let Some(native) = plan.downcast_ref::<AlgeonNativeGpuProducerExec>() {
        return Some(native.native_plan());
    }
    plan.downcast_ref::<AlgeonNativeComposedGpuProducerExec>().map(AlgeonNativeComposedGpuProducerExec::native_plan)
}

pub(super) fn edge_source_output_rows(edge_source: &dyn ExecutionPlan) -> Result<Option<u64>> {
    if let Some(source) = edge_source.downcast_ref::<DataSourceExec>()
        && let Some(rows) = exact_row_count_for_datafusion_local_parquet_scan(source)
    {
        return Ok(Some(rows));
    }
    if let Some(composed) = as_algeon_native_composed_exec(edge_source)
        && let Some(rows) = composed.hard_output_row_upper_bound()
    {
        return Ok(Some(rows));
    }
    if let Some(rows) = gpu_resident_producer_hard_output_row_upper_bound(edge_source) {
        return Ok(Some(rows));
    }
    if let Some(rows) = native_edge_source_output_rows(edge_source) {
        return Ok(Some(rows));
    }
    exact_execution_plan_output_rows(edge_source)
}

/// Returns cardinality evidence that is exact in the physical DataFusion
/// plan. Inexact optimizer estimates must never be used for bounded
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
    // fragment; a bounded graph ticket needs either all partition facts or
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

pub(super) fn resident_side_input_output_rows(plan: &dyn ExecutionPlan) -> Result<Option<u64>> {
    if let Some(rows) = gpu_resident_producer_hard_output_row_upper_bound(plan) {
        return Ok(Some(rows));
    }
    exact_execution_plan_output_rows(plan)
}

fn native_query_plan_hard_row_upper_bound(plan: &dyn ExecutionPlan) -> Option<u64> {
    lowered_native_edge_plan(plan).and_then(query_plan_hard_row_upper_bound)
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

pub(super) fn edge_source_graph_vertex_estimate(edge_source: &dyn ExecutionPlan) -> Option<u64> {
    edge_source.downcast_ref::<DataSourceExec>().and_then(graph_vertex_upper_bound_for_datafusion_local_parquet_scan)
}

pub(super) fn native_plan_is_graph_ticket_light_edge_source(native_plan: &QueryPlan) -> bool {
    native_plan.nodes().iter().all(|node| {
        matches!(node.kind, PlanNode::Source(_) | PlanNode::Filter(_) | PlanNode::Projection(_) | PlanNode::Limit(_))
    })
}

pub(super) fn task_context_with_cugraph_execution_ticket(
    execution_config: &AlgeonNativeExecutionConfig,
    task_context: Arc<TaskContext>,
) -> Arc<TaskContext> {
    if let Some(ticket) = execution_config.query_admission_ticket() {
        task_context_with_query_admission_ticket(task_context, ticket)
    } else {
        task_context
    }
}

pub(super) fn validate_standalone_bounded_cugraph_sources(
    config: &AlgeonNativeExecutionConfig,
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
    if plan_contains_algeon_native(plan) {
        return Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "algeon_datafusion.cugraph_sql.admission",
            "cuGraph execution with native child plans requires an installed AlgeonGpuBackend",
        )
        .with_log_safe_fact("reason", "gpu_backend_required")
        .with_fact("source_kind", source_kind)
        .with_log_safe_fact("remediation", "install_algeon_gpu_backend")
        .into());
    }
    Ok(())
}

pub(super) fn plan_contains_algeon_native(plan: &dyn ExecutionPlan) -> bool {
    plan.downcast_ref::<AlgeonNativeExec>().is_some()
        || plan.downcast_ref::<AlgeonNativeComposedExec>().is_some()
        || plan.downcast_ref::<AlgeonNativeGpuProducerExec>().is_some()
        || plan.downcast_ref::<AlgeonNativeComposedGpuProducerExec>().is_some()
        || plan.downcast_ref::<CugraphAlgoExec>().is_some()
        || plan.children().iter().any(|child| plan_contains_algeon_native(child.as_ref()))
}

pub(super) fn as_algeon_native_exec(plan: &dyn ExecutionPlan) -> Option<&AlgeonNativeExec> {
    if let Some(native) = plan.downcast_ref::<AlgeonNativeExec>() {
        return Some(native);
    }
    graph_input_scheduling_child(plan).and_then(as_algeon_native_exec)
}

pub(super) fn as_algeon_native_composed_exec(plan: &dyn ExecutionPlan) -> Option<&AlgeonNativeComposedExec> {
    if let Some(native) = plan.downcast_ref::<AlgeonNativeComposedExec>() {
        return Some(native);
    }
    graph_input_scheduling_child(plan).and_then(as_algeon_native_composed_exec)
}

fn graph_input_scheduling_child(plan: &dyn ExecutionPlan) -> Option<&dyn ExecutionPlan> {
    if let Some(cooperative) = plan.downcast_ref::<CooperativeExec>() {
        return Some(cooperative.input().as_ref());
    }
    plan.downcast_ref::<RepartitionExec>().map(|repartition| repartition.input().as_ref())
}

pub(crate) fn native_graph_input_execution_plan(plan: &dyn ExecutionPlan) -> Option<&dyn ExecutionPlan> {
    as_algeon_native_exec(plan)
        .map(|native| native as &dyn ExecutionPlan)
        .or_else(|| as_algeon_native_composed_exec(plan).map(|native| native as &dyn ExecutionPlan))
}
