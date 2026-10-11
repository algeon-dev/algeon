use std::sync::Arc;

use datafusion::datasource::source::DataSourceExec;
use query_planning::{
    admission::AttemptCapabilityEvidence,
    plan::{QueryPlan, row_bounds::query_plan_hard_row_upper_bound},
};

use super::{
    AlgeonNativeExec, AlgeonNativeExecutionConfig, CooperativeExec, CugraphAlgoExec, CugraphAlgorithmSpec,
    CugraphInputRowBounds, CugraphPlanDefinition, CugraphResultPlanOptions, DataFusionResult, Error, ErrorCode,
    ExecutionPlan, PlanNode, Precision, RepartitionExec, Result, StatisticsArgs, StatisticsContext, TaskContext,
    cugraph_execution_capability_evidence, gpu_resident_producer, gpu_resident_producer_hard_output_row_upper_bound,
    task_context_with_query_admission_ticket,
};
use crate::{
    native::{
        ParquetReadStrategyConfig, exact_row_count_for_datafusion_local_parquet_scan,
        graph_vertex_upper_bound_for_datafusion_local_parquet_scan,
    },
    native_exec::{
        admission::{attempt_execution_required_error, invocation_admission_ticket},
        composed_admission_capability_evidence,
    },
    planner::native::Invocation,
};

pub(super) fn edge_execution_config(
    plan: &dyn ExecutionPlan,
    default_execution_config: &AlgeonNativeExecutionConfig,
) -> AlgeonNativeExecutionConfig {
    native_graph_input_execution_plan(plan)
        .and_then(|native| native.downcast_ref::<AlgeonNativeExec>())
        .filter(|native| !native.gpu_inputs().is_empty())
        .map_or_else(|| default_execution_config.clone(), |native| native.execution_config().clone())
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
) -> Result<(AlgeonNativeExecutionConfig, QueryPlan)> {
    let execution_config =
        edge_execution_config(edge_source, default_execution_config).without_query_admission_ticket();
    // The nested state that lowers a cuGraph edge relation deliberately has
    // no admission authority: the graph root must be the only admission
    // owner once footer metadata is resolved. Its lowered child still owns
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
        edge_vertex_upper_bound: edge_source_graph_vertex_estimate(
            edge_source,
            &execution_config.source_read_strategy(),
        ),
    };
    let (graph_execution_plan, graph_execution_evidence) = cugraph_execution_capability_evidence(&definition)?;
    let mut fragment_evidence_items = vec![graph_execution_evidence];
    collect_native_fragment_capability_evidences(edge_source, &mut fragment_evidence_items)?;
    for predicate_source in predicate_sources {
        collect_native_fragment_capability_evidences(predicate_source.as_ref(), &mut fragment_evidence_items)?;
    }
    let ambient_ticket = match invocation_admission_ticket(task_context) {
        Some(ticket) => Some(ticket),
        None => match (Invocation::of_task(task_context), execution_config.query_admission_source()) {
            (Some(invocation), Some(source)) if invocation.attempt().is_some() => {
                let evidence =
                    AttemptCapabilityEvidence::aggregate(fragment_evidence_items.clone()).map_err(|error| {
                        Error::from_native(
                            "algeon_datafusion.cugraph_sql.graph_ticket_ask",
                            "failed to compute cuGraph SQL graph-aware query admission ticket ask",
                            error,
                        )
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
            .with_log_safe_fact("reason", "graph_occurrence_admission_backend_mismatch"));
        }
        let graph_is_covered = ticket
            .attempt_capability_evidence_for_rebind()
            .is_some_and(|evidence| evidence.validate_covers_plan(&graph_execution_plan).is_ok());
        if !graph_is_covered {
            ticket.bind_fragment_evidence(fragment_evidence_items).map_err(|error| {
                Error::from_native(
                    "algeon_datafusion.cugraph_sql.graph_ticket_ask",
                    "failed to bind prepared graph evidence to the invocation admission",
                    error,
                )
            })?;
        }
        return Ok((execution_config.with_existing_query_admission_ticket(ticket), graph_execution_plan));
    }
    if execution_config.query_admission_source().is_some() {
        return Err(attempt_execution_required_error("algeon_datafusion.cugraph_sql.graph_ticket_ask"));
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
        edge_vertex_upper_bound: edge_source_graph_vertex_estimate(
            edge_source,
            &execution_config.source_read_strategy(),
        ),
    };
    let _ = cugraph_execution_capability_evidence(&definition)?;
    Ok(())
}

/// Collects the evidence of every native fragment a graph input executes, in
/// the form planning admitted it: a GPU-resident producer the graph reads on
/// the device, or a host-sink fragment inside a DataFusion input the graph
/// stages on the host.
fn collect_native_fragment_capability_evidences(
    plan: &dyn ExecutionPlan,
    evidence_items: &mut Vec<AttemptCapabilityEvidence>,
) -> Result<()> {
    if let Some(producer) = gpu_resident_producer(plan) {
        evidence_items.push(producer.gpu_result_capability_evidence()?);
        return Ok(());
    }
    if let Some(composed) = plan.downcast_ref::<AlgeonNativeExec>().filter(|native| !native.gpu_inputs().is_empty()) {
        let composed_evidence = composed_admission_capability_evidence(composed.native_plan(), composed.gpu_inputs())
            .map_err(|error| {
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
    if let Some(native) = plan.downcast_ref::<AlgeonNativeExec>() {
        evidence_items.push(AttemptCapabilityEvidence::for_query_plan(native.native_plan()));
        return Ok(());
    }
    for child in plan.children() {
        collect_native_fragment_capability_evidences(child.as_ref(), evidence_items)?;
    }
    Ok(())
}

/// Returns the lowered plan of a native graph input. Keep graph admission's
/// row, vertex, capability, and native-boundary checks on this one lookup so
/// a new producer form cannot lose pricing evidence while retaining
/// graph execution authority.
pub(super) fn lowered_native_edge_plan(plan: &dyn ExecutionPlan) -> Option<&QueryPlan> {
    native_graph_input_execution_plan(plan)?.downcast_ref::<AlgeonNativeExec>().map(AlgeonNativeExec::native_plan)
}

pub(super) fn edge_source_output_rows(
    edge_source: &dyn ExecutionPlan,
    source_reads: &ParquetReadStrategyConfig,
) -> DataFusionResult<Option<u64>> {
    if let Some(source) = edge_source.downcast_ref::<DataSourceExec>()
        && let Some(rows) = exact_row_count_for_datafusion_local_parquet_scan(source, source_reads)
    {
        return Ok(Some(rows));
    }
    if let Some(rows) = gpu_resident_producer_hard_output_row_upper_bound(
        native_graph_input_execution_plan(edge_source).unwrap_or(edge_source),
    ) {
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
pub(super) fn exact_execution_plan_output_rows(plan: &dyn ExecutionPlan) -> DataFusionResult<Option<u64>> {
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

pub(super) fn resident_side_input_output_rows(plan: &dyn ExecutionPlan) -> DataFusionResult<Option<u64>> {
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

pub(super) fn edge_source_graph_vertex_estimate(
    edge_source: &dyn ExecutionPlan,
    source_reads: &ParquetReadStrategyConfig,
) -> Option<u64> {
    edge_source
        .downcast_ref::<DataSourceExec>()
        .and_then(|source| graph_vertex_upper_bound_for_datafusion_local_parquet_scan(source, source_reads))
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
        .with_log_safe_fact("remediation", "install_algeon_gpu_backend"));
    }
    Ok(())
}

pub(super) fn plan_contains_algeon_native(plan: &dyn ExecutionPlan) -> bool {
    plan.downcast_ref::<AlgeonNativeExec>().is_some()
        || plan.downcast_ref::<CugraphAlgoExec>().is_some()
        || plan.children().iter().any(|child| plan_contains_algeon_native(child.as_ref()))
}

/// The native GPU producer a graph input runs. The graph reads each input as
/// one whole relation, so the cooperative-yield and repartition wrappers
/// DataFusion places above a producer change neither its rows nor where it
/// runs, and the graph executes the producer itself.
pub(super) fn graph_input_native_producer(plan: &Arc<dyn ExecutionPlan>) -> Option<&Arc<dyn ExecutionPlan>> {
    if is_native_gpu_producer(plan.as_ref()) {
        return Some(plan);
    }
    graph_input_scheduling_child(plan.as_ref()).and_then(graph_input_native_producer)
}

pub(crate) fn native_graph_input_execution_plan(plan: &dyn ExecutionPlan) -> Option<&dyn ExecutionPlan> {
    if is_native_gpu_producer(plan) {
        return Some(plan);
    }
    graph_input_scheduling_child(plan).and_then(|child| native_graph_input_execution_plan(child.as_ref()))
}

fn is_native_gpu_producer(plan: &dyn ExecutionPlan) -> bool {
    plan.downcast_ref::<AlgeonNativeExec>().is_some_and(|native| !native.materializes_host())
}

fn graph_input_scheduling_child(plan: &dyn ExecutionPlan) -> Option<&Arc<dyn ExecutionPlan>> {
    if let Some(cooperative) = plan.downcast_ref::<CooperativeExec>() {
        return Some(cooperative.input());
    }
    plan.downcast_ref::<RepartitionExec>().map(RepartitionExec::input)
}
