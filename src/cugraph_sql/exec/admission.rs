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

pub(super) async fn exact_native_edge_source_gpu_plan(
    edge_source: &dyn ExecutionPlan,
    task_context: &TaskContext,
) -> Result<Option<nexus_query_engine::plan::QueryPlan>> {
    let (execution_config, native_plan) = if let Some(native) = as_nexus_native_exec(edge_source) {
        (native.execution_config(), native.native_plan())
    } else if let Some(composed) = as_nexus_native_composed_exec(edge_source) {
        (composed.execution_config(), composed.native_plan())
    } else {
        return Ok(None);
    };
    let plan =
        crate::native_exec::lakehouse::resolve_query_plan(execution_config, native_plan.clone(), task_context).await;
    Ok(Some(plan.clone_with_gpu_resident_sink()))
}

fn deferred_cugraph_graph_proof(
    graph_execution_plan: &nexus_query_engine::plan::QueryPlan,
    graph_execution_proof: QueryPlanCapabilityProof,
    edge_source: &dyn ExecutionPlan,
    exact_edge_source_gpu_plan: Option<&nexus_query_engine::plan::QueryPlan>,
) -> Result<Option<QueryPlanCapabilityProof>> {
    let Some(edge_plan) = exact_edge_source_gpu_plan else {
        return Ok(None);
    };
    if as_nexus_native_composed_exec(edge_source).is_some() {
        return Ok(None);
    }
    QueryPlanCapabilityProof::try_combine_cugraph_materializing_input_attempt(
        graph_execution_plan,
        graph_execution_proof,
        CUGRAPH_EDGE_INPUT_NAME,
        edge_plan,
        std::iter::empty(),
    )
    .map_err(|error| {
        Error::from_native(
            "datafusion_nexus.cugraph_sql.graph_ticket_ask",
            "failed to derive the deferred cuGraph phase admission contract",
            error,
        )
        .into()
    })
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
    let materialized_side_input_plans = predicate_sources
        .iter()
        .enumerate()
        .map(|(index, source)| {
            lowered_native_edge_plan(source.as_ref())
                .map(|plan| (super::lowering::cugraph_side_input_name(index), plan.clone_with_gpu_resident_sink()))
        })
        .collect::<Option<Vec<_>>>();
    let (graph_execution_plan, graph_execution_proof) = cugraph_execution_capability_proof_with_resident_input_plans(
        spec,
        edge_source.schema(),
        predicate_sources.iter().map(|source| source.schema()).collect(),
        input_row_bounds.predicate_row_upper_bounds(),
        result_options,
        edge_normalize_inserted,
        edge_row_upper_bound,
        edge_vertex_upper_bound,
        exact_edge_source_gpu_plan,
        materialized_side_input_plans.as_deref().unwrap_or_default(),
    )?;
    let mut graph_model_ask = if execution_config.query_admission_source().is_none() {
        None
    } else {
        cugraph_graph_model_ask_bytes(&graph_execution_plan, edge_source, &execution_config)?
    };
    let static_graph_phase_exceeds_device = execution_config
        .query_admission_source()
        .zip(graph_model_ask.as_ref())
        .is_some_and(|(source, model)| model.exceeds_device_total_memory_bytes(source.max_device_total_memory_bytes()));
    let deferred_graph_proof = if static_graph_phase_exceeds_device && materialized_side_input_plans.is_some() {
        deferred_cugraph_graph_proof(
            &graph_execution_plan,
            graph_execution_proof.clone(),
            edge_source,
            exact_edge_source_gpu_plan,
        )?
    } else {
        None
    };
    if deferred_graph_proof.is_some() {
        graph_model_ask = None;
    }
    // A cuGraph attempt executes its graph operation and may also execute
    // lowered native edge/predicate fragments under the same ticket. Keep one
    // aggregate proof for that whole invocation so the engine can verify each
    // fragment at execution time without treating the graph proof as blanket
    // authorization for a different native plan.
    let scheduled_graph_proof = match (deferred_graph_proof, exact_edge_source_gpu_plan, graph_model_ask.as_ref()) {
        (Some(deferred), _, _) => Some(deferred),
        (None, Some(edge_plan), Some(model_ask))
            if as_nexus_native_composed_exec(edge_source).is_some() && predicate_sources.is_empty() =>
        {
            let graph_with_evidence = graph_execution_proof
                .clone()
                .with_graph_memory_admission_evidence(model_ask.cold_model_evidence())
                .map_err(|error| {
                    Error::from_native(
                        "datafusion_nexus.cugraph_sql.graph_ticket_ask",
                        "failed to attach cuGraph memory evidence before composing its graph-producing edge input",
                        error,
                    )
                })?;
            let composed =
                as_nexus_native_composed_exec(edge_source).expect("match guard requires a composed native edge source");
            let edge_proof =
                composed_admission_capability_proof(edge_plan, composed.gpu_inputs()).map_err(|error| {
                    Error::new(
                        ErrorCode::InvalidExecutionPlan,
                        "datafusion_nexus.cugraph_sql.graph_ticket_ask",
                        "failed to derive the graph-producing edge-source admission proof",
                    )
                    .with_source(error)
                })?;
            Some(
                QueryPlanCapabilityProof::combine_composed_graph_attempt(
                    &graph_execution_plan,
                    graph_with_evidence,
                    [(CUGRAPH_EDGE_INPUT_NAME.to_owned(), edge_proof)],
                )
                .map_err(|error| {
                    Error::from_native(
                        "datafusion_nexus.cugraph_sql.graph_ticket_ask",
                        "failed to derive the ordered graph-to-graph admission schedule",
                        error,
                    )
                })?,
            )
        },
        (None, Some(edge_plan), Some(model_ask))
            if as_nexus_native_composed_exec(edge_source).is_none() && materialized_side_input_plans.is_some() =>
        {
            let graph_with_evidence = graph_execution_proof
                .clone()
                .with_graph_memory_admission_evidence(model_ask.cold_model_evidence())
                .map_err(|error| {
                    Error::from_native(
                        "datafusion_nexus.cugraph_sql.graph_ticket_ask",
                        "failed to attach cuGraph memory evidence before composing its native edge input",
                        error,
                    )
                })?;
            let side_input_proofs = materialized_side_input_plans
                .as_ref()
                .expect("match guard requires materialized side-input plans")
                .iter()
                .zip(predicate_sources)
                .map(|((name, plan), source)| {
                    let proof = if let Some(composed) = as_nexus_native_composed_exec(source.as_ref()) {
                        composed_admission_capability_proof(plan, composed.gpu_inputs()).map_err(|error| {
                            Error::new(
                                ErrorCode::InvalidExecutionPlan,
                                "datafusion_nexus.cugraph_sql.graph_ticket_ask",
                                "failed to derive the complete composed side-input admission proof",
                            )
                            .with_source(error)
                        })?
                    } else {
                        QueryPlanCapabilityProof::for_cugraph_materialized_input_producer_plan(plan).map_err(
                            |error| {
                                Error::from_native(
                                    "datafusion_nexus.cugraph_sql.graph_ticket_ask",
                                    "failed to derive native cuGraph side-input producer evidence",
                                    error,
                                )
                            },
                        )?
                    };
                    Ok((name.clone(), proof))
                })
                .collect::<Result<Vec<_>>>()?;
            QueryPlanCapabilityProof::try_combine_cugraph_materializing_input_attempt_with_proofs(
                &graph_execution_plan,
                graph_with_evidence,
                CUGRAPH_EDGE_INPUT_NAME,
                edge_plan,
                side_input_proofs,
            )
            .map_err(|error| {
                Error::from_native(
                    "datafusion_nexus.cugraph_sql.graph_ticket_ask",
                    "failed to derive the ordered cuGraph edge-mapping admission schedule",
                    error,
                )
            })?
        },
        _ => None,
    };
    let materialized_edge_input_scheduled = scheduled_graph_proof.is_some();
    let mut fragment_proofs = vec![scheduled_graph_proof.unwrap_or(graph_execution_proof)];
    if materialized_edge_input_scheduled {
        graph_model_ask = None;
    }
    if !materialized_edge_input_scheduled {
        if let Some(composed_plan) = exact_edge_source_gpu_plan {
            if let Some(composed) = as_nexus_native_composed_exec(edge_source) {
                let composed_proof = composed_admission_capability_proof(composed_plan, composed.gpu_inputs())
                    .map_err(|error| {
                        Error::new(
                            ErrorCode::InvalidExecutionPlan,
                            "datafusion_nexus.cugraph_sql.graph_ticket_ask",
                            "failed to derive the complete composed edge-source admission proof",
                        )
                        .with_source(error)
                    })?;
                fragment_proofs.push(composed_proof);
            } else {
                let edge_proof =
                    QueryPlanCapabilityProof::for_cugraph_graph_input_plan(composed_plan).ok_or_else(|| {
                        Error::new(
                            ErrorCode::InvalidExecutionPlan,
                            "datafusion_nexus.cugraph_sql.graph_ticket_ask",
                            "failed to derive the exact native edge-source admission proof",
                        )
                        .with_fact("reason", "native_edge_source_plan_evidence_missing")
                    })?;
                fragment_proofs.push(edge_proof);
            }
        } else {
            collect_native_fragment_capability_proofs(edge_source, &mut fragment_proofs, true, false)?;
        }
    }
    if !materialized_edge_input_scheduled {
        for predicate_source in predicate_sources {
            collect_native_fragment_capability_proofs(predicate_source.as_ref(), &mut fragment_proofs, false, true)?;
        }
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
    exact_edge_source_gpu_plan: Option<&nexus_query_engine::plan::QueryPlan>,
) -> Result<()> {
    if execution_config.query_admission_source().is_none() {
        return Ok(());
    }
    let edge_vertex_upper_bound = edge_source_graph_vertex_estimate(edge_source).map(|estimate| estimate.count);
    let (graph_execution_plan, graph_execution_proof) = cugraph_execution_capability_proof_with_edge_row_upper_bound(
        spec,
        edge_source.schema(),
        predicate_sources.iter().map(|source| source.schema()).collect(),
        input_row_bounds.predicate_row_upper_bounds(),
        result_options,
        edge_normalize_inserted,
        input_row_bounds.edge_row_upper_bound(),
        edge_vertex_upper_bound,
    )?;
    let graph_model_ask = cugraph_graph_model_ask_bytes(&graph_execution_plan, edge_source, execution_config)?;
    let static_graph_phase_exceeds_device = execution_config
        .query_admission_source()
        .zip(graph_model_ask.as_ref())
        .is_some_and(|(source, model)| model.exceeds_device_total_memory_bytes(source.max_device_total_memory_bytes()));
    if static_graph_phase_exceeds_device {
        let _ = deferred_cugraph_graph_proof(
            &graph_execution_plan,
            graph_execution_proof,
            edge_source,
            exact_edge_source_gpu_plan,
        )?;
    }
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
    graph_input_scheduling_child(plan).and_then(as_nexus_native_exec)
}

pub(super) fn as_nexus_native_composed_exec(plan: &dyn ExecutionPlan) -> Option<&NexusNativeComposedExec> {
    if let Some(native) = plan.downcast_ref::<NexusNativeComposedExec>() {
        return Some(native);
    }
    graph_input_scheduling_child(plan).and_then(as_nexus_native_composed_exec)
}

fn graph_input_scheduling_child(plan: &dyn ExecutionPlan) -> Option<&dyn ExecutionPlan> {
    if let Some(cooperative) = plan.downcast_ref::<CooperativeExec>() {
        return Some(cooperative.input().as_ref());
    }
    plan.downcast_ref::<RepartitionExec>().map(|repartition| repartition.input().as_ref())
}

pub(crate) fn native_graph_input_execution_plan(plan: &dyn ExecutionPlan) -> Option<&dyn ExecutionPlan> {
    as_nexus_native_exec(plan)
        .map(|native| native as &dyn ExecutionPlan)
        .or_else(|| as_nexus_native_composed_exec(plan).map(|native| native as &dyn ExecutionPlan))
}
