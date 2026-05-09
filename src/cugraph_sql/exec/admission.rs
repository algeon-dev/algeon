use nexus_query_engine::admission::QueryPlanCapabilityProof;

use super::*;
use crate::{
    attempt::NexusAttemptExtension,
    native_exec::{NexusNativeComposedGpuProducerExec, admission::QueryAdmissionTicket},
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

/// Resolves exact edge metadata and mints the one immutable graph-aware ticket
/// for this execution attempt.  Planning must never leave a ticket behind:
/// lakehouse footer pruning can change the graph row and vertex evidence.
pub(super) fn cugraph_execution_config_after_metadata(
    spec: &CugraphAlgorithmSpec,
    edge_source: &dyn ExecutionPlan,
    predicate_sources: &[Arc<dyn ExecutionPlan>],
    predicate_row_upper_bounds: &[Option<u64>],
    projection: Option<&Vec<usize>>,
    limit: Option<usize>,
    edge_normalize_inserted: bool,
    default_execution_config: &NexusNativeExecutionConfig,
    task_context: &TaskContext,
) -> Result<NexusNativeExecutionConfig> {
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
    let execution_config = cugraph_execution_config_with_graph_aware_ticket(
        spec,
        edge_source,
        predicate_sources,
        predicate_row_upper_bounds,
        projection,
        limit,
        edge_normalize_inserted,
        execution_config,
    )?;
    bind_cugraph_ticket_to_attempt(execution_config, task_context)
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
    predicate_row_upper_bounds: &[Option<u64>],
    projection: Option<&Vec<usize>>,
    limit: Option<usize>,
    edge_normalize_inserted: bool,
    execution_config: NexusNativeExecutionConfig,
) -> Result<NexusNativeExecutionConfig> {
    let graph_model_ask = if execution_config.query_admission_source().is_none() {
        None
    } else {
        cugraph_graph_model_ask_bytes(spec, edge_source, edge_normalize_inserted, &execution_config)?
    };
    // A cuGraph attempt executes its graph operation and may also execute
    // lowered native edge/predicate fragments under the same ticket. Keep one
    // aggregate proof for that whole invocation so the engine can verify each
    // fragment at execution time without treating the graph proof as blanket
    // authorization for a different native plan.
    let graph_execution_proof = cugraph_execution_capability_proof(
        spec,
        edge_source.schema(),
        predicate_sources.iter().map(|source| source.schema()).collect(),
        predicate_row_upper_bounds,
        projection,
        limit,
        edge_normalize_inserted,
    )?;
    let mut fragment_proofs = vec![graph_execution_proof];
    collect_native_fragment_capability_proofs(edge_source, &mut fragment_proofs, true);
    for predicate_source in predicate_sources {
        collect_native_fragment_capability_proofs(predicate_source.as_ref(), &mut fragment_proofs, false);
    }
    execution_config.with_graph_aware_plan_query_admission_ticket(graph_model_ask, fragment_proofs).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.graph_ticket_ask",
            "failed to compute cuGraph SQL graph-aware query admission ticket ask",
            error,
        ))
    })
}

fn collect_native_fragment_capability_proofs(
    plan: &dyn ExecutionPlan,
    proofs: &mut Vec<QueryPlanCapabilityProof>,
    is_graph_input_fragment: bool,
) {
    let native_plan = lowered_native_edge_plan(plan);

    // DataFusion's scheduling-only CooperativeExec is transparently unwrapped
    // by `lowered_native_edge_plan` and by the execution path. Descending
    // through it would issue a second proof for the same native edge frame:
    // only the outer proof is the graph input's exact owner. Other wrappers
    // retain their children because composed GPU producers execute those
    // distinct fragments themselves.
    if !(plan.name() == "CooperativeExec" && native_plan.is_some()) {
        for child in plan.children() {
            collect_native_fragment_capability_proofs(child.as_ref(), proofs, false);
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
        } else {
            QueryPlanCapabilityProof::for_query_plan(&native_plan)
        };
        if let Some(proof) = proof {
            proofs.push(proof);
        }
    }
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
    // The graph ticket is an immutable pre-admission contract. Only evidence
    // carried by this exact edge source may establish its cardinality; a
    // service-wide calibration value is not bound to the rowset and cannot
    // supply a grant input here.
    let raw_input_edges = edge_source_output_rows(edge_source)?;
    let Some(raw_input_edges) = raw_input_edges else {
        return Ok(None);
    };
    let vertex_estimate = edge_source_graph_vertex_estimate(edge_source);
    let input_edges = cugraph_effective_input_edges(raw_input_edges, spec, edge_normalize_inserted);
    let construction_sort_workspace_bound_bytes =
        raw_bfs_construction_sort_workspace_bound(&native_spec, spec, &edge_source.schema(), input_edges);
    let device_total_memory_bytes =
        execution_config.query_admission_source().map(|source| source.max_device_total_memory_bytes());
    let evidence = graph_memory_bounded_admission_evidence_for_edge_list_with_vertex_estimate(
        &native_spec,
        &edge_source.schema(),
        input_edges,
        Some(raw_input_edges),
        vertex_estimate,
        construction_sort_workspace_bound_bytes,
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

/// Queries the owner-provided sort-workspace bound for the exact raw BFS
/// graph-construction shape. This direct edge-list admission path turns the
/// owner contract into the typed engine model input.
///
/// An unsupported schema, an invalid owner input, or an owner-query error
/// deliberately leaves the bound absent. Bounded admission then retains its
/// typed fail-closed rejection instead of inventing a local allowance.
fn raw_bfs_construction_sort_workspace_bound(
    native_spec: &GraphAlgorithmSpec,
    spec: &CugraphAlgorithmSpec,
    edge_schema: &SchemaRef,
    input_edges: u64,
) -> Option<u64> {
    if native_spec.algorithm != GraphAlgorithm::Bfs || native_spec.edge_normalize_inserted {
        return None;
    }

    let vertex_type = cugraph_index_dtype(edge_schema, &spec.src_col)?;
    if cugraph_index_dtype(edge_schema, &spec.dst_col)? != vertex_type {
        return None;
    }
    let weight_type = if native_spec.algorithm.consumes_weight() {
        spec.weight_col.as_deref().and_then(|column| cugraph_weight_dtype(edge_schema, column))
    } else {
        None
    };
    let input = cugraph::cudf_interop::EdgelistSortPreflightInput::new(
        vertex_type,
        // Raw single-GPU graph creation dispatches its edge index type from
        // the endpoint type; it is not the optional stable edge-id dtype.
        vertex_type,
        weight_type,
        spec.edge_id_col.is_some(),
        // BFS executes on `Graph<Forward>`, so the construction is not
        // transposed. This has to match the owner query's dispatch input.
        false,
        usize::try_from(input_edges).ok()?,
    )
    .ok()?;
    cugraph::cudf_interop::edgelist_sort_workspace_preflight(&input).ok()
}

fn cugraph_index_dtype(edge_schema: &SchemaRef, column: &str) -> Option<cugraph::types::DataType> {
    match edge_schema.field_with_name(column).ok()?.data_type() {
        DataType::Int32 => Some(cugraph::types::DataType::Int32),
        DataType::Int64 => Some(cugraph::types::DataType::Int64),
        _ => None,
    }
}

fn cugraph_weight_dtype(edge_schema: &SchemaRef, column: &str) -> Option<cugraph::types::DataType> {
    match edge_schema.field_with_name(column).ok()?.data_type() {
        DataType::Float32 => Some(cugraph::types::DataType::Float32),
        DataType::Float64 => Some(cugraph::types::DataType::Float64),
        _ => None,
    }
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
    let partition_count = plan.properties().output_partitioning().partition_count();
    if partition_count <= 1 {
        let statistics = plan.partition_statistics(None)?;
        if let Precision::Exact(rows) = statistics.num_rows {
            return Ok(u64::try_from(rows).ok());
        }
    }

    // Do not treat the plan-level `Precision::Exact` field as authoritative
    // for a parallel edge scan. DataFusion may populate it from one physical
    // fragment; a strict graph ticket needs either all partition facts or
    // stronger native/footer evidence selected above.
    let mut total = 0_u64;
    for partition in 0..partition_count {
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
