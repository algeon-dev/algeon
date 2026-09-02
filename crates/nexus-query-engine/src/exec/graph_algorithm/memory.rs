use super::{
    BTreeMap, DataFrame, ExecutionContext, GpuDataFrame, GraphAlgorithmExecutionResult, NodeId, QueryError,
    QueryErrorCode, QueryMetrics, Result,
};

#[cfg(feature = "cugraph")]
pub(super) fn graph_pool_usage_snapshot(context: &ExecutionContext) -> Option<cudf::DeviceMemoryPoolUsage> {
    context.runtime().ok()?.device_memory_manager().ok()?.pool_usage().ok()
}

#[cfg(feature = "cugraph")]
pub(super) fn graph_pool_high_water_delta(
    before: Option<cudf::DeviceMemoryPoolUsage>,
    after: Option<cudf::DeviceMemoryPoolUsage>,
) -> Option<u64> {
    let before = before?;
    let after = after?;
    if after.used_high_bytes <= before.used_high_bytes {
        return None;
    }
    Some(after.used_high_bytes.saturating_sub(before.used_current_bytes))
}

/// Converts source-bound graph provenance carried by a resident edge frame to
/// the graph-memory model's typed input. External content identities do not
/// prove a footer vertex bound and therefore deliberately retain the fallback.
pub(in crate::exec) fn source_graph_vertex_estimate(
    graph_residency: Option<&crate::plan::GraphResidencyDescriptor>,
) -> Option<crate::graph_memory::GraphVertexEstimate> {
    graph_residency
        .and_then(crate::plan::GraphResidencyDescriptor::graph_vertex_upper_bound)
        .map(crate::graph_memory::GraphVertexEstimate::graph_metadata)
}

/// Returns the exact raw edge count carried by a complete source-bound graph
/// descriptor. External identities and partial sources cannot prove this
/// provenance fact; it never asserts that edge normalization retains a raw
/// DataFrame allocation.
pub(in crate::exec) fn source_graph_raw_edge_row_count(
    graph_residency: Option<&crate::plan::GraphResidencyDescriptor>,
) -> Option<u64> {
    graph_residency.and_then(crate::plan::GraphResidencyDescriptor::graph_raw_edge_row_count)
}

#[cfg(feature = "cugraph")]
#[expect(clippy::too_many_arguments, reason = "graph algorithm admission inputs are distinct proof values")]
pub(super) fn admit_graph_memory(
    context: &ExecutionContext,
    node_id: NodeId,
    spec: &crate::plan::GraphAlgorithmSpec,
    input: &DataFrame,
    side_inputs: &BTreeMap<NodeId, GpuDataFrame>,
    input_edges: u64,
    vertex_estimate: Option<crate::graph_memory::GraphVertexEstimate>,
    raw_input_edges: Option<u64>,
    metrics: &mut QueryMetrics,
) -> Result<Option<u64>> {
    let report = graph_memory_reservation_report(GraphMemoryReservationRequest {
        context,
        node_id,
        spec,
        input,
        side_inputs,
        input_edges,
        raw_input_edges,
        bounded_pair_cardinality: None,
        budget_bytes: context.bounded_device_budget_bytes(),
        device_total_memory_bytes: graph_device_total_memory_bytes(context)?,
        allow_nulls: false,
        vertex_estimate,
        statistics: None,
        already_accounted_normalized_graph_input_bytes: 0,
        exact_normalized_graph_input_charge_bytes: None,
        retained_normalization_source_bytes: 0,
        retained_normalization_source_credit_cap_bytes: None,
        already_accounted_similarity_pair_input_bytes: 0,
    })?;
    // Every non-empty graph executor gets a calibration row, including
    // algorithms whose model is currently diagnostic-only rather than a live
    // graph-tier reservation.
    metrics.record_graph_claim_calibration(report.clone(), node_id);
    if !crate::graph_memory::graph_memory_reservation_required(spec.algorithm) {
        return Ok(None);
    }
    let model = crate::graph_memory::GraphMemoryModel;
    let request = model.reservation_request(node_id, &report, context.device_memory_space());
    metrics.record_graph_reservation(report.clone());

    let Some(mut request) = request else {
        return Ok(None);
    };
    if request.query_id.is_none() && metrics.query_id().is_assigned() {
        request.query_id = Some(metrics.query_id().clone());
    }
    if report.outcome == crate::graph_memory::GraphReservationOutcome::Denied {
        metrics.record_graph_claim_calibration_graph_memory_admission_denied(node_id, spec.algorithm);
        let granted_bytes = request.budget_bytes;
        metrics.record_reservation_denied_event(&request, granted_bytes);
        return Err(report.denied_error(&request, granted_bytes));
    }

    metrics.record_reservation_granted_event(&request, request.requested_bytes);
    metrics.record_reservation_timeline_event(crate::metrics::MemoryReservationEvent::from_request(
        &request,
        crate::metrics::MemoryReservationEventOutcome::Granted,
        crate::metrics::MemoryReservationEventAction::Release,
        request.requested_bytes,
        -i128::from(request.requested_bytes),
        0,
    ));
    Ok(None)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::{source_graph_raw_edge_row_count, source_graph_vertex_estimate};
    use crate::{
        graph_memory::{GraphVertexEstimate, GraphVertexEstimateSource},
        plan::{
            GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec, GraphConstructionOptions,
            GraphEdgeColumnBindings, GraphForceAtlas2Params, GraphResidencyDescriptor, GraphResidencyPlanAnalysis,
            graph_residency_descriptor_for_local_parquet_source,
        },
        source::{LocalParquetFile, LocalParquetSource},
    };

    #[test]
    fn source_graph_vertex_estimate_uses_only_source_bound_footer_provenance() {
        let source = LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new("/tmp/source-bound-force-atlas2.parquet")
                    .with_stable_id("local:/tmp/source-bound-force-atlas2.parquet:1024:mtime=2026-08-05T00:00:00Z")
                    .with_file_size_bytes(1024)
                    .with_row_group_rows(vec![45_564_149])
                    .with_graph_vertex_upper_bound(4_894_081),
            ],
            Arc::new(Schema::new(vec![
                Field::new("src", DataType::Int64, false),
                Field::new("dst", DataType::Int64, false),
            ])),
            vec!["src".to_string(), "dst".to_string()],
        );
        let force_atlas2 = GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::ForceAtlas2,
            GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
            GraphConstructionOptions::default(),
            GraphAlgorithmParams::ForceAtlas2(GraphForceAtlas2Params::default()),
        );
        let GraphResidencyPlanAnalysis::Keyable(source_bound) =
            graph_residency_descriptor_for_local_parquet_source(&source, &force_atlas2)
        else {
            panic!("complete footer-backed Local Parquet graph source must be keyable");
        };
        let external_identity =
            GraphResidencyDescriptor::for_content_identity("external-graph-edges", DataType::Int64, None, None)
                .expect("stable external graph identity");

        let estimate = source_graph_vertex_estimate(Some(&source_bound)).expect("footer graph bound");
        assert_eq!(estimate.count, 4_894_081);
        assert_eq!(estimate.source, GraphVertexEstimateSource::GraphMetadata);
        assert_eq!(estimate, GraphVertexEstimate::graph_metadata(4_894_081));
        assert_eq!(source_graph_raw_edge_row_count(Some(&source_bound)), Some(45_564_149));
        assert_eq!(source_graph_vertex_estimate(Some(&external_identity)), None);
        assert_eq!(source_graph_vertex_estimate(None), None);
        assert_eq!(source_graph_raw_edge_row_count(Some(&external_identity)), None);
        assert_eq!(source_graph_raw_edge_row_count(None), None);
    }
}

#[cfg(feature = "cugraph")]
pub(in crate::exec) struct GraphMemoryReservationRequest<'a> {
    pub(in crate::exec) context: &'a ExecutionContext,
    pub(in crate::exec) node_id: NodeId,
    pub(in crate::exec) spec: &'a crate::plan::GraphAlgorithmSpec,
    pub(in crate::exec) input: &'a DataFrame,
    pub(in crate::exec) side_inputs: &'a BTreeMap<NodeId, GpuDataFrame>,
    pub(in crate::exec) input_edges: u64,
    pub(in crate::exec) raw_input_edges: Option<u64>,
    pub(in crate::exec) bounded_pair_cardinality: Option<(u64, crate::graph_memory::GraphPairCardinalitySource)>,
    pub(in crate::exec) budget_bytes: Option<u64>,
    pub(in crate::exec) device_total_memory_bytes: Option<u64>,
    pub(in crate::exec) allow_nulls: bool,
    pub(in crate::exec) vertex_estimate: Option<crate::graph_memory::GraphVertexEstimate>,
    pub(in crate::exec) statistics: Option<crate::graph_memory::GraphStatistics>,
    pub(in crate::exec) already_accounted_normalized_graph_input_bytes: u64,
    /// The exact tracker charge for a live `EdgeNormalize` output. Unlike a
    /// generic source frame, this is the sole logical graph-edge frame and may
    /// safely credit its full physical charge against the whole-peak model.
    pub(in crate::exec) exact_normalized_graph_input_charge_bytes: Option<u64>,
    pub(in crate::exec) retained_normalization_source_bytes: u64,
    pub(in crate::exec) retained_normalization_source_credit_cap_bytes: Option<u64>,
    pub(in crate::exec) already_accounted_similarity_pair_input_bytes: u64,
}

#[cfg(feature = "cugraph")]
pub(in crate::exec) fn graph_memory_reservation_report(
    request: GraphMemoryReservationRequest<'_>,
) -> Result<crate::graph_memory::GraphReservationReport> {
    let GraphMemoryReservationRequest {
        context,
        node_id,
        spec,
        input,
        side_inputs,
        input_edges,
        raw_input_edges,
        bounded_pair_cardinality,
        budget_bytes,
        device_total_memory_bytes,
        allow_nulls,
        vertex_estimate,
        statistics,
        already_accounted_normalized_graph_input_bytes,
        exact_normalized_graph_input_charge_bytes,
        retained_normalization_source_bytes,
        retained_normalization_source_credit_cap_bytes,
        already_accounted_similarity_pair_input_bytes,
    } = request;
    let mut model_input = graph_memory_model_input_for_frame(input, spec, input_edges, allow_nulls)?;
    if let Some(raw_input_edges) = raw_input_edges {
        model_input = model_input.with_raw_input_edges(raw_input_edges);
    }
    if let Some((runtime_pair_rows, pair_vertex_width_bytes, runtime_pair_cardinality_source)) =
        graph_similarity_pair_memory_evidence(spec, side_inputs)?
    {
        let (pair_rows, pair_cardinality_source) = match bounded_pair_cardinality {
            Some((pair_rows, pair_cardinality_source)) => (pair_rows, pair_cardinality_source),
            None => (runtime_pair_rows, runtime_pair_cardinality_source),
        };
        model_input =
            model_input.with_similarity_pair_input(pair_rows, pair_vertex_width_bytes, pair_cardinality_source);
    }
    model_input = context.graph_memory_calibration_override().apply_to_input(model_input);
    if let Some(device_total_memory_bytes) = device_total_memory_bytes {
        model_input = model_input.with_device_total_memory_bytes(device_total_memory_bytes);
    }
    model_input = model_input.with_edge_normalize_inserted(spec.edge_normalize_inserted);
    if let Some(vertex_estimate) = vertex_estimate {
        model_input = model_input.with_vertex_estimate(vertex_estimate);
    }
    if let Some(statistics) = statistics {
        model_input = model_input.with_statistics(statistics);
    }
    model_input = crate::graph_memory::with_graph_construction_workspace_bounds_from_owner(spec, model_input);
    let model = crate::graph_memory::GraphMemoryModel;
    let estimate = match budget_bytes.is_some() {
        true => model.estimate_for_bounded(spec, model_input)?,
        false => model.estimate(spec, model_input)?,
    };
    let report = crate::graph_memory::GraphReservationReport::from_estimate(node_id, estimate, budget_bytes)
        .credit_live_graph_input(
            already_accounted_normalized_graph_input_bytes,
            exact_normalized_graph_input_charge_bytes,
            retained_normalization_source_bytes,
            retained_normalization_source_credit_cap_bytes,
        )
        .credit_live_similarity_pair_input(already_accounted_similarity_pair_input_bytes);
    Ok(match statistics {
        Some(statistics) => report.with_statistics(statistics),
        None => report,
    })
}

#[cfg(feature = "cugraph")]
pub(in crate::exec) fn graph_device_total_memory_bytes(context: &ExecutionContext) -> Result<Option<u64>> {
    context.runtime()?.device().available_memory().map(|memory| Some(memory.total_bytes)).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.graph_algorithm.graph_memory_device",
            "failed to query CUDA device memory metadata for graph memory pricing",
            error,
        )
    })
}

// Common return envelope for algorithm-specific cuGraph executors.

#[cfg(feature = "cugraph")]
pub(super) struct GraphExecutionBudget {
    pub(super) runtime: std::sync::Arc<cudf::Runtime>,
    pub(super) token: cudf::ExecutionBudgetToken,
}

#[cfg(feature = "cugraph")]
#[derive(Debug, Clone)]
pub(in crate::exec) struct GraphBudgetObservation {
    operation: &'static str,
    enforcement: &'static str,
    budget_bytes: u64,
    allocator_high_water_bytes: Option<u64>,
    status: &'static str,
    context_facts: BTreeMap<String, String>,
}

#[cfg(feature = "cugraph")]
impl GraphBudgetObservation {
    fn from_reservation_usage(
        operation: &'static str,
        token: &cudf::ExecutionBudgetToken,
        usage: cudf::ReservationUsage,
        arena_usage: Option<cudf::ReservationArenaUsage>,
        status: &'static str,
    ) -> Self {
        let mut context_facts = BTreeMap::new();
        context_facts.insert("reservation_id".to_string(), usage.reservation_id.to_string());
        context_facts.insert("reservation_attached".to_string(), usage.attached.to_string());
        context_facts.insert("allocator_current_bytes".to_string(), usage.current_bytes.to_string());
        context_facts.insert("allocator_peak_aligned_bytes".to_string(), usage.peak_aligned_bytes.to_string());
        context_facts.insert("allocator_current_aligned_bytes".to_string(), usage.current_aligned_bytes.to_string());
        context_facts.insert("allocator_total_bytes".to_string(), usage.total_bytes.to_string());
        context_facts.insert("allocator_total_allocations".to_string(), usage.total_allocations.to_string());
        context_facts.insert("allocator_current_allocations".to_string(), usage.current_allocations.to_string());
        context_facts.insert("allocator_peak_allocations".to_string(), usage.peak_allocations.to_string());
        context_facts
            .insert("allocator_failed_allocation_count".to_string(), usage.failed_allocation_count.to_string());
        context_facts.insert(
            "allocator_largest_failed_allocation_bytes".to_string(),
            usage.largest_failed_allocation_bytes.to_string(),
        );
        if let Some(arena_usage) = arena_usage {
            context_facts.insert("arena_id".to_string(), arena_usage.arena_id.to_string());
            context_facts.insert("arena_active".to_string(), arena_usage.active.to_string());
            context_facts.insert("arena_active_members".to_string(), arena_usage.active_members.to_string());
            context_facts.insert("arena_limit_bytes".to_string(), arena_usage.allocation_limit_bytes.to_string());
            context_facts.insert("arena_current_bytes".to_string(), arena_usage.current_bytes.to_string());
            context_facts.insert("arena_peak_bytes".to_string(), arena_usage.peak_bytes.to_string());
            context_facts
                .insert("arena_failed_allocation_count".to_string(), arena_usage.failed_allocation_count.to_string());
            context_facts.insert(
                "arena_largest_failed_allocation_bytes".to_string(),
                arena_usage.largest_failed_allocation_bytes.to_string(),
            );
        }
        Self {
            operation,
            enforcement: token.enforcement().as_str(),
            budget_bytes: token.request().budget_bytes() as u64,
            allocator_high_water_bytes: Some(usage.peak_bytes),
            status,
            context_facts,
        }
    }

    pub(super) fn record(&self, metrics: &mut QueryMetrics) {
        self.record_with_boundary(metrics, crate::graph_memory::GRAPH_MEMORY_BOUNDARY, None);
    }

    /// Returns the typed reservation high-water mark captured with this
    /// observation. Callers that calibrate graph pricing must use this direct
    /// value rather than reconstructing a peak from error facts.
    pub(super) const fn allocator_high_water_bytes(&self) -> Option<u64> {
        self.allocator_high_water_bytes
    }

    pub(super) fn record_with_boundary(
        &self,
        metrics: &mut QueryMetrics,
        boundary: &'static str,
        unresolved_contract: Option<&'static str>,
    ) {
        metrics.record_graph_budget_tracking_with_boundary_and_enforcement(
            self.operation,
            boundary,
            unresolved_contract.map(str::to_string),
            self.budget_bytes,
            self.enforcement,
            self.allocator_high_water_bytes,
            self.status,
            self.context_facts.clone(),
        );
    }
}

#[cfg(feature = "cugraph")]
pub(super) fn graph_execution_budget(
    context: &ExecutionContext,
    graph_memory_limit_bytes: Option<u64>,
    budget_boundary: &'static str,
    operation: &'static str,
) -> Result<GraphExecutionBudget> {
    let token = graph_execution_budget_token(context, graph_memory_limit_bytes, budget_boundary, operation)?;
    let source_runtime = context.runtime()?;
    let runtime = cudf::Runtime::builder()
        .with_device(source_runtime.device())
        .with_stream(source_runtime.stream().clone())
        .with_memory_resource(token.memory_resource().clone())
        .build()
        .map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.graph_algorithm.execution_budget",
                "failed to build reservation-backed cudf runtime for cuGraph execution",
                error,
            )
            .with_fact("operation", operation)
            .with_fact("budget_boundary", budget_boundary)
            .with_fact("graph_memory_limit_bytes", graph_memory_limit_bytes.unwrap_or(usize::MAX as u64))
        })?
        .inherit_retained_lifetime_from(&source_runtime)
        .map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.graph_algorithm.execution_budget",
                "failed to retain query-service lifetime for cuGraph execution runtime",
                error,
            )
            .with_fact("operation", operation)
            .with_fact("budget_boundary", budget_boundary)
        })?;
    Ok(GraphExecutionBudget { runtime: std::sync::Arc::new(runtime), token })
}

#[cfg(feature = "cugraph")]
fn graph_execution_budget_token(
    context: &ExecutionContext,
    graph_memory_limit_bytes: Option<u64>,
    budget_boundary: &'static str,
    operation: &'static str,
) -> Result<cudf::ExecutionBudgetToken> {
    let limit_bytes = graph_memory_limit_bytes.unwrap_or(usize::MAX as u64);

    let limit = usize::try_from(limit_bytes).map_err(|_| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.execution_budget",
            "graph memory reservation does not fit this platform's usize budget limit",
        )
        .with_fact("operation", operation)
        .with_fact("budget_boundary", budget_boundary)
        .with_fact("graph_memory_limit_bytes", limit_bytes)
    })?;
    context.execution_budget_token(cudf::ExecutionBudgetRequest::new(limit).with_boundary(budget_boundary)).map_err(
        |error| {
            error
                .with_fact("operation", operation)
                .with_fact("budget_boundary", budget_boundary)
                .with_fact("graph_memory_limit_bytes", limit_bytes)
        },
    )
}

#[cfg(feature = "cugraph")]
pub(super) fn attach_graph_budget_observation(
    result: Result<GraphAlgorithmExecutionResult>,
    token: &cudf::ExecutionBudgetToken,
    operation: &'static str,
) -> Result<GraphAlgorithmExecutionResult> {
    match result {
        Ok(mut result) => {
            if let Some(observation) = graph_budget_observation(token, operation, "reservation_enforced") {
                result.budget_observations.push(observation);
            }
            Ok(result)
        },
        Err(error) => Err(attach_graph_budget_error_facts(error, token, operation)),
    }
}

#[cfg(feature = "cugraph")]
pub(super) fn graph_budget_observation(
    token: &cudf::ExecutionBudgetToken,
    operation: &'static str,
    status: &'static str,
) -> Option<GraphBudgetObservation> {
    let usage = token.reservation_usage().ok()?;
    let arena_usage = token.reservation_arena_usage().ok().flatten();
    Some(GraphBudgetObservation::from_reservation_usage(operation, token, usage, arena_usage, status))
}

#[cfg(feature = "cugraph")]
const GRAPH_BUDGET_CONTEXT_FACT_ERROR_KEYS: &[(&str, &str)] = &[
    ("reservation_id", "graph_budget_reservation_id"),
    ("reservation_attached", "graph_budget_reservation_attached"),
    ("allocator_current_bytes", "graph_budget_allocator_current_bytes"),
    ("allocator_peak_aligned_bytes", "graph_budget_allocator_peak_aligned_bytes"),
    ("allocator_current_aligned_bytes", "graph_budget_allocator_current_aligned_bytes"),
    ("allocator_total_bytes", "graph_budget_allocator_total_bytes"),
    ("allocator_total_allocations", "graph_budget_allocator_total_allocations"),
    ("allocator_current_allocations", "graph_budget_allocator_current_allocations"),
    ("allocator_peak_allocations", "graph_budget_allocator_peak_allocations"),
    ("allocator_failed_allocation_count", "graph_budget_failed_allocation_count"),
    ("allocator_largest_failed_allocation_bytes", "graph_budget_largest_failed_allocation_bytes"),
    ("arena_id", "graph_budget_arena_id"),
    ("arena_active", "graph_budget_arena_active"),
    ("arena_active_members", "graph_budget_arena_active_members"),
    ("arena_limit_bytes", "graph_budget_arena_limit_bytes"),
    ("arena_current_bytes", "graph_budget_arena_current_bytes"),
    ("arena_peak_bytes", "graph_budget_arena_peak_bytes"),
    ("arena_failed_allocation_count", "graph_budget_arena_failed_allocation_count"),
    ("arena_largest_failed_allocation_bytes", "graph_budget_arena_largest_failed_allocation_bytes"),
];

#[cfg(feature = "cugraph")]
fn graph_budget_error_fact_key(context_key: &str) -> Option<&'static str> {
    GRAPH_BUDGET_CONTEXT_FACT_ERROR_KEYS
        .iter()
        .find_map(|(candidate, fact_key)| (*candidate == context_key).then_some(*fact_key))
}

#[cfg(feature = "cugraph")]
pub(super) fn attach_graph_budget_error_facts(
    mut error: QueryError,
    token: &cudf::ExecutionBudgetToken,
    operation: &'static str,
) -> QueryError {
    let status = if error.code() == QueryErrorCode::CudfResourceExhausted {
        "reservation_resource_exhausted"
    } else {
        "reservation_failed"
    };
    let Some(observation) = graph_budget_observation(token, operation, status) else {
        return error;
    };
    error = error
        .with_fact("graph_budget_operation", observation.operation)
        .with_fact("graph_budget_enforcement", observation.enforcement)
        .with_fact("graph_budget_limit_bytes", observation.budget_bytes)
        .with_fact("graph_budget_status", observation.status);
    if let Some(peak) = observation.allocator_high_water_bytes {
        error = error.with_fact("graph_budget_peak_bytes", peak);
    }
    for (key, value) in observation.context_facts {
        if let Some(fact_key) = graph_budget_error_fact_key(&key) {
            error = error.with_fact(fact_key, value);
        }
    }
    error
}

#[cfg(feature = "cugraph")]
pub(super) fn record_graph_budget_observation_from_error(metrics: &mut QueryMetrics, error: &QueryError) {
    record_graph_budget_observation_from_error_with_boundary(
        metrics,
        error,
        crate::graph_memory::GRAPH_MEMORY_BOUNDARY,
        None,
    );
}

/// Returns the exact reservation high-water retained by a graph-execution
/// error after the token-owning executor sealed its structured evidence.
///
/// This deliberately requires the complete token projection rather than
/// treating an arbitrary numeric error fact as a physical peak. The caller
/// may use the returned value for graph-claim calibration while the existing
/// recorder retains the full diagnostic row.
#[cfg(feature = "cugraph")]
pub(super) fn graph_budget_peak_from_error(error: &QueryError) -> Option<u64> {
    error.fact_value("graph_budget_operation")?;
    error.fact_value("graph_budget_enforcement")?;
    error.fact_value("graph_budget_status")?;
    error.fact_value("graph_budget_limit_bytes")?.parse::<u64>().ok()?;
    error.fact_value("graph_budget_peak_bytes")?.parse::<u64>().ok()
}

#[cfg(feature = "cugraph")]
pub(super) fn record_graph_budget_observation_from_error_with_boundary(
    metrics: &mut QueryMetrics,
    error: &QueryError,
    boundary: &'static str,
    unresolved_contract: Option<&'static str>,
) {
    let Some(operation) = error.fact_value("graph_budget_operation") else {
        return;
    };
    let Some(budget_bytes) = error.fact_value("graph_budget_limit_bytes").and_then(|value| value.parse::<u64>().ok())
    else {
        return;
    };
    let peak_bytes = error.fact_value("graph_budget_peak_bytes").and_then(|value| value.parse::<u64>().ok());
    let status = error.fact_value("graph_budget_status").unwrap_or("reservation_failed");
    let enforcement = error.fact_value("graph_budget_enforcement").unwrap_or("stream_bound_reservation");
    let mut context_facts = BTreeMap::new();
    for (context_key, fact_key) in GRAPH_BUDGET_CONTEXT_FACT_ERROR_KEYS {
        if let Some(value) = error.fact_value(fact_key) {
            context_facts.insert((*context_key).to_string(), value.to_string());
        }
    }
    metrics.record_graph_budget_tracking_with_boundary_and_enforcement(
        operation.to_string(),
        boundary,
        unresolved_contract.map(str::to_string),
        budget_bytes,
        enforcement.to_string(),
        peak_bytes,
        status.to_string(),
        context_facts,
    );
}

#[cfg(feature = "cugraph")]
#[derive(Debug, Clone, Copy)]
pub(super) enum EdgeColumnRole {
    Vertex,
    Weight,
    EdgeId,
}

#[cfg(feature = "cugraph")]
pub(in crate::exec) fn graph_memory_model_input_for_frame(
    frame: &DataFrame,
    spec: &crate::plan::GraphAlgorithmSpec,
    input_edges: u64,
    allow_nulls: bool,
) -> Result<crate::graph_memory::GraphMemoryModelInput> {
    let source_type = inspect_edge_column(frame, &spec.edge_columns.source, EdgeColumnRole::Vertex, allow_nulls)?;
    let destination_type =
        inspect_edge_column(frame, &spec.edge_columns.destination, EdgeColumnRole::Vertex, allow_nulls)?;
    if source_type.id() != destination_type.id() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "nexus_query_engine.graph_algorithm.graph_memory_input",
            "cuGraph source and destination columns must have the same vertex dtype",
        )
        .with_fact("src_col", &spec.edge_columns.source)
        .with_fact("dst_col", &spec.edge_columns.destination)
        .with_fact("src_cudf_dtype", format!("{source_type:?}"))
        .with_fact("dst_cudf_dtype", format!("{destination_type:?}")));
    }
    if let Some(edge_id) = &spec.edge_columns.edge_id {
        let edge_id_type = inspect_edge_column(frame, edge_id, EdgeColumnRole::EdgeId, allow_nulls)?;
        if edge_id_type.id() != source_type.id() {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                "nexus_query_engine.graph_algorithm.graph_memory_input",
                "cuGraph edge-id column dtype must match the graph vertex dtype",
            )
            .with_fact("edge_id_col", edge_id)
            .with_fact("edge_id_cudf_dtype", format!("{edge_id_type:?}"))
            .with_fact("vertex_cudf_dtype", format!("{source_type:?}")));
        }
    }

    let weight_width_bytes = if spec.algorithm.consumes_weight() {
        spec.edge_columns
            .weight
            .as_ref()
            .map(|weight| {
                inspect_edge_column(frame, weight, EdgeColumnRole::Weight, allow_nulls).and_then(dtype_width_bytes)
            })
            .transpose()?
    } else {
        None
    };
    let vertex_width_bytes = dtype_width_bytes(source_type)?;
    Ok(crate::graph_memory::graph_memory_model_input(
        input_edges,
        vertex_width_bytes,
        &spec.edge_columns,
        spec.algorithm.consumes_weight(),
        weight_width_bytes,
    ))
}

/// Returns materialized candidate-pair evidence for the similarity memory
/// model.  This runs after the normal side-input materialization boundary, so
/// it neither guesses from graph cardinality nor evaluates the relation a
/// second time.
#[cfg(feature = "cugraph")]
fn graph_similarity_pair_memory_evidence(
    spec: &crate::plan::GraphAlgorithmSpec,
    side_inputs: &BTreeMap<NodeId, GpuDataFrame>,
) -> Result<Option<(u64, u64, crate::graph_memory::GraphPairCardinalitySource)>> {
    let crate::plan::GraphAlgorithmParams::Similarity(params) = &spec.params else {
        return Ok(None);
    };
    let pair_input = side_inputs.get(&params.vertex_pairs.input).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.graph_memory_input",
            "similarity graph memory admission is missing the vertex-pair side input",
        )
        .with_fact("algorithm", spec.algorithm.as_str())
        .with_fact("input_role", "vertex_pairs")
        .with_fact("side_input_node", params.vertex_pairs.input)
    })?;
    let frame = pair_input.frame();
    let first = inspect_edge_column(frame, &params.vertex_pairs.first_column, EdgeColumnRole::Vertex, false).map_err(
        |error| {
            error
                .with_fact_if_absent("algorithm", spec.algorithm.as_str())
                .with_fact_if_absent("input_role", "vertex_pairs")
                .with_fact_if_absent("pair_column_role", "first")
        },
    )?;
    let second = inspect_edge_column(frame, &params.vertex_pairs.second_column, EdgeColumnRole::Vertex, false)
        .map_err(|error| {
            error
                .with_fact_if_absent("algorithm", spec.algorithm.as_str())
                .with_fact_if_absent("input_role", "vertex_pairs")
                .with_fact_if_absent("pair_column_role", "second")
        })?;
    if first.id() != second.id() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "nexus_query_engine.graph_algorithm.graph_memory_input",
            "similarity vertex-pair columns must have matching physical types",
        )
        .with_fact("algorithm", spec.algorithm.as_str())
        .with_fact("reason", "vertex_pair_dtype_mismatch")
        .with_fact("first_column", &params.vertex_pairs.first_column)
        .with_fact("first_cudf_dtype", format!("{first:?}"))
        .with_fact("second_column", &params.vertex_pairs.second_column)
        .with_fact("second_cudf_dtype", format!("{second:?}")));
    }
    Ok(Some((
        frame.num_rows() as u64,
        dtype_width_bytes(first)?,
        crate::graph_memory::GraphPairCardinalitySource::RuntimeMaterialized,
    )))
}

#[cfg(feature = "cugraph")]
pub(super) fn validate_edge_column(frame: &DataFrame, column: &str, role: EdgeColumnRole) -> Result<cudf::DataType> {
    inspect_edge_column(frame, column, role, false)
}

#[cfg(feature = "cugraph")]
fn inspect_edge_column(
    frame: &DataFrame,
    column: &str,
    role: EdgeColumnRole,
    allow_nulls: bool,
) -> Result<cudf::DataType> {
    let series = frame.series(column).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.graph_algorithm.validate_edge_column",
            "failed to access cuGraph edge-list column",
            error,
        )
        .with_fact("column", column)
    })?;
    if !allow_nulls && series.null_count() != 0 {
        return Err(QueryError::new(
            QueryErrorCode::GraphInputInvalid,
            "nexus_query_engine.graph_algorithm.validate_edge_column",
            "cuGraph materialized input columns must be non-null",
        )
        .with_fact("column", column)
        .with_fact("null_count", series.null_count()));
    }
    let dtype = series.dtype().map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.graph_algorithm.validate_edge_column",
            "failed to inspect cuGraph edge-list column type",
            error,
        )
        .with_fact("column", column)
    })?;
    let supported = match role {
        EdgeColumnRole::Vertex => matches!(dtype.id(), cudf::TypeId::Int32 | cudf::TypeId::Int64),
        EdgeColumnRole::Weight => matches!(dtype.id(), cudf::TypeId::Float32 | cudf::TypeId::Float64),
        EdgeColumnRole::EdgeId => matches!(dtype.id(), cudf::TypeId::Int32 | cudf::TypeId::Int64),
    };
    if supported {
        return Ok(dtype);
    }
    Err(QueryError::new(
        QueryErrorCode::UnsupportedType,
        "nexus_query_engine.graph_algorithm.validate_edge_column",
        "cuGraph edge-list column has unsupported dtype",
    )
    .with_fact("column", column)
    .with_fact(
        "role",
        match role {
            EdgeColumnRole::Vertex => "vertex",
            EdgeColumnRole::Weight => "weight",
            EdgeColumnRole::EdgeId => "edge_id",
        },
    )
    .with_fact("cudf_dtype", format!("{dtype:?}")))
}

#[cfg(feature = "cugraph")]
pub(super) fn dtype_width_bytes(dtype: cudf::DataType) -> Result<u64> {
    match dtype.id() {
        cudf::TypeId::Int32 | cudf::TypeId::Float32 => Ok(4),
        cudf::TypeId::Int64 | cudf::TypeId::Float64 => Ok(8),
        other => Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "nexus_query_engine.graph_algorithm.graph_memory_input",
            "graph memory model only supports fixed-width cuGraph edge types",
        )
        .with_fact("cudf_dtype", format!("{other:?}"))),
    }
}
