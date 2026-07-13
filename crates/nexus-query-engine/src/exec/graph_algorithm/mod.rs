#[cfg(feature = "cugraph")]
use std::collections::BTreeMap;
#[cfg(feature = "cugraph")]
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(feature = "cugraph")]
use arrow::datatypes::SchemaRef;

#[cfg(feature = "cugraph")]
use super::types::{GpuResidentGraphBuildJoin, GpuResidentGraphBuildWait, GpuResidentGraphTransposition};
use super::*;

#[cfg(feature = "cugraph")]
mod centrality;
#[cfg(feature = "cugraph")]
mod community;
#[cfg(feature = "cugraph")]
mod layout;
#[cfg(feature = "cugraph")]
mod params;
#[cfg(feature = "cugraph")]
mod structure;
#[cfg(feature = "cugraph")]
mod traversal;

#[cfg(feature = "cugraph")]
use centrality::*;
#[cfg(feature = "cugraph")]
use community::*;
#[cfg(feature = "cugraph")]
use layout::*;
#[cfg(feature = "cugraph")]
use params::*;
#[cfg(feature = "cugraph")]
use structure::*;
#[cfg(feature = "cugraph")]
use traversal::{execute_bfs, execute_sssp};

#[cfg(feature = "cugraph")]
static FORCE_NEXT_GRAPH_ADAPTIVE_FULL_ATTEMPT_RESOURCE_EXHAUSTED: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "cugraph")]
static FORCE_NEXT_GRAPH_ADAPTIVE_CHUNK_ATTEMPT_RESOURCE_EXHAUSTED: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "cugraph")]
thread_local! {
    // Graph statistics probing is synchronous today. Keeping this test hook
    // thread-local prevents one integration test from consuming another test's
    // forced optional-resource exhaustion.
    static FORCE_NEXT_GRAPH_STATISTICS_OPTIONAL_RESOURCE_EXHAUSTED: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

#[cfg(feature = "cugraph")]
pub(in crate::exec) enum RuntimeGraphSpecResolution {
    Unresolved,
    CheckedNoRewrite,
    Rewritten(crate::plan::GraphAlgorithmSpec),
}

#[cfg(feature = "cugraph")]
pub(in crate::exec) struct RuntimeGraphStatisticsProbe {
    pub(in crate::exec) vertex_estimate: crate::graph_memory::GraphVertexEstimate,
    pub(in crate::exec) statistics: crate::graph_memory::GraphStatistics,
}

#[cfg(feature = "cugraph")]
impl RuntimeGraphSpecResolution {
    fn from_optional(spec: Option<crate::plan::GraphAlgorithmSpec>) -> Self {
        match spec {
            Some(spec) => Self::Rewritten(spec),
            None => Self::CheckedNoRewrite,
        }
    }

    fn spec_or<'a>(&'a self, fallback: &'a crate::plan::GraphAlgorithmSpec) -> &'a crate::plan::GraphAlgorithmSpec {
        match self {
            Self::Unresolved | Self::CheckedNoRewrite => fallback,
            Self::Rewritten(spec) => spec,
        }
    }
}

#[cfg(feature = "cugraph")]
#[doc(hidden)]
pub fn force_next_graph_adaptive_full_attempt_resource_exhausted_for_test() {
    FORCE_NEXT_GRAPH_ADAPTIVE_FULL_ATTEMPT_RESOURCE_EXHAUSTED.store(true, Ordering::SeqCst);
}

#[cfg(feature = "cugraph")]
#[doc(hidden)]
pub fn force_next_graph_adaptive_chunk_attempt_resource_exhausted_for_test() {
    FORCE_NEXT_GRAPH_ADAPTIVE_CHUNK_ATTEMPT_RESOURCE_EXHAUSTED.store(true, Ordering::SeqCst);
}

#[cfg(feature = "cugraph")]
#[doc(hidden)]
pub fn force_next_graph_statistics_optional_resource_exhausted_for_test() {
    FORCE_NEXT_GRAPH_STATISTICS_OPTIONAL_RESOURCE_EXHAUSTED.with(|force| force.set(true));
}

#[cfg(not(feature = "cugraph"))]
pub(super) fn execute_graph_algorithm(
    _context: &ExecutionContext,
    node_id: NodeId,
    node: &crate::plan::GraphAlgorithmNode,
    input: GpuDataFrame,
    side_inputs: std::collections::BTreeMap<NodeId, GpuDataFrame>,
    _metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let input_rows = input.num_rows();
    let input_columns = input.num_columns();
    let side_input_count = side_inputs.len();
    Err(QueryError::new(
        QueryErrorCode::UnsupportedOperator,
        "nexus_query_engine.graph_algorithm.execute",
        "cuGraph graph algorithm execution requires the cugraph Cargo feature",
    )
    .with_fact("node_id", node_id)
    .with_fact("algorithm", node.spec.algorithm.as_str())
    .with_fact("input_rows", input_rows)
    .with_fact("input_columns", input_columns)
    .with_fact("side_input_count", side_input_count)
    .with_fact("required_feature", "cugraph"))
}

#[cfg(feature = "cugraph")]
pub(super) fn execute_graph_algorithm(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &crate::plan::GraphAlgorithmNode,
    input: GpuDataFrame,
    side_inputs: BTreeMap<NodeId, GpuDataFrame>,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    execute_graph_algorithm_with_cache_requirement(context, node_id, node, input, side_inputs, None, metrics)
}

#[cfg(feature = "cugraph")]
pub(super) fn execute_graph_algorithm_cache_hit(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &crate::plan::GraphAlgorithmNode,
    input: GpuDataFrame,
    side_inputs: BTreeMap<NodeId, GpuDataFrame>,
    preparation_pin_id: u64,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    execute_graph_algorithm_with_cache_requirement(
        context,
        node_id,
        node,
        input,
        side_inputs,
        Some(preparation_pin_id),
        metrics,
    )
}

#[cfg(feature = "cugraph")]
fn execute_graph_algorithm_with_cache_requirement(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &crate::plan::GraphAlgorithmNode,
    input: GpuDataFrame,
    side_inputs: BTreeMap<NodeId, GpuDataFrame>,
    preparation_pin_id: Option<u64>,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let input_rows = input.num_rows() as u64;
    let input_columns = input.num_columns() as u64;
    let input_graph_residency = input.graph_residency().cloned();
    let prior_null_edge_rows_dropped = graph_input_null_edge_rows_dropped(metrics, node.input);
    let null_filter_scope = super::edge_null_filter::EdgeNullFilterScope {
        weight: node.spec.algorithm.consumes_weight(),
        edge_id: node.spec.edge_columns.edge_id.is_some(),
    };
    let filtered_input = super::edge_null_filter::filter_null_edge_rows(
        context,
        node_id,
        "graph_edge_null_filter",
        input.frame().clone(),
        &node.spec.edge_columns,
        null_filter_scope,
        metrics,
    )?;
    let sanitized_input_rows = filtered_input.num_rows() as u64;
    let null_edge_rows_dropped =
        prior_null_edge_rows_dropped.saturating_add(input_rows.saturating_sub(sanitized_input_rows));
    validate_betweenness_static_seed_policy(&node.spec)?;
    let graph_memory_limit_bytes =
        admit_graph_memory(context, node_id, &node.spec, &filtered_input, sanitized_input_rows, metrics)?;
    let runtime_spec = runtime_graph_spec_resolution_after_bounded_seed_validation(&node.spec, false);
    let mut sanitized_input = GpuDataFrame::try_new_with_source_residency(filtered_input, &input)?;
    if null_edge_rows_dropped == 0
        && let Some(graph_residency) = input_graph_residency
    {
        sanitized_input = sanitized_input.with_graph_residency(graph_residency);
    }
    execute_graph_algorithm_sanitized(
        context,
        node_id,
        node,
        runtime_spec,
        sanitized_input,
        side_inputs,
        graph_memory_limit_bytes,
        input_rows,
        input_columns,
        null_edge_rows_dropped,
        preparation_pin_id,
        metrics,
    )
}

#[cfg(feature = "cugraph")]
pub(super) fn execute_graph_algorithm_sanitized(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &crate::plan::GraphAlgorithmNode,
    runtime_spec: RuntimeGraphSpecResolution,
    input: GpuDataFrame,
    side_inputs: BTreeMap<NodeId, GpuDataFrame>,
    graph_memory_limit_bytes: Option<u64>,
    input_rows: u64,
    input_columns: u64,
    null_edge_rows_dropped: u64,
    preparation_pin_id: Option<u64>,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let boundary_before = metrics.boundary.clone();
    let graph_budget_event_start = metrics.budgeted_execution_events.len();
    let pool_usage_before = graph_pool_usage_snapshot(context);
    let timer = Instant::now();
    let runtime_spec = match runtime_spec {
        RuntimeGraphSpecResolution::Unresolved => resolve_runtime_graph_spec_for_execution(
            context,
            input.frame(),
            &node.spec,
            graph_memory_limit_bytes,
            metrics,
        )?,
        runtime_spec => runtime_spec,
    };
    let spec = runtime_spec.spec_or(&node.spec);
    let executor = graph_algorithm_executor(spec.algorithm)?;
    let adaptive_retry_inputs = graph_adaptive_retry_seed_batches(spec).map(|batches| {
        (
            GraphAlgorithmExecutionInputs {
                edge_list: input.clone(),
                side_inputs: side_inputs.clone(),
                graph_memory_limit_bytes,
                preparation_pin_id,
            },
            batches,
        )
    });
    let inputs =
        GraphAlgorithmExecutionInputs { edge_list: input, side_inputs, graph_memory_limit_bytes, preparation_pin_id };
    let result = match execute_graph_algorithm_attempt(context, spec, inputs, executor) {
        Ok(result) => result,
        Err(error) if should_retry_graph_resource_exhausted(context, &error, adaptive_retry_inputs.as_ref()) => {
            let Some((retry_inputs, seed_batches)) = adaptive_retry_inputs else {
                return Err(error);
            };
            record_graph_adaptive_memory_history_from_error(context, &node.spec, &error);
            record_graph_budget_observation_from_error(metrics, &error);
            metrics.record_adaptive_runtime_action_counts(1, 0, 0);
            let retry_result = execute_chunked_betweenness_retry(context, spec, retry_inputs, seed_batches, executor);
            if let Err(retry_error) = &retry_result {
                record_graph_adaptive_memory_history_from_error(context, &node.spec, retry_error);
                record_graph_budget_observation_from_error(metrics, retry_error);
            }
            retry_result?
        },
        Err(error) => return Err(error),
    };
    let pool_usage_after = graph_pool_usage_snapshot(context);
    let elapsed = timer.elapsed();
    metrics.operator_time += elapsed;
    for observation in &result.budget_observations {
        observation.record(metrics);
    }

    let gpu = GpuDataFrame::new(result.frame);
    let output_device_bytes = gpu.device_allocated_size_bytes().ok();
    let budget_graph_peak_bytes = metrics
        .budgeted_execution_events
        .iter()
        .skip(graph_budget_event_start)
        .filter(|event| event.boundary == crate::graph_memory::GRAPH_MEMORY_BOUNDARY)
        .filter_map(|event| event.allocator_high_water_bytes)
        .max();
    let pool_graph_peak_bytes = graph_pool_high_water_delta(pool_usage_before, pool_usage_after);
    let measured_graph_peak_bytes = budget_graph_peak_bytes.or(pool_graph_peak_bytes);
    let graph_peak_bytes = measured_graph_peak_bytes.or(Some(0));
    metrics.record_graph_model_error_ratio(
        &node_id.to_string(),
        node.spec.algorithm.as_str(),
        measured_graph_peak_bytes,
    );
    let boundary_delta = boundary_delta(&boundary_before, &metrics.boundary);
    metrics.record_graph_algorithm_runtime(crate::metrics::GraphAlgorithmRuntimeMetrics {
        query_id: None,
        node_id: node_id.to_string(),
        algorithm: node.spec.algorithm.as_str().to_string(),
        runtime_outcome: "succeeded".to_string(),
        input_boundary: "gpu_resident".to_string(),
        output_boundary: "gpu_resident".to_string(),
        construction_policy: node.spec.graph_options.policy.as_str().to_string(),
        source_column: node.spec.edge_columns.source.clone(),
        destination_column: node.spec.edge_columns.destination.clone(),
        weight_column: node.spec.edge_columns.weight.clone(),
        directed: node.spec.graph_options.directed,
        renumber: node.spec.graph_options.renumber,
        algorithm_parameters: graph_algorithm_parameters(&node.spec.params),
        algorithm_metadata: graph_algorithm_metadata_with_null_drops(result.algorithm_metadata, null_edge_rows_dropped),
        input_rows,
        input_columns,
        output_rows: gpu.num_rows() as u64,
        output_columns: gpu.num_columns() as u64,
        output_device_bytes,
        graph_peak_bytes,
        elapsed_ns: u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX),
        gpu_resident_input_handoffs: boundary_delta.gpu_resident_input_handoffs,
        host_to_device_uploads: boundary_delta.host_to_device_uploads,
        device_to_host_materializations: boundary_delta.device_to_host_materializations,
    });
    Ok(gpu)
}

#[cfg(feature = "cugraph")]
fn graph_pool_usage_snapshot(context: &ExecutionContext) -> Option<cudf::DeviceMemoryPoolUsage> {
    context.runtime().device_memory_manager().ok()?.pool_usage().ok()
}

#[cfg(feature = "cugraph")]
fn graph_pool_high_water_delta(
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

#[cfg(feature = "cugraph")]
fn execute_graph_algorithm_attempt(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
    executor: GraphAlgorithmExecutor,
) -> Result<GraphAlgorithmExecutionResult> {
    if context.native_pipeline_execution_mode() == crate::mode::NativePipelineExecutionMode::Adaptive
        && graph_adaptive_retry_seed_batches(spec).is_some()
        && FORCE_NEXT_GRAPH_ADAPTIVE_FULL_ATTEMPT_RESOURCE_EXHAUSTED.swap(false, Ordering::SeqCst)
    {
        let mut error = QueryError::new(
            QueryErrorCode::CudfResourceExhausted,
            "nexus_query_engine.graph_algorithm.adaptive_retry_test",
            "forced graph adaptive retry resource exhaustion",
        )
        .with_fact("reason", "forced_graph_adaptive_oom_retry_test")
        .with_fact("algorithm", spec.algorithm.as_str());
        if let Some(limit) = inputs.graph_memory_limit_bytes {
            error = error
                .with_fact("graph_budget_limit_bytes", limit)
                .with_fact("graph_budget_peak_bytes", limit)
                .with_fact("graph_budget_status", "reservation_resource_exhausted");
        }
        return Err(error);
    }
    executor(context, spec, inputs)
}

#[cfg(feature = "cugraph")]
fn should_retry_graph_resource_exhausted(
    context: &ExecutionContext,
    error: &QueryError,
    retry_inputs: Option<&(GraphAlgorithmExecutionInputs, Vec<crate::plan::GraphBetweennessSeeds>)>,
) -> bool {
    let Some((inputs, _)) = retry_inputs else {
        return false;
    };
    context.native_pipeline_execution_mode() == crate::mode::NativePipelineExecutionMode::Adaptive
        && error.code() == QueryErrorCode::CudfResourceExhausted
        && inputs.graph_memory_limit_bytes.is_some()
        && graph_adaptive_memory_observation_from_error(error).is_some()
}

#[cfg(feature = "cugraph")]
fn resolve_runtime_graph_spec_for_execution(
    context: &ExecutionContext,
    input: &DataFrame,
    spec: &crate::plan::GraphAlgorithmSpec,
    graph_memory_limit_bytes: Option<u64>,
    metrics: &mut QueryMetrics,
) -> Result<RuntimeGraphSpecResolution> {
    let Some(operation) = betweenness_seed_policy_operation(spec) else {
        return Ok(RuntimeGraphSpecResolution::CheckedNoRewrite);
    };
    let Some(limit_bytes) = graph_memory_limit_bytes else {
        return graph_spec_with_runtime_betweenness_seeds(input, spec).map(RuntimeGraphSpecResolution::from_optional);
    };
    let budget =
        graph_execution_budget(context, Some(limit_bytes), crate::graph_memory::GRAPH_MEMORY_BOUNDARY, operation)?;
    let result = graph_spec_with_runtime_betweenness_seeds_in_runtime(
        input,
        spec,
        Some(budget.runtime.as_ref()),
        budget.token.as_ref(),
    );
    match result {
        Ok(spec) => {
            if let Some(observation) =
                graph_budget_observation(budget.token.as_ref(), operation, "reservation_enforced")
            {
                observation.record(metrics);
            }
            Ok(RuntimeGraphSpecResolution::from_optional(spec))
        },
        Err(error) => {
            let error = attach_graph_budget_error_facts(error, budget.token.as_ref(), operation);
            record_graph_budget_observation_from_error(metrics, &error);
            Err(error)
        },
    }
}

#[cfg(feature = "cugraph")]
fn betweenness_seed_policy_operation(spec: &crate::plan::GraphAlgorithmSpec) -> Option<&'static str> {
    match &spec.params {
        crate::plan::GraphAlgorithmParams::BetweennessCentrality(_) => {
            Some("nexus_query_engine.graph_algorithm.betweenness_seed_policy")
        },
        crate::plan::GraphAlgorithmParams::EdgeBetweennessCentrality(_) => {
            Some("nexus_query_engine.graph_algorithm.edge_betweenness_seed_policy")
        },
        _ => None,
    }
}

#[cfg(feature = "cugraph")]
fn graph_spec_with_runtime_betweenness_seeds(
    input: &DataFrame,
    spec: &crate::plan::GraphAlgorithmSpec,
) -> Result<Option<crate::plan::GraphAlgorithmSpec>> {
    graph_spec_with_runtime_betweenness_seeds_in_runtime(input, spec, None, None)
}

#[cfg(feature = "cugraph")]
fn graph_spec_with_runtime_betweenness_seeds_in_runtime(
    input: &DataFrame,
    spec: &crate::plan::GraphAlgorithmSpec,
    runtime: Option<&cudf::Runtime>,
    token: Option<&cudf::ExecutionBudgetToken>,
) -> Result<Option<crate::plan::GraphAlgorithmSpec>> {
    let (seeds, k, operation) = match &spec.params {
        crate::plan::GraphAlgorithmParams::BetweennessCentrality(params) => {
            (params.seeds.as_ref(), params.k, "nexus_query_engine.graph_algorithm.betweenness_seed_policy")
        },
        crate::plan::GraphAlgorithmParams::EdgeBetweennessCentrality(params) => {
            (params.seeds.as_ref(), params.k, "nexus_query_engine.graph_algorithm.edge_betweenness_seed_policy")
        },
        _ => return Ok(None),
    };

    if seeds.is_some() {
        validate_betweenness_seed_policy_for_vertex_count(spec, 0)?;
        return Ok(None);
    }

    let vertices = sorted_graph_vertices_i64(input, spec, operation, runtime, token)?;
    let vertex_count = vertices.len() as u64;
    validate_betweenness_seed_policy_for_vertex_count(spec, vertex_count)?;
    if let Some(k) = k {
        let k_usize = usize::try_from(k).map_err(|_| {
            QueryError::new(QueryErrorCode::InvalidPlan, operation, "betweenness k does not fit this platform's usize")
                .with_fact("algorithm", spec.algorithm.as_str())
                .with_fact("k", k)
        })?;
        return graph_spec_with_betweenness_seeds(
            spec,
            crate::plan::GraphBetweennessSeeds::I64(vertices.into_iter().take(k_usize).collect()),
        )
        .map(Some);
    }

    Ok(None)
}

#[cfg(feature = "cugraph")]
pub(in crate::exec) fn validate_betweenness_seed_policy_for_vertex_count(
    spec: &crate::plan::GraphAlgorithmSpec,
    vertex_count: u64,
) -> Result<()> {
    let (seeds, k, exact_vertex_threshold, operation) = match &spec.params {
        crate::plan::GraphAlgorithmParams::BetweennessCentrality(params) => (
            params.seeds.as_ref(),
            params.k,
            params.exact_vertex_threshold,
            "nexus_query_engine.graph_algorithm.betweenness_seed_policy",
        ),
        crate::plan::GraphAlgorithmParams::EdgeBetweennessCentrality(params) => (
            params.seeds.as_ref(),
            params.k,
            params.exact_vertex_threshold,
            "nexus_query_engine.graph_algorithm.edge_betweenness_seed_policy",
        ),
        _ => return Ok(()),
    };

    validate_betweenness_static_seed_policy_parts(spec.algorithm, seeds, k, operation)?;
    if let Some(k) = k {
        if k > vertex_count {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                operation,
                "betweenness k exceeds the actual graph vertex count",
            )
            .with_fact("reason", "betweenness_k_exceeds_vertex_count")
            .with_fact("algorithm", spec.algorithm.as_str())
            .with_fact("k", k)
            .with_fact("actual_vertex_count", vertex_count));
        }
        return Ok(());
    }
    if seeds.is_some() {
        return Ok(());
    }

    if vertex_count > exact_vertex_threshold {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            operation,
            "exact betweenness requires explicit seeds or k above the configured vertex threshold",
        )
        .with_fact("reason", "exact_betweenness_vertex_threshold_exceeded")
        .with_fact("algorithm", spec.algorithm.as_str())
        .with_fact("actual_vertex_count", vertex_count)
        .with_fact("exact_vertex_threshold", exact_vertex_threshold)
        .with_fact("required_option", "seeds_or_k"));
    }
    Ok(())
}

#[cfg(feature = "cugraph")]
pub(in crate::exec) fn validate_betweenness_static_seed_policy(spec: &crate::plan::GraphAlgorithmSpec) -> Result<()> {
    let (seeds, k, operation) = match &spec.params {
        crate::plan::GraphAlgorithmParams::BetweennessCentrality(params) => {
            (params.seeds.as_ref(), params.k, "nexus_query_engine.graph_algorithm.betweenness_seed_policy")
        },
        crate::plan::GraphAlgorithmParams::EdgeBetweennessCentrality(params) => {
            (params.seeds.as_ref(), params.k, "nexus_query_engine.graph_algorithm.edge_betweenness_seed_policy")
        },
        _ => return Ok(()),
    };
    validate_betweenness_static_seed_policy_parts(spec.algorithm, seeds, k, operation)
}

#[cfg(feature = "cugraph")]
fn validate_betweenness_static_seed_policy_parts(
    algorithm: crate::plan::GraphAlgorithm,
    seeds: Option<&crate::plan::GraphBetweennessSeeds>,
    k: Option<u64>,
    operation: &'static str,
) -> Result<()> {
    if seeds.is_some() && k.is_some() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            operation,
            "betweenness seeds and k are mutually exclusive",
        )
        .with_fact("algorithm", algorithm.as_str()));
    }
    if let Some(seeds) = seeds {
        if seeds.is_empty() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                operation,
                "betweenness seed list must not be empty",
            )
            .with_fact("algorithm", algorithm.as_str()));
        }
        return Ok(());
    }
    if let Some(k) = k {
        if k == 0 {
            return Err(QueryError::new(QueryErrorCode::InvalidPlan, operation, "betweenness k must be at least 1")
                .with_fact("algorithm", algorithm.as_str())
                .with_fact("k", k));
        }
        return Ok(());
    }
    Ok(())
}

#[cfg(feature = "cugraph")]
pub(in crate::exec) fn runtime_graph_spec_resolution_after_bounded_seed_validation(
    spec: &crate::plan::GraphAlgorithmSpec,
    vertex_count_validated: bool,
) -> RuntimeGraphSpecResolution {
    match &spec.params {
        crate::plan::GraphAlgorithmParams::BetweennessCentrality(params) => {
            bounded_betweenness_runtime_resolution(params.seeds.is_some(), params.k.is_some(), vertex_count_validated)
        },
        crate::plan::GraphAlgorithmParams::EdgeBetweennessCentrality(params) => {
            bounded_betweenness_runtime_resolution(params.seeds.is_some(), params.k.is_some(), vertex_count_validated)
        },
        _ => RuntimeGraphSpecResolution::CheckedNoRewrite,
    }
}

#[cfg(feature = "cugraph")]
fn bounded_betweenness_runtime_resolution(
    has_seeds: bool,
    has_k: bool,
    vertex_count_validated: bool,
) -> RuntimeGraphSpecResolution {
    if has_k {
        RuntimeGraphSpecResolution::Unresolved
    } else if has_seeds || vertex_count_validated {
        RuntimeGraphSpecResolution::CheckedNoRewrite
    } else {
        RuntimeGraphSpecResolution::Unresolved
    }
}

#[cfg(feature = "cugraph")]
fn sorted_graph_vertices_i64(
    input: &DataFrame,
    spec: &crate::plan::GraphAlgorithmSpec,
    operation: &'static str,
    runtime: Option<&cudf::Runtime>,
    token: Option<&cudf::ExecutionBudgetToken>,
) -> Result<Vec<i64>> {
    let mut source = input
        .series(&spec.edge_columns.source)
        .map_err(|error| QueryError::from_cudf(operation, "failed to read graph source column", error))?;
    let mut destination = input
        .series(&spec.edge_columns.destination)
        .map_err(|error| QueryError::from_cudf(operation, "failed to read graph destination column", error))?;
    let source_dtype = source
        .dtype()
        .map_err(|error| QueryError::from_cudf(operation, "failed to inspect graph source dtype", error))?;
    let destination_dtype = destination
        .dtype()
        .map_err(|error| QueryError::from_cudf(operation, "failed to inspect graph destination dtype", error))?;
    if source_dtype.id() != destination_dtype.id() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            operation,
            "cuGraph source and destination columns must have the same vertex dtype",
        )
        .with_fact("src_col", &spec.edge_columns.source)
        .with_fact("dst_col", &spec.edge_columns.destination)
        .with_fact("src_cudf_dtype", format!("{source_dtype:?}"))
        .with_fact("dst_cudf_dtype", format!("{destination_dtype:?}")));
    }
    if let Some(runtime) = runtime {
        source = source.copy_into_runtime(runtime).map_err(|error| {
            QueryError::from_cudf(operation, "failed to place graph source column in budgeted runtime", error)
        })?;
        destination = destination.copy_into_runtime(runtime).map_err(|error| {
            QueryError::from_cudf(operation, "failed to place graph destination column in budgeted runtime", error)
        })?;
    }
    let vertices = if let Some(token) = token {
        let ctx = token.execution_overrides();
        cudf::Series::concat_in(&[&source, &destination], &ctx)
            .and_then(|series| series.unique_in(&ctx))
            .and_then(|series| series.sort_values_in(CudfSortOptions::new(), &ctx))
    } else {
        cudf::Series::concat(&[&source, &destination])
            .and_then(|series| series.unique())
            .and_then(|series| series.sort_values(CudfSortOptions::new()))
    }
    .map_err(|error| QueryError::from_cudf(operation, "failed to collect graph vertices", error))?;
    match source_dtype.id() {
        cudf::TypeId::Int32 => vertices
            .to_vec_strict::<i32>()
            .map(|values| values.into_iter().map(i64::from).collect())
            .map_err(|error| QueryError::from_cudf(operation, "failed to copy graph Int32 vertices to host", error)),
        cudf::TypeId::Int64 => vertices
            .to_vec_strict::<i64>()
            .map_err(|error| QueryError::from_cudf(operation, "failed to copy graph Int64 vertices to host", error)),
        dtype => Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            operation,
            "betweenness centrality requires Int32 or Int64 graph vertices",
        )
        .with_fact("vertex_dtype", format!("{dtype:?}"))),
    }
}

#[cfg(feature = "cugraph")]
fn execute_chunked_betweenness_retry(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
    seed_batches: Vec<crate::plan::GraphBetweennessSeeds>,
    executor: GraphAlgorithmExecutor,
) -> Result<GraphAlgorithmExecutionResult> {
    let requested_seed_count = seed_batches.iter().map(crate::plan::GraphBetweennessSeeds::len).sum::<usize>();
    let batch_count = seed_batches.len();
    let mut frames = Vec::with_capacity(batch_count);
    let mut budget_observations = Vec::new();
    for seeds in seed_batches {
        let chunk_spec = graph_spec_with_betweenness_seeds(spec, seeds.clone())?;
        let chunk_inputs = GraphAlgorithmExecutionInputs {
            edge_list: inputs.edge_list.clone(),
            side_inputs: inputs.side_inputs.clone(),
            graph_memory_limit_bytes: inputs.graph_memory_limit_bytes,
            preparation_pin_id: inputs.preparation_pin_id,
        };
        if FORCE_NEXT_GRAPH_ADAPTIVE_CHUNK_ATTEMPT_RESOURCE_EXHAUSTED.swap(false, Ordering::SeqCst) {
            let mut error = QueryError::new(
                QueryErrorCode::CudfResourceExhausted,
                "nexus_query_engine.graph_algorithm.adaptive_retry_test",
                "forced graph adaptive retry chunk resource exhaustion",
            )
            .with_fact("reason", "forced_graph_adaptive_chunk_oom_retry_test")
            .with_fact("algorithm", spec.algorithm.as_str());
            if let Some(limit) = chunk_inputs.graph_memory_limit_bytes {
                error = error
                    .with_fact("graph_budget_limit_bytes", limit)
                    .with_fact("graph_budget_peak_bytes", limit)
                    .with_fact("graph_budget_status", "reservation_resource_exhausted");
            }
            return Err(error);
        }
        let result = executor(context, &chunk_spec, chunk_inputs)?;
        budget_observations.extend(result.budget_observations);
        frames.push((seeds, result.frame));
    }
    let frame = sum_chunked_betweenness_results(spec, frames)?;
    let algorithm_metadata = match &spec.params {
        crate::plan::GraphAlgorithmParams::BetweennessCentrality(params) => format!(
            "approximate=true,seed_count={requested_seed_count},normalized={},adaptive_retry=oom_retry,seed_chunk_batches={batch_count}",
            params.normalized
        ),
        crate::plan::GraphAlgorithmParams::EdgeBetweennessCentrality(params) => format!(
            "approximate=true,seed_count={requested_seed_count},normalized={},adaptive_retry=oom_retry,seed_chunk_batches={batch_count}",
            params.normalized
        ),
        _ => {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.graph_algorithm.adaptive_retry",
                "graph adaptive seed chunk retry requires betweenness parameters",
            )
            .with_fact("algorithm", spec.algorithm.as_str()));
        },
    };
    Ok(GraphAlgorithmExecutionResult { frame, algorithm_metadata, budget_observations })
}

#[cfg(feature = "cugraph")]
fn graph_adaptive_retry_seed_batches(
    spec: &crate::plan::GraphAlgorithmSpec,
) -> Option<Vec<crate::plan::GraphBetweennessSeeds>> {
    let (seeds, min_chunk_size) = match &spec.params {
        crate::plan::GraphAlgorithmParams::BetweennessCentrality(params) => (params.seeds.as_ref()?, 2),
        crate::plan::GraphAlgorithmParams::EdgeBetweennessCentrality(params) => (params.seeds.as_ref()?, 1),
        _ => return None,
    };
    let crate::plan::GraphBetweennessSeeds::I64(values) = seeds;
    if values.len() < min_chunk_size * 2 {
        return None;
    }
    let first_batch_size = values.len().div_ceil(2).max(min_chunk_size);
    Some(vec![
        crate::plan::GraphBetweennessSeeds::I64(values[..first_batch_size].to_vec()),
        crate::plan::GraphBetweennessSeeds::I64(values[first_batch_size..].to_vec()),
    ])
}

#[cfg(feature = "cugraph")]
fn graph_spec_with_betweenness_seeds(
    spec: &crate::plan::GraphAlgorithmSpec,
    seeds: crate::plan::GraphBetweennessSeeds,
) -> Result<crate::plan::GraphAlgorithmSpec> {
    let mut chunk_spec = spec.clone();
    match &mut chunk_spec.params {
        crate::plan::GraphAlgorithmParams::BetweennessCentrality(params) => {
            params.seeds = Some(seeds);
            params.k = None;
        },
        crate::plan::GraphAlgorithmParams::EdgeBetweennessCentrality(params) => {
            params.seeds = Some(seeds);
            params.k = None;
        },
        _ => {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.graph_algorithm.adaptive_retry",
                "graph adaptive seed chunk retry requires betweenness parameters",
            )
            .with_fact("algorithm", spec.algorithm.as_str()));
        },
    }
    Ok(chunk_spec)
}

#[cfg(feature = "cugraph")]
fn sum_chunked_betweenness_results(
    spec: &crate::plan::GraphAlgorithmSpec,
    frames: Vec<(crate::plan::GraphBetweennessSeeds, DataFrame)>,
) -> Result<DataFrame> {
    let vertex_count = chunked_betweenness_vertex_count(spec, &frames)?;
    let (keys, score_column) = match spec.algorithm {
        crate::plan::GraphAlgorithm::BetweennessCentrality => (vec!["vertex"], "value"),
        crate::plan::GraphAlgorithm::EdgeBetweennessCentrality => (vec!["source", "destination"], "score"),
        _ => {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.graph_algorithm.adaptive_retry",
                "graph adaptive seed chunk retry can only sum betweenness results",
            )
            .with_fact("algorithm", spec.algorithm.as_str()));
        },
    };
    let requested_seeds = betweenness_seeds(spec)?;
    let mut raw_frames = frames
        .into_iter()
        .map(|(seeds, frame)| betweenness_frame_to_raw_contributions(spec, frame, &seeds, vertex_count, score_column))
        .collect::<Result<Vec<_>>>()?;
    let first = raw_frames.pop().ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.adaptive_retry",
            "graph adaptive seed chunk retry requires at least one chunk result",
        )
        .with_fact("algorithm", spec.algorithm.as_str())
    })?;
    let concatenated = if raw_frames.is_empty() {
        first
    } else {
        let refs = raw_frames.iter().collect::<Vec<_>>();
        first.concat_many(&refs).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.graph_algorithm.adaptive_retry.concat",
                "failed to concatenate chunked betweenness retry results",
                error,
            )
        })?
    };
    let summed = concatenated
        .groupby(keys)
        .agg(&[CudfAggSpec::new(score_column, GroupByAgg::Sum).with_alias(score_column)])
        .map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.graph_algorithm.adaptive_retry.sum",
                "failed to sum chunked betweenness retry results",
                error,
            )
        })?;
    let scaled = raw_betweenness_frame_to_requested_scale(spec, summed, requested_seeds, vertex_count, score_column)?;
    normalize_graph_result_schema(scaled, &spec.output_schema)
}

#[cfg(feature = "cugraph")]
fn chunked_betweenness_vertex_count(
    spec: &crate::plan::GraphAlgorithmSpec,
    frames: &[(crate::plan::GraphBetweennessSeeds, DataFrame)],
) -> Result<usize> {
    let Some((_, first)) = frames.first() else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.adaptive_retry",
            "graph adaptive seed chunk retry requires at least one chunk result",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };
    match spec.algorithm {
        crate::plan::GraphAlgorithm::BetweennessCentrality => Ok(first.num_rows()),
        crate::plan::GraphAlgorithm::EdgeBetweennessCentrality => {
            let mut vertices = std::collections::BTreeSet::new();
            let sources = first.series("source").and_then(|series| series.to_vec_strict::<i64>()).map_err(|error| {
                QueryError::from_cudf(
                    "nexus_query_engine.graph_algorithm.adaptive_retry.vertex_count",
                    "failed to read edge-betweenness source vertices for retry scaling",
                    error,
                )
            })?;
            let destinations =
                first.series("destination").and_then(|series| series.to_vec_strict::<i64>()).map_err(|error| {
                    QueryError::from_cudf(
                        "nexus_query_engine.graph_algorithm.adaptive_retry.vertex_count",
                        "failed to read edge-betweenness destination vertices for retry scaling",
                        error,
                    )
                })?;
            vertices.extend(sources);
            vertices.extend(destinations);
            Ok(vertices.len())
        },
        _ => Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.adaptive_retry",
            "graph adaptive seed chunk retry can only count betweenness vertices",
        )
        .with_fact("algorithm", spec.algorithm.as_str())),
    }
}

#[cfg(feature = "cugraph")]
fn betweenness_seeds(spec: &crate::plan::GraphAlgorithmSpec) -> Result<&crate::plan::GraphBetweennessSeeds> {
    match &spec.params {
        crate::plan::GraphAlgorithmParams::BetweennessCentrality(params) => params.seeds.as_ref(),
        crate::plan::GraphAlgorithmParams::EdgeBetweennessCentrality(params) => params.seeds.as_ref(),
        _ => None,
    }
    .ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.adaptive_retry",
            "graph adaptive seed chunk retry requires explicit betweenness seeds",
        )
        .with_fact("algorithm", spec.algorithm.as_str())
    })
}

#[cfg(feature = "cugraph")]
fn betweenness_frame_to_raw_contributions(
    spec: &crate::plan::GraphAlgorithmSpec,
    frame: DataFrame,
    seeds: &crate::plan::GraphBetweennessSeeds,
    vertex_count: usize,
    score_column: &'static str,
) -> Result<DataFrame> {
    let expr = match &spec.params {
        crate::plan::GraphAlgorithmParams::BetweennessCentrality(params) => {
            let denominators = vertex_betweenness_denominators(
                seeds.len(),
                vertex_count,
                spec.graph_options.directed,
                params.normalized,
                params.include_endpoints,
            )?;
            let seed_literals = betweenness_seed_literals(seeds);
            when(col("vertex").is_in_list(seed_literals))
                .then(col(score_column) * denominators.source)
                .otherwise(col(score_column) * denominators.nonsource)
                .alias(score_column)
        },
        crate::plan::GraphAlgorithmParams::EdgeBetweennessCentrality(params) => {
            let denominator = edge_betweenness_denominator(
                seeds.len(),
                vertex_count,
                spec.graph_options.directed,
                params.normalized,
            )?;
            (col(score_column) * denominator).alias(score_column)
        },
        _ => {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.graph_algorithm.adaptive_retry",
                "graph adaptive seed chunk retry requires betweenness parameters",
            )
            .with_fact("algorithm", spec.algorithm.as_str()));
        },
    };
    frame.with_columns([expr]).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.graph_algorithm.adaptive_retry.unscale",
            "failed to unscale chunked betweenness result to raw contributions",
            error,
        )
    })
}

#[cfg(feature = "cugraph")]
fn raw_betweenness_frame_to_requested_scale(
    spec: &crate::plan::GraphAlgorithmSpec,
    frame: DataFrame,
    seeds: &crate::plan::GraphBetweennessSeeds,
    vertex_count: usize,
    score_column: &'static str,
) -> Result<DataFrame> {
    let expr = match &spec.params {
        crate::plan::GraphAlgorithmParams::BetweennessCentrality(params) => {
            let denominators = vertex_betweenness_denominators(
                seeds.len(),
                vertex_count,
                spec.graph_options.directed,
                params.normalized,
                params.include_endpoints,
            )?;
            let seed_literals = betweenness_seed_literals(seeds);
            when(col("vertex").is_in_list(seed_literals))
                .then(col(score_column) / denominators.source)
                .otherwise(col(score_column) / denominators.nonsource)
                .alias(score_column)
        },
        crate::plan::GraphAlgorithmParams::EdgeBetweennessCentrality(params) => {
            let denominator = edge_betweenness_denominator(
                seeds.len(),
                vertex_count,
                spec.graph_options.directed,
                params.normalized,
            )?;
            (col(score_column) / denominator).alias(score_column)
        },
        _ => {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.graph_algorithm.adaptive_retry",
                "graph adaptive seed chunk retry requires betweenness parameters",
            )
            .with_fact("algorithm", spec.algorithm.as_str()));
        },
    };
    frame.with_columns([expr]).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.graph_algorithm.adaptive_retry.rescale",
            "failed to rescale raw chunked betweenness contributions",
            error,
        )
    })
}

#[cfg(feature = "cugraph")]
fn betweenness_seed_literals(seeds: &crate::plan::GraphBetweennessSeeds) -> Vec<cudf::ops::expr::Expr> {
    let crate::plan::GraphBetweennessSeeds::I64(values) = seeds;
    values.iter().copied().map(lit).collect()
}

#[cfg(feature = "cugraph")]
struct VertexBetweennessDenominators {
    source: f64,
    nonsource: f64,
}

#[cfg(feature = "cugraph")]
fn vertex_betweenness_denominators(
    seed_count: usize,
    vertex_count: usize,
    directed: bool,
    normalized: bool,
    include_endpoints: bool,
) -> Result<VertexBetweennessDenominators> {
    let num_sources = seed_count as f64;
    let mut num_vertices = vertex_count as f64;
    if !include_endpoints {
        num_vertices -= 1.0;
    }

    let (source, nonsource) = if (seed_count as f64 == num_vertices) || include_endpoints {
        let common = if normalized {
            num_sources * (num_vertices - 1.0)
        } else if directed {
            num_sources / num_vertices
        } else {
            (num_sources * 2.0) / num_vertices
        };
        (common, common)
    } else if normalized {
        ((num_sources - 1.0) * (num_vertices - 1.0), num_sources * (num_vertices - 1.0))
    } else {
        let mut source = (num_sources - 1.0) / num_vertices;
        let mut nonsource = num_sources / num_vertices;
        if !directed {
            source *= 2.0;
            nonsource *= 2.0;
        }
        (source, nonsource)
    };
    validate_betweenness_denominator(source, "source", "vertex_betweenness")?;
    validate_betweenness_denominator(nonsource, "nonsource", "vertex_betweenness")?;
    Ok(VertexBetweennessDenominators { source, nonsource })
}

#[cfg(feature = "cugraph")]
fn edge_betweenness_denominator(
    seed_count: usize,
    vertex_count: usize,
    directed: bool,
    normalized: bool,
) -> Result<f64> {
    if vertex_count <= 1 {
        return Ok(1.0);
    }
    let mut denominator = if normalized {
        let n = vertex_count as f64;
        n * (n - 1.0)
    } else if !directed {
        2.0
    } else {
        1.0
    };
    if (normalized || !directed) && seed_count < vertex_count {
        denominator *= seed_count as f64 / vertex_count as f64;
    }
    validate_betweenness_denominator(denominator, "common", "edge_betweenness")?;
    Ok(denominator)
}

#[cfg(feature = "cugraph")]
fn validate_betweenness_denominator(value: f64, role: &'static str, algorithm: &'static str) -> Result<()> {
    if value.is_finite() && value > 0.0 {
        return Ok(());
    }
    Err(QueryError::new(
        QueryErrorCode::InvalidPlan,
        "nexus_query_engine.graph_algorithm.adaptive_retry.scaling",
        "chunked betweenness retry produced an invalid scaling denominator",
    )
    .with_fact("algorithm", algorithm)
    .with_fact("denominator_role", role)
    .with_fact("denominator", value))
}

#[cfg(feature = "cugraph")]
fn graph_algorithm_metadata_with_null_drops(metadata: String, null_edge_rows_dropped: u64) -> String {
    let null_metadata = format!("null_edge_rows_dropped={null_edge_rows_dropped}");
    if metadata.is_empty() { null_metadata } else { format!("{metadata},{null_metadata}") }
}

#[cfg(feature = "cugraph")]
pub(super) fn graph_input_null_edge_rows_dropped(metrics: &QueryMetrics, input: NodeId) -> u64 {
    let input_node_id = input.to_string();
    metrics
        .operators
        .iter()
        .filter(|row| row.node_id == input_node_id && row.operator_kind == "graph_edge_null_filter")
        .map(|row| row.input_rows.saturating_sub(row.output_rows))
        .sum()
}

#[cfg(feature = "cugraph")]
fn admit_graph_memory(
    context: &ExecutionContext,
    node_id: NodeId,
    spec: &crate::plan::GraphAlgorithmSpec,
    input: &DataFrame,
    input_edges: u64,
    metrics: &mut QueryMetrics,
) -> Result<Option<u64>> {
    if !crate::graph_memory::graph_memory_reservation_required(spec.algorithm) {
        return Ok(None);
    }
    let report = graph_memory_reservation_report(
        context,
        node_id,
        spec,
        input,
        input_edges,
        context.bounded_device_budget_bytes(),
        graph_device_total_memory_bytes(context)?,
        false,
        None,
        None,
    )?;
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

#[cfg(feature = "cugraph")]
pub(super) fn graph_memory_reservation_report(
    context: &ExecutionContext,
    node_id: NodeId,
    spec: &crate::plan::GraphAlgorithmSpec,
    input: &DataFrame,
    input_edges: u64,
    budget_bytes: Option<u64>,
    device_total_memory_bytes: Option<u64>,
    allow_nulls: bool,
    vertex_estimate: Option<crate::graph_memory::GraphVertexEstimate>,
    statistics: Option<crate::graph_memory::GraphStatistics>,
) -> Result<crate::graph_memory::GraphReservationReport> {
    let mut model_input = graph_memory_model_input_for_frame(input, spec, input_edges, allow_nulls)?;
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
    let model = crate::graph_memory::GraphMemoryModel;
    let mut estimate = model.estimate(spec, model_input)?;
    if context.has_reserved_resident_graph_price() {
        estimate.requested_bytes = estimate.execution_bytes;
    }
    let report = crate::graph_memory::GraphReservationReport::from_estimate(node_id, estimate, budget_bytes);
    Ok(match statistics {
        Some(statistics) => report.with_statistics(statistics),
        None => report,
    })
}

#[cfg(feature = "cugraph")]
pub(in crate::exec) fn graph_device_total_memory_bytes(context: &ExecutionContext) -> Result<Option<u64>> {
    context.runtime().device().available_memory().map(|memory| Some(memory.total_bytes)).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.graph_algorithm.graph_memory_device",
            "failed to query CUDA device memory metadata for graph memory pricing",
            error,
        )
    })
}

// Common return envelope for algorithm-specific cuGraph executors.
#[cfg(feature = "cugraph")]
pub(super) struct GraphAlgorithmExecutionResult {
    pub(super) frame: DataFrame,
    pub(super) algorithm_metadata: String,
    pub(super) budget_observations: Vec<GraphBudgetObservation>,
}

#[cfg(feature = "cugraph")]
struct GraphExecutionBudget {
    runtime: std::sync::Arc<cudf::Runtime>,
    token: Option<cudf::ExecutionBudgetToken>,
}

#[cfg(feature = "cugraph")]
#[derive(Debug, Clone)]
pub(super) struct GraphBudgetObservation {
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

    fn record(&self, metrics: &mut QueryMetrics) {
        self.record_with_boundary(metrics, crate::graph_memory::GRAPH_MEMORY_BOUNDARY, None);
    }

    fn record_with_boundary(
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

// Shared graph inputs after edge-list preparation and side-input materialization.
#[cfg(feature = "cugraph")]
#[derive(Clone)]
pub(super) struct GraphAlgorithmExecutionInputs {
    pub(super) edge_list: GpuDataFrame,
    pub(super) side_inputs: BTreeMap<NodeId, GpuDataFrame>,
    pub(super) graph_memory_limit_bytes: Option<u64>,
    pub(super) preparation_pin_id: Option<u64>,
}

#[cfg(feature = "cugraph")]
type GraphAlgorithmExecutor = fn(
    &ExecutionContext,
    &crate::plan::GraphAlgorithmSpec,
    GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult>;

#[cfg(feature = "cugraph")]
const GRAPH_ALGORITHM_EXECUTORS: &[(crate::plan::GraphAlgorithm, GraphAlgorithmExecutor)] = &[
    (crate::plan::GraphAlgorithm::PageRank, execute_pagerank),
    (crate::plan::GraphAlgorithm::PersonalizedPageRank, execute_personalized_pagerank),
    (crate::plan::GraphAlgorithm::Bfs, execute_bfs),
    (crate::plan::GraphAlgorithm::Louvain, execute_louvain),
    (crate::plan::GraphAlgorithm::EigenvectorCentrality, execute_eigenvector_centrality),
    (crate::plan::GraphAlgorithm::KatzCentrality, execute_katz_centrality),
    (crate::plan::GraphAlgorithm::Hits, execute_hits),
    (crate::plan::GraphAlgorithm::BetweennessCentrality, execute_betweenness_centrality),
    (crate::plan::GraphAlgorithm::EdgeBetweennessCentrality, execute_edge_betweenness_centrality),
    (crate::plan::GraphAlgorithm::Sssp, execute_sssp),
    (crate::plan::GraphAlgorithm::DegreesAll, execute_degrees_all),
    (crate::plan::GraphAlgorithm::InDegreesAll, execute_in_degrees_all),
    (crate::plan::GraphAlgorithm::OutDegreesAll, execute_out_degrees_all),
    (crate::plan::GraphAlgorithm::WeaklyConnectedComponents, execute_weakly_connected_components),
    (crate::plan::GraphAlgorithm::StronglyConnectedComponents, execute_strongly_connected_components),
    (crate::plan::GraphAlgorithm::TriangleCountAll, execute_triangle_count_all),
    (crate::plan::GraphAlgorithm::CoreNumber, execute_core_number),
    (crate::plan::GraphAlgorithm::KCore, execute_k_core),
    (crate::plan::GraphAlgorithm::MinimumSpanningTree, execute_minimum_spanning_tree),
    (crate::plan::GraphAlgorithm::Leiden, execute_leiden),
    (crate::plan::GraphAlgorithm::Ecg, execute_ecg),
    (crate::plan::GraphAlgorithm::SpectralModularityMaximization, execute_spectral_modularity_maximization),
    (crate::plan::GraphAlgorithm::BalancedCutClustering, execute_balanced_cut_clustering),
    (crate::plan::GraphAlgorithm::ForceAtlas2, execute_force_atlas2),
];

#[cfg(feature = "cugraph")]
fn graph_algorithm_executor(algorithm: crate::plan::GraphAlgorithm) -> Result<GraphAlgorithmExecutor> {
    GRAPH_ALGORITHM_EXECUTORS
        .iter()
        .find(|(registered, _)| *registered == algorithm)
        .map(|(_, executor)| *executor)
        .ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::Internal,
                "nexus_query_engine.graph_algorithm.executor_lookup",
                "graph algorithm executor table is missing a registered variant",
            )
            .with_fact("algorithm", algorithm.as_str())
        })
}

/// Thread-affine cuGraph execution session.
///
/// Binds one `cugraph::ResourceHandle` to one `cudf::Runtime` for an executor
/// task.
///
/// **Do not store this in `ExecutionContext`.** `ResourceHandle` is `!Send +
/// !Sync`; keep it local to the task thread.
// Not yet used by any executor path; reserved for future multi-algorithm sequential tasks.
#[cfg(feature = "cugraph")]
#[allow(dead_code)]
struct CugraphExecutionSession {
    runtime: std::sync::Arc<cudf::Runtime>,
    handle: cugraph::ResourceHandle,
}

#[cfg(feature = "cugraph")]
#[allow(dead_code)]
impl CugraphExecutionSession {
    fn new(context: &ExecutionContext) -> Result<Self> {
        let runtime = context.shared_runtime();
        let handle =
            cugraph::ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.session",
                "failed to bind cuGraph resource handle to the native cudf runtime",
            ))?;
        Ok(Self { runtime, handle })
    }

    fn enter(&self, operation: &'static str) -> Result<cugraph::EnteredContext<'_>> {
        self.handle.enter(operation).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.enter",
            "failed to enter cuGraph execution context",
        ))
    }

    fn runtime(&self) -> &cudf::Runtime {
        &self.runtime
    }

    /// Asserts that `egress` is the same runtime used to bind the handle.
    ///
    /// Call before passing a runtime into any `to_dataframe` conversion. If the
    /// runtimes ever diverge, device-side `cudaEventRecord` +
    /// `cudaStreamWaitEvent` ordering is required — not a host synchronize —
    /// before copying cuGraph-produced memory into the egress stream.
    fn assert_egress_runtime(&self, egress: &cudf::Runtime) {
        debug_assert!(
            std::ptr::eq(self.runtime.as_ref(), egress),
            "cuGraph result egress runtime must be the same Arc<cudf::Runtime> used to \
             bind the resource handle; cross-runtime result handoff requires device-side \
             stream ordering via cudaEventRecord + cudaStreamWaitEvent"
        );
    }
}

/// Prepares edges, creates a runtime-bound cuGraph handle, enters it, builds
/// the graph, and runs `f` with the cudf runtime, entered context, and graph.
///
/// Centralizes the shared graph-algorithm execution shape and keeps
/// `context.runtime()` as the ingress/execution/egress domain.
///
/// If a future path routes graph results into a different `cudf::Runtime`
/// stream, insert a device-side `cudaEventRecord` + `cudaStreamWaitEvent`
/// dependency before copying — not a host synchronize.
#[cfg(feature = "cugraph")]
pub(super) fn run_with_cugraph_graph<T, F>(
    context: &ExecutionContext,
    graph_memory_limit_bytes: Option<u64>,
    spec: &crate::plan::GraphAlgorithmSpec,
    resident_graph_key: Option<crate::plan::ResidentGraphKey>,
    edges: DataFrame,
    options: cugraph::cudf_interop::TableGraphOptions,
    operation: &'static str,
    f: F,
) -> Result<GraphAlgorithmExecutionResult>
where
    T: GpuResidentGraphTransposition,
    F: FnOnce(
        &cudf::Runtime,
        &cugraph::EnteredContext<'_>,
        &mut cugraph::Graph<T>,
    ) -> Result<GraphAlgorithmExecutionResult>,
{
    run_with_cugraph_graph_mode(
        context,
        graph_memory_limit_bytes,
        spec,
        resident_graph_key,
        edges,
        options,
        operation,
        None,
        f,
    )
}

#[cfg(feature = "cugraph")]
pub(super) fn run_with_cugraph_graph_cache_hit<T, F>(
    context: &ExecutionContext,
    graph_memory_limit_bytes: Option<u64>,
    spec: &crate::plan::GraphAlgorithmSpec,
    resident_graph_key: Option<crate::plan::ResidentGraphKey>,
    edges: DataFrame,
    options: cugraph::cudf_interop::TableGraphOptions,
    operation: &'static str,
    preparation_pin_id: u64,
    f: F,
) -> Result<GraphAlgorithmExecutionResult>
where
    T: GpuResidentGraphTransposition,
    F: FnOnce(
        &cudf::Runtime,
        &cugraph::EnteredContext<'_>,
        &mut cugraph::Graph<T>,
    ) -> Result<GraphAlgorithmExecutionResult>,
{
    run_with_cugraph_graph_mode(
        context,
        graph_memory_limit_bytes,
        spec,
        resident_graph_key,
        edges,
        options,
        operation,
        Some(preparation_pin_id),
        f,
    )
}

#[cfg(feature = "cugraph")]
fn run_with_cugraph_graph_mode<T, F>(
    context: &ExecutionContext,
    graph_memory_limit_bytes: Option<u64>,
    spec: &crate::plan::GraphAlgorithmSpec,
    resident_graph_key: Option<crate::plan::ResidentGraphKey>,
    edges: DataFrame,
    options: cugraph::cudf_interop::TableGraphOptions,
    operation: &'static str,
    preparation_pin_id: Option<u64>,
    f: F,
) -> Result<GraphAlgorithmExecutionResult>
where
    T: GpuResidentGraphTransposition,
    F: FnOnce(
        &cudf::Runtime,
        &cugraph::EnteredContext<'_>,
        &mut cugraph::Graph<T>,
    ) -> Result<GraphAlgorithmExecutionResult>,
{
    let resident_graph_cache = context.resident_graph_cache();
    if let (Some(cache), Some(resident_graph_key)) = (resident_graph_cache.as_ref(), resident_graph_key) {
        return run_with_cached_cugraph_graph::<T, F>(
            context,
            graph_memory_limit_bytes,
            resident_graph_key,
            edges,
            options,
            operation,
            f,
            std::sync::Arc::clone(cache),
            preparation_pin_id,
        );
    }
    if preparation_pin_id.is_some() {
        return Err(required_resident_graph_cache_hit_error(operation, "cache_or_resident_key_missing"));
    }
    let result =
        run_with_uncached_cugraph_graph::<T, F>(context, graph_memory_limit_bytes, edges, options, operation, f);
    if resident_graph_cache.is_some() {
        let cache_metadata = graph_cache_bypass_metadata_for_missing_key(spec);
        result.map(|result| with_graph_cache_metadata(result, cache_metadata))
    } else {
        result
    }
}

#[cfg(feature = "cugraph")]
fn graph_cache_bypass_metadata_for_missing_key(spec: &crate::plan::GraphAlgorithmSpec) -> &'static str {
    match (&spec.algorithm, &spec.params) {
        (crate::plan::GraphAlgorithm::Bfs, crate::plan::GraphAlgorithmParams::Bfs(params))
            if params.predicates.include_edges.is_some() =>
        {
            "bypass_predicate_include_edges"
        },
        // BFS with stable edge ids normally dispatches through
        // `run_with_cugraph_graph_edge_ids`; this mirrors that label if the
        // routing ever changes.
        (crate::plan::GraphAlgorithm::Bfs, crate::plan::GraphAlgorithmParams::Bfs(_))
            if spec.edge_columns.edge_id.is_some() =>
        {
            "bypass_edge_id_graph"
        },
        (crate::plan::GraphAlgorithm::Bfs, crate::plan::GraphAlgorithmParams::Bfs(_)) => {
            "bypass_no_residency_descriptor"
        },
        (crate::plan::GraphAlgorithm::PageRank, crate::plan::GraphAlgorithmParams::PageRank(_))
            if spec.edge_columns.edge_id.is_some() =>
        {
            "bypass_edge_id_binding"
        },
        (crate::plan::GraphAlgorithm::PageRank, crate::plan::GraphAlgorithmParams::PageRank(_)) => {
            "bypass_no_residency_descriptor"
        },
        _ => "bypass_unkeyed_algorithm",
    }
}

#[cfg(feature = "cugraph")]
fn run_with_uncached_cugraph_graph<T, F>(
    context: &ExecutionContext,
    graph_memory_limit_bytes: Option<u64>,
    edges: DataFrame,
    options: cugraph::cudf_interop::TableGraphOptions,
    operation: &'static str,
    f: F,
) -> Result<GraphAlgorithmExecutionResult>
where
    T: cugraph::Transposition,
    F: FnOnce(
        &cudf::Runtime,
        &cugraph::EnteredContext<'_>,
        &mut cugraph::Graph<T>,
    ) -> Result<GraphAlgorithmExecutionResult>,
{
    let export = edges.try_prepare_device_table_export().map_err(|error| {
        QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            "nexus_query_engine.graph_algorithm.prepare_table_export",
            "failed to prepare cuGraph edge-list table export",
        )
        .with_source(error)
    })?;
    let budget = graph_execution_budget(
        context,
        graph_memory_limit_bytes,
        crate::graph_memory::GRAPH_MEMORY_BOUNDARY,
        operation,
    )?;
    let handle =
        cugraph::ResourceHandle::for_runtime(std::sync::Arc::clone(&budget.runtime)).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.resource_handle",
            "failed to bind cuGraph resource handle to the native cudf runtime",
        ))?;
    let entered = handle.enter(operation).map_err(map_cugraph_error(
        "nexus_query_engine.graph_algorithm.enter",
        "failed to enter cuGraph execution context",
    ))?;
    let mut graph =
        cugraph::cudf_interop::graph_from_prepared_table_export::<T>(&entered, &export, &options).map_err(|error| {
            attach_graph_budget_error_facts(
                map_cugraph_error(
                    "nexus_query_engine.graph_algorithm.build_graph",
                    "failed to build cuGraph graph from edge-list table",
                )(error),
                budget.token.as_ref(),
                operation,
            )
        })?;
    attach_graph_budget_observation(f(budget.runtime.as_ref(), &entered, &mut graph), budget.token.as_ref(), operation)
}

#[cfg(feature = "cugraph")]
fn run_with_cached_cugraph_graph<T, F>(
    context: &ExecutionContext,
    graph_memory_limit_bytes: Option<u64>,
    resident_graph_key: crate::plan::ResidentGraphKey,
    edges: DataFrame,
    options: cugraph::cudf_interop::TableGraphOptions,
    operation: &'static str,
    f: F,
    cache: std::sync::Arc<crate::exec::GpuResidentGraphCache>,
    preparation_pin_id: Option<u64>,
) -> Result<GraphAlgorithmExecutionResult>
where
    T: GpuResidentGraphTransposition,
    F: FnOnce(
        &cudf::Runtime,
        &cugraph::EnteredContext<'_>,
        &mut cugraph::Graph<T>,
    ) -> Result<GraphAlgorithmExecutionResult>,
{
    const BUILD_WAIT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
    let runtime = cache.companion_runtime_for_context(context, operation)?;
    let budget = graph_execution_budget_with_runtime(
        context,
        graph_memory_limit_bytes,
        crate::graph_memory::GRAPH_MEMORY_BOUNDARY,
        operation,
        std::sync::Arc::clone(&runtime),
    )?;
    let handle = cugraph::ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).map_err(map_cugraph_error(
        "nexus_query_engine.graph_algorithm.resource_handle",
        "failed to bind cuGraph resource handle to the resident graph cache runtime",
    ))?;
    let entered = handle.enter(operation).map_err(map_cugraph_error(
        "nexus_query_engine.graph_algorithm.enter",
        "failed to enter cuGraph resident graph cache execution context",
    ))?;

    let mut f = Some(f);
    let build_wait_deadline = Instant::now().checked_add(BUILD_WAIT_TIMEOUT).unwrap_or_else(Instant::now);
    let build_leader = loop {
        if let Some(mut checkout) = cache.checkout::<T>(&resident_graph_key, context, preparation_pin_id)? {
            let retained_device_bytes = checkout.retained_device_bytes();
            let parked = checkout.take_parked();
            match parked.resume(&entered) {
                Ok(mut graph) => {
                    let result = f.take().expect("graph algorithm closure is consumed once")(
                        runtime.as_ref(),
                        &entered,
                        &mut graph,
                    );
                    return match result {
                        Ok(result) => match graph.park(&entered) {
                            Ok(parked) => {
                                let cache_metadata = match checkout.return_parked(parked) {
                                    Ok(()) => format!("hit;retained_device_bytes={retained_device_bytes}"),
                                    Err(_error) => "hit_return_failed".to_string(),
                                };
                                attach_graph_budget_observation(
                                    Ok(with_graph_cache_metadata(result, &cache_metadata)),
                                    budget.token.as_ref(),
                                    operation,
                                )
                            },
                            Err(_error) => attach_graph_budget_observation(
                                Ok(with_graph_cache_metadata(result, "hit_repark_failed")),
                                budget.token.as_ref(),
                                operation,
                            ),
                        },
                        Err(error) => Err(attach_graph_budget_error_facts(error, budget.token.as_ref(), operation)),
                    };
                },
                Err(_error) => {
                    // A warm-sized grant is never allowed to fall through to
                    // graph construction after losing its reserved entry.
                    if preparation_pin_id.is_some() {
                        return Err(required_resident_graph_cache_hit_error(operation, "parked_graph_resume_failed"));
                    }
                },
            }
        }

        if preparation_pin_id.is_some() {
            return Err(required_resident_graph_cache_hit_error(operation, "exact_runtime_entry_missing"));
        }

        match cache.join_build(&resident_graph_key)? {
            GpuResidentGraphBuildJoin::Leader(leader) => break Some(leader),
            GpuResidentGraphBuildJoin::Follower(follower) => {
                if follower.wait(context.cancellation(), build_wait_deadline, operation)?
                    == GpuResidentGraphBuildWait::DeadlineExpired
                {
                    // Cold-priced queries retain enough budget to build
                    // privately when coordination outlives the bounded wait.
                    // Warm-priced queries return above before joining a build.
                    break None;
                }
            },
        }
    };
    let private_after_wait_timeout = build_leader.is_none();

    let export = edges.try_prepare_device_table_export().map_err(|error| {
        QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            "nexus_query_engine.graph_algorithm.prepare_table_export",
            "failed to prepare cuGraph edge-list table export",
        )
        .with_source(error)
    })?;
    let build_current_bytes_before = graph_budget_current_bytes(budget.token.as_ref());
    let mut graph =
        cugraph::cudf_interop::graph_from_prepared_table_export::<T>(&entered, &export, &options).map_err(|error| {
            attach_graph_budget_error_facts(
                map_cugraph_error(
                    "nexus_query_engine.graph_algorithm.build_graph",
                    "failed to build cuGraph graph from edge-list table",
                )(error),
                budget.token.as_ref(),
                operation,
            )
        })?;
    let result = f.take().expect("graph algorithm closure is consumed once")(runtime.as_ref(), &entered, &mut graph);
    match result {
        Ok(result) => {
            let retained_device_bytes = graph_retained_device_bytes_upper_bound(
                budget.token.as_ref(),
                build_current_bytes_before,
                &result.frame,
            );
            match graph.park(&entered) {
                Ok(parked) => {
                    let insert = cache.insert_if_absent(resident_graph_key, context, parked, retained_device_bytes)?;
                    let metadata = match insert {
                        crate::exec::GpuResidentGraphCacheInsert::Inserted => {
                            if private_after_wait_timeout {
                                format!(
                                    "miss_wait_timeout_private_inserted;retained_device_bytes={retained_device_bytes}"
                                )
                            } else {
                                format!("miss_inserted;retained_device_bytes={retained_device_bytes}")
                            }
                        },
                        crate::exec::GpuResidentGraphCacheInsert::AlreadyPresent if private_after_wait_timeout => {
                            "miss_wait_timeout_private_already_present".to_string()
                        },
                        crate::exec::GpuResidentGraphCacheInsert::AlreadyPresent => "miss_already_present".to_string(),
                        crate::exec::GpuResidentGraphCacheInsert::CapacitySkipped => {
                            "miss_capacity_skipped".to_string()
                        },
                        crate::exec::GpuResidentGraphCacheInsert::CopyLimitSkipped => {
                            "miss_copy_limit_skipped".to_string()
                        },
                    };
                    attach_graph_budget_observation(
                        Ok(with_graph_cache_metadata(result, &metadata)),
                        budget.token.as_ref(),
                        operation,
                    )
                },
                Err(_error) => attach_graph_budget_observation(
                    Ok(with_graph_cache_metadata(result, "miss_park_failed")),
                    budget.token.as_ref(),
                    operation,
                ),
            }
        },
        Err(error) => Err(attach_graph_budget_error_facts(error, budget.token.as_ref(), operation)),
    }
}

#[cfg(feature = "cugraph")]
fn required_resident_graph_cache_hit_error(operation: &'static str, reason: &'static str) -> QueryError {
    QueryError::new(
        QueryErrorCode::ServiceOverloaded,
        "nexus_query_engine.graph_algorithm.required_resident_graph_cache_hit",
        "warm-priced resident graph cache entry is no longer available",
    )
    .with_fact("operation", operation)
    .with_fact("reason", reason)
    .with_fact("retry_action", "readmit_query")
}

#[cfg(all(test, feature = "cugraph"))]
mod resident_graph_cache_error_tests {
    use super::*;

    #[test]
    fn warm_reservation_loss_is_structured_and_retryable() {
        let error = required_resident_graph_cache_hit_error(
            "nexus_query_engine.graph_algorithm.test",
            "exact_runtime_entry_missing",
        );

        assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
        assert_eq!(error.status(), crate::error::QueryErrorStatus::Temporary);
        assert_eq!(error.fact_value("reason"), Some("exact_runtime_entry_missing"));
        assert_eq!(error.fact_value("retry_action"), Some("readmit_query"));
    }
}

/// Build a cuGraph graph through `GraphBuilder` when the native edge list
/// carries stable edge ids. The cudf table-export path does not expose edge-id
/// binding, so this path borrows typed device slices for source, destination,
/// optional weight, and edge id columns directly for graph construction.
#[cfg(feature = "cugraph")]
pub(super) fn run_with_cugraph_graph_edge_ids<T, F>(
    context: &ExecutionContext,
    graph_memory_limit_bytes: Option<u64>,
    edges: &DataFrame,
    spec: &crate::plan::GraphAlgorithmSpec,
    operation: &'static str,
    f: F,
) -> Result<GraphAlgorithmExecutionResult>
where
    T: cugraph::Transposition,
    F: FnOnce(
        &cudf::Runtime,
        &cugraph::EnteredContext<'_>,
        &mut cugraph::Graph<T>,
    ) -> Result<GraphAlgorithmExecutionResult>,
{
    let budget = graph_execution_budget(
        context,
        graph_memory_limit_bytes,
        crate::graph_memory::GRAPH_MEMORY_BOUNDARY,
        operation,
    )?;
    let handle =
        cugraph::ResourceHandle::for_runtime(std::sync::Arc::clone(&budget.runtime)).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.resource_handle",
            "failed to bind cuGraph resource handle to the native cudf runtime",
        ))?;
    let entered = handle.enter(operation).map_err(map_cugraph_error(
        "nexus_query_engine.graph_algorithm.enter",
        "failed to enter cuGraph execution context",
    ))?;
    let mut graph = build_graph_from_edge_id_columns::<T>(&entered, edges, spec).map_err(|error| {
        if error.code() == QueryErrorCode::CudfResourceExhausted {
            attach_graph_budget_error_facts(error, budget.token.as_ref(), operation)
        } else {
            error
        }
    })?;
    let result = attach_graph_budget_observation(
        f(budget.runtime.as_ref(), &entered, &mut graph),
        budget.token.as_ref(),
        operation,
    );
    if context.resident_graph_cache().is_some() {
        result.map(|result| with_graph_cache_metadata(result, "bypass_edge_id_graph"))
    } else {
        result
    }
}

#[cfg(feature = "cugraph")]
fn graph_execution_budget(
    context: &ExecutionContext,
    graph_memory_limit_bytes: Option<u64>,
    budget_boundary: &'static str,
    operation: &'static str,
) -> Result<GraphExecutionBudget> {
    let token = graph_execution_budget_token(context, graph_memory_limit_bytes, budget_boundary, operation)?;
    let runtime = cudf::Runtime::builder()
        .with_device(context.runtime().device())
        .with_stream(context.runtime().stream().clone())
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
        })?;
    Ok(GraphExecutionBudget { runtime: std::sync::Arc::new(runtime), token: Some(token) })
}

#[cfg(feature = "cugraph")]
fn graph_execution_budget_with_runtime(
    context: &ExecutionContext,
    graph_memory_limit_bytes: Option<u64>,
    budget_boundary: &'static str,
    operation: &'static str,
    runtime: std::sync::Arc<cudf::Runtime>,
) -> Result<GraphExecutionBudget> {
    let token = graph_execution_budget_token(context, graph_memory_limit_bytes, budget_boundary, operation)?;
    Ok(GraphExecutionBudget { runtime, token: Some(token) })
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
            QueryError::from_cudf_with_budget_details(
                "nexus_query_engine.graph_algorithm.execution_budget",
                "failed to attach graph execution memory reservation",
                error,
            )
            .with_fact("operation", operation)
            .with_fact("budget_boundary", budget_boundary)
            .with_fact("graph_memory_limit_bytes", limit_bytes)
        },
    )
}

#[cfg(feature = "cugraph")]
fn attach_graph_budget_observation(
    result: Result<GraphAlgorithmExecutionResult>,
    token: Option<&cudf::ExecutionBudgetToken>,
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
fn graph_budget_observation(
    token: Option<&cudf::ExecutionBudgetToken>,
    operation: &'static str,
    status: &'static str,
) -> Option<GraphBudgetObservation> {
    let token = token?;
    let usage = token.reservation_usage().ok()?;
    let arena_usage = token.reservation_arena_usage().ok().flatten();
    Some(GraphBudgetObservation::from_reservation_usage(operation, token, usage, arena_usage, status))
}

#[cfg(feature = "cugraph")]
fn graph_budget_current_bytes(token: Option<&cudf::ExecutionBudgetToken>) -> Option<u64> {
    token?.reservation_usage().ok().map(|usage| usage.current_bytes)
}

#[cfg(feature = "cugraph")]
fn graph_retained_device_bytes_upper_bound(
    token: Option<&cudf::ExecutionBudgetToken>,
    before_current_bytes: Option<u64>,
    output: &DataFrame,
) -> u64 {
    let Some(after_current_bytes) = graph_budget_current_bytes(token) else {
        return 0;
    };
    let output_device_bytes = GpuDataFrame::new(output.clone()).device_allocated_size_bytes().unwrap_or(0);
    after_current_bytes.saturating_sub(before_current_bytes.unwrap_or(0)).saturating_sub(output_device_bytes)
}

#[cfg(feature = "cugraph")]
fn with_graph_cache_metadata(
    mut result: GraphAlgorithmExecutionResult,
    cache_metadata: &str,
) -> GraphAlgorithmExecutionResult {
    if result.algorithm_metadata.is_empty() {
        result.algorithm_metadata = format!("graph_cache={cache_metadata}");
    } else {
        result.algorithm_metadata.push_str(",graph_cache=");
        result.algorithm_metadata.push_str(cache_metadata);
    }
    result
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
fn attach_graph_budget_error_facts(
    mut error: QueryError,
    token: Option<&cudf::ExecutionBudgetToken>,
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
fn record_graph_budget_observation_from_error(metrics: &mut QueryMetrics, error: &QueryError) {
    record_graph_budget_observation_from_error_with_boundary(
        metrics,
        error,
        crate::graph_memory::GRAPH_MEMORY_BOUNDARY,
        None,
    );
}

#[cfg(feature = "cugraph")]
fn record_graph_budget_observation_from_error_with_boundary(
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
fn record_graph_adaptive_memory_history_from_error(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    error: &QueryError,
) {
    let Some(observation) = graph_adaptive_memory_observation_from_error(error) else {
        return;
    };
    let Some(memory_history) = context.adaptive_memory_history() else {
        return;
    };
    memory_history.record_graph_observation(
        crate::adaptive_runtime::AdaptiveGraphMemoryHistoryKey::for_graph_algorithm(spec),
        observation,
    );
}

#[cfg(feature = "cugraph")]
fn graph_adaptive_memory_observation_from_error(
    error: &QueryError,
) -> Option<crate::adaptive_runtime::AdaptiveGraphMemoryObservation> {
    let allocator_high_water_bytes =
        error.fact_value("graph_budget_peak_bytes").and_then(|value| value.parse::<u64>().ok())?;
    let budget_bytes = error.fact_value("graph_budget_limit_bytes").and_then(|value| value.parse::<u64>().ok())?;
    Some(crate::adaptive_runtime::AdaptiveGraphMemoryObservation { allocator_high_water_bytes, budget_bytes })
}

#[cfg(feature = "cugraph")]
fn build_graph_from_edge_id_columns<T: cugraph::Transposition>(
    entered: &cugraph::EnteredContext<'_>,
    edges: &DataFrame,
    spec: &crate::plan::GraphAlgorithmSpec,
) -> Result<cugraph::Graph<T>> {
    let vertex_type = edges
        .series(&spec.edge_columns.source)
        .and_then(|series| series.dtype())
        .map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.graph_algorithm.edge_id_graph",
                "failed to inspect source vertex dtype for edge-id graph construction",
                error,
            )
            .with_fact("src_col", &spec.edge_columns.source)
        })?
        .id();
    match vertex_type {
        cudf::TypeId::Int32 => build_graph_from_edge_id_columns_typed::<T, i32>(entered, edges, spec),
        cudf::TypeId::Int64 => build_graph_from_edge_id_columns_typed::<T, i64>(entered, edges, spec),
        _ => Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "nexus_query_engine.graph_algorithm.edge_id_graph",
            "edge-id graph construction requires Int32 or Int64 vertices",
        )
        .with_fact("vertex_dtype", format!("{vertex_type:?}"))),
    }
}

#[cfg(feature = "cugraph")]
fn build_graph_from_edge_id_columns_typed<T, V>(
    entered: &cugraph::EnteredContext<'_>,
    edges: &DataFrame,
    spec: &crate::plan::GraphAlgorithmSpec,
) -> Result<cugraph::Graph<T>>
where
    T: cugraph::Transposition,
    V: cudf::CudfType + cugraph::types::CugraphElement,
{
    let edge_id_col = spec.edge_columns.edge_id.as_deref().ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.edge_id_graph",
            "edge-id graph construction requires a stable edge-id column binding",
        )
    })?;
    let src = graph_series(edges, &spec.edge_columns.source, "source")?;
    let dst = graph_series(edges, &spec.edge_columns.destination, "destination")?;
    let edge_ids = graph_series(edges, edge_id_col, "edge_id")?;
    let src_slice = graph_device_slice::<V>(&src, "source")?;
    let dst_slice = graph_device_slice::<V>(&dst, "destination")?;
    let edge_id_slice = graph_device_slice::<V>(&edge_ids, "edge_id")?;
    let src_view = graph_device_view::<V>(&src_slice, "source")?;
    let dst_view = graph_device_view::<V>(&dst_slice, "destination")?;
    let edge_id_view = graph_device_view::<V>(&edge_id_slice, "edge_id")?;

    // Weight is only projected (and thus present in `edges`) for algorithms
    // that consume it; mirror that gate here so an unweighted edge-id graph
    // does not look up a column `project_edge_list` intentionally dropped.
    let weight_binding = if spec.algorithm.consumes_weight() { spec.edge_columns.weight.as_deref() } else { None };
    match weight_binding {
        Some(weight_col) => {
            let weight = graph_series(edges, weight_col, "weight")?;
            let weight_dtype = weight.dtype().map_err(|error| {
                QueryError::from_cudf(
                    "nexus_query_engine.graph_algorithm.edge_id_graph",
                    "failed to inspect graph weight column dtype",
                    error,
                )
                .with_fact("weight_col", weight_col)
            })?;
            match weight_dtype.id() {
                cudf::TypeId::Float32 => {
                    let weight_slice = graph_device_slice::<f32>(&weight, "weight")?;
                    let weight_view = graph_device_view::<f32>(&weight_slice, "weight")?;
                    graph_builder_base(spec, &src_view, &dst_view)
                        .weights(&weight_view)
                        .edge_ids(&edge_id_view)
                        .build(entered)
                        .map_err(map_cugraph_error(
                            "nexus_query_engine.graph_algorithm.edge_id_graph",
                            "failed to build cuGraph graph with stable edge ids",
                        ))
                },
                cudf::TypeId::Float64 => {
                    let weight_slice = graph_device_slice::<f64>(&weight, "weight")?;
                    let weight_view = graph_device_view::<f64>(&weight_slice, "weight")?;
                    graph_builder_base(spec, &src_view, &dst_view)
                        .weights(&weight_view)
                        .edge_ids(&edge_id_view)
                        .build(entered)
                        .map_err(map_cugraph_error(
                            "nexus_query_engine.graph_algorithm.edge_id_graph",
                            "failed to build cuGraph graph with stable edge ids",
                        ))
                },
                _ => Err(QueryError::new(
                    QueryErrorCode::UnsupportedType,
                    "nexus_query_engine.graph_algorithm.edge_id_graph",
                    "graph weight column dtype is unsupported",
                )
                .with_fact("weight_col", weight_col)
                .with_fact("weight_dtype", format!("{weight_dtype:?}"))),
            }
        },
        None => graph_builder_base(spec, &src_view, &dst_view).edge_ids(&edge_id_view).build(entered).map_err(
            map_cugraph_error(
                "nexus_query_engine.graph_algorithm.edge_id_graph",
                "failed to build cuGraph graph with stable edge ids",
            ),
        ),
    }
}

#[cfg(feature = "cugraph")]
fn graph_builder_base<'a, V: cugraph::types::CugraphElement>(
    spec: &crate::plan::GraphAlgorithmSpec,
    src_view: &'a cugraph::DeviceArrayView<'a, V>,
    dst_view: &'a cugraph::DeviceArrayView<'a, V>,
) -> cugraph::GraphBuilder<'a> {
    let mut builder =
        cugraph::GraphBuilder::new(src_view, dst_view).properties(graph_properties(spec.graph_options.directed));
    if spec.graph_options.renumber {
        builder = builder.renumber();
    }
    builder
}

#[cfg(feature = "cugraph")]
fn graph_series(edges: &DataFrame, column: &str, role: &'static str) -> Result<cudf::Series> {
    edges.series(column).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.graph_algorithm.edge_id_graph",
            "failed to access graph construction column",
            error,
        )
        .with_fact("role", role)
        .with_fact("column", column)
    })
}

#[cfg(feature = "cugraph")]
fn graph_device_slice<'a, T: cudf::CudfType>(
    series: &'a cudf::Series,
    role: &'static str,
) -> Result<cudf::core::DeviceSlice<'a>> {
    series.as_column().device_slice::<T>().map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.graph_algorithm.edge_id_graph",
            "failed to borrow graph construction column device slice",
            error,
        )
        .with_fact("role", role)
    })
}

#[cfg(feature = "cugraph")]
fn graph_device_view<'a, T: cugraph::types::CugraphElement>(
    slice: &'a cudf::core::DeviceSlice<'a>,
    role: &'static str,
) -> Result<cugraph::DeviceArrayView<'a, T>> {
    cugraph::cudf_interop::device_array_view_from_slice::<T>(slice)
        .map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.edge_id_graph",
            "failed to wrap graph construction column as a cuGraph device-array view",
        ))
        .map_err(|error| error.with_fact("role", role))
}

#[cfg(feature = "cugraph")]
#[derive(Debug, Clone, Copy)]
enum EdgeColumnRole {
    Vertex,
    Weight,
    EdgeId,
}

#[cfg(feature = "cugraph")]
pub(super) fn project_edge_list(
    frame: &DataFrame,
    spec: &crate::plan::GraphAlgorithmSpec,
) -> Result<(DataFrame, cudf::TypeId)> {
    let source_type = validate_edge_column(frame, &spec.edge_columns.source, EdgeColumnRole::Vertex)?;
    let destination_type = validate_edge_column(frame, &spec.edge_columns.destination, EdgeColumnRole::Vertex)?;
    if source_type.id() != destination_type.id() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "nexus_query_engine.graph_algorithm.validate_edge_column",
            "cuGraph source and destination columns must have the same vertex dtype",
        )
        .with_fact("src_col", &spec.edge_columns.source)
        .with_fact("dst_col", &spec.edge_columns.destination)
        .with_fact("src_cudf_dtype", format!("{source_type:?}"))
        .with_fact("dst_cudf_dtype", format!("{destination_type:?}")));
    }
    // Only validate/project the weight column for algorithms that consume it.
    // Unweighted algorithms (BFS, etc.) must ignore a bound weight entirely:
    // validating it would reject legitimately-null weights, and projecting it
    // would feed cuGraph a weight it does not use. Kept in lockstep with
    // `table_graph_options`, which only sets `weight_col` under the same gate.
    if spec.algorithm.consumes_weight()
        && let Some(weight) = &spec.edge_columns.weight
    {
        validate_edge_column(frame, weight, EdgeColumnRole::Weight)?;
    }
    if let Some(edge_id) = &spec.edge_columns.edge_id {
        let edge_id_type = validate_edge_column(frame, edge_id, EdgeColumnRole::EdgeId)?;
        if edge_id_type.id() != source_type.id() {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                "nexus_query_engine.graph_algorithm.validate_edge_column",
                "cuGraph edge-id column dtype must match the graph vertex dtype",
            )
            .with_fact("edge_id_col", edge_id)
            .with_fact("edge_id_cudf_dtype", format!("{edge_id_type:?}"))
            .with_fact("vertex_cudf_dtype", format!("{source_type:?}")));
        }
    }

    let mut selected = vec![spec.edge_columns.source.as_str(), spec.edge_columns.destination.as_str()];
    if spec.algorithm.consumes_weight()
        && let Some(weight) = &spec.edge_columns.weight
    {
        selected.push(weight.as_str());
    }
    if let Some(edge_id) = &spec.edge_columns.edge_id {
        selected.push(edge_id.as_str());
    }
    let edges = frame.select(selected).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.graph_algorithm.edge_projection",
            "failed to project cuGraph edge-list columns",
            error,
        )
        .with_fact("src_col", &spec.edge_columns.source)
        .with_fact("dst_col", &spec.edge_columns.destination)
        .with_fact("weight_col", spec.edge_columns.weight.as_deref().unwrap_or("<none>"))
        .with_fact("edge_id_col", spec.edge_columns.edge_id.as_deref().unwrap_or("<none>"))
    })?;
    Ok((edges, source_type.id()))
}

#[cfg(feature = "cugraph")]
pub(super) fn resident_graph_key_for_input<T>(
    input: &GpuDataFrame,
    spec: &crate::plan::GraphAlgorithmSpec,
) -> Option<crate::plan::ResidentGraphKey>
where
    T: GpuResidentGraphTransposition,
{
    input.graph_residency().and_then(|descriptor| {
        let key = crate::plan::planned_resident_graph_key(descriptor, spec)?;
        (key.orientation == T::orientation()).then_some(key)
    })
}

#[cfg(feature = "cugraph")]
pub(super) fn resident_graph_cache_preparation_pin(
    context: &ExecutionContext,
    input: &GpuDataFrame,
    spec: &crate::plan::GraphAlgorithmSpec,
) -> Result<Option<super::types::GpuResidentGraphPreparationPin>> {
    let Some(cache) = context.resident_graph_cache() else {
        context.reject_unmatched_reserved_resident_graph_price("resident_graph_cache_missing")?;
        return Ok(None);
    };
    let Some(descriptor) = input.graph_residency() else {
        context.reject_unmatched_reserved_resident_graph_price("graph_residency_descriptor_missing")?;
        return Ok(None);
    };
    let Some(key) = crate::plan::planned_resident_graph_key(descriptor, spec) else {
        context.reject_unmatched_reserved_resident_graph_price("resident_graph_key_unavailable")?;
        return Ok(None);
    };
    if let Some(pin) = context.take_reserved_resident_graph_pin(&key)? {
        return Ok(Some(pin));
    }
    cache.try_pin_available_key_for_context(&key, context)
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

#[cfg(feature = "cugraph")]
pub(in crate::exec) fn runtime_graph_statistics_probe_with_budget(
    context: &ExecutionContext,
    frame: &DataFrame,
    spec: &crate::plan::GraphAlgorithmSpec,
    budget_limit_bytes: u64,
    metrics: &mut QueryMetrics,
) -> Result<RuntimeGraphStatisticsProbe> {
    let operation = "nexus_query_engine.graph_algorithm.graph_memory_vertex_estimate";
    let budget = graph_execution_budget(
        context,
        Some(budget_limit_bytes),
        crate::graph_memory::GRAPH_VERTEX_ESTIMATE_BOUNDARY,
        operation,
    )?;
    let probe = runtime_graph_statistics_probe_in_runtime(budget.runtime.as_ref(), frame, spec);
    match probe {
        Ok(probe) => {
            if let Some(observation) =
                graph_budget_observation(budget.token.as_ref(), operation, "reservation_enforced")
            {
                observation.record_with_boundary(
                    metrics,
                    crate::graph_memory::GRAPH_VERTEX_ESTIMATE_BOUNDARY,
                    Some("graph_vertex_estimate_probe"),
                );
            }
            Ok(probe)
        },
        Err(error) => {
            let error = attach_graph_budget_error_facts(error, budget.token.as_ref(), operation);
            record_graph_budget_observation_from_error_with_boundary(
                metrics,
                &error,
                crate::graph_memory::GRAPH_VERTEX_ESTIMATE_BOUNDARY,
                Some("graph_vertex_estimate_probe"),
            );
            Err(error)
        },
    }
}

#[cfg(feature = "cugraph")]
fn runtime_graph_statistics_probe_in_runtime(
    runtime: &cudf::Runtime,
    frame: &DataFrame,
    spec: &crate::plan::GraphAlgorithmSpec,
) -> Result<RuntimeGraphStatisticsProbe> {
    let operation = GRAPH_STATISTICS_OPERATION;
    let source = frame
        .series(&spec.edge_columns.source)
        .map_err(|error| QueryError::from_cudf(operation, "failed to read graph source column", error))?;
    let destination = frame
        .series(&spec.edge_columns.destination)
        .map_err(|error| QueryError::from_cudf(operation, "failed to read graph destination column", error))?;
    let source_dtype = source
        .dtype()
        .map_err(|error| QueryError::from_cudf(operation, "failed to inspect graph source dtype", error))?;
    let destination_dtype = destination
        .dtype()
        .map_err(|error| QueryError::from_cudf(operation, "failed to inspect graph destination dtype", error))?;
    if source_dtype.id() != destination_dtype.id() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            operation,
            "cuGraph source and destination columns must have the same vertex dtype",
        )
        .with_fact("src_col", &spec.edge_columns.source)
        .with_fact("dst_col", &spec.edge_columns.destination)
        .with_fact("src_cudf_dtype", format!("{source_dtype:?}"))
        .with_fact("dst_cudf_dtype", format!("{destination_dtype:?}")));
    }
    let vertex_width_bytes = dtype_width_bytes(source_dtype)?;
    let device_total_memory_bytes = runtime.device().available_memory().ok().map(|memory| memory.total_bytes);
    let device_multiprocessor_count = runtime.device().multiprocessor_count().ok().filter(|count| *count > 0);
    let source = source.copy_into_runtime(runtime).map_err(|error| {
        QueryError::from_cudf(operation, "failed to place graph source column in budgeted runtime", error)
    })?;
    let destination = destination.copy_into_runtime(runtime).map_err(|error| {
        QueryError::from_cudf(operation, "failed to place graph destination column in budgeted runtime", error)
    })?;
    let ctx = graph_statistics_execution_overrides(runtime);
    let count = cudf::Series::concat_in(&[&source, &destination], &ctx)
        .and_then(|series| series.nunique_in(&ctx))
        .map_err(|error| QueryError::from_cudf(operation, "failed to count distinct graph endpoints", error))?;
    let count = u64::try_from(count).map_err(|error| {
        QueryError::new(QueryErrorCode::InvalidPlan, operation, "runtime graph vertex estimate does not fit u64")
            .with_source(error)
    })?;
    let vertex_estimate = crate::graph_memory::GraphVertexEstimate::runtime_exact_distinct(count);
    let statistics = crate::graph_memory::GraphStatistics::runtime_exact_vertices(count);

    let statistics = match graph_statistics_optional_algorithm_evidence(
        &ctx,
        source,
        destination,
        spec,
        vertex_width_bytes,
        device_total_memory_bytes,
        device_multiprocessor_count,
        statistics,
    ) {
        Ok(statistics) => statistics,
        Err(error) if error.code() == QueryErrorCode::CudfResourceExhausted => statistics,
        Err(error) => return Err(error),
    };

    Ok(RuntimeGraphStatisticsProbe { vertex_estimate, statistics })
}

#[cfg(feature = "cugraph")]
fn graph_statistics_optional_algorithm_evidence(
    ctx: &ExecutionOverrides<'_>,
    source: cudf::Series,
    destination: cudf::Series,
    spec: &crate::plan::GraphAlgorithmSpec,
    vertex_width_bytes: u64,
    device_total_memory_bytes: Option<u64>,
    device_multiprocessor_count: Option<u32>,
    mut statistics: crate::graph_memory::GraphStatistics,
) -> Result<crate::graph_memory::GraphStatistics> {
    let needs_sum_out_degree = graph_statistics_needs_sum_out_degree(spec.algorithm);
    let needs_triangle_degree = graph_statistics_needs_triangle_degree(spec.algorithm);
    if !needs_sum_out_degree && !needs_triangle_degree {
        return Ok(statistics);
    }
    if FORCE_NEXT_GRAPH_STATISTICS_OPTIONAL_RESOURCE_EXHAUSTED.with(|force| force.replace(false)) {
        return Err(QueryError::new(
            QueryErrorCode::CudfResourceExhausted,
            GRAPH_STATISTICS_OPERATION,
            "forced graph optional statistics resource exhaustion for test",
        ));
    }

    let edge_frame =
        DataFrame::from_series([(GRAPH_STATS_SRC, source), (GRAPH_STATS_DST, destination)]).map_err(|error| {
            QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to build graph statistics edge frame", error)
        })?;
    if edge_frame.num_rows() == 0 {
        if needs_sum_out_degree {
            statistics = statistics.with_sum_out_degree(0);
        }
        if needs_triangle_degree {
            statistics = statistics.with_degree_distribution(0, 0).with_triangle_min_degree(0, 0, 0);
        }
        return Ok(statistics);
    }

    if needs_sum_out_degree {
        let out_degrees = graph_statistics_out_degrees(&edge_frame, ctx)?;
        let sum_out_degree = graph_statistics_betweenness_resident_out_degree_sum(
            &out_degrees,
            spec,
            statistics.exact_vertices,
            vertex_width_bytes,
            device_total_memory_bytes,
            ctx,
        )?;
        statistics = statistics.with_sum_out_degree(sum_out_degree);
    }
    if needs_triangle_degree {
        let triangle_edges = graph_statistics_triangle_undirected_edges(&edge_frame, ctx)?;
        drop(edge_frame);
        if triangle_edges.num_rows() == 0 {
            statistics = statistics.with_degree_distribution(0, 0).with_triangle_min_degree(0, 0, 0);
            return Ok(statistics);
        }
        let undirected_degrees = graph_statistics_undirected_degrees(&triangle_edges, ctx)?;
        let degree = undirected_degrees.series(GRAPH_STATS_OUT_DEGREE).map_err(|error| {
            QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to read graph undirected-degree column", error)
        })?;
        let degree_max = graph_statistics_series_max_u64(&degree, ctx, GRAPH_STATS_OUT_DEGREE)?.unwrap_or(0);
        let degree_p99 = graph_statistics_series_p99_u64(&degree, ctx, GRAPH_STATS_OUT_DEGREE)?;
        let (sum_min_degree_oriented, oriented_edge_count, peak_chunk_min_degree_sum) =
            graph_statistics_triangle_min_degree(
                &triangle_edges,
                &undirected_degrees,
                ctx,
                device_multiprocessor_count,
            )?;
        statistics = statistics.with_degree_distribution(degree_max, degree_p99).with_triangle_min_degree(
            sum_min_degree_oriented,
            oriented_edge_count,
            peak_chunk_min_degree_sum,
        );
    }

    Ok(statistics)
}

#[cfg(feature = "cugraph")]
const GRAPH_STATISTICS_OPERATION: &str = "nexus_query_engine.graph_algorithm.graph_memory_vertex_estimate";
#[cfg(feature = "cugraph")]
const GRAPH_STATS_SRC: &str = "__graph_stats_src";
#[cfg(feature = "cugraph")]
const GRAPH_STATS_DST: &str = "__graph_stats_dst";
#[cfg(feature = "cugraph")]
const GRAPH_STATS_CANONICAL_SRC: &str = "__graph_stats_canonical_src";
#[cfg(feature = "cugraph")]
const GRAPH_STATS_CANONICAL_DST: &str = "__graph_stats_canonical_dst";
#[cfg(feature = "cugraph")]
const GRAPH_STATS_VERTEX: &str = "__graph_stats_vertex";
#[cfg(feature = "cugraph")]
const GRAPH_STATS_ENDPOINT: &str = "__graph_stats_endpoint";
#[cfg(feature = "cugraph")]
const GRAPH_STATS_OUT_DEGREE: &str = "__graph_stats_out_degree";
#[cfg(feature = "cugraph")]
const GRAPH_STATS_SRC_DEGREE: &str = "__graph_stats_src_degree";
#[cfg(feature = "cugraph")]
const GRAPH_STATS_DST_DEGREE: &str = "__graph_stats_dst_degree";
#[cfg(feature = "cugraph")]
const GRAPH_STATS_ORIENTED_SRC: &str = "__graph_stats_oriented_src";
#[cfg(feature = "cugraph")]
const GRAPH_STATS_ORIENTED_DST: &str = "__graph_stats_oriented_dst";
#[cfg(feature = "cugraph")]
const GRAPH_STATS_ORIENTED_DEGREE: &str = "__graph_stats_oriented_degree";
#[cfg(feature = "cugraph")]
const GRAPH_STATS_ORIENTED_SRC_DEGREE: &str = "__graph_stats_oriented_src_degree";
#[cfg(feature = "cugraph")]
const GRAPH_STATS_ORIENTED_DST_DEGREE: &str = "__graph_stats_oriented_dst_degree";
#[cfg(feature = "cugraph")]
const GRAPH_STATS_EDGE_MIN_DEGREE: &str = "__graph_stats_edge_min_degree";
#[cfg(feature = "cugraph")]
fn graph_statistics_execution_overrides(runtime: &cudf::Runtime) -> ExecutionOverrides<'_> {
    ExecutionOverrides::new().with_stream(runtime.stream()).with_memory_resource(runtime.memory_resource())
}

#[cfg(feature = "cugraph")]
const fn graph_statistics_needs_sum_out_degree(algorithm: crate::plan::GraphAlgorithm) -> bool {
    matches!(
        algorithm,
        crate::plan::GraphAlgorithm::BetweennessCentrality | crate::plan::GraphAlgorithm::EdgeBetweennessCentrality
    )
}

#[cfg(feature = "cugraph")]
const fn graph_statistics_needs_triangle_degree(algorithm: crate::plan::GraphAlgorithm) -> bool {
    matches!(algorithm, crate::plan::GraphAlgorithm::TriangleCountAll)
}

#[cfg(feature = "cugraph")]
fn graph_statistics_out_degrees(edge_frame: &DataFrame, ctx: &ExecutionOverrides<'_>) -> Result<DataFrame> {
    let out_degrees = edge_frame
        .groupby([GRAPH_STATS_SRC])
        .with_execution(ctx)
        .agg(&[CudfAggSpec::new(GRAPH_STATS_DST, GroupByAgg::Count { include_nulls: false })
            .with_alias(GRAPH_STATS_OUT_DEGREE)])
        .map_err(|error| {
            QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to compute graph out-degrees", error)
        })?;
    graph_statistics_cast_column_to_u64(&out_degrees, GRAPH_STATS_OUT_DEGREE, ctx)
}

#[cfg(feature = "cugraph")]
fn graph_statistics_betweenness_resident_out_degree_sum(
    out_degrees: &DataFrame,
    spec: &crate::plan::GraphAlgorithmSpec,
    exact_vertices: u64,
    vertex_width_bytes: u64,
    device_total_memory_bytes: Option<u64>,
    ctx: &ExecutionOverrides<'_>,
) -> Result<u64> {
    let resident_sources = graph_statistics_betweenness_resident_sources(
        spec,
        exact_vertices,
        vertex_width_bytes,
        device_total_memory_bytes,
    );
    if resident_sources == 0 || out_degrees.num_rows() == 0 {
        return Ok(0);
    }
    let out_degree = out_degrees.series(GRAPH_STATS_OUT_DEGREE).map_err(|error| {
        QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to read graph out-degree column", error)
    })?;
    if resident_sources >= out_degrees.num_rows() as u64 {
        return graph_statistics_series_sum_u64(&out_degree, ctx, GRAPH_STATS_OUT_DEGREE);
    }
    let take = usize::try_from(resident_sources).map_err(|error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            GRAPH_STATISTICS_OPERATION,
            "betweenness resident source count does not fit usize",
        )
        .with_fact("resident_sources", resident_sources)
        .with_source(error)
    })?;
    let top_out_degrees = out_degrees
        .with_execution(ctx)
        .sort_values([GRAPH_STATS_OUT_DEGREE], CudfSortOptions::new().with_descending(vec![true]))
        .and_then(|frame| frame.with_execution(ctx).head(take))
        .map_err(|error| {
            QueryError::from_cudf(
                GRAPH_STATISTICS_OPERATION,
                "failed to compute top betweenness source out-degrees",
                error,
            )
        })?;
    let top_out_degree = top_out_degrees.series(GRAPH_STATS_OUT_DEGREE).map_err(|error| {
        QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to read top graph out-degree column", error)
    })?;
    graph_statistics_series_sum_u64(&top_out_degree, ctx, GRAPH_STATS_OUT_DEGREE)
}

#[cfg(feature = "cugraph")]
fn graph_statistics_betweenness_resident_sources(
    spec: &crate::plan::GraphAlgorithmSpec,
    exact_vertices: u64,
    vertex_width_bytes: u64,
    device_total_memory_bytes: Option<u64>,
) -> u64 {
    let requested_sources = match &spec.params {
        crate::plan::GraphAlgorithmParams::BetweennessCentrality(params) => {
            params.seeds.as_ref().map_or_else(|| params.k.unwrap_or(exact_vertices), |seeds| seeds.len() as u64)
        },
        crate::plan::GraphAlgorithmParams::EdgeBetweennessCentrality(_) => return 1,
        _ => return 0,
    };
    let mut resident_sources = requested_sources.clamp(1, u16::MAX as u64);
    let Some(total_memory_bytes) = device_total_memory_bytes else {
        return resident_sources;
    };
    if resident_sources <= 1 || exact_vertices == 0 {
        return resident_sources;
    }
    let bfs_result_bytes_per_source = exact_vertices.saturating_mul(vertex_width_bytes.saturating_mul(2));
    if bfs_result_bytes_per_source == 0 {
        return resident_sources;
    }
    resident_sources = resident_sources.min(total_memory_bytes / 4 / bfs_result_bytes_per_source).max(1);
    resident_sources
}

#[cfg(feature = "cugraph")]
fn graph_statistics_triangle_undirected_edges(
    edge_frame: &DataFrame,
    ctx: &ExecutionOverrides<'_>,
) -> Result<DataFrame> {
    let non_loop_edges =
        edge_frame.with_execution(ctx).filter_expr(col(GRAPH_STATS_SRC).ne(col(GRAPH_STATS_DST))).map_err(|error| {
            QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to remove graph self-loops", error)
        })?;
    let canonical = non_loop_edges
        .with_execution(ctx)
        .with_columns([
            when(col(GRAPH_STATS_SRC).le(col(GRAPH_STATS_DST)))
                .then(col(GRAPH_STATS_SRC))
                .otherwise(col(GRAPH_STATS_DST))
                .alias(GRAPH_STATS_CANONICAL_SRC),
            when(col(GRAPH_STATS_SRC).le(col(GRAPH_STATS_DST)))
                .then(col(GRAPH_STATS_DST))
                .otherwise(col(GRAPH_STATS_SRC))
                .alias(GRAPH_STATS_CANONICAL_DST),
        ])
        .map_err(|error| {
            QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to canonicalize undirected graph edges", error)
        })?;
    let canonical = canonical
        .select([GRAPH_STATS_CANONICAL_SRC, GRAPH_STATS_CANONICAL_DST])
        .and_then(|frame| {
            frame.rename_columns([
                (GRAPH_STATS_CANONICAL_SRC, GRAPH_STATS_SRC),
                (GRAPH_STATS_CANONICAL_DST, GRAPH_STATS_DST),
            ])
        })
        .map_err(|error| {
            QueryError::from_cudf(
                GRAPH_STATISTICS_OPERATION,
                "failed to project canonical undirected graph edges",
                error,
            )
        })?;
    canonical
        .with_execution(ctx)
        .drop_duplicates(DropDuplicatesOptions::builder().subset(&[GRAPH_STATS_SRC, GRAPH_STATS_DST]).build())
        .map_err(|error| {
            QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to deduplicate undirected graph edges", error)
        })
}

#[cfg(feature = "cugraph")]
fn graph_statistics_undirected_degrees(edge_frame: &DataFrame, ctx: &ExecutionOverrides<'_>) -> Result<DataFrame> {
    let source = edge_frame.series(GRAPH_STATS_SRC).map_err(|error| {
        QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to read graph source column", error)
    })?;
    let destination = edge_frame.series(GRAPH_STATS_DST).map_err(|error| {
        QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to read graph destination column", error)
    })?;
    let endpoints = cudf::Series::concat_in(&[&source, &destination], ctx).map_err(|error| {
        QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to concat graph endpoints", error)
    })?;
    let endpoint_values = endpoints.clone();
    let endpoint_frame =
        DataFrame::from_series([(GRAPH_STATS_VERTEX, endpoints), (GRAPH_STATS_ENDPOINT, endpoint_values)]).map_err(
            |error| {
                QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to build graph endpoint degree frame", error)
            },
        )?;
    let degrees = endpoint_frame
        .groupby([GRAPH_STATS_VERTEX])
        .with_execution(ctx)
        .agg(&[CudfAggSpec::new(GRAPH_STATS_ENDPOINT, GroupByAgg::Count { include_nulls: false })
            .with_alias(GRAPH_STATS_OUT_DEGREE)])
        .map_err(|error| {
            QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to compute graph undirected degrees", error)
        })?;
    graph_statistics_cast_column_to_u64(&degrees, GRAPH_STATS_OUT_DEGREE, ctx)
}

#[cfg(feature = "cugraph")]
fn graph_statistics_triangle_min_degree(
    edge_frame: &DataFrame,
    vertex_degrees: &DataFrame,
    ctx: &ExecutionOverrides<'_>,
    device_multiprocessor_count: Option<u32>,
) -> Result<(u64, u64, u64)> {
    let source_degrees = vertex_degrees
        .rename_columns([(GRAPH_STATS_VERTEX, GRAPH_STATS_SRC), (GRAPH_STATS_OUT_DEGREE, GRAPH_STATS_SRC_DEGREE)])
        .map_err(|error| {
            QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to prepare source graph degree frame", error)
        })?;
    let destination_degrees = vertex_degrees
        .rename_columns([(GRAPH_STATS_VERTEX, GRAPH_STATS_DST), (GRAPH_STATS_OUT_DEGREE, GRAPH_STATS_DST_DEGREE)])
        .map_err(|error| {
            QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to prepare destination graph degree frame", error)
        })?;
    let edge_degrees = edge_frame
        .with_execution(ctx)
        .merge(&source_degrees, cudf::MergeOptions::new().how_left().on([GRAPH_STATS_SRC]))
        .and_then(|frame| {
            frame
                .with_execution(ctx)
                .merge(&destination_degrees, cudf::MergeOptions::new().how_left().on([GRAPH_STATS_DST]))
        })
        .map_err(|error| {
            QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to join graph endpoint degrees", error)
        })?;
    let edge_degrees = edge_degrees
        .with_execution(ctx)
        .with_columns([
            when(col(GRAPH_STATS_SRC_DEGREE).is_null())
                .then(lit(0u64))
                .otherwise(col(GRAPH_STATS_SRC_DEGREE))
                .alias(GRAPH_STATS_SRC_DEGREE),
            when(col(GRAPH_STATS_DST_DEGREE).is_null())
                .then(lit(0u64))
                .otherwise(col(GRAPH_STATS_DST_DEGREE))
                .alias(GRAPH_STATS_DST_DEGREE),
        ])
        .map_err(|error| {
            QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to normalize graph endpoint degrees", error)
        })?;
    let source_first = col(GRAPH_STATS_SRC_DEGREE).lt(col(GRAPH_STATS_DST_DEGREE))
        | (col(GRAPH_STATS_SRC_DEGREE).eq(col(GRAPH_STATS_DST_DEGREE)) & col(GRAPH_STATS_SRC).lt(col(GRAPH_STATS_DST)));
    let oriented_edges = edge_degrees
        .with_execution(ctx)
        .with_columns([
            when(source_first.clone())
                .then(col(GRAPH_STATS_SRC))
                .otherwise(col(GRAPH_STATS_DST))
                .alias(GRAPH_STATS_ORIENTED_SRC),
            when(source_first.clone())
                .then(col(GRAPH_STATS_DST))
                .otherwise(col(GRAPH_STATS_SRC))
                .alias(GRAPH_STATS_ORIENTED_DST),
        ])
        .map_err(|error| QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to orient graph edges", error))?;
    let oriented_edges = oriented_edges
        .select([GRAPH_STATS_ORIENTED_SRC, GRAPH_STATS_ORIENTED_DST])
        .and_then(|frame| {
            frame.with_execution(ctx).drop_duplicates(
                DropDuplicatesOptions::builder().subset(&[GRAPH_STATS_ORIENTED_SRC, GRAPH_STATS_ORIENTED_DST]).build(),
            )
        })
        .map_err(|error| {
            QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to deduplicate oriented graph edges", error)
        })?;
    let oriented_edge_count = oriented_edges.num_rows() as u64;
    let oriented_out_degrees = oriented_edges
        .groupby([GRAPH_STATS_ORIENTED_SRC])
        .with_execution(ctx)
        .agg(&[CudfAggSpec::new(GRAPH_STATS_ORIENTED_DST, GroupByAgg::Count { include_nulls: false })
            .with_alias(GRAPH_STATS_ORIENTED_DEGREE)])
        .map_err(|error| {
            QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to compute oriented graph degrees", error)
        })?;
    let oriented_out_degrees =
        graph_statistics_cast_column_to_u64(&oriented_out_degrees, GRAPH_STATS_ORIENTED_DEGREE, ctx)?;
    let oriented_source_degrees = oriented_out_degrees
        .rename_column(GRAPH_STATS_ORIENTED_DEGREE, GRAPH_STATS_ORIENTED_SRC_DEGREE)
        .map_err(|error| {
            QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to prepare oriented source degree frame", error)
        })?;
    let oriented_destination_degrees = oriented_out_degrees
        .rename_columns([
            (GRAPH_STATS_ORIENTED_SRC, GRAPH_STATS_ORIENTED_DST),
            (GRAPH_STATS_ORIENTED_DEGREE, GRAPH_STATS_ORIENTED_DST_DEGREE),
        ])
        .map_err(|error| {
            QueryError::from_cudf(
                GRAPH_STATISTICS_OPERATION,
                "failed to prepare oriented destination degree frame",
                error,
            )
        })?;
    let oriented_edge_degrees = oriented_edges
        .with_execution(ctx)
        .merge(&oriented_source_degrees, cudf::MergeOptions::new().how_left().on([GRAPH_STATS_ORIENTED_SRC]))
        .and_then(|frame| {
            frame.with_execution(ctx).merge(
                &oriented_destination_degrees,
                cudf::MergeOptions::new().how_left().on([GRAPH_STATS_ORIENTED_DST]),
            )
        })
        .map_err(|error| {
            QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to join oriented graph degrees", error)
        })?;
    let oriented_edge_degrees = oriented_edge_degrees
        .with_execution(ctx)
        .with_columns([
            when(col(GRAPH_STATS_ORIENTED_SRC_DEGREE).is_null())
                .then(lit(0u64))
                .otherwise(col(GRAPH_STATS_ORIENTED_SRC_DEGREE))
                .alias(GRAPH_STATS_ORIENTED_SRC_DEGREE),
            when(col(GRAPH_STATS_ORIENTED_DST_DEGREE).is_null())
                .then(lit(0u64))
                .otherwise(col(GRAPH_STATS_ORIENTED_DST_DEGREE))
                .alias(GRAPH_STATS_ORIENTED_DST_DEGREE),
        ])
        .map_err(|error| {
            QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to normalize oriented graph degrees", error)
        })?;
    let edge_terms = oriented_edge_degrees
        .with_execution(ctx)
        .with_columns([when(col(GRAPH_STATS_ORIENTED_SRC_DEGREE).le(col(GRAPH_STATS_ORIENTED_DST_DEGREE)))
            .then(col(GRAPH_STATS_ORIENTED_SRC_DEGREE))
            .otherwise(col(GRAPH_STATS_ORIENTED_DST_DEGREE))
            .alias(GRAPH_STATS_EDGE_MIN_DEGREE)])
        .and_then(|frame| frame.select([GRAPH_STATS_ORIENTED_DST, GRAPH_STATS_EDGE_MIN_DEGREE]))
        .map_err(|error| {
            QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to compute graph edge min-degree terms", error)
        })?;
    let edge_min_degree = edge_terms.series(GRAPH_STATS_EDGE_MIN_DEGREE).map_err(|error| {
        QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to read graph edge min-degree column", error)
    })?;
    let sum_min_degree_oriented = graph_statistics_series_sum_u64(&edge_min_degree, ctx, GRAPH_STATS_EDGE_MIN_DEGREE)?;
    let peak_chunk_min_degree_sum = crate::graph_memory::triangle_count_peak_chunk_min_degree_pricing_term(
        sum_min_degree_oriented,
        oriented_edge_count,
        device_multiprocessor_count,
    );
    Ok((sum_min_degree_oriented, oriented_edge_count, peak_chunk_min_degree_sum))
}

#[cfg(feature = "cugraph")]
fn graph_statistics_cast_column_to_u64(
    frame: &DataFrame,
    column: &'static str,
    ctx: &ExecutionOverrides<'_>,
) -> Result<DataFrame> {
    frame
        .with_execution(ctx)
        .with_columns([col(column).cast(cudf::DataType::new(cudf::TypeId::UInt64)).alias(column)])
        .map_err(|error| {
            QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to cast graph statistics column to uint64", error)
                .with_fact("column", column)
        })
}

#[cfg(feature = "cugraph")]
fn graph_statistics_nonnegative_i64(value: i64, column: &'static str) -> Result<u64> {
    u64::try_from(value).map_err(|error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            GRAPH_STATISTICS_OPERATION,
            "graph statistics value does not fit u64",
        )
        .with_fact("column", column)
        .with_fact("value", value)
        .with_source(error)
    })
}

#[cfg(feature = "cugraph")]
fn graph_statistics_series_sum_u64(
    series: &cudf::Series,
    ctx: &ExecutionOverrides<'_>,
    column: &'static str,
) -> Result<u64> {
    let value = series.sum_value_in::<i64>(ctx).map_err(|error| {
        QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to reduce graph statistics sum", error)
            .with_fact("column", column)
    })?;
    graph_statistics_nonnegative_i64(value, column)
}

#[cfg(feature = "cugraph")]
fn graph_statistics_series_max_u64(
    series: &cudf::Series,
    ctx: &ExecutionOverrides<'_>,
    column: &'static str,
) -> Result<Option<u64>> {
    let dtype = series.dtype().map_err(|error| {
        QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to inspect graph statistics column dtype", error)
            .with_fact("column", column)
    })?;
    let value = match dtype.id() {
        cudf::TypeId::UInt64 => graph_statistics_series_max_value::<u64>(series, ctx, column)?,
        cudf::TypeId::UInt32 => graph_statistics_series_max_value::<u32>(series, ctx, column)?.map(u64::from),
        cudf::TypeId::UInt16 => graph_statistics_series_max_value::<u16>(series, ctx, column)?.map(u64::from),
        cudf::TypeId::UInt8 => graph_statistics_series_max_value::<u8>(series, ctx, column)?.map(u64::from),
        cudf::TypeId::Int64 => graph_statistics_series_max_value::<i64>(series, ctx, column)?
            .map(|value| graph_statistics_nonnegative_i64(value, column))
            .transpose()?,
        cudf::TypeId::Int32 => graph_statistics_series_max_value::<i32>(series, ctx, column)?
            .map(|value| graph_statistics_nonnegative_i64(i64::from(value), column))
            .transpose()?,
        cudf::TypeId::Int16 => graph_statistics_series_max_value::<i16>(series, ctx, column)?
            .map(|value| graph_statistics_nonnegative_i64(i64::from(value), column))
            .transpose()?,
        cudf::TypeId::Int8 => graph_statistics_series_max_value::<i8>(series, ctx, column)?
            .map(|value| graph_statistics_nonnegative_i64(i64::from(value), column))
            .transpose()?,
        dtype => {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                GRAPH_STATISTICS_OPERATION,
                "graph statistics max requires an integer column",
            )
            .with_fact("column", column)
            .with_fact("cudf_dtype", dtype.name()));
        },
    };
    Ok(value)
}

#[cfg(feature = "cugraph")]
fn graph_statistics_series_max_value<T: cudf::CudfType>(
    series: &cudf::Series,
    ctx: &ExecutionOverrides<'_>,
    column: &'static str,
) -> Result<Option<T>> {
    series.max_value_in::<T>(ctx).map_err(|error| {
        QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to reduce graph statistics max", error)
            .with_fact("column", column)
    })
}

#[cfg(feature = "cugraph")]
fn graph_statistics_series_p99_u64(
    series: &cudf::Series,
    ctx: &ExecutionOverrides<'_>,
    column: &'static str,
) -> Result<u64> {
    if series.is_empty() {
        return Ok(0);
    }
    let rank = series.len().saturating_mul(99).div_ceil(100).saturating_sub(1);
    let dtype = series.dtype().map_err(|error| {
        QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to inspect graph statistics column dtype", error)
            .with_fact("column", column)
    })?;
    let value = match dtype.id() {
        cudf::TypeId::UInt64 => graph_statistics_sorted_rank_value::<u64>(series, ctx, rank, column),
        cudf::TypeId::UInt32 => graph_statistics_sorted_rank_value::<u32>(series, ctx, rank, column).map(u64::from),
        cudf::TypeId::UInt16 => graph_statistics_sorted_rank_value::<u16>(series, ctx, rank, column).map(u64::from),
        cudf::TypeId::UInt8 => graph_statistics_sorted_rank_value::<u8>(series, ctx, rank, column).map(u64::from),
        cudf::TypeId::Int64 => graph_statistics_sorted_rank_value::<i64>(series, ctx, rank, column)
            .and_then(|value| graph_statistics_nonnegative_i64(value, column)),
        cudf::TypeId::Int32 => graph_statistics_sorted_rank_value::<i32>(series, ctx, rank, column)
            .and_then(|value| graph_statistics_nonnegative_i64(i64::from(value), column)),
        cudf::TypeId::Int16 => graph_statistics_sorted_rank_value::<i16>(series, ctx, rank, column)
            .and_then(|value| graph_statistics_nonnegative_i64(i64::from(value), column)),
        cudf::TypeId::Int8 => graph_statistics_sorted_rank_value::<i8>(series, ctx, rank, column)
            .and_then(|value| graph_statistics_nonnegative_i64(i64::from(value), column)),
        dtype => Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            GRAPH_STATISTICS_OPERATION,
            "graph degree p99 requires an integer column",
        )
        .with_fact("column", column)
        .with_fact("cudf_dtype", dtype.name())),
    }?;
    Ok(value)
}

#[cfg(feature = "cugraph")]
fn graph_statistics_sorted_rank_value<T: cudf::CudfType>(
    series: &cudf::Series,
    ctx: &ExecutionOverrides<'_>,
    rank: usize,
    column: &'static str,
) -> Result<T> {
    let value = series
        .sort_values_in(CudfSortOptions::new(), ctx)
        .and_then(|series| series.slice_in(rank, 1, ctx))
        .and_then(|series| series.to_vec_strict::<T>())
        .map_err(|error| {
            QueryError::from_cudf(GRAPH_STATISTICS_OPERATION, "failed to read graph degree p99 statistic", error)
                .with_fact("column", column)
                .with_fact("rank", rank)
        })?;
    value.into_iter().next().ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            GRAPH_STATISTICS_OPERATION,
            "graph degree p99 statistic slice unexpectedly returned no rows",
        )
        .with_fact("column", column)
        .with_fact("rank", rank)
    })
}

#[cfg(feature = "cugraph")]
fn validate_edge_column(frame: &DataFrame, column: &str, role: EdgeColumnRole) -> Result<cudf::DataType> {
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
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.validate_edge_column",
            "cuGraph edge-list columns must be non-null",
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
fn dtype_width_bytes(dtype: cudf::DataType) -> Result<u64> {
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

#[cfg(feature = "cugraph")]
fn graph_properties(directed: bool) -> cugraph::types::GraphProperties {
    if directed { cugraph::types::GraphProperties::directed() } else { cugraph::types::GraphProperties::undirected() }
}

#[cfg(feature = "cugraph")]
pub(super) fn table_graph_options(spec: &crate::plan::GraphAlgorithmSpec) -> cugraph::cudf_interop::TableGraphOptions {
    let mut options = cugraph::cudf_interop::TableGraphOptions::new(0, 1)
        .properties(graph_properties(spec.graph_options.directed))
        .construction_policy(graph_construction_policy(spec.graph_options.policy));
    if spec.graph_options.renumber {
        options = options.renumber();
    }
    // `project_edge_list` only emits the weight column (at index 2, before any
    // edge-id column) when the algorithm consumes weights. Gate the cuGraph
    // weight binding on the same condition so the index stays valid.
    if spec.algorithm.consumes_weight() && spec.edge_columns.weight.is_some() {
        options = options.weight_col(2);
    }
    options
}

#[cfg(feature = "cugraph")]
fn graph_construction_policy(
    policy: crate::plan::GraphConstructionPolicy,
) -> cugraph::cudf_interop::GraphConstructionPolicy {
    match policy {
        crate::plan::GraphConstructionPolicy::PythonCugraph => {
            cugraph::cudf_interop::GraphConstructionPolicy::PythonCugraph
        },
        crate::plan::GraphConstructionPolicy::RawLibcugraph => {
            cugraph::cudf_interop::GraphConstructionPolicy::RawLibcugraph
        },
    }
}

#[cfg(feature = "cugraph")]
fn graph_algorithm_parameters(params: &crate::plan::GraphAlgorithmParams) -> String {
    match params {
        crate::plan::GraphAlgorithmParams::PageRank(params) => {
            format!("alpha={},epsilon={},max_iterations={}", params.alpha, params.epsilon, params.max_iterations)
        },
        crate::plan::GraphAlgorithmParams::PersonalizedPageRank(params) => format!(
            "alpha={},epsilon={},max_iterations={},personalization_node={},personalization_vertex_col={},personalization_value_col={}",
            params.pagerank.alpha,
            params.pagerank.epsilon,
            params.pagerank.max_iterations,
            params.personalization.input,
            params.personalization.vertex,
            params.personalization.value,
        ),
        crate::plan::GraphAlgorithmParams::Bfs(params) => format!(
            "source_vertex={},depth_limit={}",
            params.source_vertex,
            params.depth_limit.map_or_else(|| "none".to_string(), |value| value.to_string())
        ),
        crate::plan::GraphAlgorithmParams::Louvain(params) => {
            format!("max_level={},threshold={},resolution={}", params.max_level, params.threshold, params.resolution)
        },
        crate::plan::GraphAlgorithmParams::EigenvectorCentrality(params) => {
            format!("epsilon={},max_iterations={}", params.epsilon, params.max_iterations)
        },
        crate::plan::GraphAlgorithmParams::KatzCentrality(params) => format!(
            "alpha={},beta={},epsilon={},max_iterations={}",
            params.alpha, params.beta, params.epsilon, params.max_iterations,
        ),
        crate::plan::GraphAlgorithmParams::Hits(params) => format!(
            "epsilon={},max_iterations={},normalize={}",
            params.epsilon, params.max_iterations, params.normalize,
        ),
        crate::plan::GraphAlgorithmParams::BetweennessCentrality(params) => format!(
            "seeds={},k={},normalized={},include_endpoints={},exact_vertex_threshold={}",
            graph_betweenness_seeds_label(params.seeds.as_ref()),
            params.k.map_or_else(|| "none".to_string(), |k| k.to_string()),
            params.normalized,
            params.include_endpoints,
            params.exact_vertex_threshold,
        ),
        crate::plan::GraphAlgorithmParams::EdgeBetweennessCentrality(params) => format!(
            "seeds={},k={},normalized={},exact_vertex_threshold={}",
            graph_betweenness_seeds_label(params.seeds.as_ref()),
            params.k.map_or_else(|| "none".to_string(), |k| k.to_string()),
            params.normalized,
            params.exact_vertex_threshold,
        ),
        crate::plan::GraphAlgorithmParams::Sssp(params) => format!(
            "source_vertex={},cutoff={}",
            params.source_vertex,
            params.cutoff.map_or_else(|| "none".to_string(), |v| v.to_string()),
        ),
        crate::plan::GraphAlgorithmParams::DegreesAll(_) => String::new(),
        crate::plan::GraphAlgorithmParams::InDegreesAll(_) => String::new(),
        crate::plan::GraphAlgorithmParams::OutDegreesAll(_) => String::new(),
        crate::plan::GraphAlgorithmParams::WeaklyConnectedComponents(_) => String::new(),
        crate::plan::GraphAlgorithmParams::StronglyConnectedComponents(_) => String::new(),
        crate::plan::GraphAlgorithmParams::TriangleCountAll(_) => String::new(),
        crate::plan::GraphAlgorithmParams::CoreNumber(params) => {
            format!("degree_type={}", params.degree_type.as_str())
        },
        crate::plan::GraphAlgorithmParams::KCore(params) => {
            format!("k={},degree_type={}", params.k, params.degree_type.as_str())
        },
        crate::plan::GraphAlgorithmParams::MinimumSpanningTree(_) => String::new(),
        crate::plan::GraphAlgorithmParams::Leiden(params) => format!(
            "max_level={},resolution={},theta={},seed={}",
            params.max_level, params.resolution, params.theta, params.seed,
        ),
        crate::plan::GraphAlgorithmParams::Ecg(params) => format!(
            "min_weight={},ensemble_size={},max_level={},threshold={},resolution={},seed={}",
            params.min_weight, params.ensemble_size, params.max_level, params.threshold, params.resolution, params.seed,
        ),
        crate::plan::GraphAlgorithmParams::SpectralModularityMaximization(params) => format!(
            "n_clusters={},n_eigenvectors={},evs_tolerance={},evs_max_iterations={},\
             k_means_tolerance={},k_means_max_iterations={},seed={}",
            params.n_clusters,
            params.n_eigenvectors,
            params.evs_tolerance,
            params.evs_max_iterations,
            params.k_means_tolerance,
            params.k_means_max_iterations,
            params.seed,
        ),
        crate::plan::GraphAlgorithmParams::BalancedCutClustering(params) => format!(
            "n_clusters={},n_eigenvectors={},evs_tolerance={},evs_max_iterations={},\
             k_means_tolerance={},k_means_max_iterations={},seed={}",
            params.n_clusters,
            params.n_eigenvectors,
            params.evs_tolerance,
            params.evs_max_iterations,
            params.k_means_tolerance,
            params.k_means_max_iterations,
            params.seed,
        ),
        crate::plan::GraphAlgorithmParams::ForceAtlas2(params) => format!(
            "max_iter={},outbound_attraction_distribution={},lin_log_mode={},prevent_overlapping={},\
             overlap_scaling_ratio={},edge_weight_influence={},jitter_tolerance={},barnes_hut_optimize={},\
             barnes_hut_theta={},scaling_ratio={},strong_gravity_mode={},gravity={},verbose={},seed={}",
            params.max_iter,
            params.outbound_attraction_distribution,
            params.lin_log_mode,
            params.prevent_overlapping,
            params.overlap_scaling_ratio,
            params.edge_weight_influence,
            params.jitter_tolerance,
            params.barnes_hut_optimize,
            params.barnes_hut_theta,
            params.scaling_ratio,
            params.strong_gravity_mode,
            params.gravity,
            params.verbose,
            params.seed,
        ),
    }
}

#[cfg(feature = "cugraph")]
fn graph_betweenness_seeds_label(seeds: Option<&crate::plan::GraphBetweennessSeeds>) -> String {
    seeds.map_or_else(|| "all".to_string(), |seeds| seeds.len().to_string())
}

#[cfg(feature = "cugraph")]
fn boundary_delta(
    before: &crate::metrics::BoundaryMetrics,
    after: &crate::metrics::BoundaryMetrics,
) -> crate::metrics::BoundaryMetrics {
    crate::metrics::BoundaryMetrics {
        gpu_resident_input_handoffs: after
            .gpu_resident_input_handoffs
            .saturating_sub(before.gpu_resident_input_handoffs),
        host_to_device_uploads: after.host_to_device_uploads.saturating_sub(before.host_to_device_uploads),
        host_to_device_uploaded_rows: after
            .host_to_device_uploaded_rows
            .saturating_sub(before.host_to_device_uploaded_rows),
        host_to_device_uploaded_bytes: after
            .host_to_device_uploaded_bytes
            .saturating_sub(before.host_to_device_uploaded_bytes),
        device_to_host_materializations: after
            .device_to_host_materializations
            .saturating_sub(before.device_to_host_materializations),
        device_to_host_materialized_rows: after
            .device_to_host_materialized_rows
            .saturating_sub(before.device_to_host_materialized_rows),
        device_to_host_materialized_bytes: after
            .device_to_host_materialized_bytes
            .saturating_sub(before.device_to_host_materialized_bytes),
        device_to_device_copies: after.device_to_device_copies.saturating_sub(before.device_to_device_copies),
        device_to_device_copied_bytes: after
            .device_to_device_copied_bytes
            .saturating_sub(before.device_to_device_copied_bytes),
    }
}

#[cfg(feature = "cugraph")]
pub(super) fn degree_type_to_cugraph(
    degree_type: crate::plan::GraphKCoreDegreeType,
    directed: bool,
) -> cugraph::structure::KCoreDegreeType {
    // libcugraph's INOUT selector sums incoming and outgoing adjacency. The
    // Python cuGraph surface treats all three selectors as the same degree on
    // an undirected graph, whose physical edge list is already symmetric.
    // Passing INOUT for that representation would therefore double every
    // degree and change core membership. Use one side for undirected graphs;
    // incoming and outgoing are equivalent there.
    if !directed {
        return cugraph::structure::KCoreDegreeType::In;
    }
    match degree_type {
        crate::plan::GraphKCoreDegreeType::In => cugraph::structure::KCoreDegreeType::In,
        crate::plan::GraphKCoreDegreeType::Out => cugraph::structure::KCoreDegreeType::Out,
        crate::plan::GraphKCoreDegreeType::InOut => cugraph::structure::KCoreDegreeType::InOut,
    }
}

#[cfg(feature = "cugraph")]
pub(super) fn normalize_graph_result_schema(mut frame: DataFrame, schema: &SchemaRef) -> Result<DataFrame> {
    for field in schema.fields() {
        let field_name = field.name();
        let target_type = cudf_dtype_from_arrow_type(field.data_type()).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::UnsupportedType,
                "nexus_query_engine.graph_algorithm.schema_normalize",
                "cuGraph result schema contains an unsupported output type",
            )
            .with_fact("field_name", field_name)
            .with_fact("expected_type", format!("{:?}", field.data_type()))
        })?;
        let actual_type = frame.series(field_name).and_then(|series| series.dtype()).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.graph_algorithm.schema_normalize",
                "failed to inspect cuGraph result column",
                error,
            )
            .with_fact("field_name", field_name)
        })?;
        if actual_type != target_type {
            frame = frame.cast_column(field_name, target_type).map_err(|error| {
                QueryError::from_cudf(
                    "nexus_query_engine.graph_algorithm.schema_normalize",
                    "failed to cast cuGraph result column to the stable SQL schema",
                    error,
                )
                .with_fact("field_name", field_name)
                .with_fact("actual_cudf_type", format!("{actual_type:?}"))
                .with_fact("expected_arrow_type", format!("{:?}", field.data_type()))
            })?;
        }
    }
    let columns = schema.fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>();
    frame.select(columns).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.graph_algorithm.schema_normalize",
            "failed to order cuGraph result columns according to the stable SQL schema",
            error,
        )
    })
}

#[cfg(feature = "cugraph")]
pub(super) fn map_cugraph_error(
    operation: &'static str,
    message: &'static str,
) -> impl FnOnce(cugraph::Error) -> QueryError {
    move |error| {
        QueryError::new(code_for_cugraph_error(&error), operation, message)
            .with_fact("cugraph_kind", error.kind())
            .with_fact("cugraph_status", error.status())
            .with_fact("cugraph_operation", error.operation())
            .with_source(error)
    }
}

#[cfg(feature = "cugraph")]
fn code_for_cugraph_error(error: &cugraph::Error) -> QueryErrorCode {
    match error.kind() {
        cugraph::ErrorKind::OutOfMemory => QueryErrorCode::CudfResourceExhausted,
        cugraph::ErrorKind::InvalidInput | cugraph::ErrorKind::InvalidHandle => QueryErrorCode::CudfInvalidInput,
        cugraph::ErrorKind::UnsupportedType | cugraph::ErrorKind::NotImplemented => QueryErrorCode::CudfUnsupported,
        cugraph::ErrorKind::InvalidState => QueryErrorCode::CudfInvariantViolation,
        cugraph::ErrorKind::IoError | cugraph::ErrorKind::Unexpected => match error.status() {
            cugraph::ErrorStatus::Temporary => QueryErrorCode::CudfTemporaryFailure,
            cugraph::ErrorStatus::Persistent => QueryErrorCode::CudfPersistentFailure,
            cugraph::ErrorStatus::Permanent => QueryErrorCode::CudfDependencyFailure,
        },
        _ => QueryErrorCode::CudfDependencyFailure,
    }
}
