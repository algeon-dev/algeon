#[cfg(feature = "cugraph")]
use std::collections::BTreeMap;
#[cfg(feature = "cugraph")]
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(feature = "cugraph")]
use arrow::datatypes::SchemaRef;

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
pub(in crate::exec) enum RuntimeGraphSpecResolution {
    Unresolved,
    CheckedNoRewrite,
    Rewritten(crate::plan::GraphAlgorithmSpec),
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
    let input_rows = input.num_rows() as u64;
    let input_columns = input.num_columns() as u64;
    let prior_null_edge_rows_dropped = graph_input_null_edge_rows_dropped(metrics, node.input);
    let null_filter_scope = super::edge_null_filter::EdgeNullFilterScope {
        weight: node.spec.algorithm.consumes_weight(),
        edge_id: node.spec.edge_columns.edge_id.is_some(),
    };
    let input = super::edge_null_filter::filter_null_edge_rows(
        context,
        node_id,
        "graph_edge_null_filter",
        input.into_frame(),
        &node.spec.edge_columns,
        null_filter_scope,
        metrics,
    )?;
    let sanitized_input_rows = input.num_rows() as u64;
    let null_edge_rows_dropped =
        prior_null_edge_rows_dropped.saturating_add(input_rows.saturating_sub(sanitized_input_rows));
    let runtime_spec =
        RuntimeGraphSpecResolution::from_optional(graph_spec_with_runtime_betweenness_seeds(&input, &node.spec)?);
    let spec = runtime_spec.spec_or(&node.spec);
    admit_graph_memory(context, node_id, spec, &input, sanitized_input_rows, metrics)?;
    execute_graph_algorithm_sanitized(
        context,
        node_id,
        node,
        runtime_spec,
        GpuDataFrame::new(input),
        side_inputs,
        None,
        input_rows,
        input_columns,
        null_edge_rows_dropped,
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
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let boundary_before = metrics.boundary.clone();
    let runtime_spec = match runtime_spec {
        RuntimeGraphSpecResolution::Unresolved => {
            resolve_runtime_graph_spec_for_execution(context, input.frame(), &node.spec, graph_memory_limit_bytes)?
        },
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
            },
            batches,
        )
    });
    let inputs = GraphAlgorithmExecutionInputs { edge_list: input, side_inputs, graph_memory_limit_bytes };
    let timer = Instant::now();
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
    let elapsed = timer.elapsed();
    metrics.operator_time += elapsed;
    for observation in &result.budget_observations {
        observation.record(metrics);
    }

    let gpu = GpuDataFrame::new(result.frame);
    let output_device_bytes = gpu.device_allocated_size_bytes().ok();
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
        elapsed_ns: u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX),
        gpu_resident_input_handoffs: boundary_delta.gpu_resident_input_handoffs,
        host_to_device_uploads: boundary_delta.host_to_device_uploads,
        device_to_host_materializations: boundary_delta.device_to_host_materializations,
    });
    Ok(gpu)
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
                .with_fact("graph_budget_status", "budget_tracking_resource_exhausted");
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
) -> Result<RuntimeGraphSpecResolution> {
    let Some(operation) = betweenness_seed_policy_operation(spec) else {
        return Ok(RuntimeGraphSpecResolution::CheckedNoRewrite);
    };
    let Some(limit_bytes) = graph_memory_limit_bytes else {
        return graph_spec_with_runtime_betweenness_seeds(input, spec).map(RuntimeGraphSpecResolution::from_optional);
    };
    let runtime = graph_execution_runtime(context, Some(limit_bytes), operation)?;
    let result = graph_spec_with_runtime_betweenness_seeds_in_runtime(input, spec, Some(runtime.as_ref()));
    result
        .map(RuntimeGraphSpecResolution::from_optional)
        .map_err(|error| attach_graph_budget_error_facts(error, runtime.as_ref(), operation))
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
    graph_spec_with_runtime_betweenness_seeds_in_runtime(input, spec, None)
}

#[cfg(feature = "cugraph")]
fn graph_spec_with_runtime_betweenness_seeds_in_runtime(
    input: &DataFrame,
    spec: &crate::plan::GraphAlgorithmSpec,
    runtime: Option<&cudf::Runtime>,
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

    let vertices = sorted_graph_vertices_i64(input, spec, operation, runtime)?;
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
    let vertices = cudf::Series::concat(&[&source, &destination])
        .and_then(|series| series.unique())
        .and_then(|series| series.sort_values(CudfSortOptions::new()))
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
                    .with_fact("graph_budget_status", "budget_tracking_resource_exhausted");
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
) -> Result<()> {
    if !crate::graph_memory::graph_memory_reservation_required(spec.algorithm) {
        return Ok(());
    }
    let report = graph_memory_reservation_report(
        node_id,
        spec,
        input,
        input_edges,
        context.bounded_device_budget_bytes(),
        graph_device_total_memory_bytes(context)?,
        false,
        None,
    )?;
    let model = crate::graph_memory::GraphMemoryModel;
    let request = model.reservation_request(node_id, &report, context.device_memory_space());
    metrics.record_graph_reservation(report.clone());

    let Some(mut request) = request else {
        return Ok(());
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
    Ok(())
}

#[cfg(feature = "cugraph")]
pub(super) fn graph_memory_reservation_report(
    node_id: NodeId,
    spec: &crate::plan::GraphAlgorithmSpec,
    input: &DataFrame,
    input_edges: u64,
    budget_bytes: Option<u64>,
    device_total_memory_bytes: Option<u64>,
    allow_nulls: bool,
    vertex_estimate: Option<crate::graph_memory::GraphVertexEstimate>,
) -> Result<crate::graph_memory::GraphReservationReport> {
    let mut model_input = graph_memory_model_input_for_frame(input, spec, input_edges, allow_nulls)?;
    if let Some(device_total_memory_bytes) = device_total_memory_bytes {
        model_input = model_input.with_device_total_memory_bytes(device_total_memory_bytes);
    }
    if let Some(vertex_estimate) = vertex_estimate {
        model_input = model_input.with_vertex_estimate(vertex_estimate);
    }
    let model = crate::graph_memory::GraphMemoryModel;
    let estimate = model.estimate(spec, model_input)?;
    Ok(crate::graph_memory::GraphReservationReport::from_estimate(node_id, estimate, budget_bytes))
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
#[derive(Debug, Clone)]
pub(super) struct GraphBudgetObservation {
    operation: &'static str,
    budget_bytes: u64,
    allocator_high_water_bytes: Option<u64>,
    status: &'static str,
    context_facts: BTreeMap<String, String>,
}

#[cfg(feature = "cugraph")]
impl GraphBudgetObservation {
    fn from_usage(
        operation: &'static str,
        usage: cudf::core::memory::DeviceMemoryResourceUsage,
        status: &'static str,
    ) -> Self {
        let mut context_facts = BTreeMap::new();
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
        Self {
            operation,
            budget_bytes: usage.allocation_limit_bytes,
            allocator_high_water_bytes: Some(usage.peak_bytes),
            status,
            context_facts,
        }
    }

    fn record(&self, metrics: &mut QueryMetrics) {
        self.record_with_boundary(
            metrics,
            crate::graph_memory::GRAPH_MEMORY_BOUNDARY,
            Some("cugraph_workspace_preflight"),
        );
    }

    fn record_with_boundary(
        &self,
        metrics: &mut QueryMetrics,
        boundary: &'static str,
        unresolved_contract: Option<&'static str>,
    ) {
        metrics.record_graph_budget_tracking_with_boundary(
            self.operation,
            boundary,
            unresolved_contract.map(str::to_string),
            self.budget_bytes,
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

/// Transposition layout as a value type, used in [`GraphCacheKey`].
#[cfg(feature = "cugraph")]
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum GraphOrientation {
    Forward,
    Transposed,
}

/// Stable dimensions for a future thread-affine cuGraph graph cache.
///
/// **Current executors do not cache on this key** — they rebuild the graph per
/// node until edge-buffer identity and graph lifetimes are modeled.
///
/// Graph reuse additionally requires:
/// - Stable edge-buffer identity (a pointer/generation pair that survives
///   column projection, or a content-addressable hash of device memory).
/// - A GPU-resident lifetime model: drop the cached graph when the source
///   `DataFrame` is released, or scope the cache to the executor task.
/// - Verification that no algorithm mutates the graph in place (check cuGraph
///   C API ownership semantics per algorithm).
#[cfg(feature = "cugraph")]
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GraphCacheKey {
    /// Runtime data pointer used only to distinguish device/stream domains.
    ///
    /// ABA risk: a future cache must be scoped so the runtime cannot be dropped
    /// while this pointer is used as a key.
    runtime_ptr: usize,
    source_column: String,
    destination_column: String,
    weight_column: Option<String>,
    edge_id_column: Option<String>,
    construction_policy: &'static str,
    directed: bool,
    renumber: bool,
    orientation: GraphOrientation,
}

#[cfg(feature = "cugraph")]
#[allow(dead_code)]
impl GraphCacheKey {
    fn forward(context: &ExecutionContext, spec: &crate::plan::GraphAlgorithmSpec) -> Self {
        Self::new(context, spec, GraphOrientation::Forward)
    }

    fn transposed(context: &ExecutionContext, spec: &crate::plan::GraphAlgorithmSpec) -> Self {
        Self::new(context, spec, GraphOrientation::Transposed)
    }

    fn new(context: &ExecutionContext, spec: &crate::plan::GraphAlgorithmSpec, orientation: GraphOrientation) -> Self {
        Self {
            runtime_ptr: context.runtime() as *const cudf::Runtime as usize,
            source_column: spec.edge_columns.source.clone(),
            destination_column: spec.edge_columns.destination.clone(),
            weight_column: spec.edge_columns.weight.clone(),
            edge_id_column: spec.edge_columns.edge_id.clone(),
            construction_policy: spec.graph_options.policy.as_str(),
            directed: spec.graph_options.directed,
            renumber: spec.graph_options.renumber,
            orientation,
        }
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
    _spec: &crate::plan::GraphAlgorithmSpec,
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
    let runtime = graph_execution_runtime(context, graph_memory_limit_bytes, operation)?;
    let handle = cugraph::ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).map_err(map_cugraph_error(
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
                runtime.as_ref(),
                operation,
            )
        })?;
    attach_graph_budget_observation(f(runtime.as_ref(), &entered, &mut graph), runtime.as_ref(), operation)
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
    let runtime = graph_execution_runtime(context, graph_memory_limit_bytes, operation)?;
    let handle = cugraph::ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).map_err(map_cugraph_error(
        "nexus_query_engine.graph_algorithm.resource_handle",
        "failed to bind cuGraph resource handle to the native cudf runtime",
    ))?;
    let entered = handle.enter(operation).map_err(map_cugraph_error(
        "nexus_query_engine.graph_algorithm.enter",
        "failed to enter cuGraph execution context",
    ))?;
    let mut graph = build_graph_from_edge_id_columns::<T>(&entered, edges, spec).map_err(|error| {
        if error.code() == QueryErrorCode::CudfResourceExhausted {
            attach_graph_budget_error_facts(error, runtime.as_ref(), operation)
        } else {
            error
        }
    })?;
    attach_graph_budget_observation(f(runtime.as_ref(), &entered, &mut graph), runtime.as_ref(), operation)
}

#[cfg(feature = "cugraph")]
fn graph_execution_runtime(
    context: &ExecutionContext,
    graph_memory_limit_bytes: Option<u64>,
    operation: &'static str,
) -> Result<std::sync::Arc<cudf::Runtime>> {
    let Some(limit_bytes) = graph_memory_limit_bytes else {
        return Ok(context.shared_runtime());
    };
    let limit = usize::try_from(limit_bytes).map_err(|_| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.budget_tracking_runtime",
            "graph memory reservation does not fit this platform's usize budget limit",
        )
        .with_fact("operation", operation)
        .with_fact("graph_memory_limit_bytes", limit_bytes)
    })?;
    let budget_mr = context.runtime().memory_resource().budget_tracking_limit(limit).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.graph_algorithm.budget_tracking_runtime",
            "failed to create budget-tracking memory resource for cuGraph execution",
            error,
        )
        .with_fact("operation", operation)
        .with_fact("graph_memory_limit_bytes", limit_bytes)
    })?;
    let runtime = cudf::Runtime::builder()
        .with_device(context.runtime().device())
        .with_stream(context.runtime().stream().clone())
        .with_memory_resource(budget_mr)
        .build()
        .map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.graph_algorithm.budget_tracking_runtime",
                "failed to build budget-tracking cudf runtime for cuGraph execution",
                error,
            )
            .with_fact("operation", operation)
            .with_fact("graph_memory_limit_bytes", limit_bytes)
        })?;
    Ok(std::sync::Arc::new(runtime))
}

#[cfg(feature = "cugraph")]
fn attach_graph_budget_observation(
    result: Result<GraphAlgorithmExecutionResult>,
    runtime: &cudf::Runtime,
    operation: &'static str,
) -> Result<GraphAlgorithmExecutionResult> {
    match result {
        Ok(mut result) => {
            if let Some(observation) = graph_budget_observation(runtime, operation, "budget_tracking_enforced") {
                result.budget_observations.push(observation);
            }
            Ok(result)
        },
        Err(error) => Err(attach_graph_budget_error_facts(error, runtime, operation)),
    }
}

#[cfg(feature = "cugraph")]
fn graph_budget_observation(
    runtime: &cudf::Runtime,
    operation: &'static str,
    status: &'static str,
) -> Option<GraphBudgetObservation> {
    runtime
        .memory_resource()
        .budget_tracking_usage()
        .ok()
        .map(|usage| GraphBudgetObservation::from_usage(operation, usage, status))
}

#[cfg(feature = "cugraph")]
fn attach_graph_budget_error_facts(
    mut error: QueryError,
    runtime: &cudf::Runtime,
    operation: &'static str,
) -> QueryError {
    let status = if error.code() == QueryErrorCode::CudfResourceExhausted {
        "budget_tracking_resource_exhausted"
    } else {
        "budget_tracking_failed"
    };
    let Some(observation) = graph_budget_observation(runtime, operation, status) else {
        return error;
    };
    error = error
        .with_fact("graph_budget_operation", observation.operation)
        .with_fact("graph_budget_limit_bytes", observation.budget_bytes)
        .with_fact("graph_budget_status", observation.status);
    if let Some(peak) = observation.allocator_high_water_bytes {
        error = error.with_fact("graph_budget_peak_bytes", peak);
    }
    for (key, value) in observation.context_facts {
        error = match key.as_str() {
            "allocator_failed_allocation_count" => error.with_fact("graph_budget_failed_allocation_count", value),
            "allocator_largest_failed_allocation_bytes" => {
                error.with_fact("graph_budget_largest_failed_allocation_bytes", value)
            },
            _ => error,
        };
    }
    error
}

#[cfg(feature = "cugraph")]
fn record_graph_budget_observation_from_error(metrics: &mut QueryMetrics, error: &QueryError) {
    record_graph_budget_observation_from_error_with_boundary(
        metrics,
        error,
        crate::graph_memory::GRAPH_MEMORY_BOUNDARY,
        Some("cugraph_workspace_preflight"),
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
    let status = error.fact_value("graph_budget_status").unwrap_or("budget_tracking_failed");
    let mut context_facts = BTreeMap::new();
    for key in ["graph_budget_failed_allocation_count", "graph_budget_largest_failed_allocation_bytes"] {
        if let Some(value) = error.fact_value(key) {
            context_facts.insert(key.to_string(), value.to_string());
        }
    }
    metrics.record_graph_budget_tracking_with_boundary(
        operation.to_string(),
        boundary,
        unresolved_contract.map(str::to_string),
        budget_bytes,
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
fn graph_memory_model_input_for_frame(
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
pub(in crate::exec) fn runtime_exact_vertex_estimate_with_budget(
    context: &ExecutionContext,
    frame: &DataFrame,
    spec: &crate::plan::GraphAlgorithmSpec,
    budget_limit_bytes: u64,
    metrics: &mut QueryMetrics,
) -> Result<crate::graph_memory::GraphVertexEstimate> {
    let operation = "nexus_query_engine.graph_algorithm.graph_memory_vertex_estimate";
    let runtime = graph_execution_runtime(context, Some(budget_limit_bytes), operation)?;
    let estimate = runtime_exact_vertex_estimate_in_runtime(runtime.as_ref(), frame, spec);
    match estimate {
        Ok(estimate) => {
            if let Some(observation) = graph_budget_observation(runtime.as_ref(), operation, "budget_tracking_enforced")
            {
                observation.record_with_boundary(
                    metrics,
                    crate::graph_memory::GRAPH_VERTEX_ESTIMATE_BOUNDARY,
                    Some("graph_vertex_estimate_probe"),
                );
            }
            Ok(estimate)
        },
        Err(error) => {
            let error = attach_graph_budget_error_facts(error, runtime.as_ref(), operation);
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
fn runtime_exact_vertex_estimate_in_runtime(
    runtime: &cudf::Runtime,
    frame: &DataFrame,
    spec: &crate::plan::GraphAlgorithmSpec,
) -> Result<crate::graph_memory::GraphVertexEstimate> {
    let operation = "nexus_query_engine.graph_algorithm.graph_memory_vertex_estimate";
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
    let source = source.copy_into_runtime(runtime).map_err(|error| {
        QueryError::from_cudf(operation, "failed to place graph source column in budgeted runtime", error)
    })?;
    let destination = destination.copy_into_runtime(runtime).map_err(|error| {
        QueryError::from_cudf(operation, "failed to place graph destination column in budgeted runtime", error)
    })?;
    let count = cudf::Series::concat(&[&source, &destination])
        .and_then(|series| series.nunique())
        .map_err(|error| QueryError::from_cudf(operation, "failed to count distinct graph endpoints", error))?;
    let count = u64::try_from(count).map_err(|error| {
        QueryError::new(QueryErrorCode::InvalidPlan, operation, "runtime graph vertex estimate does not fit u64")
            .with_source(error)
    })?;
    Ok(crate::graph_memory::GraphVertexEstimate::runtime_exact_distinct(count))
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
) -> cugraph::structure::KCoreDegreeType {
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
