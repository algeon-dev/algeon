use std::sync::atomic::{AtomicBool, Ordering};

use super::*;

#[cfg(feature = "cugraph")]
pub(super) static FORCE_NEXT_GRAPH_ADAPTIVE_FULL_ATTEMPT_RESOURCE_EXHAUSTED: AtomicBool = AtomicBool::new(false);
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

    pub(super) fn spec_or<'a>(
        &'a self,
        fallback: &'a crate::plan::GraphAlgorithmSpec,
    ) -> &'a crate::plan::GraphAlgorithmSpec {
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
pub(super) fn resolve_runtime_graph_spec_for_execution(
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
pub(super) fn execute_chunked_betweenness_retry(
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
pub(super) fn graph_adaptive_retry_seed_batches(
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
pub(super) fn record_graph_adaptive_memory_history_from_error(
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
pub(super) fn graph_adaptive_memory_observation_from_error(
    error: &QueryError,
) -> Option<crate::adaptive_runtime::AdaptiveGraphMemoryObservation> {
    let allocator_high_water_bytes =
        error.fact_value("graph_budget_peak_bytes").and_then(|value| value.parse::<u64>().ok())?;
    let budget_bytes = error.fact_value("graph_budget_limit_bytes").and_then(|value| value.parse::<u64>().ok())?;
    Some(crate::adaptive_runtime::AdaptiveGraphMemoryObservation { allocator_high_water_bytes, budget_bytes })
}
