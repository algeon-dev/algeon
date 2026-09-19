//! Runtime betweenness seed policy and validation.

use super::{CudfSortOptions, DataFrame, QueryError, QueryErrorCode, Result, cudf_query_error};

#[cfg(feature = "cugraph")]
pub(in crate::exec) enum RuntimeGraphSpecResolution {
    Unresolved,
    CheckedNoRewrite,
    Rewritten(Box<crate::plan::GraphAlgorithmSpec>),
}

#[cfg(feature = "cugraph")]
impl RuntimeGraphSpecResolution {
    fn from_optional(spec: Option<crate::plan::GraphAlgorithmSpec>) -> Self {
        match spec {
            Some(spec) => Self::Rewritten(Box::new(spec)),
            None => Self::CheckedNoRewrite,
        }
    }

    pub(super) fn spec_or<'a>(
        &'a self,
        fallback: &'a crate::plan::GraphAlgorithmSpec,
    ) -> &'a crate::plan::GraphAlgorithmSpec {
        match self {
            Self::Unresolved | Self::CheckedNoRewrite => fallback,
            Self::Rewritten(spec) => spec.as_ref(),
        }
    }
}

#[cfg(feature = "cugraph")]
pub(super) fn resolve_runtime_graph_spec_for_execution(
    input: &DataFrame,
    spec: &crate::plan::GraphAlgorithmSpec,
) -> Result<RuntimeGraphSpecResolution> {
    if betweenness_seed_policy_operation(spec).is_none() {
        return Ok(RuntimeGraphSpecResolution::CheckedNoRewrite);
    }
    graph_spec_with_runtime_betweenness_seeds(input, spec).map(RuntimeGraphSpecResolution::from_optional)
}

#[cfg(feature = "cugraph")]
fn betweenness_seed_policy_operation(spec: &crate::plan::GraphAlgorithmSpec) -> Option<&'static str> {
    match &spec.params {
        crate::plan::GraphAlgorithmParams::BetweennessCentrality(_) => {
            Some("query_engine.graph_algorithm.betweenness_seed_policy")
        },
        crate::plan::GraphAlgorithmParams::EdgeBetweennessCentrality(_) => {
            Some("query_engine.graph_algorithm.edge_betweenness_seed_policy")
        },
        _ => None,
    }
}

#[cfg(feature = "cugraph")]
fn graph_spec_with_runtime_betweenness_seeds(
    input: &DataFrame,
    spec: &crate::plan::GraphAlgorithmSpec,
) -> Result<Option<crate::plan::GraphAlgorithmSpec>> {
    let (seeds, k, operation) = match &spec.params {
        crate::plan::GraphAlgorithmParams::BetweennessCentrality(params) => {
            (params.seeds.as_ref(), params.k, "query_engine.graph_algorithm.betweenness_seed_policy")
        },
        crate::plan::GraphAlgorithmParams::EdgeBetweennessCentrality(params) => {
            (params.seeds.as_ref(), params.k, "query_engine.graph_algorithm.edge_betweenness_seed_policy")
        },
        _ => return Ok(None),
    };

    if seeds.is_some() {
        validate_betweenness_seed_policy_for_vertex_count(spec, 0)?;
        return Ok(None);
    }

    let vertices = sorted_graph_vertices_i64(input, spec, operation)?;
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
            "query_engine.graph_algorithm.betweenness_seed_policy",
        ),
        crate::plan::GraphAlgorithmParams::EdgeBetweennessCentrality(params) => (
            params.seeds.as_ref(),
            params.k,
            params.exact_vertex_threshold,
            "query_engine.graph_algorithm.edge_betweenness_seed_policy",
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
            (params.seeds.as_ref(), params.k, "query_engine.graph_algorithm.betweenness_seed_policy")
        },
        crate::plan::GraphAlgorithmParams::EdgeBetweennessCentrality(params) => {
            (params.seeds.as_ref(), params.k, "query_engine.graph_algorithm.edge_betweenness_seed_policy")
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
) -> Result<Vec<i64>> {
    let source = input
        .series(&spec.edge_columns.source)
        .map_err(|error| cudf_query_error(operation, "failed to read graph source column", error))?;
    let destination = input
        .series(&spec.edge_columns.destination)
        .map_err(|error| cudf_query_error(operation, "failed to read graph destination column", error))?;
    let source_dtype =
        source.dtype().map_err(|error| cudf_query_error(operation, "failed to inspect graph source dtype", error))?;
    let destination_dtype = destination
        .dtype()
        .map_err(|error| cudf_query_error(operation, "failed to inspect graph destination dtype", error))?;
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
    let vertices = cudf::Series::concat(&[&source, &destination])
        .and_then(|series| series.unique())
        .and_then(|series| series.sort_values(CudfSortOptions::new()))
        .map_err(|error| cudf_query_error(operation, "failed to collect graph vertices", error))?;
    match source_dtype.id() {
        cudf::TypeId::Int32 => vertices
            .to_vec_strict::<i32>()
            .map(|values| values.into_iter().map(i64::from).collect())
            .map_err(|error| cudf_query_error(operation, "failed to copy graph Int32 vertices to host", error)),
        cudf::TypeId::Int64 => vertices
            .to_vec_strict::<i64>()
            .map_err(|error| cudf_query_error(operation, "failed to copy graph Int64 vertices to host", error)),
        dtype => Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            operation,
            "betweenness centrality requires Int32 or Int64 graph vertices",
        )
        .with_fact("vertex_dtype", format!("{dtype:?}"))),
    }
}

#[cfg(feature = "cugraph")]
fn graph_spec_with_betweenness_seeds(
    spec: &crate::plan::GraphAlgorithmSpec,
    seeds: crate::plan::GraphBetweennessSeeds,
) -> Result<crate::plan::GraphAlgorithmSpec> {
    let mut resolved = spec.clone();
    match &mut resolved.params {
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
                "query_engine.graph_algorithm.betweenness_seed_policy",
                "betweenness seed resolution requires betweenness parameters",
            )
            .with_fact("algorithm", spec.algorithm.as_str()));
        },
    }
    Ok(resolved)
}
