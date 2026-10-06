use cudf::{Exec, Series};
use query_planning::plan::GraphAlgorithmSpec;

use super::{
    DataFrame, ExecutionContext, QueryError, QueryErrorCode, Result,
    bounded_betweenness::{RuntimeGraphSpecResolution, runtime_graph_spec_resolution_after_bounded_seed_validation},
    cudf_query_error,
};

/// Runs the runtime graph vertex-count probe on the attempt's own runtime.
///
/// Runtime sizing runs only when the seed policy requires a vertex count.
/// The caller may omit the count after a typed allocation failure; seed-policy
/// validation remains part of graph execution.
#[cfg(feature = "cugraph")]
pub(in crate::exec) fn runtime_graph_statistics_probe(
    context: &ExecutionContext,
    frame: &DataFrame,
    spec: &GraphAlgorithmSpec,
) -> Result<Option<u64>> {
    if !matches!(
        runtime_graph_spec_resolution_after_bounded_seed_validation(spec, false),
        RuntimeGraphSpecResolution::Unresolved
    ) {
        return Ok(None);
    }
    let runtime = context.runtime()?;
    runtime_graph_statistics_probe_in_runtime(&runtime, frame, spec).map(Some)
}

#[cfg(feature = "cugraph")]
fn runtime_graph_statistics_probe_in_runtime(
    runtime: &cudf::Runtime,
    frame: &DataFrame,
    spec: &GraphAlgorithmSpec,
) -> Result<u64> {
    let operation = GRAPH_STATISTICS_OPERATION;
    let ctx = runtime.exec();
    let (source, destination) = distinct_graph_endpoints(frame, spec, ctx, operation)?;
    let endpoint_count = source.len() as u64 + destination.len() as u64;
    if source.is_empty() || destination.is_empty() {
        return Ok(endpoint_count);
    }
    let overlap = source
        .on(ctx)
        .isin(&destination)
        .and_then(|mask| mask.on(ctx).sum_value::<i64>())
        .map_err(|error| cudf_query_error(operation, "failed to count overlapping graph endpoints", error))?;
    let overlap = u64::try_from(overlap).map_err(|error| {
        QueryError::new(QueryErrorCode::InternalInvariantViolation, operation, "graph endpoint overlap is negative")
            .with_source(error)
    })?;
    Ok(endpoint_count - overlap)
}

pub(super) fn distinct_graph_endpoints(
    frame: &DataFrame,
    spec: &GraphAlgorithmSpec,
    ctx: Exec<'_>,
    operation: &'static str,
) -> Result<(Series, Series)> {
    let source = frame
        .series(&spec.edge_columns.source)
        .map_err(|error| cudf_query_error(operation, "failed to read graph source column", error))?;
    let destination = frame
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
    let source = source
        .on(ctx)
        .unique()
        .map_err(|error| cudf_query_error(operation, "failed to collect distinct graph source vertices", error))?;
    let destination = destination
        .on(ctx)
        .unique()
        .map_err(|error| cudf_query_error(operation, "failed to collect distinct graph destination vertices", error))?;
    Ok((source, destination))
}

#[cfg(feature = "cugraph")]
const GRAPH_STATISTICS_OPERATION: &str = "query_engine.graph_algorithm.graph_memory_vertex_estimate";
