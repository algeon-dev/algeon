use super::{DataFrame, ExecutionContext, ExecutionOverrides, QueryError, QueryErrorCode, Result};

#[cfg(feature = "cugraph")]
thread_local! {
    // Graph statistics probing is synchronous today. Keeping this test hook
    // thread-local prevents one integration test from consuming another test's
    // forced optional-resource exhaustion.
    static FORCE_NEXT_GRAPH_STATISTICS_OPTIONAL_RESOURCE_EXHAUSTED: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

#[cfg(all(any(test, feature = "test-hooks"), feature = "cugraph"))]
pub fn force_next_graph_statistics_optional_resource_exhausted_for_test() {
    FORCE_NEXT_GRAPH_STATISTICS_OPTIONAL_RESOURCE_EXHAUSTED.with(|force| force.set(true));
}

/// Runs the runtime graph vertex-count probe on the attempt's own runtime.
///
/// This runtime sizing only informs betweenness seed selection. It carries no
/// reservation or rejection authority, so failure degrades to a missing
/// estimate instead of failing the query.
#[cfg(feature = "cugraph")]
pub(in crate::exec) fn runtime_graph_statistics_probe(
    context: &ExecutionContext,
    frame: &DataFrame,
    spec: &crate::plan::GraphAlgorithmSpec,
) -> Result<u64> {
    let runtime = context.runtime()?;
    runtime_graph_statistics_probe_in_runtime(&runtime, frame, spec)
}

#[cfg(feature = "cugraph")]
fn runtime_graph_statistics_probe_in_runtime(
    runtime: &cudf::Runtime,
    frame: &DataFrame,
    spec: &crate::plan::GraphAlgorithmSpec,
) -> Result<u64> {
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
    u64::try_from(count).map_err(|error| {
        QueryError::new(QueryErrorCode::InvalidPlan, operation, "runtime graph vertex estimate does not fit u64")
            .with_source(error)
    })
}

#[cfg(feature = "cugraph")]
const GRAPH_STATISTICS_OPERATION: &str = "nexus_query_engine.graph_algorithm.graph_memory_vertex_estimate";

#[cfg(feature = "cugraph")]
fn graph_statistics_execution_overrides(runtime: &cudf::Runtime) -> ExecutionOverrides<'_> {
    ExecutionOverrides::new().with_stream(runtime.stream()).with_memory_resource(runtime.memory_resource())
}
