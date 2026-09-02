use super::{
    DataFrame, EdgeColumnRole, ExecutionContext, GraphAlgorithmExecutionResult, QueryError, QueryErrorCode, Result,
    attach_graph_budget_error_facts, attach_graph_budget_observation, graph_build_range, graph_execution_budget,
    map_cugraph_error, validate_edge_column,
};

#[cfg(feature = "cugraph")]
pub(super) fn run_with_uncached_cugraph_graph<T, F>(
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
            "failed to bind cuGraph resource handle to the nexus native runtime",
        ))?;
    let entered = handle.enter(operation).map_err(map_cugraph_error(
        "nexus_query_engine.graph_algorithm.enter",
        "failed to enter cuGraph execution context",
    ))?;
    let _graph_build_range = graph_build_range(operation);
    let mut graph =
        cugraph::cudf_interop::graph_from_prepared_table_export::<T>(&entered, &export, &options).map_err(|error| {
            attach_graph_budget_error_facts(
                map_cugraph_error(
                    "nexus_query_engine.graph_algorithm.build_graph",
                    "failed to build cuGraph graph from edge-list table",
                )(error),
                &budget.token,
                operation,
            )
        })?;
    attach_graph_budget_observation(f(budget.runtime.as_ref(), &entered, &mut graph), &budget.token, operation)
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
    if spec.edge_columns.edge_id.is_some() {
        let edge_id_col = 2 + usize::from(options.weight_col.is_some());
        options = options.edge_id_col(edge_id_col);
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
