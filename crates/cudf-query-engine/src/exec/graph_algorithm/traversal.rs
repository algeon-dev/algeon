use super::*;

pub(super) fn execute_bfs(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    input: GpuDataFrame,
) -> Result<GraphAlgorithmExecutionResult> {
    let crate::plan::GraphAlgorithmParams::Bfs(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.graph_algorithm.bfs",
            "BFS graph algorithm node carried non-BFS parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, vertex_type) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);
    let bfs_p = bfs_params(params)?;

    run_with_cugraph_graph::<cugraph::Forward, _, _>(
        context,
        edges,
        options,
        "datafusion_cudf_cugraph_bfs",
        |runtime, entered, graph| {
            let frame = match vertex_type {
                cudf::TypeId::Int32 => {
                    let source = i32::try_from(params.source_vertex).map_err(|_| {
                        QueryError::new(
                            QueryErrorCode::InvalidPlan,
                            "cudf_query_engine.graph_algorithm.bfs_source",
                            "BFS source vertex does not fit the Int32 graph vertex dtype",
                        )
                        .with_fact("source_vertex", params.source_vertex)
                        .with_fact("vertex_dtype", "Int32")
                    })?;
                    let sources = cugraph::DeviceArray::from_slice(entered, &[source]).map_err(map_cugraph_error(
                        "cudf_query_engine.graph_algorithm.bfs_source",
                        "failed to copy BFS source vertex to device memory",
                    ))?;
                    let sources_view = sources.view().map_err(map_cugraph_error(
                        "cudf_query_engine.graph_algorithm.bfs_source",
                        "failed to create BFS source vertex view",
                    ))?;
                    let result = graph
                        .bfs(entered, &sources_view, &bfs_p)
                        .map_err(map_cugraph_error("cudf_query_engine.graph_algorithm.bfs", "cuGraph BFS failed"))?;
                    result.to_dataframe(runtime).map_err(map_cugraph_error(
                        "cudf_query_engine.graph_algorithm.bfs_result",
                        "failed to convert cuGraph BFS result to cudf DataFrame",
                    ))?
                },
                cudf::TypeId::Int64 => {
                    let sources = cugraph::DeviceArray::from_slice(entered, &[params.source_vertex]).map_err(
                        map_cugraph_error(
                            "cudf_query_engine.graph_algorithm.bfs_source",
                            "failed to copy BFS source vertex to device memory",
                        ),
                    )?;
                    let sources_view = sources.view().map_err(map_cugraph_error(
                        "cudf_query_engine.graph_algorithm.bfs_source",
                        "failed to create BFS source vertex view",
                    ))?;
                    let result = graph
                        .bfs(entered, &sources_view, &bfs_p)
                        .map_err(map_cugraph_error("cudf_query_engine.graph_algorithm.bfs", "cuGraph BFS failed"))?;
                    result.to_dataframe(runtime).map_err(map_cugraph_error(
                        "cudf_query_engine.graph_algorithm.bfs_result",
                        "failed to convert cuGraph BFS result to cudf DataFrame",
                    ))?
                },
                _ => {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedType,
                        "cudf_query_engine.graph_algorithm.bfs_source",
                        "BFS graph vertex dtype is unsupported",
                    )
                    .with_fact("vertex_dtype", format!("{vertex_type:?}")));
                },
            };
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata: format!(
                    "source_vertex={},depth_limit={},compute_predecessors=true",
                    params.source_vertex,
                    params.depth_limit.map_or_else(|| "none".to_string(), |value| value.to_string())
                ),
            })
        },
    )
}

pub(super) fn execute_sssp(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    input: GpuDataFrame,
) -> Result<GraphAlgorithmExecutionResult> {
    let crate::plan::GraphAlgorithmParams::Sssp(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.graph_algorithm.sssp",
            "SSSP graph algorithm node carried non-SSSP parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, vertex_type) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);
    let sssp_p = sssp_params(params)?;

    run_with_cugraph_graph::<cugraph::Forward, _, _>(
        context,
        edges,
        options,
        "datafusion_cudf_cugraph_sssp",
        |runtime, entered, graph| {
            let frame = match vertex_type {
                cudf::TypeId::Int32 => {
                    let source = i32::try_from(params.source_vertex).map_err(|_| {
                        QueryError::new(
                            QueryErrorCode::InvalidPlan,
                            "cudf_query_engine.graph_algorithm.sssp_source",
                            "SSSP source vertex does not fit the Int32 graph vertex dtype",
                        )
                        .with_fact("source_vertex", params.source_vertex)
                        .with_fact("vertex_dtype", "Int32")
                    })?;
                    let result = graph
                        .sssp(entered, source, &sssp_p)
                        .map_err(map_cugraph_error("cudf_query_engine.graph_algorithm.sssp", "cuGraph SSSP failed"))?;
                    result.to_dataframe(runtime).map_err(map_cugraph_error(
                        "cudf_query_engine.graph_algorithm.sssp_result",
                        "failed to convert cuGraph SSSP result to cudf DataFrame",
                    ))?
                },
                cudf::TypeId::Int64 => {
                    let result = graph
                        .sssp(entered, params.source_vertex, &sssp_p)
                        .map_err(map_cugraph_error("cudf_query_engine.graph_algorithm.sssp", "cuGraph SSSP failed"))?;
                    result.to_dataframe(runtime).map_err(map_cugraph_error(
                        "cudf_query_engine.graph_algorithm.sssp_result",
                        "failed to convert cuGraph SSSP result to cudf DataFrame",
                    ))?
                },
                _ => {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedType,
                        "cudf_query_engine.graph_algorithm.sssp_source",
                        "SSSP graph vertex dtype is unsupported",
                    )
                    .with_fact("vertex_dtype", format!("{vertex_type:?}")));
                },
            };
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata: format!(
                    "source_vertex={},cutoff={}",
                    params.source_vertex,
                    params.cutoff.map_or_else(|| "none".to_string(), |v| v.to_string()),
                ),
            })
        },
    )
}
