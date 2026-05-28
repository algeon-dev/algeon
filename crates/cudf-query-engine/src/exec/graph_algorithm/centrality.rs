use super::*;

pub(super) fn execute_pagerank(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult> {
    let input = inputs.edge_list;
    let crate::plan::GraphAlgorithmParams::PageRank(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.graph_algorithm.pagerank",
            "PageRank graph algorithm node carried non-PageRank parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);
    let pr_params = pagerank_params(params)?;

    run_with_cugraph_graph::<cugraph::Transposed, _, _>(
        context,
        edges,
        options,
        "datafusion_cudf_cugraph_pagerank",
        |runtime, entered, graph| {
            let result = graph
                .pagerank(entered, &pr_params)
                .map_err(map_cugraph_error("cudf_query_engine.graph_algorithm.pagerank", "cuGraph PageRank failed"))?;
            let converged = result.converged().map_or_else(|_| "unknown".to_string(), |value| value.to_string());
            let algorithm_metadata = format!("converged={},iterations={}", converged, result.num_iterations());
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.pagerank_result",
                "failed to convert cuGraph PageRank result to cudf DataFrame",
            ))?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata,
            })
        },
    )
}

pub(super) fn execute_eigenvector_centrality(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult> {
    let input = inputs.edge_list;
    let crate::plan::GraphAlgorithmParams::EigenvectorCentrality(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.graph_algorithm.eigenvector_centrality",
            "EigenvectorCentrality graph algorithm node carried non-EigenvectorCentrality parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);
    let evc_params = eigenvector_centrality_params(params)?;

    run_with_cugraph_graph::<cugraph::Transposed, _, _>(
        context,
        edges,
        options,
        "datafusion_cudf_cugraph_eigenvector_centrality",
        |runtime, entered, graph| {
            let result = graph.eigenvector_centrality(entered, &evc_params).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.eigenvector_centrality",
                "cuGraph eigenvector centrality failed",
            ))?;
            let converged = result.converged().map_or_else(|_| "unknown".to_string(), |value| value.to_string());
            let algorithm_metadata = format!("converged={},iterations={}", converged, result.num_iterations());
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.eigenvector_centrality_result",
                "failed to convert cuGraph eigenvector centrality result to cudf DataFrame",
            ))?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata,
            })
        },
    )
}

pub(super) fn execute_hits(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult> {
    let input = inputs.edge_list;
    let crate::plan::GraphAlgorithmParams::Hits(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.graph_algorithm.hits",
            "HITS graph algorithm node carried non-HITS parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);
    let h_params = hits_params(params)?;

    run_with_cugraph_graph::<cugraph::Transposed, _, _>(
        context,
        edges,
        options,
        "datafusion_cudf_cugraph_hits",
        |runtime, entered, graph| {
            let result = graph
                .hits(entered, &h_params)
                .map_err(map_cugraph_error("cudf_query_engine.graph_algorithm.hits", "cuGraph HITS failed"))?;
            let algorithm_metadata = format!(
                "iterations={},hub_score_differences={}",
                result.number_of_iterations(),
                result.hub_score_differences(),
            );
            // Upstream HITS emits `vertex` / `hub` / `authority`; the stable
            // SQL schema renames the score columns to `hub_score` /
            // `authority_score` so the meaning is unambiguous at query time.
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.hits_result",
                "failed to convert cuGraph HITS result to cudf DataFrame",
            ))?;
            let frame =
                frame.rename_columns([("hub", "hub_score"), ("authority", "authority_score")]).map_err(|error| {
                    QueryError::from_cudf(
                        "cudf_query_engine.graph_algorithm.hits_result",
                        "failed to rename HITS score columns to the stable SQL schema",
                        error,
                    )
                })?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata,
            })
        },
    )
}

pub(super) fn execute_katz_centrality(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult> {
    let input = inputs.edge_list;
    let crate::plan::GraphAlgorithmParams::KatzCentrality(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.graph_algorithm.katz_centrality",
            "KatzCentrality graph algorithm node carried non-KatzCentrality parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);
    let kc_params = katz_centrality_params(params)?;

    run_with_cugraph_graph::<cugraph::Transposed, _, _>(
        context,
        edges,
        options,
        "datafusion_cudf_cugraph_katz_centrality",
        |runtime, entered, graph| {
            let result = graph.katz_centrality(entered, &kc_params).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.katz_centrality",
                "cuGraph Katz centrality failed",
            ))?;
            let converged = result.converged().map_or_else(|_| "unknown".to_string(), |value| value.to_string());
            let algorithm_metadata = format!("converged={},iterations={}", converged, result.num_iterations());
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.katz_centrality_result",
                "failed to convert cuGraph Katz centrality result to cudf DataFrame",
            ))?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata,
            })
        },
    )
}

pub(super) fn execute_pagerank_strict(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult> {
    let input = inputs.edge_list;
    let crate::plan::GraphAlgorithmParams::PageRankStrict(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.graph_algorithm.pagerank_strict",
            "PageRankStrict graph algorithm node carried non-PageRankStrict parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);
    let pr_params = pagerank_params(params)?;

    run_with_cugraph_graph::<cugraph::Transposed, _, _>(
        context,
        edges,
        options,
        "datafusion_cudf_cugraph_pagerank_strict",
        |runtime, entered, graph| {
            let result = graph.pagerank_strict(entered, &pr_params).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.pagerank_strict",
                "cuGraph PageRank strict failed",
            ))?;
            let converged = result.converged().map_or_else(|_| "unknown".to_string(), |value| value.to_string());
            let algorithm_metadata = format!("converged={},iterations={}", converged, result.num_iterations());
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.pagerank_strict_result",
                "failed to convert cuGraph PageRank strict result to cudf DataFrame",
            ))?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata,
            })
        },
    )
}
