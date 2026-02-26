use super::*;

pub(super) fn execute_degrees_all(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult> {
    let input = inputs.edge_list;
    let crate::plan::GraphAlgorithmParams::DegreesAll(_) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.degrees_all",
            "DegreesAll graph algorithm node carried non-DegreesAll parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);

    run_with_cugraph_graph::<cugraph::Transposed, _, _>(
        context,
        edges,
        options,
        "datafusion_nexus_cugraph_degrees_all",
        |runtime, entered, graph| {
            let result = graph.degrees_all(entered).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.degrees_all",
                "cuGraph degrees_all failed",
            ))?;
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.degrees_all_result",
                "failed to convert cuGraph degrees result to cudf DataFrame",
            ))?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata: String::new(),
            })
        },
    )
}

pub(super) fn execute_in_degrees_all(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult> {
    let input = inputs.edge_list;
    let crate::plan::GraphAlgorithmParams::InDegreesAll(_) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.in_degrees_all",
            "InDegreesAll graph algorithm node carried non-InDegreesAll parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);

    run_with_cugraph_graph::<cugraph::Transposed, _, _>(
        context,
        edges,
        options,
        "datafusion_nexus_cugraph_in_degrees_all",
        |runtime, entered, graph| {
            let result = graph.in_degrees_all(entered).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.in_degrees_all",
                "cuGraph in_degrees_all failed",
            ))?;
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.in_degrees_all_result",
                "failed to convert cuGraph in-degrees result to cudf DataFrame",
            ))?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata: String::new(),
            })
        },
    )
}

pub(super) fn execute_out_degrees_all(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult> {
    let input = inputs.edge_list;
    let crate::plan::GraphAlgorithmParams::OutDegreesAll(_) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.out_degrees_all",
            "OutDegreesAll graph algorithm node carried non-OutDegreesAll parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);

    // out_degrees_all is layout-agnostic in cugraph-rs; build Forward for
    // consistency with other Any-layout algorithms in this adapter.
    run_with_cugraph_graph::<cugraph::Forward, _, _>(
        context,
        edges,
        options,
        "datafusion_nexus_cugraph_out_degrees_all",
        |runtime, entered, graph| {
            let result = graph.out_degrees_all(entered).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.out_degrees_all",
                "cuGraph out_degrees_all failed",
            ))?;
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.out_degrees_all_result",
                "failed to convert cuGraph out-degrees result to cudf DataFrame",
            ))?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata: String::new(),
            })
        },
    )
}

pub(super) fn execute_weakly_connected_components(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult> {
    let input = inputs.edge_list;
    let crate::plan::GraphAlgorithmParams::WeaklyConnectedComponents(_) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.weakly_connected_components",
            "WeaklyConnectedComponents graph algorithm node carried wrong parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);

    run_with_cugraph_graph::<cugraph::Forward, _, _>(
        context,
        edges,
        options,
        "datafusion_nexus_cugraph_weakly_connected_components",
        |runtime, entered, graph| {
            let wcc_params = cugraph::labeling::ConnectedComponentsParams::default();
            let result = graph.weakly_connected_components(entered, &wcc_params).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.weakly_connected_components",
                "cuGraph weakly_connected_components failed",
            ))?;
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.weakly_connected_components_result",
                "failed to convert cuGraph WCC result to cudf DataFrame",
            ))?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata: String::new(),
            })
        },
    )
}

pub(super) fn execute_strongly_connected_components(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult> {
    let input = inputs.edge_list;
    let crate::plan::GraphAlgorithmParams::StronglyConnectedComponents(_) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.strongly_connected_components",
            "StronglyConnectedComponents graph algorithm node carried wrong parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);

    run_with_cugraph_graph::<cugraph::Forward, _, _>(
        context,
        edges,
        options,
        "datafusion_nexus_cugraph_strongly_connected_components",
        |runtime, entered, graph| {
            let scc_params = cugraph::labeling::ConnectedComponentsParams::default();
            let result = graph.strongly_connected_components(entered, &scc_params).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.strongly_connected_components",
                "cuGraph strongly_connected_components failed",
            ))?;
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.strongly_connected_components_result",
                "failed to convert cuGraph SCC result to cudf DataFrame",
            ))?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata: String::new(),
            })
        },
    )
}

pub(super) fn execute_triangle_count_all(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult> {
    let input = inputs.edge_list;
    let crate::plan::GraphAlgorithmParams::TriangleCountAll(_) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.triangle_count_all",
            "TriangleCountAll graph algorithm node carried wrong parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);

    run_with_cugraph_graph::<cugraph::Forward, _, _>(
        context,
        edges,
        options,
        "datafusion_nexus_cugraph_triangle_count_all",
        |runtime, entered, graph| {
            let tc_params = cugraph::community::TriangleCountParams::default();
            let result = graph.triangle_count_all(entered, &tc_params).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.triangle_count_all",
                "cuGraph triangle_count_all failed",
            ))?;
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.triangle_count_all_result",
                "failed to convert cuGraph triangle-count result to cudf DataFrame",
            ))?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata: String::new(),
            })
        },
    )
}

pub(super) fn execute_core_number(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult> {
    let input = inputs.edge_list;
    let crate::plan::GraphAlgorithmParams::CoreNumber(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.core_number",
            "CoreNumber graph algorithm node carried non-CoreNumber parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);
    let cn_params = cugraph::structure::CoreNumberParams {
        degree_type: degree_type_to_cugraph(params.degree_type),
        do_expensive_check: false,
    };

    run_with_cugraph_graph::<cugraph::Forward, _, _>(
        context,
        edges,
        options,
        "datafusion_nexus_cugraph_core_number",
        |runtime, entered, graph| {
            let result = graph.core_number(entered, &cn_params).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.core_number",
                "cuGraph core_number failed",
            ))?;
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.core_number_result",
                "failed to convert cuGraph core-number result to cudf DataFrame",
            ))?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata: format!("degree_type={}", params.degree_type.as_str()),
            })
        },
    )
}

pub(super) fn execute_k_core(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult> {
    let input = inputs.edge_list;
    let crate::plan::GraphAlgorithmParams::KCore(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.k_core",
            "KCore graph algorithm node carried non-KCore parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);
    let kc_params = k_core_params(params)?;

    run_with_cugraph_graph::<cugraph::Forward, _, _>(
        context,
        edges,
        options,
        "datafusion_nexus_cugraph_k_core",
        |runtime, entered, graph| {
            let result = graph
                .k_core(entered, &kc_params)
                .map_err(map_cugraph_error("nexus_query_engine.graph_algorithm.k_core", "cuGraph k_core failed"))?;
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.k_core_result",
                "failed to convert cuGraph k-core result to cudf DataFrame",
            ))?;
            // cudf_interop produces "source" / "destination"; stable schema uses "src" / "dst".
            let frame = frame.rename_columns([("source", "src"), ("destination", "dst")]).map_err(|error| {
                QueryError::from_cudf(
                    "nexus_query_engine.graph_algorithm.k_core_result",
                    "failed to rename k-core edge columns to the stable SQL schema",
                    error,
                )
            })?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata: format!("k={},degree_type={}", params.k, params.degree_type.as_str()),
            })
        },
    )
}

pub(super) fn execute_minimum_spanning_tree(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult> {
    let input = inputs.edge_list;
    let crate::plan::GraphAlgorithmParams::MinimumSpanningTree(_) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.minimum_spanning_tree",
            "MinimumSpanningTree graph algorithm node carried wrong parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);

    run_with_cugraph_graph::<cugraph::Forward, _, _>(
        context,
        edges,
        options,
        "datafusion_nexus_cugraph_minimum_spanning_tree",
        |runtime, entered, graph| {
            let mst_params = cugraph::tree::MinimumSpanningTreeParams::default();
            let result = graph.minimum_spanning_tree(entered, &mst_params).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.minimum_spanning_tree",
                "cuGraph minimum_spanning_tree failed",
            ))?;
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.minimum_spanning_tree_result",
                "failed to convert cuGraph MST result to cudf DataFrame",
            ))?;
            // cudf_interop produces "source" / "destination"; stable schema uses "src" / "dst".
            let frame = frame.rename_columns([("source", "src"), ("destination", "dst")]).map_err(|error| {
                QueryError::from_cudf(
                    "nexus_query_engine.graph_algorithm.minimum_spanning_tree_result",
                    "failed to rename MST edge columns to the stable SQL schema",
                    error,
                )
            })?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata: String::new(),
            })
        },
    )
}
