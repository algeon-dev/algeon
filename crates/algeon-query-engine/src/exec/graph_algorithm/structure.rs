use cugraph::{
    community::TriangleCountParams, labeling::ConnectedComponentsParams, structure::CoreNumberParams,
    tree::MinimumSpanningTreeParams,
};
use query_planning::plan::{GraphAlgorithmParams, GraphAlgorithmSpec};

use super::{
    DataFrame, ExecutionContext, GraphAlgorithmExecutionInputs, GraphAlgorithmExecutionResult, QueryError,
    QueryErrorCode, Result, cudf_query_error, degree_type_to_cugraph, k_core_params, map_cugraph_error,
    normalize_graph_result_schema, project_edge_list, run_with_cugraph_graph, table_graph_options,
};

/// Renames cuGraph component result columns to the SQL-facing names declared
/// by the stable result schema.
pub(super) fn rename_result_columns(
    frame: DataFrame,
    renames: &[(&str, &str)],
    operation: &'static str,
) -> Result<DataFrame> {
    frame.rename_columns(renames.iter().copied()).map_err(|error| {
        cudf_query_error(operation, "failed to rename cuGraph result columns to the stable SQL schema", error)
    })
}

pub(super) fn execute_degrees_all(
    context: &ExecutionContext,
    spec: &GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult> {
    let input = inputs.edge_list;
    let GraphAlgorithmParams::DegreesAll(_) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.graph_algorithm.degrees_all",
            "DegreesAll graph algorithm node carried non-DegreesAll parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(input, spec)?;
    let options = table_graph_options(spec);

    run_with_cugraph_graph::<cugraph::Transposed, _>(
        context,
        edges,
        options,
        "algeon_datafusion_cugraph_degrees_all",
        |runtime, entered, graph| {
            let result = graph
                .degrees_all(entered)
                .map_err(map_cugraph_error("query_engine.graph_algorithm.degrees_all", "cuGraph degrees_all failed"))?;
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "query_engine.graph_algorithm.degrees_all_result",
                "failed to convert cuGraph degrees result to cudf DataFrame",
            ))?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata: String::new(),
                cache_scope_peak_bytes: None,
            })
        },
    )
}

pub(super) fn execute_in_degrees_all(
    context: &ExecutionContext,
    spec: &GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult> {
    let input = inputs.edge_list;
    let GraphAlgorithmParams::InDegreesAll(_) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.graph_algorithm.in_degrees_all",
            "InDegreesAll graph algorithm node carried non-InDegreesAll parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(input, spec)?;
    let options = table_graph_options(spec);

    run_with_cugraph_graph::<cugraph::Transposed, _>(
        context,
        edges,
        options,
        "algeon_datafusion_cugraph_in_degrees_all",
        |runtime, entered, graph| {
            let result = graph.in_degrees_all(entered).map_err(map_cugraph_error(
                "query_engine.graph_algorithm.in_degrees_all",
                "cuGraph in_degrees_all failed",
            ))?;
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "query_engine.graph_algorithm.in_degrees_all_result",
                "failed to convert cuGraph in-degrees result to cudf DataFrame",
            ))?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata: String::new(),
                cache_scope_peak_bytes: None,
            })
        },
    )
}

pub(super) fn execute_out_degrees_all(
    context: &ExecutionContext,
    spec: &GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult> {
    let input = inputs.edge_list;
    let GraphAlgorithmParams::OutDegreesAll(_) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.graph_algorithm.out_degrees_all",
            "OutDegreesAll graph algorithm node carried non-OutDegreesAll parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(input, spec)?;
    let options = table_graph_options(spec);

    // out_degrees_all is layout-agnostic in the local `cugraph` crate; build Forward for
    // consistency with other Any-layout algorithms in this adapter.
    run_with_cugraph_graph::<cugraph::Forward, _>(
        context,
        edges,
        options,
        "algeon_datafusion_cugraph_out_degrees_all",
        |runtime, entered, graph| {
            let result = graph.out_degrees_all(entered).map_err(map_cugraph_error(
                "query_engine.graph_algorithm.out_degrees_all",
                "cuGraph out_degrees_all failed",
            ))?;
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "query_engine.graph_algorithm.out_degrees_all_result",
                "failed to convert cuGraph out-degrees result to cudf DataFrame",
            ))?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata: String::new(),
                cache_scope_peak_bytes: None,
            })
        },
    )
}

pub(super) fn execute_weakly_connected_components(
    context: &ExecutionContext,
    spec: &GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult> {
    let input = inputs.edge_list;
    let GraphAlgorithmParams::WeaklyConnectedComponents(_) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.graph_algorithm.weakly_connected_components",
            "WeaklyConnectedComponents graph algorithm node carried wrong parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(input, spec)?;
    let options = table_graph_options(spec);

    run_with_cugraph_graph::<cugraph::Forward, _>(
        context,
        edges,
        options,
        "algeon_datafusion_cugraph_weakly_connected_components",
        |runtime, entered, graph| {
            let wcc_params = ConnectedComponentsParams::default();
            let result = graph.weakly_connected_components(entered, &wcc_params).map_err(map_cugraph_error(
                "query_engine.graph_algorithm.weakly_connected_components",
                "cuGraph weakly_connected_components failed",
            ))?;
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "query_engine.graph_algorithm.weakly_connected_components_result",
                "failed to convert cuGraph WCC result to cudf DataFrame",
            ))?;
            let frame = rename_result_columns(
                frame,
                &[("label", "component")],
                "query_engine.graph_algorithm.weakly_connected_components_result",
            )?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata: String::new(),
                cache_scope_peak_bytes: None,
            })
        },
    )
}

pub(super) fn execute_strongly_connected_components(
    context: &ExecutionContext,
    spec: &GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult> {
    let input = inputs.edge_list;
    let GraphAlgorithmParams::StronglyConnectedComponents(_) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.graph_algorithm.strongly_connected_components",
            "StronglyConnectedComponents graph algorithm node carried wrong parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(input, spec)?;
    let options = table_graph_options(spec);

    run_with_cugraph_graph::<cugraph::Forward, _>(
        context,
        edges,
        options,
        "algeon_datafusion_cugraph_strongly_connected_components",
        |runtime, entered, graph| {
            let scc_params = ConnectedComponentsParams::default();
            let result = graph.strongly_connected_components(entered, &scc_params).map_err(map_cugraph_error(
                "query_engine.graph_algorithm.strongly_connected_components",
                "cuGraph strongly_connected_components failed",
            ))?;
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "query_engine.graph_algorithm.strongly_connected_components_result",
                "failed to convert cuGraph SCC result to cudf DataFrame",
            ))?;
            let frame = rename_result_columns(
                frame,
                &[("label", "component")],
                "query_engine.graph_algorithm.strongly_connected_components_result",
            )?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata: String::new(),
                cache_scope_peak_bytes: None,
            })
        },
    )
}

pub(super) fn execute_triangle_count_all(
    context: &ExecutionContext,
    spec: &GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult> {
    let input = inputs.edge_list;
    let GraphAlgorithmParams::TriangleCountAll(_) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.graph_algorithm.triangle_count_all",
            "TriangleCountAll graph algorithm node carried wrong parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(input, spec)?;
    let options = table_graph_options(spec);

    run_with_cugraph_graph::<cugraph::Forward, _>(
        context,
        edges,
        options,
        "algeon_datafusion_cugraph_triangle_count_all",
        |runtime, entered, graph| {
            let tc_params = TriangleCountParams::default();
            let result = graph.triangle_count_all(entered, &tc_params).map_err(map_cugraph_error(
                "query_engine.graph_algorithm.triangle_count_all",
                "cuGraph triangle_count_all failed",
            ))?;
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "query_engine.graph_algorithm.triangle_count_all_result",
                "failed to convert cuGraph triangle-count result to cudf DataFrame",
            ))?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata: String::new(),
                cache_scope_peak_bytes: None,
            })
        },
    )
}

pub(super) fn execute_core_number(
    context: &ExecutionContext,
    spec: &GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult> {
    let input = inputs.edge_list;
    let GraphAlgorithmParams::CoreNumber(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.graph_algorithm.core_number",
            "CoreNumber graph algorithm node carried non-CoreNumber parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(input, spec)?;
    let options = table_graph_options(spec);
    let cn_params = CoreNumberParams {
        degree_type: degree_type_to_cugraph(params.degree_type, spec.graph_options.directed),
        do_expensive_check: false,
    };

    run_with_cugraph_graph::<cugraph::Forward, _>(
        context,
        edges,
        options,
        "algeon_datafusion_cugraph_core_number",
        |runtime, entered, graph| {
            let result = graph
                .core_number(entered, &cn_params)
                .map_err(map_cugraph_error("query_engine.graph_algorithm.core_number", "cuGraph core_number failed"))?;
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "query_engine.graph_algorithm.core_number_result",
                "failed to convert cuGraph core-number result to cudf DataFrame",
            ))?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata: format!("degree_type={}", params.degree_type.as_str()),
                cache_scope_peak_bytes: None,
            })
        },
    )
}

pub(super) fn execute_k_core(
    context: &ExecutionContext,
    spec: &GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult> {
    let input = inputs.edge_list;
    let GraphAlgorithmParams::KCore(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.graph_algorithm.k_core",
            "KCore graph algorithm node carried non-KCore parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(input, spec)?;
    let options = table_graph_options(spec);
    let kc_params = k_core_params(params, spec.graph_options.directed)?;

    run_with_cugraph_graph::<cugraph::Forward, _>(
        context,
        edges,
        options,
        "algeon_datafusion_cugraph_k_core",
        |runtime, entered, graph| {
            let result = graph
                .k_core(entered, &kc_params)
                .map_err(map_cugraph_error("query_engine.graph_algorithm.k_core", "cuGraph k_core failed"))?;
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "query_engine.graph_algorithm.k_core_result",
                "failed to convert cuGraph k-core result to cudf DataFrame",
            ))?;
            let frame = rename_result_columns(
                frame,
                &[("source", "src"), ("destination", "dst")],
                "query_engine.graph_algorithm.k_core_result",
            )?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata: format!("k={},degree_type={}", params.k, params.degree_type.as_str()),
                cache_scope_peak_bytes: None,
            })
        },
    )
}

pub(super) fn execute_minimum_spanning_tree(
    context: &ExecutionContext,
    spec: &GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult> {
    let input = inputs.edge_list;
    let GraphAlgorithmParams::MinimumSpanningTree(_) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.graph_algorithm.minimum_spanning_tree",
            "MinimumSpanningTree graph algorithm node carried wrong parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(input, spec)?;
    let options = table_graph_options(spec);

    run_with_cugraph_graph::<cugraph::Forward, _>(
        context,
        edges,
        options,
        "algeon_datafusion_cugraph_minimum_spanning_tree",
        |runtime, entered, graph| {
            let mst_params = MinimumSpanningTreeParams::default();
            let result = graph.minimum_spanning_tree(entered, &mst_params).map_err(map_cugraph_error(
                "query_engine.graph_algorithm.minimum_spanning_tree",
                "cuGraph minimum_spanning_tree failed",
            ))?;
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "query_engine.graph_algorithm.minimum_spanning_tree_result",
                "failed to convert cuGraph MST result to cudf DataFrame",
            ))?;
            let frame = rename_result_columns(
                frame,
                &[("source", "src"), ("destination", "dst")],
                "query_engine.graph_algorithm.minimum_spanning_tree_result",
            )?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata: String::new(),
                cache_scope_peak_bytes: None,
            })
        },
    )
}
