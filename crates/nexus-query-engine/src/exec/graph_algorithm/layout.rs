use super::*;

pub(super) fn execute_force_atlas2(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult> {
    let input = inputs.edge_list;
    let crate::plan::GraphAlgorithmParams::ForceAtlas2(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.force_atlas2",
            "ForceAtlas2 graph algorithm node carried non-ForceAtlas2 parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);
    let fa2_params = force_atlas2_params(params)?;
    let seed = params.seed;

    run_with_cugraph_graph::<cugraph::Forward, _>(
        context,
        inputs.graph_memory_limit_bytes,
        edges,
        options,
        "datafusion_nexus_cugraph_force_atlas2",
        |runtime, entered, graph| {
            let mut rng_state = cugraph::random::RngState::new(entered, seed).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.force_atlas2_rng",
                "failed to create cuGraph RNG state for ForceAtlas2",
            ))?;
            let result = graph.force_atlas2(entered, &mut rng_state, &fa2_params).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.force_atlas2",
                "cuGraph ForceAtlas2 failed",
            ))?;
            let algorithm_metadata = format!("seed={seed}");
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.force_atlas2_result",
                "failed to convert cuGraph ForceAtlas2 result to cudf DataFrame",
            ))?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata,
                budget_observations: Vec::new(),
                cache_scope_peak_bytes: None,
            })
        },
    )
}
