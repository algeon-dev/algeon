use super::*;

pub(super) fn execute_louvain(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult> {
    let input = inputs.edge_list;
    let crate::plan::GraphAlgorithmParams::Louvain(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.louvain",
            "Louvain graph algorithm node carried non-Louvain parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);
    let lv_params = louvain_params(params)?;

    run_with_cugraph_graph::<cugraph::Forward, _>(
        context,
        inputs.graph_memory_limit_bytes,
        spec,
        None,
        edges,
        options,
        "datafusion_nexus_cugraph_louvain",
        |runtime, entered, graph| {
            let result = graph
                .louvain(entered, &lv_params)
                .map_err(map_cugraph_error("nexus_query_engine.graph_algorithm.louvain", "cuGraph Louvain failed"))?;
            let algorithm_metadata = format!("modularity={}", result.modularity());
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.louvain_result",
                "failed to convert cuGraph Louvain result to cudf DataFrame",
            ))?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata,
                budget_observations: Vec::new(),
            })
        },
    )
}

pub(super) fn execute_leiden(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult> {
    let input = inputs.edge_list;
    let crate::plan::GraphAlgorithmParams::Leiden(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.leiden",
            "Leiden graph algorithm node carried non-Leiden parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);
    let leiden_p = leiden_params(params)?;
    let seed = params.seed;

    run_with_cugraph_graph::<cugraph::Forward, _>(
        context,
        inputs.graph_memory_limit_bytes,
        spec,
        None,
        edges,
        options,
        "datafusion_nexus_cugraph_leiden",
        |runtime, entered, graph| {
            let mut rng_state = cugraph::random::RngState::new(entered, seed).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.leiden_rng",
                "failed to create cuGraph RNG state for Leiden",
            ))?;
            let result = graph
                .leiden(entered, &mut rng_state, &leiden_p)
                .map_err(map_cugraph_error("nexus_query_engine.graph_algorithm.leiden", "cuGraph Leiden failed"))?;
            let algorithm_metadata = format!("modularity={},seed={}", result.modularity(), seed);
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.leiden_result",
                "failed to convert cuGraph Leiden result to cudf DataFrame",
            ))?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata,
                budget_observations: Vec::new(),
            })
        },
    )
}

pub(super) fn execute_ecg(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult> {
    let input = inputs.edge_list;
    let crate::plan::GraphAlgorithmParams::Ecg(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.ecg",
            "ECG graph algorithm node carried non-ECG parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);
    let ecg_p = ecg_params(params)?;
    let seed = params.seed;

    run_with_cugraph_graph::<cugraph::Forward, _>(
        context,
        inputs.graph_memory_limit_bytes,
        spec,
        None,
        edges,
        options,
        "datafusion_nexus_cugraph_ecg",
        |runtime, entered, graph| {
            let mut rng_state = cugraph::random::RngState::new(entered, seed).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.ecg_rng",
                "failed to create cuGraph RNG state for ECG",
            ))?;
            let result = graph
                .ecg(entered, &mut rng_state, &ecg_p)
                .map_err(map_cugraph_error("nexus_query_engine.graph_algorithm.ecg", "cuGraph ECG failed"))?;
            let algorithm_metadata = format!("modularity={},seed={}", result.modularity(), seed);
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.ecg_result",
                "failed to convert cuGraph ECG result to cudf DataFrame",
            ))?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata,
                budget_observations: Vec::new(),
            })
        },
    )
}

pub(super) fn execute_spectral_modularity_maximization(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult> {
    let input = inputs.edge_list;
    let crate::plan::GraphAlgorithmParams::SpectralModularityMaximization(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.spectral_modularity_maximization",
            "SpectralModularityMaximization graph algorithm node carried wrong parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);
    let spectral_p = spectral_clustering_params(params)?;
    let (n_clusters, seed) = (params.n_clusters, params.seed);

    run_with_cugraph_graph::<cugraph::Forward, _>(
        context,
        inputs.graph_memory_limit_bytes,
        spec,
        None,
        edges,
        options,
        "datafusion_nexus_cugraph_spectral_modularity_maximization",
        |runtime, entered, graph| {
            let mut rng_state = cugraph::random::RngState::new(entered, seed).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.spectral_modularity_maximization_rng",
                "failed to create cuGraph RNG state for SpectralModularityMaximization",
            ))?;
            let result = graph.spectral_modularity_maximization(entered, &mut rng_state, &spectral_p).map_err(
                map_cugraph_error(
                    "nexus_query_engine.graph_algorithm.spectral_modularity_maximization",
                    "cuGraph spectral_modularity_maximization failed",
                ),
            )?;
            let algorithm_metadata = format!("n_clusters={},seed={}", n_clusters, seed);
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.spectral_modularity_maximization_result",
                "failed to convert cuGraph SpectralModularityMaximization result to cudf DataFrame",
            ))?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata,
                budget_observations: Vec::new(),
            })
        },
    )
}
