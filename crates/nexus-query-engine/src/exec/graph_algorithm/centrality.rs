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
            "nexus_query_engine.graph_algorithm.pagerank",
            "PageRank graph algorithm node carried non-PageRank parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let resident_key =
        input.graph_residency().and_then(|descriptor| crate::plan::planned_resident_graph_key(descriptor, spec));
    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);
    let pr_params = pagerank_params(params)?;

    let run_pagerank = |runtime: &cudf::Runtime,
                        entered: &cugraph::EnteredContext<'_>,
                        graph: &mut cugraph::Graph<cugraph::Transposed>| {
        let result = graph
            .pagerank(entered, &pr_params)
            .map_err(map_cugraph_error("nexus_query_engine.graph_algorithm.pagerank", "cuGraph PageRank failed"))?;
        let converged = result.converged().map_or_else(|_| "unknown".to_string(), |value| value.to_string());
        let algorithm_metadata = format!("converged={},iterations={}", converged, result.num_iterations());
        let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.pagerank_result",
            "failed to convert cuGraph PageRank result to cudf DataFrame",
        ))?;
        Ok(GraphAlgorithmExecutionResult {
            frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
            algorithm_metadata,
            budget_observations: Vec::new(),
            cache_scope_peak_bytes: None,
        })
    };
    {
        run_with_accounted_resident_cugraph_graph::<cugraph::Transposed, _>(
            context,
            inputs.graph_memory_limit_bytes,
            edges,
            options,
            resident_key,
            "datafusion_nexus_cugraph_pagerank",
            run_pagerank,
        )
    }
}

pub(super) fn execute_personalized_pagerank(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    mut inputs: GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult> {
    let input = inputs.edge_list;
    let crate::plan::GraphAlgorithmParams::PersonalizedPageRank(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.personalized_pagerank",
            "PersonalizedPageRank graph algorithm node carried non-PersonalizedPageRank parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };
    let personalization_frame = inputs.side_inputs.remove(&params.personalization.input).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.personalized_pagerank",
            "PersonalizedPageRank graph algorithm node is missing its personalization side input",
        )
        .with_fact("side_input_node", params.personalization.input)
    })?;

    let input = input.into_frame();
    let (edges, vertex_type) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);
    let ppr_params = personalized_pagerank_params(&params.pagerank)?;
    let personalization_frame = personalization_frame.into_frame();
    let personalization = &params.personalization;

    run_with_cugraph_graph::<cugraph::Transposed, _>(
        context,
        inputs.graph_memory_limit_bytes,
        edges,
        options,
        "datafusion_nexus_cugraph_personalized_pagerank",
        |runtime, entered, graph| match vertex_type {
            cudf::TypeId::Int32 => match graph.weight_dtype() {
                cugraph::types::DataType::Float32 => personalized_pagerank_typed::<i32, f32>(
                    &personalization_frame,
                    personalization,
                    runtime,
                    entered,
                    graph,
                    &ppr_params,
                    spec,
                ),
                cugraph::types::DataType::Float64 => personalized_pagerank_typed::<i32, f64>(
                    &personalization_frame,
                    personalization,
                    runtime,
                    entered,
                    graph,
                    &ppr_params,
                    spec,
                ),
                dtype => Err(personalized_pagerank_unsupported_weight_dtype(dtype)),
            },
            cudf::TypeId::Int64 => match graph.weight_dtype() {
                cugraph::types::DataType::Float32 => personalized_pagerank_typed::<i64, f32>(
                    &personalization_frame,
                    personalization,
                    runtime,
                    entered,
                    graph,
                    &ppr_params,
                    spec,
                ),
                cugraph::types::DataType::Float64 => personalized_pagerank_typed::<i64, f64>(
                    &personalization_frame,
                    personalization,
                    runtime,
                    entered,
                    graph,
                    &ppr_params,
                    spec,
                ),
                dtype => Err(personalized_pagerank_unsupported_weight_dtype(dtype)),
            },
            dtype => Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                "nexus_query_engine.graph_algorithm.personalized_pagerank",
                "personalized PageRank requires Int32 or Int64 graph vertices",
            )
            .with_fact("vertex_dtype", format!("{dtype:?}"))),
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
            "nexus_query_engine.graph_algorithm.eigenvector_centrality",
            "EigenvectorCentrality graph algorithm node carried non-EigenvectorCentrality parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);
    let evc_params = eigenvector_centrality_params(params)?;

    run_with_cugraph_graph::<cugraph::Transposed, _>(
        context,
        inputs.graph_memory_limit_bytes,
        edges,
        options,
        "datafusion_nexus_cugraph_eigenvector_centrality",
        |runtime, entered, graph| {
            let result = graph.eigenvector_centrality(entered, &evc_params).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.eigenvector_centrality",
                "cuGraph eigenvector centrality failed",
            ))?;
            let converged = result.converged().map_or_else(|_| "unknown".to_string(), |value| value.to_string());
            let algorithm_metadata = format!("converged={},iterations={}", converged, result.num_iterations());
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.eigenvector_centrality_result",
                "failed to convert cuGraph eigenvector centrality result to cudf DataFrame",
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

pub(super) fn execute_hits(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult> {
    let input = inputs.edge_list;
    let crate::plan::GraphAlgorithmParams::Hits(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.hits",
            "HITS graph algorithm node carried non-HITS parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);
    let h_params = hits_params(params)?;

    run_with_cugraph_graph::<cugraph::Transposed, _>(
        context,
        inputs.graph_memory_limit_bytes,
        edges,
        options,
        "datafusion_nexus_cugraph_hits",
        |runtime, entered, graph| {
            let result = graph
                .hits(entered, &h_params)
                .map_err(map_cugraph_error("nexus_query_engine.graph_algorithm.hits", "cuGraph HITS failed"))?;
            let algorithm_metadata = format!(
                "iterations={},hub_score_differences={}",
                result.number_of_iterations(),
                result.hub_score_differences(),
            );
            // Upstream HITS emits `vertex` / `hub` / `authority`; the stable
            // SQL schema renames the score columns to `hub_score` /
            // `authority_score` so the meaning is unambiguous at query time.
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.hits_result",
                "failed to convert cuGraph HITS result to cudf DataFrame",
            ))?;
            let frame =
                frame.rename_columns([("hub", "hub_score"), ("authority", "authority_score")]).map_err(|error| {
                    QueryError::from_cudf(
                        "nexus_query_engine.graph_algorithm.hits_result",
                        "failed to rename HITS score columns to the stable SQL schema",
                        error,
                    )
                })?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata,
                budget_observations: Vec::new(),
                cache_scope_peak_bytes: None,
            })
        },
    )
}

pub(super) fn execute_betweenness_centrality(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult> {
    let input = inputs.edge_list;
    let crate::plan::GraphAlgorithmParams::BetweennessCentrality(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.betweenness_centrality",
            "BetweennessCentrality graph algorithm node carried non-BetweennessCentrality parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, vertex_type) = project_edge_list(&input, spec)?;
    let seed_count =
        params.seeds.as_ref().map(|seeds| betweenness_seed_count(seeds, &inputs.side_inputs)).transpose()?;
    let options = table_graph_options(spec);
    let bc_params = betweenness_centrality_params(params);

    run_with_cugraph_graph::<cugraph::Forward, _>(
        context,
        inputs.graph_memory_limit_bytes,
        edges,
        options,
        "datafusion_nexus_cugraph_betweenness_centrality",
        |runtime, entered, graph| {
            let result = match vertex_type {
                cudf::TypeId::Int32 => match &params.seeds {
                    Some(seeds) => {
                        let seeds = betweenness_i32_seeds(seeds, &inputs.side_inputs)?;
                        graph.betweenness_centrality(entered, seeds.as_slice(), &bc_params)
                    },
                    None => graph.betweenness_centrality_all(entered, &bc_params),
                },
                cudf::TypeId::Int64 => match &params.seeds {
                    Some(seeds) => {
                        let seeds = betweenness_i64_seeds(seeds, &inputs.side_inputs)?;
                        graph.betweenness_centrality(entered, seeds.as_slice(), &bc_params)
                    },
                    None => graph.betweenness_centrality_all(entered, &bc_params),
                },
                dtype => {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedType,
                        "nexus_query_engine.graph_algorithm.betweenness_centrality",
                        "betweenness centrality requires Int32 or Int64 graph vertices",
                    )
                    .with_fact("vertex_dtype", format!("{dtype:?}")));
                },
            }
            .map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.betweenness_centrality",
                "cuGraph betweenness centrality failed",
            ))?;
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.betweenness_centrality_result",
                "failed to convert cuGraph betweenness centrality result to cudf DataFrame",
            ))?;
            let algorithm_metadata = betweenness_algorithm_metadata(seed_count, params.normalized);
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata,
                budget_observations: Vec::new(),
                cache_scope_peak_bytes: None,
            })
        },
    )
}

pub(super) fn execute_edge_betweenness_centrality(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult> {
    let input = inputs.edge_list;
    let crate::plan::GraphAlgorithmParams::EdgeBetweennessCentrality(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.edge_betweenness_centrality",
            "EdgeBetweennessCentrality graph algorithm node carried non-EdgeBetweennessCentrality parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, vertex_type) = project_edge_list(&input, spec)?;
    let seed_count =
        params.seeds.as_ref().map(|seeds| betweenness_seed_count(seeds, &inputs.side_inputs)).transpose()?;
    let options = table_graph_options(spec);
    let ebc_params = edge_betweenness_centrality_params(params);

    run_with_cugraph_graph::<cugraph::Forward, _>(
        context,
        inputs.graph_memory_limit_bytes,
        edges,
        options,
        "datafusion_nexus_cugraph_edge_betweenness_centrality",
        |runtime, entered, graph| {
            let result = match vertex_type {
                cudf::TypeId::Int32 => match &params.seeds {
                    Some(seeds) => {
                        let seeds = betweenness_i32_seeds(seeds, &inputs.side_inputs)?;
                        graph.edge_betweenness_centrality(entered, seeds.as_slice(), &ebc_params)
                    },
                    None => graph.edge_betweenness_centrality_all(entered, &ebc_params),
                },
                cudf::TypeId::Int64 => match &params.seeds {
                    Some(seeds) => {
                        let seeds = betweenness_i64_seeds(seeds, &inputs.side_inputs)?;
                        graph.edge_betweenness_centrality(entered, seeds.as_slice(), &ebc_params)
                    },
                    None => graph.edge_betweenness_centrality_all(entered, &ebc_params),
                },
                dtype => {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedType,
                        "nexus_query_engine.graph_algorithm.edge_betweenness_centrality",
                        "edge betweenness centrality requires Int32 or Int64 graph vertices",
                    )
                    .with_fact("vertex_dtype", format!("{dtype:?}")));
                },
            }
            .map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.edge_betweenness_centrality",
                "cuGraph edge betweenness centrality failed",
            ))?;
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.edge_betweenness_centrality_result",
                "failed to convert cuGraph edge betweenness centrality result to cudf DataFrame",
            ))?;
            let algorithm_metadata = betweenness_algorithm_metadata(seed_count, params.normalized);
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata,
                budget_observations: Vec::new(),
                cache_scope_peak_bytes: None,
            })
        },
    )
}

fn betweenness_i32_seeds(
    seeds: &crate::plan::GraphBetweennessSeeds,
    side_inputs: &std::collections::BTreeMap<NodeId, GpuDataFrame>,
) -> Result<Vec<i32>> {
    match seeds {
        crate::plan::GraphBetweennessSeeds::I64(values) => values
            .iter()
            .map(|value| {
                i32::try_from(*value).map_err(|_| {
                    QueryError::new(
                        QueryErrorCode::InvalidPlan,
                        "nexus_query_engine.graph_algorithm.betweenness_seeds",
                        "betweenness seed vertex does not fit Int32 graph vertex dtype",
                    )
                    .with_fact("seed_vertex", *value)
                })
            })
            .collect(),
        crate::plan::GraphBetweennessSeeds::Input { input, .. } => {
            betweenness_input_series(input, side_inputs, cudf::TypeId::Int32)?.to_vec_strict::<i32>().map_err(|error| {
                QueryError::from_cudf(
                    "nexus_query_engine.graph_algorithm.betweenness_seeds",
                    "failed to read encoded Int32 betweenness seeds",
                    error,
                )
            })
        },
    }
}

fn betweenness_i64_seeds(
    seeds: &crate::plan::GraphBetweennessSeeds,
    side_inputs: &std::collections::BTreeMap<NodeId, GpuDataFrame>,
) -> Result<Vec<i64>> {
    match seeds {
        crate::plan::GraphBetweennessSeeds::I64(values) => Ok(values.clone()),
        crate::plan::GraphBetweennessSeeds::Input { input, .. } => {
            betweenness_input_series(input, side_inputs, cudf::TypeId::Int64)?.to_vec_strict::<i64>().map_err(|error| {
                QueryError::from_cudf(
                    "nexus_query_engine.graph_algorithm.betweenness_seeds",
                    "failed to read encoded Int64 betweenness seeds",
                    error,
                )
            })
        },
    }
}

fn betweenness_seed_count(
    seeds: &crate::plan::GraphBetweennessSeeds,
    side_inputs: &std::collections::BTreeMap<NodeId, GpuDataFrame>,
) -> Result<u64> {
    match seeds {
        crate::plan::GraphBetweennessSeeds::I64(values) => Ok(values.len() as u64),
        crate::plan::GraphBetweennessSeeds::Input { input, row_count_upper_bound } => {
            let rows = side_inputs.get(&input.input).map(|frame| frame.num_rows() as u64).ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.graph_algorithm.betweenness_seeds",
                    "betweenness seed side input was not executed",
                )
                .with_fact("side_input_node", input.input)
                .with_fact("column", &input.column)
            })?;
            if rows > *row_count_upper_bound {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.graph_algorithm.betweenness_seeds",
                    "betweenness seed side input exceeded its proven row-count bound",
                )
                .with_fact("side_input_rows", rows)
                .with_fact("row_count_upper_bound", *row_count_upper_bound));
            }
            Ok(rows)
        },
    }
}

fn betweenness_input_series(
    input: &crate::plan::GraphPredicateInput,
    side_inputs: &std::collections::BTreeMap<NodeId, GpuDataFrame>,
    vertex_type: cudf::TypeId,
) -> Result<cudf::Series> {
    let frame = side_inputs.get(&input.input).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.betweenness_seeds",
            "betweenness seed side input was not executed",
        )
        .with_fact("side_input_node", input.input)
        .with_fact("column", &input.column)
    })?;
    let series = frame.frame().series(&input.column).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.graph_algorithm.betweenness_seeds",
            "failed to resolve the betweenness seed side-input column",
            error,
        )
    })?;
    let side_input_dtype = series.dtype().map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.graph_algorithm.betweenness_seeds",
            "failed to inspect the betweenness seed side-input dtype",
            error,
        )
    })?;
    if series.is_empty() || series.null_count() != 0 || side_input_dtype.id() != vertex_type {
        return Err(QueryError::new(
            QueryErrorCode::GraphInputInvalid,
            "nexus_query_engine.graph_algorithm.betweenness_seeds",
            "betweenness seed side input must be non-empty, non-null, and match the graph vertex dtype",
        )
        .with_fact("side_input_rows", series.len())
        .with_fact("side_input_null_rows", series.null_count())
        .with_fact("side_input_dtype", format!("{:?}", side_input_dtype.id()))
        .with_fact("vertex_dtype", format!("{vertex_type:?}")));
    }
    Ok(series)
}

fn betweenness_algorithm_metadata(seed_count: Option<u64>, normalized: bool) -> String {
    match seed_count {
        Some(seed_count) => format!("approximate=true,seed_count={seed_count},normalized={normalized}"),
        None => format!("approximate=false,seed_count=all,normalized={normalized}"),
    }
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
            "nexus_query_engine.graph_algorithm.katz_centrality",
            "KatzCentrality graph algorithm node carried non-KatzCentrality parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);
    let kc_params = katz_centrality_params(params)?;

    run_with_cugraph_graph::<cugraph::Transposed, _>(
        context,
        inputs.graph_memory_limit_bytes,
        edges,
        options,
        "datafusion_nexus_cugraph_katz_centrality",
        |runtime, entered, graph| {
            let result = graph.katz_centrality(entered, &kc_params).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.katz_centrality",
                "cuGraph Katz centrality failed",
            ))?;
            let converged = result.converged().map_or_else(|_| "unknown".to_string(), |value| value.to_string());
            let algorithm_metadata = format!("converged={},iterations={}", converged, result.num_iterations());
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.katz_centrality_result",
                "failed to convert cuGraph Katz centrality result to cudf DataFrame",
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

fn personalized_pagerank_typed<V, W>(
    personalization_frame: &DataFrame,
    personalization: &crate::plan::GraphPageRankPersonalizationInput,
    runtime: &cudf::Runtime,
    entered: &cugraph::EnteredContext<'_>,
    graph: &mut cugraph::Graph<cugraph::Transposed>,
    params: &cugraph::centrality::PersonalizedPageRankParams,
    spec: &crate::plan::GraphAlgorithmSpec,
) -> Result<GraphAlgorithmExecutionResult>
where
    V: cudf::CudfType + cugraph::types::CugraphElement,
    W: cudf::CudfType + cugraph::types::CugraphElement,
{
    let vertices = graph_series(personalization_frame, &personalization.vertex, "personalization_vertex")?;
    let values = personalization_values_series::<W>(personalization_frame, &personalization.value)?;
    validate_personalization_series(&vertices, "personalization_vertex")?;
    validate_personalization_series(&values, "personalization_value")?;
    let vertex_slice = graph_device_slice::<V>(&vertices, "personalization_vertex")?;
    let value_slice = graph_device_slice::<W>(&values, "personalization_value")?;
    let vertex_view = graph_device_view::<V>(&vertex_slice, "personalization_vertex")?;
    let value_view = graph_device_view::<W>(&value_slice, "personalization_value")?;
    let personalization = cugraph::centrality::PersonalizationVector { vertices: &vertex_view, values: &value_view };
    let result = graph.personalized_pagerank(entered, params, &personalization).map_err(map_cugraph_error(
        "nexus_query_engine.graph_algorithm.personalized_pagerank",
        "cuGraph personalized PageRank failed",
    ))?;
    let converged = result.converged().map_or_else(|_| "unknown".to_string(), |value| value.to_string());
    let algorithm_metadata = format!("converged={},iterations={}", converged, result.num_iterations());
    let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
        "nexus_query_engine.graph_algorithm.personalized_pagerank_result",
        "failed to convert cuGraph personalized PageRank result to cudf DataFrame",
    ))?;
    Ok(GraphAlgorithmExecutionResult {
        frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
        algorithm_metadata,
        budget_observations: Vec::new(),
        cache_scope_peak_bytes: None,
    })
}

fn personalization_values_series<W>(frame: &DataFrame, column: &str) -> Result<cudf::Series>
where W: cudf::CudfType + cugraph::types::CugraphElement {
    let values = graph_series(frame, column, "personalization_value")?;
    let target = match W::data_type() {
        cugraph::types::DataType::Float32 => cudf::TypeId::Float32,
        cugraph::types::DataType::Float64 => cudf::TypeId::Float64,
        dtype => return Err(personalized_pagerank_unsupported_weight_dtype(dtype)),
    };
    let dtype = values.dtype().map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.graph_algorithm.personalized_pagerank",
            "failed to inspect personalized PageRank value dtype",
            error,
        )
        .with_fact("column", column)
    })?;
    if dtype.id() == target {
        return Ok(values);
    }
    if !matches!(dtype.id(), cudf::TypeId::Float32 | cudf::TypeId::Float64) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "nexus_query_engine.graph_algorithm.personalized_pagerank",
            "personalized PageRank value column type is unsupported",
        )
        .with_fact("column", column)
        .with_fact("dtype", format!("{dtype:?}")));
    }
    values.astype(cudf::DataType::new(target)).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.graph_algorithm.personalized_pagerank",
            "failed to cast personalized PageRank values to the graph weight dtype",
            error,
        )
        .with_fact("column", column)
        .with_fact("source_dtype", format!("{dtype:?}"))
        .with_fact("target_dtype", format!("{target:?}"))
    })
}

fn validate_personalization_series(series: &cudf::Series, role: &'static str) -> Result<()> {
    if series.null_count() == 0 {
        return Ok(());
    }
    Err(QueryError::new(
        QueryErrorCode::InvalidPlan,
        "nexus_query_engine.graph_algorithm.personalized_pagerank",
        "personalized PageRank side-input columns must be non-null",
    )
    .with_fact("role", role)
    .with_fact("null_count", series.null_count()))
}

fn personalized_pagerank_unsupported_weight_dtype(dtype: cugraph::types::DataType) -> QueryError {
    QueryError::new(
        QueryErrorCode::UnsupportedType,
        "nexus_query_engine.graph_algorithm.personalized_pagerank",
        "personalized PageRank requires Float32 or Float64 graph weights",
    )
    .with_fact("weight_dtype", dtype.to_string())
}
