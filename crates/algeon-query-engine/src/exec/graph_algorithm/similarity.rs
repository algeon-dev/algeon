//! Shared explicit-pair similarity execution.

use cugraph::cudf_interop::vertex_pairs_from_prepared_columns;
use query_planning::plan::{GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec, planned_resident_graph_key};

use super::{
    ExecutionContext, GraphAlgorithmExecutionInputs, GraphAlgorithmExecutionResult, NvtxRange, QueryError,
    QueryErrorCode, Result, cudf_query_error, empty_graph_execution_result, map_cugraph_error,
    normalize_graph_result_schema, project_edge_list, run_with_accounted_resident_cugraph_graph, similarity_params,
    table_graph_options,
};

pub(super) fn execute_similarity(
    context: &ExecutionContext,
    spec: &GraphAlgorithmSpec,
    mut inputs: GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult> {
    let GraphAlgorithmParams::Similarity(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.graph_algorithm.similarity",
            "similarity graph algorithm node carried non-similarity parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };
    if !spec.algorithm.is_similarity() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.graph_algorithm.similarity",
            "similarity parameter payload was attached to a non-similarity algorithm",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    }

    let pair_input = inputs.side_inputs.remove(&params.vertex_pairs.input).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.graph_algorithm.similarity",
            "similarity graph algorithm node is missing its vertex-pair side input",
        )
        .with_fact("algorithm", spec.algorithm.as_str())
        .with_fact("input_role", "vertex_pairs")
        .with_fact("side_input_ordinal", params.vertex_pairs.input)
    })?;
    let pair_frame = pair_input.into_frame();
    let pair_rows = pair_frame.num_rows();
    if pair_rows == 0 {
        return empty_graph_execution_result(
            context,
            spec,
            spec.algorithm.definition().empty_input_policy(),
            "vertex_pairs_empty",
        );
    }

    let input = inputs.edge_list;
    let resident_key = input.graph_residency().and_then(|descriptor| planned_resident_graph_key(descriptor, spec));
    let input = input.into_frame();
    let (edges, _) = project_edge_list(input, spec)?;
    let graph_options = table_graph_options(spec);
    let cugraph_params = similarity_params(spec);
    let algorithm = spec.algorithm;
    let run_similarity = |runtime: &cudf::Runtime,
                          entered: &cugraph::EnteredContext<'_>,
                          graph: &mut cugraph::Graph<cugraph::Forward>| {
        let _vertex_pairs_range = NvtxRange::static_label(c"algeon_datafusion.cugraph_similarity.vertex_pairs");
        let first = pair_frame.series(&params.vertex_pairs.first_column).map_err(|error| {
            cudf_query_error(
                "query_engine.graph_algorithm.similarity_pairs",
                "failed to resolve the first selected vertex-pair column",
                error,
            )
            .with_fact("algorithm", spec.algorithm.as_str())
            .with_fact("pair_column_role", "first")
            .with_fact("first_column", &params.vertex_pairs.first_column)
            .with_fact("second_column", &params.vertex_pairs.second_column)
        })?;
        let second = pair_frame.series(&params.vertex_pairs.second_column).map_err(|error| {
            cudf_query_error(
                "query_engine.graph_algorithm.similarity_pairs",
                "failed to resolve the second selected vertex-pair column",
                error,
            )
            .with_fact("algorithm", spec.algorithm.as_str())
            .with_fact("pair_column_role", "second")
            .with_fact("first_column", &params.vertex_pairs.first_column)
            .with_fact("second_column", &params.vertex_pairs.second_column)
        })?;
        if first.null_count() != 0 || second.null_count() != 0 {
            return Err(QueryError::new(
                QueryErrorCode::GraphInputInvalid,
                "query_engine.graph_algorithm.similarity_pairs",
                "similarity vertex-pair columns must be non-null",
            )
            .with_fact("algorithm", spec.algorithm.as_str())
            .with_fact("input_role", "vertex_pairs")
            .with_fact("first_null_count", first.null_count())
            .with_fact("second_null_count", second.null_count()));
        }
        let first_export = first.try_prepare_device_column_export().map_err(|error| {
            QueryError::new(
                QueryErrorCode::CudfDependencyFailure,
                "query_engine.graph_algorithm.prepare_vertex_pair_column_export",
                "failed to prepare the first cuGraph vertex-pair column export",
            )
            .with_source(error)
        })?;
        let second_export = second.try_prepare_device_column_export().map_err(|error| {
            QueryError::new(
                QueryErrorCode::CudfDependencyFailure,
                "query_engine.graph_algorithm.prepare_vertex_pair_column_export",
                "failed to prepare the second cuGraph vertex-pair column export",
            )
            .with_source(error)
        })?;
        let vertex_pairs =
            vertex_pairs_from_prepared_columns(entered, graph, &first_export, &second_export).map_err(|error| {
                similarity_cugraph_error(
                    spec,
                    "query_engine.graph_algorithm.vertex_pairs",
                    "failed to create cuGraph vertex pairs from the prepared pair relation",
                    error,
                )
            })?;
        // End the input conversion range before timing the algorithm itself.
        drop(_vertex_pairs_range);

        let frame = {
            let _metric_range = match algorithm {
                GraphAlgorithm::Jaccard => NvtxRange::static_label(c"algeon_datafusion.cugraph_similarity.jaccard"),
                GraphAlgorithm::Sorensen => NvtxRange::static_label(c"algeon_datafusion.cugraph_similarity.sorensen"),
                GraphAlgorithm::Overlap => NvtxRange::static_label(c"algeon_datafusion.cugraph_similarity.overlap"),
                GraphAlgorithm::Cosine => NvtxRange::static_label(c"algeon_datafusion.cugraph_similarity.cosine"),
                _ => unreachable!("similarity executor accepts only similarity algorithms"),
            };
            match algorithm {
                GraphAlgorithm::Jaccard => graph
                    .jaccard_coefficients(entered, vertex_pairs, &cugraph_params)
                    .map_err(|error| {
                        similarity_cugraph_error(
                            spec,
                            "query_engine.graph_algorithm.jaccard",
                            "cuGraph Jaccard similarity failed",
                            error,
                        )
                    })?
                    .to_dataframe(runtime)
                    .map_err(|error| {
                        similarity_cugraph_error(
                            spec,
                            "query_engine.graph_algorithm.jaccard_result",
                            "failed to convert cuGraph Jaccard similarity result to cudf DataFrame",
                            error,
                        )
                    })?,
                GraphAlgorithm::Sorensen => graph
                    .sorensen_coefficients(entered, vertex_pairs, &cugraph_params)
                    .map_err(|error| {
                        similarity_cugraph_error(
                            spec,
                            "query_engine.graph_algorithm.sorensen",
                            "cuGraph Sorensen similarity failed",
                            error,
                        )
                    })?
                    .to_dataframe(runtime)
                    .map_err(|error| {
                        similarity_cugraph_error(
                            spec,
                            "query_engine.graph_algorithm.sorensen_result",
                            "failed to convert cuGraph Sorensen similarity result to cudf DataFrame",
                            error,
                        )
                    })?,
                GraphAlgorithm::Overlap => graph
                    .overlap_coefficients(entered, vertex_pairs, &cugraph_params)
                    .map_err(|error| {
                        similarity_cugraph_error(
                            spec,
                            "query_engine.graph_algorithm.overlap",
                            "cuGraph overlap similarity failed",
                            error,
                        )
                    })?
                    .to_dataframe(runtime)
                    .map_err(|error| {
                        similarity_cugraph_error(
                            spec,
                            "query_engine.graph_algorithm.overlap_result",
                            "failed to convert cuGraph overlap similarity result to cudf DataFrame",
                            error,
                        )
                    })?,
                GraphAlgorithm::Cosine => graph
                    .cosine_similarity_coefficients(entered, vertex_pairs, &cugraph_params)
                    .map_err(|error| {
                        similarity_cugraph_error(
                            spec,
                            "query_engine.graph_algorithm.cosine",
                            "cuGraph cosine similarity failed",
                            error,
                        )
                    })?
                    .to_dataframe(runtime)
                    .map_err(|error| {
                        similarity_cugraph_error(
                            spec,
                            "query_engine.graph_algorithm.cosine_result",
                            "failed to convert cuGraph cosine similarity result to cudf DataFrame",
                            error,
                        )
                    })?,
                _ => unreachable!("similarity executor accepts only similarity algorithms"),
            }
        };
        let frame = {
            let _output_normalize_range =
                NvtxRange::static_label(c"algeon_datafusion.cugraph_similarity.output_normalize");
            normalize_graph_result_schema(frame, &spec.output_schema)?
        };
        Ok(GraphAlgorithmExecutionResult {
            frame,
            algorithm_metadata: format!(
                "vertex_pair_rows={pair_rows},use_weight={},do_expensive_check=true",
                cugraph_params.use_weight,
            ),
            cache_scope_peak_bytes: None,
        })
    };

    {
        run_with_accounted_resident_cugraph_graph::<cugraph::Forward, _>(
            context,
            edges,
            graph_options,
            resident_key,
            "algeon_datafusion_cugraph_similarity",
            run_similarity,
        )
    }
}

fn similarity_cugraph_error(
    spec: &GraphAlgorithmSpec,
    operation: &'static str,
    message: &'static str,
    error: cugraph::Error,
) -> QueryError {
    let GraphAlgorithmParams::Similarity(params) = &spec.params else {
        unreachable!("similarity executor validated its parameter payload before calling cuGraph")
    };
    map_cugraph_error(operation, message)(error)
        .with_fact("algorithm", spec.algorithm.as_str())
        .with_fact("input_role", "vertex_pairs")
        .with_fact("first_column", &params.vertex_pairs.first_column)
        .with_fact("second_column", &params.vertex_pairs.second_column)
}
