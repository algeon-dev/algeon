//! Engine-owned empty graph input behavior.

use super::*;

pub(super) fn resolve_empty_graph_input(
    context: &ExecutionContext,
    node: &crate::plan::GraphAlgorithmNode,
    sanitized_edges: &DataFrame,
    side_inputs: &std::collections::BTreeMap<NodeId, GpuDataFrame>,
    input_rows: u64,
    sanitized_input_rows: u64,
) -> Result<Option<GraphAlgorithmExecutionResult>> {
    let policy = node.spec.algorithm.definition().empty_input_policy();
    validate_required_side_inputs(&node.spec, side_inputs, policy.side_inputs)?;
    if policy.side_inputs == crate::plan::EmptyGraphSideInputPolicy::RequireVertexPairs {
        let pairs = similarity_vertex_pairs(&node.spec, side_inputs)?;
        if pairs.num_rows() == 0 {
            return empty_graph_execution_result(context, &node.spec, policy, "vertex_pairs_empty").map(Some);
        }
    }
    if sanitized_input_rows != 0 {
        return Ok(None);
    }

    match policy.edge_input {
        crate::plan::EmptyGraphEdgeInputAction::ReturnEmpty => {
            empty_graph_execution_result(context, &node.spec, policy, "edge_source_empty").map(Some)
        },
        crate::plan::EmptyGraphEdgeInputAction::ValidateSelectorsThenReturnEmpty => {
            validate_empty_graph_selectors(&node.spec, input_rows, sanitized_input_rows)?;
            empty_graph_execution_result(context, &node.spec, policy, "edge_source_empty").map(Some)
        },
        crate::plan::EmptyGraphEdgeInputAction::Reject => Err(empty_graph_input_error(
            &node.spec,
            "edge",
            "operation_requires_nonempty_graph",
            input_rows,
            sanitized_input_rows,
        )),
        crate::plan::EmptyGraphEdgeInputAction::BfsEndpointPolicy => {
            traversal::resolve_empty_bfs_edge_input(context, &node.spec, sanitized_edges, side_inputs).map(Some)
        },
        crate::plan::EmptyGraphEdgeInputAction::SimilarityVertexPairs => {
            let pairs = similarity_vertex_pairs(&node.spec, side_inputs)?;
            Err(empty_graph_input_error(
                &node.spec,
                "edge",
                "nonempty_vertex_pairs_require_graph",
                input_rows,
                sanitized_input_rows,
            )
            .with_fact("vertex_pairs_node", similarity_vertex_pairs_params(&node.spec)?.vertex_pairs.input)
            .with_fact("vertex_pair_rows", pairs.num_rows()))
        },
    }
}

fn validate_required_side_inputs(
    spec: &crate::plan::GraphAlgorithmSpec,
    side_inputs: &std::collections::BTreeMap<NodeId, GpuDataFrame>,
    policy: crate::plan::EmptyGraphSideInputPolicy,
) -> Result<()> {
    match policy {
        crate::plan::EmptyGraphSideInputPolicy::None | crate::plan::EmptyGraphSideInputPolicy::BfsPredicates => Ok(()),
        crate::plan::EmptyGraphSideInputPolicy::RequireVertexPairs => {
            similarity_vertex_pairs(spec, side_inputs).map(|_| ())
        },
        crate::plan::EmptyGraphSideInputPolicy::RequirePersonalization => {
            let crate::plan::GraphAlgorithmParams::PersonalizedPageRank(params) = &spec.params else {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.graph_algorithm.empty_input",
                    "Personalized PageRank empty-input policy was attached to incompatible parameters",
                )
                .with_fact("algorithm", spec.algorithm.as_str()));
            };
            let input = side_inputs.get(&params.personalization.input).ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "nexus_query_engine.graph_algorithm.personalized_pagerank",
                    "Personalized PageRank graph algorithm node is missing its personalization side input",
                )
                .with_fact("algorithm", spec.algorithm.as_str())
                .with_fact("input_role", "personalization")
                .with_fact("side_input_node", params.personalization.input)
            })?;
            let rows = input.num_rows();
            if rows != 0 {
                return Ok(());
            }
            Err(QueryError::new(
                QueryErrorCode::GraphInputInvalid,
                "nexus_query_engine.graph_algorithm.empty_input",
                "personalized PageRank requires a non-empty personalization side input",
            )
            .with_fact("algorithm", spec.algorithm.as_str())
            .with_fact("input_role", "personalization")
            .with_fact("side_input_node", params.personalization.input)
            .with_fact("side_input_rows", rows)
            .with_fact("reason", "required_side_input_empty")
            .with_fact("empty_input_policy", policy.as_str()))
        },
    }
}

fn similarity_vertex_pairs_params(
    spec: &crate::plan::GraphAlgorithmSpec,
) -> Result<&crate::plan::GraphSimilarityParams> {
    let crate::plan::GraphAlgorithmParams::Similarity(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.empty_input",
            "similarity empty-input policy was attached to incompatible parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };
    Ok(params)
}

fn similarity_vertex_pairs<'a>(
    spec: &crate::plan::GraphAlgorithmSpec,
    side_inputs: &'a std::collections::BTreeMap<NodeId, GpuDataFrame>,
) -> Result<&'a GpuDataFrame> {
    let params = similarity_vertex_pairs_params(spec)?;
    side_inputs.get(&params.vertex_pairs.input).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.similarity",
            "similarity graph algorithm node is missing its vertex-pair side input",
        )
        .with_fact("algorithm", spec.algorithm.as_str())
        .with_fact("input_role", "vertex_pairs")
        .with_fact("side_input_node", params.vertex_pairs.input)
    })
}

fn validate_empty_graph_selectors(
    spec: &crate::plan::GraphAlgorithmSpec,
    input_rows: u64,
    sanitized_input_rows: u64,
) -> Result<()> {
    let (seeds, k) = match &spec.params {
        crate::plan::GraphAlgorithmParams::BetweennessCentrality(params) => (&params.seeds, params.k),
        crate::plan::GraphAlgorithmParams::EdgeBetweennessCentrality(params) => (&params.seeds, params.k),
        _ => {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.graph_algorithm.empty_input",
                "selector-validation empty-input policy was attached to incompatible parameters",
            )
            .with_fact("algorithm", spec.algorithm.as_str()));
        },
    };
    if let Some(seeds) = seeds {
        return Err(empty_graph_input_error(
            spec,
            "seed_vertices",
            "explicit_seed_vertices_require_graph_vertices",
            input_rows,
            sanitized_input_rows,
        )
        .with_fact("selector_rows", seeds.len()));
    }
    if let Some(k) = k {
        return Err(empty_graph_input_error(
            spec,
            "seed_count",
            "requested_seed_count_requires_graph_vertices",
            input_rows,
            sanitized_input_rows,
        )
        .with_fact("requested_seed_count", k));
    }
    Ok(())
}

fn empty_graph_input_error(
    spec: &crate::plan::GraphAlgorithmSpec,
    input_role: &'static str,
    reason: &'static str,
    input_rows: u64,
    sanitized_input_rows: u64,
) -> QueryError {
    QueryError::new(
        QueryErrorCode::GraphInputInvalid,
        "nexus_query_engine.graph_algorithm.empty_input",
        "graph operation requires input that is absent after edge sanitization",
    )
    .with_fact("algorithm", spec.algorithm.as_str())
    .with_fact("input_role", input_role)
    .with_fact("reason", reason)
    .with_fact("input_rows", input_rows)
    .with_fact("sanitized_input_rows", sanitized_input_rows)
    .with_fact("empty_input_policy", spec.algorithm.definition().empty_input_policy().edge_input.as_str())
}

pub(super) fn empty_graph_execution_result(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    policy: crate::plan::EmptyGraphInputPolicy,
    reason: &'static str,
) -> Result<GraphAlgorithmExecutionResult> {
    Ok(GraphAlgorithmExecutionResult {
        frame: empty_graph_result_frame(context, &spec.output_schema)?,
        algorithm_metadata: format!(
            "empty_result_reason={reason},empty_edge_input_policy={},empty_side_input_policy={}",
            policy.edge_input.as_str(),
            policy.side_inputs.as_str(),
        ),
        budget_observations: Vec::new(),
        cache_scope_peak_bytes: None,
    })
}

pub(super) fn empty_graph_result_frame(
    context: &ExecutionContext,
    schema: &arrow::datatypes::SchemaRef,
) -> Result<DataFrame> {
    if schema.fields().is_empty() {
        return Ok(context.runtime()?.empty_dataframe());
    }
    let columns = schema
        .fields()
        .iter()
        .map(|field| {
            let dtype = crate::datatypes::cudf_dtype_from_arrow_type(field.data_type()).ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::UnsupportedType,
                    "nexus_query_engine.graph_algorithm.empty_input",
                    "empty graph result column type is not supported by native execution",
                )
                .with_fact("column", field.name().as_str())
                .with_fact("data_type", format!("{:?}", field.data_type()))
            })?;
            let series = context.runtime()?.empty_series(dtype).map_err(|error| {
                QueryError::from_cudf(
                    "nexus_query_engine.graph_algorithm.empty_input",
                    "failed to build empty graph result column",
                    error,
                )
                .with_fact("column", field.name().as_str())
                .with_fact("data_type", format!("{:?}", field.data_type()))
            })?;
            Ok((field.name().clone(), series))
        })
        .collect::<Result<Vec<_>>>()?;
    DataFrame::from_series(columns).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.graph_algorithm.empty_input",
            "failed to build empty graph result frame",
            error,
        )
    })
}
