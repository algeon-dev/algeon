#[cfg(feature = "cugraph")]
use arrow::datatypes::DataType;

#[cfg(feature = "cugraph")]
use super::model::hard_row_upper_bound_for_plan_node;
use super::{evidence::GraphMemoryEstimate, model::missing_graph_input_error};
use crate::{
    error::{QueryError, QueryErrorCode, Result},
    plan::{GraphAlgorithm, GraphAlgorithmParams, NodeId, PlanNode, QueryPlan},
};

pub(super) fn graph_side_input_workspace_peak_bytes(
    plan: &QueryPlan,
    graph_node_id: NodeId,
    _estimate: &GraphMemoryEstimate,
) -> Result<Option<u64>> {
    let graph = match &plan.node(graph_node_id).ok_or_else(|| missing_graph_input_error(graph_node_id))?.kind {
        PlanNode::GraphAlgorithm(graph) => graph,
        _ => {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.graph_memory.side_input_workspace_preflight",
                "graph memory evidence referenced a non-graph node",
            )
            .with_fact("reason", "side_input_workspace_preflight_unavailable")
            .with_fact("side_input_reason", "graph_node_missing"));
        },
    };

    #[cfg(feature = "cugraph")]
    {
        graph_side_input_workspace_peak_bytes_with_owner(plan, graph, _estimate)
    }
    #[cfg(not(feature = "cugraph"))]
    {
        let has_bounded_side_input = match &graph.spec.params {
            GraphAlgorithmParams::Bfs(params) => params.has_predicates() || params.source_vertices.is_some(),
            GraphAlgorithmParams::PersonalizedPageRank(_) => true,
            _ => false,
        };
        if has_bounded_side_input {
            return Err(side_input_workspace_preflight_unavailable("owner_contract_unavailable", None, None));
        }
        Ok(None)
    }
}

fn side_input_workspace_preflight_unavailable(
    side_input_reason: &'static str,
    role: Option<&'static str>,
    algorithm: Option<GraphAlgorithm>,
) -> QueryError {
    side_input_workspace_error(QueryErrorCode::UnsupportedPlanShape, side_input_reason, role, algorithm)
}

fn side_input_workspace_error(
    code: QueryErrorCode,
    side_input_reason: &'static str,
    role: Option<&'static str>,
    algorithm: Option<GraphAlgorithm>,
) -> QueryError {
    let error = QueryError::new(
        code,
        "nexus_query_engine.graph_memory.side_input_workspace_preflight",
        "cuGraph side-input workspace preflight is unavailable",
    )
    .with_fact("reason", "side_input_workspace_preflight_unavailable")
    .with_fact("side_input_reason", side_input_reason);
    let error = match role {
        Some(role) => error.with_fact("side_input_role", role),
        None => error,
    };
    match algorithm {
        Some(algorithm) => error.with_fact("algorithm", algorithm.as_str()),
        None => error,
    }
}

#[cfg(feature = "cugraph")]
#[derive(Debug, Clone, Copy)]
enum GraphSideInputColumns<'a> {
    Vertex(&'a str),
    Personalization { vertex: &'a str, value: &'a str },
}

#[cfg(feature = "cugraph")]
#[derive(Debug, Clone, Copy)]
struct GraphSideInputBinding<'a> {
    role: &'static str,
    input: NodeId,
    columns: GraphSideInputColumns<'a>,
}

#[cfg(feature = "cugraph")]
fn graph_side_input_workspace_peak_bytes_with_owner(
    plan: &QueryPlan,
    graph: &crate::plan::GraphAlgorithmNode,
    estimate: &GraphMemoryEstimate,
) -> Result<Option<u64>> {
    match &graph.spec.params {
        GraphAlgorithmParams::Bfs(_params) => {
            let bindings = graph_side_input_bindings(graph)?;
            let graph_vertex_type = graph_side_input_graph_vertex_type(plan, graph)?;
            let source_rows = bindings
                .iter()
                .find(|binding| binding.role == "source_vertices")
                .map(|binding| graph_side_input_rows(plan, binding))
                .transpose()?
                .unwrap_or(1);
            let mut include_vertices_rows = None;
            let mut exclude_vertices_rows = None;
            let mut target_vertices_rows = None;
            let mut include_edge_ids_rows = None;
            for binding in &bindings {
                let rows = graph_side_input_rows(plan, binding)?;
                let GraphSideInputColumns::Vertex(column) = binding.columns else {
                    return Err(side_input_workspace_preflight_unavailable(
                        "invalid_bfs_binding",
                        Some(binding.role),
                        Some(graph.spec.algorithm),
                    ));
                };
                graph_side_input_vertex_type(
                    plan,
                    binding.input,
                    column,
                    graph_vertex_type,
                    binding.role,
                    graph.spec.algorithm,
                )?;
                match binding.role {
                    "source_vertices" => {},
                    "include_vertices" => include_vertices_rows = Some(rows),
                    "exclude_vertices" => exclude_vertices_rows = Some(rows),
                    "target_vertices" => target_vertices_rows = Some(rows),
                    "include_edge_ids" => include_edge_ids_rows = Some(rows),
                    _ => {
                        return Err(side_input_workspace_preflight_unavailable(
                            "unknown_bfs_binding",
                            Some(binding.role),
                            Some(graph.spec.algorithm),
                        ));
                    },
                }
            }
            let num_vertices = graph_side_input_count(estimate.vertex_estimate, "num_vertices", graph.spec.algorithm)?;
            let num_edges = graph_side_input_count(estimate.input_edges, "num_edges", graph.spec.algorithm)?;
            if include_vertices_rows.is_some() && exclude_vertices_rows.is_some() {
                return Err(side_input_workspace_preflight_unavailable(
                    "owner_input_rejected",
                    None,
                    Some(graph.spec.algorithm),
                ));
            }
            if source_rows == 0 {
                return Ok(Some(0));
            }
            let input = cugraph::cudf_interop::BfsSideInputPreflightInput::new(
                graph_vertex_type,
                graph_vertex_type,
                num_vertices,
                num_edges,
                source_rows,
                include_vertices_rows,
                exclude_vertices_rows,
                target_vertices_rows,
                include_edge_ids_rows,
            )
            .map_err(|error| {
                side_input_workspace_preflight_unavailable("owner_input_rejected", None, Some(graph.spec.algorithm))
                    .with_source(error)
            })?;
            let preflight = cugraph::cudf_interop::bfs_side_input_workspace_preflight(&input).map_err(|error| {
                side_input_workspace_preflight_unavailable("owner_query_failed", None, Some(graph.spec.algorithm))
                    .with_source(error)
            })?;
            Ok(Some(preflight.peak_bytes()))
        },
        GraphAlgorithmParams::PersonalizedPageRank(_params) => {
            let bindings = graph_side_input_bindings(graph)?;
            let [binding] = bindings.as_slice() else {
                return Err(side_input_workspace_preflight_unavailable(
                    "personalization_binding_missing",
                    Some("personalization"),
                    Some(graph.spec.algorithm),
                ));
            };
            let GraphSideInputColumns::Personalization { vertex, value } = binding.columns else {
                return Err(side_input_workspace_preflight_unavailable(
                    "invalid_personalization_binding",
                    Some(binding.role),
                    Some(graph.spec.algorithm),
                ));
            };
            let graph_vertex_type = graph_side_input_graph_vertex_type(plan, graph)?;
            graph_side_input_vertex_type(
                plan,
                binding.input,
                vertex,
                graph_vertex_type,
                binding.role,
                graph.spec.algorithm,
            )?;
            let value_type =
                graph_side_input_value_type(plan, binding.input, value, binding.role, graph.spec.algorithm)?;
            let rows = graph_side_input_rows(plan, binding)?;
            let input = cugraph::cudf_interop::PersonalizedPageRankSideInputPreflightInput::new(
                graph_vertex_type,
                value_type,
                rows,
            )
            .map_err(|error| {
                side_input_workspace_preflight_unavailable(
                    "owner_input_rejected",
                    Some(binding.role),
                    Some(graph.spec.algorithm),
                )
                .with_source(error)
            })?;
            let preflight = cugraph::cudf_interop::personalized_pagerank_side_input_workspace_preflight(&input)
                .map_err(|error| {
                    side_input_workspace_preflight_unavailable(
                        "owner_query_failed",
                        Some(binding.role),
                        Some(graph.spec.algorithm),
                    )
                    .with_source(error)
                })?;
            Ok(Some(preflight.peak_bytes()))
        },
        _ => Ok(None),
    }
}

#[cfg(feature = "cugraph")]
fn graph_side_input_bindings<'a>(graph: &'a crate::plan::GraphAlgorithmNode) -> Result<Vec<GraphSideInputBinding<'a>>> {
    let mut bindings = Vec::new();
    match &graph.spec.params {
        GraphAlgorithmParams::Bfs(params) => {
            if params.predicates.include_edges.is_some() {
                return Err(side_input_workspace_preflight_unavailable(
                    "include_edges_owner_contract_missing",
                    Some("include_edges"),
                    Some(graph.spec.algorithm),
                ));
            }
            if let Some(input) = params.source_vertices.as_ref() {
                bindings.push(GraphSideInputBinding {
                    role: "source_vertices",
                    input: input.input,
                    columns: GraphSideInputColumns::Vertex(input.column.as_str()),
                });
            }
            if let Some(input) = params.predicates.include_vertices.as_ref() {
                bindings.push(GraphSideInputBinding {
                    role: "include_vertices",
                    input: input.input,
                    columns: GraphSideInputColumns::Vertex(input.column.as_str()),
                });
            }
            if let Some(input) = params.predicates.exclude_vertices.as_ref() {
                bindings.push(GraphSideInputBinding {
                    role: "exclude_vertices",
                    input: input.input,
                    columns: GraphSideInputColumns::Vertex(input.column.as_str()),
                });
            }
            if let Some(input) = params.predicates.target_vertices.as_ref() {
                bindings.push(GraphSideInputBinding {
                    role: "target_vertices",
                    input: input.input,
                    columns: GraphSideInputColumns::Vertex(input.column.as_str()),
                });
            }
            if let Some(input) = params.predicates.include_edge_ids.as_ref() {
                bindings.push(GraphSideInputBinding {
                    role: "include_edge_ids",
                    input: input.input,
                    columns: GraphSideInputColumns::Vertex(input.column.as_str()),
                });
            }
        },
        GraphAlgorithmParams::PersonalizedPageRank(params) => {
            let input = &params.personalization;
            bindings.push(GraphSideInputBinding {
                role: "personalization",
                input: input.input,
                columns: GraphSideInputColumns::Personalization {
                    vertex: input.vertex.as_str(),
                    value: input.value.as_str(),
                },
            });
        },
        _ => return Ok(bindings),
    }

    let mut expected_inputs = Vec::with_capacity(bindings.len() + 1);
    expected_inputs.push(graph.input);
    expected_inputs.extend(bindings.iter().map(|binding| binding.input));
    if graph.inputs != expected_inputs {
        return Err(side_input_workspace_preflight_unavailable(
            "graph_input_binding_mismatch",
            None,
            Some(graph.spec.algorithm),
        ));
    }
    let mut seen = std::collections::HashSet::with_capacity(expected_inputs.len());
    for input in expected_inputs {
        if !seen.insert(input) {
            return Err(side_input_workspace_preflight_unavailable(
                "side_input_node_reused",
                None,
                Some(graph.spec.algorithm),
            )
            .with_fact("node_id", input));
        }
    }
    Ok(bindings)
}

#[cfg(feature = "cugraph")]
fn graph_side_input_rows(plan: &QueryPlan, binding: &GraphSideInputBinding<'_>) -> Result<usize> {
    let Some((rows, _)) = hard_row_upper_bound_for_plan_node(plan, binding.input) else {
        return Err(side_input_workspace_preflight_unavailable("missing_row_upper_bound", Some(binding.role), None)
            .with_fact("node_id", binding.input));
    };
    usize::try_from(rows).map_err(|_| {
        side_input_workspace_preflight_unavailable("row_count_conversion_overflow", Some(binding.role), None)
            .with_fact("row_count", rows)
            .with_fact("node_id", binding.input)
    })
}

#[cfg(feature = "cugraph")]
fn graph_side_input_count(value: u64, term: &'static str, algorithm: GraphAlgorithm) -> Result<usize> {
    usize::try_from(value).map_err(|_| {
        side_input_workspace_preflight_unavailable("graph_count_conversion_overflow", None, Some(algorithm))
            .with_fact("term", term)
            .with_fact("value", value)
    })
}

#[cfg(feature = "cugraph")]
fn graph_side_input_graph_vertex_type(
    plan: &QueryPlan,
    graph: &crate::plan::GraphAlgorithmNode,
) -> Result<cugraph::types::DataType> {
    let schema = plan.node(graph.input).ok_or_else(|| missing_graph_input_error(graph.input))?.output_schema();
    let field = schema.field_with_name(&graph.spec.edge_columns.source).map_err(|_| {
        side_input_workspace_preflight_unavailable(
            "graph_vertex_column_missing",
            Some("graph_vertex"),
            Some(graph.spec.algorithm),
        )
    })?;
    cugraph_index_type_for_arrow(field.data_type(), "graph_vertex", graph.spec.algorithm)
}

#[cfg(feature = "cugraph")]
fn graph_side_input_vertex_type(
    plan: &QueryPlan,
    node_id: NodeId,
    column: &str,
    expected: cugraph::types::DataType,
    role: &'static str,
    algorithm: GraphAlgorithm,
) -> Result<()> {
    let schema = plan
        .node(node_id)
        .ok_or_else(|| {
            side_input_workspace_preflight_unavailable("side_input_node_missing", Some(role), Some(algorithm))
                .with_fact("node_id", node_id)
        })?
        .output_schema();
    let field = schema.field_with_name(column).map_err(|_| {
        side_input_workspace_preflight_unavailable("side_input_column_missing", Some(role), Some(algorithm))
            .with_fact("column", column)
            .with_fact("node_id", node_id)
    })?;
    let actual = cugraph_index_type_for_arrow(field.data_type(), role, algorithm)?;
    if actual != expected {
        return Err(side_input_workspace_error(
            QueryErrorCode::UnsupportedType,
            "side_input_vertex_dtype_mismatch",
            Some(role),
            Some(algorithm),
        )
        .with_fact("expected_dtype", expected)
        .with_fact("actual_dtype", actual)
        .with_fact("column", column));
    }
    Ok(())
}

#[cfg(feature = "cugraph")]
fn graph_side_input_value_type(
    plan: &QueryPlan,
    node_id: NodeId,
    column: &str,
    role: &'static str,
    algorithm: GraphAlgorithm,
) -> Result<cugraph::types::DataType> {
    let schema = plan
        .node(node_id)
        .ok_or_else(|| {
            side_input_workspace_preflight_unavailable("side_input_node_missing", Some(role), Some(algorithm))
                .with_fact("node_id", node_id)
        })?
        .output_schema();
    let field = schema.field_with_name(column).map_err(|_| {
        side_input_workspace_preflight_unavailable("side_input_column_missing", Some(role), Some(algorithm))
            .with_fact("column", column)
            .with_fact("node_id", node_id)
    })?;
    match field.data_type() {
        DataType::Float32 => Ok(cugraph::types::DataType::Float32),
        DataType::Float64 => Ok(cugraph::types::DataType::Float64),
        _ => Err(side_input_workspace_error(
            QueryErrorCode::UnsupportedType,
            "side_input_value_dtype_unsupported",
            Some(role),
            Some(algorithm),
        )
        .with_fact("column", column)),
    }
}

#[cfg(feature = "cugraph")]
fn cugraph_index_type_for_arrow(
    data_type: &DataType,
    role: &'static str,
    algorithm: GraphAlgorithm,
) -> Result<cugraph::types::DataType> {
    match data_type {
        DataType::Int32 => Ok(cugraph::types::DataType::Int32),
        DataType::Int64 => Ok(cugraph::types::DataType::Int64),
        _ => Err(side_input_workspace_error(
            QueryErrorCode::UnsupportedType,
            "side_input_vertex_dtype_unsupported",
            Some(role),
            Some(algorithm),
        )
        .with_fact("data_type", data_type)),
    }
}
