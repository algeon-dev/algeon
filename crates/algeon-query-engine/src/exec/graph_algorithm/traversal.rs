use cudf::{
    DataFrameJoinOptions, DataType, ExecutionOverrides, NullEquality, Series, TypeId, core::DeviceSlice,
    frame::dataframe::DataFrameFilteredJoinOptions,
};
use cugraph::{
    DeviceArray, ErrorKind,
    cudf_interop::device_array_view_from_slice,
    traversal::{BfsPredicateParams, BfsResult, BfsTargetInfo},
    types::CugraphElement,
};

use super::{
    DataFrame, ExecutionContext, GpuDataFrame, GraphAlgorithmExecutionInputs, GraphAlgorithmExecutionResult, NodeId,
    QueryError, QueryErrorCode, Result, bfs_params, col, cudf_query_error, empty_graph_result_frame,
    execution_overrides, map_cugraph_error, normalize_graph_result_schema, project_edge_list,
    run_with_accounted_resident_cugraph_graph, run_with_cugraph_graph, sssp_params, table_graph_options,
};
use crate::plan::{
    GraphAlgorithmParams, GraphAlgorithmSpec, GraphBfsEndpointPolicy, GraphBfsParams, GraphEdgePredicateInput,
    GraphPredicateInput, GraphSsspSource, planned_resident_graph_key,
};

pub(super) fn execute_bfs(
    context: &ExecutionContext,
    spec: &GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult> {
    let GraphAlgorithmExecutionInputs { edge_list: input, side_inputs } = inputs;
    let GraphAlgorithmParams::Bfs(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.graph_algorithm.bfs",
            "BFS graph algorithm node carried non-BFS parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let resident_key = input.graph_residency().and_then(|descriptor| planned_resident_graph_key(descriptor, spec));
    let input = input.into_frame();
    let (mut edges, vertex_type) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);
    let bfs_p = bfs_params(params)?;
    let predicate_columns = BfsPredicateColumns::from_side_inputs(
        context,
        params,
        &side_inputs,
        vertex_type,
        spec.edge_columns.edge_id.is_some(),
        spec,
        &mut edges,
    )?;
    if let Some(result) = empty_bfs_result_for_trivial_path_endpoint(context, spec, params, &edges, &predicate_columns)?
    {
        return Ok(result);
    }
    let endpoint_edges = edges.clone();

    let run_bfs = |runtime: &cudf::Runtime,
                   entered: &cugraph::EnteredContext<'_>,
                   graph: &mut cugraph::Graph<cugraph::Forward>| {
        let (frame, target_metadata) = match vertex_type {
            TypeId::Int32 => {
                let source = i32::try_from(params.source_vertex).map_err(|_| {
                    QueryError::new(
                        QueryErrorCode::InvalidPlan,
                        "query_engine.graph_algorithm.bfs_source",
                        "BFS source vertex does not fit the Int32 graph vertex dtype",
                    )
                    .with_fact("source_vertex", params.source_vertex)
                    .with_fact("vertex_dtype", "Int32")
                })?;
                let source_slice =
                    predicate_column_slice::<i32>(predicate_columns.source_vertices.as_ref(), "source_vertices")?;
                let source_array = if source_slice.is_none() {
                    Some(DeviceArray::from_slice(entered, &[source]).map_err(map_cugraph_error(
                        "query_engine.graph_algorithm.bfs_source",
                        "failed to copy BFS source vertex to device memory",
                    ))?)
                } else {
                    None
                };
                let sources_view = match (source_slice.as_ref(), source_array.as_ref()) {
                    (Some(slice), None) => device_array_view_from_slice::<i32>(slice).map_err(map_cugraph_error(
                        "query_engine.graph_algorithm.bfs_source",
                        "failed to wrap BFS source-vertex side input as a cuGraph device-array view",
                    ))?,
                    (None, Some(array)) => array.view().map_err(map_cugraph_error(
                        "query_engine.graph_algorithm.bfs_source",
                        "failed to create BFS source vertex view",
                    ))?,
                    _ => unreachable!("source array is created iff no source side input exists"),
                };
                let include_vertices_slice =
                    predicate_column_slice::<i32>(predicate_columns.include_vertices.as_ref(), "include_vertices")?;
                let exclude_vertices_slice =
                    predicate_column_slice::<i32>(predicate_columns.exclude_vertices.as_ref(), "exclude_vertices")?;
                let target_vertices_slice =
                    predicate_column_slice::<i32>(predicate_columns.target_vertices.as_ref(), "target_vertices")?;
                let include_edge_ids_slice =
                    predicate_column_slice::<i32>(predicate_columns.include_edge_ids.as_ref(), "include_edge_ids")?;
                let include_vertices_view =
                    predicate_device_view::<i32>(include_vertices_slice.as_ref(), "include_vertices")?;
                let exclude_vertices_view =
                    predicate_device_view::<i32>(exclude_vertices_slice.as_ref(), "exclude_vertices")?;
                let target_vertices_view =
                    predicate_device_view::<i32>(target_vertices_slice.as_ref(), "target_vertices")?;
                let include_edge_ids_view =
                    predicate_device_view::<i32>(include_edge_ids_slice.as_ref(), "include_edge_ids")?;
                let predicates = BfsPredicateParams::<i32, i32> {
                    include_vertices: include_vertices_view.as_ref(),
                    exclude_vertices: exclude_vertices_view.as_ref(),
                    target_vertices: target_vertices_view.as_ref(),
                    include_edge_ids: include_edge_ids_view.as_ref(),
                    stop_when_any_target_found: target_vertices_view.is_some(),
                };
                let result = if predicate_columns.requires_predicate_bfs_call() {
                    match graph.bfs_with_predicates(entered, &sources_view, &bfs_p, &predicates) {
                        Ok(result) => result,
                        Err(error) => {
                            return graph_bfs_error_for_missing_path_endpoint(
                                runtime,
                                spec,
                                params,
                                &endpoint_edges,
                                vertex_type,
                                &predicate_columns,
                                error,
                                "cuGraph predicate BFS failed",
                            );
                        },
                    }
                } else {
                    match graph.bfs(entered, &sources_view, &bfs_p) {
                        Ok(result) => result,
                        Err(error) => {
                            return graph_bfs_error_for_missing_path_endpoint(
                                runtime,
                                spec,
                                params,
                                &endpoint_edges,
                                vertex_type,
                                &predicate_columns,
                                error,
                                "cuGraph BFS failed",
                            );
                        },
                    }
                };
                let target_info = result.target_info().copied();
                let target_metadata = bfs_target_metadata(target_info.as_ref());
                let frame = if params.path_output.is_some() {
                    extract_bfs_path_frame(
                        runtime,
                        spec,
                        params,
                        entered,
                        graph,
                        &result,
                        &sources_view,
                        target_vertices_view.as_ref(),
                        predicate_columns.target_vertices.as_ref(),
                        target_info,
                    )?
                } else {
                    let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                        "query_engine.graph_algorithm.bfs_result",
                        "failed to convert cuGraph BFS result to cudf DataFrame",
                    ))?;
                    append_bfs_target_metadata_columns(runtime, params, frame, target_info)?
                };
                (frame, target_metadata)
            },
            TypeId::Int64 => {
                let source_slice =
                    predicate_column_slice::<i64>(predicate_columns.source_vertices.as_ref(), "source_vertices")?;
                let source_array = if source_slice.is_none() {
                    Some(DeviceArray::from_slice(entered, &[params.source_vertex]).map_err(map_cugraph_error(
                        "query_engine.graph_algorithm.bfs_source",
                        "failed to copy BFS source vertex to device memory",
                    ))?)
                } else {
                    None
                };
                let sources_view = match (source_slice.as_ref(), source_array.as_ref()) {
                    (Some(slice), None) => device_array_view_from_slice::<i64>(slice).map_err(map_cugraph_error(
                        "query_engine.graph_algorithm.bfs_source",
                        "failed to wrap BFS source-vertex side input as a cuGraph device-array view",
                    ))?,
                    (None, Some(array)) => array.view().map_err(map_cugraph_error(
                        "query_engine.graph_algorithm.bfs_source",
                        "failed to create BFS source vertex view",
                    ))?,
                    _ => unreachable!("source array is created iff no source side input exists"),
                };
                let include_vertices_slice =
                    predicate_column_slice::<i64>(predicate_columns.include_vertices.as_ref(), "include_vertices")?;
                let exclude_vertices_slice =
                    predicate_column_slice::<i64>(predicate_columns.exclude_vertices.as_ref(), "exclude_vertices")?;
                let target_vertices_slice =
                    predicate_column_slice::<i64>(predicate_columns.target_vertices.as_ref(), "target_vertices")?;
                let include_edge_ids_slice =
                    predicate_column_slice::<i64>(predicate_columns.include_edge_ids.as_ref(), "include_edge_ids")?;
                let include_vertices_view =
                    predicate_device_view::<i64>(include_vertices_slice.as_ref(), "include_vertices")?;
                let exclude_vertices_view =
                    predicate_device_view::<i64>(exclude_vertices_slice.as_ref(), "exclude_vertices")?;
                let target_vertices_view =
                    predicate_device_view::<i64>(target_vertices_slice.as_ref(), "target_vertices")?;
                let include_edge_ids_view =
                    predicate_device_view::<i64>(include_edge_ids_slice.as_ref(), "include_edge_ids")?;
                let predicates = BfsPredicateParams::<i64, i64> {
                    include_vertices: include_vertices_view.as_ref(),
                    exclude_vertices: exclude_vertices_view.as_ref(),
                    target_vertices: target_vertices_view.as_ref(),
                    include_edge_ids: include_edge_ids_view.as_ref(),
                    stop_when_any_target_found: target_vertices_view.is_some(),
                };
                let result = if predicate_columns.requires_predicate_bfs_call() {
                    match graph.bfs_with_predicates(entered, &sources_view, &bfs_p, &predicates) {
                        Ok(result) => result,
                        Err(error) => {
                            return graph_bfs_error_for_missing_path_endpoint(
                                runtime,
                                spec,
                                params,
                                &endpoint_edges,
                                vertex_type,
                                &predicate_columns,
                                error,
                                "cuGraph predicate BFS failed",
                            );
                        },
                    }
                } else {
                    match graph.bfs(entered, &sources_view, &bfs_p) {
                        Ok(result) => result,
                        Err(error) => {
                            return graph_bfs_error_for_missing_path_endpoint(
                                runtime,
                                spec,
                                params,
                                &endpoint_edges,
                                vertex_type,
                                &predicate_columns,
                                error,
                                "cuGraph BFS failed",
                            );
                        },
                    }
                };
                let target_info = result.target_info().copied();
                let target_metadata = bfs_target_metadata(target_info.as_ref());
                let frame = if params.path_output.is_some() {
                    extract_bfs_path_frame(
                        runtime,
                        spec,
                        params,
                        entered,
                        graph,
                        &result,
                        &sources_view,
                        target_vertices_view.as_ref(),
                        predicate_columns.target_vertices.as_ref(),
                        target_info,
                    )?
                } else {
                    let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                        "query_engine.graph_algorithm.bfs_result",
                        "failed to convert cuGraph BFS result to cudf DataFrame",
                    ))?;
                    append_bfs_target_metadata_columns(runtime, params, frame, target_info)?
                };
                (frame, target_metadata)
            },
            _ => {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedType,
                    "query_engine.graph_algorithm.bfs_source",
                    "BFS graph vertex dtype is unsupported",
                )
                .with_fact("vertex_dtype", format!("{vertex_type:?}")));
            },
        };
        Ok(GraphAlgorithmExecutionResult {
            frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
            algorithm_metadata: format!(
                "source_vertex={},depth_limit={},compute_predecessors=true,{},{}",
                params.source_vertex,
                params.depth_limit.map_or_else(|| "none".to_string(), |value| value.to_string()),
                predicate_columns.metadata(),
                target_metadata,
            ),
            cache_scope_peak_bytes: None,
        })
    };
    run_with_accounted_resident_cugraph_graph::<cugraph::Forward, _>(
        context,
        edges,
        options,
        resident_key,
        "algeon_datafusion_cugraph_bfs",
        run_bfs,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "BFS path extraction passes the native call state explicitly at the FFI boundary"
)]
fn extract_bfs_path_frame<V>(
    runtime: &cudf::Runtime,
    spec: &GraphAlgorithmSpec,
    params: &GraphBfsParams,
    entered: &cugraph::EnteredContext<'_>,
    graph: &mut cugraph::Graph<cugraph::Forward>,
    result: &BfsResult,
    sources: &cugraph::DeviceArrayView<'_, V>,
    destinations: Option<&cugraph::DeviceArrayView<'_, V>>,
    destination_series: Option<&cudf::Series>,
    target_info: Option<BfsTargetInfo>,
) -> Result<DataFrame>
where
    V: CugraphElement + cudf::CudfType,
{
    let path_output = params.path_output.ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.graph_algorithm.bfs_path",
            "BFS path extraction requires a bounded path output request",
        )
    })?;
    let destinations = destinations.ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.graph_algorithm.bfs_path",
            "BFS path output target device view is missing",
        )
        .with_fact("reason", "bfs_path_target_binding_missing")
    })?;
    if destinations.len() != 1 {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.graph_algorithm.bfs_path",
            "BFS path output requires exactly one target vertex",
        )
        .with_fact("reason", "bfs_path_target_cardinality_invalid")
        .with_fact("target_rows", destinations.len()));
    }
    let target_info = target_info.ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.graph_algorithm.bfs_path",
            "BFS path output requires predicate target metadata",
        )
    })?;
    if !target_info.target_found {
        return empty_graph_result_frame(runtime, &spec.output_schema);
    }

    let extracted = result.extract_paths(entered, graph, sources, destinations).map_err(map_cugraph_error(
        "query_engine.graph_algorithm.bfs_path",
        "failed to extract bounded BFS path on device",
    ))?;
    let vertex = extracted.paths().and_then(|paths| paths.to_series(runtime)).map_err(map_cugraph_error(
        "query_engine.graph_algorithm.bfs_path",
        "failed to convert extracted BFS path to a cudf series",
    ))?;
    let rows = vertex.len();
    if rows as u64 > path_output.max_output_rows {
        return Err(QueryError::new(
            QueryErrorCode::CudfInvariantViolation,
            "query_engine.graph_algorithm.bfs_path",
            "native BFS path output exceeded its admitted row bound",
        )
        .with_fact("path_rows", rows)
        .with_fact("max_output_rows", path_output.max_output_rows));
    }
    let destination_series = destination_series.ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.graph_algorithm.bfs_path",
            "BFS path output target series is missing",
        )
    })?;
    let overrides =
        ExecutionOverrides::new().with_stream(runtime.stream()).with_memory_resource(runtime.memory_resource());
    let repeat = runtime.sequence(0_i32, 0_i32, rows).map_err(|error| {
        cudf_query_error("query_engine.graph_algorithm.bfs_path", "failed to build BFS path scalar gather map", error)
    })?;
    let source = vertex.slice_in(0, 1, &overrides).and_then(|source| source.gather_in(&repeat, &overrides)).map_err(
        |error| {
            cudf_query_error(
                "query_engine.graph_algorithm.bfs_path",
                "failed to materialize BFS path source column",
                error,
            )
        },
    )?;
    let target = destination_series
        .slice_in(0, 1, &overrides)
        .and_then(|target| target.gather_in(&repeat, &overrides))
        .map_err(|error| {
            cudf_query_error(
                "query_engine.graph_algorithm.bfs_path",
                "failed to materialize BFS path target column",
                error,
            )
        })?;
    let path_index = runtime.sequence(0_i64, 1_i64, rows).map_err(|error| {
        cudf_query_error("query_engine.graph_algorithm.bfs_path", "failed to materialize BFS path index", error)
    })?;
    let distance = path_index.clone();
    DataFrame::from_series([
        ("path_index", path_index),
        ("source", source),
        ("target", target),
        ("vertex", vertex),
        ("distance", distance),
    ])
    .map_err(|error| {
        cudf_query_error("query_engine.graph_algorithm.bfs_path", "failed to assemble bounded BFS path frame", error)
    })
}

struct BfsPredicateColumns {
    source_vertices: Option<cudf::Series>,
    include_vertices: Option<cudf::Series>,
    exclude_vertices: Option<cudf::Series>,
    target_vertices: Option<cudf::Series>,
    include_edge_ids: Option<cudf::Series>,
    include_edges_rows: Option<usize>,
    include_edges_aligned_rows: Option<usize>,
    include_edges_null_rows_dropped: Option<usize>,
}

impl BfsPredicateColumns {
    fn from_side_inputs(
        context: &ExecutionContext,
        params: &GraphBfsParams,
        side_inputs: &std::collections::BTreeMap<NodeId, GpuDataFrame>,
        vertex_type: cudf::TypeId,
        graph_has_edge_id_binding: bool,
        spec: &GraphAlgorithmSpec,
        edges: &mut DataFrame,
    ) -> Result<Self> {
        if params.predicates.include_vertices.is_some() && params.predicates.exclude_vertices.is_some() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.graph_algorithm.bfs_predicates",
                "BFS include_vertices and exclude_vertices predicates are mutually exclusive",
            ));
        }
        if params.return_target_info && params.predicates.target_vertices.is_none() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.graph_algorithm.bfs_target_metadata",
                "BFS return_target_info requires a target_vertices predicate input",
            ));
        }
        if params.predicates.include_edge_ids.is_some() && !graph_has_edge_id_binding {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.graph_algorithm.bfs_predicates",
                "BFS include_edge_ids requires the primary edge list to declare a stable edge-id column binding",
            )
            .with_fact("predicate_role", "include_edge_ids")
            .with_fact("missing_binding", "edge_id_col"));
        }
        let edge_alignment = align_include_edges(
            context,
            edges,
            spec,
            params.predicates.include_edges.as_ref(),
            side_inputs,
            vertex_type,
        )?;
        Ok(Self {
            source_vertices: source_vertex_series(
                side_inputs,
                params.source_vertices.as_ref(),
                vertex_type,
                params.endpoint_policy,
            )?,
            include_vertices: predicate_series(
                side_inputs,
                params.predicates.include_vertices.as_ref(),
                "include_vertices",
                vertex_type,
            )?,
            exclude_vertices: predicate_series(
                side_inputs,
                params.predicates.exclude_vertices.as_ref(),
                "exclude_vertices",
                vertex_type,
            )?,
            target_vertices: predicate_series(
                side_inputs,
                params.predicates.target_vertices.as_ref(),
                "target_vertices",
                vertex_type,
            )?,
            include_edge_ids: predicate_series(
                side_inputs,
                params.predicates.include_edge_ids.as_ref(),
                "include_edge_ids",
                vertex_type,
            )?,
            include_edges_rows: edge_alignment.input_rows,
            include_edges_aligned_rows: edge_alignment.aligned_rows,
            include_edges_null_rows_dropped: edge_alignment.null_rows_dropped,
        })
    }

    fn requires_predicate_bfs_call(&self) -> bool {
        self.include_vertices.is_some()
            || self.exclude_vertices.is_some()
            || self.target_vertices.is_some()
            || self.include_edge_ids.is_some()
    }

    fn metadata(&self) -> String {
        format!(
            "source_vertices_rows={},predicate_include_vertices_rows={},predicate_exclude_vertices_rows={},predicate_target_vertices_rows={},predicate_include_edge_ids_rows={},predicate_include_edges_rows={},predicate_include_edges_aligned_rows={},predicate_include_edges_null_rows_dropped={},target_early_stop_supported={}",
            optional_series_len(self.source_vertices.as_ref()),
            optional_series_len(self.include_vertices.as_ref()),
            optional_series_len(self.exclude_vertices.as_ref()),
            optional_series_len(self.target_vertices.as_ref()),
            optional_series_len(self.include_edge_ids.as_ref()),
            optional_usize(self.include_edges_rows),
            optional_usize(self.include_edges_aligned_rows),
            optional_usize(self.include_edges_null_rows_dropped),
            self.target_vertices.is_some(),
        )
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct EdgePredicateAlignment {
    input_rows: Option<usize>,
    aligned_rows: Option<usize>,
    null_rows_dropped: Option<usize>,
}

fn align_include_edges(
    context: &ExecutionContext,
    edges: &mut DataFrame,
    spec: &GraphAlgorithmSpec,
    input: Option<&GraphEdgePredicateInput>,
    side_inputs: &std::collections::BTreeMap<NodeId, GpuDataFrame>,
    vertex_type: cudf::TypeId,
) -> Result<EdgePredicateAlignment> {
    let Some(input) = input else {
        return Ok(EdgePredicateAlignment::default());
    };
    let frame = side_inputs.get(&input.input).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.graph_algorithm.bfs_predicates",
            "BFS include_edges side input was not executed",
        )
        .with_fact("predicate_role", "include_edges")
        .with_fact("input", input.input)
    })?;
    let frame = frame.clone().into_frame();
    let input_rows = frame.num_rows();

    let mut left_keys = vec![spec.edge_columns.source.clone(), spec.edge_columns.destination.clone()];
    let mut right_keys = vec![input.source.clone(), input.destination.clone()];
    let mut predicate_columns =
        vec![("include_edges_source", input.source.clone()), ("include_edges_destination", input.destination.clone())];
    if let Some(edge_id) = &input.edge_id {
        let primary_edge_id = spec.edge_columns.edge_id.as_ref().ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.graph_algorithm.bfs_predicates",
                "BFS include_edges edge-id alignment requires a primary edge-id column binding",
            )
            .with_fact("predicate_role", "include_edges")
            .with_fact("input", input.input)
            .with_fact("missing_binding", "edge_id_col")
        })?;
        left_keys.push(primary_edge_id.clone());
        right_keys.push(edge_id.clone());
        predicate_columns.push(("include_edges_edge_id", edge_id.clone()));
    }

    // include_edges is a relational filter: a null src/dst/edge-id cannot identify
    // any allowed edge, so dropping the row preserves user intent. Decide on the
    // actual values (null_count), never schema nullability. Dtype is still
    // validated (it must match the graph vertex dtype for the semi-join keys).
    for (role, column) in &predicate_columns {
        let series = predicate_series_for_column(&frame, input.input, column, role)?;
        validate_predicate_series_dtype(&series, input.input, column, role, vertex_type)?;
    }
    let (allowed_edges, null_rows_dropped) = drop_null_include_edge_rows(context, input.input, &frame, &right_keys)?;

    let aligned_allowed_rows = allowed_edges.num_rows();
    if aligned_allowed_rows == 0 {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.graph_algorithm.bfs_predicates",
            "BFS include_edges whitelist is empty after dropping null rows",
        )
        .with_fact("predicate_role", "include_edges")
        .with_fact("input", input.input)
        .with_fact("input_rows", input_rows)
        .with_fact("null_rows_dropped", null_rows_dropped));
    }

    let indices = {
        let allowed_edges = allowed_edges;
        let overrides_lease = execution_overrides(context)?;
        let overrides = overrides_lease.overrides();
        let options = DataFrameFilteredJoinOptions::new().null_equality(NullEquality::Unequal);
        let join = allowed_edges
            .with_execution(&overrides)
            .build_filtered_join(right_keys.iter().map(String::as_str), options)
            .map_err(|error| {
                cudf_query_error(
                    "query_engine.graph_algorithm.bfs_predicates",
                    "failed to build reusable BFS include_edges alignment",
                    error,
                )
                .with_fact("predicate_role", "include_edges")
                .with_fact("input", input.input)
            })?;
        let probe = join.begin_left_semi_probe_in(edges, left_keys.iter().map(String::as_str), &overrides).map_err(
            |error| {
                cudf_query_error(
                    "query_engine.graph_algorithm.bfs_predicates",
                    "failed to probe BFS include_edges alignment",
                    error,
                )
                .with_fact("predicate_role", "include_edges")
                .with_fact("input", input.input)
            },
        )?;
        probe.materialize_indices_in(&overrides).map_err(|error| {
            cudf_query_error(
                "query_engine.graph_algorithm.bfs_predicates",
                "failed to materialize BFS include_edges alignment indices",
                error,
            )
            .with_fact("predicate_role", "include_edges")
            .with_fact("input", input.input)
        })?
    };
    let output_columns = edges.column_names().into_iter().map(str::to_string).collect::<Vec<_>>();
    let overrides_lease = execution_overrides(context)?;
    let overrides = overrides_lease.overrides();
    *edges = edges
        .with_execution(&overrides)
        .materialize_join_indices(&indices, output_columns.iter().map(String::as_str))
        .map_err(|error| {
            cudf_query_error(
                "query_engine.graph_algorithm.bfs_predicates",
                "failed to align BFS include_edges predicate with the primary edge list",
                error,
            )
            .with_fact("predicate_role", "include_edges")
            .with_fact("input", input.input)
            .with_fact("left_keys", left_keys.join(","))
            .with_fact("right_keys", right_keys.join(","))
        })?;

    Ok(EdgePredicateAlignment {
        input_rows: Some(input_rows),
        aligned_rows: Some(edges.num_rows()),
        null_rows_dropped: Some(null_rows_dropped),
    })
}

/// Projects the include_edges key columns and drops any row containing a null in
/// a key column, returning the cleaned whitelist frame and the dropped-row count.
///
/// Mirrors [`super::edge_null_filter::filter_null_edge_rows`]: the reject-vs-drop
/// axis is the actual `null_count`, not schema nullability.
fn drop_null_include_edge_rows(
    context: &ExecutionContext,
    input: NodeId,
    frame: &DataFrame,
    right_keys: &[String],
) -> Result<(DataFrame, usize)> {
    let allowed_edges = frame.select(right_keys.iter().map(String::as_str)).map_err(|error| {
        cudf_query_error(
            "query_engine.graph_algorithm.bfs_predicates",
            "failed to project BFS include_edges predicate columns",
            error,
        )
        .with_fact("predicate_role", "include_edges")
        .with_fact("input", input)
    })?;

    let mut has_nulls = false;
    for column in right_keys {
        let series = predicate_series_for_column(&allowed_edges, input, column, "include_edges")?;
        if series.null_count() != 0 {
            has_nulls = true;
            break;
        }
    }
    if !has_nulls {
        return Ok((allowed_edges, 0));
    }

    let input_rows = allowed_edges.num_rows();
    let predicates = right_keys.iter().map(|column| col(column.as_str()).is_not_null()).collect::<Vec<_>>();
    let overrides_lease = execution_overrides(context)?;
    let overrides = overrides_lease.overrides();
    let filtered = allowed_edges.with_execution(&overrides).filter_all(predicates).map_err(|error| {
        cudf_query_error(
            "query_engine.graph_algorithm.bfs_predicates",
            "failed to drop null BFS include_edges predicate rows",
            error,
        )
        .with_fact("predicate_role", "include_edges")
        .with_fact("input", input)
    })?;
    let null_rows_dropped = input_rows.saturating_sub(filtered.num_rows());
    Ok((filtered, null_rows_dropped))
}

#[derive(Debug, Clone, Copy)]
struct BfsEndpointMembership {
    source_rows: usize,
    valid_source_rows: usize,
    target_rows: usize,
    valid_target_rows: usize,
}

fn empty_bfs_result_for_missing_path_endpoint(
    runtime: &cudf::Runtime,
    spec: &GraphAlgorithmSpec,
    params: &GraphBfsParams,
    edges: &DataFrame,
    vertex_type: cudf::TypeId,
    predicate_columns: &BfsPredicateColumns,
) -> Result<Option<GraphAlgorithmExecutionResult>> {
    if params.endpoint_policy != GraphBfsEndpointPolicy::EmptyOnMissingPathEndpoint {
        return Ok(None);
    }

    let source_vertices = concrete_bfs_source_vertices(runtime, params, predicate_columns, vertex_type)?;
    let target_vertices = predicate_columns.target_vertices.as_ref().ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.graph_algorithm.bfs_endpoint_membership",
            "BFS empty endpoint policy requires target_vertices",
        )
    })?;
    if target_vertices.len() != 1 {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.graph_algorithm.bfs_endpoint_membership",
            "BFS empty endpoint policy requires exactly one target vertex",
        )
        .with_fact("target_rows", target_vertices.len()));
    }

    if source_vertices.is_empty() {
        let membership = BfsEndpointMembership {
            source_rows: 0,
            valid_source_rows: 0,
            target_rows: target_vertices.len(),
            valid_target_rows: 0,
        };
        return empty_bfs_result(runtime, spec, params, predicate_columns, "source_vertices_empty", membership)
            .map(Some);
    }

    if edges.num_rows() == 0 {
        let membership = BfsEndpointMembership {
            source_rows: source_vertices.len(),
            valid_source_rows: 0,
            target_rows: target_vertices.len(),
            valid_target_rows: 0,
        };
        return empty_bfs_result(runtime, spec, params, predicate_columns, "edge_source_empty", membership).map(Some);
    }

    let domain = graph_vertex_domain_frame(runtime, edges, spec)?;
    let source_membership = valid_vertex_rows(runtime, &source_vertices, &domain, "source_vertices")?;
    let target_membership = valid_vertex_rows(runtime, target_vertices, &domain, "target_vertices")?;
    let membership = BfsEndpointMembership {
        source_rows: source_vertices.len(),
        valid_source_rows: source_membership,
        target_rows: target_vertices.len(),
        valid_target_rows: target_membership,
    };

    if source_membership == 0 {
        return empty_bfs_result(runtime, spec, params, predicate_columns, "source_vertices_absent", membership)
            .map(Some);
    }
    if source_membership < source_vertices.len() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.graph_algorithm.bfs_endpoint_membership",
            "BFS source vertices contain values absent from the graph after edge-source filters",
        )
        .with_fact("reason", "partial_source_vertices_absent")
        .with_fact("predicate_role", "source_vertices")
        .with_fact("source_rows", source_vertices.len())
        .with_fact("valid_source_rows", source_membership)
        .with_fact("target_rows", target_vertices.len())
        .with_fact("valid_target_rows", target_membership)
        .with_fact("endpoint_policy", params.endpoint_policy.as_str()));
    }
    if target_membership == 0 {
        return empty_bfs_result(runtime, spec, params, predicate_columns, "target_vertices_absent", membership)
            .map(Some);
    }

    Ok(None)
}

fn empty_bfs_result_for_trivial_path_endpoint(
    context: &ExecutionContext,
    spec: &GraphAlgorithmSpec,
    params: &GraphBfsParams,
    edges: &DataFrame,
    predicate_columns: &BfsPredicateColumns,
) -> Result<Option<GraphAlgorithmExecutionResult>> {
    if params.endpoint_policy != GraphBfsEndpointPolicy::EmptyOnMissingPathEndpoint {
        return Ok(None);
    }

    let target_vertices = predicate_columns.target_vertices.as_ref().ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.graph_algorithm.bfs_endpoint_membership",
            "BFS empty endpoint policy requires target_vertices",
        )
    })?;
    if target_vertices.len() != 1 {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.graph_algorithm.bfs_endpoint_membership",
            "BFS empty endpoint policy requires exactly one target vertex",
        )
        .with_fact("target_rows", target_vertices.len()));
    }
    let runtime = context.runtime()?;
    validate_bfs_non_endpoint_predicate_membership(&runtime, edges, spec, predicate_columns)?;

    let source_rows = predicate_columns.source_vertices.as_ref().map_or(1, Series::len);
    if source_rows == 0 {
        let valid_target_rows = if edges.num_rows() == 0 {
            0
        } else {
            let domain = graph_vertex_domain_frame(&runtime, edges, spec)?;
            valid_vertex_rows(&runtime, target_vertices, &domain, "target_vertices")?
        };
        let membership = BfsEndpointMembership {
            source_rows: 0,
            valid_source_rows: 0,
            target_rows: target_vertices.len(),
            valid_target_rows,
        };
        return empty_bfs_result(&runtime, spec, params, predicate_columns, "source_vertices_empty", membership)
            .map(Some);
    }

    if edges.num_rows() == 0 {
        let membership = BfsEndpointMembership {
            source_rows,
            valid_source_rows: 0,
            target_rows: target_vertices.len(),
            valid_target_rows: 0,
        };
        return empty_bfs_result(&runtime, spec, params, predicate_columns, "edge_source_empty", membership).map(Some);
    }

    Ok(None)
}

pub(super) fn resolve_empty_bfs_edge_input(
    context: &ExecutionContext,
    spec: &GraphAlgorithmSpec,
    sanitized_edges: &DataFrame,
    side_inputs: &std::collections::BTreeMap<NodeId, GpuDataFrame>,
) -> Result<GraphAlgorithmExecutionResult> {
    let params = empty_bfs_params(spec)?;
    let (edges, predicate_columns) = project_bfs_inputs(context, spec, params, sanitized_edges, side_inputs)?;
    debug_assert_eq!(edges.num_rows(), 0, "empty-edge policy must run after all BFS edge filters");

    if let Some(result) = empty_bfs_result_for_trivial_path_endpoint(context, spec, params, &edges, &predicate_columns)?
    {
        return Ok(result);
    }
    let runtime = context.runtime()?;
    validate_bfs_non_endpoint_predicate_membership(&runtime, &edges, spec, &predicate_columns)?;
    Err(QueryError::new(
        QueryErrorCode::GraphInputInvalid,
        "query_engine.graph_algorithm.empty_input",
        "BFS requires a graph containing its source vertex",
    )
    .with_fact("algorithm", spec.algorithm.as_str())
    .with_fact("input_role", "edge")
    .with_fact("reason", "bfs_source_requires_nonempty_graph")
    .with_fact("input_rows", sanitized_edges.num_rows())
    .with_fact("sanitized_input_rows", edges.num_rows())
    .with_fact("empty_input_policy", spec.algorithm.definition().empty_input_policy().edge_input.as_str())
    .with_fact("source_rows", predicate_columns.source_vertices.as_ref().map_or(1, Series::len)))
}

pub(super) fn resolve_empty_bfs_source_input(
    context: &ExecutionContext,
    spec: &GraphAlgorithmSpec,
    sanitized_edges: &DataFrame,
    side_inputs: &std::collections::BTreeMap<NodeId, GpuDataFrame>,
) -> Result<Option<GraphAlgorithmExecutionResult>> {
    let params = empty_bfs_params(spec)?;
    let (edges, predicate_columns) = project_bfs_inputs(context, spec, params, sanitized_edges, side_inputs)?;
    if !predicate_columns.source_vertices.as_ref().is_some_and(Series::is_empty) {
        return Ok(None);
    }
    empty_bfs_result_for_trivial_path_endpoint(context, spec, params, &edges, &predicate_columns)
}

fn empty_bfs_params(spec: &GraphAlgorithmSpec) -> Result<&GraphBfsParams> {
    let GraphAlgorithmParams::Bfs(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.graph_algorithm.empty_input",
            "BFS empty-input policy was attached to incompatible parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };
    Ok(params)
}

fn project_bfs_inputs(
    context: &ExecutionContext,
    spec: &GraphAlgorithmSpec,
    params: &GraphBfsParams,
    sanitized_edges: &DataFrame,
    side_inputs: &std::collections::BTreeMap<NodeId, GpuDataFrame>,
) -> Result<(DataFrame, BfsPredicateColumns)> {
    let (mut edges, vertex_type) = project_edge_list(sanitized_edges, spec)?;
    let predicate_columns = BfsPredicateColumns::from_side_inputs(
        context,
        params,
        side_inputs,
        vertex_type,
        spec.edge_columns.edge_id.is_some(),
        spec,
        &mut edges,
    )?;
    Ok((edges, predicate_columns))
}

#[expect(clippy::too_many_arguments, reason = "graph algorithm admission inputs are distinct proof values")]
fn graph_bfs_error_for_missing_path_endpoint(
    runtime: &cudf::Runtime,
    spec: &GraphAlgorithmSpec,
    params: &GraphBfsParams,
    edges: &DataFrame,
    vertex_type: cudf::TypeId,
    predicate_columns: &BfsPredicateColumns,
    error: cugraph::Error,
    message: &'static str,
) -> Result<GraphAlgorithmExecutionResult> {
    if error.kind() == ErrorKind::InvalidInput {
        validate_bfs_non_endpoint_predicate_membership(runtime, edges, spec, predicate_columns)?;
        if let Some(result) =
            empty_bfs_result_for_missing_path_endpoint(runtime, spec, params, edges, vertex_type, predicate_columns)?
        {
            return Ok(result);
        }
    }
    Err(map_cugraph_error("query_engine.graph_algorithm.bfs", message)(error))
}

fn validate_bfs_non_endpoint_predicate_membership(
    runtime: &cudf::Runtime,
    edges: &DataFrame,
    spec: &GraphAlgorithmSpec,
    predicate_columns: &BfsPredicateColumns,
) -> Result<()> {
    if edges.num_rows() == 0 {
        if let Some(vertices) = &predicate_columns.include_vertices
            && !vertices.is_empty()
        {
            return Err(predicate_membership_error("include_vertices", vertices.len(), 0));
        }
        if let Some(vertices) = &predicate_columns.exclude_vertices
            && !vertices.is_empty()
        {
            return Err(predicate_membership_error("exclude_vertices", vertices.len(), 0));
        }
        if let Some(edge_ids) = &predicate_columns.include_edge_ids
            && !edge_ids.is_empty()
        {
            return Err(predicate_membership_error("include_edge_ids", edge_ids.len(), 0));
        }
        if let Some(input_rows) = predicate_columns.include_edges_rows
            && input_rows > 0
        {
            return Err(QueryError::new(
                QueryErrorCode::GraphInputInvalid,
                "query_engine.graph_algorithm.bfs_endpoint_membership",
                "BFS include_edges predicate cannot match an empty graph after edge-source filters",
            )
            .with_fact("reason", "predicate_edges_absent")
            .with_fact("predicate_role", "include_edges")
            .with_fact("predicate_rows", input_rows)
            .with_fact("valid_predicate_rows", 0));
        }
        return Ok(());
    }
    if predicate_columns.include_vertices.is_some() || predicate_columns.exclude_vertices.is_some() {
        let domain = graph_vertex_domain_frame(runtime, edges, spec)?;
        if let Some(vertices) = &predicate_columns.include_vertices {
            validate_predicate_membership(runtime, vertices, &domain, "include_vertices")?;
        }
        if let Some(vertices) = &predicate_columns.exclude_vertices {
            validate_predicate_membership(runtime, vertices, &domain, "exclude_vertices")?;
        }
    }
    if let Some(edge_ids) = &predicate_columns.include_edge_ids {
        let edge_id_col = spec.edge_columns.edge_id.as_ref().ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.graph_algorithm.bfs_endpoint_membership",
                "BFS include_edge_ids membership validation requires an edge-id column binding",
            )
            .with_fact("predicate_role", "include_edge_ids")
        })?;
        let edge_id_domain = graph_edge_id_domain_frame(edges, edge_id_col)?;
        validate_predicate_membership(runtime, edge_ids, &edge_id_domain, "include_edge_ids")?;
    }
    Ok(())
}

fn validate_predicate_membership(
    runtime: &cudf::Runtime,
    vertices: &cudf::Series,
    domain: &DataFrame,
    role: &'static str,
) -> Result<()> {
    let valid_rows = valid_vertex_rows(runtime, vertices, domain, role)?;
    if valid_rows == vertices.len() {
        return Ok(());
    }
    Err(predicate_membership_error(role, vertices.len(), valid_rows))
}

fn predicate_membership_error(role: &'static str, predicate_rows: usize, valid_rows: usize) -> QueryError {
    QueryError::new(
        QueryErrorCode::GraphInputInvalid,
        "query_engine.graph_algorithm.bfs_endpoint_membership",
        "BFS predicate contains values absent from the graph after edge-source filters",
    )
    .with_fact("reason", "predicate_vertices_absent")
    .with_fact("predicate_role", role)
    .with_fact("predicate_rows", predicate_rows)
    .with_fact("valid_predicate_rows", valid_rows)
}

fn concrete_bfs_source_vertices(
    runtime: &cudf::Runtime,
    params: &GraphBfsParams,
    predicate_columns: &BfsPredicateColumns,
    vertex_type: cudf::TypeId,
) -> Result<cudf::Series> {
    if let Some(source_vertices) = &predicate_columns.source_vertices {
        return Ok(source_vertices.clone());
    }
    match vertex_type {
        TypeId::Int32 => {
            let source = i32::try_from(params.source_vertex).map_err(|_| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "query_engine.graph_algorithm.bfs_source",
                    "BFS source vertex does not fit the Int32 graph vertex dtype",
                )
                .with_fact("source_vertex", params.source_vertex)
                .with_fact("vertex_dtype", "Int32")
            })?;
            scalar_series(runtime, source, "source_vertices")
        },
        TypeId::Int64 => scalar_series(runtime, params.source_vertex, "source_vertices"),
        _ => Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "query_engine.graph_algorithm.bfs_source",
            "BFS graph vertex dtype is unsupported",
        )
        .with_fact("vertex_dtype", format!("{vertex_type:?}"))),
    }
}

fn scalar_series<T>(runtime: &cudf::Runtime, value: T, role: &'static str) -> Result<cudf::Series>
where T: cudf::CudfType {
    runtime.scalar(value).and_then(|scalar| runtime.series_from_scalar(&scalar, 1)).map_err(|error| {
        cudf_query_error(
            "query_engine.graph_algorithm.bfs_endpoint_membership",
            "failed to materialize BFS endpoint scalar",
            error,
        )
        .with_fact("predicate_role", role)
    })
}

fn graph_vertex_domain_frame(
    runtime: &cudf::Runtime,
    edges: &DataFrame,
    spec: &GraphAlgorithmSpec,
) -> Result<DataFrame> {
    let source = graph_edge_series_for_column(edges, &spec.edge_columns.source, "graph_source_domain")?;
    let destination = graph_edge_series_for_column(edges, &spec.edge_columns.destination, "graph_destination_domain")?;
    let overrides =
        ExecutionOverrides::new().with_stream(runtime.stream()).with_memory_resource(runtime.memory_resource());
    let domain = Series::concat_in(&[&source, &destination], &overrides).map_err(|error| {
        cudf_query_error(
            "query_engine.graph_algorithm.bfs_endpoint_membership",
            "failed to build BFS graph vertex domain",
            error,
        )
    })?;
    DataFrame::from_series([("__vertex", domain)]).map_err(|error| {
        cudf_query_error(
            "query_engine.graph_algorithm.bfs_endpoint_membership",
            "failed to bind BFS graph vertex domain",
            error,
        )
    })
}

fn graph_edge_series_for_column(frame: &DataFrame, column: &str, role: &'static str) -> Result<cudf::Series> {
    frame.series(column).map_err(|error| {
        cudf_query_error(
            "query_engine.graph_algorithm.bfs_endpoint_membership",
            "failed to access BFS graph domain column",
            error,
        )
        .with_fact("predicate_role", role)
        .with_fact("column", column)
    })
}

fn graph_edge_id_domain_frame(edges: &DataFrame, edge_id_column: &str) -> Result<DataFrame> {
    let edge_ids = graph_edge_series_for_column(edges, edge_id_column, "include_edge_ids_domain")?;
    DataFrame::from_series([("__vertex", edge_ids)]).map_err(|error| {
        cudf_query_error(
            "query_engine.graph_algorithm.bfs_endpoint_membership",
            "failed to bind BFS edge-id predicate domain",
            error,
        )
        .with_fact("predicate_role", "include_edge_ids")
    })
}

fn valid_vertex_rows(
    runtime: &cudf::Runtime,
    vertices: &cudf::Series,
    domain: &DataFrame,
    role: &'static str,
) -> Result<usize> {
    let vertices = DataFrame::from_series([("__vertex", vertices.clone())]).map_err(|error| {
        cudf_query_error(
            "query_engine.graph_algorithm.bfs_endpoint_membership",
            "failed to bind BFS endpoint vertex set",
            error,
        )
        .with_fact("predicate_role", role)
    })?;
    let options =
        DataFrameJoinOptions::new().left_on(["__vertex"]).right_on(["__vertex"]).null_equality(NullEquality::Unequal);
    let overrides =
        ExecutionOverrides::new().with_stream(runtime.stream()).with_memory_resource(runtime.memory_resource());
    vertices.with_execution(&overrides).left_semi_join(domain, options).map(|frame| frame.num_rows()).map_err(|error| {
        cudf_query_error(
            "query_engine.graph_algorithm.bfs_endpoint_membership",
            "failed to test BFS endpoint graph membership",
            error,
        )
        .with_fact("predicate_role", role)
    })
}

fn empty_bfs_result(
    runtime: &cudf::Runtime,
    spec: &GraphAlgorithmSpec,
    params: &GraphBfsParams,
    predicate_columns: &BfsPredicateColumns,
    reason: &'static str,
    membership: BfsEndpointMembership,
) -> Result<GraphAlgorithmExecutionResult> {
    let frame = empty_graph_result_frame(runtime, &spec.output_schema)?;
    Ok(GraphAlgorithmExecutionResult {
        frame,
        algorithm_metadata: format!(
            "source_vertex={},depth_limit={},compute_predecessors=true,{},target_found=false,target_distance=none,endpoint_policy={},empty_result_reason={},source_rows={},valid_source_rows={},target_rows={},valid_target_rows={}",
            params.source_vertex,
            params.depth_limit.map_or_else(|| "none".to_string(), |value| value.to_string()),
            predicate_columns.metadata(),
            params.endpoint_policy.as_str(),
            reason,
            membership.source_rows,
            membership.valid_source_rows,
            membership.target_rows,
            membership.valid_target_rows,
        ),
        cache_scope_peak_bytes: None,
    })
}

fn predicate_series(
    side_inputs: &std::collections::BTreeMap<NodeId, GpuDataFrame>,
    input: Option<&GraphPredicateInput>,
    role: &'static str,
    vertex_type: cudf::TypeId,
) -> Result<Option<cudf::Series>> {
    let Some(input) = input else {
        return Ok(None);
    };
    let frame = side_inputs.get(&input.input).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.graph_algorithm.bfs_predicates",
            "BFS predicate side input was not executed",
        )
        .with_fact("predicate_role", role)
        .with_fact("input", input.input)
        .with_fact("column", &input.column)
    })?;
    let frame = frame.clone().into_frame();
    let series = validate_predicate_series(&frame, input.input, &input.column, role, vertex_type)?;
    Ok(Some(series))
}

fn source_vertex_series(
    side_inputs: &std::collections::BTreeMap<NodeId, GpuDataFrame>,
    input: Option<&GraphPredicateInput>,
    vertex_type: cudf::TypeId,
    endpoint_policy: GraphBfsEndpointPolicy,
) -> Result<Option<cudf::Series>> {
    let Some(input) = input else {
        return Ok(None);
    };
    let series =
        predicate_series(side_inputs, Some(input), "source_vertices", vertex_type)?.expect("input was Some above");
    if series.is_empty() && endpoint_policy != GraphBfsEndpointPolicy::EmptyOnMissingPathEndpoint {
        return Err(QueryError::new(
            QueryErrorCode::GraphInputInvalid,
            "query_engine.graph_algorithm.bfs_source",
            "BFS source-vertices side input must contain at least one row",
        )
        .with_fact("input_role", "source_vertices")
        .with_fact("side_input_node", input.input)
        .with_fact("column", &input.column)
        .with_fact("side_input_rows", series.len())
        .with_fact("reason", "required_side_input_empty"));
    }
    Ok(Some(series))
}

/// Validates a vertex-set predicate column (`include_vertices`,
/// `exclude_vertices`, `target_vertices`, `include_edge_ids`, `source_vertices`).
///
/// These predicates are set constraints the user explicitly enumerated, so a
/// null id is an unknown set member and is rejected on the actual *values*
/// (`null_count() > 0`), never on schema nullability. Schema nullability is
/// accepted at planning time; the reject decision happens here, where the data
/// is materialized.
fn validate_predicate_series(
    frame: &DataFrame,
    input: NodeId,
    column: &str,
    role: &'static str,
    vertex_type: cudf::TypeId,
) -> Result<cudf::Series> {
    let series = predicate_series_for_column(frame, input, column, role)?;
    if series.null_count() != 0 {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.graph_algorithm.bfs_predicates",
            "BFS vertex-set predicate column contains null values",
        )
        .with_fact("reason", "predicate_column_contains_nulls")
        .with_fact("predicate_role", role)
        .with_fact("input", input)
        .with_fact("column", column)
        .with_fact("null_count", series.null_count()));
    }
    validate_predicate_series_dtype(&series, input, column, role, vertex_type)?;
    Ok(series)
}

fn predicate_series_for_column(
    frame: &DataFrame,
    input: NodeId,
    column: &str,
    role: &'static str,
) -> Result<cudf::Series> {
    frame.series(column).map_err(|error| {
        cudf_query_error("query_engine.graph_algorithm.bfs_predicates", "failed to access BFS predicate column", error)
            .with_fact("predicate_role", role)
            .with_fact("input", input)
            .with_fact("column", column)
    })
}

fn validate_predicate_series_dtype(
    series: &cudf::Series,
    input: NodeId,
    column: &str,
    role: &'static str,
    vertex_type: cudf::TypeId,
) -> Result<()> {
    let dtype = series.dtype().map_err(|error| {
        cudf_query_error(
            "query_engine.graph_algorithm.bfs_predicates",
            "failed to inspect BFS predicate column type",
            error,
        )
        .with_fact("predicate_role", role)
        .with_fact("input", input)
        .with_fact("column", column)
    })?;
    if dtype.id() != vertex_type {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "query_engine.graph_algorithm.bfs_predicates",
            "BFS predicate id column dtype must match the graph vertex dtype",
        )
        .with_fact("predicate_role", role)
        .with_fact("input", input)
        .with_fact("column", column)
        .with_fact("predicate_dtype", format!("{:?}", dtype.id()))
        .with_fact("vertex_dtype", format!("{vertex_type:?}")));
    }
    Ok(())
}

fn predicate_column_slice<'a, T: cudf::CudfType>(
    series: Option<&'a cudf::Series>,
    role: &'static str,
) -> Result<Option<DeviceSlice<'a>>> {
    series
        .map(|series| {
            series.as_column().device_slice::<T>().map_err(|error| {
                cudf_query_error(
                    "query_engine.graph_algorithm.bfs_predicates",
                    "failed to borrow BFS predicate column device slice",
                    error,
                )
                .with_fact("predicate_role", role)
            })
        })
        .transpose()
}

fn predicate_device_view<'a, T: CugraphElement>(
    slice: Option<&'a DeviceSlice<'a>>,
    role: &'static str,
) -> Result<Option<cugraph::DeviceArrayView<'a, T>>> {
    slice
        .map(|slice| {
            device_array_view_from_slice::<T>(slice).map_err(map_cugraph_error(
                "query_engine.graph_algorithm.bfs_predicates",
                "failed to wrap BFS predicate column as a cuGraph device-array view",
            ))
        })
        .transpose()
        .map_err(|error| error.with_fact("predicate_role", role))
}

fn optional_series_len(series: Option<&cudf::Series>) -> String {
    series.map_or_else(|| "none".to_string(), |series| series.len().to_string())
}

fn optional_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "none".to_string(), |value| value.to_string())
}

fn bfs_target_metadata(target_info: Option<&BfsTargetInfo>) -> String {
    match target_info {
        Some(info) => format!(
            "target_found={},target_distance={}",
            info.target_found,
            info.target_distance.map_or_else(|| "none".to_string(), |distance| distance.to_string())
        ),
        None => "target_found=not_requested,target_distance=none".to_string(),
    }
}

fn append_bfs_target_metadata_columns(
    runtime: &cudf::Runtime,
    params: &GraphBfsParams,
    frame: DataFrame,
    target_info: Option<BfsTargetInfo>,
) -> Result<DataFrame> {
    if !params.return_target_info {
        return Ok(frame);
    }
    let frame = frame.copy_into_runtime(runtime).map_err(|error| {
        cudf_query_error(
            "query_engine.graph_algorithm.bfs_target_metadata",
            "failed to place BFS result in metadata runtime",
            error,
        )
    })?;
    let info = target_info.ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.graph_algorithm.bfs_target_metadata",
            "BFS target metadata was requested but cuGraph returned no target metadata",
        )
    })?;
    let row_count = frame.num_rows();
    let target_found = runtime
        .scalar(info.target_found)
        .and_then(|scalar| runtime.series_from_scalar(&scalar, row_count))
        .map_err(|error| {
            cudf_query_error(
                "query_engine.graph_algorithm.bfs_target_metadata",
                "failed to materialize BFS target_found metadata column",
                error,
            )
        })?;
    let target_distance = match info.target_distance {
        Some(distance) => runtime.scalar(distance).and_then(|scalar| runtime.series_from_scalar(&scalar, row_count)),
        None => runtime.nulls(DataType::new(TypeId::Int64), row_count),
    }
    .map_err(|error| {
        cudf_query_error(
            "query_engine.graph_algorithm.bfs_target_metadata",
            "failed to materialize BFS target_distance metadata column",
            error,
        )
    })?;
    frame
        .with_column("target_found", target_found)
        .and_then(|frame| frame.with_column("target_distance", target_distance))
        .map_err(|error| {
            cudf_query_error(
                "query_engine.graph_algorithm.bfs_target_metadata",
                "failed to append BFS target metadata columns",
                error,
            )
        })
}

pub(super) fn execute_sssp(
    context: &ExecutionContext,
    spec: &GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult> {
    let input = inputs.edge_list;
    let GraphAlgorithmParams::Sssp(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.graph_algorithm.sssp",
            "SSSP graph algorithm node carried non-SSSP parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, vertex_type) = project_edge_list(&input, spec)?;
    let source_vertex = sssp_source_vertex(&params.source_vertex, &inputs.side_inputs, vertex_type)?;
    let options = table_graph_options(spec);
    let sssp_p = sssp_params(params)?;

    run_with_cugraph_graph::<cugraph::Forward, _>(
        context,
        edges,
        options,
        "algeon_datafusion_cugraph_sssp",
        |runtime, entered, graph| {
            let frame = match vertex_type {
                TypeId::Int32 => {
                    let source = i32::try_from(source_vertex).map_err(|_| {
                        QueryError::new(
                            QueryErrorCode::InvalidPlan,
                            "query_engine.graph_algorithm.sssp_source",
                            "SSSP source vertex does not fit the Int32 graph vertex dtype",
                        )
                        .with_fact("source_vertex", source_vertex)
                        .with_fact("vertex_dtype", "Int32")
                    })?;
                    let result = graph
                        .sssp(entered, source, &sssp_p)
                        .map_err(map_cugraph_error("query_engine.graph_algorithm.sssp", "cuGraph SSSP failed"))?;
                    result.to_dataframe(runtime).map_err(map_cugraph_error(
                        "query_engine.graph_algorithm.sssp_result",
                        "failed to convert cuGraph SSSP result to cudf DataFrame",
                    ))?
                },
                TypeId::Int64 => {
                    let result = graph
                        .sssp(entered, source_vertex, &sssp_p)
                        .map_err(map_cugraph_error("query_engine.graph_algorithm.sssp", "cuGraph SSSP failed"))?;
                    result.to_dataframe(runtime).map_err(map_cugraph_error(
                        "query_engine.graph_algorithm.sssp_result",
                        "failed to convert cuGraph SSSP result to cudf DataFrame",
                    ))?
                },
                _ => {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedType,
                        "query_engine.graph_algorithm.sssp_source",
                        "SSSP graph vertex dtype is unsupported",
                    )
                    .with_fact("vertex_dtype", format!("{vertex_type:?}")));
                },
            };
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata: format!(
                    "source_vertex={},cutoff={}",
                    source_vertex,
                    params.cutoff.map_or_else(|| "none".to_string(), |v| v.to_string()),
                ),
                cache_scope_peak_bytes: None,
            })
        },
    )
}

fn sssp_source_vertex(
    source: &GraphSsspSource,
    side_inputs: &std::collections::BTreeMap<NodeId, GpuDataFrame>,
    vertex_type: cudf::TypeId,
) -> Result<i64> {
    let GraphSsspSource::Input(input) = source else {
        let GraphSsspSource::I64(value) = source else { unreachable!() };
        return Ok(*value);
    };
    let frame = side_inputs.get(&input.input).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.graph_algorithm.sssp_source",
            "SSSP source side input was not executed",
        )
        .with_fact("side_input_node", input.input)
        .with_fact("column", &input.column)
    })?;
    let series = frame.frame().series(&input.column).map_err(|error| {
        cudf_query_error(
            "query_engine.graph_algorithm.sssp_source",
            "failed to resolve the SSSP source side-input column",
            error,
        )
        .with_fact("side_input_node", input.input)
        .with_fact("column", &input.column)
    })?;
    if series.len() != 1 || series.null_count() != 0 {
        return Err(QueryError::new(
            QueryErrorCode::GraphInputInvalid,
            "query_engine.graph_algorithm.sssp_source",
            "SSSP source side input must contain exactly one non-null row",
        )
        .with_fact("side_input_rows", series.len())
        .with_fact("side_input_null_rows", series.null_count()));
    }
    match vertex_type {
        TypeId::Int32 => series.to_vec_strict::<i32>().map(|values| i64::from(values[0])).map_err(|error| {
            cudf_query_error(
                "query_engine.graph_algorithm.sssp_source",
                "failed to read the encoded Int32 SSSP source",
                error,
            )
        }),
        TypeId::Int64 => series.to_vec_strict::<i64>().map(|values| values[0]).map_err(|error| {
            cudf_query_error(
                "query_engine.graph_algorithm.sssp_source",
                "failed to read the encoded Int64 SSSP source",
                error,
            )
        }),
        dtype => Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "query_engine.graph_algorithm.sssp_source",
            "SSSP source side input requires an Int32 or Int64 graph vertex dtype",
        )
        .with_fact("vertex_dtype", format!("{dtype:?}"))),
    }
}
