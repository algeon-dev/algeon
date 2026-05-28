use super::*;

pub(super) fn execute_bfs(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult> {
    let GraphAlgorithmExecutionInputs { edge_list: input, side_inputs } = inputs;
    let crate::plan::GraphAlgorithmParams::Bfs(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.graph_algorithm.bfs",
            "BFS graph algorithm node carried non-BFS parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (mut edges, vertex_type) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);
    let bfs_p = bfs_params(params)?;
    let predicate_columns = BfsPredicateColumns::from_side_inputs(
        params,
        &side_inputs,
        vertex_type,
        spec.edge_columns.edge_id.is_some(),
        spec,
        &mut edges,
    )?;

    let run_bfs = |runtime: &cudf::Runtime,
                   entered: &cugraph::EnteredContext<'_>,
                   graph: &mut cugraph::Graph<cugraph::Forward>| {
        let (frame, target_metadata) =
            match vertex_type {
                cudf::TypeId::Int32 => {
                    let source = i32::try_from(params.source_vertex).map_err(|_| {
                        QueryError::new(
                            QueryErrorCode::InvalidPlan,
                            "cudf_query_engine.graph_algorithm.bfs_source",
                            "BFS source vertex does not fit the Int32 graph vertex dtype",
                        )
                        .with_fact("source_vertex", params.source_vertex)
                        .with_fact("vertex_dtype", "Int32")
                    })?;
                    let source_slice =
                        predicate_column_slice::<i32>(predicate_columns.source_vertices.as_ref(), "source_vertices")?;
                    let source_array = if source_slice.is_none() {
                        Some(cugraph::DeviceArray::from_slice(entered, &[source]).map_err(map_cugraph_error(
                            "cudf_query_engine.graph_algorithm.bfs_source",
                            "failed to copy BFS source vertex to device memory",
                        ))?)
                    } else {
                        None
                    };
                    let sources_view = match (source_slice.as_ref(), source_array.as_ref()) {
                        (Some(slice), None) => cugraph::cudf_interop::device_array_view_from_slice::<i32>(slice)
                            .map_err(map_cugraph_error(
                                "cudf_query_engine.graph_algorithm.bfs_source",
                                "failed to wrap BFS source-vertex side input as a cuGraph device-array view",
                            ))?,
                        (None, Some(array)) => array.view().map_err(map_cugraph_error(
                            "cudf_query_engine.graph_algorithm.bfs_source",
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
                    let predicates = cugraph::traversal::BfsPredicateParams::<i32, i32> {
                        include_vertices: include_vertices_view.as_ref(),
                        exclude_vertices: exclude_vertices_view.as_ref(),
                        target_vertices: target_vertices_view.as_ref(),
                        include_edge_ids: include_edge_ids_view.as_ref(),
                        stop_when_any_target_found: target_vertices_view.is_some(),
                    };
                    let result = if predicate_columns.requires_predicate_bfs_call() {
                        graph.bfs_with_predicates(entered, &sources_view, &bfs_p, &predicates).map_err(
                            map_cugraph_error("cudf_query_engine.graph_algorithm.bfs", "cuGraph predicate BFS failed"),
                        )?
                    } else {
                        graph
                            .bfs(entered, &sources_view, &bfs_p)
                            .map_err(map_cugraph_error("cudf_query_engine.graph_algorithm.bfs", "cuGraph BFS failed"))?
                    };
                    let target_info = result.target_info().copied();
                    let target_metadata = bfs_target_metadata(target_info.as_ref());
                    let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                        "cudf_query_engine.graph_algorithm.bfs_result",
                        "failed to convert cuGraph BFS result to cudf DataFrame",
                    ))?;
                    let frame = append_bfs_target_metadata_columns(context, params, frame, target_info)?;
                    (frame, target_metadata)
                },
                cudf::TypeId::Int64 => {
                    let source_slice =
                        predicate_column_slice::<i64>(predicate_columns.source_vertices.as_ref(), "source_vertices")?;
                    let source_array = if source_slice.is_none() {
                        Some(cugraph::DeviceArray::from_slice(entered, &[params.source_vertex]).map_err(
                            map_cugraph_error(
                                "cudf_query_engine.graph_algorithm.bfs_source",
                                "failed to copy BFS source vertex to device memory",
                            ),
                        )?)
                    } else {
                        None
                    };
                    let sources_view = match (source_slice.as_ref(), source_array.as_ref()) {
                        (Some(slice), None) => cugraph::cudf_interop::device_array_view_from_slice::<i64>(slice)
                            .map_err(map_cugraph_error(
                                "cudf_query_engine.graph_algorithm.bfs_source",
                                "failed to wrap BFS source-vertex side input as a cuGraph device-array view",
                            ))?,
                        (None, Some(array)) => array.view().map_err(map_cugraph_error(
                            "cudf_query_engine.graph_algorithm.bfs_source",
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
                    let predicates = cugraph::traversal::BfsPredicateParams::<i64, i64> {
                        include_vertices: include_vertices_view.as_ref(),
                        exclude_vertices: exclude_vertices_view.as_ref(),
                        target_vertices: target_vertices_view.as_ref(),
                        include_edge_ids: include_edge_ids_view.as_ref(),
                        stop_when_any_target_found: target_vertices_view.is_some(),
                    };
                    let result = if predicate_columns.requires_predicate_bfs_call() {
                        graph.bfs_with_predicates(entered, &sources_view, &bfs_p, &predicates).map_err(
                            map_cugraph_error("cudf_query_engine.graph_algorithm.bfs", "cuGraph predicate BFS failed"),
                        )?
                    } else {
                        graph
                            .bfs(entered, &sources_view, &bfs_p)
                            .map_err(map_cugraph_error("cudf_query_engine.graph_algorithm.bfs", "cuGraph BFS failed"))?
                    };
                    let target_info = result.target_info().copied();
                    let target_metadata = bfs_target_metadata(target_info.as_ref());
                    let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                        "cudf_query_engine.graph_algorithm.bfs_result",
                        "failed to convert cuGraph BFS result to cudf DataFrame",
                    ))?;
                    let frame = append_bfs_target_metadata_columns(context, params, frame, target_info)?;
                    (frame, target_metadata)
                },
                _ => {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedType,
                        "cudf_query_engine.graph_algorithm.bfs_source",
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
        })
    };
    if spec.edge_columns.edge_id.is_some() {
        run_with_cugraph_graph_edge_ids::<cugraph::Forward, _, _>(
            context,
            &edges,
            spec,
            "datafusion_cudf_cugraph_bfs",
            run_bfs,
        )
    } else {
        run_with_cugraph_graph::<cugraph::Forward, _, _>(
            context,
            edges,
            options,
            "datafusion_cudf_cugraph_bfs",
            run_bfs,
        )
    }
}

struct BfsPredicateColumns {
    source_vertices: Option<cudf::Series>,
    include_vertices: Option<cudf::Series>,
    exclude_vertices: Option<cudf::Series>,
    target_vertices: Option<cudf::Series>,
    include_edge_ids: Option<cudf::Series>,
    include_edges_rows: Option<usize>,
    include_edges_aligned_rows: Option<usize>,
}

impl BfsPredicateColumns {
    fn from_side_inputs(
        params: &crate::plan::GraphBfsParams,
        side_inputs: &std::collections::BTreeMap<NodeId, GpuDataFrame>,
        vertex_type: cudf::TypeId,
        graph_has_edge_id_binding: bool,
        spec: &crate::plan::GraphAlgorithmSpec,
        edges: &mut DataFrame,
    ) -> Result<Self> {
        if params.predicates.include_vertices.is_some() && params.predicates.exclude_vertices.is_some() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.graph_algorithm.bfs_predicates",
                "BFS include_vertices and exclude_vertices predicates are mutually exclusive",
            ));
        }
        if params.return_target_info && params.predicates.target_vertices.is_none() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.graph_algorithm.bfs_target_metadata",
                "BFS return_target_info requires a target_vertices predicate input",
            ));
        }
        if params.predicates.include_edge_ids.is_some() && !graph_has_edge_id_binding {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.graph_algorithm.bfs_predicates",
                "BFS include_edge_ids requires the primary edge list to declare a stable edge-id column binding",
            )
            .with_fact("predicate", "include_edge_ids")
            .with_fact("missing_binding", "edge_id_col"));
        }
        let edge_alignment =
            align_include_edges(edges, spec, params.predicates.include_edges.as_ref(), side_inputs, vertex_type)?;
        Ok(Self {
            source_vertices: source_vertex_series(side_inputs, params.source_vertices.as_ref(), vertex_type)?,
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
            "source_vertices_rows={},predicate_include_vertices_rows={},predicate_exclude_vertices_rows={},predicate_target_vertices_rows={},predicate_include_edge_ids_rows={},predicate_include_edges_rows={},predicate_include_edges_aligned_rows={},target_early_stop_supported={}",
            optional_series_len(self.source_vertices.as_ref()),
            optional_series_len(self.include_vertices.as_ref()),
            optional_series_len(self.exclude_vertices.as_ref()),
            optional_series_len(self.target_vertices.as_ref()),
            optional_series_len(self.include_edge_ids.as_ref()),
            optional_usize(self.include_edges_rows),
            optional_usize(self.include_edges_aligned_rows),
            self.target_vertices.is_some(),
        )
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct EdgePredicateAlignment {
    input_rows: Option<usize>,
    aligned_rows: Option<usize>,
}

fn align_include_edges(
    edges: &mut DataFrame,
    spec: &crate::plan::GraphAlgorithmSpec,
    input: Option<&crate::plan::GraphEdgePredicateInput>,
    side_inputs: &std::collections::BTreeMap<NodeId, GpuDataFrame>,
    vertex_type: cudf::TypeId,
) -> Result<EdgePredicateAlignment> {
    let Some(input) = input else {
        return Ok(EdgePredicateAlignment::default());
    };
    let frame = side_inputs.get(&input.input).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.graph_algorithm.bfs_predicates",
            "BFS include_edges side input was not executed",
        )
        .with_fact("predicate", "include_edges")
        .with_fact("input", input.input)
    })?;
    let frame = frame.clone().into_frame();
    validate_predicate_series(&frame, input.input, &input.source, "include_edges_source", vertex_type)?;
    validate_predicate_series(&frame, input.input, &input.destination, "include_edges_destination", vertex_type)?;

    let mut left_keys = vec![spec.edge_columns.source.clone(), spec.edge_columns.destination.clone()];
    let mut right_keys = vec![input.source.clone(), input.destination.clone()];
    if let Some(edge_id) = &input.edge_id {
        let primary_edge_id = spec.edge_columns.edge_id.as_ref().ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.graph_algorithm.bfs_predicates",
                "BFS include_edges edge-id alignment requires a primary edge-id column binding",
            )
            .with_fact("predicate", "include_edges")
            .with_fact("input", input.input)
            .with_fact("missing_binding", "edge_id_col")
        })?;
        validate_predicate_series(&frame, input.input, edge_id, "include_edges_edge_id", vertex_type)?;
        left_keys.push(primary_edge_id.clone());
        right_keys.push(edge_id.clone());
    }

    let allowed_edges = frame.select(right_keys.iter().map(String::as_str)).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.graph_algorithm.bfs_predicates",
            "failed to project BFS include_edges predicate columns",
            error,
        )
        .with_fact("predicate", "include_edges")
        .with_fact("input", input.input)
    })?;
    let options = cudf::DataFrameJoinOptions::new()
        .left_on(left_keys.iter().map(String::as_str))
        .right_on(right_keys.iter().map(String::as_str))
        .null_equality(cudf::NullEquality::Unequal);
    *edges = edges.left_semi_join(&allowed_edges, options).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.graph_algorithm.bfs_predicates",
            "failed to align BFS include_edges predicate with the primary edge list",
            error,
        )
        .with_fact("predicate", "include_edges")
        .with_fact("input", input.input)
        .with_fact("left_keys", left_keys.join(","))
        .with_fact("right_keys", right_keys.join(","))
    })?;
    Ok(EdgePredicateAlignment { input_rows: Some(frame.num_rows()), aligned_rows: Some(edges.num_rows()) })
}

fn predicate_series(
    side_inputs: &std::collections::BTreeMap<NodeId, GpuDataFrame>,
    input: Option<&crate::plan::GraphPredicateInput>,
    role: &'static str,
    vertex_type: cudf::TypeId,
) -> Result<Option<cudf::Series>> {
    let Some(input) = input else {
        return Ok(None);
    };
    let frame = side_inputs.get(&input.input).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.graph_algorithm.bfs_predicates",
            "BFS predicate side input was not executed",
        )
        .with_fact("predicate", role)
        .with_fact("input", input.input)
        .with_fact("column", &input.column)
    })?;
    let frame = frame.clone().into_frame();
    let series = validate_predicate_series(&frame, input.input, &input.column, role, vertex_type)?;
    Ok(Some(series))
}

fn source_vertex_series(
    side_inputs: &std::collections::BTreeMap<NodeId, GpuDataFrame>,
    input: Option<&crate::plan::GraphPredicateInput>,
    vertex_type: cudf::TypeId,
) -> Result<Option<cudf::Series>> {
    let Some(input) = input else {
        return Ok(None);
    };
    let series =
        predicate_series(side_inputs, Some(input), "source_vertices", vertex_type)?.expect("input was Some above");
    if series.len() != 1 {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.graph_algorithm.bfs_source",
            "BFS source-vertex side input must contain exactly one row",
        )
        .with_fact("input", input.input)
        .with_fact("column", &input.column)
        .with_fact("actual_rows", series.len()));
    }
    Ok(Some(series))
}

fn validate_predicate_series(
    frame: &DataFrame,
    input: NodeId,
    column: &str,
    role: &'static str,
    vertex_type: cudf::TypeId,
) -> Result<cudf::Series> {
    let series = frame.series(column).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.graph_algorithm.bfs_predicates",
            "failed to access BFS predicate column",
            error,
        )
        .with_fact("predicate", role)
        .with_fact("input", input)
        .with_fact("column", column)
    })?;
    if series.null_count() != 0 {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.graph_algorithm.bfs_predicates",
            "BFS predicate id columns must be non-null",
        )
        .with_fact("predicate", role)
        .with_fact("input", input)
        .with_fact("column", column)
        .with_fact("null_count", series.null_count()));
    }
    let dtype = series.dtype().map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.graph_algorithm.bfs_predicates",
            "failed to inspect BFS predicate column type",
            error,
        )
        .with_fact("predicate", role)
        .with_fact("input", input)
        .with_fact("column", column)
    })?;
    if dtype.id() != vertex_type {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "cudf_query_engine.graph_algorithm.bfs_predicates",
            "BFS predicate id column dtype must match the graph vertex dtype",
        )
        .with_fact("predicate", role)
        .with_fact("input", input)
        .with_fact("column", column)
        .with_fact("predicate_dtype", format!("{:?}", dtype.id()))
        .with_fact("vertex_dtype", format!("{vertex_type:?}")));
    }
    Ok(series)
}

fn predicate_column_slice<'a, T: cudf::CudfType>(
    series: Option<&'a cudf::Series>,
    role: &'static str,
) -> Result<Option<cudf::core::DeviceSlice<'a>>> {
    series
        .map(|series| {
            series.as_column().device_slice::<T>().map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.graph_algorithm.bfs_predicates",
                    "failed to borrow BFS predicate column device slice",
                    error,
                )
                .with_fact("predicate", role)
            })
        })
        .transpose()
}

fn predicate_device_view<'a, T: cugraph::types::CugraphElement>(
    slice: Option<&'a cudf::core::DeviceSlice<'a>>,
    role: &'static str,
) -> Result<Option<cugraph::DeviceArrayView<'a, T>>> {
    slice
        .map(|slice| {
            cugraph::cudf_interop::device_array_view_from_slice::<T>(slice).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.bfs_predicates",
                "failed to wrap BFS predicate column as a cuGraph device-array view",
            ))
        })
        .transpose()
        .map_err(|error| error.with_fact("predicate", role))
}

fn optional_series_len(series: Option<&cudf::Series>) -> String {
    series.map_or_else(|| "none".to_string(), |series| series.len().to_string())
}

fn optional_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "none".to_string(), |value| value.to_string())
}

fn bfs_target_metadata(target_info: Option<&cugraph::traversal::BfsTargetInfo>) -> String {
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
    context: &ExecutionContext,
    params: &crate::plan::GraphBfsParams,
    frame: DataFrame,
    target_info: Option<cugraph::traversal::BfsTargetInfo>,
) -> Result<DataFrame> {
    if !params.return_target_info {
        return Ok(frame);
    }
    let info = target_info.ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.graph_algorithm.bfs_target_metadata",
            "BFS target metadata was requested but cuGraph returned no target metadata",
        )
    })?;
    let row_count = frame.num_rows();
    let target_found = context
        .runtime()
        .scalar(info.target_found)
        .and_then(|scalar| context.runtime().series_from_scalar(&scalar, row_count))
        .map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.graph_algorithm.bfs_target_metadata",
                "failed to materialize BFS target_found metadata column",
                error,
            )
        })?;
    let target_distance = match info.target_distance {
        Some(distance) => context
            .runtime()
            .scalar(distance)
            .and_then(|scalar| context.runtime().series_from_scalar(&scalar, row_count)),
        None => context.runtime().nulls(cudf::DataType::new(cudf::TypeId::Int64), row_count),
    }
    .map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.graph_algorithm.bfs_target_metadata",
            "failed to materialize BFS target_distance metadata column",
            error,
        )
    })?;
    frame
        .with_column("target_found", target_found)
        .and_then(|frame| frame.with_column("target_distance", target_distance))
        .map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.graph_algorithm.bfs_target_metadata",
                "failed to append BFS target metadata columns",
                error,
            )
        })
}

pub(super) fn execute_sssp(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult> {
    let input = inputs.edge_list;
    let crate::plan::GraphAlgorithmParams::Sssp(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.graph_algorithm.sssp",
            "SSSP graph algorithm node carried non-SSSP parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, vertex_type) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);
    let sssp_p = sssp_params(params)?;

    run_with_cugraph_graph::<cugraph::Forward, _, _>(
        context,
        edges,
        options,
        "datafusion_cudf_cugraph_sssp",
        |runtime, entered, graph| {
            let frame = match vertex_type {
                cudf::TypeId::Int32 => {
                    let source = i32::try_from(params.source_vertex).map_err(|_| {
                        QueryError::new(
                            QueryErrorCode::InvalidPlan,
                            "cudf_query_engine.graph_algorithm.sssp_source",
                            "SSSP source vertex does not fit the Int32 graph vertex dtype",
                        )
                        .with_fact("source_vertex", params.source_vertex)
                        .with_fact("vertex_dtype", "Int32")
                    })?;
                    let result = graph
                        .sssp(entered, source, &sssp_p)
                        .map_err(map_cugraph_error("cudf_query_engine.graph_algorithm.sssp", "cuGraph SSSP failed"))?;
                    result.to_dataframe(runtime).map_err(map_cugraph_error(
                        "cudf_query_engine.graph_algorithm.sssp_result",
                        "failed to convert cuGraph SSSP result to cudf DataFrame",
                    ))?
                },
                cudf::TypeId::Int64 => {
                    let result = graph
                        .sssp(entered, params.source_vertex, &sssp_p)
                        .map_err(map_cugraph_error("cudf_query_engine.graph_algorithm.sssp", "cuGraph SSSP failed"))?;
                    result.to_dataframe(runtime).map_err(map_cugraph_error(
                        "cudf_query_engine.graph_algorithm.sssp_result",
                        "failed to convert cuGraph SSSP result to cudf DataFrame",
                    ))?
                },
                _ => {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedType,
                        "cudf_query_engine.graph_algorithm.sssp_source",
                        "SSSP graph vertex dtype is unsupported",
                    )
                    .with_fact("vertex_dtype", format!("{vertex_type:?}")));
                },
            };
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata: format!(
                    "source_vertex={},cutoff={}",
                    params.source_vertex,
                    params.cutoff.map_or_else(|| "none".to_string(), |v| v.to_string()),
                ),
            })
        },
    )
}
