#[cfg(feature = "cugraph")]
use arrow::datatypes::SchemaRef;

use super::*;

#[cfg(not(feature = "cugraph"))]
pub(super) fn execute_graph_algorithm(
    _context: &ExecutionContext,
    node_id: NodeId,
    node: &crate::plan::GraphAlgorithmNode,
    input: GpuDataFrame,
    _metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let input_rows = input.num_rows();
    let input_columns = input.num_columns();
    Err(QueryError::new(
        QueryErrorCode::UnsupportedOperator,
        "cudf_query_engine.graph_algorithm.execute",
        "cuGraph graph algorithm execution requires the cugraph Cargo feature",
    )
    .with_fact("node_id", node_id)
    .with_fact("algorithm", node.spec.algorithm.as_str())
    .with_fact("input_rows", input_rows)
    .with_fact("input_columns", input_columns)
    .with_fact("required_feature", "cugraph"))
}

#[cfg(feature = "cugraph")]
pub(super) fn execute_graph_algorithm(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &crate::plan::GraphAlgorithmNode,
    input: GpuDataFrame,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let input_rows = input.num_rows() as u64;
    let input_columns = input.num_columns() as u64;
    let boundary_before = metrics.boundary.clone();
    let timer = Instant::now();
    let result = match node.spec.algorithm {
        crate::plan::GraphAlgorithm::PageRank => execute_pagerank(context, &node.spec, input)?,
        crate::plan::GraphAlgorithm::Bfs => execute_bfs(context, &node.spec, input)?,
        crate::plan::GraphAlgorithm::Louvain => execute_louvain(context, &node.spec, input)?,
    };
    let elapsed = timer.elapsed();
    metrics.operator_time += elapsed;

    let gpu = GpuDataFrame::new(result.frame);
    let output_device_bytes = gpu.device_allocated_size_bytes().ok();
    let boundary_delta = boundary_delta(&boundary_before, &metrics.boundary);
    metrics.record_graph_algorithm_runtime(crate::metrics::GraphAlgorithmRuntimeMetrics {
        node_id: node_id.to_string(),
        algorithm: node.spec.algorithm.as_str().to_string(),
        runtime_outcome: "succeeded".to_string(),
        input_boundary: "gpu_resident".to_string(),
        output_boundary: "gpu_resident".to_string(),
        construction_policy: node.spec.graph_options.policy.as_str().to_string(),
        source_column: node.spec.edge_columns.source.clone(),
        destination_column: node.spec.edge_columns.destination.clone(),
        weight_column: node.spec.edge_columns.weight.clone(),
        directed: node.spec.graph_options.directed,
        renumber: node.spec.graph_options.renumber,
        algorithm_parameters: graph_algorithm_parameters(&node.spec.params),
        algorithm_metadata: result.algorithm_metadata,
        input_rows,
        input_columns,
        output_rows: gpu.num_rows() as u64,
        output_columns: gpu.num_columns() as u64,
        output_device_bytes,
        elapsed_ns: u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX),
        gpu_resident_input_handoffs: boundary_delta.gpu_resident_input_handoffs,
        host_to_device_uploads: boundary_delta.host_to_device_uploads,
        device_to_host_materializations: boundary_delta.device_to_host_materializations,
    });
    Ok(gpu)
}

#[cfg(feature = "cugraph")]
struct GraphAlgorithmExecutionResult {
    frame: DataFrame,
    algorithm_metadata: String,
}

#[cfg(feature = "cugraph")]
fn execute_pagerank(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    input: GpuDataFrame,
) -> Result<GraphAlgorithmExecutionResult> {
    let crate::plan::GraphAlgorithmParams::PageRank(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.graph_algorithm.pagerank",
            "PageRank graph algorithm node carried non-PageRank parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;

    let export = edges.try_prepare_device_table_export().map_err(|error| {
        QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            "cudf_query_engine.graph_algorithm.prepare_table_export",
            "failed to prepare cuGraph edge-list table export",
        )
        .with_source(error)
    })?;
    let handle = cugraph::ResourceHandle::for_runtime(context.shared_runtime()).map_err(map_cugraph_error(
        "cudf_query_engine.graph_algorithm.resource_handle",
        "failed to bind cuGraph resource handle to the native cudf runtime",
    ))?;
    let entered = handle.enter("datafusion_cudf_cugraph_pagerank").map_err(map_cugraph_error(
        "cudf_query_engine.graph_algorithm.enter",
        "failed to enter cuGraph execution context",
    ))?;

    let options = table_graph_options(spec);

    let mut graph =
        cugraph::cudf_interop::graph_from_prepared_table_export::<cugraph::Transposed>(&entered, &export, &options)
            .map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.build_graph",
                "failed to build cuGraph graph from edge-list table",
            ))?;

    let result = graph
        .pagerank(&entered, &pagerank_params(params)?)
        .map_err(map_cugraph_error("cudf_query_engine.graph_algorithm.pagerank", "cuGraph PageRank failed"))?;
    let converged = result.converged().map_or_else(|_| "unknown".to_string(), |value| value.to_string());
    let algorithm_metadata = format!("converged={},iterations={}", converged, result.num_iterations());
    let frame = result.to_dataframe(context.runtime()).map_err(map_cugraph_error(
        "cudf_query_engine.graph_algorithm.pagerank_result",
        "failed to convert cuGraph PageRank result to cudf DataFrame",
    ))?;
    Ok(GraphAlgorithmExecutionResult {
        frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
        algorithm_metadata,
    })
}

#[cfg(feature = "cugraph")]
fn execute_bfs(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    input: GpuDataFrame,
) -> Result<GraphAlgorithmExecutionResult> {
    let crate::plan::GraphAlgorithmParams::Bfs(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.graph_algorithm.bfs",
            "BFS graph algorithm node carried non-BFS parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, vertex_type) = project_edge_list(&input, spec)?;

    let export = edges.try_prepare_device_table_export().map_err(|error| {
        QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            "cudf_query_engine.graph_algorithm.prepare_table_export",
            "failed to prepare cuGraph edge-list table export",
        )
        .with_source(error)
    })?;
    let handle = cugraph::ResourceHandle::for_runtime(context.shared_runtime()).map_err(map_cugraph_error(
        "cudf_query_engine.graph_algorithm.resource_handle",
        "failed to bind cuGraph resource handle to the native cudf runtime",
    ))?;
    let entered = handle.enter("datafusion_cudf_cugraph_bfs").map_err(map_cugraph_error(
        "cudf_query_engine.graph_algorithm.enter",
        "failed to enter cuGraph execution context",
    ))?;

    let options = table_graph_options(spec);
    let mut graph =
        cugraph::cudf_interop::graph_from_prepared_table_export::<cugraph::Forward>(&entered, &export, &options)
            .map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.build_graph",
                "failed to build cuGraph graph from edge-list table",
            ))?;
    let bfs_params = bfs_params(params)?;
    let frame = match vertex_type {
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
            let sources = cugraph::DeviceArray::from_slice(&entered, &[source]).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.bfs_source",
                "failed to copy BFS source vertex to device memory",
            ))?;
            let sources_view = sources.view().map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.bfs_source",
                "failed to create BFS source vertex view",
            ))?;
            let result = graph
                .bfs(&entered, &sources_view, &bfs_params)
                .map_err(map_cugraph_error("cudf_query_engine.graph_algorithm.bfs", "cuGraph BFS failed"))?;
            result.to_dataframe(context.runtime()).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.bfs_result",
                "failed to convert cuGraph BFS result to cudf DataFrame",
            ))?
        },
        cudf::TypeId::Int64 => {
            let sources =
                cugraph::DeviceArray::from_slice(&entered, &[params.source_vertex]).map_err(map_cugraph_error(
                    "cudf_query_engine.graph_algorithm.bfs_source",
                    "failed to copy BFS source vertex to device memory",
                ))?;
            let sources_view = sources.view().map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.bfs_source",
                "failed to create BFS source vertex view",
            ))?;
            let result = graph
                .bfs(&entered, &sources_view, &bfs_params)
                .map_err(map_cugraph_error("cudf_query_engine.graph_algorithm.bfs", "cuGraph BFS failed"))?;
            result.to_dataframe(context.runtime()).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.bfs_result",
                "failed to convert cuGraph BFS result to cudf DataFrame",
            ))?
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
            "source_vertex={},depth_limit={},compute_predecessors=true",
            params.source_vertex,
            params.depth_limit.map_or_else(|| "none".to_string(), |value| value.to_string())
        ),
    })
}

#[cfg(feature = "cugraph")]
fn execute_louvain(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    input: GpuDataFrame,
) -> Result<GraphAlgorithmExecutionResult> {
    let crate::plan::GraphAlgorithmParams::Louvain(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.graph_algorithm.louvain",
            "Louvain graph algorithm node carried non-Louvain parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;

    let export = edges.try_prepare_device_table_export().map_err(|error| {
        QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            "cudf_query_engine.graph_algorithm.prepare_table_export",
            "failed to prepare cuGraph edge-list table export",
        )
        .with_source(error)
    })?;
    let handle = cugraph::ResourceHandle::for_runtime(context.shared_runtime()).map_err(map_cugraph_error(
        "cudf_query_engine.graph_algorithm.resource_handle",
        "failed to bind cuGraph resource handle to the native cudf runtime",
    ))?;
    let entered = handle.enter("datafusion_cudf_cugraph_louvain").map_err(map_cugraph_error(
        "cudf_query_engine.graph_algorithm.enter",
        "failed to enter cuGraph execution context",
    ))?;

    let options = table_graph_options(spec);
    let mut graph =
        cugraph::cudf_interop::graph_from_prepared_table_export::<cugraph::Forward>(&entered, &export, &options)
            .map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.build_graph",
                "failed to build cuGraph graph from edge-list table",
            ))?;
    let result = graph
        .louvain(&entered, &louvain_params(params)?)
        .map_err(map_cugraph_error("cudf_query_engine.graph_algorithm.louvain", "cuGraph Louvain failed"))?;
    let algorithm_metadata = format!("modularity={}", result.modularity());
    let frame = result.to_dataframe(context.runtime()).map_err(map_cugraph_error(
        "cudf_query_engine.graph_algorithm.louvain_result",
        "failed to convert cuGraph Louvain result to cudf DataFrame",
    ))?;
    Ok(GraphAlgorithmExecutionResult {
        frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
        algorithm_metadata,
    })
}

#[cfg(feature = "cugraph")]
#[derive(Debug, Clone, Copy)]
enum EdgeColumnRole {
    Vertex,
    Weight,
}

#[cfg(feature = "cugraph")]
fn project_edge_list(frame: &DataFrame, spec: &crate::plan::GraphAlgorithmSpec) -> Result<(DataFrame, cudf::TypeId)> {
    let source_type = validate_edge_column(frame, &spec.edge_columns.source, EdgeColumnRole::Vertex)?;
    let destination_type = validate_edge_column(frame, &spec.edge_columns.destination, EdgeColumnRole::Vertex)?;
    if source_type.id() != destination_type.id() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "cudf_query_engine.graph_algorithm.validate_edge_column",
            "cuGraph source and destination columns must have the same vertex dtype",
        )
        .with_fact("src_col", &spec.edge_columns.source)
        .with_fact("dst_col", &spec.edge_columns.destination)
        .with_fact("src_cudf_dtype", format!("{source_type:?}"))
        .with_fact("dst_cudf_dtype", format!("{destination_type:?}")));
    }
    if let Some(weight) = &spec.edge_columns.weight {
        validate_edge_column(frame, weight, EdgeColumnRole::Weight)?;
    }

    let mut selected = vec![spec.edge_columns.source.as_str(), spec.edge_columns.destination.as_str()];
    if let Some(weight) = &spec.edge_columns.weight {
        selected.push(weight.as_str());
    }
    let edges = frame.select(selected).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.graph_algorithm.edge_projection",
            "failed to project cuGraph edge-list columns",
            error,
        )
        .with_fact("src_col", &spec.edge_columns.source)
        .with_fact("dst_col", &spec.edge_columns.destination)
        .with_fact("weight_col", spec.edge_columns.weight.as_deref().unwrap_or("<none>"))
    })?;
    Ok((edges, source_type.id()))
}

#[cfg(feature = "cugraph")]
fn validate_edge_column(frame: &DataFrame, column: &str, role: EdgeColumnRole) -> Result<cudf::DataType> {
    let series = frame.series(column).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.graph_algorithm.validate_edge_column",
            "failed to access cuGraph edge-list column",
            error,
        )
        .with_fact("column", column)
    })?;
    if series.null_count() != 0 {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.graph_algorithm.validate_edge_column",
            "cuGraph edge-list columns must be non-null",
        )
        .with_fact("column", column)
        .with_fact("null_count", series.null_count()));
    }
    let dtype = series.dtype().map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.graph_algorithm.validate_edge_column",
            "failed to inspect cuGraph edge-list column type",
            error,
        )
        .with_fact("column", column)
    })?;
    let supported = match role {
        EdgeColumnRole::Vertex => matches!(dtype.id(), cudf::TypeId::Int32 | cudf::TypeId::Int64),
        EdgeColumnRole::Weight => matches!(dtype.id(), cudf::TypeId::Float32 | cudf::TypeId::Float64),
    };
    if supported {
        return Ok(dtype);
    }
    Err(QueryError::new(
        QueryErrorCode::UnsupportedType,
        "cudf_query_engine.graph_algorithm.validate_edge_column",
        "cuGraph edge-list column has unsupported dtype",
    )
    .with_fact("column", column)
    .with_fact(
        "role",
        match role {
            EdgeColumnRole::Vertex => "vertex",
            EdgeColumnRole::Weight => "weight",
        },
    )
    .with_fact("cudf_dtype", format!("{dtype:?}")))
}

#[cfg(feature = "cugraph")]
fn graph_properties(directed: bool) -> cugraph::types::GraphProperties {
    if directed { cugraph::types::GraphProperties::directed() } else { cugraph::types::GraphProperties::undirected() }
}

#[cfg(feature = "cugraph")]
fn table_graph_options(spec: &crate::plan::GraphAlgorithmSpec) -> cugraph::cudf_interop::TableGraphOptions {
    let mut options = cugraph::cudf_interop::TableGraphOptions::new(0, 1)
        .properties(graph_properties(spec.graph_options.directed))
        .construction_policy(graph_construction_policy(spec.graph_options.policy));
    if spec.graph_options.renumber {
        options = options.renumber();
    }
    if spec.edge_columns.weight.is_some() {
        options = options.weight_col(2);
    }
    options
}

#[cfg(feature = "cugraph")]
fn graph_construction_policy(
    policy: crate::plan::GraphConstructionPolicy,
) -> cugraph::cudf_interop::GraphConstructionPolicy {
    match policy {
        crate::plan::GraphConstructionPolicy::PythonCugraph => {
            cugraph::cudf_interop::GraphConstructionPolicy::PythonCugraph
        },
        crate::plan::GraphConstructionPolicy::RawLibcugraph => {
            cugraph::cudf_interop::GraphConstructionPolicy::RawLibcugraph
        },
    }
}

#[cfg(feature = "cugraph")]
fn graph_algorithm_parameters(params: &crate::plan::GraphAlgorithmParams) -> String {
    match params {
        crate::plan::GraphAlgorithmParams::PageRank(params) => {
            format!("alpha={},epsilon={},max_iterations={}", params.alpha, params.epsilon, params.max_iterations)
        },
        crate::plan::GraphAlgorithmParams::Bfs(params) => format!(
            "source_vertex={},depth_limit={}",
            params.source_vertex,
            params.depth_limit.map_or_else(|| "none".to_string(), |value| value.to_string())
        ),
        crate::plan::GraphAlgorithmParams::Louvain(params) => format!("resolution={}", params.resolution),
    }
}

#[cfg(feature = "cugraph")]
fn boundary_delta(
    before: &crate::metrics::BoundaryMetrics,
    after: &crate::metrics::BoundaryMetrics,
) -> crate::metrics::BoundaryMetrics {
    crate::metrics::BoundaryMetrics {
        gpu_resident_input_handoffs: after
            .gpu_resident_input_handoffs
            .saturating_sub(before.gpu_resident_input_handoffs),
        host_to_device_uploads: after.host_to_device_uploads.saturating_sub(before.host_to_device_uploads),
        host_to_device_uploaded_rows: after
            .host_to_device_uploaded_rows
            .saturating_sub(before.host_to_device_uploaded_rows),
        host_to_device_uploaded_bytes: after
            .host_to_device_uploaded_bytes
            .saturating_sub(before.host_to_device_uploaded_bytes),
        device_to_host_materializations: after
            .device_to_host_materializations
            .saturating_sub(before.device_to_host_materializations),
        device_to_host_materialized_rows: after
            .device_to_host_materialized_rows
            .saturating_sub(before.device_to_host_materialized_rows),
        device_to_host_materialized_bytes: after
            .device_to_host_materialized_bytes
            .saturating_sub(before.device_to_host_materialized_bytes),
        device_to_device_copies: after.device_to_device_copies.saturating_sub(before.device_to_device_copies),
        device_to_device_copied_bytes: after
            .device_to_device_copied_bytes
            .saturating_sub(before.device_to_device_copied_bytes),
    }
}

#[cfg(feature = "cugraph")]
fn pagerank_params(params: &crate::plan::GraphPageRankParams) -> Result<cugraph::centrality::PageRankParams> {
    Ok(cugraph::centrality::PageRankParams {
        alpha: cugraph::params::ClosedUnitF64::new(params.alpha).map_err(map_cugraph_error(
            "cudf_query_engine.graph_algorithm.pagerank_params",
            "invalid cuGraph PageRank alpha",
        ))?,
        epsilon: cugraph::params::NonNegativeF64::new(params.epsilon).map_err(map_cugraph_error(
            "cudf_query_engine.graph_algorithm.pagerank_params",
            "invalid cuGraph PageRank epsilon",
        ))?,
        max_iterations: cugraph::params::IterationCount::new(params.max_iterations).map_err(map_cugraph_error(
            "cudf_query_engine.graph_algorithm.pagerank_params",
            "invalid cuGraph PageRank max_iterations",
        ))?,
        do_expensive_check: false,
    })
}

#[cfg(feature = "cugraph")]
fn bfs_params(params: &crate::plan::GraphBfsParams) -> Result<cugraph::traversal::BfsParams> {
    let depth_limit = match params.depth_limit {
        None | Some(0) => None,
        Some(value) if value > 0 => {
            let value = usize::try_from(value).map_err(|_| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "cudf_query_engine.graph_algorithm.bfs_params",
                    "BFS depth_limit does not fit usize",
                )
                .with_fact("depth_limit", value)
            })?;
            Some(cugraph::params::BfsDepthLimit::new(value).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.bfs_params",
                "invalid cuGraph BFS depth_limit",
            ))?)
        },
        Some(value) => {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.graph_algorithm.bfs_params",
                "BFS depth_limit must be non-negative",
            )
            .with_fact("depth_limit", value));
        },
    };
    Ok(cugraph::traversal::BfsParams { depth_limit, compute_predecessors: true, do_expensive_check: false })
}

#[cfg(feature = "cugraph")]
fn louvain_params(params: &crate::plan::GraphLouvainParams) -> Result<cugraph::community::LouvainParams> {
    let mut cugraph_params = cugraph::community::LouvainParams::default();
    cugraph_params.resolution = cugraph::params::PositiveF64::new(params.resolution).map_err(map_cugraph_error(
        "cudf_query_engine.graph_algorithm.louvain_params",
        "invalid cuGraph Louvain resolution",
    ))?;
    Ok(cugraph_params)
}

#[cfg(feature = "cugraph")]
fn normalize_graph_result_schema(mut frame: DataFrame, schema: &SchemaRef) -> Result<DataFrame> {
    for field in schema.fields() {
        let field_name = field.name();
        let target_type = cudf_dtype_from_arrow_type(field.data_type()).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::UnsupportedType,
                "cudf_query_engine.graph_algorithm.schema_normalize",
                "cuGraph result schema contains an unsupported output type",
            )
            .with_fact("field_name", field_name)
            .with_fact("expected_type", format!("{:?}", field.data_type()))
        })?;
        let actual_type = frame.series(field_name).and_then(|series| series.dtype()).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.graph_algorithm.schema_normalize",
                "failed to inspect cuGraph result column",
                error,
            )
            .with_fact("field_name", field_name)
        })?;
        if actual_type != target_type {
            frame = frame.cast_column(field_name, target_type).map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.graph_algorithm.schema_normalize",
                    "failed to cast cuGraph result column to the stable SQL schema",
                    error,
                )
                .with_fact("field_name", field_name)
                .with_fact("actual_cudf_type", format!("{actual_type:?}"))
                .with_fact("expected_arrow_type", format!("{:?}", field.data_type()))
            })?;
        }
    }
    let columns = schema.fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>();
    frame.select(columns).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.graph_algorithm.schema_normalize",
            "failed to order cuGraph result columns according to the stable SQL schema",
            error,
        )
    })
}

#[cfg(feature = "cugraph")]
fn map_cugraph_error(operation: &'static str, message: &'static str) -> impl FnOnce(cugraph::Error) -> QueryError {
    move |error| {
        QueryError::new(code_for_cugraph_error(&error), operation, message)
            .with_fact("cugraph_kind", error.kind())
            .with_fact("cugraph_status", error.status())
            .with_fact("cugraph_operation", error.operation())
            .with_source(error)
    }
}

#[cfg(feature = "cugraph")]
fn code_for_cugraph_error(error: &cugraph::Error) -> QueryErrorCode {
    match error.kind() {
        cugraph::ErrorKind::OutOfMemory => QueryErrorCode::CudfResourceExhausted,
        cugraph::ErrorKind::InvalidInput | cugraph::ErrorKind::InvalidHandle => QueryErrorCode::CudfInvalidInput,
        cugraph::ErrorKind::UnsupportedType | cugraph::ErrorKind::NotImplemented => QueryErrorCode::CudfUnsupported,
        cugraph::ErrorKind::InvalidState => QueryErrorCode::CudfInvariantViolation,
        cugraph::ErrorKind::IoError | cugraph::ErrorKind::Unexpected => match error.status() {
            cugraph::ErrorStatus::Temporary => QueryErrorCode::CudfTemporaryFailure,
            cugraph::ErrorStatus::Persistent => QueryErrorCode::CudfPersistentFailure,
            cugraph::ErrorStatus::Permanent => QueryErrorCode::CudfDependencyFailure,
        },
        _ => QueryErrorCode::CudfDependencyFailure,
    }
}
