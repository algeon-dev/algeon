use super::*;

/// Thread-affine cuGraph execution session.
///
/// Binds one `cugraph::ResourceHandle` to one `cudf::Runtime` for an executor
/// task.
///
/// **Do not store this in `ExecutionContext`.** `ResourceHandle` is `!Send +
/// !Sync`; keep it local to the task thread.
// Not yet used by any executor path; reserved for future multi-algorithm sequential tasks.
#[cfg(feature = "cugraph")]
#[allow(dead_code)]
struct CugraphExecutionSession {
    runtime: std::sync::Arc<cudf::Runtime>,
    handle: cugraph::ResourceHandle,
}

#[cfg(feature = "cugraph")]
#[allow(dead_code)]
impl CugraphExecutionSession {
    fn new(context: &ExecutionContext) -> Result<Self> {
        let runtime = context.shared_runtime()?;
        let handle =
            cugraph::ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).map_err(map_cugraph_error(
                "nexus_query_engine.graph_algorithm.session",
                "failed to bind cuGraph resource handle to the native cudf runtime",
            ))?;
        Ok(Self { runtime, handle })
    }

    fn enter(&self, operation: &'static str) -> Result<cugraph::EnteredContext<'_>> {
        self.handle.enter(operation).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.enter",
            "failed to enter cuGraph execution context",
        ))
    }

    fn runtime(&self) -> &cudf::Runtime {
        &self.runtime
    }

    /// Asserts that `egress` is the same runtime used to bind the handle.
    ///
    /// Call before passing a runtime into any `to_dataframe` conversion. If the
    /// runtimes ever diverge, device-side `cudaEventRecord` +
    /// `cudaStreamWaitEvent` ordering is required — not a host synchronize —
    /// before copying cuGraph-produced memory into the egress stream.
    fn assert_egress_runtime(&self, egress: &cudf::Runtime) {
        debug_assert!(
            std::ptr::eq(self.runtime.as_ref(), egress),
            "cuGraph result egress runtime must be the same Arc<cudf::Runtime> used to \
             bind the resource handle; cross-runtime result handoff requires device-side \
             stream ordering via cudaEventRecord + cudaStreamWaitEvent"
        );
    }
}

#[cfg(feature = "cugraph")]
pub(super) fn run_with_uncached_cugraph_graph<T, F>(
    context: &ExecutionContext,
    graph_memory_limit_bytes: Option<u64>,
    edges: DataFrame,
    options: cugraph::cudf_interop::TableGraphOptions,
    operation: &'static str,
    f: F,
) -> Result<GraphAlgorithmExecutionResult>
where
    T: cugraph::Transposition,
    F: FnOnce(
        &cudf::Runtime,
        &cugraph::EnteredContext<'_>,
        &mut cugraph::Graph<T>,
    ) -> Result<GraphAlgorithmExecutionResult>,
{
    let export = edges.try_prepare_device_table_export().map_err(|error| {
        QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            "nexus_query_engine.graph_algorithm.prepare_table_export",
            "failed to prepare cuGraph edge-list table export",
        )
        .with_source(error)
    })?;
    let budget = graph_execution_budget(
        context,
        graph_memory_limit_bytes,
        crate::graph_memory::GRAPH_MEMORY_BOUNDARY,
        operation,
    )?;
    let handle =
        cugraph::ResourceHandle::for_runtime(std::sync::Arc::clone(&budget.runtime)).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.resource_handle",
            "failed to bind cuGraph resource handle to the native cudf runtime",
        ))?;
    let entered = handle.enter(operation).map_err(map_cugraph_error(
        "nexus_query_engine.graph_algorithm.enter",
        "failed to enter cuGraph execution context",
    ))?;
    let mut graph =
        cugraph::cudf_interop::graph_from_prepared_table_export::<T>(&entered, &export, &options).map_err(|error| {
            attach_graph_budget_error_facts(
                map_cugraph_error(
                    "nexus_query_engine.graph_algorithm.build_graph",
                    "failed to build cuGraph graph from edge-list table",
                )(error),
                budget.token.as_ref(),
                operation,
            )
        })?;
    attach_graph_budget_observation(f(budget.runtime.as_ref(), &entered, &mut graph), budget.token.as_ref(), operation)
}

/// Build a cuGraph graph through `GraphBuilder` when the native edge list
/// carries stable edge ids. The cudf table-export path does not expose edge-id
/// binding, so this path borrows typed device slices for source, destination,
/// optional weight, and edge id columns directly for graph construction.
#[cfg(feature = "cugraph")]
pub(super) fn run_with_cugraph_graph_edge_ids<T, F>(
    context: &ExecutionContext,
    graph_memory_limit_bytes: Option<u64>,
    edges: &DataFrame,
    spec: &crate::plan::GraphAlgorithmSpec,
    operation: &'static str,
    f: F,
) -> Result<GraphAlgorithmExecutionResult>
where
    T: cugraph::Transposition,
    F: FnOnce(
        &cudf::Runtime,
        &cugraph::EnteredContext<'_>,
        &mut cugraph::Graph<T>,
    ) -> Result<GraphAlgorithmExecutionResult>,
{
    let budget = graph_execution_budget(
        context,
        graph_memory_limit_bytes,
        crate::graph_memory::GRAPH_MEMORY_BOUNDARY,
        operation,
    )?;
    let handle =
        cugraph::ResourceHandle::for_runtime(std::sync::Arc::clone(&budget.runtime)).map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.resource_handle",
            "failed to bind cuGraph resource handle to the native cudf runtime",
        ))?;
    let entered = handle.enter(operation).map_err(map_cugraph_error(
        "nexus_query_engine.graph_algorithm.enter",
        "failed to enter cuGraph execution context",
    ))?;
    let mut graph = build_graph_from_edge_id_columns::<T>(&entered, edges, spec).map_err(|error| {
        if error.code() == QueryErrorCode::CudfResourceExhausted {
            attach_graph_budget_error_facts(error, budget.token.as_ref(), operation)
        } else {
            error
        }
    })?;
    let result = attach_graph_budget_observation(
        f(budget.runtime.as_ref(), &entered, &mut graph),
        budget.token.as_ref(),
        operation,
    );
    if context.resident_graph_cache().is_some() {
        result.map(|result| with_graph_cache_metadata(result, "bypass_edge_id_graph"))
    } else {
        result
    }
}

#[cfg(feature = "cugraph")]
fn build_graph_from_edge_id_columns<T: cugraph::Transposition>(
    entered: &cugraph::EnteredContext<'_>,
    edges: &DataFrame,
    spec: &crate::plan::GraphAlgorithmSpec,
) -> Result<cugraph::Graph<T>> {
    let vertex_type = edges
        .series(&spec.edge_columns.source)
        .and_then(|series| series.dtype())
        .map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.graph_algorithm.edge_id_graph",
                "failed to inspect source vertex dtype for edge-id graph construction",
                error,
            )
            .with_fact("src_col", &spec.edge_columns.source)
        })?
        .id();
    match vertex_type {
        cudf::TypeId::Int32 => build_graph_from_edge_id_columns_typed::<T, i32>(entered, edges, spec),
        cudf::TypeId::Int64 => build_graph_from_edge_id_columns_typed::<T, i64>(entered, edges, spec),
        _ => Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "nexus_query_engine.graph_algorithm.edge_id_graph",
            "edge-id graph construction requires Int32 or Int64 vertices",
        )
        .with_fact("vertex_dtype", format!("{vertex_type:?}"))),
    }
}

#[cfg(feature = "cugraph")]
fn build_graph_from_edge_id_columns_typed<T, V>(
    entered: &cugraph::EnteredContext<'_>,
    edges: &DataFrame,
    spec: &crate::plan::GraphAlgorithmSpec,
) -> Result<cugraph::Graph<T>>
where
    T: cugraph::Transposition,
    V: cudf::CudfType + cugraph::types::CugraphElement,
{
    let edge_id_col = spec.edge_columns.edge_id.as_deref().ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.graph_algorithm.edge_id_graph",
            "edge-id graph construction requires a stable edge-id column binding",
        )
    })?;
    let src = graph_series(edges, &spec.edge_columns.source, "source")?;
    let dst = graph_series(edges, &spec.edge_columns.destination, "destination")?;
    let edge_ids = graph_series(edges, edge_id_col, "edge_id")?;
    let src_slice = graph_device_slice::<V>(&src, "source")?;
    let dst_slice = graph_device_slice::<V>(&dst, "destination")?;
    let edge_id_slice = graph_device_slice::<V>(&edge_ids, "edge_id")?;
    let src_view = graph_device_view::<V>(&src_slice, "source")?;
    let dst_view = graph_device_view::<V>(&dst_slice, "destination")?;
    let edge_id_view = graph_device_view::<V>(&edge_id_slice, "edge_id")?;

    // Weight is only projected (and thus present in `edges`) for algorithms
    // that consume it; mirror that gate here so an unweighted edge-id graph
    // does not look up a column `project_edge_list` intentionally dropped.
    let weight_binding = if spec.algorithm.consumes_weight() { spec.edge_columns.weight.as_deref() } else { None };
    match weight_binding {
        Some(weight_col) => {
            let weight = graph_series(edges, weight_col, "weight")?;
            let weight_dtype = weight.dtype().map_err(|error| {
                QueryError::from_cudf(
                    "nexus_query_engine.graph_algorithm.edge_id_graph",
                    "failed to inspect graph weight column dtype",
                    error,
                )
                .with_fact("weight_col", weight_col)
            })?;
            match weight_dtype.id() {
                cudf::TypeId::Float32 => {
                    let weight_slice = graph_device_slice::<f32>(&weight, "weight")?;
                    let weight_view = graph_device_view::<f32>(&weight_slice, "weight")?;
                    graph_builder_base(spec, &src_view, &dst_view)
                        .weights(&weight_view)
                        .edge_ids(&edge_id_view)
                        .build(entered)
                        .map_err(map_cugraph_error(
                            "nexus_query_engine.graph_algorithm.edge_id_graph",
                            "failed to build cuGraph graph with stable edge ids",
                        ))
                },
                cudf::TypeId::Float64 => {
                    let weight_slice = graph_device_slice::<f64>(&weight, "weight")?;
                    let weight_view = graph_device_view::<f64>(&weight_slice, "weight")?;
                    graph_builder_base(spec, &src_view, &dst_view)
                        .weights(&weight_view)
                        .edge_ids(&edge_id_view)
                        .build(entered)
                        .map_err(map_cugraph_error(
                            "nexus_query_engine.graph_algorithm.edge_id_graph",
                            "failed to build cuGraph graph with stable edge ids",
                        ))
                },
                _ => Err(QueryError::new(
                    QueryErrorCode::UnsupportedType,
                    "nexus_query_engine.graph_algorithm.edge_id_graph",
                    "graph weight column dtype is unsupported",
                )
                .with_fact("weight_col", weight_col)
                .with_fact("weight_dtype", format!("{weight_dtype:?}"))),
            }
        },
        None => graph_builder_base(spec, &src_view, &dst_view).edge_ids(&edge_id_view).build(entered).map_err(
            map_cugraph_error(
                "nexus_query_engine.graph_algorithm.edge_id_graph",
                "failed to build cuGraph graph with stable edge ids",
            ),
        ),
    }
}

#[cfg(feature = "cugraph")]
fn graph_builder_base<'a, V: cugraph::types::CugraphElement>(
    spec: &crate::plan::GraphAlgorithmSpec,
    src_view: &'a cugraph::DeviceArrayView<'a, V>,
    dst_view: &'a cugraph::DeviceArrayView<'a, V>,
) -> cugraph::GraphBuilder<'a> {
    let mut builder =
        cugraph::GraphBuilder::new(src_view, dst_view).properties(graph_properties(spec.graph_options.directed));
    if spec.graph_options.renumber {
        builder = builder.renumber();
    }
    builder
}

#[cfg(feature = "cugraph")]
pub(super) fn graph_series(edges: &DataFrame, column: &str, role: &'static str) -> Result<cudf::Series> {
    edges.series(column).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.graph_algorithm.edge_id_graph",
            "failed to access graph construction column",
            error,
        )
        .with_fact("role", role)
        .with_fact("column", column)
    })
}

#[cfg(feature = "cugraph")]
pub(super) fn graph_device_slice<'a, T: cudf::CudfType>(
    series: &'a cudf::Series,
    role: &'static str,
) -> Result<cudf::core::DeviceSlice<'a>> {
    series.as_column().device_slice::<T>().map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.graph_algorithm.edge_id_graph",
            "failed to borrow graph construction column device slice",
            error,
        )
        .with_fact("role", role)
    })
}

#[cfg(feature = "cugraph")]
pub(super) fn graph_device_view<'a, T: cugraph::types::CugraphElement>(
    slice: &'a cudf::core::DeviceSlice<'a>,
    role: &'static str,
) -> Result<cugraph::DeviceArrayView<'a, T>> {
    cugraph::cudf_interop::device_array_view_from_slice::<T>(slice)
        .map_err(map_cugraph_error(
            "nexus_query_engine.graph_algorithm.edge_id_graph",
            "failed to wrap graph construction column as a cuGraph device-array view",
        ))
        .map_err(|error| error.with_fact("role", role))
}

#[cfg(feature = "cugraph")]
pub(super) fn project_edge_list(
    frame: &DataFrame,
    spec: &crate::plan::GraphAlgorithmSpec,
) -> Result<(DataFrame, cudf::TypeId)> {
    let source_type = validate_edge_column(frame, &spec.edge_columns.source, EdgeColumnRole::Vertex)?;
    let destination_type = validate_edge_column(frame, &spec.edge_columns.destination, EdgeColumnRole::Vertex)?;
    if source_type.id() != destination_type.id() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "nexus_query_engine.graph_algorithm.validate_edge_column",
            "cuGraph source and destination columns must have the same vertex dtype",
        )
        .with_fact("src_col", &spec.edge_columns.source)
        .with_fact("dst_col", &spec.edge_columns.destination)
        .with_fact("src_cudf_dtype", format!("{source_type:?}"))
        .with_fact("dst_cudf_dtype", format!("{destination_type:?}")));
    }
    // Only validate/project the weight column for algorithms that consume it.
    // Unweighted algorithms (BFS, etc.) must ignore a bound weight entirely:
    // validating it would reject legitimately-null weights, and projecting it
    // would feed cuGraph a weight it does not use. Kept in lockstep with
    // `table_graph_options`, which only sets `weight_col` under the same gate.
    if spec.algorithm.consumes_weight()
        && let Some(weight) = &spec.edge_columns.weight
    {
        validate_edge_column(frame, weight, EdgeColumnRole::Weight)?;
    }
    if let Some(edge_id) = &spec.edge_columns.edge_id {
        let edge_id_type = validate_edge_column(frame, edge_id, EdgeColumnRole::EdgeId)?;
        if edge_id_type.id() != source_type.id() {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                "nexus_query_engine.graph_algorithm.validate_edge_column",
                "cuGraph edge-id column dtype must match the graph vertex dtype",
            )
            .with_fact("edge_id_col", edge_id)
            .with_fact("edge_id_cudf_dtype", format!("{edge_id_type:?}"))
            .with_fact("vertex_cudf_dtype", format!("{source_type:?}")));
        }
    }

    let mut selected = vec![spec.edge_columns.source.as_str(), spec.edge_columns.destination.as_str()];
    if spec.algorithm.consumes_weight()
        && let Some(weight) = &spec.edge_columns.weight
    {
        selected.push(weight.as_str());
    }
    if let Some(edge_id) = &spec.edge_columns.edge_id {
        selected.push(edge_id.as_str());
    }
    let edges = frame.select(selected).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.graph_algorithm.edge_projection",
            "failed to project cuGraph edge-list columns",
            error,
        )
        .with_fact("src_col", &spec.edge_columns.source)
        .with_fact("dst_col", &spec.edge_columns.destination)
        .with_fact("weight_col", spec.edge_columns.weight.as_deref().unwrap_or("<none>"))
        .with_fact("edge_id_col", spec.edge_columns.edge_id.as_deref().unwrap_or("<none>"))
    })?;
    Ok((edges, source_type.id()))
}

#[cfg(feature = "cugraph")]
fn graph_properties(directed: bool) -> cugraph::types::GraphProperties {
    if directed { cugraph::types::GraphProperties::directed() } else { cugraph::types::GraphProperties::undirected() }
}

#[cfg(feature = "cugraph")]
pub(super) fn table_graph_options(spec: &crate::plan::GraphAlgorithmSpec) -> cugraph::cudf_interop::TableGraphOptions {
    let mut options = cugraph::cudf_interop::TableGraphOptions::new(0, 1)
        .properties(graph_properties(spec.graph_options.directed))
        .construction_policy(graph_construction_policy(spec.graph_options.policy));
    if spec.graph_options.renumber {
        options = options.renumber();
    }
    // `project_edge_list` only emits the weight column (at index 2, before any
    // edge-id column) when the algorithm consumes weights. Gate the cuGraph
    // weight binding on the same condition so the index stays valid.
    if spec.algorithm.consumes_weight() && spec.edge_columns.weight.is_some() {
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
