#[cfg(feature = "cugraph")]
use std::collections::BTreeMap;

#[cfg(feature = "cugraph")]
use arrow::datatypes::SchemaRef;

use super::*;

#[cfg(feature = "cugraph")]
mod centrality;
#[cfg(feature = "cugraph")]
mod community;
#[cfg(feature = "cugraph")]
mod params;
#[cfg(feature = "cugraph")]
mod structure;
#[cfg(feature = "cugraph")]
mod traversal;

#[cfg(feature = "cugraph")]
use centrality::{
    execute_eigenvector_centrality, execute_hits, execute_katz_centrality, execute_pagerank, execute_pagerank_strict,
};
#[cfg(feature = "cugraph")]
use community::{
    execute_balanced_cut_clustering, execute_ecg, execute_leiden, execute_louvain,
    execute_spectral_modularity_maximization,
};
#[cfg(feature = "cugraph")]
use params::{
    bfs_params, ecg_params, eigenvector_centrality_params, hits_params, k_core_params, katz_centrality_params,
    leiden_params, louvain_params, pagerank_params, spectral_clustering_params, sssp_params,
};
#[cfg(feature = "cugraph")]
use structure::{
    execute_core_number, execute_degrees_all, execute_in_degrees_all, execute_k_core, execute_minimum_spanning_tree,
    execute_out_degrees_all, execute_strongly_connected_components, execute_triangle_count_all,
    execute_weakly_connected_components,
};
#[cfg(feature = "cugraph")]
use traversal::{execute_bfs, execute_sssp};

#[cfg(not(feature = "cugraph"))]
pub(super) fn execute_graph_algorithm(
    _context: &ExecutionContext,
    node_id: NodeId,
    node: &crate::plan::GraphAlgorithmNode,
    input: GpuDataFrame,
    side_inputs: std::collections::BTreeMap<NodeId, GpuDataFrame>,
    _metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let input_rows = input.num_rows();
    let input_columns = input.num_columns();
    let side_input_count = side_inputs.len();
    Err(QueryError::new(
        QueryErrorCode::UnsupportedOperator,
        "cudf_query_engine.graph_algorithm.execute",
        "cuGraph graph algorithm execution requires the cugraph Cargo feature",
    )
    .with_fact("node_id", node_id)
    .with_fact("algorithm", node.spec.algorithm.as_str())
    .with_fact("input_rows", input_rows)
    .with_fact("input_columns", input_columns)
    .with_fact("side_input_count", side_input_count)
    .with_fact("required_feature", "cugraph"))
}

#[cfg(feature = "cugraph")]
pub(super) fn execute_graph_algorithm(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &crate::plan::GraphAlgorithmNode,
    input: GpuDataFrame,
    side_inputs: BTreeMap<NodeId, GpuDataFrame>,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let input_rows = input.num_rows() as u64;
    let input_columns = input.num_columns() as u64;
    let boundary_before = metrics.boundary.clone();
    let timer = Instant::now();
    let executor = graph_algorithm_executor(node.spec.algorithm);
    let inputs = GraphAlgorithmExecutionInputs { edge_list: input, side_inputs };
    let result = executor(context, &node.spec, inputs)?;
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
pub(super) struct GraphAlgorithmExecutionResult {
    pub(super) frame: DataFrame,
    pub(super) algorithm_metadata: String,
}

#[cfg(feature = "cugraph")]
pub(super) struct GraphAlgorithmExecutionInputs {
    pub(super) edge_list: GpuDataFrame,
    pub(super) side_inputs: BTreeMap<NodeId, GpuDataFrame>,
}

#[cfg(feature = "cugraph")]
type GraphAlgorithmExecutor = fn(
    &ExecutionContext,
    &crate::plan::GraphAlgorithmSpec,
    GraphAlgorithmExecutionInputs,
) -> Result<GraphAlgorithmExecutionResult>;

#[cfg(feature = "cugraph")]
const GRAPH_ALGORITHM_EXECUTORS: &[(crate::plan::GraphAlgorithm, GraphAlgorithmExecutor)] = &[
    (crate::plan::GraphAlgorithm::PageRank, execute_pagerank),
    (crate::plan::GraphAlgorithm::Bfs, execute_bfs),
    (crate::plan::GraphAlgorithm::Louvain, execute_louvain),
    (crate::plan::GraphAlgorithm::EigenvectorCentrality, execute_eigenvector_centrality),
    (crate::plan::GraphAlgorithm::KatzCentrality, execute_katz_centrality),
    (crate::plan::GraphAlgorithm::Hits, execute_hits),
    (crate::plan::GraphAlgorithm::Sssp, execute_sssp),
    (crate::plan::GraphAlgorithm::DegreesAll, execute_degrees_all),
    (crate::plan::GraphAlgorithm::InDegreesAll, execute_in_degrees_all),
    (crate::plan::GraphAlgorithm::OutDegreesAll, execute_out_degrees_all),
    (crate::plan::GraphAlgorithm::WeaklyConnectedComponents, execute_weakly_connected_components),
    (crate::plan::GraphAlgorithm::StronglyConnectedComponents, execute_strongly_connected_components),
    (crate::plan::GraphAlgorithm::TriangleCountAll, execute_triangle_count_all),
    (crate::plan::GraphAlgorithm::CoreNumber, execute_core_number),
    (crate::plan::GraphAlgorithm::KCore, execute_k_core),
    (crate::plan::GraphAlgorithm::MinimumSpanningTree, execute_minimum_spanning_tree),
    (crate::plan::GraphAlgorithm::PageRankStrict, execute_pagerank_strict),
    (crate::plan::GraphAlgorithm::Leiden, execute_leiden),
    (crate::plan::GraphAlgorithm::Ecg, execute_ecg),
    (crate::plan::GraphAlgorithm::SpectralModularityMaximization, execute_spectral_modularity_maximization),
    (crate::plan::GraphAlgorithm::BalancedCutClustering, execute_balanced_cut_clustering),
];

#[cfg(feature = "cugraph")]
fn graph_algorithm_executor(algorithm: crate::plan::GraphAlgorithm) -> GraphAlgorithmExecutor {
    GRAPH_ALGORITHM_EXECUTORS
        .iter()
        .find(|(registered, _)| *registered == algorithm)
        .map(|(_, executor)| *executor)
        .unwrap_or_else(|| {
            panic!("graph algorithm executor table is missing variant {}", algorithm.as_str());
        })
}

/// Thread-affine cuGraph execution session.
///
/// Binds a single `cugraph::ResourceHandle` to a `cudf::Runtime` for the
/// duration of one executor task. The retained handle allows a future
/// multi-algorithm executor to call `enter` multiple times without rebuilding
/// the RAFT wrapper on each call.
///
/// **Do not store this in `ExecutionContext`.** `ResourceHandle` is `!Send +
/// !Sync`; placing it in a shared context would break the thread-affinity
/// contract. Create a session as a local variable, use it on the same thread,
/// and let it drop at the end of the task.
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
        let runtime = context.shared_runtime();
        let handle =
            cugraph::ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.session",
                "failed to bind cuGraph resource handle to the native cudf runtime",
            ))?;
        Ok(Self { runtime, handle })
    }

    fn enter(&self, operation: &'static str) -> Result<cugraph::EnteredContext<'_>> {
        self.handle.enter(operation).map_err(map_cugraph_error(
            "cudf_query_engine.graph_algorithm.enter",
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

/// Transposition layout as a value type, used in [`GraphCacheKey`].
#[cfg(feature = "cugraph")]
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum GraphOrientation {
    Forward,
    Transposed,
}

/// Stable key over all dimensions that determine a materialised cuGraph graph.
///
/// Two plan nodes that produce the same key from the same edge source could
/// share a single `cugraph::Graph` object. The type makes every reuse
/// dimension explicit and is the intended key for a future thread-affine
/// graph cache.
///
/// **Current executors do not cache on this key** — they rebuild the graph per
/// node, which is conservative and correct until edge-buffer identity and
/// GPU-resident graph lifetime management are implemented.
///
/// Graph reuse additionally requires:
/// - Stable edge-buffer identity (a pointer/generation pair that survives
///   column projection, or a content-addressable hash of device memory).
/// - A GPU-resident lifetime model: drop the cached graph when the source
///   `DataFrame` is released, or scope the cache to the executor task.
/// - Verification that no algorithm mutates the graph in place (check cuGraph
///   C API ownership semantics per algorithm).
#[cfg(feature = "cugraph")]
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GraphCacheKey {
    /// `Arc<cudf::Runtime>` data-pointer cast to `usize`, identifying the CUDA
    /// device and stream pair used to build the graph. Stored as `usize` so
    /// the key does not retain the runtime.
    ///
    /// **ABA risk**: this pointer is *not* a stable identifier across runtime
    /// drops. If the captured runtime is dropped and a new `Arc<cudf::Runtime>`
    /// is allocated at the same heap address, a stale key could falsely match
    /// the new runtime. A future cache must therefore scope its lifetime to the
    /// owning `ExecutionContext`'s `Arc<cudf::Runtime>` (e.g. cache lives in a
    /// thread-local on the executor task) so the captured pointer cannot be
    /// recycled while the cache is alive.
    runtime_ptr: usize,
    source_column: String,
    destination_column: String,
    weight_column: Option<String>,
    edge_id_column: Option<String>,
    construction_policy: &'static str,
    directed: bool,
    renumber: bool,
    orientation: GraphOrientation,
}

#[cfg(feature = "cugraph")]
#[allow(dead_code)]
impl GraphCacheKey {
    fn forward(context: &ExecutionContext, spec: &crate::plan::GraphAlgorithmSpec) -> Self {
        Self::new(context, spec, GraphOrientation::Forward)
    }

    fn transposed(context: &ExecutionContext, spec: &crate::plan::GraphAlgorithmSpec) -> Self {
        Self::new(context, spec, GraphOrientation::Transposed)
    }

    fn new(context: &ExecutionContext, spec: &crate::plan::GraphAlgorithmSpec, orientation: GraphOrientation) -> Self {
        Self {
            runtime_ptr: context.runtime() as *const cudf::Runtime as usize,
            source_column: spec.edge_columns.source.clone(),
            destination_column: spec.edge_columns.destination.clone(),
            weight_column: spec.edge_columns.weight.clone(),
            edge_id_column: spec.edge_columns.edge_id.clone(),
            construction_policy: spec.graph_options.policy.as_str(),
            directed: spec.graph_options.directed,
            renumber: spec.graph_options.renumber,
            orientation,
        }
    }
}

/// Prepares edges, creates a runtime-bound cuGraph handle, enters it, builds
/// the graph, and runs `f` with the cudf runtime, entered context, and graph.
///
/// All graph algorithm executors share this shape. Centralising it here keeps
/// `context.runtime()` as the authoritative execution domain for ingress,
/// cuGraph execution, and egress, and prevents the pattern from diverging as
/// more algorithms are added.
///
/// If a future path routes graph results into a different `cudf::Runtime`
/// stream, insert a device-side `cudaEventRecord` + `cudaStreamWaitEvent`
/// dependency before copying — not a host synchronize.
#[cfg(feature = "cugraph")]
pub(super) fn run_with_cugraph_graph<T, F, R>(
    context: &ExecutionContext,
    edges: DataFrame,
    options: cugraph::cudf_interop::TableGraphOptions,
    operation: &'static str,
    f: F,
) -> Result<R>
where
    T: cugraph::Transposition,
    F: FnOnce(&cudf::Runtime, &cugraph::EnteredContext<'_>, &mut cugraph::Graph<T>) -> Result<R>,
{
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
    let entered = handle.enter(operation).map_err(map_cugraph_error(
        "cudf_query_engine.graph_algorithm.enter",
        "failed to enter cuGraph execution context",
    ))?;
    let mut graph = cugraph::cudf_interop::graph_from_prepared_table_export::<T>(&entered, &export, &options).map_err(
        map_cugraph_error(
            "cudf_query_engine.graph_algorithm.build_graph",
            "failed to build cuGraph graph from edge-list table",
        ),
    )?;
    f(context.runtime(), &entered, &mut graph)
}

/// Build a cuGraph graph through `GraphBuilder` when the native edge list
/// carries stable edge ids. The cudf table-export path does not expose edge-id
/// binding, so this path borrows typed device slices for source, destination,
/// optional weight, and edge id columns directly for graph construction.
#[cfg(feature = "cugraph")]
pub(super) fn run_with_cugraph_graph_edge_ids<T, F, R>(
    context: &ExecutionContext,
    edges: &DataFrame,
    spec: &crate::plan::GraphAlgorithmSpec,
    operation: &'static str,
    f: F,
) -> Result<R>
where
    T: cugraph::Transposition,
    F: FnOnce(&cudf::Runtime, &cugraph::EnteredContext<'_>, &mut cugraph::Graph<T>) -> Result<R>,
{
    let handle = cugraph::ResourceHandle::for_runtime(context.shared_runtime()).map_err(map_cugraph_error(
        "cudf_query_engine.graph_algorithm.resource_handle",
        "failed to bind cuGraph resource handle to the native cudf runtime",
    ))?;
    let entered = handle.enter(operation).map_err(map_cugraph_error(
        "cudf_query_engine.graph_algorithm.enter",
        "failed to enter cuGraph execution context",
    ))?;
    let mut graph = build_graph_from_edge_id_columns::<T>(&entered, edges, spec)?;
    f(context.runtime(), &entered, &mut graph)
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
                "cudf_query_engine.graph_algorithm.edge_id_graph",
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
            "cudf_query_engine.graph_algorithm.edge_id_graph",
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
            "cudf_query_engine.graph_algorithm.edge_id_graph",
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

    match spec.edge_columns.weight.as_deref() {
        Some(weight_col) => {
            let weight = graph_series(edges, weight_col, "weight")?;
            let weight_dtype = weight.dtype().map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.graph_algorithm.edge_id_graph",
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
                            "cudf_query_engine.graph_algorithm.edge_id_graph",
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
                            "cudf_query_engine.graph_algorithm.edge_id_graph",
                            "failed to build cuGraph graph with stable edge ids",
                        ))
                },
                _ => Err(QueryError::new(
                    QueryErrorCode::UnsupportedType,
                    "cudf_query_engine.graph_algorithm.edge_id_graph",
                    "graph weight column dtype is unsupported",
                )
                .with_fact("weight_col", weight_col)
                .with_fact("weight_dtype", format!("{weight_dtype:?}"))),
            }
        },
        None => graph_builder_base(spec, &src_view, &dst_view).edge_ids(&edge_id_view).build(entered).map_err(
            map_cugraph_error(
                "cudf_query_engine.graph_algorithm.edge_id_graph",
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
fn graph_series(edges: &DataFrame, column: &str, role: &'static str) -> Result<cudf::Series> {
    edges.series(column).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.graph_algorithm.edge_id_graph",
            "failed to access graph construction column",
            error,
        )
        .with_fact("role", role)
        .with_fact("column", column)
    })
}

#[cfg(feature = "cugraph")]
fn graph_device_slice<'a, T: cudf::CudfType>(
    series: &'a cudf::Series,
    role: &'static str,
) -> Result<cudf::core::DeviceSlice<'a>> {
    series.as_column().device_slice::<T>().map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.graph_algorithm.edge_id_graph",
            "failed to borrow graph construction column device slice",
            error,
        )
        .with_fact("role", role)
    })
}

#[cfg(feature = "cugraph")]
fn graph_device_view<'a, T: cugraph::types::CugraphElement>(
    slice: &'a cudf::core::DeviceSlice<'a>,
    role: &'static str,
) -> Result<cugraph::DeviceArrayView<'a, T>> {
    cugraph::cudf_interop::device_array_view_from_slice::<T>(slice)
        .map_err(map_cugraph_error(
            "cudf_query_engine.graph_algorithm.edge_id_graph",
            "failed to wrap graph construction column as a cuGraph device-array view",
        ))
        .map_err(|error| error.with_fact("role", role))
}

#[cfg(feature = "cugraph")]
#[derive(Debug, Clone, Copy)]
enum EdgeColumnRole {
    Vertex,
    Weight,
    EdgeId,
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
    if let Some(edge_id) = &spec.edge_columns.edge_id {
        let edge_id_type = validate_edge_column(frame, edge_id, EdgeColumnRole::EdgeId)?;
        if edge_id_type.id() != source_type.id() {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                "cudf_query_engine.graph_algorithm.validate_edge_column",
                "cuGraph edge-id column dtype must match the graph vertex dtype",
            )
            .with_fact("edge_id_col", edge_id)
            .with_fact("edge_id_cudf_dtype", format!("{edge_id_type:?}"))
            .with_fact("vertex_cudf_dtype", format!("{source_type:?}")));
        }
    }

    let mut selected = vec![spec.edge_columns.source.as_str(), spec.edge_columns.destination.as_str()];
    if let Some(weight) = &spec.edge_columns.weight {
        selected.push(weight.as_str());
    }
    if let Some(edge_id) = &spec.edge_columns.edge_id {
        selected.push(edge_id.as_str());
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
        .with_fact("edge_id_col", spec.edge_columns.edge_id.as_deref().unwrap_or("<none>"))
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
        EdgeColumnRole::EdgeId => matches!(dtype.id(), cudf::TypeId::Int32 | cudf::TypeId::Int64),
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
            EdgeColumnRole::EdgeId => "edge_id",
        },
    )
    .with_fact("cudf_dtype", format!("{dtype:?}")))
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
        crate::plan::GraphAlgorithmParams::EigenvectorCentrality(params) => {
            format!("epsilon={},max_iterations={}", params.epsilon, params.max_iterations)
        },
        crate::plan::GraphAlgorithmParams::KatzCentrality(params) => format!(
            "alpha={},beta={},epsilon={},max_iterations={}",
            params.alpha, params.beta, params.epsilon, params.max_iterations,
        ),
        crate::plan::GraphAlgorithmParams::Hits(params) => format!(
            "epsilon={},max_iterations={},normalize={}",
            params.epsilon, params.max_iterations, params.normalize,
        ),
        crate::plan::GraphAlgorithmParams::Sssp(params) => format!(
            "source_vertex={},cutoff={}",
            params.source_vertex,
            params.cutoff.map_or_else(|| "none".to_string(), |v| v.to_string()),
        ),
        crate::plan::GraphAlgorithmParams::DegreesAll(_) => String::new(),
        crate::plan::GraphAlgorithmParams::InDegreesAll(_) => String::new(),
        crate::plan::GraphAlgorithmParams::OutDegreesAll(_) => String::new(),
        crate::plan::GraphAlgorithmParams::WeaklyConnectedComponents(_) => String::new(),
        crate::plan::GraphAlgorithmParams::StronglyConnectedComponents(_) => String::new(),
        crate::plan::GraphAlgorithmParams::TriangleCountAll(_) => String::new(),
        crate::plan::GraphAlgorithmParams::CoreNumber(params) => {
            format!("degree_type={}", params.degree_type.as_str())
        },
        crate::plan::GraphAlgorithmParams::KCore(params) => {
            format!("k={},degree_type={}", params.k, params.degree_type.as_str())
        },
        crate::plan::GraphAlgorithmParams::MinimumSpanningTree(_) => String::new(),
        crate::plan::GraphAlgorithmParams::PageRankStrict(params) => {
            format!("alpha={},epsilon={},max_iterations={}", params.alpha, params.epsilon, params.max_iterations)
        },
        crate::plan::GraphAlgorithmParams::Leiden(params) => format!(
            "max_level={},resolution={},theta={},seed={}",
            params.max_level, params.resolution, params.theta, params.seed,
        ),
        crate::plan::GraphAlgorithmParams::Ecg(params) => format!(
            "min_weight={},ensemble_size={},max_level={},threshold={},resolution={},seed={}",
            params.min_weight, params.ensemble_size, params.max_level, params.threshold, params.resolution, params.seed,
        ),
        crate::plan::GraphAlgorithmParams::SpectralModularityMaximization(params) => format!(
            "n_clusters={},n_eigenvectors={},evs_tolerance={},evs_max_iterations={},\
             k_means_tolerance={},k_means_max_iterations={},seed={}",
            params.n_clusters,
            params.n_eigenvectors,
            params.evs_tolerance,
            params.evs_max_iterations,
            params.k_means_tolerance,
            params.k_means_max_iterations,
            params.seed,
        ),
        crate::plan::GraphAlgorithmParams::BalancedCutClustering(params) => format!(
            "n_clusters={},n_eigenvectors={},evs_tolerance={},evs_max_iterations={},\
             k_means_tolerance={},k_means_max_iterations={},seed={}",
            params.n_clusters,
            params.n_eigenvectors,
            params.evs_tolerance,
            params.evs_max_iterations,
            params.k_means_tolerance,
            params.k_means_max_iterations,
            params.seed,
        ),
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
pub(super) fn degree_type_to_cugraph(
    degree_type: crate::plan::GraphKCoreDegreeType,
) -> cugraph::structure::KCoreDegreeType {
    match degree_type {
        crate::plan::GraphKCoreDegreeType::In => cugraph::structure::KCoreDegreeType::In,
        crate::plan::GraphKCoreDegreeType::Out => cugraph::structure::KCoreDegreeType::Out,
        crate::plan::GraphKCoreDegreeType::InOut => cugraph::structure::KCoreDegreeType::InOut,
    }
}

#[cfg(feature = "cugraph")]
pub(super) fn normalize_graph_result_schema(mut frame: DataFrame, schema: &SchemaRef) -> Result<DataFrame> {
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
pub(super) fn map_cugraph_error(
    operation: &'static str,
    message: &'static str,
) -> impl FnOnce(cugraph::Error) -> QueryError {
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
