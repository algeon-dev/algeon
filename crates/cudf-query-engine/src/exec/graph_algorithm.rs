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
    let executor = graph_algorithm_executor(node.spec.algorithm);
    let result = executor(context, &node.spec, input)?;
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
type GraphAlgorithmExecutor =
    fn(&ExecutionContext, &crate::plan::GraphAlgorithmSpec, GpuDataFrame) -> Result<GraphAlgorithmExecutionResult>;

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
fn run_with_cugraph_graph<T, F, R>(
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
    let options = table_graph_options(spec);
    let pr_params = pagerank_params(params)?;

    run_with_cugraph_graph::<cugraph::Transposed, _, _>(
        context,
        edges,
        options,
        "datafusion_cudf_cugraph_pagerank",
        |runtime, entered, graph| {
            let result = graph
                .pagerank(entered, &pr_params)
                .map_err(map_cugraph_error("cudf_query_engine.graph_algorithm.pagerank", "cuGraph PageRank failed"))?;
            let converged = result.converged().map_or_else(|_| "unknown".to_string(), |value| value.to_string());
            let algorithm_metadata = format!("converged={},iterations={}", converged, result.num_iterations());
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.pagerank_result",
                "failed to convert cuGraph PageRank result to cudf DataFrame",
            ))?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata,
            })
        },
    )
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
    let options = table_graph_options(spec);
    let bfs_p = bfs_params(params)?;

    run_with_cugraph_graph::<cugraph::Forward, _, _>(
        context,
        edges,
        options,
        "datafusion_cudf_cugraph_bfs",
        |runtime, entered, graph| {
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
                    let sources = cugraph::DeviceArray::from_slice(entered, &[source]).map_err(map_cugraph_error(
                        "cudf_query_engine.graph_algorithm.bfs_source",
                        "failed to copy BFS source vertex to device memory",
                    ))?;
                    let sources_view = sources.view().map_err(map_cugraph_error(
                        "cudf_query_engine.graph_algorithm.bfs_source",
                        "failed to create BFS source vertex view",
                    ))?;
                    let result = graph
                        .bfs(entered, &sources_view, &bfs_p)
                        .map_err(map_cugraph_error("cudf_query_engine.graph_algorithm.bfs", "cuGraph BFS failed"))?;
                    result.to_dataframe(runtime).map_err(map_cugraph_error(
                        "cudf_query_engine.graph_algorithm.bfs_result",
                        "failed to convert cuGraph BFS result to cudf DataFrame",
                    ))?
                },
                cudf::TypeId::Int64 => {
                    let sources = cugraph::DeviceArray::from_slice(entered, &[params.source_vertex]).map_err(
                        map_cugraph_error(
                            "cudf_query_engine.graph_algorithm.bfs_source",
                            "failed to copy BFS source vertex to device memory",
                        ),
                    )?;
                    let sources_view = sources.view().map_err(map_cugraph_error(
                        "cudf_query_engine.graph_algorithm.bfs_source",
                        "failed to create BFS source vertex view",
                    ))?;
                    let result = graph
                        .bfs(entered, &sources_view, &bfs_p)
                        .map_err(map_cugraph_error("cudf_query_engine.graph_algorithm.bfs", "cuGraph BFS failed"))?;
                    result.to_dataframe(runtime).map_err(map_cugraph_error(
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
        },
    )
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
    let options = table_graph_options(spec);
    let lv_params = louvain_params(params)?;

    run_with_cugraph_graph::<cugraph::Forward, _, _>(
        context,
        edges,
        options,
        "datafusion_cudf_cugraph_louvain",
        |runtime, entered, graph| {
            let result = graph
                .louvain(entered, &lv_params)
                .map_err(map_cugraph_error("cudf_query_engine.graph_algorithm.louvain", "cuGraph Louvain failed"))?;
            let algorithm_metadata = format!("modularity={}", result.modularity());
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.louvain_result",
                "failed to convert cuGraph Louvain result to cudf DataFrame",
            ))?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata,
            })
        },
    )
}

#[cfg(feature = "cugraph")]
fn execute_eigenvector_centrality(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    input: GpuDataFrame,
) -> Result<GraphAlgorithmExecutionResult> {
    let crate::plan::GraphAlgorithmParams::EigenvectorCentrality(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.graph_algorithm.eigenvector_centrality",
            "EigenvectorCentrality graph algorithm node carried non-EigenvectorCentrality parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);
    let evc_params = eigenvector_centrality_params(params)?;

    run_with_cugraph_graph::<cugraph::Transposed, _, _>(
        context,
        edges,
        options,
        "datafusion_cudf_cugraph_eigenvector_centrality",
        |runtime, entered, graph| {
            let result = graph.eigenvector_centrality(entered, &evc_params).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.eigenvector_centrality",
                "cuGraph eigenvector centrality failed",
            ))?;
            let converged = result.converged().map_or_else(|_| "unknown".to_string(), |value| value.to_string());
            let algorithm_metadata = format!("converged={},iterations={}", converged, result.num_iterations());
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.eigenvector_centrality_result",
                "failed to convert cuGraph eigenvector centrality result to cudf DataFrame",
            ))?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata,
            })
        },
    )
}

#[cfg(feature = "cugraph")]
fn execute_hits(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    input: GpuDataFrame,
) -> Result<GraphAlgorithmExecutionResult> {
    let crate::plan::GraphAlgorithmParams::Hits(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.graph_algorithm.hits",
            "HITS graph algorithm node carried non-HITS parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);
    let h_params = hits_params(params)?;

    run_with_cugraph_graph::<cugraph::Transposed, _, _>(
        context,
        edges,
        options,
        "datafusion_cudf_cugraph_hits",
        |runtime, entered, graph| {
            let result = graph
                .hits(entered, &h_params)
                .map_err(map_cugraph_error("cudf_query_engine.graph_algorithm.hits", "cuGraph HITS failed"))?;
            let algorithm_metadata = format!(
                "iterations={},hub_score_differences={}",
                result.number_of_iterations(),
                result.hub_score_differences(),
            );
            // Upstream HITS emits `vertex` / `hub` / `authority`; the stable
            // SQL schema renames the score columns to `hub_score` /
            // `authority_score` so the meaning is unambiguous at query time.
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.hits_result",
                "failed to convert cuGraph HITS result to cudf DataFrame",
            ))?;
            let frame =
                frame.rename_columns([("hub", "hub_score"), ("authority", "authority_score")]).map_err(|error| {
                    QueryError::from_cudf(
                        "cudf_query_engine.graph_algorithm.hits_result",
                        "failed to rename HITS score columns to the stable SQL schema",
                        error,
                    )
                })?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata,
            })
        },
    )
}

#[cfg(feature = "cugraph")]
fn execute_katz_centrality(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    input: GpuDataFrame,
) -> Result<GraphAlgorithmExecutionResult> {
    let crate::plan::GraphAlgorithmParams::KatzCentrality(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.graph_algorithm.katz_centrality",
            "KatzCentrality graph algorithm node carried non-KatzCentrality parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);
    let kc_params = katz_centrality_params(params)?;

    run_with_cugraph_graph::<cugraph::Transposed, _, _>(
        context,
        edges,
        options,
        "datafusion_cudf_cugraph_katz_centrality",
        |runtime, entered, graph| {
            let result = graph.katz_centrality(entered, &kc_params).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.katz_centrality",
                "cuGraph Katz centrality failed",
            ))?;
            let converged = result.converged().map_or_else(|_| "unknown".to_string(), |value| value.to_string());
            let algorithm_metadata = format!("converged={},iterations={}", converged, result.num_iterations());
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.katz_centrality_result",
                "failed to convert cuGraph Katz centrality result to cudf DataFrame",
            ))?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata,
            })
        },
    )
}

#[cfg(feature = "cugraph")]
fn execute_sssp(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    input: GpuDataFrame,
) -> Result<GraphAlgorithmExecutionResult> {
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

#[cfg(feature = "cugraph")]
fn execute_degrees_all(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    input: GpuDataFrame,
) -> Result<GraphAlgorithmExecutionResult> {
    let crate::plan::GraphAlgorithmParams::DegreesAll(_) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.graph_algorithm.degrees_all",
            "DegreesAll graph algorithm node carried non-DegreesAll parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);

    run_with_cugraph_graph::<cugraph::Transposed, _, _>(
        context,
        edges,
        options,
        "datafusion_cudf_cugraph_degrees_all",
        |runtime, entered, graph| {
            let result = graph.degrees_all(entered).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.degrees_all",
                "cuGraph degrees_all failed",
            ))?;
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.degrees_all_result",
                "failed to convert cuGraph degrees result to cudf DataFrame",
            ))?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata: String::new(),
            })
        },
    )
}

#[cfg(feature = "cugraph")]
fn execute_in_degrees_all(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    input: GpuDataFrame,
) -> Result<GraphAlgorithmExecutionResult> {
    let crate::plan::GraphAlgorithmParams::InDegreesAll(_) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.graph_algorithm.in_degrees_all",
            "InDegreesAll graph algorithm node carried non-InDegreesAll parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);

    run_with_cugraph_graph::<cugraph::Transposed, _, _>(
        context,
        edges,
        options,
        "datafusion_cudf_cugraph_in_degrees_all",
        |runtime, entered, graph| {
            let result = graph.in_degrees_all(entered).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.in_degrees_all",
                "cuGraph in_degrees_all failed",
            ))?;
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.in_degrees_all_result",
                "failed to convert cuGraph in-degrees result to cudf DataFrame",
            ))?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata: String::new(),
            })
        },
    )
}

#[cfg(feature = "cugraph")]
fn execute_out_degrees_all(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    input: GpuDataFrame,
) -> Result<GraphAlgorithmExecutionResult> {
    let crate::plan::GraphAlgorithmParams::OutDegreesAll(_) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.graph_algorithm.out_degrees_all",
            "OutDegreesAll graph algorithm node carried non-OutDegreesAll parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);

    // out_degrees_all is layout-agnostic in cugraph-rs; build Forward for
    // consistency with other Any-layout algorithms in this adapter.
    run_with_cugraph_graph::<cugraph::Forward, _, _>(
        context,
        edges,
        options,
        "datafusion_cudf_cugraph_out_degrees_all",
        |runtime, entered, graph| {
            let result = graph.out_degrees_all(entered).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.out_degrees_all",
                "cuGraph out_degrees_all failed",
            ))?;
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.out_degrees_all_result",
                "failed to convert cuGraph out-degrees result to cudf DataFrame",
            ))?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata: String::new(),
            })
        },
    )
}

#[cfg(feature = "cugraph")]
fn execute_weakly_connected_components(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    input: GpuDataFrame,
) -> Result<GraphAlgorithmExecutionResult> {
    let crate::plan::GraphAlgorithmParams::WeaklyConnectedComponents(_) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.graph_algorithm.weakly_connected_components",
            "WeaklyConnectedComponents graph algorithm node carried wrong parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);

    run_with_cugraph_graph::<cugraph::Forward, _, _>(
        context,
        edges,
        options,
        "datafusion_cudf_cugraph_weakly_connected_components",
        |runtime, entered, graph| {
            let wcc_params = cugraph::labeling::ConnectedComponentsParams::default();
            let result = graph.weakly_connected_components(entered, &wcc_params).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.weakly_connected_components",
                "cuGraph weakly_connected_components failed",
            ))?;
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.weakly_connected_components_result",
                "failed to convert cuGraph WCC result to cudf DataFrame",
            ))?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata: String::new(),
            })
        },
    )
}

#[cfg(feature = "cugraph")]
fn execute_strongly_connected_components(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    input: GpuDataFrame,
) -> Result<GraphAlgorithmExecutionResult> {
    let crate::plan::GraphAlgorithmParams::StronglyConnectedComponents(_) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.graph_algorithm.strongly_connected_components",
            "StronglyConnectedComponents graph algorithm node carried wrong parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);

    run_with_cugraph_graph::<cugraph::Forward, _, _>(
        context,
        edges,
        options,
        "datafusion_cudf_cugraph_strongly_connected_components",
        |runtime, entered, graph| {
            let scc_params = cugraph::labeling::ConnectedComponentsParams::default();
            let result = graph.strongly_connected_components(entered, &scc_params).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.strongly_connected_components",
                "cuGraph strongly_connected_components failed",
            ))?;
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.strongly_connected_components_result",
                "failed to convert cuGraph SCC result to cudf DataFrame",
            ))?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata: String::new(),
            })
        },
    )
}

#[cfg(feature = "cugraph")]
fn execute_triangle_count_all(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    input: GpuDataFrame,
) -> Result<GraphAlgorithmExecutionResult> {
    let crate::plan::GraphAlgorithmParams::TriangleCountAll(_) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.graph_algorithm.triangle_count_all",
            "TriangleCountAll graph algorithm node carried wrong parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);

    run_with_cugraph_graph::<cugraph::Forward, _, _>(
        context,
        edges,
        options,
        "datafusion_cudf_cugraph_triangle_count_all",
        |runtime, entered, graph| {
            let tc_params = cugraph::community::TriangleCountParams::default();
            let result = graph.triangle_count_all(entered, &tc_params).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.triangle_count_all",
                "cuGraph triangle_count_all failed",
            ))?;
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.triangle_count_all_result",
                "failed to convert cuGraph triangle-count result to cudf DataFrame",
            ))?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata: String::new(),
            })
        },
    )
}

#[cfg(feature = "cugraph")]
fn execute_core_number(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    input: GpuDataFrame,
) -> Result<GraphAlgorithmExecutionResult> {
    let crate::plan::GraphAlgorithmParams::CoreNumber(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.graph_algorithm.core_number",
            "CoreNumber graph algorithm node carried non-CoreNumber parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);
    let cn_params = cugraph::structure::CoreNumberParams {
        degree_type: degree_type_to_cugraph(params.degree_type),
        do_expensive_check: false,
    };

    run_with_cugraph_graph::<cugraph::Forward, _, _>(
        context,
        edges,
        options,
        "datafusion_cudf_cugraph_core_number",
        |runtime, entered, graph| {
            let result = graph.core_number(entered, &cn_params).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.core_number",
                "cuGraph core_number failed",
            ))?;
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.core_number_result",
                "failed to convert cuGraph core-number result to cudf DataFrame",
            ))?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata: format!("degree_type={}", params.degree_type.as_str()),
            })
        },
    )
}

#[cfg(feature = "cugraph")]
fn execute_k_core(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    input: GpuDataFrame,
) -> Result<GraphAlgorithmExecutionResult> {
    let crate::plan::GraphAlgorithmParams::KCore(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.graph_algorithm.k_core",
            "KCore graph algorithm node carried non-KCore parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);
    let kc_params = k_core_params(params)?;

    run_with_cugraph_graph::<cugraph::Forward, _, _>(
        context,
        edges,
        options,
        "datafusion_cudf_cugraph_k_core",
        |runtime, entered, graph| {
            let result = graph
                .k_core(entered, &kc_params)
                .map_err(map_cugraph_error("cudf_query_engine.graph_algorithm.k_core", "cuGraph k_core failed"))?;
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.k_core_result",
                "failed to convert cuGraph k-core result to cudf DataFrame",
            ))?;
            // cudf_interop produces "source" / "destination"; stable schema uses "src" / "dst".
            let frame = frame.rename_columns([("source", "src"), ("destination", "dst")]).map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.graph_algorithm.k_core_result",
                    "failed to rename k-core edge columns to the stable SQL schema",
                    error,
                )
            })?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata: format!("k={},degree_type={}", params.k, params.degree_type.as_str()),
            })
        },
    )
}

#[cfg(feature = "cugraph")]
fn execute_minimum_spanning_tree(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    input: GpuDataFrame,
) -> Result<GraphAlgorithmExecutionResult> {
    let crate::plan::GraphAlgorithmParams::MinimumSpanningTree(_) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.graph_algorithm.minimum_spanning_tree",
            "MinimumSpanningTree graph algorithm node carried wrong parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);

    run_with_cugraph_graph::<cugraph::Forward, _, _>(
        context,
        edges,
        options,
        "datafusion_cudf_cugraph_minimum_spanning_tree",
        |runtime, entered, graph| {
            let mst_params = cugraph::tree::MinimumSpanningTreeParams::default();
            let result = graph.minimum_spanning_tree(entered, &mst_params).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.minimum_spanning_tree",
                "cuGraph minimum_spanning_tree failed",
            ))?;
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.minimum_spanning_tree_result",
                "failed to convert cuGraph MST result to cudf DataFrame",
            ))?;
            // cudf_interop produces "source" / "destination"; stable schema uses "src" / "dst".
            let frame = frame.rename_columns([("source", "src"), ("destination", "dst")]).map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.graph_algorithm.minimum_spanning_tree_result",
                    "failed to rename MST edge columns to the stable SQL schema",
                    error,
                )
            })?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata: String::new(),
            })
        },
    )
}

#[cfg(feature = "cugraph")]
fn execute_pagerank_strict(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    input: GpuDataFrame,
) -> Result<GraphAlgorithmExecutionResult> {
    let crate::plan::GraphAlgorithmParams::PageRankStrict(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.graph_algorithm.pagerank_strict",
            "PageRankStrict graph algorithm node carried non-PageRankStrict parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);
    let pr_params = pagerank_params(params)?;

    run_with_cugraph_graph::<cugraph::Transposed, _, _>(
        context,
        edges,
        options,
        "datafusion_cudf_cugraph_pagerank_strict",
        |runtime, entered, graph| {
            let result = graph.pagerank_strict(entered, &pr_params).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.pagerank_strict",
                "cuGraph PageRank strict failed",
            ))?;
            let converged = result.converged().map_or_else(|_| "unknown".to_string(), |value| value.to_string());
            let algorithm_metadata = format!("converged={},iterations={}", converged, result.num_iterations());
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.pagerank_strict_result",
                "failed to convert cuGraph PageRank strict result to cudf DataFrame",
            ))?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata,
            })
        },
    )
}

#[cfg(feature = "cugraph")]
fn execute_leiden(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    input: GpuDataFrame,
) -> Result<GraphAlgorithmExecutionResult> {
    let crate::plan::GraphAlgorithmParams::Leiden(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.graph_algorithm.leiden",
            "Leiden graph algorithm node carried non-Leiden parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);
    let leiden_p = leiden_params(params)?;
    let seed = params.seed;

    run_with_cugraph_graph::<cugraph::Forward, _, _>(
        context,
        edges,
        options,
        "datafusion_cudf_cugraph_leiden",
        |runtime, entered, graph| {
            let mut rng_state = cugraph::random::RngState::new(entered, seed).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.leiden_rng",
                "failed to create cuGraph RNG state for Leiden",
            ))?;
            let result = graph
                .leiden(entered, &mut rng_state, &leiden_p)
                .map_err(map_cugraph_error("cudf_query_engine.graph_algorithm.leiden", "cuGraph Leiden failed"))?;
            let algorithm_metadata = format!("modularity={},seed={}", result.modularity(), seed);
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.leiden_result",
                "failed to convert cuGraph Leiden result to cudf DataFrame",
            ))?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata,
            })
        },
    )
}

#[cfg(feature = "cugraph")]
fn execute_ecg(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    input: GpuDataFrame,
) -> Result<GraphAlgorithmExecutionResult> {
    let crate::plan::GraphAlgorithmParams::Ecg(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.graph_algorithm.ecg",
            "ECG graph algorithm node carried non-ECG parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);
    let ecg_p = ecg_params(params)?;
    let seed = params.seed;

    run_with_cugraph_graph::<cugraph::Forward, _, _>(
        context,
        edges,
        options,
        "datafusion_cudf_cugraph_ecg",
        |runtime, entered, graph| {
            let mut rng_state = cugraph::random::RngState::new(entered, seed).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.ecg_rng",
                "failed to create cuGraph RNG state for ECG",
            ))?;
            let result = graph
                .ecg(entered, &mut rng_state, &ecg_p)
                .map_err(map_cugraph_error("cudf_query_engine.graph_algorithm.ecg", "cuGraph ECG failed"))?;
            let algorithm_metadata = format!("modularity={},seed={}", result.modularity(), seed);
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.ecg_result",
                "failed to convert cuGraph ECG result to cudf DataFrame",
            ))?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata,
            })
        },
    )
}

#[cfg(feature = "cugraph")]
fn execute_spectral_modularity_maximization(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    input: GpuDataFrame,
) -> Result<GraphAlgorithmExecutionResult> {
    let crate::plan::GraphAlgorithmParams::SpectralModularityMaximization(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.graph_algorithm.spectral_modularity_maximization",
            "SpectralModularityMaximization graph algorithm node carried wrong parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);
    let spectral_p = spectral_clustering_params(params)?;
    let (n_clusters, seed) = (params.n_clusters, params.seed);

    run_with_cugraph_graph::<cugraph::Forward, _, _>(
        context,
        edges,
        options,
        "datafusion_cudf_cugraph_spectral_modularity_maximization",
        |runtime, entered, graph| {
            let mut rng_state = cugraph::random::RngState::new(entered, seed).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.spectral_modularity_maximization_rng",
                "failed to create cuGraph RNG state for SpectralModularityMaximization",
            ))?;
            let result = graph.spectral_modularity_maximization(entered, &mut rng_state, &spectral_p).map_err(
                map_cugraph_error(
                    "cudf_query_engine.graph_algorithm.spectral_modularity_maximization",
                    "cuGraph spectral_modularity_maximization failed",
                ),
            )?;
            let algorithm_metadata = format!("n_clusters={},seed={}", n_clusters, seed);
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.spectral_modularity_maximization_result",
                "failed to convert cuGraph SpectralModularityMaximization result to cudf DataFrame",
            ))?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata,
            })
        },
    )
}

#[cfg(feature = "cugraph")]
#[allow(deprecated)]
fn execute_balanced_cut_clustering(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    input: GpuDataFrame,
) -> Result<GraphAlgorithmExecutionResult> {
    let crate::plan::GraphAlgorithmParams::BalancedCutClustering(params) = &spec.params else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.graph_algorithm.balanced_cut_clustering",
            "BalancedCutClustering graph algorithm node carried wrong parameters",
        )
        .with_fact("algorithm", spec.algorithm.as_str()));
    };

    let input = input.into_frame();
    let (edges, _) = project_edge_list(&input, spec)?;
    let options = table_graph_options(spec);
    let spectral_p = spectral_clustering_params(params)?;
    let (n_clusters, seed) = (params.n_clusters, params.seed);

    run_with_cugraph_graph::<cugraph::Forward, _, _>(
        context,
        edges,
        options,
        "datafusion_cudf_cugraph_balanced_cut_clustering",
        |runtime, entered, graph| {
            let mut rng_state = cugraph::random::RngState::new(entered, seed).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.balanced_cut_clustering_rng",
                "failed to create cuGraph RNG state for BalancedCutClustering",
            ))?;
            let result =
                graph.balanced_cut_clustering(entered, &mut rng_state, &spectral_p).map_err(map_cugraph_error(
                    "cudf_query_engine.graph_algorithm.balanced_cut_clustering",
                    "cuGraph balanced_cut_clustering failed",
                ))?;
            let algorithm_metadata = format!("n_clusters={},seed={}", n_clusters, seed);
            let frame = result.to_dataframe(runtime).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.balanced_cut_clustering_result",
                "failed to convert cuGraph BalancedCutClustering result to cudf DataFrame",
            ))?;
            Ok(GraphAlgorithmExecutionResult {
                frame: normalize_graph_result_schema(frame, &spec.output_schema)?,
                algorithm_metadata,
            })
        },
    )
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
fn hits_params(params: &crate::plan::GraphHitsParams) -> Result<cugraph::centrality::HitsParams> {
    Ok(cugraph::centrality::HitsParams {
        epsilon: cugraph::params::NonNegativeF64::new(params.epsilon).map_err(map_cugraph_error(
            "cudf_query_engine.graph_algorithm.hits_params",
            "invalid cuGraph HITS epsilon",
        ))?,
        max_iterations: cugraph::params::IterationCount::new(params.max_iterations).map_err(map_cugraph_error(
            "cudf_query_engine.graph_algorithm.hits_params",
            "invalid cuGraph HITS max_iterations",
        ))?,
        normalize: params.normalize,
        do_expensive_check: false,
    })
}

#[cfg(feature = "cugraph")]
fn katz_centrality_params(
    params: &crate::plan::GraphKatzCentralityParams,
) -> Result<cugraph::centrality::KatzCentralityParams> {
    Ok(cugraph::centrality::KatzCentralityParams {
        alpha: cugraph::params::NonNegativeF64::new(params.alpha).map_err(map_cugraph_error(
            "cudf_query_engine.graph_algorithm.katz_centrality_params",
            "invalid cuGraph Katz centrality alpha",
        ))?,
        beta: cugraph::params::NonNegativeF64::new(params.beta).map_err(map_cugraph_error(
            "cudf_query_engine.graph_algorithm.katz_centrality_params",
            "invalid cuGraph Katz centrality beta",
        ))?,
        epsilon: cugraph::params::NonNegativeF64::new(params.epsilon).map_err(map_cugraph_error(
            "cudf_query_engine.graph_algorithm.katz_centrality_params",
            "invalid cuGraph Katz centrality epsilon",
        ))?,
        max_iterations: cugraph::params::IterationCount::new(params.max_iterations).map_err(map_cugraph_error(
            "cudf_query_engine.graph_algorithm.katz_centrality_params",
            "invalid cuGraph Katz centrality max_iterations",
        ))?,
        do_expensive_check: false,
    })
}

#[cfg(feature = "cugraph")]
fn eigenvector_centrality_params(
    params: &crate::plan::GraphEigenvectorCentralityParams,
) -> Result<cugraph::centrality::EigenvectorCentralityParams> {
    Ok(cugraph::centrality::EigenvectorCentralityParams {
        epsilon: cugraph::params::NonNegativeF64::new(params.epsilon).map_err(map_cugraph_error(
            "cudf_query_engine.graph_algorithm.eigenvector_centrality_params",
            "invalid cuGraph eigenvector centrality epsilon",
        ))?,
        max_iterations: cugraph::params::IterationCount::new(params.max_iterations).map_err(map_cugraph_error(
            "cudf_query_engine.graph_algorithm.eigenvector_centrality_params",
            "invalid cuGraph eigenvector centrality max_iterations",
        ))?,
        do_expensive_check: false,
    })
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
    Ok(cugraph::community::LouvainParams {
        resolution: cugraph::params::PositiveF64::new(params.resolution).map_err(map_cugraph_error(
            "cudf_query_engine.graph_algorithm.louvain_params",
            "invalid cuGraph Louvain resolution",
        ))?,
        ..cugraph::community::LouvainParams::default()
    })
}

#[cfg(feature = "cugraph")]
fn leiden_params(params: &crate::plan::GraphLeidenParams) -> Result<cugraph::community::LeidenParams> {
    Ok(cugraph::community::LeidenParams {
        max_level: cugraph::params::IterationCount::new(params.max_level).map_err(map_cugraph_error(
            "cudf_query_engine.graph_algorithm.leiden_params",
            "invalid cuGraph Leiden max_level",
        ))?,
        resolution: cugraph::params::PositiveF64::new(params.resolution).map_err(map_cugraph_error(
            "cudf_query_engine.graph_algorithm.leiden_params",
            "invalid cuGraph Leiden resolution",
        ))?,
        theta: cugraph::params::PositiveF64::new(params.theta).map_err(map_cugraph_error(
            "cudf_query_engine.graph_algorithm.leiden_params",
            "invalid cuGraph Leiden theta",
        ))?,
        do_expensive_check: false,
    })
}

#[cfg(feature = "cugraph")]
fn ecg_params(params: &crate::plan::GraphEcgParams) -> Result<cugraph::community::EcgParams> {
    Ok(cugraph::community::EcgParams {
        min_weight: cugraph::params::NonNegativeF64::new(params.min_weight).map_err(map_cugraph_error(
            "cudf_query_engine.graph_algorithm.ecg_params",
            "invalid cuGraph ECG min_weight",
        ))?,
        ensemble_size: cugraph::params::IterationCount::new(params.ensemble_size).map_err(map_cugraph_error(
            "cudf_query_engine.graph_algorithm.ecg_params",
            "invalid cuGraph ECG ensemble_size",
        ))?,
        max_level: cugraph::params::IterationCount::new(params.max_level).map_err(map_cugraph_error(
            "cudf_query_engine.graph_algorithm.ecg_params",
            "invalid cuGraph ECG max_level",
        ))?,
        threshold: cugraph::params::NonNegativeF64::new(params.threshold).map_err(map_cugraph_error(
            "cudf_query_engine.graph_algorithm.ecg_params",
            "invalid cuGraph ECG threshold",
        ))?,
        resolution: cugraph::params::PositiveF64::new(params.resolution).map_err(map_cugraph_error(
            "cudf_query_engine.graph_algorithm.ecg_params",
            "invalid cuGraph ECG resolution",
        ))?,
        do_expensive_check: false,
    })
}

#[cfg(feature = "cugraph")]
fn spectral_clustering_params(
    params: &crate::plan::GraphSpectralClusteringParams,
) -> Result<cugraph::community::SpectralClusteringParams> {
    let n_clusters = std::num::NonZeroUsize::new(params.n_clusters as usize).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.graph_algorithm.spectral_clustering_params",
            "spectral clustering n_clusters must be at least 1",
        )
        .with_fact("n_clusters", params.n_clusters)
    })?;
    let n_eigenvectors = std::num::NonZeroUsize::new(params.n_eigenvectors as usize).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.graph_algorithm.spectral_clustering_params",
            "spectral clustering n_eigenvectors must be at least 1",
        )
        .with_fact("n_eigenvectors", params.n_eigenvectors)
    })?;
    Ok(cugraph::community::SpectralClusteringParams {
        n_clusters,
        n_eigenvectors,
        evs_tolerance: cugraph::params::NonNegativeF64::new(params.evs_tolerance).map_err(map_cugraph_error(
            "cudf_query_engine.graph_algorithm.spectral_clustering_params",
            "invalid cuGraph spectral clustering evs_tolerance",
        ))?,
        evs_max_iterations: cugraph::params::IterationCount::new(params.evs_max_iterations).map_err(
            map_cugraph_error(
                "cudf_query_engine.graph_algorithm.spectral_clustering_params",
                "invalid cuGraph spectral clustering evs_max_iterations",
            ),
        )?,
        k_means_tolerance: cugraph::params::NonNegativeF64::new(params.k_means_tolerance).map_err(
            map_cugraph_error(
                "cudf_query_engine.graph_algorithm.spectral_clustering_params",
                "invalid cuGraph spectral clustering k_means_tolerance",
            ),
        )?,
        k_means_max_iterations: cugraph::params::IterationCount::new(params.k_means_max_iterations).map_err(
            map_cugraph_error(
                "cudf_query_engine.graph_algorithm.spectral_clustering_params",
                "invalid cuGraph spectral clustering k_means_max_iterations",
            ),
        )?,
        do_expensive_check: false,
    })
}

#[cfg(feature = "cugraph")]
fn sssp_params(params: &crate::plan::GraphSsspParams) -> Result<cugraph::traversal::SsspParams> {
    let cutoff = params
        .cutoff
        .map(|value| {
            cugraph::params::NonNegativeF64::new(value).map_err(map_cugraph_error(
                "cudf_query_engine.graph_algorithm.sssp_params",
                "invalid cuGraph SSSP cutoff",
            ))
        })
        .transpose()?;
    let mut p = cugraph::traversal::SsspParams::default().with_predecessors();
    if let Some(cutoff) = cutoff {
        p = p.with_cutoff(cutoff);
    }
    Ok(p)
}

#[cfg(feature = "cugraph")]
fn degree_type_to_cugraph(degree_type: crate::plan::GraphKCoreDegreeType) -> cugraph::structure::KCoreDegreeType {
    match degree_type {
        crate::plan::GraphKCoreDegreeType::In => cugraph::structure::KCoreDegreeType::In,
        crate::plan::GraphKCoreDegreeType::Out => cugraph::structure::KCoreDegreeType::Out,
        crate::plan::GraphKCoreDegreeType::InOut => cugraph::structure::KCoreDegreeType::InOut,
    }
}

#[cfg(feature = "cugraph")]
fn k_core_params(params: &crate::plan::GraphKCoreParams) -> Result<cugraph::structure::KCoreParams> {
    let k = std::num::NonZeroUsize::new(params.k as usize).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.graph_algorithm.k_core_params",
            "KCore k must be at least 1",
        )
        .with_fact("k", params.k)
    })?;
    Ok(cugraph::structure::KCoreParams {
        k,
        degree_type: degree_type_to_cugraph(params.degree_type),
        do_expensive_check: false,
    })
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
