//! Breadth-First Search (BFS) traversal algorithm.

use std::ptr;

use cugraph_sys::{
    cugraph_bfs, cugraph_bfs_predicate_result_free, cugraph_bfs_predicate_result_get_target_distance,
    cugraph_bfs_predicate_result_get_target_found, cugraph_bfs_predicate_result_t, cugraph_bfs_with_predicates,
    cugraph_error_t, cugraph_paths_result_t, cugraph_type_erased_device_array_view_t,
};

use crate::{
    array::DeviceArrayView,
    error::{Error, ErrorKind, Result, check_error, require_non_null},
    graph::{Forward, Graph},
    handle::EnteredContext,
    params::BfsDepthLimit,
    structure::{
        VertexSelection,
        typed_results::{Distances, VertexIds},
    },
    traversal::paths::{ExtractPathsResult, PathsResult},
    types::{CugraphElement, to_bool_t, try_from_bool_t},
};

/// Parameters for the BFS traversal algorithm.
///
/// Like the other algorithm parameter types in this crate, this struct uses
/// public fields plus [`Default`] for the baseline configuration. The builder
/// methods remain available as shorthand for the common optional toggles.
///
/// # Example
///
/// ```
/// use cugraph::{params::BfsDepthLimit, traversal::BfsParams};
///
/// let params = BfsParams {
///     depth_limit: Some(BfsDepthLimit::new(3).unwrap()),
///     compute_predecessors: true,
///     ..Default::default()
/// };
/// assert!(params.compute_predecessors);
/// assert_eq!(params.depth_limit.unwrap().get(), 3);
/// ```
#[derive(Debug, Clone, Copy, Default)]
#[must_use]
pub struct BfsParams {
    /// Maximum BFS traversal depth. `None` means no depth limit.
    pub depth_limit: Option<BfsDepthLimit>,
    /// Whether predecessor vertices should be included in the result.
    pub compute_predecessors: bool,
    /// Whether to run expensive input validation checks in the C library.
    pub do_expensive_check: bool,
}

impl BfsParams {
    /// Enable computation of predecessor vertices in the BFS result.
    pub fn with_predecessors(mut self) -> Self {
        self.compute_predecessors = true;
        self
    }

    /// Set the maximum BFS traversal depth.
    pub fn with_depth_limit(mut self, limit: BfsDepthLimit) -> Self {
        self.depth_limit = Some(limit);
        self
    }

    /// Enable expensive input validation checks in the C library.
    pub fn with_expensive_check(mut self) -> Self {
        self.do_expensive_check = true;
        self
    }

    /// Whether predecessor vertices will be computed.
    pub fn compute_predecessors(&self) -> bool {
        self.compute_predecessors
    }

    /// The maximum BFS depth, or `None` for unlimited traversal.
    pub fn depth_limit(&self) -> Option<BfsDepthLimit> {
        self.depth_limit
    }

    /// Whether expensive input validation checks are enabled.
    pub fn do_expensive_check(&self) -> bool {
        self.do_expensive_check
    }
}

/// Concrete GPU-resident predicate inputs for predicate-aware BFS.
///
/// This type intentionally accepts only device array views. SQL expressions,
/// table names, logical-id mapping, and relation-backed edge alignment belong
/// in higher query-engine layers before they call into `cugraph-rs`.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct BfsPredicateParams<'a, V: CugraphElement, E: CugraphElement = i32> {
    /// Vertices allowed to be reached, using the graph's external vertex ID dtype.
    pub include_vertices: Option<&'a DeviceArrayView<'a, V>>,
    /// Vertices forbidden from being reached, using the graph's external vertex ID dtype.
    pub exclude_vertices: Option<&'a DeviceArrayView<'a, V>>,
    /// Target vertices used for target-hit metadata and optional early stop.
    pub target_vertices: Option<&'a DeviceArrayView<'a, V>>,
    /// External edge IDs allowed to be traversed. Requires a graph built with edge IDs.
    pub include_edge_ids: Option<&'a DeviceArrayView<'a, E>>,
    /// Stop after the first BFS depth where any target vertex is discovered.
    pub stop_when_any_target_found: bool,
}

impl<V: CugraphElement, E: CugraphElement> Default for BfsPredicateParams<'_, V, E> {
    fn default() -> Self {
        Self {
            include_vertices: None,
            exclude_vertices: None,
            target_vertices: None,
            include_edge_ids: None,
            stop_when_any_target_found: false,
        }
    }
}

/// Target-discovery metadata returned by predicate-aware BFS when target vertices are provided.
///
/// The current cuGraph predicate C API exposes a stable `target_found` flag and
/// minimum target distance. It deliberately does not expose a target vertex
/// because multiple targets can be discovered at the same BFS depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BfsTargetInfo {
    /// Whether any target vertex was discovered.
    pub target_found: bool,
    /// The first BFS depth containing a target vertex, or `None` if no target was reached.
    pub target_distance: Option<i64>,
}

/// Perform breadth-first search from one or more source vertices.
///
/// Prefer using [`Graph::bfs`](Graph::bfs) instead of calling
/// this function directly.
pub(crate) fn bfs<V: CugraphElement>(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<Forward>,
    sources: &DeviceArrayView<'_, V>,
    params: &BfsParams,
) -> Result<BfsResult> {
    let mut result: *mut cugraph_paths_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();

    // The C++ BFS defaults to std::numeric_limits<vertex_t>::max() for unlimited
    // depth. Since vertex_t is typically int32_t, we use i32::MAX.
    let limit = params.depth_limit.map(usize::from).unwrap_or(i32::MAX as usize);
    let graph_ptr = graph.as_ptr(handle, "bfs")?;

    // SAFETY: `graph` and `sources` stay valid for the duration of the call,
    // `sources` matches the graph vertex dtype by prior validation, and
    // `result`/`error` are valid out-pointers.
    unsafe {
        let code = cugraph_bfs(
            handle.handle_ptr(),
            graph_ptr,
            sources.as_ptr(),
            to_bool_t(false), // direction_optimizing
            limit,
            to_bool_t(params.compute_predecessors),
            to_bool_t(params.do_expensive_check),
            &mut result,
            &mut error,
        );
        check_error(code, error, "bfs")?;
    }

    // SAFETY: on success, cuGraph transfers ownership of the paths result to
    // the caller.
    let paths = unsafe { PathsResult::from_raw(result, handle.device(), "bfs") }?;

    Ok(BfsResult { paths, target_info: None })
}

fn optional_view_ptr<T: CugraphElement>(
    view: Option<&DeviceArrayView<'_, T>>,
) -> *const cugraph_type_erased_device_array_view_t {
    view.map(|view| view.as_ptr() as *const _).unwrap_or(ptr::null())
}

fn bfs_depth_limit(params: &BfsParams) -> usize {
    // The C++ BFS defaults to std::numeric_limits<vertex_t>::max() for unlimited
    // depth. Since vertex_t is typically int32_t, use i32::MAX as the existing
    // safe wrapper does for non-predicate BFS.
    params.depth_limit.map(usize::from).unwrap_or(i32::MAX as usize)
}

fn validate_predicates<V: CugraphElement, E: CugraphElement>(
    graph: &Graph<Forward>,
    predicates: &BfsPredicateParams<'_, V, E>,
) -> Result<()> {
    let operation = "Graph::bfs_with_predicates";

    if predicates.include_vertices.is_some() && predicates.exclude_vertices.is_some() {
        return Err(Error::new(ErrorKind::InvalidInput, operation)
            .with_message("include_vertices and exclude_vertices are mutually exclusive"));
    }

    if predicates.include_edge_ids.is_some() {
        if !graph.has_edge_ids() {
            return Err(Error::new(ErrorKind::InvalidInput, operation)
                .with_message("include_edge_ids requires a graph with edge IDs"));
        }

        let dtype = E::data_type();
        if !dtype.is_supported_index_type() {
            return Err(Error::new(ErrorKind::InvalidInput, operation)
                .with_message(format!("include_edge_ids dtype must be Int32 or Int64, got {dtype}")));
        }

        if let Some(edge_id_dtype) = graph.edge_id_dtype() {
            if dtype != edge_id_dtype {
                return Err(Error::new(ErrorKind::UnsupportedType, operation).with_message(format!(
                    "include_edge_ids dtype {dtype} does not match graph edge-id dtype {edge_id_dtype}"
                )));
            }
        }
    }

    Ok(())
}

struct BfsPredicateResult {
    raw: *mut cugraph_bfs_predicate_result_t,
}

impl BfsPredicateResult {
    unsafe fn from_raw(raw: *mut cugraph_bfs_predicate_result_t) -> Result<Self> {
        let raw = require_non_null(raw, "bfs_with_predicates", "predicate_result")?;
        Ok(Self { raw })
    }

    fn target_info(&self) -> Result<BfsTargetInfo> {
        // SAFETY: `self.raw` is a live predicate-result handle owned by this wrapper.
        let found = unsafe { cugraph_bfs_predicate_result_get_target_found(self.raw) };
        let target_found = try_from_bool_t(found).ok_or_else(|| {
            Error::new(ErrorKind::InvalidState, "bfs_with_predicates")
                .with_message(format!("invalid target_found bool_t value: {}", found.0))
        })?;

        if !target_found {
            return Ok(BfsTargetInfo { target_found, target_distance: None });
        }

        // SAFETY: `self.raw` is a live predicate-result handle owned by this wrapper.
        let distance = unsafe { cugraph_bfs_predicate_result_get_target_distance(self.raw) };
        let distance = i64::try_from(distance).map_err(|_| {
            Error::new(ErrorKind::InvalidState, "bfs_with_predicates")
                .with_message("target_distance does not fit in i64")
                .with_context("target_distance", distance.to_string())
        })?;

        Ok(BfsTargetInfo { target_found, target_distance: Some(distance) })
    }
}

impl Drop for BfsPredicateResult {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: `self.raw` is owned by this wrapper and freed at most once.
            unsafe { cugraph_bfs_predicate_result_free(self.raw) };
        }
    }
}

/// Perform breadth-first search with optional edge, vertex, and target predicates.
pub(crate) fn bfs_with_predicates<V: CugraphElement, E: CugraphElement>(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<Forward>,
    sources: &DeviceArrayView<'_, V>,
    params: &BfsParams,
    predicates: &BfsPredicateParams<'_, V, E>,
) -> Result<BfsResult> {
    validate_predicates(graph, predicates)?;

    let mut result: *mut cugraph_paths_result_t = ptr::null_mut();
    let mut predicate_result: *mut cugraph_bfs_predicate_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();

    let predicate_result_out: *mut *mut cugraph_bfs_predicate_result_t =
        if predicates.target_vertices.is_some() { &mut predicate_result } else { ptr::null_mut() };
    let graph_ptr = graph.as_ptr(handle, "bfs_with_predicates")?;

    // SAFETY: `graph`, `sources`, and all optional predicate views stay valid
    // for the duration of the call. The public graph method validated source
    // vertex dtype and this function validated Rust-side predicate invariants.
    unsafe {
        let code = cugraph_bfs_with_predicates(
            handle.handle_ptr(),
            graph_ptr,
            sources.as_ptr(),
            optional_view_ptr(predicates.include_vertices),
            optional_view_ptr(predicates.exclude_vertices),
            optional_view_ptr(predicates.target_vertices),
            optional_view_ptr(predicates.include_edge_ids),
            to_bool_t(false), // direction_optimizing
            bfs_depth_limit(params),
            to_bool_t(params.compute_predecessors),
            to_bool_t(predicates.stop_when_any_target_found),
            to_bool_t(params.do_expensive_check),
            &mut result,
            predicate_result_out,
            &mut error,
        );
        check_error(code, error, "bfs_with_predicates")?;
    }

    let predicate_result = if predicates.target_vertices.is_some() {
        // SAFETY: `predicate_result` is initialized on successful calls when
        // a predicate result out-pointer was supplied.
        Some(unsafe { BfsPredicateResult::from_raw(predicate_result) }?)
    } else {
        None
    };
    let target_info = predicate_result.as_ref().map(BfsPredicateResult::target_info).transpose()?;

    // SAFETY: on success, cuGraph transfers ownership of the paths result to
    // the caller.
    let paths = unsafe { PathsResult::from_raw(result, handle.device(), "bfs_with_predicates") }?;

    Ok(BfsResult { paths, target_info })
}

/// Result of a BFS computation.
///
/// Provides access to vertex ids, distances, and optionally predecessors.
/// Freed on drop.
pub struct BfsResult {
    pub(crate) paths: PathsResult,
    target_info: Option<BfsTargetInfo>,
}

impl std::fmt::Debug for BfsResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BfsResult").finish_non_exhaustive()
    }
}

impl BfsResult {
    /// Get the vertex ids as a runtime-typed device array view.
    ///
    /// Returns [`VertexIds::I32`] or [`VertexIds::I64`] depending on the
    /// graph's vertex type.
    pub fn vertex_ids(&self) -> Result<VertexIds<'_>> {
        self.paths.vertex_ids("BfsResult::vertex_ids")
    }

    /// Get the BFS distances as a runtime-typed device array view.
    ///
    /// Returns [`Distances::I32`] or [`Distances::I64`] depending on the
    /// graph's vertex type. BFS stores distances as integer hop counts
    /// (`vertex_t`).
    pub fn distances(&self) -> Result<Distances<'_>> {
        self.paths.distances("BfsResult::distances")
    }

    /// Get the predecessor vertices as a runtime-typed device array view.
    ///
    /// Returns `None` if `compute_predecessors` was `false` in the BFS call.
    /// cuGraph may signal that state with either a null pointer or a
    /// zero-length array view; both are normalized to `None`.
    /// Returns [`VertexIds::I32`] or [`VertexIds::I64`] depending on the
    /// graph's vertex type.
    pub fn predecessors(&self) -> Result<Option<VertexIds<'_>>> {
        self.paths.predecessors("BfsResult::predecessors")
    }

    /// Return target-discovery metadata for predicate-aware BFS calls.
    ///
    /// Non-predicate BFS and predicate BFS calls without `target_vertices`
    /// return `None`.
    pub fn target_info(&self) -> Option<&BfsTargetInfo> {
        self.target_info.as_ref()
    }

    /// Extract concrete row-major paths to the requested destination vertices.
    ///
    /// `sources` must match the source vertex set originally used to compute
    /// this BFS result. cuGraph requires this input again when extracting
    /// paths from the predecessor forest.
    pub fn extract_paths<'a, V: CugraphElement + 'a>(
        &self,
        handle: &EnteredContext<'_>,
        graph: &mut Graph<Forward>,
        sources: impl Into<VertexSelection<'a, V>>,
        destinations: impl Into<VertexSelection<'a, V>>,
    ) -> Result<ExtractPathsResult> {
        crate::traversal::paths::extract_paths(
            handle,
            graph,
            &self.paths,
            sources,
            destinations,
            "BfsResult::extract_paths",
        )
    }

    /// Convert this BFS result to a cudf DataFrame.
    ///
    /// Returns a DataFrame with columns:
    /// - `"vertex"` — vertex ids with the underlying result dtype (`Int32` or `Int64`)
    /// - `"distance"` — BFS distances with the underlying result dtype;
    ///   unreachable vertices use that integer dtype's max sentinel value
    /// - `"predecessor"` — predecessor vertex ids with the underlying result
    ///   dtype, only if predecessors were computed; the column is omitted
    ///   otherwise
    ///
    /// Device-to-device copy: unavoidable because the cugraph C API owns
    /// result memory and frees it on drop. Matches Python cudf/cugraph behavior.
    pub fn to_dataframe(&self, runtime: &cudf::Runtime) -> Result<cudf::DataFrame> {
        crate::cudf_interop::bfs_result_to_dataframe(self, runtime)
            .map_err(|e: Error| e.with_operation("BfsResult::to_dataframe"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ResourceHandle,
        array::DeviceArray,
        graph::{Forward, GraphBuilder},
        handle::EnteredContext,
    };

    fn bfs_distance_map(handle: &EnteredContext<'_>, result: &BfsResult) -> std::collections::HashMap<i32, i32> {
        let vertices = result.vertex_ids().expect("vertex_ids").to_vec_i32(handle).expect("vertices");
        let distances = result.distances().expect("distances").to_vec_i32(handle).expect("distances");
        vertices.into_iter().zip(distances).collect()
    }

    #[test]
    fn test_bfs_directed_graph() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_bfs_directed_graph").expect("entered");

        // Directed graph matching the C API test pattern:
        // 0->1, 1->3, 1->4, 2->0, 2->1, 2->3, 3->5, 4->5
        let src_data: Vec<i32> = vec![0, 1, 1, 2, 2, 2, 3, 4];
        let dst_data: Vec<i32> = vec![1, 3, 4, 0, 1, 3, 5, 5];
        let wgt_data: Vec<f32> = vec![0.1, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2];

        let src = DeviceArray::from_slice(&entered, &src_data).expect("src");
        let dst = DeviceArray::from_slice(&entered, &dst_data).expect("dst");
        let wgt = DeviceArray::from_slice(&entered, &wgt_data).expect("wgt");
        let src_view = src.view().expect("src_view");
        let dst_view = dst.view().expect("dst_view");
        let wgt_view = wgt.view().expect("wgt_view");

        let mut graph = GraphBuilder::new(&src_view, &dst_view)
            .weights(&wgt_view)
            .renumber()
            .build::<Forward>(&entered)
            .expect("graph");

        // BFS from vertex 0
        let sources_arr = DeviceArray::from_slice(&entered, &[0i32]).expect("sources");
        let sources_view = sources_arr.view().expect("sources_view");

        let params = BfsParams::default().with_predecessors();
        let result = graph.bfs(&entered, &sources_view, &params).expect("bfs");

        let vids = result.vertex_ids().expect("vertex_ids");
        let dists = result.distances().expect("distances");
        assert_eq!(vids.len(), 6, "should have 6 vertices");
        assert_eq!(dists.len(), 6, "should have 6 distances");

        let dist_map = bfs_distance_map(&entered, &result);

        // Expected: {0: 0, 1: 1, 2: unreachable, 3: 2, 4: 2, 5: 3}
        assert_eq!(dist_map[&0], 0, "source should have distance 0");
        assert_eq!(dist_map[&1], 1, "vertex 1 should have distance 1");
        assert_eq!(dist_map[&2], i32::MAX, "vertex 2 should be unreachable from 0");
        assert_eq!(dist_map[&3], 2, "vertex 3 should have distance 2");
        assert_eq!(dist_map[&4], 2, "vertex 4 should have distance 2");
        assert_eq!(dist_map[&5], 3, "vertex 5 should have distance 3");

        // Check predecessors are present
        let preds = result.predecessors().expect("predecessors");
        assert!(preds.is_some(), "predecessors should be computed");
    }

    #[test]
    fn test_bfs_depth_limit_works() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_bfs_depth_limit_works").expect("entered");

        // Path graph 0->1->2->3->4
        let src_data: Vec<i32> = vec![0, 1, 2, 3];
        let dst_data: Vec<i32> = vec![1, 2, 3, 4];

        let src = DeviceArray::from_slice(&entered, &src_data).expect("src");
        let dst = DeviceArray::from_slice(&entered, &dst_data).expect("dst");
        let src_view = src.view().expect("src_view");
        let dst_view = dst.view().expect("dst_view");

        let mut graph = GraphBuilder::new(&src_view, &dst_view).build::<Forward>(&entered).expect("graph");

        let sources_arr = DeviceArray::from_slice(&entered, &[0i32]).expect("sources");
        let sources_view = sources_arr.view().expect("sources_view");

        let params = BfsParams::default().with_depth_limit(BfsDepthLimit::new(2).unwrap());
        let result = graph.bfs(&entered, &sources_view, &params).expect("bfs");

        let dist_map = bfs_distance_map(&entered, &result);

        assert_eq!(dist_map[&0], 0);
        assert_eq!(dist_map[&1], 1);
        assert_eq!(dist_map[&2], 2);
        assert_eq!(dist_map[&3], i32::MAX);
        assert_eq!(dist_map[&4], i32::MAX);
    }

    #[test]
    fn test_bfs_no_predecessors() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_bfs_no_predecessors").expect("entered");

        let src_data: Vec<i32> = vec![0, 1];
        let dst_data: Vec<i32> = vec![1, 2];

        let src = DeviceArray::from_slice(&entered, &src_data).expect("src");
        let dst = DeviceArray::from_slice(&entered, &dst_data).expect("dst");
        let src_view = src.view().expect("src_view");
        let dst_view = dst.view().expect("dst_view");

        let mut graph = GraphBuilder::new(&src_view, &dst_view).build::<Forward>(&entered).expect("graph");

        let sources_arr = DeviceArray::from_slice(&entered, &[0i32]).expect("sources");
        let sources_view = sources_arr.view().expect("sources_view");

        let params = BfsParams::default();
        let result = graph.bfs(&entered, &sources_view, &params).expect("bfs");
        assert!(result.predecessors().expect("predecessors").is_none(), "predecessors should be omitted");
    }

    #[test]
    fn test_bfs_with_excluded_vertices() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_bfs_with_excluded_vertices").expect("entered");

        let src_data = vec![0i32, 0, 1, 3];
        let dst_data = vec![1i32, 3, 2, 2];
        let src = DeviceArray::from_slice(&entered, &src_data).expect("src");
        let dst = DeviceArray::from_slice(&entered, &dst_data).expect("dst");
        let src_view = src.view().expect("src_view");
        let dst_view = dst.view().expect("dst_view");
        let mut graph = GraphBuilder::new(&src_view, &dst_view).build::<Forward>(&entered).expect("graph");

        let sources = DeviceArray::from_slice(&entered, &[0i32]).expect("sources");
        let excluded = DeviceArray::from_slice(&entered, &[1i32]).expect("excluded");
        let sources_view = sources.view().expect("sources_view");
        let excluded_view = excluded.view().expect("excluded_view");
        let predicates: BfsPredicateParams<'_, i32> =
            BfsPredicateParams { exclude_vertices: Some(&excluded_view), ..BfsPredicateParams::default() };

        let result =
            graph.bfs_with_predicates(&entered, &sources_view, &BfsParams::default(), &predicates).expect("bfs");
        let dist_map = bfs_distance_map(&entered, &result);

        assert_eq!(dist_map[&0], 0);
        assert_eq!(dist_map[&1], i32::MAX);
        assert_eq!(dist_map[&3], 1);
        assert_eq!(dist_map[&2], 2);
    }

    #[test]
    fn test_bfs_with_included_vertices() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_bfs_with_included_vertices").expect("entered");

        let src_data = vec![0i32, 0, 1, 3];
        let dst_data = vec![1i32, 3, 2, 2];
        let src = DeviceArray::from_slice(&entered, &src_data).expect("src");
        let dst = DeviceArray::from_slice(&entered, &dst_data).expect("dst");
        let src_view = src.view().expect("src_view");
        let dst_view = dst.view().expect("dst_view");
        let mut graph = GraphBuilder::new(&src_view, &dst_view).build::<Forward>(&entered).expect("graph");

        let sources = DeviceArray::from_slice(&entered, &[0i32]).expect("sources");
        let included = DeviceArray::from_slice(&entered, &[0i32, 1, 2]).expect("included");
        let sources_view = sources.view().expect("sources_view");
        let included_view = included.view().expect("included_view");
        let predicates: BfsPredicateParams<'_, i32> =
            BfsPredicateParams { include_vertices: Some(&included_view), ..BfsPredicateParams::default() };

        let result =
            graph.bfs_with_predicates(&entered, &sources_view, &BfsParams::default(), &predicates).expect("bfs");
        let dist_map = bfs_distance_map(&entered, &result);

        assert_eq!(dist_map[&0], 0);
        assert_eq!(dist_map[&1], 1);
        assert_eq!(dist_map[&3], i32::MAX);
        assert_eq!(dist_map[&2], 2);
    }

    #[test]
    fn test_bfs_rejects_include_and_exclude_vertices_together() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_bfs_rejects_include_and_exclude_vertices_together").expect("entered");

        let src = DeviceArray::from_slice(&entered, &[0i32, 1]).expect("src");
        let dst = DeviceArray::from_slice(&entered, &[1i32, 2]).expect("dst");
        let src_view = src.view().expect("src_view");
        let dst_view = dst.view().expect("dst_view");
        let mut graph = GraphBuilder::new(&src_view, &dst_view).build::<Forward>(&entered).expect("graph");

        let sources = DeviceArray::from_slice(&entered, &[0i32]).expect("sources");
        let included = DeviceArray::from_slice(&entered, &[0i32, 1]).expect("included");
        let excluded = DeviceArray::from_slice(&entered, &[1i32]).expect("excluded");
        let sources_view = sources.view().expect("sources_view");
        let included_view = included.view().expect("included_view");
        let excluded_view = excluded.view().expect("excluded_view");
        let predicates: BfsPredicateParams<'_, i32> = BfsPredicateParams {
            include_vertices: Some(&included_view),
            exclude_vertices: Some(&excluded_view),
            ..BfsPredicateParams::default()
        };

        let err = graph
            .bfs_with_predicates(&entered, &sources_view, &BfsParams::default(), &predicates)
            .expect_err("include/exclude vertex predicates are mutually exclusive");
        assert_eq!(err.kind(), crate::error::ErrorKind::InvalidInput);
        assert_eq!(err.operation(), "Graph::bfs_with_predicates");
        assert!(err.message().contains("include_vertices and exclude_vertices are mutually exclusive"));
    }

    #[test]
    fn test_bfs_with_include_edge_ids() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_bfs_with_include_edge_ids").expect("entered");

        let src_data = vec![0i32, 0, 1, 2];
        let dst_data = vec![1i32, 2, 3, 3];
        let edge_id_data = vec![10i32, 11, 12, 13];
        let src = DeviceArray::from_slice(&entered, &src_data).expect("src");
        let dst = DeviceArray::from_slice(&entered, &dst_data).expect("dst");
        let edge_ids = DeviceArray::from_slice(&entered, &edge_id_data).expect("edge_ids");
        let src_view = src.view().expect("src_view");
        let dst_view = dst.view().expect("dst_view");
        let edge_ids_view = edge_ids.view().expect("edge_ids_view");
        let mut graph =
            GraphBuilder::new(&src_view, &dst_view).edge_ids(&edge_ids_view).build::<Forward>(&entered).expect("graph");

        let sources = DeviceArray::from_slice(&entered, &[0i32]).expect("sources");
        let allowed_edge_ids = DeviceArray::from_slice(&entered, &[11i32, 13]).expect("allowed_edge_ids");
        let sources_view = sources.view().expect("sources_view");
        let allowed_edge_ids_view = allowed_edge_ids.view().expect("allowed_edge_ids_view");
        let predicates =
            BfsPredicateParams { include_edge_ids: Some(&allowed_edge_ids_view), ..BfsPredicateParams::default() };

        let result =
            graph.bfs_with_predicates(&entered, &sources_view, &BfsParams::default(), &predicates).expect("bfs");
        let dist_map = bfs_distance_map(&entered, &result);

        assert_eq!(dist_map[&0], 0);
        assert_eq!(dist_map[&1], i32::MAX);
        assert_eq!(dist_map[&2], 1);
        assert_eq!(dist_map[&3], 2);
    }

    #[test]
    fn test_bfs_include_edge_ids_rejects_graph_without_edge_ids() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_bfs_include_edge_ids_rejects_graph_without_edge_ids").expect("entered");

        let src = DeviceArray::from_slice(&entered, &[0i32, 1]).expect("src");
        let dst = DeviceArray::from_slice(&entered, &[1i32, 2]).expect("dst");
        let src_view = src.view().expect("src_view");
        let dst_view = dst.view().expect("dst_view");
        let mut graph = GraphBuilder::new(&src_view, &dst_view).build::<Forward>(&entered).expect("graph");

        let sources = DeviceArray::from_slice(&entered, &[0i32]).expect("sources");
        let allowed_edge_ids = DeviceArray::from_slice(&entered, &[0i32]).expect("allowed_edge_ids");
        let sources_view = sources.view().expect("sources_view");
        let allowed_edge_ids_view = allowed_edge_ids.view().expect("allowed_edge_ids_view");
        let predicates =
            BfsPredicateParams { include_edge_ids: Some(&allowed_edge_ids_view), ..BfsPredicateParams::default() };

        let err = graph
            .bfs_with_predicates(&entered, &sources_view, &BfsParams::default(), &predicates)
            .expect_err("edge-id predicates require graph edge ids");
        assert_eq!(err.kind(), crate::error::ErrorKind::InvalidInput);
        assert_eq!(err.operation(), "Graph::bfs_with_predicates");
        assert!(err.message().contains("include_edge_ids requires a graph with edge IDs"));
    }

    #[test]
    fn test_bfs_include_edge_ids_rejects_mismatched_dtype() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_bfs_include_edge_ids_rejects_mismatched_dtype").expect("entered");

        let src = DeviceArray::from_slice(&entered, &[0i32, 1]).expect("src");
        let dst = DeviceArray::from_slice(&entered, &[1i32, 2]).expect("dst");
        let edge_ids = DeviceArray::from_slice(&entered, &[10i32, 11]).expect("edge_ids");
        let src_view = src.view().expect("src_view");
        let dst_view = dst.view().expect("dst_view");
        let edge_ids_view = edge_ids.view().expect("edge_ids_view");
        let mut graph =
            GraphBuilder::new(&src_view, &dst_view).edge_ids(&edge_ids_view).build::<Forward>(&entered).expect("graph");

        let sources = DeviceArray::from_slice(&entered, &[0i32]).expect("sources");
        let allowed_edge_ids = DeviceArray::from_slice(&entered, &[10i64]).expect("allowed_edge_ids");
        let sources_view = sources.view().expect("sources_view");
        let allowed_edge_ids_view = allowed_edge_ids.view().expect("allowed_edge_ids_view");
        let predicates =
            BfsPredicateParams { include_edge_ids: Some(&allowed_edge_ids_view), ..BfsPredicateParams::default() };

        let err = graph
            .bfs_with_predicates(&entered, &sources_view, &BfsParams::default(), &predicates)
            .expect_err("include_edge_ids dtype should match graph edge-id dtype");
        assert_eq!(err.kind(), crate::error::ErrorKind::UnsupportedType);
        assert_eq!(err.operation(), "Graph::bfs_with_predicates");
        assert!(err.message().contains("include_edge_ids dtype Int64 does not match graph edge-id dtype Int32"));
    }

    #[test]
    fn test_bfs_target_vertices_return_metadata() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_bfs_target_vertices_return_metadata").expect("entered");

        let src = DeviceArray::from_slice(&entered, &[0i32, 1, 2]).expect("src");
        let dst = DeviceArray::from_slice(&entered, &[1i32, 2, 3]).expect("dst");
        let src_view = src.view().expect("src_view");
        let dst_view = dst.view().expect("dst_view");
        let mut graph = GraphBuilder::new(&src_view, &dst_view).build::<Forward>(&entered).expect("graph");

        let sources = DeviceArray::from_slice(&entered, &[0i32]).expect("sources");
        let targets = DeviceArray::from_slice(&entered, &[2i32]).expect("targets");
        let sources_view = sources.view().expect("sources_view");
        let targets_view = targets.view().expect("targets_view");
        let predicates: BfsPredicateParams<'_, i32> = BfsPredicateParams {
            target_vertices: Some(&targets_view),
            stop_when_any_target_found: true,
            ..BfsPredicateParams::default()
        };

        let result =
            graph.bfs_with_predicates(&entered, &sources_view, &BfsParams::default(), &predicates).expect("bfs");
        let target_info = result.target_info().expect("target metadata should be returned");

        assert!(target_info.target_found);
        assert_eq!(target_info.target_distance, Some(2));
        let dist_map = bfs_distance_map(&entered, &result);
        assert_eq!(dist_map[&3], i32::MAX, "early stop should avoid depth after target discovery");
    }
}
