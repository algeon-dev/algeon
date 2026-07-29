//! Breadth-First Search (BFS) traversal algorithm.

use std::ptr;

use cugraph_sys::{cugraph_bfs, cugraph_error_t, cugraph_paths_result_t};

use crate::{
    array::DeviceArrayView,
    error::{Error, Result, check_error},
    graph::{Forward, Graph},
    handle::EnteredContext,
    params::BfsDepthLimit,
    structure::{
        VertexSelection,
        typed_results::{Distances, VertexIds},
    },
    traversal::paths::{ExtractPathsResult, PathsResult},
    types::{CugraphElement, to_bool_t},
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

    // SAFETY: `graph` and `sources` stay valid for the duration of the call,
    // `sources` matches the graph vertex dtype by prior validation, and
    // `result`/`error` are valid out-pointers.
    unsafe {
        let code = cugraph_bfs(
            handle.handle_ptr(),
            graph.as_ptr(),
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

    Ok(BfsResult { paths })
}

/// Result of a BFS computation.
///
/// Provides access to vertex ids, distances, and optionally predecessors.
/// Freed on drop.
pub struct BfsResult {
    pub(crate) paths: PathsResult,
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
}
