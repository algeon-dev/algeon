//! Breadth-First Search (BFS) traversal algorithm.

use std::ptr;

use cugraph_sys::{
    cugraph_bfs, cugraph_error_t, cugraph_paths_result_free, cugraph_paths_result_get_distances,
    cugraph_paths_result_get_predecessors, cugraph_paths_result_get_vertices, cugraph_paths_result_t,
};

use crate::{
    array::{DeviceArrayView, RawDeviceArrayView},
    error::{Error, ErrorKind, Result, check_error},
    graph::{Forward, Graph},
    handle::ResourceHandle,
    params::BfsDepthLimit,
    structure::typed_results::{Distances, VertexIds},
    types::{CugraphElement, to_bool_t},
};

/// Parameters for the BFS traversal algorithm.
///
/// Use the builder methods to configure optional settings. All fields
/// default to "no limit, no predecessors" via [`Default`].
///
/// # Example
///
/// ```
/// use cugraph::{params::BfsDepthLimit, traversal::BfsParams};
///
/// let params = BfsParams::default()
///     .with_predecessors()
///     .with_depth_limit(BfsDepthLimit::new(3).unwrap());
/// assert!(params.compute_predecessors());
/// assert_eq!(params.depth_limit().unwrap().get(), 3);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct BfsParams {
    depth_limit: Option<BfsDepthLimit>,
    compute_predecessors: bool,
    do_expensive_check: bool,
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
/// Prefer using [`Graph::bfs`](crate::graph::Graph::bfs) instead of calling
/// this function directly.
pub(crate) fn bfs<V: CugraphElement>(
    handle: &ResourceHandle,
    graph: &mut Graph<Forward>,
    sources: &DeviceArrayView<'_, V>,
    params: &BfsParams,
) -> Result<BfsResult> {
    let mut result: *mut cugraph_paths_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();

    // The C++ BFS defaults to std::numeric_limits<vertex_t>::max() for unlimited
    // depth. Since vertex_t is typically int32_t, we use i32::MAX.
    let limit = params.depth_limit.map(usize::from).unwrap_or(i32::MAX as usize);

    unsafe {
        let code = cugraph_bfs(
            handle.as_ptr(),
            graph.raw,
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

    Ok(BfsResult { raw: result, _nosend_nosync: std::marker::PhantomData })
}

/// Result of a BFS computation.
///
/// Provides access to vertex ids, distances, and optionally predecessors.
/// Freed on drop.
pub struct BfsResult {
    raw: *mut cugraph_paths_result_t,
    _nosend_nosync: std::marker::PhantomData<*mut ()>,
}

impl std::fmt::Debug for BfsResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BfsResult").finish_non_exhaustive()
    }
}

impl BfsResult {
    fn normalized_predecessors_raw(&self) -> Option<RawDeviceArrayView<'_>> {
        let view_ptr = unsafe { cugraph_paths_result_get_predecessors(self.raw) };
        if view_ptr.is_null() {
            return None;
        }

        let raw = unsafe { RawDeviceArrayView::from_borrowed(view_ptr) };
        if raw.is_empty() { None } else { Some(raw) }
    }

    /// Get the vertex ids as a runtime-typed device array view.
    ///
    /// Returns [`VertexIds::I32`] or [`VertexIds::I64`] depending on the
    /// graph's vertex type.
    pub fn vertex_ids(&self) -> Result<VertexIds<'_>> {
        VertexIds::from_raw(self.vertices_raw()?)
    }

    /// Get the BFS distances as a runtime-typed device array view.
    ///
    /// Returns [`Distances::I32`] or [`Distances::I64`] depending on the
    /// graph's vertex type. BFS stores distances as integer hop counts
    /// (`vertex_t`). A future SSSP binding will return [`Distances::F32`] or
    /// [`Distances::F64`] (float weighted distances) from the same
    /// `cugraph_paths_result_t` C API type.
    pub fn distances(&self) -> Result<Distances<'_>> {
        Distances::from_raw(self.distances_raw()?)
    }

    /// Get the predecessor vertices as a runtime-typed device array view.
    ///
    /// Returns `None` if `compute_predecessors` was `false` in the BFS call.
    /// cuGraph may signal that state with either a null pointer or a
    /// zero-length array view; both are normalized to `None`.
    /// Returns [`VertexIds::I32`] or [`VertexIds::I64`] depending on the
    /// graph's vertex type.
    pub fn predecessors(&self) -> Result<Option<VertexIds<'_>>> {
        match self.normalized_predecessors_raw() {
            Some(raw) => VertexIds::from_raw(raw).map(Some),
            None => Ok(None),
        }
    }

    /// Get the raw vertices view.
    pub(crate) fn vertices_raw(&self) -> Result<RawDeviceArrayView<'_>> {
        let view_ptr = unsafe { cugraph_paths_result_get_vertices(self.raw) };
        if view_ptr.is_null() {
            return Err(Error::new(ErrorKind::Unexpected, "BfsResult::vertices_raw")
                .with_message("C API returned null pointer for vertices"));
        }
        Ok(unsafe { RawDeviceArrayView::from_borrowed(view_ptr) })
    }

    /// Get the raw distances view.
    pub(crate) fn distances_raw(&self) -> Result<RawDeviceArrayView<'_>> {
        let view_ptr = unsafe { cugraph_paths_result_get_distances(self.raw) };
        if view_ptr.is_null() {
            return Err(Error::new(ErrorKind::Unexpected, "BfsResult::distances_raw")
                .with_message("C API returned null pointer for distances"));
        }
        Ok(unsafe { RawDeviceArrayView::from_borrowed(view_ptr) })
    }

    /// Get the raw predecessors view.
    pub(crate) fn predecessors_raw(&self) -> Option<RawDeviceArrayView<'_>> {
        self.normalized_predecessors_raw()
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
    pub fn to_dataframe(&self) -> crate::error::Result<cudf::DataFrame> {
        cudf::DataFrame::try_from(self).map_err(|e: crate::error::Error| e.with_operation("BfsResult::to_dataframe"))
    }
}

impl Drop for BfsResult {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { cugraph_paths_result_free(self.raw) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        array::DeviceArray,
        graph::{Forward, GraphBuilder},
    };

    #[test]
    fn test_bfs_directed_graph() {
        let handle = ResourceHandle::new().expect("handle");

        // Directed graph matching the C API test pattern:
        // 0->1, 1->3, 1->4, 2->0, 2->1, 2->3, 3->5, 4->5
        let src_data: Vec<i32> = vec![0, 1, 1, 2, 2, 2, 3, 4];
        let dst_data: Vec<i32> = vec![1, 3, 4, 0, 1, 3, 5, 5];
        let wgt_data: Vec<f32> = vec![0.1, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2];

        let src = DeviceArray::from_slice(&handle, &src_data).expect("src");
        let dst = DeviceArray::from_slice(&handle, &dst_data).expect("dst");
        let wgt = DeviceArray::from_slice(&handle, &wgt_data).expect("wgt");

        let mut graph = GraphBuilder::new(&src.view(), &dst.view())
            .weights(&wgt.view())
            .renumber()
            .build::<Forward>(&handle)
            .expect("graph");

        // BFS from vertex 0
        let sources_arr = DeviceArray::from_slice(&handle, &[0i32]).expect("sources");
        let sources_view = sources_arr.view();

        let params = BfsParams::default().with_predecessors();
        let result = graph.bfs(&handle, &sources_view, &params).expect("bfs");

        let vids = result.vertex_ids().expect("vertex_ids");
        let dists = result.distances().expect("distances");
        assert_eq!(vids.len(), 6, "should have 6 vertices");
        assert_eq!(dists.len(), 6, "should have 6 distances");

        // Copy to host using typed helpers
        let vertices: Vec<i32> = vids.to_vec_i32(&handle).expect("vert copy");
        let distances: Vec<i32> = dists.to_vec_i32(&handle).expect("dist copy (BFS int hops)");

        // Build a map of vertex -> distance
        let mut dist_map = std::collections::HashMap::new();
        for (v, d) in vertices.iter().zip(distances.iter()) {
            dist_map.insert(*v, *d);
        }

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

        // Path graph 0->1->2->3->4
        let src_data: Vec<i32> = vec![0, 1, 2, 3];
        let dst_data: Vec<i32> = vec![1, 2, 3, 4];

        let src = DeviceArray::from_slice(&handle, &src_data).expect("src");
        let dst = DeviceArray::from_slice(&handle, &dst_data).expect("dst");

        let mut graph = GraphBuilder::new(&src.view(), &dst.view()).build::<Forward>(&handle).expect("graph");

        let sources_arr = DeviceArray::from_slice(&handle, &[0i32]).expect("sources");
        let sources_view = sources_arr.view();

        let params = BfsParams::default().with_depth_limit(BfsDepthLimit::new(2).unwrap());
        let result = graph.bfs(&handle, &sources_view, &params).expect("bfs");

        let vertices: Vec<i32> = result.vertex_ids().expect("vertex_ids").to_vec_i32(&handle).expect("vert copy");
        let distances: Vec<i32> =
            result.distances().expect("distances").to_vec_i32(&handle).expect("dist copy (BFS int hops)");

        let mut dist_map = std::collections::HashMap::new();
        for (v, d) in vertices.iter().zip(distances.iter()) {
            dist_map.insert(*v, *d);
        }

        assert_eq!(dist_map[&0], 0);
        assert_eq!(dist_map[&1], 1);
        assert_eq!(dist_map[&2], 2);
        assert_eq!(dist_map[&3], i32::MAX);
        assert_eq!(dist_map[&4], i32::MAX);
    }

    #[test]
    fn test_bfs_no_predecessors() {
        let handle = ResourceHandle::new().expect("handle");

        let src_data: Vec<i32> = vec![0, 1];
        let dst_data: Vec<i32> = vec![1, 2];

        let src = DeviceArray::from_slice(&handle, &src_data).expect("src");
        let dst = DeviceArray::from_slice(&handle, &dst_data).expect("dst");

        let mut graph = GraphBuilder::new(&src.view(), &dst.view()).build::<Forward>(&handle).expect("graph");

        let sources_arr = DeviceArray::from_slice(&handle, &[0i32]).expect("sources");
        let sources_view = sources_arr.view();

        let params = BfsParams::default();
        let result = graph.bfs(&handle, &sources_view, &params).expect("bfs");
        assert!(result.predecessors().expect("predecessors").is_none(), "predecessors should be omitted");
    }
}
