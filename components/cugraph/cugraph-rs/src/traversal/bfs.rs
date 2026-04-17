//! Breadth-First Search (BFS) traversal algorithm.

use std::{marker::PhantomData, ptr};

use cugraph_sys::{
    cugraph_bfs, cugraph_error_t, cugraph_paths_result_free, cugraph_paths_result_get_distances,
    cugraph_paths_result_get_predecessors, cugraph_paths_result_get_vertices, cugraph_paths_result_t,
};

use crate::{
    array::{DeviceArrayView, RawDeviceArrayView},
    cuda::DeviceGuard,
    error::{Error, Result, check_error, require_non_null},
    graph::{Forward, Graph},
    handle::ResourceHandle,
    params::BfsDepthLimit,
    structure::typed_results::{Distances, VertexIds},
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

    // SAFETY: `graph` and `sources` stay valid for the duration of the call,
    // `sources` matches the graph vertex dtype by prior validation, and
    // `result`/`error` are valid out-pointers.
    unsafe {
        let code = cugraph_bfs(
            handle.as_ptr(),
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

    let result = require_non_null(result, "bfs", "paths_result")?;

    Ok(BfsResult { raw: result, device: handle.device(), _nosend_nosync: PhantomData })
}

/// Result of a BFS computation.
///
/// Provides access to vertex ids, distances, and optionally predecessors.
/// Freed on drop.
pub struct BfsResult {
    raw: *mut cugraph_paths_result_t,
    device: i32,
    _nosend_nosync: PhantomData<*mut ()>,
}

impl std::fmt::Debug for BfsResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BfsResult").finish_non_exhaustive()
    }
}

unsafe fn normalize_optional_raw_view<'a>(
    view_ptr: *mut cugraph_sys::cugraph_type_erased_device_array_view_t,
    device: i32,
    operation: &'static str,
    pointer_name: &'static str,
) -> Result<Option<RawDeviceArrayView<'a>>> {
    if view_ptr.is_null() {
        return Ok(None);
    }

    // SAFETY: the caller guarantees that `view_ptr` belongs to a live cuGraph
    // result for the requested lifetime.
    let raw = unsafe { RawDeviceArrayView::from_borrowed(view_ptr, Some(device), operation, pointer_name)? };
    if raw.is_empty() { Ok(None) } else { Ok(Some(raw)) }
}

impl BfsResult {
    fn normalized_predecessors_raw(&self) -> Result<Option<RawDeviceArrayView<'_>>> {
        // SAFETY: `self.raw` is a live paths result owned by this wrapper for
        // the duration of the call.
        let view_ptr = unsafe { cugraph_paths_result_get_predecessors(self.raw) };
        // SAFETY: `view_ptr` comes from the live cuGraph result object owned by
        // `self` for the duration of `&self`.
        unsafe {
            normalize_optional_raw_view(view_ptr, self.device, "BfsResult::normalized_predecessors_raw", "predecessors")
        }
    }

    /// Get the vertex ids as a runtime-typed device array view.
    ///
    /// Returns [`VertexIds::I32`] or [`VertexIds::I64`] depending on the
    /// graph's vertex type.
    pub fn vertex_ids(&self) -> Result<VertexIds<'_>> {
        // SAFETY: `self.raw` is a live paths result owned by this wrapper for
        // the duration of the call.
        let view_ptr = require_non_null(
            unsafe { cugraph_paths_result_get_vertices(self.raw) },
            "BfsResult::vertex_ids",
            "vertices",
        )?;
        // SAFETY: `view_ptr` belongs to the live result object and was
        // validated to be non-null above.
        let raw = unsafe {
            RawDeviceArrayView::from_borrowed(view_ptr, Some(self.device), "BfsResult::vertex_ids", "vertices")
        }?;
        VertexIds::from_raw(raw)
    }

    /// Get the BFS distances as a runtime-typed device array view.
    ///
    /// Returns [`Distances::I32`] or [`Distances::I64`] depending on the
    /// graph's vertex type. BFS stores distances as integer hop counts
    /// (`vertex_t`). A future SSSP binding will return [`Distances::F32`] or
    /// [`Distances::F64`] (float weighted distances) from the same
    /// `cugraph_paths_result_t` C API type.
    pub fn distances(&self) -> Result<Distances<'_>> {
        // SAFETY: `self.raw` is a live paths result owned by this wrapper for
        // the duration of the call.
        let view_ptr = require_non_null(
            unsafe { cugraph_paths_result_get_distances(self.raw) },
            "BfsResult::distances",
            "distances",
        )?;
        // SAFETY: `view_ptr` belongs to the live result object and was
        // validated to be non-null above.
        let raw = unsafe {
            RawDeviceArrayView::from_borrowed(view_ptr, Some(self.device), "BfsResult::distances", "distances")
        }?;
        Distances::from_raw(raw)
    }

    /// Get the predecessor vertices as a runtime-typed device array view.
    ///
    /// Returns `None` if `compute_predecessors` was `false` in the BFS call.
    /// cuGraph may signal that state with either a null pointer or a
    /// zero-length array view; both are normalized to `None`.
    /// Returns [`VertexIds::I32`] or [`VertexIds::I64`] depending on the
    /// graph's vertex type.
    pub fn predecessors(&self) -> Result<Option<VertexIds<'_>>> {
        match self.normalized_predecessors_raw()? {
            Some(raw) => VertexIds::from_raw(raw).map(Some),
            None => Ok(None),
        }
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

impl Drop for BfsResult {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _guard = DeviceGuard::switch_to_for_drop(self.device, "BfsResult::drop").ok();
            // SAFETY: `self.raw` is owned by this wrapper and has not been
            // freed yet.
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

    fn bfs_distance_map(handle: &ResourceHandle, result: &BfsResult) -> std::collections::HashMap<i32, i32> {
        let vertices = result.vertex_ids().expect("vertex_ids").to_vec_i32(handle).expect("vertices");
        let distances = result.distances().expect("distances").to_vec_i32(handle).expect("distances");
        vertices.into_iter().zip(distances).collect()
    }

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
        let src_view = src.view().expect("src_view");
        let dst_view = dst.view().expect("dst_view");
        let wgt_view = wgt.view().expect("wgt_view");

        let mut graph = GraphBuilder::new(&src_view, &dst_view)
            .weights(&wgt_view)
            .renumber()
            .build::<Forward>(&handle)
            .expect("graph");

        // BFS from vertex 0
        let sources_arr = DeviceArray::from_slice(&handle, &[0i32]).expect("sources");
        let sources_view = sources_arr.view().expect("sources_view");

        let params = BfsParams::default().with_predecessors();
        let result = graph.bfs(&handle, &sources_view, &params).expect("bfs");

        let vids = result.vertex_ids().expect("vertex_ids");
        let dists = result.distances().expect("distances");
        assert_eq!(vids.len(), 6, "should have 6 vertices");
        assert_eq!(dists.len(), 6, "should have 6 distances");

        let dist_map = bfs_distance_map(&handle, &result);

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
        let src_view = src.view().expect("src_view");
        let dst_view = dst.view().expect("dst_view");

        let mut graph = GraphBuilder::new(&src_view, &dst_view).build::<Forward>(&handle).expect("graph");

        let sources_arr = DeviceArray::from_slice(&handle, &[0i32]).expect("sources");
        let sources_view = sources_arr.view().expect("sources_view");

        let params = BfsParams::default().with_depth_limit(BfsDepthLimit::new(2).unwrap());
        let result = graph.bfs(&handle, &sources_view, &params).expect("bfs");

        let dist_map = bfs_distance_map(&handle, &result);

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
        let src_view = src.view().expect("src_view");
        let dst_view = dst.view().expect("dst_view");

        let mut graph = GraphBuilder::new(&src_view, &dst_view).build::<Forward>(&handle).expect("graph");

        let sources_arr = DeviceArray::from_slice(&handle, &[0i32]).expect("sources");
        let sources_view = sources_arr.view().expect("sources_view");

        let params = BfsParams::default();
        let result = graph.bfs(&handle, &sources_view, &params).expect("bfs");
        assert!(result.predecessors().expect("predecessors").is_none(), "predecessors should be omitted");
    }

    #[test]
    fn test_normalize_optional_raw_view_treats_null_as_none() {
        // SAFETY: this validates the null-path contract and does not
        // dereference the pointer.
        let normalized = unsafe {
            normalize_optional_raw_view(
                std::ptr::null_mut(),
                0,
                "test_normalize_optional_raw_view_treats_null_as_none",
                "predecessors",
            )
        }
        .expect("null predecessor pointer should not error");
        assert!(normalized.is_none());
    }

    #[test]
    fn test_normalize_optional_raw_view_treats_empty_view_as_none() {
        let handle = ResourceHandle::new().expect("handle");
        let empty = DeviceArray::<i32>::new(&handle, 0).expect("empty");
        let view_ptr = empty.view().expect("view").raw_view().as_ptr();

        // SAFETY: `view_ptr` comes from a live empty device array view and
        // remains valid while `empty` is alive.
        let normalized = unsafe {
            normalize_optional_raw_view(
                view_ptr,
                handle.device(),
                "test_normalize_optional_raw_view_treats_empty_view_as_none",
                "predecessors",
            )
        }
        .expect("empty predecessor view should not error");
        assert!(normalized.is_none());
    }

    #[test]
    fn test_normalize_optional_raw_view_keeps_non_empty_view() {
        let handle = ResourceHandle::new().expect("handle");
        let preds = DeviceArray::from_slice(&handle, &[1i32, 2, 3]).expect("preds");
        let view_ptr = preds.view().expect("view").raw_view().as_ptr();

        // SAFETY: `view_ptr` comes from a live non-empty device array view and
        // remains valid while `preds` is alive.
        let normalized = unsafe {
            normalize_optional_raw_view(
                view_ptr,
                handle.device(),
                "test_normalize_optional_raw_view_keeps_non_empty_view",
                "predecessors",
            )
        }
        .expect("non-empty predecessor view should not error");

        let raw = normalized.expect("non-empty view should be preserved");
        assert_eq!(raw.len(), 3);
    }
}
