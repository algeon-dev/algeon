//! Breadth-First Search (BFS) traversal algorithm.

use std::ptr;

use cugraph_sys::{
    bool_t, cugraph_bfs, cugraph_error_t, cugraph_paths_result_free,
    cugraph_paths_result_get_distances, cugraph_paths_result_get_predecessors,
    cugraph_paths_result_get_vertices, cugraph_paths_result_t,
};

use crate::array::DeviceArrayView;
use crate::error::{check_error, Result};
use crate::graph::Graph;
use crate::handle::ResourceHandle;
use crate::types::to_bool_t;

/// Perform breadth-first search from one or more source vertices.
///
/// # Arguments
///
/// - `handle` — resource handle for GPU operations
/// - `graph` — the graph to traverse
/// - `sources` — source vertex ids (device array view, mutable because the C
///   API may modify it if renumbering is enabled)
/// - `depth_limit` — maximum BFS depth, or `None` for unlimited traversal.
///   The C API treats depth_limit=0 as "stop at depth 0", so `None` maps to
///   `i32::MAX` (matching the C++ default of `std::numeric_limits<vertex_t>::max()`).
/// - `compute_predecessors` — whether to compute predecessor vertices
pub fn bfs(
    handle: &ResourceHandle,
    graph: &mut Graph,
    sources: &mut DeviceArrayView<'_>,
    depth_limit: Option<usize>,
    compute_predecessors: bool,
) -> Result<BfsResult> {
    let mut result: *mut cugraph_paths_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();

    // The C++ BFS defaults to std::numeric_limits<vertex_t>::max() for unlimited
    // depth. Since vertex_t is typically int32_t, we use i32::MAX.
    let limit = depth_limit.unwrap_or(i32::MAX as usize);

    unsafe {
        let code = cugraph_bfs(
            handle.as_ptr(),
            graph.raw,
            sources.as_ptr(),
            bool_t::FALSE, // direction_optimizing
            limit,
            to_bool_t(compute_predecessors),
            bool_t::FALSE, // do_expensive_check
            &mut result,
            &mut error,
        );
        check_error(code, error, "bfs")?;
    }

    Ok(BfsResult {
        raw: result,
        _nosend_nosync: std::marker::PhantomData,
    })
}

/// Result of a BFS computation.
///
/// Provides access to vertex ids, distances, and optionally predecessors.
/// Freed on drop.
pub struct BfsResult {
    raw: *mut cugraph_paths_result_t,
    _nosend_nosync: std::marker::PhantomData<*mut ()>,
}

impl BfsResult {
    /// Get the vertex ids as a device array view.
    pub fn vertices(&self) -> DeviceArrayView<'_> {
        let view_ptr = unsafe { cugraph_paths_result_get_vertices(self.raw) };
        unsafe { DeviceArrayView::from_borrowed(view_ptr) }
    }

    /// Get the BFS distances as a device array view.
    pub fn distances(&self) -> DeviceArrayView<'_> {
        let view_ptr = unsafe { cugraph_paths_result_get_distances(self.raw) };
        unsafe { DeviceArrayView::from_borrowed(view_ptr) }
    }

    /// Get the predecessor vertices as a device array view.
    ///
    /// Returns `None` if `compute_predecessors` was `false` in the BFS call.
    pub fn predecessors(&self) -> Option<DeviceArrayView<'_>> {
        let view_ptr = unsafe { cugraph_paths_result_get_predecessors(self.raw) };
        if view_ptr.is_null() {
            None
        } else {
            Some(unsafe { DeviceArrayView::from_borrowed(view_ptr) })
        }
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
    use crate::array::device_array_from_host;
    use crate::types::{DataType, GraphProperties};

    #[test]
    fn test_bfs_directed_graph() {
        let handle = ResourceHandle::new().expect("handle");

        // Directed graph matching the C API test pattern:
        // 0->1, 1->3, 1->4, 2->0, 2->1, 2->3, 3->5, 4->5
        let src_data: Vec<i32> = vec![0, 1, 1, 2, 2, 2, 3, 4];
        let dst_data: Vec<i32> = vec![1, 3, 4, 0, 1, 3, 5, 5];
        let wgt_data: Vec<f32> = vec![0.1, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2];

        let src = device_array_from_host(&handle, &src_data, DataType::Int32).expect("src");
        let dst = device_array_from_host(&handle, &dst_data, DataType::Int32).expect("dst");
        let wgt = device_array_from_host(&handle, &wgt_data, DataType::Float32).expect("wgt");

        let props = GraphProperties {
            is_symmetric: false,
            is_multigraph: false,
        };

        let mut graph = Graph::from_edge_list(
            &handle,
            props,
            &src.view(),
            &dst.view(),
            Some(&wgt.view()),
            false, // store_transposed (BFS needs non-transposed)
            true,  // renumber
        )
        .expect("graph");

        // BFS from vertex 0
        let sources_arr =
            device_array_from_host(&handle, &[0i32], DataType::Int32).expect("sources");
        let mut sources_view = sources_arr.view();

        let result = bfs(&handle, &mut graph, &mut sources_view, None, true).expect("bfs");

        let vert_view = result.vertices();
        let dist_view = result.distances();

        assert_eq!(vert_view.len(), 6, "should have 6 vertices");
        assert_eq!(dist_view.len(), 6, "should have 6 distances");

        // Copy to host
        let vert_bytes = vert_view.to_host_vec(&handle).expect("vert copy");
        let dist_bytes = dist_view.to_host_vec(&handle).expect("dist copy");

        let vertices: Vec<i32> = vert_bytes
            .chunks_exact(4)
            .map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let distances: Vec<i32> = dist_bytes
            .chunks_exact(4)
            .map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Build a map of vertex -> distance
        let mut dist_map = std::collections::HashMap::new();
        for (v, d) in vertices.iter().zip(distances.iter()) {
            dist_map.insert(*v, *d);
        }

        // Expected: {0: 0, 1: 1, 2: unreachable, 3: 2, 4: 2, 5: 3}
        assert_eq!(dist_map[&0], 0, "source should have distance 0");
        assert_eq!(dist_map[&1], 1, "vertex 1 should have distance 1");
        assert_eq!(
            dist_map[&2],
            i32::MAX,
            "vertex 2 should be unreachable from 0"
        );
        assert_eq!(dist_map[&3], 2, "vertex 3 should have distance 2");
        assert_eq!(dist_map[&4], 2, "vertex 4 should have distance 2");
        assert_eq!(dist_map[&5], 3, "vertex 5 should have distance 3");

        // Check predecessors are present
        let preds = result.predecessors();
        assert!(preds.is_some(), "predecessors should be computed");
    }

    #[test]
    fn test_bfs_no_predecessors() {
        let handle = ResourceHandle::new().expect("handle");

        let src_data: Vec<i32> = vec![0, 1];
        let dst_data: Vec<i32> = vec![1, 2];

        let src = device_array_from_host(&handle, &src_data, DataType::Int32).expect("src");
        let dst = device_array_from_host(&handle, &dst_data, DataType::Int32).expect("dst");

        let props = GraphProperties {
            is_symmetric: false,
            is_multigraph: false,
        };

        let mut graph = Graph::from_edge_list(
            &handle,
            props,
            &src.view(),
            &dst.view(),
            None,
            false, // store_transposed (BFS needs non-transposed)
            false, // renumber (not needed, IDs are 0-based sequential)
        )
        .expect("graph");

        let sources_arr =
            device_array_from_host(&handle, &[0i32], DataType::Int32).expect("sources");
        let mut sources_view = sources_arr.view();

        let result = bfs(&handle, &mut graph, &mut sources_view, None, false).expect("bfs");
        // When compute_predecessors is false, predecessors may be NULL
        // (behavior depends on C implementation)
        let _ = result.predecessors();
    }
}
