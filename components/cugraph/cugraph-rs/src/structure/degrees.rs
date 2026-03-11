//! Degree computation for graph vertices.
//!
//! Computes in-degrees, out-degrees, or both for all (or a subset of) vertices
//! in a graph.

use std::ptr;

use cugraph_sys::{
    bool_t, cugraph_degrees, cugraph_degrees_result_free, cugraph_degrees_result_get_in_degrees,
    cugraph_degrees_result_get_out_degrees, cugraph_degrees_result_get_vertices,
    cugraph_degrees_result_t, cugraph_error_t, cugraph_in_degrees, cugraph_out_degrees,
};

use crate::array::DeviceArrayView;
use crate::error::{check_error, Result};
use crate::graph::Graph;
use crate::handle::ResourceHandle;

/// Compute both in-degrees and out-degrees for vertices in a graph.
///
/// If `source_vertices` is `None`, computes degrees for all vertices.
/// Otherwise, computes degrees only for the specified vertices.
///
/// Returns a [`DegreesResult`] containing vertex ids, in-degrees, and
/// out-degrees. The result must be kept alive as long as any
/// [`DeviceArrayView`] borrowed from it is in use.
pub fn degrees(
    handle: &ResourceHandle,
    graph: &mut Graph,
    source_vertices: Option<&DeviceArrayView<'_>>,
) -> Result<DegreesResult> {
    let mut result: *mut cugraph_degrees_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();

    let src_ptr = source_vertices
        .map(|v| v.as_ptr() as *const _)
        .unwrap_or(ptr::null());

    unsafe {
        let code = cugraph_degrees(
            handle.as_ptr(),
            graph.raw,
            src_ptr,
            bool_t::FALSE, // do_expensive_check
            &mut result,
            &mut error,
        );
        check_error(code, error, "degrees")?;
    }

    Ok(DegreesResult {
        raw: result,
        _nosend_nosync: std::marker::PhantomData,
    })
}

/// Compute only in-degrees for vertices in a graph.
///
/// If `source_vertices` is `None`, computes in-degrees for all vertices.
/// Otherwise, computes in-degrees only for the specified vertices.
///
/// Returns a [`DegreesResult`]. [`DegreesResult::in_degrees()`] is expected to
/// be available. [`DegreesResult::out_degrees()`] may be unavailable for this
/// result and is therefore optional in the Rust API.
pub fn in_degrees(
    handle: &ResourceHandle,
    graph: &mut Graph,
    source_vertices: Option<&DeviceArrayView<'_>>,
) -> Result<DegreesResult> {
    let mut result: *mut cugraph_degrees_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();

    let src_ptr = source_vertices
        .map(|v| v.as_ptr() as *const _)
        .unwrap_or(ptr::null());

    unsafe {
        let code = cugraph_in_degrees(
            handle.as_ptr(),
            graph.raw,
            src_ptr,
            bool_t::FALSE, // do_expensive_check
            &mut result,
            &mut error,
        );
        check_error(code, error, "in_degrees")?;
    }

    Ok(DegreesResult {
        raw: result,
        _nosend_nosync: std::marker::PhantomData,
    })
}

/// Compute only out-degrees for vertices in a graph.
///
/// If `source_vertices` is `None`, computes out-degrees for all vertices.
/// Otherwise, computes out-degrees only for the specified vertices.
///
/// Returns a [`DegreesResult`]. [`DegreesResult::out_degrees()`] is expected to
/// be available. [`DegreesResult::in_degrees()`] may be unavailable for this
/// result and is therefore optional in the Rust API.
pub fn out_degrees(
    handle: &ResourceHandle,
    graph: &mut Graph,
    source_vertices: Option<&DeviceArrayView<'_>>,
) -> Result<DegreesResult> {
    let mut result: *mut cugraph_degrees_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();

    let src_ptr = source_vertices
        .map(|v| v.as_ptr() as *const _)
        .unwrap_or(ptr::null());

    unsafe {
        let code = cugraph_out_degrees(
            handle.as_ptr(),
            graph.raw,
            src_ptr,
            bool_t::FALSE, // do_expensive_check
            &mut result,
            &mut error,
        );
        check_error(code, error, "out_degrees")?;
    }

    Ok(DegreesResult {
        raw: result,
        _nosend_nosync: std::marker::PhantomData,
    })
}

/// Result of a degree computation.
///
/// Provides access to vertex ids, in-degrees, and out-degrees as device array
/// views. Freed on drop via `cugraph_degrees_result_free`.
pub struct DegreesResult {
    raw: *mut cugraph_degrees_result_t,
    _nosend_nosync: std::marker::PhantomData<*mut ()>,
}

impl DegreesResult {
    /// Get the vertex ids as a device array view.
    ///
    /// The view borrows `self`, so this result must outlive the returned view.
    pub fn vertices(&self) -> DeviceArrayView<'_> {
        let view_ptr = unsafe { cugraph_degrees_result_get_vertices(self.raw) };
        unsafe { DeviceArrayView::from_borrowed(view_ptr) }
    }

    /// Get the in-degrees as a device array view.
    ///
    /// Returns `None` if this result does not contain in-degrees (for example,
    /// a result produced by [`out_degrees`]).
    ///
    /// The view borrows `self`, so this result must outlive the returned view.
    pub fn in_degrees(&self) -> Option<DeviceArrayView<'_>> {
        let view_ptr = unsafe { cugraph_degrees_result_get_in_degrees(self.raw) };
        if view_ptr.is_null() {
            None
        } else {
            Some(unsafe { DeviceArrayView::from_borrowed(view_ptr) })
        }
    }

    /// Get the out-degrees as a device array view.
    ///
    /// Returns `None` if this result does not contain out-degrees (for example,
    /// a result produced by [`in_degrees`] on a non-symmetric graph).
    ///
    /// If the graph is symmetric, in-degrees and out-degrees are equal and may
    /// share the same underlying memory.
    ///
    /// The view borrows `self`, so this result must outlive the returned view.
    pub fn out_degrees(&self) -> Option<DeviceArrayView<'_>> {
        let view_ptr = unsafe { cugraph_degrees_result_get_out_degrees(self.raw) };
        if view_ptr.is_null() {
            None
        } else {
            Some(unsafe { DeviceArrayView::from_borrowed(view_ptr) })
        }
    }
}

impl Drop for DegreesResult {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { cugraph_degrees_result_free(self.raw) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::device_array_from_host;
    use crate::types::{DataType, GraphProperties};

    /// Helper: create a star graph where vertices 1,2,3 all point to vertex 0.
    /// With store_transposed=true and renumber=true (needed for degree computation).
    fn make_star_graph(handle: &ResourceHandle) -> Graph {
        // Directed star: 1->0, 2->0, 3->0
        let src_data: Vec<i32> = vec![1, 2, 3];
        let dst_data: Vec<i32> = vec![0, 0, 0];

        let src = device_array_from_host(handle, &src_data, DataType::Int32).expect("src");
        let dst = device_array_from_host(handle, &dst_data, DataType::Int32).expect("dst");

        let src_view = src.view();
        let dst_view = dst.view();

        let props = GraphProperties {
            is_symmetric: false,
            is_multigraph: false,
        };

        Graph::from_edge_list(
            handle, props, &src_view, &dst_view, None, None, None,
            true, // store_transposed (needed for in-degree computation)
            true, // renumber
        )
        .expect("star graph")
    }

    /// Helper: decode device array bytes into Vec<i32>
    fn to_i32_vec(bytes: &[u8]) -> Vec<i32> {
        bytes
            .chunks_exact(4)
            .map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Helper: decode device array bytes into Vec<i64>
    fn to_i64_vec(bytes: &[u8]) -> Vec<i64> {
        bytes
            .chunks_exact(8)
            .map(|c| i64::from_ne_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect()
    }

    /// Helper: build a vertex -> (in_degree, out_degree) map from a DegreesResult.
    /// The degree type may be i32 or i64 depending on the graph's edge_t, so we
    /// use the dtype to decide.
    fn degree_map(
        handle: &ResourceHandle,
        result: &DegreesResult,
    ) -> std::collections::HashMap<i32, (i64, i64)> {
        let vert_view = result.vertices();
        let in_view = result
            .in_degrees()
            .expect("degrees() result should include in_degrees");
        let out_view = result
            .out_degrees()
            .expect("degrees() result should include out_degrees");

        let vert_bytes = vert_view.to_host_vec(handle).expect("vert copy");
        let in_bytes = in_view.to_host_vec(handle).expect("in copy");
        let out_bytes = out_view.to_host_vec(handle).expect("out copy");

        let vertices = to_i32_vec(&vert_bytes);

        // Degree arrays can be edge_t which may be i32 or i64.
        let in_dtype = in_view.dtype().expect("in dtype");
        let in_degs: Vec<i64> = match in_dtype.size_of() {
            4 => to_i32_vec(&in_bytes)
                .into_iter()
                .map(|v| v as i64)
                .collect(),
            8 => to_i64_vec(&in_bytes),
            other => panic!("unexpected degree element size: {}", other),
        };

        let out_dtype = out_view.dtype().expect("out dtype");
        let out_degs: Vec<i64> = match out_dtype.size_of() {
            4 => to_i32_vec(&out_bytes)
                .into_iter()
                .map(|v| v as i64)
                .collect(),
            8 => to_i64_vec(&out_bytes),
            other => panic!("unexpected degree element size: {}", other),
        };

        let mut map = std::collections::HashMap::new();
        for i in 0..vertices.len() {
            map.insert(vertices[i], (in_degs[i], out_degs[i]));
        }
        map
    }

    #[test]
    fn test_degrees_star_graph() {
        let handle = ResourceHandle::new().expect("handle");
        let mut graph = make_star_graph(&handle);

        let result = degrees(&handle, &mut graph, None).expect("degrees");

        let vert_view = result.vertices();
        let in_view = result
            .in_degrees()
            .expect("degrees() result should include in_degrees");
        let out_view = result
            .out_degrees()
            .expect("degrees() result should include out_degrees");

        // Star with 4 vertices
        assert_eq!(vert_view.len(), 4);
        assert_eq!(in_view.len(), 4);
        assert_eq!(out_view.len(), 4);

        let map = degree_map(&handle, &result);

        // Vertex 0: hub — receives edges from 1,2,3 → in=3, out=0
        assert_eq!(map[&0], (3, 0), "hub should have in=3, out=0");

        // Vertices 1,2,3: leaves — each sends one edge → in=0, out=1
        for v in [1, 2, 3] {
            assert_eq!(map[&v], (0, 1), "leaf vertex {} should have in=0, out=1", v);
        }
    }

    #[test]
    fn test_in_degrees_star_graph() {
        let handle = ResourceHandle::new().expect("handle");
        let mut graph = make_star_graph(&handle);

        let result = in_degrees(&handle, &mut graph, None).expect("in_degrees");

        let vert_view = result.vertices();
        let in_view = result
            .in_degrees()
            .expect("in_degrees() result should include in_degrees");

        assert_eq!(vert_view.len(), 4);
        assert_eq!(in_view.len(), 4);
        assert!(
            result.out_degrees().is_none(),
            "in_degrees() result should not include out_degrees on non-symmetric graphs"
        );

        let vert_bytes = vert_view.to_host_vec(&handle).expect("vert copy");
        let in_bytes = in_view.to_host_vec(&handle).expect("in copy");
        let vertices = to_i32_vec(&vert_bytes);

        let in_dtype = in_view.dtype().expect("in dtype");
        let in_degs: Vec<i64> = match in_dtype.size_of() {
            4 => to_i32_vec(&in_bytes)
                .into_iter()
                .map(|v| v as i64)
                .collect(),
            8 => to_i64_vec(&in_bytes),
            other => panic!("unexpected degree element size: {}", other),
        };

        let mut map = std::collections::HashMap::new();
        for i in 0..vertices.len() {
            map.insert(vertices[i], in_degs[i]);
        }

        // Hub vertex 0 receives 3 edges
        assert_eq!(map[&0], 3, "hub should have in_degree=3");
        for v in [1, 2, 3] {
            assert_eq!(map[&v], 0, "leaf {} should have in_degree=0", v);
        }
    }

    #[test]
    fn test_out_degrees_star_graph() {
        let handle = ResourceHandle::new().expect("handle");
        let mut graph = make_star_graph(&handle);

        let result = out_degrees(&handle, &mut graph, None).expect("out_degrees");

        let vert_view = result.vertices();
        let out_view = result
            .out_degrees()
            .expect("out_degrees() result should include out_degrees");

        assert_eq!(vert_view.len(), 4);
        assert_eq!(out_view.len(), 4);
        assert!(
            result.in_degrees().is_none(),
            "out_degrees() result should not include in_degrees on non-symmetric graphs"
        );

        let vert_bytes = vert_view.to_host_vec(&handle).expect("vert copy");
        let out_bytes = out_view.to_host_vec(&handle).expect("out copy");
        let vertices = to_i32_vec(&vert_bytes);

        let out_dtype = out_view.dtype().expect("out dtype");
        let out_degs: Vec<i64> = match out_dtype.size_of() {
            4 => to_i32_vec(&out_bytes)
                .into_iter()
                .map(|v| v as i64)
                .collect(),
            8 => to_i64_vec(&out_bytes),
            other => panic!("unexpected degree element size: {}", other),
        };

        let mut map = std::collections::HashMap::new();
        for i in 0..vertices.len() {
            map.insert(vertices[i], out_degs[i]);
        }

        // Hub vertex 0 sends 0 edges
        assert_eq!(map[&0], 0, "hub should have out_degree=0");
        // Leaf vertices each send 1 edge
        for v in [1, 2, 3] {
            assert_eq!(map[&v], 1, "leaf {} should have out_degree=1", v);
        }
    }

    #[test]
    fn test_degrees_cycle_graph() {
        let handle = ResourceHandle::new().expect("handle");

        // Directed cycle: 0->1, 1->2, 2->3, 3->0
        let src_data: Vec<i32> = vec![0, 1, 2, 3];
        let dst_data: Vec<i32> = vec![1, 2, 3, 0];

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
            None,
            None,
            true,
            true,
        )
        .expect("graph");

        let result = degrees(&handle, &mut graph, None).expect("degrees");
        let map = degree_map(&handle, &result);

        // Every vertex in a directed cycle has in=1, out=1
        assert_eq!(map.len(), 4);
        for v in 0..4 {
            assert_eq!(
                map[&v],
                (1, 1),
                "vertex {} in cycle should have in=1, out=1",
                v
            );
        }
    }

    #[test]
    fn test_degrees_with_source_vertices_subset() {
        let handle = ResourceHandle::new().expect("handle");
        let mut graph = make_star_graph(&handle);

        // Only query degrees for vertices 0 and 1
        let query = device_array_from_host(&handle, &[0i32, 1], DataType::Int32).expect("query");

        let result = degrees(&handle, &mut graph, Some(&query.view())).expect("degrees subset");

        let vert_view = result.vertices();
        assert_eq!(vert_view.len(), 2, "should return degrees for 2 vertices");

        let map = degree_map(&handle, &result);
        assert!(map.contains_key(&0), "should contain vertex 0");
        assert!(map.contains_key(&1), "should contain vertex 1");
        assert_eq!(map[&0], (3, 0), "hub should have in=3, out=0");
        assert_eq!(map[&1], (0, 1), "leaf 1 should have in=0, out=1");
    }

    #[test]
    fn test_degrees_isolated_vertex() {
        let handle = ResourceHandle::new().expect("handle");

        // Graph: 0->1 (vertex 2 exists but is isolated via explicit edge 2->2
        // trick doesn't work cleanly; instead we create 0->1, 2->2 self-loop)
        // Actually, for cuGraph, isolated vertices that aren't endpoints of any
        // edge won't appear. Let's use a graph where vertex 2 has no edges by
        // making edges: 0->1, 3->4 — so vertices 0,1,3,4 appear, and we check
        // that all degrees are correct.
        let src_data: Vec<i32> = vec![0, 3];
        let dst_data: Vec<i32> = vec![1, 4];

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
            None,
            None,
            true,
            true,
        )
        .expect("graph");

        let result = degrees(&handle, &mut graph, None).expect("degrees");
        let map = degree_map(&handle, &result);

        // Vertex 0: out=1, in=0
        assert_eq!(map[&0], (0, 1), "vertex 0 should have in=0, out=1");
        // Vertex 1: out=0, in=1
        assert_eq!(map[&1], (1, 0), "vertex 1 should have in=1, out=0");
        // Vertex 3: out=1, in=0
        assert_eq!(map[&3], (0, 1), "vertex 3 should have in=0, out=1");
        // Vertex 4: out=0, in=1
        assert_eq!(map[&4], (1, 0), "vertex 4 should have in=1, out=0");
    }

    #[test]
    fn test_degrees_symmetric_graph() {
        let handle = ResourceHandle::new().expect("handle");

        // Symmetric (undirected) triangle: 0<->1, 1<->2, 2<->0
        let src_data: Vec<i32> = vec![0, 1, 1, 2, 2, 0];
        let dst_data: Vec<i32> = vec![1, 0, 2, 1, 0, 2];

        let src = device_array_from_host(&handle, &src_data, DataType::Int32).expect("src");
        let dst = device_array_from_host(&handle, &dst_data, DataType::Int32).expect("dst");

        let props = GraphProperties {
            is_symmetric: true,
            is_multigraph: false,
        };

        let mut graph = Graph::from_edge_list(
            &handle,
            props,
            &src.view(),
            &dst.view(),
            None,
            None,
            None,
            true,
            true,
        )
        .expect("graph");

        let result = degrees(&handle, &mut graph, None).expect("degrees");
        let map = degree_map(&handle, &result);

        // Symmetric graph: in_degree == out_degree for every vertex
        assert_eq!(map.len(), 3);
        for v in 0..3 {
            let (in_d, out_d) = map[&v];
            assert_eq!(
                in_d, out_d,
                "symmetric graph vertex {} should have in_degree == out_degree",
                v
            );
            assert_eq!(in_d, 2, "vertex {} should have degree 2 in triangle", v);
        }
    }
}
