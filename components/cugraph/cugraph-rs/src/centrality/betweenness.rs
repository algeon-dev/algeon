//! Vertex betweenness centrality algorithm.
//!
//! Betweenness centrality measures how often a vertex appears on shortest paths
//! between all pairs of vertices in the graph. It can be computed exactly
//! (traversal from every source) or approximately (using a subset of seed
//! vertices).
//!
//! # Example
//!
//! ```no_run
//! use cugraph::{
//!     array::device_array_from_host,
//!     centrality::{BetweennessCentralityParams, betweenness_centrality},
//!     graph::Graph,
//!     handle::ResourceHandle,
//!     types::{DataType, GraphProperties},
//! };
//!
//! let handle = ResourceHandle::new().unwrap();
//!
//! let src =
//!     device_array_from_host(&handle, &[0i32, 1, 2, 3], DataType::Int32)
//!         .unwrap();
//! let dst =
//!     device_array_from_host(&handle, &[1i32, 2, 3, 0], DataType::Int32)
//!         .unwrap();
//!
//! let props = GraphProperties { is_symmetric: false, is_multigraph: false };
//! let mut graph = Graph::from_edge_list(
//!     &handle,
//!     props,
//!     &src.view(),
//!     &dst.view(),
//!     None,
//!     None,
//!     None,
//!     false,
//!     true,
//! )
//! .unwrap();
//!
//! let params = BetweennessCentralityParams::default();
//! let result =
//!     betweenness_centrality(&handle, &mut graph, None, &params).unwrap();
//! // result.vertices() and result.values() provide the centrality scores.
//! ```

use std::ptr;

use cugraph_sys::{bool_t, cugraph_betweenness_centrality, cugraph_centrality_result_t, cugraph_error_t};

use super::CentralityResult;
use crate::{
    array::DeviceArrayView,
    error::{Result, check_error},
    graph::Graph,
    handle::ResourceHandle,
    types::to_bool_t,
};

/// Parameters for the betweenness centrality algorithm.
#[derive(Debug, Clone, Copy)]
pub struct BetweennessCentralityParams {
    /// Whether to normalize the result. When `true`, values are scaled so that
    /// the maximum possible betweenness is 1.0 (default: `true`).
    pub normalized: bool,
    /// Whether to include endpoints when considering a vertex to be on a
    /// shortest path. The traditional formulation does not include endpoints
    /// (default: `false`).
    pub include_endpoints: bool,
}

impl Default for BetweennessCentralityParams {
    fn default() -> Self {
        Self { normalized: true, include_endpoints: false }
    }
}

/// Compute vertex betweenness centrality on a graph.
///
/// Betweenness centrality can be computed exactly by passing `vertex_list` as
/// `None`. This performs a traversal from every source vertex.
///
/// Approximate betweenness centrality can be computed by passing a
/// `vertex_list` containing seed vertices to use as traversal sources. This
/// corresponds to the Python `cugraph.betweenness_centrality(G, k=seeds)`
/// usage.
///
/// Returns a [`CentralityResult`] containing vertex ids and their betweenness
/// centrality values.
pub fn betweenness_centrality(
    handle: &ResourceHandle,
    graph: &mut Graph,
    vertex_list: Option<&DeviceArrayView<'_>>,
    params: &BetweennessCentralityParams,
) -> Result<CentralityResult> {
    let mut result: *mut cugraph_centrality_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();

    let vertex_list_ptr = vertex_list.map(|v| v.as_ptr() as *const _).unwrap_or(ptr::null());

    unsafe {
        let code = cugraph_betweenness_centrality(
            handle.as_ptr(),
            graph.raw,
            vertex_list_ptr,
            to_bool_t(params.normalized),
            to_bool_t(params.include_endpoints),
            bool_t::FALSE, // do_expensive_check
            &mut result,
            &mut error,
        );
        check_error(code, error, "betweenness_centrality")?;
    }

    Ok(unsafe { CentralityResult::from_raw(result) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        array::device_array_from_host,
        types::{DataType, GraphProperties},
    };

    /// Helper: decode device array bytes into Vec<i32>.
    fn to_i32_vec(bytes: &[u8]) -> Vec<i32> {
        bytes.chunks_exact(4).map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    /// Helper: decode device array bytes into Vec<f32>.
    fn to_f32_vec(bytes: &[u8]) -> Vec<f32> {
        bytes.chunks_exact(4).map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    /// Helper: build a vertex -> betweenness map from a CentralityResult.
    fn betweenness_map(handle: &ResourceHandle, result: &CentralityResult) -> std::collections::HashMap<i32, f32> {
        let vert_bytes = result.vertices().to_host_vec(handle).expect("vert copy");
        let val_bytes = result.values().to_host_vec(handle).expect("val copy");

        let vertices = to_i32_vec(&vert_bytes);
        let values = to_f32_vec(&val_bytes);

        vertices.into_iter().zip(values.into_iter()).collect()
    }

    #[test]
    fn test_betweenness_centrality_star_graph() {
        // Star graph: 1->0, 2->0, 3->0, 0->1, 0->2, 0->3
        // The hub (vertex 0) lies on shortest paths between all leaf pairs,
        // so it should have the highest betweenness centrality.
        let handle = ResourceHandle::new().expect("handle");

        let src_data: Vec<i32> = vec![1, 2, 3, 0, 0, 0];
        let dst_data: Vec<i32> = vec![0, 0, 0, 1, 2, 3];

        let src = device_array_from_host(&handle, &src_data, DataType::Int32).expect("src");
        let dst = device_array_from_host(&handle, &dst_data, DataType::Int32).expect("dst");

        let props = GraphProperties { is_symmetric: false, is_multigraph: false };

        let mut graph = Graph::from_edge_list(
            &handle,
            props,
            &src.view(),
            &dst.view(),
            None,
            None,
            None,
            false, // store_transposed not required for betweenness
            true,  // renumber
        )
        .expect("graph");

        let params = BetweennessCentralityParams::default();
        let result = betweenness_centrality(&handle, &mut graph, None, &params).expect("betweenness");

        let map = betweenness_map(&handle, &result);

        assert_eq!(map.len(), 4, "should have 4 vertices");

        // Hub vertex 0 should have the highest betweenness
        let hub_bc = map[&0];
        for v in [1, 2, 3] {
            assert!(hub_bc > map[&v], "hub betweenness {} should be > leaf {} betweenness {}", hub_bc, v, map[&v]);
        }

        // All leaves should have equal betweenness (by symmetry)
        let leaf_bc = map[&1];
        for v in [2, 3] {
            assert!(
                (map[&v] - leaf_bc).abs() < 1e-6,
                "leaf {} betweenness {} should equal leaf 1 betweenness {}",
                v,
                map[&v],
                leaf_bc
            );
        }
    }

    #[test]
    fn test_betweenness_centrality_line_graph() {
        // Line graph: 0->1->2->3->4 (directed path)
        // The middle vertex (vertex 2) should have the highest betweenness.
        let handle = ResourceHandle::new().expect("handle");

        let src_data: Vec<i32> = vec![0, 1, 2, 3];
        let dst_data: Vec<i32> = vec![1, 2, 3, 4];

        let src = device_array_from_host(&handle, &src_data, DataType::Int32).expect("src");
        let dst = device_array_from_host(&handle, &dst_data, DataType::Int32).expect("dst");

        let props = GraphProperties { is_symmetric: false, is_multigraph: false };

        let mut graph = Graph::from_edge_list(&handle, props, &src.view(), &dst.view(), None, None, None, false, true)
            .expect("graph");

        let params = BetweennessCentralityParams::default();
        let result = betweenness_centrality(&handle, &mut graph, None, &params).expect("betweenness");

        let map = betweenness_map(&handle, &result);

        assert_eq!(map.len(), 5, "should have 5 vertices");

        // In a directed path, interior vertices have higher betweenness than
        // endpoints. Vertex 2 is the most central.
        let bc_0 = map[&0];
        let bc_2 = map[&2];
        let bc_4 = map[&4];

        assert!(bc_2 >= bc_0, "middle vertex bc {} should be >= endpoint bc {}", bc_2, bc_0);
        assert!(bc_2 >= bc_4, "middle vertex bc {} should be >= endpoint bc {}", bc_2, bc_4);
    }

    #[test]
    fn test_betweenness_centrality_with_seed_vertices() {
        // Use a subset of vertices as seeds for approximate betweenness.
        // This corresponds to the Python `k=seed_series` parameter.
        let handle = ResourceHandle::new().expect("handle");

        // Directed star: 1->0, 2->0, 3->0, 0->1, 0->2, 0->3
        let src_data: Vec<i32> = vec![1, 2, 3, 0, 0, 0];
        let dst_data: Vec<i32> = vec![0, 0, 0, 1, 2, 3];

        let src = device_array_from_host(&handle, &src_data, DataType::Int32).expect("src");
        let dst = device_array_from_host(&handle, &dst_data, DataType::Int32).expect("dst");

        let props = GraphProperties { is_symmetric: false, is_multigraph: false };

        let mut graph = Graph::from_edge_list(&handle, props, &src.view(), &dst.view(), None, None, None, false, true)
            .expect("graph");

        // Use only vertices 0 and 1 as seeds
        let seeds = device_array_from_host(&handle, &[0i32, 1], DataType::Int32).expect("seeds");

        let params = BetweennessCentralityParams::default();
        let result =
            betweenness_centrality(&handle, &mut graph, Some(&seeds.view()), &params).expect("betweenness with seeds");

        let map = betweenness_map(&handle, &result);

        // Should still have results for all 4 vertices
        assert_eq!(map.len(), 4, "should have results for all 4 vertices");

        // All betweenness values should be non-negative
        for (&v, &bc) in &map {
            assert!(bc >= 0.0, "vertex {} betweenness {} should be non-negative", v, bc);
        }
    }

    #[test]
    fn test_betweenness_centrality_unnormalized() {
        // Test with normalized=false
        let handle = ResourceHandle::new().expect("handle");

        // Triangle: 0->1, 1->2, 2->0
        let src_data: Vec<i32> = vec![0, 1, 2];
        let dst_data: Vec<i32> = vec![1, 2, 0];

        let src = device_array_from_host(&handle, &src_data, DataType::Int32).expect("src");
        let dst = device_array_from_host(&handle, &dst_data, DataType::Int32).expect("dst");

        let props = GraphProperties { is_symmetric: false, is_multigraph: false };

        let mut graph = Graph::from_edge_list(&handle, props, &src.view(), &dst.view(), None, None, None, false, true)
            .expect("graph");

        let params = BetweennessCentralityParams { normalized: false, include_endpoints: false };
        let result = betweenness_centrality(&handle, &mut graph, None, &params).expect("betweenness");

        let map = betweenness_map(&handle, &result);

        assert_eq!(map.len(), 3, "should have 3 vertices");

        // All values should be non-negative
        for (&v, &bc) in &map {
            assert!(bc >= 0.0, "vertex {} betweenness {} should be non-negative", v, bc);
        }
    }

    #[test]
    fn test_betweenness_centrality_with_endpoints() {
        // Test with include_endpoints=true — values should generally be higher
        // than without endpoints.
        let handle = ResourceHandle::new().expect("handle");

        // Line graph: 0->1->2
        let src_data: Vec<i32> = vec![0, 1];
        let dst_data: Vec<i32> = vec![1, 2];

        let src = device_array_from_host(&handle, &src_data, DataType::Int32).expect("src");
        let dst = device_array_from_host(&handle, &dst_data, DataType::Int32).expect("dst");

        let props = GraphProperties { is_symmetric: false, is_multigraph: false };

        // Without endpoints
        let mut graph_a =
            Graph::from_edge_list(&handle, props, &src.view(), &dst.view(), None, None, None, false, true)
                .expect("graph_a");

        let params_no_ep = BetweennessCentralityParams { normalized: false, include_endpoints: false };
        let result_no_ep = betweenness_centrality(&handle, &mut graph_a, None, &params_no_ep).expect("bc no ep");
        let map_no_ep = betweenness_map(&handle, &result_no_ep);

        // With endpoints
        let mut graph_b =
            Graph::from_edge_list(&handle, props, &src.view(), &dst.view(), None, None, None, false, true)
                .expect("graph_b");

        let params_with_ep = BetweennessCentralityParams { normalized: false, include_endpoints: true };
        let result_with_ep = betweenness_centrality(&handle, &mut graph_b, None, &params_with_ep).expect("bc with ep");
        let map_with_ep = betweenness_map(&handle, &result_with_ep);

        // With endpoints, the sum of betweenness values should be >=
        // without endpoints (endpoints adds credit for being a source/target).
        let sum_no_ep: f32 = map_no_ep.values().sum();
        let sum_with_ep: f32 = map_with_ep.values().sum();

        assert!(
            sum_with_ep >= sum_no_ep,
            "sum with endpoints ({}) should be >= sum without ({})",
            sum_with_ep,
            sum_no_ep
        );
    }

    #[test]
    fn test_betweenness_centrality_result_vertex_count_matches() {
        // Ensure the result has the same number of vertices as the graph.
        let handle = ResourceHandle::new().expect("handle");

        // Complete directed graph on 5 vertices (20 edges)
        let mut src_data = Vec::new();
        let mut dst_data = Vec::new();
        for i in 0..5i32 {
            for j in 0..5i32 {
                if i != j {
                    src_data.push(i);
                    dst_data.push(j);
                }
            }
        }

        let src = device_array_from_host(&handle, &src_data, DataType::Int32).expect("src");
        let dst = device_array_from_host(&handle, &dst_data, DataType::Int32).expect("dst");

        let props = GraphProperties { is_symmetric: false, is_multigraph: false };

        let mut graph = Graph::from_edge_list(&handle, props, &src.view(), &dst.view(), None, None, None, false, true)
            .expect("graph");

        let params = BetweennessCentralityParams::default();
        let result = betweenness_centrality(&handle, &mut graph, None, &params).expect("betweenness");

        assert_eq!(result.vertices().len(), 5, "should have 5 vertices");
        assert_eq!(result.values().len(), 5, "should have 5 values");

        // In a complete directed graph, all vertices should have equal
        // betweenness centrality (by symmetry).
        let map = betweenness_map(&handle, &result);
        let first_bc = *map.values().next().unwrap();
        for (&v, &bc) in &map {
            assert!((bc - first_bc).abs() < 1e-6, "vertex {} betweenness {} should equal {}", v, bc, first_bc);
        }
    }
}
