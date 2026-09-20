//! Vertex betweenness centrality algorithm.
//!
//! Betweenness centrality measures how often a vertex appears on shortest paths
//! between all pairs of vertices in the graph. It can be computed exactly
//! (traversal from every source) or approximately (using a subset of seed
//! vertices).

use std::ptr;

use cugraph_sys::{cugraph_betweenness_centrality, cugraph_centrality_result_t, cugraph_error_t};

use super::BetweennessCentralityResult;
use crate::{
    array::DeviceArrayView,
    error::{Result, check_error, require_non_null},
    graph::Graph,
    handle::EnteredContext,
    types::{CugraphElement, to_bool_t},
};

/// Parameters for the betweenness centrality algorithm.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct BetweennessCentralityParams {
    /// Whether to normalize the result. When `true`, values are scaled so that
    /// the maximum possible betweenness is 1.0 (default: `true`).
    pub normalized: bool,
    /// Whether to include endpoints when considering a vertex to be on a
    /// shortest path. The traditional formulation does not include endpoints
    /// (default: `false`).
    pub include_endpoints: bool,
    /// Whether to run expensive input validation checks in the C library
    /// (default: `false`).
    pub do_expensive_check: bool,
}

impl Default for BetweennessCentralityParams {
    fn default() -> Self {
        Self { normalized: true, include_endpoints: false, do_expensive_check: false }
    }
}

/// Compute vertex betweenness centrality on a graph.
///
/// Prefer using [`Graph::betweenness_centrality`](crate::graph::Graph::betweenness_centrality)
/// instead of calling this function directly.
pub(crate) fn betweenness_centrality<T, V: CugraphElement>(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<T>,
    vertex_list: Option<&DeviceArrayView<'_, V>>,
    params: &BetweennessCentralityParams,
) -> Result<BetweennessCentralityResult> {
    let mut result: *mut cugraph_centrality_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();

    let vertex_list_ptr = vertex_list.map(|v| v.as_ptr() as *const _).unwrap_or(ptr::null());
    let do_expensive_check = params.do_expensive_check || vertex_list.is_some();
    let graph_ptr = graph.as_ptr(handle, "betweenness_centrality")?;

    // SAFETY: `vertex_list_ptr` is either null or borrowed from a live device
    // view validated against the graph vertex dtype. Non-null vertex lists
    // force libcugraph validation so out-of-range vertices cannot drive
    // unchecked native indexing. `result`/`error` are valid out-pointers.
    unsafe {
        let code = cugraph_betweenness_centrality(
            handle.handle_ptr(),
            graph_ptr,
            vertex_list_ptr,
            to_bool_t(params.normalized),
            to_bool_t(params.include_endpoints),
            to_bool_t(do_expensive_check),
            &mut result,
            &mut error,
        );
        check_error(code, error, handle.device(), "betweenness_centrality")?;
    }

    let result = require_non_null(result, "betweenness_centrality", "centrality_result")?;

    // SAFETY: on success, cuGraph returns ownership of `result` to the caller.
    unsafe { BetweennessCentralityResult::from_raw(result, handle.device(), handle.runtime_bound_lease()) }
}

#[cfg(test)]
mod tests {
    use super::BetweennessCentralityParams;
    use crate::{
        ResourceHandle,
        array::DeviceArray,
        centrality::BetweennessCentralityResult,
        graph::{Forward, GraphBuilder},
        handle::EnteredContext,
    };

    /// Helper: build a vertex -> betweenness map from a BetweennessCentralityResult.
    fn betweenness_map(
        handle: &EnteredContext<'_>,
        result: &BetweennessCentralityResult,
    ) -> std::collections::HashMap<i32, f32> {
        let vertices: Vec<i32> = result.vertex_ids().expect("vertex_ids").to_vec_i32(handle).expect("vert copy");
        let values: Vec<f32> = result.scores().expect("scores").to_vec_f32(handle).expect("val copy");

        vertices.into_iter().zip(values).collect()
    }

    #[test]
    fn test_betweenness_centrality_star_graph() {
        // Star graph: 1->0, 2->0, 3->0, 0->1, 0->2, 0->3
        // The hub (vertex 0) lies on shortest paths between all leaf pairs,
        // so it should have the highest betweenness centrality.
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_betweenness_centrality_star_graph").expect("entered");

        let src_data: Vec<i32> = vec![1, 2, 3, 0, 0, 0];
        let dst_data: Vec<i32> = vec![0, 0, 0, 1, 2, 3];

        let src = DeviceArray::from_slice(&entered, &src_data).expect("src");
        let dst = DeviceArray::from_slice(&entered, &dst_data).expect("dst");
        let src_view = src.view().expect("src_view");
        let dst_view = dst.view().expect("dst_view");

        let mut graph = GraphBuilder::new(&src_view, &dst_view).renumber().build::<Forward>(&entered).expect("graph");

        let params = BetweennessCentralityParams::default();
        let result = graph.betweenness_centrality_all(&entered, &params).expect("betweenness");

        let map = betweenness_map(&entered, &result);

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
        let entered = handle.enter("test_betweenness_centrality_line_graph").expect("entered");

        let src_data: Vec<i32> = vec![0, 1, 2, 3];
        let dst_data: Vec<i32> = vec![1, 2, 3, 4];

        let src = DeviceArray::from_slice(&entered, &src_data).expect("src");
        let dst = DeviceArray::from_slice(&entered, &dst_data).expect("dst");
        let src_view = src.view().expect("src_view");
        let dst_view = dst.view().expect("dst_view");

        let mut graph = GraphBuilder::new(&src_view, &dst_view).renumber().build::<Forward>(&entered).expect("graph");

        let params = BetweennessCentralityParams::default();
        let result = graph.betweenness_centrality_all(&entered, &params).expect("betweenness");

        let map = betweenness_map(&entered, &result);

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
        let entered = handle.enter("test_betweenness_centrality_with_seed_vertices").expect("entered");

        // Directed star: 1->0, 2->0, 3->0, 0->1, 0->2, 0->3
        let src_data: Vec<i32> = vec![1, 2, 3, 0, 0, 0];
        let dst_data: Vec<i32> = vec![0, 0, 0, 1, 2, 3];

        let src = DeviceArray::from_slice(&entered, &src_data).expect("src");
        let dst = DeviceArray::from_slice(&entered, &dst_data).expect("dst");
        let src_view = src.view().expect("src_view");
        let dst_view = dst.view().expect("dst_view");

        let mut graph = GraphBuilder::new(&src_view, &dst_view).renumber().build::<Forward>(&entered).expect("graph");

        // Use only vertices 0 and 1 as seeds
        let seeds = DeviceArray::from_slice(&entered, &[0i32, 1]).expect("seeds");
        let seeds_view = seeds.view().expect("seeds_view");

        let params = BetweennessCentralityParams::default();
        let result = graph.betweenness_centrality(&entered, &seeds_view, &params).expect("betweenness with seeds");

        let map = betweenness_map(&entered, &result);

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
        let entered = handle.enter("test_betweenness_centrality_unnormalized").expect("entered");

        // Triangle: 0->1, 1->2, 2->0
        let src_data: Vec<i32> = vec![0, 1, 2];
        let dst_data: Vec<i32> = vec![1, 2, 0];

        let src = DeviceArray::from_slice(&entered, &src_data).expect("src");
        let dst = DeviceArray::from_slice(&entered, &dst_data).expect("dst");
        let src_view = src.view().expect("src_view");
        let dst_view = dst.view().expect("dst_view");

        let mut graph = GraphBuilder::new(&src_view, &dst_view).renumber().build::<Forward>(&entered).expect("graph");

        let params = BetweennessCentralityParams { normalized: false, include_endpoints: false, ..Default::default() };
        let result = graph.betweenness_centrality_all(&entered, &params).expect("betweenness");

        let map = betweenness_map(&entered, &result);

        assert_eq!(map.len(), 3, "should have 3 vertices");

        // All values should be non-negative
        for (&v, &bc) in &map {
            assert!(bc >= 0.0, "vertex {} betweenness {} should be non-negative", v, bc);
        }
    }

    #[test]
    fn test_betweenness_centrality_with_endpoints() {
        // Test with include_endpoints=true; values should generally be higher
        // than without endpoints.
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_betweenness_centrality_with_endpoints").expect("entered");

        // Line graph: 0->1->2
        let src_data: Vec<i32> = vec![0, 1];
        let dst_data: Vec<i32> = vec![1, 2];

        let src = DeviceArray::from_slice(&entered, &src_data).expect("src");
        let dst = DeviceArray::from_slice(&entered, &dst_data).expect("dst");
        let src_view = src.view().expect("src_view");
        let dst_view = dst.view().expect("dst_view");

        // Without endpoints
        let mut graph_a =
            GraphBuilder::new(&src_view, &dst_view).renumber().build::<Forward>(&entered).expect("graph_a");

        let params_no_ep =
            BetweennessCentralityParams { normalized: false, include_endpoints: false, ..Default::default() };
        let result_no_ep = graph_a.betweenness_centrality_all(&entered, &params_no_ep).expect("bc no ep");
        let map_no_ep = betweenness_map(&entered, &result_no_ep);

        // With endpoints
        let mut graph_b =
            GraphBuilder::new(&src_view, &dst_view).renumber().build::<Forward>(&entered).expect("graph_b");

        let params_with_ep =
            BetweennessCentralityParams { normalized: false, include_endpoints: true, ..Default::default() };
        let result_with_ep = graph_b.betweenness_centrality_all(&entered, &params_with_ep).expect("bc with ep");
        let map_with_ep = betweenness_map(&entered, &result_with_ep);

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
        let entered = handle.enter("test_betweenness_centrality_result_vertex_count_matches").expect("entered");

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

        let src = DeviceArray::from_slice(&entered, &src_data).expect("src");
        let dst = DeviceArray::from_slice(&entered, &dst_data).expect("dst");
        let src_view = src.view().expect("src_view");
        let dst_view = dst.view().expect("dst_view");

        let mut graph = GraphBuilder::new(&src_view, &dst_view).renumber().build::<Forward>(&entered).expect("graph");

        let params = BetweennessCentralityParams::default();
        let result = graph.betweenness_centrality_all(&entered, &params).expect("betweenness");

        assert_eq!(result.vertex_ids().expect("vertex_ids").len(), 5, "should have 5 vertices");
        assert_eq!(result.scores().expect("scores").len(), 5, "should have 5 values");

        // In a complete directed graph, all vertices should have equal
        // betweenness centrality (by symmetry).
        let map = betweenness_map(&entered, &result);
        let first_bc = *map.values().next().unwrap();
        for (&v, &bc) in &map {
            assert!((bc - first_bc).abs() < 1e-6, "vertex {} betweenness {} should equal {}", v, bc, first_bc);
        }
    }
}
