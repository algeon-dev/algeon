//! PageRank centrality algorithm.

use std::ptr;

use cugraph_sys::{
    bool_t, cugraph_centrality_result_converged, cugraph_centrality_result_free,
    cugraph_centrality_result_get_num_iterations, cugraph_centrality_result_get_values,
    cugraph_centrality_result_get_vertices, cugraph_centrality_result_t, cugraph_error_t,
    cugraph_pagerank,
};

use crate::array::DeviceArrayView;
use crate::error::{check_error, Error, ErrorKind, Result};
use crate::graph::Graph;
use crate::handle::ResourceHandle;
use crate::types::try_from_bool_t;

/// Parameters for the PageRank algorithm.
#[derive(Debug, Clone, Copy)]
pub struct PageRankParams {
    /// Damping factor (default 0.85).
    pub alpha: f64,
    /// Convergence tolerance (default 1e-5).
    pub epsilon: f64,
    /// Maximum number of iterations (default 100).
    pub max_iterations: usize,
}

impl Default for PageRankParams {
    fn default() -> Self {
        Self {
            alpha: 0.85,
            epsilon: 1e-5,
            max_iterations: 100,
        }
    }
}

/// Compute PageRank on a graph.
///
/// Returns a [`PageRankResult`] containing vertex ids and their PageRank
/// values. The result must be kept alive as long as any [`DeviceArrayView`]
/// borrowed from it is in use.
pub fn pagerank(
    handle: &ResourceHandle,
    graph: &mut Graph,
    params: &PageRankParams,
) -> Result<PageRankResult> {
    let mut result: *mut cugraph_centrality_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();

    unsafe {
        let code = cugraph_pagerank(
            handle.as_ptr(),
            graph.raw,
            ptr::null(), // precomputed_vertex_out_weight_vertices
            ptr::null(), // precomputed_vertex_out_weight_sums
            ptr::null(), // initial_guess_vertices
            ptr::null(), // initial_guess_values
            params.alpha,
            params.epsilon,
            params.max_iterations,
            bool_t::FALSE, // do_expensive_check
            &mut result,
            &mut error,
        );
        check_error(code, error, "pagerank")?;
    }

    Ok(PageRankResult {
        raw: result,
        _nosend_nosync: std::marker::PhantomData,
    })
}

/// Result of a PageRank computation.
///
/// Provides access to vertex ids, their PageRank values, iteration count, and
/// convergence status. Freed on drop.
pub struct PageRankResult {
    raw: *mut cugraph_centrality_result_t,
    _nosend_nosync: std::marker::PhantomData<*mut ()>,
}

impl PageRankResult {
    /// Get the vertex ids as a device array view.
    ///
    /// The view borrows `self`, so this result must outlive the returned view.
    pub fn vertices(&self) -> DeviceArrayView<'_> {
        let view_ptr = unsafe { cugraph_centrality_result_get_vertices(self.raw) };
        unsafe { DeviceArrayView::from_borrowed(view_ptr) }
    }

    /// Get the PageRank values as a device array view.
    ///
    /// The view borrows `self`, so this result must outlive the returned view.
    pub fn values(&self) -> DeviceArrayView<'_> {
        let view_ptr = unsafe { cugraph_centrality_result_get_values(self.raw) };
        unsafe { DeviceArrayView::from_borrowed(view_ptr) }
    }

    /// Number of iterations performed.
    pub fn num_iterations(&self) -> usize {
        unsafe { cugraph_centrality_result_get_num_iterations(self.raw) }
    }

    /// Whether the algorithm converged within the maximum iterations.
    pub fn converged(&self) -> Result<bool> {
        let b = unsafe { cugraph_centrality_result_converged(self.raw) };
        try_from_bool_t(b).ok_or_else(|| {
            Error::new(ErrorKind::Unexpected, "PageRankResult::converged").with_context(
                "message",
                format!("invalid bool_t value returned by cuGraph: {}", b.0),
            )
        })
    }
}

impl Drop for PageRankResult {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { cugraph_centrality_result_free(self.raw) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::device_array_from_host;
    use crate::types::{DataType, GraphProperties};

    #[test]
    fn test_pagerank_star_graph() {
        let handle = ResourceHandle::new().expect("handle");

        // Star graph: 1->0, 2->0, 3->0 (node 0 is the hub)
        let src_data: Vec<i32> = vec![1, 2, 3, 0, 0, 0];
        let dst_data: Vec<i32> = vec![0, 0, 0, 1, 2, 3];

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
            true,  // store_transposed (required for PageRank)
            false, // renumber (not needed, IDs are 0-based sequential)
        )
        .expect("graph");

        let params = PageRankParams::default();
        let result = pagerank(&handle, &mut graph, &params).expect("pagerank");

        assert!(
            result.converged().expect("converged"),
            "PageRank should converge"
        );
        assert!(
            result.num_iterations() > 0,
            "should do at least 1 iteration"
        );

        let vertices_view = result.vertices();
        let values_view = result.values();

        assert_eq!(vertices_view.len(), 4);
        assert_eq!(values_view.len(), 4);

        // Copy results to host and verify hub (node 0) has the highest rank
        let vert_bytes = vertices_view.to_host_vec(&handle).expect("vert copy");
        let val_bytes = values_view.to_host_vec(&handle).expect("val copy");

        let vertices: Vec<i32> = vert_bytes
            .chunks_exact(4)
            .map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let values: Vec<f32> = val_bytes
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Find the PageRank of vertex 0 (hub) — should be highest
        let hub_idx = vertices.iter().position(|&v| v == 0).expect("hub vertex");
        let hub_rank = values[hub_idx];

        for (i, &v) in vertices.iter().enumerate() {
            if v != 0 {
                assert!(
                    hub_rank > values[i],
                    "hub rank {} should be > leaf rank {} (vertex {})",
                    hub_rank,
                    values[i],
                    v
                );
            }
        }
    }
}
