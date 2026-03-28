//! PageRank centrality algorithm.

use std::ptr;

use cugraph_sys::{bool_t, cugraph_centrality_result_t, cugraph_error_t, cugraph_pagerank};

use super::PageRankResult;
use crate::{
    error::{Result, check_error},
    graph::{Graph, Transposed},
    handle::ResourceHandle,
    params::{ClosedUnitF64, IterationCount, NonNegativeF64},
};

/// Parameters for the PageRank algorithm.
///
/// All fields use strongly-typed wrappers that enforce invariants at
/// construction time:
///
/// - `alpha` ([`ClosedUnitF64`]): damping factor, must be in `[0.0, 1.0]`
/// - `epsilon` ([`NonNegativeF64`]): convergence tolerance, must be `>= 0.0`
///   (zero means "never converge early")
/// - `max_iterations` ([`IterationCount`]): must be `>= 1`
///
/// Use the [`Default`] implementation for standard values, or construct
/// individual fields using `TryFrom`/`.new()`.
///
/// # Example
///
/// ```
/// use cugraph::centrality::PageRankParams;
///
/// // Default parameters
/// let params = PageRankParams::default();
/// assert_eq!(params.alpha.get(), 0.85);
///
/// // Custom parameters
/// use cugraph::params::{ClosedUnitF64, IterationCount, NonNegativeF64};
/// let params = PageRankParams {
///     alpha: ClosedUnitF64::new(0.9).unwrap(),
///     epsilon: NonNegativeF64::new(1e-6).unwrap(),
///     max_iterations: IterationCount::new(200).unwrap(),
/// };
/// ```
#[derive(Debug, Clone, Copy)]
pub struct PageRankParams {
    /// Damping factor (default 0.85). Must be in the closed interval `[0.0, 1.0]`.
    pub alpha: ClosedUnitF64,
    /// Convergence tolerance (default 1e-5). Must be non-negative. Zero
    /// disables early convergence (always runs for `max_iterations`).
    pub epsilon: NonNegativeF64,
    /// Maximum number of iterations (default 100). Must be at least 1.
    pub max_iterations: IterationCount,
}

impl Default for PageRankParams {
    fn default() -> Self {
        Self {
            // SAFETY: these literals are known-valid constants.
            alpha: ClosedUnitF64::new(0.85).unwrap(),
            epsilon: NonNegativeF64::new(1e-5).unwrap(),
            max_iterations: IterationCount::new(100).unwrap(),
        }
    }
}

/// Compute PageRank on a graph.
///
/// Prefer using [`Graph::pagerank`](crate::graph::Graph::pagerank) instead
/// of calling this function directly.
pub(crate) fn pagerank(
    handle: &ResourceHandle,
    graph: &mut Graph<Transposed>,
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
            params.alpha.get(),
            params.epsilon.get(),
            usize::from(params.max_iterations),
            bool_t::FALSE, // do_expensive_check
            &mut result,
            &mut error,
        );
        check_error(code, error, "pagerank")?;
    }

    Ok(unsafe { PageRankResult::from_raw(result) })
}

#[cfg(test)]
mod tests {
    use super::PageRankParams;
    use crate::{
        array::DeviceArray,
        graph::{GraphBuilder, Transposed},
        handle::ResourceHandle,
    };

    #[test]
    fn test_pagerank_star_graph() {
        let handle = ResourceHandle::new().expect("handle");

        // Star graph: 1->0, 2->0, 3->0 (node 0 is the hub)
        let src_data: Vec<i32> = vec![1, 2, 3, 0, 0, 0];
        let dst_data: Vec<i32> = vec![0, 0, 0, 1, 2, 3];

        let src = DeviceArray::from_slice(&handle, &src_data).expect("src");
        let dst = DeviceArray::from_slice(&handle, &dst_data).expect("dst");

        let src_view = src.view();
        let dst_view = dst.view();
        let mut graph = GraphBuilder::new(&src_view, &dst_view).build::<Transposed>(&handle).expect("graph");

        let params = PageRankParams::default();
        let result = graph.pagerank(&handle, &params).expect("pagerank");

        assert!(result.converged().expect("converged"), "PageRank should converge");
        assert!(result.num_iterations() > 0, "should do at least 1 iteration");

        let vertices_view = result.vertex_ids().expect("vertex_ids");
        let values_view = result.scores().expect("scores");

        assert_eq!(vertices_view.len(), 4);
        assert_eq!(values_view.len(), 4);

        // Copy results to host and verify hub (node 0) has the highest rank
        let vertices: Vec<i32> = vertices_view.to_vec_i32(&handle).expect("vert copy");
        let values: Vec<f32> = values_view.to_vec_f32(&handle).expect("val copy");

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
