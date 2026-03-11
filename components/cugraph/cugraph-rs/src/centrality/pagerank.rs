//! PageRank centrality algorithm.

use std::ptr;

use cugraph_sys::{bool_t, cugraph_centrality_result_t, cugraph_error_t, cugraph_pagerank};

use super::CentralityResult;
use crate::error::{check_error, Result};
use crate::graph::Graph;
use crate::handle::ResourceHandle;
use crate::params::{IterationCount, PositiveF64, UnitF64};

/// Parameters for the PageRank algorithm.
///
/// All fields use strongly-typed wrappers that enforce invariants at
/// construction time:
///
/// - `alpha` ([`UnitF64`]): damping factor, must be in `(0.0, 1.0)`
/// - `epsilon` ([`PositiveF64`]): convergence tolerance, must be `> 0.0`
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
/// use cugraph::params::{IterationCount, PositiveF64, UnitF64};
/// let params = PageRankParams {
///     alpha: UnitF64::new(0.9).unwrap(),
///     epsilon: PositiveF64::new(1e-6).unwrap(),
///     max_iterations: IterationCount::new(200).unwrap(),
/// };
/// ```
#[derive(Debug, Clone, Copy)]
pub struct PageRankParams {
    /// Damping factor (default 0.85). Must be in the open interval `(0.0, 1.0)`.
    pub alpha: UnitF64,
    /// Convergence tolerance (default 1e-5). Must be strictly positive.
    pub epsilon: PositiveF64,
    /// Maximum number of iterations (default 100). Must be at least 1.
    pub max_iterations: IterationCount,
}

impl Default for PageRankParams {
    fn default() -> Self {
        Self {
            // SAFETY: these literals are known-valid constants.
            alpha: UnitF64::new(0.85).unwrap(),
            epsilon: PositiveF64::new(1e-5).unwrap(),
            max_iterations: IterationCount::new(100).unwrap(),
        }
    }
}

/// Compute PageRank on a graph.
///
/// Returns a [`CentralityResult`] containing vertex ids and their PageRank
/// values. The result must be kept alive as long as any
/// [`crate::array::DeviceArrayView`] borrowed from it is in use.
pub fn pagerank(
    handle: &ResourceHandle,
    graph: &mut Graph,
    params: &PageRankParams,
) -> Result<CentralityResult> {
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

    Ok(unsafe { CentralityResult::from_raw(result) })
}

/// Type alias for backward compatibility.
pub type PageRankResult = CentralityResult;

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
            None,
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
