//! PageRank centrality algorithm.

use std::ptr;

use cugraph_sys::{cugraph_centrality_result_t, cugraph_error_t, cugraph_pagerank_allow_nonconvergence};

use super::PageRankResult;
use crate::{
    array::DeviceArrayView,
    error::{Error, ErrorKind, Result, check_error, require_non_null},
    graph::{Graph, Transposed},
    handle::EnteredContext,
    params::{ClosedUnitF64, IterationCount, NonNegativeF64, defaults},
    types::{CugraphElement, to_bool_t},
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
///     do_expensive_check: false,
/// };
/// ```
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct PageRankParams {
    /// Damping factor (default 0.85). Must be in the closed interval `[0.0, 1.0]`.
    pub alpha: ClosedUnitF64,
    /// Convergence tolerance (default 1e-5). Must be non-negative. Zero
    /// disables early convergence (always runs for `max_iterations`).
    pub epsilon: NonNegativeF64,
    /// Maximum number of iterations (default 100). Must be at least 1.
    pub max_iterations: IterationCount,
    /// Whether to run expensive input validation checks in the C library
    /// (default: `false`).
    pub do_expensive_check: bool,
}

impl Default for PageRankParams {
    fn default() -> Self {
        Self {
            alpha: defaults::PAGERANK_ALPHA,
            epsilon: defaults::PAGERANK_EPSILON,
            max_iterations: defaults::PAGERANK_MAX_ITERATIONS,
            do_expensive_check: false,
        }
    }
}

/// Precomputed vertex out-weight sums for PageRank.
///
/// Providing these avoids recomputation inside the C library and can
/// improve performance on repeated runs with the same graph.
#[derive(Debug)]
pub struct PrecomputedWeights<'a, V: CugraphElement = i32, W: CugraphElement = f32> {
    /// Vertex ids for which out-weight sums are provided.
    pub vertices: &'a DeviceArrayView<'a, V>,
    /// Precomputed sum of out-edge weights for each vertex.
    pub sums: &'a DeviceArrayView<'a, W>,
}

/// Initial guess of PageRank values.
///
/// When provided, the solver starts from these values instead of the
/// uniform `1/N` default. Useful for iterative refinement.
#[derive(Debug)]
pub struct InitialGuess<'a, V: CugraphElement = i32, W: CugraphElement = f32> {
    /// Vertex ids for the initial guess.
    pub vertices: &'a DeviceArrayView<'a, V>,
    /// Initial PageRank values for each vertex.
    pub values: &'a DeviceArrayView<'a, W>,
}

/// Compute PageRank on a graph.
///
/// Prefer using [`Graph::pagerank`](crate::graph::Graph::pagerank) or
/// [`Graph::pagerank_with`](crate::graph::Graph::pagerank_with) instead
/// of calling this function directly.
pub(crate) fn pagerank<V: CugraphElement, W: CugraphElement>(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<Transposed>,
    params: &PageRankParams,
    precomputed_weights: Option<&PrecomputedWeights<'_, V, W>>,
    initial_guess: Option<&InitialGuess<'_, V, W>>,
) -> Result<PageRankResult> {
    // Validate precomputed weights.
    if let Some(pw) = precomputed_weights {
        if pw.vertices.len() != pw.sums.len() {
            return Err(Error::new(ErrorKind::InvalidInput, "pagerank")
                .with_message("precomputed weights: vertices and sums must have the same length")
                .with_context("vertices_len", pw.vertices.len().to_string())
                .with_context("sums_len", pw.sums.len().to_string()));
        }
        validate_vertex_dtype::<V>(graph, "precomputed weights vertices")?;
        validate_weight_dtype::<W>("precomputed weights sums")?;
    }

    // Validate initial guess.
    if let Some(ig) = initial_guess {
        if ig.vertices.len() != ig.values.len() {
            return Err(Error::new(ErrorKind::InvalidInput, "pagerank")
                .with_message("initial guess: vertices and values must have the same length")
                .with_context("vertices_len", ig.vertices.len().to_string())
                .with_context("values_len", ig.values.len().to_string()));
        }
        validate_vertex_dtype::<V>(graph, "initial guess vertices")?;
        validate_weight_dtype::<W>("initial guess values")?;
    }

    // Extract raw pointers for optional arrays.
    let (pw_verts_ptr, pw_sums_ptr) = match precomputed_weights {
        Some(pw) => (pw.vertices.as_ptr() as *const _, pw.sums.as_ptr() as *const _),
        None => (ptr::null(), ptr::null()),
    };

    let (ig_verts_ptr, ig_vals_ptr) = match initial_guess {
        Some(ig) => (ig.vertices.as_ptr() as *const _, ig.values.as_ptr() as *const _),
        None => (ptr::null(), ptr::null()),
    };

    let mut result: *mut cugraph_centrality_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();

    // SAFETY: all optional input pointers are either null or borrowed from
    // validated device views that outlive the call, and `result`/`error` are
    // valid out-pointers.
    unsafe {
        let code = cugraph_pagerank_allow_nonconvergence(
            handle.handle_ptr(),
            graph.as_ptr(),
            pw_verts_ptr,
            pw_sums_ptr,
            ig_verts_ptr,
            ig_vals_ptr,
            params.alpha.get(),
            params.epsilon.get(),
            usize::from(params.max_iterations),
            to_bool_t(params.do_expensive_check),
            &mut result,
            &mut error,
        );
        check_error(code, error, "pagerank")?;
    }

    let result = require_non_null(result, "pagerank", "centrality_result")?;

    // SAFETY: on success, cuGraph returns ownership of `result` to the caller.
    unsafe { PageRankResult::from_raw(result, handle.device()) }
}

fn validate_vertex_dtype<V: CugraphElement>(graph: &Graph<Transposed>, input_name: &str) -> Result<()> {
    let dtype = V::data_type();
    if !dtype.is_supported_index_type() {
        return Err(Error::new(ErrorKind::UnsupportedType, "pagerank")
            .with_message(format!("{input_name} require Int32 or Int64 vertex ids, got {dtype}")));
    }
    if dtype != graph.vertex_dtype() {
        return Err(Error::new(ErrorKind::UnsupportedType, "pagerank").with_message(format!(
            "{input_name} vertex dtype {dtype} does not match graph vertex dtype {}",
            graph.vertex_dtype()
        )));
    }
    Ok(())
}

fn validate_weight_dtype<W: CugraphElement>(input_name: &str) -> Result<()> {
    let dtype = W::data_type();
    if !dtype.is_float() {
        return Err(Error::new(ErrorKind::UnsupportedType, "pagerank")
            .with_message(format!("{input_name} require Float32 or Float64 values, got {dtype}")));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        array::DeviceArray,
        error::ErrorKind,
        graph::{GraphBuilder, Transposed},
        handle::ResourceHandle,
    };

    #[test]
    fn test_pagerank_star_graph() {
        let handle = ResourceHandle::new().expect("handle");
        let handle = handle.enter("test_pagerank_star_graph").expect("entered");

        // Star graph: 1->0, 2->0, 3->0 (node 0 is the hub)
        let src_data: Vec<i32> = vec![1, 2, 3, 0, 0, 0];
        let dst_data: Vec<i32> = vec![0, 0, 0, 1, 2, 3];

        let src = DeviceArray::from_slice(&handle, &src_data).expect("src");
        let dst = DeviceArray::from_slice(&handle, &dst_data).expect("dst");

        let src_view = src.view().expect("src_view");
        let dst_view = dst.view().expect("dst_view");
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

    #[test]
    fn test_pagerank_simple_convenience() {
        let handle = ResourceHandle::new().expect("handle");
        let handle = handle.enter("test_pagerank_simple_convenience").expect("entered");

        let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).expect("src");
        let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 0]).expect("dst");
        let src_view = src.view().expect("src_view");
        let dst_view = dst.view().expect("dst_view");
        let mut graph = GraphBuilder::new(&src_view, &dst_view).build::<Transposed>(&handle).expect("graph");

        let params = PageRankParams::default();
        let result = graph.pagerank(&handle, &params).expect("pagerank");

        assert_eq!(result.vertex_ids().expect("vertex_ids").len(), 3);
        assert_eq!(result.scores().expect("scores").len(), 3);
    }

    #[test]
    fn test_pagerank_rejects_mismatched_precomputed_weights() {
        let handle = ResourceHandle::new().expect("handle");
        let handle = handle.enter("test_pagerank_rejects_mismatched_precomputed_weights").expect("entered");

        let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).expect("src");
        let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 0]).expect("dst");
        let src_view = src.view().expect("src_view");
        let dst_view = dst.view().expect("dst_view");
        let mut graph = GraphBuilder::new(&src_view, &dst_view).build::<Transposed>(&handle).expect("graph");

        let verts = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).expect("verts");
        let sums = DeviceArray::from_slice(&handle, &[1.0f32, 2.0]).expect("sums"); // wrong length
        let verts_view = verts.view().expect("verts_view");
        let sums_view = sums.view().expect("sums_view");

        let pw = PrecomputedWeights { vertices: &verts_view, sums: &sums_view };
        let params = PageRankParams::default();

        let err = graph.pagerank_with(&handle, &params, Some(&pw), None).expect_err("should reject mismatched lengths");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.message().contains("precomputed weights"));
    }

    #[test]
    fn test_pagerank_rejects_mismatched_initial_guess() {
        let handle = ResourceHandle::new().expect("handle");
        let handle = handle.enter("test_pagerank_rejects_mismatched_initial_guess").expect("entered");

        let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).expect("src");
        let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 0]).expect("dst");
        let src_view = src.view().expect("src_view");
        let dst_view = dst.view().expect("dst_view");
        let mut graph = GraphBuilder::new(&src_view, &dst_view).build::<Transposed>(&handle).expect("graph");

        let verts = DeviceArray::from_slice(&handle, &[0i32, 1]).expect("verts");
        let vals = DeviceArray::from_slice(&handle, &[0.5f32]).expect("vals"); // wrong length
        let verts_view = verts.view().expect("verts_view");
        let vals_view = vals.view().expect("vals_view");

        let ig = InitialGuess { vertices: &verts_view, values: &vals_view };
        let params = PageRankParams::default();

        let err = graph.pagerank_with(&handle, &params, None, Some(&ig)).expect_err("should reject mismatched lengths");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.message().contains("initial guess"));
    }
}
