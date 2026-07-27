//! PageRank centrality algorithm.

use std::ptr;

use cugraph_sys::{cugraph_centrality_result_t, cugraph_error_t, cugraph_pagerank_allow_nonconvergence};

use super::PageRankResult;
use crate::{
    array::DeviceArrayView,
    error::{Error, ErrorKind, Result, check_error},
    graph::{Graph, Transposed},
    handle::ResourceHandle,
    params::{ClosedUnitF64, IterationCount, NonNegativeF64},
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
            // SAFETY: these literals are known-valid constants.
            alpha: ClosedUnitF64::new(0.85).unwrap(),
            epsilon: NonNegativeF64::new(1e-5).unwrap(),
            max_iterations: IterationCount::new(100).unwrap(),
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

/// Consolidated options for a PageRank computation.
///
/// Bundles [`PageRankParams`] with optional precomputed weights and an
/// initial guess. This is the single argument passed to
/// [`Graph::pagerank`](crate::graph::Graph::pagerank).
///
/// # Example
///
/// ```
/// use cugraph::centrality::pagerank::{PageRankOptions, PageRankParams};
///
/// // Simple: just params, no optional inputs
/// let options = PageRankOptions::<'_, i32, f32>::default();
/// assert_eq!(options.params.alpha.get(), 0.85);
/// ```
#[derive(Debug)]
pub struct PageRankOptions<'a, V: CugraphElement = i32, W: CugraphElement = f32> {
    /// Algorithm parameters (damping, tolerance, iterations, etc.).
    pub params: PageRankParams,
    /// Optional precomputed vertex out-weight sums.
    pub precomputed_weights: Option<PrecomputedWeights<'a, V, W>>,
    /// Optional initial guess of PageRank values.
    pub initial_guess: Option<InitialGuess<'a, V, W>>,
}

impl<'a, V: CugraphElement, W: CugraphElement> Default for PageRankOptions<'a, V, W> {
    fn default() -> Self {
        Self { params: PageRankParams::default(), precomputed_weights: None, initial_guess: None }
    }
}

/// Compute PageRank on a graph.
///
/// Prefer using [`Graph::pagerank`](crate::graph::Graph::pagerank) instead
/// of calling this function directly.
pub(crate) fn pagerank<V: CugraphElement, W: CugraphElement>(
    handle: &ResourceHandle,
    graph: &mut Graph<Transposed>,
    options: &PageRankOptions<'_, V, W>,
) -> Result<PageRankResult> {
    let params = &options.params;

    // Validate precomputed weights.
    if let Some(ref pw) = options.precomputed_weights {
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
    if let Some(ref ig) = options.initial_guess {
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
    let (pw_verts_ptr, pw_sums_ptr) = match options.precomputed_weights {
        Some(ref pw) => (pw.vertices.as_ptr() as *const _, pw.sums.as_ptr() as *const _),
        None => (ptr::null(), ptr::null()),
    };

    let (ig_verts_ptr, ig_vals_ptr) = match options.initial_guess {
        Some(ref ig) => (ig.vertices.as_ptr() as *const _, ig.values.as_ptr() as *const _),
        None => (ptr::null(), ptr::null()),
    };

    let mut result: *mut cugraph_centrality_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();

    unsafe {
        let code = cugraph_pagerank_allow_nonconvergence(
            handle.as_ptr(),
            graph.raw,
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

    Ok(unsafe { PageRankResult::from_raw(result) })
}

/// Convenience wrapper: run PageRank with just [`PageRankParams`] (no
/// optional inputs).
pub(crate) fn pagerank_simple(
    handle: &ResourceHandle,
    graph: &mut Graph<Transposed>,
    params: &PageRankParams,
) -> Result<PageRankResult> {
    let options = PageRankOptions::<'_, i32, f32> { params: *params, ..Default::default() };
    pagerank(handle, graph, &options)
}

fn validate_vertex_dtype<V: CugraphElement>(graph: &Graph<Transposed>, input_name: &str) -> Result<()> {
    let dtype = V::data_type();
    if !dtype.is_integer() {
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

        // Star graph: 1->0, 2->0, 3->0 (node 0 is the hub)
        let src_data: Vec<i32> = vec![1, 2, 3, 0, 0, 0];
        let dst_data: Vec<i32> = vec![0, 0, 0, 1, 2, 3];

        let src = DeviceArray::from_slice(&handle, &src_data).expect("src");
        let dst = DeviceArray::from_slice(&handle, &dst_data).expect("dst");

        let src_view = src.view();
        let dst_view = dst.view();
        let mut graph = GraphBuilder::new(&src_view, &dst_view).build::<Transposed>(&handle).expect("graph");

        let options = PageRankOptions::<'_, i32, f32>::default();
        let result = graph.pagerank(&handle, &options).expect("pagerank");

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

        let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).expect("src");
        let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 0]).expect("dst");
        let mut graph = GraphBuilder::new(&src.view(), &dst.view()).build::<Transposed>(&handle).expect("graph");

        let params = PageRankParams::default();
        let result = graph.pagerank_simple(&handle, &params).expect("pagerank_simple");

        assert_eq!(result.vertex_ids().expect("vertex_ids").len(), 3);
        assert_eq!(result.scores().expect("scores").len(), 3);
    }

    #[test]
    fn test_pagerank_rejects_mismatched_precomputed_weights() {
        let handle = ResourceHandle::new().expect("handle");

        let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).expect("src");
        let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 0]).expect("dst");
        let mut graph = GraphBuilder::new(&src.view(), &dst.view()).build::<Transposed>(&handle).expect("graph");

        let verts = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).expect("verts");
        let sums = DeviceArray::from_slice(&handle, &[1.0f32, 2.0]).expect("sums"); // wrong length

        let options = PageRankOptions {
            precomputed_weights: Some(PrecomputedWeights { vertices: &verts.view(), sums: &sums.view() }),
            ..Default::default()
        };

        let err = graph.pagerank(&handle, &options).expect_err("should reject mismatched lengths");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.message().contains("precomputed weights"));
    }

    #[test]
    fn test_pagerank_rejects_mismatched_initial_guess() {
        let handle = ResourceHandle::new().expect("handle");

        let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).expect("src");
        let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 0]).expect("dst");
        let mut graph = GraphBuilder::new(&src.view(), &dst.view()).build::<Transposed>(&handle).expect("graph");

        let verts = DeviceArray::from_slice(&handle, &[0i32, 1]).expect("verts");
        let vals = DeviceArray::from_slice(&handle, &[0.5f32]).expect("vals"); // wrong length

        let options = PageRankOptions {
            initial_guess: Some(InitialGuess { vertices: &verts.view(), values: &vals.view() }),
            ..Default::default()
        };

        let err = graph.pagerank(&handle, &options).expect_err("should reject mismatched lengths");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.message().contains("initial guess"));
    }
}
