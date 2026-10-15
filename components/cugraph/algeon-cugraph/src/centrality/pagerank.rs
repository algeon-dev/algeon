//! PageRank centrality algorithm.

use std::ptr;

use cugraph_sys::{
    cugraph_centrality_result_t, cugraph_error_t, cugraph_pagerank_allow_nonconvergence,
    cugraph_personalized_pagerank_allow_nonconvergence,
};

use super::CentralityResult;
use crate::{
    array::DeviceArrayView,
    cuda::{DeviceResourceKind, drop_device_resource},
    error::{Error, ErrorKind, Result, check_error, require_non_null},
    graph::{Graph, Transposed},
    handle::EnteredContext,
    params::{ClosedUnitF64, IterationCount, NonNegativeF64, defaults},
    types::{CugraphElement, to_bool_t},
};

/// Parameters for the PageRank algorithm.
///
/// Baseline configuration is provided by [`Default`].
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

/// Parameters for personalized PageRank.
///
/// These have the same invariants and defaults as [`PageRankParams`], but are
/// kept as a distinct public type because personalized PageRank requires an
/// explicit personalization vector.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct PersonalizedPageRankParams {
    /// Damping factor (default 0.85). Must be in the closed interval `[0.0, 1.0]`.
    pub alpha: ClosedUnitF64,
    /// Convergence tolerance (default 1e-5). Must be non-negative.
    pub epsilon: NonNegativeF64,
    /// Maximum number of iterations (default 100). Must be at least 1.
    pub max_iterations: IterationCount,
    /// Whether to run expensive input validation checks in the C library
    /// (default: `false`).
    pub do_expensive_check: bool,
}

impl Default for PersonalizedPageRankParams {
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
/// When provided, the solver starts from these values. Without an initial
/// guess, the solver starts from the uniform `1/N` default. Useful for
/// iterative refinement.
#[derive(Debug)]
pub struct InitialGuess<'a, V: CugraphElement = i32, W: CugraphElement = f32> {
    /// Vertex ids for the initial guess.
    pub vertices: &'a DeviceArrayView<'a, V>,
    /// Initial PageRank values for each vertex.
    pub values: &'a DeviceArrayView<'a, W>,
}

/// Personalization vector for personalized PageRank.
#[derive(Debug)]
pub struct PersonalizationVector<'a, V: CugraphElement = i32, W: CugraphElement = f32> {
    /// Vertex ids in the personalization set.
    pub vertices: &'a DeviceArrayView<'a, V>,
    /// Personalization values for each vertex.
    pub values: &'a DeviceArrayView<'a, W>,
}

#[derive(Debug, Clone, Copy)]
enum PageRankCall {
    Standard,
    Personalized,
}

/// Compute PageRank on a graph.
///
/// Callers use [`Graph::pagerank`](crate::graph::Graph::pagerank) or
/// [`Graph::pagerank_with`](crate::graph::Graph::pagerank_with), which call
/// this function.
pub(crate) fn pagerank<V: CugraphElement, W: CugraphElement>(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<Transposed>,
    params: &PageRankParams,
    precomputed_weights: Option<&PrecomputedWeights<'_, V, W>>,
    initial_guess: Option<&InitialGuess<'_, V, W>>,
) -> Result<CentralityResult> {
    pagerank_impl(
        handle,
        graph,
        params.alpha,
        params.epsilon,
        params.max_iterations,
        params.do_expensive_check,
        precomputed_weights,
        initial_guess,
        None,
        PageRankCall::Standard,
        "pagerank",
    )
}

pub(crate) fn personalized_pagerank<V: CugraphElement, W: CugraphElement>(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<Transposed>,
    params: &PersonalizedPageRankParams,
    personalization: &PersonalizationVector<'_, V, W>,
    precomputed_weights: Option<&PrecomputedWeights<'_, V, W>>,
    initial_guess: Option<&InitialGuess<'_, V, W>>,
) -> Result<CentralityResult> {
    pagerank_impl(
        handle,
        graph,
        params.alpha,
        params.epsilon,
        params.max_iterations,
        params.do_expensive_check,
        precomputed_weights,
        initial_guess,
        Some(personalization),
        PageRankCall::Personalized,
        "personalized_pagerank",
    )
}

#[expect(clippy::too_many_arguments, reason = "all parameters required to invoke PageRank FFI")]
fn pagerank_impl<V: CugraphElement, W: CugraphElement>(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<Transposed>,
    alpha: ClosedUnitF64,
    epsilon: NonNegativeF64,
    max_iterations: IterationCount,
    do_expensive_check: bool,
    precomputed_weights: Option<&PrecomputedWeights<'_, V, W>>,
    initial_guess: Option<&InitialGuess<'_, V, W>>,
    personalization: Option<&PersonalizationVector<'_, V, W>>,
    call: PageRankCall,
    operation: &'static str,
) -> Result<CentralityResult> {
    if let Some(pw) = precomputed_weights {
        if pw.vertices.len() != pw.sums.len() {
            return Err(Error::new(ErrorKind::InvalidInput, operation)
                .with_message("precomputed weights: vertices and sums must have the same length")
                .with_context("vertices_len", pw.vertices.len().to_string())
                .with_context("sums_len", pw.sums.len().to_string()));
        }
        validate_vertex_dtype::<V>(graph, operation, "precomputed weights vertices")?;
        validate_weight_dtype::<W>(operation, "precomputed weights sums")?;
    }

    if let Some(ig) = initial_guess {
        if ig.vertices.len() != ig.values.len() {
            return Err(Error::new(ErrorKind::InvalidInput, operation)
                .with_message("initial guess: vertices and values must have the same length")
                .with_context("vertices_len", ig.vertices.len().to_string())
                .with_context("values_len", ig.values.len().to_string()));
        }
        validate_vertex_dtype::<V>(graph, operation, "initial guess vertices")?;
        validate_weight_dtype::<W>(operation, "initial guess values")?;
    }

    if let Some(personalization) = personalization {
        if personalization.vertices.len() != personalization.values.len() {
            return Err(Error::new(ErrorKind::InvalidInput, operation)
                .with_message("personalization: vertices and values must have the same length")
                .with_context("vertices_len", personalization.vertices.len().to_string())
                .with_context("values_len", personalization.values.len().to_string()));
        }
        validate_vertex_dtype::<V>(graph, operation, "personalization vertices")?;
        validate_weight_dtype::<W>(operation, "personalization values")?;
    }

    let (pw_verts_ptr, pw_sums_ptr) = match precomputed_weights {
        Some(pw) => (pw.vertices.as_ptr() as *const _, pw.sums.as_ptr() as *const _),
        None => (ptr::null(), ptr::null()),
    };

    let (ig_verts_ptr, ig_vals_ptr) = match initial_guess {
        Some(ig) => (ig.vertices.as_ptr() as *const _, ig.values.as_ptr() as *const _),
        None => (ptr::null(), ptr::null()),
    };

    let (personalization_verts_ptr, personalization_vals_ptr) = match personalization {
        Some(p) => (p.vertices.as_ptr() as *const _, p.values.as_ptr() as *const _),
        None => (ptr::null(), ptr::null()),
    };
    let do_expensive_check =
        do_expensive_check || precomputed_weights.is_some() || initial_guess.is_some() || personalization.is_some();

    let mut result: *mut cugraph_centrality_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();
    let graph_ptr = graph.as_ptr(handle, operation)?;

    // SAFETY: all optional input pointers are either null or borrowed from
    // validated device views that outlive the call. Non-null user vertex
    // arrays force libcugraph validation so out-of-range vertices cannot drive
    // unchecked native indexing. `result`/`error` are valid out-pointers.
    unsafe {
        let code = match call {
            PageRankCall::Standard => cugraph_pagerank_allow_nonconvergence(
                handle.handle_ptr(),
                graph_ptr,
                pw_verts_ptr,
                pw_sums_ptr,
                ig_verts_ptr,
                ig_vals_ptr,
                alpha.get(),
                epsilon.get(),
                usize::from(max_iterations),
                to_bool_t(do_expensive_check),
                &mut result,
                &mut error,
            ),
            PageRankCall::Personalized => cugraph_personalized_pagerank_allow_nonconvergence(
                handle.handle_ptr(),
                graph_ptr,
                pw_verts_ptr,
                pw_sums_ptr,
                ig_verts_ptr,
                ig_vals_ptr,
                personalization_verts_ptr,
                personalization_vals_ptr,
                alpha.get(),
                epsilon.get(),
                usize::from(max_iterations),
                to_bool_t(do_expensive_check),
                &mut result,
                &mut error,
            ),
        };
        if let Err(err) = check_error(code, error, handle.device(), operation) {
            if !result.is_null() {
                drop_device_resource(&mut result, handle.device(), operation, DeviceResourceKind::CentralityResult);
            }
            return Err(err);
        }
    }

    let result = require_non_null(result, operation, "centrality_result")?;

    // SAFETY: on success, cuGraph returns ownership of `result` to the caller.
    unsafe { CentralityResult::from_raw(result, handle.device(), handle.runtime_bound_lease()) }
}

fn validate_vertex_dtype<V: CugraphElement>(
    graph: &Graph<Transposed>,
    operation: &'static str,
    input_name: &str,
) -> Result<()> {
    let dtype = V::data_type();
    if !dtype.is_supported_index_type() {
        return Err(Error::new(ErrorKind::UnsupportedType, operation)
            .with_message(format!("{input_name} require Int32 or Int64 vertex ids, got {dtype}")));
    }
    if dtype != graph.vertex_dtype() {
        return Err(Error::new(ErrorKind::UnsupportedType, operation).with_message(format!(
            "{input_name} vertex dtype {dtype} does not match graph vertex dtype {}",
            graph.vertex_dtype()
        )));
    }
    Ok(())
}

fn validate_weight_dtype<W: CugraphElement>(operation: &'static str, input_name: &str) -> Result<()> {
    let dtype = W::data_type();
    if !dtype.is_float() {
        return Err(Error::new(ErrorKind::UnsupportedType, operation)
            .with_message(format!("{input_name} require Float32 or Float64 values, got {dtype}")));
    }
    Ok(())
}
