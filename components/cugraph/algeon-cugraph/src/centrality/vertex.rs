//! Vertex centrality algorithms beyond PageRank and betweenness.

use std::ptr;

use cugraph_sys::{
    cugraph_centrality_result_t, cugraph_eigenvector_centrality_allow_nonconvergence, cugraph_error_t,
    cugraph_katz_centrality_allow_nonconvergence,
};

use super::{CentralityResult, non_convergence_error};
use crate::{
    error::{Result, check_error, require_non_null},
    graph::{Graph, Transposed},
    handle::EnteredContext,
    params::{IterationCount, NonNegativeF64},
    types::to_bool_t,
};

/// Parameters for eigenvector centrality.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct EigenvectorCentralityParams {
    /// Convergence tolerance. Must be non-negative.
    pub epsilon: NonNegativeF64,
    /// Maximum number of power iterations. Must be at least 1.
    pub max_iterations: IterationCount,
    /// Whether to run expensive input validation checks in the C library.
    pub do_expensive_check: bool,
}

impl Default for EigenvectorCentralityParams {
    fn default() -> Self {
        Self {
            epsilon: NonNegativeF64::new(1e-6).expect("literal is non-negative"),
            max_iterations: IterationCount::new(200).expect("literal is non-zero"),
            do_expensive_check: false,
        }
    }
}

/// Parameters for Katz centrality.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct KatzCentralityParams {
    /// Katz attenuation factor. cuGraph requires this to be smaller than the
    /// inverse of the graph's maximum eigenvalue.
    pub alpha: NonNegativeF64,
    /// Constant added to each vertex score in every iteration.
    pub beta: NonNegativeF64,
    /// Convergence tolerance. Must be non-negative.
    pub epsilon: NonNegativeF64,
    /// Maximum number of iterations. Must be at least 1.
    pub max_iterations: IterationCount,
    /// Whether to run expensive input validation checks in the C library.
    pub do_expensive_check: bool,
}

impl Default for KatzCentralityParams {
    fn default() -> Self {
        Self {
            alpha: NonNegativeF64::new(0.01).expect("literal is non-negative"),
            beta: NonNegativeF64::new(1.0).expect("literal is non-negative"),
            epsilon: NonNegativeF64::new(1e-6).expect("literal is non-negative"),
            max_iterations: IterationCount::new(1000).expect("literal is non-zero"),
            do_expensive_check: false,
        }
    }
}

pub(crate) fn eigenvector_centrality(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<Transposed>,
    params: &EigenvectorCentralityParams,
) -> Result<CentralityResult> {
    let mut result: *mut cugraph_centrality_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();
    let graph_ptr = graph.as_ptr(handle, "eigenvector_centrality")?;

    // SAFETY: `graph` is a live cuGraph handle and `result`/`error` are valid
    // out-pointers.
    unsafe {
        let code = cugraph_eigenvector_centrality_allow_nonconvergence(
            handle.handle_ptr(),
            graph_ptr,
            params.epsilon.get(),
            usize::from(params.max_iterations),
            to_bool_t(params.do_expensive_check),
            &mut result,
            &mut error,
        );
        check_error(code, error, handle.device(), "eigenvector_centrality")?;
    }

    let result = require_non_null(result, "eigenvector_centrality", "centrality_result")?;
    // SAFETY: on success, cuGraph returns ownership of `result` to the caller.
    let result = unsafe { CentralityResult::from_raw(result, handle.device(), handle.runtime_bound_lease())? };
    if !result.converged()? {
        return Err(non_convergence_error(
            "eigenvector_centrality",
            result.num_iterations(),
            params.max_iterations,
            params.epsilon,
            "raise max_iterations or epsilon".to_string(),
        ));
    }
    Ok(result)
}

pub(crate) fn katz_centrality(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<Transposed>,
    params: &KatzCentralityParams,
) -> Result<CentralityResult> {
    let mut result: *mut cugraph_centrality_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();
    let graph_ptr = graph.as_ptr(handle, "katz_centrality")?;

    // SAFETY: this safe wrapper intentionally exposes the scalar-beta path, so
    // the optional beta vector pointer is null. `result`/`error` are valid
    // out-pointers.
    unsafe {
        let code = cugraph_katz_centrality_allow_nonconvergence(
            handle.handle_ptr(),
            graph_ptr,
            ptr::null(),
            params.alpha.get(),
            params.beta.get(),
            params.epsilon.get(),
            usize::from(params.max_iterations),
            to_bool_t(params.do_expensive_check),
            &mut result,
            &mut error,
        );
        check_error(code, error, handle.device(), "katz_centrality")?;
    }

    let result = require_non_null(result, "katz_centrality", "centrality_result")?;
    // SAFETY: on success, cuGraph returns ownership of `result` to the caller.
    let result = unsafe { CentralityResult::from_raw(result, handle.device(), handle.runtime_bound_lease())? };
    if !result.converged()? {
        // The Katz series converges only for alpha below 1/lambda_max, the reciprocal of the
        // adjacency matrix's largest eigenvalue; above it the iteration diverges at any
        // max_iterations.
        return Err(non_convergence_error(
            "katz_centrality",
            result.num_iterations(),
            params.max_iterations,
            params.epsilon,
            format!(
                "lower alpha (currently {}) below 1/lambda_max, the reciprocal of the largest eigenvalue of the \
                 adjacency matrix, or raise max_iterations",
                params.alpha.get()
            ),
        ));
    }
    Ok(result)
}
