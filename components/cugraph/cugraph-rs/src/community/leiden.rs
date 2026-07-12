//! Leiden community detection.

use std::ptr;

use cugraph_sys::{cugraph_error_t, cugraph_hierarchical_clustering_result_t, cugraph_leiden};

use super::LeidenResult;
use crate::{
    error::{Result, check_error, require_non_null},
    graph::{Forward, Graph},
    handle::EnteredContext,
    params::{IterationCount, PositiveF64, defaults},
    random::RngState,
    types::to_bool_t,
};

/// Parameters for the Leiden community-detection algorithm.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct LeidenParams {
    /// Maximum hierarchy depth / iteration count (default: `100`).
    pub max_level: IterationCount,
    /// Resolution parameter (`gamma`) in the modularity formula (default: `1.0`).
    pub resolution: PositiveF64,
    /// Modularity-gain scaling parameter for the Leiden refinement phase
    /// (default: `1.0`).
    pub theta: PositiveF64,
    /// Whether to run expensive input validation checks in the C library
    /// (default: `false`).
    pub do_expensive_check: bool,
}

impl Default for LeidenParams {
    fn default() -> Self {
        Self {
            max_level: defaults::LOUVAIN_MAX_LEVEL,
            resolution: defaults::LOUVAIN_RESOLUTION,
            theta: PositiveF64::new(1.0).expect("literal is positive"),
            do_expensive_check: false,
        }
    }
}

/// Compute Leiden communities on a forward-stored graph.
pub(crate) fn leiden(
    handle: &EnteredContext<'_>,
    rng_state: &mut RngState,
    graph: &mut Graph<Forward>,
    params: &LeidenParams,
) -> Result<LeidenResult> {
    let mut result: *mut cugraph_hierarchical_clustering_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();
    let graph_ptr = graph.as_ptr(handle, "leiden")?;

    // SAFETY: `graph` is a live cuGraph graph object matching Leiden's
    // forward-layout requirement, `rng_state` is a live cuGraph RNG state, and
    // `result`/`error` are valid out-pointers.
    unsafe {
        let code = cugraph_leiden(
            handle.handle_ptr(),
            rng_state.as_ptr(),
            graph_ptr,
            usize::from(params.max_level),
            params.resolution.get(),
            params.theta.get(),
            to_bool_t(params.do_expensive_check),
            &mut result,
            &mut error,
        );
        check_error(code, error, "leiden")?;
    }

    let result = require_non_null(result, "leiden", "hierarchical_clustering_result")?;

    // SAFETY: on success, cuGraph transfers ownership of the result pointer.
    unsafe { LeidenResult::from_raw(result, handle.device(), handle.runtime_bound_lease()) }
}
