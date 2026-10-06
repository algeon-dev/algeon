//! ECG community detection.

use std::ptr;

use cugraph_sys::{cugraph_ecg, cugraph_error_t, cugraph_hierarchical_clustering_result_t};

use super::EcgResult;
use crate::{
    error::{Result, check_error, require_non_null},
    graph::{Forward, Graph},
    handle::EnteredContext,
    params::{FractionF64, IterationCount, NonNegativeF64, defaults},
    random::RngState,
    types::to_bool_t,
};

/// Parameters for ECG community detection.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct EcgParams {
    /// Minimum edge weight in the final graph (default: `0.001`).
    pub min_weight: NonNegativeF64,
    /// Number of Louvain ensemble iterations to run (default: `10`).
    pub ensemble_size: IterationCount,
    /// Maximum hierarchy depth for final Louvain (default: `100`).
    pub max_level: IterationCount,
    /// Convergence threshold for final Louvain, in `(0.0, 1.0]` (default: `1e-7`).
    pub threshold: FractionF64,
    /// Resolution parameter (`gamma`) in the modularity formula, in `(0.0, 1.0]`
    /// (default: `1.0`).
    pub resolution: FractionF64,
    /// Whether to run expensive input validation checks in the C library
    /// (default: `false`).
    pub do_expensive_check: bool,
}

impl Default for EcgParams {
    fn default() -> Self {
        Self {
            min_weight: NonNegativeF64::new(0.001).expect("literal is non-negative"),
            ensemble_size: IterationCount::new(10).expect("literal is non-zero"),
            max_level: defaults::LOUVAIN_MAX_LEVEL,
            threshold: defaults::ECG_THRESHOLD,
            resolution: defaults::ECG_RESOLUTION,
            do_expensive_check: false,
        }
    }
}

/// Compute ECG communities on a forward-stored graph.
pub(crate) fn ecg(
    handle: &EnteredContext<'_>,
    rng_state: &mut RngState,
    graph: &mut Graph<Forward>,
    params: &EcgParams,
) -> Result<EcgResult> {
    let mut result: *mut cugraph_hierarchical_clustering_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();
    let graph_ptr = graph.as_ptr(handle, "ecg")?;

    // SAFETY: `graph` is a live cuGraph graph object matching ECG's
    // forward-layout requirement, `rng_state` is a live cuGraph RNG state, and
    // `result`/`error` are valid out-pointers.
    unsafe {
        let code = cugraph_ecg(
            handle.handle_ptr(),
            rng_state.as_ptr(),
            graph_ptr,
            params.min_weight.get(),
            usize::from(params.ensemble_size),
            usize::from(params.max_level),
            params.threshold.get(),
            params.resolution.get(),
            to_bool_t(params.do_expensive_check),
            &mut result,
            &mut error,
        );
        check_error(code, error, handle.device(), "ecg")?;
    }

    let result = require_non_null(result, "ecg", "hierarchical_clustering_result")?;

    // SAFETY: on success, cuGraph transfers ownership of the result pointer.
    unsafe { EcgResult::from_raw(result, handle.device(), handle.runtime_bound_lease()) }
}
