//! Louvain community detection.

use std::ptr;

use cugraph_sys::{cugraph_error_t, cugraph_hierarchical_clustering_result_t, cugraph_louvain};

use super::LouvainResult;
use crate::{
    error::{Result, check_error, require_non_null},
    graph::{Forward, Graph},
    handle::EnteredContext,
    params::{IterationCount, NonNegativeF64, PositiveF64, defaults},
    types::to_bool_t,
};

/// Parameters for the Louvain community-detection algorithm.
///
/// All numeric fields use strongly-typed wrappers that enforce invariants at
/// construction time:
///
/// - `max_level` ([`IterationCount`]): must be at least 1
/// - `threshold` ([`NonNegativeF64`]): must be finite and `>= 0.0`
/// - `resolution` ([`PositiveF64`]): must be finite and `> 0.0`
///
/// # Example
///
/// ```
/// use cugraph::{
///     community::LouvainParams,
///     params::{IterationCount, NonNegativeF64, PositiveF64},
/// };
///
/// let params = LouvainParams {
///     max_level: IterationCount::new(50).unwrap(),
///     threshold: NonNegativeF64::new(1e-6).unwrap(),
///     resolution: PositiveF64::new(1.5).unwrap(),
///     do_expensive_check: false,
/// };
///
/// assert_eq!(params.max_level.get(), 50);
/// assert_eq!(params.resolution.get(), 1.5);
/// ```
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct LouvainParams {
    /// Maximum hierarchy depth / iteration count (default: `100`).
    pub max_level: IterationCount,
    /// Convergence threshold for modularity gain at each level (default: `1e-7`).
    pub threshold: NonNegativeF64,
    /// Resolution parameter (`gamma`) in the modularity formula (default: `1.0`).
    pub resolution: PositiveF64,
    /// Whether to run expensive input validation checks in the C library
    /// (default: `false`).
    pub do_expensive_check: bool,
}

impl Default for LouvainParams {
    fn default() -> Self {
        Self {
            max_level: defaults::LOUVAIN_MAX_LEVEL,
            threshold: defaults::LOUVAIN_THRESHOLD,
            resolution: defaults::LOUVAIN_RESOLUTION,
            do_expensive_check: false,
        }
    }
}

/// Compute Louvain communities on a forward-stored graph.
///
/// Prefer using [`Graph::louvain`](crate::graph::Graph::louvain), which calls
/// this function.
pub(crate) fn louvain(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<Forward>,
    params: &LouvainParams,
) -> Result<LouvainResult> {
    let mut result: *mut cugraph_hierarchical_clustering_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();
    let graph_ptr = graph.as_ptr(handle, "louvain")?;

    // SAFETY: `graph` is a live cuGraph graph object matching Louvain's
    // forward-layout requirement, and `result`/`error` are valid out-pointers.
    unsafe {
        let code = cugraph_louvain(
            handle.handle_ptr(),
            graph_ptr,
            usize::from(params.max_level),
            params.threshold.get(),
            params.resolution.get(),
            to_bool_t(params.do_expensive_check),
            &mut result,
            &mut error,
        );
        check_error(code, error, handle.device(), "louvain")?;
    }

    let result = require_non_null(result, "louvain", "hierarchical_clustering_result")?;

    // SAFETY: on success, cuGraph transfers ownership of the result pointer.
    unsafe { LouvainResult::from_raw(result, handle.device(), handle.runtime_bound_lease()) }
}
