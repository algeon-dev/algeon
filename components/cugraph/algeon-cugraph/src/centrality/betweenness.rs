//! Vertex betweenness centrality algorithm.
//!
//! Betweenness centrality measures how often a vertex appears on shortest paths
//! between all pairs of vertices in the graph. It can be computed exactly
//! (traversal from every source) or approximately (using a subset of seed
//! vertices).

use std::ptr;

use cugraph_sys::{cugraph_betweenness_centrality, cugraph_centrality_result_t, cugraph_error_t};

use super::BetweennessCentralityResult;
use crate::{
    array::DeviceArrayView,
    error::{Result, check_error, require_non_null},
    graph::Graph,
    handle::EnteredContext,
    types::{CugraphElement, to_bool_t},
};

/// Parameters for the betweenness centrality algorithm.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct BetweennessCentralityParams {
    /// Whether to normalize the result. When `true`, values are scaled so that
    /// the maximum possible betweenness is 1.0 (default: `true`).
    pub normalized: bool,
    /// Whether to include endpoints when considering a vertex to be on a
    /// shortest path. The traditional formulation does not include endpoints
    /// (default: `false`).
    pub include_endpoints: bool,
    /// Whether to run expensive input validation checks in the C library
    /// (default: `false`).
    pub do_expensive_check: bool,
}

impl Default for BetweennessCentralityParams {
    fn default() -> Self {
        Self { normalized: true, include_endpoints: false, do_expensive_check: false }
    }
}

/// Compute vertex betweenness centrality on a graph.
///
/// Prefer using [`Graph::betweenness_centrality`](crate::graph::Graph::betweenness_centrality)
/// instead of calling this function directly.
pub(crate) fn betweenness_centrality<T, V: CugraphElement>(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<T>,
    vertex_list: Option<&DeviceArrayView<'_, V>>,
    params: &BetweennessCentralityParams,
) -> Result<BetweennessCentralityResult> {
    let mut result: *mut cugraph_centrality_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();

    let vertex_list_ptr = vertex_list.map(|v| v.as_ptr() as *const _).unwrap_or(ptr::null());
    let do_expensive_check = params.do_expensive_check || vertex_list.is_some();
    let graph_ptr = graph.as_ptr(handle, "betweenness_centrality")?;

    // SAFETY: `vertex_list_ptr` is either null or borrowed from a live device
    // view validated against the graph vertex dtype. Non-null vertex lists
    // force libcugraph validation so out-of-range vertices cannot drive
    // unchecked native indexing. `result`/`error` are valid out-pointers.
    unsafe {
        let code = cugraph_betweenness_centrality(
            handle.handle_ptr(),
            graph_ptr,
            vertex_list_ptr,
            to_bool_t(params.normalized),
            to_bool_t(params.include_endpoints),
            to_bool_t(do_expensive_check),
            &mut result,
            &mut error,
        );
        check_error(code, error, handle.device(), "betweenness_centrality")?;
    }

    let result = require_non_null(result, "betweenness_centrality", "centrality_result")?;

    // SAFETY: on success, cuGraph returns ownership of `result` to the caller.
    unsafe { BetweennessCentralityResult::from_raw(result, handle.device(), handle.runtime_bound_lease()) }
}
