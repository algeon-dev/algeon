//! Edge betweenness centrality algorithm.

use std::ptr;

use cugraph_sys::{cugraph_edge_betweenness_centrality, cugraph_edge_centrality_result_t, cugraph_error_t};

use super::EdgeBetweennessCentralityResult;
use crate::{
    array::DeviceArrayView,
    error::{Result, check_error, require_non_null},
    graph::{Graph, Transposition},
    handle::EnteredContext,
    types::{CugraphElement, to_bool_t},
};

/// Parameters for edge betweenness centrality.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct EdgeBetweennessCentralityParams {
    /// Whether to normalize the result. When `true`, values are scaled so that
    /// the maximum possible edge betweenness is 1.0.
    pub normalized: bool,
    /// Whether to run expensive input validation checks in the C library.
    pub do_expensive_check: bool,
}

impl Default for EdgeBetweennessCentralityParams {
    fn default() -> Self {
        Self { normalized: true, do_expensive_check: false }
    }
}

pub(crate) fn edge_betweenness_centrality<T: Transposition, V: CugraphElement>(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<T>,
    vertex_list: Option<&DeviceArrayView<'_, V>>,
    params: &EdgeBetweennessCentralityParams,
) -> Result<EdgeBetweennessCentralityResult> {
    let mut result: *mut cugraph_edge_centrality_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();

    let vertex_list_ptr = vertex_list.map(|v| v.as_ptr() as *const _).unwrap_or(ptr::null());
    let do_expensive_check = params.do_expensive_check || vertex_list.is_some();
    let graph_ptr = graph.as_ptr(handle, "edge_betweenness_centrality")?;

    // SAFETY: `vertex_list_ptr` is either null or borrowed from a live device
    // view validated against the graph vertex dtype. Non-null vertex lists
    // force libcugraph validation so out-of-range vertices cannot drive
    // unchecked native indexing. `result`/`error` are valid out-pointers.
    unsafe {
        let code = cugraph_edge_betweenness_centrality(
            handle.handle_ptr(),
            graph_ptr,
            vertex_list_ptr,
            to_bool_t(params.normalized),
            to_bool_t(do_expensive_check),
            &mut result,
            &mut error,
        );
        check_error(code, error, handle.device(), "edge_betweenness_centrality")?;
    }

    let result = require_non_null(result, "edge_betweenness_centrality", "edge_centrality_result")?;
    // SAFETY: on success, cuGraph returns ownership of `result` to the caller.
    unsafe {
        EdgeBetweennessCentralityResult::from_raw(
            result,
            handle.device(),
            graph.has_edge_ids(),
            handle.runtime_bound_lease(),
        )
    }
}
