//! Connected-components algorithms.

use std::ptr;

use cugraph_sys::{
    cugraph_error_t, cugraph_labeling_result_t, cugraph_strongly_connected_components,
    cugraph_weakly_connected_components,
};

use super::{StronglyConnectedComponentsResult, WeaklyConnectedComponentsResult};
use crate::{
    error::{Result, check_error, require_non_null},
    graph::Graph,
    handle::EnteredContext,
    types::to_bool_t,
};

/// Parameters shared by weakly and strongly connected components.
#[derive(Debug, Clone, Copy, Default)]
#[must_use]
pub struct ConnectedComponentsParams {
    /// Whether to run expensive input validation in the C library.
    pub do_expensive_check: bool,
}

/// Compute weakly connected components.
///
/// The native cuGraph API requires a symmetric graph. This wrapper does not
/// add a Rust-side graph-property hard failure; native validation reports
/// detailed graph-property errors when checks are enabled.
pub(crate) fn weakly_connected_components<T>(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<T>,
    params: &ConnectedComponentsParams,
) -> Result<WeaklyConnectedComponentsResult> {
    let mut result: *mut cugraph_labeling_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();
    let graph_ptr = graph.as_ptr(handle, "weakly_connected_components")?;

    // SAFETY: `graph` and `handle` are live for the duration of the call, and
    // `result`/`error` are valid out-pointers.
    unsafe {
        let code = cugraph_weakly_connected_components(
            handle.handle_ptr(),
            graph_ptr,
            to_bool_t(params.do_expensive_check),
            &mut result,
            &mut error,
        );
        check_error(code, error, "weakly_connected_components")?;
    }

    let result = require_non_null(result, "weakly_connected_components", "labeling_result")?;

    // SAFETY: on success, cuGraph transfers ownership of the result pointer.
    unsafe { WeaklyConnectedComponentsResult::from_raw(result, handle.device()) }
}

/// Compute strongly connected components.
pub(crate) fn strongly_connected_components<T>(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<T>,
    params: &ConnectedComponentsParams,
) -> Result<StronglyConnectedComponentsResult> {
    let mut result: *mut cugraph_labeling_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();
    let graph_ptr = graph.as_ptr(handle, "strongly_connected_components")?;

    // SAFETY: `graph` and `handle` are live for the duration of the call, and
    // `result`/`error` are valid out-pointers.
    unsafe {
        let code = cugraph_strongly_connected_components(
            handle.handle_ptr(),
            graph_ptr,
            to_bool_t(params.do_expensive_check),
            &mut result,
            &mut error,
        );
        check_error(code, error, "strongly_connected_components")?;
    }

    let result = require_non_null(result, "strongly_connected_components", "labeling_result")?;

    // SAFETY: on success, cuGraph transfers ownership of the result pointer.
    unsafe { StronglyConnectedComponentsResult::from_raw(result, handle.device()) }
}
