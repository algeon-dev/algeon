//! HITS centrality algorithm.

use std::ptr;

use cugraph_sys::{cugraph_error_t, cugraph_hits, cugraph_hits_result_t};

use super::HitsResult;
use crate::{
    array::DeviceArrayView,
    error::{Error, ErrorKind, Result, check_error, require_non_null},
    graph::{Graph, Transposed},
    handle::EnteredContext,
    params::{IterationCount, NonNegativeF64},
    types::{CugraphElement, to_bool_t},
};

/// Parameters for HITS.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct HitsParams {
    /// Convergence tolerance. Must be non-negative.
    pub epsilon: NonNegativeF64,
    /// Maximum number of iterations. Must be at least 1.
    pub max_iterations: IterationCount,
    /// Whether to normalize the resulting hub and authority scores.
    pub normalize: bool,
    /// Whether to run expensive input validation checks in the C library.
    pub do_expensive_check: bool,
}

impl Default for HitsParams {
    fn default() -> Self {
        Self {
            epsilon: NonNegativeF64::new(1e-5).expect("literal is non-negative"),
            max_iterations: IterationCount::new(100).expect("literal is non-zero"),
            normalize: false,
            do_expensive_check: false,
        }
    }
}

/// Initial hub scores for HITS.
#[derive(Debug)]
pub struct HitsInitialHubs<'a, V: CugraphElement = i32, W: CugraphElement = f32> {
    /// Vertex ids for the initial hub scores.
    pub vertices: &'a DeviceArrayView<'a, V>,
    /// Initial hub score values.
    pub values: &'a DeviceArrayView<'a, W>,
}

pub(crate) fn hits<V: CugraphElement, W: CugraphElement>(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<Transposed>,
    params: &HitsParams,
    initial_hubs: Option<&HitsInitialHubs<'_, V, W>>,
) -> Result<HitsResult> {
    if let Some(initial_hubs) = initial_hubs {
        if initial_hubs.vertices.len() != initial_hubs.values.len() {
            return Err(Error::new(ErrorKind::InvalidInput, "hits")
                .with_message("initial hubs: vertices and values must have the same length")
                .with_context("vertices_len", initial_hubs.vertices.len().to_string())
                .with_context("values_len", initial_hubs.values.len().to_string()));
        }
        graph.validate_vertex_input_dtype(V::data_type(), "hits", "initial hubs vertices")?;
        let dtype = W::data_type();
        if !dtype.is_float() {
            return Err(Error::new(ErrorKind::UnsupportedType, "hits")
                .with_message(format!("initial hubs values require Float32 or Float64 values, got {dtype}")));
        }
    }

    let (initial_vertices_ptr, initial_values_ptr) = match initial_hubs {
        Some(initial_hubs) => (initial_hubs.vertices.as_ptr() as *const _, initial_hubs.values.as_ptr() as *const _),
        None => (ptr::null(), ptr::null()),
    };
    let do_expensive_check = params.do_expensive_check || initial_hubs.is_some();

    let mut result: *mut cugraph_hits_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();
    let graph_ptr = graph.as_ptr(handle, "hits")?;

    // SAFETY: optional input pointers are either null or borrowed from live
    // device views validated above. Non-null initial hub vertices force
    // libcugraph validation so out-of-range vertices cannot drive unchecked
    // native indexing. `result`/`error` are valid out-pointers.
    unsafe {
        let code = cugraph_hits(
            handle.handle_ptr(),
            graph_ptr,
            params.epsilon.get(),
            usize::from(params.max_iterations),
            initial_vertices_ptr,
            initial_values_ptr,
            to_bool_t(params.normalize),
            to_bool_t(do_expensive_check),
            &mut result,
            &mut error,
        );
        check_error(code, error, handle.device(), "hits")?;
    }

    let result = require_non_null(result, "hits", "hits_result")?;
    // SAFETY: on success, cuGraph returns ownership of `result` to the caller.
    unsafe { HitsResult::from_raw(result, handle.device(), handle.runtime_bound_lease()) }
}
