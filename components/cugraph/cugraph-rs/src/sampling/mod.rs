//! Random-walk and sampling algorithms.

use std::{num::NonZeroUsize, ptr};

use cugraph_sys::{
    cugraph_biased_random_walks, cugraph_error_t, cugraph_random_walk_result_free,
    cugraph_random_walk_result_get_max_path_length, cugraph_random_walk_result_get_paths,
    cugraph_random_walk_result_get_weights, cugraph_random_walk_result_t, cugraph_select_random_vertices,
    cugraph_type_erased_device_array_t, cugraph_uniform_random_walks,
};

use crate::{
    array::{DeviceArray, DeviceArrayView, RawDeviceArrayView},
    cuda::DeviceGuard,
    error::{Result, check_error, require_non_null},
    graph::{Forward, Graph},
    handle::EnteredContext,
    params::PositiveF64,
    random::RngState,
    structure::{ScoreValues, VertexIds},
    types::CugraphElement,
};

/// Parameters shared by uniform and biased random walks.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct RandomWalkParams {
    /// Maximum number of edges traversed per walk.
    pub max_length: NonZeroUsize,
}

impl Default for RandomWalkParams {
    fn default() -> Self {
        Self { max_length: NonZeroUsize::new(1).expect("literal is non-zero") }
    }
}

/// Parameters for node2vec random walks.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct Node2VecRandomWalkParams {
    /// Maximum number of edges traversed per walk.
    pub max_length: NonZeroUsize,
    /// Return parameter.
    pub p: PositiveF64,
    /// In/out parameter.
    pub q: PositiveF64,
}

impl Default for Node2VecRandomWalkParams {
    fn default() -> Self {
        Self {
            max_length: NonZeroUsize::new(1).expect("literal is non-zero"),
            p: PositiveF64::new(1.0).expect("literal is positive"),
            q: PositiveF64::new(1.0).expect("literal is positive"),
        }
    }
}

/// Result of a random-walk computation.
pub struct RandomWalkResult {
    raw: *mut cugraph_random_walk_result_t,
    device: i32,
    has_weights: bool,
    _nosend_nosync: std::marker::PhantomData<*mut ()>,
}

impl std::fmt::Debug for RandomWalkResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RandomWalkResult")
            .field("max_path_length", &self.max_path_length())
            .field("has_weights", &self.has_weights)
            .finish_non_exhaustive()
    }
}

impl RandomWalkResult {
    /// # Safety
    ///
    /// `raw` must be a valid, non-null pointer returned by a cuGraph
    /// random-walk C API. Ownership is transferred to this struct.
    pub(crate) unsafe fn from_raw(
        raw: *mut cugraph_random_walk_result_t,
        device: i32,
        has_weights: bool,
    ) -> Result<Self> {
        let raw = require_non_null(raw, "RandomWalkResult::from_raw", "random_walk_result")?;
        Ok(Self { raw, device, has_weights, _nosend_nosync: std::marker::PhantomData })
    }

    /// Maximum number of edges requested per walk.
    pub fn max_path_length(&self) -> usize {
        // SAFETY: `self.raw` is a live random-walk result handle.
        unsafe { cugraph_random_walk_result_get_max_path_length(self.raw) }
    }

    /// Dense row-major vertex path matrix.
    ///
    /// The row count is the number of start vertices and each row contains
    /// `max_path_length() + 1` vertex ids. Native cuGraph uses a negative
    /// sentinel in later columns when a walk terminates early.
    pub fn paths(&self) -> Result<VertexIds<'_>> {
        VertexIds::from_raw(self.raw_view(cugraph_random_walk_result_get_paths, "RandomWalkResult::paths", "paths")?)
    }

    /// Dense row-major edge-weight matrix, if the input graph had weights.
    ///
    /// Each row contains `max_path_length()` edge weights. For unweighted
    /// graphs the native result has no weight column, so this returns `None`.
    pub fn weights(&self) -> Result<Option<ScoreValues<'_>>> {
        if !self.has_weights {
            return Ok(None);
        }

        let raw = self.raw_view(cugraph_random_walk_result_get_weights, "RandomWalkResult::weights", "weights")?;
        ScoreValues::from_raw(raw).map(Some)
    }

    fn raw_view(
        &self,
        getter: unsafe extern "C" fn(
            *mut cugraph_random_walk_result_t,
        ) -> *mut cugraph_sys::cugraph_type_erased_device_array_view_t,
        operation: &'static str,
        pointer_name: &'static str,
    ) -> Result<RawDeviceArrayView<'_>> {
        let view_ptr = require_non_null(
            // SAFETY: `self.raw` is owned by this wrapper and remains live for
            // the duration of this call.
            unsafe { getter(self.raw) },
            operation,
            pointer_name,
        )?;
        // SAFETY: random-walk getters allocate view metadata for the caller.
        // The backing device arrays are owned by `self.raw`.
        unsafe { RawDeviceArrayView::from_owned_view(view_ptr, Some(self.device), operation, pointer_name) }
    }
}

impl Drop for RandomWalkResult {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _guard = DeviceGuard::switch_to_for_drop(self.device, "RandomWalkResult::drop").ok();
            // SAFETY: `self.raw` is owned by this wrapper and freed at most
            // once here.
            unsafe { cugraph_random_walk_result_free(self.raw) };
        }
    }
}

type RandomWalkFn = unsafe extern "C" fn(
    *const cugraph_sys::cugraph_resource_handle_t,
    *mut cugraph_sys::cugraph_rng_state_t,
    *mut cugraph_sys::cugraph_graph_t,
    *const cugraph_sys::cugraph_type_erased_device_array_view_t,
    usize,
    *mut *mut cugraph_random_walk_result_t,
    *mut *mut cugraph_error_t,
) -> cugraph_sys::cugraph_error_code_t;

pub(crate) fn uniform_random_walks<V: CugraphElement>(
    handle: &EnteredContext<'_>,
    rng_state: &mut RngState,
    graph: &mut Graph<Forward>,
    start_vertices: &DeviceArrayView<'_, V>,
    params: &RandomWalkParams,
) -> Result<RandomWalkResult> {
    run_random_walks(
        handle,
        rng_state,
        graph,
        start_vertices,
        params.max_length.get(),
        cugraph_uniform_random_walks,
        "uniform_random_walks",
    )
}

pub(crate) fn biased_random_walks<V: CugraphElement>(
    handle: &EnteredContext<'_>,
    rng_state: &mut RngState,
    graph: &mut Graph<Forward>,
    start_vertices: &DeviceArrayView<'_, V>,
    params: &RandomWalkParams,
) -> Result<RandomWalkResult> {
    run_random_walks(
        handle,
        rng_state,
        graph,
        start_vertices,
        params.max_length.get(),
        cugraph_biased_random_walks,
        "biased_random_walks",
    )
}

fn run_random_walks<V: CugraphElement>(
    handle: &EnteredContext<'_>,
    rng_state: &mut RngState,
    graph: &mut Graph<Forward>,
    start_vertices: &DeviceArrayView<'_, V>,
    max_length: usize,
    function: RandomWalkFn,
    operation: &'static str,
) -> Result<RandomWalkResult> {
    graph.validate_vertex_input_dtype(V::data_type(), operation, "start_vertices")?;
    let mut result: *mut cugraph_random_walk_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();

    // SAFETY: `graph`, `rng_state`, and `start_vertices` are live cuGraph
    // objects, and `result`/`error` are valid out-pointers.
    unsafe {
        let code = function(
            handle.handle_ptr(),
            rng_state.as_ptr(),
            graph.as_ptr(),
            start_vertices.as_ptr(),
            max_length,
            &mut result,
            &mut error,
        );
        check_error(code, error, operation)?;
    }

    // SAFETY: on success, cuGraph transfers ownership of the result pointer.
    unsafe { RandomWalkResult::from_raw(result, handle.device(), graph.has_weights()) }
}

pub(crate) fn node2vec_random_walks<V: CugraphElement>(
    handle: &EnteredContext<'_>,
    rng_state: &mut RngState,
    graph: &mut Graph<Forward>,
    start_vertices: &DeviceArrayView<'_, V>,
    params: &Node2VecRandomWalkParams,
) -> Result<RandomWalkResult> {
    graph.validate_vertex_input_dtype(V::data_type(), "node2vec_random_walks", "start_vertices")?;
    let mut result: *mut cugraph_random_walk_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();

    // SAFETY: `graph`, `rng_state`, and `start_vertices` are live cuGraph
    // objects, and `result`/`error` are valid out-pointers.
    unsafe {
        let code = cugraph_sys::cugraph_node2vec_random_walks(
            handle.handle_ptr(),
            rng_state.as_ptr(),
            graph.as_ptr(),
            start_vertices.as_ptr(),
            params.max_length.get(),
            params.p.get(),
            params.q.get(),
            &mut result,
            &mut error,
        );
        check_error(code, error, "node2vec_random_walks")?;
    }

    // SAFETY: on success, cuGraph transfers ownership of the result pointer.
    unsafe { RandomWalkResult::from_raw(result, handle.device(), graph.has_weights()) }
}

pub(crate) fn select_random_vertices<T, V>(
    handle: &EnteredContext<'_>,
    graph: &Graph<T>,
    rng_state: &mut RngState,
    num_vertices: usize,
) -> Result<DeviceArray<V>>
where
    V: CugraphElement,
{
    graph.validate_vertex_input_dtype(V::data_type(), "select_random_vertices", "vertices")?;
    let mut vertices: *mut cugraph_type_erased_device_array_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();

    // SAFETY: `graph` and `rng_state` are live cuGraph objects, and
    // `vertices`/`error` are valid out-pointers.
    unsafe {
        let code = cugraph_select_random_vertices(
            handle.handle_ptr(),
            graph.as_ptr(),
            rng_state.as_ptr(),
            num_vertices,
            &mut vertices,
            &mut error,
        );
        check_error(code, error, "select_random_vertices")?;
    }

    // SAFETY: on success, cuGraph transfers ownership of the device array.
    unsafe { DeviceArray::from_raw(vertices, handle.device(), "select_random_vertices", "vertices") }
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use super::*;
    use crate::error::ErrorKind;

    #[test]
    fn random_walk_result_from_raw_rejects_null() {
        // SAFETY: this exercises the constructor's null guard only.
        let err = match unsafe { RandomWalkResult::from_raw(ptr::null_mut(), 0, false) } {
            Err(err) => err,
            Ok(_) => panic!("null must be rejected"),
        };
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert_eq!(err.operation(), "RandomWalkResult::from_raw");
    }
}
