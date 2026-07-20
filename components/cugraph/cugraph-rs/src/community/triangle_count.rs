//! Triangle-counting algorithm.

use std::{marker::PhantomData, ptr};

use cugraph_sys::{
    cugraph_error_t, cugraph_triangle_count, cugraph_triangle_count_result_get_counts,
    cugraph_triangle_count_result_get_vertices, cugraph_triangle_count_result_t,
    cugraph_type_erased_device_array_view_t,
};

use crate::{
    array::{DeviceArrayView, RawDeviceArrayView},
    cuda::{DeviceResourceKind, drop_device_resource},
    error::{Error, Result, check_error, require_non_null},
    graph::Graph,
    handle::{EnteredContext, ResourceHandleLease},
    structure::typed_results::{IntegerValues, VertexIds},
    types::{CugraphElement, to_bool_t},
};

/// Parameters for [`Graph::triangle_count`](crate::graph::Graph::triangle_count).
#[derive(Debug, Clone, Copy, Default)]
#[must_use]
pub struct TriangleCountParams {
    /// Whether to run expensive input validation in the C library.
    pub do_expensive_check: bool,
}

/// Compute triangle counts for all vertices or a selected start list.
///
/// Prefer using [`Graph::triangle_count`](crate::graph::Graph::triangle_count)
/// instead of calling this function directly.
pub(crate) fn triangle_count<T, V: CugraphElement>(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<T>,
    start: Option<&DeviceArrayView<'_, V>>,
    params: &TriangleCountParams,
) -> Result<TriangleCountResult> {
    let mut result: *mut cugraph_triangle_count_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();
    let start_ptr = start.map(|v| v.as_ptr().cast_const()).unwrap_or(ptr::null());
    let graph_ptr = graph.as_ptr(handle, "triangle_count")?;

    // SAFETY: `start_ptr` is either null or borrowed from a live device view,
    // and `result`/`error` are valid out-pointers.
    unsafe {
        let code = cugraph_triangle_count(
            handle.handle_ptr(),
            graph_ptr,
            start_ptr,
            to_bool_t(params.do_expensive_check),
            &mut result,
            &mut error,
        );
        check_error(code, error, handle.device(), "triangle_count")?;
    }

    let result = require_non_null(result, "triangle_count", "triangle_count_result")?;

    // SAFETY: on success, cuGraph transfers ownership of the result pointer.
    unsafe { TriangleCountResult::from_raw(result, handle.device(), handle.runtime_bound_lease()) }
}

/// Result of a triangle-count computation.
///
/// Provides vertex ids and per-vertex triangle counts. Freed on drop via
/// `cugraph_triangle_count_result_free`.
pub struct TriangleCountResult {
    raw: *mut cugraph_triangle_count_result_t,
    device: i32,
    // Keeps a runtime-bound cuGraph handle alive until after Drop frees `raw`.
    _resource_handle_lease: Option<ResourceHandleLease>,
    _nosend_nosync: PhantomData<*mut ()>,
}

impl std::fmt::Debug for TriangleCountResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TriangleCountResult").finish_non_exhaustive()
    }
}

impl TriangleCountResult {
    /// Wrap a raw triangle-count result pointer.
    ///
    /// # Safety
    ///
    /// `raw` must be a valid, non-null pointer returned by
    /// `cugraph_triangle_count`. Ownership is transferred to this struct.
    pub(crate) unsafe fn from_raw(
        raw: *mut cugraph_triangle_count_result_t,
        device: i32,
        resource_handle_lease: Option<ResourceHandleLease>,
    ) -> Result<Self> {
        let raw = require_non_null(raw, "TriangleCountResult::from_raw", "triangle_count_result")?;
        Ok(Self { raw, device, _resource_handle_lease: resource_handle_lease, _nosend_nosync: PhantomData })
    }

    fn raw_view(
        &self,
        getter: unsafe extern "C" fn(
            *mut cugraph_triangle_count_result_t,
        ) -> *mut cugraph_type_erased_device_array_view_t,
        operation: &'static str,
        pointer_name: &'static str,
    ) -> Result<RawDeviceArrayView<'_>> {
        let view_ptr = require_non_null(
            // SAFETY: `self.raw` is a live triangle-count result owned by this wrapper.
            unsafe { getter(self.raw) },
            operation,
            pointer_name,
        )?;
        // SAFETY: `view_ptr` is a borrowed child view tied to the live result.
        unsafe { RawDeviceArrayView::from_borrowed(view_ptr, Some(self.device), operation, pointer_name) }
    }

    /// Get vertex ids as a runtime-typed device array view.
    pub fn vertex_ids(&self) -> Result<VertexIds<'_>> {
        VertexIds::from_raw(self.raw_view(
            cugraph_triangle_count_result_get_vertices,
            "TriangleCountResult::vertex_ids",
            "vertices",
        )?)
    }

    /// Get triangle counts as a runtime-typed integer device array view.
    pub fn counts(&self) -> Result<IntegerValues<'_>> {
        IntegerValues::from_raw(self.raw_view(
            cugraph_triangle_count_result_get_counts,
            "TriangleCountResult::counts",
            "counts",
        )?)
    }

    /// Convert this result to a cudf DataFrame with `"vertex"` and
    /// `"triangle_count"` columns.
    pub fn to_dataframe(&self, runtime: &cudf::Runtime) -> Result<cudf::DataFrame> {
        crate::cudf_interop::triangle_count_result_to_dataframe(self, runtime)
            .map_err(|e: Error| e.with_operation("TriangleCountResult::to_dataframe"))
    }
}

impl Drop for TriangleCountResult {
    fn drop(&mut self) {
        drop_device_resource(
            &mut self.raw,
            self.device,
            "TriangleCountResult::drop",
            DeviceResourceKind::TriangleCountResult,
        );
    }
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use super::*;
    use crate::error::ErrorKind;

    #[test]
    fn triangle_count_result_from_raw_rejects_null() {
        // SAFETY: this exercises the constructor's null guard only.
        let err = match unsafe { TriangleCountResult::from_raw(ptr::null_mut(), 0, None) } {
            Err(err) => err,
            Ok(_) => panic!("null must be rejected"),
        };
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert_eq!(err.operation(), "TriangleCountResult::from_raw");
    }
}
