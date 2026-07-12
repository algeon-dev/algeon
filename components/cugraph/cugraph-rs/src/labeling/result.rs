//! Result wrappers for vertex labeling algorithms.

use std::marker::PhantomData;

use cugraph_sys::{
    cugraph_labeling_result_free, cugraph_labeling_result_get_labels, cugraph_labeling_result_get_vertices,
    cugraph_labeling_result_t, cugraph_type_erased_device_array_view_t,
};

use crate::{
    array::RawDeviceArrayView,
    cuda::DeviceGuard,
    error::{Error, Result, require_non_null},
    handle::ResourceHandleLease,
    structure::typed_results::{IntegerValues, VertexIds},
};

struct LabelingResult {
    raw: *mut cugraph_labeling_result_t,
    device: i32,
    // Keeps a runtime-bound cuGraph handle alive until after Drop frees `raw`.
    _resource_handle_lease: Option<ResourceHandleLease>,
    _nosend_nosync: PhantomData<*mut ()>,
}

impl LabelingResult {
    unsafe fn from_raw(
        raw: *mut cugraph_labeling_result_t,
        device: i32,
        resource_handle_lease: Option<ResourceHandleLease>,
    ) -> Result<Self> {
        let raw = require_non_null(raw, "LabelingResult::from_raw", "labeling_result")?;
        Ok(Self { raw, device, _resource_handle_lease: resource_handle_lease, _nosend_nosync: PhantomData })
    }

    fn raw_view(
        &self,
        getter: unsafe extern "C" fn(*mut cugraph_labeling_result_t) -> *mut cugraph_type_erased_device_array_view_t,
        operation: &'static str,
        pointer_name: &'static str,
    ) -> Result<RawDeviceArrayView<'_>> {
        let view_ptr = require_non_null(
            // SAFETY: `self.raw` is a live labeling result owned by this wrapper.
            unsafe { getter(self.raw) },
            operation,
            pointer_name,
        )?;
        // SAFETY: `view_ptr` is a borrowed child view tied to the live result.
        unsafe { RawDeviceArrayView::from_borrowed(view_ptr, Some(self.device), operation, pointer_name) }
    }

    fn vertex_ids(&self, operation: &'static str) -> Result<VertexIds<'_>> {
        VertexIds::from_raw(self.raw_view(cugraph_labeling_result_get_vertices, operation, "vertices")?)
    }

    fn labels(&self, operation: &'static str) -> Result<IntegerValues<'_>> {
        IntegerValues::from_raw(self.raw_view(cugraph_labeling_result_get_labels, operation, "labels")?)
    }
}

impl Drop for LabelingResult {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _guard = DeviceGuard::switch_to_for_drop(self.device, "LabelingResult::drop").ok();
            // SAFETY: `self.raw` is owned by this wrapper and freed exactly once.
            unsafe { cugraph_labeling_result_free(self.raw) };
        }
    }
}

/// Result of weakly connected components.
pub struct WeaklyConnectedComponentsResult(LabelingResult);

impl std::fmt::Debug for WeaklyConnectedComponentsResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeaklyConnectedComponentsResult").finish_non_exhaustive()
    }
}

impl WeaklyConnectedComponentsResult {
    /// Wrap a raw labeling result pointer.
    ///
    /// # Safety
    ///
    /// `raw` must be a valid, non-null pointer returned by
    /// `cugraph_weakly_connected_components`. Ownership is transferred.
    pub(crate) unsafe fn from_raw(
        raw: *mut cugraph_labeling_result_t,
        device: i32,
        resource_handle_lease: Option<ResourceHandleLease>,
    ) -> Result<Self> {
        // SAFETY: the caller transfers ownership of a live cuGraph result.
        Ok(Self(unsafe { LabelingResult::from_raw(raw, device, resource_handle_lease) }?))
    }

    /// Get vertex ids as a runtime-typed device array view.
    pub fn vertex_ids(&self) -> Result<VertexIds<'_>> {
        self.0.vertex_ids("WeaklyConnectedComponentsResult::vertex_ids")
    }

    /// Get component labels as a runtime-typed integer device array view.
    pub fn labels(&self) -> Result<IntegerValues<'_>> {
        self.0.labels("WeaklyConnectedComponentsResult::labels")
    }

    /// Convert this result to a cudf DataFrame with `"vertex"` and `"label"` columns.
    pub fn to_dataframe(&self, runtime: &cudf::Runtime) -> Result<cudf::DataFrame> {
        crate::cudf_interop::weakly_connected_components_result_to_dataframe(self, runtime)
            .map_err(|e: Error| e.with_operation("WeaklyConnectedComponentsResult::to_dataframe"))
    }
}

/// Result of strongly connected components.
pub struct StronglyConnectedComponentsResult(LabelingResult);

impl std::fmt::Debug for StronglyConnectedComponentsResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StronglyConnectedComponentsResult").finish_non_exhaustive()
    }
}

impl StronglyConnectedComponentsResult {
    /// Wrap a raw labeling result pointer.
    ///
    /// # Safety
    ///
    /// `raw` must be a valid, non-null pointer returned by
    /// `cugraph_strongly_connected_components`. Ownership is transferred.
    pub(crate) unsafe fn from_raw(
        raw: *mut cugraph_labeling_result_t,
        device: i32,
        resource_handle_lease: Option<ResourceHandleLease>,
    ) -> Result<Self> {
        // SAFETY: the caller transfers ownership of a live cuGraph result.
        Ok(Self(unsafe { LabelingResult::from_raw(raw, device, resource_handle_lease) }?))
    }

    /// Get vertex ids as a runtime-typed device array view.
    pub fn vertex_ids(&self) -> Result<VertexIds<'_>> {
        self.0.vertex_ids("StronglyConnectedComponentsResult::vertex_ids")
    }

    /// Get component labels as a runtime-typed integer device array view.
    pub fn labels(&self) -> Result<IntegerValues<'_>> {
        self.0.labels("StronglyConnectedComponentsResult::labels")
    }

    /// Convert this result to a cudf DataFrame with `"vertex"` and `"label"` columns.
    pub fn to_dataframe(&self, runtime: &cudf::Runtime) -> Result<cudf::DataFrame> {
        crate::cudf_interop::strongly_connected_components_result_to_dataframe(self, runtime)
            .map_err(|e: Error| e.with_operation("StronglyConnectedComponentsResult::to_dataframe"))
    }
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use super::*;
    use crate::error::ErrorKind;

    #[test]
    fn weakly_connected_components_result_from_raw_rejects_null() {
        // SAFETY: this exercises the constructor's null guard only.
        let err = match unsafe { WeaklyConnectedComponentsResult::from_raw(ptr::null_mut(), 0, None) } {
            Err(err) => err,
            Ok(_) => panic!("null must be rejected"),
        };
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert_eq!(err.operation(), "LabelingResult::from_raw");
    }

    #[test]
    fn strongly_connected_components_result_from_raw_rejects_null() {
        // SAFETY: this exercises the constructor's null guard only.
        let err = match unsafe { StronglyConnectedComponentsResult::from_raw(ptr::null_mut(), 0, None) } {
            Err(err) => err,
            Ok(_) => panic!("null must be rejected"),
        };
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert_eq!(err.operation(), "LabelingResult::from_raw");
    }
}
