//! Result wrapper for vertex labeling algorithms.

use std::marker::PhantomData;

use cugraph_sys::{
    cugraph_labeling_result_get_labels, cugraph_labeling_result_get_vertices, cugraph_labeling_result_t,
    cugraph_type_erased_device_array_view_t,
};

use crate::{
    array::RawDeviceArrayView,
    cuda::{DeviceResourceKind, drop_device_resource},
    cudf_interop::labeling_result_to_dataframe,
    error::{Error, Result, require_non_null},
    handle::ResourceHandleLease,
    structure::typed_results::{IntegerValues, VertexIds},
};

/// Result of weakly or strongly connected components.
///
/// Both algorithms return a `cugraph_labeling_result_t` from the C API with
/// the same field layout, so they share this result type.
pub struct LabelingResult {
    raw: *mut cugraph_labeling_result_t,
    device: i32,
    // Keeps a runtime-bound cuGraph handle alive until after Drop frees `raw`.
    _resource_handle_lease: Option<ResourceHandleLease>,
    _nosend_nosync: PhantomData<*mut ()>,
}

impl std::fmt::Debug for LabelingResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LabelingResult").finish_non_exhaustive()
    }
}

impl LabelingResult {
    /// Wrap a raw labeling result pointer.
    ///
    /// # Safety
    ///
    /// `raw` must be a valid, non-null pointer returned by
    /// `cugraph_weakly_connected_components` or
    /// `cugraph_strongly_connected_components`. Ownership is transferred to
    /// this struct.
    pub(crate) unsafe fn from_raw(
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
        // SAFETY: cuGraph allocates this view metadata; this wrapper frees it, and the result retains its backing arrays.
        unsafe { RawDeviceArrayView::from_owned_view(view_ptr, Some(self.device), operation, pointer_name) }
    }

    /// Get vertex ids as a runtime-typed device array view.
    pub fn vertex_ids(&self) -> Result<VertexIds<'_>> {
        VertexIds::from_raw(self.raw_view(
            cugraph_labeling_result_get_vertices,
            "LabelingResult::vertex_ids",
            "vertices",
        )?)
    }

    /// Get component labels as a runtime-typed integer device array view.
    pub fn labels(&self) -> Result<IntegerValues<'_>> {
        IntegerValues::from_raw(self.raw_view(
            cugraph_labeling_result_get_labels,
            "LabelingResult::labels",
            "labels",
        )?)
    }

    /// Convert this result to a cudf DataFrame with `"vertex"` and `"label"` columns.
    ///
    /// Each `"label"` is the minimum vertex id of its component, so labels are
    /// stable across runs.
    pub fn to_dataframe(&self, runtime: &cudf::Runtime) -> Result<cudf::DataFrame> {
        labeling_result_to_dataframe(self, runtime).map_err(|e: Error| e.with_operation("LabelingResult::to_dataframe"))
    }
}

impl Drop for LabelingResult {
    fn drop(&mut self) {
        drop_device_resource(&mut self.raw, self.device, "LabelingResult::drop", DeviceResourceKind::LabelingResult);
    }
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use super::*;
    use crate::error::ErrorKind;

    #[test]
    fn labeling_result_from_raw_rejects_null() {
        // SAFETY: this exercises the constructor's null guard only.
        let err = match unsafe { LabelingResult::from_raw(ptr::null_mut(), 0, None) } {
            Err(err) => err,
            Ok(_) => panic!("null must be rejected"),
        };
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert_eq!(err.operation(), "LabelingResult::from_raw");
    }
}
