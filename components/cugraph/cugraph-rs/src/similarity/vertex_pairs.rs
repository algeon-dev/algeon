//! RAII wrapper for `cugraph_vertex_pairs_t`.

use std::marker::PhantomData;

use cugraph_sys::{
    cugraph_type_erased_device_array_view_t, cugraph_vertex_pairs_get_first, cugraph_vertex_pairs_get_second,
    cugraph_vertex_pairs_t,
};

use crate::{
    array::RawDeviceArrayView,
    cuda::{DeviceResourceKind, drop_device_resource},
    error::{Error, ErrorKind, Result, require_non_null},
    graph::DeferredInputCompletion,
    handle::ResourceHandleLease,
    structure::typed_results::VertexIds,
};

/// Pair-shaped vertex input or output owned by cuGraph.
///
/// Pairwise similarity algorithms consume this type as their explicit input.
/// All-pairs similarity results also expose generated vertex pairs through the
/// same wrapper.
pub struct VertexPairs {
    raw: *mut cugraph_vertex_pairs_t,
    device: i32,
    // Prepared cuDF pair columns can still be read asynchronously while the
    // native pair object is live.  Retain their producer owners and completion
    // witnesses until drop so the zero-copy ingress path cannot outlive input
    // storage or race its producer stream.
    pending_input_completion: Option<DeferredInputCompletion>,
    // Keeps a runtime-bound cuGraph handle alive until after Drop frees `raw`.
    _resource_handle_lease: Option<ResourceHandleLease>,
    _nosend_nosync: PhantomData<*mut ()>,
}

impl std::fmt::Debug for VertexPairs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VertexPairs").finish_non_exhaustive()
    }
}

impl VertexPairs {
    /// Wrap a raw vertex-pairs pointer transferred from cuGraph.
    ///
    /// # Safety
    ///
    /// `raw` must be a valid, non-null pointer returned by cuGraph. Ownership
    /// is transferred to this wrapper and freed exactly once on drop.
    pub(crate) unsafe fn from_raw(
        raw: *mut cugraph_vertex_pairs_t,
        device: i32,
        operation: &'static str,
        resource_handle_lease: Option<ResourceHandleLease>,
    ) -> Result<Self> {
        let raw = require_non_null(raw, operation, "vertex_pairs")?;
        Ok(Self {
            raw,
            device,
            pending_input_completion: None,
            _resource_handle_lease: resource_handle_lease,
            _nosend_nosync: PhantomData,
        })
    }

    pub(crate) fn as_ptr_const(&self) -> *const cugraph_vertex_pairs_t {
        self.raw.cast_const()
    }

    pub(crate) fn with_pending_input_completion(mut self, pending: DeferredInputCompletion) -> Self {
        self.pending_input_completion = Some(pending);
        self
    }

    fn raw_view(
        &self,
        getter: unsafe extern "C" fn(*mut cugraph_vertex_pairs_t) -> *mut cugraph_type_erased_device_array_view_t,
        operation: &'static str,
        pointer_name: &'static str,
    ) -> Result<RawDeviceArrayView<'_>> {
        let view_ptr = require_non_null(
            // SAFETY: `self.raw` is a live vertex-pairs object owned by this wrapper.
            unsafe { getter(self.raw) },
            operation,
            pointer_name,
        )?;
        // SAFETY: cuGraph vertex-pair getters allocate view metadata for the
        // caller. The backing device allocation remains owned by `self.raw`.
        unsafe { RawDeviceArrayView::from_owned_view(view_ptr, Some(self.device), operation, pointer_name) }
    }

    /// Get the first vertex id column.
    pub fn first(&self) -> Result<VertexIds<'_>> {
        VertexIds::from_raw(self.raw_view(cugraph_vertex_pairs_get_first, "VertexPairs::first", "first")?)
    }

    /// Get the second vertex id column.
    pub fn second(&self) -> Result<VertexIds<'_>> {
        VertexIds::from_raw(self.raw_view(cugraph_vertex_pairs_get_second, "VertexPairs::second", "second")?)
    }

    /// Number of vertex pairs.
    ///
    /// Returns an error if native state ever reports mismatched first/second
    /// lengths.
    pub fn len(&self) -> Result<usize> {
        let first_len = self.first()?.len();
        let second_len = self.second()?.len();
        if first_len != second_len {
            return Err(Error::new(ErrorKind::InvalidState, "VertexPairs::len")
                .with_message("vertex-pair columns have different lengths")
                .with_context("first_len", first_len.to_string())
                .with_context("second_len", second_len.to_string()));
        }
        Ok(first_len)
    }

    /// Whether this pair collection is empty.
    pub fn is_empty(&self) -> Result<bool> {
        self.len().map(|len| len == 0)
    }

    /// Convert this pair collection to a cudf DataFrame with `"first"` and
    /// `"second"` columns.
    pub fn to_dataframe(&self, runtime: &cudf::Runtime) -> Result<cudf::DataFrame> {
        crate::cudf_interop::vertex_pairs_to_dataframe(self, runtime)
            .map_err(|error| error.with_operation("VertexPairs::to_dataframe"))
    }
}

impl Drop for VertexPairs {
    fn drop(&mut self) {
        if let Some(pending) = self.pending_input_completion.take() {
            pending.synchronize();
        }
        drop_device_resource(&mut self.raw, self.device, "VertexPairs::drop", DeviceResourceKind::VertexPairs);
    }
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use super::*;

    #[test]
    fn vertex_pairs_from_raw_rejects_null() {
        // SAFETY: this exercises the constructor's null guard only.
        let err = match unsafe { VertexPairs::from_raw(ptr::null_mut(), 0, "test_vertex_pairs", None) } {
            Err(err) => err,
            Ok(_) => panic!("null must be rejected"),
        };
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert_eq!(err.operation(), "test_vertex_pairs");
    }
}
