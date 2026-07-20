//! Internal wrappers for induced-subgraph shaped C API results.

use std::marker::PhantomData;

use cugraph_sys::{
    cugraph_induced_subgraph_get_destinations, cugraph_induced_subgraph_get_edge_ids,
    cugraph_induced_subgraph_get_edge_type_ids, cugraph_induced_subgraph_get_edge_weights,
    cugraph_induced_subgraph_get_sources, cugraph_induced_subgraph_get_subgraph_offsets,
    cugraph_induced_subgraph_result_t, cugraph_type_erased_device_array_view_t,
};

use crate::{
    array::RawDeviceArrayView,
    cuda::{DeviceResourceKind, drop_device_resource},
    error::{Result, require_non_null},
    handle::ResourceHandleLease,
    structure::typed_results::{IntegerValues, ScoreValues, SizeValues, VertexIds},
};

pub(crate) struct InducedSubgraphResult {
    raw: *mut cugraph_induced_subgraph_result_t,
    device: i32,
    // Keeps a runtime-bound cuGraph handle alive until after Drop frees `raw`.
    _resource_handle_lease: Option<ResourceHandleLease>,
    _nosend_nosync: PhantomData<*mut ()>,
}

impl InducedSubgraphResult {
    pub(crate) unsafe fn from_raw(
        raw: *mut cugraph_induced_subgraph_result_t,
        device: i32,
        operation: &'static str,
        resource_handle_lease: Option<ResourceHandleLease>,
    ) -> Result<Self> {
        let raw = require_non_null(raw, operation, "induced_subgraph_result")?;
        Ok(Self { raw, device, _resource_handle_lease: resource_handle_lease, _nosend_nosync: PhantomData })
    }

    fn raw_view(
        &self,
        getter: unsafe extern "C" fn(
            *mut cugraph_induced_subgraph_result_t,
        ) -> *mut cugraph_type_erased_device_array_view_t,
        operation: &'static str,
        pointer_name: &'static str,
    ) -> Result<RawDeviceArrayView<'_>> {
        let view_ptr = require_non_null(
            // SAFETY: `self.raw` is a live induced-subgraph result owned by this wrapper.
            unsafe { getter(self.raw) },
            operation,
            pointer_name,
        )?;
        // SAFETY: cuGraph result getters allocate view metadata for the
        // caller. The backing device allocation remains owned by `self.raw`.
        unsafe { RawDeviceArrayView::from_owned_view(view_ptr, Some(self.device), operation, pointer_name) }
    }

    fn optional_raw_view(
        &self,
        getter: unsafe extern "C" fn(
            *mut cugraph_induced_subgraph_result_t,
        ) -> *mut cugraph_type_erased_device_array_view_t,
        operation: &'static str,
    ) -> Result<Option<RawDeviceArrayView<'_>>> {
        // SAFETY: `self.raw` is a live induced-subgraph result owned by this wrapper.
        let view_ptr = unsafe { getter(self.raw) };
        if view_ptr.is_null() {
            return Ok(None);
        }

        // SAFETY: cuGraph result getters allocate view metadata for the
        // caller. The backing device allocation remains owned by `self.raw`.
        unsafe { RawDeviceArrayView::from_owned_view(view_ptr, Some(self.device), operation, "optional_view") }
            .map(Some)
    }

    pub(crate) fn sources(&self, operation: &'static str) -> Result<VertexIds<'_>> {
        VertexIds::from_raw(self.raw_view(cugraph_induced_subgraph_get_sources, operation, "sources")?)
    }

    pub(crate) fn destinations(&self, operation: &'static str) -> Result<VertexIds<'_>> {
        VertexIds::from_raw(self.raw_view(cugraph_induced_subgraph_get_destinations, operation, "destinations")?)
    }

    pub(crate) fn weights(&self, operation: &'static str) -> Result<Option<ScoreValues<'_>>> {
        self.optional_raw_view(cugraph_induced_subgraph_get_edge_weights, operation)?
            .map(ScoreValues::from_raw)
            .transpose()
    }

    pub(crate) fn edge_ids(&self, operation: &'static str) -> Result<Option<IntegerValues<'_>>> {
        self.optional_raw_view(cugraph_induced_subgraph_get_edge_ids, operation)?
            .map(IntegerValues::from_raw)
            .transpose()
    }

    pub(crate) fn edge_type_ids(&self, operation: &'static str) -> Result<Option<IntegerValues<'_>>> {
        self.optional_raw_view(cugraph_induced_subgraph_get_edge_type_ids, operation)?
            .map(IntegerValues::from_raw)
            .transpose()
    }

    pub(crate) fn subgraph_offsets(&self, operation: &'static str) -> Result<Option<SizeValues<'_>>> {
        self.optional_raw_view(cugraph_induced_subgraph_get_subgraph_offsets, operation)?
            .map(SizeValues::from_raw)
            .transpose()
    }
}

impl Drop for InducedSubgraphResult {
    fn drop(&mut self) {
        drop_device_resource(
            &mut self.raw,
            self.device,
            "InducedSubgraphResult::drop",
            DeviceResourceKind::InducedSubgraphResult,
        );
    }
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use super::*;
    use crate::error::ErrorKind;

    #[test]
    fn induced_subgraph_result_from_raw_rejects_null() {
        // SAFETY: this exercises the constructor's null guard only.
        let err = match unsafe { InducedSubgraphResult::from_raw(ptr::null_mut(), 0, "test", None) } {
            Err(err) => err,
            Ok(_) => panic!("null must be rejected"),
        };
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert_eq!(err.operation(), "test");
    }
}
