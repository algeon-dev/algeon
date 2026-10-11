//! Transient decompressed-edgelist views for structural queries.

use std::marker::PhantomData;

use cugraph_sys::{
    cugraph_edgelist_get_destinations, cugraph_edgelist_get_sources, cugraph_edgelist_t,
    cugraph_type_erased_device_array_view_t,
};

use crate::{
    array::RawDeviceArrayView,
    cuda::{DeviceResourceKind, drop_device_resource},
    error::{Error, ErrorKind, Result, require_non_null},
    handle::{EnteredContext, ResourceHandleLease},
    structure::VertexIds,
    types::GraphProperties,
};

pub(crate) struct DecompressedEdgelist {
    raw: *mut cugraph_edgelist_t,
    device: i32,
    // Keeps a runtime-bound cuGraph handle alive until after Drop frees `raw`.
    _resource_handle_lease: Option<ResourceHandleLease>,
    _nosend_nosync: PhantomData<*mut ()>,
}

impl DecompressedEdgelist {
    pub(super) unsafe fn from_raw(
        raw: *mut cugraph_edgelist_t,
        device: i32,
        resource_handle_lease: Option<ResourceHandleLease>,
    ) -> Result<Self> {
        let raw = require_non_null(raw, "DecompressedEdgelist::from_raw", "edgelist")?;
        Ok(Self { raw, device, _resource_handle_lease: resource_handle_lease, _nosend_nosync: PhantomData })
    }

    fn raw_view(
        &self,
        getter: unsafe extern "C" fn(*mut cugraph_edgelist_t) -> *mut cugraph_type_erased_device_array_view_t,
        operation: &'static str,
        pointer_name: &'static str,
    ) -> Result<RawDeviceArrayView<'_>> {
        let view_ptr = require_non_null(
            // SAFETY: `self.raw` is a live decompressed edgelist owned by this
            // wrapper for the duration of the call.
            unsafe { getter(self.raw) },
            operation,
            pointer_name,
        )?;

        // SAFETY: cuGraph allocates the view metadata; this wrapper frees it while the edgelist retains the backing arrays.
        unsafe { RawDeviceArrayView::from_owned_view(view_ptr, Some(self.device), operation, pointer_name) }
    }

    pub(super) fn edge_count(
        &self,
        handle: &EnteredContext<'_>,
        properties: GraphProperties,
        operation: &'static str,
    ) -> Result<usize> {
        let source_len = self.raw_view(cugraph_edgelist_get_sources, operation, "sources")?.len();
        let destination_len = self.raw_view(cugraph_edgelist_get_destinations, operation, "destinations")?.len();

        if source_len != destination_len {
            return Err(Error::new(ErrorKind::InvalidState, operation)
                .with_message("decompressed edgelist returned mismatched source/destination lengths")
                .with_context("source_len", source_len.to_string())
                .with_context("destination_len", destination_len.to_string()));
        }

        if properties.is_multigraph && !properties.is_symmetric {
            return Ok(source_len);
        }

        let sources = VertexIds::from_raw(self.raw_view(cugraph_edgelist_get_sources, operation, "sources")?)?;
        let destinations =
            VertexIds::from_raw(self.raw_view(cugraph_edgelist_get_destinations, operation, "destinations")?)?;

        match (sources, destinations) {
            (VertexIds::I32(src), VertexIds::I32(dst)) => {
                let src = src.to_vec(handle)?;
                let dst = dst.to_vec(handle)?;
                logical_edge_count_from_host(src, dst, properties, operation)
            },
            (VertexIds::I64(src), VertexIds::I64(dst)) => {
                let src = src.to_vec(handle)?;
                let dst = dst.to_vec(handle)?;
                logical_edge_count_from_host(src, dst, properties, operation)
            },
            (VertexIds::I32(_), VertexIds::I64(_)) | (VertexIds::I64(_), VertexIds::I32(_)) => {
                Err(Error::new(ErrorKind::InvalidState, operation)
                    .with_message("decompressed edgelist source/destination vertex dtypes differ"))
            },
        }
    }
}

fn logical_edge_count_from_host<T>(
    src: Vec<T>,
    dst: Vec<T>,
    properties: GraphProperties,
    operation: &'static str,
) -> Result<usize>
where
    T: Ord + Copy + std::fmt::Debug,
{
    debug_assert_eq!(src.len(), dst.len(), "source/destination vectors should already be length-checked");

    let mut edges = src
        .into_iter()
        .zip(dst)
        .map(|(src, dst)| if properties.is_symmetric && dst < src { (dst, src) } else { (src, dst) })
        .collect::<Vec<_>>();
    edges.sort_unstable();

    if !properties.is_multigraph {
        edges.dedup();
        return Ok(edges.len());
    }

    if !properties.is_symmetric {
        return Ok(edges.len());
    }

    let mut logical_edge_count = 0usize;
    let mut index = 0usize;
    while index < edges.len() {
        let edge = edges[index];
        let mut run_len = 1usize;
        while index + run_len < edges.len() && edges[index + run_len] == edge {
            run_len += 1;
        }

        if edge.0 == edge.1 {
            logical_edge_count += run_len;
        } else if !run_len.is_multiple_of(2) {
            return Err(Error::new(ErrorKind::InvalidState, operation)
                .with_message(
                    "undirected multigraph decompressed edgelist contained an odd count for a non-self-loop edge",
                )
                .with_context("edge", format!("{edge:?}"))
                .with_context("run_len", run_len.to_string()));
        } else {
            logical_edge_count += run_len / 2;
        }

        index += run_len;
    }

    Ok(logical_edge_count)
}

impl Drop for DecompressedEdgelist {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let mut free = || {
                drop_device_resource(
                    &mut self.raw,
                    self.device,
                    "DecompressedEdgelist::drop",
                    DeviceResourceKind::Edgelist,
                );
            };
            if let Some(lease) = self._resource_handle_lease.as_ref() {
                // Teardown recovers exclusive access even after a use-path
                // panic poisoned the stable gate; normal entry stays closed.
                lease.with_teardown_gate(free);
            } else {
                free();
            }
        }
    }
}
