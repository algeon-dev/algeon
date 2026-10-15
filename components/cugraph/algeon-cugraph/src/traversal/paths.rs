//! Shared traversal path result ownership and path extraction.

use std::{marker::PhantomData, ptr};

use cugraph_sys::{
    cugraph_error_t, cugraph_extract_paths, cugraph_extract_paths_result_get_max_path_length,
    cugraph_extract_paths_result_get_paths, cugraph_extract_paths_result_t, cugraph_paths_result_get_distances,
    cugraph_paths_result_get_predecessors, cugraph_paths_result_get_vertices, cugraph_paths_result_t,
    cugraph_type_erased_device_array_view_t,
};

use crate::{
    array::RawDeviceArrayView,
    cuda::{DeviceResourceKind, drop_device_resource},
    error::{Error, ErrorKind, Result, check_error, require_non_null},
    graph::{Forward, Graph},
    handle::{EnteredContext, ResourceHandleLease},
    structure::{
        VertexSelection,
        typed_results::{Distances, VertexIds},
    },
    types::CugraphElement,
};

pub(crate) struct PathsResult {
    raw: *mut cugraph_paths_result_t,
    device: i32,
    // Keeps a runtime-bound cuGraph handle alive until after Drop frees `raw`.
    _resource_handle_lease: Option<ResourceHandleLease>,
    _nosend_nosync: PhantomData<*mut ()>,
}

impl std::fmt::Debug for PathsResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PathsResult").finish_non_exhaustive()
    }
}

impl PathsResult {
    pub(crate) unsafe fn from_raw(
        raw: *mut cugraph_paths_result_t,
        device: i32,
        operation: &'static str,
        resource_handle_lease: Option<ResourceHandleLease>,
    ) -> Result<Self> {
        let raw = require_non_null(raw, operation, "paths_result")?;
        Ok(Self { raw, device, _resource_handle_lease: resource_handle_lease, _nosend_nosync: PhantomData })
    }

    fn raw_view(
        &self,
        getter: unsafe extern "C" fn(*mut cugraph_paths_result_t) -> *mut cugraph_type_erased_device_array_view_t,
        operation: &'static str,
        pointer_name: &'static str,
    ) -> Result<RawDeviceArrayView<'_>> {
        let view_ptr = require_non_null(
            // SAFETY: `self.raw` is a live paths result owned by this wrapper
            // for the duration of the call.
            unsafe { getter(self.raw) },
            operation,
            pointer_name,
        )?;

        // SAFETY: cuGraph allocates view metadata; this wrapper frees it while `self.raw` retains the backing arrays.
        unsafe { RawDeviceArrayView::from_owned_view(view_ptr, Some(self.device), operation, pointer_name) }
    }

    fn optional_raw_view(
        &self,
        getter: unsafe extern "C" fn(*mut cugraph_paths_result_t) -> *mut cugraph_type_erased_device_array_view_t,
        operation: &'static str,
        pointer_name: &'static str,
    ) -> Result<Option<RawDeviceArrayView<'_>>> {
        // SAFETY: `self.raw` is a live paths result owned by this wrapper for
        // the duration of the call.
        let view_ptr = unsafe { getter(self.raw) };
        if view_ptr.is_null() {
            return Ok(None);
        }

        // SAFETY: cuGraph allocates view metadata; this wrapper frees it while `self.raw` retains the backing arrays.
        let raw = unsafe { RawDeviceArrayView::from_owned_view(view_ptr, Some(self.device), operation, pointer_name)? };
        if raw.is_empty() { Ok(None) } else { Ok(Some(raw)) }
    }

    pub(crate) fn vertex_ids(&self, operation: &'static str) -> Result<VertexIds<'_>> {
        VertexIds::from_raw(self.raw_view(cugraph_paths_result_get_vertices, operation, "vertices")?)
    }

    pub(crate) fn distances(&self, operation: &'static str) -> Result<Distances<'_>> {
        Distances::from_raw(self.raw_view(cugraph_paths_result_get_distances, operation, "distances")?)
    }

    pub(crate) fn predecessors(&self, operation: &'static str) -> Result<Option<VertexIds<'_>>> {
        match self.optional_raw_view(cugraph_paths_result_get_predecessors, operation, "predecessors")? {
            Some(raw) => VertexIds::from_raw(raw).map(Some),
            None => Ok(None),
        }
    }

    fn ensure_predecessors(&self, operation: &'static str) -> Result<()> {
        if self.predecessors(operation)?.is_some() {
            return Ok(());
        }

        Err(Error::new(ErrorKind::InvalidInput, operation).with_message(
            "paths result has no predecessor column; enable predecessor computation before extracting paths",
        ))
    }
}

impl Drop for PathsResult {
    fn drop(&mut self) {
        drop_device_resource(&mut self.raw, self.device, "PathsResult::drop", DeviceResourceKind::PathsResult);
    }
}

/// Result of extracting concrete paths from a BFS or SSSP paths result.
pub struct ExtractPathsResult {
    raw: *mut cugraph_extract_paths_result_t,
    device: i32,
    // Keeps a runtime-bound cuGraph handle alive until after Drop frees native storage.
    _resource_handle_lease: Option<ResourceHandleLease>,
    _nosend_nosync: PhantomData<*mut ()>,
}

impl std::fmt::Debug for ExtractPathsResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExtractPathsResult").finish_non_exhaustive()
    }
}

impl ExtractPathsResult {
    pub(crate) unsafe fn from_raw(
        raw: *mut cugraph_extract_paths_result_t,
        device: i32,
        operation: &'static str,
        resource_handle_lease: Option<ResourceHandleLease>,
    ) -> Result<Self> {
        let raw = require_non_null(raw, operation, "extract_paths_result")?;
        Ok(Self { raw, device, _resource_handle_lease: resource_handle_lease, _nosend_nosync: PhantomData })
    }

    /// Maximum row length in the extracted row-major path matrix.
    pub fn max_path_length(&self) -> usize {
        // SAFETY: `self.raw` is a live extract-paths result owned by this wrapper.
        unsafe { cugraph_extract_paths_result_get_max_path_length(self.raw) }
    }

    /// Row-major path matrix as vertex ids.
    ///
    /// The array contains `destination_count * max_path_length()` elements.
    pub fn paths(&self) -> Result<VertexIds<'_>> {
        let view_ptr = require_non_null(
            // SAFETY: `self.raw` is a live extract-paths result owned by this wrapper.
            unsafe { cugraph_extract_paths_result_get_paths(self.raw) },
            "ExtractPathsResult::paths",
            "paths",
        )?;
        // SAFETY: cuGraph transfers the view metadata. This owner retains its backing array.
        let raw = unsafe {
            RawDeviceArrayView::from_owned_view(view_ptr, Some(self.device), "ExtractPathsResult::paths", "paths")
        }?;
        VertexIds::from_raw(raw)
    }
}

impl Drop for ExtractPathsResult {
    fn drop(&mut self) {
        drop_device_resource(
            &mut self.raw,
            self.device,
            "ExtractPathsResult::drop",
            DeviceResourceKind::ExtractPathsResult,
        );
    }
}

pub(crate) fn extract_paths<'a, V: CugraphElement + 'a>(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<Forward>,
    paths: &PathsResult,
    sources: impl Into<VertexSelection<'a, V>>,
    destinations: impl Into<VertexSelection<'a, V>>,
    operation: &'static str,
) -> Result<ExtractPathsResult> {
    paths.ensure_predecessors(operation)?;

    let sources = sources.into();
    graph.validate_vertex_selection(&sources, operation, "sources", false)?;
    let sources = sources.resolve(handle)?;
    let sources_view = graph.resolved_vertex_view(&sources, operation, "sources")?;
    let weighted_paths = matches!(paths.distances(operation)?, Distances::F32(_) | Distances::F64(_));
    if weighted_paths && sources_view.len() != 1 {
        return Err(Error::new(ErrorKind::InvalidInput, operation)
            .with_message("SSSP path extraction requires exactly one source vertex")
            .with_context("source_count", sources_view.len().to_string()));
    }

    let destinations = destinations.into();
    graph.validate_vertex_selection(&destinations, operation, "destinations", false)?;
    let destinations = destinations.resolve(handle)?;
    let destinations_view = graph.resolved_vertex_view(&destinations, operation, "destinations")?;
    if weighted_paths {
        let vertex_count = paths.vertex_ids(operation)?.len();
        let predecessor_count = paths.predecessors(operation)?.map_or(0, |predecessors| predecessors.len());
        if vertex_count != predecessor_count {
            return Err(Error::new(ErrorKind::InvalidState, operation)
                .with_message("SSSP result returned mismatched vertex and predecessor lengths")
                .with_context("vertex_count", vertex_count.to_string())
                .with_context("predecessor_count", predecessor_count.to_string()));
        }
    }

    let mut result: *mut cugraph_extract_paths_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();
    let graph_ptr = graph.as_ptr(handle, operation)?;

    // SAFETY: `graph`, `paths`, `sources_view`, and `destinations_view` stay
    // valid for the duration of the call, and `result`/`error` are valid
    // out-pointers.
    unsafe {
        let code = cugraph_extract_paths(
            handle.handle_ptr(),
            graph_ptr,
            sources_view.as_ptr(),
            paths.raw,
            destinations_view.as_ptr(),
            &mut result,
            &mut error,
        );
        check_error(code, error, handle.device(), operation)?;
    }

    // SAFETY: on success, cuGraph transfers ownership of the result pointer to
    // the caller.
    unsafe { ExtractPathsResult::from_raw(result, handle.device(), operation, handle.runtime_bound_lease()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_result_from_raw_rejects_null() {
        // SAFETY: passing a null pointer is an intentional invalid input; the
        // function's null check rejects it before any dereference happens.
        let err = unsafe { PathsResult::from_raw(std::ptr::null_mut(), 0, "test_paths_result", None) }
            .expect_err("null paths result should fail");
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert_eq!(err.operation(), "test_paths_result");
    }

    #[test]
    fn extract_paths_result_from_raw_rejects_null() {
        // SAFETY: passing a null pointer is an intentional invalid input; the
        // function's null check rejects it before any dereference happens.
        let err = unsafe { ExtractPathsResult::from_raw(std::ptr::null_mut(), 0, "test_extract_paths_result", None) }
            .expect_err("null extract paths result should fail");
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert_eq!(err.operation(), "test_extract_paths_result");
    }
}
