//! Shared traversal path result ownership and path extraction.

use std::{marker::PhantomData, ptr};

use cugraph_sys::{
    cugraph_error_t, cugraph_extract_paths, cugraph_extract_paths_result_free,
    cugraph_extract_paths_result_get_max_path_length, cugraph_extract_paths_result_get_paths,
    cugraph_extract_paths_result_t, cugraph_paths_result_free, cugraph_paths_result_get_distances,
    cugraph_paths_result_get_predecessors, cugraph_paths_result_get_vertices, cugraph_paths_result_t,
    cugraph_type_erased_device_array_view_t,
};

use crate::{
    array::{DeviceArray, RawDeviceArrayView},
    cuda::DeviceGuard,
    error::{Error, ErrorKind, Result, check_error, require_non_null},
    graph::{Forward, Graph},
    handle::EnteredContext,
    structure::{
        VertexSelection,
        typed_results::{Distances, VertexIds},
    },
    types::CugraphElement,
};

pub(crate) struct PathsResult {
    raw: *mut cugraph_paths_result_t,
    device: i32,
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
    ) -> Result<Self> {
        let raw = require_non_null(raw, operation, "paths_result")?;
        Ok(Self { raw, device, _nosend_nosync: PhantomData })
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

        // SAFETY: the returned view borrows data owned by `self.raw`, which
        // outlives the returned wrapper. This preserves the existing BFS
        // result-accessor behavior.
        unsafe { RawDeviceArrayView::from_borrowed(view_ptr, Some(self.device), operation, pointer_name) }
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

        // SAFETY: same borrowing contract as `raw_view`.
        let raw = unsafe { RawDeviceArrayView::from_borrowed(view_ptr, Some(self.device), operation, pointer_name)? };
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
        if !self.raw.is_null() {
            let _guard = DeviceGuard::switch_to_for_drop(self.device, "PathsResult::drop").ok();
            // SAFETY: `self.raw` is owned by this wrapper and has not been
            // freed yet.
            unsafe { cugraph_paths_result_free(self.raw) };
        }
    }
}

/// Result of extracting concrete paths from a BFS or SSSP paths result.
pub struct ExtractPathsResult {
    storage: ExtractPathsStorage,
    _nosend_nosync: PhantomData<*mut ()>,
}

enum ExtractPathsStorage {
    Native { raw: *mut cugraph_extract_paths_result_t, device: i32 },
    I32 { paths: DeviceArray<i32>, max_path_length: usize },
    I64 { paths: DeviceArray<i64>, max_path_length: usize },
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
    ) -> Result<Self> {
        let raw = require_non_null(raw, operation, "extract_paths_result")?;
        Ok(Self { storage: ExtractPathsStorage::Native { raw, device }, _nosend_nosync: PhantomData })
    }

    pub(crate) fn from_i32_paths(paths: DeviceArray<i32>, max_path_length: usize) -> Self {
        Self { storage: ExtractPathsStorage::I32 { paths, max_path_length }, _nosend_nosync: PhantomData }
    }

    pub(crate) fn from_i64_paths(paths: DeviceArray<i64>, max_path_length: usize) -> Self {
        Self { storage: ExtractPathsStorage::I64 { paths, max_path_length }, _nosend_nosync: PhantomData }
    }

    /// Maximum row length in the extracted row-major path matrix.
    pub fn max_path_length(&self) -> usize {
        match &self.storage {
            ExtractPathsStorage::Native { raw, .. } => {
                // SAFETY: `raw` is a live extract-paths result owned by this
                // wrapper for the duration of the call.
                unsafe { cugraph_extract_paths_result_get_max_path_length(*raw) }
            },
            ExtractPathsStorage::I32 { max_path_length, .. } | ExtractPathsStorage::I64 { max_path_length, .. } => {
                *max_path_length
            },
        }
    }

    /// Row-major path matrix as vertex ids.
    ///
    /// The array contains `destination_count * max_path_length()` elements.
    pub fn paths(&self) -> Result<VertexIds<'_>> {
        match &self.storage {
            ExtractPathsStorage::Native { raw, device } => {
                let view_ptr = require_non_null(
                    // SAFETY: `raw` is a live extract-paths result owned by
                    // this wrapper for the duration of the call.
                    unsafe { cugraph_extract_paths_result_get_paths(*raw) },
                    "ExtractPathsResult::paths",
                    "paths",
                )?;
                // SAFETY: cuGraph transfers ownership of the view metadata
                // returned by the result accessor. The backing allocation
                // remains owned by `raw`, which outlives the returned view.
                let raw = unsafe {
                    RawDeviceArrayView::from_owned_view(view_ptr, Some(*device), "ExtractPathsResult::paths", "paths")
                }?;
                VertexIds::from_raw(raw)
            },
            ExtractPathsStorage::I32 { paths, .. } => Ok(VertexIds::I32(paths.view()?)),
            ExtractPathsStorage::I64 { paths, .. } => Ok(VertexIds::I64(paths.view()?)),
        }
    }
}

impl Drop for ExtractPathsResult {
    fn drop(&mut self) {
        if let ExtractPathsStorage::Native { raw, device } = &mut self.storage {
            if !raw.is_null() {
                let _guard = DeviceGuard::switch_to_for_drop(*device, "ExtractPathsResult::drop").ok();
                // SAFETY: `raw` is owned by this wrapper and has not been
                // freed yet.
                unsafe { cugraph_extract_paths_result_free(*raw) };
            }
        }
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

    let destinations = destinations.into();
    graph.validate_vertex_selection(&destinations, operation, "destinations", false)?;
    let destinations = destinations.resolve(handle)?;
    let destinations_view = graph.resolved_vertex_view(&destinations, operation, "destinations")?;

    let mut result: *mut cugraph_extract_paths_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();

    // SAFETY: `graph`, `paths`, `sources_view`, and `destinations_view` stay
    // valid for the duration of the call, and `result`/`error` are valid
    // out-pointers.
    unsafe {
        let code = cugraph_extract_paths(
            handle.handle_ptr(),
            graph.as_ptr(),
            sources_view.as_ptr(),
            paths.raw,
            destinations_view.as_ptr(),
            &mut result,
            &mut error,
        );
        check_error(code, error, operation)?;
    }

    // SAFETY: on success, cuGraph transfers ownership of the result pointer to
    // the caller.
    unsafe { ExtractPathsResult::from_raw(result, handle.device(), operation) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_result_from_raw_rejects_null() {
        let err = unsafe { PathsResult::from_raw(std::ptr::null_mut(), 0, "test_paths_result") }
            .expect_err("null paths result should fail");
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert_eq!(err.operation(), "test_paths_result");
    }

    #[test]
    fn extract_paths_result_from_raw_rejects_null() {
        let err = unsafe { ExtractPathsResult::from_raw(std::ptr::null_mut(), 0, "test_extract_paths_result") }
            .expect_err("null extract paths result should fail");
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert_eq!(err.operation(), "test_extract_paths_result");
    }
}
