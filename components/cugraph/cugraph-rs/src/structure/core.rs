//! Core-number algorithms.

use std::{marker::PhantomData, num::NonZeroUsize, ptr};

use cugraph_sys::{
    cugraph_core_number, cugraph_core_result_free, cugraph_core_result_get_core_numbers,
    cugraph_core_result_get_vertices, cugraph_core_result_t, cugraph_error_t, cugraph_k_core,
    cugraph_k_core_degree_type_t, cugraph_k_core_degree_type_t_K_CORE_DEGREE_TYPE_IN,
    cugraph_k_core_degree_type_t_K_CORE_DEGREE_TYPE_INOUT, cugraph_k_core_degree_type_t_K_CORE_DEGREE_TYPE_OUT,
    cugraph_k_core_result_free, cugraph_k_core_result_get_dst_vertices, cugraph_k_core_result_get_src_vertices,
    cugraph_k_core_result_get_weights, cugraph_k_core_result_t, cugraph_type_erased_device_array_view_t,
};

use crate::{
    array::RawDeviceArrayView,
    cuda::DeviceGuard,
    error::{Error, Result, check_error, require_non_null},
    graph::Graph,
    handle::EnteredContext,
    structure::typed_results::{IntegerValues, ScoreValues, VertexIds},
    types::to_bool_t,
};

/// Degree direction used when computing core numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum KCoreDegreeType {
    /// Compute core numbers from incoming edges.
    In,
    /// Compute core numbers from outgoing edges.
    Out,
    /// Compute core numbers from both incoming and outgoing edges.
    #[default]
    InOut,
}

impl From<KCoreDegreeType> for cugraph_k_core_degree_type_t {
    fn from(value: KCoreDegreeType) -> Self {
        match value {
            KCoreDegreeType::In => cugraph_k_core_degree_type_t_K_CORE_DEGREE_TYPE_IN,
            KCoreDegreeType::Out => cugraph_k_core_degree_type_t_K_CORE_DEGREE_TYPE_OUT,
            KCoreDegreeType::InOut => cugraph_k_core_degree_type_t_K_CORE_DEGREE_TYPE_INOUT,
        }
    }
}

/// Parameters for [`Graph::core_number`](crate::graph::Graph::core_number).
#[derive(Debug, Clone, Copy, Default)]
#[must_use]
pub struct CoreNumberParams {
    /// Which edge direction to use when computing core numbers.
    pub degree_type: KCoreDegreeType,
    /// Whether to run expensive input validation in the C library.
    pub do_expensive_check: bool,
}

/// Parameters for [`Graph::k_core`](crate::graph::Graph::k_core).
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct KCoreParams {
    /// K value for the extracted core. Defaults to 2.
    pub k: NonZeroUsize,
    /// Degree direction to use if no precomputed core-number result is supplied.
    pub degree_type: KCoreDegreeType,
    /// Whether to run expensive input validation in the C library.
    pub do_expensive_check: bool,
}

impl Default for KCoreParams {
    fn default() -> Self {
        Self {
            // SAFETY: the literal `2` is non-zero.
            k: unsafe { NonZeroUsize::new_unchecked(2) },
            degree_type: KCoreDegreeType::default(),
            do_expensive_check: false,
        }
    }
}

/// Compute core numbers for all vertices in a graph.
///
/// Prefer using [`Graph::core_number`](crate::graph::Graph::core_number)
/// instead of calling this function directly.
pub(crate) fn core_number<T>(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<T>,
    params: &CoreNumberParams,
) -> Result<CoreNumberResult> {
    let mut result: *mut cugraph_core_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();

    // SAFETY: `graph` and `handle` are live for the duration of the call, and
    // `result`/`error` are valid out-pointers.
    unsafe {
        let code = cugraph_core_number(
            handle.handle_ptr(),
            graph.as_ptr(),
            params.degree_type.into(),
            to_bool_t(params.do_expensive_check),
            &mut result,
            &mut error,
        );
        check_error(code, error, "core_number")?;
    }

    let result = require_non_null(result, "core_number", "core_result")?;

    // SAFETY: on success, cuGraph transfers ownership of the result pointer.
    unsafe { CoreNumberResult::from_raw(result, handle.device()) }
}

/// Extract a k-core subgraph.
pub(crate) fn k_core<T>(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<T>,
    params: &KCoreParams,
    core_numbers: Option<&CoreNumberResult>,
) -> Result<KCoreResult> {
    let mut result: *mut cugraph_k_core_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();
    let core_result_ptr = core_numbers.map(CoreNumberResult::as_ptr).unwrap_or(ptr::null());

    // SAFETY: `graph` and optional `core_result_ptr` are live for the call,
    // and `result`/`error` are valid out-pointers.
    unsafe {
        let code = cugraph_k_core(
            handle.handle_ptr(),
            graph.as_ptr(),
            params.k.get(),
            params.degree_type.into(),
            core_result_ptr,
            to_bool_t(params.do_expensive_check),
            &mut result,
            &mut error,
        );
        check_error(code, error, "k_core")?;
    }

    let result = require_non_null(result, "k_core", "k_core_result")?;

    // SAFETY: on success, cuGraph transfers ownership of the result pointer.
    unsafe { KCoreResult::from_raw(result, handle.device()) }
}

/// Result of a core-number computation.
///
/// Provides vertex ids and integer core numbers. Freed on drop via
/// `cugraph_core_result_free`.
pub struct CoreNumberResult {
    raw: *mut cugraph_core_result_t,
    device: i32,
    _nosend_nosync: PhantomData<*mut ()>,
}

impl std::fmt::Debug for CoreNumberResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoreNumberResult").finish_non_exhaustive()
    }
}

impl CoreNumberResult {
    /// Wrap a raw core-number result pointer.
    ///
    /// # Safety
    ///
    /// `raw` must be a valid, non-null pointer returned by
    /// `cugraph_core_number`. Ownership is transferred to this struct.
    pub(crate) unsafe fn from_raw(raw: *mut cugraph_core_result_t, device: i32) -> Result<Self> {
        let raw = require_non_null(raw, "CoreNumberResult::from_raw", "core_result")?;
        Ok(Self { raw, device, _nosend_nosync: PhantomData })
    }

    pub(crate) fn as_ptr(&self) -> *const cugraph_core_result_t {
        self.raw.cast_const()
    }

    fn raw_view(
        &self,
        getter: unsafe extern "C" fn(
            *mut cugraph_core_result_t,
        ) -> *mut cugraph_sys::cugraph_type_erased_device_array_view_t,
        operation: &'static str,
        pointer_name: &'static str,
    ) -> Result<RawDeviceArrayView<'_>> {
        let view_ptr = require_non_null(
            // SAFETY: `self.raw` is a live core result owned by this wrapper.
            unsafe { getter(self.raw) },
            operation,
            pointer_name,
        )?;
        // SAFETY: cuGraph result getters allocate view metadata for the
        // caller. The backing device allocation remains owned by `self.raw`.
        unsafe { RawDeviceArrayView::from_owned_view(view_ptr, Some(self.device), operation, pointer_name) }
    }

    /// Get the vertex ids as a runtime-typed device array view.
    pub fn vertex_ids(&self) -> Result<VertexIds<'_>> {
        VertexIds::from_raw(self.raw_view(
            cugraph_core_result_get_vertices,
            "CoreNumberResult::vertex_ids",
            "vertices",
        )?)
    }

    /// Get the core numbers as a runtime-typed integer device array view.
    pub fn core_numbers(&self) -> Result<IntegerValues<'_>> {
        IntegerValues::from_raw(self.raw_view(
            cugraph_core_result_get_core_numbers,
            "CoreNumberResult::core_numbers",
            "core_numbers",
        )?)
    }

    /// Convert this core-number result to a cudf DataFrame.
    ///
    /// Returns columns `"vertex"` and `"core_number"`.
    pub fn to_dataframe(&self, runtime: &cudf::Runtime) -> Result<cudf::DataFrame> {
        crate::cudf_interop::core_number_result_to_dataframe(self, runtime)
            .map_err(|e: Error| e.with_operation("CoreNumberResult::to_dataframe"))
    }
}

impl Drop for CoreNumberResult {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _guard = DeviceGuard::switch_to_for_drop(self.device, "CoreNumberResult::drop").ok();
            // SAFETY: `self.raw` is owned by this wrapper and freed exactly once.
            unsafe { cugraph_core_result_free(self.raw) };
        }
    }
}

/// Result of a k-core extraction.
///
/// Provides source and destination vertex ids plus optional edge weights.
/// Freed on drop via `cugraph_k_core_result_free`.
pub struct KCoreResult {
    raw: *mut cugraph_k_core_result_t,
    device: i32,
    _nosend_nosync: PhantomData<*mut ()>,
}

impl std::fmt::Debug for KCoreResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KCoreResult").finish_non_exhaustive()
    }
}

impl KCoreResult {
    /// Wrap a raw k-core result pointer.
    ///
    /// # Safety
    ///
    /// `raw` must be a valid, non-null pointer returned by `cugraph_k_core`.
    /// Ownership is transferred to this struct.
    pub(crate) unsafe fn from_raw(raw: *mut cugraph_k_core_result_t, device: i32) -> Result<Self> {
        let raw = require_non_null(raw, "KCoreResult::from_raw", "k_core_result")?;
        Ok(Self { raw, device, _nosend_nosync: PhantomData })
    }

    fn raw_view(
        &self,
        getter: unsafe extern "C" fn(*mut cugraph_k_core_result_t) -> *mut cugraph_type_erased_device_array_view_t,
        operation: &'static str,
        pointer_name: &'static str,
    ) -> Result<RawDeviceArrayView<'_>> {
        let view_ptr = require_non_null(
            // SAFETY: `self.raw` is a live k-core result owned by this wrapper.
            unsafe { getter(self.raw) },
            operation,
            pointer_name,
        )?;
        // SAFETY: `view_ptr` is a borrowed child view tied to the live result.
        unsafe { RawDeviceArrayView::from_borrowed(view_ptr, Some(self.device), operation, pointer_name) }
    }

    fn optional_raw_view(
        &self,
        getter: unsafe extern "C" fn(*mut cugraph_k_core_result_t) -> *mut cugraph_type_erased_device_array_view_t,
        operation: &'static str,
    ) -> Result<Option<RawDeviceArrayView<'_>>> {
        // SAFETY: `self.raw` is a live k-core result owned by this wrapper.
        let view_ptr = unsafe { getter(self.raw) };
        if view_ptr.is_null() {
            return Ok(None);
        }

        // SAFETY: cuGraph result getters allocate view metadata for the
        // caller. The backing device allocation remains owned by `self.raw`.
        unsafe { RawDeviceArrayView::from_owned_view(view_ptr, Some(self.device), operation, "optional_view") }
            .map(Some)
    }

    /// Get source vertex ids.
    pub fn sources(&self) -> Result<VertexIds<'_>> {
        VertexIds::from_raw(self.raw_view(cugraph_k_core_result_get_src_vertices, "KCoreResult::sources", "sources")?)
    }

    /// Get destination vertex ids.
    pub fn destinations(&self) -> Result<VertexIds<'_>> {
        VertexIds::from_raw(self.raw_view(
            cugraph_k_core_result_get_dst_vertices,
            "KCoreResult::destinations",
            "destinations",
        )?)
    }

    /// Get optional edge weights.
    pub fn weights(&self) -> Result<Option<ScoreValues<'_>>> {
        self.optional_raw_view(cugraph_k_core_result_get_weights, "KCoreResult::weights")?
            .map(ScoreValues::from_raw)
            .transpose()
    }

    /// Convert this result to a cudf DataFrame.
    ///
    /// Returns `"source"` and `"destination"` columns, plus `"weight"` when
    /// the native result contains an edge-weight column.
    pub fn to_dataframe(&self, runtime: &cudf::Runtime) -> Result<cudf::DataFrame> {
        crate::cudf_interop::k_core_result_to_dataframe(self, runtime)
            .map_err(|e: Error| e.with_operation("KCoreResult::to_dataframe"))
    }
}

impl Drop for KCoreResult {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _guard = DeviceGuard::switch_to_for_drop(self.device, "KCoreResult::drop").ok();
            // SAFETY: `self.raw` is owned by this wrapper and freed exactly once.
            unsafe { cugraph_k_core_result_free(self.raw) };
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use super::*;
    use crate::error::ErrorKind;

    #[test]
    fn core_number_result_from_raw_rejects_null() {
        // SAFETY: this exercises the constructor's null guard only.
        let err = match unsafe { CoreNumberResult::from_raw(ptr::null_mut(), 0) } {
            Err(err) => err,
            Ok(_) => panic!("null must be rejected"),
        };
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert_eq!(err.operation(), "CoreNumberResult::from_raw");
    }

    #[test]
    fn k_core_result_from_raw_rejects_null() {
        // SAFETY: this exercises the constructor's null guard only.
        let err = match unsafe { KCoreResult::from_raw(ptr::null_mut(), 0) } {
            Err(err) => err,
            Ok(_) => panic!("null must be rejected"),
        };
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert_eq!(err.operation(), "KCoreResult::from_raw");
    }
}
