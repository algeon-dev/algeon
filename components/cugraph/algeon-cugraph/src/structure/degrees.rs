//! Degree computation for graph vertices.
//!
//! Computes in-degrees, out-degrees, or both for all (or a subset of) vertices
//! in a graph.

use std::{marker::PhantomData, ptr};

use cugraph_sys::{
    cugraph_degrees, cugraph_degrees_result_get_in_degrees, cugraph_degrees_result_get_out_degrees,
    cugraph_degrees_result_get_vertices, cugraph_degrees_result_t, cugraph_error_t, cugraph_in_degrees,
    cugraph_out_degrees, cugraph_type_erased_device_array_view_t,
};

use crate::{
    array::{DeviceArrayView, RawDeviceArrayView},
    cuda::{DeviceResourceKind, drop_device_resource},
    cudf_interop::{degrees_result_to_dataframe, in_degrees_result_to_dataframe, out_degrees_result_to_dataframe},
    error::{Error, ErrorKind, Result, check_error, require_non_null},
    graph::{Graph, Transposed},
    handle::{EnteredContext, ResourceHandleLease},
    structure::typed_results::VertexIds,
    types::{CugraphElement, DataType, to_bool_t},
};

/// Compute both in-degrees and out-degrees for vertices in a graph.
///
/// Prefer using [`Graph::degrees`](crate::graph::Graph::degrees) instead
/// of calling this function directly.
pub(crate) fn degrees<V: CugraphElement>(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<Transposed>,
    vertices: Option<&DeviceArrayView<'_, V>>,
    do_expensive_check: bool,
) -> Result<DegreesResult> {
    let mut result: *mut cugraph_degrees_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();

    let vertices_ptr = vertices.map(|v| v.as_ptr() as *const _).unwrap_or(ptr::null());
    let graph_ptr = graph.as_ptr(handle, "degrees")?;

    // SAFETY: `graph` and `handle` outlive the call, `vertices_ptr` is either
    // null or borrowed from a live device view, and `result`/`error` are valid
    // output locations.
    unsafe {
        let code = cugraph_degrees(
            handle.handle_ptr(),
            graph_ptr,
            vertices_ptr,
            to_bool_t(do_expensive_check),
            &mut result,
            &mut error,
        );
        check_error(code, error, handle.device(), "degrees")?;
    }

    DegreesResult::from_raw(result, handle.device(), handle.runtime_bound_lease())
}

/// Compute only in-degrees for vertices in a graph.
///
/// Prefer using [`Graph::in_degrees`](crate::graph::Graph::in_degrees)
/// instead of calling this function directly.
pub(crate) fn in_degrees<V: CugraphElement>(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<Transposed>,
    vertices: Option<&DeviceArrayView<'_, V>>,
    do_expensive_check: bool,
) -> Result<InDegreesResult> {
    let mut result: *mut cugraph_degrees_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();

    let vertices_ptr = vertices.map(|v| v.as_ptr() as *const _).unwrap_or(ptr::null());
    let graph_ptr = graph.as_ptr(handle, "in_degrees")?;

    // SAFETY: `graph` and `handle` outlive the call, `vertices_ptr` is either
    // null or borrowed from a live device view, and `result`/`error` are valid
    // output locations.
    unsafe {
        let code = cugraph_in_degrees(
            handle.handle_ptr(),
            graph_ptr,
            vertices_ptr,
            to_bool_t(do_expensive_check),
            &mut result,
            &mut error,
        );
        check_error(code, error, handle.device(), "in_degrees")?;
    }

    InDegreesResult::from_raw(result, handle.device(), handle.runtime_bound_lease())
}

/// Compute only out-degrees for vertices in a graph.
///
/// Prefer using [`Graph::out_degrees`](crate::graph::Graph::out_degrees)
/// instead of calling this function directly.
pub(crate) fn out_degrees<T, V: CugraphElement>(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<T>,
    vertices: Option<&DeviceArrayView<'_, V>>,
    do_expensive_check: bool,
) -> Result<OutDegreesResult> {
    let mut result: *mut cugraph_degrees_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();

    let vertices_ptr = vertices.map(|v| v.as_ptr() as *const _).unwrap_or(ptr::null());
    let graph_ptr = graph.as_ptr(handle, "out_degrees")?;

    // SAFETY: `graph` and `handle` outlive the call, `vertices_ptr` is either
    // null or borrowed from a live device view, and `result`/`error` are valid
    // output locations.
    unsafe {
        let code = cugraph_out_degrees(
            handle.handle_ptr(),
            graph_ptr,
            vertices_ptr,
            to_bool_t(do_expensive_check),
            &mut result,
            &mut error,
        );
        check_error(code, error, handle.device(), "out_degrees")?;
    }

    OutDegreesResult::from_raw(result, handle.device(), handle.runtime_bound_lease())
}

// ═══════════════════════════════════════════════════════════════════════════
// DegreeValues; typed enum for runtime-width degree arrays
// ═══════════════════════════════════════════════════════════════════════════

/// Runtime-typed degree values returned by cuGraph.
///
/// The cuGraph C API may return degree arrays as either `i32` or `i64`
/// depending on the graph's edge count width. This enum wraps the two
/// possibilities so callers can handle both without raw byte manipulation.
///
/// For convenience, [`to_vec_i64`](DegreeValues::to_vec_i64) widens `i32`
/// values to `i64` automatically.
#[derive(Debug)]
pub enum DegreeValues<'a> {
    /// Degree values stored as 32-bit integers.
    I32(DeviceArrayView<'a, i32>),
    /// Degree values stored as 64-bit integers.
    I64(DeviceArrayView<'a, i64>),
}

impl<'a> DegreeValues<'a> {
    /// Construct from a type-erased raw view after checking the dtype.
    ///
    /// Returns an error if the dtype is neither `Int32` nor `Int64`.
    pub(crate) fn from_raw(raw: RawDeviceArrayView<'a>) -> Result<Self> {
        match raw.dtype()? {
            DataType::Int32 => Ok(Self::I32(DeviceArrayView::from_raw_unchecked(raw))),
            DataType::Int64 => Ok(Self::I64(DeviceArrayView::from_raw_unchecked(raw))),
            other => Err(Error::new(ErrorKind::InvalidState, "DegreeValues::from_raw")
                .with_message(format!("unexpected degree dtype: {other}"))),
        }
    }

    /// Number of elements in the degree array.
    pub fn len(&self) -> usize {
        match self {
            Self::I32(v) => v.len(),
            Self::I64(v) => v.len(),
        }
    }

    /// Whether the degree array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Copy to host as `Vec<i64>`, widening `i32` values if needed.
    pub fn to_vec_i64(&self, handle: &EnteredContext<'_>) -> Result<Vec<i64>> {
        match self {
            Self::I32(v) => {
                let vals = v.to_vec(handle)?;
                Ok(vals.into_iter().map(|x| x as i64).collect())
            },
            Self::I64(v) => v.to_vec(handle),
        }
    }

    /// Copy to host as `Vec<i32>`.
    ///
    /// Returns an error if the underlying values are `i64`, since narrowing
    /// could lose data.
    pub fn to_vec_i32(&self, handle: &EnteredContext<'_>) -> Result<Vec<i32>> {
        match self {
            Self::I32(v) => v.to_vec(handle),
            Self::I64(_) => Err(Error::new(ErrorKind::InvalidState, "DegreeValues::to_vec_i32")
                .with_message("degree values are i64; use to_vec_i64() to avoid data loss")),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Degree result types
// ═══════════════════════════════════════════════════════════════════════════

struct DegreeResultInner {
    raw: *mut cugraph_degrees_result_t,
    device: i32,
    // Keeps a runtime-bound cuGraph handle alive until after Drop frees `raw`.
    _resource_handle_lease: Option<ResourceHandleLease>,
    _nosend_nosync: PhantomData<*mut ()>,
}

impl DegreeResultInner {
    fn new(
        raw: *mut cugraph_degrees_result_t,
        device: i32,
        operation: &'static str,
        resource_handle_lease: Option<ResourceHandleLease>,
    ) -> Result<Self> {
        let raw = require_non_null(raw, operation, "degrees_result")?;
        Ok(Self { raw, device, _resource_handle_lease: resource_handle_lease, _nosend_nosync: PhantomData })
    }

    fn raw_view(
        &self,
        getter: unsafe extern "C" fn(*mut cugraph_degrees_result_t) -> *mut cugraph_type_erased_device_array_view_t,
        operation: &'static str,
        pointer_name: &'static str,
    ) -> Result<RawDeviceArrayView<'_>> {
        // SAFETY: `self.raw` is a live cuGraph degrees result and `getter`
        // returns a borrowed child view tied to that result object.
        unsafe { RawDeviceArrayView::from_borrowed(getter(self.raw), Some(self.device), operation, pointer_name) }
    }

    fn vertex_ids(&self, operation: &'static str) -> Result<VertexIds<'_>> {
        VertexIds::from_raw(self.raw_view(cugraph_degrees_result_get_vertices, operation, "vertices")?)
    }

    fn in_degrees(&self, operation: &'static str) -> Result<DegreeValues<'_>> {
        DegreeValues::from_raw(self.raw_view(cugraph_degrees_result_get_in_degrees, operation, "in_degrees")?)
    }

    fn out_degrees(&self, operation: &'static str) -> Result<DegreeValues<'_>> {
        DegreeValues::from_raw(self.raw_view(cugraph_degrees_result_get_out_degrees, operation, "out_degrees")?)
    }
}

impl Drop for DegreeResultInner {
    fn drop(&mut self) {
        drop_device_resource(&mut self.raw, self.device, "DegreeResultInner::drop", DeviceResourceKind::DegreesResult);
    }
}

/// Result of a degree computation returning both in-degrees and out-degrees.
///
/// Both [`in_degrees()`](DegreesResult::in_degrees) and
/// [`out_degrees()`](DegreesResult::out_degrees) are non-optional: the C API
/// always populates both arrays when `cugraph_degrees` is called.
///
/// Freed on drop via `cugraph_degrees_result_free`.
pub struct DegreesResult(DegreeResultInner);

impl std::fmt::Debug for DegreesResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DegreesResult").finish_non_exhaustive()
    }
}

impl DegreesResult {
    fn from_raw(
        raw: *mut cugraph_degrees_result_t,
        device: i32,
        resource_handle_lease: Option<ResourceHandleLease>,
    ) -> Result<Self> {
        Ok(Self(DegreeResultInner::new(raw, device, "degrees", resource_handle_lease)?))
    }

    /// Get the vertex ids as a runtime-typed device array view.
    pub fn vertex_ids(&self) -> Result<VertexIds<'_>> {
        self.0.vertex_ids("DegreesResult::vertex_ids")
    }

    /// Get the in-degrees as typed degree values.
    pub fn in_degrees(&self) -> Result<DegreeValues<'_>> {
        self.0.in_degrees("DegreesResult::in_degrees")
    }

    /// Get the out-degrees as typed degree values.
    pub fn out_degrees(&self) -> Result<DegreeValues<'_>> {
        self.0.out_degrees("DegreesResult::out_degrees")
    }

    /// Convert this degree result to a cudf DataFrame in the provided runtime.
    pub fn to_dataframe(&self, runtime: &cudf::Runtime) -> Result<cudf::DataFrame> {
        degrees_result_to_dataframe(self, runtime).map_err(|e| e.with_operation("DegreesResult::to_dataframe"))
    }
}

/// Result of an in-degree computation.
///
/// The [`in_degrees()`](InDegreesResult::in_degrees) accessor is non-optional.
/// Freed on drop via `cugraph_degrees_result_free`.
pub struct InDegreesResult(DegreeResultInner);

impl std::fmt::Debug for InDegreesResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InDegreesResult").finish_non_exhaustive()
    }
}

impl InDegreesResult {
    fn from_raw(
        raw: *mut cugraph_degrees_result_t,
        device: i32,
        resource_handle_lease: Option<ResourceHandleLease>,
    ) -> Result<Self> {
        Ok(Self(DegreeResultInner::new(raw, device, "in_degrees", resource_handle_lease)?))
    }

    /// Get the vertex ids as a runtime-typed device array view.
    pub fn vertex_ids(&self) -> Result<VertexIds<'_>> {
        self.0.vertex_ids("InDegreesResult::vertex_ids")
    }

    /// Get the in-degrees as typed degree values.
    pub fn in_degrees(&self) -> Result<DegreeValues<'_>> {
        self.0.in_degrees("InDegreesResult::in_degrees")
    }

    /// Convert this in-degree result to a cudf DataFrame in the provided runtime.
    pub fn to_dataframe(&self, runtime: &cudf::Runtime) -> Result<cudf::DataFrame> {
        in_degrees_result_to_dataframe(self, runtime).map_err(|e| e.with_operation("InDegreesResult::to_dataframe"))
    }
}

/// Result of an out-degree computation.
///
/// The [`out_degrees()`](OutDegreesResult::out_degrees) accessor is non-optional.
/// Freed on drop via `cugraph_degrees_result_free`.
pub struct OutDegreesResult(DegreeResultInner);

impl std::fmt::Debug for OutDegreesResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OutDegreesResult").finish_non_exhaustive()
    }
}

impl OutDegreesResult {
    fn from_raw(
        raw: *mut cugraph_degrees_result_t,
        device: i32,
        resource_handle_lease: Option<ResourceHandleLease>,
    ) -> Result<Self> {
        Ok(Self(DegreeResultInner::new(raw, device, "out_degrees", resource_handle_lease)?))
    }

    /// Get the vertex ids as a runtime-typed device array view.
    pub fn vertex_ids(&self) -> Result<VertexIds<'_>> {
        self.0.vertex_ids("OutDegreesResult::vertex_ids")
    }

    /// Get the out-degrees as typed degree values.
    pub fn out_degrees(&self) -> Result<DegreeValues<'_>> {
        self.0.out_degrees("OutDegreesResult::out_degrees")
    }

    /// Convert this out-degree result to a cudf DataFrame in the provided runtime.
    pub fn to_dataframe(&self, runtime: &cudf::Runtime) -> Result<cudf::DataFrame> {
        out_degrees_result_to_dataframe(self, runtime).map_err(|e| e.with_operation("OutDegreesResult::to_dataframe"))
    }
}
