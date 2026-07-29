//! Degree computation for graph vertices.
//!
//! Computes in-degrees, out-degrees, or both for all (or a subset of) vertices
//! in a graph.

use std::{marker::PhantomData, ptr};

use cugraph_sys::{
    cugraph_degrees, cugraph_degrees_result_free, cugraph_degrees_result_get_in_degrees,
    cugraph_degrees_result_get_out_degrees, cugraph_degrees_result_get_vertices, cugraph_degrees_result_t,
    cugraph_error_t, cugraph_in_degrees, cugraph_out_degrees, cugraph_type_erased_device_array_view_t,
};

use crate::{
    array::{DeviceArrayView, RawDeviceArrayView},
    cuda::DeviceGuard,
    error::{Error, ErrorKind, Result, check_error, require_non_null},
    graph::{Graph, Transposed},
    handle::EnteredContext,
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

    // SAFETY: `graph` and `handle` outlive the call, `vertices_ptr` is either
    // null or borrowed from a live device view, and `result`/`error` are valid
    // output locations.
    unsafe {
        let code = cugraph_degrees(
            handle.handle_ptr(),
            graph.as_ptr(),
            vertices_ptr,
            to_bool_t(do_expensive_check),
            &mut result,
            &mut error,
        );
        check_error(code, error, "degrees")?;
    }

    DegreesResult::from_raw(result, handle.device())
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

    // SAFETY: `graph` and `handle` outlive the call, `vertices_ptr` is either
    // null or borrowed from a live device view, and `result`/`error` are valid
    // output locations.
    unsafe {
        let code = cugraph_in_degrees(
            handle.handle_ptr(),
            graph.as_ptr(),
            vertices_ptr,
            to_bool_t(do_expensive_check),
            &mut result,
            &mut error,
        );
        check_error(code, error, "in_degrees")?;
    }

    InDegreesResult::from_raw(result, handle.device())
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

    // SAFETY: `graph` and `handle` outlive the call, `vertices_ptr` is either
    // null or borrowed from a live device view, and `result`/`error` are valid
    // output locations.
    unsafe {
        let code = cugraph_out_degrees(
            handle.handle_ptr(),
            graph.as_ptr(),
            vertices_ptr,
            to_bool_t(do_expensive_check),
            &mut result,
            &mut error,
        );
        check_error(code, error, "out_degrees")?;
    }

    OutDegreesResult::from_raw(result, handle.device())
}

// ═══════════════════════════════════════════════════════════════════════════
// DegreeValues — typed enum for runtime-width degree arrays
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
    _nosend_nosync: PhantomData<*mut ()>,
}

impl DegreeResultInner {
    fn new(raw: *mut cugraph_degrees_result_t, device: i32, operation: &'static str) -> Result<Self> {
        let raw = require_non_null(raw, operation, "degrees_result")?;
        Ok(Self { raw, device, _nosend_nosync: PhantomData })
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
        if !self.raw.is_null() {
            let _guard = DeviceGuard::switch_to_for_drop(self.device, "DegreeResultInner::drop").ok();
            // SAFETY: `self.raw` is owned by this wrapper and freed exactly
            // once here.
            unsafe { cugraph_degrees_result_free(self.raw) };
        }
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
    fn from_raw(raw: *mut cugraph_degrees_result_t, device: i32) -> Result<Self> {
        Ok(Self(DegreeResultInner::new(raw, device, "degrees")?))
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
        crate::cudf_interop::degrees_result_to_dataframe(self, runtime)
            .map_err(|e| e.with_operation("DegreesResult::to_dataframe"))
    }
}

// ---------------------------------------------------------------------------
// InDegreesResult
// ---------------------------------------------------------------------------

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
    fn from_raw(raw: *mut cugraph_degrees_result_t, device: i32) -> Result<Self> {
        Ok(Self(DegreeResultInner::new(raw, device, "in_degrees")?))
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
        crate::cudf_interop::in_degrees_result_to_dataframe(self, runtime)
            .map_err(|e| e.with_operation("InDegreesResult::to_dataframe"))
    }
}

// ---------------------------------------------------------------------------
// OutDegreesResult
// ---------------------------------------------------------------------------

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
    fn from_raw(raw: *mut cugraph_degrees_result_t, device: i32) -> Result<Self> {
        Ok(Self(DegreeResultInner::new(raw, device, "out_degrees")?))
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
        crate::cudf_interop::out_degrees_result_to_dataframe(self, runtime)
            .map_err(|e| e.with_operation("OutDegreesResult::to_dataframe"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ResourceHandle,
        array::{DeviceArray, RawDeviceArrayView},
        error::ErrorKind,
        graph::{GraphBuilder, Transposed},
        handle::EnteredContext,
        types::{DataType, GraphProperties},
    };

    /// Helper: create a star graph where vertices 1,2,3 all point to vertex 0.
    /// With store_transposed=true and renumber=true (needed for degree computation).
    fn make_star_graph(handle: &EnteredContext<'_>) -> Graph<Transposed> {
        // Directed star: 1->0, 2->0, 3->0
        let src_data: Vec<i32> = vec![1, 2, 3];
        let dst_data: Vec<i32> = vec![0, 0, 0];

        let src = DeviceArray::from_slice(handle, &src_data).expect("src");
        let dst = DeviceArray::from_slice(handle, &dst_data).expect("dst");

        let src_view = src.view().expect("src_view");
        let dst_view = dst.view().expect("dst_view");

        GraphBuilder::new(&src_view, &dst_view).renumber().build::<Transposed>(handle).expect("star graph")
    }

    /// Helper: build a vertex -> (in_degree, out_degree) map from a DegreesResult
    /// using the DegreeValues API.
    fn degree_map(handle: &EnteredContext<'_>, result: &DegreesResult) -> std::collections::HashMap<i32, (i64, i64)> {
        let vertices: Vec<i32> = result.vertex_ids().expect("vertex_ids").to_vec_i32(handle).expect("vert copy");
        let in_degs = result.in_degrees().expect("in_degrees").to_vec_i64(handle).expect("in copy");
        let out_degs = result.out_degrees().expect("out_degrees").to_vec_i64(handle).expect("out copy");

        let mut map = std::collections::HashMap::new();
        for i in 0..vertices.len() {
            map.insert(vertices[i], (in_degs[i], out_degs[i]));
        }
        map
    }

    #[test]
    fn test_degrees_star_graph() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_degrees_star_graph").expect("entered");
        let mut graph = make_star_graph(&entered);

        let result = graph.degrees_all(&entered).expect("degrees");

        let vert_view = result.vertex_ids().expect("vertex_ids");
        let in_vals = result.in_degrees().expect("in_degrees");
        let out_vals = result.out_degrees().expect("out_degrees");

        // Star with 4 vertices
        assert_eq!(vert_view.len(), 4);
        assert_eq!(in_vals.len(), 4);
        assert_eq!(out_vals.len(), 4);

        let map = degree_map(&entered, &result);

        // Vertex 0: hub — receives edges from 1,2,3 → in=3, out=0
        assert_eq!(map[&0], (3, 0), "hub should have in=3, out=0");

        // Vertices 1,2,3: leaves — each sends one edge → in=0, out=1
        for v in [1, 2, 3] {
            assert_eq!(map[&v], (0, 1), "leaf vertex {} should have in=0, out=1", v);
        }
    }

    #[test]
    fn test_in_degrees_star_graph() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_in_degrees_star_graph").expect("entered");
        let mut graph = make_star_graph(&entered);

        let result = graph.in_degrees_all(&entered).expect("in_degrees");

        let vert_ids = result.vertex_ids().expect("vertex_ids");
        let in_vals = result.in_degrees().expect("in_degrees");

        assert_eq!(vert_ids.len(), 4);
        assert_eq!(in_vals.len(), 4);

        let vertices: Vec<i32> = vert_ids.to_vec_i32(&entered).expect("vert copy");
        let in_degs = in_vals.to_vec_i64(&entered).expect("in copy");

        let mut map = std::collections::HashMap::new();
        for i in 0..vertices.len() {
            map.insert(vertices[i], in_degs[i]);
        }

        // Hub vertex 0 receives 3 edges
        assert_eq!(map[&0], 3, "hub should have in_degree=3");
        for v in [1, 2, 3] {
            assert_eq!(map[&v], 0, "leaf {} should have in_degree=0", v);
        }
    }

    #[test]
    fn test_out_degrees_star_graph() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_out_degrees_star_graph").expect("entered");
        let mut graph = make_star_graph(&entered);

        let result = graph.out_degrees_all(&entered).expect("out_degrees");

        let vert_ids = result.vertex_ids().expect("vertex_ids");
        let out_vals = result.out_degrees().expect("out_degrees");

        assert_eq!(vert_ids.len(), 4);
        assert_eq!(out_vals.len(), 4);

        let vertices: Vec<i32> = vert_ids.to_vec_i32(&entered).expect("vert copy");
        let out_degs = out_vals.to_vec_i64(&entered).expect("out copy");

        let mut map = std::collections::HashMap::new();
        for i in 0..vertices.len() {
            map.insert(vertices[i], out_degs[i]);
        }

        // Hub vertex 0 sends 0 edges
        assert_eq!(map[&0], 0, "hub should have out_degree=0");
        // Leaf vertices each send 1 edge
        for v in [1, 2, 3] {
            assert_eq!(map[&v], 1, "leaf {} should have out_degree=1", v);
        }
    }

    #[test]
    fn test_degrees_cycle_graph() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_degrees_cycle_graph").expect("entered");

        // Directed cycle: 0->1, 1->2, 2->3, 3->0
        let src_data: Vec<i32> = vec![0, 1, 2, 3];
        let dst_data: Vec<i32> = vec![1, 2, 3, 0];

        let src = DeviceArray::from_slice(&entered, &src_data).expect("src");
        let dst = DeviceArray::from_slice(&entered, &dst_data).expect("dst");

        let src_view = src.view().expect("src_view");
        let dst_view = dst.view().expect("dst_view");
        let mut graph =
            GraphBuilder::new(&src_view, &dst_view).renumber().build::<Transposed>(&entered).expect("graph");

        let result = graph.degrees_all(&entered).expect("degrees");
        let map = degree_map(&entered, &result);

        // Every vertex in a directed cycle has in=1, out=1
        assert_eq!(map.len(), 4);
        for v in 0..4 {
            assert_eq!(map[&v], (1, 1), "vertex {} in cycle should have in=1, out=1", v);
        }
    }

    #[test]
    fn test_degrees_with_source_vertices_subset() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_degrees_with_source_vertices_subset").expect("entered");
        let mut graph = make_star_graph(&entered);

        // Only query degrees for vertices 0 and 1
        let query = DeviceArray::from_slice(&entered, &[0i32, 1]).expect("query");
        let query_view = query.view().expect("query_view");

        let result = graph.degrees(&entered, &query_view).expect("degrees subset");

        let vert_view = result.vertex_ids().expect("vertex_ids");
        assert_eq!(vert_view.len(), 2, "should return degrees for 2 vertices");

        let map = degree_map(&entered, &result);
        assert!(map.contains_key(&0), "should contain vertex 0");
        assert!(map.contains_key(&1), "should contain vertex 1");
        assert_eq!(map[&0], (3, 0), "hub should have in=3, out=0");
        assert_eq!(map[&1], (0, 1), "leaf 1 should have in=0, out=1");
    }

    #[test]
    fn test_degrees_isolated_vertex() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_degrees_isolated_vertex").expect("entered");

        // Graph: 0->1, 3->4 — vertices 0,1,3,4 appear, and we check
        // that all degrees are correct.
        let src_data: Vec<i32> = vec![0, 3];
        let dst_data: Vec<i32> = vec![1, 4];

        let src = DeviceArray::from_slice(&entered, &src_data).expect("src");
        let dst = DeviceArray::from_slice(&entered, &dst_data).expect("dst");

        let src_view = src.view().expect("src_view");
        let dst_view = dst.view().expect("dst_view");
        let mut graph =
            GraphBuilder::new(&src_view, &dst_view).renumber().build::<Transposed>(&entered).expect("graph");

        let result = graph.degrees_all(&entered).expect("degrees");
        let map = degree_map(&entered, &result);

        // Vertex 0: out=1, in=0
        assert_eq!(map[&0], (0, 1), "vertex 0 should have in=0, out=1");
        // Vertex 1: out=0, in=1
        assert_eq!(map[&1], (1, 0), "vertex 1 should have in=1, out=0");
        // Vertex 3: out=1, in=0
        assert_eq!(map[&3], (0, 1), "vertex 3 should have in=0, out=1");
        // Vertex 4: out=0, in=1
        assert_eq!(map[&4], (1, 0), "vertex 4 should have in=1, out=0");
    }

    #[test]
    fn test_degrees_symmetric_graph() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_degrees_symmetric_graph").expect("entered");

        // Symmetric (undirected) triangle: 0<->1, 1<->2, 2<->0
        let src_data: Vec<i32> = vec![0, 1, 1, 2, 2, 0];
        let dst_data: Vec<i32> = vec![1, 0, 2, 1, 0, 2];

        let src = DeviceArray::from_slice(&entered, &src_data).expect("src");
        let dst = DeviceArray::from_slice(&entered, &dst_data).expect("dst");

        let src_view = src.view().expect("src_view");
        let dst_view = dst.view().expect("dst_view");
        let mut graph = GraphBuilder::new(&src_view, &dst_view)
            .properties(GraphProperties::undirected())
            .renumber()
            .build::<Transposed>(&entered)
            .expect("graph");

        let result = graph.degrees_all(&entered).expect("degrees");
        let map = degree_map(&entered, &result);

        // Symmetric graph: in_degree == out_degree for every vertex
        assert_eq!(map.len(), 3);
        for v in 0..3 {
            let (in_d, out_d) = map[&v];
            assert_eq!(in_d, out_d, "symmetric graph vertex {} should have in_degree == out_degree", v);
            assert_eq!(in_d, 2, "vertex {} should have degree 2 in triangle", v);
        }
    }

    #[test]
    fn degree_values_i64_to_vec_i32_is_invalid_state() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("degree_values_i64_to_vec_i32_is_invalid_state").expect("entered");
        let degrees = DeviceArray::from_slice(&entered, &[1i64, 2, 3]).expect("degrees");
        let values = DegreeValues::I64(degrees.view().expect("view"));

        let err = values.to_vec_i32(&entered).expect_err("i64 degree values should reject i32 copy");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert_eq!(err.operation(), "DegreeValues::to_vec_i32");
    }

    #[test]
    fn degree_values_from_raw_unexpected_dtype_is_invalid_state() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("degree_values_from_raw_unexpected_dtype_is_invalid_state").expect("entered");
        let degrees = DeviceArray::from_slice(&entered, &[1.0f32, 2.0, 3.0]).expect("degrees");
        let view = degrees.view().expect("view");
        // SAFETY: The pointer/length pair comes from a live `DeviceArrayView`, and the dtype is
        // intentionally mismatched to exercise Rust-side validation in `DegreeValues::from_raw`.
        let raw = unsafe {
            RawDeviceArrayView::try_from_raw(view.device_ptr(), view.len(), DataType::Float32, "test_raw")
                .expect("raw view")
        };

        let err = DegreeValues::from_raw(raw).expect_err("unexpected degree dtype should be rejected");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert_eq!(err.operation(), "DegreeValues::from_raw");
    }
}
