//! Degree computation for graph vertices.
//!
//! Computes in-degrees, out-degrees, or both for all (or a subset of) vertices
//! in a graph.

use std::ptr;

use cugraph_sys::{
    bool_t, cugraph_degrees, cugraph_degrees_result_free, cugraph_degrees_result_get_in_degrees,
    cugraph_degrees_result_get_out_degrees, cugraph_degrees_result_get_vertices, cugraph_degrees_result_t,
    cugraph_error_t, cugraph_in_degrees, cugraph_out_degrees,
};

use crate::{
    array::{DeviceArrayView, RawDeviceArrayView},
    error::{Error, ErrorKind, Result, check_error},
    graph::{Graph, Transposed},
    handle::ResourceHandle,
    structure::typed_results::VertexIds,
    types::{CugraphElement, DataType},
};

/// Specifies which vertices to query degrees for.
///
/// Used as a parameter to [`Graph::degrees`], [`Graph::in_degrees`], and
/// [`Graph::out_degrees`] instead of a bare `Option`.
///
/// # Examples
///
/// ```no_run
/// use cugraph::DegreeVertices;
///
/// // Query all vertices (the default):
/// let all: DegreeVertices<'_> = DegreeVertices::All;
///
/// // Query a subset via From conversion:
/// # use cugraph::{DeviceArray, ResourceHandle};
/// # let handle = ResourceHandle::new().unwrap();
/// # let arr = DeviceArray::from_slice(&handle, &[0i32, 1]).unwrap();
/// let subset = DegreeVertices::from(&arr.view());
/// ```
#[derive(Debug, Default)]
pub enum DegreeVertices<'a, V: CugraphElement = i32> {
    /// Compute degrees for all vertices in the graph.
    #[default]
    All,
    /// Compute degrees only for the specified vertex subset.
    Subset(&'a DeviceArrayView<'a, V>),
}

impl<'a, V: CugraphElement> From<&'a DeviceArrayView<'a, V>> for DegreeVertices<'a, V> {
    fn from(view: &'a DeviceArrayView<'a, V>) -> Self {
        Self::Subset(view)
    }
}

/// Compute both in-degrees and out-degrees for vertices in a graph.
///
/// Prefer using [`Graph::degrees`](crate::graph::Graph::degrees) instead
/// of calling this function directly.
pub(crate) fn degrees<V: CugraphElement>(
    handle: &ResourceHandle,
    graph: &mut Graph<Transposed>,
    vertices: &DegreeVertices<'_, V>,
) -> Result<DegreesResult> {
    let mut result: *mut cugraph_degrees_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();

    let src_ptr = match vertices {
        DegreeVertices::All => ptr::null(),
        DegreeVertices::Subset(v) => v.as_ptr() as *const _,
    };

    unsafe {
        let code = cugraph_degrees(
            handle.as_ptr(),
            graph.raw,
            src_ptr,
            bool_t::FALSE, // do_expensive_check
            &mut result,
            &mut error,
        );
        check_error(code, error, "degrees")?;
    }

    Ok(DegreesResult { raw: result, _nosend_nosync: std::marker::PhantomData })
}

/// Compute only in-degrees for vertices in a graph.
///
/// Prefer using [`Graph::in_degrees`](crate::graph::Graph::in_degrees)
/// instead of calling this function directly.
pub(crate) fn in_degrees<V: CugraphElement>(
    handle: &ResourceHandle,
    graph: &mut Graph<Transposed>,
    vertices: &DegreeVertices<'_, V>,
) -> Result<InDegreesResult> {
    let mut result: *mut cugraph_degrees_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();

    let src_ptr = match vertices {
        DegreeVertices::All => ptr::null(),
        DegreeVertices::Subset(v) => v.as_ptr() as *const _,
    };

    unsafe {
        let code = cugraph_in_degrees(
            handle.as_ptr(),
            graph.raw,
            src_ptr,
            bool_t::FALSE, // do_expensive_check
            &mut result,
            &mut error,
        );
        check_error(code, error, "in_degrees")?;
    }

    Ok(InDegreesResult { raw: result, _nosend_nosync: std::marker::PhantomData })
}

/// Compute only out-degrees for vertices in a graph.
///
/// Prefer using [`Graph::out_degrees`](crate::graph::Graph::out_degrees)
/// instead of calling this function directly.
pub(crate) fn out_degrees<T, V: CugraphElement>(
    handle: &ResourceHandle,
    graph: &mut Graph<T>,
    vertices: &DegreeVertices<'_, V>,
) -> Result<OutDegreesResult> {
    let mut result: *mut cugraph_degrees_result_t = ptr::null_mut();
    let mut error: *mut cugraph_error_t = ptr::null_mut();

    let src_ptr = match vertices {
        DegreeVertices::All => ptr::null(),
        DegreeVertices::Subset(v) => v.as_ptr() as *const _,
    };

    unsafe {
        let code = cugraph_out_degrees(
            handle.as_ptr(),
            graph.raw,
            src_ptr,
            bool_t::FALSE, // do_expensive_check
            &mut result,
            &mut error,
        );
        check_error(code, error, "out_degrees")?;
    }

    Ok(OutDegreesResult { raw: result, _nosend_nosync: std::marker::PhantomData })
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
            other => Err(Error::new(ErrorKind::InvalidInput, "DegreeValues::from_raw")
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
    pub fn to_vec_i64(&self, handle: &ResourceHandle) -> Result<Vec<i64>> {
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
    pub fn to_vec_i32(&self, handle: &ResourceHandle) -> Result<Vec<i32>> {
        match self {
            Self::I32(v) => v.to_vec(handle),
            Self::I64(_) => Err(Error::new(ErrorKind::InvalidInput, "DegreeValues::to_vec_i32")
                .with_message("degree values are i64; use to_vec_i64() to avoid data loss")),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Degree result types
// ═══════════════════════════════════════════════════════════════════════════

/// Result of a degree computation returning both in-degrees and out-degrees.
///
/// Both [`in_degrees()`](DegreesResult::in_degrees) and
/// [`out_degrees()`](DegreesResult::out_degrees) are non-optional: the C API
/// always populates both arrays when `cugraph_degrees` is called.
///
/// Freed on drop via `cugraph_degrees_result_free`.
pub struct DegreesResult {
    raw: *mut cugraph_degrees_result_t,
    _nosend_nosync: std::marker::PhantomData<*mut ()>,
}

impl std::fmt::Debug for DegreesResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DegreesResult").finish_non_exhaustive()
    }
}

impl DegreesResult {
    /// Get the vertex ids as a runtime-typed device array view.
    ///
    /// Returns [`VertexIds::I32`] or [`VertexIds::I64`] depending on the
    /// graph's vertex type.
    ///
    /// # Errors
    ///
    /// Returns an error if the C API returned a null pointer or an
    /// unrecognised dtype.
    pub fn vertex_ids(&self) -> Result<VertexIds<'_>> {
        let view_ptr = unsafe { cugraph_degrees_result_get_vertices(self.raw) };
        if view_ptr.is_null() {
            return Err(Error::new(ErrorKind::Unexpected, "DegreesResult::vertex_ids")
                .with_message("C API returned null pointer for vertices"));
        }
        let raw = unsafe { RawDeviceArrayView::from_borrowed(view_ptr) };
        VertexIds::from_raw(raw)
    }

    /// Get the in-degrees as typed degree values.
    ///
    /// The C API always populates this when `cugraph_degrees` is called.
    ///
    /// # Errors
    ///
    /// Returns an error if the C API returned a null pointer or an
    /// unrecognised dtype.
    pub fn in_degrees(&self) -> Result<DegreeValues<'_>> {
        let view_ptr = unsafe { cugraph_degrees_result_get_in_degrees(self.raw) };
        if view_ptr.is_null() {
            return Err(Error::new(ErrorKind::Unexpected, "DegreesResult::in_degrees")
                .with_message("C API returned null pointer for in_degrees"));
        }
        let raw = unsafe { RawDeviceArrayView::from_borrowed(view_ptr) };
        DegreeValues::from_raw(raw)
    }

    /// Get the out-degrees as typed degree values.
    ///
    /// The C API always populates this when `cugraph_degrees` is called.
    ///
    /// # Errors
    ///
    /// Returns an error if the C API returned a null pointer or an
    /// unrecognised dtype.
    pub fn out_degrees(&self) -> Result<DegreeValues<'_>> {
        let view_ptr = unsafe { cugraph_degrees_result_get_out_degrees(self.raw) };
        if view_ptr.is_null() {
            return Err(Error::new(ErrorKind::Unexpected, "DegreesResult::out_degrees")
                .with_message("C API returned null pointer for out_degrees"));
        }
        let raw = unsafe { RawDeviceArrayView::from_borrowed(view_ptr) };
        DegreeValues::from_raw(raw)
    }
}

impl Drop for DegreesResult {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { cugraph_degrees_result_free(self.raw) };
        }
    }
}

// ---------------------------------------------------------------------------
// InDegreesResult
// ---------------------------------------------------------------------------

/// Result of an in-degree computation.
///
/// The [`in_degrees()`](InDegreesResult::in_degrees) accessor is non-optional.
/// Freed on drop via `cugraph_degrees_result_free`.
pub struct InDegreesResult {
    raw: *mut cugraph_degrees_result_t,
    _nosend_nosync: std::marker::PhantomData<*mut ()>,
}

impl std::fmt::Debug for InDegreesResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InDegreesResult").finish_non_exhaustive()
    }
}

impl InDegreesResult {
    /// Get the vertex ids as a runtime-typed device array view.
    ///
    /// Returns [`VertexIds::I32`] or [`VertexIds::I64`] depending on the
    /// graph's vertex type.
    ///
    /// # Errors
    ///
    /// Returns an error if the C API returned a null pointer or an
    /// unrecognised dtype.
    pub fn vertex_ids(&self) -> Result<VertexIds<'_>> {
        let view_ptr = unsafe { cugraph_degrees_result_get_vertices(self.raw) };
        if view_ptr.is_null() {
            return Err(Error::new(ErrorKind::Unexpected, "InDegreesResult::vertex_ids")
                .with_message("C API returned null pointer for vertices"));
        }
        let raw = unsafe { RawDeviceArrayView::from_borrowed(view_ptr) };
        VertexIds::from_raw(raw)
    }

    /// Get the in-degrees as typed degree values.
    ///
    /// # Errors
    ///
    /// Returns an error if the C API returned a null pointer or an
    /// unrecognised dtype.
    pub fn in_degrees(&self) -> Result<DegreeValues<'_>> {
        let view_ptr = unsafe { cugraph_degrees_result_get_in_degrees(self.raw) };
        if view_ptr.is_null() {
            return Err(Error::new(ErrorKind::Unexpected, "InDegreesResult::in_degrees")
                .with_message("C API returned null pointer for in_degrees"));
        }
        let raw = unsafe { RawDeviceArrayView::from_borrowed(view_ptr) };
        DegreeValues::from_raw(raw)
    }
}

impl Drop for InDegreesResult {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { cugraph_degrees_result_free(self.raw) };
        }
    }
}

// ---------------------------------------------------------------------------
// OutDegreesResult
// ---------------------------------------------------------------------------

/// Result of an out-degree computation.
///
/// The [`out_degrees()`](OutDegreesResult::out_degrees) accessor is non-optional.
/// Freed on drop via `cugraph_degrees_result_free`.
pub struct OutDegreesResult {
    raw: *mut cugraph_degrees_result_t,
    _nosend_nosync: std::marker::PhantomData<*mut ()>,
}

impl std::fmt::Debug for OutDegreesResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OutDegreesResult").finish_non_exhaustive()
    }
}

impl OutDegreesResult {
    /// Get the vertex ids as a runtime-typed device array view.
    ///
    /// Returns [`VertexIds::I32`] or [`VertexIds::I64`] depending on the
    /// graph's vertex type.
    ///
    /// # Errors
    ///
    /// Returns an error if the C API returned a null pointer or an
    /// unrecognised dtype.
    pub fn vertex_ids(&self) -> Result<VertexIds<'_>> {
        let view_ptr = unsafe { cugraph_degrees_result_get_vertices(self.raw) };
        if view_ptr.is_null() {
            return Err(Error::new(ErrorKind::Unexpected, "OutDegreesResult::vertex_ids")
                .with_message("C API returned null pointer for vertices"));
        }
        let raw = unsafe { RawDeviceArrayView::from_borrowed(view_ptr) };
        VertexIds::from_raw(raw)
    }

    /// Get the out-degrees as typed degree values.
    ///
    /// # Errors
    ///
    /// Returns an error if the C API returned a null pointer or an
    /// unrecognised dtype.
    pub fn out_degrees(&self) -> Result<DegreeValues<'_>> {
        let view_ptr = unsafe { cugraph_degrees_result_get_out_degrees(self.raw) };
        if view_ptr.is_null() {
            return Err(Error::new(ErrorKind::Unexpected, "OutDegreesResult::out_degrees")
                .with_message("C API returned null pointer for out_degrees"));
        }
        let raw = unsafe { RawDeviceArrayView::from_borrowed(view_ptr) };
        DegreeValues::from_raw(raw)
    }
}

impl Drop for OutDegreesResult {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { cugraph_degrees_result_free(self.raw) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        array::DeviceArray,
        graph::{GraphBuilder, Transposed},
        types::GraphProperties,
    };

    /// Helper: create a star graph where vertices 1,2,3 all point to vertex 0.
    /// With store_transposed=true and renumber=true (needed for degree computation).
    fn make_star_graph(handle: &ResourceHandle) -> Graph<Transposed> {
        // Directed star: 1->0, 2->0, 3->0
        let src_data: Vec<i32> = vec![1, 2, 3];
        let dst_data: Vec<i32> = vec![0, 0, 0];

        let src = DeviceArray::from_slice(handle, &src_data).expect("src");
        let dst = DeviceArray::from_slice(handle, &dst_data).expect("dst");

        let src_view = src.view();
        let dst_view = dst.view();

        GraphBuilder::new(&src_view, &dst_view).renumber().build::<Transposed>(handle).expect("star graph")
    }

    /// Helper: build a vertex -> (in_degree, out_degree) map from a DegreesResult
    /// using the DegreeValues API.
    fn degree_map(handle: &ResourceHandle, result: &DegreesResult) -> std::collections::HashMap<i32, (i64, i64)> {
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
        let mut graph = make_star_graph(&handle);

        let result = graph.degrees_all(&handle).expect("degrees");

        let vert_view = result.vertex_ids().expect("vertex_ids");
        let in_vals = result.in_degrees().expect("in_degrees");
        let out_vals = result.out_degrees().expect("out_degrees");

        // Star with 4 vertices
        assert_eq!(vert_view.len(), 4);
        assert_eq!(in_vals.len(), 4);
        assert_eq!(out_vals.len(), 4);

        let map = degree_map(&handle, &result);

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
        let mut graph = make_star_graph(&handle);

        let result = graph.in_degrees_all(&handle).expect("in_degrees");

        let vert_ids = result.vertex_ids().expect("vertex_ids");
        let in_vals = result.in_degrees().expect("in_degrees");

        assert_eq!(vert_ids.len(), 4);
        assert_eq!(in_vals.len(), 4);

        let vertices: Vec<i32> = vert_ids.to_vec_i32(&handle).expect("vert copy");
        let in_degs = in_vals.to_vec_i64(&handle).expect("in copy");

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
        let mut graph = make_star_graph(&handle);

        let result = graph.out_degrees_all(&handle).expect("out_degrees");

        let vert_ids = result.vertex_ids().expect("vertex_ids");
        let out_vals = result.out_degrees().expect("out_degrees");

        assert_eq!(vert_ids.len(), 4);
        assert_eq!(out_vals.len(), 4);

        let vertices: Vec<i32> = vert_ids.to_vec_i32(&handle).expect("vert copy");
        let out_degs = out_vals.to_vec_i64(&handle).expect("out copy");

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

        // Directed cycle: 0->1, 1->2, 2->3, 3->0
        let src_data: Vec<i32> = vec![0, 1, 2, 3];
        let dst_data: Vec<i32> = vec![1, 2, 3, 0];

        let src = DeviceArray::from_slice(&handle, &src_data).expect("src");
        let dst = DeviceArray::from_slice(&handle, &dst_data).expect("dst");

        let src_view = src.view();
        let dst_view = dst.view();
        let mut graph = GraphBuilder::new(&src_view, &dst_view).renumber().build::<Transposed>(&handle).expect("graph");

        let result = graph.degrees_all(&handle).expect("degrees");
        let map = degree_map(&handle, &result);

        // Every vertex in a directed cycle has in=1, out=1
        assert_eq!(map.len(), 4);
        for v in 0..4 {
            assert_eq!(map[&v], (1, 1), "vertex {} in cycle should have in=1, out=1", v);
        }
    }

    #[test]
    fn test_degrees_with_source_vertices_subset() {
        let handle = ResourceHandle::new().expect("handle");
        let mut graph = make_star_graph(&handle);

        // Only query degrees for vertices 0 and 1
        let query = DeviceArray::from_slice(&handle, &[0i32, 1]).expect("query");

        let result = graph.degrees(&handle, &DegreeVertices::from(&query.view())).expect("degrees subset");

        let vert_view = result.vertex_ids().expect("vertex_ids");
        assert_eq!(vert_view.len(), 2, "should return degrees for 2 vertices");

        let map = degree_map(&handle, &result);
        assert!(map.contains_key(&0), "should contain vertex 0");
        assert!(map.contains_key(&1), "should contain vertex 1");
        assert_eq!(map[&0], (3, 0), "hub should have in=3, out=0");
        assert_eq!(map[&1], (0, 1), "leaf 1 should have in=0, out=1");
    }

    #[test]
    fn test_degrees_isolated_vertex() {
        let handle = ResourceHandle::new().expect("handle");

        // Graph: 0->1, 3->4 — vertices 0,1,3,4 appear, and we check
        // that all degrees are correct.
        let src_data: Vec<i32> = vec![0, 3];
        let dst_data: Vec<i32> = vec![1, 4];

        let src = DeviceArray::from_slice(&handle, &src_data).expect("src");
        let dst = DeviceArray::from_slice(&handle, &dst_data).expect("dst");

        let src_view = src.view();
        let dst_view = dst.view();
        let mut graph = GraphBuilder::new(&src_view, &dst_view).renumber().build::<Transposed>(&handle).expect("graph");

        let result = graph.degrees_all(&handle).expect("degrees");
        let map = degree_map(&handle, &result);

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

        // Symmetric (undirected) triangle: 0<->1, 1<->2, 2<->0
        let src_data: Vec<i32> = vec![0, 1, 1, 2, 2, 0];
        let dst_data: Vec<i32> = vec![1, 0, 2, 1, 0, 2];

        let src = DeviceArray::from_slice(&handle, &src_data).expect("src");
        let dst = DeviceArray::from_slice(&handle, &dst_data).expect("dst");

        let src_view = src.view();
        let dst_view = dst.view();
        let mut graph = GraphBuilder::new(&src_view, &dst_view)
            .properties(GraphProperties::undirected())
            .renumber()
            .build::<Transposed>(&handle)
            .expect("graph");

        let result = graph.degrees_all(&handle).expect("degrees");
        let map = degree_map(&handle, &result);

        // Symmetric graph: in_degree == out_degree for every vertex
        assert_eq!(map.len(), 3);
        for v in 0..3 {
            let (in_d, out_d) = map[&v];
            assert_eq!(in_d, out_d, "symmetric graph vertex {} should have in_degree == out_degree", v);
            assert_eq!(in_d, 2, "vertex {} should have degree 2 in triangle", v);
        }
    }
}
