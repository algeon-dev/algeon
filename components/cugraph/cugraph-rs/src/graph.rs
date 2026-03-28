//! Graph construction and management.
//!
//! A [`Graph`] is the central data structure for cuGraph algorithms. It is
//! constructed from an edge list via [`GraphBuilder`] and freed on drop.
//!
//! # Type-state encoding
//!
//! The `Graph<T>` type parameter encodes whether the graph is stored in
//! transposed form ([`Transposed`]) or forward form ([`Forward`]). Algorithms
//! that require a specific layout accept only the correct variant, so
//! mismatches are caught at compile time:
//!
//! - **[`Forward`]** — required by [`bfs`](crate::traversal::bfs::bfs)
//! - **[`Transposed`]** — required by [`pagerank`](crate::centrality::pagerank::pagerank),
//!   [`in_degrees`](crate::structure::degrees::in_degrees), and
//!   [`degrees`](crate::structure::degrees::degrees)
//!
//! Algorithms that work with either layout (e.g. betweenness centrality,
//! Force Atlas 2, out_degrees) are generic over `T: Transposition`.
//!
//! # Example
//!
//! ```no_run
//! # use cugraph::graph::GraphBuilder;
//! # use cugraph::types::GraphProperties;
//! # use cugraph::array::device_array_from_host;
//! # use cugraph::handle::ResourceHandle;
//! # use cugraph::types::DataType;
//! # let handle = ResourceHandle::new().unwrap();
//! # let src = device_array_from_host(&handle, &[0i32, 1, 2], DataType::Int32).unwrap();
//! # let dst = device_array_from_host(&handle, &[1i32, 2, 0], DataType::Int32).unwrap();
//! let props = GraphProperties { is_symmetric: false, is_multigraph: false };
//! let graph = GraphBuilder::new(props, &src.view(), &dst.view())
//!     .renumber(true)
//!     .build_transposed(&handle)
//!     .unwrap();
//! ```

use std::ptr;

use cugraph_sys::{cugraph_error_t, cugraph_graph_create_with_times_sg, cugraph_graph_free, cugraph_graph_t};

use crate::{
    array::DeviceArrayView,
    error::{Error, ErrorKind, Result, check_error},
    handle::ResourceHandle,
    types::{GraphProperties, to_bool_t},
};

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::Forward {}
    impl Sealed for super::Transposed {}
}

/// Trait implemented by the graph transposition marker types.
///
/// This trait is [sealed] — it cannot be implemented outside this crate.
///
/// [sealed]: https://rust-lang.github.io/api-guidelines/future-proofing.html#sealed-traits-protect-against-downstream-implementations-c-sealed
pub trait Transposition: sealed::Sealed {
    /// Whether this transposition state stores the graph transposed.
    #[doc(hidden)]
    fn store_transposed() -> bool;
}

/// Marker type indicating a graph stored in forward (non-transposed) form.
///
/// Required by algorithms like [`bfs`](crate::traversal::bfs::bfs).
pub struct Forward;
impl Transposition for Forward {
    fn store_transposed() -> bool {
        false
    }
}

/// Marker type indicating a graph stored in transposed form.
///
/// Required by algorithms like [`pagerank`](crate::centrality::pagerank::pagerank) and
/// [`in_degrees`](crate::structure::degrees::in_degrees).
pub struct Transposed;
impl Transposition for Transposed {
    fn store_transposed() -> bool {
        true
    }
}

/// RAII wrapper for a cuGraph graph object.
///
/// The type parameter `T` encodes the transposition state of the graph:
///
/// - [`Forward`] — stored in forward (non-transposed) form (the default)
/// - [`Transposed`] — stored in transposed form
///
/// Use [`GraphBuilder::build_forward`] or [`GraphBuilder::build_transposed`]
/// to construct a graph with the appropriate type state.
pub struct Graph<T = Forward> {
    pub(crate) raw: *mut cugraph_graph_t,
    _nosend_nosync: std::marker::PhantomData<*mut ()>,
    _transposition: std::marker::PhantomData<T>,
}

impl<T> std::fmt::Debug for Graph<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Graph").finish_non_exhaustive()
    }
}

/// Builder for constructing a [`Graph`] from edge lists.
///
/// Provides a fluent API for setting optional arrays and configuration flags.
///
/// # Required parameters
///
/// - `properties` — whether the graph is symmetric / multigraph
/// - `src` — source vertex ids (device array view)
/// - `dst` — destination vertex ids (device array view)
///
/// # Example
///
/// ```no_run
/// # use cugraph::graph::GraphBuilder;
/// # use cugraph::types::GraphProperties;
/// # use cugraph::array::device_array_from_host;
/// # use cugraph::handle::ResourceHandle;
/// # use cugraph::types::DataType;
/// # let handle = ResourceHandle::new().unwrap();
/// # let src = device_array_from_host(&handle, &[0i32, 1, 2], DataType::Int32).unwrap();
/// # let dst = device_array_from_host(&handle, &[1i32, 2, 0], DataType::Int32).unwrap();
/// # let wgt = device_array_from_host(&handle, &[1.0f32, 2.0, 3.0], DataType::Float32).unwrap();
/// let props = GraphProperties { is_symmetric: false, is_multigraph: false };
/// let graph = GraphBuilder::new(props, &src.view(), &dst.view())
///     .weights(&wgt.view())
///     .renumber(true)
///     .build_transposed(&handle)?;
/// # Ok::<(), cugraph::error::Error>(())
/// ```
#[must_use]
#[derive(Debug, Clone)]
pub struct GraphBuilder<'a> {
    properties: GraphProperties,
    src: &'a DeviceArrayView<'a>,
    dst: &'a DeviceArrayView<'a>,
    weights: Option<&'a DeviceArrayView<'a>>,
    edge_ids: Option<&'a DeviceArrayView<'a>>,
    edge_type_ids: Option<&'a DeviceArrayView<'a>>,
    vertices: Option<&'a DeviceArrayView<'a>>,
    edge_start_time_ids: Option<&'a DeviceArrayView<'a>>,
    edge_end_time_ids: Option<&'a DeviceArrayView<'a>>,
    renumber: bool,
    drop_self_loops: bool,
    drop_multi_edges: bool,
    symmetrize: bool,
    do_expensive_check: bool,
}

impl<'a> GraphBuilder<'a> {
    /// Create a new builder with the required parameters.
    ///
    /// All optional arrays default to `None` and all boolean flags default to
    /// `false`.
    pub fn new(properties: GraphProperties, src: &'a DeviceArrayView<'a>, dst: &'a DeviceArrayView<'a>) -> Self {
        Self {
            properties,
            src,
            dst,
            weights: None,
            edge_ids: None,
            edge_type_ids: None,
            vertices: None,
            edge_start_time_ids: None,
            edge_end_time_ids: None,
            renumber: false,
            drop_self_loops: false,
            drop_multi_edges: false,
            symmetrize: false,
            do_expensive_check: false,
        }
    }

    /// Set the optional edge weights array.
    ///
    /// Must have the same length as `src`/`dst`.
    pub fn weights(&mut self, weights: &'a DeviceArrayView<'a>) -> &mut Self {
        self.weights = Some(weights);
        self
    }

    /// Set the optional edge ids array.
    ///
    /// Must have the same length as `src`/`dst`.
    pub fn edge_ids(&mut self, edge_ids: &'a DeviceArrayView<'a>) -> &mut Self {
        self.edge_ids = Some(edge_ids);
        self
    }

    /// Set the optional edge type ids array.
    ///
    /// Must have the same length as `src`/`dst`.
    pub fn edge_type_ids(&mut self, edge_type_ids: &'a DeviceArrayView<'a>) -> &mut Self {
        self.edge_type_ids = Some(edge_type_ids);
        self
    }

    /// Set the optional vertex array.
    ///
    /// When `None` (the default), vertices are auto-detected from the edge list.
    pub fn vertices(&mut self, vertices: &'a DeviceArrayView<'a>) -> &mut Self {
        self.vertices = Some(vertices);
        self
    }

    /// Set the optional edge start time ids array.
    ///
    /// Must have the same length as `src`/`dst`.
    pub fn edge_start_time_ids(&mut self, times: &'a DeviceArrayView<'a>) -> &mut Self {
        self.edge_start_time_ids = Some(times);
        self
    }

    /// Set the optional edge end time ids array.
    ///
    /// Must have the same length as `src`/`dst`.
    pub fn edge_end_time_ids(&mut self, times: &'a DeviceArrayView<'a>) -> &mut Self {
        self.edge_end_time_ids = Some(times);
        self
    }

    /// Whether to renumber vertices (default: `false`).
    ///
    /// Strongly recommended when vertex IDs are not sequential from 0.
    pub fn renumber(&mut self, renumber: bool) -> &mut Self {
        self.renumber = renumber;
        self
    }

    /// Whether to drop self-loop edges during graph construction (default: `false`).
    pub fn drop_self_loops(&mut self, drop_self_loops: bool) -> &mut Self {
        self.drop_self_loops = drop_self_loops;
        self
    }

    /// Whether to drop multi-edges during graph construction (default: `false`).
    pub fn drop_multi_edges(&mut self, drop_multi_edges: bool) -> &mut Self {
        self.drop_multi_edges = drop_multi_edges;
        self
    }

    /// Whether to symmetrize the graph (default: `false`).
    pub fn symmetrize(&mut self, symmetrize: bool) -> &mut Self {
        self.symmetrize = symmetrize;
        self
    }

    /// Whether to perform expensive validation checks (default: `false`).
    pub fn do_expensive_check(&mut self, do_expensive_check: bool) -> &mut Self {
        self.do_expensive_check = do_expensive_check;
        self
    }

    /// Validate inputs and construct a [`Graph<Forward>`] on the GPU.
    ///
    /// The graph is stored in forward (non-transposed) form, suitable for
    /// algorithms like [`bfs`](crate::traversal::bfs::bfs).
    pub fn build_forward(&self, handle: &ResourceHandle) -> Result<Graph<Forward>> {
        self.build_inner(handle, false)
    }

    /// Validate inputs and construct a [`Graph<Transposed>`] on the GPU.
    ///
    /// The graph is stored in transposed form, suitable for algorithms like
    /// [`pagerank`](crate::centrality::pagerank::pagerank) and
    /// [`in_degrees`](crate::structure::degrees::in_degrees).
    pub fn build_transposed(&self, handle: &ResourceHandle) -> Result<Graph<Transposed>> {
        self.build_inner(handle, true)
    }

    /// Internal build logic shared by [`build_forward`](Self::build_forward)
    /// and [`build_transposed`](Self::build_transposed).
    pub(crate) fn build_inner<T>(&self, handle: &ResourceHandle, store_transposed: bool) -> Result<Graph<T>> {
        self.validate()?;

        let c_props = self.properties.into();
        let vertices_ptr = self.vertices.map(|v| v.as_ptr() as *const _).unwrap_or(ptr::null());
        let weights_ptr = self.weights.map(|w| w.as_ptr() as *const _).unwrap_or(ptr::null());
        let edge_ids_ptr = self.edge_ids.map(|e| e.as_ptr() as *const _).unwrap_or(ptr::null());
        let edge_type_ids_ptr = self.edge_type_ids.map(|e| e.as_ptr() as *const _).unwrap_or(ptr::null());
        let start_times_ptr = self.edge_start_time_ids.map(|t| t.as_ptr() as *const _).unwrap_or(ptr::null());
        let end_times_ptr = self.edge_end_time_ids.map(|t| t.as_ptr() as *const _).unwrap_or(ptr::null());

        let mut graph: *mut cugraph_graph_t = ptr::null_mut();
        let mut error: *mut cugraph_error_t = ptr::null_mut();

        unsafe {
            let code = cugraph_graph_create_with_times_sg(
                handle.as_ptr(),
                &c_props,
                vertices_ptr,
                self.src.as_ptr(),
                self.dst.as_ptr(),
                weights_ptr,
                edge_ids_ptr,
                edge_type_ids_ptr,
                start_times_ptr,
                end_times_ptr,
                to_bool_t(store_transposed),
                to_bool_t(self.renumber),
                to_bool_t(self.drop_self_loops),
                to_bool_t(self.drop_multi_edges),
                to_bool_t(self.symmetrize),
                to_bool_t(self.do_expensive_check),
                &mut graph,
                &mut error,
            );
            check_error(code, error, "GraphBuilder::build")?;
        }

        Ok(Graph { raw: graph, _nosend_nosync: std::marker::PhantomData, _transposition: std::marker::PhantomData })
    }

    /// Validate that all array lengths are consistent.
    fn validate(&self) -> Result<()> {
        let edge_count = self.src.len();

        if self.dst.len() != edge_count {
            return Err(Error::new(ErrorKind::InvalidInput, "GraphBuilder::build")
                .with_message("src and dst must have the same number of elements")
                .with_context("src_len", edge_count.to_string())
                .with_context("dst_len", self.dst.len().to_string()));
        }

        Self::check_optional_len(self.weights, edge_count, "weights")?;
        Self::check_optional_len(self.edge_ids, edge_count, "edge_ids")?;
        Self::check_optional_len(self.edge_type_ids, edge_count, "edge_type_ids")?;
        Self::check_optional_len(self.edge_start_time_ids, edge_count, "edge_start_time_ids")?;
        Self::check_optional_len(self.edge_end_time_ids, edge_count, "edge_end_time_ids")?;

        Ok(())
    }

    /// Check that an optional array, if present, has the expected length.
    fn check_optional_len(array: Option<&DeviceArrayView<'_>>, expected: usize, name: &'static str) -> Result<()> {
        if let Some(arr) = array {
            if arr.len() != expected {
                return Err(Error::new(ErrorKind::InvalidInput, "GraphBuilder::build")
                    .with_message(format!("{name} must have the same number of elements as src/dst"))
                    .with_context("src_len", expected.to_string())
                    .with_context(name, arr.len().to_string()));
            }
        }
        Ok(())
    }
}

impl<T> Drop for Graph<T> {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { cugraph_graph_free(self.raw) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{array::device_array_from_host, error::ErrorKind, handle::ResourceHandle, types::DataType};

    #[test]
    fn test_graph_creation_triangle() {
        let handle = ResourceHandle::new().expect("handle");

        // Triangle: 0->1, 1->2, 2->0
        let src_data: Vec<i32> = vec![0, 1, 2];
        let dst_data: Vec<i32> = vec![1, 2, 0];

        let src = device_array_from_host(&handle, &src_data, DataType::Int32).expect("src");
        let dst = device_array_from_host(&handle, &dst_data, DataType::Int32).expect("dst");

        let props = GraphProperties { is_symmetric: false, is_multigraph: false };

        let graph = GraphBuilder::new(props, &src.view(), &dst.view()).build_forward(&handle);
        assert!(graph.is_ok(), "graph creation failed: {:?}", graph.err());
    }

    #[test]
    fn test_graph_creation_rejects_mismatched_src_dst_lengths() {
        let handle = ResourceHandle::new().expect("handle");

        let src = device_array_from_host(&handle, &[0i32, 1, 2], DataType::Int32).expect("src");
        let dst = device_array_from_host(&handle, &[1i32, 2], DataType::Int32).expect("dst");

        let props = GraphProperties { is_symmetric: false, is_multigraph: false };

        let err = match GraphBuilder::new(props, &src.view(), &dst.view()).build_forward(&handle) {
            Ok(_) => panic!("should reject mismatched edge-list lengths"),
            Err(err) => err,
        };
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn test_graph_creation_with_edge_ids_and_edge_type_ids() {
        let handle = ResourceHandle::new().expect("handle");

        // Triangle: 0->1, 1->2, 2->0 with edge IDs and edge type IDs
        let src_data: Vec<i32> = vec![0, 1, 2];
        let dst_data: Vec<i32> = vec![1, 2, 0];
        let edge_id_data: Vec<i32> = vec![100, 101, 102];
        let edge_type_data: Vec<i32> = vec![0, 1, 0];

        let src = device_array_from_host(&handle, &src_data, DataType::Int32).expect("src");
        let dst = device_array_from_host(&handle, &dst_data, DataType::Int32).expect("dst");
        let edge_ids = device_array_from_host(&handle, &edge_id_data, DataType::Int32).expect("edge_ids");
        let edge_types = device_array_from_host(&handle, &edge_type_data, DataType::Int32).expect("edge_types");

        let props = GraphProperties { is_symmetric: false, is_multigraph: false };

        let graph = GraphBuilder::new(props, &src.view(), &dst.view())
            .edge_ids(&edge_ids.view())
            .edge_type_ids(&edge_types.view())
            .renumber(true) // renumber required when using edge properties
            .build_forward(&handle);
        assert!(graph.is_ok(), "graph creation with edge_ids and edge_type_ids failed: {:?}", graph.err());
    }

    #[test]
    fn test_graph_creation_with_edge_ids_only() {
        let handle = ResourceHandle::new().expect("handle");

        let src_data: Vec<i32> = vec![0, 1, 2];
        let dst_data: Vec<i32> = vec![1, 2, 0];
        let edge_id_data: Vec<i32> = vec![10, 20, 30];

        let src = device_array_from_host(&handle, &src_data, DataType::Int32).expect("src");
        let dst = device_array_from_host(&handle, &dst_data, DataType::Int32).expect("dst");
        let edge_ids = device_array_from_host(&handle, &edge_id_data, DataType::Int32).expect("edge_ids");

        let props = GraphProperties { is_symmetric: false, is_multigraph: false };

        let graph = GraphBuilder::new(props, &src.view(), &dst.view())
            .edge_ids(&edge_ids.view())
            .renumber(true)
            .build_forward(&handle);
        assert!(graph.is_ok(), "graph creation with edge_ids only failed: {:?}", graph.err());
    }

    #[test]
    fn test_graph_creation_with_edge_type_ids_only() {
        let handle = ResourceHandle::new().expect("handle");

        let src_data: Vec<i32> = vec![0, 1, 2];
        let dst_data: Vec<i32> = vec![1, 2, 0];
        let edge_type_data: Vec<i32> = vec![0, 1, 2];

        let src = device_array_from_host(&handle, &src_data, DataType::Int32).expect("src");
        let dst = device_array_from_host(&handle, &dst_data, DataType::Int32).expect("dst");
        let edge_types = device_array_from_host(&handle, &edge_type_data, DataType::Int32).expect("edge_types");

        let props = GraphProperties { is_symmetric: false, is_multigraph: false };

        let graph = GraphBuilder::new(props, &src.view(), &dst.view())
            .edge_type_ids(&edge_types.view())
            .renumber(true)
            .build_forward(&handle);
        assert!(graph.is_ok(), "graph creation with edge_type_ids only failed: {:?}", graph.err());
    }

    #[test]
    fn test_graph_creation_with_weights_and_edge_properties() {
        let handle = ResourceHandle::new().expect("handle");

        let src_data: Vec<i32> = vec![0, 1, 2, 3];
        let dst_data: Vec<i32> = vec![1, 2, 3, 0];
        let wgt_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let edge_id_data: Vec<i32> = vec![0, 1, 2, 3];
        let edge_type_data: Vec<i32> = vec![0, 0, 1, 1];

        let src = device_array_from_host(&handle, &src_data, DataType::Int32).expect("src");
        let dst = device_array_from_host(&handle, &dst_data, DataType::Int32).expect("dst");
        let wgt = device_array_from_host(&handle, &wgt_data, DataType::Float32).expect("wgt");
        let edge_ids = device_array_from_host(&handle, &edge_id_data, DataType::Int32).expect("edge_ids");
        let edge_types = device_array_from_host(&handle, &edge_type_data, DataType::Int32).expect("edge_types");

        let props = GraphProperties { is_symmetric: false, is_multigraph: false };

        let graph = GraphBuilder::new(props, &src.view(), &dst.view())
            .weights(&wgt.view())
            .edge_ids(&edge_ids.view())
            .edge_type_ids(&edge_types.view())
            .renumber(true)
            .build_forward(&handle);
        assert!(graph.is_ok(), "graph creation with weights + edge properties failed: {:?}", graph.err());
    }

    #[test]
    fn test_graph_creation_rejects_mismatched_edge_ids_length() {
        let handle = ResourceHandle::new().expect("handle");

        let src = device_array_from_host(&handle, &[0i32, 1, 2], DataType::Int32).expect("src");
        let dst = device_array_from_host(&handle, &[1i32, 2, 0], DataType::Int32).expect("dst");
        let edge_ids = device_array_from_host(&handle, &[10i32, 20], DataType::Int32).expect("edge_ids");

        let props = GraphProperties { is_symmetric: false, is_multigraph: false };

        let err = GraphBuilder::new(props, &src.view(), &dst.view())
            .edge_ids(&edge_ids.view())
            .build_forward(&handle)
            .expect_err("should reject mismatched edge_ids length");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.message().contains("edge_ids"));
    }

    #[test]
    fn test_graph_creation_rejects_mismatched_edge_type_ids_length() {
        let handle = ResourceHandle::new().expect("handle");

        let src = device_array_from_host(&handle, &[0i32, 1, 2], DataType::Int32).expect("src");
        let dst = device_array_from_host(&handle, &[1i32, 2, 0], DataType::Int32).expect("dst");
        let edge_types = device_array_from_host(&handle, &[0i32, 1], DataType::Int32).expect("edge_types");

        let props = GraphProperties { is_symmetric: false, is_multigraph: false };

        let err = GraphBuilder::new(props, &src.view(), &dst.view())
            .edge_type_ids(&edge_types.view())
            .build_forward(&handle)
            .expect_err("should reject mismatched edge_type_ids length");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.message().contains("edge_type_ids"));
    }
}
