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
//! - **[`Forward`]** — required by [`Graph::bfs`]
//! - **[`Transposed`]** — required by [`Graph::pagerank`],
//!   [`Graph::in_degrees`], and [`Graph::degrees`]
//!
//! Algorithms that work with either layout (e.g. betweenness centrality,
//! Force Atlas 2, out_degrees) are generic over `T: Transposition`.
//!
//! # Example
//!
//! ```no_run
//! # use cugraph::{DeviceArray, GraphBuilder, ResourceHandle, Transposed};
//! # let handle = ResourceHandle::new().unwrap();
//! # let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).unwrap();
//! # let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 0]).unwrap();
//! let graph = GraphBuilder::new(&src.view(), &dst.view())
//!     .renumber()
//!     .build::<Transposed>(&handle)
//!     .unwrap();
//! ```

use std::ptr;

use cugraph_sys::{
    cugraph_error_t, cugraph_graph_create_with_times_sg, cugraph_graph_free, cugraph_graph_t,
    cugraph_type_erased_device_array_view_t, cugraph_type_erased_device_array_view_type,
};

#[cfg(feature = "cudf-interop")]
use crate::array::RawDeviceArrayView;
use crate::{
    array::DeviceArrayView,
    error::{Error, ErrorKind, Result, check_error},
    handle::ResourceHandle,
    types::{CugraphElement, DataType, GraphProperties, to_bool_t},
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
/// Use [`GraphBuilder::build::<Forward>`](GraphBuilder::build) or
/// [`GraphBuilder::build::<Transposed>`](GraphBuilder::build) to construct a
/// graph with the appropriate type state.
///
/// The graph carries runtime dtype information for its vertex IDs
/// ([`vertex_dtype`](Graph::vertex_dtype)) and edge weights
/// ([`weight_dtype`](Graph::weight_dtype)), enabling downstream code to
/// interpret algorithm results without guessing.
pub struct Graph<T = Forward> {
    pub(crate) raw: *mut cugraph_graph_t,
    vertex_dtype: DataType,
    weight_dtype: DataType,
    _nosend_nosync: std::marker::PhantomData<*mut ()>,
    _transposition: std::marker::PhantomData<T>,
}

impl<T> Graph<T> {
    /// The data type of vertex IDs in this graph (`Int32` or `Int64`).
    pub fn vertex_dtype(&self) -> DataType {
        self.vertex_dtype
    }

    /// The data type of edge weights in this graph (`Float32` or `Float64`).
    ///
    /// For unweighted graphs, this defaults to `Float32`.
    pub fn weight_dtype(&self) -> DataType {
        self.weight_dtype
    }

    fn validate_vertex_input_dtype(&self, dtype: DataType, op: &'static str, input_name: &'static str) -> Result<()> {
        if !dtype.is_integer() {
            return Err(Error::new(ErrorKind::UnsupportedType, op)
                .with_message(format!("{input_name} must use Int32 or Int64 vertex ids, got {dtype}")));
        }

        if dtype != self.vertex_dtype() {
            return Err(Error::new(ErrorKind::UnsupportedType, op).with_message(format!(
                "{input_name} dtype {dtype} does not match graph vertex dtype {}",
                self.vertex_dtype()
            )));
        }

        Ok(())
    }

    /// Validate that a `DegreeVertices::Subset` is compatible with this
    /// graph's vertex dtype. `All` always passes.
    fn validate_degree_vertices<V: CugraphElement>(
        &self,
        vertices: &crate::structure::DegreeVertices<'_, V>,
        op: &'static str,
    ) -> Result<()> {
        if let crate::structure::DegreeVertices::Subset(_) = vertices {
            self.validate_vertex_input_dtype(V::data_type(), op, "degree vertex subset")?;
        }
        Ok(())
    }
}

impl<T> std::fmt::Debug for Graph<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Graph")
            .field("vertex_dtype", &self.vertex_dtype)
            .field("weight_dtype", &self.weight_dtype)
            .finish_non_exhaustive()
    }
}

/// Builder for constructing a [`Graph`] from edge lists.
///
/// Provides a fluent by-value API for setting optional arrays and
/// configuration flags. All setters consume `self` and return `Self`.
///
/// # Required parameters
///
/// - `src` — source vertex ids (device array view)
/// - `dst` — destination vertex ids (device array view)
///
/// Properties default to a directed simple graph. Use
/// [`properties()`](Self::properties) to override.
///
/// # Example
///
/// ```no_run
/// # use cugraph::{DeviceArray, GraphBuilder, ResourceHandle, Transposed};
/// # let handle = ResourceHandle::new().unwrap();
/// # let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).unwrap();
/// # let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 0]).unwrap();
/// # let wgt = DeviceArray::from_slice(&handle, &[1.0f32, 2.0, 3.0]).unwrap();
/// let graph = GraphBuilder::new(&src.view(), &dst.view())
///     .weights(&wgt.view())
///     .renumber()
///     .build::<Transposed>(&handle)?;
/// # Ok::<(), cugraph::Error>(())
/// ```
#[must_use]
#[derive(Debug)]
pub struct GraphBuilder<'a> {
    properties: GraphProperties,
    src: *mut cugraph_type_erased_device_array_view_t,
    src_len: usize,
    dst: *mut cugraph_type_erased_device_array_view_t,
    dst_len: usize,
    weights: Option<(*mut cugraph_type_erased_device_array_view_t, usize)>,
    edge_ids: Option<(*mut cugraph_type_erased_device_array_view_t, usize)>,
    edge_type_ids: Option<(*mut cugraph_type_erased_device_array_view_t, usize)>,
    vertices: Option<(*mut cugraph_type_erased_device_array_view_t, usize)>,
    edge_start_time_ids: Option<(*mut cugraph_type_erased_device_array_view_t, usize)>,
    edge_end_time_ids: Option<(*mut cugraph_type_erased_device_array_view_t, usize)>,
    vertex_dtype: Option<DataType>,
    weight_dtype: Option<DataType>,
    renumber: bool,
    drop_self_loops: bool,
    drop_multi_edges: bool,
    symmetrize: bool,
    do_expensive_check: bool,
    _lifetime: std::marker::PhantomData<&'a ()>,
}

impl<'a> Clone for GraphBuilder<'a> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a> Copy for GraphBuilder<'a> {}

impl<'a> GraphBuilder<'a> {
    /// Create a new builder with the required edge arrays.
    ///
    /// Properties default to a directed simple graph. All optional arrays
    /// default to `None` and all boolean flags default to `false`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use cugraph::{DeviceArray, Forward, GraphBuilder, ResourceHandle};
    /// # let handle = ResourceHandle::new().unwrap();
    /// let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).unwrap();
    /// let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 0]).unwrap();
    ///
    /// let graph = GraphBuilder::new(&src.view(), &dst.view())
    ///     .renumber()
    ///     .build::<Forward>(&handle)
    ///     .unwrap();
    /// ```
    pub fn new<V: CugraphElement>(src: &'a DeviceArrayView<'a, V>, dst: &'a DeviceArrayView<'a, V>) -> Self {
        Self {
            properties: GraphProperties::default(),
            src: src.as_ptr(),
            src_len: src.len(),
            dst: dst.as_ptr(),
            dst_len: dst.len(),
            weights: None,
            edge_ids: None,
            edge_type_ids: None,
            vertices: None,
            edge_start_time_ids: None,
            edge_end_time_ids: None,
            vertex_dtype: Some(V::data_type()),
            weight_dtype: None,
            renumber: false,
            drop_self_loops: false,
            drop_multi_edges: false,
            symmetrize: false,
            do_expensive_check: false,
            _lifetime: std::marker::PhantomData,
        }
    }

    /// Set the graph properties (symmetric / multigraph).
    ///
    /// Defaults to [`GraphProperties::directed()`] if not called.
    pub fn properties(mut self, properties: GraphProperties) -> Self {
        self.properties = properties;
        self
    }

    /// Set the optional edge weights array.
    ///
    /// Must have the same length as `src`/`dst`.
    pub fn weights<W: CugraphElement>(mut self, weights: &'a DeviceArrayView<'a, W>) -> Self {
        self.weights = Some((weights.as_ptr(), weights.len()));
        self.weight_dtype = Some(W::data_type());
        self
    }

    /// Set the optional edge ids array.
    ///
    /// Must have the same length as `src`/`dst`.
    pub fn edge_ids<E: CugraphElement>(mut self, edge_ids: &'a DeviceArrayView<'a, E>) -> Self {
        self.edge_ids = Some((edge_ids.as_ptr(), edge_ids.len()));
        self
    }

    /// Set the optional edge type ids array.
    ///
    /// Must have the same length as `src`/`dst`.
    pub fn edge_type_ids<E: CugraphElement>(mut self, edge_type_ids: &'a DeviceArrayView<'a, E>) -> Self {
        self.edge_type_ids = Some((edge_type_ids.as_ptr(), edge_type_ids.len()));
        self
    }

    /// Set the optional vertex array.
    ///
    /// When `None` (the default), vertices are auto-detected from the edge list.
    pub fn vertices<V: CugraphElement>(mut self, vertices: &'a DeviceArrayView<'a, V>) -> Self {
        self.vertices = Some((vertices.as_ptr(), vertices.len()));
        self
    }

    /// Set the optional edge start time ids array.
    ///
    /// Must have the same length as `src`/`dst`.
    pub fn edge_start_time_ids<E: CugraphElement>(mut self, times: &'a DeviceArrayView<'a, E>) -> Self {
        self.edge_start_time_ids = Some((times.as_ptr(), times.len()));
        self
    }

    /// Set the optional edge end time ids array.
    ///
    /// Must have the same length as `src`/`dst`.
    pub fn edge_end_time_ids<E: CugraphElement>(mut self, times: &'a DeviceArrayView<'a, E>) -> Self {
        self.edge_end_time_ids = Some((times.as_ptr(), times.len()));
        self
    }

    /// Enable vertex renumbering (default: disabled).
    ///
    /// Strongly recommended when vertex IDs are not sequential from 0.
    pub fn renumber(mut self) -> Self {
        self.renumber = true;
        self
    }

    /// Enable dropping self-loop edges during graph construction (default: disabled).
    pub fn drop_self_loops(mut self) -> Self {
        self.drop_self_loops = true;
        self
    }

    /// Enable dropping multi-edges during graph construction (default: disabled).
    pub fn drop_multi_edges(mut self) -> Self {
        self.drop_multi_edges = true;
        self
    }

    /// Enable graph symmetrization (default: disabled).
    ///
    /// Cannot be combined with [`edge_ids`](Self::edge_ids) or
    /// [`edge_type_ids`](Self::edge_type_ids).
    pub fn symmetrize(mut self) -> Self {
        self.symmetrize = true;
        self
    }

    /// Enable expensive validation checks (default: disabled).
    pub fn do_expensive_check(mut self) -> Self {
        self.do_expensive_check = true;
        self
    }

    /// Validate inputs and construct a [`Graph<T>`] on the GPU.
    ///
    /// The type parameter `T` determines the transposition state:
    ///
    /// - `build::<Forward>` — forward form, for [`bfs`](crate::traversal::bfs::bfs)
    /// - `build::<Transposed>` — transposed form, for [`pagerank`](crate::centrality::pagerank::pagerank)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - array lengths are inconsistent
    /// - `symmetrize` is combined with `edge_ids` or `edge_type_ids`
    /// - only one of `edge_start_time_ids` / `edge_end_time_ids` is set
    /// - the C API rejects the inputs
    pub fn build<T: Transposition>(&self, handle: &ResourceHandle) -> Result<Graph<T>> {
        self.build_inner(handle, T::store_transposed())
    }

    /// Internal build logic.
    pub(crate) fn build_inner<T>(&self, handle: &ResourceHandle, store_transposed: bool) -> Result<Graph<T>> {
        self.validate()?;

        // Resolve dtypes: prefer the statically-known value (from the typed
        // API), falling back to a runtime query of the raw view pointer (the
        // cudf_interop path).
        let vertex_dtype = match self.vertex_dtype {
            Some(dt) => dt,
            None => Self::view_dtype(self.src)?,
        };
        let weight_dtype = match self.weight_dtype {
            Some(dt) => dt,
            None => match self.weights {
                Some((ptr, _)) => Self::view_dtype(ptr)?,
                None => DataType::Float32,
            },
        };

        let c_props = self.properties.into();
        let vertices_ptr = self.vertices.map(|(p, _)| p as *const _).unwrap_or(ptr::null());
        let weights_ptr = self.weights.map(|(p, _)| p as *const _).unwrap_or(ptr::null());
        let edge_ids_ptr = self.edge_ids.map(|(p, _)| p as *const _).unwrap_or(ptr::null());
        let edge_type_ids_ptr = self.edge_type_ids.map(|(p, _)| p as *const _).unwrap_or(ptr::null());
        let start_times_ptr = self.edge_start_time_ids.map(|(p, _)| p as *const _).unwrap_or(ptr::null());
        let end_times_ptr = self.edge_end_time_ids.map(|(p, _)| p as *const _).unwrap_or(ptr::null());

        let mut graph: *mut cugraph_graph_t = ptr::null_mut();
        let mut error: *mut cugraph_error_t = ptr::null_mut();

        unsafe {
            let code = cugraph_graph_create_with_times_sg(
                handle.as_ptr(),
                &c_props,
                vertices_ptr,
                self.src,
                self.dst,
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

        Ok(Graph {
            raw: graph,
            vertex_dtype,
            weight_dtype,
            _nosend_nosync: std::marker::PhantomData,
            _transposition: std::marker::PhantomData,
        })
    }

    /// Build from raw (type-erased) view pointers. Used by cudf_interop.
    #[cfg(feature = "cudf-interop")]
    pub(crate) fn from_raw_views(
        properties: GraphProperties,
        src: &'a RawDeviceArrayView<'a>,
        dst: &'a RawDeviceArrayView<'a>,
    ) -> Self {
        Self {
            properties,
            src: src.as_ptr(),
            src_len: src.len(),
            dst: dst.as_ptr(),
            dst_len: dst.len(),
            weights: None,
            edge_ids: None,
            edge_type_ids: None,
            vertices: None,
            edge_start_time_ids: None,
            edge_end_time_ids: None,
            vertex_dtype: None,
            weight_dtype: None,
            renumber: false,
            drop_self_loops: false,
            drop_multi_edges: false,
            symmetrize: false,
            do_expensive_check: false,
            _lifetime: std::marker::PhantomData,
        }
    }

    /// Set weights from a raw view. Used by cudf_interop.
    #[cfg(feature = "cudf-interop")]
    pub(crate) fn weights_raw(mut self, raw: &'a RawDeviceArrayView<'a>) -> Self {
        self.weights = Some((raw.as_ptr(), raw.len()));
        self
    }

    /// Set renumber from a bool. Used by cudf_interop where the flag comes
    /// from a function parameter rather than a builder toggle.
    #[cfg(feature = "cudf-interop")]
    pub(crate) fn renumber_flag(mut self, renumber: bool) -> Self {
        self.renumber = renumber;
        self
    }

    /// Validate that all array lengths are consistent, option combinations
    /// are valid, and array dtypes are compatible with the cuGraph C API.
    fn validate(&self) -> Result<()> {
        let edge_count = self.src_len;

        if self.dst_len != edge_count {
            return Err(Error::new(ErrorKind::InvalidInput, "GraphBuilder::build")
                .with_message("src and dst must have the same number of elements")
                .with_context("src_len", edge_count.to_string())
                .with_context("dst_len", self.dst_len.to_string()));
        }

        Self::check_optional_len(self.weights, edge_count, "weights")?;
        Self::check_optional_len(self.edge_ids, edge_count, "edge_ids")?;
        Self::check_optional_len(self.edge_type_ids, edge_count, "edge_type_ids")?;
        Self::check_optional_len(self.edge_start_time_ids, edge_count, "edge_start_time_ids")?;
        Self::check_optional_len(self.edge_end_time_ids, edge_count, "edge_end_time_ids")?;

        // symmetrize is incompatible with edge_ids and edge_type_ids
        if self.symmetrize && self.edge_ids.is_some() {
            return Err(Error::new(ErrorKind::InvalidInput, "GraphBuilder::build")
                .with_message("symmetrize cannot be combined with edge_ids"));
        }
        if self.symmetrize && self.edge_type_ids.is_some() {
            return Err(Error::new(ErrorKind::InvalidInput, "GraphBuilder::build")
                .with_message("symmetrize cannot be combined with edge_type_ids"));
        }

        // temporal edges: both start and end must be provided together
        let has_start = self.edge_start_time_ids.is_some();
        let has_end = self.edge_end_time_ids.is_some();
        if has_start != has_end {
            let missing = if has_start { "edge_end_time_ids" } else { "edge_start_time_ids" };
            let present = if has_start { "edge_start_time_ids" } else { "edge_end_time_ids" };
            return Err(Error::new(ErrorKind::InvalidInput, "GraphBuilder::build").with_message(format!(
                "{present} was provided but {missing} was not; both must be provided or both omitted"
            )));
        }

        // ── dtype validation (safety net for the from_raw_views path) ────
        self.validate_dtypes()?;

        Ok(())
    }

    /// Validate that array dtypes are compatible with the cuGraph C API.
    ///
    /// For the typed public API (`GraphBuilder::new`), the compiler already
    /// guarantees type safety through `DeviceArrayView<T>`. This validation
    /// acts as a safety net for the `from_raw_views()` internal path, which
    /// takes type-erased `RawDeviceArrayView` and bypasses compile-time type
    /// checking.
    fn validate_dtypes(&self) -> Result<()> {
        let src_dtype = Self::view_dtype(self.src)?;
        let dst_dtype = Self::view_dtype(self.dst)?;

        // src and dst must have the same dtype
        if src_dtype != dst_dtype {
            return Err(Error::new(ErrorKind::InvalidInput, "GraphBuilder::build")
                .with_message("src and dst must have the same dtype")
                .with_context("src_dtype", src_dtype.to_string())
                .with_context("dst_dtype", dst_dtype.to_string()));
        }

        // src/dst must be integer (i32 or i64)
        if !src_dtype.is_integer() {
            return Err(Error::new(ErrorKind::InvalidInput, "GraphBuilder::build")
                .with_message(format!("src/dst dtype must be Int32 or Int64, got {src_dtype}")));
        }

        // weights must be float (f32 or f64)
        if let Some((ptr, _)) = self.weights {
            let dtype = Self::view_dtype(ptr)?;
            if !dtype.is_float() {
                return Err(Error::new(ErrorKind::InvalidInput, "GraphBuilder::build")
                    .with_message(format!("weights dtype must be Float32 or Float64, got {dtype}")));
            }
        }

        // edge_ids must be integer (i32 or i64)
        if let Some((ptr, _)) = self.edge_ids {
            let dtype = Self::view_dtype(ptr)?;
            if !dtype.is_integer() {
                return Err(Error::new(ErrorKind::InvalidInput, "GraphBuilder::build")
                    .with_message(format!("edge_ids dtype must be Int32 or Int64, got {dtype}")));
            }
        }

        // edge_type_ids must be i32
        if let Some((ptr, _)) = self.edge_type_ids {
            let dtype = Self::view_dtype(ptr)?;
            if dtype != DataType::Int32 {
                return Err(Error::new(ErrorKind::InvalidInput, "GraphBuilder::build")
                    .with_message(format!("edge_type_ids dtype must be Int32, got {dtype}")));
            }
        }

        // temporal arrays must be integer (i32 or i64)
        if let Some((ptr, _)) = self.edge_start_time_ids {
            let dtype = Self::view_dtype(ptr)?;
            if !dtype.is_integer() {
                return Err(Error::new(ErrorKind::InvalidInput, "GraphBuilder::build")
                    .with_message(format!("edge_start_time_ids dtype must be Int32 or Int64, got {dtype}")));
            }
        }
        if let Some((ptr, _)) = self.edge_end_time_ids {
            let dtype = Self::view_dtype(ptr)?;
            if !dtype.is_integer() {
                return Err(Error::new(ErrorKind::InvalidInput, "GraphBuilder::build")
                    .with_message(format!("edge_end_time_ids dtype must be Int32 or Int64, got {dtype}")));
            }
        }

        // temporal arrays must have the same dtype if both present
        if let (Some((start_ptr, _)), Some((end_ptr, _))) = (self.edge_start_time_ids, self.edge_end_time_ids) {
            let start_dtype = Self::view_dtype(start_ptr)?;
            let end_dtype = Self::view_dtype(end_ptr)?;
            if start_dtype != end_dtype {
                return Err(Error::new(ErrorKind::InvalidInput, "GraphBuilder::build")
                    .with_message("edge_start_time_ids and edge_end_time_ids must have the same dtype")
                    .with_context("start_dtype", start_dtype.to_string())
                    .with_context("end_dtype", end_dtype.to_string()));
            }
        }

        // vertices must be integer (i32 or i64) AND match src/dst dtype.
        // The upstream C API silently casts mismatched vertex types via
        // static_cast (copy_or_transform in graph_helper_impl.cuh), which
        // can cause silent truncation (e.g. i64 → i32). We reject the
        // mismatch here to prevent data corruption.
        if let Some((ptr, _)) = self.vertices {
            let dtype = Self::view_dtype(ptr)?;
            if !dtype.is_integer() {
                return Err(Error::new(ErrorKind::InvalidInput, "GraphBuilder::build")
                    .with_message(format!("vertices dtype must be Int32 or Int64, got {dtype}")));
            }
            if dtype != src_dtype {
                return Err(Error::new(ErrorKind::InvalidInput, "GraphBuilder::build")
                    .with_message(
                        "vertices must have the same dtype as src/dst to prevent silent truncation by the C API",
                    )
                    .with_context("vertices_dtype", dtype.to_string())
                    .with_context("src_dst_dtype", src_dtype.to_string()));
            }
        }

        Ok(())
    }

    /// Query the runtime dtype of a raw device array view pointer.
    ///
    /// Returns an error if the C API reports an unrecognised type id.
    fn view_dtype(ptr: *mut cugraph_type_erased_device_array_view_t) -> Result<DataType> {
        let c_type = unsafe { cugraph_type_erased_device_array_view_type(ptr) };
        DataType::try_from(c_type).map_err(|_| {
            Error::new(ErrorKind::UnsupportedType, "view_dtype")
                .with_message(format!("unknown C API dtype id: {}", c_type.0))
        })
    }

    /// Check that an optional array, if present, has the expected length.
    fn check_optional_len(
        array: Option<(*mut cugraph_type_erased_device_array_view_t, usize)>,
        expected: usize,
        name: &'static str,
    ) -> Result<()> {
        if let Some((_, len)) = array {
            if len != expected {
                return Err(Error::new(ErrorKind::InvalidInput, "GraphBuilder::build")
                    .with_message(format!("{name} must have the same number of elements as src/dst"))
                    .with_context("src_len", expected.to_string())
                    .with_context(name, len.to_string()));
            }
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Algorithm methods on Graph<T>
// ═══════════════════════════════════════════════════════════════════════════

impl Graph<Transposed> {
    /// Compute PageRank on this transposed graph with full options.
    ///
    /// Returns a [`PageRankResult`](crate::centrality::PageRankResult)
    /// containing vertex ids, scores, iteration count, and convergence status.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use cugraph::{DeviceArray, GraphBuilder, PageRankOptions, ResourceHandle, Transposed};
    /// # let handle = ResourceHandle::new().unwrap();
    /// # let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).unwrap();
    /// # let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 0]).unwrap();
    /// # let mut graph = GraphBuilder::new(&src.view(), &dst.view())
    /// #     .renumber()
    /// #     .build::<Transposed>(&handle).unwrap();
    /// let result = graph.pagerank(&handle, &PageRankOptions::default()).unwrap();
    /// let scores = result.scores().unwrap();
    /// ```
    pub fn pagerank<V: CugraphElement, W: CugraphElement>(
        &mut self,
        handle: &ResourceHandle,
        options: &crate::centrality::PageRankOptions<'_, V, W>,
    ) -> Result<crate::centrality::PageRankResult> {
        crate::centrality::pagerank::pagerank(handle, self, options)
    }

    /// Compute PageRank with only basic parameters (no optional inputs).
    ///
    /// Convenience wrapper around [`pagerank`](Self::pagerank) for the common
    /// case where no precomputed weights or initial guess are needed.
    pub fn pagerank_simple(
        &mut self,
        handle: &ResourceHandle,
        params: &crate::centrality::PageRankParams,
    ) -> Result<crate::centrality::PageRankResult> {
        crate::centrality::pagerank::pagerank_simple(handle, self, params)
    }

    /// Compute both in-degrees and out-degrees for vertices in this graph.
    ///
    /// Pass [`DegreeVertices::All`] to compute degrees for every vertex, or
    /// [`DegreeVertices::Subset`] to restrict to a specific set.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::UnsupportedType`] if `vertices` uses a vertex
    /// dtype that does not match the graph.
    pub fn degrees<V: CugraphElement>(
        &mut self,
        handle: &ResourceHandle,
        vertices: &crate::structure::DegreeVertices<'_, V>,
    ) -> Result<crate::structure::DegreesResult> {
        self.validate_degree_vertices(vertices, "Graph::degrees")?;
        crate::structure::degrees::degrees(handle, self, vertices, false)
    }

    /// Compute degrees with optional expensive input validation.
    pub fn degrees_checked<V: CugraphElement>(
        &mut self,
        handle: &ResourceHandle,
        vertices: &crate::structure::DegreeVertices<'_, V>,
        do_expensive_check: bool,
    ) -> Result<crate::structure::DegreesResult> {
        self.validate_degree_vertices(vertices, "Graph::degrees")?;
        crate::structure::degrees::degrees(handle, self, vertices, do_expensive_check)
    }

    /// Compute degrees for every vertex in the graph.
    pub fn degrees_all(&mut self, handle: &ResourceHandle) -> Result<crate::structure::DegreesResult> {
        self.degrees::<i32>(handle, &crate::structure::DegreeVertices::All)
    }

    /// Compute only in-degrees for vertices in this graph.
    ///
    /// Pass [`DegreeVertices::All`] to compute degrees for every vertex, or
    /// [`DegreeVertices::Subset`] to restrict to a specific set.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::UnsupportedType`] if `vertices` uses a vertex
    /// dtype that does not match the graph.
    pub fn in_degrees<V: CugraphElement>(
        &mut self,
        handle: &ResourceHandle,
        vertices: &crate::structure::DegreeVertices<'_, V>,
    ) -> Result<crate::structure::InDegreesResult> {
        self.validate_degree_vertices(vertices, "Graph::in_degrees")?;
        crate::structure::degrees::in_degrees(handle, self, vertices, false)
    }

    /// Compute in-degrees with optional expensive input validation.
    pub fn in_degrees_checked<V: CugraphElement>(
        &mut self,
        handle: &ResourceHandle,
        vertices: &crate::structure::DegreeVertices<'_, V>,
        do_expensive_check: bool,
    ) -> Result<crate::structure::InDegreesResult> {
        self.validate_degree_vertices(vertices, "Graph::in_degrees")?;
        crate::structure::degrees::in_degrees(handle, self, vertices, do_expensive_check)
    }

    /// Compute in-degrees for every vertex in the graph.
    pub fn in_degrees_all(&mut self, handle: &ResourceHandle) -> Result<crate::structure::InDegreesResult> {
        self.in_degrees::<i32>(handle, &crate::structure::DegreeVertices::All)
    }
}

impl Graph<Forward> {
    /// Perform breadth-first search from one or more source vertices.
    ///
    /// Returns a [`BfsResult`](crate::traversal::BfsResult) containing vertex
    /// ids, distances, and optionally predecessors.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::UnsupportedType`] if `sources` uses a vertex
    /// dtype that does not match the graph.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use cugraph::{BfsParams, DeviceArray, Forward, GraphBuilder, ResourceHandle};
    /// # let handle = ResourceHandle::new().unwrap();
    /// # let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).unwrap();
    /// # let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 0]).unwrap();
    /// # let mut graph = GraphBuilder::new(&src.view(), &dst.view())
    /// #     .renumber()
    /// #     .build::<Forward>(&handle).unwrap();
    /// let sources = DeviceArray::from_slice(&handle, &[0i32]).unwrap();
    /// let result = graph.bfs(&handle, &sources.view(), &BfsParams::default()).unwrap();
    /// let distances = result.distances().unwrap();
    /// ```
    pub fn bfs<V: CugraphElement>(
        &mut self,
        handle: &ResourceHandle,
        sources: &DeviceArrayView<'_, V>,
        params: &crate::traversal::BfsParams,
    ) -> Result<crate::traversal::BfsResult> {
        self.validate_vertex_input_dtype(V::data_type(), "Graph::bfs", "sources")?;
        crate::traversal::bfs::bfs(handle, self, sources, params)
    }
}

impl<T: Transposition> Graph<T> {
    /// Compute vertex betweenness centrality on this graph.
    ///
    /// Pass `vertex_list` as `None` for exact computation, or provide seed
    /// vertices for approximate computation.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::UnsupportedType`] if `vertex_list` uses a vertex
    /// dtype that does not match the graph.
    pub fn betweenness_centrality<V: CugraphElement>(
        &mut self,
        handle: &ResourceHandle,
        vertex_list: Option<&DeviceArrayView<'_, V>>,
        params: &crate::centrality::BetweennessCentralityParams,
    ) -> Result<crate::centrality::BetweennessCentralityResult> {
        if vertex_list.is_some() {
            self.validate_vertex_input_dtype(V::data_type(), "Graph::betweenness_centrality", "vertex_list")?;
        }
        crate::centrality::betweenness::betweenness_centrality(handle, self, vertex_list, params)
    }

    /// Compute exact betweenness centrality over all vertices.
    pub fn betweenness_centrality_all(
        &mut self,
        handle: &ResourceHandle,
        params: &crate::centrality::BetweennessCentralityParams,
    ) -> Result<crate::centrality::BetweennessCentralityResult> {
        crate::centrality::betweenness::betweenness_centrality::<T, i32>(handle, self, None, params)
    }

    /// Compute a Force Atlas 2 layout for this graph.
    ///
    /// Returns a [`LayoutResult`](crate::layout::LayoutResult) containing
    /// vertex ids and (x, y) coordinates.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use cugraph::*;
    /// # let handle = ResourceHandle::new().unwrap();
    /// # let mut rng = RngState::new(&handle, 42).unwrap();
    /// # let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).unwrap();
    /// # let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 0]).unwrap();
    /// # let mut graph = GraphBuilder::new(&src.view(), &dst.view())
    /// #     .renumber().build::<Forward>(&handle).unwrap();
    /// let options: ForceAtlas2Options<'_> = ForceAtlas2Options::default();
    /// let result = graph.force_atlas2(&handle, &mut rng, &options)?;
    /// # Ok::<(), Error>(())
    /// ```
    pub fn force_atlas2<V: CugraphElement, W: CugraphElement>(
        &mut self,
        handle: &ResourceHandle,
        rng_state: &mut crate::random::RngState,
        options: &crate::layout::ForceAtlas2Options<'_, V, W>,
    ) -> Result<crate::layout::LayoutResult> {
        crate::layout::force_atlas2::force_atlas2(handle, rng_state, self, options)
    }

    /// Compute only out-degrees for vertices in this graph.
    ///
    /// Pass [`DegreeVertices::All`] to compute degrees for every vertex, or
    /// [`DegreeVertices::Subset`] to restrict to a specific set.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::UnsupportedType`] if `vertices` uses a vertex
    /// dtype that does not match the graph.
    pub fn out_degrees<V: CugraphElement>(
        &mut self,
        handle: &ResourceHandle,
        vertices: &crate::structure::DegreeVertices<'_, V>,
    ) -> Result<crate::structure::OutDegreesResult> {
        self.validate_degree_vertices(vertices, "Graph::out_degrees")?;
        crate::structure::degrees::out_degrees(handle, self, vertices, false)
    }

    /// Compute out-degrees with optional expensive input validation.
    pub fn out_degrees_checked<V: CugraphElement>(
        &mut self,
        handle: &ResourceHandle,
        vertices: &crate::structure::DegreeVertices<'_, V>,
        do_expensive_check: bool,
    ) -> Result<crate::structure::OutDegreesResult> {
        self.validate_degree_vertices(vertices, "Graph::out_degrees")?;
        crate::structure::degrees::out_degrees(handle, self, vertices, do_expensive_check)
    }

    /// Compute out-degrees for every vertex in the graph.
    pub fn out_degrees_all(&mut self, handle: &ResourceHandle) -> Result<crate::structure::OutDegreesResult> {
        self.out_degrees::<i32>(handle, &crate::structure::DegreeVertices::All)
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
    use crate::{array::DeviceArray, error::ErrorKind, handle::ResourceHandle};

    #[test]
    fn test_graph_creation_triangle() {
        let handle = ResourceHandle::new().expect("handle");

        let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).expect("src");
        let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 0]).expect("dst");

        let graph = GraphBuilder::new(&src.view(), &dst.view()).build::<Forward>(&handle);
        assert!(graph.is_ok(), "graph creation failed: {:?}", graph.err());
    }

    #[test]
    fn test_graph_creation_rejects_mismatched_src_dst_lengths() {
        let handle = ResourceHandle::new().expect("handle");

        let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).expect("src");
        let dst = DeviceArray::from_slice(&handle, &[1i32, 2]).expect("dst");

        let err = match GraphBuilder::new(&src.view(), &dst.view()).build::<Forward>(&handle) {
            Ok(_) => panic!("should reject mismatched edge-list lengths"),
            Err(err) => err,
        };
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn test_graph_creation_with_edge_ids_and_edge_type_ids() {
        let handle = ResourceHandle::new().expect("handle");

        let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).expect("src");
        let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 0]).expect("dst");
        let edge_ids = DeviceArray::from_slice(&handle, &[100i32, 101, 102]).expect("edge_ids");
        let edge_types = DeviceArray::from_slice(&handle, &[0i32, 1, 0]).expect("edge_types");

        let graph = GraphBuilder::new(&src.view(), &dst.view())
            .edge_ids(&edge_ids.view())
            .edge_type_ids(&edge_types.view())
            .renumber()
            .build::<Forward>(&handle);
        assert!(graph.is_ok(), "graph creation with edge_ids and edge_type_ids failed: {:?}", graph.err());
    }

    #[test]
    fn test_graph_creation_with_weights_and_edge_properties() {
        let handle = ResourceHandle::new().expect("handle");

        let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2, 3]).expect("src");
        let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 3, 0]).expect("dst");
        let wgt = DeviceArray::from_slice(&handle, &[1.0f32, 2.0, 3.0, 4.0]).expect("wgt");
        let edge_ids = DeviceArray::from_slice(&handle, &[0i32, 1, 2, 3]).expect("edge_ids");
        let edge_types = DeviceArray::from_slice(&handle, &[0i32, 0, 1, 1]).expect("edge_types");

        let graph = GraphBuilder::new(&src.view(), &dst.view())
            .weights(&wgt.view())
            .edge_ids(&edge_ids.view())
            .edge_type_ids(&edge_types.view())
            .renumber()
            .build::<Forward>(&handle);
        assert!(graph.is_ok(), "graph creation with weights + edge properties failed: {:?}", graph.err());
    }

    #[test]
    fn test_graph_creation_rejects_mismatched_edge_ids_length() {
        let handle = ResourceHandle::new().expect("handle");

        let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).expect("src");
        let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 0]).expect("dst");
        let edge_ids = DeviceArray::from_slice(&handle, &[10i32, 20]).expect("edge_ids");

        let err = GraphBuilder::new(&src.view(), &dst.view())
            .edge_ids(&edge_ids.view())
            .build::<Forward>(&handle)
            .expect_err("should reject mismatched edge_ids length");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.message().contains("edge_ids"));
    }

    #[test]
    fn test_graph_creation_rejects_mismatched_edge_type_ids_length() {
        let handle = ResourceHandle::new().expect("handle");

        let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).expect("src");
        let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 0]).expect("dst");
        let edge_types = DeviceArray::from_slice(&handle, &[0i32, 1]).expect("edge_types");

        let err = GraphBuilder::new(&src.view(), &dst.view())
            .edge_type_ids(&edge_types.view())
            .build::<Forward>(&handle)
            .expect_err("should reject mismatched edge_type_ids length");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.message().contains("edge_type_ids"));
    }

    #[test]
    fn test_rejects_symmetrize_with_edge_ids() {
        let handle = ResourceHandle::new().expect("handle");

        let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).expect("src");
        let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 0]).expect("dst");
        let eids = DeviceArray::from_slice(&handle, &[10i32, 20, 30]).expect("eids");

        let err = GraphBuilder::new(&src.view(), &dst.view())
            .edge_ids(&eids.view())
            .symmetrize()
            .build::<Forward>(&handle)
            .expect_err("should reject symmetrize + edge_ids");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.message().contains("symmetrize"));
        assert!(err.message().contains("edge_ids"));
    }

    #[test]
    fn test_rejects_symmetrize_with_edge_type_ids() {
        let handle = ResourceHandle::new().expect("handle");

        let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).expect("src");
        let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 0]).expect("dst");
        let etids = DeviceArray::from_slice(&handle, &[0i32, 1, 0]).expect("etids");

        let err = GraphBuilder::new(&src.view(), &dst.view())
            .edge_type_ids(&etids.view())
            .symmetrize()
            .build::<Forward>(&handle)
            .expect_err("should reject symmetrize + edge_type_ids");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.message().contains("symmetrize"));
        assert!(err.message().contains("edge_type_ids"));
    }

    #[test]
    fn test_rejects_start_time_without_end_time() {
        let handle = ResourceHandle::new().expect("handle");

        let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).expect("src");
        let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 0]).expect("dst");
        let start = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).expect("start");

        let err = GraphBuilder::new(&src.view(), &dst.view())
            .edge_start_time_ids(&start.view())
            .build::<Forward>(&handle)
            .expect_err("should reject start_time without end_time");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.message().contains("edge_end_time_ids"));
        assert!(err.message().contains("was not"));
    }

    #[test]
    fn test_builder_defaults_to_directed() {
        let handle = ResourceHandle::new().expect("handle");

        let src = DeviceArray::from_slice(&handle, &[0i32, 1, 2]).expect("src");
        let dst = DeviceArray::from_slice(&handle, &[1i32, 2, 0]).expect("dst");

        let src_view = src.view();
        let dst_view = dst.view();
        let builder = GraphBuilder::new(&src_view, &dst_view);
        let debug = format!("{:?}", builder);

        // properties defaults to directed simple graph
        assert!(debug.contains("is_symmetric: false"), "should default to directed");
        assert!(debug.contains("is_multigraph: false"), "should default to simple graph");
    }
}
