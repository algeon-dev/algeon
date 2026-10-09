//! Low-level graph construction from device edge lists.

use std::{
    cell::{Cell, RefCell},
    marker::PhantomData,
    ptr,
};

use cugraph_sys::{
    cugraph_error_t, cugraph_graph_create_with_times_sg, cugraph_graph_t, cugraph_type_erased_device_array_view_t,
    cugraph_type_erased_device_array_view_type,
};

use super::{Graph, Transposition};
use crate::{
    array::{DeviceArrayView, RawDeviceArrayView},
    error::{Error, ErrorKind, Result, check_error, require_non_null},
    handle::EnteredContext,
    types::{CugraphElement, DataType, GraphProperties, to_bool_t},
};

/// Builder for constructing a [`Graph`] from edge lists.
///
/// Provides a fluent by-value API for setting optional arrays and
/// configuration flags. All setters consume `self` and return `Self`.
///
/// Most applications should prefer the higher-level constructors in
/// [`crate::cudf_interop`] so the primary data model stays in cudf
/// [`Series`](cudf::Series) or [`DataFrame`](cudf::DataFrame). Use
/// `GraphBuilder` when you already have typed [`DeviceArrayView`] edge columns
/// and want direct control over graph-construction options.
///
/// # Required parameters
///
/// - `src`: source vertex ids (device array view)
/// - `dst`: destination vertex ids (device array view)
///
/// Properties default to a directed simple graph. Use
/// [`properties()`](Self::properties) to override.
///
/// # Example
///
/// ```no_run
/// # use cudf::Runtime;
/// # use cugraph::{DeviceArray, GraphBuilder, ResourceHandle, Transposed};
/// # let runtime = std::sync::Arc::new(Runtime::new().unwrap());
/// # let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).unwrap();
/// # let entered = handle.enter("GraphBuilder docs").unwrap();
/// # let src = DeviceArray::from_slice(&entered, &[0i32, 1, 2]).unwrap();
/// # let dst = DeviceArray::from_slice(&entered, &[1i32, 2, 0]).unwrap();
/// # let wgt = DeviceArray::from_slice(&entered, &[1.0f32, 2.0, 3.0]).unwrap();
/// let src_view = src.view().unwrap();
/// let dst_view = dst.view().unwrap();
/// let wgt_view = wgt.view().unwrap();
/// let graph = GraphBuilder::new(&src_view, &dst_view)
///     .weights(&wgt_view)
///     .renumber()
///     .build::<Transposed>(&entered)?;
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
    pub(super) vertex_dtype: DataType,
    pub(super) weight_dtype: DataType,
    renumber: bool,
    drop_self_loops: bool,
    drop_multi_edges: bool,
    symmetrize: bool,
    do_expensive_check: bool,
    _lifetime: PhantomData<&'a ()>,
}

impl Clone for GraphBuilder<'_> {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for GraphBuilder<'_> {}

impl<'a> GraphBuilder<'a> {
    /// Create a new builder with the required edge arrays.
    ///
    /// Properties default to a directed simple graph. All optional arrays
    /// default to `None` and all boolean flags default to `false`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use cudf::Runtime;
    /// # use cugraph::{DeviceArray, Forward, GraphBuilder, ResourceHandle};
    /// # let runtime = std::sync::Arc::new(Runtime::new().unwrap());
    /// # let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).unwrap();
    /// # let entered = handle.enter("GraphBuilder::new docs").unwrap();
    /// let src = DeviceArray::from_slice(&entered, &[0i32, 1, 2]).unwrap();
    /// let dst = DeviceArray::from_slice(&entered, &[1i32, 2, 0]).unwrap();
    ///
    /// let src_view = src.view().unwrap();
    /// let dst_view = dst.view().unwrap();
    /// let graph = GraphBuilder::new(&src_view, &dst_view)
    ///     .renumber()
    ///     .build::<Forward>(&entered)
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
            vertex_dtype: V::data_type(),
            weight_dtype: DataType::Float32,
            renumber: false,
            drop_self_loops: false,
            drop_multi_edges: false,
            symmetrize: false,
            do_expensive_check: false,
            _lifetime: PhantomData,
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
        self.weight_dtype = W::data_type();
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
    /// - `build::<Forward>`: forward form, for [`Graph::bfs`]
    /// - `build::<Transposed>`: transposed form, for [`Graph::pagerank`]
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - array lengths are inconsistent
    /// - `symmetrize` is combined with `edge_ids` or `edge_type_ids`
    /// - only one of `edge_start_time_ids` / `edge_end_time_ids` is set
    /// - the C API rejects the inputs
    pub fn build<T: Transposition>(&self, handle: &EnteredContext<'_>) -> Result<Graph<T>> {
        self.build_inner(handle, T::store_transposed())
    }

    /// Internal build logic.
    pub(crate) fn build_inner<T>(&self, handle: &EnteredContext<'_>, store_transposed: bool) -> Result<Graph<T>> {
        self.validate()?;

        let edge_id_dtype = self.edge_ids.map(|(ptr, _)| Self::view_dtype(ptr)).transpose()?;

        let c_props = self.properties.into();
        let vertices_ptr = self.vertices.map(|(p, _)| p as *const _).unwrap_or(ptr::null());
        let weights_ptr = self.weights.map(|(p, _)| p as *const _).unwrap_or(ptr::null());
        let edge_ids_ptr = self.edge_ids.map(|(p, _)| p as *const _).unwrap_or(ptr::null());
        let edge_type_ids_ptr = self.edge_type_ids.map(|(p, _)| p as *const _).unwrap_or(ptr::null());
        let start_times_ptr = self.edge_start_time_ids.map(|(p, _)| p as *const _).unwrap_or(ptr::null());
        let end_times_ptr = self.edge_end_time_ids.map(|(p, _)| p as *const _).unwrap_or(ptr::null());

        let mut graph: *mut cugraph_graph_t = ptr::null_mut();
        let mut error: *mut cugraph_error_t = ptr::null_mut();

        // SAFETY: every optional pointer is either null or comes from a live
        // device array view with a validated length and dtype; `graph`/`error`
        // are valid out-pointers owned by this stack frame.
        unsafe {
            let code = cugraph_graph_create_with_times_sg(
                handle.handle_ptr(),
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
            check_error(code, error, handle.device(), "GraphBuilder::build")?;
        }

        let graph = require_non_null(graph, "GraphBuilder::build", "graph")?;

        Ok(Graph {
            raw: graph,
            device: handle.device(),
            properties: self.properties,
            vertex_dtype: self.vertex_dtype,
            weight_dtype: self.weight_dtype,
            has_weights: self.weights.is_some(),
            edge_id_dtype,
            pending_input_completion: None,
            resource_handle_leases: RefCell::new(handle.runtime_bound_lease().into_iter().collect()),
            ordered_binding: Cell::new(handle.runtime_bound_lease().map(|lease| lease.runtime().binding_id())),
            untracked_resource_handle_use: Cell::new(false),
            _nosend_nosync: PhantomData,
            _transposition: PhantomData,
        })
    }

    /// Build from raw (type-erased) view pointers. Used by cudf_interop.
    pub(crate) fn from_raw_views(
        properties: GraphProperties,
        src: &'a RawDeviceArrayView<'a>,
        dst: &'a RawDeviceArrayView<'a>,
    ) -> Result<Self> {
        Ok(Self {
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
            vertex_dtype: src.dtype()?,
            weight_dtype: DataType::Float32,
            renumber: false,
            drop_self_loops: false,
            drop_multi_edges: false,
            symmetrize: false,
            do_expensive_check: false,
            _lifetime: PhantomData,
        })
    }

    /// Set weights from a raw view. Used by cudf_interop.
    pub(crate) fn weights_raw(mut self, raw: &'a RawDeviceArrayView<'a>) -> Result<Self> {
        self.weights = Some((raw.as_ptr(), raw.len()));
        self.weight_dtype = raw.dtype()?;
        Ok(self)
    }

    /// Set edge IDs from a raw view. Used by cudf prepared interop.
    pub(crate) fn edge_ids_raw(mut self, raw: &'a RawDeviceArrayView<'a>) -> Result<Self> {
        self.edge_ids = Some((raw.as_ptr(), raw.len()));
        Ok(self)
    }

    /// Set renumber from a bool. Used by cudf_interop, where the flag comes from
    /// a function parameter. The `renumber()` toggle is not used on this path.
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
        if !src_dtype.is_supported_index_type() {
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
            if !dtype.is_supported_index_type() {
                return Err(Error::new(ErrorKind::InvalidInput, "GraphBuilder::build")
                    .with_message(format!("edge_ids dtype must be Int32 or Int64, got {dtype}")));
            }
            if dtype != src_dtype {
                return Err(Error::new(ErrorKind::InvalidInput, "GraphBuilder::build")
                    .with_message(
                        "edge_ids must have the same dtype as src/dst to prevent silent truncation by the C API",
                    )
                    .with_context("edge_ids_dtype", dtype.to_string())
                    .with_context("src_dst_dtype", src_dtype.to_string()));
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
            if !dtype.is_supported_index_type() {
                return Err(Error::new(ErrorKind::InvalidInput, "GraphBuilder::build")
                    .with_message(format!("edge_start_time_ids dtype must be Int32 or Int64, got {dtype}")));
            }
        }
        if let Some((ptr, _)) = self.edge_end_time_ids {
            let dtype = Self::view_dtype(ptr)?;
            if !dtype.is_supported_index_type() {
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
        // The upstream C API casts mismatched vertex types via static_cast
        // (copy_or_transform in graph_helper_impl.cuh), which can truncate
        // values without an error (e.g. i64 → i32). The check below rejects
        // the mismatch to prevent data corruption.
        if let Some((ptr, _)) = self.vertices {
            let dtype = Self::view_dtype(ptr)?;
            if !dtype.is_supported_index_type() {
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
        // SAFETY: callers pass a cuGraph-owned device-array view pointer that
        // remains valid for the duration of this query.
        let c_type = unsafe { cugraph_type_erased_device_array_view_type(ptr) };
        DataType::try_from(c_type).map_err(|err| err.with_operation("view_dtype"))
    }

    /// Check that an optional array, if present, has the expected length.
    fn check_optional_len(
        array: Option<(*mut cugraph_type_erased_device_array_view_t, usize)>,
        expected: usize,
        name: &'static str,
    ) -> Result<()> {
        if let Some((_, len)) = array
            && len != expected
        {
            return Err(Error::new(ErrorKind::InvalidInput, "GraphBuilder::build")
                .with_message(format!("{name} must have the same number of elements as src/dst"))
                .with_context("src_len", expected.to_string())
                .with_context(name, len.to_string()));
        }
        Ok(())
    }
}
