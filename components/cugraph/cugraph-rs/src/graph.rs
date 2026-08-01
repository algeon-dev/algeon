//! Graph construction and management.
//!
//! A [`Graph`] is the central data structure for cuGraph algorithms.
//!
//! Most callers should construct graphs through
//! [`crate::cudf_interop`] from cudf
//! [`Series`](cudf::Series)/[`DataFrame`](cudf::DataFrame) inputs. This module
//! documents the lower-level graph object itself plus [`GraphBuilder`], which
//! is useful when edge lists are already materialized as typed device arrays.
//!
//! # Type-state encoding
//!
//! The `Graph<T>` type parameter encodes whether the graph is stored in
//! transposed form ([`Transposed`]) or forward form ([`Forward`]). Algorithms
//! that require a specific layout accept only the correct variant, so
//! mismatches are caught at compile time:
//!
//! - **[`Forward`]** — required by [`Graph::bfs`] and [`Graph::louvain`]
//! - **[`Transposed`]** — required by [`Graph::pagerank`],
//!   [`Graph::in_degrees`], and [`Graph::degrees`]
//!
//! Algorithms that work with either layout (e.g. betweenness centrality,
//! Force Atlas 2, out_degrees) are generic over `T: Transposition`.
//!
//! # Low-level builder example
//!
//! ```no_run
//! # use cudf::Runtime;
//! # use cugraph::{DeviceArray, GraphBuilder, ResourceHandle, Transposed};
//! # let runtime = std::sync::Arc::new(Runtime::new().unwrap());
//! # let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).unwrap();
//! # let entered = handle.enter("graph module docs").unwrap();
//! # let src = DeviceArray::from_slice(&entered, &[0i32, 1, 2]).unwrap();
//! # let dst = DeviceArray::from_slice(&entered, &[1i32, 2, 0]).unwrap();
//! let src_view = src.view().unwrap();
//! let dst_view = dst.view().unwrap();
//! let graph = GraphBuilder::new(&src_view, &dst_view)
//!     .renumber()
//!     .build::<Transposed>(&entered)
//!     .unwrap();
//! ```

use std::{
    any::Any,
    cell::{Cell, RefCell},
    marker::PhantomData,
    mem::ManuallyDrop,
    num::NonZeroUsize,
    ptr,
    sync::Arc,
};

use cugraph_sys::{
    cugraph_decompress_to_edgelist, cugraph_edgelist_free, cugraph_edgelist_get_destinations,
    cugraph_edgelist_get_sources, cugraph_edgelist_t, cugraph_error_t, cugraph_graph_create_with_times_sg,
    cugraph_graph_free, cugraph_graph_t, cugraph_type_erased_device_array_view_t,
    cugraph_type_erased_device_array_view_type,
};
use rapids_interop::{CompletionWitness, InteropError};

use crate::{
    array::{DeviceArray, DeviceArrayView, RawDeviceArrayView},
    cuda::DeviceGuard,
    error::{Error, ErrorKind, Result, check_error, require_non_null},
    handle::{EnteredContext, ResourceHandleLease},
    structure::{ResolvedVertexSelection, VertexIds, VertexSelection},
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
/// Required by algorithms like [`Graph::bfs`] and [`Graph::louvain`].
pub struct Forward;
impl Transposition for Forward {
    fn store_transposed() -> bool {
        false
    }
}

/// Marker type indicating a graph stored in transposed form.
///
/// Required by algorithms like [`Graph::pagerank`] and [`Graph::in_degrees`].
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
///
/// When a graph is constructed or operated on with a
/// [`ResourceHandle`](crate::handle::ResourceHandle) created by
/// [`ResourceHandle::for_runtime`](crate::handle::ResourceHandle::for_runtime),
/// the graph retains a private lease on that runtime-bound native handle until
/// the graph is dropped. This keeps native graph teardown safe when cuGraph
/// records allocations or RAFT-side state associated with the runtime-bound
/// handle used by construction or later algorithms.
pub struct Graph<T = Forward> {
    raw: *mut cugraph_graph_t,
    device: i32,
    properties: GraphProperties,
    vertex_dtype: DataType,
    weight_dtype: DataType,
    has_weights: bool,
    has_edge_ids: bool,
    edge_id_dtype: Option<DataType>,
    pending_input_completion: Option<DeferredInputCompletion>,
    // Runtime-bound resource handles used to build or operate on the graph
    // can own native state referenced during cugraph_graph_free. Keep these
    // leases alive until after Graph::drop runs.
    resource_handle_leases: RefCell<Vec<ResourceHandleLease>>,
    // A graph touched through local or external handles has work whose stream
    // is not represented by the runtime-bound lease ledger. Such a graph
    // cannot be parked because one runtime event cannot fence those uses.
    untracked_resource_handle_use: Cell<bool>,
    _nosend_nosync: PhantomData<*mut ()>,
    _transposition: PhantomData<T>,
}

/// A parked cuGraph graph that can be moved across Rust threads.
///
/// `Graph<T>` itself remains `!Send`/`!Sync` because native cuGraph operations
/// require a thread-local entered execution context. `ParkedGraph<T>` is the
/// transfer state for an idle graph: it owns the native graph pointer, retains
/// runtime-bound handle leases needed for teardown, and carries a CUDA event
/// recorded after tracked work on the graph's runtime stream. Resume requires
/// an entered context for the same cudf runtime owner that parked the graph.
pub struct ParkedGraph<T = Forward> {
    raw: *mut cugraph_graph_t,
    device: i32,
    properties: GraphProperties,
    vertex_dtype: DataType,
    weight_dtype: DataType,
    has_weights: bool,
    has_edge_ids: bool,
    edge_id_dtype: Option<DataType>,
    park_event: cudf::runtime::CudaEvent,
    resource_handle_leases: Vec<ResourceHandleLease>,
    _not_sync: PhantomData<Cell<()>>,
    _transposition: PhantomData<T>,
}

// SAFETY: parked graphs do not permit native graph use. The raw graph pointer
// is only resumed into a non-Send `Graph<T>` under a matching entered runtime
// context, or freed after the park event completes. Runtime-bound leases are
// retention-only metadata and are Send under their own safety contract.
unsafe impl<T: Send> Send for ParkedGraph<T> {}

pub(crate) struct DeferredInputCompletion {
    owners: Vec<Arc<dyn Any + Send + Sync>>,
    witnesses: Vec<CompletionWitness>,
}

impl DeferredInputCompletion {
    pub(crate) fn new(owners: Vec<Arc<dyn Any + Send + Sync>>, witnesses: Vec<CompletionWitness>) -> Option<Self> {
        if owners.is_empty() || witnesses.is_empty() {
            return None;
        }

        Some(Self { owners, witnesses })
    }

    pub(crate) fn synchronize_witnesses(&self, operation: &'static str) -> Result<()> {
        for witness in &self.witnesses {
            witness.synchronize().map_err(|error| map_completion_witness_error(error, operation))?;
        }
        Ok(())
    }

    pub(crate) fn synchronize(self) {
        for witness in &self.witnesses {
            let _ = witness.synchronize();
        }
        drop(self.owners);
    }
}

fn map_completion_witness_error(error: InteropError, operation: &'static str) -> Error {
    let mut mapped = Error::new(ErrorKind::InvalidState, operation)
        .with_message("failed to synchronize graph input completion witness")
        .with_context("interop_kind", format!("{:?}", error.kind()))
        .with_context("interop_message", error.message().to_string());

    for (key, value) in error.context() {
        mapped = mapped.with_context(key, value.clone());
    }

    mapped.set_source(error)
}

fn map_cudf_runtime_error(error: cudf::Error, operation: &'static str, message: &'static str) -> Error {
    let kind = match error.code() {
        cudf::ErrorCode::OutOfMemory | cudf::ErrorCode::ForeignResourceExhausted => ErrorKind::OutOfMemory,
        cudf::ErrorCode::CudaRuntime
        | cudf::ErrorCode::FatalCuda
        | cudf::ErrorCode::ReadinessUnavailable
        | cudf::ErrorCode::ForeignTransientDependencyFailure
        | cudf::ErrorCode::ForeignPersistentDependencyFailure => ErrorKind::Unexpected,
        cudf::ErrorCode::RuntimeBindingMismatch
        | cudf::ErrorCode::ExecutionBindingUnavailable
        | cudf::ErrorCode::InvariantViolation
        | cudf::ErrorCode::ForeignInvariantViolation => ErrorKind::InvalidState,
        cudf::ErrorCode::CrossDeviceInputs | cudf::ErrorCode::CrossProvenanceInputs => ErrorKind::InvalidInput,
        cudf::ErrorCode::UnsupportedDtype
        | cudf::ErrorCode::UnsupportedLayout
        | cudf::ErrorCode::UnsupportedOperation => ErrorKind::UnsupportedType,
        _ => ErrorKind::Unexpected,
    };

    let source_status = error.status();
    let mut mapped = Error::new(kind, operation)
        .with_message(message)
        .with_context("cudf_code", error.code().to_string())
        .with_context("cudf_kind", error.kind().to_string())
        .with_context("cudf_status", source_status.to_string())
        .with_context("cudf_operation", error.operation().to_string());

    mapped = match source_status {
        cudf::ErrorStatus::Permanent => mapped.set_permanent(),
        cudf::ErrorStatus::Temporary => mapped.set_temporary(),
        cudf::ErrorStatus::Persistent => mapped.set_persistent(),
    };

    mapped.set_source(error)
}

struct DecompressedEdgelist {
    raw: *mut cugraph_edgelist_t,
    device: i32,
    // Keeps a runtime-bound cuGraph handle alive until after Drop frees `raw`.
    _resource_handle_lease: Option<ResourceHandleLease>,
    _nosend_nosync: PhantomData<*mut ()>,
}

impl DecompressedEdgelist {
    unsafe fn from_raw(
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

        // SAFETY: `view_ptr` belongs to the live edgelist object and was
        // checked for null above.
        unsafe { RawDeviceArrayView::from_borrowed(view_ptr, Some(self.device), operation, pointer_name) }
    }

    fn edge_count(
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
            logical_edge_count = logical_edge_count.checked_add(run_len).ok_or_else(|| {
                Error::new(ErrorKind::InvalidState, operation)
                    .with_message("logical edge count overflow while accumulating undirected self-loops")
            })?;
        } else if !run_len.is_multiple_of(2) {
            return Err(Error::new(ErrorKind::InvalidState, operation)
                .with_message(
                    "undirected multigraph decompressed edgelist contained an odd count for a non-self-loop edge",
                )
                .with_context("edge", format!("{edge:?}"))
                .with_context("run_len", run_len.to_string()));
        } else {
            logical_edge_count = logical_edge_count.checked_add(run_len / 2).ok_or_else(|| {
                Error::new(ErrorKind::InvalidState, operation)
                    .with_message("logical edge count overflow while accumulating undirected multiedges")
            })?;
        }

        index += run_len;
    }

    Ok(logical_edge_count)
}

impl Drop for DecompressedEdgelist {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _guard = DeviceGuard::switch_to_for_drop(self.device, "DecompressedEdgelist::drop").ok();
            // SAFETY: `self.raw` is owned by this wrapper and has not been
            // freed yet.
            unsafe { cugraph_edgelist_free(self.raw) };
        }
    }
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

    pub(crate) fn has_edge_ids(&self) -> bool {
        self.has_edge_ids
    }

    pub(crate) fn edge_id_dtype(&self) -> Option<DataType> {
        self.edge_id_dtype
    }

    pub(crate) fn has_weights(&self) -> bool {
        self.has_weights
    }

    pub(crate) fn with_pending_input_completion(mut self, pending: DeferredInputCompletion) -> Self {
        self.pending_input_completion = Some(pending);
        self
    }

    /// Park this graph for transfer to another Rust thread.
    ///
    /// Parking consumes the non-Send [`Graph`] and returns a [`ParkedGraph`]
    /// that is `Send` but not `Sync`. The graph must have been built and used
    /// only through runtime-bound handles for the same cudf [`Runtime`]
    /// owner as `handle`; otherwise a single park event cannot prove all
    /// graph work is ordered on one stream.
    ///
    /// Prepared-input completion witnesses are synchronized before parking so
    /// cached graphs do not retain their input column owners indefinitely.
    /// The ordering proof relies on runtime-bound cuGraph handles using the
    /// cudf runtime stream directly rather than a RAFT stream pool.
    ///
    /// If this method returns an error, the consumed graph is dropped.
    pub fn park(mut self, handle: &EnteredContext<'_>) -> Result<ParkedGraph<T>>
    where T: Send {
        let operation = "Graph::park";
        self.validate_runtime_bound_transfer(handle, operation)?;

        if let Some(pending) = self.pending_input_completion.as_ref() {
            pending.synchronize_witnesses(operation)?;
        }
        drop(self.pending_input_completion.take());

        let _guard = DeviceGuard::switch_to(self.device, operation)?;
        let event = cudf::runtime::CudaEvent::new()
            .map_err(|error| map_cudf_runtime_error(error, operation, "failed to create graph park CUDA event"))?;
        let runtime = handle.runtime().expect("validated runtime-bound handle");
        event
            .record(runtime.stream())
            .map_err(|error| map_cudf_runtime_error(error, operation, "failed to record graph park CUDA event"))?;

        let this = ManuallyDrop::new(self);
        // SAFETY: `this` will not run `Graph::drop`, so moving the lease
        // ledger out here transfers the only owned `RefCell`.
        let resource_handle_leases = unsafe { ptr::read(&this.resource_handle_leases).into_inner() };

        Ok(ParkedGraph {
            raw: this.raw,
            device: this.device,
            properties: this.properties,
            vertex_dtype: this.vertex_dtype,
            weight_dtype: this.weight_dtype,
            has_weights: this.has_weights,
            has_edge_ids: this.has_edge_ids,
            edge_id_dtype: this.edge_id_dtype,
            park_event: event,
            resource_handle_leases,
            _not_sync: PhantomData,
            _transposition: PhantomData,
        })
    }

    fn validate_runtime_bound_transfer(&self, handle: &EnteredContext<'_>, operation: &'static str) -> Result<()> {
        if handle.device() != self.device {
            return Err(Error::new(ErrorKind::InvalidInput, operation)
                .with_message("graph and resource handle must be on the same CUDA device")
                .with_context("graph_device", self.device.to_string())
                .with_context("handle_device", handle.device().to_string()));
        }

        let Some(handle_lease) = handle.runtime_bound_lease() else {
            return Err(Error::new(ErrorKind::InvalidState, operation)
                .with_message("parked graph transfer requires a runtime-bound resource handle"));
        };
        if self.untracked_resource_handle_use.get() {
            return Err(Error::new(ErrorKind::InvalidState, operation)
                .with_message("graph was used with a non-runtime-bound resource handle and cannot be parked"));
        }

        let leases = self.resource_handle_leases.try_borrow().map_err(|_| {
            Error::new(ErrorKind::InvalidState, operation)
                .with_message("graph runtime-bound resource-handle lease ledger is already borrowed")
        })?;
        if leases.is_empty() {
            return Err(Error::new(ErrorKind::InvalidState, operation)
                .with_message("graph has no runtime-bound handle leases to prove park stream ownership"));
        }
        if !leases.iter().all(|lease| lease.same_runtime_owner(&handle_lease)) {
            return Err(Error::new(ErrorKind::InvalidState, operation)
                .with_message("graph has runtime-bound handle leases from multiple cudf runtime owners")
                .with_context("lease_count", leases.len().to_string()));
        }

        Ok(())
    }

    fn retain_runtime_bound_handle(&self, handle: &EnteredContext<'_>, operation: &'static str) -> Result<()> {
        if handle.device() != self.device {
            return Err(Error::new(ErrorKind::InvalidInput, operation)
                .with_message("graph and resource handle must be on the same CUDA device")
                .with_context("graph_device", self.device.to_string())
                .with_context("handle_device", handle.device().to_string()));
        }

        let Some(lease) = handle.runtime_bound_lease() else {
            self.untracked_resource_handle_use.set(true);
            return Ok(());
        };

        let mut leases = self.resource_handle_leases.try_borrow_mut().map_err(|_| {
            Error::new(ErrorKind::InvalidState, operation)
                .with_message("graph runtime-bound resource-handle lease ledger is already borrowed")
        })?;

        if !leases.iter().any(|existing| existing.same_runtime_owner(&lease)) {
            leases.push(lease);
        }

        Ok(())
    }

    /// Structural properties declared when this graph was constructed.
    ///
    /// These are the graph-construction properties Rust passed to cuGraph
    /// when creating this graph. They are reused by structural helpers such
    /// as [`edge_count`](Self::edge_count) to interpret decompressed edge
    /// lists with the same simple-vs-multigraph and directed-vs-undirected
    /// semantics the graph was built with.
    pub fn properties(&self) -> GraphProperties {
        self.properties
    }

    /// Return the number of edges currently stored in this graph.
    ///
    /// This materializes a transient edgelist view via the cuGraph C API, so
    /// it should be treated as a structural query rather than a constant-time
    /// metadata lookup.
    ///
    /// The returned count follows the graph's declared
    /// [`GraphProperties`]:
    ///
    /// - directed simple graphs count unique directed edges
    /// - undirected simple graphs count unique unordered edge pairs
    /// - directed multigraphs count all stored directed edge rows
    /// - undirected multigraphs count unordered edge multiplicities, with
    ///   self-loops counted once each
    pub fn edge_count(&mut self, handle: &EnteredContext<'_>) -> Result<usize> {
        self.edge_count_impl(handle, false, "Graph::edge_count")
    }

    /// Return the number of edges currently stored in this graph with
    /// optional expensive validation checks.
    ///
    /// This has the same counting semantics as [`edge_count`](Self::edge_count).
    pub fn edge_count_checked(&mut self, handle: &EnteredContext<'_>, do_expensive_check: bool) -> Result<usize> {
        self.edge_count_impl(handle, do_expensive_check, "Graph::edge_count_checked")
    }

    /// Select random vertices from this graph.
    ///
    /// The returned [`DeviceArray`] owns the sampled vertex ids on the handle
    /// device. The requested vertex dtype must match this graph's vertex dtype.
    pub fn select_random_vertices<V: CugraphElement>(
        &self,
        handle: &EnteredContext<'_>,
        rng_state: &mut crate::random::RngState,
        num_vertices: usize,
    ) -> Result<DeviceArray<V>> {
        crate::sampling::select_random_vertices(handle, self, rng_state, num_vertices)
    }

    pub(crate) fn as_ptr(&self, handle: &EnteredContext<'_>, operation: &'static str) -> Result<*mut cugraph_graph_t> {
        self.retain_runtime_bound_handle(handle, operation)?;
        Ok(self.raw)
    }

    fn edge_count_impl(
        &mut self,
        handle: &EnteredContext<'_>,
        do_expensive_check: bool,
        operation: &'static str,
    ) -> Result<usize> {
        let mut edgelist: *mut cugraph_edgelist_t = ptr::null_mut();
        let mut error: *mut cugraph_error_t = ptr::null_mut();

        // SAFETY: `self.raw` is a live cuGraph graph handle and
        // `edgelist`/`error` are valid out-pointers for the C API.
        unsafe {
            let code = cugraph_decompress_to_edgelist(
                handle.handle_ptr(),
                self.as_ptr(handle, operation)?,
                to_bool_t(do_expensive_check),
                &mut edgelist,
                &mut error,
            );
            check_error(code, error, operation)?;
        }

        // SAFETY: on success, cuGraph transfers ownership of the edgelist
        // pointer to the caller.
        let edgelist =
            unsafe { DecompressedEdgelist::from_raw(edgelist, handle.device(), handle.runtime_bound_lease()) }?;
        edgelist.edge_count(handle, self.properties, operation)
    }

    pub(crate) fn validate_vertex_input_dtype(
        &self,
        dtype: DataType,
        op: &'static str,
        input_name: &'static str,
    ) -> Result<()> {
        if !dtype.is_supported_index_type() {
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

    pub(crate) fn validate_vertex_selection<V: CugraphElement>(
        &self,
        vertices: &VertexSelection<'_, V>,
        op: &'static str,
        input_name: &'static str,
        allow_all: bool,
    ) -> Result<()> {
        match vertices {
            VertexSelection::All if allow_all => {},
            VertexSelection::All => {
                return Err(Error::new(ErrorKind::InvalidInput, op).with_message(format!(
                    "{input_name} must specify an explicit vertex subset; VertexSelection::All is not supported"
                )));
            },
            VertexSelection::Device(_) | VertexSelection::Host(_) => {
                self.validate_vertex_input_dtype(V::data_type(), op, input_name)?;
            },
        }
        Ok(())
    }

    pub(crate) fn resolved_vertex_view<'a, V: CugraphElement>(
        &self,
        resolved: &'a ResolvedVertexSelection<'_, V>,
        op: &'static str,
        input_name: &'static str,
    ) -> Result<DeviceArrayView<'a, V>> {
        resolved.view()?.ok_or_else(|| {
            Error::new(ErrorKind::InvalidInput, op).with_message(format!(
                "{input_name} must specify an explicit vertex subset; VertexSelection::All is not supported"
            ))
        })
    }
}

impl<T> std::fmt::Debug for Graph<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Graph")
            .field("properties", &self.properties)
            .field("vertex_dtype", &self.vertex_dtype)
            .field("weight_dtype", &self.weight_dtype)
            .field("has_weights", &self.has_weights)
            .finish_non_exhaustive()
    }
}

impl<T> ParkedGraph<T> {
    /// The CUDA device this parked graph resides on.
    pub fn device(&self) -> i32 {
        self.device
    }

    /// Structural properties declared when this graph was constructed.
    pub fn properties(&self) -> GraphProperties {
        self.properties
    }

    /// The data type of vertex IDs in this graph.
    pub fn vertex_dtype(&self) -> DataType {
        self.vertex_dtype
    }

    /// The data type of edge weights in this graph.
    pub fn weight_dtype(&self) -> DataType {
        self.weight_dtype
    }

    /// Resume this parked graph under a runtime-bound entered context.
    ///
    /// V1 resume is same-runtime by construction: `handle` must use the same
    /// cudf `Runtime` owner as every lease retained by the parked graph. The
    /// runtime stream waits on the park event before the live [`Graph`] is
    /// returned to the caller.
    ///
    /// If this method returns an error, the consumed parked graph is dropped.
    pub fn resume(self, handle: &EnteredContext<'_>) -> Result<Graph<T>> {
        let operation = "ParkedGraph::resume";
        self.validate_resume_handle(handle, operation)?;

        let runtime = handle.runtime().expect("validated runtime-bound handle");
        runtime
            .stream()
            .wait_event(&self.park_event)
            .map_err(|error| map_cudf_runtime_error(error, operation, "failed to wait on graph park CUDA event"))?;

        let this = ManuallyDrop::new(self);
        // SAFETY: `this` will not run `ParkedGraph::drop`, so moving fields
        // out here transfers native graph ownership to the returned Graph.
        let resource_handle_leases = unsafe { ptr::read(&this.resource_handle_leases) };
        // SAFETY: the park event is no longer needed once the runtime stream
        // has been ordered after it. Drop the moved event normally.
        let park_event = unsafe { ptr::read(&this.park_event) };
        drop(park_event);

        Ok(Graph {
            raw: this.raw,
            device: this.device,
            properties: this.properties,
            vertex_dtype: this.vertex_dtype,
            weight_dtype: this.weight_dtype,
            has_weights: this.has_weights,
            has_edge_ids: this.has_edge_ids,
            edge_id_dtype: this.edge_id_dtype,
            pending_input_completion: None,
            resource_handle_leases: RefCell::new(resource_handle_leases),
            untracked_resource_handle_use: Cell::new(false),
            _nosend_nosync: PhantomData,
            _transposition: PhantomData,
        })
    }

    fn validate_resume_handle(&self, handle: &EnteredContext<'_>, operation: &'static str) -> Result<()> {
        if handle.device() != self.device {
            return Err(Error::new(ErrorKind::InvalidInput, operation)
                .with_message("parked graph and resource handle must be on the same CUDA device")
                .with_context("graph_device", self.device.to_string())
                .with_context("handle_device", handle.device().to_string()));
        }

        let Some(handle_lease) = handle.runtime_bound_lease() else {
            return Err(Error::new(ErrorKind::InvalidState, operation)
                .with_message("parked graph resume requires a runtime-bound resource handle"));
        };
        if self.resource_handle_leases.is_empty() {
            return Err(Error::new(ErrorKind::InvalidState, operation)
                .with_message("parked graph has no runtime-bound handle leases"));
        }
        if !self.resource_handle_leases.iter().all(|lease| lease.same_runtime_owner(&handle_lease)) {
            return Err(Error::new(ErrorKind::InvalidState, operation)
                .with_message("parked graph runtime owner does not match the resume handle")
                .with_context("lease_count", self.resource_handle_leases.len().to_string()));
        }

        Ok(())
    }
}

impl<T> std::fmt::Debug for ParkedGraph<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParkedGraph")
            .field("device", &self.device)
            .field("properties", &self.properties)
            .field("vertex_dtype", &self.vertex_dtype)
            .field("weight_dtype", &self.weight_dtype)
            .field("has_weights", &self.has_weights)
            .finish_non_exhaustive()
    }
}

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
/// - `src` — source vertex ids (device array view)
/// - `dst` — destination vertex ids (device array view)
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
    vertex_dtype: Option<DataType>,
    weight_dtype: Option<DataType>,
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
            vertex_dtype: Some(V::data_type()),
            weight_dtype: None,
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
    /// - `build::<Forward>` — forward form, for [`Graph::bfs`]
    /// - `build::<Transposed>` — transposed form, for [`Graph::pagerank`]
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

        // Resolve dtypes: prefer the cached builder-state value (from either
        // the typed API or the raw-view cudf interop path), falling back to a
        // runtime query of the raw view pointer only when necessary.
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
            check_error(code, error, "GraphBuilder::build")?;
        }

        let graph = require_non_null(graph, "GraphBuilder::build", "graph")?;

        Ok(Graph {
            raw: graph,
            device: handle.device(),
            properties: self.properties,
            vertex_dtype,
            weight_dtype,
            has_weights: self.weights.is_some(),
            has_edge_ids: self.edge_ids.is_some(),
            edge_id_dtype,
            pending_input_completion: None,
            resource_handle_leases: RefCell::new(handle.runtime_bound_lease().into_iter().collect()),
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
            vertex_dtype: Some(src.dtype()?),
            weight_dtype: None,
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
        self.weight_dtype = Some(raw.dtype()?);
        Ok(self)
    }

    /// Set renumber from a bool. Used by cudf_interop where the flag comes
    /// from a function parameter rather than a builder toggle.
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
        // The upstream C API silently casts mismatched vertex types via
        // static_cast (copy_or_transform in graph_helper_impl.cuh), which
        // can cause silent truncation (e.g. i64 → i32). We reject the
        // mismatch here to prevent data corruption.
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
    /// Compute PageRank on this transposed graph.
    ///
    /// Returns a [`PageRankResult`](crate::centrality::PageRankResult)
    /// containing vertex ids, scores, iteration count, and convergence status.
    ///
    /// This convenience method does not expose any typed auxiliary arrays, so
    /// it can be used regardless of the graph's runtime vertex-id or weight
    /// dtype. For advanced features (precomputed weights, initial guess), use
    /// [`pagerank_with`](Self::pagerank_with) and ensure those typed inputs
    /// match the graph's runtime dtypes.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use cugraph::{
    /// #     ResourceHandle, Transposed,
    /// #     centrality::PageRankParams,
    /// #     cudf_interop::{DataFrameGraphOptions, graph_from_dataframe},
    /// # };
    /// # let runtime = std::sync::Arc::new(cudf::Runtime::new().unwrap());
    /// # let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).unwrap();
    /// # let entered = handle.enter("Graph::pagerank docs").unwrap();
    /// # let src = runtime.series(&[0i32, 1, 2]).unwrap();
    /// # let dst = runtime.series(&[1i32, 2, 0]).unwrap();
    /// # let edges = cudf::DataFrame::from_series([("src", src), ("dst", dst)]).unwrap();
    /// # let opts = DataFrameGraphOptions::new("src", "dst").renumber();
    /// # let mut graph = graph_from_dataframe::<Transposed>(&entered, &edges, &opts).unwrap();
    /// let result = graph
    ///     .pagerank(&entered, &PageRankParams::default())
    ///     .unwrap();
    /// let scores = result.scores().unwrap();
    /// ```
    pub fn pagerank(
        &mut self,
        handle: &EnteredContext<'_>,
        params: &crate::centrality::PageRankParams,
    ) -> Result<crate::centrality::PageRankResult> {
        crate::centrality::pagerank::pagerank::<i32, f32>(handle, self, params, None, None)
    }

    /// Compute PageRank with optional precomputed weights and/or initial guess.
    ///
    /// This is the advanced variant of [`pagerank`](Self::pagerank). Use it
    /// when you have precomputed out-weight sums or want to supply an initial
    /// guess for iterative refinement.
    pub fn pagerank_with<V: CugraphElement, W: CugraphElement>(
        &mut self,
        handle: &EnteredContext<'_>,
        params: &crate::centrality::PageRankParams,
        precomputed_weights: Option<&crate::centrality::PrecomputedWeights<'_, V, W>>,
        initial_guess: Option<&crate::centrality::InitialGuess<'_, V, W>>,
    ) -> Result<crate::centrality::PageRankResult> {
        crate::centrality::pagerank::pagerank(handle, self, params, precomputed_weights, initial_guess)
    }

    /// Compute personalized PageRank using the allow-nonconvergence native API.
    pub fn personalized_pagerank<V: CugraphElement, W: CugraphElement>(
        &mut self,
        handle: &EnteredContext<'_>,
        params: &crate::centrality::PersonalizedPageRankParams,
        personalization: &crate::centrality::PersonalizationVector<'_, V, W>,
    ) -> Result<crate::centrality::PageRankResult> {
        crate::centrality::pagerank::personalized_pagerank(handle, self, params, personalization, None, None)
    }

    /// Compute personalized PageRank with optional precomputed weights and/or initial guess.
    pub fn personalized_pagerank_with<V: CugraphElement, W: CugraphElement>(
        &mut self,
        handle: &EnteredContext<'_>,
        params: &crate::centrality::PersonalizedPageRankParams,
        personalization: &crate::centrality::PersonalizationVector<'_, V, W>,
        precomputed_weights: Option<&crate::centrality::PrecomputedWeights<'_, V, W>>,
        initial_guess: Option<&crate::centrality::InitialGuess<'_, V, W>>,
    ) -> Result<crate::centrality::PageRankResult> {
        crate::centrality::pagerank::personalized_pagerank(
            handle,
            self,
            params,
            personalization,
            precomputed_weights,
            initial_guess,
        )
    }

    /// Compute eigenvector centrality.
    pub fn eigenvector_centrality(
        &mut self,
        handle: &EnteredContext<'_>,
        params: &crate::centrality::EigenvectorCentralityParams,
    ) -> Result<crate::centrality::EigenvectorCentralityResult> {
        crate::centrality::vertex::eigenvector_centrality(handle, self, params)
    }

    /// Compute Katz centrality using a scalar beta value.
    ///
    /// The local cuGraph C API accepts a per-vertex beta array parameter, but
    /// the current implementation does not forward it to the native algorithm.
    /// This safe method therefore exposes only the scalar-beta path.
    pub fn katz_centrality(
        &mut self,
        handle: &EnteredContext<'_>,
        params: &crate::centrality::KatzCentralityParams,
    ) -> Result<crate::centrality::KatzCentralityResult> {
        crate::centrality::vertex::katz_centrality(handle, self, params)
    }

    /// Compute HITS hub and authority scores.
    pub fn hits(
        &mut self,
        handle: &EnteredContext<'_>,
        params: &crate::centrality::HitsParams,
    ) -> Result<crate::centrality::HitsResult> {
        crate::centrality::hits::hits::<i32, f32>(handle, self, params, None)
    }

    /// Compute HITS hub and authority scores with an optional initial hub guess.
    pub fn hits_with_initial_hubs<V: CugraphElement, W: CugraphElement>(
        &mut self,
        handle: &EnteredContext<'_>,
        params: &crate::centrality::HitsParams,
        initial_hubs: Option<&crate::centrality::HitsInitialHubs<'_, V, W>>,
    ) -> Result<crate::centrality::HitsResult> {
        crate::centrality::hits::hits(handle, self, params, initial_hubs)
    }

    /// Compute both in-degrees and out-degrees for vertices in this graph.
    ///
    /// Pass [`VertexSelection::All`] to compute degrees for every vertex,
    /// [`VertexSelection::Device`] with a device view, or
    /// [`VertexSelection::Host`] with a host slice (uploaded on demand).
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::UnsupportedType`] if `vertices` uses a vertex
    /// dtype that does not match the graph.
    pub fn degrees<'a, V: CugraphElement + 'a>(
        &mut self,
        handle: &EnteredContext<'_>,
        vertices: impl Into<VertexSelection<'a, V>>,
    ) -> Result<crate::structure::DegreesResult> {
        let vertices = vertices.into();
        self.validate_vertex_selection(&vertices, "Graph::degrees", "vertices", true)?;
        let resolved = vertices.resolve(handle)?;
        let view = resolved.view()?;
        let result = crate::structure::degrees::degrees(handle, self, view.as_ref(), false);
        result
    }

    /// Compute degrees with optional expensive input validation.
    pub fn degrees_checked<'a, V: CugraphElement + 'a>(
        &mut self,
        handle: &EnteredContext<'_>,
        vertices: impl Into<VertexSelection<'a, V>>,
        do_expensive_check: bool,
    ) -> Result<crate::structure::DegreesResult> {
        let vertices = vertices.into();
        self.validate_vertex_selection(&vertices, "Graph::degrees", "vertices", true)?;
        let resolved = vertices.resolve(handle)?;
        let view = resolved.view()?;
        let result = crate::structure::degrees::degrees(handle, self, view.as_ref(), do_expensive_check);
        result
    }

    /// Compute degrees for every vertex in the graph.
    pub fn degrees_all(&mut self, handle: &EnteredContext<'_>) -> Result<crate::structure::DegreesResult> {
        self.degrees(handle, crate::structure::VertexSelection::<i32>::All)
    }

    /// Compute only in-degrees for vertices in this graph.
    ///
    /// Pass [`VertexSelection::All`] to compute degrees for every vertex,
    /// [`VertexSelection::Device`] with a device view, or
    /// [`VertexSelection::Host`] with a host slice (uploaded on demand).
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::UnsupportedType`] if `vertices` uses a vertex
    /// dtype that does not match the graph.
    pub fn in_degrees<'a, V: CugraphElement + 'a>(
        &mut self,
        handle: &EnteredContext<'_>,
        vertices: impl Into<VertexSelection<'a, V>>,
    ) -> Result<crate::structure::InDegreesResult> {
        let vertices = vertices.into();
        self.validate_vertex_selection(&vertices, "Graph::in_degrees", "vertices", true)?;
        let resolved = vertices.resolve(handle)?;
        let view = resolved.view()?;
        let result = crate::structure::degrees::in_degrees(handle, self, view.as_ref(), false);
        result
    }

    /// Compute in-degrees with optional expensive input validation.
    pub fn in_degrees_checked<'a, V: CugraphElement + 'a>(
        &mut self,
        handle: &EnteredContext<'_>,
        vertices: impl Into<VertexSelection<'a, V>>,
        do_expensive_check: bool,
    ) -> Result<crate::structure::InDegreesResult> {
        let vertices = vertices.into();
        self.validate_vertex_selection(&vertices, "Graph::in_degrees", "vertices", true)?;
        let resolved = vertices.resolve(handle)?;
        let view = resolved.view()?;
        let result = crate::structure::degrees::in_degrees(handle, self, view.as_ref(), do_expensive_check);
        result
    }

    /// Compute in-degrees for every vertex in the graph.
    pub fn in_degrees_all(&mut self, handle: &EnteredContext<'_>) -> Result<crate::structure::InDegreesResult> {
        self.in_degrees(handle, crate::structure::VertexSelection::<i32>::All)
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
    /// # use cugraph::{
    /// #     Forward, ResourceHandle,
    /// #     cudf_interop::{DataFrameGraphOptions, device_array_view_from_slice, graph_from_dataframe},
    /// #     traversal::BfsParams,
    /// # };
    /// # let runtime = std::sync::Arc::new(cudf::Runtime::new().unwrap());
    /// # let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).unwrap();
    /// # let entered = handle.enter("Graph::bfs docs").unwrap();
    /// # let src = runtime.series(&[0i32, 1, 2]).unwrap();
    /// # let dst = runtime.series(&[1i32, 2, 0]).unwrap();
    /// # let edges = cudf::DataFrame::from_series([("src", src), ("dst", dst)]).unwrap();
    /// # let opts = DataFrameGraphOptions::new("src", "dst").renumber();
    /// # let mut graph = graph_from_dataframe::<Forward>(&entered, &edges, &opts).unwrap();
    /// let sources = runtime.series(&[0i32]).unwrap();
    /// let source_column = sources.as_column();
    /// let source_slice = source_column.device_slice::<i32>().unwrap();
    /// let sources_view = device_array_view_from_slice::<i32>(&source_slice).unwrap();
    /// let result =
    ///     graph.bfs(&entered, &sources_view, &BfsParams::default()).unwrap();
    /// let distances = result.distances().unwrap();
    /// ```
    pub fn bfs<'a, V: CugraphElement + 'a>(
        &mut self,
        handle: &EnteredContext<'_>,
        sources: impl Into<VertexSelection<'a, V>>,
        params: &crate::traversal::BfsParams,
    ) -> Result<crate::traversal::BfsResult> {
        let sources = sources.into();
        self.validate_vertex_selection(&sources, "Graph::bfs", "sources", false)?;
        let resolved = sources.resolve(handle)?;
        let view = self.resolved_vertex_view(&resolved, "Graph::bfs", "sources")?;
        crate::traversal::bfs::bfs(handle, self, &view, params)
    }

    /// Perform breadth-first search with optional edge, vertex, and target predicates.
    ///
    /// Predicate inputs must already be concrete device arrays. Higher layers
    /// are responsible for SQL planning, relation-backed edge alignment, and
    /// logical-to-physical vertex-id mapping before calling this API.
    pub fn bfs_with_predicates<'a, V, E>(
        &mut self,
        handle: &EnteredContext<'_>,
        sources: impl Into<VertexSelection<'a, V>>,
        params: &crate::traversal::BfsParams,
        predicates: &crate::traversal::BfsPredicateParams<'a, V, E>,
    ) -> Result<crate::traversal::BfsResult>
    where
        V: CugraphElement + 'a,
        E: CugraphElement + 'a,
    {
        let sources = sources.into();
        self.validate_vertex_selection(&sources, "Graph::bfs_with_predicates", "sources", false)?;
        let resolved = sources.resolve(handle)?;
        let view = self.resolved_vertex_view(&resolved, "Graph::bfs_with_predicates", "sources")?;
        crate::traversal::bfs::bfs_with_predicates(handle, self, &view, params, predicates)
    }

    /// Perform single-source shortest path from one source vertex.
    ///
    /// Returns an [`SsspResult`](crate::traversal::SsspResult) containing
    /// vertex ids, weighted distances, and optionally predecessors.
    pub fn sssp<V>(
        &mut self,
        handle: &EnteredContext<'_>,
        source: V,
        params: &crate::traversal::SsspParams,
    ) -> Result<crate::traversal::SsspResult>
    where
        V: CugraphElement + TryInto<usize> + Copy + std::fmt::Debug,
    {
        self.validate_vertex_input_dtype(V::data_type(), "Graph::sssp", "source")?;
        crate::traversal::sssp::sssp(handle, self, source, params)
    }

    /// Compute uniform random walks from explicit start vertices.
    pub fn uniform_random_walks<V: CugraphElement>(
        &mut self,
        handle: &EnteredContext<'_>,
        rng_state: &mut crate::random::RngState,
        start_vertices: &DeviceArrayView<'_, V>,
        params: &crate::sampling::RandomWalkParams,
    ) -> Result<crate::sampling::RandomWalkResult> {
        crate::sampling::uniform_random_walks(handle, rng_state, self, start_vertices, params)
    }

    /// Compute biased random walks from explicit start vertices.
    pub fn biased_random_walks<V: CugraphElement>(
        &mut self,
        handle: &EnteredContext<'_>,
        rng_state: &mut crate::random::RngState,
        start_vertices: &DeviceArrayView<'_, V>,
        params: &crate::sampling::RandomWalkParams,
    ) -> Result<crate::sampling::RandomWalkResult> {
        crate::sampling::biased_random_walks(handle, rng_state, self, start_vertices, params)
    }

    /// Compute node2vec random walks from explicit start vertices.
    pub fn node2vec_random_walks<V: CugraphElement>(
        &mut self,
        handle: &EnteredContext<'_>,
        rng_state: &mut crate::random::RngState,
        start_vertices: &DeviceArrayView<'_, V>,
        params: &crate::sampling::Node2VecRandomWalkParams,
    ) -> Result<crate::sampling::RandomWalkResult> {
        crate::sampling::node2vec_random_walks(handle, rng_state, self, start_vertices, params)
    }

    /// Sample homogeneous neighborhoods uniformly from explicit start vertices.
    pub fn homogeneous_uniform_neighbor_sample<V: CugraphElement>(
        &mut self,
        handle: &EnteredContext<'_>,
        rng_state: &mut crate::random::RngState,
        start_vertices: &DeviceArrayView<'_, V>,
        starting_vertex_label_offsets: Option<&DeviceArrayView<'_, usize>>,
        params: &crate::sampling::NeighborSamplingParams,
    ) -> Result<crate::sampling::SampleResult> {
        crate::sampling::homogeneous_uniform_neighbor_sample(
            handle,
            rng_state,
            self,
            start_vertices,
            starting_vertex_label_offsets,
            params,
        )
    }

    /// Sample homogeneous neighborhoods with biased sampling from explicit start vertices.
    pub fn homogeneous_biased_neighbor_sample<V: CugraphElement>(
        &mut self,
        handle: &EnteredContext<'_>,
        rng_state: &mut crate::random::RngState,
        start_vertices: &DeviceArrayView<'_, V>,
        starting_vertex_label_offsets: Option<&DeviceArrayView<'_, usize>>,
        params: &crate::sampling::NeighborSamplingParams,
    ) -> Result<crate::sampling::SampleResult> {
        crate::sampling::homogeneous_biased_neighbor_sample(
            handle,
            rng_state,
            self,
            start_vertices,
            starting_vertex_label_offsets,
            params,
        )
    }

    /// Sample homogeneous temporal neighborhoods uniformly from explicit start vertices.
    pub fn homogeneous_uniform_temporal_neighbor_sample<V, Tm>(
        &mut self,
        handle: &EnteredContext<'_>,
        rng_state: &mut crate::random::RngState,
        temporal_property_name: &str,
        start_vertices: &DeviceArrayView<'_, V>,
        starting_vertex_times: Option<&DeviceArrayView<'_, Tm>>,
        starting_vertex_label_offsets: Option<&DeviceArrayView<'_, usize>>,
        params: &crate::sampling::NeighborSamplingParams,
    ) -> Result<crate::sampling::SampleResult>
    where
        V: CugraphElement,
        Tm: CugraphElement,
    {
        crate::sampling::homogeneous_uniform_temporal_neighbor_sample(
            handle,
            rng_state,
            self,
            temporal_property_name,
            start_vertices,
            starting_vertex_times,
            starting_vertex_label_offsets,
            params,
        )
    }

    /// Sample homogeneous temporal neighborhoods with biased sampling from explicit start vertices.
    pub fn homogeneous_biased_temporal_neighbor_sample<V, Tm>(
        &mut self,
        handle: &EnteredContext<'_>,
        rng_state: &mut crate::random::RngState,
        temporal_property_name: &str,
        start_vertices: &DeviceArrayView<'_, V>,
        starting_vertex_times: Option<&DeviceArrayView<'_, Tm>>,
        starting_vertex_label_offsets: Option<&DeviceArrayView<'_, usize>>,
        params: &crate::sampling::NeighborSamplingParams,
    ) -> Result<crate::sampling::SampleResult>
    where
        V: CugraphElement,
        Tm: CugraphElement,
    {
        crate::sampling::homogeneous_biased_temporal_neighbor_sample(
            handle,
            rng_state,
            self,
            temporal_property_name,
            start_vertices,
            starting_vertex_times,
            starting_vertex_label_offsets,
            params,
        )
    }

    /// Compute Louvain communities on this forward-stored graph.
    ///
    /// Louvain is an undirected community-detection algorithm. This wrapper
    /// preserves the crate's type-state requirement that the graph must be
    /// stored in forward (non-transposed) layout, but it does not add a new
    /// Rust-side hard check that the graph was constructed with undirected
    /// properties.
    ///
    /// Returns a [`LouvainResult`](crate::community::LouvainResult)
    /// containing vertex ids, per-vertex partitions, and modularity.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use cugraph::{
    /// #     Forward, ResourceHandle,
    /// #     community::LouvainParams,
    /// #     cudf_interop::{DataFrameGraphOptions, graph_from_dataframe},
    /// #     types::GraphProperties,
    /// # };
    /// # let runtime = std::sync::Arc::new(cudf::Runtime::new().unwrap());
    /// # let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).unwrap();
    /// # let entered = handle.enter("Graph::louvain docs").unwrap();
    /// # let src = runtime.series(&[0i32, 1, 1, 2]).unwrap();
    /// # let dst = runtime.series(&[1i32, 0, 2, 1]).unwrap();
    /// # let weight = runtime.series(&[1.0f32, 1.0, 1.0, 1.0]).unwrap();
    /// # let edges = cudf::DataFrame::from_series([("src", src), ("dst", dst), ("weight", weight)]).unwrap();
    /// let opts = DataFrameGraphOptions::new("src", "dst")
    ///     .weight_col("weight")
    ///     .properties(GraphProperties::undirected())
    ///     .renumber();
    /// let mut graph = graph_from_dataframe::<Forward>(&entered, &edges, &opts).unwrap();
    ///
    /// let result = graph.louvain(&entered, &LouvainParams::default()).unwrap();
    /// let modularity = result.modularity();
    /// # let _ = modularity;
    /// ```
    pub fn louvain(
        &mut self,
        handle: &EnteredContext<'_>,
        params: &crate::community::LouvainParams,
    ) -> Result<crate::community::LouvainResult> {
        crate::community::louvain::louvain(handle, self, params)
    }

    /// Compute Leiden communities on this forward-stored graph.
    ///
    /// Leiden is an undirected community-detection algorithm. This wrapper
    /// requires forward graph storage but leaves graph-property validation to
    /// the native library.
    pub fn leiden(
        &mut self,
        handle: &EnteredContext<'_>,
        rng_state: &mut crate::random::RngState,
        params: &crate::community::LeidenParams,
    ) -> Result<crate::community::LeidenResult> {
        crate::community::leiden::leiden(handle, rng_state, self, params)
    }

    /// Compute ECG communities on this forward-stored graph.
    ///
    /// ECG is an undirected community-detection algorithm built from a Louvain
    /// ensemble. This wrapper requires forward graph storage but leaves
    /// graph-property validation to the native library.
    pub fn ecg(
        &mut self,
        handle: &EnteredContext<'_>,
        rng_state: &mut crate::random::RngState,
        params: &crate::community::EcgParams,
    ) -> Result<crate::community::EcgResult> {
        crate::community::ecg::ecg(handle, rng_state, self, params)
    }

    /// Compute legacy spectral modularity maximization clusters.
    ///
    /// cuGraph requires a symmetric graph for this algorithm. This wrapper
    /// keeps that validation in the native library, matching the rest of the
    /// community-detection surface.
    pub fn spectral_modularity_maximization(
        &mut self,
        handle: &EnteredContext<'_>,
        rng_state: &mut crate::random::RngState,
        params: &crate::community::SpectralClusteringParams,
    ) -> Result<crate::community::SpectralClusteringResult> {
        crate::community::clustering::spectral_modularity_maximization(handle, rng_state, self, params)
    }

    /// Compute deprecated balanced-cut clustering.
    #[allow(deprecated)]
    #[deprecated(note = "cuGraph deprecates balanced cut clustering; use spectral_modularity_maximization instead")]
    pub fn balanced_cut_clustering(
        &mut self,
        handle: &EnteredContext<'_>,
        rng_state: &mut crate::random::RngState,
        params: &crate::community::BalancedCutClusteringParams,
    ) -> Result<crate::community::BalancedCutClusteringResult> {
        crate::community::clustering::balanced_cut_clustering(handle, rng_state, self, params)
    }

    /// Compute modularity for explicit clustering columns.
    pub fn analyze_clustering_modularity<V: CugraphElement>(
        &mut self,
        handle: &EnteredContext<'_>,
        n_clusters: NonZeroUsize,
        vertices: &DeviceArrayView<'_, V>,
        clusters: &DeviceArrayView<'_, V>,
    ) -> Result<f64> {
        crate::community::clustering::analyze_clustering_modularity(handle, self, n_clusters, vertices, clusters)
    }

    /// Compute edge cut for explicit clustering columns.
    pub fn analyze_clustering_edge_cut<V: CugraphElement>(
        &mut self,
        handle: &EnteredContext<'_>,
        n_clusters: NonZeroUsize,
        vertices: &DeviceArrayView<'_, V>,
        clusters: &DeviceArrayView<'_, V>,
    ) -> Result<f64> {
        crate::community::clustering::analyze_clustering_edge_cut(handle, self, n_clusters, vertices, clusters)
    }

    /// Compute ratio cut for explicit clustering columns.
    pub fn analyze_clustering_ratio_cut<V: CugraphElement>(
        &mut self,
        handle: &EnteredContext<'_>,
        n_clusters: NonZeroUsize,
        vertices: &DeviceArrayView<'_, V>,
        clusters: &DeviceArrayView<'_, V>,
    ) -> Result<f64> {
        crate::community::clustering::analyze_clustering_ratio_cut(handle, self, n_clusters, vertices, clusters)
    }
}

impl<T: Transposition> Graph<T> {
    /// Compute vertex betweenness centrality on this graph.
    ///
    /// Pass [`VertexSelection::All`] for exact computation, or provide seed
    /// vertices for approximate computation via [`VertexSelection::Device`] or
    /// [`VertexSelection::Host`].
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::UnsupportedType`] if `vertex_list` uses a vertex
    /// dtype that does not match the graph.
    pub fn betweenness_centrality<'a, V: CugraphElement + 'a>(
        &mut self,
        handle: &EnteredContext<'_>,
        vertex_list: impl Into<VertexSelection<'a, V>>,
        params: &crate::centrality::BetweennessCentralityParams,
    ) -> Result<crate::centrality::BetweennessCentralityResult> {
        let vertex_list = vertex_list.into();
        self.validate_vertex_selection(&vertex_list, "Graph::betweenness_centrality", "vertex_list", true)?;
        let resolved = vertex_list.resolve(handle)?;
        let view = resolved.view()?;
        let result = crate::centrality::betweenness::betweenness_centrality(handle, self, view.as_ref(), params);
        result
    }

    /// Compute exact betweenness centrality over all vertices.
    pub fn betweenness_centrality_all(
        &mut self,
        handle: &EnteredContext<'_>,
        params: &crate::centrality::BetweennessCentralityParams,
    ) -> Result<crate::centrality::BetweennessCentralityResult> {
        self.betweenness_centrality(handle, crate::structure::VertexSelection::<i32>::All, params)
    }

    /// Compute edge betweenness centrality on this graph.
    ///
    /// Pass [`VertexSelection::All`] for exact computation, or provide seed
    /// vertices for approximate computation via [`VertexSelection::Device`] or
    /// [`VertexSelection::Host`].
    pub fn edge_betweenness_centrality<'a, V: CugraphElement + 'a>(
        &mut self,
        handle: &EnteredContext<'_>,
        vertex_list: impl Into<VertexSelection<'a, V>>,
        params: &crate::centrality::EdgeBetweennessCentralityParams,
    ) -> Result<crate::centrality::EdgeBetweennessCentralityResult> {
        let vertex_list = vertex_list.into();
        self.validate_vertex_selection(&vertex_list, "Graph::edge_betweenness_centrality", "vertex_list", true)?;
        let resolved = vertex_list.resolve(handle)?;
        let view = resolved.view()?;
        crate::centrality::edge_betweenness::edge_betweenness_centrality(handle, self, view.as_ref(), params)
    }

    /// Compute exact edge betweenness centrality over all vertices.
    pub fn edge_betweenness_centrality_all(
        &mut self,
        handle: &EnteredContext<'_>,
        params: &crate::centrality::EdgeBetweennessCentralityParams,
    ) -> Result<crate::centrality::EdgeBetweennessCentralityResult> {
        self.edge_betweenness_centrality(handle, crate::structure::VertexSelection::<i32>::All, params)
    }

    /// Compute weakly connected components.
    ///
    /// cuGraph requires the input graph to be symmetric for this algorithm.
    /// This wrapper does not add a Rust-side graph-property hard failure; use
    /// `do_expensive_check` in [`ConnectedComponentsParams`] to request native
    /// validation.
    ///
    /// [`ConnectedComponentsParams`]: crate::labeling::ConnectedComponentsParams
    pub fn weakly_connected_components(
        &mut self,
        handle: &EnteredContext<'_>,
        params: &crate::labeling::ConnectedComponentsParams,
    ) -> Result<crate::labeling::WeaklyConnectedComponentsResult> {
        crate::labeling::connected_components::weakly_connected_components(handle, self, params)
    }

    /// Compute strongly connected components.
    pub fn strongly_connected_components(
        &mut self,
        handle: &EnteredContext<'_>,
        params: &crate::labeling::ConnectedComponentsParams,
    ) -> Result<crate::labeling::StronglyConnectedComponentsResult> {
        crate::labeling::connected_components::strongly_connected_components(handle, self, params)
    }

    /// Compute core numbers for all vertices.
    pub fn core_number(
        &mut self,
        handle: &EnteredContext<'_>,
        params: &crate::structure::CoreNumberParams,
    ) -> Result<crate::structure::CoreNumberResult> {
        crate::structure::core::core_number(handle, self, params)
    }

    /// Extract a k-core subgraph, computing core numbers internally.
    pub fn k_core(
        &mut self,
        handle: &EnteredContext<'_>,
        params: &crate::structure::KCoreParams,
    ) -> Result<crate::structure::KCoreResult> {
        crate::structure::core::k_core(handle, self, params, None)
    }

    /// Extract a k-core subgraph using a precomputed core-number result.
    pub fn k_core_with_core_numbers(
        &mut self,
        handle: &EnteredContext<'_>,
        params: &crate::structure::KCoreParams,
        core_numbers: &crate::structure::CoreNumberResult,
    ) -> Result<crate::structure::KCoreResult> {
        crate::structure::core::k_core(handle, self, params, Some(core_numbers))
    }

    /// Compute a minimum spanning tree.
    pub fn minimum_spanning_tree(
        &mut self,
        handle: &EnteredContext<'_>,
        params: &crate::tree::MinimumSpanningTreeParams,
    ) -> Result<crate::tree::MinimumSpanningTreeResult> {
        crate::tree::minimum_spanning_tree::minimum_spanning_tree(handle, self, params)
    }

    /// Create an owned vertex-pair collection for pairwise similarity algorithms.
    ///
    /// `first` and `second` must be explicit vertex selections with matching
    /// lengths and the same vertex dtype as this graph. `VertexSelection::All`
    /// is rejected because pairwise similarity requires aligned input pairs.
    pub fn vertex_pairs<'a, V: CugraphElement + 'a>(
        &mut self,
        handle: &EnteredContext<'_>,
        first: impl Into<VertexSelection<'a, V>>,
        second: impl Into<VertexSelection<'a, V>>,
    ) -> Result<crate::similarity::VertexPairs> {
        let first = first.into();
        let second = second.into();
        self.validate_vertex_selection(&first, "Graph::vertex_pairs", "first", false)?;
        self.validate_vertex_selection(&second, "Graph::vertex_pairs", "second", false)?;
        let first = first.resolve(handle)?;
        let second = second.resolve(handle)?;
        let first_view = self.resolved_vertex_view(&first, "Graph::vertex_pairs", "first")?;
        let second_view = self.resolved_vertex_view(&second, "Graph::vertex_pairs", "second")?;
        crate::similarity::create_vertex_pairs(handle, self, &first_view, &second_view)
            .map_err(|error| error.with_operation("Graph::vertex_pairs"))
    }

    /// Compute Jaccard similarity coefficients for explicit vertex pairs.
    ///
    /// cuGraph requires a symmetric graph for this algorithm; native
    /// validation reports detailed graph-property errors.
    pub fn jaccard_coefficients(
        &mut self,
        handle: &EnteredContext<'_>,
        pairs: crate::similarity::VertexPairs,
        params: &crate::similarity::SimilarityParams,
    ) -> Result<crate::similarity::JaccardSimilarityResult> {
        crate::similarity::jaccard_coefficients(handle, self, pairs, params)
    }

    /// Compute Sorensen similarity coefficients for explicit vertex pairs.
    ///
    /// cuGraph requires a symmetric graph for this algorithm; native
    /// validation reports detailed graph-property errors.
    pub fn sorensen_coefficients(
        &mut self,
        handle: &EnteredContext<'_>,
        pairs: crate::similarity::VertexPairs,
        params: &crate::similarity::SimilarityParams,
    ) -> Result<crate::similarity::SorensenSimilarityResult> {
        crate::similarity::sorensen_coefficients(handle, self, pairs, params)
    }

    /// Compute overlap similarity coefficients for explicit vertex pairs.
    ///
    /// cuGraph requires a symmetric graph for this algorithm; native
    /// validation reports detailed graph-property errors.
    pub fn overlap_coefficients(
        &mut self,
        handle: &EnteredContext<'_>,
        pairs: crate::similarity::VertexPairs,
        params: &crate::similarity::SimilarityParams,
    ) -> Result<crate::similarity::OverlapSimilarityResult> {
        crate::similarity::overlap_coefficients(handle, self, pairs, params)
    }

    /// Compute cosine similarity coefficients for explicit vertex pairs.
    ///
    /// cuGraph requires a symmetric graph for this algorithm; native
    /// validation reports detailed graph-property errors.
    pub fn cosine_similarity_coefficients(
        &mut self,
        handle: &EnteredContext<'_>,
        pairs: crate::similarity::VertexPairs,
        params: &crate::similarity::SimilarityParams,
    ) -> Result<crate::similarity::CosineSimilarityResult> {
        crate::similarity::cosine_similarity_coefficients(handle, self, pairs, params)
    }

    /// Compute all-pairs Jaccard similarity coefficients.
    ///
    /// Pass [`VertexSelection::All`] to use all vertices or an explicit subset
    /// to seed pair generation. cuGraph requires a symmetric graph.
    pub fn all_pairs_jaccard_coefficients<'a, V: CugraphElement + 'a>(
        &mut self,
        handle: &EnteredContext<'_>,
        vertices: impl Into<VertexSelection<'a, V>>,
        params: &crate::similarity::AllPairsSimilarityParams,
    ) -> Result<crate::similarity::JaccardSimilarityResult> {
        let vertices = vertices.into();
        self.validate_vertex_selection(&vertices, "Graph::all_pairs_jaccard_coefficients", "vertices", true)?;
        let resolved = vertices.resolve(handle)?;
        let view = resolved.view()?;
        crate::similarity::all_pairs_jaccard_coefficients(handle, self, view.as_ref(), params)
    }

    /// Compute all-pairs Sorensen similarity coefficients.
    ///
    /// Pass [`VertexSelection::All`] to use all vertices or an explicit subset
    /// to seed pair generation. cuGraph requires a symmetric graph.
    pub fn all_pairs_sorensen_coefficients<'a, V: CugraphElement + 'a>(
        &mut self,
        handle: &EnteredContext<'_>,
        vertices: impl Into<VertexSelection<'a, V>>,
        params: &crate::similarity::AllPairsSimilarityParams,
    ) -> Result<crate::similarity::SorensenSimilarityResult> {
        let vertices = vertices.into();
        self.validate_vertex_selection(&vertices, "Graph::all_pairs_sorensen_coefficients", "vertices", true)?;
        let resolved = vertices.resolve(handle)?;
        let view = resolved.view()?;
        crate::similarity::all_pairs_sorensen_coefficients(handle, self, view.as_ref(), params)
    }

    /// Compute all-pairs overlap similarity coefficients.
    ///
    /// Pass [`VertexSelection::All`] to use all vertices or an explicit subset
    /// to seed pair generation. cuGraph requires a symmetric graph.
    pub fn all_pairs_overlap_coefficients<'a, V: CugraphElement + 'a>(
        &mut self,
        handle: &EnteredContext<'_>,
        vertices: impl Into<VertexSelection<'a, V>>,
        params: &crate::similarity::AllPairsSimilarityParams,
    ) -> Result<crate::similarity::OverlapSimilarityResult> {
        let vertices = vertices.into();
        self.validate_vertex_selection(&vertices, "Graph::all_pairs_overlap_coefficients", "vertices", true)?;
        let resolved = vertices.resolve(handle)?;
        let view = resolved.view()?;
        crate::similarity::all_pairs_overlap_coefficients(handle, self, view.as_ref(), params)
    }

    /// Compute all-pairs cosine similarity coefficients.
    ///
    /// Pass [`VertexSelection::All`] to use all vertices or an explicit subset
    /// to seed pair generation. cuGraph requires a symmetric graph.
    pub fn all_pairs_cosine_similarity_coefficients<'a, V: CugraphElement + 'a>(
        &mut self,
        handle: &EnteredContext<'_>,
        vertices: impl Into<VertexSelection<'a, V>>,
        params: &crate::similarity::AllPairsSimilarityParams,
    ) -> Result<crate::similarity::CosineSimilarityResult> {
        let vertices = vertices.into();
        self.validate_vertex_selection(&vertices, "Graph::all_pairs_cosine_similarity_coefficients", "vertices", true)?;
        let resolved = vertices.resolve(handle)?;
        let view = resolved.view()?;
        crate::similarity::all_pairs_cosine_similarity_coefficients(handle, self, view.as_ref(), params)
    }

    /// Compute a Force Atlas 2 layout for this graph.
    ///
    /// Returns a [`LayoutResult`](crate::layout::LayoutResult) containing
    /// vertex ids and (x, y) coordinates.
    ///
    /// For advanced features (per-vertex mass/radius, initial positions for
    /// refinement passes), use [`force_atlas2_with`](Self::force_atlas2_with).
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use cugraph::{
    /// #     Forward, ResourceHandle,
    /// #     cudf_interop::{DataFrameGraphOptions, graph_from_dataframe},
    /// #     layout::ForceAtlas2Params,
    /// #     random::RngState,
    /// # };
    /// # let runtime = std::sync::Arc::new(cudf::Runtime::new().unwrap());
    /// # let handle = ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).unwrap();
    /// # let entered = handle.enter("Graph::force_atlas2 docs").unwrap();
    /// # let mut rng = RngState::new(&entered, 42).unwrap();
    /// # let src = runtime.series(&[0i32, 1, 2]).unwrap();
    /// # let dst = runtime.series(&[1i32, 2, 0]).unwrap();
    /// # let edges = cudf::DataFrame::from_series([("src", src), ("dst", dst)]).unwrap();
    /// # let opts = DataFrameGraphOptions::new("src", "dst").renumber();
    /// # let mut graph = graph_from_dataframe::<Forward>(&entered, &edges, &opts).unwrap();
    /// let params = ForceAtlas2Params::default();
    /// let result = graph.force_atlas2(&entered, &mut rng, &params)?;
    /// # Ok::<(), cugraph::Error>(())
    /// ```
    pub fn force_atlas2(
        &mut self,
        handle: &EnteredContext<'_>,
        rng_state: &mut crate::random::RngState,
        params: &crate::layout::ForceAtlas2Params,
    ) -> Result<crate::layout::LayoutResult> {
        crate::layout::force_atlas2::force_atlas2::<T, i32, f32>(handle, rng_state, self, params, None, None)
    }

    /// Compute a Force Atlas 2 layout with optional per-vertex attributes
    /// and/or initial positions.
    ///
    /// This is the advanced variant of [`force_atlas2`](Self::force_atlas2).
    /// Use it when you need refinement passes (supplying a previous layout as
    /// initial positions) or per-vertex radius/mobility/mass.
    pub fn force_atlas2_with<V: CugraphElement, W: CugraphElement>(
        &mut self,
        handle: &EnteredContext<'_>,
        rng_state: &mut crate::random::RngState,
        params: &crate::layout::ForceAtlas2Params,
        vertex_attrs: Option<&crate::layout::ForceAtlas2VertexAttrs<'_, V, W>>,
        initial_positions: Option<&crate::layout::InitialPositions<'_, V, W>>,
    ) -> Result<crate::layout::LayoutResult> {
        crate::layout::force_atlas2::force_atlas2(handle, rng_state, self, params, vertex_attrs, initial_positions)
    }

    /// Compute only out-degrees for vertices in this graph.
    ///
    /// Pass [`VertexSelection::All`] to compute degrees for every vertex,
    /// [`VertexSelection::Device`] with a device view, or
    /// [`VertexSelection::Host`] with a host slice (uploaded on demand).
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::UnsupportedType`] if `vertices` uses a vertex
    /// dtype that does not match the graph.
    pub fn out_degrees<'a, V: CugraphElement + 'a>(
        &mut self,
        handle: &EnteredContext<'_>,
        vertices: impl Into<VertexSelection<'a, V>>,
    ) -> Result<crate::structure::OutDegreesResult> {
        let vertices = vertices.into();
        self.validate_vertex_selection(&vertices, "Graph::out_degrees", "vertices", true)?;
        let resolved = vertices.resolve(handle)?;
        let view = resolved.view()?;
        let result = crate::structure::degrees::out_degrees(handle, self, view.as_ref(), false);
        result
    }

    /// Compute out-degrees with optional expensive input validation.
    pub fn out_degrees_checked<'a, V: CugraphElement + 'a>(
        &mut self,
        handle: &EnteredContext<'_>,
        vertices: impl Into<VertexSelection<'a, V>>,
        do_expensive_check: bool,
    ) -> Result<crate::structure::OutDegreesResult> {
        let vertices = vertices.into();
        self.validate_vertex_selection(&vertices, "Graph::out_degrees", "vertices", true)?;
        let resolved = vertices.resolve(handle)?;
        let view = resolved.view()?;
        let result = crate::structure::degrees::out_degrees(handle, self, view.as_ref(), do_expensive_check);
        result
    }

    /// Compute out-degrees for every vertex in the graph.
    pub fn out_degrees_all(&mut self, handle: &EnteredContext<'_>) -> Result<crate::structure::OutDegreesResult> {
        self.out_degrees(handle, crate::structure::VertexSelection::<i32>::All)
    }

    /// Compute triangle counts for all vertices or an explicit start-vertex list.
    ///
    /// Pass [`VertexSelection::All`] to count triangles for every vertex,
    /// [`VertexSelection::Device`] with a device view, or
    /// [`VertexSelection::Host`] with a host slice uploaded on demand.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::UnsupportedType`] if `start_vertices` uses a
    /// vertex dtype that does not match the graph.
    pub fn triangle_count<'a, V: CugraphElement + 'a>(
        &mut self,
        handle: &EnteredContext<'_>,
        start_vertices: impl Into<VertexSelection<'a, V>>,
        params: &crate::community::TriangleCountParams,
    ) -> Result<crate::community::TriangleCountResult> {
        let start_vertices = start_vertices.into();
        self.validate_vertex_selection(&start_vertices, "Graph::triangle_count", "start_vertices", true)?;
        let resolved = start_vertices.resolve(handle)?;
        let view = resolved.view()?;
        crate::community::triangle_count::triangle_count(handle, self, view.as_ref(), params)
    }

    /// Compute triangle counts for every vertex in the graph.
    pub fn triangle_count_all(
        &mut self,
        handle: &EnteredContext<'_>,
        params: &crate::community::TriangleCountParams,
    ) -> Result<crate::community::TriangleCountResult> {
        self.triangle_count(handle, crate::structure::VertexSelection::<i32>::All, params)
    }

    /// Extract a k-truss subgraph.
    pub fn k_truss_subgraph(
        &mut self,
        handle: &EnteredContext<'_>,
        params: &crate::community::KTrussSubgraphParams,
    ) -> Result<crate::community::KTrussSubgraphResult> {
        crate::community::subgraph::k_truss_subgraph(handle, self, params)
    }

    /// Extract ego subgraphs for explicit source vertices.
    ///
    /// `VertexSelection::All` is rejected because the native API requires an
    /// explicit source vertex list.
    pub fn extract_ego<'a, V: CugraphElement + 'a>(
        &mut self,
        handle: &EnteredContext<'_>,
        source_vertices: impl Into<VertexSelection<'a, V>>,
        params: &crate::community::ExtractEgoParams,
    ) -> Result<crate::community::ExtractEgoResult> {
        let source_vertices = source_vertices.into();
        self.validate_vertex_selection(&source_vertices, "Graph::extract_ego", "source_vertices", false)?;
        let resolved = source_vertices.resolve(handle)?;
        let view = self.resolved_vertex_view(&resolved, "Graph::extract_ego", "source_vertices")?;
        crate::community::subgraph::extract_ego(handle, self, &view, params)
    }
}

fn free_graph_raw(raw: &mut *mut cugraph_graph_t, device: i32, operation: &'static str) {
    if !raw.is_null() {
        let _guard = DeviceGuard::switch_to_for_drop(device, operation).ok();
        // SAFETY: `raw` is owned by the caller and is freed at most once. The
        // pointer is nulled immediately after free to make repeated best-effort
        // cleanup paths harmless.
        unsafe { cugraph_graph_free(*raw) };
        *raw = ptr::null_mut();
    }
}

impl<T> Drop for Graph<T> {
    fn drop(&mut self) {
        if let Some(pending) = self.pending_input_completion.take() {
            pending.synchronize();
        }
        free_graph_raw(&mut self.raw, self.device, "Graph::drop");
    }
}

impl<T> Drop for ParkedGraph<T> {
    fn drop(&mut self) {
        {
            let _guard = DeviceGuard::switch_to_for_drop(self.device, "ParkedGraph::drop").ok();
            let _ = self.park_event.synchronize();
        }
        free_graph_raw(&mut self.raw, self.device, "ParkedGraph::drop");
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rapids_interop::{CompletionWitnessBackend, CompletionWitnessKind};

    use super::*;
    use crate::{array::DeviceArray, error::ErrorKind, handle::ResourceHandle};

    #[derive(Debug)]
    struct FailingCompletionWitness;

    impl CompletionWitnessBackend for FailingCompletionWitness {
        fn is_ready(&self) -> rapids_interop::Result<bool> {
            Ok(false)
        }

        fn synchronize(&self) -> rapids_interop::Result<()> {
            Err(InteropError::new(rapids_interop::InteropErrorKind::ReadinessUnavailable, "injected sync failure"))
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    fn dummy_runtime_bound_graph(
        handle: &EnteredContext<'_>,
        pending_input_completion: Option<DeferredInputCompletion>,
    ) -> Graph<Forward> {
        Graph {
            raw: ptr::null_mut(),
            device: handle.device(),
            properties: GraphProperties::default(),
            vertex_dtype: DataType::Int32,
            weight_dtype: DataType::Float32,
            has_weights: false,
            has_edge_ids: false,
            edge_id_dtype: None,
            pending_input_completion,
            resource_handle_leases: RefCell::new(handle.runtime_bound_lease().into_iter().collect()),
            untracked_resource_handle_use: Cell::new(false),
            _nosend_nosync: PhantomData,
            _transposition: PhantomData,
        }
    }

    #[test]
    fn test_graph_creation_triangle() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_graph_creation_triangle").expect("entered");

        let src = DeviceArray::from_slice(&entered, &[0i32, 1, 2]).expect("src");
        let dst = DeviceArray::from_slice(&entered, &[1i32, 2, 0]).expect("dst");
        let src_view = src.view().expect("src_view");
        let dst_view = dst.view().expect("dst_view");

        let graph = GraphBuilder::new(&src_view, &dst_view).build::<Forward>(&entered);
        assert!(graph.is_ok(), "graph creation failed: {:?}", graph.err());
    }

    #[test]
    fn test_graph_park_releases_pending_input_owner() {
        let runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));
        let handle = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("handle");
        let entered = handle.enter("test_graph_park_releases_pending_input_owner").expect("entered");

        let owner = Arc::new(());
        let retained_owner: Arc<dyn Any + Send + Sync> = owner.clone();
        let pending = DeferredInputCompletion::new(
            vec![retained_owner],
            vec![CompletionWitness::new(CompletionWitnessKind::ConsumerStream)],
        )
        .expect("pending");
        let graph = dummy_runtime_bound_graph(&entered, Some(pending));

        assert_eq!(Arc::strong_count(&owner), 2, "graph should retain the prepared input owner before park");
        let parked = graph.park(&entered).expect("park");
        assert_eq!(Arc::strong_count(&owner), 1, "park should release prepared input owners");
        drop(parked);
    }

    #[test]
    fn test_graph_park_surfaces_pending_input_completion_error() {
        let runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));
        let handle = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("handle");
        let entered = handle.enter("test_graph_park_surfaces_pending_input_completion_error").expect("entered");

        let owner = Arc::new(());
        let retained_owner: Arc<dyn Any + Send + Sync> = owner.clone();
        let pending = DeferredInputCompletion::new(
            vec![retained_owner],
            vec![CompletionWitness::with_backend(
                CompletionWitnessKind::ConsumerStream,
                Arc::new(FailingCompletionWitness),
            )],
        )
        .expect("pending");
        let graph = dummy_runtime_bound_graph(&entered, Some(pending));

        let err = graph.park(&entered).expect_err("park should fail");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert_eq!(err.operation(), "Graph::park");
        assert_eq!(Arc::strong_count(&owner), 1, "failed park should drop graph and retained input owner");
    }

    #[test]
    fn test_graph_park_rejects_non_runtime_bound_transfer_handle() {
        let runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));
        let build_handle = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("build handle");
        let build_entered =
            build_handle.enter("test_graph_park_rejects_non_runtime_bound_transfer_handle_build").expect("entered");
        let local_handle = ResourceHandle::new().expect("local handle");
        let local_entered =
            local_handle.enter("test_graph_park_rejects_non_runtime_bound_transfer_handle_local").expect("entered");
        let graph = dummy_runtime_bound_graph(&build_entered, None);

        let err = graph.park(&local_entered).expect_err("park should reject non-runtime-bound handle");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert_eq!(err.operation(), "Graph::park");
    }

    #[test]
    fn test_graph_runtime_bound_handle_leases_dedup_by_runtime_owner() {
        let runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));
        let build_handle = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("build handle");
        let build_entered =
            build_handle.enter("test_graph_runtime_bound_handle_leases_dedup_by_runtime_owner_build").expect("entered");
        let graph = dummy_runtime_bound_graph(&build_entered, None);
        assert_eq!(graph.resource_handle_leases.borrow().len(), 1);

        let algorithm_handle = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("algorithm handle");
        let algorithm_entered = algorithm_handle
            .enter("test_graph_runtime_bound_handle_leases_dedup_by_runtime_owner_algorithm")
            .expect("entered");
        graph
            .retain_runtime_bound_handle(
                &algorithm_entered,
                "test_graph_runtime_bound_handle_leases_dedup_by_runtime_owner",
            )
            .expect("retain runtime handle");

        assert_eq!(graph.resource_handle_leases.borrow().len(), 1);
    }

    #[test]
    fn test_graph_park_rejects_graph_tainted_by_non_runtime_bound_use() {
        let runtime = Arc::new(cudf::Runtime::builder().build().expect("runtime"));
        let runtime_handle = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("runtime handle");
        let runtime_entered = runtime_handle
            .enter("test_graph_park_rejects_graph_tainted_by_non_runtime_bound_use_runtime")
            .expect("entered");
        let local_handle = ResourceHandle::new().expect("local handle");
        let local_entered = local_handle
            .enter("test_graph_park_rejects_graph_tainted_by_non_runtime_bound_use_local")
            .expect("entered");
        let graph = dummy_runtime_bound_graph(&runtime_entered, None);

        graph
            .retain_runtime_bound_handle(
                &local_entered,
                "test_graph_park_rejects_graph_tainted_by_non_runtime_bound_use",
            )
            .expect("taint graph");
        let err = graph.park(&runtime_entered).expect_err("park should reject tainted graph");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert_eq!(err.operation(), "Graph::park");
    }

    #[test]
    fn test_graph_creation_rejects_mismatched_src_dst_lengths() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_graph_creation_rejects_mismatched_src_dst_lengths").expect("entered");

        let src = DeviceArray::from_slice(&entered, &[0i32, 1, 2]).expect("src");
        let dst = DeviceArray::from_slice(&entered, &[1i32, 2]).expect("dst");
        let src_view = src.view().expect("src_view");
        let dst_view = dst.view().expect("dst_view");

        let err = match GraphBuilder::new(&src_view, &dst_view).build::<Forward>(&entered) {
            Ok(_) => panic!("should reject mismatched edge-list lengths"),
            Err(err) => err,
        };
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn test_graph_creation_with_edge_ids_and_edge_type_ids() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_graph_creation_with_edge_ids_and_edge_type_ids").expect("entered");

        let src = DeviceArray::from_slice(&entered, &[0i32, 1, 2]).expect("src");
        let dst = DeviceArray::from_slice(&entered, &[1i32, 2, 0]).expect("dst");
        let edge_ids = DeviceArray::from_slice(&entered, &[100i32, 101, 102]).expect("edge_ids");
        let edge_types = DeviceArray::from_slice(&entered, &[0i32, 1, 0]).expect("edge_types");
        let src_view = src.view().expect("src_view");
        let dst_view = dst.view().expect("dst_view");
        let edge_ids_view = edge_ids.view().expect("edge_ids_view");
        let edge_types_view = edge_types.view().expect("edge_types_view");

        let graph = GraphBuilder::new(&src_view, &dst_view)
            .edge_ids(&edge_ids_view)
            .edge_type_ids(&edge_types_view)
            .renumber()
            .build::<Forward>(&entered);
        assert!(graph.is_ok(), "graph creation with edge_ids and edge_type_ids failed: {:?}", graph.err());
    }

    #[test]
    fn test_graph_creation_with_weights_and_edge_properties() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_graph_creation_with_weights_and_edge_properties").expect("entered");

        let src = DeviceArray::from_slice(&entered, &[0i32, 1, 2, 3]).expect("src");
        let dst = DeviceArray::from_slice(&entered, &[1i32, 2, 3, 0]).expect("dst");
        let wgt = DeviceArray::from_slice(&entered, &[1.0f32, 2.0, 3.0, 4.0]).expect("wgt");
        let edge_ids = DeviceArray::from_slice(&entered, &[0i32, 1, 2, 3]).expect("edge_ids");
        let edge_types = DeviceArray::from_slice(&entered, &[0i32, 0, 1, 1]).expect("edge_types");
        let src_view = src.view().expect("src_view");
        let dst_view = dst.view().expect("dst_view");
        let wgt_view = wgt.view().expect("wgt_view");
        let edge_ids_view = edge_ids.view().expect("edge_ids_view");
        let edge_types_view = edge_types.view().expect("edge_types_view");

        let graph = GraphBuilder::new(&src_view, &dst_view)
            .weights(&wgt_view)
            .edge_ids(&edge_ids_view)
            .edge_type_ids(&edge_types_view)
            .renumber()
            .build::<Forward>(&entered);
        assert!(graph.is_ok(), "graph creation with weights + edge properties failed: {:?}", graph.err());
    }

    #[test]
    fn test_graph_creation_rejects_mismatched_edge_ids_length() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_graph_creation_rejects_mismatched_edge_ids_length").expect("entered");

        let src = DeviceArray::from_slice(&entered, &[0i32, 1, 2]).expect("src");
        let dst = DeviceArray::from_slice(&entered, &[1i32, 2, 0]).expect("dst");
        let edge_ids = DeviceArray::from_slice(&entered, &[10i32, 20]).expect("edge_ids");
        let src_view = src.view().expect("src_view");
        let dst_view = dst.view().expect("dst_view");
        let edge_ids_view = edge_ids.view().expect("edge_ids_view");

        let err = GraphBuilder::new(&src_view, &dst_view)
            .edge_ids(&edge_ids_view)
            .build::<Forward>(&entered)
            .expect_err("should reject mismatched edge_ids length");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.message().contains("edge_ids"));
    }

    #[test]
    fn test_graph_creation_rejects_mismatched_edge_ids_dtype() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_graph_creation_rejects_mismatched_edge_ids_dtype").expect("entered");

        let src = DeviceArray::from_slice(&entered, &[0i32, 1, 2]).expect("src");
        let dst = DeviceArray::from_slice(&entered, &[1i32, 2, 0]).expect("dst");
        let edge_ids = DeviceArray::from_slice(&entered, &[10i64, 20, 30]).expect("edge_ids");
        let src_view = src.view().expect("src_view");
        let dst_view = dst.view().expect("dst_view");
        let edge_ids_view = edge_ids.view().expect("edge_ids_view");

        let err = GraphBuilder::new(&src_view, &dst_view)
            .edge_ids(&edge_ids_view)
            .build::<Forward>(&entered)
            .expect_err("should reject edge_ids dtype mismatch");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.message().contains("edge_ids must have the same dtype as src/dst"));
    }

    #[test]
    fn test_graph_creation_rejects_mismatched_edge_type_ids_length() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_graph_creation_rejects_mismatched_edge_type_ids_length").expect("entered");

        let src = DeviceArray::from_slice(&entered, &[0i32, 1, 2]).expect("src");
        let dst = DeviceArray::from_slice(&entered, &[1i32, 2, 0]).expect("dst");
        let edge_types = DeviceArray::from_slice(&entered, &[0i32, 1]).expect("edge_types");
        let src_view = src.view().expect("src_view");
        let dst_view = dst.view().expect("dst_view");
        let edge_types_view = edge_types.view().expect("edge_types_view");

        let err = GraphBuilder::new(&src_view, &dst_view)
            .edge_type_ids(&edge_types_view)
            .build::<Forward>(&entered)
            .expect_err("should reject mismatched edge_type_ids length");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.message().contains("edge_type_ids"));
    }

    #[test]
    fn test_rejects_symmetrize_with_edge_ids() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_rejects_symmetrize_with_edge_ids").expect("entered");

        let src = DeviceArray::from_slice(&entered, &[0i32, 1, 2]).expect("src");
        let dst = DeviceArray::from_slice(&entered, &[1i32, 2, 0]).expect("dst");
        let eids = DeviceArray::from_slice(&entered, &[10i32, 20, 30]).expect("eids");
        let src_view = src.view().expect("src_view");
        let dst_view = dst.view().expect("dst_view");
        let eids_view = eids.view().expect("eids_view");

        let err = GraphBuilder::new(&src_view, &dst_view)
            .edge_ids(&eids_view)
            .symmetrize()
            .build::<Forward>(&entered)
            .expect_err("should reject symmetrize + edge_ids");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.message().contains("symmetrize"));
        assert!(err.message().contains("edge_ids"));
    }

    #[test]
    fn test_rejects_symmetrize_with_edge_type_ids() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_rejects_symmetrize_with_edge_type_ids").expect("entered");

        let src = DeviceArray::from_slice(&entered, &[0i32, 1, 2]).expect("src");
        let dst = DeviceArray::from_slice(&entered, &[1i32, 2, 0]).expect("dst");
        let etids = DeviceArray::from_slice(&entered, &[0i32, 1, 0]).expect("etids");
        let src_view = src.view().expect("src_view");
        let dst_view = dst.view().expect("dst_view");
        let etids_view = etids.view().expect("etids_view");

        let err = GraphBuilder::new(&src_view, &dst_view)
            .edge_type_ids(&etids_view)
            .symmetrize()
            .build::<Forward>(&entered)
            .expect_err("should reject symmetrize + edge_type_ids");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.message().contains("symmetrize"));
        assert!(err.message().contains("edge_type_ids"));
    }

    #[test]
    fn test_rejects_start_time_without_end_time() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_rejects_start_time_without_end_time").expect("entered");

        let src = DeviceArray::from_slice(&entered, &[0i32, 1, 2]).expect("src");
        let dst = DeviceArray::from_slice(&entered, &[1i32, 2, 0]).expect("dst");
        let start = DeviceArray::from_slice(&entered, &[0i32, 1, 2]).expect("start");
        let src_view = src.view().expect("src_view");
        let dst_view = dst.view().expect("dst_view");
        let start_view = start.view().expect("start_view");

        let err = GraphBuilder::new(&src_view, &dst_view)
            .edge_start_time_ids(&start_view)
            .build::<Forward>(&entered)
            .expect_err("should reject start_time without end_time");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.message().contains("edge_end_time_ids"));
        assert!(err.message().contains("was not"));
    }

    #[test]
    fn test_builder_defaults_to_directed() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_builder_defaults_to_directed").expect("entered");

        let src = DeviceArray::from_slice(&entered, &[0i32, 1, 2]).expect("src");
        let dst = DeviceArray::from_slice(&entered, &[1i32, 2, 0]).expect("dst");

        let src_view = src.view().expect("src_view");
        let dst_view = dst.view().expect("dst_view");
        let builder = GraphBuilder::new(&src_view, &dst_view);
        let debug = format!("{:?}", builder);

        // properties defaults to directed simple graph
        assert!(debug.contains("is_symmetric: false"), "should default to directed");
        assert!(debug.contains("is_multigraph: false"), "should default to simple graph");
    }

    #[test]
    fn test_from_raw_views_preserves_known_dtypes_in_builder_state() {
        let handle = ResourceHandle::new().expect("handle");
        let entered = handle.enter("test_from_raw_views_preserves_known_dtypes_in_builder_state").expect("entered");

        let src = DeviceArray::from_slice(&entered, &[0i32, 1, 2]).expect("src");
        let dst = DeviceArray::from_slice(&entered, &[1i32, 2, 0]).expect("dst");
        let wgt = DeviceArray::from_slice(&entered, &[1.0f64, 2.0, 3.0]).expect("wgt");

        let src_raw = src.raw_view().expect("src_raw");
        let dst_raw = dst.raw_view().expect("dst_raw");
        let wgt_raw = wgt.raw_view().expect("wgt_raw");

        let builder = GraphBuilder::from_raw_views(GraphProperties::default(), &src_raw, &dst_raw)
            .expect("builder from raw views")
            .weights_raw(&wgt_raw)
            .expect("weights_raw");

        assert_eq!(builder.vertex_dtype, Some(DataType::Int32));
        assert_eq!(builder.weight_dtype, Some(DataType::Float64));
    }
}
