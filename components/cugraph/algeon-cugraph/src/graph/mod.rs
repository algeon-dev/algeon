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
//! - **[`Forward`]**: required by [`Graph::bfs`] and [`Graph::louvain`]
//! - **[`Transposed`]**: required by [`Graph::pagerank`],
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
    cell::{Cell, RefCell},
    marker::PhantomData,
    mem::ManuallyDrop,
    ptr,
};

use cudf::{
    interop::contract::{InteropError, LeaseCompletion},
    runtime::{CudaEvent, ExecutionBindingId},
};
use cugraph_sys::{cugraph_decompress_to_edgelist, cugraph_edgelist_t, cugraph_error_t, cugraph_graph_t};

use crate::{
    array::{DeviceArray, DeviceArrayView},
    cuda::{DeviceGuard, DeviceResourceKind, drop_device_resource},
    error::{Error, ErrorKind, Result, check_error},
    handle::{EnteredContext, ResourceHandleLease},
    random::RngState,
    sampling::select_random_vertices,
    structure::{ResolvedVertexSelection, VertexSelection},
    types::{CugraphElement, DataType, GraphProperties, to_bool_t},
};

mod algorithms;
mod builder;
mod edgelist;
mod parked;

pub use builder::GraphBuilder;
pub(crate) use edgelist::DecompressedEdgelist;
pub use parked::ParkedGraph;

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::Forward {}
    impl Sealed for super::Transposed {}
}

/// Trait implemented by the graph transposition marker types.
///
/// This trait is [sealed]; it cannot be implemented outside this crate.
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
/// - [`Forward`]: stored in forward (non-transposed) form (the default)
/// - [`Transposed`]: stored in transposed form
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
    edge_id_dtype: Option<DataType>,
    pending_input_completion: Option<DeferredInputCompletion>,
    // Runtime-bound resource handles of the binding that owns the graph's
    // native allocations can own state referenced during cugraph_graph_free.
    // Keep those leases alive until after Graph::drop runs. Handles of a
    // borrowed execution binding are not retained here: their allocations
    // are call-scoped and must retire with the borrowing attempt.
    resource_handle_leases: RefCell<Vec<ResourceHandleLease>>,
    // Binding whose stream is ordered after every tracked graph operation.
    // Established by construction or resume; every runtime-bound use must
    // enter under it so one park event keeps proving full stream ordering.
    // `None` until the first runtime-bound use of a graph built outside a
    // runtime binding.
    ordered_binding: Cell<Option<ExecutionBindingId>>,
    // A graph touched through local or external handles has work whose stream
    // is not represented by the runtime-bound lease ledger. Such a graph
    // cannot be parked because one runtime event cannot fence those uses.
    untracked_resource_handle_use: Cell<bool>,
    _nosend_nosync: PhantomData<*mut ()>,
    _transposition: PhantomData<T>,
}

pub(crate) struct DeferredInputCompletion {
    completions: Vec<LeaseCompletion>,
}

impl DeferredInputCompletion {
    pub(crate) fn new(completions: Vec<LeaseCompletion>) -> Option<Self> {
        if completions.is_empty() {
            return None;
        }

        Some(Self { completions })
    }

    pub(crate) fn synchronize(&mut self, operation: &'static str) -> Result<()> {
        for completion in &mut self.completions {
            completion.synchronize().map_err(|error| map_completion_witness_error(error, operation))?;
        }
        Ok(())
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
        self.edge_id_dtype.is_some()
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

    /// Waits for graph construction to finish reading prepared inputs and
    /// releases their retained producer owners.
    ///
    /// Prepared construction remains asynchronous until this boundary, parking,
    /// or graph destruction. This method waits on the construction completion
    /// witnesses; it does not wait for algorithm work submitted afterward.
    /// On failure, unfinished witnesses and their owners remain retained.
    pub fn synchronize_input_reads(&mut self) -> Result<()> {
        self.finish_input_reads("Graph::synchronize_input_reads")
    }

    fn finish_input_reads(&mut self, operation: &'static str) -> Result<()> {
        if let Some(pending) = self.pending_input_completion.as_mut() {
            pending.synchronize(operation)?;
        }
        self.pending_input_completion = None;
        Ok(())
    }

    /// Park this graph for transfer to another Rust thread.
    ///
    /// Parking consumes the non-Send [`Graph`] and returns a [`ParkedGraph`]
    /// that is `Send` but not `Sync`. The graph must have been built and used
    /// only through runtime-bound handles for the same exact cudf execution
    /// binding as `handle`; otherwise a single park event cannot prove all
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

        self.finish_input_reads(operation)?;
        self.park_recorded(handle, operation)
    }

    /// Park this graph under the ordered execution binding without requiring
    /// it to match the construction binding (storage/execution split).
    ///
    /// The entering binding must be the ordered binding: every tracked use
    /// ran on its stream, so the park event recorded below stays ordered
    /// after all of them. Teardown still runs under the retained creation
    /// leases. Use this after [`resume_under`](ParkedGraph::resume_under);
    /// parking a graph that ran only under its construction binding must keep
    /// using [`park`](Graph::park).
    ///
    /// If this method returns an error, the consumed graph is dropped.
    pub fn park_under(mut self, handle: &EnteredContext<'_>) -> Result<ParkedGraph<T>>
    where T: Send {
        let operation = "Graph::park_under";
        if handle.device() != self.device {
            return Err(Error::new(ErrorKind::InvalidInput, operation)
                .with_message("graph and resource handle must be on the same CUDA device")
                .with_context("graph_device", self.device.to_string())
                .with_context("handle_device", handle.device().to_string()));
        }
        let Some(handle_lease) = handle.runtime_bound_lease() else {
            return Err(Error::new(ErrorKind::InvalidState, operation)
                .with_message("park_under requires a runtime-bound resource handle"));
        };
        if self.untracked_resource_handle_use.get() {
            return Err(Error::new(ErrorKind::InvalidState, operation)
                .with_message("graph was used with a non-runtime-bound resource handle and cannot be parked"));
        }
        if self.resource_handle_leases.get_mut().is_empty() {
            return Err(Error::new(ErrorKind::InvalidState, operation)
                .with_message("graph has no runtime-bound handle leases to prove park stream ownership"));
        }
        let entering = handle_lease.runtime().binding_id();
        if self.ordered_binding.get() != Some(entering) {
            return Err(Error::new(ErrorKind::InvalidState, operation)
                .with_message("park handle binding does not match the ordered graph execution binding")
                .with_context(
                    "graph_binding_id",
                    self.ordered_binding.get().map(|id| id.to_string()).as_deref().unwrap_or("none"),
                )
                .with_context("handle_binding_id", entering.to_string()));
        }
        self.finish_input_reads(operation)?;
        self.park_recorded(handle, operation)
    }

    fn park_recorded(self, handle: &EnteredContext<'_>, operation: &'static str) -> Result<ParkedGraph<T>> {
        let _guard = DeviceGuard::enter_owner_device(self.device, operation)?;
        let event = CudaEvent::new()
            .map_err(|error| map_cudf_runtime_error(error, operation, "failed to create graph park CUDA event"))?;
        let runtime = handle.runtime().expect("validated runtime-bound handle");
        event
            .record(runtime.stream())
            .map_err(|error| map_cudf_runtime_error(error, operation, "failed to record graph park CUDA event"))?;
        let park_binding = runtime.binding_id();

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
            edge_id_dtype: this.edge_id_dtype,
            park_event: event,
            park_binding,
            resource_handle_leases,
            _not_sync: PhantomData,
            _transposition: PhantomData,
        })
    }

    fn validate_runtime_bound_transfer(&mut self, handle: &EnteredContext<'_>, operation: &'static str) -> Result<()> {
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

        let leases = self.resource_handle_leases.get_mut();
        if leases.is_empty() {
            return Err(Error::new(ErrorKind::InvalidState, operation)
                .with_message("graph has no runtime-bound handle leases to prove park stream ownership"));
        }
        if !leases.iter().all(|lease| lease.same_binding(&handle_lease)) {
            return Err(Error::new(ErrorKind::InvalidState, operation)
                .with_message("graph execution binding does not match the park handle")
                .with_context("lease_count", leases.len().to_string()));
        }
        // Matching the storage leases is not enough once a graph can execute
        // under a borrowed binding: the park event has to be recorded on the
        // stream every tracked use actually ran on.
        let entering = handle_lease.runtime().binding_id();
        if self.ordered_binding.get().is_some_and(|ordered| ordered != entering) {
            return Err(Error::new(ErrorKind::InvalidState, operation)
                .with_message("park handle binding does not match the ordered graph execution binding")
                .with_context(
                    "graph_binding_id",
                    self.ordered_binding.get().map(|id| id.to_string()).as_deref().unwrap_or("none"),
                )
                .with_context("handle_binding_id", entering.to_string()));
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

        let mut leases = self.resource_handle_leases.borrow_mut();

        // Every runtime-bound use must enter under the ordered binding, so
        // the park event recorded later is ordered after all tracked work.
        // The first runtime-bound use of a graph built outside a runtime
        // binding establishes the order.
        let entering = lease.runtime().binding_id();
        match self.ordered_binding.get() {
            Some(ordered) if ordered != entering => {
                return Err(Error::new(ErrorKind::InvalidState, operation)
                    .with_message("graph execution binding does not match the resource handle")
                    .with_context("graph_binding_id", ordered.to_string())
                    .with_context("handle_binding_id", entering.to_string()));
            },
            None => self.ordered_binding.set(Some(entering)),
            Some(_) => {},
        }

        // Only the binding that owns the graph's native allocations is
        // retained for teardown. A borrowed execution binding allocates just
        // call-scoped workspace, so retaining its lease would keep its
        // current-resource stream route; and the query domain behind it;
        // attached for as long as the parked graph lives, long after the
        // borrowing attempt released that stream.
        let owns_teardown = leases.first().is_none_or(|existing| existing.same_binding(&lease));
        if owns_teardown && !leases.iter().any(|existing| existing.same_native_owner(&lease)) {
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
        rng_state: &mut RngState,
        num_vertices: usize,
    ) -> Result<DeviceArray<V>> {
        select_random_vertices(handle, self, rng_state, num_vertices)
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
            check_error(code, error, handle.device(), operation)?;
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

fn free_graph_raw(raw: &mut *mut cugraph_graph_t, device: i32, operation: &'static str) {
    drop_device_resource(raw, device, operation, DeviceResourceKind::Graph);
}

fn free_graph_raw_with_leases(
    raw: &mut *mut cugraph_graph_t,
    device: i32,
    operation: &'static str,
    leases: &[ResourceHandleLease],
) {
    if let Some(lease) = leases.first() {
        // Teardown recovers exclusive access after use-path poison without
        // reopening normal entry or exposing the native owner value.
        lease.with_teardown_gate(|| free_graph_raw(raw, device, operation));
    } else {
        free_graph_raw(raw, device, operation);
    }
}

impl<T> Drop for Graph<T> {
    fn drop(&mut self) {
        // Fence input reads before native teardown; a failed fence leaks the
        // input owners (see `LeaseCompletion`).
        drop(self.pending_input_completion.take());
        free_graph_raw_with_leases(&mut self.raw, self.device, "Graph::drop", self.resource_handle_leases.get_mut());
    }
}

#[cfg(test)]
mod tests;
