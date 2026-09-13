//! Parked graph transfer state for cross-thread moves.

use std::{
    cell::{Cell, RefCell},
    marker::PhantomData,
    mem::ManuallyDrop,
    ptr,
};

use cudf::runtime::{CudaEvent, ExecutionBindingId};
use cugraph_sys::cugraph_graph_t;

use super::{Forward, Graph, map_cudf_runtime_error};
use crate::{
    cuda::{DeviceResourceKind, drop_device_resource, quarantine_device_drop},
    error::{Error, ErrorKind, Result},
    handle::{EnteredContext, ResourceHandleLease},
    types::{DataType, GraphProperties},
};

/// A parked cuGraph graph that can be moved across Rust threads.
///
/// `Graph<T>` itself remains `!Send`/`!Sync` because native cuGraph operations
/// require a thread-local entered execution context. `ParkedGraph<T>` is the
/// transfer state for an idle graph: it owns the native graph pointer, retains
/// runtime-bound handle leases needed for teardown, and carries a CUDA event
/// recorded after tracked work on the parking binding's stream, plus that
/// binding's identity. Exact [`resume`](ParkedGraph::resume) requires an
/// entered context for the binding that parked the graph;
/// [`resume_under`](ParkedGraph::resume_under) executes under a different
/// same-device binding after ordering it behind the park event.
pub struct ParkedGraph<T = Forward> {
    pub(super) raw: *mut cugraph_graph_t,
    pub(super) device: i32,
    pub(super) properties: GraphProperties,
    pub(super) vertex_dtype: DataType,
    pub(super) weight_dtype: DataType,
    pub(super) has_weights: bool,
    pub(super) edge_id_dtype: Option<DataType>,
    pub(super) park_event: CudaEvent,
    // Binding whose stream the park event was recorded on. Exact resume
    // requires this binding; `resume_under` orders another same-device
    // binding behind the event instead.
    pub(super) park_binding: ExecutionBindingId,
    pub(super) resource_handle_leases: Vec<ResourceHandleLease>,
    pub(super) _not_sync: PhantomData<Cell<()>>,
    pub(super) _transposition: PhantomData<T>,
}

// SAFETY: parked graphs do not permit native graph use. The raw graph pointer
// is only resumed into a non-Send `Graph<T>` under a matching entered runtime
// context, or freed after the park event completes. Runtime-bound leases are
// retention-only metadata and are Send under their own safety contract.
unsafe impl<T: Send> Send for ParkedGraph<T> {}

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
    /// Resume is exact-binding by construction: `handle` must use the cudf
    /// runtime stream/MR binding that parked the graph. The runtime stream
    /// waits on the park event before the live [`Graph`] is returned to the
    /// caller. Retained creation leases stay untouched for teardown; use
    /// [`resume_under`](ParkedGraph::resume_under) to execute under a
    /// different same-device binding.
    ///
    /// If this method returns an error, the consumed parked graph is dropped.
    pub fn resume(self, handle: &EnteredContext<'_>) -> Result<Graph<T>> {
        let operation = "ParkedGraph::resume";
        self.validate_resume_handle(handle, operation)?;
        self.resume_ordered(handle, operation)
    }

    /// Resume this parked graph under a different runtime-bound execution
    /// binding on the same device (storage/execution split).
    ///
    /// The resume stream waits on the park event, so every storage write that
    /// produced the parked state happens before any borrower kernel reads it.
    /// Algorithm temporaries allocate through the entering handle, so
    /// workspace, auxiliary streams, and outputs charge to the borrower's
    /// domain; never to the parking domain. cuGraph algorithms observe the
    /// graph through a const view: the borrower must not mutate shared
    /// storage, and the exclusive checkout guard that owns this parked graph
    /// must outlive every kernel launched under the borrower binding.
    /// Teardown still runs under the retained creation leases. Re-park with
    /// [`park_under`](Graph::park_under) under the same borrower binding.
    ///
    /// If this method returns an error, the consumed parked graph is dropped.
    pub fn resume_under(self, handle: &EnteredContext<'_>) -> Result<Graph<T>> {
        let operation = "ParkedGraph::resume_under";
        if handle.device() != self.device {
            return Err(Error::new(ErrorKind::InvalidInput, operation)
                .with_message("parked graph and resource handle must be on the same CUDA device")
                .with_context("graph_device", self.device.to_string())
                .with_context("handle_device", handle.device().to_string()));
        }
        if handle.runtime().is_none() {
            return Err(Error::new(ErrorKind::InvalidState, operation)
                .with_message("parked graph resume requires a runtime-bound resource handle"));
        }
        self.resume_ordered(handle, operation)
    }

    fn resume_ordered(self, handle: &EnteredContext<'_>, operation: &'static str) -> Result<Graph<T>> {
        let runtime = handle.runtime().expect("validated runtime-bound handle");
        runtime
            .stream()
            .wait_event(&self.park_event)
            .map_err(|error| map_cudf_runtime_error(error, operation, "failed to wait on graph park CUDA event"))?;
        let ordered_binding = runtime.binding_id();

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
            edge_id_dtype: this.edge_id_dtype,
            pending_input_completion: None,
            resource_handle_leases: RefCell::new(resource_handle_leases),
            ordered_binding: Cell::new(Some(ordered_binding)),
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

        let Some(runtime) = handle.runtime() else {
            return Err(Error::new(ErrorKind::InvalidState, operation)
                .with_message("parked graph resume requires a runtime-bound resource handle"));
        };
        if runtime.binding_id() != self.park_binding {
            return Err(Error::new(ErrorKind::InvalidState, operation)
                .with_message("parked graph execution binding does not match the resume handle")
                .with_context("graph_binding_id", self.park_binding.to_string())
                .with_context("handle_binding_id", runtime.binding_id().to_string()));
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

fn free_parked_graph_raw_with_leases(
    raw: &mut *mut cugraph_graph_t,
    device: i32,
    park_event: &CudaEvent,
    leases: &[ResourceHandleLease],
) {
    let mut free = || match park_event.synchronize() {
        Ok(()) => drop_device_resource(raw, device, "ParkedGraph::drop", DeviceResourceKind::Graph),
        Err(error) => {
            let owned = std::mem::replace(raw, ptr::null_mut());
            if !owned.is_null() {
                quarantine_device_drop("ParkedGraph::drop", device, &error);
            }
        },
    };
    if let Some(lease) = leases.first() {
        lease.with_teardown_gate(free);
    } else {
        free();
    }
}

impl<T> Drop for ParkedGraph<T> {
    fn drop(&mut self) {
        free_parked_graph_raw_with_leases(&mut self.raw, self.device, &self.park_event, &self.resource_handle_leases);
    }
}
