//! Query-oriented borrowed execution facade.

use rapids_interop::{ExecutionBinding, InteropError};

use super::{CudaStream, DeviceId, ExecutionOverrides, ExecutionPlacement, Runtime};
use crate::core::memory::{DeviceMemoryManager, DeviceMemoryResource, StreamReservationLease};

/// Borrowed query execution context over a [`Runtime`].
///
/// This is a convenience facade for query pipelines that need to pass the
/// same runtime-owned stream and memory resource through many operations. It
/// does not own scheduling policy, memory reservation, retry behavior, or a
/// separate execution domain. Outputs still retain provenance through the
/// underlying operation's normal [`ExecutionOverrides`] resolution.
///
/// Construct via [`Runtime::query_execution`].
#[derive(Clone, Copy)]
#[must_use]
pub struct QueryExecutionContext<'a> {
    runtime: &'a Runtime,
    overrides: ExecutionOverrides<'a>,
}

impl Runtime {
    /// Borrow this runtime as a query execution context.
    pub fn query_execution(&self) -> QueryExecutionContext<'_> {
        QueryExecutionContext::new(self)
    }
}

impl<'a> QueryExecutionContext<'a> {
    fn new(runtime: &'a Runtime) -> Self {
        let overrides =
            ExecutionOverrides::new().with_stream(runtime.stream()).with_memory_resource(runtime.memory_resource());
        Self { runtime, overrides }
    }

    /// The runtime that owns this query context's execution resources.
    pub fn runtime(&self) -> &'a Runtime {
        self.runtime
    }

    /// Borrowed per-operation overrides carrying the runtime stream and memory resource.
    pub fn overrides(&self) -> &ExecutionOverrides<'a> {
        &self.overrides
    }

    /// Diagnostic retained placement for this query context.
    pub fn placement(&self) -> ExecutionPlacement {
        self.runtime.placement()
    }

    /// CUDA device pinned by the runtime.
    pub fn device(&self) -> DeviceId {
        self.runtime.device()
    }

    /// Runtime-owned CUDA stream used by this query context.
    pub fn stream(&self) -> &'a CudaStream {
        self.runtime.stream()
    }

    /// Runtime-owned device memory resource used by this query context.
    pub fn memory_resource(&self) -> &'a DeviceMemoryResource {
        self.runtime.memory_resource()
    }

    /// Resolve the device-level reservation-aware memory manager for this query context.
    pub fn device_memory_manager(&self) -> crate::Result<DeviceMemoryManager> {
        DeviceMemoryManager::for_device(self.device())
    }

    /// Attach a hard memory reservation to this query context's runtime stream.
    ///
    /// The returned lease supplies execution overrides whose stream is this
    /// context's stream and whose memory resource is the device-level
    /// reservation-aware adaptor. Dropping the lease detaches the reservation
    /// from the stream while retained allocations remain accounted to the
    /// reservation identity that created them.
    pub fn reserve_stream(&self, reservation_id: u64, limit_bytes: usize) -> crate::Result<StreamReservationLease> {
        self.device_memory_manager()?.reserve_stream(self.stream(), reservation_id, limit_bytes)
    }

    /// Borrow the runtime's explicit execution binding for cross-crate interop.
    pub fn execution_binding(&self) -> std::result::Result<ExecutionBinding<'_>, InteropError> {
        self.runtime.execution_binding()
    }
}

impl std::fmt::Debug for QueryExecutionContext<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryExecutionContext")
            .field("device", &self.device())
            .field("placement", &self.placement())
            .finish()
    }
}
