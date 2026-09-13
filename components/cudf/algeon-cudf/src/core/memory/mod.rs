//! Device memory resource management (RMM).
//!
//! A device memory resource is an RMM allocator handle.
//!
//! In `cudf`, these handles participate in execution in two distinct ways:
//!
//! - [`crate::Runtime`] carries one default memory-resource handle as part of
//!   its retained-provenance bundle. By default this is the process-local
//!   [`DeviceMemoryManager`] resource for the runtime's CUDA device.
//! - [`crate::ExecutionOverrides`] can borrow `&DeviceMemoryResource` to
//!   override allocator selection for one supported operation without mutating
//!   the runtime's default.
//!
//! The low-level `_ex` FFI shims in `cudf-sys` accept explicit
//! memory-resource handles, and `cudf` routes supported operations and
//! constructors through the resolved allocator for the active runtime/override
//! context.
//!
//! Query-style bounded execution should use the device-level manager plus
//! stream-bound reservations: allocate a [`crate::Runtime`], create an
//! [`crate::ExecutionBudgetToken`] or [`StreamReservationLease`], and execute
//! with the lease's stream and manager resource. Query engines that need one
//! hard cap across several streams can create a [`ReservationArena`] and attach
//! stream leases as members under it. This keeps one long-lived allocator
//! adaptor per device while charging allocations to the active stream-bound
//! reservation and, when present, its parent arena.
//!
//! Explicit CUDA and pool resources remain available for advanced library use,
//! diagnostics, tests, and non-query workflows. Use
//! [`crate::RuntimeBuilder::with_pool_memory_resource`] when you specifically
//! want one fresh pool owned by a runtime, or [`DeviceMemoryResource::pool_on`]
//! / [`DeviceMemoryResource::cuda_on`] when you need an explicit handle first
//! so you can install it into one or more runtimes, or borrow it for a one-off
//! override. In all cases, the runtime, stream, and memory resource must be on
//! the same CUDA device.
//!
//! # Examples
//!
//! ```no_run
//! use cudf::{
//!     DeviceId, ExecutionOverrides, Runtime, SortOptions,
//!     core::memory::DeviceMemoryResource,
//! };
//!
//! let device = DeviceId::new(0).unwrap();
//!
//! // Default query-backend path: use the device-level manager resource.
//! let runtime = Runtime::builder().with_device(device).build().unwrap();
//! let manager = runtime.device_memory_manager().unwrap();
//! let lease = runtime.stream_reservation(7, 64 * 1024 * 1024).unwrap();
//!
//! assert_eq!(runtime.memory_resource().device(), device);
//! assert_eq!(manager.resource().device(), device);
//! assert_eq!(lease.memory_resource().device(), device);
//!
//! // Advanced: create a runtime-owned pool when you specifically need
//! // allocator isolation instead of stream-bound query reservations.
//! let runtime = Runtime::builder()
//!     .with_device(device)
//!     .with_pool_memory_resource(256 * 1024 * 1024, None)
//!     .build()
//!     .unwrap();
//!
//! // Advanced: create an allocator handle immediately on an explicit device.
//! let shared_pool = DeviceMemoryResource::pool_on(
//!     device,
//!     512 * 1024 * 1024, // 512 MiB initial pool
//!     None,              // None = unbounded maximum
//! )
//! .unwrap();
//!
//! // Install that handle as another runtime's owned default allocator.
//! let runtime_with_explicit_mr = Runtime::builder()
//!     .with_device(device)
//!     .with_memory_resource(shared_pool.clone())
//!     .build()
//!     .unwrap();
//!
//! // Or borrow it for one supported operation without changing the runtime's
//! // default allocator.
//! let series = runtime.series(&[3i32, 1, 2]).unwrap();
//! let ctx = ExecutionOverrides::new().with_memory_resource(&shared_pool);
//! let sorted = series.sort_values_in(SortOptions::new(), &ctx).unwrap();
//!
//! assert_eq!(runtime.memory_resource().device(), device);
//! assert_eq!(runtime_with_explicit_mr.memory_resource().device(), device);
//! assert_eq!(sorted.len(), 3);
//! ```

mod domain;
mod failure;
mod installation;
mod manager;
mod options;
mod resource;
mod usage;

#[cfg(test)]
mod tests;

pub use domain::{QueryMemoryDomain, ReservationArena, StreamReservationLease};
pub use failure::{
    AllocatorEvent, AllocatorEventAction, AllocatorEventCompleteness, AllocatorEventDomainIdentity, AllocatorEventRead,
    AllocatorEventRingBounds, DeviceMemoryFailureAllocationDomain, DeviceMemoryFailureInfo, DeviceMemoryFailureSource,
};
pub use installation::{
    DeviceMemoryManagerFailureCallbackState, DeviceMemoryManagerInstallationCause,
    DeviceMemoryManagerInstallationError, DeviceMemoryManagerInstallationIdentity,
    DeviceMemoryManagerInstallationOutcome, DeviceMemoryManagerInstallationOutcomeKind,
    DeviceMemoryManagerInstallationRetry, DeviceMemoryManagerInstallationState, DeviceMemoryManagerInstallationStatus,
};
pub use manager::{CurrentResourceRouteLease, DeviceMemoryManager};
pub use options::{DeviceMemoryManagerBase, DeviceMemoryManagerOptions, QueryMemoryDomainOptions};
pub use resource::DeviceMemoryResource;
pub use usage::{
    DeviceMemoryPoolUsage, DeviceMemoryResourceUsage, QueryMemoryDomainAllocation, QueryMemoryDomainIdentity,
    QueryMemoryDomainUsage, ReservationArenaUsage, ReservationUsage,
};
