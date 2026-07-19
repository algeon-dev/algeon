//! Runtime utilities for device-pinned execution resources and retained provenance.
//!
//! This module exposes the public runtime control surface:
//!
//! - [`Runtime`] owns one CUDA device binding plus default [`CudaStream`] and
//!   [`crate::core::memory::DeviceMemoryResource`] handles
//! - [`crate::core::memory::DeviceMemoryManager`] provides one reservation-aware
//!   allocator adaptor per CUDA device for query backends
//! - [`ExecutionOverrides`] borrows a stream and/or memory resource for one
//!   operation
//! - [`CudaStream`] and [`CudaEvent`] expose low-level CUDA ordering handles
//! - [`DeviceId`] is a validated CUDA device ordinal

mod affinity;
mod binding_extension;
mod boundary;
mod budget;
pub mod device;
pub mod device_guard;
pub mod device_health;
pub mod event;
pub mod execution_context;
pub(crate) mod execution_domain;
pub(crate) mod execution_resolver;
pub(crate) mod host_staging;
mod matrix;
mod owner;
mod placement;
mod profiler;
mod query_execution_context;
pub mod stream;

pub use affinity::{ExecutionBindingId, RuntimeAffinityAnchor, RuntimeAffinityId, RuntimeLeaseAnchor};
pub use binding_extension::{
    BindingExtensionGuard, BindingExtensionInitError, BindingExtensionLease, MAX_BINDING_EXTENSION_SLOTS_PER_BINDING,
};
pub use boundary::{BoundaryKind, BoundaryObservation, BoundaryObserver, QueryMemoryObservation, QueryStageKind};
pub(crate) use boundary::{
    BoundaryObserverRef, last_boundary_observation, record_boundary_observation,
    record_boundary_observation_for_observer, record_boundary_observation_in_domain,
};
pub use budget::{
    BudgetedExecutionOutcome, ExecutionBudgetEnforcement, ExecutionBudgetReport, ExecutionBudgetRequest,
    ExecutionBudgetStatus, ExecutionBudgetToken,
};
pub use device::DeviceId;
pub use device_guard::DeviceEntryGuard;
pub use device_health::{
    DeviceFault, DeviceFaultKind, DeviceHealthSnapshot, DeviceHealthStatus, device_health_snapshot,
    drop_device_resource, quarantine_device_resource, report_device_fault,
};
pub use event::CudaEvent;
pub use execution_context::ExecutionOverrides;
pub use matrix::{DenseMatrixLayout, PackedFloat32Matrix, ValidatedPackedFloat32Matrix};
pub use owner::{Runtime, RuntimeBuilder, StreamDependencyOutcome};
pub use placement::ExecutionPlacement;
pub use profiler::{cuda_profiler_start, cuda_profiler_stop};
pub use query_execution_context::QueryExecutionContext;
pub use stream::CudaStream;
