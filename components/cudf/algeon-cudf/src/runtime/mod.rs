//! Runtime utilities for device-pinned execution resources and retained provenance.
//!
//! # Control Surface
//!
//! - [`Runtime`] owns one CUDA device binding plus default [`CudaStream`] and
//!   [`crate::core::memory::DeviceMemoryResource`] handles.
//! - [`crate::core::memory::DeviceMemoryManager`] provides one reservation-aware
//!   allocator adaptor per CUDA device for query backends.
//! - [`Exec`] selects the stream and memory resource of one operation.
//! - [`CudaStream`] and [`CudaEvent`] expose low-level CUDA ordering handles.
//! - [`DeviceId`] is a validated CUDA device ordinal.

mod binding_extension;
mod binding_id;
mod budget;
mod cache_allocation;
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
pub mod stream;

pub use binding_extension::{
    BindingExtensionGuard, BindingExtensionInitError, BindingExtensionLease, MAX_BINDING_EXTENSION_SLOTS_PER_BINDING,
};
pub use binding_id::ExecutionBindingId;
pub(crate) use budget::CompleteReport;
pub use budget::{
    BudgetedExecutionOutcome, BudgetedOutput, ExecutionBudgetEnforcement, ExecutionBudgetReport,
    ExecutionBudgetRequest, ExecutionBudgetToken,
};
pub use cache_allocation::{
    CacheAllocationDomain, CacheAllocationDomainId, CacheAllocationLease, CacheAllocationScope, CacheAllocationScopeId,
    CacheAllocationScopeIdentity,
};
pub use device::DeviceId;
pub use device_guard::DeviceEntryGuard;
pub use device_health::{
    DeviceFault, DeviceFaultKind, DeviceFaultNativeStatus, DeviceFaultObservation, DeviceHealthSnapshot,
    DeviceHealthStatus, device_health_snapshot, drop_device_resource, observe_device_faults,
    quarantine_device_resource, report_device_fault,
};
pub use event::CudaEvent;
pub use execution_context::Exec;
pub(crate) use matrix::matrix_element_bytes;
pub use matrix::{DenseMatrixLayout, PackedMatrix, ValidatedPackedMatrix};
pub use owner::{Runtime, RuntimeBuilder, StreamDependencyOutcome};
pub use placement::ExecutionPlacement;
pub use profiler::{cuda_profiler_start, cuda_profiler_stop};
pub use stream::{AttemptStreamProviderAlias, AttemptStreamProviderRegistration, CudaStream};
