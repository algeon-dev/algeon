//! GPU runtime inventory, registry, and placement helpers.

pub(crate) mod device_context;
pub mod inventory;
pub(crate) mod lanes;
pub mod ledger;
pub mod memory_failure;
mod owner;
pub(crate) mod placement;
pub mod profile;
pub(crate) mod registry;

pub(crate) use device_context::DeviceContext;
pub use inventory::{
    DeviceFacts, DeviceTelemetrySnapshot, DeviceTelemetryStatus, GpuInventory, GpuTelemetry, NexusGpuInventory,
    NexusNvmlTelemetry, device_budget_bytes_from_facts,
};
pub(crate) use lanes::{DeviceLanePermit, DeviceWorkerScope};
pub(crate) use ledger::LedgerTransition;
#[cfg(test)]
pub(crate) use ledger::LedgerTransitionKind;
pub use ledger::{
    DeviceAdmissionHealth, DeviceLedger, DeviceLedgerGrantRow, DeviceLedgerReport, EvictableMemoryObserver,
    EvictableMemoryPressure, LedgerSnapshot, MemoryGrant,
};
pub use memory_failure::{DeviceMemoryFailureReport, DeviceMemoryFailureTracker};
pub(crate) use owner::{DeviceOwnerToken, LiveBackendLease, claim_live_backend, normalize_device_ordinals};
pub(crate) use placement::{PlacementPolicy, PlacementRequest};
pub use profile::{DeviceCapabilities, DeviceCapability, DeviceResourceProfile};
pub(crate) use registry::{DeviceRegistry, DeviceRegistryConfig};

/// Query placement selected by the service scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Placement {
    /// All work for the query runs on one CUDA device.
    Single(cudf::DeviceId),
}
