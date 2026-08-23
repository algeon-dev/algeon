//! GPU runtime inventory, registry, and placement helpers.

#[doc(hidden)]
pub mod device_context;
pub mod inventory;
#[doc(hidden)]
pub mod lanes;
pub mod ledger;
pub mod memory_failure;
mod owner;
#[doc(hidden)]
pub mod placement;
pub mod profile;
#[doc(hidden)]
pub mod registry;

#[doc(hidden)]
pub use device_context::DeviceContext;
pub use inventory::{
    DeviceFacts, DeviceTelemetrySnapshot, DeviceTelemetryStatus, GpuInventory, GpuTelemetry, NexusGpuInventory,
    NexusNvmlTelemetry, device_budget_bytes_from_facts,
};
#[doc(hidden)]
pub use lanes::{DeviceLanePermit, DeviceWorkerScope};
#[cfg(test)]
#[doc(hidden)]
pub use ledger::LedgerTransitionKind;
pub use ledger::{
    DeviceAdmissionHealth, DeviceLedger, DeviceLedgerGrantRow, DeviceLedgerReport, GrantIdentity, LedgerSnapshot,
    MemoryGrant,
};
#[doc(hidden)]
pub use ledger::{LedgerTransition, MemoryGrantPhase};
pub use memory_failure::{DeviceMemoryFailureReport, DeviceMemoryFailureTracker};
#[doc(hidden)]
pub use owner::{DeviceOwnerToken, LiveBackendLease, claim_live_backend, normalize_device_ordinals};
#[doc(hidden)]
pub use placement::PlacementRequest;
pub use profile::{DeviceCapabilities, DeviceCapability, DeviceResourceEnvelope, DeviceResourceProfile};
pub(crate) use registry::DeviceRegistry;

/// Query placement selected by the service scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Placement {
    /// All work for the query runs on one CUDA device.
    Single(cudf::DeviceId),
}
