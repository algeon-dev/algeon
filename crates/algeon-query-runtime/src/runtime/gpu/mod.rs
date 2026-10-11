//! GPU runtime inventory, registry, and placement helpers.
//!
//! `inventory` reads device facts and diagnostics telemetry; `profile`
//! freezes one `DeviceResourceProfile` per device from them; `registry`
//! installs the memory manager, stream pool, `ledger`, and `device_context`
//! for each configured ordinal; `placement` ranks ledgers for a new attempt;
//! `lanes` assigns each admitted attempt its identity slot; `memory_failure`
//! counts allocator failures for reports; `owner` makes sure only one live
//! backend exists per process. The capacity and admission contract these
//! modules implement is `website/docs/design/admission.mdx`.

pub(crate) mod device_context;
pub(crate) mod inventory;
pub(crate) mod lanes;
pub(crate) mod ledger;
pub(crate) mod memory_failure;
mod owner;
pub(crate) mod placement;
pub(crate) mod profile;
pub(crate) mod registry;

pub(crate) use device_context::DeviceContext;
#[cfg(test)]
pub(crate) use inventory::device_budget_bytes_from_facts;
pub(crate) use inventory::{AlgeonGpuInventory, AlgeonNvmlTelemetry, GpuInventory, GpuTelemetry};
pub use inventory::{DeviceFacts, DeviceTelemetrySnapshot, DeviceTelemetryStatus};
pub(crate) use lanes::{DeviceLanePermit, DeviceWorkerScope};
#[cfg(test)]
pub(crate) use ledger::LedgerSnapshot;
pub(crate) use ledger::{
    DeviceAdmissionHealth, DeviceLedger, GrantIdentity, LedgerTransition, LedgerTransitionKind, MemoryGrant,
};
pub use ledger::{DeviceLedgerGrantRow, DeviceLedgerReport};
pub use memory_failure::DeviceMemoryFailureReport;
pub(crate) use memory_failure::DeviceMemoryFailureTracker;
pub(crate) use owner::{DeviceOwnerToken, LiveBackendLease, claim_live_backend, normalize_device_ordinals};
pub(crate) use placement::PlacementRequest;
pub(crate) use profile::DeviceResourceEnvelope;
pub use profile::{DeviceCapabilities, DeviceCapability, DeviceResourceProfile};
pub(crate) use registry::DeviceRegistry;
