//! GPU runtime inventory, registry, and placement helpers.

pub mod comms;
pub(crate) mod device_context;
pub mod governor;
pub mod inventory;
pub mod memory_failure;
mod owner;
pub(crate) mod placement;
pub mod qos;
pub(crate) mod registry;

pub use comms::{GpuComms, SingleDeviceComms};
pub(crate) use device_context::{DeviceActiveQuerySlot, DeviceContext};
pub(crate) use governor::BudgetChangeObserver;
pub use governor::{
    EvictableMemoryObserver, EvictableMemoryPressure, GrantRequest, MemoryGovernor, MemoryGovernorGrantRow,
    MemoryGovernorReport, MemoryGrant,
};
pub use inventory::{
    DeviceFacts, DeviceTelemetrySnapshot, DeviceTelemetryStatus, GpuInventory, GpuTelemetry, NexusGpuInventory,
    NexusNvmlTelemetry, device_budget_bytes_from_facts,
};
pub use memory_failure::{DeviceMemoryFailureReport, DeviceMemoryFailureTracker};
pub(crate) use owner::{DeviceOwnerToken, LiveBackendLease, claim_live_backend, normalize_device_ordinals};
pub(crate) use placement::PlacementPolicy;
pub use placement::{PlacementRequest, QueryPlacementHint, QueryPlacementPolicy};
pub use qos::{Priority, QueryQos};
pub(crate) use registry::{DeviceRegistry, DeviceRegistryConfig};

/// Query placement selected by the service scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Placement {
    /// All work for the query runs on one CUDA device.
    Single(cudf::DeviceId),
}
