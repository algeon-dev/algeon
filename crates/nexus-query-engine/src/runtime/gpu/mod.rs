//! GPU runtime inventory, registry, and placement helpers.

pub mod comms;
pub mod device_context;
pub mod governor;
pub mod inventory;
pub mod memory_failure;
pub mod placement;
pub mod qos;
pub mod registry;

pub use comms::{GpuComms, SingleDeviceComms};
pub(crate) use device_context::DeviceActiveQuerySlot;
pub use device_context::DeviceContext;
pub(crate) use governor::BudgetChangeObserver;
pub use governor::{
    EvictableMemoryObserver, EvictableMemoryPressure, GrantRequest, MemoryGovernor, MemoryGovernorGrantRow,
    MemoryGovernorReport, MemoryGrant,
};
pub use inventory::{
    CudfInventory, CudfNvmlTelemetry, DeviceFacts, DeviceTelemetrySnapshot, DeviceTelemetryStatus, GpuInventory,
    GpuTelemetry, device_budget_bytes_from_facts,
};
pub use memory_failure::{DeviceMemoryFailureReport, DeviceMemoryFailureTracker};
pub use placement::{
    BestFitPlacement, LeastLoadedPlacement, PinnedDevicePlacement, PlacementPolicy, PlacementRequest,
    QueryPlacementHint, QueryPlacementPolicy,
};
pub use qos::{Priority, QueryQos};
pub use registry::{DeviceRegistry, DeviceRegistryConfig};

/// Query placement selected by the service scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Placement {
    /// All work for the query runs on one CUDA device.
    Single(cudf::DeviceId),
}
