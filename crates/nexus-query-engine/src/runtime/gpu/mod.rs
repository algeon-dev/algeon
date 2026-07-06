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
pub use device_context::DeviceContext;
pub use governor::{GrantRequest, MemoryGovernor, MemoryGovernorGrantRow, MemoryGovernorReport, MemoryGrant};
pub use inventory::{CudfInventory, DeviceFacts, GpuInventory, device_budget_bytes_from_facts};
pub use memory_failure::{DeviceMemoryFailureReport, DeviceMemoryFailureTracker};
pub use placement::{LeastLoadedPlacement, PinnedDevicePlacement, PlacementPolicy, PlacementRequest};
pub use qos::{Priority, QueryQos};
pub use registry::{DeviceRegistry, DeviceRegistryConfig};

/// Query placement selected by the service scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Placement {
    /// All work for the query runs on one CUDA device.
    Single(cudf::DeviceId),
}
