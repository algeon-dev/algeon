//! GPU runtime inventory and policy helpers.

pub mod inventory;

pub use inventory::{CudfInventory, DeviceFacts, GpuInventory, device_budget_bytes_from_facts};
